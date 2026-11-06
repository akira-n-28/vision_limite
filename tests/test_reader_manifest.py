#!/usr/bin/env python3
"""Test CPU di C7 per UniMER-1M e MAVIS-Geometry: reader col manifest delle ammissioni per record, split e compito
(limite_vl.manifest.TASKED, stream.make_sources, SourceReader, validation_samples, MixedRowStream e i loader di
limite_vl.data). I manifest si costruiscono coi fixture di tests/test_unimer_manifest.py e tests/test_mavis_manifest.py
(scripts/unimer_manifest.py, scripts/mavis_manifest.py); dei dati reali si usa solo l'esistenza degli shard, per gli
arresti senza manifest, che avvengono prima di leggerli. Solo in un job (slurm/tests_cpu.sbatch).

Controlla:
- UniMER: un'epoca consegna esattamente le formule ammesse dello split (train o dev), ciascuna una volta; quarantena,
  esclusi e l'altro split non entrano, coi motivi contati; validazione dal dev del manifest nell'ordine degli shard;
  righe ritagliate con la loro decisione (stessa chiave e record_id della formula, ma la formula ammessa non ammette la
  riga); rietichettature dalla lista registrata nel manifest, applicate e verificate (un'etichetta non rietichettata col
  manifest ferma il loader); tipi e insiemi incompatibili, manifest non identificato o diverso, liste della config,
  lista delle rietichettature alterata, revisione degli shard diversa, shard alterato;
- MAVIS: epoca e validazione come sopra; compiti scelti solo fra quelli ammessi, shape_type mai, gli altri compiti del
  record disponibili e le domande bloccate contate; digest delle domande e della descrizione legati al manifest;
  selezione riproducibile per chiave; un record senza compiti utilizzabili saltato col suo motivo (nessun ripiego, nessun
  ciclo); il flusso si ferma se una sorgente non costruisce più esempi;
- shard reali di UniMER e MAVIS senza manifest rifiutati (MANIFEST_REQUIRED, anche le varianti non registrate); liste
  storiche non richieste con il manifest;
- MixedRowStream col processor di Violetto: solo chiavi ammesse di train, contatori per sorgente e compito."""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import mavis_manifest as mm  # noqa: E402
import test_mavis_manifest as tm  # noqa: E402
import test_unimer_manifest as tu  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl import data as dt  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl import stream as st  # noqa: E402
from limite_vl.data_lists import missing_required, provenance  # noqa: E402
from limite_vl.f1_mavis import mavis_questions, mavis_truth  # noqa: E402
from limite_vl.latex import effective_label  # noqa: E402
from limite_vl.provenance import sha256_file  # noqa: E402

VIOLETTO = ROOT / "hf/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
FAILS: list[str] = []
DONE: list[str] = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:400]})" if detail != "" else ""))
    DONE.append(name)
    if not cond:
        FAILS.append(name)


def refused(fn) -> str:
    """Il messaggio del ManifestError atteso, vuoto se fn non lo solleva."""
    try:
        fn()
    except mf.ManifestError as exc:
        return str(exc) or "rifiutato"
    except Exception as exc:  # noqa: BLE001 - un errore di altro tipo non è l'arresto esplicito atteso
        return ""
    return ""


def epoch(source, n_workers: int = 2) -> list[tuple[str, dict]]:
    """(chiave, file) consegnati in un'epoca da n_workers reader."""
    out = []
    for w in range(n_workers):
        reader = st.SourceReader(source, seed=3, worker=w, n_workers=n_workers, val_count=0)
        while True:
            key, files = reader.next()
            if reader.epoch > 0:
                break
            out.append((key, files))
    return out


class EpochReader(st.SourceReader):
    """SourceReader che fotografa i contatori quando finisce la prima epoca: next() li fa avanzare fino al primo
    campione dell'epoca dopo."""

    def _advance_shard(self) -> None:
        super()._advance_shard()
        if self.epoch == 1 and not hasattr(self, "epoch0"):
            self.epoch0 = dict(self.counts)


def counts(source) -> dict:
    """Contatori di un reader che legge esattamente un'epoca con un worker."""
    reader = EpochReader(source, seed=3, worker=0, n_workers=1, val_count=0)
    while True:
        reader.next()
        if reader.epoch > 0:
            break
    return reader.epoch0


def shard_files(paths, wanted: set[str]) -> dict[str, dict]:
    out = {}
    for p in paths:
        for key, files in dt.iter_shard(p):
            if key in wanted:
                out[key] = files
    return out


def rewritten(folder: Path, dst: Path, edit, summarize) -> Path:
    """Copia del manifest coi record alterati da edit(record), sha256 dei record e conteggi aggiornati."""
    shutil.copytree(folder, dst)
    path = dst / "records.jsonl.gz"
    recs = [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()]
    for r in recs:
        edit(r)
    path.unlink()
    sha, _ = um.write_records_gz(path, (json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                                        for r in recs))
    meta = json.loads((dst / "manifest.json").read_text())
    meta["records"]["sha256"] = sha
    meta["conteggi"] = summarize(recs)
    (dst / "manifest.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    return dst


# ---------------------------------------------------------------- UniMER

def test_unimer(tmp: Path) -> dict:
    s = tu.spec()
    args = tu.write_fixture(tmp, s)
    um.run(args)
    folder, sha = args.out, sha256_file(args.out / "manifest.json")
    by = {(r["insieme"], r["record"]): r for r in tu.records_of(folder)}
    formula_glob, lines_glob = str(args.shards / "unimer-train-*.tar"), str(args.shards / "lines" / "*.tar")

    def spec(name, kind, split, shards=None, **over):
        return {"name": name, "kind": kind, "weight": 1.0, "manifest": str(folder), "manifest_sha256": sha,
                "split": split, "shards": shards or (formula_glob if kind == "unimer_formula" else lines_glob), **over}

    def want(insieme, split):
        return sorted(k for (i, k), r in by.items() if i == insieme and r["split"] == split and r["stato"] == "ammesso")

    st_of = {k: r["stato"] for (i, k), r in by.items() if i == "train"}
    check("preparazione UniMER: stati del fixture (errori certi e immagine del test esclusi, incerti e formule lunghe in "
          "quarantena, rietichettature ammesse)",
          st_of["train-0000003"] == st_of["train-0000009"] == st_of["train-0000016"] == "escluso"
          and st_of["train-0000004"] == st_of["train-0000008"] == "quarantena"
          and st_of["train-0000002"] == st_of["train-0000013"] == "ammesso", st_of)
    f_train, f_dev = st.make_sources([spec("f", "unimer_formula", "train"), spec("fd", "unimer_formula", "dev")])
    got_train, got_dev = epoch(f_train), epoch(f_dev)
    k_train, k_dev = [k for k, _ in got_train], [k for k, _ in got_dev]
    check("UniMER formule: un'epoca con 2 worker consegna esattamente le ammesse dello split, una volta (train e dev)",
          sorted(k_train) == want("train", "train") and sorted(k_dev) == want("train", "dev")
          and len(set(k_train)) == len(k_train) and len(set(k_dev)) == len(k_dev), (len(k_train), len(k_dev)))
    groups = {split: {by[("train", k)]["gruppo"] for k in ks} for split, ks in (("train", k_train), ("dev", k_dev))}
    check("UniMER: train e dev senza chiavi né gruppi in comune; quarantena ed esclusi mai consegnati",
          not set(k_train) & set(k_dev) and not groups["train"] & groups["dev"]
          and all(st_of[k] == "ammesso" for k in k_train + k_dev))
    c = counts(f_train)
    exp = {"quarantena": 0, "escluso": 0, "split_dev": 0}
    for (i, k), r in by.items():
        if i != "train":
            continue
        if r["split"] != "train":
            exp["split_dev"] += 1
        elif r["stato"] != "ammesso":
            exp[r["stato"]] += 1
    check("UniMER reader: motivi dei record saltati uguali a quelli del manifest (altro split, quarantena, escluso)",
          all(c.get(f"saltati_manifest/{w}", 0) == n for w, n in exp.items())
          and c["consegnati"] == len(want("train", "train")) and not c.get("errori_manifest"), (c, exp))
    order = [r["record"] for r in sorted((r for (i, _), r in by.items() if i == "train"), key=lambda r: (r["shard"], r["indice"]))
             if r["split"] == "dev" and r["stato"] == "ammesso"]
    val = st.validation_samples(f_train, 1000)
    check("UniMER validazione: il dev del manifest, nell'ordine degli shard, con la decisione allegata",
          [k for k, _ in val] == order and all(mf.ADMISSION_FILE in f for _, f in val), len(val))
    # ---- rietichettature
    delivered = dict(got_train + got_dev)
    relabeled = {"train-0000002": "3", "train-0000013": effective_label(tu.ARRAY_CODED)[0]}
    targets = {k: dt.SAMPLE_LOADERS["unimer_formula"](k, delivered[k]).messages[-1]["content"] for k in relabeled}
    check("UniMER rietichettature: applicate dalla lista registrata nel manifest, l'etichetta nuova è il target",
          targets == relabeled and dt.SAMPLE_LOADERS["unimer_formula"]("train-0000001", delivered["train-0000001"])
          .messages[-1]["content"] == "x^{2}+1", targets)
    raw = shard_files(f_train.shards, {"train-0000002"})["train-0000002"]
    adm_source = f_train if f_train.admission.allows("train-0000002") else f_dev
    unrelabeled = adm_source.admission.attach("train-0000002", raw)
    check("UniMER: una formula rietichettata nel manifest senza la sua rietichettatura ferma il loader",
          bool(refused(lambda: dt.SAMPLE_LOADERS["unimer_formula"]("train-0000002", unrelabeled))))
    check("UniMER: rietichettature della sorgente tutte e sole quelle del manifest, con le etichette della lista",
          f_train.relabel == f_dev.relabel == {r["key"]: (r["label"], r["new_label"]) for r in s["relabel"]},
          f_train.relabel)
    # ---- righe ritagliate: stessa chiave, decisione propria
    righe = {k: r for (i, k), r in by.items() if i == "righe"}
    line_split = righe["train-0000012"]["split"]
    other = "dev" if line_split == "train" else "train"
    l_src, ls_src = st.make_sources([spec("l", "unimer_line", line_split), spec("ls", "unimer_line_select", line_split)])
    got_l, got_ls = epoch(l_src, 1), epoch(ls_src, 1)
    check("UniMER righe: consegnate solo le righe ammesse dello split (train-0000012), per entrambi i compiti",
          [k for k, _ in got_l] == [k for k, _ in got_ls] == want("righe", line_split) == ["train-0000012"],
          [k for k, _ in got_l])
    shared = righe["train-0000013"]
    check("UniMER chiavi condivise: la formula rietichettata train-0000013 è ammessa, la sua riga no (stesso record_id)",
          shared["record_id"] == by[("train", "train-0000013")]["record_id"] and shared["stato"] == "quarantena"
          and st_of["train-0000013"] == "ammesso" and "train-0000013" in k_train + k_dev
          and l_src.admission.refusal("train-0000013") in ("quarantena", f"split_{other}"))
    s_line = dt.SAMPLE_LOADERS["unimer_line"]("train-0000012", got_l[0][1])
    s_sel = dt.SAMPLE_LOADERS["unimer_line_select"]("train-0000012", got_ls[0][1])
    meta = json.loads(got_l[0][1]["json"])
    check("UniMER righe: ritaglio e scelta della riga invariati, target la riga, compiti dichiarati",
          s_line.messages[-1]["content"] in meta["rows"] and s_sel.messages[-1]["content"] in meta["rows"]
          and s_line.tasks == ("transcribe_line",) and s_sel.tasks == ("transcribe_line_select",))
    check("UniMER: una riga senza il suo compito ammesso ferma il loader (NoAdmittedTask, nessun ripiego)",
          isinstance(_raises(lambda: dt.SAMPLE_LOADERS["unimer_line"]("train-0000012", {
              **got_l[0][1], mf.ADMISSION_FILE: json.dumps({"compiti_ammessi": ["transcribe_line_select"]}).encode()})),
              dt.NoAdmittedTask))
    check("UniMER righe: nessuna riga ammessa nell'altro split, errore esplicito invece di una sorgente vuota",
          "nessun record ammesso" in refused(lambda: st.make_sources([spec("lx", "unimer_line", other)])))
    # ---- arresti
    stops = {
        "tipo delle formule sugli shard delle righe": spec("x", "unimer_formula", "train", shards=lines_glob),
        "tipo delle righe sugli shard delle formule": spec("x", "unimer_line", "train", shards=formula_glob),
        "manifest senza sha256 dichiarato": {k: v for k, v in spec("x", "unimer_formula", "train").items()
                                             if k != "manifest_sha256"},
        "sha256 dichiarato diverso": spec("x", "unimer_formula", "train", manifest_sha256="0" * 64),
        "lista di esclusione nella config": spec("x", "unimer_formula", "train", exclude=[str(args.check / "uncertain.txt")]),
        "rietichettature nella config": spec("x", "unimer_formula", "train", relabel=[str(args.check / "relabel.jsonl")]),
        "tipo di MAVIS col manifest di UniMER": spec("x", "mavis_f1", "train"),
        "manifest mancante": spec("x", "unimer_formula", "train", manifest=str(tmp / "nessuno")),
    }
    for name, sp in stops.items():
        why = refused(lambda sp=sp: st.make_sources([sp]))
        check(f"UniMER arresto: {name}", bool(why), why[:160])
    relabel_file = args.check / "relabel.jsonl"
    saved = relabel_file.read_bytes()
    relabel_file.write_bytes(saved.replace(b'"new_label": "3"', b'"new_label": "4"'))
    why = refused(lambda: st.make_sources([spec("x", "unimer_formula", "train")]))
    relabel_file.write_bytes(saved)
    check("UniMER arresto: lista delle rietichettature diversa da quella registrata nel manifest", "relabel" in why, why)
    sharding = args.shards / "manifest.json"
    saved = sharding.read_bytes()
    sharding.write_bytes(saved.replace(tu.REV.encode(), b"f" * 40))
    why = refused(lambda: st.make_sources([spec("x", "unimer_formula", "train")]))
    sharding.write_bytes(saved)
    check("UniMER arresto: shard di un'altra revisione (manifest dello sharding)", "revisione" in why or "@" in why, why)
    lines_tar = args.shards / "lines" / "unimer-lines-00000.tar"
    raw_tar = bytearray(lines_tar.read_bytes())
    raw_tar[600] ^= 1
    lines_tar.write_bytes(bytes(raw_tar))
    why = refused(lambda: st.make_sources([spec("x", "unimer_line", line_split)]))
    check("UniMER arresto: shard alterato dopo la costruzione del manifest (sha256)", "sha256" in why, why)
    lines_tar.write_bytes(bytes(raw_tar[:600]) + bytes([raw_tar[600] ^ 1]) + bytes(raw_tar[601:]))
    return {"sources": [f_train, l_src] if line_split == "train" else [f_train], "admitted": set(want("train", "train"))
            | (set(want("righe", "train")) if line_split == "train" else set())}


def _raises(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


# ---------------------------------------------------------------- MAVIS

def test_mavis(tmp: Path) -> dict:
    s = tm.spec()
    args = tm.write_fixture(tmp, s)
    mm.run(args)
    folder, sha = args.out, sha256_file(args.out / "manifest.json")
    recs = {r["record"]: r for r in tu.records_of(folder)}
    glob_ = str(args.shards / "*.tar")

    def spec(name, split, kind="mavis_f1", **over):
        return {"name": name, "kind": kind, "shards": glob_, "weight": 1.0, "manifest": str(folder),
                "manifest_sha256": sha, "split": split, **over}

    def admitted_tasks(r):
        return {t["compito"] for t in r["compiti"] if t["stato"] == "ammesso"} if r["stato"] == "ammesso" else set()

    def want(split):
        return sorted(k for k, r in recs.items() if r["split"] == split and admitted_tasks(r))

    m_train, m_dev = st.make_sources([spec("m", "train"), spec("md", "dev")])
    got_train, got_dev = epoch(m_train), epoch(m_dev)
    k_train, k_dev = [k for k, _ in got_train], [k for k, _ in got_dev]
    quarantined = sorted(k for k, r in recs.items() if r["stato"] != "ammesso")
    check("MAVIS: un'epoca consegna esattamente i record ammessi dello split con un compito ammesso (train e dev)",
          sorted(k_train) == want("train") and sorted(k_dev) == want("dev") and not set(k_train) & set(k_dev)
          and not set(quarantined) & set(k_train + k_dev) and len(quarantined) >= 3, (len(k_train), len(k_dev), quarantined))
    check("MAVIS validazione: lo split dev del manifest, ogni record una volta",
          sorted(k for k, _ in st.validation_samples(m_train, 1000)) == want("dev"))
    built, bad, blocked_seen, shape_pool = {}, [], 0, 0
    for key, files in got_train + got_dev:
        smp = dt.SAMPLE_LOADERS["mavis_f1"](key, files)
        adm = dt.admission_of(files)
        allowed = set(adm["compiti_ammessi"])
        built[key] = smp
        pool_types = [q["type"] for q in mavis_questions(key, mavis_truth(json.loads(files["json"])["text_en"]["caption"],
                                                                          json.loads(files["json"])["text_en"][
                                                                              "diagram_generation"])[0])]
        shape_pool += "shape_type" in pool_types
        if not set(smp.tasks) <= allowed or "shape_type" in smp.tasks:
            bad.append(key)
        if smp.tasks != ("describe",):
            blocked_seen += smp.blocked.get("shape_type", 0) == pool_types.count("shape_type") > 0
        else:
            desc = smp.messages[-1]["content"]
            if hashlib.sha256(desc.encode()).hexdigest() != recs[key]["descrizione_sha256"]:
                bad.append(key)
    check("MAVIS: compiti emessi solo fra quelli ammessi, shape_type mai, descrizioni coi digest del manifest",
          not bad and shape_pool > 0, (bad, shape_pool))
    check("MAVIS: domande di shape_type tolte dal pool prima della scelta e contate fra le bloccate",
          blocked_seen > 0, blocked_seen)
    # shape_type bloccata con gli altri compiti disponibili: la descrizione non ammessa obbliga alle domande
    key = next(k for k, f in got_train if "shape_type" in json.dumps(dt.admission_of(f)["compiti_bloccati"]))
    files = dict(got_train)[key]
    adm = dt.admission_of(files)
    no_desc = {**files, mf.ADMISSION_FILE: mf.canonical({**adm, "compiti_ammessi": [t for t in adm["compiti_ammessi"]
                                                                                   if t != "describe"]}).encode()}
    smp = dt.SAMPLE_LOADERS["mavis_f1"](key, no_desc)
    check("MAVIS: record con shape_type bloccata e altri compiti disponibili, scelti fra le domande ammesse",
          smp.tasks and "shape_type" not in smp.tasks and set(smp.tasks) <= set(adm["compiti_ammessi"])
          and smp.blocked.get("shape_type", 0) >= 1, (key, smp.tasks, smp.blocked))
    only_desc = {**files, mf.ADMISSION_FILE: mf.canonical({**adm, "compiti_ammessi": ["describe"]}).encode()}
    none = {**files, mf.ADMISSION_FILE: mf.canonical({**adm, "compiti_ammessi": []}).encode()}
    check("MAVIS: famiglia senza compiti ammessi sostituita dall'altra ammessa; senza nessun compito NoAdmittedTask",
          dt.SAMPLE_LOADERS["mavis_f1"](key, only_desc).tasks == ("describe",)
          and isinstance(_raises(lambda: dt.SAMPLE_LOADERS["mavis_f1"](key, none)), dt.NoAdmittedTask))
    wrong_qa = {**files, mf.ADMISSION_FILE: mf.canonical({**adm, "qa_sha256": "0" * 64}).encode()}
    check("MAVIS: domande diverse dai digest del manifest fermano il loader",
          isinstance(_raises(lambda: dt.SAMPLE_LOADERS["mavis_f1"](key, wrong_qa)), mf.ManifestError))
    again = {k: dt.SAMPLE_LOADERS["mavis_f1"](k, f) for k, f in epoch(m_train, 1)}
    check("MAVIS: selezione riproducibile per chiave (stessi messaggi con 1 e 2 worker)",
          all(again[k].messages == built[k].messages and again[k].tasks == built[k].tasks for k in again))
    raw = shard_files(m_train.shards, {key})[key]
    old = dt.SAMPLE_LOADERS["mavis_f1"](key, raw)
    check("MAVIS senza manifest: la scelta storica, su tutti i compiti (percorso diretto, non protetto)",
          old.blocked == {} and (old.tasks == ("describe",) or set(old.tasks) <= {q["type"] for q in mavis_questions(
              key, mavis_truth(json.loads(raw["json"])["text_en"]["caption"],
                               json.loads(raw["json"])["text_en"]["diagram_generation"])[0])}))
    # record ammesso senza compiti utilizzabili: saltato dal reader col suo motivo
    victim = sorted(want("train"))[0]

    def no_tasks(r):
        if r["record"] == victim:
            for t in r["compiti"]:
                t["stato"] = "quarantena"

    alt = rewritten(folder, tmp / "senza-compiti", no_tasks, mm.summarize)
    (alt_src,) = st.make_sources([{**spec("a", "train"), "manifest": str(alt), "manifest_sha256": sha256_file(
        alt / "manifest.json")}])
    c = counts(alt_src)
    check("MAVIS: record ammesso senza compiti utilizzabili saltato dal reader, contato, mai consegnato",
          c.get("saltati_manifest/nessun_compito_ammesso") == 1 and victim not in [k for k, _ in epoch(alt_src)]
          and alt_src.admission.refusal(victim) == "nessun_compito_ammesso", c)
    stops = {"tipo mavis_caption (compiti non registrati)": spec("x", "train", kind="mavis_caption"),
             "tipo mavis_qa (compiti non registrati)": spec("x", "train", kind="mavis_qa"),
             "tipo di UniMER col manifest di MAVIS": spec("x", "train", kind="unimer_formula"),
             "sha256 dichiarato diverso": spec("x", "train", manifest_sha256="1" * 64)}
    for name, sp in stops.items():
        why = refused(lambda sp=sp: st.make_sources([sp]))
        check(f"MAVIS arresto: {name}", bool(why), why[:160])
    # una sorgente che non costruisce più esempi ferma il flusso, invece di girare a vuoto
    saved_loader, saved_limit = dt.SAMPLE_LOADERS["mavis_f1"], st.FAILURE_LIMIT

    def always(key, files):
        raise dt.NoAdmittedTask(f"{key}: prova")

    dt.SAMPLE_LOADERS["mavis_f1"], st.FAILURE_LIMIT = always, 25
    try:
        stream = st.MixedRowStream([m_train], None, max_len=4096, micro_rows=1, seed=0, shuffle_buffer=4, pack_buffer=4)
        err = _raises(lambda: next(stream.iter_worker(0, 1)))
    finally:
        dt.SAMPLE_LOADERS["mavis_f1"], st.FAILURE_LIMIT = saved_loader, saved_limit
    check("flusso: una sorgente senza esempi costruibili lo ferma con un errore (nessun ciclo infinito)",
          isinstance(err, RuntimeError) and "di seguito" in str(err), repr(err)[:200])
    d = m_train.admission.describe()
    check("MAVIS: descrizione dell'ammissione con identità, insieme, compiti attesi e decisione non gold",
          d["sorgente"] == "mavis-geometry" and d["insieme"] == "train" and d["tipo"] == "mavis_f1"
          and any(k.endswith("/quarantena/shape_type") for k in d["attesi_per_compito"]) and "non gold" in d["decisione"],
          d)
    return {"sources": [m_train], "admitted": set(want("train"))}


# ---------------------------------------------------------------- shard reali e liste

def test_real_paths() -> None:
    real = {"UniMER denso": "data/shards/unimer/2343ddd9-dense/unimer-train-*.tar",
            "UniMER x4 (non registrato)": "data/shards/unimer/2343ddd9-x4/unimer-train-*.tar",
            "UniMER righe": "data/shards/unimer/2343ddd9-dense/lines/*.tar",
            "MAVIS": "data/shards/mavis-geometry/9c80fff3-caption/*.tar"}
    for name, glob_ in real.items():
        kind = "mavis_f1" if "mavis" in glob_ else "unimer_line" if "lines" in glob_ else "unimer_formula"
        why = refused(lambda g=glob_, k=kind: st.make_sources([{"name": "r", "kind": k, "shards": str(ROOT / g),
                                                               "weight": 1.0}]))
        check(f"shard reali senza manifest rifiutati prima di leggerli: {name}", "si leggono solo" in why, why[:120])
    spec = {"name": "u", "kind": "unimer_formula", "shards": "data/shards/unimer/2343ddd9-dense/unimer-train-*.tar",
            "manifest": "data/manifests/unimer-candidato-2", "manifest_sha256": "0" * 64, "split": "train"}
    check("liste storiche: non richieste con il manifest, richieste senza (comportamento di prima)",
          missing_required([spec]) == [] and missing_required([{k: v for k, v in spec.items() if k != "manifest"}]))
    check("provenienza: manifest con lo sha256 dichiarato dalla config",
          provenance([{**spec, "manifest": str(ROOT / "data/manifests/unimer-candidato-2")}])["u"]["manifest"][
              "sha256_dichiarato"] == "0" * 64)


def test_stream(cases: list[dict]) -> None:
    from limite_vl.processing import LimiteVLProcessor

    processor = LimiteVLProcessor.from_pretrained(str(VIOLETTO))
    sources = [s for c in cases for s in c["sources"]]
    admitted = {s.name: c["admitted"] for c in cases for s in c["sources"]}
    stream = st.MixedRowStream(sources, processor, max_len=4096, micro_rows=2, seed=0, shuffle_buffer=8, pack_buffer=8)
    keys, meta = [], {}
    for n, batch in enumerate(stream.iter_worker(0, 1)):
        keys += [k.split("/", 1) for k in batch["meta"]["keys"]]
        meta = batch["meta"]
        if n >= 15:
            break
    stats = meta.get("stats", {})
    wrong = [f"{s}/{k}" for s, k in keys if k not in admitted[s]]
    check("flusso col processor di Violetto: solo chiavi ammesse di train per ogni sorgente, nessun errore",
          keys and not wrong and not any(k.startswith(("errors/", "manifest_errors/", "senza_compiti/")) for k in stats),
          (wrong[:5], {k: v for k, v in stats.items() if "/" in k}))
    check("flusso: contatori per sorgente e compito nel meta (caricati, compiti, nessun shape_type emesso)",
          all(stats.get(f"loaded/{s.name}", 0) > 0 for s in sources)
          and any(k.startswith("compiti/m/") for k in stats) and "compiti/m/shape_type" not in stats
          and stats.get("compiti/f/transcribe", 0) > 0 and set(meta["readers"]) == {s.name for s in sources},
          {k: v for k, v in stats.items() if k.startswith(("compiti", "loaded", "too_long"))})


def main() -> None:
    cases = []
    for test in (test_unimer, test_mavis):
        tmp = Path(tempfile.mkdtemp(prefix="reader_manifest_"))
        cases.append((tmp, test(tmp)))
    test_real_paths()
    test_stream([c for _, c in cases])
    for tmp, _ in cases:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\ntest_reader_manifest: {len(DONE) - len(FAILS)} controlli superati su {len(DONE)}"
          + (f", FALLITI: {len(FAILS)}" if FAILS else ""))
    if FAILS:
        for f in FAILS:
            print(" -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
