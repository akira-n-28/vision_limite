#!/usr/bin/env python3
"""Manifest candidato di UniMER-1M (docs/F0_CAMPIONE_CANDIDATO.md §6 punto 3; S1 §§1-3, 5, 7, 9; S2 §4), col sottoinsieme
di S1 già usato per i sintetici v4 (limite_vl.manifest). Solo in un job SLURM CPU: shard ed evidenze in sola lettura,
scrive cartelle nuove. Organizza candidati ed evidenze: non certifica la qualità, non assegna gold e non autorizza il
training.

Ingressi, verificati prima di scrivere:
- shard densi di `wanderkid/UniMER_Dataset@2343ddd9` (--shards, job 29729): ogni tar con lo sha256 e la dimensione del
  manifest dello sharding, letto e hashato in un solo passaggio; le righe ritagliate (lines/) e UniMER-Test (test/)
  dello stesso sharding; lo snapshot di Hugging Face con i blob dichiarati dal manifest dello sharding;
- unimer_check/run4 (--check, job 30047, unimer-check/4): errori certi, incerti, rietichettature e motivi;
- unimer_overlap/run2 (--overlap, job 30048): formule e immagini del training che compaiono in UniMER-Test;
- split_duplicates/run1/unimer (--duplicates, job 30043): solo controllo incrociato dei gruppi.

Record, una riga JSON per campione di uno shard:
- formule intere (insieme `train`), compito transcribe;
- righe ritagliate (insieme `righe`): lo stesso campione originale, quindi lo stesso record_id della formula, con le
  fasce delle righe; compiti transcribe_line e transcribe_line_select; trasformazione `righe` e il genitore (shard,
  posizione e hash della formula intera).
UniMER-Test non entra: è valutazione, coi ruoli di pannello-v3, e qui serve solo a riconoscere le contaminazioni.

Gruppi (S1 §3): componenti dei legami esatti, cioè gli stessi byte del PNG negli shard densi; ogni riga ritagliata sta
nel gruppo della sua formula. Formule uguali con immagini diverse non si uniscono, perché una formula comune non
identifica un problema (S1 §5): il dev misura immagini nuove, non formule nuove.

Stati (vince il più grave):
- `escluso`: errori certi di run4 (per record); stessa immagine di un campione di UniMER-Test (per gruppo: legame
  confermato con un elemento di valutazione, S1 §5);
- `quarantena`: incerti di run4, cioè conflitti irrisolti, proporzioni fuori norma e marcatori non riparabili (per
  record); formula di UniMER-Test con più di 20 token visibili e immagine diversa (per gruppo, da riesaminare, S1 §5);
- `ammesso`: candidato del manifest, non certificato; una formula rietichettata o compilabile non diventa gold;
- le righe ritagliate prendono lo stato della loro formula, e vanno in quarantena se la formula è rietichettata,
  perché le righe vengono dall'etichetta originale.
Le formule del training uguali a una formula breve del test restano un'evidenza senza cambio di stato: la metrica
senza contaminazione usa i casi mai visti di run2.

Riconciliazione prima di scrivere: ogni chiave delle liste si risolve in esattamente un record degli shard (zero, o più
di uno, ferma il job). run4 e run2 si ricalcolano dalle regole correnti (latex.effective_label, structure_problems,
unrendered_commands, visible_tokens; conflitti per sha256 dei byte; proporzioni col z robusto di run4; corrispondenze col
test e soglia dei 20 token di run2) e devono coincidere con i loro file. Solo `immagine_vuota` di run4, che decodifica i
pixel, non si ricalcola: in run4 non ha casi.

Split (manifest.assign_splits): dev per circa --dev-fraction delle formule ammesse, gruppi indivisibili in ordine di
sha256({seme, gruppo}) con --seed; il resto train. Il rango di estrazione del campione candidato (F0_CAMPIONE_CANDIDATO
§1) è registrato per ogni gruppo: rango_f0, e per i gruppi di dev anche rango_f0_sviluppo.

Uscite: --out (data/manifests/<nome>-<versione>, scritta una volta sola) con records.jsonl.gz, manifest.json e keys/;
--report (cartella nuova) con verifica.json, i controlli fatti rileggendo il manifest; --summary, rapporto sintetico
versionato in git. Con --probe si legge un solo tar per insieme e si scrive soltanto --report/probe.json: tempi,
memoria, stima del lavoro completo e dello spazio.

    srun $RUN_PYTHON -u -B scripts/unimer_manifest.py --shards DIR --check DIR --overlap DIR --duplicates DIR \\
        --out DIR --report DIR --summary FILE [--name unimer-candidato] [--version 1] [--dev-fraction 0.05] \\
        [--seed 0] [--workers 4] [--probe]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import resource
import statistics
import struct
import sys
import tarfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.f1c import array_rows  # noqa: E402
from limite_vl.latex import effective_label, structure_problems, unrendered_commands, visible_tokens  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, refuse_existing, sha256_file, write_new_json  # noqa: E402

SOURCE = "unimer"
SAMPLE = "f0-candidato-1"  # campione candidato di F0 (F0_CAMPIONE_CANDIDATO §1)
REVISION = re.compile(r"^wanderkid/UniMER_Dataset@([0-9a-f]{40})$")
# categorie e stati di unimer-check/4 (scripts/unimer_check.py, CERTAIN e UNCERTAIN; tests/test_unimer_manifest.py
# controlla che coincidano)
CERTAIN = {"graffe", "left_right", "ambiente", "formula_vuota", "immagine_vuota", "dimensioni_diverse_dal_json",
           "comando_non_reso", "etichetta_diversa_dall_immagine"}
UNCERTAIN = {"conflitto_irrisolto", "proporzioni_fuori_norma", "marcatori_non_riparabili"}
NOT_REPRODUCED = {"immagine_vuota"}  # decodifica dei pixel: importato da run4, non ricalcolato
Z = 5.0  # z robusto delle proporzioni di run4 (--z di unimer_check.py)
LONG = 20  # formule del test oltre questi token visibili: contaminazione di run2 (--long di unimer_test_overlap.py)
SEVERITY = {"ammesso": 0, "quarantena": 1, "escluso": 2}
TASKS = {"train": ["transcribe"], "righe": ["transcribe_line", "transcribe_line_select"]}
NOT_COVERED = [
    "variant_id e uid (S1 §1): dipendono dal contratto di risposta (S3, D3). L'ammissione è per record e compito; il "
    "ritaglio della riga e la scelta della riga sono decisi dal loader con la chiave (data._unimer_line)",
    "hash dei token di input e di target, tokenizer, template e contratto (S1 §2, §7): dipendono dal contratto",
    "indice globale fra sorgenti (S1 §4): i gruppi sono i legami esatti dentro UniMER-1M; nessun confronto con le altre "
    "sorgenti di training",
    "decontaminazione esterna contro pannello-v3 (pixel, pHash e testo): non fatta, è il lotto successivo. Qui c'è solo "
    "il confronto interno con UniMER-Test di run2, ricalcolato",
    "reader col manifest (C7): gli shard di UniMER non sono in manifest.MANIFEST_REQUIRED e i loader non applicano "
    "questo manifest; è un candidato, non autorizza il training",
    "campo gold (S2 §4): nessuno; target_controllato richiede la stima campionaria di S4 e la revisione visiva",
    "equivalenza matematica delle formule: il target è l'uguaglianza dei token LaTeX senza spazi (latex.same_latex)",
]
MISSING_CHECKS = [
    "decontaminazione esterna di UniMER contro pannello-v3 (immagini e formule di tutti i benchmark del pannello)",
    "stima campionaria della correttezza e dell'osservabilità alla risoluzione scelta (S4, revisione competente)",
    "controllo immagine_vuota di run4 non ricalcolato (decodifica dei pixel); in run4 nessun caso",
    "quarantene da riesaminare: conflitti irrisolti di run4 e formule lunghe del test con immagine diversa",
    "manoscritti incidentali di CROHME (meno dell'1%): non si separano per indice e restano (v1.5)",
]


class Problem(Exception):
    """Ingresso o evidenza incoerente: il manifest non si scrive."""


# ---------------------------------------------------------------- lettura degli shard

class HashingReader(io.RawIOBase):
    """Lettura di un file che ne calcola sha256 e byte: lo shard si legge e si verifica in un solo passaggio."""

    def __init__(self, path: Path):
        self.fh, self.h, self.n = open(path, "rb"), hashlib.sha256(), 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        n = self.fh.readinto(b)
        if n:
            self.h.update(memoryview(b)[:n])
            self.n += n
        return n

    def finish(self) -> tuple[str, int]:
        while block := self.fh.read(1 << 20):
            self.h.update(block)
            self.n += len(block)
        self.fh.close()
        return self.h.hexdigest(), self.n


def iter_samples(tar: tarfile.TarFile):
    """(chiave, {estensione: byte}) nell'ordine del tar, come data.iter_shard."""
    key, files = None, {}
    for member in tar:
        if not member.isfile():
            continue
        stem, _, ext = member.name.rpartition("/")[2].partition(".")
        if stem != key and files:
            yield key, files
            files = {}
        key = stem
        files[ext] = tar.extractfile(member).read()
    if files:
        yield key, files


def png_size(data: bytes) -> tuple[int, int] | None:
    """(larghezza, altezza) dall'intestazione IHDR del PNG, senza decodificare."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def vis_digest(label: str) -> tuple[bytes, int]:
    """Digest della formula visibile (token visibili dell'etichetta effettiva, uniti da spazi, come vis_string di
    unimer_test_overlap.py) e numero dei suoi token come li conta run2 (split della stringa)."""
    vis = " ".join(visible_tokens(effective_label(label)[0]))
    return hashlib.sha256(vis.encode()).digest(), len(vis.split())


def label_categories(label: str, meta: dict, png: bytes) -> tuple[list[str], str | None, str, float | None]:
    """Categorie di unimer-check/4 per un campione (check_shard di scripts/unimer_check.py, senza immagine_vuota):
    (categorie, etichetta rietichettata o None, regole, rapporto per le proporzioni)."""
    effective, rules = effective_label(label)
    vis = visible_tokens(effective)
    cats = list(structure_problems(effective))
    if unrendered_commands(effective):
        cats.append("comando_non_reso")
    if not vis:
        cats.append("formula_vuota")
    if png_size(png) != (meta["width"], meta["height"]):
        cats.append("dimensioni_diverse_dal_json")
    if "Ḋ" in effective or "Ḍ" in effective:
        cats.append("marcatori_non_riparabili")
    cats += [f"rietichettato_{r}" for r in rules]
    area = meta["orig_width"] * meta["orig_height"]
    ratio = math.log(area / len(vis)) if vis and area else None
    return cats, (effective if rules else None), "+".join(sorted(rules)), ratio


def read_shard(task: tuple[str, str, int]) -> dict:
    """Un tar letto e hashato in un passaggio: righe compatte per insieme (train, righe, test). limit > 0 legge solo i
    primi campioni (prova)."""
    kind, path, limit = task
    t0 = time.time()
    raw = HashingReader(Path(path))
    rows, problems = [], []
    with tarfile.open(fileobj=io.BufferedReader(raw, buffer_size=1 << 20), mode="r|") as tar:
        for index, (key, files) in enumerate(iter_samples(tar)):
            if limit and index >= limit:
                break
            if set(files) != {"json", "png"}:
                problems.append(f"{key}: file {sorted(files)}")
                continue
            meta = json.loads(files["json"])
            if meta.get("id") != key:
                problems.append(f"{key}: id del JSON {meta.get('id')!r}")
                continue
            label, png = meta["label"], files["png"]
            base = (key, index, hashlib.sha256(files["json"]).digest(), hashlib.sha256(png).digest(),
                    hashlib.sha256(label.encode()).digest())
            if kind == "train":
                cats, new, rules, ratio = label_categories(label, meta, png)
                vis, n_vis = vis_digest(label)
                rows.append(base + (vis, n_vis, ratio, tuple(cats),
                                    hashlib.sha256(new.encode()).digest() if new is not None else None, rules,
                                    meta["width"], meta["height"], meta["subset"], meta["index"]))
            elif kind == "righe":
                rows_ok = bool(meta.get("rows")) and meta["rows"] == array_rows(label) \
                    and len(meta.get("lines") or []) == len(meta["rows"])
                rows.append(base + (len(meta.get("rows") or []), rows_ok, meta["width"], meta["height"]))
            else:
                vis, n_vis = vis_digest(label)
                rows.append(base + (vis, n_vis, meta["subset"]))
    sha, size = raw.finish() if not limit else (None, None)
    return {"kind": kind, "path": path, "sha256": sha, "bytes": size, "rows": rows, "problems": problems,
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- liste storiche

def read_keys(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def resolve(keys: list[str], index: dict[str, int], what: str) -> list[str]:
    """Problemi di una lista storica: ogni chiave deve corrispondere a esattamente un record (S1 §5)."""
    out, seen = [], Counter(keys)
    for k, n in seen.items():
        if n > 1:
            out.append(f"{what}: chiave {k} ripetuta {n} volte")
        if k not in index:
            out.append(f"{what}: chiave {k} senza record negli shard")
    return out


# ---------------------------------------------------------------- gruppi, ranghi, scrittura

def components(n: int, links) -> list[int]:
    """Rappresentante (indice minimo) della componente di ciascuno di n elementi, dati i legami (i, j)."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in links:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(n)]


def same_key_links(keys) -> list[tuple[int, int]]:
    """Legami fra elementi con la stessa chiave (per esempio lo sha256 dei byte del PNG)."""
    first, links = {}, []
    for i, k in enumerate(keys):
        if k is None:
            continue
        if k in first:
            links.append((first[k], i))
        else:
            first[k] = i
    return links


def f0_rank(seed: int, group: str, role: str | None = None) -> str:
    """Rango di estrazione del campione candidato di F0 (F0_CAMPIONE_CANDIDATO §1): sha256 del JSON canonico
    {campione, seme, gruppo} e, per lo sviluppo, {..., ruolo: "sviluppo"}."""
    obj = {"campione": SAMPLE, "seme": seed, "gruppo": mf.require_digest(group, "gruppo")}
    if role is not None:
        obj["ruolo"] = role
    return mf.sha256_text(mf.canonical(obj))


def worst(*states: str) -> str:
    return max(states, key=SEVERITY.__getitem__)


def rel(path: Path) -> str:
    """Percorso relativo alla radice del progetto; assoluto se sta fuori (prove in una cartella temporanea)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(mf.ROOT))
    except ValueError:
        return str(path)


def write_records_gz(path: Path, lines) -> tuple[str, int]:
    """records.jsonl.gz riproducibile, scritto in streaming: righe in ordine fisso, gzip senza nome né data."""
    refuse_existing(path)
    n = 0
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as gz:
        for line in lines:
            gz.write(line.encode("utf-8") + b"\n")
            n += 1
    return sha256_file(path), n


def rss_mib() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(max(own, kids) / 1024, 1)


# ---------------------------------------------------------------- costruzione

def shard_plan(shards: Path) -> dict:
    """Manifest dello sharding (job 29729), con la revisione e i tar di ciascun insieme."""
    main = json.loads((shards / "manifest.json").read_text())
    lines = json.loads((shards / "lines" / "manifest.json").read_text())
    test = json.loads((shards / "test" / "manifest.json").read_text())
    m = REVISION.match(main.get("source", ""))
    if not m or lines.get("source") != main["source"] or test.get("source") != main["source"]:
        raise Problem(f"sorgente degli shard non riconosciuta o diversa fra gli insiemi: {main.get('source')!r}")
    sets = {"train": [(shards / s["name"], s) for s in main["shards"]],
            "righe": [(shards / "lines" / s["name"], s) for s in lines["shards"]],
            "test": [(shards / "test" / s["name"], s) for s in test["shards"]]}
    return {"revision": m.group(1), "main": main, "sets": sets,
            "files": {"manifest": file_entry(shards / "manifest.json"),
                      "lines_manifest": file_entry(shards / "lines" / "manifest.json"),
                      "test_manifest": file_entry(shards / "test" / "manifest.json")}}


def snapshot_identity(main: dict) -> dict:
    """Lo snapshot di Hugging Face: gli zip dichiarati dal manifest dello sharding esistono e il loro blob ha il nome
    dello sha256 atteso (i blob LFS si chiamano col proprio sha256; hash verificati nel job 29554)."""
    out = {}
    for field, expect in (("zip", "zip_sha256_expected"), ("test_zip", "test_zip_sha256_expected")):
        path = Path(main[field])
        blob = path.resolve()
        out[field] = {"percorso": str(path), "esiste": path.is_file(), "byte": path.stat().st_size if path.is_file() else None,
                      "blob": blob.name, "sha256_atteso": main[expect], "blob_coincide": blob.name == main[expect]}
    return out


def build(args, plan: dict) -> dict:
    """Righe del manifest e metadati, dopo tutte le riconciliazioni; Problem se un ingresso è incoerente."""
    problems: list[str] = []
    tasks = [(kind, str(p), 0) for kind in ("train", "righe", "test") for p, _ in plan["sets"][kind]]
    results: dict[str, dict] = {}
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(read_shard, tasks):
            results[res["path"]] = res
            print(f"{Path(res['path']).name}: {len(res['rows'])} campioni in {res['seconds']} s", flush=True)
    shards_meta = []
    for kind in ("train", "righe", "test"):
        for p, s in plan["sets"][kind]:
            res = results[str(p)]
            problems += [f"{p.name}: {x}" for x in res["problems"]]
            if res["sha256"] != s["sha256"] or res["bytes"] != s["bytes"] or len(res["rows"]) != s["samples"]:
                problems.append(f"{p.name}: sha256, byte o campioni diversi dal manifest dello sharding")
            shards_meta.append({"insieme": kind, "nome": p.name, "percorso": rel(p),
                                "sha256": res["sha256"], "bytes": res["bytes"], "record": len(res["rows"])})
    if problems:
        raise Problem("; ".join(problems[:10]))

    # ---- indici: formule (train), righe ritagliate, test
    train, train_shard, righe, righe_shard, test = [], [], [], [], []
    for kind, store, where in (("train", train, train_shard), ("righe", righe, righe_shard), ("test", test, None)):
        for p, _ in plan["sets"][kind]:
            path = rel(p)
            for row in results[str(p)]["rows"]:
                store.append(row)
                if where is not None:
                    where.append(path)
    index = {}
    for i, row in enumerate(train):
        if row[0] in index:
            raise Problem(f"{row[0]}: in due shard di train ({train_shard[index[row[0]]]}, {train_shard[i]})")
        index[row[0]] = i
    parent_of = []
    for j, row in enumerate(righe):
        i = index.get(row[0])
        if i is None:
            raise Problem(f"riga ritagliata {row[0]} senza la formula negli shard di train")
        if row[4] != train[i][4]:
            raise Problem(f"riga ritagliata {row[0]}: etichetta diversa da quella della formula")
        if not row[6]:
            raise Problem(f"riga ritagliata {row[0]}: righe o fasce diverse da f1c.array_rows dell'etichetta")
        parent_of.append(i)
    if len(set(parent_of)) != len(parent_of):
        raise Problem("una formula ha due righe ritagliate")

    # ---- run4 ricalcolato: conflitti (stessi byte, formule visibili diverse), proporzioni, stati
    check = Path(args.check)
    report4 = json.loads((check / "report.json").read_text())
    if report4.get("schema") != "unimer-check/4":
        raise Problem(f"run4: schema {report4.get('schema')!r}")
    train_sha = {s["sha256"] for s in shards_meta if s["insieme"] == "train"}
    if {s["sha256"] for s in report4.get("shards", [])} != train_sha or report4.get("samples") != len(train):
        raise Problem("run4: shard o campioni diversi da quelli letti")
    cats = [list(row[8]) for row in train]
    by_png = defaultdict(list)
    for i, row in enumerate(train):
        by_png[row[3]].append(i)
    conflict_groups = [m for m in by_png.values() if len({train[i][5] for i in m}) > 1]
    for m in conflict_groups:
        for i in m:
            cats[i].append("conflitto_irrisolto")  # run4: il disegno non si usa (render_used false), nessun vincitore
    if report4.get("render_used"):
        raise Problem("run4 ha usato il disegno per i conflitti: la regola ricalcolata qui non vale")
    ratios = [row[7] for row in train if row[7] is not None]
    med = statistics.median(ratios)
    mad = statistics.median(abs(x - med) for x in ratios) * 1.4826
    for i, row in enumerate(train):
        if row[7] is not None and mad and abs(row[7] - med) / mad > Z:
            cats[i].append("proporzioni_fuori_norma")
    ours = {}
    for i, row in enumerate(train):
        cs = sorted(set(cats[i]))
        if cs:
            state = "excluded" if any(c in CERTAIN for c in cs) else "uncertain" if any(c in UNCERTAIN for c in cs) \
                else "accepted"
            ours[row[0]] = (state, cs)
    files4 = {name: check / name for name in ("quality_excluded.txt", "uncertain.txt", "relabel.jsonl",
                                              "reasons.jsonl", "report.json")}
    excluded4, uncertain4 = read_keys(files4["quality_excluded.txt"]), read_keys(files4["uncertain.txt"])
    relabel4, reasons4 = read_jsonl(files4["relabel.jsonl"]), read_jsonl(files4["reasons.jsonl"])
    problems += resolve(excluded4, index, "run4 quality_excluded")
    problems += resolve(uncertain4, index, "run4 uncertain")
    problems += resolve([r["key"] for r in relabel4], index, "run4 relabel")
    problems += resolve([r["key"] for r in reasons4], index, "run4 reasons")
    if set(excluded4) & set(uncertain4):
        problems.append(f"run4: chiavi sia escluse sia incerte: {sorted(set(excluded4) & set(uncertain4))[:3]}")
    reasons = {r["key"]: r for r in reasons4}
    for r in reasons4:
        if r["key"] in index and hashlib.sha256(r["label"].encode()).digest() != train[index[r["key"]]][4]:
            problems.append(f"run4 reasons: {r['key']} con un'etichetta diversa da quella dello shard")
    if {k for k, r in reasons.items() if r["state"] == "excluded"} != set(excluded4) \
            or {k for k, r in reasons.items() if r["state"] == "uncertain"} != set(uncertain4):
        problems.append("run4: stati di reasons.jsonl diversi dalle liste quality_excluded e uncertain")
    relabel = {}
    for r in relabel4:
        i = index.get(r["key"])
        if i is None:
            continue
        if hashlib.sha256(r["label"].encode()).digest() != train[i][4]:
            problems.append(f"run4 relabel: {r['key']} con un'etichetta attesa diversa da quella dello shard")
        relabel[r["key"]] = r
    reproduced = {"stati": {}, "categorie": {}}
    their = {k: (r["state"], sorted(r["categories"])) for k, r in reasons.items()}
    ours_cmp = {k: (s, [c for c in cs if c not in NOT_REPRODUCED]) for k, (s, cs) in ours.items()}
    their_cmp = {k: (s, [c for c in cs if c not in NOT_REPRODUCED]) for k, (s, cs) in their.items()}
    differ = sorted(k for k in set(ours_cmp) | set(their_cmp) if ours_cmp.get(k) != their_cmp.get(k))
    our_relabel = {train[i][0]: (train[i][9], train[i][10]) for i in range(len(train))
                   if train[i][9] is not None and ours.get(train[i][0], ("",))[0] == "accepted"}
    their_relabel = {k: (hashlib.sha256(r["new_label"].encode()).digest(), r["rule"]) for k, r in relabel.items()}
    reproduced.update({
        "reasons_uguali": not differ, "reasons_diversi": len(differ), "esempi_diversi": differ[:10],
        "relabel_uguali": our_relabel == their_relabel,
        "relabel_diversi": len(set(our_relabel.items()) ^ set(their_relabel.items())),
        "gruppi_in_conflitto": len(conflict_groups), "campioni_in_conflitto": sum(len(m) for m in conflict_groups),
        "duplicati_stessa_immagine_stessa_formula": sum(len(m) - 1 for m in by_png.values()
                                                        if len({train[i][5] for i in m}) == 1),
        "non_ricalcolato": sorted(NOT_REPRODUCED)})
    reproduced["stati"] = dict(Counter(s for s, _ in ours.values()))
    reproduced["categorie"] = dict(Counter(c for _, cs in ours.values() for c in cs))
    documented = {"quality_excluded": report4.get("quality_excluded"), "uncertain": report4.get("uncertain"),
                  "relabeled": report4.get("relabeled"), "conflict_groups": report4.get("conflict_groups"),
                  "conflict_samples": report4.get("conflict_samples")}
    counted = {"quality_excluded": len(excluded4), "uncertain": len(uncertain4), "relabeled": len(relabel4),
               "conflict_groups": len(conflict_groups), "conflict_samples": reproduced["campioni_in_conflitto"]}
    if documented != counted:
        problems.append(f"run4: conteggi del report {documented} diversi da quelli dei file e del ricalcolo {counted}")
    if differ or our_relabel != their_relabel:
        problems.append(f"run4 non riprodotto dalle regole correnti: {len(differ)} motivi e "
                        f"{reproduced['relabel_diversi']} rietichettature diverse, es. {differ[:3]}")

    # ---- run2 ricalcolato: formule e immagini del training in UniMER-Test
    overlap = Path(args.overlap)
    report2 = json.loads((overlap / "report.json").read_text())
    if report2.get("schema") != "unimer-test-overlap/2" or report2.get("long") != LONG:
        raise Problem(f"run2: schema {report2.get('schema')!r} o soglia {report2.get('long')!r}")
    test_sha = {s["sha256"] for s in shards_meta if s["insieme"] == "test"}
    if {s["sha256"] for s in report2.get("test_shards", [])} != test_sha \
            or {s["sha256"] for s in report2.get("train_shards", [])} != train_sha:
        raise Problem("run2: shard diversi da quelli letti")
    test_vis, test_png = defaultdict(list), defaultdict(list)
    for row in test:
        test_vis[row[5]].append(row[0])
        test_png[row[3]].append(row[0])
    match_formula = {i for i, row in enumerate(train) if row[5] in test_vis}
    match_image = {i for i, row in enumerate(train) if row[3] in test_png}
    long_formula = {i for i in match_formula if train[i][6] > LONG}
    matches2 = read_keys(overlap / "train_matches_test.txt")
    contamination2 = read_keys(overlap / "train_contamination_long.txt")
    problems += resolve(matches2, index, "run2 train_matches_test")
    problems += resolve(contamination2, index, "run2 train_contamination_long")
    ours_matches = {train[i][0] for i in match_formula | match_image}
    ours_contamination = {train[i][0] for i in match_image | long_formula}
    if ours_matches != set(matches2) or ours_contamination != set(contamination2) \
            or report2.get("train_samples_matching") != len(matches2) \
            or report2.get("train_contamination_long") != len(contamination2):
        problems.append(f"run2 non riprodotto: corrispondenze {len(ours_matches)} contro {len(matches2)}, "
                        f"contaminazioni {len(ours_contamination)} contro {len(contamination2)}")
    reproduced["run2"] = {"corrispondenze": len(ours_matches), "per_formula": len(match_formula),
                          "per_immagine": len(match_image), "formula_lunga": len(long_formula),
                          "contaminazioni": len(ours_contamination),
                          "contaminazioni_solo_formula_lunga": len(long_formula - match_image),
                          "uguali_ai_file": ours_matches == set(matches2) and ours_contamination == set(contamination2)}

    # ---- duplicati della vecchia validazione (controllo incrociato dei gruppi per immagine)
    dup_rows = read_jsonl(Path(args.duplicates) / "val_with_duplicates.jsonl")
    problems += resolve([r["val_key"] for r in dup_rows], index, "split_duplicates val_with_duplicates")
    if problems:
        raise Problem("; ".join(problems[:10]))

    # ---- gruppi: stessi byte del PNG; ogni riga ritagliata con la sua formula
    n = len(train) + len(righe)
    links = same_key_links([row[3] for row in train] + [row[3] for row in righe])
    links += [(parent_of[j], len(train) + j) for j in range(len(righe))]
    rep = components(n, links)
    dup_check = [r["val_key"] for r in dup_rows if r.get("immagine")]
    members = defaultdict(list)
    for i in range(len(train)):
        members[rep[i]].append(i)
    dup_bad = [k for k in dup_check if len(members[rep[index[k]]]) < 2]
    if dup_bad:
        raise Problem(f"split_duplicates: {len(dup_bad)} chiavi con immagine ripetuta in un gruppo da solo, es. {dup_bad[:3]}")
    digests = mf.Digests()
    revision = plan["revision"]
    rid = [mf.record_id(digests, SOURCE, revision, row[0]) for row in train]
    index_name = {"nome": "legami esatti di UniMER-1M: stessi byte del PNG negli shard densi; ogni riga ritagliata nel "
                          "gruppo della sua formula", "sharding_manifest_sha256": plan["files"]["manifest"]["sha256"]}
    gid_of_rep = {r: mf.group_id(digests, index_name, sorted({rid[i] for i in ms})) for r, ms in members.items()}

    # ---- stati
    state, motive, evid = ["ammesso"] * len(train), [[] for _ in train], [[] for _ in train]
    for k in excluded4:
        i = index[k]
        state[i] = worst(state[i], "escluso")
    for k in uncertain4:
        i = index[k]
        state[i] = worst(state[i], "quarantena")
    for k, r in reasons.items():
        i = index[k]
        motive[i] += [f"unimer_check/4: {c}" for c in r["categories"]]
        evid[i].append({"lista": "unimer_check_run4", "esito": r["state"], "categorie": r["categories"]})
    for i in range(len(train)):
        if train[i][0] not in reasons:
            evid[i].append({"lista": "unimer_check_run4", "esito": "nessuna_segnalazione"})
    group_state, group_motive = {}, defaultdict(list)
    for i in match_image:
        g = rep[i]
        group_state[g] = worst(group_state.get(g, "ammesso"), "escluso")
        group_motive[g].append(f"stessa immagine di UniMER-Test ({', '.join(sorted(test_png[train[i][3]])[:3])})")
    for i in long_formula - match_image:
        g = rep[i]
        group_state[g] = worst(group_state.get(g, "ammesso"), "quarantena")
        group_motive[g].append(f"formula di UniMER-Test con più di {LONG} token visibili, immagine diversa "
                               f"({', '.join(sorted(test_vis[train[i][5]])[:3])}): da riesaminare")
    for i in range(len(train)):
        esito = ("immagine_del_test" if i in match_image else "formula_lunga_del_test" if i in long_formula
                 else "formula_breve_del_test" if i in match_formula else "nessuna_corrispondenza")
        evid[i].append({"lista": "unimer_overlap_run2", "esito": esito})
        g = rep[i]
        if g in group_state:
            state[i] = worst(state[i], group_state[g])
            motive[i] += sorted(set(group_motive[g]))

    # ---- split e ranghi
    info = {}
    for r, ms in members.items():
        info[gid_of_rep[r]] = {"famiglia": SOURCE, "valutazione": False,
                               "ammessi": sum(state[i] == "ammesso" for i in ms)}
    split_of = mf.assign_splits(info, args.dev_fraction, args.seed)
    rank = {g: f0_rank(args.seed, g) for g in info}
    rank_dev = {g: f0_rank(args.seed, g, "sviluppo") for g in info if split_of[g] == "dev"}
    return {"train": train, "train_shard": train_shard, "righe": righe, "righe_shard": righe_shard, "test": test,
            "parent_of": parent_of, "rep": rep, "gid_of_rep": gid_of_rep, "rid": rid, "state": state,
            "motive": motive, "evid": evid, "relabel": relabel, "split_of": split_of, "rank": rank,
            "rank_dev": rank_dev, "info": info, "index_name": index_name, "shards_meta": shards_meta,
            "reproduced": reproduced, "files4": files4, "match_formula": match_formula, "match_image": match_image,
            "long_formula": long_formula, "conflict_groups": conflict_groups, "dup_checked": len(dup_check),
            "report4": report4, "report2": report2, "revision": revision}


def records(b: dict):
    """Righe del manifest, in ordine fisso: formule negli shard e nell'ordine dei tar, poi le righe ritagliate."""
    train, righe = b["train"], b["righe"]
    for i, row in enumerate(train):
        g = b["gid_of_rep"][b["rep"][i]]
        key = row[0]
        rec = {"record_id": b["rid"][i], "sorgente": SOURCE, "revisione": b["revision"], "record": key,
               "insieme": "train", "shard": b["train_shard"][i], "indice": row[1], "compiti": TASKS["train"],
               "gruppo": g, "split": b["split_of"][g], "stato": b["state"][i], "motivo": b["motive"][i], "gold": None,
               "immagini": [{"sha256": row[3].hex(), "ordine": 0, "griglia": [row[12] // 16, row[11] // 16]}],
               "json_sha256": row[2].hex(), "etichetta_sha256": row[4].hex(), "token_visibili": row[6],
               "evidenze": b["evid"][i], "trasformazione": None, "genitori": [],
               "rango_f0": b["rank"][g], "rango_f0_sviluppo": b["rank_dev"].get(g)}
        r4 = b["relabel"].get(key)
        if r4 is not None:
            rec["trasformazione"] = {"nome": "rietichettatura", "regola": r4["rule"], "versione": "unimer-check/4",
                                     "lista": "unimer_check_run4",
                                     "etichetta_sha256": hashlib.sha256(r4["new_label"].encode()).hexdigest()}
            rec["genitori"] = [{"record_id": b["rid"][i], "insieme": "train", "json_sha256": row[2].hex(),
                                "etichetta_sha256": row[4].hex()}]
        yield rec
    for j, row in enumerate(righe):
        i = b["parent_of"][j]
        parent = train[i]
        g = b["gid_of_rep"][b["rep"][i]]
        st, mot = b["state"][i], [f"dalla formula: {m}" for m in b["motive"][i]]
        if row[0] in b["relabel"]:
            st = worst(st, "quarantena")
            mot.append("formula rietichettata: le righe vengono dall'etichetta originale")
        yield {"record_id": b["rid"][i], "sorgente": SOURCE, "revisione": b["revision"], "record": row[0],
               "insieme": "righe", "shard": b["righe_shard"][j], "indice": row[1], "compiti": TASKS["righe"],
               "gruppo": g, "split": b["split_of"][g], "stato": st, "motivo": mot, "gold": None,
               "immagini": [{"sha256": row[3].hex(), "ordine": 0, "griglia": [row[8] // 16, row[7] // 16]}],
               "json_sha256": row[2].hex(), "etichetta_sha256": row[4].hex(), "righe": row[5], "evidenze": [],
               "trasformazione": {"nome": "righe", "regola": "f1c.array_rows e f1c.split_lines (scripts/shard_unimer.py "
                                                             "--lines, job 29729)", "righe": row[5],
                                  "stessa_immagine_del_genitore": row[3] == parent[3]},
               "genitori": [{"record_id": b["rid"][i], "insieme": "train", "shard": b["train_shard"][i],
                             "indice": parent[1], "json_sha256": parent[2].hex(), "png_sha256": parent[3].hex()}],
               "rango_f0": b["rank"][g], "rango_f0_sviluppo": b["rank_dev"].get(g)}


def summarize(recs: list[dict]) -> dict:
    """Conteggi per insieme, split e stato; per compito e stato; per motivo; gruppi per split."""
    c = {"per_insieme_split_stato": Counter(), "per_compito_split_stato": Counter(), "per_motivo": Counter(),
         "gruppi_per_split": Counter(), "trasformazioni": Counter()}
    groups = {}
    for r in recs:
        c["per_insieme_split_stato"][f"{r['insieme']}/{r['split']}/{r['stato']}"] += 1
        for t in r["compiti"]:
            c["per_compito_split_stato"][f"{t}/{r['split']}/{r['stato']}"] += 1
        for m in r["motivo"]:
            c["per_motivo"][re.sub(r" \(.*\)", "", m).split(": da riesaminare")[0]] += 1
        if r["trasformazione"]:
            c["trasformazioni"][r["trasformazione"]["nome"]] += 1
        groups[r["gruppo"]] = r["split"]
    for s in groups.values():
        c["gruppi_per_split"][s] += 1
    return {k: dict(sorted(v.items())) for k, v in c.items()}


# ---------------------------------------------------------------- verifica dopo la scrittura

def verify(folder: Path, check: Path, overlap: Path, seed_check: bool = True) -> dict:
    """Controlli rileggendo il manifest scritto (manifest.load) e le liste storiche, senza gli shard: identità uniche,
    gruppi e split ricalcolati, trasformazioni e genitori, evidenze riconciliate, conteggi, ranghi, file derivati."""
    checks = []

    def ok(name: str, cond: bool, detail=None) -> None:
        checks.append({"controllo": name, "esito": bool(cond), "dettaglio": detail})

    m = mf.load(folder)
    ok("manifest letto e verificato (sha256 dei record, record_id ricalcolati, un solo split per gruppo, conteggi per "
       "shard)", True)
    recs = [r for rows in m.by_shard.values() for r in rows.values()]
    meta = m.meta
    base = [r for r in recs if r["insieme"] == "train"]
    ok("identità uniche: (record_id, insieme) e chiave per insieme",
       len({(r["record_id"], r["insieme"]) for r in recs}) == len(recs)
       and len({r["record"] for r in base}) == len(base))
    by_key = {r["record"]: r for r in base}
    # gruppi ricalcolati dai legami salvati
    items = base + [r for r in recs if r["insieme"] == "righe"]
    base_pos = {r["record_id"]: n for n, r in enumerate(items) if r["insieme"] == "train"}
    links = same_key_links([r["immagini"][0]["sha256"] for r in items])
    links += [(base_pos[g["record_id"]], n) for n, r in enumerate(items) if r["insieme"] == "righe"
              for g in r["genitori"] if g.get("record_id") in base_pos]
    rep = components(len(items), links)
    ours = defaultdict(set)
    for n, r in enumerate(items):
        ours[rep[n]].add(n)
    theirs = defaultdict(set)
    for n, r in enumerate(items):
        theirs[r["gruppo"]].add(n)
    ok("gruppi ricalcolati dai byte del PNG e dai genitori: stessa partizione",
       sorted(map(sorted, ours.values())) == sorted(map(sorted, theirs.values())), len(theirs))
    digests = mf.Digests()
    bad_gid = 0
    for members in theirs.values():
        rids = sorted({items[n]["record_id"] for n in members})
        g = items[next(iter(members))]["gruppo"]
        bad_gid += mf.group_id(digests, meta["indice"], rids) != g
    ok("group_id ricalcolati dai membri", bad_gid == 0, bad_gid)
    split = {}
    for r in recs:
        split.setdefault(r["gruppo"], set()).add(r["split"])
    ok("nessun gruppo in due split (in particolare fra train e dev)", all(len(s) == 1 for s in split.values()))
    info = {}
    for r in base:
        i = info.setdefault(r["gruppo"], {"famiglia": SOURCE, "valutazione": False, "ammessi": 0})
        i["ammessi"] += r["stato"] == "ammesso"
    rule = meta["split"]
    again = mf.assign_splits(info, rule["quota_dev"], rule["seme"])
    ok("split ricalcolato con quota e seme del manifest: uguale a quello scritto",
       all(again[g] == next(iter(split[g])) for g in info))
    if seed_check:
        other = mf.assign_splits(info, rule["quota_dev"], rule["seme"] + 1)
        moved = sum(other[g] != again[g] for g in info)
        ok("con un altro seme il dev cambia (la scelta dipende dal seme)", moved > 0, moved)
    admitted = sum(i["ammessi"] for i in info.values())
    dev = sum(i["ammessi"] for g, i in info.items() if again[g] == "dev")
    ok("dev: almeno l'obiettivo di formule ammesse, superato al più dall'ultimo gruppo",
       round(rule["quota_dev"] * admitted) <= dev <= round(rule["quota_dev"] * admitted)
       + max((i["ammessi"] for i in info.values()), default=0), {"ammesse": admitted, "dev": dev})
    # trasformazioni e genitori
    bad_righe, bad_relabel = [], []
    relabel = {r["key"]: r for r in read_jsonl(check / "relabel.jsonl")}
    for r in recs:
        t = r["trasformazione"]
        if r["insieme"] == "righe":
            p = by_key.get(r["record"])
            g = r["genitori"][0] if r["genitori"] else {}
            if p is None or t is None or t["nome"] != "righe" or g.get("record_id") != p["record_id"] \
                    or g.get("json_sha256") != p["json_sha256"] or g.get("png_sha256") != p["immagini"][0]["sha256"] \
                    or g.get("shard") != p["shard"] or g.get("indice") != p["indice"] or r["gruppo"] != p["gruppo"] \
                    or r["split"] != p["split"] or SEVERITY[r["stato"]] < SEVERITY[p["stato"]] \
                    or r["etichetta_sha256"] != p["etichetta_sha256"]:
                bad_righe.append(r["record"])
        elif t is not None:
            src = relabel.get(r["record"])
            g = r["genitori"][0] if r["genitori"] else {}
            if t["nome"] != "rietichettatura" or src is None \
                    or t["etichetta_sha256"] != hashlib.sha256(src["new_label"].encode()).hexdigest() \
                    or g.get("etichetta_sha256") != r["etichetta_sha256"] \
                    or hashlib.sha256(src["label"].encode()).hexdigest() != r["etichetta_sha256"]:
                bad_relabel.append(r["record"])
    ok("righe ritagliate: genitore presente con gli stessi hash, stesso gruppo e split, stato non meno grave",
       not bad_righe, bad_righe[:5])
    ok("rietichettature: genitore con l'etichetta originale e nuova etichetta uguali a relabel.jsonl",
       not bad_relabel, bad_relabel[:5])
    got_relabel = {r["record"] for r in base if r["trasformazione"]}
    ok("rietichettature: tutte e sole quelle di relabel.jsonl", got_relabel == set(relabel),
       {"attese": len(relabel), "trovate": len(got_relabel)})
    righe_relabel = [r["record"] for r in recs if r["insieme"] == "righe" and r["record"] in relabel
                     and r["stato"] == "ammesso"]
    ok("righe ritagliate di formule rietichettate non ammesse", not righe_relabel, righe_relabel[:5])
    # evidenze riconciliate con le liste storiche
    lists = {"quality_excluded": read_keys(check / "quality_excluded.txt"), "uncertain": read_keys(check / "uncertain.txt"),
             "train_contamination_long": read_keys(overlap / "train_contamination_long.txt"),
             "train_matches_test": read_keys(overlap / "train_matches_test.txt")}
    unresolved = {k: sum(1 for x in v if x not in by_key) for k, v in lists.items()}
    ok("liste storiche: ogni chiave risolta in un solo record", not any(unresolved.values()), unresolved)
    ex = {r["record"] for r in base if any(e["lista"] == "unimer_check_run4" and e["esito"] == "excluded"
                                          for e in r["evidenze"])}
    un = {r["record"] for r in base if any(e["lista"] == "unimer_check_run4" and e["esito"] == "uncertain"
                                          for e in r["evidenze"])}
    ok("evidenze di run4 nei record uguali alle liste", ex == set(lists["quality_excluded"])
       and un == set(lists["uncertain"]))
    ok("errori certi esclusi, incerti in quarantena o esclusi", all(by_key[k]["stato"] == "escluso" for k in ex)
       and all(by_key[k]["stato"] != "ammesso" for k in un))
    esiti = Counter(e["esito"] for r in base for e in r["evidenze"] if e["lista"] == "unimer_overlap_run2")
    cont = {r["record"] for r in base for e in r["evidenze"] if e["lista"] == "unimer_overlap_run2"
            and e["esito"] in ("immagine_del_test", "formula_lunga_del_test")}
    matched = {r["record"] for r in base for e in r["evidenze"] if e["lista"] == "unimer_overlap_run2"
               and e["esito"] != "nessuna_corrispondenza"}
    ok("evidenze di run2 nei record uguali alle liste", cont == set(lists["train_contamination_long"])
       and matched == set(lists["train_matches_test"]), dict(esiti))
    bad_cont = [k for k in cont if by_key[k]["stato"] == "ammesso"]
    ok("contaminazioni con UniMER-Test non ammesse (immagine: escluse; formula lunga: almeno in quarantena)",
       not bad_cont, bad_cont[:5])
    group_cont = defaultdict(set)
    for r in base:
        if r["record"] in cont:
            group_cont[r["gruppo"]].add(r["record"])
    spread = [r["record"] for r in base if r["gruppo"] in group_cont and r["stato"] == "ammesso"]
    ok("contaminazioni estese a tutto il gruppo", not spread, spread[:5])
    # conteggi, ranghi, file derivati
    ok("conteggi per insieme, split, stato, compito e motivo uguali a manifest.json",
       summarize(recs) == meta["conteggi"])
    bad_rank = [r["record"] for r in recs if r["rango_f0"] != f0_rank(rule["seme"], r["gruppo"])
                or r["rango_f0_sviluppo"] != (f0_rank(rule["seme"], r["gruppo"], "sviluppo") if r["split"] == "dev"
                                              else None)]
    ok("ranghi del campione candidato ricalcolati", not bad_rank, bad_rank[:5])
    derived_ok = []
    for name, entry in meta["derivati"].items():
        text = (folder / name).read_text(encoding="utf-8")
        derived_ok.append(mf.sha256_text(text) == entry["sha256"] and text.count("\n") == entry["righe"])
    ok("file derivati (keys/) con gli sha256 registrati", all(derived_ok))
    return {"manifest": str(folder), "manifest_sha256": m.manifest_sha256, "controlli": checks,
            "superati": sum(c["esito"] for c in checks), "totale": len(checks),
            "esito": all(c["esito"] for c in checks)}


# ---------------------------------------------------------------- job

def probe(args, plan: dict) -> None:
    """Un tar per insieme, per misurare il carico prima del lavoro completo."""
    tasks = [(kind, str(plan["sets"][kind][0][0]), 0) for kind in ("train", "righe", "test")]
    t0 = time.time()
    with Pool(min(args.workers, len(tasks))) as pool:
        out = pool.map(read_shard, tasks)
    per = {}
    for res, kind in zip(out, ("train", "righe", "test")):
        s = plan["sets"][kind][0][1]
        rate = len(res["rows"]) / max(res["seconds"], 0.1)
        per[kind] = {"tar": Path(res["path"]).name, "campioni": len(res["rows"]), "secondi": res["seconds"],
                     "campioni_al_secondo": round(rate, 1), "sha256_coincide": res["sha256"] == s["sha256"],
                     "byte_letti": res["bytes"], "problemi": res["problems"][:5]}
    train = out[0]["rows"]
    sample = [{"record_id": "0" * 64, "sorgente": SOURCE, "revisione": plan["revision"], "record": row[0],
               "insieme": "train", "shard": rel(plan["sets"]["train"][0][0]),
               "indice": row[1], "compiti": TASKS["train"], "gruppo": "0" * 64, "split": "train", "stato": "ammesso",
               "motivo": [], "gold": None, "immagini": [{"sha256": row[3].hex(), "ordine": 0,
                                                       "griglia": [row[12] // 16, row[11] // 16]}],
               "json_sha256": row[2].hex(), "etichetta_sha256": row[4].hex(), "token_visibili": row[6],
               "evidenze": [{"lista": "unimer_check_run4", "esito": "nessuna_segnalazione"},
                            {"lista": "unimer_overlap_run2", "esito": "nessuna_corrispondenza"}],
               "trasformazione": None, "genitori": [], "rango_f0": "0" * 64, "rango_f0_sviluppo": None}
              for row in train[:20000]]
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for r in sample)
    packed = gzip.compress(text.encode(), compresslevel=6, mtime=0)
    totals = {kind: sum(s["samples"] for _, s in plan["sets"][kind]) for kind in plan["sets"]}
    seconds = sum(per[k]["secondi"] * totals[k] / max(per[k]["campioni"], 1) for k in per) / max(args.workers, 1)
    report = {"schema": "unimer-manifest-probe/1", "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "per_insieme": per, "campioni_totali": totals, "secondi_della_prova": round(time.time() - t0, 1),
              "rss_max_mib": rss_mib(), "workers": args.workers,
              "stima_lettura_completa_secondi": round(seconds, 1),
              "byte_per_riga_gz": round(len(packed) / len(sample), 1), "byte_per_riga_json": round(len(text) / len(sample), 1),
              "stima_records_jsonl_gz_mib": round(len(packed) / len(sample) * (totals["train"] + totals["righe"]) / 2**20, 1),
              "nota": "stima della lettura: secondi per campione di ogni insieme per i campioni totali, divisi per i "
                      "worker; memoria del processo principale per tutte le righe da misurare nel lavoro completo"}
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "probe.json", report)
    print(json.dumps(report, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--check", required=True, type=Path)
    ap.add_argument("--overlap", required=True, type=Path)
    ap.add_argument("--duplicates", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--name", default="unimer-candidato")
    ap.add_argument("--version", default="1")
    ap.add_argument("--dev-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("unimer_manifest.py va eseguito in un job SLURM")
    try:
        run(args)
    except Problem as exc:
        sys.exit(str(exc))


def run(args) -> dict | None:
    """Prova (--probe) oppure costruzione, scrittura e verifica; Problem a ogni incoerenza, prima di scrivere il
    manifest o, dopo, se la verifica fallisce. Restituisce il riepilogo."""
    if args.report.exists() and any(args.report.iterdir()):
        raise Problem(f"{args.report} non è vuota")
    t0 = time.time()
    try:
        plan = shard_plan(args.shards)
    except Problem as exc:
        raise Problem(f"ingressi incoerenti: {exc}") from None
    if args.probe:
        probe(args, plan)
        return None
    if args.out is None or args.summary is None:
        raise Problem("servono --out e --summary")
    if args.out.exists() and any(args.out.iterdir()):
        raise Problem(f"{args.out} non è vuota: una versione del manifest si scrive una volta sola")
    snapshot = snapshot_identity(plan["main"])
    if not all(s["esiste"] and s["blob_coincide"] for s in snapshot.values()):
        raise Problem(f"snapshot di Hugging Face diverso da quello dello sharding: {snapshot}")
    try:
        b = build(args, plan)
    except Problem as exc:
        raise Problem(f"controlli prima della scrittura falliti: {exc}") from None
    t_build = time.time() - t0

    # ---- scrittura
    args.out.mkdir(parents=True, exist_ok=True)
    recs = []

    def lines():
        for rec in records(b):
            recs.append({k: rec[k] for k in ("record", "insieme", "split", "stato", "gruppo", "compiti", "motivo",
                                             "trasformazione")})
            yield json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    data_path = args.out / "records.jsonl.gz"
    data_sha, n_lines = write_records_gz(data_path, lines())
    keys = {"train.txt": [r["record"] for r in recs if r["insieme"] == "train" and r["split"] == "train"
                          and r["stato"] == "ammesso"],
            "dev.txt": [r["record"] for r in recs if r["insieme"] == "train" and r["split"] == "dev"
                        and r["stato"] == "ammesso"],
            "non_ammessi.txt": [r["record"] for r in recs if r["insieme"] == "train" and r["stato"] != "ammesso"],
            "righe_train.txt": [r["record"] for r in recs if r["insieme"] == "righe" and r["split"] == "train"
                                and r["stato"] == "ammesso"],
            "righe_dev.txt": [r["record"] for r in recs if r["insieme"] == "righe" and r["split"] == "dev"
                              and r["stato"] == "ammesso"],
            "righe_non_ammesse.txt": [r["record"] for r in recs if r["insieme"] == "righe" and r["stato"] != "ammesso"]}
    (args.out / "keys").mkdir()
    derived = {}
    for name, ks in keys.items():
        text = "".join(k + "\n" for k in sorted(ks))
        (args.out / "keys" / name).write_text(text, encoding="utf-8")
        derived[f"keys/{name}"] = {**file_entry(args.out / "keys" / name, mf.sha256_text(text)), "righe": len(ks)}
    counts = summarize(recs)
    admitted = sum(i["ammessi"] for i in b["info"].values())
    dev = sum(i["ammessi"] for g, i in b["info"].items() if b["split_of"][g] == "dev")
    files4 = b["files4"]
    meta = {
        "schema": mf.SCHEMA, "nome": args.name, "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"),
        "codice": code_identity(), "sorgente": SOURCE, "revisione": b["revision"],
        "identita": {"repository": "wanderkid/UniMER_Dataset", "revisione": b["revision"], "snapshot": snapshot,
                     "catalogo": "downloads/catalog_status.tsv (hash dello snapshot verificati nei job 29429 e 29554)"},
        "indice": b["index_name"],
        "ingressi": {"sharding": plan["files"], "sharding_job": plan["main"].get("job_id"),
                     "parametri_sharding": {k: plan["main"].get(k) for k in ("scale", "max_patches",
                                                                            "tokens_per_symbol", "dense_cap")}},
        "shards": [s for s in b["shards_meta"] if s["insieme"] != "test"],
        "shards_di_valutazione_letti": [s for s in b["shards_meta"] if s["insieme"] == "test"],
        "liste_importate": {
            "unimer_check_run4": {
                "controllo": "scripts/unimer_check.py", "versione": "unimer-check/4", "job": b["report4"]["job_id"],
                "file": {n: file_entry(p) for n, p in files4.items()},
                "stato_proposto": {"quality_excluded.txt": "escluso (errore certo trovato dal codice)",
                                   "uncertain.txt": "quarantena (incerti: non diventano scarti né validi)",
                                   "relabel.jsonl": "trasformazione tracciata, stato invariato"},
                "applicazione": "per record; le righe ritagliate prendono lo stato della formula",
                "riproduzione": b["reproduced"]},
            "unimer_overlap_run2": {
                "controllo": "scripts/unimer_test_overlap.py", "versione": "unimer-test-overlap/2",
                "job": b["report2"]["job_id"],
                "file": {n: file_entry(Path(args.overlap) / n) for n in ("report.json", "train_matches_test.txt",
                                                                         "train_contamination_long.txt")},
                "stato_proposto": {"immagine_del_test": "escluso, per gruppo (stessa immagine di un elemento di "
                                                        "valutazione)",
                                   "formula_lunga_del_test": f"quarantena, per gruppo (formula di più di {LONG} token "
                                                             "visibili, immagine diversa: da riesaminare)",
                                   "formula_breve_del_test": "evidenza, stato invariato"},
                "riproduzione": b["reproduced"]["run2"]},
            "split_duplicates_run1": {
                "controllo": "scripts/split_duplicates.py", "job": "30043",
                "file": {"val_with_duplicates.jsonl": file_entry(Path(args.duplicates) / "val_with_duplicates.jsonl")},
                "uso": "solo controllo incrociato: ogni chiave della vecchia validazione con l'immagine ripetuta sta in "
                       "un gruppo di almeno due formule; la vecchia validazione (prefisso dell'ultimo shard) non vale "
                       "più",
                "chiavi_controllate": b["dup_checked"]}},
        "tokenizer": None, "template": None, "contratti": None,
        "split": {"valori": list(mf.SPLITS), "unita": "gruppo", "quota_dev": args.dev_fraction, "seme": args.seed,
                  "strati": "un solo strato (UniMER-1M)",
                  "regola": "in ordine di sha256({seme, gruppo}), dev finché le formule ammesse di dev sono meno di "
                            "round(quota * formule ammesse); il resto train (manifest.assign_splits); i gruppi senza "
                            "formule ammesse restano in train, dove nessun record si legge",
                  "formule_ammesse": admitted, "formule_ammesse_dev": dev,
                  "quota_ottenuta": round(dev / admitted, 5) if admitted else None},
        "rango_f0": {"campione": SAMPLE, "seme": args.seed,
                     "regola": "sha256 del JSON canonico {campione, seme, gruppo}; per lo sviluppo anche ruolo "
                               "'sviluppo' (F0_CAMPIONE_CANDIDATO §1)"},
        "stati": {"ammesso": "candidato del manifest, non certificato",
                  "quarantena": "da riesaminare: incerti di run4, formule lunghe del test con immagine diversa, righe "
                                "di formule rietichettate",
                  "escluso": "errori certi di run4, stessa immagine di UniMER-Test"},
        "compiti": TASKS,
        "conteggi": counts,
        "records": {"file": "records.jsonl.gz", "sha256": data_sha, "righe": n_lines},
        "derivati": derived,
        "fuori_dal_sottoinsieme": NOT_COVERED,
        "verifiche_mancanti": MISSING_CHECKS,
        "secondi_costruzione": round(t_build, 1), "rss_max_mib": rss_mib()}
    write_new_json(args.out / "manifest.json", meta)
    del recs, b
    verification = verify(args.out, args.check, args.overlap)
    verification.update({"job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
                         "secondi_totali": round(time.time() - t0, 1), "rss_max_mib": rss_mib()})
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "verifica.json", verification)
    summary = {"schema": "limite-vl-manifest-riepilogo/1", "manifest": rel(args.out),
               "nome": args.name, "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"),
               "manifest_sha256": verification["manifest_sha256"], "records_sha256": data_sha, "righe": n_lines,
               "sorgente": SOURCE, "revisione": meta["revisione"], "shards": meta["shards"],
               "liste_importate": {k: {n: e["sha256"] for n, e in v["file"].items()}
                                   for k, v in meta["liste_importate"].items()},
               "split": {k: meta["split"][k] for k in ("quota_dev", "seme", "formule_ammesse", "formule_ammesse_dev",
                                                       "quota_ottenuta")},
               "conteggi": counts, "riproduzione_run4": {k: v for k, v in meta["liste_importate"]["unimer_check_run4"]
                                                         ["riproduzione"].items() if k != "esempi_diversi"},
               "controlli": [{k: c[k] for k in ("controllo", "esito")} for c in verification["controlli"]],
               "esito_verifica": verification["esito"], "fuori_dal_sottoinsieme": NOT_COVERED,
               "verifiche_mancanti": MISSING_CHECKS,
               "nota": "organizza candidati ed evidenze: non certifica la qualità e non autorizza il training"}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.summary, summary)
    print(json.dumps({"righe": n_lines, "conteggi": counts["per_insieme_split_stato"], "split": summary["split"],
                      "verifica": f"{verification['superati']}/{verification['totale']}",
                      "secondi": verification["secondi_totali"], "rss_max_mib": verification["rss_max_mib"]},
                     ensure_ascii=False))
    if not verification["esito"]:
        raise Problem("verifica dopo la scrittura fallita: vedi " + str(args.report / "verifica.json"))
    return summary


if __name__ == "__main__":
    main()
