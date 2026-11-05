#!/usr/bin/env python3
"""Test CPU del manifest candidato di UniMER-1M (scripts/unimer_manifest.py; F0_CAMPIONE_CANDIDATO §6 punto 3), su shard,
liste storiche e snapshot fittizi costruiti qui: nessun dato reale. Solo in un job (slurm/tests_cpu.sbatch o
slurm/unimer_manifest.sbatch).

Il fixture copre: rietichettatura di un indice isolato e di graffe codificate, un errore certo (comando non reso), un
conflitto (stessa immagine, formule diverse), un duplicato (stessa immagine e formula), una formula lunga del test con
un'immagine diversa, due formule con l'immagine di un campione del test, una formula breve del test, una formula lunga
del test che è anche un errore certo (vince il più grave), righe ritagliate di una formula ammessa, di una formula
rietichettata, di una formula con l'immagine del test e di una formula lunga del test, e 40 formule semplici. Controlla:
- categorie e stati di unimer-check/4 uguali a quelli di scripts/unimer_check.py;
- costruzione, scrittura e verifica col percorso del job (unimer_manifest.run): stati, gruppi, split, trasformazioni e
  genitori, evidenze, ranghi, conteggi, verifica dopo la scrittura;
- righe ritagliate: un PNG scambiato con quello di una formula estranea (con chiave ed etichetta giuste) si rifiuta
  prima dei gruppi; nella verifica di un manifest alterato non unisce le due formule; fasce strutturalmente valide
  (band_problems: casi validi e vuote, invertite, fuori dall'immagine, sovrapposte, fuori ordine, malformate, in
  numero diverso dalle righe, dimensioni del JSON diverse dal PNG);
- stati richiesti dalle evidenze nella verifica, su copie alterate del manifest scritto: immagine del test lasciata in
  quarantena, riga ritagliata meno grave del suo gruppo, errore certo e formula lunga non escluso, stato più grave di
  quello richiesto;
- seme: due costruzioni uguali; su più semi gli split cambiano e stati e gruppi no; la verifica non chiede che un altro
  seme cambi il dev (quota zero, quota piccola con due semi consecutivi che danno lo stesso dev);
- rifiuti prima della scrittura: chiave storica senza record o ripetuta, chiave in due shard, etichetta attesa diversa,
  stati incoerenti fra le liste, run4 o run2 non riprodotti, riga ritagliata senza formula, con righe diverse, col PNG di
  un'altra formula, con fasce non valide o dimensioni incoerenti, shard con sha256 diverso, snapshot diverso."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import unimer_manifest as um  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.f1c import array_rows  # noqa: E402
from limite_vl.latex import effective_label, visible_tokens  # noqa: E402

FAILS = []
REV = "2343ddd963290469da36ca83e3a56c66e068add9"
SRC = f"wanderkid/UniMER_Dataset@{REV}"
LONG = "a+b+c+d+e+f+g+h+i+j+k"  # 21 token visibili
ARRAY = "\\begin{array}{l}a\\\\b\\end{array}"
ARRAY_CODED = "\\begin{array}{l}aḊ1Ḍ\\\\b\\end{array}"
ARRAY_IMG = "\\begin{array}{l}p\\\\q\\end{array}"  # formula su due righe con l'immagine di un campione del test
ARRAY_LONG = "\\begin{array}{l}a+b+c+d+e+f\\\\g+h+i+j+k+l\\end{array}"  # formula su due righe, lunga, del test
LONG_BAD = LONG + "\\footnote{b}"  # formula lunga del test e comando non reso: escluso
BANDS = [[0, 5], [6, 10]]


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def png(n: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (32 + n % 50, 16 + n // 50), color=(n * 37) % 256).save(buf, format="PNG")
    return buf.getvalue()


def png_wh(width: int, height: int, color: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def sample(key: str, label: str, image: bytes, subset: str = "train", **extra) -> tuple[str, bytes, dict]:
    with Image.open(io.BytesIO(image)) as im:
        w, h = im.size
    n_vis = max(1, len(visible_tokens(effective_label(label)[0])))
    meta = {"id": key, "subset": subset, "index": int(key.split("-")[1]), "label": label, "label_spaced": label,
            "orig_width": 10 * n_vis, "orig_height": 10, "width": w, "height": h, "max_patches": 1024,
            "format": "PNG", "source": SRC, **extra}
    return key, image, meta


def line(key: str, label: str, image: bytes, bands=None) -> tuple[str, bytes, dict]:
    """Campione delle righe ritagliate: l'immagine della formula, le righe di array_rows e le fasce."""
    return sample(key, label, image, rows=array_rows(label), lines=BANDS if bands is None else bands)


def spec() -> dict:
    """Campioni per shard e liste storiche coerenti (a mano) con le regole di run4 e run2."""
    p_conf, p_dup, p_test, p_img = png(901), png(902), png(903), png(904)
    s0 = [sample("train-0000001", "x^{2}+1", png(1)), sample("train-0000002", "^{3}", png(2)),
          sample("train-0000003", "a\\footnote{b}", png(3)), sample("train-0000004", "a+b", p_conf),
          sample("train-0000005", "a-b", p_conf), sample("train-0000006", "y=mx", p_dup),
          sample("train-0000007", "y=mx", p_dup)]
    s1 = [sample("train-0000008", LONG, png(8)), sample("train-0000009", "w", p_test),
          sample("train-0000010", "w", p_test), sample("train-0000011", "z", png(11)),
          sample("train-0000012", ARRAY, png(12)), sample("train-0000013", ARRAY_CODED, png(13)),
          sample("train-0000014", ARRAY_IMG, p_img), sample("train-0000015", ARRAY_LONG, png(15)),
          sample("train-0000016", LONG_BAD, png(16))]
    s1 += [sample(f"train-{100 + n:07d}", f"x_{{{n}}}+y", png(100 + n)) for n in range(40)]
    lines = [line("train-0000012", ARRAY, png(12)), line("train-0000013", ARRAY_CODED, png(13)),
             line("train-0000014", ARRAY_IMG, p_img), line("train-0000015", ARRAY_LONG, png(15))]
    test = [sample("spe-0000001", LONG, png(951), "spe"), sample("spe-0000002", "q", p_test, "spe"),
            sample("spe-0000003", "z", png(953), "spe"), sample("spe-0000004", "r", p_img, "spe"),
            sample("spe-0000005", ARRAY_LONG, png(955), "spe"), sample("spe-0000006", LONG_BAD, png(956), "spe")]
    relabel = [{"key": "train-0000002", "label": "^{3}", "new_label": "3", "rule": "indice_isolato"},
               {"key": "train-0000013", "label": ARRAY_CODED, "new_label": effective_label(ARRAY_CODED)[0],
                "rule": "graffe_codificate"}]
    reasons = [{"key": "train-0000002", "state": "accepted", "categories": ["rietichettato_indice_isolato"],
                "label": "^{3}"},
               {"key": "train-0000003", "state": "excluded", "categories": ["comando_non_reso"], "label": "a\\footnote{b}"},
               {"key": "train-0000004", "state": "uncertain", "categories": ["conflitto_irrisolto"], "label": "a+b"},
               {"key": "train-0000005", "state": "uncertain", "categories": ["conflitto_irrisolto"], "label": "a-b"},
               {"key": "train-0000013", "state": "accepted", "categories": ["rietichettato_graffe_codificate"],
                "label": ARRAY_CODED},
               {"key": "train-0000016", "state": "excluded", "categories": ["comando_non_reso"], "label": LONG_BAD}]
    return {"train": [("unimer-train-00000.tar", s0), ("unimer-train-00001.tar", s1)],
            "righe": [("unimer-lines-00000.tar", lines)], "test": [("unimer-test-spe.tar", test)],
            "excluded": ["train-0000003", "train-0000016"], "uncertain": ["train-0000004", "train-0000005"],
            "relabel": relabel, "reasons": reasons,
            "matches": ["train-0000008", "train-0000009", "train-0000010", "train-0000011", "train-0000014",
                        "train-0000015", "train-0000016"],
            "contamination": ["train-0000008", "train-0000009", "train-0000010", "train-0000014", "train-0000015",
                              "train-0000016"],
            "duplicates": [{"val_key": "train-0000006", "immagine": 1}, {"val_key": "train-0000011", "target": 1}],
            "report4": {"conflict_groups": 1, "conflict_samples": 2}}


def write_tar(path: Path, items) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for key, image, meta in items:
            for name, data in ((f"{key}.png", image), (f"{key}.json", json.dumps(meta).encode())):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
    data = path.read_bytes()
    return {"name": path.name, "samples": len(items), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_fixture(tmp: Path, s: dict, tamper=None) -> argparse.Namespace:
    """Shard, manifest dello sharding, snapshot e liste storiche su disco; tamper(cartelle) altera i file dopo."""
    shards, check, overlap, dups = tmp / "shards", tmp / "run4", tmp / "run2", tmp / "dup"
    blobs, snap = tmp / "blobs", tmp / "snapshot"
    for d in (check, overlap, dups, blobs, snap):
        d.mkdir(parents=True)
    zips = {}
    for name, content in (("UniMER-1M.zip", b"zip finto"), ("UniMER-Test.zip", b"zip di test finto")):
        sha = hashlib.sha256(content).hexdigest()
        (blobs / sha).write_bytes(content)
        (snap / name).symlink_to(blobs / sha)
        zips[name] = (str(snap / name), sha)
    train = [write_tar(shards / n, items) for n, items in s["train"]]
    righe = [write_tar(shards / "lines" / n, items) for n, items in s["righe"]]
    test = [write_tar(shards / "test" / n, items) for n, items in s["test"]]
    common = {"source": SRC, "scale": 4.0, "max_patches": 1024, "tokens_per_symbol": 4.0, "dense_cap": 2304}
    (shards / "manifest.json").write_text(json.dumps({**common, "job_id": "prova", "shards": train,
                                                      "zip": zips["UniMER-1M.zip"][0],
                                                      "zip_sha256_expected": zips["UniMER-1M.zip"][1],
                                                      "test_zip": zips["UniMER-Test.zip"][0],
                                                      "test_zip_sha256_expected": zips["UniMER-Test.zip"][1]}))
    (shards / "lines" / "manifest.json").write_text(json.dumps({**common, "shards": righe}))
    (shards / "test" / "manifest.json").write_text(json.dumps({**common, "shards": test}))
    n_train = sum(len(items) for _, items in s["train"])
    (check / "report.json").write_text(json.dumps({
        "schema": "unimer-check/4", "job_id": "prova4", "shards": [{"sha256": t["sha256"]} for t in train],
        "samples": n_train, "quality_excluded": len(s["excluded"]), "uncertain": len(s["uncertain"]),
        "relabeled": len(s["relabel"]), "render_used": False, **s["report4"]}))
    (check / "quality_excluded.txt").write_text("".join(k + "\n" for k in s["excluded"]))
    (check / "uncertain.txt").write_text("".join(k + "\n" for k in s["uncertain"]))
    (check / "relabel.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in s["relabel"]))
    (check / "reasons.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in s["reasons"]))
    (overlap / "report.json").write_text(json.dumps({
        "schema": "unimer-test-overlap/2", "job_id": "prova2", "long": 20,
        "test_shards": [{"sha256": t["sha256"]} for t in test], "train_shards": [{"sha256": t["sha256"]} for t in train],
        "train_samples_matching": len(s["matches"]), "train_contamination_long": len(s["contamination"])}))
    (overlap / "train_matches_test.txt").write_text("".join(k + "\n" for k in sorted(s["matches"])))
    (overlap / "train_contamination_long.txt").write_text("".join(k + "\n" for k in sorted(s["contamination"])))
    (dups / "val_with_duplicates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in s["duplicates"]))
    if tamper:
        tamper({"shards": shards, "check": check, "overlap": overlap, "dup": dups, "snapshot": snap})
    return argparse.Namespace(shards=shards, check=check, overlap=overlap, duplicates=dups, out=tmp / "manifest",
                              report=tmp / "verifica", summary=tmp / "riepilogo.json", name="unimer-prova",
                              version="1", dev_fraction=0.25, seed=0, workers=1, probe=False)


def refused(s: dict, tamper=None, run: bool = False) -> str:
    """Il motivo del rifiuto della costruzione (o di run, con lo snapshot), vuoto se passa."""
    with tempfile.TemporaryDirectory() as tmp:
        args = write_fixture(Path(tmp), s, tamper)
        try:
            if run:
                um.run(args)
            else:
                um.build(args, um.shard_plan(args.shards))
        except um.Problem as exc:
            return str(exc) or "rifiutato"
    return ""


def run_quietly(args) -> tuple[dict | None, str]:
    try:
        return um.run(args), ""
    except um.Problem as exc:
        return None, str(exc)


def records_of(folder: Path) -> list[dict]:
    m = mf.load(folder)
    return [r for rows in m.by_shard.values() for r in rows.values()]


def altered(folder: Path, dst: Path, edit) -> Path:
    """Copia del manifest scritto con i record alterati da edit(record) e manifest.json aggiornato (sha256 dei record e
    conteggi): la verifica vede un manifest coerente salvo l'alterazione."""
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
    meta["conteggi"] = um.summarize(recs)
    (dst / "manifest.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    return dst


def failing(result: dict) -> set[str]:
    return {c["controllo"] for c in result["controlli"] if not c["esito"]}


def named(result: dict, fragment: str) -> dict:
    found = [c for c in result["controlli"] if fragment in c["controllo"]]
    assert len(found) == 1, (fragment, [c["controllo"] for c in result["controlli"]])
    return found[0]


def test_fixture():
    check("fixture: formule lunghe oltre la soglia, formule su due righe",
          um.vis_digest(LONG_BAD)[1] > um.LONG and um.vis_digest(ARRAY_LONG)[1] > um.LONG
          and um.vis_digest(ARRAY_IMG)[1] <= um.LONG and len(array_rows(ARRAY_IMG)) == len(array_rows(ARRAY_LONG)) == 2,
          [um.vis_digest(x)[1] for x in (LONG_BAD, ARRAY_LONG, ARRAY_IMG)])


def test_sets():
    import unimer_check

    check("categorie certe e incerte uguali a quelle di unimer-check/4 (scripts/unimer_check.py)",
          um.CERTAIN == unimer_check.CERTAIN and um.UNCERTAIN == unimer_check.UNCERTAIN)
    check("esiti delle evidenze con lo stato richiesto: tutti gli esiti scritti da build()",
          set(um.RUN4_REQUIRED) == {"excluded", "uncertain", "accepted", "nessuna_segnalazione"}
          and set(um.RUN2_REQUIRED) == {"immagine_del_test", "formula_lunga_del_test", "formula_breve_del_test",
                                        "nessuna_corrispondenza"})


def test_build_and_verify():
    with tempfile.TemporaryDirectory() as tmp:
        args = write_fixture(Path(tmp), spec())
        summary = um.run(args)
        recs = records_of(args.out)
        verification = json.loads((args.report / "verifica.json").read_text())
        saved = json.loads(args.summary.read_text())
    base = {r["record"]: r for r in recs if r["insieme"] == "train"}
    righe = {r["record"]: r for r in recs if r["insieme"] == "righe"}
    st = {k: r["stato"] for k, r in base.items()}
    check("costruzione: tutte le formule e le righe ritagliate, verifica dopo la scrittura superata",
          len(base) == 56 and len(righe) == 4 and verification["esito"] and summary["esito_verifica"]
          and saved["manifest_sha256"] == verification["manifest_sha256"],
          f"{verification['superati']}/{verification['totale']}: "
          f"{[c['controllo'] for c in verification['controlli'] if not c['esito']]}")
    check("verifica: nessun controllo sul cambiamento del dev con un altro seme",
          not any("altro seme" in c["controllo"] for c in verification["controlli"]))
    check("stati da run4: errore certo escluso, conflitto in quarantena, rietichettature ammesse",
          st["train-0000003"] == "escluso" and st["train-0000004"] == st["train-0000005"] == "quarantena"
          and st["train-0000002"] == st["train-0000013"] == "ammesso" and st["train-0000001"] == "ammesso", st)
    check("stati da run2: stessa immagine del test esclusa per gruppo, formula lunga in quarantena, formula breve "
          "solo evidenza",
          st["train-0000009"] == st["train-0000010"] == st["train-0000014"] == "escluso"
          and st["train-0000008"] == st["train-0000015"] == "quarantena" and st["train-0000011"] == "ammesso"
          and {"lista": "unimer_overlap_run2", "esito": "formula_breve_del_test"} in base["train-0000011"]["evidenze"]
          and any("stessa immagine di UniMER-Test (spe-0000002)" in x for x in base["train-0000009"]["motivo"])
          and any(x.startswith("formula di UniMER-Test") for x in base["train-0000008"]["motivo"]), st)
    check("errore certo e formula lunga del test: escluso, vince il più grave",
          st["train-0000016"] == "escluso"
          and {"lista": "unimer_overlap_run2", "esito": "formula_lunga_del_test"} in base["train-0000016"]["evidenze"])
    g = {k: r["gruppo"] for k, r in base.items()}
    check("gruppi per stessi byte del PNG: conflitto, duplicato e immagine del test insieme; formule diverse separate",
          g["train-0000004"] == g["train-0000005"] and g["train-0000006"] == g["train-0000007"]
          and g["train-0000009"] == g["train-0000010"] and g["train-0000001"] != g["train-0000002"]
          and len(set(g.values())) == 53)
    check("righe ritagliate: stesso record_id, gruppo e split della formula, genitore e PNG della formula",
          all(r["record_id"] == base[k]["record_id"] and r["gruppo"] == base[k]["gruppo"]
              and r["split"] == base[k]["split"] and r["genitori"][0]["json_sha256"] == base[k]["json_sha256"]
              and r["genitori"][0]["png_sha256"] == base[k]["immagini"][0]["sha256"]
              and r["immagini"][0] == base[k]["immagini"][0]
              and r["trasformazione"]["nome"] == "righe" and r["trasformazione"]["stessa_immagine_del_genitore"]
              for k, r in righe.items()))
    check("righe ritagliate: lo stato della formula (ammessa, esclusa per l'immagine del test, in quarantena per la "
          "formula lunga), in quarantena se la formula è rietichettata",
          righe["train-0000012"]["stato"] == "ammesso" and righe["train-0000013"]["stato"] == "quarantena"
          and righe["train-0000014"]["stato"] == "escluso" and righe["train-0000015"]["stato"] == "quarantena"
          and any("rietichettata" in x for x in righe["train-0000013"]["motivo"]),
          {k: r["stato"] for k, r in righe.items()})
    t2, t13 = base["train-0000002"]["trasformazione"], base["train-0000013"]["trasformazione"]
    check("rietichettature tracciate: regola, nuova etichetta e genitore con l'etichetta originale",
          t2["nome"] == "rietichettatura" and t2["regola"] == "indice_isolato"
          and t2["etichetta_sha256"] == hashlib.sha256(b"3").hexdigest()
          and base["train-0000002"]["genitori"][0]["etichetta_sha256"] == hashlib.sha256(b"^{3}").hexdigest()
          and t13["regola"] == "graffe_codificate" and base["train-0000001"]["trasformazione"] is None)
    splits = {}
    for r in recs:
        splits.setdefault(r["gruppo"], set()).add(r["split"])
    dev = [r for r in recs if r["split"] == "dev"]
    check("split: nessun gruppo in due split, dev presente, gruppi non ammessi in train",
          all(len(v) == 1 for v in splits.values()) and dev
          and all(base[k]["split"] == "train" for k in ("train-0000003", "train-0000009", "train-0000010")))
    check("ranghi del campione candidato: rango_f0 per tutti, rango_f0_sviluppo solo nel dev",
          all(r["rango_f0"] == um.f0_rank(0, r["gruppo"]) for r in recs)
          and all((r["rango_f0_sviluppo"] is not None) == (r["split"] == "dev") for r in recs))
    check("gold assente e compiti per insieme", all(r["gold"] is None for r in recs)
          and all(r["compiti"] == um.TASKS[r["insieme"]] for r in recs))
    check("riproduzione di run4 e run2 registrata nel riepilogo",
          saved["riproduzione_run4"]["reasons_uguali"] and saved["riproduzione_run4"]["relabel_uguali"]
          and saved["riproduzione_run4"]["run2"]["uguali_ai_file"]
          and saved["riproduzione_run4"]["gruppi_in_conflitto"] == 1)
    image = named(verification, "stessa immagine di UniMER-Test")["dettaglio"]
    long_ = named(verification, "almeno in quarantena tutto il gruppo")["dettaglio"]
    state = named(verification, "stato di ogni record uguale")["dettaglio"]
    check("verifica degli stati: gruppi con l'immagine del test e con una formula lunga, righe comprese; ogni stato "
          "uguale a quello richiesto",
          (image["gruppi"], image["record"], long_["gruppi"], long_["record"]) == (2, 4, 3, 4)
          and state["con_stato_atteso"] == state["record"] == 60 and state["diversi"] == 0, (image, long_, state))


def test_righe_png():
    """Il PNG di una riga ritagliata è quello della formula: altrimenti rifiuto prima dei gruppi; in un manifest
    alterato la verifica lo segnala e non unisce le due formule."""
    twin = png_wh(44, 16, 7)  # stesse dimensioni di png(12), byte diversi: il PNG di una formula estranea
    assert Image.open(io.BytesIO(twin)).size == Image.open(io.BytesIO(png(12))).size and twin != png(12)
    s = spec()
    s["train"][1][1].append(sample("train-0000099", "u+v", twin))
    s["righe"][0][1][0] = line("train-0000012", ARRAY, twin)
    same_size = refused(s)
    s = spec()
    s["righe"][0][1][0] = line("train-0000012", ARRAY, png(1))
    other_size = refused(s)
    check("riga ritagliata col PNG di un'altra formula (chiave ed etichetta giuste): rifiutata prima dei gruppi, anche "
          "con le stesse dimensioni",
          "train-0000012" in same_size and "PNG diverso da quello della formula" in same_size
          and "dimensioni" not in same_size.split("train-0000012", 1)[1]
          and "PNG diverso" in other_size and "dimensioni 33x16 diverse da quelle della formula 44x16" in other_size,
          f"{same_size[:160]} | {other_size[:160]}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        args = write_fixture(tmp, spec())
        um.run(args)
        base = {r["record"]: r for r in records_of(args.out) if r["insieme"] == "train"}
        foreign = base["train-0000001"]["immagini"][0]["sha256"]

        def swap(r):
            if r["insieme"] == "righe" and r["record"] == "train-0000012":
                r["immagini"][0]["sha256"] = foreign

        result = um.verify(altered(args.out, tmp / "scambio", swap), args.check, args.overlap)
    png_check = named(result, "lo stesso PNG della formula")
    check("manifest con il PNG di una riga ritagliata scambiato: la verifica lo segnala, i gruppi ricalcolati restano "
          "quelli scritti (le due formule non si uniscono) e nient'altro fallisce",
          failing(result) == {png_check["controllo"]} and png_check["dettaglio"]["esempi"] == ["train-0000012"]
          and base["train-0000001"]["gruppo"] != base["train-0000012"]["gruppo"], sorted(failing(result)))


def test_bands():
    """band_problems: validità strutturale delle fasce, su un PNG 44x16 con due righe."""
    image = png(12)

    def problems(bands="assenti", rows="giuste", **meta):
        m = {"width": 44, "height": 16, **meta}
        if rows == "giuste":
            m["rows"] = array_rows(ARRAY)
        elif rows is not None:
            m["rows"] = rows
        if bands != "assenti":
            m["lines"] = bands
        return um.band_problems(m, ARRAY, image)

    valid = {"fasce separate": [[0, 5], [6, 10]], "fasce adiacenti": [[0, 6], [6, 16]],
             "fino al bordo": [[0, 1], [15, 16]]}
    bad_valid = {k: problems(v) for k, v in valid.items() if problems(v)}
    check("fasce valide: separate, adiacenti (intervalli semiaperti), fino al bordo dell'immagine", not bad_valid,
          bad_valid)
    cases = {
        "fasce assenti": (problems(), "nessuna fascia"),
        "lista di fasce vuota": (problems([]), "nessuna fascia"),
        "fascia vuota": (problems([[0, 5], [6, 6]]), "vuota o invertita"),
        "fascia invertita": (problems([[5, 0], [6, 10]]), "vuota o invertita"),
        "fascia oltre l'altezza": (problems([[0, 5], [6, 17]]), "fuori dall'immagine"),
        "fascia con inizio negativo": (problems([[-1, 5], [6, 10]]), "fuori dall'immagine"),
        "fasce sovrapposte": (problems([[0, 7], [6, 10]]), "sovrapposta alla precedente"),
        "fasce fuori ordine": (problems([[6, 10], [0, 5]]), "non in ordine"),
        "fascia con tre numeri": (problems([[0, 5, 7], [6, 10]]), "non è una coppia di interi"),
        "fascia con una stringa": (problems([[0, "5"], [6, 10]]), "non è una coppia di interi"),
        "fascia con numeri decimali": (problems([[0.0, 5.0], [6, 10]]), "non è una coppia di interi"),
        "fascia con un booleano": (problems([[True, 5], [6, 10]]), "non è una coppia di interi"),
        "fasce come lista piatta": (problems([0, 5, 6, 10]), "non è una coppia di interi"),
        "fasce non in una lista": (problems({"0": [0, 5]}), "nessuna fascia"),
        "più fasce che righe": (problems([[0, 4], [5, 9], [10, 14]]), "3 fasce per 2 righe"),
        "meno fasce che righe": (problems([[0, 5]]), "1 fasce per 2 righe"),
        "righe diverse da array_rows": (problems([[0, 5], [6, 10]], rows=["a", "c"]), "righe diverse"),
        "righe assenti": (problems([[0, 5], [6, 10]], rows=None), "nessuna riga del LaTeX"),
        "altezza del JSON diversa dal PNG": (problems([[0, 5], [6, 10]], height=17), "dimensioni del JSON 44x17"),
        "larghezza del JSON diversa dal PNG": (problems([[0, 5], [6, 10]], width=40), "dimensioni del JSON 40x16"),
    }
    missed = {k: v for k, (v, want) in cases.items() if not any(want in x for x in v)}
    check(f"fasce non valide: {len(cases)} casi, ciascuno col suo motivo", not missed, missed)
    for k, (v, _) in cases.items():
        print(f"     {k}: {'; '.join(v)[:150]}")
    check("PNG illeggibile: nessuna dimensione", any("IHDR" in x for x in um.band_problems(
        {"width": 44, "height": 16, "rows": array_rows(ARRAY), "lines": [[0, 5], [6, 10]]}, ARRAY, b"non un png")))


def test_states():
    """Stati richiesti dalle evidenze, su copie alterate del manifest scritto (manifest.json coerente con i record)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        args = write_fixture(tmp, spec())
        um.run(args)
        clean = um.verify(args.out, args.check, args.overlap)

        def state(insieme, key, value):
            def edit(r):
                if r["insieme"] == insieme and r["record"] == key:
                    r["stato"] = value
            return edit

        image, long_, equal, parent, certain = (named(clean, x)["controllo"] for x in (
            "stessa immagine di UniMER-Test", "almeno in quarantena tutto il gruppo", "stato di ogni record uguale",
            "stato non meno grave", "errori certi esclusi"))
        cases = {
            "immagine del test lasciata in quarantena": (("train", "train-0000009", "quarantena"), {image, equal}),
            "riga ritagliata di una formula con l'immagine del test in quarantena": (
                ("righe", "train-0000014", "quarantena"), {image, equal, parent}),
            "riga ritagliata di una formula lunga del test ammessa": (
                ("righe", "train-0000015", "ammesso"), {long_, equal, parent}),
            "errore certo e formula lunga in quarantena invece che escluso": (
                ("train", "train-0000016", "quarantena"), {equal, certain}),
            "stato più grave di quello richiesto (formula lunga esclusa)": (
                ("train", "train-0000008", "escluso"), {equal}),
        }
        results = {k: failing(um.verify(altered(args.out, tmp / f"caso-{n}", state(*edit)), args.check,
                                        args.overlap))
                   for n, (k, (edit, _)) in enumerate(cases.items())}
    check("manifest corretto: nessun falso rifiuto", clean["esito"], sorted(failing(clean)))
    wrong = {k: sorted(results[k]) for k, (_, want) in cases.items() if results[k] != want}
    check(f"stati alterati: {len(cases)} casi, ciascuno rifiutato dai soli controlli attesi", not wrong, wrong)


def test_seed():
    """Il seme: due costruzioni uguali; su più semi gli split cambiano e stati e gruppi no; la verifica non chiede che
    un altro seme cambi il dev."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        args = write_fixture(tmp, spec())
        plan = um.shard_plan(args.shards)
        lines = [[json.dumps(r, sort_keys=True) for r in um.records(um.build(args, plan))] for _ in range(2)]
        by_seed = {}
        for seed in range(6):
            args.seed = seed
            by_seed[seed] = list(um.records(um.build(args, plan)))
        args.seed, args.dev_fraction = 0, 0.0
        info = um.build(args, plan)["info"]
        zero, why_zero = run_quietly(argparse.Namespace(**{**vars(args), "out": tmp / "zero", "report": tmp / "zero-v",
                                                           "summary": tmp / "zero.json"}))
        zero_dev = [r for r in records_of(tmp / "zero") if r["split"] == "dev"] if zero else None
        small = 0.02
        target = round(small * sum(i["ammessi"] for i in info.values()))
        same = next(s for s in range(1000) if mf.assign_splits(info, small, s) == mf.assign_splits(info, small, s + 1))
        steady, why_steady = run_quietly(argparse.Namespace(**{**vars(args), "dev_fraction": small, "seed": same,
                                                               "out": tmp / "small", "report": tmp / "small-v",
                                                               "summary": tmp / "small.json"}))
        steady_dev = {r["gruppo"] for r in records_of(tmp / "small") if r["split"] == "dev"} if steady else None
    check("riproducibilità: due costruzioni danno righe identiche", lines[0] == lines[1] and len(lines[0]) == 60)
    splits = {s: [r["split"] for r in recs] for s, recs in by_seed.items()}
    check("su sei semi gli split non sono tutti uguali, stati e gruppi sì",
          len({tuple(v) for v in splits.values()}) > 1
          and len({tuple(r["stato"] for r in recs) for recs in by_seed.values()}) == 1
          and len({tuple(r["gruppo"] for r in recs) for recs in by_seed.values()}) == 1)
    check("quota di dev zero: nessun dev, verifica superata", zero is not None and zero["esito_verifica"]
          and not zero_dev, why_zero[:200])
    check(f"quota piccola ({small}, obiettivo {target}) con un seme e il successivo che danno lo stesso dev: "
          "verifica superata", steady is not None and steady["esito_verifica"] and target == 1
          and len(steady_dev) == 1, f"seme {same} {why_steady[:200]}")


def test_refusals():
    cases = {}
    s = spec()
    s["excluded"] = s["excluded"] + ["train-9999999"]
    cases["chiave storica senza record"] = refused(s)
    s = spec()
    s["excluded"] = s["excluded"] * 2
    cases["chiave storica ripetuta (evidenza ambigua)"] = refused(s)
    s = spec()
    s["train"][1][1].append(sample("train-0000001", "x^{2}+1", png(1)))
    cases["stessa chiave in due shard di train"] = refused(s)
    s = spec()
    s["relabel"][0] = {**s["relabel"][0], "label": "^{4}"}
    cases["rietichettatura con un'etichetta attesa diversa da quella dello shard"] = refused(s)
    s = spec()
    s["reasons"][1] = {**s["reasons"][1], "state": "uncertain"}
    cases["stati di reasons.jsonl incoerenti con le liste"] = refused(s)
    s = spec()
    s["excluded"] = ["train-0000016"]
    s["reasons"] = [r for r in s["reasons"] if r["key"] != "train-0000003"]
    cases["run4 non riprodotto (errore certo tolto dalle liste)"] = refused(s)
    s = spec()
    s["contamination"] = [k for k in s["contamination"] if k != "train-0000010"]
    cases["run2 non riprodotto (contaminazione tolta)"] = refused(s)
    s = spec()
    s["righe"][0][1].append(line("train-0000777", ARRAY, png(777)))
    cases["riga ritagliata senza la formula"] = refused(s)
    s = spec()
    s["righe"][0][1][0] = sample("train-0000012", ARRAY, png(12), rows=["a"], lines=[[0, 5]])
    cases["riga ritagliata con righe diverse da array_rows"] = refused(s)
    s = spec()
    s["righe"][0][1][0] = line("train-0000012", ARRAY, png(1))
    cases["riga ritagliata col PNG di un'altra formula"] = refused(s)
    s = spec()
    s["righe"][0][1][0] = line("train-0000012", ARRAY, png(12), bands=[[0, 7], [6, 10]])
    cases["riga ritagliata con fasce sovrapposte"] = refused(s)
    s = spec()
    s["righe"][0][1][0] = line("train-0000012", ARRAY, png(12), bands=[[0, 5], [6, 99]])
    cases["riga ritagliata con una fascia fuori dall'immagine"] = refused(s)
    s = spec()
    key, image, meta = line("train-0000012", ARRAY, png(12))
    s["righe"][0][1][0] = (key, image, {**meta, "height": meta["height"] + 1})
    cases["riga ritagliata con dimensioni del JSON diverse dal PNG"] = refused(s)

    def wrong_sha(dirs):
        path = dirs["shards"] / "manifest.json"
        meta = json.loads(path.read_text())
        meta["shards"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(meta))

    cases["shard con sha256 diverso dal manifest dello sharding"] = refused(spec(), wrong_sha)

    def wrong_blob(dirs):
        path = dirs["shards"] / "manifest.json"
        meta = json.loads(path.read_text())
        meta["zip_sha256_expected"] = "1" * 64
        path.write_text(json.dumps(meta))

    cases["snapshot con un blob diverso da quello dello sharding"] = refused(spec(), wrong_blob, run=True)
    missed = [k for k, v in cases.items() if not v]
    check(f"rifiuti prima della scrittura: {len(cases)} casi, ciascuno con un motivo", not missed, missed)
    for k, v in cases.items():
        print(f"     {k}: {v[:150]}")


if __name__ == "__main__":
    test_fixture()
    test_sets()
    test_build_and_verify()
    test_righe_png()
    test_bands()
    test_states()
    test_seed()
    test_refusals()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
