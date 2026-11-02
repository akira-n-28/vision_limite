#!/usr/bin/env python3
"""Test CPU della decontaminazione esterna dei sintetici v4 (scripts/decontam.py scan --kind synth,
scripts/synth_decontam.py), su un pannello, un manifest e shard piccoli costruiti qui. Solo in un job
(slurm/tests_cpu.sbatch o slurm/synth_decontam.sbatch).

Coprono: domande dei sintetici e della coppia nella scansione; identità dei pixel con un indice del pannello e pHash
come solo candidato; immagine illeggibile; esito per record con shard, posizione e sha256 dei byte; perimetro con
tutti i ruoli del pannello; identificativi ricondotti al record (posizione alterata rifiutata, scansione parziale
accettata solo con --partial); stato per gruppo che non migliora mai; manifest nuovo con split ereditati ed evidenze."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import decontam  # noqa: E402
import synth_decontam as sd  # noqa: E402
from limite_vl import decontam as dc  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402

FAILS = []
ENV = dict(os.environ, SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID", "test"))
TEMPLATE = "Is the vertex of the graph of f to the left or to the right of the y-axis?"


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def drawing(seed: int, fmt: str = "PNG", quality: int = 95) -> bytes:
    """Un disegno a tratti su fondo bianco, diverso per ogni seme."""
    rng = random.Random(seed)
    im = Image.new("RGB", (192, 160), "white")
    d = ImageDraw.Draw(im)
    for _ in range(14):
        x0, y0 = rng.randrange(0, 150), rng.randrange(0, 120)
        d.rectangle((x0, y0, x0 + rng.randrange(10, 40), y0 + rng.randrange(10, 40)),
                    fill=tuple(rng.randrange(0, 256) for _ in range(3)))
    buf = io.BytesIO()
    im.save(buf, format=fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    return buf.getvalue()


def write_tar(path: Path, items: list[tuple[str, dict, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for key, meta, image in items:
            for name, data in ((f"{key}.png", image), (f"{key}.json", json.dumps(meta).encode())):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))


def qa(q: str, answer: str = "yes") -> dict:
    return {"type": "exists", "q": q, "options": ["yes", "no"], "answer": answer}


# ---------------------------------------------------------------- 1. decisioni per gruppo
def test_decide():
    def rec(key, group, state="ammesso", split="train", insieme="train"):
        return {"record": key, "gruppo": group, "stato": state, "split": split, "insieme": insieme}

    def row(pixel=0, image=0, text=0, status="letta", pixel_check=True):
        return {"immagine": status, "controlli": {"pixel": pixel_check, "phash": status == "letta"},
                "corrispondenze": {k: v for k, v in (("pixel", pixel), ("image", image), ("text", text)) if v}}

    records = [rec("a", "G1"), rec("b", "G1"), rec("c", "G2", "quarantena"), rec("d", "G3", "escluso"),
               rec("e", "G4"), rec("f", "G5", split="dev"), rec("g", "G5", split="dev"), rec("h", "G6"),
               rec("i", "G7", split="test", insieme="test"), rec("j", "G7", "escluso", split="test")]
    scan = {"a": row(pixel=1, image=1), "b": row(), "c": row(), "d": row(image=2), "e": row(text=3),
            "f": row(status="illeggibile"), "g": row(), "h": row(), "i": row(image=1), "j": row()}
    d = sd.decide(records, scan)
    check("decisioni: pixel identici escludono il gruppo intero",
          d["a"]["stato"] == d["b"]["stato"] == "escluso" and d["b"]["esito"] == "nessuna_corrispondenza"
          and d["b"]["esito_gruppo"] == "corrispondenza_confermata" and d["b"]["cause"] == ["a"])
    check("decisioni: un record già in quarantena senza corrispondenze non torna ammesso", d["c"]["stato"] == "quarantena")
    check("decisioni: un record già escluso resta escluso anche con un candidato", d["d"]["stato"] == "escluso")
    check("decisioni: un candidato testuale mette il record in quarantena, non lo esclude",
          d["e"]["stato"] == "quarantena" and d["e"]["esito"] == "candidato")
    check("decisioni: un'immagine illeggibile mette in quarantena tutto il gruppo",
          d["f"]["esito"] == "non_verificabile" and d["f"]["stato"] == d["g"]["stato"] == "quarantena")
    check("decisioni: nessuna corrispondenza, stato invariato", d["h"]["stato"] == "ammesso"
          and d["h"]["esito"] == "nessuna_corrispondenza" and d["h"]["stato_gruppo"] is None)
    check("decisioni: valutazione col candidato in quarantena, il train del gruppo resta escluso",
          d["i"]["stato"] == "quarantena" and d["j"]["stato"] == "escluso")
    check("esito: senza identità dei pixel il controllo è incompleto",
          sd.record_outcome(row(pixel_check=False)) == "non_verificabile")


# ---------------------------------------------------------------- 2. pannello, manifest e shard piccoli
def build_fixture(tmp: Path) -> dict:
    same = drawing(1)                     # anche nel pannello, byte per byte
    near = drawing(2)                     # nel pannello in JPEG: pHash vicino, pixel diversi
    near_jpeg = drawing(2, "JPEG", 95)
    others = {i: drawing(10 + i) for i in range(12)}
    h_near, p_near, _ = dc.hash_image_bytes(near)
    h_jpeg, p_jpeg, _ = dc.hash_image_bytes(near_jpeg)
    check("preparazione: il JPEG ha pixel diversi e pHash entro 4 bit", p_near != p_jpeg and dc.hamming(h_near, h_jpeg) <= 4,
          f"distanza {dc.hamming(h_near, h_jpeg)}")
    # ---- indice e pannello: un elemento per ruolo
    entries = [("mathvista", "data/testmini-00000-of-00001.parquet", "testmini-00000-of-00001:1", near_jpeg,
                "Which bar is tallest?", "sviluppo", None),
               ("mathvista", "data/test-00000-of-00002.parquet", "test-00000-of-00002:2", same,
                "What is shown?", "solo_decontaminazione", None),
               ("dynamath", "data/sample1-00000-of-00001.parquet", "sample1-00000-of-00001:3", others[0],
                "Find x.", "test", True),
               ("chartqa", "data/test-00000-of-00001.parquet", "test-00000-of-00001:4", others[1],
                "What is the value?", "test", False),
               ("unimer_test", "hwe", "hwe-0000001", others[2], "a+b", "diagnostica", None),
               ("gsm8k", "main/test-00000-of-00001.parquet", "test-00000-of-00001:5", None,
                "A long question: " + TEMPLATE.replace("y-axis", "y axis") + " Explain.", "sviluppo", None)]
    items, elements = [], []
    for n, (bench, file, item, data, text, role, ok) in enumerate(entries):
        h, pixels, size = dc.hash_image_bytes(data) if data else (None, None, None)
        items.append({"benchmark": bench, "file": file, "item": item, "id": item, "pos": 0 if data else None,
                      "phash": None if h is None else f"{h:016x}", "pixel_sha256": pixels, "size": size,
                      "text": text, "hints": {}})
        elements.append({"chiave": f"{bench}/{item}", "benchmark": bench, "ruolo": role, "gruppo": f"{n:064x}",
                         "metrica_primaria": ok})
    index_dir, panel_dir = tmp / "indice", tmp / "pannello"
    index_dir.mkdir()
    panel_dir.mkdir()
    (index_dir / "index.json").write_text(json.dumps({"schema": "panel-index/2",
                                                      "convenzione_immagini": dc.IMAGE_CONVENTION, "items": items}))
    (index_dir / "verifica.json").write_text(json.dumps({
        "verificato": True, "errori": 0, "voci_con_immagine": sum(it["pos"] is not None for it in items),
        "convenzione_immagini": dc.IMAGE_CONVENTION}))
    (panel_dir / "elements.jsonl.gz").write_bytes(gzip.compress(
        "".join(json.dumps(e) + "\n" for e in elements).encode(), mtime=0))
    (panel_dir / "summary.json").write_text(json.dumps({"indice": {"sha256": hashlib.sha256(
        (index_dir / "index.json").read_bytes()).hexdigest()}, "elementi": len(elements), "job_id": "test"}))
    # ---- shard e manifest della versione 1
    plan = [  # chiave, immagine, domande, stato v1, split, gruppo
        ("synth-0000000", same, [qa("Is there a point labeled A?")], "ammesso", "train", "G0"),
        ("synth-0000001", others[3], [qa("Is there a point labeled B?")], "ammesso", "train", "G0"),
        ("synth-0000002", near, [qa("Is there a point labeled C?")], "ammesso", "train", "G2"),
        ("synth-0000003", others[4], [qa(TEMPLATE, "left")], "ammesso", "train", "G3"),
        ("synth-0000004", others[5], [qa("Is there a point labeled D?")], "quarantena", "train", "G4"),
        ("synth-0000005", others[6], [qa("Is there a point labeled E?")], "escluso", "test", "Gt"),
        ("synth-0000006", others[7], [qa("Is there a point labeled F?")], "ammesso", "train", "G6"),
        ("synth-0000007", others[8], [qa("Is there a point labeled G?")], "ammesso", "dev", "G7"),
        ("synth-0000008", b"not a png", [qa("Is there a point labeled H?")], "ammesso", "train", "G8"),
    ]
    test_plan = [("synth-test-00000", others[9], [qa("Is there a shape labeled K?")], "ammesso", "test", "Gt")]
    pair_plan = [("synth-pair-00000-a", others[10], [], "ammesso", "test", "Gp"),
                 ("synth-pair-00000-b", others[11], [], "ammesso", "test", "Gp")]
    sets = {"train": ("train/synth-train-00000.tar", plan), "test": ("test/synth-test.tar", test_plan),
            "coppie": ("pairs/synth-pairs.tar", pair_plan)}
    digests = mf.Digests()
    shards, lines = [], []
    for insieme, (relpath, rows) in sets.items():
        path = tmp / "shards" / relpath
        metas = []
        for key, image, qas, *_ in rows:
            meta = {"id": key, "family": "coord", "qa": qas, "width": 192, "height": 160}
            if insieme == "coppie":
                meta["pair"] = qa("Is point A above or below the x-axis?", "above" if key.endswith("a") else "below")
            metas.append((key, meta, image))
        write_tar(path, metas)
        shards.append({"insieme": insieme, "nome": path.name, "percorso": str(path),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                       "record": len(rows)})
        for index, ((key, image, qas, state, split, group), (_, meta, _)) in enumerate(zip(rows, metas)):
            rec = {"record_id": mf.record_id(digests, "synth", "v4@" + "0" * 40, key), "sorgente": "synth",
                   "revisione": "v4@" + "0" * 40, "record": key, "insieme": insieme, "shard": str(path),
                   "indice": index, "famiglia": "coord", "coppia": "synth-pair-00000" if insieme == "coppie" else None,
                   "gruppo": mf.sha256_text(group), "split": split, "stato": state,
                   "motivo": {"incerto": ["testi_sovrapposti"]} if state == "quarantena" else
                   {"contaminazione": ["stato"]} if state == "escluso" else {},
                   "gold": None, "immagini": [{"sha256": hashlib.sha256(image).hexdigest(), "ordine": 0}],
                   "json_sha256": hashlib.sha256(json.dumps(meta).encode()).hexdigest(),
                   "qa": [{"tipo": q["type"], "sha256": mf.qa_digest(digests, q)} for q in qas],
                   "evidenze": [{"controllo": "synth_candidate", "esito": state}], "trasformazione": None,
                   "genitori": []}
            lines.append(json.dumps(rec, sort_keys=True))
    folder = tmp / "manifest-1"
    folder.mkdir()
    data = gzip.compress("".join(x + "\n" for x in lines).encode(), mtime=0)
    (folder / "records.jsonl.gz").write_bytes(data)
    (folder / "manifest.json").write_text(json.dumps({
        "schema": mf.SCHEMA, "nome": "synth-test", "versione": "1", "shards": shards, "derivati": {},
        "records": {"file": "records.jsonl.gz", "sha256": hashlib.sha256(data).hexdigest(), "righe": len(lines)},
        "split": {"valori": list(mf.SPLITS), "quota_dev": 0.05, "seme": 0, "dev_per_famiglia": {}},
        "fuori_dal_sottoinsieme": ["indice globale fra sorgenti (S1 §4): non confrontato", "altro"]}))
    return {"index": index_dir, "panel": panel_dir, "manifest": folder, "shards": tmp / "shards"}


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", *map(str, args)], env=ENV, capture_output=True, text=True)


def scan_rows(folder: Path) -> dict[str, dict]:
    with gzip.open(folder / "records.jsonl.gz", "rt") as fh:
        return {r["key"]: r for r in map(json.loads, fh)}


def test_end_to_end(tmp: Path):
    fx = build_fixture(tmp)
    check("texts_of synth: domande della scena e della coppia",
          decontam.texts_of("synth", {"qa": [qa("a"), qa("b")], "pair": qa("c")}) == ["a", "b", "c"]
          and decontam.texts_of("synth", {"qa": []}) == [])
    scan = [ROOT / "scripts/decontam.py", "scan", "--kind", "synth", "--index", fx["index"] / "index.json",
            "--workers", "1", "--max-lines", "50"]
    res = run(*scan, "--shards", fx["shards"] / "*/*.tar", "--out", tmp / "scan")
    check("scan: eseguita", res.returncode == 0, res.stderr[-300:])
    rows = scan_rows(tmp / "scan")
    summary = json.loads((tmp / "scan/summary.json").read_text())
    r0, r2, r3, r8 = rows["synth-0000000"], rows["synth-0000002"], rows["synth-0000003"], rows["synth-0000008"]
    check("scan: pixel identici a una voce del pannello (con elemento e posizione)",
          r0["corrispondenze"].get("pixel") == 1 and any(
              d["match"] == "pixel" and d["elemento"] == "mathvista/test-00000-of-00002:2" and d["pos"] == 0
              for d in r0["dettagli"]), str(r0)[:300])
    check("scan: il JPEG del pannello è solo un candidato pHash",
          r2["corrispondenze"].get("image", 0) >= 1 and not r2["corrispondenze"].get("pixel"), str(r2)[:200])
    check("scan: la domanda a modello che coincide per 10-gram è un candidato testuale",
          r3["corrispondenze"].get("text") == 1 and r3["controlli"]["testo"] == {"testi": 1, "confrontabili": 1})
    check("scan: immagine illeggibile registrata", r8["immagine"] == "illeggibile" and not r8["controlli"]["phash"])
    check("scan: posizione e sha256 dei byte del campione",
          r2["posizione"] == 2 and r2["byte_sha256"] == hashlib.sha256(drawing(2)).hexdigest())
    check("scan: la domanda della coppia entra nel confronto testuale",
          rows["synth-pair-00000-a"]["controlli"]["testo"]["testi"] == 1)
    check("scan: riepilogo con indice, identità dei pixel e sha256 delle uscite",
          summary["identita_pixel"] and summary["indice"]["voci_con_pixel_sha256"] == 5 and summary["samples"] == 12
          and summary["files"]["records.jsonl.gz"]["sha256"] == hashlib.sha256(
              (tmp / "scan/records.jsonl.gz").read_bytes()).hexdigest())
    res = run(*scan, "--shards", fx["shards"] / "*/*.tar", "--out", tmp / "scan")
    check("scan: una cartella di uscita non vuota è rifiutata", res.returncode != 0 and "non è vuota" in res.stderr)
    common = ["--manifest", fx["manifest"], "--index", fx["index"], "--panel", fx["panel"]]
    tool = ROOT / "scripts/synth_decontam.py"
    res = run(tool, "verifica", *common, "--scan", tmp / "scan", "--out", tmp / "verifica")
    report = json.loads((tmp / "verifica/report.json").read_text()) if (tmp / "verifica/report.json").exists() else {}
    check("verifica: indice, perimetro, scansione e identificativi compatibili", res.returncode == 0
          and not report.get("falliti"), (report.get("falliti"), res.stderr[-300:]))
    check("verifica: perimetro con tutti i ruoli del pannello", report.get("perimetro_pannello") == {
        "sviluppo": 2, "solo_decontaminazione": 1, "test_ammesso": 1, "test_non_ammesso": 1, "diagnostica": 1})
    # scansione limitata: accettata solo con --partial
    res = run(*scan, "--shards", fx["shards"] / "test/*.tar", "--out", tmp / "scan-test")
    ok_partial = run(tool, "verifica", *common, "--scan", tmp / "scan-test", "--out", tmp / "v-parziale",
                     "--partial").returncode == 0
    no_partial = run(tool, "verifica", *common, "--scan", tmp / "scan-test", "--out", tmp / "v-completa").returncode != 0
    check("verifica: una scansione limitata passa con --partial e non come completa", ok_partial and no_partial)
    # posizione alterata: rifiutata, nessun manifest
    bad = tmp / "scan-alterata"
    shutil.copytree(tmp / "scan", bad)
    altered = [dict(r, posizione=r["posizione"] + 1) if r["key"] == "synth-0000006" else r
               for r in scan_rows(tmp / "scan").values()]
    (bad / "records.jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in altered).encode(), mtime=0))
    s = json.loads((bad / "summary.json").read_text())
    s["files"]["records.jsonl.gz"]["sha256"] = hashlib.sha256((bad / "records.jsonl.gz").read_bytes()).hexdigest()
    (bad / "summary.json").write_text(json.dumps(s))
    res = run(tool, "manifest", *common, "--scan", bad, "--out", tmp / "manifest-x", "--version", "2")
    check("manifest: una riga della scansione con un'altra posizione ferma tutto",
          res.returncode != 0 and "posizione" in (res.stderr + res.stdout) and not (tmp / "manifest-x").exists(),
          res.stderr[-200:])
    # manifest nuovo
    res = run(tool, "manifest", *common, "--scan", tmp / "scan", "--out", tmp / "manifest-2", "--version", "2")
    check("manifest: versione 2 scritta", res.returncode == 0, res.stderr[-400:])
    if res.returncode != 0:
        return
    m1, m2 = mf.load(fx["manifest"]), mf.load(tmp / "manifest-2")
    old = {k: r for s in m1.by_shard.values() for k, r in s.items()}
    new = {k: r for s in m2.by_shard.values() for k, r in s.items()}
    state = {k: r["stato"] for k, r in new.items()}
    check("manifest: pixel identici escludono il gruppo, compreso il membro senza corrispondenze",
          state["synth-0000000"] == state["synth-0000001"] == "escluso"
          and sd.CHECK in new["synth-0000001"]["motivo"])
    check("manifest: candidati pHash e testo e immagine illeggibile in quarantena",
          state["synth-0000002"] == state["synth-0000003"] == state["synth-0000008"] == "quarantena")
    check("manifest: nessuna riammissione (quarantena ed esclusione precedenti restano)",
          state["synth-0000004"] == "quarantena" and state["synth-0000005"] == "escluso"
          and new["synth-0000004"]["motivo"] == old["synth-0000004"]["motivo"])
    check("manifest: record senza corrispondenze ammessi, split e gruppi invariati",
          state["synth-0000006"] == state["synth-0000007"] == state["synth-test-00000"] == "ammesso"
          and all((new[k]["split"], new[k]["gruppo"], new[k]["record_id"]) == (o["split"], o["gruppo"], o["record_id"])
                  for k, o in old.items()))
    ev = [e for e in new["synth-0000000"]["evidenze"] if e["controllo"] == sd.CHECK]
    check("manifest: evidenza con esito, elemento e ruolo del pannello, riferimento alla scansione",
          len(ev) == 1 and ev[0]["esito"] == "corrispondenza_confermata"
          and any(x["elemento"] == "mathvista/test-00000-of-00002:2" and x["ruolo"] == "solo_decontaminazione"
                  for x in ev[0]["elementi"])
          and ev[0]["rif"]["sha256"] == summary["files"]["records.jsonl.gz"]["sha256"]
          and all(sum(e["controllo"] == sd.CHECK for e in r["evidenze"]) == 1 for r in new.values()))
    meta = m2.meta
    check("manifest: split ereditati, precedente e ingressi con sha256",
          meta["split"]["ereditato_da"]["manifest_sha256"] == m1.manifest_sha256
          and meta["decontaminazione"]["indice"]["sha256"] == json.loads(
              (fx["panel"] / "summary.json").read_text())["indice"]["sha256"]
          and meta["decontaminazione"]["esiti_record"] == {"candidato": 2, "corrispondenza_confermata": 1,
                                                           "nessuna_corrispondenza": 8, "non_verificabile": 1})
    train_keys = (tmp / "manifest-2/keys/train.txt").read_text().split()
    check("manifest: chiavi derivate solo dei record ammessi", train_keys == ["synth-0000006"], train_keys)


if __name__ == "__main__":
    test_decide()
    tmp = ROOT / "outputs" / "tmp" / f"test-synth-decontam-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        test_end_to_end(tmp)
    finally:
        shutil.rmtree(tmp)
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
