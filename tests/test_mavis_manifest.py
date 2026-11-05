#!/usr/bin/env python3
"""Test CPU del manifest candidato di MAVIS-Geometry (scripts/mavis_manifest.py; F0_CAMPIONE_CANDIDATO §6 punto 3), su
shard fittizi con annotazioni vere di MAVIS (tests/test_mavis_gen.py): nessun dato reale. Il report di run2 del fixture
lo scrive lo script storico scripts/f1_mavis_check.py, eseguito sugli shard fittizi, così il ricalcolo si confronta col
suo vero produttore. Solo in un job (slurm/tests_cpu.sbatch o slurm/mavis_manifest.sbatch).

Il fixture ha sei coppie vere (didascalia e diagram_generation), una figura con più forme, 20 rettangoli con misure
diverse, un duplicato (stessa immagine e stesse annotazioni), un conflitto (stessa immagine, annotazioni diverse) e una
didascalia alterata. Controlla:
- costruzione, scrittura e verifica col percorso del job (mavis_manifest.run): stati del record e dei compiti, shape_type
  in quarantena senza togliere gli altri compiti, coerenza con le annotazioni e fedeltà all'immagine separate, gruppi,
  split per profondità, ranghi, misure razionali, evidenze;
- riproducibilità e dipendenza degli split dal seme;
- rifiuti prima della scrittura: run2 non riprodotto, sha256 di uno shard, chiave segnalata senza record, caption check
  incoerente, escluso dello sharding presente negli shard, duplicato della vecchia validazione da solo nel gruppo,
  snapshot diverso."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import mavis_manifest as mm  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from test_mavis_gen import MULTI_CAP, MULTI_GEN, REAL  # noqa: E402

FAILS = []
REV = "9c80fff300f4f2cb3a76f71e03428eebdd571907"
SRC = f"CaraJ/MAVIS-Geometry@{REV}"


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def png(n: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64 + n % 40, 32 + n // 40), color=((n * 37) % 256, (n * 11) % 256, 90)).save(buf, format="PNG")
    return buf.getvalue()


def sample(key: str, caption: str, generation: str, image: bytes) -> tuple[str, bytes, dict]:
    with Image.open(io.BytesIO(image)) as im:
        w, h = im.size
    depth = int(key[1:key.index("-")])
    meta = {"id": key, "zip_dir": f"RuleBaseGeo/depth{depth}/{key.split('-')[1]}", "depth": depth, "variant": "caption",
            "text_en": {"caption": caption, "diagram_generation": generation}, "choices": [], "answer_index": 0,
            "orig_width": w, "orig_height": h, "width": w, "height": h, "max_patches": 2304, "format": "PNG",
            "source": SRC}
    return key, image, meta


def spec() -> dict:
    cap, gen = REAL["d0-9323"]
    items = [sample(k, c, g, png(i)) for i, (k, (c, g)) in enumerate(REAL.items())]
    items.append(sample("d2-90010", MULTI_CAP, MULTI_GEN, png(50)))
    items += [sample(f"d0-{91000 + n}", cap.replace("Side AB is 3", f"Side AB is {n}"),
                     gen.replace("side ab is 3", f"side ab is {n}"), png(100 + n)) for n in range(6, 26)]
    items.append(sample("d0-90001", cap, gen, png(list(REAL).index("d0-9323"))))  # duplicato: immagine e annotazioni
    # conflitto: l'immagine di d0-48357 con annotazioni proprie (un diagram_generation che nessun altro record ha)
    items.append(sample("d0-90002", cap.replace("Side AB is 3", "Side AB is 30"),
                        gen.replace("side ab is 3", "side ab is 30"), png(list(REAL).index("d0-48357"))))
    # didascalia diversa dal suo diagram_generation, anch'esso unico
    items.append(sample("d0-90003", cap.replace("Side AB is 3", "Side AB is 32"),
                        gen.replace("side ab is 3", "side ab is 31"), png(60)))
    return {"shards": [("mavis-geometry-caption-00000.tar", items[:15]), ("mavis-geometry-caption-00001.tar", items[15:])],
            "issues": [{"key": "d0-90003", "issues": ["misura_diversa"]}], "violations": [],
            "duplicates": [{"val_key": "d0-90001", "immagine": 1}],
            "excluded": {"incomplete": ["d3-77777"], "duplicate_captions": [["d0-88888", "d0-9323"]]}}


def write_tar(path: Path, items) -> dict:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for key, image, meta in items:
            for name, data in ((f"{key}.png", image), (f"{key}.json", json.dumps(meta).encode())):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
    data = path.read_bytes()
    return {"name": path.name, "samples": len(items), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_fixture(tmp: Path, s: dict, tamper=None) -> argparse.Namespace:
    shards, ev, blobs, snap = tmp / "shards", tmp / "outputs", tmp / "blobs", tmp / "snapshot"
    for d in (shards, blobs, snap):
        d.mkdir(parents=True)
    sha = hashlib.sha256(b"zip finto").hexdigest()
    (blobs / sha).write_bytes(b"zip finto")
    (snap / "RuleBaseGeo.zip").symlink_to(blobs / sha)
    rows = [write_tar(shards / name, items) for name, items in s["shards"]]
    (shards / "manifest.json").write_text(json.dumps({
        "source": SRC, "job_id": "prova", "zip": str(snap / "RuleBaseGeo.zip"), "zip_sha256_expected": sha,
        "variant": "caption", "max_patches": 2304, "shards": rows, "excluded": s["excluded"],
        "counts": {"incomplete": len(s["excluded"]["incomplete"]),
                   "duplicate_captions": len(s["excluded"]["duplicate_captions"])}}))
    (shards / "plan.json").write_text(json.dumps({"source": SRC, "shards": len(rows)}))
    run2 = ev / "f1_mavis" / "run2"
    done = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "f1_mavis_check.py"), "--shards",
                           *[str(shards / r["name"]) for r in rows], "--out", str(run2)],
                          env={**os.environ, "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "prova")},
                          capture_output=True, text=True)
    if done.returncode:
        print(done.stdout[-2000:], done.stderr[-2000:])
        raise RuntimeError("f1_mavis_check.py fallito sul fixture")
    n = sum(r["samples"] for r in rows)
    entries = [{"sha256": r["sha256"]} for r in rows]
    for sub, report, lines in (("mavis_caption_check/run1", {"samples_with_issues": len(s["issues"])}, s["issues"]),
                               ("mavis_geometry/run1", {"samples_with_violations": len(s["violations"])},
                                s["violations"])):
        d = ev / sub
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps({"job_id": "prova", "shards": entries, "samples": n, **report}))
        name = "issues.jsonl" if "caption" in sub else "violations.jsonl"
        (d / name).write_text("".join(json.dumps(x) + "\n" for x in lines))
    frac = ev / "f1_mavis" / "fractions_run1"
    frac.mkdir(parents=True)
    (frac / "report.json").write_text(json.dumps({"job_id": "prova"}))
    dup = ev / "split_duplicates" / "run1" / "mavis"
    dup.mkdir(parents=True)
    (dup.parent / "report.json").write_text(json.dumps({"job_id": "prova"}))
    (dup / "val_with_duplicates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in s["duplicates"]))
    if tamper:
        tamper({"shards": shards, "evidence": ev})
    return argparse.Namespace(shards=shards, evidence=ev, out=tmp / "manifest", report=tmp / "verifica",
                              summary=tmp / "riepilogo.json", name="mavis-prova", version="1", dev_fraction=0.3,
                              seed=0, workers=1, probe=False)


def refused(s: dict, tamper=None, run: bool = False) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        args = write_fixture(Path(tmp), s, tamper)
        try:
            if run:
                mm.run(args)
            else:
                mm.build(args, mm.shard_plan(args.shards))
        except um.Problem as exc:
            return str(exc) or "rifiutato"
    return ""


def test_build_and_verify():
    with tempfile.TemporaryDirectory() as tmp:
        args = write_fixture(Path(tmp), spec())
        summary = mm.run(args)
        m = mf.load(args.out)
        verification = json.loads((args.report / "verifica.json").read_text())
    recs = {r["record"]: r for rows in m.by_shard.values() for r in rows.values()}
    st = {k: r["stato"] for k, r in recs.items()}
    check("costruzione: tutti i record, verifica dopo la scrittura superata",
          len(recs) == 30 and verification["esito"] and summary["esito_verifica"],
          f"{verification['superati']}/{verification['totale']}: "
          f"{[c['controllo'] for c in verification['controlli'] if not c['esito']]}")
    tasks = {k: {t["compito"]: t["stato"] for t in r["compiti"]} for k, r in recs.items()}
    check("shape_type in quarantena per compito, gli altri compiti del record ammesso restano ammessi",
          st["d0-9323"] == "ammesso" and tasks["d0-9323"]["shape_type"] == "quarantena"
          and tasks["d0-9323"]["side_len"] == "ammesso" and tasks["d0-9323"]["describe"] == "ammesso"
          and tasks["d2-90010"]["shape_type"] == "quarantena", tasks["d0-9323"])
    check("duplicato con le stesse annotazioni: stesso gruppo, ammesso",
          recs["d0-90001"]["gruppo"] == recs["d0-9323"]["gruppo"] and st["d0-90001"] == "ammesso")
    check("stessa immagine con annotazioni diverse: stesso gruppo, entrambi in quarantena",
          recs["d0-90002"]["gruppo"] == recs["d0-48357"]["gruppo"]
          and st["d0-90002"] == st["d0-48357"] == "quarantena"
          and any("verità diverse" in x for x in recs["d0-48357"]["motivo"]), recs["d0-48357"]["motivo"])
    r3 = recs["d0-90003"]
    check("didascalia diversa da diagram_generation: in quarantena, coerenza non verificata, segnalata dalle liste",
          r3["stato"] == "quarantena" and r3["verifiche"]["coerenza_annotazioni"] == "non_verificata"
          and r3["verita_sha256"] is None and r3["qa"] == []
          and {"lista": "mavis_caption_check_run1", "esito": "segnalato"} in r3["evidenze"], r3["motivo"])
    check("coerenza con le annotazioni e fedeltà all'immagine separate: fedeltà da verificare per tutti",
          all(r["verifiche"]["fedelta_immagine"] == "da_verificare" for r in recs.values())
          and recs["d0-9323"]["verifiche"]["coerenza_annotazioni"] == "verificata")
    check("misure razionali disegnate come decimali contate (3/2 in d0-13996), nessuna risposta inventata",
          recs["d0-13996"]["misure_razionali_come_decimali"] == 1 and recs["d0-9323"]["misure_razionali_come_decimali"] == 0
          and all(r["gold"] is None for r in recs.values()))
    splits = {}
    for r in recs.values():
        splits.setdefault(r["gruppo"], set()).add(r["split"])
    check("split: nessun gruppo in due split, dev presente nella profondità 0",
          all(len(v) == 1 for v in splits.values())
          and any(r["split"] == "dev" and r["famiglia"] == "d0" for r in recs.values()))
    check("ranghi del campione candidato e QA col digest del loader (chiave del tar)",
          all(r["rango_f0"] == um.f0_rank(0, r["gruppo"]) for r in recs.values())
          and all(len(q["sha256"]) == 64 for q in recs["d0-9323"]["qa"]))


def test_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        args = write_fixture(Path(tmp), spec())
        plan = mm.shard_plan(args.shards)
        lines = [[json.dumps(r, sort_keys=True) for r in mm.records(mm.build(args, plan))] for _ in range(2)]
        args.seed = 1
        other = [json.loads(x) for x in (json.dumps(r, sort_keys=True) for r in mm.records(mm.build(args, plan)))]
    first = [json.loads(x) for x in lines[0]]
    check("riproducibilità: due costruzioni danno righe identiche", lines[0] == lines[1] and len(lines[0]) == 30)
    check("con un altro seme cambiano gli split, non gli stati né i gruppi",
          [r["split"] for r in first] != [r["split"] for r in other]
          and [r["stato"] for r in first] == [r["stato"] for r in other]
          and [r["gruppo"] for r in first] == [r["gruppo"] for r in other])


def test_refusals():
    cases = {}

    def edit_json(path: Path, fn) -> None:
        obj = json.loads(path.read_text())
        fn(obj)
        path.write_text(json.dumps(obj))

    cases["run2 non riprodotto (conteggio delle domande cambiato)"] = refused(spec(), lambda d: edit_json(
        d["evidence"] / "f1_mavis" / "run2" / "report.json",
        lambda o: o["question_types_in_pool"].update(side_len=o["question_types_in_pool"]["side_len"] + 1)))
    cases["shard con sha256 diverso dal manifest dello sharding"] = refused(spec(), lambda d: edit_json(
        d["shards"] / "manifest.json", lambda o: o["shards"][0].update(sha256="0" * 64)))
    s = spec()
    s["issues"] = s["issues"] + [{"key": "d9-99999"}]
    cases["chiave segnalata senza record"] = refused(s)
    s = spec()
    s["issues"] = []
    cases["caption check incoerente con la didascalia diversa ricalcolata"] = refused(s)
    s = spec()
    s["excluded"] = {"incomplete": ["d0-9323"], "duplicate_captions": []}
    cases["escluso dello sharding presente negli shard"] = refused(s)
    s = spec()
    s["duplicates"] = [{"val_key": "d0-9323", "immagine": 1}, {"val_key": "d0-41141", "immagine": 1}]
    cases["duplicato della vecchia validazione da solo nel gruppo"] = refused(s)
    cases["snapshot con un blob diverso da quello dello sharding"] = refused(spec(), lambda d: edit_json(
        d["shards"] / "manifest.json", lambda o: o.update(zip_sha256_expected="1" * 64)), run=True)
    missed = [k for k, v in cases.items() if not v]
    check(f"rifiuti prima della scrittura: {len(cases)} casi, ciascuno con un motivo", not missed, missed)
    for k, v in cases.items():
        print(f"     {k}: {v[:150]}")


if __name__ == "__main__":
    test_build_and_verify()
    test_reproducible()
    test_refusals()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
