#!/usr/bin/env python3
"""Verifica semantica e del renderer dei sintetici (limite_vl.synth_check) in un job SLURM CPU. Non scrive shard e
non modifica quelli esistenti.

Due modi, anche insieme:
- --shards GLOB: campioni di shard esistenti, in sola lettura.
  - Il layout si prende dal JSON se c'è (v4). Altrimenti la scena si ridisegna dalla specifica salvata, e si conta se
    il PNG ridisegnato coincide con quello dello shard: è la riproducibilità, per esempio di v3.
  - Ogni shard letto è registrato con il suo sha256.
- --generate N --generator-version V: N scene generate in memoria (chiavi synth-check-NNNNNNN, famiglie a rotazione),
  disegnate e controllate, senza salvare immagini.
- --cases CHIAVE:FAMIGLIA:VERSIONE ...: casi singoli disegnati e salvati in OUT/cases/ (PNG, layout ed esiti), per
  esempio la scena tagliata synth-0000043 in v3 e la stessa chiave in v4.

Un campione è verificato solo senza violazioni del renderer e con tutte le domande ok; con una violazione bloccante
(synth_check.BLOCKING, per esempio una figura tagliata dagli assi) le domande ok diventano non ammissibili.

Uscite nella cartella nuova --out:
- report.json: esiti per famiglia, tipo di domanda e stato; campioni verificati e bloccati; violazioni del renderer per
  tipo; PNG identici; ingressi; identità del codice;
- problems.jsonl: una riga per domanda non ok o violazione, al più PROBLEM_CAP esempi per shard (o blocco generato) e per
  famiglia, tipo ed esito; i conteggi del report sono sempre completi.

    srun $RUN_PYTHON -u -B scripts/synth_semantics_check.py --out DIR [--shards GLOB ...] \\
        [--generate N --generator-version 4] [--cases KEY:FAMILY:VERSION ...] [--workers W]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import synth, synth_check  # noqa: E402
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.provenance import HashingTextWriter, code_identity, sha256_file, write_new_json  # noqa: E402

PROBLEM_CAP = 200


def _keep(problems: list[dict], kept: Counter, new: list[dict]) -> None:
    """Aggiunge gli esempi di problemi fino a PROBLEM_CAP per famiglia, tipo ed esito."""
    for p in new:
        k = (p["family"], p["what"], p.get("type") or p.get("kind"), p.get("status"))
        if kept[k] < PROBLEM_CAP:
            kept[k] += 1
            problems.append(p)


def _check(key: str, meta: dict, png: bytes | None) -> tuple[Counter, list[dict], bool | None]:
    """Esiti di un campione: contatori, problemi e, se il layout è stato ridisegnato, PNG identico."""
    spec, identical = meta["spec"], None
    layout = meta.get("layout")
    if layout is None:
        image, layout = synth.render(spec, layout=True)
        identical = png is not None and synth.png_bytes(image) == png
    result = synth_check.check_sample(meta, layout)
    counts, problems = Counter(), []
    fam = spec["family"]
    for r in result["questions"]:
        counts[f"questions/{fam}/{r['type']}/{r['status']}"] += 1
        if r["status"] != "ok":
            problems.append({"key": key, "family": fam, "what": "question", **r})
    for v in result["violations"]:
        counts[f"violations/{fam}/{v['kind']}"] += 1
        problems.append({"key": key, "family": fam, "what": "violation", **v})
    counts["samples"] += 1
    counts[f"verified/{fam}/{'verificati' if result['verified'] else 'non_verificati'}"] += 1
    if result["blocking"]:
        counts[f"verified/{fam}/con_violazioni_bloccanti"] += 1
    return counts, problems, identical


def shard_task(path: str) -> dict:
    counts, problems, png, kept = Counter(), [], Counter(), Counter()
    for key, files in iter_shard(path):
        c, p, identical = _check(key, json.loads(files["json"]), files.get("png"))
        counts.update(c)
        _keep(problems, kept, p)
        if identical is not None:
            png["checked"] += 1
            png["identical"] += identical
    return {"input": {"path": path, "sha256": sha256_file(path)}, "counts": counts, "problems": problems, "png": png}


def generate_task(task: tuple[int, int, int]) -> dict:
    start, stop, version = task
    counts, problems, kept = Counter(), [], Counter()
    for i in range(start, stop):
        key, family = f"synth-check-{i:07d}", synth.FAMILIES[i % len(synth.FAMILIES)]
        spec = synth.make_spec(key, family, version)
        image, layout = synth.render(spec, layout=True)
        meta = {"id": key, "family": family, "qa": synth.questions(spec), "spec": spec, "layout": layout}
        c, p, _ = _check(key, meta, None)
        counts.update(c)
        _keep(problems, kept, p)
    return {"input": {"generate": [start, stop], "generator_version": version}, "counts": counts, "problems": problems,
            "png": Counter()}


def save_cases(cases: list[str], out: Path) -> list[dict]:
    """Disegna i casi CHIAVE:FAMIGLIA:VERSIONE e salva PNG, layout ed esiti in out/ (file nuovi)."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        key, family, version = case.rsplit(":", 2)
        spec = synth.make_spec(key, family, int(version))
        image, layout = synth.render(spec, layout=True)
        meta = {"id": key, "family": family, "qa": synth.questions(spec), "spec": spec}
        result = synth_check.check_sample(meta, layout)
        stem = f"{key}-v{version}"
        png = out / f"{stem}.png"
        if png.exists():
            sys.exit(f"{png} esiste già")
        png.write_bytes(synth.png_bytes(image))
        write_new_json(out / f"{stem}.json", {"case": case, "spec": spec, "layout": layout, "result": result})
        rows.append({"case": case, "png": png.name, "verified": result["verified"], "blocking": result["blocking"],
                     "questions": dict(Counter(r["status"] for r in result["questions"]))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shards", nargs="*", default=[], help="glob di shard tar da controllare (sola lettura)")
    ap.add_argument("--generate", type=int, default=0, help="scene da generare e controllare in memoria")
    ap.add_argument("--generator-version", type=int, choices=synth.VERSIONS, help="versione per --generate")
    ap.add_argument("--cases", nargs="*", default=[], help="casi CHIAVE:FAMIGLIA:VERSIONE da disegnare e salvare")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("synth_semantics_check.py va eseguito in un job SLURM")
    if args.generate and args.generator_version is None:
        sys.exit("--generate richiede --generator-version")
    paths = sorted({p for g in args.shards for p in glob.glob(g)})
    if args.shards and not paths:
        sys.exit(f"nessuno shard per {args.shards}")
    if not paths and not args.generate and not args.cases:
        sys.exit("niente da controllare: --shards, --generate o --cases")
    for case in args.cases:
        parts = case.rsplit(":", 2)
        if len(parts) != 3 or parts[1] not in synth.FAMILIES or not parts[2].isdigit() or int(parts[2]) not in synth.VERSIONS:
            sys.exit(f"caso non valido: {case} (atteso CHIAVE:FAMIGLIA:VERSIONE)")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: scegliere un'altra cartella")
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    case_rows = save_cases(args.cases, args.out / "cases") if args.cases else []
    counts, png, inputs = Counter(), Counter(), []
    writer = HashingTextWriter(args.out / "problems.jsonl")
    with Pool(args.workers) as pool:
        jobs = [pool.imap_unordered(shard_task, paths)]
        if args.generate:
            chunks = [(i, min(i + args.chunk, args.generate), args.generator_version)
                      for i in range(0, args.generate, args.chunk)]
            jobs.append(pool.imap_unordered(generate_task, chunks))
        for it in jobs:
            for res in it:
                counts.update(res["counts"])
                png.update(res["png"])
                inputs.append(res["input"])
                for p in res["problems"]:
                    writer.write_json_line(p)
                print(f"{res['input']}: {res['counts']['samples']} campioni, {len(res['problems'])} problemi "
                      f"({time.time() - t0:.0f} s)", flush=True)
    problems_file = writer.close()

    def group(prefix: str) -> dict:
        out: dict = {}
        for k, v in sorted(counts.items()):
            if k.startswith(prefix):
                parts = k[len(prefix):].split("/")
                node = out
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = v
        return out

    statuses, verified = Counter(), Counter()
    for k, v in counts.items():
        if k.startswith("verified/"):
            verified[k.rsplit("/", 1)[1]] += v
    for k, v in counts.items():
        if k.startswith("questions/"):
            statuses[k.rsplit("/", 1)[1]] += v
    report = {"schema": "synth-check/2", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
              "args": {k: str(v) for k, v in vars(args).items()},
              "thresholds": {k: getattr(synth_check, k) for k in ("EDGE", "CLOSE_ROOTS", "TANGENT", "SIDE", "ON_LINE",
                                                                   "OFF_LINE", "ANGLE_TOL", "LENGTH_TOL", "EXTREME",
                                                                   "MIN_TEXT_PX", "OVERLAP_PX", "CLIP_PX")},
              "blocking_kinds": list(synth_check.BLOCKING),
              "inputs": sorted(inputs, key=json.dumps), "samples": counts["samples"],
              "samples_verified": dict(verified), "verified_by_family": group("verified/"),
              "questions": dict(statuses), "by_type": group("questions/"), "violations": group("violations/"),
              "cases": case_rows,
              "png_redrawn": dict(png), "problems": problems_file, "problem_cap": PROBLEM_CAP,
              "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("samples", "samples_verified", "questions", "png_redrawn", "seconds")},
                     ensure_ascii=False))
    for row in case_rows:
        print("caso", json.dumps(row, ensure_ascii=False))
    print(json.dumps(report["violations"], ensure_ascii=False))


if __name__ == "__main__":
    main()
