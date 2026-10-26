#!/usr/bin/env python3
"""Controllo di MAVIS-Geometry su tutta la popolazione: didascalia contro diagram_generation (limite_vl.mavis_gen).

Le due descrizioni vengono dallo stesso programma del motore. Per ogni campione si confrontano misure (angoli e lati),
figure (tipo, nome, lato tolto e sostituito, direzione) e punti. Ogni differenza va in una categoria. La prova
riguarda la coerenza fra i due testi; che l'immagine li rispetti lo dirà l'audit di leggibilità.

Uscite in --out (cartella nuova):
- report.json: conteggi per categoria e profondità, esempi, identità di codice e shard;
- issues.jsonl: un record per ogni campione con almeno una segnalazione.

Da eseguire in un job CPU (slurm/mavis_caption_check.sbatch): legge solo i JSON degli shard.

    srun $RUN_PYTHON -u -B scripts/mavis_caption_check.py --shards DIR/*.tar --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.mavis_gen import compare  # noqa: E402
from limite_vl.provenance import code_identity, file_entry  # noqa: E402

EXAMPLES = 20


def iter_json(path: Path):
    with tarfile.open(path, "r|") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".json"):
                yield json.loads(tar.extractfile(member).read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("mavis_caption_check.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} esiste e non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)

    n, flagged = 0, 0
    samples, occurrences = Counter(), Counter()
    info = Counter()
    by_depth = defaultdict(Counter)
    examples = defaultdict(list)
    figures, measures = Counter(), Counter()
    with open(args.out / "issues.jsonl", "w") as fh:
        for shard in args.shards:
            for meta in iter_json(shard):
                n += 1
                te, depth = meta["text_en"], meta.get("depth")
                r = compare(te["caption"], te["diagram_generation"])
                figures[r["figures"]] += 1
                measures[r["measures"]] += 1
                info.update(c for c, _ in r["info"])
                by_depth[depth]["campioni"] += 1
                if not r["issues"]:
                    continue
                flagged += 1
                by_depth[depth]["con_segnalazioni"] += 1
                cats = Counter(c for c, _ in r["issues"])
                occurrences.update(cats)
                samples.update(cats.keys())
                for c in cats:
                    by_depth[depth][c] += 1
                    if len(examples[c]) < EXAMPLES:
                        examples[c].append({"id": meta["id"], "depth": depth,
                                            "detail": [d for k, d in r["issues"] if k == c],
                                            "caption": te["caption"], "generation": te["diagram_generation"]})
                fh.write(json.dumps({"id": meta["id"], "depth": depth, "issues": r["issues"]}, ensure_ascii=False) + "\n")
            print(f"{shard.name}: {n} campioni, {flagged} con segnalazioni", flush=True)

    report = {
        "schema": "mavis-caption-check/1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code": code_identity(),
        "shards": [file_entry(s) for s in args.shards],
        "samples": n,
        "samples_with_issues": flagged,
        "issue_samples": dict(samples.most_common()),
        "issue_occurrences": dict(occurrences.most_common()),
        "info": dict(info.most_common()),
        "by_depth": {str(k): dict(v) for k, v in sorted(by_depth.items(), key=lambda kv: str(kv[0]))},
        "figures_per_sample": dict(sorted(figures.items())),
        "measures_per_sample": dict(sorted(measures.items())),
        "examples": examples,
        "note": "coerenza fra didascalia e diagram_generation; non prova che l'immagine le rispetti",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("samples", "samples_with_issues", "issue_samples", "info", "by_depth")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
