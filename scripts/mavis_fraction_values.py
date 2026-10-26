#!/usr/bin/env python3
"""MAVIS: come sono scritti nella figura i valori razionali non interi? Nelle 40 immagini controllate a vista
(outputs/f1_mavis/contact_run1) 1/2 e 3/2 della didascalia sono disegnati 0.5 e 1.5. Questo script conta i valori
frazionari senza radice per denominatore e copia, per ogni denominatore, alcune immagini da guardare con il valore
atteso, per fissare la regola della descrizione canonica (frazione o decimale).

Uscite in --out (cartella nuova): report.json (conteggi per denominatore, esempi) e <id>.png. Job CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.f1_mavis import mavis_truth  # noqa: E402

FRACTION = re.compile(r"^(\d+)/(\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-denominator", type=int, default=3)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} esiste e non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)
    counts, samples = Counter(), Counter()
    examples = defaultdict(list)
    for shard in args.shards:
        pending: dict[str, dict] = {}
        with tarfile.open(shard, "r|") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                key, ext = m.name.rsplit("/", 1)[-1].split(".", 1)
                pending.setdefault(key, {})[ext] = tar.extractfile(m).read()
                if not {"json", "png"} <= pending[key].keys():
                    continue
                files = pending.pop(key)
                meta = json.loads(files["json"])
                truth, _ = mavis_truth(meta["text_en"]["caption"], meta["text_en"]["diagram_generation"])
                if truth is None:
                    continue
                dens = set()
                for name, v in list(truth["sides"].items()):
                    fm = FRACTION.match(v)
                    if fm:
                        d = int(fm.group(2))
                        counts[d] += 1
                        dens.add(d)
                        if len(examples[d]) < args.per_denominator:
                            examples[d].append({"id": meta["id"], "segment": name, "value": v})
                            (args.out / f"{meta['id']}.png").write_bytes(files["png"])
                samples.update(dens)
    report = {"values_by_denominator": dict(sorted(counts.items())), "samples_by_denominator": dict(sorted(samples.items())),
              "examples": {str(k): v for k, v in sorted(examples.items())}, "job_id": os.environ.get("SLURM_JOB_ID")}
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
