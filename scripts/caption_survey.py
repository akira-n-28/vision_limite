#!/usr/bin/env python3
"""Struttura delle descrizioni di Qwen3-VL-235B in MMFineReason: righe di intestazione più frequenti (titoli markdown,
"**Sezione:**", "Figure N: ...") ed esempi completi per sorgente. Serve a togliere i titoli delle sezioni dal ponte
di F2, che Violetto cita nel ragionamento (63% delle soluzioni scelte, job 29614).

Da eseguire in un job CPU (slurm/caption_survey.sbatch); legge solo le colonne source e caption.

    srun $RUN_PYTHON -u -B scripts/caption_survey.py --parquet FILE... [--examples 2] [--rows 20000]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HEADER = re.compile(r"^\s*(?:#{1,6}\s.*|\*\*[^*]{1,80}:?\*\*:?.*|figure\s*\d*\s*:.*|image\s*\d*\s*:.*)$", re.IGNORECASE)


def normalize(line: str) -> str:
    """Intestazione senza numeri, dettagli dopo i due punti e spazi ripetuti, per contarla."""
    line = re.sub(r"\d+", "N", line.strip())
    m = re.match(r"^(#{1,6}\s*[^:]*:?|\*\*[^*]*\*\*:?|figure\s*N?\s*:|image\s*N?\s*:)", line, re.IGNORECASE)
    return re.sub(r"\s+", " ", m.group(1) if m else line)[:80]


def main() -> None:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", nargs="+", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=20000, help="righe lette per file")
    ap.add_argument("--examples", type=int, default=2, help="descrizioni complete per sorgente")
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("caption_survey.py va eseguito in un job SLURM")
    headers, per_source_headers = Counter(), defaultdict(Counter)
    examples = defaultdict(list)
    n, n_lines = 0, Counter()
    for path in args.parquet:
        table = pq.read_table(path, columns=["source", "caption"]).slice(0, args.rows)
        for r in table.to_pylist():
            cap = r["caption"] or ""
            if not cap.strip():
                continue
            n += 1
            seen = set()
            for line in cap.splitlines():
                if HEADER.match(line):
                    h = normalize(line)
                    if h not in seen:
                        seen.add(h)
                        headers[h] += 1
                        per_source_headers[r["source"]][h] += 1
                n_lines["header" if HEADER.match(line) else "other"] += 1
            if len(examples[r["source"]]) < args.examples:
                examples[r["source"]].append(cap)
        del table
    print(f"{n} descrizioni da {len(args.parquet)} parquet; righe {dict(n_lines)}")
    print(f"\nintestazioni più frequenti (descrizioni che le contengono, su {n}):")
    for h, c in headers.most_common(args.top):
        print(f"  {c:7d} {c / max(1, n):6.1%}  {h!r}")
    for source, caps in sorted(examples.items()):
        top = ", ".join(f"{h!r} {c}" for h, c in per_source_headers[source].most_common(6))
        print(f"\n######## {source}: {top}")
        for cap in caps:
            print("-" * 40)
            print(cap[:2500])


if __name__ == "__main__":
    main()
