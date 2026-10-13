#!/usr/bin/env python3
"""Distribuzione delle sorgenti di MMFineReason: in tutti i parquet `sft` e nel sottoinsieme scelto da una
generazione di F2 (ricalcolato con limite_vl.mmfine.choose, che è deterministico). Da eseguire in un job CPU.

    srun $RUN_PYTHON -B scripts/f2_sources.py --all FILE... --chosen-from N --select 120000 --easy-frac 0.5
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.mmfine import choose, scan_problems  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", nargs="+", type=Path, required=True, help="tutti i parquet sft, in ordine")
    ap.add_argument("--chosen-from", type=int, required=True, help="quanti dei primi file usava la generazione")
    ap.add_argument("--select", type=int, default=120000)
    ap.add_argument("--easy-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    files = sorted(args.all)
    scanned = scan_problems(files)
    first = {str(p) for p in files[: args.chosen_from]}
    in_first = [p for p in scanned if p["file"] in first]
    chosen = choose(in_first, args.select, args.easy_frac, args.seed)
    full = Counter(p["source_dataset"] for p in scanned)
    head = Counter(p["source_dataset"] for p in in_first)
    pick = Counter(p["source_dataset"] for p in chosen)
    easy_full = Counter(p["source_dataset"] for p in scanned if p["qwen4b_pass_rate"] == 1.0)
    print(f"problemi validi: tutti i {len(files)} file {len(scanned)}, primi {args.chosen_from} {len(in_first)}, "
          f"scelti {len(chosen)}")
    print(f"{'sorgente':28s} {'tutti':>9s} {'%':>6s} {'primi':>9s} {'%':>6s} {'scelti':>8s} {'%':>6s} {'facili%':>8s}")
    for src, n in full.most_common():
        print(f"{str(src)[:28]:28s} {n:9d} {100 * n / len(scanned):6.2f} {head[src]:9d} {100 * head[src] / len(in_first):6.2f} "
              f"{pick[src]:8d} {100 * pick[src] / len(chosen):6.2f} {100 * easy_full[src] / n:8.1f}")
    print("\nprimo e ultimo file in cui compare ogni sorgente:")
    span: dict = {}
    order = {str(f): i for i, f in enumerate(files)}
    for p in scanned:
        i = order[p["file"]]
        lo, hi = span.get(p["source_dataset"], (i, i))
        span[p["source_dataset"]] = (min(lo, i), max(hi, i))
    for src, (lo, hi) in sorted(span.items(), key=lambda x: x[1]):
        print(f"  {str(src)[:28]:28s} file {lo:3d}-{hi:3d}")


if __name__ == "__main__":
    main()
