#!/usr/bin/env python3
"""Confronto problema per problema fra run di limite-eval sullo stesso taskset (es. un checkpoint F2 contro la
baseline di Violetto).

Legge metrics/samples.jsonl di ogni run (solo le righe del taskset scelto) e stampa, per ogni problema, le risposte
giuste secondo il protocollo scelto, i token medi e i troncamenti; poi la differenza di ogni run dalla prima con un
intervallo bootstrap appaiato: si ricampionano i problemi (gli stessi per le due run) e poi i rollout di ciascun
problema, come l'intervallo del report del valutatore. I run devono avere la stessa versione del correttore e lo
stesso rendering del taskset.

Da eseguire in un job (slurm/eval_compare.sbatch): i samples.jsonl della baseline pesano centinaia di MB.

    srun $RUN_PYTHON -u -B scripts/eval_compare.py --task aime26 RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def load(run: Path, task: str, protocol: str) -> dict[int, list[tuple[float, int, bool]]]:
    """problema -> [(punteggio, token, troncato)] per rollout; controlla versione del correttore e rendering."""
    out: dict[int, list[tuple[float, int, bool]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    meta: set[tuple[str, str]] = set()
    with open(run / "metrics" / "samples.jsonl") as f:
        for line in f:
            if f'"task": "{task}"' not in line:
                continue
            r = json.loads(line)
            if r["task"] != task:
                continue
            key = (r["problem_index"], r["rollout_index"])
            if key in seen:
                sys.exit(f"{run}: rollout duplicato {key}")
            seen.add(key)
            meta.add((r["scorer_version"], r["template_sha256"]))
            out[r["problem_index"]].append((float(r["metrics"][protocol]), int(r["completion_tokens"]), bool(r["truncated"])))
    if len(meta) != 1:
        sys.exit(f"{run}: versioni del correttore o rendering misti: {meta}")
    return dict(out)


def mean(xs):
    return sum(xs) / len(xs)


def paired_bootstrap(a, b, problems, n_boot, seed):
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        ps = [rng.choice(problems) for _ in problems]
        da = mean([mean([rng.choice(a[p])[0] for _ in a[p]]) for p in ps])
        db = mean([mean([rng.choice(b[p])[0] for _ in b[p]]) for p in ps])
        diffs.append(db - da)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="cartelle dei run (la prima è il riferimento)")
    ap.add_argument("--task", default="aime26")
    ap.add_argument("--protocol", default="exact")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("eval_compare.py va eseguito in un job SLURM")

    data, metas = [], []
    for run in args.runs:
        d = load(run, args.task, args.protocol)
        data.append(d)
        print(f"{run.name}: {len(d)} problemi, {sum(len(v) for v in d.values())} rollout", flush=True)
    problems = sorted(data[0])
    for run, d in zip(args.runs[1:], data[1:]):
        if sorted(d) != problems:
            sys.exit(f"{run.name}: problemi diversi dal riferimento")

    names = [run.name.rsplit("-text-", 1)[0] for run in args.runs]
    print(f"\n{args.task}, protocollo {args.protocol}: giuste/rollout, token medi in migliaia, troncati")
    print("problema  " + "  ".join(f"{n[:24]:>24}" for n in names))
    for p in problems:
        cells = []
        for d in data:
            rs = d[p]
            cells.append(f"{sum(s for s, _, _ in rs):3.0f}/{len(rs):<3d} {mean([t for _, t, _ in rs]) / 1000:5.1f}k "
                         f"tr {sum(tr for _, _, tr in rs):2d}")
        mark = "" if len({round(mean([s for s, _, _ in d[p]]), 6) for d in data}) == 1 else "  *"
        print(f"{p:8d}  " + "  ".join(f"{c:>24}" for c in cells) + mark)
    print()
    ref = data[0]
    for name, d in zip(names, data):
        acc = mean([mean([s for s, _, _ in d[p]]) for p in problems])
        toks = mean([t for p in problems for _, t, _ in d[p]])
        trunc = mean([tr for p in problems for _, _, tr in d[p]])
        line = f"{name}: {100 * acc:.1f}%, token medi {toks:.0f}, troncati {100 * trunc:.1f}%"
        if d is not ref:
            lo, hi = paired_bootstrap(ref, d, problems, args.boot, args.seed)
            ref_acc = mean([mean([s for s, _, _ in ref[p]]) for p in problems])
            worse = [p for p in problems if mean([s for s, _, _ in d[p]]) < mean([s for s, _, _ in ref[p]]) - 0.1]
            better = [p for p in problems if mean([s for s, _, _ in d[p]]) > mean([s for s, _, _ in ref[p]]) + 0.1]
            line += (f"; differenza dal riferimento {100 * (acc - ref_acc):+.1f} punti (95% appaiato "
                     f"{100 * lo:+.1f}..{100 * hi:+.1f}); problemi peggiori di oltre 10 punti {worse}, migliori {better}")
        print(line, flush=True)


if __name__ == "__main__":
    main()
