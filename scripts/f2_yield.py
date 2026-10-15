#!/usr/bin/env python3
"""F2: resa della generazione per numero di campioni, problemi rimasti per un secondo lotto, risposte delle
sorgenti quasi mai risolte.

Da eseguire in un job CPU (slurm/f2_yield.sbatch). Legge le parti di scripts/f2_generate.py e rifà la verifica di
scripts/f2_select.py, con le stesse esclusioni. Le n generazioni di un problema sono campioni indipendenti: la resa
che si sarebbe avuta generandone solo k è la media esatta sui sottoinsiemi di k delle n. Per k da 1 a n e per
fascia (Qwen3-VL-4B sempre giusto, "easy", o no, "hard") stampa per problema la probabilità di almeno una
soluzione usabile e di almeno una corretta, la probabilità di finire nel pool di RL, soluzioni e token scelti con
--keep, token generati; poi la resa a parità di token generati (il budget del lotto). Il tempo di GPU si considera
proporzionale ai token generati: il prompt si calcola una volta per problema qualunque sia k (prefix caching).
Con --parquet-glob conta i problemi sft non ancora generati né esclusi, per sorgente e fascia. Per le sorgenti con
pass rate sotto --low stampa esiti ed esempi di risposte sbagliate (risposta del dataset e \\boxed{} di Violetto),
per distinguere i problemi difficili da un confronto delle risposte troppo severo.

    srun $RUN_PYTHON -u -B scripts/f2_yield.py --gen outputs/f2_gen/RUN --out DIR [--exclude FILE...]
        [--parquet-glob 'SNAPSHOT/data/sft-*.parquet']
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import grade  # noqa: E402


def grade_problem(line: str) -> dict:
    rec = json.loads(line)
    grades = [grade(c["text"], rec["answer"], c["finish_reason"]) for c in rec["completions"]]
    return {"key": rec["key"], "source": rec.get("source_dataset") or "?", "answer": rec["answer"],
            "easy": rec.get("qwen4b_pass_rate") == 1.0, "prompt_tokens": rec.get("prompt_tokens") or 0,
            "tokens": [c["tokens"] for c in rec["completions"]],
            "stopped": [g["stopped"] for g in grades], "correct": [g["correct"] for g in grades],
            "usable": [g["usable"] for g in grades],
            "wrong": [g["answer"] for g in grades if g["stopped"] and g["answer"] is not None and not g["correct"]]}


def subset_yield(item: dict, k: int, keep: int) -> dict[str, float]:
    """Resa attesa con k generazioni invece di n: media sui sottoinsiemi di k indici (come f2_select)."""
    subsets = list(itertools.combinations(range(len(item["tokens"])), k))
    acc = Counter()
    for s in subsets:
        usable = [i for i in s if item["usable"][i]]
        correct = any(item["correct"][i] for i in s)
        kept = min(keep, len(usable))
        acc["usable_any"] += bool(usable)
        acc["solved"] += correct
        acc["rl_pool"] += not correct and any(item["stopped"][i] for i in s)
        acc["selected"] += kept
        acc["selected_tokens"] += kept * sum(item["tokens"][i] for i in usable) / len(usable) if usable else 0
        acc["generated"] += sum(item["tokens"][i] for i in s)
    return {name: v / len(subsets) for name, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, type=Path, help="cartella di f2_generate.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keep", type=int, default=2, help="soluzioni tenute per problema, come in f2_select")
    ap.add_argument("--exclude", nargs="*", default=[], help="file di chiavi da saltare, come in f2_select")
    ap.add_argument("--parquet-glob", help="parquet sft dello snapshot di MMFineReason, per i problemi rimasti")
    ap.add_argument("--low", type=float, default=0.2, help="pass rate sotto cui stampare esempi di una sorgente")
    ap.add_argument("--examples", type=int, default=15, help="esempi di risposte sbagliate per sorgente")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_yield.py va eseguito in un job SLURM")
    t0 = time.time()
    parts = sorted(args.gen.glob("part-*.jsonl"))
    if not parts:
        sys.exit(f"nessuna parte in {args.gen}")
    parquet = sorted(glob.glob(args.parquet_glob)) if args.parquet_glob else []
    if args.parquet_glob and not parquet:
        sys.exit(f"nessun parquet per {args.parquet_glob}")
    exclude: set[str] = set()
    for path in args.exclude:
        exclude.update(line.strip() for line in open(path) if line.strip())
    args.out.mkdir(parents=True, exist_ok=True)

    generated: set[str] = set()
    n_samples = None
    n_excluded = 0
    sums: dict[tuple[str, int], Counter] = defaultdict(Counter)
    n_group, prompt_tokens = Counter(), Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list] = defaultdict(list)
    with Pool(args.workers) as workers:
        for part in parts:
            with open(part) as fh:
                for item in workers.imap(grade_problem, fh, chunksize=16):
                    generated.add(item["key"])
                    if item["key"] in exclude:
                        n_excluded += 1
                        continue
                    n = len(item["tokens"])
                    if n_samples is None:
                        n_samples = n
                    elif n != n_samples:
                        sys.exit(f"{item['key']}: {n} generazioni invece di {n_samples}")
                    groups = ("all", "easy" if item["easy"] else "hard")
                    for g in groups:
                        n_group[g] += 1
                        prompt_tokens[g] += item["prompt_tokens"]
                    for k in range(1, n + 1):
                        y = subset_yield(item, k, args.keep)
                        for g in groups:
                            sums[(g, k)].update(y)
                    src = by_source[item["source"]]
                    src["problems"] += 1
                    src["completions"] += n
                    src["correct"] += sum(item["correct"])
                    src["usable"] += sum(item["usable"])
                    src["truncated"] += n - sum(item["stopped"])
                    src["wrong"] += len(item["wrong"])
                    if item["wrong"] and len(examples[item["source"]]) < args.examples:
                        examples[item["source"]].append({"key": item["key"], "gold": item["answer"],
                                                         "boxed": item["wrong"][0]})
    print(f"{len(parts)} parti, {n_group['all']} problemi, esclusi {n_excluded}, {n_samples} generazioni per "
          f"problema, keep {args.keep} ({time.time() - t0:.0f} s)", flush=True)

    report = {"gen": str(args.gen), "exclude": args.exclude, "keep": args.keep, "samples": n_samples,
              "excluded": n_excluded, "problems": dict(n_group), "by_k": {}}
    for g in ("all", "easy", "hard"):
        m = n_group[g]
        if not m:
            continue
        budget = sums[(g, n_samples)]["generated"]
        print(f"\nfascia {g}: {m} problemi, token di prompt medi {prompt_tokens[g] / m:.0f}, "
              f"token generati {budget / 1e9:.3f} B", flush=True)
        print("  k | per problema: usabile>=1 corretta>=1 pool_RL scelte token_scelti token_generati | "
              "a parità di token generati: problemi usabili>=1 corretti>=1 scelte token_scelti")
        for k in range(1, n_samples + 1):
            s = sums[(g, k)]
            per = {name: v / m for name, v in s.items()}
            scale = budget / s["generated"]
            row = {**{name: round(v, 4) for name, v in per.items()}, "problems_for_budget": round(m * scale),
                   "usable_any_for_budget": round(s["usable_any"] * scale),
                   "solved_for_budget": round(s["solved"] * scale),
                   "selected_for_budget": round(s["selected"] * scale),
                   "selected_tokens_for_budget": round(s["selected_tokens"] * scale)}
            report["by_k"][f"{g}/k{k}"] = row
            print(f"  {k} | {per['usable_any']:.3f} {per['solved']:.3f} {per['rl_pool']:.3f} {per['selected']:.3f} "
                  f"{per['selected_tokens']:.0f} {per['generated']:.0f} | {row['problems_for_budget']} "
                  f"{row['usable_any_for_budget']} {row['solved_for_budget']} {row['selected_for_budget']} "
                  f"{row['selected_tokens_for_budget'] / 1e6:.1f}M", flush=True)

    print("\nesiti per sorgente, dal pass rate più basso:")
    report["by_source"] = {}
    for name, c in sorted(by_source.items(), key=lambda x: x[1]["correct"] / x[1]["completions"]):
        tot = c["completions"]
        other = tot - c["truncated"] - c["correct"] - c["wrong"]
        row = {"problems": c["problems"], "correct": round(c["correct"] / tot, 3), "usable": round(c["usable"] / tot, 3),
               "truncated": round(c["truncated"] / tot, 3), "wrong": round(c["wrong"] / tot, 3),
               "no_answer": round(other / tot, 3)}
        report["by_source"][name] = row
        print(f"  {name:42s} problemi {c['problems']:6d}  corrette {row['correct']:.3f}  usabili {row['usable']:.3f}  "
              f"troncate {row['truncated']:.3f}  sbagliate {row['wrong']:.3f}  senza risposta {row['no_answer']:.3f}")
    report["examples"] = {}
    for name, row in report["by_source"].items():
        if row["correct"] >= args.low:
            continue
        report["examples"][name] = examples[name]
        print(f"\n-- {name}: risposte sbagliate (dataset | \\boxed di Violetto)")
        for ex in examples[name]:
            print(f"  {ex['key']}: {ex['gold'][:120]!r} | {ex['boxed'][:120]!r}")

    if parquet:
        from limite_vl.mmfine import scan_problems

        scanned = scan_problems([Path(p) for p in parquet])
        remaining: dict[str, Counter] = defaultdict(Counter)
        for p in scanned:
            state = "generated" if p["key"] in generated else "excluded" if p["key"] in exclude else "available"
            band = "easy" if p["qwen4b_pass_rate"] == 1.0 else "hard"
            remaining[p["source_dataset"] or "?"][f"{state}_{band}"] += 1
        totals = Counter()
        for c in remaining.values():
            totals.update(c)
        report["remaining"] = {"parquet": len(parquet), "scanned": len(scanned), "totals": dict(totals),
                               "per_source": {k: dict(v) for k, v in sorted(remaining.items())}}
        print(f"\nproblemi sft con domanda, risposta e descrizione: {len(scanned)} in {len(parquet)} parquet; "
              f"{dict(sorted(totals.items()))}")
        print("disponibili per sorgente (easy, hard):")
        for name, c in sorted(remaining.items(), key=lambda x: -(x[1]["available_easy"] + x[1]["available_hard"])):
            print(f"  {name:42s} {c['available_easy']:7d} {c['available_hard']:7d}")
    report["seconds"] = round(time.time() - t0, 1)
    (args.out / "yield.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\nfine in {report['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
