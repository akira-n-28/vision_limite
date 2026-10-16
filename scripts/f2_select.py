#!/usr/bin/env python3
"""F2, ponte testuale: sceglie le soluzioni di Violetto da usare come target (dopo scripts/f2_generate.py).

Da eseguire in un job CPU (slurm/f2_select.sbatch). Per ogni generazione: limite_vl.f2.grade (finita con
stop, con </think>, \\boxed{} corretto rispetto alla risposta verificata; nelle corrette i riferimenti alla
descrizione diventano riferimenti alla figura, e si scartano quelle che dicono di non vedere l'immagine, quelle
con caratteri CJK e quelle che citano la struttura della descrizione, come "Explicit Component Breakdown"). Per
ogni problema tiene fino a --keep soluzioni utilizzabili (scelta deterministica per chiave), e scrive in --out:
- selected.jsonl: chiave, sorgente, risposta, soluzione (testo generato da Violetto, da `<think>` in poi),
  pass rate di Violetto con la descrizione;
- rl_pool.jsonl: problemi con nessuna soluzione corretta ma almeno una finita (al pool di RL di F4);
- stats.json: conteggi, pass rate per sorgente, lunghezze, motivi di scarto;
- solved_unselected.txt: chiavi dei problemi con almeno una soluzione corretta ma nessuna utilizzabile (tutte
  scartate dai filtri), da rigenerare con f2_generate.py --include. Con --keys-only si scrivono solo questo file e
  stats.json.
Con --exclude salta i problemi delle liste di decontaminazione (data/decontam/p6/<sorgente>/exclude_final.txt):
non entrano né nelle soluzioni né nel pool di RL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import grade, no_image_phrase  # noqa: E402


def grade_problem(line: str) -> dict:
    rec = json.loads(line)
    grades = [grade(c["text"], rec["answer"], c["finish_reason"]) for c in rec["completions"]]
    return {"rec": rec, "grades": grades}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, type=Path, help="cartella di f2_generate.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keep", type=int, default=2, help="soluzioni tenute per problema")
    ap.add_argument("--exclude", nargs="*", default=[], help="file di chiavi da saltare (decontaminazione)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--keys-only", action="store_true",
                    help="solo stats.json e solved_unselected.txt, senza selected.jsonl e rl_pool.jsonl")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_select.py va eseguito in un job SLURM")
    t0 = time.time()
    parts = sorted(args.gen.glob("part-*.jsonl"))
    if not parts:
        sys.exit(f"nessuna parte in {args.gen}")
    exclude: set[str] = set()
    for path in args.exclude:
        exclude.update(line.strip() for line in open(path) if line.strip())
    args.out.mkdir(parents=True, exist_ok=True)
    reasons, per_source, lengths, pass_rates = Counter(), defaultdict(lambda: [0, 0]), [], []
    no_image_phrases, no_image_examples = Counter(), defaultdict(list)
    n_problems = n_selected = n_rl = n_excluded = n_with_selection = 0
    solved_unselected = []
    sel_path = os.devnull if args.keys_only else args.out / "selected.jsonl"
    rl_path = os.devnull if args.keys_only else args.out / "rl_pool.jsonl"
    with Pool(args.workers) as pool, open(sel_path, "w") as sel, open(rl_path, "w") as rl:
        for part in parts:
            with open(part) as fh:
                for item in pool.imap(grade_problem, fh, chunksize=16):
                    rec, grades = item["rec"], item["grades"]
                    if rec["key"] in exclude:
                        n_excluded += 1
                        continue
                    n_problems += 1
                    usable = [i for i, g in enumerate(grades) if g["usable"]]
                    correct = sum(g["correct"] for g in grades)
                    for i, g in enumerate(grades):
                        if not g["stopped"]:
                            reasons["truncated"] += 1
                        elif not g["has_think"]:
                            reasons["no_think"] += 1
                        elif g["answer"] is None:
                            reasons["no_boxed"] += 1
                        elif not g["correct"]:
                            reasons["wrong"] += 1
                        elif g["reject"]:
                            reasons[g["reject"]] += 1
                            if g["reject"] == "no_image":  # frase che ha fatto scartare, con un po' di contesto
                                text = rec["completions"][i]["text"]
                                phrase = no_image_phrase(text)
                                no_image_phrases[phrase.lower()] += 1
                                if len(no_image_examples[phrase.lower()]) < 3:
                                    at = text.find(phrase)
                                    no_image_examples[phrase.lower()].append(
                                        {"key": rec["key"], "context": text[max(0, at - 200):at + len(phrase) + 100]})
                        else:
                            reasons["usable"] += 1
                            reasons["rewritten"] += g["cleaned"] != rec["completions"][i]["text"]
                    rate = correct / max(1, len(grades))
                    pass_rates.append(rate)
                    src = per_source[rec.get("source_dataset") or "?"]
                    src[0] += correct
                    src[1] += len(grades)
                    if usable:
                        n_with_selection += 1
                        rng = random.Random(int(hashlib.blake2b(rec["key"].encode(), digest_size=8).hexdigest(), 16))
                        for i in sorted(rng.sample(usable, min(args.keep, len(usable)))):
                            c = rec["completions"][i]
                            lengths.append(c["tokens"])
                            sel.write(json.dumps({"key": rec["key"], "source_dataset": rec.get("source_dataset"),
                                                  "answer": rec["answer"], "solution": grades[i]["cleaned"],
                                                  "rewritten": grades[i]["cleaned"] != c["text"], "tokens": c["tokens"],
                                                  "violetto_pass_rate": rate}, ensure_ascii=False) + "\n")
                            n_selected += 1
                    elif correct:
                        solved_unselected.append(rec["key"])
                    elif any(g["stopped"] for g in grades):
                        rl.write(json.dumps({"key": rec["key"], "source_dataset": rec.get("source_dataset"),
                                             "answer": rec["answer"], "qwen4b_pass_rate": rec.get("qwen4b_pass_rate")},
                                            ensure_ascii=False) + "\n")
                        n_rl += 1
    stats = {"excluded_contaminated": n_excluded, "exclude_files": args.exclude,
             "problems": n_problems, "selected_solutions": n_selected, "problems_with_selection": n_with_selection,
             "problems_solved": sum(1 for r in pass_rates if r > 0), "solved_unselected": len(solved_unselected),
             "rl_pool": n_rl,
             "completion_outcomes": dict(reasons), "no_image_phrases": dict(no_image_phrases.most_common()),
             "no_image_examples": no_image_examples,
             "violetto_pass_rate_mean": statistics.mean(pass_rates) if pass_rates else None,
             "pass_rate_by_source": {k: round(v[0] / max(1, v[1]), 3) for k, v in sorted(per_source.items())},
             "selected_tokens": {"mean": statistics.mean(lengths), "median": statistics.median(lengths),
                                 "max": max(lengths)} if lengths else None,
             "seconds": round(time.time() - t0, 1)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    (args.out / "solved_unselected.txt").write_text("".join(f"{k}\n" for k in sorted(solved_unselected)))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
