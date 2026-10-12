#!/usr/bin/env python3
"""Esempi e statistiche di una generazione del ponte testuale (scripts/f2_generate.py), per capire gli scarti.
Da eseguire in un job CPU (slurm/f2_inspect.sbatch).

Stampa: lunghezze delle soluzioni finite e troncate; se le troncate ripetono le stesse frasi (cicli); pass rate di
Violetto per fascia di pass rate di Qwen3-VL-4B; le frasi in cui le soluzioni corrette citano la descrizione;
qualche finale di soluzione troncata.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import _MENTIONS_BRIDGE, grade  # noqa: E402


def repetition(text: str) -> float:
    """Frazione delle frasi (righe non vuote) che sono ripetizioni di una frase già vista."""
    lines = [s.strip() for s in re.split(r"[\n.]", text) if len(s.strip()) > 20]
    return 1 - len(set(lines)) / len(lines) if lines else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, type=Path)
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    finished, truncated, reps_trunc, reps_fin = [], [], [], []
    by_bucket = defaultdict(lambda: [0, 0])
    mention_snippets, trunc_tails, mention_words = [], [], Counter()
    for part in sorted(args.gen.glob("part-*.jsonl")):
        for line in open(part):
            rec = json.loads(line)
            bucket = rec.get("qwen4b_pass_rate")
            for c in rec["completions"]:
                g = grade(c["text"], rec["answer"], c["finish_reason"])
                b = by_bucket[bucket]
                b[0] += g["correct"]
                b[1] += 1
                if c["finish_reason"] == "stop":
                    finished.append(c["tokens"])
                    reps_fin.append(repetition(c["text"]))
                else:
                    truncated.append(c["tokens"])
                    reps_trunc.append(repetition(c["text"]))
                    if len(trunc_tails) < args.examples:
                        trunc_tails.append(c["text"][-400:])
                if g["correct"] and g["mentions_bridge"]:
                    for m in _MENTIONS_BRIDGE.finditer(c["text"]):
                        mention_words[m.group(0).lower()] += 1
                    if len(mention_snippets) < args.examples * 2:
                        m = _MENTIONS_BRIDGE.search(c["text"])
                        mention_snippets.append(c["text"][max(0, m.start() - 160):m.end() + 120].replace("\n", " "))
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else None  # noqa: E731
    print(f"finite: {len(finished)}, token mediana {q(finished, .5)}, p90 {q(finished, .9)}; "
          f"ripetizione media {statistics.mean(reps_fin) if reps_fin else 0:.2f}")
    print(f"troncate: {len(truncated)}; ripetizione media {statistics.mean(reps_trunc) if reps_trunc else 0:.2f}, "
          f"con più del 30% di frasi ripetute {sum(r > 0.3 for r in reps_trunc)}")
    print("pass rate di Violetto (con descrizione) per pass rate di Qwen3-VL-4B:")
    for k in sorted(by_bucket, key=lambda x: (x is None, x)):
        c, n = by_bucket[k]
        print(f"  qwen4b {k}: {c}/{n} = {c / max(1, n):.2f}")
    print(f"parole che citano il ponte nelle soluzioni corrette: {dict(mention_words)}")
    print("\n-- frasi che citano la descrizione")
    for s in mention_snippets:
        print(f"  ...{s}...")
    print("\n-- finali di soluzioni troncate")
    for t in trunc_tails:
        print("  >>> " + t.replace("\n", " ")[-400:])


if __name__ == "__main__":
    main()
