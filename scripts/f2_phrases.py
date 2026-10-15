#!/usr/bin/env python3
"""F2: quante soluzioni scelte (selected.jsonl di scripts/f2_select.py) contengono tracce del ponte testuale che la
riscrittura non toglie: titoli delle sezioni delle descrizioni di MMFineReason ("Explicit Component Breakdown",
"Identified Ambiguities", "Figure 1: Geometric Diagram"), riscritture goffe ("a figure of a figure"), riferimenti
al testo. Da eseguire in un job CPU (slurm/f2_phrases.sbatch). Stampa, per frase, soluzioni e problemi che la
contengono ed esempi con il contesto; con --pattern si aggiungono altre espressioni regolari.

    srun $RUN_PYTHON -u -B scripts/f2_phrases.py --selected outputs/f2_select/RUN/selected.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PATTERNS = {
    "component_breakdown": r"component breakdown",
    "inferred_properties": r"inferred properties",
    "identified_ambiguities": r"identified ambiguities|ambiguous statements",
    "figure_header": r"figure(?: \d+)?: (?:###\s*)?(?:geometric|puzzle|mathematical|diagram|image|chart|table)",
    "the_summary": r"\bthe summary\b|\bsummary says\b",
    "figure_of_figure": r"\bfigure of (?:a |the )?(?:figure|diagram|image|picture|chart)\b",
}
# Non contano come tracce (job 29613): "in text" è quasi sempre "include the reasoning in text", e "the user shows"
# è la riscrittura di "the user describes", che con l'immagine regge. Si possono contare con --pattern.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", required=True, type=Path)
    ap.add_argument("--pattern", action="append", default=[], help="NOME=REGEX in più")
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_phrases.py va eseguito in un job SLURM")
    patterns = dict(PATTERNS)
    for item in args.pattern:
        name, _, rx = item.partition("=")
        patterns[name] = rx
    compiled = {k: re.compile(v, re.IGNORECASE) for k, v in patterns.items()}
    hits, problems, examples = Counter(), defaultdict(set), defaultdict(list)
    any_hit, n = 0, 0
    for line in open(args.selected):
        rec = json.loads(line)
        n += 1
        text = rec["solution"]
        found = False
        for name, rx in compiled.items():
            m = rx.search(text)
            if not m:
                continue
            found = True
            hits[name] += 1
            problems[name].add(rec["key"])
            if len(examples[name]) < args.examples:
                examples[name].append((rec["key"], text[max(0, m.start() - 160):m.end() + 120].replace("\n", " ")))
        any_hit += found
    print(f"{n} soluzioni; con almeno una frase {any_hit} ({any_hit / max(1, n):.1%})")
    for name in patterns:
        print(f"\n== {name} ({patterns[name]}): {hits[name]} soluzioni ({hits[name] / max(1, n):.1%}), "
              f"{len(problems[name])} problemi")
        for key, ctx in examples[name]:
            print(f"  [{key}] {ctx}")


if __name__ == "__main__":
    main()
