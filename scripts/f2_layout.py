#!/usr/bin/env python3
"""Soluzioni F2 che descrivono la disposizione della figura (audit mavis-f2sol, 2026-10-21: Violetto, che vede solo la
descrizione, mette spesso lati orizzontali o verticali, figure sopra o a destra, coordinate, in modo diverso dal disegno).

Per ogni soluzione di selected.jsonl (f2_select.py) conta le parole di disposizione e le coordinate assegnate ai punti;
con --audit (cartella di un audit delle soluzioni, audit_f2.jsonl) le confronta con visual_ok degli agenti.
Da eseguire in un job CPU (slurm/f2_layout.sbatch).

    srun $RUN_PYTHON -u -B scripts/f2_layout.py --selected SEL.jsonl --out OUT.json [--audit DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

LAYOUT = re.compile(r"\b(horizontal(?:ly)?|vertical(?:ly)?|top|bottom|left|right(?!\s*(?:angle|triangle))|above|below|"
                    r"upward|downward|x-axis|y-axis|leftmost|rightmost)\b", re.IGNORECASE)
COORD = re.compile(r"\b[A-Z]\s*=\s*\(\s*-?[\d.]+[^()]{0,20},\s*-?[^()]{1,25}\)|\bcoordinates?\b|\(\s*0\s*,\s*0\s*\)")
# tracce del ponte che f2.grade non riscrive: il prompt citato ("Figure: ...") e la figura "a parole"
TEXT_TRACE = re.compile(r"\"Figure:|\bFigure:|\bin words\b|\bin (?:the )?text\b|\bthe text (?:says|states|gives)\b", re.IGNORECASE)


def features(text: str) -> dict:
    words = Counter(m.group(1).lower() for m in LAYOUT.finditer(text))
    return {"layout_words": sum(words.values()), "coords": len(COORD.findall(text)), "words": dict(words),
            "text_traces": len(TEXT_TRACE.findall(text)), "think_close": text.count("</think>")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--audit", type=Path, help="cartella dell'audit delle soluzioni (audit_f2.jsonl, _gold/items_f2.json)")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_layout.py va eseguito in un job SLURM")
    rows, totals = [], Counter()
    with open(args.selected) as fh:
        for line in fh:
            r = json.loads(line)
            f = features(r["solution"])
            rows.append({"key": r["key"], "source": r.get("source_dataset"), **f})
            totals.update(f["words"])
    n = len(rows)
    out = {"solutions": n,
           "with_layout_words": sum(r["layout_words"] > 0 for r in rows),
           "with_coords": sum(r["coords"] > 0 for r in rows),
           "with_either": sum(r["layout_words"] > 0 or r["coords"] > 0 for r in rows),
           "with_text_traces": sum(r["text_traces"] > 0 for r in rows),
           "think_close_not_1": sum(r["think_close"] != 1 for r in rows),
           "by_source": {}, "words": dict(totals.most_common())}
    for src in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == src]
        out["by_source"][src] = {"solutions": len(rs), "with_either": sum(r["layout_words"] > 0 or r["coords"] > 0 for r in rs)}
    if args.audit:  # confronto con il giudizio degli agenti sulle soluzioni auditate (testo esatto della soluzione)
        items = {it["fid"]: it for it in json.loads((args.audit / "_gold" / "items_f2.json").read_text())}
        table, words = Counter(), Counter()
        for line in open(args.audit / "audit_f2.jsonl"):
            a = json.loads(line)
            it = items.get(a["fid"])
            if not it or it.get("calibration") or a["visual_ok"]["status"] != "resolved":
                continue
            f = features(it["solution"])
            table[(f["layout_words"] > 0 or f["coords"] > 0, a["visual_ok"]["value"])] += 1
            words[(min(f["layout_words"] + f["coords"], 10), a["visual_ok"]["value"])] += 1
        out["audit_vs_layout"] = {f"layout={k[0]} visual_ok={k[1]}": v for k, v in sorted(table.items(), key=str)}
        out["audit_by_count"] = {f"{k[0]}{'+' if k[0] == 10 else ''} visual_ok={k[1]}": v for k, v in sorted(words.items(), key=str)}
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "words"}, indent=1, ensure_ascii=False))
    print("parole:", list(totals.most_common(15)))


if __name__ == "__main__":
    main()
