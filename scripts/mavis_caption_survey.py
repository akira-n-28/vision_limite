#!/usr/bin/env python3
"""Formule delle didascalie di MAVIS-Geometry: frasi raggruppate per schema (nomi dei punti -> P, numeri -> N),
con conteggi ed esempi, e le frasi sulle figure modificate ("is a rectangle ... extended into an equilateral
triangle", semicerchi, figure tagliate). Serve a riscrivere in modo esatto le frasi letteralmente false prima di
usare la didascalia come ponte di F2 (es. "ABCED is a rectangle" per un pentagono, audit STEM 2026-10-20), e a
controllare le domande F2 di structure_question (frasi di struttura, richieste con cifre, esempi).

Da eseguire in un job CPU (slurm/mavis_caption_survey.sbatch); legge solo i JSON degli shard.

    srun $RUN_PYTHON -u -B scripts/mavis_caption_survey.py --shards DIR/*.tar --out DIR [--top 80]
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
from limite_vl.mavis import bridge_caption, f2_problem, sentences as bridge_sentences, structure_question  # noqa: E402

POINTS = re.compile(r"\b[A-Z]{2,7}\b")
NUMBER = re.compile(r"\d+(?:\.\d+)?(?:/\d+)?(?:\*?√\(\d+\))?|√\(\d+\)")
MODIFIED = re.compile(r"extend|prolong|semicircle|semi-circle|removed|inside|outside|beyond|cut|inward|outward", re.IGNORECASE)
DIRECTION = re.compile(r"inward|internal|inside|within|outward|external|outside|beyond", re.IGNORECASE)
# dopo la riscrittura non deve restare nessun "è un rettangolo" con un triangolo o un semicerchio
RESIDUAL = re.compile(r"\bis a (?:rectangle|square)\b.*(?:equilateral triangle|semi-?circ)", re.IGNORECASE)


def sentences(caption: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", caption.strip()) if s.strip()]


def schema(sentence: str) -> str:
    return NUMBER.sub("N", POINTS.sub("P", sentence))


def iter_json(path: Path):
    with tarfile.open(path, "r|") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".json"):
                yield json.loads(tar.extractfile(member).read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=80)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("mavis_caption_survey.py va eseguito in un job SLURM")
    args.out.mkdir(parents=True, exist_ok=True)
    schemas, modified, examples = Counter(), Counter(), defaultdict(list)
    gen_dirs, vd_words = Counter(), Counter()
    bridge, bridge_fail, bridge_ok, residual = Counter(), [], [], []
    question, q_struct, q_examples, q_ask_digits = Counter(), Counter(), [], []
    arc_schemas, arc_no_pi = Counter(), []
    n = 0
    for shard in args.shards:
        for meta in iter_json(shard):
            n += 1
            te = meta["text_en"]
            for s in sentences(te["caption"]):
                k = schema(s)
                schemas[k] += 1
                if MODIFIED.search(s):
                    modified[k] += 1
                    if len(examples[k]) < 3:
                        examples[k].append({"id": meta["id"], "sentence": s, "generation": te["diagram_generation"][:600]})
            for m in DIRECTION.finditer(te["diagram_generation"]):
                gen_dirs[m.group(0).lower()] += 1
            bridged, counts = bridge_caption(te["caption"], te["diagram_generation"])
            bridge["rewritten"] += counts["rewritten"]
            bridge["failed"] += counts["failed"]
            bridge["captions_failed"] += bool(counts["failed"])
            bridge["captions_rewritten"] += bool(counts["rewritten"])
            if counts["failed"] and len(bridge_fail) < 30:
                bridge_fail.append({"id": meta["id"], "caption": te["caption"], "generation": te["diagram_generation"][:800]})
            for s in bridge_sentences(bridged):
                if RESIDUAL.search(s) and not counts["failed"]:
                    bridge["residual"] += 1
                    if len(residual) < 20:
                        residual.append({"id": meta["id"], "sentence": s})
            if len(bridge_ok) < 12 and counts["rewritten"]:
                bridge_ok.append({"id": meta["id"], "before": te["caption"], "after": bridged})
            bridge["f2_none"] += f2_problem(meta) is None
            # domanda F2 con la struttura della figura (structure_question)
            q, qc = structure_question(bridged, te["vision_dominant_question"])
            q_struct[qc["structure"]] += 1
            question["ask_digits"] += qc["ask_digits"]
            question["connector"] += qc["connector"]
            question["no_structure"] += qc["structure"] == 0
            question["ask_no_question_mark"] += not q.endswith("?")
            question["ask_says_arc"] += bool(re.search(r"\barc\b", q, re.IGNORECASE))
            if qc["arc"]:  # richiesta sull'arco di un semicerchio, riscritta da arc_ask
                question["arc_rewritten"] += 1
                arc_schemas[schema(q.rsplit(". ", 1)[-1])] += 1
                gold = str(meta["choices"][meta["answer_index"]])
                if "π" not in gold:
                    question["arc_gold_without_pi"] += 1
                    if len(arc_no_pi) < 20:
                        arc_no_pi.append({"id": meta["id"], "question": q, "choices": meta["choices"], "gold": gold})
            if qc["ask_digits"] and len(q_ask_digits) < 20:
                q_ask_digits.append({"id": meta["id"], "vision_dominant": te["vision_dominant_question"]})
            if len(q_examples) < 12 and (qc["connector"] or meta["depth"] == len(q_examples) % 4):
                q_examples.append({"id": meta["id"], "vision_dominant": te["vision_dominant_question"], "f2": q})
            vd_words[len(te["vision_dominant_question"].split()) // 10 * 10] += 1
        print(f"{shard.name}: {n} didascalie", flush=True)
    report = {"captions": n, "distinct_schemas": len(schemas),
              "top_schemas": schemas.most_common(args.top),
              "modified_schemas": [{"schema": k, "count": c, "examples": examples[k]} for k, c in modified.most_common()],
              "generation_direction_words": dict(gen_dirs), "vision_dominant_question_words": dict(sorted(vd_words.items())),
              "bridge": dict(bridge), "bridge_failures": bridge_fail, "bridge_residual": residual, "bridge_examples": bridge_ok,
              "question": dict(question), "question_structure_sentences": dict(sorted(q_struct.items())),
              "question_ask_digits": q_ask_digits, "question_examples": q_examples,
              "arc_ask_schemas": arc_schemas.most_common(40), "arc_gold_without_pi": arc_no_pi}
    (args.out / "mavis_caption_survey.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("captions", "distinct_schemas", "generation_direction_words", "bridge", "question",
                                             "question_structure_sentences")},
                     ensure_ascii=False))
    print(f"schemi di figure modificate: {len(modified)}; i primi:")
    for k, c in modified.most_common(25):
        print(f"{c:7d}  {k[:200]}")


if __name__ == "__main__":
    main()
