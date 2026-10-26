#!/usr/bin/env python3
"""Costruzione dei dati MAVIS di F1-clean su tutta la popolazione (limite_vl.f1_mavis) con i controlli del codice.

Per ogni campione: verità da diagram_generation confermata dalla didascalia, descrizione canonica, domande brevi.
Controlli:
- andata e ritorno: la descrizione riletta da f1_schema.parse_geometry dà la verità senza perdite (metriche perfette);
- domande: tipi, risposte "yes"/"no" delle presenze, nessuna risposta vuota.
Nessun file per campione: report.json (conteggi per profondità, lunghezze, esempi), sample.jsonl (60 esempi
stratificati per profondità, da leggere) e failures.jsonl (campioni senza verità o con andata e ritorno imperfetto).

Job CPU: slurm/f1_mavis_check.sbatch.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.f1_mavis import describe, mavis_questions, mavis_truth, questions_example  # noqa: E402
from limite_vl.f1_schema import geometry_scores  # noqa: E402
from limite_vl.provenance import code_identity, file_entry  # noqa: E402

SAMPLE_PER_DEPTH = 15


def iter_json(path: Path):
    with tarfile.open(path, "r|") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".json"):
                yield json.loads(tar.extractfile(member).read())


def perfect(s: dict) -> bool:
    return (s["parsed"] and s["points"]["precision"] == s["points"]["recall"] == 1 and s["shapes"]["precision"] ==
            s["shapes"]["recall"] == 1 and s["shape_kind_ok"] == s["shapes_matched"] == s["shape_modification_ok"]
            and all(s[k]["pairs_ok"] == s[k]["true"] == s[k]["pred"] for k in ("sides", "angles")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f1_mavis_check.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} esiste e non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)

    n = 0
    reasons, by_depth, qtypes, answers = Counter(), defaultdict(Counter), Counter(), Counter()
    desc_chars, qa_chars, n_questions = [], [], []
    samples = defaultdict(list)
    with open(args.out / "failures.jsonl", "w") as fail:
        for shard in args.shards:
            for meta in iter_json(shard):
                n += 1
                key, depth, te = f"mavis-{meta['id']}", meta.get("depth"), meta["text_en"]
                by_depth[depth]["campioni"] += 1
                truth, why = mavis_truth(te["caption"], te["diagram_generation"])
                reasons[why] += 1
                if truth is None:
                    fail.write(json.dumps({"id": meta["id"], "reason": why}) + "\n")
                    continue
                prompt, description = describe(key, truth)
                scores = geometry_scores(description, truth)
                if not perfect(scores):
                    reasons["andata_e_ritorno_imperfetto"] += 1
                    fail.write(json.dumps({"id": meta["id"], "reason": "andata_e_ritorno", "scores": scores,
                                           "description": description}, ensure_ascii=False) + "\n")
                    continue
                by_depth[depth]["pronti"] += 1
                pool = mavis_questions(key, truth)
                qtypes.update(q["type"] for q in pool)
                answers.update(q["answer"] for q in pool if q["type"].startswith("exists"))
                if any(not q["answer"] for q in pool):
                    reasons["risposta_vuota"] += 1
                q_prompt, q_answer, picked = questions_example(key, truth)
                n_questions.append(len(picked))
                desc_chars.append(len(description))
                qa_chars.append(len(q_prompt) + len(q_answer))
                if len(samples[depth]) < SAMPLE_PER_DEPTH and n % 97 == 0:
                    samples[depth].append({"id": meta["id"], "depth": depth, "describe_prompt": prompt,
                                           "description": description, "questions_prompt": q_prompt,
                                           "questions_answer": q_answer, "caption": te["caption"],
                                           "generation": te["diagram_generation"]})
            print(f"{shard.name}: {n} campioni", flush=True)

    with open(args.out / "sample.jsonl", "w") as fh:
        for depth in sorted(samples, key=str):
            for s in samples[depth]:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    def dist(xs):
        xs = sorted(xs)
        return {"mean": round(statistics.mean(xs), 1), "p50": xs[len(xs) // 2], "p90": xs[int(len(xs) * 0.9)],
                "max": xs[-1]} if xs else {}

    report = {
        "schema": "f1-mavis-check/1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code": code_identity(),
        "shards": [file_entry(s) for s in args.shards],
        "samples": n,
        "reasons": dict(reasons),
        "by_depth": {str(k): dict(v) for k, v in sorted(by_depth.items(), key=lambda kv: str(kv[0]))},
        "question_types_in_pool": dict(qtypes.most_common()),
        "presence_answers": dict(answers),
        "questions_per_example": dict(Counter(n_questions)),
        "description_chars": dist(desc_chars),
        "questions_chars": dist(qa_chars),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("samples", "reasons", "by_depth", "presence_answers", "description_chars")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
