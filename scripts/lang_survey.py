#!/usr/bin/env python3
"""Testi non in inglese in MMFineReason (domande e descrizioni con ideogrammi CJK, kana o hangul).

Violetto non sa il cinese e ragiona solo in inglese: una domanda che non può leggere non dà soluzioni utili in F2
(una risposta giusta sarebbe un caso), e una descrizione con trascrizioni in cinese è un target che il decoder di
F1 non sa modellare. Da eseguire in un job CPU (slurm/lang_survey.sbatch). Legge solo le colonne di testo dei
parquet dello snapshot e scrive in --out:
- question_non_english.txt: chiavi con almeno un carattere CJK/kana/hangul nella domanda (da escludere in F2 e F4);
- caption_cjk.txt: chiavi con almeno un carattere CJK/kana/hangul nella descrizione (da escludere in F1);
- stats.json: conteggi per sorgente e, con --gen, il pass rate di Violetto nella generazione di F2 per domande in
  inglese e non, e quante soluzioni corrette contengono caratteri CJK.

    srun $RUN_PYTHON -u -B scripts/lang_survey.py --parquet FILE... --out DIR [--gen outputs/f2_gen/RUN]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import grade  # noqa: E402
from limite_vl.mmfine import mmfine_key, split_of  # noqa: E402

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")


def n_cjk(text: str | None) -> int:
    return len(CJK.findall(text or ""))


def grade_line(line: str) -> tuple[str, int, int, int]:
    rec = json.loads(line)
    grades = [grade(c["text"], rec["answer"], c["finish_reason"]) for c in rec["completions"]]
    correct = [i for i, g in enumerate(grades) if g["correct"]]
    cjk_correct = sum(n_cjk(rec["completions"][i]["text"]) > 0 for i in correct)
    return rec["key"], len(correct), len(grades), cjk_correct


def main() -> None:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", nargs="+", type=Path, required=True, help="parquet dello snapshot (sft e rl)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gen", type=Path, help="cartella di f2_generate.py, per il pass rate di Violetto")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("lang_survey.py va eseguito in un job SLURM")
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    q_keys, c_keys = [], []
    per_source = defaultdict(Counter)
    q_counts = Counter()
    for path in sorted(args.parquet):
        split = split_of(path)
        table = pq.read_table(path, columns=["id", "source", "question", "caption"])
        for r in table.to_pylist():
            key = mmfine_key(split, r["source"], r["id"])
            src = per_source[f"{split}/{r['source']}"]
            src["rows"] += 1
            nq, nc = n_cjk(r["question"]), n_cjk(r["caption"])
            q_counts["0" if nq == 0 else "1-2" if nq < 3 else "3-19" if nq < 20 else ">=20"] += 1
            if nq:
                q_keys.append(key)
                src["question_non_english"] += 1
            if nc:
                c_keys.append(key)
                src["caption_cjk"] += 1
        del table
    (args.out / "question_non_english.txt").write_text("".join(f"{k}\n" for k in sorted(q_keys)))
    (args.out / "caption_cjk.txt").write_text("".join(f"{k}\n" for k in sorted(c_keys)))
    rows = sum(s["rows"] for s in per_source.values())
    stats = {"rows": rows, "question_non_english": len(q_keys), "caption_cjk": len(c_keys),
             "question_cjk_chars": dict(q_counts),
             "per_source": {k: dict(v) for k, v in sorted(per_source.items(), key=lambda x: -x[1]["question_non_english"])}}
    print(f"testi: {rows} righe, domande non in inglese {len(q_keys)}, descrizioni con CJK {len(c_keys)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    if args.gen:
        non_en = set(q_keys)
        acc = {"english": Counter(), "non_english": Counter()}
        parts = sorted(args.gen.glob("part-*.jsonl"))
        with Pool(args.workers) as pool:
            for part in parts:
                with open(part) as fh:
                    for key, correct, total, cjk_correct in pool.imap(grade_line, fh, chunksize=16):
                        a = acc["non_english" if key in non_en else "english"]
                        a["problems"] += 1
                        a["solved"] += correct > 0
                        a["correct"] += correct
                        a["completions"] += total
                        a["correct_with_cjk"] += cjk_correct
        stats["f2_generation"] = {"parts": len(parts), **{k: {**dict(v), "pass_rate": round(v["correct"] / max(1, v["completions"]), 3),
                                                              "solved_frac": round(v["solved"] / max(1, v["problems"]), 3)}
                                                          for k, v in acc.items()}}
        print(f"generazione F2 ({len(parts)} parti): {json.dumps(stats['f2_generation'])}", flush=True)
    stats["seconds"] = round(time.time() - t0, 1)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    top = list(stats["per_source"].items())[:12]
    print("sorgenti con più domande non in inglese:", flush=True)
    for k, v in top:
        print(f"  {k:45s} righe {v['rows']:7d}  domande {v.get('question_non_english', 0):6d}  descrizioni "
              f"{v.get('caption_cjk', 0):6d}", flush=True)


if __name__ == "__main__":
    main()
