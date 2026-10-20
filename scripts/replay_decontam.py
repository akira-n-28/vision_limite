#!/usr/bin/env python3
"""Decontaminazione del replay testuale di F2 (limite_vl.replay) contro i benchmark testuali della suite
math-extended: AIME 2024/2025/2026, HMMT febbraio 2025/2026, BeyondAIME, APEX shortlist, MATH-500, OlympiadBench.

Stessi strumenti dei benchmark visivi (limite_vl.decontam): n-gram di 10 parole normalizzate, senza quelli comuni
a più di --max-df testi dei benchmark; per ogni problema con un n-gram in comune, la copertura in entrambe le
direzioni (n-gram condivisi / n-gram di ciascun testo) con il benchmark più vicino. In più l'uguaglianza delle
parole normalizzate, che copre anche le domande più corte di 10 parole.

Regola: escluso se la copertura è >= --cov (default 0.2) o se le parole coincidono. La regola di p6 (job 29577:
copertura >= 0.8 con almeno 15 parole) qui ne prende solo 17 su 270 (job 29699): senza immagini, le riscritture in
LaTeX diverso (OlympiadBench, MATH-500) e le versioni con i numeri cambiati di BeyondAIME restano fra 0.2 e 0.8,
spesso con la stessa risposta. Fra 0.2 e 0.35 ci sono anche falsi positivi (la chiusa "can be written as m/n where
m and n are relatively prime positive integers" degli AIME), ma escluderli costa lo 0.2% dei problemi.
summary.json conta i candidati per soglia, con e senza risposta uguale, e quanti ne escluderebbe la regola di p6.

Uscita in --out: exclude_final.txt, matches.jsonl, summary.json. Da eseguire in un job CPU
(slurm/replay_decontam.sbatch).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.decontam import TextIndex, normalize_words, word_ngrams  # noqa: E402
from limite_vl.f2 import answers_match  # noqa: E402
from limite_vl.replay import load_problems  # noqa: E402

TEXT_FIELDS = ("problem", "question", "Problem", "Question")
ANSWER_FIELDS = ("answer", "final_answer", "Answer")


def bench_rows(name: str, root: Path) -> list[dict]:
    """Righe di un benchmark (parquet, jsonl o json sotto root) con testo e risposta."""
    import pyarrow.parquet as pq

    files = sorted(p for p in root.rglob("*") if p.suffix in (".parquet", ".jsonl", ".json") and p.is_file())
    out = []
    for f in files:
        if f.suffix == ".parquet":
            cols = pq.read_schema(f).names
            keep = [c for c in cols if c in TEXT_FIELDS + ANSWER_FIELDS]
            rows = pq.read_table(f, columns=keep).to_pylist()
        elif f.suffix == ".jsonl":
            rows = [json.loads(line) for line in f.open() if line.strip()]
        else:
            data = json.loads(f.read_text())
            rows = data if isinstance(data, list) else []
        for i, r in enumerate(rows):
            text = next((r[k] for k in TEXT_FIELDS if isinstance(r.get(k), str) and r[k].strip()), None)
            if text is None:
                continue
            ans = next((r[k] for k in ANSWER_FIELDS if r.get(k) not in (None, "", [])), None)
            if isinstance(ans, list):
                ans = ans[0] if len(ans) == 1 else None
            out.append({"benchmark": name, "id": f"{f.relative_to(root)}:{i}", "text": text,
                        "answer": None if ans is None else str(ans)})
    if not out:
        raise SystemExit(f"{name}: nessuna riga con un campo di testo fra {TEXT_FIELDS} in {root}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deepscaler", type=Path, required=True)
    ap.add_argument("--deepmath", nargs="+", type=Path, required=True)
    ap.add_argument("--benchmark", action="append", required=True, help="nome=cartella dello snapshot")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ngram", type=int, default=10)
    ap.add_argument("--max-df", type=int, default=25)
    ap.add_argument("--cov", type=float, default=0.2)
    ap.add_argument("--words", type=int, default=0)
    ap.add_argument("--keep-cov", type=float, default=0.2, help="candidati da scrivere in matches.jsonl")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("replay_decontam.py va eseguito in un job SLURM")
    t0 = time.time()

    bench = []
    for spec in args.benchmark:
        name, _, path = spec.partition("=")
        rows = bench_rows(name, Path(path))
        print(f"{name}: {len(rows)} problemi ({sum(r['answer'] is not None for r in rows)} con risposta)", flush=True)
        bench.extend(rows)
    tidx = TextIndex.build([b["text"] for b in bench], n=args.ngram, max_df=args.max_df)
    bench_grams = [word_ngrams(normalize_words(b["text"]), args.ngram) for b in bench]
    bench_words = {}
    for j, b in enumerate(bench):
        bench_words.setdefault(tuple(normalize_words(b["text"])), j)
    print(f"indice: {len(bench)} testi, {len(tidx.grams)} n-gram, {tidx.dropped} comuni scartati", flush=True)

    problems = load_problems(args.deepscaler, args.deepmath)
    print(f"replay: {len(problems)} problemi ({Counter(p['source'] for p in problems)}) in {time.time() - t0:.0f} s",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    excluded, by_bench, by_threshold = [], Counter(), Counter()
    with open(args.out / "matches.jsonl", "w") as fh:
        for p in problems:
            words = normalize_words(p["question"] or "")
            exact = bench_words.get(tuple(words)) if words else None
            best = None
            tg = word_ngrams(words, args.ngram)
            for j in tidx.query(p["question"] or ""):
                shared = len(tg & bench_grams[j])
                cov_b, cov_t = shared / max(1, len(bench_grams[j])), shared / max(1, len(tg))
                cov = min(cov_b, cov_t)
                if best is None or cov > best["coverage"]:
                    best = {"coverage": round(cov, 3), "cov_bench": round(cov_b, 3), "cov_train": round(cov_t, 3), "j": j}
            if exact is not None:
                best = {"coverage": 1.0, "cov_bench": 1.0, "cov_train": 1.0, "j": exact}
            if best is None or best["coverage"] < args.keep_cov:
                continue
            b = bench[best.pop("j")]
            same_answer = b["answer"] is not None and answers_match(p["answer"], b["answer"])
            rule = exact is not None or (best["coverage"] >= args.cov and len(words) >= args.words)
            if exact is not None or (best["coverage"] >= 0.8 and len(words) >= 15):
                by_threshold["regola p6"] += 1
            for t in (0.2, 0.3, 0.5, 0.8):
                if best["coverage"] >= t:
                    by_threshold[f"cov>={t}"] += 1
                    if same_answer:
                        by_threshold[f"cov>={t},stessa risposta"] += 1
            if rule:
                excluded.append(p["key"])
                by_bench[b["benchmark"]] += 1
            fh.write(json.dumps({"key": p["key"], "source": p["source"], **best, "words": len(words),
                                 "exact": exact is not None, "same_answer": same_answer, "excluded": rule,
                                 "benchmark": b["benchmark"], "bench_id": b["id"], "answer": p["answer"],
                                 "bench_answer": b["answer"], "text": p["question"][:400], "bench_text": b["text"][:400]},
                                ensure_ascii=False) + "\n")
    (args.out / "exclude_final.txt").write_text("".join(f"{k}\n" for k in sorted(excluded)))
    summary = {"problems": len(problems), "by_source": Counter(p["source"] for p in problems),
               "excluded": len(excluded), "excluded_by_benchmark": by_bench, "candidates_by_threshold": by_threshold,
               "rule": {"cov": args.cov, "words": args.words, "ngram": args.ngram, "max_df": args.max_df},
               "difficulty": Counter(str(p["difficulty"]) for p in problems if p["source"] == "deepmath"),
               "seconds": round(time.time() - t0, 1)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
