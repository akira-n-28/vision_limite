#!/usr/bin/env python3
"""F2, replay testuale: correzione delle generazioni di scripts/replay_generate.py e shard di training.

Una soluzione si tiene se Violetto ha finito da sé (finish_reason stop), ha chiuso `</think>`, la risposta
nell'ultimo \\boxed{} della parte finale coincide con quella del dataset (limite_vl.f2.answers_match, come il ponte
visivo), non contiene caratteri CJK e, con il prompt e il template, sta in --max-len token (le righe di training
di F2). Le soluzioni giuste più lunghe di --max-len si contano a parte: misurano quanto del ragionamento lungo di
Violetto il replay non può coprire.

Uscita:
- --out: stats.json (esiti, lunghezze per sorgente e per esito, pass rate per sorgente e difficoltà) e
  selected.jsonl (metadati delle soluzioni tenute, senza testo);
- --shards: replay-NNNNN.tar con un campione `<chiave>-s<k>` per soluzione: JSON con prompt (domanda con
  l'istruzione, come in generazione), soluzione (da `<think>` in poi), risposta, sorgente, difficoltà, stile e token
  (data.text_replay_sample). Ogni tar si scrive in `.tmp` e si rinomina.

Da eseguire in un job CPU (slurm/replay_select.sbatch).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import tarfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from shard_llava_pretrain import HashingWriter, add_member  # noqa: E402

from limite_vl.f2 import CJK, answers_match, last_boxed, split_completion  # noqa: E402
from limite_vl.replay import load_problems, replay_prompt  # noqa: E402

TEMPLATE_TOKENS = 16  # <|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n e <|endoftext|>: margine


def grade(line: str) -> dict:
    rec = json.loads(line)
    out = []
    for k, c in enumerate(rec["completions"]):
        text = c["text"]
        reasoning, final = split_completion(text)
        answer = last_boxed(final) if reasoning is not None else None
        if c["finish_reason"] != "stop":
            outcome = "truncated"
        elif reasoning is None:
            outcome = "no_think"
        elif answer is None:
            outcome = "no_boxed"
        elif not answers_match(answer, rec["answer"]):
            outcome = "wrong"
        elif CJK.search(text):
            outcome = "non_english"
        else:
            outcome = "correct"
        out.append({"k": k, "outcome": outcome, "tokens": c["tokens"], "text": text if outcome == "correct" else None})
    return {"key": rec["key"], "source": rec["source"], "difficulty": rec.get("difficulty"), "style": rec["style"],
            "answer": rec["answer"], "prompt_tokens": rec["prompt_tokens"], "grades": out}


def summary(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    return {"n": len(v), "mean": round(statistics.fmean(v)), "median": v[len(v) // 2], "p90": v[int(0.9 * (len(v) - 1))],
            "max": v[-1], "over_8k": sum(x > 8192 for x in v), "over_16k": sum(x > 16384 for x in v),
            "over_32k": sum(x > 32768 for x in v)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", nargs="+", type=Path, required=True, help="cartelle di replay_generate.py")
    ap.add_argument("--deepscaler", type=Path, required=True)
    ap.add_argument("--deepmath", nargs="+", type=Path, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--prefix", default="replay")
    ap.add_argument("--per-shard", type=int, default=5000)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("replay_select.py va eseguito in un job SLURM")
    t0 = time.time()
    problems = {p["key"]: p for p in load_problems(args.deepscaler, args.deepmath)}
    parts = sorted(f for d in args.gen for f in glob.glob(str(d / "part-*.jsonl")))
    if not parts:
        sys.exit(f"nessuna parte in {args.gen}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.shards.mkdir(parents=True, exist_ok=True)

    outcomes, lengths = Counter(), defaultdict(list)
    per_source = defaultdict(lambda: [0, 0])  # [problemi con una soluzione giusta, problemi]
    per_difficulty = defaultdict(lambda: [0, 0])
    kept, kept_tokens, too_long, n_problems = 0, 0, 0, 0
    shard_idx, in_shard, tar, fh, hw = 0, 0, None, None, None
    shard_list = []

    def close():
        nonlocal tar, fh, hw
        if tar is None:
            return
        tar.close()
        fh.close()
        final = args.shards / f"{args.prefix}-{shard_idx:05d}.tar"
        os.replace(final.with_suffix(".tar.tmp"), final)
        shard_list.append({"shard": final.name, "samples": in_shard, "bytes": hw.n, "sha256": hw.sha.hexdigest()})
        tar = fh = hw = None

    with Pool(args.workers) as pool, open(args.out / "selected.jsonl", "w") as sel:
        for part in parts:
            with open(part) as f:
                for item in pool.imap(grade, f, chunksize=4):
                    n_problems += 1
                    p = problems[item["key"]]
                    prompt = replay_prompt(p["question"], item["style"])
                    solved = False
                    for g in item["grades"]:
                        outcomes[g["outcome"]] += 1
                        lengths[g["outcome"]].append(g["tokens"])
                        lengths[f"{item['source']}:{g['outcome']}"].append(g["tokens"])
                        if g["outcome"] != "correct":
                            continue
                        solved = True
                        if item["prompt_tokens"] + g["tokens"] + TEMPLATE_TOKENS > args.max_len:
                            too_long += 1
                            continue
                        if tar is None:
                            fh = open((args.shards / f"{args.prefix}-{shard_idx:05d}.tar").with_suffix(".tar.tmp"), "wb")
                            hw = HashingWriter(fh)
                            tar = tarfile.open(fileobj=hw, mode="w", format=tarfile.USTAR_FORMAT)
                        name = f"{item['key']}-s{g['k']}"
                        meta = {"key": item["key"], "source": item["source"], "difficulty": item["difficulty"],
                                "style": item["style"], "answer": item["answer"], "tokens": g["tokens"],
                                "prompt": prompt, "solution": g["text"]}
                        add_member(tar, f"{name}.json", json.dumps(meta, ensure_ascii=False).encode())
                        sel.write(json.dumps({k: v for k, v in meta.items() if k not in ("prompt", "solution")}) + "\n")
                        kept += 1
                        kept_tokens += item["prompt_tokens"] + g["tokens"]
                        in_shard += 1
                        if in_shard >= args.per_shard:
                            close()
                            shard_idx, in_shard = shard_idx + 1, 0
                    per_source[item["source"]][1] += 1
                    per_source[item["source"]][0] += solved
                    if item["difficulty"] is not None:
                        per_difficulty[item["difficulty"]][1] += 1
                        per_difficulty[item["difficulty"]][0] += solved
        close()

    stats = {"parts": len(parts), "problems": n_problems, "outcomes": outcomes, "kept": kept,
             "correct_too_long": too_long, "max_len": args.max_len,
             "kept_tokens": kept_tokens,
             "solved_by_source": {s: {"solved": a, "problems": b, "rate": round(a / max(1, b), 3)} for s, (a, b) in per_source.items()},
             "solved_by_difficulty": {str(d): {"solved": a, "problems": b, "rate": round(a / max(1, b), 3)}
                                      for d, (a, b) in sorted(per_difficulty.items())},
             "tokens": {k: summary(v) for k, v in sorted(lengths.items())}, "shards": shard_list,
             "seconds": round(time.time() - t0, 1)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "shards"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
