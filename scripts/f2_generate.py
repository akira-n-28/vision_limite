#!/usr/bin/env python3
"""F2, ponte testuale: Violetto risolve i problemi di MMFineReason leggendo la descrizione della figura.

Da eseguire SOLO in un job SLURM con GPU (slurm/f2_generate.sbatch; VLLM_PLUGINS=limite). Legge i problemi
dagli shard di MMFineReason (scripts/shard_parquet.py) oppure direttamente dai parquet dello snapshot
(--parquet, solo le colonne di testo: non servono le immagini), con domanda, risposta verificata e descrizione
di Qwen3-VL-235B. Salta quelli senza domanda, risposta o descrizione e quelli esclusi dalla decontaminazione.
Con --select N sceglie N problemi, --easy-frac fra quelli che Qwen3-VL-4B risolve sempre (Violetto ne risolve il
57% contro il 20% degli altri, job 29498) e il resto fra gli altri, in ordine pseudo-casuale fissato dal seed,
e per ognuno campiona --samples soluzioni con vLLM (temperature e top_p dai default del checkpoint, come il
valutatore). Prompt: limite_vl.f2.bridge_prompt.

Uscita in --out, a parti da --part-size problemi: part-NNNNN.jsonl con, per problema, chiave, sorgente,
risposta, pass_rate di Qwen3-VL-4B e le generazioni (testo, finish_reason, token). Ogni parte si scrive in
`.tmp` e si rinomina; un nuovo avvio con la stessa --out salta le parti già scritte (il piano dei problemi
dipende solo da shard, esclusioni e --limit, e si controlla con plan.json). Non inizia una parte che, con la
durata della più lunga già fatta nel job, finirebbe a meno di --margin-min minuti dalla scadenza: una parte
interrotta dal limite di tempo andrebbe persa (job 29514). Le soluzioni si scelgono poi con scripts/f2_select.py.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.f2 import BRIDGE_TEMPLATE, bridge_prompt  # noqa: E402
from limite_vl.mmfine import choose, load_texts, scan_problems  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_f1 import seconds_left_in_job  # noqa: E402


def load_problems(shards: list[str], exclude: set[str], limit: int | None) -> tuple[list[dict], dict]:
    problems, skipped = [], {"missing": 0, "excluded": 0}
    for shard in shards:
        for key, files in iter_shard(shard):
            meta = json.loads(files["json"])
            if not (meta.get("question") and meta.get("answer") and meta.get("caption")):
                skipped["missing"] += 1
                continue
            if key in exclude:
                skipped["excluded"] += 1
                continue
            problems.append({"key": key, "source_dataset": meta.get("source_dataset"), "question": meta["question"],
                             "answer": meta["answer"], "caption": meta["caption"], "qwen4b_pass_rate": meta.get("pass_rate")})
            if limit and len(problems) >= limit:
                return problems, skipped
    return problems, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="snapshot di Violetto")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shards", help="glob degli shard di MMFineReason")
    src.add_argument("--parquet", nargs="+", type=Path, help="parquet dello snapshot di MMFineReason")
    ap.add_argument("--select", type=int, help="con --parquet: quanti problemi scegliere")
    ap.add_argument("--easy-frac", type=float, default=0.5, help="frazione con pass_rate di Qwen3-VL-4B = 1.0")
    ap.add_argument("--exclude", nargs="*", default=[], help="file di chiavi da saltare (decontaminazione)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--part-size", type=int, default=2048)
    ap.add_argument("--limit", type=int, help="solo i primi N problemi (prova)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin-min", type=float, default=20,
                    help="minuti di scarto fra la fine prevista di una parte e la scadenza del job")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_generate.py va eseguito in un job SLURM")
    if os.environ.get("VLLM_PLUGINS") != "limite":
        sys.exit("impostare VLLM_PLUGINS=limite (il plugin registra LimiteForCausalLM)")
    t_start = time.time()
    left = seconds_left_in_job()
    deadline = t_start + left - args.margin_min * 60 if left else None

    exclude: set[str] = set()
    for path in args.exclude:
        exclude.update(line.strip() for line in open(path) if line.strip())
    if args.parquet:
        missing = [str(p) for p in args.parquet if not p.is_file()]
        if missing:
            sys.exit(f"parquet mancanti: {missing[:5]}")
        inputs = [p.name for p in args.parquet]
        scanned = scan_problems(args.parquet)
        chosen = choose(scanned, args.select or len(scanned), args.easy_frac, args.seed, exclude)
        problems = load_texts(chosen)
        skipped = {"scanned": len(scanned), "chosen": len(chosen), "empty_question": len(chosen) - len(problems),
                   "easy": sum(p["qwen4b_pass_rate"] == 1.0 for p in problems)}
    else:
        shards = sorted(glob.glob(args.shards))
        if not shards:
            sys.exit(f"nessuno shard per {args.shards}")
        inputs = [Path(s).name for s in shards]
        problems, skipped = load_problems(shards, exclude, args.limit)
    parts = [problems[i:i + args.part_size] for i in range(0, len(problems), args.part_size)]
    args.out.mkdir(parents=True, exist_ok=True)
    plan = {"inputs": inputs, "select": args.select, "easy_frac": args.easy_frac,
            "exclude": [str(p) for p in args.exclude], "limit": args.limit,
            "part_size": args.part_size, "samples": args.samples, "max_tokens": args.max_tokens, "seed": args.seed,
            "template": BRIDGE_TEMPLATE, "keys_digest": hashlib.sha256("\n".join(p["key"] for p in problems).encode()).hexdigest()}
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if {k: old.get(k) for k in plan} != plan:
            sys.exit(f"piano diverso da quello in {plan_path}: usare un'altra cartella --out")
    else:
        plan_path.write_text(json.dumps({**plan, "problems": len(problems), "parts": len(parts), "skipped": skipped}, indent=1))
    todo = [i for i in range(len(parts)) if not (args.out / f"part-{i:05d}.jsonl").exists()]
    print(f"{len(problems)} problemi in {len(parts)} parti, da fare {len(todo)}; saltati {skipped} "
          f"({time.time() - t_start:.0f} s)", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=args.seed, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()
    default = llm.get_default_sampling_params()
    params = SamplingParams(n=args.samples, temperature=default.temperature, top_p=default.top_p, top_k=default.top_k,
                            max_tokens=args.max_tokens, seed=args.seed)
    print(f"engine pronto in {time.time() - t_start:.0f} s; temperature {params.temperature}, top_p {params.top_p}, "
          f"max_tokens {params.max_tokens}, n {params.n}", flush=True)

    durations: list[float] = []
    for i in todo:
        expected = max(durations, default=0.0)
        if deadline and time.time() + expected > deadline:
            print(f"fermo prima della scadenza del job: parti rimaste {len([j for j in todo if j >= i])}, "
                  f"una parte richiede circa {expected / 60:.0f} min", flush=True)
            break
        t0 = time.time()
        part = parts[i]
        prompts = [tok.apply_chat_template([{"role": "user", "content": bridge_prompt(p["question"], p["caption"])}],
                                           tokenize=False, add_generation_prompt=True) for p in part]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        n_out = 0
        path = args.out / f"part-{i:05d}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as fh:
            for p, out in zip(part, outputs):
                completions = [{"text": c.text, "finish_reason": c.finish_reason, "tokens": len(c.token_ids)}
                               for c in out.outputs]
                n_out += sum(c["tokens"] for c in completions)
                fh.write(json.dumps({"key": p["key"], "source_dataset": p["source_dataset"], "answer": p["answer"],
                                     "qwen4b_pass_rate": p["qwen4b_pass_rate"], "prompt_tokens": len(out.prompt_token_ids),
                                     "completions": completions}, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        dt = time.time() - t0
        durations.append(dt)
        print(f"parte {i + 1}/{len(parts)}: {len(part)} problemi, {n_out} token generati in {dt:.0f} s "
              f"({n_out / dt:.0f} token/s, {n_out / (len(part) * args.samples):.0f} token per soluzione)", flush=True)
    print(f"fine in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
