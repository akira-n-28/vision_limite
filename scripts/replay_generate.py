#!/usr/bin/env python3
"""F2, replay testuale: Violetto risolve problemi di testo (DeepScaleR, DeepMath-103K; limite_vl.replay) con il
budget e il campionamento della valutazione (contesto nativo meno 4096 token di riserva, temperature e top_p dai
default del checkpoint), così le tracce conservano la lunghezza dei ragionamenti di Violetto. Le soluzioni di
training del ponte visivo stanno sotto 8192 token e le ablation 29678/29679 hanno perso punti di AIME proprio sui
problemi lunghi (job 29696).

Da eseguire SOLO in un job SLURM con GPU (slurm/replay_generate.sbatch; VLLM_PLUGINS=limite). Con --select N
sceglie N problemi (limite_vl.replay.choose: --dsr-frac da DeepScaleR, il resto da DeepMath con difficulty >=
--min-difficulty) saltando le chiavi di --exclude (decontaminazione, data/decontam/text/replay/exclude_final.txt)
e i problemi delle generazioni in --exclude-gen. Prompt: limite_vl.replay.replay_prompt con le due istruzioni del
valutatore, scelte dalla chiave.

Uscita in --out come scripts/f2_generate.py: plan.json, keys.txt e part-NNNNN.jsonl (chiave, sorgente, risposta,
difficoltà, stile del prompt, token del prompt, generazioni con testo, finish_reason e token), scritte in `.tmp` e
rinominate; un nuovo avvio con la stessa --out riprende dalle parti mancanti e non inizia una parte che finirebbe a
meno di --margin-min minuti dalla scadenza.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import generated_keys  # noqa: E402
from limite_vl.replay import AIME_INSTRUCTION, MATH500_INSTRUCTION, choose, load_problems, prompt_style, replay_prompt  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_f1 import seconds_left_in_job  # noqa: E402

PROMPT_RESERVE = 4096  # come il valutatore: budget = contesto nativo - riserva del prompt (126976 su 131072)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="snapshot di Violetto")
    ap.add_argument("--deepscaler", type=Path, required=True)
    ap.add_argument("--deepmath", nargs="+", type=Path, required=True)
    ap.add_argument("--select", type=int, required=True)
    ap.add_argument("--dsr-frac", type=float, default=0.5)
    ap.add_argument("--min-difficulty", type=float, default=6.0)
    ap.add_argument("--exclude", nargs="*", default=[], help="file di chiavi da saltare (decontaminazione)")
    ap.add_argument("--exclude-gen", nargs="*", type=Path, default=[], help="generazioni precedenti del replay")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, help="default: contesto del modello meno 4096")
    ap.add_argument("--part-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin-min", type=float, default=30)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("replay_generate.py va eseguito in un job SLURM")
    if not args.plan_only and os.environ.get("VLLM_PLUGINS") != "limite":
        sys.exit("impostare VLLM_PLUGINS=limite (il plugin registra LimiteForCausalLM)")
    t_start = time.time()
    left = seconds_left_in_job()
    deadline = t_start + left - args.margin_min * 60 if left else None

    exclude: set[str] = set()
    for path in args.exclude:
        exclude.update(line.strip() for line in open(path) if line.strip())
    n_decontam = len(exclude)
    for run_dir in args.exclude_gen:
        if run_dir.resolve() == args.out.resolve():
            sys.exit("--exclude-gen non può contenere la cartella --out")
        exclude |= generated_keys(run_dir)
    problems = choose(load_problems(args.deepscaler, args.deepmath), args.select, args.dsr_frac,
                      args.min_difficulty, args.seed, exclude)
    if len(problems) < args.select:
        print(f"attenzione: {len(problems)} problemi disponibili su {args.select} richiesti", flush=True)
    parts = [problems[i:i + args.part_size] for i in range(0, len(problems), args.part_size)]
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(args.model) / "config.json").read_text())
    context = int(config["max_position_embeddings"])
    max_tokens = args.max_tokens or context - PROMPT_RESERVE
    plan = {"deepscaler": args.deepscaler.name, "deepmath": sorted(p.name for p in args.deepmath),
            "select": args.select, "dsr_frac": args.dsr_frac, "min_difficulty": args.min_difficulty,
            "exclude": [str(p) for p in args.exclude], "exclude_gen": [str(p) for p in args.exclude_gen],
            "part_size": args.part_size, "samples": args.samples, "max_tokens": max_tokens, "seed": args.seed,
            "instructions": [AIME_INSTRUCTION, MATH500_INSTRUCTION],
            "keys_digest": hashlib.sha256("\n".join(p["key"] for p in problems).encode()).hexdigest()}
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if {k: old.get(k) for k in plan} != plan:
            sys.exit(f"piano diverso da quello in {plan_path}: usare un'altra cartella --out")
    else:
        stats = {"problems": len(problems), "parts": len(parts), "decontam_keys": n_decontam,
                 "by_source": {s: sum(p["source"] == s for p in problems) for s in ("deepscaler", "deepmath")},
                 "by_style": {s: sum(prompt_style(p["key"]) == s for p in problems) for s in ("aime", "math500")}}
        plan_path.write_text(json.dumps({**plan, **stats}, indent=1))
    keys_path = args.out / "keys.txt"
    if not keys_path.exists():
        keys_path.write_text("".join(f"{p['key']}\n" for p in problems))
    todo = [i for i in range(len(parts)) if not (args.out / f"part-{i:05d}.jsonl").exists()]
    print(f"{len(problems)} problemi in {len(parts)} parti, da fare {len(todo)}; max_tokens {max_tokens} "
          f"({time.time() - t_start:.0f} s)", flush=True)
    if not todo or args.plan_only:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=args.seed, gpu_memory_utilization=0.90,
              max_model_len=context)
    tok = llm.get_tokenizer()
    default = llm.get_default_sampling_params()
    params = SamplingParams(n=args.samples, temperature=default.temperature, top_p=default.top_p, top_k=default.top_k,
                            max_tokens=max_tokens, seed=args.seed)
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
        styles = [prompt_style(p["key"]) for p in part]
        prompts = [tok.apply_chat_template([{"role": "user", "content": replay_prompt(p["question"], s)}],
                                           tokenize=False, add_generation_prompt=True) for p, s in zip(part, styles)]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        n_out, n_stop = 0, 0
        path = args.out / f"part-{i:05d}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as fh:
            for p, s, out in zip(part, styles, outputs):
                completions = [{"text": c.text, "finish_reason": c.finish_reason, "tokens": len(c.token_ids)}
                               for c in out.outputs]
                n_out += sum(c["tokens"] for c in completions)
                n_stop += sum(c["finish_reason"] == "stop" for c in completions)
                fh.write(json.dumps({"key": p["key"], "source": p["source"], "answer": p["answer"],
                                     "difficulty": p["difficulty"], "topic": p["topic"], "style": s,
                                     "prompt_tokens": len(out.prompt_token_ids), "completions": completions},
                                    ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        dt = time.time() - t0
        durations.append(dt)
        n = len(part) * args.samples
        print(f"parte {i + 1}/{len(parts)}: {len(part)} problemi, {n_out} token generati in {dt:.0f} s "
              f"({n_out / dt:.0f} token/s, {n_out / n:.0f} token per soluzione, finite {n_stop}/{n})", flush=True)
    print(f"fine in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
