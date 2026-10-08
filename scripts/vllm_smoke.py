#!/usr/bin/env python3
"""Prova di generazione con vLLM e il plugin Limite su Violetto, in un job SLURM con 1 GPU.

Verifica che vLLM (build cu129) parta sul driver del nodo, che il plugin `limite` registri
LimiteForCausalLM e che il modello risponda in modo sensato a pochi problemi con risposta nota.
Non è una valutazione: quattro problemi facili campionati una volta non misurano l'accuratezza.

Offline: modello, tokenizer, chat template e generation_config.json dallo snapshot locale.
Impostazioni come `vllm serve` nel valutatore ufficiale (limite-evals/serve.py): dtype del
checkpoint, max_model_len dalla config, gpu_memory_utilization 0.90; temperature e top_p
dai default del checkpoint (0.6 e 0.95), registrati nel risultato.

Risultati in JSON in --out, anche in caso di errore.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

PROBLEMS = [
    ("Solve for $x$: $2x + 3 = 11$.", "4"),
    ("What is the sum of the first $100$ positive integers?", "5050"),
    ("Find the remainder when $2^{100}$ is divided by $7$.", "2"),
    ("How many positive divisors does $360$ have?", "24"),
]


def last_boxed(text: str) -> str | None:
    """Contenuto dell'ultimo \\boxed{...}, rispettando le graffe annidate."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def normalize(ans: str | None) -> str | None:
    return None if ans is None else re.sub(r"[\s$]", "", ans).rstrip(".")


def run(args: argparse.Namespace, result: dict) -> None:
    import torch
    import vllm
    from vllm import LLM, ModelRegistry, SamplingParams

    # Niente CUDA nel processo principale prima dell'engine: vLLM avvia i propri processi.
    result["env"] = {"vllm": vllm.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                     "model_path": args.model, "enforce_eager": args.enforce_eager,
                     "max_tokens": args.max_tokens, "seed": args.seed}

    t0 = time.time()
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=0.90, enforce_eager=args.enforce_eager)
    result["engine"] = {
        "init_seconds": round(time.time() - t0, 1),
        "limite_registered": "LimiteForCausalLM" in ModelRegistry.get_supported_archs(),
        "max_model_len": llm.llm_engine.model_config.max_model_len,
    }
    print("engine:", result["engine"], flush=True)

    default = llm.get_default_sampling_params()
    params = SamplingParams(temperature=default.temperature, top_p=default.top_p, top_k=default.top_k,
                            max_tokens=args.max_tokens, seed=args.seed)
    result["sampling"] = {"temperature": params.temperature, "top_p": params.top_p, "top_k": params.top_k,
                          "max_tokens": params.max_tokens, "seed": params.seed}
    print("sampling:", result["sampling"], flush=True)

    conversations = [[{"role": "user", "content": problem}] for problem, _ in PROBLEMS]
    t0 = time.time()
    outputs = llm.chat(conversations, sampling_params=params, use_tqdm=False)
    gen_seconds = time.time() - t0

    rows = []
    for (problem, expected), out in zip(PROBLEMS, outputs):
        completion = out.outputs[0]
        got = normalize(last_boxed(completion.text))
        rows.append({"problem": problem, "expected": expected, "boxed": got, "correct": got == expected,
                     "finish_reason": completion.finish_reason, "output_tokens": len(completion.token_ids),
                     "text": completion.text})
        print(f"[{'ok' if got == expected else 'NO'}] atteso {expected}, boxed {got!r}, "
              f"{len(completion.token_ids)} token, fine: {completion.finish_reason}", flush=True)
    total_tokens = sum(r["output_tokens"] for r in rows)
    result["prompt_example"] = outputs[0].prompt
    result["problems"] = rows
    result["summary"] = {
        "correct": sum(r["correct"] for r in rows), "total": len(rows),
        "all_stopped": all(r["finish_reason"] == "stop" for r in rows),
        "generation_seconds": round(gen_seconds, 1), "output_tokens": total_tokens,
        "output_tokens_per_s": round(total_tokens / gen_seconds) if gen_seconds > 0 else None,
    }
    print("summary:", result["summary"], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="cartella locale dello snapshot (snapshots/<commit>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enforce-eager", action="store_true", help="senza torch.compile e CUDA graph (diagnosi)")
    args = ap.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("vllm_smoke.py va eseguito dentro un job SLURM")
    if os.environ.get("VLLM_PLUGINS") != "limite":
        sys.exit("impostare VLLM_PLUGINS=limite (il plugin registra LimiteForCausalLM)")
    if not Path(args.model, "model.safetensors").is_file():
        sys.exit(f"snapshot di Violetto mancante: {args.model}")

    result: dict = {"job_id": os.environ["SLURM_JOB_ID"]}
    try:
        run(args, result)
        ok = True
    except Exception as exc:  # noqa: BLE001 - l'errore va registrato nel JSON prima di uscire
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
        ok = False
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"risultati: {args.out}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
