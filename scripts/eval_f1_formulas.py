#!/usr/bin/env python3
"""Criterio di F1: trascrizione di formule renderizzate (UniMER-Test) con un adapter.

Da eseguire in un job con GPU (slurm/eval_f1_formulas.sbatch). Per ogni sottoinsieme (default spe e cpe,
formule stampate semplici e complesse) prende i primi --per-subset campioni dello shard di test
(scripts/shard_unimer.py), con la stessa richiesta e lo stesso prefill del training
(data.generation_prompt), e genera in greedy fino a <|endoftext|> o --max-new-tokens. Metriche:
- exact: predizione e riferimento uguali a meno degli spazi (latex.same_latex);
- token_acc: 1 - distanza di edit sui token LaTeX / token del riferimento (al minimo 0).
Uscite: OUT.json con le metriche per sottoinsieme e OUT.jsonl con un campione per riga.

    srun $RUN_PYTHON -u -B scripts/eval_f1_formulas.py --config CONFIG --adapter DIR --test-dir DIR --out OUT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_f1 import load_config, to_cuda  # noqa: E402  (stessa cartella scripts/)

from limite_vl.collator import PackedCollator  # noqa: E402
from limite_vl.data import generation_prompt, iter_shard  # noqa: E402
from limite_vl.latex import latex_tokens  # noqa: E402
from limite_vl.modeling import LimiteVLForConditionalGeneration, VLConfig  # noqa: E402
from limite_vl.processing import LimiteVLProcessor  # noqa: E402


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="config di training (decoder, vision, attn)")
    ap.add_argument("--adapter", required=True, help="cartella dell'adapter, oppure 'random'")
    ap.add_argument("--test-dir", required=True, type=Path)
    ap.add_argument("--subsets", nargs="+", default=["spe", "cpe"])
    ap.add_argument("--per-subset", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    processor = LimiteVLProcessor.from_pretrained(cfg["decoder"])
    torch.manual_seed(cfg["seed"])
    model = LimiteVLForConditionalGeneration.from_pretrained(
        VLConfig(decoder_path=cfg["decoder"], vision_path=cfg["vision"], vision_precision=cfg.get("vision_precision", "tf32")), dtype=torch.bfloat16,
        adapter_dir=None if args.adapter == "random" else args.adapter)
    model.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_out = open(args.out.with_suffix(".jsonl"), "w")
    summary = {"adapter": args.adapter, "config": str(args.config), "max_new_tokens": args.max_new_tokens, "subsets": {}}
    for subset in args.subsets:
        shard = args.test_dir / f"unimer-test-{subset}.tar"
        if not shard.is_file():
            sys.exit(f"shard di test mancante: {shard}")
        exact, accs, truncated, t0 = 0, [], 0, time.time()
        n = 0
        for key, files in iter_shard(shard):
            if n >= args.per_subset:
                break
            example, reference = generation_prompt(processor, "unimer_formula", key, files)
            batch = to_cuda(PackedCollator(len(example))([[example]]))
            with torch.no_grad():
                out = model.generate(batch["input_ids"], pixel_values=batch["pixel_values"],
                                     pixel_attention_mask=batch["pixel_attention_mask"],
                                     spatial_shapes=batch["spatial_shapes"], do_sample=False,
                                     max_new_tokens=args.max_new_tokens, use_cache=True)
            new = out[0, len(example):].tolist()
            hit_eos = processor.tokenizer.eos_token_id in new
            truncated += not hit_eos
            prediction = processor.tokenizer.decode(new, skip_special_tokens=True).strip()
            ref_tokens, pred_tokens = latex_tokens(reference), latex_tokens(prediction)
            same = ref_tokens == pred_tokens
            acc = max(0.0, 1 - edit_distance(pred_tokens, ref_tokens) / max(1, len(ref_tokens)))
            exact += same
            accs.append(acc)
            rows_out.write(json.dumps({"subset": subset, "key": key, "exact": same, "token_acc": acc,
                                       "eos": hit_eos, "reference": reference, "prediction": prediction},
                                      ensure_ascii=False) + "\n")
            n += 1
        summary["subsets"][subset] = {"samples": n, "exact": exact / max(1, n), "token_acc": sum(accs) / max(1, n),
                                      "truncated": truncated, "seconds": round(time.time() - t0, 1)}
        print(f"{subset}: {n} campioni, esatte {exact}/{n}, token_acc {sum(accs) / max(1, n):.3f}, "
              f"senza EOS {truncated} ({time.time() - t0:.0f} s)", flush=True)
    total = sum(v["samples"] for v in summary["subsets"].values())
    summary["exact"] = sum(v["exact"] * v["samples"] for v in summary["subsets"].values()) / max(1, total)
    summary["token_acc"] = sum(v["token_acc"] * v["samples"] for v in summary["subsets"].values()) / max(1, total)
    args.out.write_text(json.dumps(summary, indent=1))
    rows_out.close()
    print(f"totale: esatte {summary['exact']:.3f}, token_acc {summary['token_acc']:.3f}; risultati in {args.out}")


if __name__ == "__main__":
    main()
