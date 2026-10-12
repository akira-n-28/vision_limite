#!/usr/bin/env python3
"""L'output di SigLIP2 NaFlex dipende dal padding? Controllo mirato dopo il job 29443 (gruppi più piccoli,
cioè meno padding, davano feature diverse dell'8.4%).

Da eseguire in un job con GPU (slurm/check_vision_padding.sbatch). Prende alcune immagini vere da ogni
sorgente (patch come in training), e per ogni combinazione di dtype (fp32, bf16), implementazione
dell'attenzione (sdpa, eager) e tipo della maschera (int32 come il collator, bool) confronta, immagine
per immagine, l'ultimo hidden state sulle patch vere fra:
- l'immagine da sola, senza padding (riferimento);
- il batch allineato alla più lunga;
- il batch con 512 patch di padding in più.

    srun $RUN_PYTHON -u -B scripts/check_vision_padding.py --vision SIGLIP2 --config CONFIG --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_f1 import load_config  # noqa: E402  (stessa cartella scripts/)

from limite_vl.data import build_example, iter_shard  # noqa: E402
from limite_vl.modeling import PIXEL_MEAN, PIXEL_STD, load_vision_tower  # noqa: E402
from limite_vl.processing import LimiteVLProcessor  # noqa: E402
from limite_vl.stream import make_sources  # noqa: E402


def batch_of(patches: list, grids: list, pad_to: int, mask_dtype: torch.dtype, dtype: torch.dtype):
    pixel = torch.zeros(len(patches), pad_to, patches[0].shape[1], dtype=torch.uint8)
    mask = torch.zeros(len(patches), pad_to, dtype=mask_dtype)
    for i, p in enumerate(patches):
        pixel[i, : p.shape[0]] = torch.from_numpy(p)
        mask[i, : p.shape[0]] = 1
    pixel = ((pixel.float() / 255.0 - PIXEL_MEAN) / PIXEL_STD).to(dtype).cuda()
    return pixel, mask.cuda(), torch.tensor(grids, dtype=torch.long).cuda()


@torch.no_grad()
def hidden(model, patches, grids, pad_to, mask_dtype, dtype) -> list[torch.Tensor]:
    pixel, mask, shapes = batch_of(patches, grids, pad_to, mask_dtype, dtype)
    out = model(pixel_values=pixel, pixel_attention_mask=mask, spatial_shapes=shapes).last_hidden_state.float()
    return [out[i, : p.shape[0]] for i, p in enumerate(patches)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--per-source", type=int, default=3)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    processor = LimiteVLProcessor.from_pretrained(cfg["decoder"])
    patches, grids, labels = [], [], []
    for s in make_sources(cfg["sources"]):
        for n, (key, files) in enumerate(iter_shard(s.shards[0])):
            if n >= args.per_source:
                break
            ex = build_example(processor, s.kind, key, files)
            patches.extend(ex.patches)
            grids.extend(ex.grids)
            labels.append(f"{s.name}/{key} ({ex.patches[0].shape[0]} patch)")
    longest = max(p.shape[0] for p in patches)
    print(f"{len(patches)} immagini, patch da {min(p.shape[0] for p in patches)} a {longest}", flush=True)

    results = {}
    for attn in ("sdpa", "eager"):
        for dtype in (torch.float32, torch.bfloat16):
            model = load_vision_tower(cfg["vision"], dtype, attn_implementation=attn).cuda()
            for mask_dtype in (torch.int32, torch.bool):
                name = f"{attn}/{str(dtype).split('.')[-1]}/{str(mask_dtype).split('.')[-1]}"
                alone = [hidden(model, [p], [g], p.shape[0], mask_dtype, dtype)[0] for p, g in zip(patches, grids)]
                row = {}
                for label, pad in (("allineato", longest), ("padding+512", longest + 512)):
                    batched = hidden(model, patches, grids, pad, mask_dtype, dtype)
                    rel = [((b - a).norm() / a.norm()).item() for a, b in zip(alone, batched)]
                    row[label] = rel
                    print(f"{name:22s} {label:12s} diff. relativa per immagine: "
                          + " ".join(f"{r:.1e}" for r in rel), flush=True)
                results[name] = row
            del model
            torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"images": labels, "results": results}, indent=1))
    print(f"immagini: {labels}\nrisultati: {args.out}")


if __name__ == "__main__":
    main()
