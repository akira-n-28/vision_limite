#!/usr/bin/env python3
"""Precisione e costo dell'encoder SigLIP2 in vari modi, contro il riferimento fp32 di ogni immagine da sola.

Dopo il job 29445: in fp32 l'output non dipende dal padding (1e-5), in bf16 cambia del 5-13% fra immagine
da sola e in batch. Questo script misura, per le stesse immagini in batch, la distanza dal riferimento
(fp32, immagine da sola, matmul senza TF32) e il tempo per batch, nei modi:
- bf16: pesi e calcolo bf16 (come finora in training);
- autocast: pesi fp32 con torch.autocast(bfloat16): matmul bf16, norme, softmax e residuo fp32;
- tf32: pesi fp32, matmul TF32;
- fp32: pesi fp32, matmul fp32.
Metriche sull'ultimo hidden state delle patch vere: differenza relativa in norma, coseno medio per token,
e lo stesso dopo il proiettore di un adapter addestrato (se dato), che è ciò che entra nel decoder.

    srun $RUN_PYTHON -u -B scripts/check_vision_precision.py --config CONFIG [--adapter DIR] --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from check_vision_padding import batch_of  # noqa: E402  (stessa cartella scripts/)
from train_f1 import load_config  # noqa: E402

from limite_vl.data import build_example, iter_shard  # noqa: E402
from limite_vl.modeling import PixelShuffleProjector, load_vision_tower  # noqa: E402
from limite_vl.processing import LimiteVLProcessor  # noqa: E402
from limite_vl.stream import make_sources  # noqa: E402

MODES = {"bf16": (torch.bfloat16, False, False), "autocast": (torch.float32, True, False),
         "tf32": (torch.float32, False, True), "fp32": (torch.float32, False, False)}


def run(model, patches, grids, dtype, autocast) -> list[torch.Tensor]:
    longest = max(p.shape[0] for p in patches)
    pixel, mask, shapes = batch_of(patches, grids, longest, torch.int32, dtype)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else nullcontext()
    with torch.no_grad(), ctx:
        out = model(pixel_values=pixel, pixel_attention_mask=mask, spatial_shapes=shapes).last_hidden_state.float()
    return [out[i, : p.shape[0]] for i, p in enumerate(patches)]


def compare(ref: list[torch.Tensor], got: list[torch.Tensor]) -> dict:
    rel = [((g - r).norm() / r.norm()).item() for r, g in zip(ref, got)]
    cos = [F.cosine_similarity(g, r, dim=-1).mean().item() for r, g in zip(ref, got)]
    return {"rel_max": max(rel), "rel_mean": sum(rel) / len(rel), "cos_min": min(cos), "cos_mean": sum(cos) / len(cos)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--adapter", help="cartella di un adapter (proiettore addestrato) per misurare dopo il proiettore")
    ap.add_argument("--per-source", type=int, default=3)
    ap.add_argument("--timing-batch", type=int, default=32)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    processor = LimiteVLProcessor.from_pretrained(cfg["decoder"])
    patches, grids = [], []
    for s in make_sources(cfg["sources"]):
        for n, (key, files) in enumerate(iter_shard(s.shards[0])):
            if n >= args.per_source:
                break
            ex = build_example(processor, s.kind, key, files)
            patches.extend(ex.patches)
            grids.extend(ex.grids)
    timing_patches = (patches * (args.timing_batch // len(patches) + 1))[: args.timing_batch]
    timing_grids = (grids * (args.timing_batch // len(grids) + 1))[: args.timing_batch]

    projector = None
    if args.adapter:
        from safetensors.torch import load_file
        state = load_file(str(Path(args.adapter) / "vl_adapter.safetensors"))
        projector = PixelShuffleProjector(1152, 1280, 2560).cuda()
        projector.load_state_dict({k[len("projector."):]: v for k, v in state.items() if k.startswith("projector.")})

    def project(feats: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        with torch.no_grad():
            for f, g in zip(feats, grids):
                out.append(projector(f.unsqueeze(0), torch.tensor([g], device="cuda")).float())
        return out

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fp32_model = load_vision_tower(cfg["vision"], torch.float32).cuda()
    ref = [run(fp32_model, [p], [g], torch.float32, False)[0] for p, g in zip(patches, grids)]
    ref_proj = project(ref) if projector is not None else None
    bf16_model = load_vision_tower(cfg["vision"], torch.bfloat16).cuda()
    results = {}
    for name, (dtype, autocast, tf32) in MODES.items():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        model = bf16_model if dtype == torch.bfloat16 else fp32_model
        got = run(model, patches, grids, dtype, autocast)
        row = {"hidden": compare(ref, got)}
        if projector is not None:
            row["projector"] = compare(ref_proj, project(got))
        for _ in range(2):  # riscaldamento
            run(model, timing_patches, timing_grids, dtype, autocast)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            run(model, timing_patches, timing_grids, dtype, autocast)
        torch.cuda.synchronize()
        row["seconds_per_batch"] = (time.time() - t0) / 5
        results[name] = row
        print(f"{name:9s} hidden {row['hidden']}  " + (f"proiettore {row['projector']}  " if projector else "")
              + f"{row['seconds_per_batch']:.3f} s per batch di {args.timing_batch}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"risultati: {args.out}")


if __name__ == "__main__":
    main()
