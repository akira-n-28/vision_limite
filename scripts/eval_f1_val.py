#!/usr/bin/env python3
"""Loss di validazione di F1 per sorgente con immagine vera, bianca o assente, per uno o più adapter.

Da eseguire in un job con GPU (slurm/f1_diag.sbatch). Usa gli stessi campioni di validazione del training
(i primi val_count dell'ultimo shard di ogni sorgente della config). Per ogni adapter, sorgente e modo:
- real: l'immagine del campione;
- white: un'immagine bianca della stessa dimensione (stesso numero di token immagine);
- none: nessuna immagine, solo il testo della richiesta.
Riporta la loss media sui token con label e la sua scomposizione: primo token della risposta, token
centrali, <|endoftext|> finale. Se real e white sono vicine, il modello non usa l'immagine.

    srun $RUN_PYTHON -u -B scripts/eval_f1_val.py --config CONFIG --adapter random --adapter nome=DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_f1 import load_config, to_cuda  # noqa: E402  (stessa cartella scripts/)

from limite_vl.collator import PackedCollator, iter_packed_rows  # noqa: E402
from limite_vl.data import IMAGE_MODES, build_example  # noqa: E402
from limite_vl.modeling import LimiteVLForConditionalGeneration, VLConfig  # noqa: E402
from limite_vl.processing import IGNORE, LimiteVLProcessor  # noqa: E402
from limite_vl.stream import make_sources, validation_samples  # noqa: E402

@torch.no_grad()
def token_losses(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CE per token con label, e maschere del primo e dell'ultimo token di ogni risposta."""
    labels = batch.pop("labels")
    hidden = model(**batch).hidden_states[:, :-1]
    y = labels[:, 1:]
    mask = y != IGNORE
    h, target = hidden[mask], y[mask]
    losses = [F.cross_entropy(model.logits(h[s:s + 1024]).float(), target[s:s + 1024], reduction="none")
              for s in range(0, h.shape[0], 1024)]
    ignore = torch.full_like(y[:, :1], IGNORE)
    first = (mask & (torch.cat([ignore, y[:, :-1]], dim=1) == IGNORE))[mask]
    last = (mask & (torch.cat([y[:, 1:], ignore], dim=1) == IGNORE))[mask]
    return torch.cat(losses), first, last


def evaluate(model, rows_by_key: dict, micro_rows: int, max_len: int) -> dict:
    collate = PackedCollator(max_len)
    out = {}
    for (source, mode), rows in rows_by_key.items():
        parts = [token_losses(model, to_cuda({k: v for k, v in collate(rows[i:i + micro_rows]).items()
                                              if k != "num_examples"}))
                 for i in range(0, len(rows), micro_rows)]
        loss = torch.cat([p[0] for p in parts])
        first = torch.cat([p[1] for p in parts])
        last = torch.cat([p[2] for p in parts])
        middle = ~(first | last)
        out.setdefault(source, {})[mode] = {
            "loss": loss.mean().item(), "first_token": loss[first].mean().item(),
            "middle_tokens": loss[middle].mean().item() if middle.any() else None,
            "eos": loss[last].mean().item(), "tokens": int(loss.numel()), "answers": int(first.sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--adapter", action="append", required=True,
                    help="'random' (proiettore appena inizializzato) oppure nome=cartella dell'adapter")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    processor = LimiteVLProcessor.from_pretrained(cfg["decoder"])
    sources = make_sources(cfg["sources"])
    rows_by_key = {}
    for s in sources:
        raw = validation_samples(s, cfg["val_count"])
        for mode in IMAGE_MODES:
            examples = [build_example(processor, s.kind, k, f, image_mode=mode) for k, f in raw]
            rows_by_key[(s.name, mode)] = list(iter_packed_rows(examples, cfg["max_len"], buffer_size=len(examples)))
    print(f"esempi pronti in {time.time() - t0:.0f} s: " +
          ", ".join(f"{s}/{m} {len(r)} righe" for (s, m), r in rows_by_key.items()), flush=True)

    torch.manual_seed(cfg["seed"])  # stessa inizializzazione del proiettore del passo 0 del training
    model = LimiteVLForConditionalGeneration.from_pretrained(
        VLConfig(decoder_path=cfg["decoder"], vision_path=cfg["vision"]), dtype=torch.bfloat16)
    model.language_model.config._attn_implementation = cfg["attn"]
    model.train()  # come la validazione del training (Limite non ha dropout)

    results = {}
    for spec in args.adapter:
        name, _, path = spec.partition("=")
        if name == "random":
            if results:
                sys.exit("'random' va indicato per primo, prima di caricare altri adapter")
        else:
            model.load_adapter(path)
        t = time.time()
        results[name] = evaluate(model, rows_by_key, cfg["micro_rows"], cfg["max_len"])
        for source, modes in results[name].items():
            for mode, r in modes.items():
                print(f"{name:12s} {source:6s} {mode:5s} loss {r['loss']:.3f}  primo {r['first_token']:.3f}  "
                      f"centrali {r['middle_tokens'] if r['middle_tokens'] is None else round(r['middle_tokens'], 3)}  "
                      f"eos {r['eos']:.3f}  ({r['tokens']} token, {r['answers']} risposte)", flush=True)
        print(f"{name}: {time.time() - t:.0f} s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": str(args.config), "adapters": args.adapter, "results": results}, indent=1))
    print(f"risultati: {args.out}")


if __name__ == "__main__":
    main()
