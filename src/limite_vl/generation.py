"""Generazione a batch per il modello VL (valutazioni di F1c): prompt con padding a sinistra e attention_mask.

Limite ricava le posizioni dalla maschera (cumsum - 1) e gestisce la cache con padding (modeling_limite.py), quindi
basta allineare i prompt a destra. Le immagini vanno nello stesso ordine dei loro token <|image_pad|> nel batch
(riga per riga): l'hook degli embedding le inserisce con masked_scatter in quell'ordine. Il controllo contro la
generazione a un esempio alla volta è scripts/check_batched_generation.py.
"""
from __future__ import annotations

import numpy as np
import torch

from .processing import EOS_ID, Example


def left_pad(examples: list[Example], pad_id: int = EOS_ID) -> dict[str, torch.Tensor]:
    """Prompt allineati a destra: input_ids e attention_mask (B, L) e, se ci sono immagini, pixel_values uint8
    (N, P, 768), pixel_attention_mask (N, P), spatial_shapes (N, 2) nell'ordine degli esempi."""
    width = max(len(e) for e in examples)
    input_ids = np.full((len(examples), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(examples), width), dtype=np.int64)
    patches, grids = [], []
    for i, e in enumerate(examples):
        n = len(e)
        input_ids[i, width - n:] = e.input_ids
        mask[i, width - n:] = 1
        patches.extend(e.patches)
        grids.extend(e.grids)
    batch = {"input_ids": torch.from_numpy(input_ids), "attention_mask": torch.from_numpy(mask)}
    if patches:
        longest = max(p.shape[0] for p in patches)
        pixel_values = np.zeros((len(patches), longest, patches[0].shape[1]), dtype=np.uint8)
        pmask = np.zeros((len(patches), longest), dtype=np.int32)
        for i, p in enumerate(patches):
            pixel_values[i, : p.shape[0]] = p
            pmask[i, : p.shape[0]] = 1
        batch.update(pixel_values=torch.from_numpy(pixel_values), pixel_attention_mask=torch.from_numpy(pmask),
                     spatial_shapes=torch.tensor(grids, dtype=torch.long))
    return batch


@torch.no_grad()
def generate_batch(model, tokenizer, examples: list[Example], max_new_tokens: int,
                   sampling: dict | None = None, seed: int = 0) -> list[dict]:
    """[{text, eos, new_tokens}] per ogni esempio; greedy, oppure campionata con sampling (temperature, top_p)."""
    batch = {k: v.cuda(non_blocking=True) for k, v in left_pad(examples).items()}
    kwargs = {"do_sample": False}
    if sampling:
        torch.manual_seed(seed)
        kwargs = {"do_sample": True, "top_k": 0, **sampling}
    out = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"],
                         pixel_values=batch.get("pixel_values"), pixel_attention_mask=batch.get("pixel_attention_mask"),
                         spatial_shapes=batch.get("spatial_shapes"), max_new_tokens=max_new_tokens, use_cache=True,
                         pad_token_id=EOS_ID, **kwargs)
    results = []
    for row in out[:, batch["input_ids"].shape[1]:].tolist():
        eos = EOS_ID in row
        if eos:
            row = row[: row.index(EOS_ID)]
        results.append({"text": tokenizer.decode(row, skip_special_tokens=False), "eos": eos, "new_tokens": len(row)})
    return results


def generate_all(model, tokenizer, examples: list[Example], max_new_tokens: int, batch_size: int = 16,
                 sampling: dict | None = None, seed: int = 0) -> list[dict]:
    """Tutti gli esempi, a batch di prompt di lunghezza simile; risultati nell'ordine originale."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]))
    results: list[dict | None] = [None] * len(examples)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        for i, r in zip(idx, generate_batch(model, tokenizer, [examples[i] for i in idx], max_new_tokens,
                                            sampling, seed + start)):
            results[i] = r
    return results
