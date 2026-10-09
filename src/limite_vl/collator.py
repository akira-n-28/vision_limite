"""Packing degli esempi in righe di lunghezza fissa e collate in tensori per LimiteVLForConditionalGeneration.

Ogni esempio è un documento a sé: position_ids ripartono da 0 a ogni esempio e transformers
(masking_utils.find_packed_sequence_indices, con attention_mask=None) ne ricava la maschera a blocchi,
così un esempio non vede quelli impacchettati prima. La coda di una riga è riempita con
<|endoftext|> come documento separato, con label ignorate. Le immagini seguono l'ordine di apparizione
dei loro token nella riga (righe in ordine, esempi in ordine), che è l'ordine in cui il modello
sostituisce i token <|image_pad|>.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Iterator

import numpy as np
import torch

from .processing import EOS_ID, IGNORE, Example


def pack_first_fit(lengths: list[int], max_len: int) -> list[list[int]]:
    """First-fit decreasing: gruppi di indici con somma delle lunghezze <= max_len.

    Gli indici più lunghi di max_len restano fuori. Dentro un gruppo gli indici sono in ordine crescente.
    """
    bins: list[list[int]] = []
    room: list[int] = []
    for i in sorted(range(len(lengths)), key=lambda j: (-lengths[j], j)):
        n = lengths[i]
        if n > max_len:
            continue
        for b, free in enumerate(room):
            if free >= n:
                bins[b].append(i)
                room[b] -= n
                break
        else:
            bins.append([i])
            room.append(max_len - n)
    return [sorted(b) for b in bins]


def iter_packed_rows(examples: Iterable[Example], max_len: int, buffer_size: int = 1024,
                     min_fill: float = 0.98, stats: Counter | None = None) -> Iterator[list[Example]]:
    """Righe impacchettate da un flusso di esempi, deterministiche a parità di ordine in ingresso.

    Raccoglie `buffer_size` esempi, li impacchetta e consegna le righe piene almeno `min_fill`;
    gli esempi delle altre restano nel buffer per il giro successivo. A fine flusso consegna tutto.
    `stats` (facoltativo) conta esempi, righe, token utili e scartati perché più lunghi di max_len.
    """
    stats = stats if stats is not None else Counter()
    buffer: list[Example] = []

    def emit(final: bool) -> Iterator[list[Example]]:
        nonlocal buffer
        bins = pack_first_fit([len(e) for e in buffer], max_len)
        fills = [sum(len(buffer[i]) for i in b) for b in bins]
        ready = [b for b, f in zip(bins, fills) if final or f >= min_fill * max_len]
        if not ready and bins:  # garantisce progresso con buffer di esempi lunghi
            ready = [bins[max(range(len(bins)), key=fills.__getitem__)]]
        used = {i for b in ready for i in b}
        for b in ready:
            stats["rows"] += 1
            stats["tokens"] += sum(len(buffer[i]) for i in b)
            yield [buffer[i] for i in b]
        buffer = [e for i, e in enumerate(buffer) if i not in used]

    for example in examples:
        stats["examples"] += 1
        if len(example) > max_len:
            stats["too_long"] += 1
            continue
        buffer.append(example)
        if len(buffer) >= buffer_size:
            yield from emit(final=False)
    while buffer:
        yield from emit(final=True)


class PackedCollator:
    """Righe di esempi -> tensori: input_ids, labels, position_ids (B, max_len) e, se ci sono immagini,
    pixel_values uint8 (N, P, 768), pixel_attention_mask (N, P), spatial_shapes (N, 2)."""

    def __init__(self, max_len: int, pad_id: int = EOS_ID):
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, rows: list[list[Example]]) -> dict[str, torch.Tensor]:
        shape = (len(rows), self.max_len)
        input_ids = np.full(shape, self.pad_id, dtype=np.int64)
        labels = np.full(shape, IGNORE, dtype=np.int64)
        position_ids = np.zeros(shape, dtype=np.int64)
        patches, grids, n_examples = [], [], 0
        for r, row in enumerate(rows):
            pos = 0
            for example in row:
                n = len(example)
                if pos + n > self.max_len:
                    raise ValueError(f"riga {r}: {pos + n} token oltre max_len={self.max_len}")
                input_ids[r, pos:pos + n] = example.input_ids
                labels[r, pos:pos + n] = example.labels
                position_ids[r, pos:pos + n] = np.arange(n)
                patches.extend(example.patches)
                grids.extend(example.grids)
                pos += n
                n_examples += 1
            position_ids[r, pos:] = np.arange(self.max_len - pos)  # coda: documento a sé
        batch = {"input_ids": torch.from_numpy(input_ids), "labels": torch.from_numpy(labels),
                 "position_ids": torch.from_numpy(position_ids), "num_examples": torch.tensor(n_examples)}
        if patches:
            longest = max(p.shape[0] for p in patches)
            pixel_values = np.zeros((len(patches), longest, patches[0].shape[1]), dtype=np.uint8)
            mask = np.zeros((len(patches), longest), dtype=np.int32)
            for i, p in enumerate(patches):
                pixel_values[i, : p.shape[0]] = p
                mask[i, : p.shape[0]] = 1
            batch.update(pixel_values=torch.from_numpy(pixel_values), pixel_attention_mask=torch.from_numpy(mask),
                         spatial_shapes=torch.tensor(grids, dtype=torch.long))
        return batch
