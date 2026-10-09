"""Lettura degli shard tar in stile webdataset (scripts/shard_llava_pretrain.py) ed esempi F1."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Iterator

from PIL import Image

from .processing import F1_PREFILL, Example, LimiteVLProcessor


def iter_shard(path: Path | str) -> Iterator[tuple[str, dict[str, bytes]]]:
    """(chiave, {estensione: byte}) per ogni campione, nell'ordine del tar (file consecutivi per chiave)."""
    key, files = None, {}
    with tarfile.open(path, "r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            stem, _, ext = member.name.rpartition("/")[2].partition(".")
            if stem != key and files:
                yield key, files
                files = {}
            key = stem
            files[ext] = tar.extractfile(member).read()
    if files:
        yield key, files


def llava_pretrain_messages(meta: dict) -> list[dict]:
    """LLaVA-Pretrain: il prompt contiene <image> (prima o dopo il testo); l'immagine va sempre prima.
    Didascalia come target F1, dopo il ragionamento vuoto F1_PREFILL (senza label)."""
    text = meta["prompt"].replace("<image>", "").strip()
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
            {"role": "assistant", "content": meta["caption"], "prefill": F1_PREFILL}]


def llava_pretrain_example(processor: LimiteVLProcessor, files: dict[str, bytes],
                           max_patches: int | None = None) -> Example:
    meta = json.loads(files["json"])
    with Image.open(io.BytesIO(files["jpg"])) as image:
        image.load()
        return processor.encode(llava_pretrain_messages(meta), [image], max_patches=max_patches)
