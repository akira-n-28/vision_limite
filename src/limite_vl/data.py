"""Lettura degli shard tar in stile webdataset ed esempi F1 per tipo di sorgente.

Ogni tipo di sorgente ha un loader `(chiave, file del campione) -> Sample`: messaggi completi
(l'ultimo è la risposta dell'assistente con il suo prefill), immagine e budget di patch. Da un Sample
si costruiscono sia l'esempio di training (`build_example`) sia il prompt per la generazione
(`generation_prompt`: gli stessi messaggi senza la risposta, con lo stesso prefill).
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image

from .processing import DIAGRAM_MAX_PATCHES, F1_PREFILL, Example, LimiteVLProcessor

# Richieste per le didascalie di MAVIS-Geometry, scelte in modo deterministico dalla chiave del campione.
MAVIS_CAPTION_PROMPTS = (
    "Describe the diagram.",
    "Describe this geometric figure in detail.",
    "Write a precise description of the figure, including the labelled lengths and angles.",
    "What does this diagram show? Describe the shapes and their measurements.",
    "Give a detailed caption for this geometry diagram.",
    "Describe the shapes in the figure and how they are attached to each other.",
    "List the shapes in this diagram and the measurements marked on it.",
    "Provide a careful description of this figure.",
)


@dataclass
class Sample:
    messages: list[dict]  # l'ultimo è l'assistente (content = target, prefill facoltativo)
    image: Image.Image
    max_patches: int | None = None  # None = budget di default del processor


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


def open_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def llava_pretrain_messages(meta: dict) -> list[dict]:
    """LLaVA-Pretrain: il prompt contiene <image> (prima o dopo il testo); l'immagine va sempre prima.
    Didascalia come target F1, dopo il ragionamento vuoto F1_PREFILL (senza label)."""
    text = meta["prompt"].replace("<image>", "").strip()
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
            {"role": "assistant", "content": meta["caption"], "prefill": F1_PREFILL}]


def llava_pretrain_sample(key: str, files: dict[str, bytes]) -> Sample:
    return Sample(llava_pretrain_messages(json.loads(files["json"])), open_image(files["jpg"]))


def mavis_caption_prompt(key: str) -> str:
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return MAVIS_CAPTION_PROMPTS[int.from_bytes(digest, "little") % len(MAVIS_CAPTION_PROMPTS)]


def mavis_caption_sample(key: str, files: dict[str, bytes]) -> Sample:
    """MAVIS-Geometry (scripts/shard_mavis_geometry.py): PNG caption.jpg già a 2304 patch, didascalia
    di text_en come target, richiesta scelta dalla chiave."""
    meta = json.loads(files["json"])
    if meta.get("variant") != "caption":
        raise ValueError(f"{key}: variante {meta.get('variant')!r}, per le didascalie serve 'caption'")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": mavis_caption_prompt(key)}]},
                {"role": "assistant", "content": meta["text_en"]["caption"].strip(), "prefill": F1_PREFILL}]
    return Sample(messages, open_image(files["png"]), meta.get("max_patches", DIAGRAM_MAX_PATCHES))


SAMPLE_LOADERS: dict[str, Callable[[str, dict[str, bytes]], Sample]] = {
    "llava": llava_pretrain_sample,
    "mavis_caption": mavis_caption_sample,
}


IMAGE_MODES = ("real", "white", "none")


def build_example(processor: LimiteVLProcessor, kind: str, key: str, files: dict[str, bytes],
                  image_mode: str = "real") -> Example:
    """Esempio di training. image_mode: real; white = immagine bianca della stessa dimensione (stessi token
    immagine, test dell'immagine bianca); none = senza immagine, solo il testo."""
    sample = SAMPLE_LOADERS[kind](key, files)
    messages, images = sample.messages, [sample.image]
    if image_mode == "white":
        images = [Image.new("RGB", sample.image.size, "white")]
    elif image_mode == "none":
        messages = [{**m, "content": [c for c in m["content"] if c["type"] != "image"]}
                    if isinstance(m["content"], list) else m for m in messages]
        images = []
    elif image_mode != "real":
        raise ValueError(f"image_mode {image_mode!r}: atteso uno di {IMAGE_MODES}")
    example = processor.encode(messages, images, max_patches=sample.max_patches)
    example.source, example.key = kind, key
    return example


def generation_prompt(processor: LimiteVLProcessor, kind: str, key: str,
                      files: dict[str, bytes]) -> tuple[Example, str]:
    """Prompt per generate() (con il prefill dell'esempio) e risposta di riferimento."""
    sample = SAMPLE_LOADERS[kind](key, files)
    *history, answer = sample.messages
    example = processor.encode(history, [sample.image], max_patches=sample.max_patches,
                               add_generation_prompt=True, prefill=answer.get("prefill", ""))
    return example, answer["content"]


def llava_pretrain_example(processor: LimiteVLProcessor, files: dict[str, bytes],
                           max_patches: int | None = None) -> Example:
    sample = llava_pretrain_sample("", files)
    return processor.encode(sample.messages, [sample.image], max_patches=max_patches)
