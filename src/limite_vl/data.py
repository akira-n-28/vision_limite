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

from .processing import (DIAGRAM_MAX_PATCHES, F1_PREFILL, Example, LimiteVLProcessor, capped_size,
                         exact_patches, to_rgb)

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

# Richieste per le descrizioni dense (PixMo-Cap: foto; MMFineReason: immagini STEM), scelte allo stesso modo.
DETAILED_CAPTION_PROMPTS = (
    "Describe this image in detail.",
    "Give a detailed description of the image.",
    "What is shown in this image? Describe it thoroughly.",
    "Write a comprehensive description of everything visible in the image.",
    "Describe the image, including all the details you can see.",
    "Provide a thorough, detailed caption for this image.",
)
STEM_CAPTION_PROMPTS = (
    "Describe the figure in detail.",
    "Describe everything shown in this figure, including labels and values.",
    "Give a detailed description of this diagram.",
    "What does this figure show? Describe all of its components.",
    "Write a precise, detailed description of the image, listing every labelled element.",
    "Describe this mathematical figure thoroughly.",
)

# Richieste per la trascrizione delle formule di UniMER, scelte allo stesso modo.
FORMULA_PROMPTS = (
    "Transcribe the formula in the image into LaTeX.",
    "Write the LaTeX code for this formula.",
    "Convert the mathematical expression in the image to LaTeX.",
    "What is the LaTeX source of this expression?",
    "Give the LaTeX for the formula shown.",
    "Transcribe this equation in LaTeX.",
    "Read the formula in the image and write it in LaTeX.",
    "Write this expression as LaTeX code.",
)


@dataclass
class Sample:
    messages: list[dict]  # l'ultimo è l'assistente (content = target, prefill facoltativo)
    image: Image.Image | None  # None: esempio di solo testo (replay testuale)
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


def pick_prompt(key: str, prompts: tuple[str, ...]) -> str:
    """Richiesta deterministica per chiave (stessa scelta in ogni processo e a ogni epoca)."""
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return prompts[int.from_bytes(digest, "little") % len(prompts)]


def mavis_caption_prompt(key: str) -> str:
    return pick_prompt(key, MAVIS_CAPTION_PROMPTS)


def mavis_caption_sample(key: str, files: dict[str, bytes]) -> Sample:
    """MAVIS-Geometry (scripts/shard_mavis_geometry.py): PNG caption.jpg già a 2304 patch, didascalia
    di text_en come target, richiesta scelta dalla chiave."""
    meta = json.loads(files["json"])
    if meta.get("variant") != "caption":
        raise ValueError(f"{key}: variante {meta.get('variant')!r}, per le didascalie serve 'caption'")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": mavis_caption_prompt(key)}]},
                {"role": "assistant", "content": meta["text_en"]["caption"].strip(), "prefill": F1_PREFILL}]
    return Sample(messages, open_image(files["png"]), meta.get("max_patches", DIAGRAM_MAX_PATCHES))


def unimer_formula_sample(key: str, files: dict[str, bytes]) -> Sample:
    """UniMER (scripts/shard_unimer.py): PNG già alla dimensione finale (capped_size), LaTeX senza gli spazi
    fra token (latex.despace) come target, richiesta scelta dalla chiave."""
    meta = json.loads(files["json"])
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pick_prompt(key, FORMULA_PROMPTS)}]},
                {"role": "assistant", "content": meta["label"], "prefill": F1_PREFILL}]
    return Sample(messages, open_image(files["png"]), meta["max_patches"])


def pixmo_caption_sample(key: str, files: dict[str, bytes]) -> Sample:
    """PixMo-Cap (scripts/shard_parquet.py --kind pixmo): foto già a 1024 patch, didascalia densa (circa 200
    parole) come target."""
    meta = json.loads(files["json"])
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pick_prompt(key, DETAILED_CAPTION_PROMPTS)}]},
                {"role": "assistant", "content": meta["caption"], "prefill": F1_PREFILL}]
    return Sample(messages, open_image(files["jpg"]), meta["max_patches"])


# Ingrandimento massimo delle immagini STEM (2.0 nel run f1, job 29618). Con 4x le immagini di MMFineReason (lato
# corto mediano 136 px) passano da 120 a 450 token immagine di mediana, vicino ai benchmark (circa 540, job 29649), e
# il proiettore di F1 legge meglio le immagini piccole (formule: token giusti dal 45 al 53%, job 29647).
STEM_MAX_SCALE = 4.0


def stem_image(files: dict[str, bytes]) -> tuple[Image.Image, int]:
    """Immagine STEM originale di MMFineReason portata a capped_size (al più STEM_MAX_SCALE e al più 2304 patch),
    con il max_patches che il processor lascia invariato. Stessa regola in valutazione."""
    image = to_rgb(open_image(files["png"] if "png" in files else files["jpg"]))
    h, w = capped_size(image.height, image.width, STEM_MAX_SCALE, DIAGRAM_MAX_PATCHES)
    if (w, h) != image.size:
        image = image.resize((w, h), Image.BICUBIC)
    return image, exact_patches(h, w)


def stem_caption_sample(key: str, files: dict[str, bytes]) -> Sample:
    """MMFineReason (scripts/shard_parquet.py --kind mmfinereason): immagine STEM originale (stem_image),
    descrizione di Qwen3-VL-235B (markdown strutturato, circa 640 token) come target."""
    meta = json.loads(files["json"])
    image, max_patches = stem_image(files)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pick_prompt(key, STEM_CAPTION_PROMPTS)}]},
                {"role": "assistant", "content": meta["caption"], "prefill": F1_PREFILL}]
    return Sample(messages, image, max_patches)


# ---- F1c (limite_vl.f1c, limite_vl.synth): domande verificabili e testi solo visivi ----

def _qa_sample(key: str, questions: list[dict], image: Image.Image, max_patches: int) -> Sample:
    from .f1c import pick_questions, qa_messages

    prompt, answer = qa_messages(pick_questions(key, questions))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": answer, "prefill": F1_PREFILL}]
    return Sample(messages, image, max_patches)


def _mavis_meta(key: str, files: dict[str, bytes]) -> dict:
    meta = json.loads(files["json"])
    if meta.get("variant") != "caption":
        raise ValueError(f"{key}: variante {meta.get('variant')!r}, per le domande serve 'caption'")
    return meta


def mavis_qa_sample(key: str, files: dict[str, bytes]) -> Sample:
    """MAVIS-Geometry: da 1 a 5 domande verificabili ricavate dalla didascalia (f1c.mavis_questions), su caption.jpg
    che disegna tutte le misure citate."""
    from .f1c import mavis_questions

    meta = _mavis_meta(key, files)
    questions = mavis_questions(meta["text_en"]["caption"].strip(), meta["depth"], key)
    return _qa_sample(key, questions, open_image(files["png"]), meta.get("max_patches", DIAGRAM_MAX_PATCHES))


def mavis_count_caption_sample(key: str, files: dict[str, bytes]) -> Sample:
    """MAVIS-Geometry: didascalia preceduta da "The figure contains N shapes: ..." (f1c.mavis_count_caption)."""
    from .f1c import mavis_count_caption

    meta = _mavis_meta(key, files)
    caption = mavis_count_caption(meta["text_en"]["caption"].strip(), meta["depth"])
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": mavis_caption_prompt(key)}]},
                {"role": "assistant", "content": caption, "prefill": F1_PREFILL}]
    return Sample(messages, open_image(files["png"]), meta.get("max_patches", DIAGRAM_MAX_PATCHES))


def stem_visible_sample(key: str, files: dict[str, bytes]) -> Sample:
    """MMFineReason: descrizione di Qwen3-VL ridotta al contenuto visibile (f1c.visible_caption)."""
    from .f1c import visible_caption

    meta = json.loads(files["json"])
    caption = visible_caption(meta["caption"])
    if len(caption) < 40:
        raise ValueError(f"{key}: descrizione visibile vuota")
    image, max_patches = stem_image(files)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pick_prompt(key, STEM_CAPTION_PROMPTS)}]},
                {"role": "assistant", "content": caption, "prefill": F1_PREFILL}]
    return Sample(messages, image, max_patches)


def _unimer_line(key: str, meta: dict) -> int:
    from .f1c import key_rng

    if not meta.get("rows"):
        raise ValueError(f"{key}: nessuna riga (serve uno shard di scripts/shard_unimer.py --lines)")
    return key_rng(key + "/line").randrange(len(meta["rows"]))


def unimer_line_sample(key: str, files: dict[str, bytes]) -> Sample:
    """UniMER, formula su più righe: una riga ritagliata (ingrandita al più 2x entro 1024 patch) e la sua
    trascrizione."""
    meta = json.loads(files["json"])
    k = _unimer_line(key, meta)
    image = open_image(files["png"])
    lines = meta["lines"]
    y0, y1 = lines[k]
    pad = max(2, int(0.15 * (y1 - y0)))
    # al più metà dello spazio verso la riga vicina, per non includerne pezzi
    top = max(0, y0 - min(pad, (y0 - lines[k - 1][1]) // 2 if k else pad))
    bottom = min(image.height, y1 + min(pad, (lines[k + 1][0] - y1) // 2 if k + 1 < len(lines) else pad))
    crop = image.crop((0, top, image.width, bottom))
    h, w = capped_size(crop.height, crop.width, 2.0, 1024)
    if (w, h) != crop.size:
        crop = crop.resize((w, h), Image.BICUBIC)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pick_prompt(key, FORMULA_PROMPTS)}]},
                {"role": "assistant", "content": meta["rows"][k], "prefill": F1_PREFILL}]
    return Sample(messages, crop, exact_patches(h, w))


def unimer_line_select_sample(key: str, files: dict[str, bytes]) -> Sample:
    """UniMER, formula su più righe: l'immagine intera e la richiesta di trascrivere solo la riga k."""
    meta = json.loads(files["json"])
    k = _unimer_line(key, meta)
    prompt = f"Transcribe only line {k + 1} of the formula in the image into LaTeX."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": meta["rows"][k], "prefill": F1_PREFILL}]
    return Sample(messages, open_image(files["png"]), meta["max_patches"])


def synth_qa_sample(key: str, files: dict[str, bytes]) -> Sample:
    """Diagramma sintetico (scripts/synth_render.py): da 1 a 5 domande con risposta esatta per costruzione."""
    meta = json.loads(files["json"])
    return _qa_sample(key, meta["qa"], open_image(files["png"]), meta["max_patches"])


def f2_solution_sample(key: str, files: dict[str, bytes]) -> Sample:
    """F2 (scripts/f2_build.py): immagine STEM + domanda con l'istruzione del valutatore; target la soluzione
    di Violetto generata con il ponte testuale (`<think>...</think>` e risposta con \\boxed{}), senza prefill."""
    from .f2 import training_question

    meta = json.loads(files["json"])
    image, max_patches = stem_image(files)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": training_question(meta["question"])}]},
                {"role": "assistant", "content": meta["solution"]}]
    return Sample(messages, image, max_patches)


def text_replay_sample(key: str, files: dict[str, bytes]) -> Sample:
    """Replay testuale di F2 (scripts/replay_select.py): problema di testo con l'istruzione del valutatore, già
    composto come nella generazione (limite_vl.replay.replay_prompt); target la soluzione di Violetto
    (`<think>...</think>` e risposta con \\boxed{}), senza prefill e senza immagine."""
    meta = json.loads(files["json"])
    return Sample([{"role": "user", "content": meta["prompt"]}, {"role": "assistant", "content": meta["solution"]}], None)


TEXT_ONLY_KINDS = frozenset({"text_replay"})

SAMPLE_LOADERS: dict[str, Callable[[str, dict[str, bytes]], Sample]] = {
    "llava": llava_pretrain_sample,
    "mavis_caption": mavis_caption_sample,
    "unimer_formula": unimer_formula_sample,
    "pixmo_caption": pixmo_caption_sample,
    "stem_caption": stem_caption_sample,
    "mavis_qa": mavis_qa_sample,
    "mavis_count_caption": mavis_count_caption_sample,
    "stem_visible": stem_visible_sample,
    "unimer_line": unimer_line_sample,
    "unimer_line_select": unimer_line_select_sample,
    "synth_qa": synth_qa_sample,
    "f2_solution": f2_solution_sample,
    "text_replay": text_replay_sample,
}


IMAGE_MODES = ("real", "white", "none")


def build_example(processor: LimiteVLProcessor, kind: str, key: str, files: dict[str, bytes],
                  image_mode: str = "real") -> Example:
    """Esempio di training. image_mode: real; white = immagine bianca della stessa dimensione (stessi token
    immagine, test dell'immagine bianca); none = senza immagine, solo il testo."""
    sample = SAMPLE_LOADERS[kind](key, files)
    messages, images = sample.messages, [sample.image]
    if sample.image is None:  # solo testo: nessuna immagine in nessun modo
        images = []
    elif image_mode == "white":
        images = [Image.new("RGB", sample.image.size, "white")]
    elif image_mode == "none":
        messages = [{**m, "content": [c for c in m["content"] if c["type"] != "image"]}
                    if isinstance(m["content"], list) else m for m in messages]
        images = []
    if image_mode not in IMAGE_MODES:
        raise ValueError(f"image_mode {image_mode!r}: atteso uno di {IMAGE_MODES}")
    example = processor.encode(messages, images, max_patches=sample.max_patches)
    example.source, example.key = kind, key
    return example


def generation_prompt(processor: LimiteVLProcessor, kind: str, key: str,
                      files: dict[str, bytes], image_mode: str = "real") -> tuple[Example, str]:
    """Prompt per generate() (con il prefill dell'esempio) e risposta di riferimento. image_mode: real, oppure white
    (immagine bianca della stessa dimensione, come in build_example)."""
    if image_mode not in ("real", "white"):
        raise ValueError(f"image_mode {image_mode!r}: atteso real o white")
    sample = SAMPLE_LOADERS[kind](key, files)
    *history, answer = sample.messages
    images = [] if sample.image is None else [sample.image]
    if images and image_mode == "white":
        images = [Image.new("RGB", sample.image.size, "white")]
    example = processor.encode(history, images, max_patches=sample.max_patches,
                               add_generation_prompt=True, prefill=answer.get("prefill", ""))
    return example, answer["content"]


def llava_pretrain_example(processor: LimiteVLProcessor, files: dict[str, bytes],
                           max_patches: int | None = None) -> Example:
    sample = llava_pretrain_sample("", files)
    return processor.encode(sample.messages, [sample.image], max_patches=max_patches)
