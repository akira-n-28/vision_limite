"""Processor di Limite-VL: immagini in patch NaFlex a griglia pari, conversazioni in token con label.

Immagini. Patch 16 e pixel shuffle 2x2 danno 1 token ogni 32x32 pixel, quindi la griglia di patch
deve essere pari in entrambe le direzioni. Si riusa la ricerca della scala di SigLIP2 NaFlex
(`get_image_size_for_max_num_patches` di transformers) con passo 32 e un quarto del budget di patch:
lati multipli di 32, al più `max_patches` patch da 16, aspect ratio conservato come in NaFlex
(immagini piccole comprese, che vengono ingrandite fino al budget). Le patch restano uint8 nell'ordine
di SigLIP2 (riga per riga, canali in fondo); la normalizzazione (x/255 - 0.5) / 0.5 si fa in GPU.

Testo. La conversazione passa dal chat template ufficiale di Violetto (system prompt fisso); ogni
immagine diventa `<|vision_start|><|image_pad|><|vision_end|>` nel messaggio utente, prima del
testo, e `<|image_pad|>` si espande poi in N token. Le label coprono solo il contenuto dei turni
dell'assistente. Il formato della risposta segue quello che Violetto genera, non il template: il
ragionamento è `<think>...</think>` senza a capo (soluzioni del valutatore, job 29393) e l'ultimo
turno finisce con `<|endoftext|>` subito dopo il testo, senza `<|im_end|>\\n` (generazione greedy,
job 29400). Nei turni intermedi `<|im_end|>\\n` resta nell'input, come lo scrive il template nella
cronologia, ma senza label. Nelle didascalie di F1 il turno dell'assistente comincia con un
ragionamento vuoto dato in input senza label (F1_PREFILL).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

PATCH_SIZE = 16
MERGE = 2
UNIT = PATCH_SIZE * MERGE  # pixel per lato di un token
DEFAULT_MAX_PATCHES = 1024  # 256 token
DIAGRAM_MAX_PATCHES = 2304  # 576 token, diagrammi con testo piccolo
MAX_IMAGE_TOKENS = 768  # tetto rigido: 36 layer su 48 vedono solo 1024 token

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
IMAGE_PLACEHOLDER = VISION_START + IMAGE_PAD + VISION_END
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655
EOS_ID = 151643  # <|endoftext|>: fine generazione di Violetto, pad e inizio del chat template
# Ragionamento vuoto dato prima delle didascalie di F1 (scelta dell'utente, opzione B, 2026-10-09):
# Violetto comincia sempre con <think>, così la loss resta solo sui token che dipendono dall'immagine
# e il proiettore non impara a far saltare al decoder il suo formato.
F1_PREFILL = "<think></think>"
# Token aggiunti (non speciali) del formato di Violetto, ammessi solo nei testi dell'assistente.
THINK_TOKENS = ("<think>", "</think>")  # 151665, 151666
IM_END_ID = 151645
NEWLINE_ID = 198
IGNORE = -100


def target_size(height: int, width: int, max_patches: int = DEFAULT_MAX_PATCHES) -> tuple[int, int]:
    """(altezza, larghezza) in pixel, multipli di 32, con (h/16)·(w/16) <= max_patches."""
    from transformers.models.siglip2.image_processing_pil_siglip2 import get_image_size_for_max_num_patches

    per_token = MERGE * MERGE
    if max_patches % per_token or not 0 < max_patches // per_token <= MAX_IMAGE_TOKENS:
        raise ValueError(f"max_patches={max_patches}: serve un multiplo di 4 fino a {MAX_IMAGE_TOKENS * 4}")
    return get_image_size_for_max_num_patches(height, width, patch_size=UNIT, max_num_patches=max_patches // per_token)


def to_rgb(image: Image.Image) -> Image.Image:
    """RGB; le trasparenze (diagrammi PNG) vanno su fondo bianco invece che nero."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    return image.convert("RGB")


def image_to_patches(image: Image.Image, max_patches: int = DEFAULT_MAX_PATCHES) -> tuple[np.ndarray, tuple[int, int]]:
    """Patch uint8 (n, 16·16·3) nell'ordine di SigLIP2 e griglia (righe, colonne), entrambe pari."""
    image = to_rgb(image)
    height, width = target_size(image.height, image.width, max_patches)
    if (image.width, image.height) != (width, height):
        image = image.resize((width, height), Image.BILINEAR)  # resample=2 di preprocessor_config.json
    pixels = np.asarray(image, dtype=np.uint8)
    rows, cols = height // PATCH_SIZE, width // PATCH_SIZE
    patches = pixels.reshape(rows, PATCH_SIZE, cols, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4).reshape(rows * cols, -1)
    return np.ascontiguousarray(patches), (rows, cols)


def image_tokens(grid: tuple[int, int]) -> int:
    return (grid[0] // MERGE) * (grid[1] // MERGE)


@dataclass
class Example:
    """Un esempio tokenizzato: label = IGNORE fuori dai turni dell'assistente."""

    input_ids: np.ndarray  # int32
    labels: np.ndarray  # int32
    patches: list[np.ndarray] = field(default_factory=list)  # uint8 (n_i, 768), in ordine di apparizione
    grids: list[tuple[int, int]] = field(default_factory=list)
    source: str = ""  # tipo di sorgente e chiave del campione (data.build_example), per statistiche e test
    key: str = ""

    def __len__(self) -> int:
        return len(self.input_ids)


class LimiteVLProcessor:
    """Messaggi: {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ...}]} oppure
    content stringa; {"role": "assistant", "content": ..., "reasoning_content": ... (facoltativo)}."""

    def __init__(self, tokenizer, max_patches: int = DEFAULT_MAX_PATCHES):
        self.tokenizer = tokenizer
        self.max_patches = max_patches
        for token, expected in ((VISION_START, VISION_START_ID), (VISION_END, VISION_END_ID),
                                (IMAGE_PAD, IMAGE_PAD_ID), ("<|endoftext|>", EOS_ID), ("<|im_end|>", IM_END_ID)):
            if tokenizer.convert_tokens_to_ids(token) != expected:
                raise ValueError(f"{token}: id {tokenizer.convert_tokens_to_ids(token)}, atteso {expected}")
        if tokenizer("\n", add_special_tokens=False)["input_ids"] != [NEWLINE_ID]:
            raise ValueError(f"'\\n' non è il token {NEWLINE_ID}")
        self.special_tokens = sorted({t for t in tokenizer.all_special_tokens} |
                                     set(tokenizer.added_tokens_encoder), key=len, reverse=True)
        target_size(UNIT, UNIT, max_patches)  # valida il budget

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "LimiteVLProcessor":
        # La classe di tokenizer_config.json, non AutoTokenizer: la config di Violetto dichiara codice
        # remoto (auto_map) e AutoTokenizer chiederebbe se eseguirlo.
        import json
        from pathlib import Path

        from transformers import Qwen2Tokenizer

        declared = json.loads((Path(path) / "tokenizer_config.json").read_text()).get("tokenizer_class")
        if declared != "Qwen2Tokenizer":
            raise ValueError(f"tokenizer_class {declared!r}, atteso Qwen2Tokenizer")
        return cls(Qwen2Tokenizer.from_pretrained(path, local_files_only=True), **kwargs)

    def _check_text(self, text: str, allowed: tuple[str, ...] = ()) -> str:
        for token in self.special_tokens:
            if token in text and token not in allowed:
                raise ValueError(f"il testo contiene il token speciale {token!r}")
        return text

    def render(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Messaggi con contenuto stringa per il chat template e numero di immagini."""
        out, n_images = [], 0
        for message in messages:
            role, content = message["role"], message["content"]
            if isinstance(content, str):
                text = self._check_text(content, THINK_TOKENS if role == "assistant" else ())
            elif role == "user":
                parts, seen_text = [], False
                for item in content:
                    if item["type"] == "image":
                        if seen_text:
                            raise ValueError("immagine dopo il testo: il piano vuole l'immagine prima della domanda")
                        parts.append(IMAGE_PLACEHOLDER)
                        n_images += 1
                    elif item["type"] == "text":
                        parts.append(self._check_text(item["text"]))
                        seen_text = True
                    else:
                        raise ValueError(f"tipo di contenuto non supportato: {item['type']!r}")
                text = "".join(parts)
            else:
                raise ValueError(f"contenuto strutturato ammesso solo per l'utente, non per {role!r}")
            if message.get("reasoning_content"):
                if role != "assistant":
                    raise ValueError("reasoning_content ammesso solo per l'assistente")
                # Formato generato da Violetto, non quello del template (<think>\n...\n</think>\n\n).
                text = "<think>" + self._check_text(message["reasoning_content"]) + "</think>" + text
            out.append({"role": role, "content": text})
        return out, n_images

    def _ids(self, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        return self._text_ids(text)

    def _text_ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode(self, messages: list[dict], images: list = (), max_patches: int | None = None,
               add_generation_prompt: bool = False, prefill: str = "") -> Example:
        """Tokenizza una conversazione. `images`: PIL.Image oppure coppie (patch, griglia) già calcolate.

        L'ultimo turno dell'assistente si costruisce come in generazione: il prompt del template fino a
        `<|im_start|>assistant\\n`, poi l'eventuale `prefill` del messaggio (dato al modello, senza label,
        es. F1_PREFILL), poi il contenuto e `<|endoftext|>` con label. Prefill e contenuto si tokenizzano
        a parte, come i token che il modello genera dopo il prompt. Con add_generation_prompt=True
        (inferenza) il risultato finisce con il prompt e l'argomento `prefill`, senza label.
        """
        rendered, n_images = self.render(messages)
        if n_images != len(images):
            raise ValueError(f"{n_images} immagini nei messaggi, {len(images)} passate")
        final = not add_generation_prompt and bool(rendered) and rendered[-1]["role"] == "assistant"
        if any(m.get("prefill") for m in messages[: len(messages) - final]):
            raise ValueError("prefill ammesso solo nell'ultimo turno dell'assistente")
        if final:
            if prefill:
                raise ValueError("in training il prefill va nel messaggio dell'assistente")
            prefill = messages[-1].get("prefill", "")
        elif prefill and not add_generation_prompt:
            raise ValueError("prefill senza un turno dell'assistente da generare")
        history = rendered[:-1] if final else rendered
        ids = self._ids(history, add_generation_prompt=add_generation_prompt or final)
        labels = [IGNORE] * len(ids)
        if not add_generation_prompt:
            for i, message in enumerate(history):
                if message["role"] != "assistant":
                    continue
                start = self._ids(history[:i], add_generation_prompt=True)
                end = self._ids(history[: i + 1])
                if ids[: len(end)] != end or end[: len(start)] != start:
                    raise ValueError(f"turno {i}: la tokenizzazione del prefisso non è un prefisso di quella completa")
                if end[-2:] != [IM_END_ID, NEWLINE_ID]:
                    raise ValueError(f"turno {i}: il template non chiude il turno con <|im_end|>\\n")
                labels[len(start):len(end) - 2] = ids[len(start):len(end) - 2]  # <|im_end|>\n senza label
        if prefill:
            given = self._text_ids(self._check_text(prefill, THINK_TOKENS))
            ids += given
            labels += [IGNORE] * len(given)
        if final:
            content = self._text_ids(rendered[-1]["content"]) + [EOS_ID]  # Violetto chiude con EOS
            ids += content
            labels += content

        patches, grids = [], []
        for image in images:
            p, g = image if isinstance(image, tuple) else image_to_patches(image, max_patches or self.max_patches)
            patches.append(p)
            grids.append(g)
        if ids.count(IMAGE_PAD_ID) != len(grids) or ids.count(VISION_START_ID) != len(grids):
            raise ValueError("segnaposti delle immagini incoerenti dopo il chat template")
        out_ids, out_labels, k = [], [], 0
        for token, label in zip(ids, labels):
            if token == IMAGE_PAD_ID:
                n = image_tokens(grids[k])
                out_ids.extend([IMAGE_PAD_ID] * n)
                out_labels.extend([IGNORE] * n)
                k += 1
            else:
                out_ids.append(token)
                out_labels.append(label)
        return Example(np.asarray(out_ids, dtype=np.int32), np.asarray(out_labels, dtype=np.int32), patches, grids)
