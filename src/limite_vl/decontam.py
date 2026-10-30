"""Decontaminazione dei dati di training contro i benchmark visivi (MathVista, MathVerse, MathVision, We-Math,
DynaMath): hash percettivo delle immagini e n-gram delle domande, come previsto dal piano.

- Immagini: pHash a 64 bit (imagehash, DCT 32x32 in scala di grigi), robusto a ridimensionamento e
  ricompressione. Due immagini sono "la stessa" se distano al più `radius` bit. La ricerca usa il principio
  dei cassetti: l'hash si divide in radius+1 blocchi, e due hash entro radius bit coincidono esattamente in
  almeno un blocco; si confrontano solo gli hash che condividono un blocco.
- Domande: testo normalizzato (minuscole, solo lettere e cifre, LaTeX ridotto ai nomi dei comandi), poi
  n-gram di parole. Una domanda di training è contaminata se condivide un n-gram con una domanda di un
  benchmark. Gli n-gram presenti in più di `max_df` testi diversi dei benchmark sono formule di rito
  (istruzioni come "Hint: Please answer the question and provide the correct option letter", aperture come
  "As shown in the figure"), non indizi di un problema preciso, e non contano. Le domande più corte di n
  parole ("Find x.") non si confrontano sul testo: decide l'immagine. Questa forma toglie operatori e segni: dà
  candidati, non identità (`exact_text` per i legami confermati).
- Convenzione delle immagini: indice, scansione e revisione normalizzano le immagini nello stesso modo, e un indice
  dichiara il suo (`IMAGE_CONVENTION`, `index_convention`).
"""
from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from PIL import Image

from .processing import to_rgb

# parole latine e cifre; ogni ideogramma CJK è una parola (senza, le domande in cinese si riducevano a numeri e
# lettere delle scelte e si somigliavano tutte: revisione del job 29572)
_WORD = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


# Convenzioni delle immagini per pHash e sha256 dei pixel. Un indice dichiara la sua nel campo "convenzione_immagini";
# scansione e revisione usano quella dell'indice, così due hash fatti in modi diversi non si confrontano mai. Un indice
# che non la dichiara va accompagnato da una convenzione indicata esplicitamente (index_convention): non si deduce.
# - "rgb-su-bianco/1": processing.to_rgb (trasparenze composte su bianco), poi pHash in scala di grigi; lo sha256 dei
#   pixel copre dimensioni e valori RGB. È quella di scripts/panel_index.py (anche di pannello-v1 e v2, che non la
#   dichiaravano) e dei nuovi indici di scripts/decontam.py.
# - "grigio-diretto/0": pHash su image.convert("L"), che ignora l'alfa (le zone trasparenti prendono il colore dei
#   pixel sottostanti, di solito nero). È quella degli indici storici p6 e f1eval e delle scansioni fatte prima; non ha
#   sha256 dei pixel.
IMAGE_CONVENTION = "rgb-su-bianco/1"
LEGACY_CONVENTION = "grigio-diretto/0"
CONVENTIONS = (IMAGE_CONVENTION, LEGACY_CONVENTION)


def phash(image: Image.Image) -> int:
    """pHash dell'immagine così com'è (`convert("L")` di Pillow). Per le convenzioni usare image_phash."""
    import imagehash

    return int(str(imagehash.phash(image.convert("L"), hash_size=8)), 16)


def normalize_image(image: Image.Image, convention: str = IMAGE_CONVENTION) -> Image.Image:
    if convention == IMAGE_CONVENTION:
        return to_rgb(image)
    if convention == LEGACY_CONVENTION:
        return image
    raise ValueError(f"convenzione delle immagini sconosciuta: {convention!r}")


def image_phash(image: Image.Image, convention: str = IMAGE_CONVENTION) -> int:
    return phash(normalize_image(image, convention))


def pixel_sha256(rgb: Image.Image) -> str:
    """sha256 di dimensioni e valori di un'immagine già normalizzata."""
    return hashlib.sha256(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes()).hexdigest()


def hash_image_bytes(data: bytes, convention: str = IMAGE_CONVENTION) -> tuple[int | None, str | None, list[int] | None]:
    """(pHash, sha256 dei pixel, [larghezza, altezza]) nella convenzione data; (None, None, None) se l'immagine non si
    decodifica. Nella convenzione storica lo sha256 dei pixel non esiste."""
    if convention not in CONVENTIONS:
        raise ValueError(f"convenzione delle immagini sconosciuta: {convention!r}")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            norm = normalize_image(im, convention)
            h = phash(norm)
            pixels = pixel_sha256(norm) if convention == IMAGE_CONVENTION else None
            return h, pixels, [norm.size[0], norm.size[1]]
    except Exception:  # noqa: BLE001 - immagine illeggibile: la conta il chiamante
        return None, None, None


def index_convention(index: dict, override: str | None = None) -> str:
    """Convenzione di un indice: quella dichiarata, oppure quella indicata per un indice che non la dichiara (gli
    storici p6 e f1eval sono "grigio-diretto/0", pannello-v1 e v2 "rgb-su-bianco/1"). Mai dedotta dal contenuto."""
    declared = index.get("convenzione_immagini")
    if declared and override and declared != override:
        raise ValueError(f"l'indice dichiara la convenzione {declared!r}, richiesta {override!r}")
    convention = declared or override
    if convention is None:
        raise ValueError("indice senza convenzione delle immagini: indicarla esplicitamente "
                         f"({' o '.join(CONVENTIONS)})")
    if convention not in CONVENTIONS:
        raise ValueError(f"convenzione delle immagini sconosciuta: {convention!r}")
    return convention


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def normalize_words(text: str) -> list[str]:
    """Parole minuscole di lettere e cifre (forme a larghezza piena ridotte con NFKC) e ideogrammi CJK uno per uno;
    `\\frac` resta `frac`, simboli e graffe spariscono. Toglie anche operatori, segni e disuguaglianze: è un segnale
    per i candidati, non una prova d'identità."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).lower().replace("\\", " "))


def exact_text(text: str) -> str:
    """Forma conservativa di un testo, per i legami confermati: NFC e spazi compressi. Operatori, segni, numeri,
    simboli, maiuscole e condizioni restano come sono (NFKC no: trasforma ² in 2)."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def word_ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class HashIndex:
    """Vicini entro `radius` bit di distanza di Hamming fra hash a 64 bit."""

    def __init__(self, hashes: list[int], radius: int = 4):
        self.hashes, self.radius = list(hashes), radius
        blocks = radius + 1
        size = 64 // blocks
        self.spans = [(i * size, 64 if i == blocks - 1 else (i + 1) * size) for i in range(blocks)]
        self.tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in self.spans]
        for idx, h in enumerate(self.hashes):
            for table, key in zip(self.tables, self._keys(h)):
                table[key].append(idx)

    def _keys(self, h: int) -> list[int]:
        return [(h >> lo) & ((1 << (hi - lo)) - 1) for lo, hi in self.spans]

    def query(self, h: int) -> list[tuple[int, int]]:
        """(indice, distanza) degli hash entro radius bit, ordinati per distanza."""
        seen, out = set(), []
        for table, key in zip(self.tables, self._keys(h)):
            for idx in table.get(key, ()):
                if idx in seen:
                    continue
                seen.add(idx)
                d = hamming(h, self.hashes[idx])
                if d <= self.radius:
                    out.append((idx, d))
        return sorted(out, key=lambda x: x[1])


@dataclass
class TextIndex:
    """n-gram delle domande dei benchmark, senza quelli presenti in più di max_df testi diversi."""

    n: int
    grams: dict[tuple[str, ...], list[int]]
    dropped: int = 0

    @classmethod
    def build(cls, texts: list[str], n: int = 10, max_df: int = 25) -> "TextIndex":
        grams: dict[tuple[str, ...], list[int]] = defaultdict(list)
        df: Counter = Counter()
        seen: set[tuple[str, ...]] = set()
        for idx, text in enumerate(texts):
            words = tuple(normalize_words(text))
            if len(words) < n:
                continue
            ngrams = word_ngrams(list(words), n)
            for g in ngrams:
                grams[g].append(idx)
            if words not in seen:  # varianti identiche (es. DynaMath) contano una volta
                seen.add(words)
                df.update(ngrams)
        common = {g for g, c in df.items() if c > max_df}
        return cls(n, {g: v for g, v in grams.items() if g not in common}, len(common))

    def query(self, text: str) -> list[int]:
        hits: set[int] = set()
        for g in word_ngrams(normalize_words(text), self.n):
            hits.update(self.grams.get(g, ()))
        return sorted(hits)
