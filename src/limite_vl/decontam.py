"""Decontaminazione dei dati di training contro i benchmark visivi (MathVista, MathVerse, MathVision, We-Math,
DynaMath): hash percettivo delle immagini e n-gram delle domande, come previsto dal piano.

- Immagini: pHash a 64 bit (imagehash, DCT 32x32 in scala di grigi), robusto a ridimensionamento e
  ricompressione. Due immagini sono "la stessa" se distano al più `radius` bit. La ricerca usa il principio
  dei cassetti: l'hash si divide in radius+1 blocchi, e due hash entro radius bit coincidono esattamente in
  almeno un blocco; si confrontano solo gli hash che condividono un blocco.
- Domande: testo normalizzato (minuscole, solo lettere e cifre, LaTeX ridotto ai nomi dei comandi), poi
  n-gram di parole. Una domanda di training è contaminata se condivide un n-gram con una domanda di un
  benchmark; le domande più corte di n parole devono coincidere per intero.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from PIL import Image

_WORD = re.compile(r"[a-z0-9]+")


def phash(image: Image.Image) -> int:
    import imagehash

    return int(str(imagehash.phash(image.convert("L"), hash_size=8)), 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def normalize_words(text: str) -> list[str]:
    """Parole minuscole di lettere e cifre; `\\frac` resta `frac`, simboli e graffe spariscono."""
    return _WORD.findall(text.lower().replace("\\", " "))


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
    """n-gram delle domande dei benchmark e domande corte intere."""

    n: int
    grams: dict[tuple[str, ...], list[int]]
    short: dict[tuple[str, ...], list[int]]

    @classmethod
    def build(cls, texts: list[str], n: int = 10) -> "TextIndex":
        grams: dict[tuple[str, ...], list[int]] = defaultdict(list)
        short: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for idx, text in enumerate(texts):
            words = normalize_words(text)
            if not words:
                continue
            if len(words) < n:
                short[tuple(words)].append(idx)
            for g in word_ngrams(words, n):
                grams[g].append(idx)
        return cls(n, dict(grams), dict(short))

    def query(self, text: str) -> list[int]:
        words = normalize_words(text)
        hits: set[int] = set(self.short.get(tuple(words), ()))
        for g in word_ngrams(words, self.n):
            hits.update(self.grams.get(g, ()))
        return sorted(hits)
