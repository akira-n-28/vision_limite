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
  parole ("Find x.") non si confrontano sul testo: decide l'immagine.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from PIL import Image

# parole latine e cifre; ogni ideogramma CJK è una parola (senza, le domande in cinese si riducevano a numeri e
# lettere delle scelte e si somigliavano tutte: revisione del job 29572)
_WORD = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def phash(image: Image.Image) -> int:
    import imagehash

    return int(str(imagehash.phash(image.convert("L"), hash_size=8)), 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def normalize_words(text: str) -> list[str]:
    """Parole minuscole di lettere e cifre (forme a larghezza piena ridotte con NFKC) e ideogrammi CJK uno per uno;
    `\\frac` resta `frac`, simboli e graffe spariscono."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).lower().replace("\\", " "))


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
