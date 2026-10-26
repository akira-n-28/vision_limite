"""Descrizioni di PixMo-Cap contro le trascrizioni parlate degli annotatori (controllo di F1-clean senza LLM).

PixMo-Cap nasce da annotatori che descrivono l'immagine a voce per 60-90 secondi; un LLM riscrive le trascrizioni in una
descrizione. Ciò che la descrizione afferma e che nessuna trascrizione contiene è un'aggiunta del riscrittore: un
candidato errore, da confermare. Si confrontano tre tipi di contenuto che il riscrittore non dovrebbe inventare:
numeri (cifre o parole), colori e testi citati fra virgolette. Gli errori percettivi degli annotatori restano fuori:
qui l'annotatore è la verità.
"""
from __future__ import annotations

import re

UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "dozen": 12}
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "brown", "black", "white", "gray",
          "beige", "tan", "gold", "golden", "silver", "teal", "turquoise", "navy", "maroon", "cyan", "magenta", "cream",
          "ivory", "khaki", "olive", "lavender", "peach", "burgundy", "crimson", "indigo", "aqua", "lime", "mint"}
SYNONYM = {"grey": "gray", "golden": "gold"}
WORD = re.compile(r"[a-z]+(?:-[a-z]+)?|\d+(?:[.,]\d+)?")
QUOTE = re.compile(r"\"([^\"]{2,120})\"|“([^”]{2,120})”")
# parole numeriche che fanno parte di espressioni comuni, non di un conteggio ("one of the", "no one")
NOT_COUNT = re.compile(r"\b(?:one|ones)\s+(?:of|another|side|end|hand|another's)\b|\bno one\b|\bsomeone\b|\banyone\b"
                       r"|\beveryone\b|\bthe one\b|\bthis one\b|\bthat one\b|\bone (?:is|are|has)\b")


def _norm(word: str) -> str:
    return SYNONYM.get(word, word)


def numbers(text: str) -> set[float]:
    """Valori numerici nel testo: cifre (3,5 come decimale, 1,000 come migliaia) e parole da zero a cento, "dozen",
    con le decine seguite dalle unità ("twenty four", "twenty-four") lette anche come un solo numero."""
    words = [p for w in WORD.findall(NOT_COUNT.sub(" ", text.lower())) for p in w.split("-")]
    out, i = set(), 0
    while i < len(words):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if w[0].isdigit():
            try:
                comma = w.count(",") == 1 and len(w.split(",")[1]) != 3
                out.add(float(w.replace(",", ".")) if comma else float(w.replace(",", "")))
            except ValueError:
                pass
        elif w in UNITS and UNITS[w] in (20, 30, 40, 50, 60, 70, 80, 90) and nxt in UNITS and UNITS[nxt] < 10:
            out.add(float(UNITS[w] + UNITS[nxt]))  # "twenty four": un solo numero
            i += 1
        elif w in UNITS:
            out.add(float(UNITS[w]))
        i += 1
    return out


def colors(text: str) -> set[str]:
    return {_norm(w) for w in WORD.findall(text.lower()) if _norm(w) in COLORS or w in SYNONYM}


def quotes(text: str) -> list[str]:
    return [a or b for a, b in QUOTE.findall(text)]


def quote_coverage(quote: str, transcript: str) -> float:
    """Frazione delle parole del testo citato presenti nelle trascrizioni (le trascrizioni automatiche sbagliano grafie)."""
    words = [w for w in WORD.findall(quote.lower())]
    if not words:
        return 1.0
    have = set(WORD.findall(transcript.lower()))
    return sum(w in have for w in words) / len(words)


def compare(caption: str, transcripts: list[str]) -> dict:
    """Contenuti della descrizione che nessuna trascrizione contiene: {"numbers", "colors", "quotes"} con i valori non
    trovati, più i conteggi di quelli controllati."""
    joined = " ".join(t for t in transcripts if t)
    cap_numbers = {n for n in numbers(caption) if n >= 2}  # 0 e 1 troppo ambigui ("a dog" / "one dog")
    tr_numbers = numbers(joined)
    cap_colors, tr_colors = colors(caption), colors(joined)
    cap_quotes = quotes(caption)
    return {"numbers": sorted(cap_numbers - tr_numbers), "colors": sorted(cap_colors - tr_colors),
            "quotes": [q for q in cap_quotes if quote_coverage(q, joined) < 0.5],
            "checked": {"numbers": len(cap_numbers), "colors": len(cap_colors), "quotes": len(cap_quotes)}}
