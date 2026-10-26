"""Descrizioni di PixMo-Cap contro le trascrizioni parlate degli annotatori (controllo di F1-clean senza LLM).

PixMo-Cap nasce da annotatori che descrivono l'immagine a voce per 60-90 secondi; un LLM riscrive le trascrizioni in una
descrizione. Ciò che la descrizione afferma e che nessuna trascrizione contiene è un'aggiunta del riscrittore: un
candidato errore, da confermare. Si confrontano tre tipi di contenuto che il riscrittore non dovrebbe inventare:
numeri (cifre o parole), colori e testi citati fra virgolette. Gli errori percettivi degli annotatori restano fuori:
qui l'annotatore è la verità.

Forme equivalenti che non contano come aggiunte (seconda versione, dopo la lettura degli esempi del job 30028):
orari e numeri detti in altro modo ("16:19" e "1619", "29000" e "two nine zero zero zero"), lettere sillabate
("FRUUGO" e "F R U U G O"), colori con suffisso ("grayish", "pinkness"), parole storpiate dalla trascrizione automatica
("Supermarkt" e "supermarket"), numeri degli elenchi ("2. **Second ...**"); i testi citati si leggono una volta sola
(numeri e colori dentro le virgolette non si contano due volte). Terza versione (esempi del job 30031): "creamy" e
"golden hour" non sono colori, orari e numeri detti a parole ("eight twenty seven") e citazioni numeriche corte ("8:43"
contro "843", "7" contro "seven"), parole come "zero" lette anche come parole ("IsNotZero" contro "is not zero").
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "dozen": 12}
DIGIT_WORDS = {w: str(v) for w, v in UNITS.items() if v < 10}
# senza i nomi che sono spesso oggetti o sapori (lime, mint, peach, olive, cream: quarta versione, dopo il job 30032)
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "brown", "black", "white", "gray",
          "beige", "tan", "gold", "silver", "teal", "turquoise", "navy", "maroon", "cyan", "magenta", "ivory", "khaki",
          "lavender", "burgundy", "crimson", "indigo", "aqua"}
WORD = re.compile(r"[a-z]+(?:-[a-z]+)?|\d+(?:[.,]\d+)?")
QUOTE = re.compile(r"\"([^\"\n]{1,120})\"|“([^”\n]{1,120})”")
TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
LIST_MARKER = re.compile(r"(?m)^\s*(?:\*\*)?\d+[.)](?:\*\*)?\s+")
# parole numeriche che fanno parte di espressioni comuni, non di un conteggio ("one of the", "no one")
NOT_COUNT = re.compile(r"\b(?:one|ones)\s+(?:of|another|side|end|hand|another's)\b|\bno one\b|\bsomeone\b|\banyone\b"
                       r"|\beveryone\b|\bthe one\b|\bthis one\b|\bthat one\b|\bone (?:is|are|has)\b")


def _color_forms() -> dict[str, str]:
    forms = {"grey": "gray", "greyish": "gray", "golden": "gold", "bluish": "blue", "silvery": "silver",
             "orangey": "orange"}
    for c in COLORS:
        for f in (c, c + "ish", c[:-1] + "ish", c + c[-1] + "ish", c + "ness"):
            forms.setdefault(f, c)
    return forms


COLOR_FORMS = _color_forms()


def numbers(text: str) -> set[float]:
    """Valori numerici nel testo: cifre (3,5 come decimale, 1,000 come migliaia) e parole da zero a cento, "dozen",
    con le decine seguite dalle unità ("twenty four", "twenty-four") lette come un solo numero."""
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
    return {COLOR_FORMS[w] for w in WORD.findall(text.lower()) if w in COLOR_FORMS}


def quotes(text: str) -> list[str]:
    """Testi fra virgolette. Uno spazio dopo l'apertura o prima della chiusura vuol dire virgolette accoppiate male
    (un testo fra due citazioni): quei tratti non sono citazioni e si scartano."""
    out = []
    for a, b in QUOTE.findall(text):
        q = a or b
        if q.strip() and q == q.strip():
            out.append(q)
    return out


def compact(text: str, spoken: bool = True) -> str:
    """Lettere e cifre minuscole senza separatori; con `spoken` i numeri detti a parole diventano cifre ("two nine" ->
    "29", "eight twenty seven" -> "827")."""
    words = [p for w in WORD.findall(text.lower()) for p in w.split("-")]
    out, i = [], 0
    while i < len(words):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if spoken and w in UNITS and UNITS[w] in (20, 30, 40, 50, 60, 70, 80, 90) and nxt in UNITS and UNITS[nxt] < 10:
            out.append(str(UNITS[w] + UNITS[nxt]))
            i += 1
        elif spoken and w in UNITS and UNITS[w] < 100:
            out.append(str(UNITS[w]))
        else:
            out.append(w)
        i += 1
    return "".join(out).replace(",", "").replace(".", "")


def _number_text(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


def _word_present(word: str, have: set[str]) -> bool:
    if word in have:
        return True
    return len(word) >= 4 and any(abs(len(h) - len(word)) <= 2 and SequenceMatcher(None, word, h).ratio() >= 0.8
                                  for h in have)


def quote_supported(quote: str, transcript: str, compacts: tuple[str, str], tr_numbers: set[float]) -> bool:
    """Un testo citato è nelle trascrizioni se almeno metà delle sue parole c'è (anche storpiata), o se la sua forma
    compatta (lettere e cifre) compare in una delle forme compatte delle trascrizioni (lettere sillabate, cifre dette una
    a una, numeri a parole). Una citazione fatta solo di numeri (anche un orario) basta che abbia i suoi numeri nelle
    trascrizioni."""
    words = WORD.findall(quote.lower())
    if not words:
        return True
    q = compact(quote, spoken=False)
    numeric = all(w[0].isdigit() for w in words)
    if (len(q) >= 4 or (numeric and len(q) >= 3)) and any(q in c for c in compacts):
        return True
    if numeric and numbers(quote) <= tr_numbers:
        return True
    have = set(WORD.findall(transcript.lower()))
    return sum(_word_present(w, have) for w in words) / len(words) >= 0.5


YEAR = re.compile(r"^(1[89]\d\d|20\d\d)$")


def number_kind(value: str) -> str:
    """Tipo di un numero non trovato, per i segnali di rischio: orario, anno, piccolo conteggio (2-12), altro."""
    if ":" in value:
        return "orario"
    if YEAR.match(value):
        return "anno"
    try:
        return "conteggio_2_12" if 2 <= float(value) <= 12 else "altro_numero"
    except ValueError:
        return "altro_numero"


def compare(caption: str, transcripts: list[str]) -> dict:
    """Contenuti della descrizione che nessuna trascrizione contiene: {"numbers", "colors", "quotes"} con i valori non
    trovati, più i conteggi di quelli controllati."""
    joined = " ".join(t for t in transcripts if t)
    compacts = (compact(joined), compact(joined, spoken=False))
    joined_compact = compacts[0]
    cap_quotes = quotes(caption)
    body = LIST_MARKER.sub(" ", QUOTE.sub(" ", caption))
    body = re.sub(r"\bgolden hours?\b", " ", body, flags=re.IGNORECASE)
    tr_numbers = numbers(joined)
    missing_numbers, checked_numbers = [], 0
    for m in TIME.finditer(body):  # un orario si controlla intero: "16:19" contro "1619", "16 19", "sixteen nineteen"
        checked_numbers += 1
        h, mm = float(m.group(1)), float(m.group(2))
        if not (m.group(1) + m.group(2) in joined_compact or {h, mm} <= tr_numbers):
            missing_numbers.append(m.group(0))
    body = TIME.sub(" ", body)
    for n in sorted(x for x in numbers(body) if x >= 2):  # 0 e 1 troppo ambigui ("a dog" / "one dog")
        checked_numbers += 1
        s = _number_text(n).replace(".", "")
        if n in tr_numbers or (len(s) >= 2 and s in joined_compact):
            continue
        missing_numbers.append(_number_text(n))
    cap_colors = colors(body)
    return {"numbers": missing_numbers, "colors": sorted(cap_colors - colors(joined)),
            "quotes": [q for q in cap_quotes if not quote_supported(q, joined, compacts, tr_numbers)],
            "checked": {"numbers": checked_numbers, "colors": len(cap_colors), "quotes": len(cap_quotes)}}
