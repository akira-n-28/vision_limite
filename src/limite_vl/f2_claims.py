"""Affermazioni sulla figura nelle soluzioni F2 MAVIS: confronto deterministico delle misure, alterazioni con verità nota,
controlli negativi verificabili, legame fra citazioni e testo, risoluzione dei giudizi per affermazione.

Solo libreria standard (più limite_vl.stem_audit): serve al pilota dell'audit v2 (scripts/f2_pilot.py) e ai suoi test.

Le didascalie di MAVIS-Geometry (riscritte da limite_vl.mavis.bridge_caption e certificate) dicono una misura per frase:
"Side AB is measured at 72 units.", "The distance between points C and E is 9*√(2).", "Angle CBA measures 60 degrees.".
Ogni misura è disegnata in caption.jpg, quindi è la verità della figura per quel segmento o angolo. Una soluzione che
scrive "AB = 16" dove la didascalia dice 12 contraddice la figura: lo dice il codice, senza agenti.
"""
from __future__ import annotations

import math
import random
import re

from .stem_audit import _norm_map, resolve_field

_SENT = re.compile(r"(?<=\.)\s+")
_NUM = r"(\d+(?:\.\d+)?)"
# valore di una misura della didascalia: numero, eventualmente per una radice (9*√(2)); si prende l'ultimo della frase
_CAP_VALUE = re.compile(_NUM + r"(?:\s*\*\s*√\((\d+)\))?(?:\s*/\s*" + _NUM + ")?")  # 10*√(3)/3
# parole chiave senza maiuscole/minuscole, lettere dei punti solo maiuscole ("side is 5" non è il segmento IS)
_CAP_ANGLE = re.compile(r"\b(?i:angle)\s+([A-Z])([A-Z])([A-Z])\b")
_CAP_SEG = [re.compile(r"\b(?i:side|segment|edge|line)\s+([A-Z])([A-Z])\b"),
            re.compile(r"\b(?i:points?)\s+([A-Z])\s+and\s+([A-Z])\b"),
            re.compile(r"\b(?i:from)\s+(?:(?i:point)\s+)?([A-Z])\s+to\s+(?:(?i:point)\s+)?([A-Z])\b")]
# valore nella soluzione: numero con radice facoltativa (4\sqrt{2}, 4√2, 4*√(2)) che non apre un'espressione (2*16, 3/4, 5+x)
# la parentesi della radice si chiude se si apre (altrimenti "10*√(3)/3" si fermerebbe a "10*√(3"); /c facoltativo
_SOL_VALUE = (r"(?:\$|\\\(|\\\[)?\s*" + _NUM + r"(?:\s*\*?\s*(?:√\s*(?:\((\d+)\)|(\d+))|\\sqrt\s*(?:\{(\d+)\}|(\d+))))?"
              r"(?:\s*/\s*" + _NUM + ")?"
              r"(?!\d|\.\d|\s*[*/+\-×·]|\s*\^(?!\{?\\circ)|\s*\\(?:times|cdot|frac|pi|sqrt)|\s*[π√])")  # 30^\circ resta una misura
_DEGREES = re.compile(r"\s*(?:°|\^\{?\\circ|degrees?\b)")
# "r = half of DC = 6", "AD = AB + BD = 14": il valore finale è del primo termine della catena, non del lato in mezzo.
# Senza cifre fra "=" e il lato: "angle ABC = 90° and AB = 5" non è una catena
_CHAINED = re.compile(r"=\s*[^=.;,\n\d]{0,25}$")
_REL = r"\s*(?:=|is|measures|equals|has length|of length)\s*"
_SOL_SEG = re.compile(r"(?<![\w*/+\-^\\{])(?:\\overline\{)?([A-Z])([A-Z])\}?(?![A-Za-z])" + _REL + _SOL_VALUE)
_SOL_ANGLE = re.compile(r"(?:\\angle\s*|∠\s*|\b(?i:angle)\s+)\{?([A-Z])([A-Z])([A-Z])\}?(?![A-Za-z])" + _REL + _SOL_VALUE)
_HYPOTHETICAL = re.compile(r"\b(?:if|suppose|assume|assuming|were|would|could|maybe|perhaps|say)\b[^.\n]{0,40}$", re.IGNORECASE)
_PROMPT_QUOTE = re.compile(r'"Figure:[^"]*"')
TOL = 0.005  # differenza relativa oltre la quale due misure sono diverse (4√2 scritto 5.66 resta uguale)


def _value(coef: str, rad: str | None, den: str | None = None) -> float:
    return float(coef) * (math.sqrt(float(rad)) if rad else 1.0) / (float(den) if den else 1.0)


def _seg_key(a: str, b: str) -> tuple:
    return ("seg", "".join(sorted(a.upper() + b.upper())))


def _ang_key(a: str, v: str, b: str) -> tuple:
    return ("ang", v.upper(), "".join(sorted(a.upper() + b.upper())))


def caption_measures(caption: str) -> dict[tuple, float]:
    """Misure della didascalia: {("seg", "AB"): 12.0, ("ang", "B", "AC"): 60.0}. Una frase per misura; frasi con due
    elementi o senza valore si saltano."""
    facts = {}
    for s in _SENT.split(caption.strip()):
        vals = list(_CAP_VALUE.finditer(s))
        if not vals:
            continue
        v = _value(vals[-1].group(1), vals[-1].group(2), vals[-1].group(3))
        m = _CAP_ANGLE.search(s)
        if m:
            facts[_ang_key(*m.groups())] = v
            continue
        for pat in _CAP_SEG:
            m = pat.search(s)
            if m:
                facts[_seg_key(*m.groups())] = v
                break
    return facts


def prompt_regions(text: str) -> list[tuple[int, int]]:
    """Tratti che citano il prompt ("Figure: ..."): le alterazioni non si mettono lì dentro."""
    return [(m.start(), m.end()) for m in _PROMPT_QUOTE.finditer(text)]


def solution_measures(solution: str, caption: str) -> list[dict]:
    """Misure scritte nella soluzione con l'esito del confronto con la didascalia: match, contradiction (stesso
    segmento o angolo, valore diverso), absent (elemento non misurato nella didascalia: una deduzione, non giudicata),
    hypothetical (dopo "if", "suppose"...) e chained (in mezzo a una catena "x = ... DC = 6"): queste ultime non contano.
    Un "lato" seguito da gradi ("the angle between BC and BA is 45°") non è una lunghezza e si salta. Ogni voce porta
    lo span del valore."""
    facts = caption_measures(caption)
    out = []
    for kind, pat in (("ang", _SOL_ANGLE), ("seg", _SOL_SEG)):
        for m in pat.finditer(solution):
            g = m.groups()
            if kind == "seg" and _DEGREES.match(solution, m.end()):
                continue
            key = _ang_key(g[0], g[1], g[2]) if kind == "ang" else _seg_key(g[0], g[1])
            coef, rad, den = (g[3], g[4] or g[5] or g[6] or g[7], g[8]) if kind == "ang" else (g[2], g[3] or g[4] or g[5] or g[6], g[7])
            v = _value(coef, rad, den)
            start = m.start(4 if kind == "ang" else 3)
            w = facts.get(key)
            before = solution[max(0, m.start() - 60):m.start()].split("\n")[-1]
            if before.rstrip()[-1:] in tuple("*/+-^×·"):  # "AB * BC = ...", "180 - angle ABC = 135": un'espressione
                continue
            if _HYPOTHETICAL.search(before):
                verdict = "hypothetical"
            elif _CHAINED.search(before):
                verdict = "chained"
            elif w is None:
                verdict = "absent"
            else:
                verdict = "match" if abs(v - w) <= TOL * max(1.0, abs(w)) else "contradiction"
            out.append({"key": "-".join(key), "value": v, "caption_value": w, "verdict": verdict, "start": start,
                        "end": m.end(), "text": solution[m.start():m.end()]})
    return sorted(out, key=lambda d: d["start"])


def _outside(spans: list[tuple[int, int]], pos: int) -> bool:
    return not any(a <= pos < b for a, b in spans)


def _think_end(text: str) -> int:
    e = text.find("</think>")
    return e if e > 0 else len(text)


def inject_measure(solution: str, caption: str, rng: random.Random) -> tuple[str, dict] | None:
    """Misura cambiata con verità nota: in un punto dove la soluzione scrive un segmento o un angolo con lo stesso valore
    della didascalia (verdict match, numero senza radice, nel ragionamento, fuori dalle citazioni del prompt), il valore
    diventa un altro: l'affermazione nuova contraddice la figura per costruzione."""
    regions, end = prompt_regions(solution), _think_end(solution)
    spots = [d for d in solution_measures(solution, caption) if d["verdict"] == "match" and d["start"] < end
             and _outside(regions, d["start"]) and d["value"] == float(_num_at(solution, d["start"]))]  # senza radice
    if not spots:
        return None
    d = rng.choice(spots)
    old = _num_at(solution, d["start"])
    if "." in old:
        new = f"{float(old) + rng.choice([0.5, 1.5, 2.5]):.{len(old.split('.')[1])}f}"
    else:
        iv = int(old)
        new = str(iv + rng.choice([2, 3, 4, 5]) if iv < 20 or rng.random() < 0.5 else iv * 2)
    text = solution[:d["start"]] + new + solution[d["start"] + len(old):]
    return text, {"family": "measure", "key": d["key"], "original": old, "changed": new, "truth": "contradicts the caption",
                  "start": d["start"], "end": d["start"] + len(new), "context": text[max(0, d["start"] - 80):d["start"] + 80]}


def _num_at(text: str, pos: int) -> str:
    m = re.match(_NUM, text[pos:])
    return m.group(1) if m else ""


_POLY = re.compile(r"(?<!\b[Aa]ngle )\b([A-Z]{3,6})\b")  # "angle ABC" nomina un angolo, non una forma ("rectangle DCEF" sì)
_ATTACH = re.compile(r"(?:shar\w*|attach\w*|adjacent|common|join\w*|built|constructed|erected|drawn|extend\w*|connect\w*|abut\w*)"
                     r"\b[^.\n]{0,30}$", re.IGNORECASE)
_EDGE = re.compile(r"\b(?i:side|edge|base)\s+([A-Z])([A-Z])(?![A-Za-z])")


def shared_edges(caption: str) -> tuple[set[str], dict[str, list[str]]]:
    """Lati in comune fra due forme della didascalia (le forme sono i nomi di 3-6 lettere: ABCD, CBEF, DCE) e lati di
    ogni forma: {"BC"}, {"ABCD": ["AB", "BC", "CD", "AD"], ...}."""
    polys = {p for s in _SENT.split(caption) for p in _POLY.findall(s) if len(set(p)) == len(p)}
    edges = {p: ["".join(sorted(p[i] + p[(i + 1) % len(p)])) for i in range(len(p))] for p in polys}
    count: dict[str, int] = {}
    for p, es in edges.items():
        for e in set(es):
            count[e] = count.get(e, 0) + 1
    return {e for e, n in count.items() if n > 1}, edges


def inject_construction(solution: str, caption: str, rng: random.Random) -> tuple[str, dict] | None:
    """Costruzione sbagliata con verità nota: dove il ragionamento dice che una forma è attaccata, condivide o si
    costruisce su un lato che la didascalia dà in comune fra due forme, quel lato diventa un altro lato della stessa
    forma che non è in comune con nessuna: la nuova affermazione contraddice la costruzione disegnata."""
    shared, edges = shared_edges(caption)
    regions, end = prompt_regions(solution), _think_end(solution)
    spots = []
    for m in _EDGE.finditer(solution[:end]):
        e = "".join(sorted(m.group(1) + m.group(2)))
        if e in shared and _outside(regions, m.start()) and _ATTACH.search(solution[max(0, m.start() - 40):m.start()]) \
                and not solution[m.end():m.end() + 3].lstrip().startswith("?"):  # una domanda non è un'affermazione
            named_after = set(_POLY.findall(solution[m.end():m.end() + 40]))
            others = sorted({x for p, es in edges.items() if e in es for x in es if x not in shared}
                            - {x for p in named_after for x in edges.get(p, [])})
            if others:
                spots.append((m, others))
    if not spots:
        return None
    m, others = rng.choice(spots)
    new = rng.choice(others)
    new = new if m.group(1) + m.group(2) == "".join(sorted(m.group(1) + m.group(2))) else new[::-1]
    text = solution[:m.start(1)] + new + solution[m.end(2):]
    return text, {"family": "construction", "original": m.group(1) + m.group(2), "changed": new,
                  "truth": "the figure attaches the shapes along another side", "start": m.start(1), "end": m.start(1) + 2,
                  "context": text[max(0, m.start(1) - 80):m.start(1) + 80]}


_RIGHT_ANGLE = re.compile(r"right[\s-]*(?:angle|angled|triangle|isosceles)", re.IGNORECASE)  # tipo di angolo, non posizione
_ELEMENT = re.compile(r"(?<![A-Za-z\\])(?:[A-Z]{2,4}|(?i:point|vertex)\s+[A-Z])(?![A-Za-z])")  # un lato, una forma o un punto
LAYOUT_SWAPS = [("left", "right"), ("above", "below"), ("horizontal", "vertical"), ("upward", "downward"),
                ("inside", "outside"), ("top", "bottom"), ("upper", "lower"), ("clockwise", "counterclockwise")]


def inject_orientation(solution: str, rng: random.Random) -> tuple[str, dict] | None:
    """Parola di posizione o orientamento sostituita con l'opposta nel ragionamento, fuori dalle citazioni del prompt, a
    meno di 30 caratteri da un elemento della figura (lato, forma o punto nominato) e mai in "right angle". Verità NON
    garantita: se il testo originale era già falso, lo scambio può renderlo vero, e l'errore non trovato fa
    scendere la sensibilità (stima prudente)."""
    regions, end = prompt_regions(solution), _think_end(solution)
    head = solution[:end]
    spots = sorted([(m, y) for a, b in LAYOUT_SWAPS for x, y in ((a, b), (b, a))
                    for m in re.finditer(rf"\b{x}\b", head, re.IGNORECASE) if _outside(regions, m.start())
                    and not _RIGHT_ANGLE.match(head, m.start()) and _ELEMENT.search(head[max(0, m.start() - 30):m.end() + 30])],
                   key=lambda s: s[0].start())
    if not spots:
        return None
    m, repl = rng.choice(spots)
    word = m.group(0)
    repl = repl.upper() if word.isupper() else repl.capitalize() if word[0].isupper() else repl
    text = solution[:m.start()] + repl + solution[m.end():]
    return text, {"family": "orientation", "original": word, "changed": repl, "truth": "not guaranteed",
                  "start": m.start(), "end": m.start() + len(repl), "context": text[max(0, m.start() - 80):m.start() + 80]}


def caption_negative(caption: str, letter: str) -> str:
    """Controllo negativo verificabile: ogni affermazione sulla figura è una frase della didascalia certificata."""
    sents = [s.strip() for s in _SENT.split(caption.strip()) if s.strip()]
    return ("<think>From the figure: " + " ".join(sents) + " These facts determine the answer.</think> "
            f"The answer is \\boxed{{{letter}}}.")


def cot_negative(cot: str, caption: str, letter: str) -> str | None:
    """Controllo negativo dal CoT del motore MAVIS: ammesso solo se ogni misura che scrive coincide con la didascalia
    (nessuna contraddizione, nessun valore ipotetico) e ne scrive almeno due."""
    if not cot.strip():
        return None
    ms = solution_measures(cot, caption)
    if any(d["verdict"] in ("contradiction", "hypothetical") for d in ms) or sum(d["verdict"] == "match" for d in ms) < 2:
        return None
    return f"<think>{cot.strip()}</think> The answer is \\boxed{{{letter}}}."


# ---------------------------------------------------------------- citazioni e risoluzione

def quote_spans(text: str, quote: str, min_len: int = 3) -> list[tuple[int, int]]:
    """Span (nel testo originale) di ogni occorrenza della citazione, con le stesse normalizzazioni di
    stem_audit.quote_covers (delimitatori LaTeX, spazi, maiuscole); i pezzi separati da "..." contano come una citazione
    sola solo se il primo pezzo si ritrova."""
    norm, pos = _norm_map(text)
    frags = [f for f in (_norm_map(x)[0].strip(" \"'“”‘’`.,;:") for x in re.split(r"\.\.\.|…", quote or "")) if len(f) >= min_len]
    spans = []
    for q in frags[:1]:
        at = norm.find(q)
        while at >= 0:
            spans.append((pos[at], pos[at + len(q) - 1] + 1))
            at = norm.find(q, at + 1)
    return spans


def context_sentence(text: str, span: tuple[int, int], limit: int = 320) -> str:
    """La frase che contiene lo span (fra punti, a capo o inizio del ragionamento), al più limit caratteri."""
    dot, nl = text.rfind(". ", 0, span[0]), text.rfind("\n", 0, span[0])
    a = max(dot + 2 if dot >= 0 else 0, nl + 1 if nl >= 0 else 0)
    dot, nl = text.find(". ", span[1]), text.find("\n", span[1])
    b = min(dot + 1 if dot >= 0 else len(text), nl if nl >= 0 else len(text))
    s = text[a:b].strip()
    if len(s) > limit:
        mid = (span[0] + span[1]) // 2 - a
        s = s[max(0, mid - limit // 2):max(0, mid - limit // 2) + limit]
    return s


ERROR_VERDICTS = ("false", "not_visible")
OK_VERDICTS = ("true", "deduction")


def verdict_class(v) -> str | None:
    """Classe di un giudizio su un'affermazione: E errore (falsa o presentata come letta ma non visibile), K giusta (vera o
    deduzione dai dati), None incerta (unclear o mancante)."""
    v = v.strip().lower() if isinstance(v, str) else v
    return "E" if v in ERROR_VERDICTS else "K" if v in OK_VERDICTS else None


def resolve_claim(judgments: list) -> dict:
    """Stato di un'affermazione dai giudizi alla cieca VA, VB (e VC): due concordi e certi risolvono (consenso); con tre
    giudizi decide la maggioranza di due certi, altrimenti quarantena. Il pilota non ha arbitri: la maggioranza di tre
    è una semplificazione dichiarata nel report."""
    classes = [verdict_class(v) for v in judgments]
    st = resolve_field(classes, [])
    if st["status"] != "need_arbiter":
        return st
    sure = [c for c in classes if c]
    majority = next((c for c in ("E", "K") if sure.count(c) >= 2), None)
    if majority:
        return {"status": "resolved", "value": majority, "how": "maggioranza di tre"}
    return {"status": "quarantine", "value": None, "how": "nessuna maggioranza dopo il terzo giudizio"}
