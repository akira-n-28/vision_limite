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
_QUOTED = re.compile(r'"[^"\n]{1,400}"|“[^”\n]{1,400}”')  # citazioni del prompt anche senza "Figure:"
_PAREN = re.compile(r"\([^()\n]{0,60}\)")
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
    """Tratti che citano il prompt ("Figure: ..." e ogni altro testo fra virgolette): le alterazioni non si mettono lì
    dentro, perché devono stare nel ragionamento della soluzione."""
    return sorted({(m.start(), m.end()) for pat in (_PROMPT_QUOTE, _QUOTED) for m in pat.finditer(text)})


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


# frasi che non asseriscono: ipotesi, condizioni, domande, negazioni, verifiche, correzioni ("Wait", "Actually")
_HEDGE = re.compile(r"\?|\b(?:if|suppose|supposing|assume|assuming|assumed|maybe|perhaps|possibly|probably|likely|unlikely|"
                    r"might|could|would|should|whether|either|or|unless|otherwise|alternatively|guess|seem|seems|seemingly|"
                    r"hypothes\w*|consider|considering|imagine|let|let's|try|trying|check|checking|wonder|not|no|never|neither|"
                    r"nor|cannot|without|wrong|incorrect|actually|wait|hmm|typically|usually|interpret\w*|mean|means|say|case|need|needs|"
                    r"confirm|verify|ensure)\b|n't\b",
                    re.IGNORECASE)
_CORRECTION = re.compile(r"(?:no\b|wait|actually|but|hmm|hold on|that's wrong|that is wrong|not\b|oops|however|or\b|alternatively)",
                         re.IGNORECASE)


def _sentence_at(text: str, pos: int) -> tuple[int, int]:
    for a, b in sentence_spans(text):
        if a <= pos < b:
            return a, b
    return pos, pos


def asserted_at(text: str, pos: int) -> bool:
    """La frase che contiene pos asserisce un fatto: niente ipotesi, condizioni, domande, negazioni o verifiche, e la frase
    dopo non la corregge ("No, ...", "Wait", "Actually"). Le alterazioni con verità certa si mettono solo in frasi così:
    in una frase ipotetica o scartata la verità dell'affermazione non è definita."""
    a, b = _sentence_at(text, pos)
    if b <= a or _HEDGE.search(text[a:b]):
        return False
    nxt = text[b:b + 40].lstrip(" .\n\"")
    return not _CORRECTION.match(nxt)


def inject_measure(solution: str, caption: str, rng: random.Random) -> tuple[str, dict] | None:
    """Misura cambiata con verità nota: in un punto dove la soluzione scrive un segmento o un angolo con lo stesso valore
    della didascalia (verdict match, numero senza radice, nel ragionamento, fuori dalle citazioni del prompt), il valore
    diventa un altro: l'affermazione nuova contraddice la figura per costruzione. Solo in frasi che asseriscono
    (asserted_at)."""
    regions, end = prompt_regions(solution), _think_end(solution)
    spots = [d for d in solution_measures(solution, caption) if d["verdict"] == "match" and d["start"] < end
             and _outside(regions, d["start"]) and d["value"] == float(_num_at(solution, d["start"]))  # senza radice
             and asserted_at(solution, d["start"])]
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
# "adjacent" no: "adjacent side BA" è il cateto vicino a un angolo, non un attacco fra forme
_ATTACH = re.compile(r"(?:shar\w*|attach\w*|common|join\w*|built|constructed|erected|drawn|extend\w*|connect\w*|abut\w*)"
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
        parens = [(p.start(), p.end()) for p in _PAREN.finditer(solution, max(0, m.start() - 60), m.end() + 60)]
        if e in shared and _outside(regions + parens, m.start()) and _ATTACH.search(solution[max(0, m.start() - 40):m.start()]) \
                and asserted_at(solution, m.start()):  # niente domande, ipotesi o frasi scartate
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


# "right" come tipo di angolo, non come posizione: right angle, right-angled, right isosceles, right at B, isosceles right
_RIGHT_ANGLE = re.compile(r"right[\s-]*(?:angle|angled|triangle|isosceles|at\b)", re.IGNORECASE)
_ISOSCELES_BEFORE = re.compile(r"isosceles\s+$", re.IGNORECASE)
_ELEMENT = re.compile(r"(?<![A-Za-z\\])(?:[A-Z]{2,4}|(?i:point|vertex)\s+[A-Z])(?![A-Za-z])")  # un lato, una forma o un punto
# direzioni nella pagina; dentro/fuori è configurazione (inject_configuration, con verità certa dalla didascalia)
LAYOUT_SWAPS = [("left", "right"), ("above", "below"), ("horizontal", "vertical"), ("upward", "downward"),
                ("top", "bottom"), ("upper", "lower"), ("clockwise", "counterclockwise")]


def inject_orientation(solution: str, rng: random.Random) -> tuple[str, dict] | None:
    """Parola di posizione o orientamento sostituita con l'opposta nel ragionamento, fuori dalle citazioni del prompt, a
    meno di 30 caratteri da un elemento della figura (lato, forma o punto nominato), mai in "right angle" e solo in frasi
    che asseriscono (asserted_at). Verità NON
    garantita: se il testo originale era già falso, lo scambio può renderlo vero, e l'errore non trovato fa
    scendere la sensibilità (stima prudente)."""
    regions, end = prompt_regions(solution), _think_end(solution)
    head = solution[:end]
    spots = sorted([(m, y) for a, b in LAYOUT_SWAPS for x, y in ((a, b), (b, a))
                    for m in re.finditer(rf"\b{x}\b", head, re.IGNORECASE) if _outside(regions, m.start())
                    and not _RIGHT_ANGLE.match(head, m.start()) and not _ISOSCELES_BEFORE.search(head[max(0, m.start() - 12):m.start()])
                    and _ELEMENT.search(head[max(0, m.start() - 30):m.end() + 30])
                    and asserted_at(head, m.start())],
                   key=lambda s: s[0].start())
    if not spots:
        return None
    m, repl = rng.choice(spots)
    word = m.group(0)
    repl = repl.upper() if word.isupper() else repl.capitalize() if word[0].isupper() else repl
    text = solution[:m.start()] + repl + solution[m.end():]
    return text, {"family": "orientation", "original": word, "changed": repl, "truth": "not guaranteed",
                  "start": m.start(), "end": m.start() + len(repl), "context": text[max(0, m.start() - 80):m.start() + 80]}


_QUAD_DECL = re.compile(r"\b([A-Z]{4})\b[^.]{0,40}?\b(?:square|rectangle|parallelogram|rhombus)\b|"
                        r"\b(?:square|rectangle|parallelogram|rhombus)\s+([A-Z]{4})\b", re.IGNORECASE)
_TRI_DECL = re.compile(r"\b([A-Z]{3})\b")
_REMOVED = re.compile(r"\bside\s+([A-Z])([A-Z])\s+removed", re.IGNORECASE)
_PLACED = re.compile(r"\b(?:the\s+)?(?:rectangle|square|parallelogram)\s+([A-Z]{4})\b[^.]*?\bside\s+([A-Z]{2})\s+removed and replaced by\s+"
                     r"an?\s+[^.]*?\b(semicircle|triangle|circle|sector)\b[^.]*?\b(inside|outside)\b", re.IGNORECASE)
_INOUT = {"inside": "inside", "inward": "inside", "inwards": "inside", "interior": "inside",
          "outside": "outside", "outward": "outside", "outwards": "outside", "exterior": "outside"}
_INOUT_SWAP = {"inside": "outside", "outside": "inside", "inward": "outward", "outward": "inward", "inwards": "outwards",
               "outwards": "inwards", "interior": "exterior", "exterior": "interior"}
_SIDE_PAIR = [re.compile(r"\b(?:sides?\s+)?([A-Z]{2})\s+and\s+(?:side\s+)?([A-Z]{2})\s+are\s+(?:both\s+)?(opposite|adjacent)\b"),
              re.compile(r"\b(?:side\s+)?([A-Z]{2})\s+is\s+(opposite|adjacent)\s+(?:to\s+)?(?:side\s+)?([A-Z]{2})\b")]
_VERTEX_PAIR = re.compile(r"\b(?:points?\s+|vertices\s+)?([A-Z])\s+and\s+([A-Z])\s+are\s+(?:both\s+)?(adjacent|opposite)\b")
_RIGHT_AT = re.compile(r"\bright angle\s+(?:is\s+)?at\s+(?:vertex\s+|point\s+)?([A-Z])(?![A-Za-z])", re.IGNORECASE)


def caption_quads(caption: str) -> dict[str, set[str]]:
    """Quadrilateri dichiarati nella didascalia (quadrato, rettangolo, parallelogramma, rombo) con i lati ancora disegnati:
    {"ABCD": {"AB", "BC", "CD", "AD"}} meno i lati "removed"."""
    removed = {"".join(sorted(m.group(1) + m.group(2))) for m in _REMOVED.finditer(caption)}
    out = {}
    for m in _QUAD_DECL.finditer(caption):
        q = m.group(1) or m.group(2)
        if q and len(set(q)) == 4:
            out[q] = {"".join(sorted(q[i] + q[(i + 1) % 4])) for i in range(4)} - removed
    return out


def _quad_relation(q: str, x: str, y: str) -> str | None:
    """Relazione fra due lati (due lettere) o due vertici (una lettera) del quadrilatero q: adjacent / opposite."""
    if len(x) == 1:
        if x == y or x not in q or y not in q:
            return None
        return "adjacent" if abs(q.index(x) - q.index(y)) in (1, 3) else "opposite"
    es = {"".join(sorted(q[i] + q[(i + 1) % 4])) for i in range(4)}
    a, b = "".join(sorted(x)), "".join(sorted(y))
    if a == b or a not in es or b not in es:
        return None
    return "adjacent" if set(a) & set(b) else "opposite"


def _swap_case(word: str, repl: str) -> str:
    return repl.upper() if word.isupper() else repl.capitalize() if word[0].isupper() else repl


def inject_configuration(solution: str, caption: str, rng: random.Random) -> tuple[str, dict] | None:
    """Configurazione sbagliata con verità certa, decisa dal codice sulla didascalia certificata, in frasi che asseriscono
    (asserted_at), nel ragionamento e fuori dalle citazioni del prompt. Tre modi:
    - inside/outside: la didascalia dice "side CD removed and replaced by a semicircle ... drawn outside the rectangle";
      dove la soluzione dice la stessa cosa della stessa forma, la parola diventa l'opposta;
    - opposite/adjacent: due lati o due vertici di un quadrilatero della didascalia; si altera solo se la parola originale
      è quella giusta per l'ordine dei vertici, quindi la nuova è falsa;
    - angolo retto: la didascalia dà 90° all'angolo XYZ di un triangolo XYZ; "right angle at Y" diventa un altro vertice
      dello stesso triangolo (un triangolo ha un solo angolo retto)."""
    regions, end = prompt_regions(solution), _think_end(solution)
    head = solution[:end]
    parens = [(p.start(), p.end()) for p in _PAREN.finditer(head)]
    ok = lambda pos: _outside(regions + parens, pos) and asserted_at(head, pos)  # noqa: E731
    spots = []  # (start, end, original, changed, mode, truth)
    placed = [(m.group(1), "".join(sorted(m.group(2))), m.group(3).lower(), m.group(4).lower()) for m in _PLACED.finditer(caption)]
    for m in re.finditer(r"\b(?:inside|outside|inward|outward|inwards|outwards|interior|exterior)\b", head, re.IGNORECASE):
        a, b = _sentence_at(head, m.start())
        sent = head[a:b]
        hits = [p for p in placed if re.search(rf"\b{p[2]}\b", sent, re.IGNORECASE)]
        named = {"".join(sorted(x)) for x in re.findall(r"(?<![A-Za-z])([A-Z]{2})(?![A-Za-z])", sent)}
        if len({p[3] for p in hits}) > 1:
            hits = [p for p in hits if p[1] in named]
        after = set(_POLY.findall(head[m.end():m.end() + 30]))
        word = m.group(0)
        if hits and len({p[3] for p in hits}) == 1 and _INOUT[word.lower()] == hits[0][3] and after <= {hits[0][0]} and ok(m.start()):
            spots.append((m.start(), m.end(), word, _swap_case(word, _INOUT_SWAP[word.lower()]), "inside/outside",
                          f"the caption draws the {hits[0][2]} {hits[0][3]}"))
    quads = caption_quads(caption)
    pairs = [(m, m.group(1), m.group(2), m.group(3), 3) for m in _SIDE_PAIR[0].finditer(head)]
    pairs += [(m, m.group(1), m.group(3), m.group(2), 2) for m in _SIDE_PAIR[1].finditer(head)]
    pairs += [(m, m.group(1), m.group(2), m.group(3), 3) for m in _VERTEX_PAIR.finditer(head)]
    for m, x, y, word, gi in pairs:
        drawn = lambda q: len(x) == 1 or {"".join(sorted(x)), "".join(sorted(y))} <= quads[q]  # noqa: E731  lati non rimossi
        rel = {_quad_relation(q, x, y) for q in quads if drawn(q)} - {None}
        if len(rel) == 1 and word.lower() == rel.pop() and ok(m.start(gi)):
            new = "adjacent" if word.lower() == "opposite" else "opposite"
            spots.append((m.start(gi), m.end(gi), word, _swap_case(word, new), "opposite/adjacent",
                          f"{x} and {y} are {word.lower()} in the caption's quadrilateral"))
    facts = caption_measures(caption)
    tris = {t for t in _TRI_DECL.findall(caption) if len(set(t)) == 3}
    right = {(k[1], k[2]) for k, v in facts.items() if k[0] == "ang" and abs(v - 90) < 1e-9}  # (vertice, altri due)
    for m in _RIGHT_AT.finditer(head):
        y = m.group(1)
        a, b = _sentence_at(head, m.start())
        named = {t for t in re.findall(r"(?<![A-Za-z])([A-Z]{3})(?![A-Za-z])", head[a:b])
                 if "".join(sorted(t)) in {"".join(sorted(x)) for x in tris}}
        for t in named:
            others = "".join(sorted(set(t) - {y}))
            if y in t and (y, others) in right and ok(m.start(1)):
                spots.append((m.start(1), m.end(1), y, rng.choice(sorted(set(t) - {y})), "right angle",
                              f"the caption gives angle {others[0]}{y}{others[1]} = 90 in triangle {t}"))
                break
    if not spots:
        return None
    a, b, old, new, mode, why = rng.choice(sorted(spots))
    text = solution[:a] + new + solution[b:]
    return text, {"family": "configuration", "mode": mode, "original": old, "changed": new, "truth": why, "certain": True,
                  "start": a, "end": a + len(new), "context": text[max(0, a - 80):a + 80]}


def caption_value_texts(caption: str) -> dict[tuple, str]:
    """Come caption_measures, ma il valore come è scritto, senza "*" né parentesi della radice: "28√2", "3/2"."""
    out = {}
    for s in _SENT.split(caption.strip()):
        vals = list(_CAP_VALUE.finditer(s))
        if not vals:
            continue
        txt = re.sub(r"\s*\*\s*√\((\d+)\)", r"√\1", vals[-1].group(0)).replace(" ", "")
        m = _CAP_ANGLE.search(s)
        if m:
            out[_ang_key(*m.groups())] = txt
            continue
        for pat in _CAP_SEG:
            m = pat.search(s)
            if m:
                out[_seg_key(*m.groups())] = txt
                break
    return out


def frame_control(caption: str, rng: random.Random, broken: bool) -> dict | None:
    """Affermazione di controllo sui sistemi di riferimento, con verità certa, per i verificatori (non viene da una
    soluzione). Serve un rettangolo o un quadrato ABCD della didascalia con i lati AB e AD (o BC) noti. Coordinate canoniche
    A(0,0), B(a,0), C(a,b), D(0,b) girate di k·90° e forse specchiate:
    - lecito (broken=False): una scelta di coordinate qualsiasi è coerente con la figura, comunque sia disegnata: non è
      un errore anche se gli assi non seguono il disegno;
    - rotto (broken=True): lunghezze scambiate (AB = b) se a != b, altrimenti C e D scambiati (A e C diventano vicini,
      mentre nel quadrato disegnato sono opposti): nessuna rotazione o riflessione della figura lo rende vero."""
    vals, texts = caption_measures(caption), caption_value_texts(caption)
    for q, kind in re.findall(r"\b([A-Z]{4})\b[^.]{0,40}?\b(square|rectangle)\b", caption, re.IGNORECASE):
        if len(set(q)) != 4:
            continue
        A, B, C, D = q
        ab, ad = _seg_key(A, B), _seg_key(A, D)
        a, at = vals.get(ab), texts.get(ab)
        b, bt = vals.get(ad, vals.get(_seg_key(B, C))), texts.get(ad, texts.get(_seg_key(B, C)))
        if kind.lower() == "square" and a is not None:
            b, bt = a, at
        if a is None or b is None or not at or not bt:
            continue
        same = abs(a - b) <= TOL * max(1.0, abs(a))
        pa, pb = (bt, at) if broken and not same else (at, bt)
        pts = {A: (0, 0), B: (1, 0), C: (1, 1), D: (0, 1)}  # in unità (lato AB, lato AD)
        if broken and same:
            pts[C], pts[D] = pts[D], pts[C]
        k, mirror = rng.randrange(4), rng.random() < 0.5
        fmt = lambda c: "0" if c[0] == 0 else ("-" if c[0] < 0 else "") + (pa if c[1] == "a" else pb)  # noqa: E731
        coords = {}
        for p, (x, y) in pts.items():
            # ogni coordinata è (coefficiente, lato): x lungo AB (lunghezza a), y lungo AD (lunghezza b)
            cx, cy = (-x if mirror else x, "a"), (y, "b")
            for _ in range(k):  # rotazione di 90°: (u, v) -> (-v, u)
                cx, cy = (-cy[0], cy[1]), cx
            coords[p] = f"({fmt(cx)}, {fmt(cy)})"
        st = "Coordinates chosen to compute: " + ", ".join(f"{p} = {coords[p]}" for p in q) + "."
        return {"statement": st, "quote": st, "context": st, "basis": "chosen", "kind": "coordinates",
                "family": "frame_broken" if broken else "frame_legit", "expected": "E" if broken else "K", "polygon": q,
                "rotation": k * 90, "mirror": mirror}
    return None


# ---------------------------------------------------------------- frasi e candidati deterministici

# fine frase: . ! ? seguiti da spazi e da un carattere che non è minuscolo ("3.5" e "e.g. the" restano interi), a capo, tag
_BOUNDARY = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”')]))\s+(?=[^a-z\s])|\n+|</?think>")
_CAPS = re.compile(r"(?<![A-Za-z\\])([A-Z]{1,6})(?![A-Za-z])")
# lettere di risposta, non punti: "answer D", "option (B)", "\boxed{C}", "A. 7056" a inizio riga
_ANSWER_LETTER = re.compile(r"(?:answer|option|choice|choices|boxed\{|is)\s*[:(]?\s*$", re.IGNORECASE)
_COORD = re.compile(r"\(\s*[-+]?[\w.√/*^{}\\ ]{1,24},\s*[-+]?[\w.√/*^{}\\ ]{1,24}\)")
# parole che parlano della figura anche senza nominare punti: forme, parti, relazioni, posizioni, la figura stessa
FIGURE_WORDS = (
    "square|squares|rectangle|rectangles|triangle|triangles|circle|circles|semicircle|semi-circle|sector|sectors|arc|arcs|"
    "parallelogram|rhombus|trapezoid|trapezium|kite|polygon|pentagon|hexagon|quadrilateral|shape|shapes|figure|diagram|image|"
    "picture|drawing|drawn|shown|depicted|labeled|labelled|marked|label|mark|side|sides|edge|edges|vertex|vertices|corner|"
    "corners|angle|angles|segment|segments|line|lines|ray|rays|diameter|radius|radii|chord|tangent|center|centre|midpoint|"
    "diagonal|diagonals|base|height|leg|legs|hypotenuse|perimeter|area|point|points|left|right|above|below|top|bottom|upper|"
    "lower|horizontal|horizontally|vertical|vertically|inside|outside|outward|outwards|inward|inwards|interior|exterior|"
    "clockwise|counterclockwise|anticlockwise|beside|adjacent|opposite|parallel|perpendicular|beyond|toward|towards|"
    "facing|extends|extend|extended|extending|attached|attach|attaches|shares|sharing|shared|connected|connects|joined|joins|"
    "meets|intersects|intersect|touches|lies|removed|replaced|inscribed|circumscribed|collinear|bisects|bisector|origin|"
    "axis|coordinates|coordinate|x-axis|y-axis|orientation|oriented|upward|upwards|downward|downwards|north|south|east|"
    "west|equilateral|isosceles")
_FIGURE_WORD = re.compile(rf"\b(?:{FIGURE_WORDS})\b", re.IGNORECASE)
# valori di grandezze della figura senza nominarla: "Total = 9 + 2π", "So h = 3/2", "w = 18√3" (job 29871: un errore trovato
# dai verificatori stava in una frase così)
_VALUE_OF = re.compile(r"(?<![\w\\])(?:[a-zθφαβγ]|total)\s*=(?!=)", re.IGNORECASE)
# parole di misura ammesse in una frase "solo misure" ("Side AB is 72 units", "Angle ABC = 60°")
_MEASURE_WORDS = {"side", "sides", "segment", "segments", "edge", "angle", "angles", "length", "distance", "measure",
                  "measures", "units", "unit", "degrees", "degree"}


def caption_points(caption: str) -> set[str]:
    """Lettere dei punti: quelle dei nomi di 2-6 maiuscole della didascalia (ABCD, CB, CBA); in MAVIS ogni punto compare
    in almeno un nome di forma, lato o angolo."""
    return {ch for tok in _CAPS.findall(caption) if len(tok) >= 2 for ch in tok}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Frasi del testo come span senza spazi ai bordi; i tag <think> separano. Frasi oltre 600 caratteri si spezzano anche
    a "; " (le tracce di Violetto hanno periodi lunghissimi)."""
    out, start = [], 0
    for m in list(_BOUNDARY.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        a, b = start, end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            if b - a > 600:
                cut = a
                for p in re.finditer(r";\s+", text[a:b]):
                    if a + p.start() - cut > 200:
                        out.append((cut, a + p.start() + 1))
                        cut = a + p.end()
                out.append((cut, b))
            else:
                out.append((a, b))
        start = m.end() if m else len(text)
    return out


def _norm(s: str) -> str:
    return _norm_map(s)[0].strip(" \"'“”‘’`.,;:")


def figure_triggers(sentence: str, points: set[str]) -> list[str]:
    """Perché una frase parla della figura: nomi fatti di lettere dei punti (AB, CBEF, punto G), coordinate, parole di
    forma, relazione o posizione. Una sola lettera dopo "answer"/"option"/"boxed{" è una lettera di risposta."""
    out = []
    for m in _CAPS.finditer(sentence):
        tok = m.group(1)
        if set(tok) <= points and not (len(tok) == 1 and _ANSWER_LETTER.search(sentence[max(0, m.start() - 12):m.start()])):
            out.append(tok)
    out += ["(coordinate)" for _ in _COORD.finditer(sentence)]
    out += ["(valore)" for _ in _VALUE_OF.finditer(sentence)]
    out += [w.group(0).lower() for w in _FIGURE_WORD.finditer(sentence)]
    return out


def code_only_measures(text: str, caption: str) -> list[tuple[str, float]] | None:
    """Le misure (elemento, valore) del testo se le sole cose che dice sulla figura sono misure di segmenti o angoli che il
    codice trova uguali alla didascalia; None se c'è altro (una forma, una relazione, un valore diverso o non in
    didascalia). Serve alla scorciatoia per i fatti di misura: il valore uguale non prova il ragionamento."""
    ms = solution_measures(text, caption)
    if not ms or any(d["verdict"] != "match" for d in ms):
        return None
    rest = text
    for d in sorted(ms, key=lambda d: -d["end"]):
        rest = rest[:max(0, d["end"] - len(d["text"]))] + " " + rest[d["end"]:]
    if [t for t in figure_triggers(rest, caption_points(caption)) if t.lower() not in _MEASURE_WORDS]:
        return None
    return sorted({(d["key"], round(d["value"], 6)) for d in ms})


def classify_sentences(solution: str, caption: str, measures: list[dict] | None = None) -> list[dict]:
    """Ogni frase della soluzione in una categoria, con gli span di tutte le occorrenze delle frasi uguali (gruppo):
    - caption_verbatim: copia di una frase della didascalia certificata (anche dentro la citazione "Figure: ...");
    - measure_only: le sole cose sulla figura sono misure di segmenti o angoli che il codice trova uguali alla didascalia;
    - candidate: parla della figura (figure_triggers), anche senza valori: va agli agenti;
    - other: niente che nomini la figura (aritmetica, piani, la lettera della risposta): il generatore la scarta, e la
      sua copertura si misura mandando agli agenti, alla cieca, un campione di queste frasi.
    Le prime due categorie saltano la verifica ordinaria: il valore uguale alla didascalia non prova il ragionamento.
    Ritorna [{"g": gruppo, "spans": [...], "category", "triggers", "text"}] in ordine di prima occorrenza."""
    points = caption_points(caption)
    cap_norm = _norm(caption)
    ms = solution_measures(solution, caption) if measures is None else measures
    groups: dict[str, dict] = {}
    for a, b in sentence_spans(solution):
        s = solution[a:b]
        key = _norm(s)
        if not key:
            continue
        if key in groups:
            groups[key]["spans"].append((a, b))
            continue
        inside = [d for d in ms if a <= d["start"] < b]
        trig = figure_triggers(s, points)
        body = _norm(s.split("Figure:", 1)[1] if "Figure:" in s else s)
        if len(body) >= 15 and body in cap_norm:
            cat = "caption_verbatim"
        elif inside and all(d["verdict"] == "match" for d in inside):
            rest = s
            for d in sorted(inside, key=lambda d: -d["end"]):  # tolte le misure (dal nome al valore), resta il contorno
                lo = d["end"] - len(d["text"]) - a
                rest = rest[:max(0, lo)] + " " + rest[d["end"] - a:]
            left = [t for t in figure_triggers(rest, points) if t.lower() not in _MEASURE_WORDS]
            cat = "measure_only" if not left else "candidate"
        elif trig:
            cat = "candidate"
        else:
            cat = "other"
        groups[key] = {"spans": [(a, b)], "category": cat, "triggers": sorted(set(trig)), "text": s}
    return [{"g": f"s{i:03d}", **v} for i, v in enumerate(groups.values(), 1)]


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


# ---------------------------------------------------------------- pilota v3: fatti per frase, due giudizi, categorie

STANCES = ("asserted", "hypothesis", "rejected")
BASES = ("given", "figure", "chosen", "computed")
KINDS = ("measure", "construction", "relation", "orientation", "existence", "coordinates")
PAGE_ERROR, PAGE_OK = ("contradicts",), ("matches", "none")


def fact_spans(text: str, sentence_spans_: list[tuple[int, int]], quote: str) -> list[tuple[int, int]]:
    """Span della citazione di un fatto dentro ogni occorrenza della sua frase (le frasi uguali sono un gruppo): la
    citazione deve stare nella frase indicata, non altrove nel testo."""
    out = []
    for a, b in sentence_spans_:
        out += [(a + s, a + e) for s, e in quote_spans(text[a:b], quote)]
    return out


def page_class(v) -> str | None:
    """Giudizio sulle direzioni nella pagina: C contraddice il disegno, M non lo contraddice (concorda o non ne parla),
    None incerto o mancante."""
    v = v.strip().lower() if isinstance(v, str) else v
    return "C" if v in PAGE_ERROR else "M" if v in PAGE_OK else None


def resolve_page(judgments: list) -> dict:
    """Come resolve_claim, sulle classi di page_class."""
    classes = [page_class(v) for v in judgments]
    st = resolve_field(classes, [])
    if st["status"] != "need_arbiter":
        return st
    sure = [c for c in classes if c]
    majority = next((c for c in ("C", "M") if sure.count(c) >= 2), None)
    if majority:
        return {"status": "resolved", "value": majority, "how": "maggioranza di tre"}
    return {"status": "quarantine", "value": None, "how": "nessuna maggioranza dopo il terzo giudizio"}


# categorie di un fatto (definizioni dell'utente del 2026-10-22, da tradurre poi in f2_policy.json):
COUNTED = ("visual_error",)  # l'unico errore visivo contato nei tassi
CATEGORIES = {
    "visual_error": "configurazione falsa o non visibile in un fatto asserito: errore, comunque sia orientato il disegno",
    "orientation_error": "direzione nella pagina contraddetta in un fatto asserito come proprietà visibile (base given o "
                         "figure), con la configurazione giusta: conteggio separato",
    "frame_change": "assi o direzioni scelti per calcolare (base chosen o computed) diversi dal disegno, configurazione "
                    "giusta: cambio di coordinate lecito, non è un errore",
    "discarded_hypothesis": "ipotesi falsa scartata esplicitamente e non usata per la risposta: non è un errore visivo",
    "reasoning_flag": "ipotesi falsa (anche scartata) usata per la risposta finale: possibile errore di ragionamento",
    "open_hypothesis": "ipotesi falsa né scartata né usata: conteggio separato, scelta aperta della politica",
    "ok": "configurazione giusta (vera o deduzione compatibile) e nessuna direzione contraddetta che conti",
    "code_match": "solo misure uguali alla didascalia per il codice (scorciatoia, non verificate dagli agenti salvo il "
                  "campione di controllo): non prova il ragionamento",
    "quarantine": "giudizi senza maggioranza sulla configurazione o, se conta, sulla pagina",
}


def fact_category(stance: str, basis: str, used: bool, config: dict, page: dict | None) -> str:
    """Categoria di un'occorrenza di fatto dagli esiti risolti dei verificatori (config: E/K, page: C/M) e da come la
    soluzione lo presenta (stance, basis, used_in_answer dell'agente di classificazione)."""
    if config.get("status") != "resolved":
        return "quarantine"
    if config["value"] == "E":
        if stance == "asserted":
            return "visual_error"
        if used:
            return "reasoning_flag"
        return "discarded_hypothesis" if stance == "rejected" else "open_hypothesis"
    counts_page = basis in ("given", "figure") and stance == "asserted"
    if page is None or page.get("status") != "resolved":
        return "quarantine" if counts_page and page is not None and page.get("status") == "quarantine" else "ok"
    if page["value"] == "C":
        return "orientation_error" if counts_page else "frame_change" if basis in ("chosen", "computed") else "ok"
    return "ok"
