"""Fatti delle didascalie di MAVIS-Geometry estratti dal testo con regole (analisi degli errori di F1, domande di F1c).

Le didascalie di MAVIS sono generate da modelli di frase: figure ("ABCD is a square", "the side CD is extended into
an equilateral triangle", "Triangle EFG is isosceles"), angoli ("Angle BAD has a measure of 90 degrees") e lati
("Side AB is 4 units long", "The distance between points A and B is 80"). `mavis_facts` li riduce a chiavi
canoniche (lettere ordinate; per gli angoli vertice|estremi) con il tipo o il valore normalizzato, così due testi
si confrontano a meno della formulazione; tiene anche il nome e il valore come sono scritti, per le domande.
"""
from __future__ import annotations

import re

SHAPE_WORDS = ("semicircle", "sector", "square", "rectangle", "parallelogram", "rhombus", "trapezoid", "kite",
               "triangle", "circle")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
NAME = re.compile(r"\b[A-Z]{2,6}\b")
POINTS_PAIR = re.compile(r"\bpoints? ([A-Z]) and ([A-Z])\b")
# anche con il denominatore dopo la radice (10*√(3)/3): senza, la parte /3 si perdeva (audit di F1c)
VALUE = re.compile(r"(?:\d+(?:\.\d+)?\s*\*?\s*)?√\s*\(?\s*\d+(?:\.\d+)?\s*\)?(?:\s*/\s*\d+)?|\d+(?:\.\d+)?(?:\s*/\s*\d+)?")
SHAPE = re.compile(r"\b(?:(isosceles|equilateral|right(?:-angled)?)\s+)?(" + "|".join(SHAPE_WORDS) + r")\b", re.I)


def norm_value(v: str) -> str:
    return re.sub(r"[\s*()]", "", v)


def mavis_facts(text: str) -> dict:
    """Fatti di una didascalia MAVIS: {"shape": {lettere: tipo}, "angle": {vertice|estremi: valore},
    "side": {lettere: valore}}, l'ordine delle figure, e per ogni fatto ("shape:CHIAVE", ...) il nome scritto
    ("display") e il valore scritto ("raw"). Una figura è un nome di 3+ lettere (o un lato di 2 lettere
    prolungato in una figura: chiave ext:XY) con la parola di forma che lo precede subito ("triangle ABC") o, se
    manca, la prima che lo segue prima del nome successivo."""
    facts = {"shape": {}, "angle": {}, "side": {}, "shape_order": [], "display": {}, "raw": {}}
    text = re.sub(r"semi-circle", "semicircle", text, flags=re.I)
    for s in SENTENCE.split(text.strip()):
        names = [(m.start(), m.end(), m.group()) for m in NAME.finditer(s)]
        shapes = [(m.start(), m.end(), ((m.group(1) or "").lower().replace("-angled", "") + " " + m.group(2).lower()).strip())
                  for m in SHAPE.finditer(s)]
        used, assigned = set(), {}
        for idx, (ns, _, name) in enumerate(names):  # forma subito prima del nome
            for si, (ss, se, kind) in enumerate(shapes):
                if si not in used and 0 <= ns - se <= 3:
                    assigned[idx] = kind
                    used.add(si)
        for idx, (_, ne, name) in enumerate(names):  # altrimenti la prima forma dopo, prima del nome successivo
            if idx in assigned:
                continue
            nxt = names[idx + 1][0] if idx + 1 < len(names) else len(s)
            for si, (ss, _, kind) in enumerate(shapes):
                if si not in used and ne <= ss < nxt:
                    assigned[idx] = kind
                    used.add(si)
                    break
        for idx, kind in assigned.items():
            name = names[idx][2]
            key = ("ext:" if len(name) == 2 else "") + "".join(sorted(set(name)))
            if key not in facts["shape"]:
                facts["shape_order"].append(key)
                facts["display"][f"shape:{key}"] = name
            facts["shape"][key] = kind
        raw_values = [m.group().strip() for m in VALUE.finditer(s)]
        if not raw_values:
            continue
        three = [n for _, _, n in names if len(n) == 3]
        two = [n for _, _, n in names if len(n) == 2] + [a + b for a, b in POINTS_PAIR.findall(s)]
        if re.search(r"\bangle\b", s, re.I) and three:
            n = three[0]
            key = n[1] + "|" + "".join(sorted(n[0] + n[2]))
            facts["angle"][key] = norm_value(raw_values[0])
            facts["display"][f"angle:{key}"] = n
            facts["raw"][f"angle:{key}"] = raw_values[0]
        elif two:
            key = "".join(sorted(two[0]))
            facts["side"][key] = norm_value(raw_values[0])
            facts["display"][f"side:{key}"] = two[0]
            facts["raw"][f"side:{key}"] = raw_values[0]
    return facts


def compare_facts(ref: dict, gen: dict) -> dict:
    out = {}
    for kind in ("shape", "angle", "side"):
        r, g = ref[kind], gen[kind]
        out[kind] = {"ok": sum(k in g and g[k] == v for k, v in r.items()),
                     "wrong": sum(k in g and g[k] != v for k, v in r.items()),
                     "missing": sum(k not in g for k in r), "extra": sum(k not in r for k in g),
                     "ref": len(r), "gen": len(g)}
    out["shape_hits"] = [k in gen["shape"] and gen["shape"][k] == ref["shape"][k] for k in ref["shape_order"]]
    return out
