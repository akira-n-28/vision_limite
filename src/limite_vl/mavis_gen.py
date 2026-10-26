"""diagram_generation di MAVIS-Geometry contro la didascalia: due rese dello stesso programma del motore.

diagram_generation descrive la costruzione con modelli fissi in minuscolo: per ogni figura "[ordinale,] [attaccata al
lato xy,] <verbo> a <forma> <nome> [modifica]", poi le misure ("angle cba is 30 degrees", "side ab measures 42 units").
La didascalia descrive lo stesso programma con altri modelli e i nomi in maiuscolo.

- `measures` legge angoli e lati da tutti e due i testi con le stesse regole, dopo il minuscolo;
- `figures` legge le figure di diagram_generation: tipo, nome, lato di attacco, lato tolto e sostituito da un triangolo
  equilatero o da un semicerchio, direzione;
- `compare` elenca dove le due descrizioni non coincidono, per categoria.

Il confronto prova la coerenza fra le due rese, non che l'immagine le rispetti: quello resta all'audit della leggibilità.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .facts import VALUE, norm_value
from .mavis import _direction

SENTENCE = re.compile(r"(?<=\.)\s+")
ORDINAL = re.compile(r"^(?:firstly|secondly|thirdly|fourthly|fifthly|sixthly|seventhly|eighthly),\s*")
# forme di diagram_generation, dalla più lunga (l'alternanza prova le voci in ordine)
FORMS = (
    (r"quadrilateral with opposite sides parallel and equal in length", "parallelogram"),
    (r"circle and its circumscribing square", "square"),
    (r"right-angled triangle", "right triangle"),
    (r"right triangle", "right triangle"),
    (r"triangle with a right angle", "right triangle"),
    (r"triangle with a 90-degree angle", "right triangle"),
    (r"triangle with an? \d+(?:\.\d+)?-degree angle", "triangle"),
    (r"isosceles triangle", "isosceles triangle"),
    (r"equilateral triangle", "equilateral triangle"),
    (r"modified rectangle", "rectangle"),
    (r"rectangle", "rectangle"),
    (r"parallelogram", "parallelogram"),
    (r"rhombus", "rhombus"),
    (r"square", "square"),
    (r"kite", "kite"),
    (r"trapezoid", "trapezoid"),
    (r"sector", "sector"),
    (r"semi-?circle", "semicircle"),
    (r"triangle", "triangle"),
    (r"circle", "circle"),
)
FIGURE = re.compile(r"\ban? (?P<form>" + "|".join(p for p, _ in FORMS) + r")\s+(?P<name>[a-z]{3,6})\b(?P<rest>.*)$")
NOT_NAMES = {"with", "and", "that", "which", "where", "whose", "has", "having", "its", "the", "inside", "outside"}
ISOSCELES = re.compile(r"\btwo (?:congruent|equal-length|equal) sides\b|\bwhere [a-z]{2} = [a-z]{2}\b|\bas its base\b"
                       r"|\bas its vertex angle\b")
INSCRIBED_CIRCLE = re.compile(r"inscribed circle|circle that fits")
ATTACH = re.compile(r"\b(?:edge|side) (?P<edge>[a-z]{2}),")
MODIFICATION = re.compile(r"equilateral triangle|semi-?circ")
ANCHOR = re.compile(r"\bangle (?P<angle>[a-z]{3})\b"
                    r"|\b(?:side|segment|edge|base)\s+(?P<side>[a-z]{2})\b"
                    r"|\bpoints? (?P<p1>[a-z]) and (?:points? )?(?P<p2>[a-z])\b"
                    r"|\bfrom (?:point )?(?P<f1>[a-z]) to (?:point )?(?P<f2>[a-z])\b")
NAME = re.compile(r"\b[A-Z]{2,6}\b")
POINT_PAIRS = re.compile(r"\b(?:points? ([A-Z]) and (?:points? )?([A-Z])|from (?:point )?([A-Z]) to (?:point )?([A-Z]))\b",
                         re.IGNORECASE)
FAMILY_WORDS = {
    "square": ("square",), "rectangle": ("rectangle", "rectangular"), "parallelogram": ("parallelogram",),
    "rhombus": ("rhombus",), "kite": ("kite",), "trapezoid": ("trapezoid",), "sector": ("sector",),
    "semicircle": ("semicircle", "semi-circle"), "circle": ("circle",), "triangle": ("triangle", "isosceles"),
}
QUALIFIERS = {"right triangle": ("right triangle", "right-angled", "right angle"), "isosceles triangle": ("isosceles",),
              "equilateral triangle": ("equilateral",)}


def family(kind: str) -> str:
    return "triangle" if kind.endswith("triangle") else kind


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text.strip()) if s.strip()]


def _clean(sentence: str) -> str:
    return ORDINAL.sub("", sentence.strip()).rstrip(".")


def _form_kind(form: str) -> str:
    for pattern, kind in FORMS:
        if re.fullmatch(pattern, form):
            return kind
    raise ValueError(form)


def is_side(name: str, side: str) -> bool:
    """`side` è un lato del poligono `name` (vertici consecutivi, anche l'ultimo col primo)."""
    if len(side) != 2 or side[0] not in name or side[1] not in name:
        return False
    return abs(name.index(side[0]) - name.index(side[1])) in (1, len(name) - 1)


@dataclass
class Figure:
    kind: str
    name: str
    attach: str | None = None
    modification: str | None = None  # "equilateral triangle" o "semicircle" al posto di un lato del rettangolo
    removed: str | None = None
    direction: str | None = None
    inscribed_circle: bool = False
    sentence: str = ""


def figures(generation: str) -> tuple[list[Figure], list[str]]:
    """Figure di diagram_generation e frasi di costruzione non lette (senza cifre e senza una forma nota)."""
    out, unread = [], []
    for raw in sentences(generation.lower()):
        s = _clean(raw)
        m = FIGURE.search(s)
        if not m or m["name"] in NOT_NAMES or len(set(m["name"])) != len(m["name"]):
            if s and not re.search(r"\d", s):
                unread.append(s)
            continue
        kind, rest, name = _form_kind(m["form"]), m["rest"], m["name"]
        if kind == "triangle" and ISOSCELES.search(rest):
            kind = "isosceles triangle"
        f = Figure(kind=kind, name=name, sentence=s)
        a = ATTACH.search(s[:m.start()])
        f.attach = a["edge"] if a else None
        if kind == "square":
            f.inscribed_circle = m["form"].startswith("circle") or bool(INSCRIBED_CIRCLE.search(rest))
        mod = MODIFICATION.search(rest)
        if kind == "rectangle" and mod:
            f.modification = "equilateral triangle" if mod.group().startswith("equilateral") else "semicircle"
            sides = {t for t in re.findall(r"\b([a-z]{2})\b", rest) if is_side(name, t)}
            f.removed = sides.pop() if len(sides) == 1 else None
            f.direction = _direction(rest)
        out.append(f)
    return out, unread


def measures(text: str) -> dict:
    """Angoli e lati con valore: {"angle": {vertice|estremi: valore}, "side": {lettere: valore}, "conflicts", "unread"}.
    Stesse regole per didascalia e diagram_generation; le frasi di costruzione si saltano."""
    out = {"angle": {}, "side": {}, "conflicts": [], "unread": []}
    for raw in sentences(text.lower()):
        s = _clean(raw)
        if FIGURE.search(s):
            continue
        values = [(m.start(), m.group().strip()) for m in VALUE.finditer(s)]
        if not values:
            continue
        anchors = []
        for m in ANCHOR.finditer(s):
            if m["angle"]:
                n = m["angle"]
                anchors.append((m.start(), "angle", n[1] + "|" + "".join(sorted(n[0] + n[2]))))
            else:
                pair = m["side"] or (m["p1"] or m["f1"]) + (m["p2"] or m["f2"])
                anchors.append((m.start(), "side", "".join(sorted(pair))))
        if not anchors:
            out["unread"].append(s)
            continue
        for i, (pos, kind, key) in enumerate(anchors):
            end = anchors[i + 1][0] if i + 1 < len(anchors) else len(s)
            vals = [v for p, v in values if pos < p < end]
            if not vals:
                out["unread"].append(s)
                continue
            value = norm_value(vals[0])
            if key in out[kind] and out[kind][key] != value:
                out["conflicts"].append(f"{kind}:{key} {out[kind][key]} / {value}")
            out[kind][key] = value
    return out


def _letters(text: str) -> set[str]:
    letters = {c for n in NAME.findall(text) for c in n}
    for m in POINT_PAIRS.finditer(text):
        letters |= {c.upper() for c in m.groups() if c}
    return letters


def compare(caption: str, generation: str) -> dict:
    """Differenze fra didascalia e diagram_generation: {"issues": [(categoria, dettaglio)], "info": [...]}.
    Le categorie in "issues" indicano una contraddizione o qualcosa che le regole non leggono; "info" raccoglie le
    differenze di formulazione note (qualificatori o direzioni non ripetuti nella didascalia)."""
    issues, info = [], []
    cm, gm = measures(caption), measures(generation)
    for kind in ("angle", "side"):
        for k in sorted(set(cm[kind]) | set(gm[kind])):
            if k not in gm[kind]:
                issues.append(("misura_solo_didascalia", f"{kind}:{k}={cm[kind][k]}"))
            elif k not in cm[kind]:
                issues.append(("misura_solo_generazione", f"{kind}:{k}={gm[kind][k]}"))
            elif cm[kind][k] != gm[kind][k]:
                issues.append(("misura_diversa", f"{kind}:{k} {cm[kind][k]} / {gm[kind][k]}"))
    for text, m in (("didascalia", cm), ("generazione", gm)):
        issues += [(f"misura_ripetuta_diversa_{text}", c) for c in m["conflicts"]]
        issues += [(f"misura_non_letta_{text}", s[:160]) for s in m["unread"]]
    figs, unread = figures(generation)
    issues += [("costruzione_non_letta", s[:160]) for s in unread]
    if not figs:
        issues.append(("nessuna_figura", generation[:160]))
    cap_sents = sentences(caption)
    names = set(NAME.findall(caption))
    apexes: set[str] = set()  # vertice del triangolo equilatero: nella didascalia (ABCED) ma non nel nome del rettangolo
    for f in figs:
        up = f.name.upper()
        pentagon = ({n for n in names if len(n) == len(up) + 1 and set(up) <= set(n)}
                    if f.modification == "equilateral triangle" else set())
        apexes |= {c for n in pentagon for c in n} - set(up)
        wanted = {up} | pentagon
        sents = [s for s in cap_sents if any(re.search(rf"\b{w}\b", s) for w in wanted)]
        if not sents:
            issues.append(("figura_non_nominata", f"{f.kind} {up}"))
            continue
        text = " ".join(sents).lower()
        fam = family(f.kind)
        if not any(w in text for w in FAMILY_WORDS[fam]):
            others = sorted(k for k, ws in FAMILY_WORDS.items() if k != fam and any(w in text for w in ws))
            issues.append(("figura_tipo_diverso" if others else "figura_tipo_assente", f"{up}: {f.kind} / {others}"))
        elif f.kind in QUALIFIERS and not any(w in text for w in QUALIFIERS[f.kind]):
            info.append(("qualificatore_non_ripetuto", f"{up}: {f.kind}"))
        if f.modification:
            mod_sents = [s for s in sents if MODIFICATION.search(s.lower())]
            if not mod_sents:
                issues.append(("modifica_assente", f"{up}: {f.modification}"))
                continue
            mod_text = " ".join(mod_sents).lower()
            if f.modification == "equilateral triangle" and "equilateral" not in mod_text:
                issues.append(("modifica_diversa", f"{up}: {f.modification}"))
            d = _direction(mod_text)
            if f.direction is None:
                info.append(("direzione_assente_generazione", up))
            elif d is None:
                info.append(("direzione_non_ripetuta", up))
            elif d != f.direction:
                issues.append(("direzione_opposta", f"{up}: {f.direction} / {d}"))
            if f.removed is None:
                issues.append(("lato_tolto_non_letto", f"{up}: {f.sentence[:120]}"))
            elif not re.search(rf"\b{f.removed}\b", mod_text):
                issues.append(("lato_tolto_diverso", f"{up}: {f.removed.upper()}"))
    gen_letters = {c.upper() for f in figs for c in f.name} | apexes
    gen_letters |= {c.upper() for k in list(gm["angle"]) + list(gm["side"]) for c in k if c != "|"}
    cap_letters = _letters(caption)
    if cap_letters - gen_letters:
        issues.append(("punti_solo_didascalia", "".join(sorted(cap_letters - gen_letters))))
    if gen_letters - cap_letters:
        issues.append(("punti_solo_generazione", "".join(sorted(gen_letters - cap_letters))))
    return {"issues": issues, "info": info, "figures": len(figs),
            "measures": len(gm["angle"]) + len(gm["side"])}
