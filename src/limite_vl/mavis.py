"""MAVIS-Geometry come dati di F2 esatti per costruzione: domanda, risposta e ponte testuale per Violetto.

La didascalia di MAVIS è generata dal motore che disegna la figura (tutte le misure citate sono disegnate in
caption.jpg), ma due formule sono letteralmente false (mavis_caption_survey.py, job 29799, su 199.050 didascalie):
- "ABCED is a rectangle, with the side CD extended into an equilateral triangle" (66.517 frasi): la figura è un
  pentagono, il rettangolo ABCD con il lato CD tolto e sostituito dal triangolo equilatero CED;
- "ABCD is a rectangle, ... the side CD ... into a semi-circle" (29.789 frasi): il lato CD è tolto e sostituito da
  un semicerchio di diametro CD.
La direzione (dentro o fuori il rettangolo) manca in molte frasi; diagram_generation la dice sempre, nella parte
che costruisce quella figura. bridge_caption riscrive queste frasi in modo esatto e lascia le altre invariate.

Anche le domande vision_dominant nominano il lato tolto ("Side CD extends into an equilateral triangle") e non ripetono
le premesse sulle forme ("ECFG is a rectangle"): nell'audit del 2026-10-21 (outputs/stem_audit/mavis-f2) 32 domande su
152 escluse o in quarantena per coerenza, nessuna risposta sbagliata. structure_question le sostituisce con le frasi
senza numeri della didascalia riscritta (forme e costruzioni) e la richiesta finale: le misure restano solo nella figura.
"""
from __future__ import annotations

import re

_PREFIX = r"(?P<prefix>(?:[A-Z][\w']*(?: [\w']+){0,3}, )?)"  # "Besides, ", "What's more, ", "On top of that, "
_RECTANGLE = re.compile(_PREFIX + r"(?P<poly>[A-Z]{4,5}) is a rectangle\b(?P<rest>.*)$")
_SHAPE = re.compile(r"equilateral triangle|semi-?circ\w*", re.IGNORECASE)
_SIDE = re.compile(r"\bside (?P<side>[A-Z]{2})\b")
_INWARD = re.compile(r"\b(inside|inward|inwards|internal|internally|within|inscribe|inscribed|inscribing)\b", re.IGNORECASE)
_OUTWARD = re.compile(r"\b(outside|outward|outwards|external|externally|beyond)\b", re.IGNORECASE)
_ORDINAL = re.compile(r"\b(?:firstly|secondly|thirdly|fourthly|fifthly|sixthly),\s*", re.IGNORECASE)
# connettivo iniziale ("Besides, ", "What's more, "): parole minuscole dopo l'iniziale, quindi mai nomi di punti come "ABCD"
_CONNECTOR = re.compile(r"^[A-Z][a-z']*(?: [a-z']+){0,3}, (?=[A-Z])")
_DIGIT = re.compile(r"\d")
_DIAMETER = re.compile(r"semicircle with diameter (?P<side>[A-Z]{2})\b")
_MC_LETTERS = "ABCD"


def sentences(caption: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", caption.strip()) if s.strip()]


def question_sentences(question: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", question.strip()) if s.strip()]


def _direction(text: str) -> str | None:
    inward, outward = bool(_INWARD.search(text)), bool(_OUTWARD.search(text))
    return "inside" if inward and not outward else "outside" if outward and not inward else None


def _generation_direction(generation: str, rect: str, side: str) -> str | None:
    """Direzione dalla parte di diagram_generation che costruisce il rettangolo `rect` con il lato `side`."""
    for part in _ORDINAL.split(generation):
        low = part.lower()
        if rect.lower() in low and side.lower() in low and ("triangle" in low or "semicircle" in low or "semi-circle" in low):
            clause = re.split(r"\.\s", part)[0]
            return _direction(clause)
    return None


def rewrite_sentence(sentence: str, generation: str) -> tuple[str, str]:
    """(frase, esito): esito "same" se non è una figura modificata, "rewritten", oppure "failed" (frase invariata)."""
    m = _RECTANGLE.match(sentence)
    shape = _SHAPE.search(m["rest"]) if m else None
    if not shape:
        return sentence, "same"
    s = _SIDE.search(m["rest"])
    if not s:
        return sentence, "failed"
    poly, side, shape = m["poly"], s["side"], shape.group(0).lower()
    a, b = poly.find(side[0]), poly.find(side[1])
    if a < 0 or b < 0:
        return sentence, "failed"
    if shape.startswith("equilateral"):
        if len(poly) != 5 or abs(a - b) != 2:
            return sentence, "failed"
        new = poly[min(a, b) + 1]
        rect = poly.replace(new, "")
        what = f"an equilateral triangle {poly[min(a, b)]}{new}{poly[max(a, b)]}"
        name = f"{poly} is a pentagon: the rectangle {rect}"
    else:
        if len(poly) != 4 or abs(a - b) not in (1, 3):
            return sentence, "failed"
        rect = poly
        what = f"a semicircle with diameter {side}"
        name = f"{poly} is the rectangle {rect}"
    where = _direction(m["rest"]) or _generation_direction(generation, rect, side)
    if where is None:
        return sentence, "failed"
    return (f"{m['prefix']}{name} with side {side} removed and replaced by {what} drawn {where} the rectangle.",
            "rewritten")


def bridge_caption(caption: str, generation: str) -> tuple[str, dict]:
    """Didascalia con le figure modificate descritte in modo esatto, e conteggi {"rewritten", "failed"}."""
    out, counts = [], {"rewritten": 0, "failed": 0}
    for s in sentences(caption):
        new, what = rewrite_sentence(s, generation)
        if what != "same":
            counts[what] += 1
        out.append(new)
    return " ".join(out), counts


def arc_ask(ask: str, diameters: list[str]) -> tuple[str, bool]:
    """"The length of IJ" quando il lato IJ è sostituito da un semicerchio: MAVIS intende l'arco (risposta con π in tutti
    i 19 casi del campione mavis-f2b), ma dopo "side IJ removed" la domanda è ambigua. La richiesta nomina l'arco."""
    if re.search(r"\barc\b", ask, re.IGNORECASE):
        return ask, False
    for d in diameters:
        pat = re.compile(rf"\b({d}|{d[::-1]})\b")
        if pat.search(ask):
            return pat.sub(lambda m: f"the semicircular arc {m[1]}", ask, count=1), True
    return ask, False


def structure_question(bridged_caption: str, question: str) -> tuple[str, dict]:
    """Domanda F2: le frasi senza cifre della didascalia riscritta (tipi di forma e costruzioni, nessuna misura), poi
    l'ultima frase della domanda vision_dominant (la richiesta, con arc_ask). Conteggi {"structure", "ask_digits",
    "connector", "arc"}."""
    struct = [s for s in sentences(bridged_caption) if not _DIGIT.search(s)]
    connector = bool(struct and _CONNECTOR.match(struct[0]))
    if connector:
        struct[0] = _CONNECTOR.sub("", struct[0])
    ask, arc = arc_ask(question_sentences(question)[-1], [m["side"] for s in struct for m in _DIAMETER.finditer(s)])
    return " ".join(struct + [ask]), {"structure": len(struct), "ask_digits": bool(_DIGIT.search(ask)), "connector": connector,
                                      "arc": arc}


def mc_question(question: str, choices: list[str]) -> tuple[str, str]:
    """Domanda con le opzioni A-D (stesso formato dell'audit STEM) e lettere valide."""
    letters = _MC_LETTERS[:len(choices)]
    return question.strip() + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, choices)), letters


def f2_problem(meta: dict) -> dict | None:
    """Problema F2 da un JSON dello shard caption: domanda con la struttura della figura e la richiesta di
    vision_dominant (structure_question: le misure vanno lette dalla figura), risposta (lettera e contenuto
    dell'opzione), ponte esatto. None se la didascalia non si riscrive tutta, se la richiesta contiene cifre o se chiede
    l'arco di un semicerchio e la risposta non contiene π."""
    te = meta["text_en"]
    choices, ai = meta.get("choices"), meta.get("answer_index")
    if not choices or not isinstance(ai, int) or not 0 <= ai < len(choices):
        return None
    caption, counts = bridge_caption(te["caption"], te["diagram_generation"])
    if counts["failed"]:
        return None
    text, qc = structure_question(caption, te["vision_dominant_question"])
    if qc["ask_digits"] or (qc["arc"] and "π" not in str(choices[ai])):
        return None
    question, letters = mc_question(text, [str(c) for c in choices])
    return {"key": f"mavis-{meta['id']}", "depth": meta["depth"], "question": question, "answer_letter": letters[ai],
            "answer_text": str(choices[ai]), "caption": caption, "rewritten": counts["rewritten"],
            "structure": qc["structure"], "question_vd": te["vision_dominant_question"], "cot": te.get("CoT_reasoning", "")}
