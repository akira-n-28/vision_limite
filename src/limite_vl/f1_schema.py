"""Schema delle uscite di F1-clean: descrizione strutturata di una figura, parser inverso e metriche calcolate dal codice.

La descrizione è fatta di righe `Etichetta: contenuto.` in un ordine canonico, e contiene solo ciò che è disegnato.
Lo stesso schema sarà il preambolo percettivo di F2. Per le figure geometriche (MAVIS, sintetici):

    Figure: geometry diagram.
    Points: A, B, C, D, E.
    Shapes: rectangle ABCD, side CD replaced by equilateral triangle CED outside the rectangle; square DCFG on side DC.
    Segments: AB = 30; AD = 42.
    Angles: ∠BAD = 90°.

- punti in ordine alfabetico; figure in ordine di costruzione; segmenti e angoli in ordine alfabetico;
- i valori si scrivono come compaiono nella figura, in LaTeX senza `$` (2\\sqrt{3}, 1.5 per 3/2);
- un angolo si scrive con il vertice al centro e gli estremi in ordine alfabetico (∠BAD).

`render_geometry` scrive la descrizione dalla verità, `parse_geometry` la rilegge (anche dall'uscita del modello),
`geometry_scores` confronta un'uscita con la verità: inventario dei punti e delle figure (P2), coppie elemento-valore
(P3), tipi e modifiche delle figure (P4). Il confronto è sugli insiemi, quindi l'ordine conta per l'addestramento e non
per la valutazione.
"""
from __future__ import annotations

import re

LINE = re.compile(r"^(?P<label>[A-Z][a-z]+(?: [a-z]+)?):\s*(?P<content>.*?)\.?\s*$")
SHAPE_ITEM = re.compile(
    r"^(?P<kind>[a-z][a-z -]*?) (?P<name>[A-Z]{3,6})"
    r"(?P<circle> with an inscribed circle)?"
    r"(?: on side (?P<attach>[A-Z]{2}))?"
    r"(?:, side (?P<removed>[A-Z]{2}) replaced by (?:equilateral triangle (?P<triangle>[A-Z]{3})"
    r"|a semicircle with diameter (?P<diameter>[A-Z]{2})) (?P<direction>inside|outside) the rectangle)?$")
PAIR = re.compile(r"^(?:∠)?(?P<name>[A-Z]{2,3})\s*=\s*(?P<value>.+)$")


def latex_value(value: str) -> str:
    """Valore normalizzato di facts.norm_value (2√3, 10√3/3, √2, 3/2) come è scritto nella figura, in LaTeX senza `$`.
    Il motore di MAVIS scrive i razionali senza radice come decimali con al più due cifre (3/2 -> 1.5, 44/3 -> 14.67:
    immagini del job 30049) e le frazioni con la radice come frazioni (22√3/3 -> 22\\sqrt{3}/3)."""
    m = re.fullmatch(r"(\d+)/(\d+)", value)
    if m:
        return f"{int(m.group(1)) / int(m.group(2)):.2f}".rstrip("0").rstrip(".")
    return re.sub(r"√(\d+(?:\.\d+)?)", r"\\sqrt{\1}", value)


def canonical_value(value: str) -> str:
    """Forma di confronto: senza spazi, `$`, `°` e `{}` attorno a numeri semplici (\\sqrt3 = \\sqrt{3})."""
    v = value.replace("$", "").replace("°", "").replace("^\\circ", "").replace(" ", "")
    return re.sub(r"\\sqrt\{(\d+(?:\.\d+)?)\}", r"\\sqrt\1", v)


def angle_name(vertex: str, ends: str) -> str:
    a, b = sorted(ends)
    return a + vertex + b


def angle_key(name: str) -> str:
    """∠BAD, DAB -> chiave A|BD (vertice al centro, estremi ordinati)."""
    name = name.lstrip("∠")
    return name[1] + "|" + "".join(sorted(name[0] + name[2]))


def side_key(name: str) -> str:
    return "".join(sorted(name))


def shape_text(s: dict) -> str:
    text = f"{s['kind']} {s['name']}"
    if s.get("inscribed_circle"):
        text += " with an inscribed circle"
    if s.get("attach"):
        text += f" on side {s['attach']}"
    if s.get("modification") == "equilateral triangle":
        text += (f", side {s['removed']} replaced by equilateral triangle {s['triangle']} {s['direction']} the "
                 "rectangle")
    elif s.get("modification") == "semicircle":
        text += f", side {s['removed']} replaced by a semicircle with diameter {s['removed']} {s['direction']} the rectangle"
    return text


def render_geometry(truth: dict) -> str:
    """Descrizione canonica da {"points": [..], "shapes": [..], "sides": {AB: valore}, "angles": {BAD: valore}}."""
    lines = ["Figure: geometry diagram.", f"Points: {', '.join(sorted(truth['points']))}."]
    if truth["shapes"]:
        lines.append("Shapes: " + "; ".join(shape_text(s) for s in truth["shapes"]) + ".")
    if truth["sides"]:
        lines.append("Segments: " + "; ".join(f"{k} = {v}" for k, v in sorted(truth["sides"].items())) + ".")
    if truth["angles"]:
        lines.append("Angles: " + "; ".join(f"∠{k} = {v}°" for k, v in sorted(truth["angles"].items())) + ".")
    return "\n".join(lines)


def parse_geometry(text: str) -> dict:
    """Rilegge una descrizione (anche imperfetta): {"figure", "points", "shapes", "sides", "angles", "unparsed"}.
    Le chiavi di lati e angoli sono canoniche (side_key, angle_key) e i valori in forma di confronto."""
    out = {"figure": None, "points": set(), "shapes": [], "sides": {}, "angles": {}, "unparsed": []}
    for raw in text.strip().splitlines():
        if not raw.strip():
            continue
        m = LINE.match(raw.strip())
        if not m:
            out["unparsed"].append(raw)
            continue
        label, content = m["label"], m["content"].strip()
        if label == "Figure":
            out["figure"] = content
        elif label == "Points":
            out["points"] = {p.strip() for p in content.split(",") if p.strip()}
        elif label == "Shapes":
            for item in content.split(";"):
                sm = SHAPE_ITEM.match(item.strip())
                if sm:
                    out["shapes"].append({k: v for k, v in sm.groupdict().items() if v} | {"circle": bool(sm["circle"])})
                else:
                    out["unparsed"].append(item.strip())
        elif label in ("Segments", "Angles"):
            for item in content.split(";"):
                pm = PAIR.match(item.strip())
                if not pm:
                    out["unparsed"].append(item.strip())
                elif label == "Segments" and len(pm["name"]) == 2:
                    out["sides"][side_key(pm["name"])] = canonical_value(pm["value"])
                elif label == "Angles" and len(pm["name"]) == 3:
                    out["angles"][angle_key(pm["name"])] = canonical_value(pm["value"])
                else:
                    out["unparsed"].append(item.strip())
        else:
            out["unparsed"].append(raw)
    return out


def _pr(pred: set, true: set) -> dict:
    tp = len(pred & true)
    return {"tp": tp, "pred": len(pred), "true": len(true),
            "precision": tp / len(pred) if pred else 1.0, "recall": tp / len(true) if true else 1.0}


def geometry_scores(pred_text: str, truth: dict) -> dict:
    """Metriche di un'uscita contro la verità. Punti e figure: precisione e richiamo (inventati = pred - tp).
    Lati e angoli: coppie (chiave, valore) giuste, valori sbagliati su chiavi giuste, chiavi inventate."""
    p = parse_geometry(pred_text)
    t = parse_geometry(render_geometry(truth))
    scores = {"parsed": not p["unparsed"] and p["figure"] is not None, "unparsed": len(p["unparsed"]),
              "points": _pr(p["points"], t["points"]),
              "shapes": _pr({s["name"] for s in p["shapes"]}, {s["name"] for s in t["shapes"]})}
    true_kinds = {s["name"]: s for s in t["shapes"]}
    matched = [s for s in p["shapes"] if s["name"] in true_kinds]
    scores["shape_kind_ok"] = sum(s["kind"] == true_kinds[s["name"]]["kind"] for s in matched)
    scores["shape_modification_ok"] = sum(
        (s.get("removed"), s.get("direction")) == (true_kinds[s["name"]].get("removed"), true_kinds[s["name"]].get("direction"))
        for s in matched)
    scores["shapes_matched"] = len(matched)
    for kind in ("sides", "angles"):
        pv, tv = p[kind], t[kind]
        scores[kind] = {"true": len(tv), "pred": len(pv),
                        "pairs_ok": sum(k in tv and tv[k] == v for k, v in pv.items()),
                        "wrong_value": sum(k in tv and tv[k] != v for k, v in pv.items()),
                        "invented": sum(k not in tv for k in pv)}
    return scores
