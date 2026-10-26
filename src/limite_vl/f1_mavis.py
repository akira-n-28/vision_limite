"""Dati di F1-clean da MAVIS-Geometry: verità della figura da diagram_generation, descrizione canonica e domande brevi.

La verità viene dal programma del motore (diagram_generation, letto da mavis_gen.figures e mavis_gen.measures). Si
usa solo se la didascalia lo conferma in tutto (mavis_gen.compare senza segnalazioni: job 30025, 0 su 199.050). Dalla
didascalia si prende soltanto il vertice del triangolo equilatero che sostituisce un lato, che diagram_generation non
nomina (ABCED: il vertice è E).

Compiti (scelti dalla chiave del campione):
- descrizione: f1_schema.render_geometry, la stessa che sarà il preambolo di F2;
- domande brevi (formato f1c.qa_messages): lati, angoli, tipi di figura, lato sostituito e direzione, lato in comune,
  numero di punti, presenza di punti e figure con risposte "yes" e "no" in egual numero.
"""
from __future__ import annotations

from .f1_schema import angle_name, latex_value, render_geometry
from .f1c import key_rng, pick_questions, qa_messages
from .mavis_gen import NAME, compare, figures, measures

DESCRIBE_PROMPTS = ("Describe the figure.", "Describe the figure in the structured format.",
                    "List the points, shapes, segments and angles shown in the figure.")


def mavis_truth(caption: str, generation: str) -> tuple[dict | None, str]:
    """(verità, motivo): verità None se didascalia e diagram_generation non coincidono o il vertice del triangolo non
    si trova in modo univoco."""
    if compare(caption, generation)["issues"]:
        return None, "didascalia_diversa"
    figs, _ = figures(generation)
    names = set(NAME.findall(caption))
    shapes, letters = [], set()
    for f in figs:
        up = f.name.upper()
        s = {"kind": f.kind, "name": up}
        if f.attach:
            s["attach"] = f.attach.upper()
        if f.inscribed_circle:
            s["inscribed_circle"] = True
        if f.modification:
            removed = f.removed.upper()
            s.update(modification=f.modification, removed=removed, direction=f.direction)
            if f.modification == "equilateral triangle":
                apex = {c for n in names if len(n) == len(up) + 1 and set(up) <= set(n) for c in n} - set(up)
                if len(apex) != 1:
                    return None, "vertice_non_univoco"
                (a,) = apex
                s["triangle"] = removed[0] + a + removed[1]
                letters.add(a)
        letters |= set(up)
        shapes.append(s)
    gm = measures(generation)
    sides = {k.upper(): latex_value(v) for k, v in gm["side"].items()}
    angles = {angle_name(k[0].upper(), k[2:].upper()): latex_value(v) for k, v in gm["angle"].items()}
    letters |= {c for k in list(sides) + list(angles) for c in k}
    return {"points": sorted(letters), "shapes": shapes, "sides": sides, "angles": angles}, "ok"


def describe(key: str, truth: dict) -> tuple[str, str]:
    """(richiesta, risposta) per il compito di descrizione."""
    return key_rng(key + "/describe").choice(DESCRIBE_PROMPTS), render_geometry(truth)


def _next_letter(points: list[str]) -> str:
    return chr(ord(max(points)) + 1) if max(points) < "Z" else "Z"


def mavis_questions(key: str, truth: dict) -> list[dict]:
    """Domande con risposta presa dalla verità: {"type", "q", "answer"}."""
    rng = key_rng(key + "/mavis-f1")
    out = [{"type": "side_len", "q": f"What is the length of {k}?", "answer": v} for k, v in truth["sides"].items()]
    out += [{"type": "angle_val", "q": f"What is the measure of angle {k}?", "answer": f"{v}°"}
            for k, v in truth["angles"].items()]
    shapes = truth["shapes"]
    for i, s in enumerate(shapes):
        if s.get("modification"):
            what = "an equilateral triangle" if s["modification"] == "equilateral triangle" else "a semicircle"
            out.append({"type": "removed_side", "q": f"Which side of rectangle {s['name']} is replaced by {what}?",
                        "answer": s["removed"]})
            part = (f"the equilateral triangle {s['triangle']}" if s["modification"] == "equilateral triangle"
                    else f"the semicircle on {s['removed']}")
            out.append({"type": "direction", "q": f"Is {part} drawn inside or outside rectangle {s['name']}?",
                        "answer": s["direction"]})
        else:
            out.append({"type": "shape_type", "q": f"What type of shape is {s['name']}?", "answer": s["kind"]})
        if s.get("attach"):
            owners = [p["name"] for p in shapes[:i] if set(s["attach"]) <= set(p["name"])]
            owners += [p["triangle"] for p in shapes[:i] if p.get("triangle") and set(s["attach"]) <= set(p["triangle"])]
            if len(owners) == 1:
                out.append({"type": "shared_side", "q": f"Which side do {owners[0]} and {s['name']} share?",
                            "answer": s["attach"]})
    points = truth["points"]
    out.append({"type": "count_points", "q": "How many labeled points are in the figure?", "answer": str(len(points))})
    out.append({"type": "exists_point", "q": f"Is there a point labeled {rng.choice(points)}?", "answer": "yes"})
    out.append({"type": "exists_point", "q": f"Is there a point labeled {_next_letter(points)}?", "answer": "no"})
    real = rng.choice(shapes)["name"]
    fake = real[:-1] + _next_letter(points)
    out.append({"type": "exists_shape", "q": f"Is there a shape named {real}?", "answer": "yes"})
    out.append({"type": "exists_shape", "q": f"Is there a shape named {fake}?", "answer": "no"})
    return out


def questions_example(key: str, truth: dict) -> tuple[str, str, list[dict]]:
    """(richiesta, risposta, domande scelte) per il compito di domande brevi."""
    picked = pick_questions(key, mavis_questions(key, truth))
    prompt, answer = qa_messages(picked)
    return prompt, answer, picked
