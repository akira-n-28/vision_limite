"""Verifica indipendente delle domande dei sintetici contro il disegno (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md
§§1, 2.3-2.4).

Non usa le funzioni del generatore che calcolano le risposte (synth._*_qa, synth._intersections, synth.visible_bound).
I fatti si ricavano dagli oggetti disegnati, come li registra synth.render(spec, layout=True): testi con i riquadri in
pixel, marcatori, spezzate, figure, intervalli in cui un grafico è disegnato. Le risposte si ricalcolano con predicati
geometrici scritti qui. Dalla specifica si prendono solo le formule dei grafici. Solo libreria standard.

Esito per domanda (`check_questions`):
- ok: la risposta è quella che si vede;
- errata: si vede un'altra risposta;
- non_ammissibile: il disegno non determina una sola risposta (vicino a un bordo, sotto un margine, più risposte vere);
- non_verificabile: manca un oggetto nel layout o la domanda non si riconosce.
Invarianti del renderer (`renderer_violations`):
- testi dentro l'immagine e alti almeno MIN_TEXT_PX;
- testi non sovrapposti fra loro né ai marcatori di altri oggetti;
- figure delle shapes non sovrapposte;
- valori scritti coerenti con il disegno.
"""
from __future__ import annotations

import math
import re

EDGE = 0.4         # margine dal bordo del tratto disegnato per le intersezioni (S2 §2.1)
CLOSE_ROOTS = 0.6  # due intersezioni più vicine di così (in x) non si distinguono a vista
TANGENT = 0.25     # grafici più vicini di così senza incrociarsi sembrano toccarsi (v4 non chiede sotto synth.NEAR_MISS)
SIDE = 0.5         # distanza minima da un asse, o fra due punti, per dire da che parte sta (unità del disegno)
ON_LINE, OFF_LINE = 0.05, 0.5  # un punto sta su una retta o su un lato sotto ON_LINE, ne è fuori oltre OFF_LINE
ANGLE_TOL = 0.6    # gradi: valore scritto contro angolo disegnato, angolo retto
LENGTH_TOL = 0.02  # relativa: lunghezza scritta contro lunghezza disegnata
EXTREME = 0.15     # il vertice estremo deve staccare il secondo di questa frazione del lato del riquadro
MIN_TEXT_PX = 10.0
OVERLAP_PX = 1.0   # due riquadri si sovrappongono se l'intersezione supera questo in entrambe le direzioni

TRIANGLE_VOCAB = ("right triangle", "isosceles triangle", "equilateral triangle", "scalene triangle", "acute triangle",
                  "obtuse triangle")
QUAD_VOCAB = ("square", "rectangle", "rhombus", "parallelogram", "trapezoid", "kite")
SHAPE_VOCAB = ("circle", "triangle", "square", "rectangle")
# Implicazioni fra le classi: una classe è più specifica di quelle che implica.
IMPLIES = {"square": {"rectangle", "rhombus", "parallelogram", "trapezoid", "kite"},
           "rectangle": {"parallelogram", "trapezoid"}, "rhombus": {"parallelogram", "trapezoid", "kite"},
           "parallelogram": {"trapezoid"}, "equilateral triangle": {"isosceles triangle", "acute triangle"}}
NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


# ---------------------------------------------------------------- geometria

def _dist_segment(p, a, b) -> float:
    (px, py), (ax_, ay), (bx, by) = p, a, b
    dx, dy = bx - ax_, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax_) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax_ - t * dx, py - ay - t * dy)


def _dist_line(p, a, b) -> float:
    (px, py), (ax_, ay), (bx, by) = p, a, b
    return abs((bx - ax_) * (py - ay) - (by - ay) * (px - ax_)) / math.hypot(bx - ax_, by - ay)


def _angle(a, v, b) -> float:
    """Angolo in v, in gradi."""
    u, w = (a[0] - v[0], a[1] - v[1]), (b[0] - v[0], b[1] - v[1])
    c = (u[0] * w[0] + u[1] * w[1]) / (math.hypot(*u) * math.hypot(*w))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _close(a: float, b: float, rel: float = 1e-6) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


def triangle_classes(pts) -> set[str]:
    """Classi vere di un triangolo, con le definizioni inclusive (un equilatero è anche isoscele)."""
    sides = [math.dist(pts[i], pts[(i + 1) % 3]) for i in range(3)]
    angles = [_angle(pts[i - 1], pts[i], pts[(i + 1) % 3]) for i in range(3)]
    iso = any(_close(sides[i], sides[j]) for i in range(3) for j in range(i + 1, 3))
    out = {"isosceles triangle"} if iso else {"scalene triangle"}
    if all(_close(sides[0], s) for s in sides):
        out.add("equilateral triangle")
    top = max(angles)
    out.add("right triangle" if abs(top - 90) <= 0.5 else "acute triangle" if top < 90 else "obtuse triangle")
    return out


def quad_classes(pts) -> set[str]:
    """Classi vere di un quadrilatero convesso, inclusive (un quadrato è anche rettangolo, rombo, aquilone…)."""
    sides = [math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    angles = [_angle(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]

    def parallel(i: int) -> bool:  # lato i parallelo al lato i + 2
        (a, b), (c, d) = (pts[i], pts[(i + 1) % 4]), (pts[(i + 2) % 4], pts[(i + 3) % 4])
        u, w = (b[0] - a[0], b[1] - a[1]), (d[0] - c[0], d[1] - c[1])
        return abs(u[0] * w[1] - u[1] * w[0]) <= 1e-6 * math.hypot(*u) * math.hypot(*w)
    out = set()
    if all(abs(a - 90) <= 0.5 for a in angles):
        out.add("rectangle")
    if all(_close(sides[0], s) for s in sides):
        out.add("rhombus")
    if "rectangle" in out and "rhombus" in out:
        out.add("square")
    pairs = parallel(0) + parallel(1)
    if pairs == 2:
        out.add("parallelogram")
    if pairs >= 1:
        out.add("trapezoid")
    if (_close(sides[0], sides[1]) and _close(sides[2], sides[3])) or (_close(sides[1], sides[2]) and _close(sides[3], sides[0])):
        out.add("kite")
    return out


def _options(question: str) -> list[str]:
    """Opzioni scritte nel testo, dopo i due punti: "…: a, b or c?"."""
    if ":" not in question:
        return []
    tail = question.rsplit(":", 1)[1].strip().rstrip("?")
    return [o.strip() for o in re.split(r",\s*|\s+or\s+", tail) if o.strip()]


def _classify(question: str, answer: str, true: set[str], vocab: tuple[str, ...]) -> tuple[str, str | None, str]:
    """(esito, risposta attesa, motivo) per una domanda di classificazione: aperta, con le opzioni o con la classe più
    specifica fra le opzioni."""
    options = _options(question)
    if not options:
        valid = true & set(vocab)
        if len(valid) != 1:
            return "non_ammissibile", None, f"domanda aperta con {len(valid)} risposte vere: {sorted(valid)}"
        return _status(answer, next(iter(valid))), next(iter(valid)), ""
    hit = [o for o in options if o in true]
    if "most specific" in question:
        minimal = [o for o in hit if all(other == o or other in IMPLIES.get(o, ()) for other in hit)]
        if len(minimal) != 1:
            return "non_ammissibile", None, f"opzioni vere {hit}: nessuna più specifica di tutte"
        return _status(answer, minimal[0]), minimal[0], ""
    if len(hit) != 1:
        return "non_ammissibile", None, f"{len(hit)} opzioni vere: {hit}"
    return _status(answer, hit[0]), hit[0], ""


def _status(answer: str, expected: str) -> str:
    return "ok" if answer == expected else "errata"


# ---------------------------------------------------------------- layout

def _to_data(layout: dict, px: tuple[float, float]) -> tuple[float, float]:
    t = layout["transform"]
    return (px[0] - t["x0"]) / t["ux"], (px[1] - t["y0"]) / t["uy"]


def _center(bbox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _by_prefix(layout: dict, prefix: str) -> list[dict]:
    return [it for it in layout["items"] if it["gid"].startswith(prefix)]


def _label_texts(layout: dict) -> set[str]:
    return {it["text"] for it in layout["items"] if it["kind"] == "text"
            and it["gid"].split(":")[0] in ("label", "vertex_label", "curve_label")}


def _exists(q: dict, layout: dict) -> tuple[str, str | None, str]:
    m = re.search(r"labeled (\w+)\?$", q["q"])
    if not m:
        return "non_verificabile", None, "etichetta non riconosciuta"
    expected = "yes" if m.group(1) in _label_texts(layout) else "no"
    return _status(q["answer"], expected), expected, ""


# ---------------------------------------------------------------- famiglie

def _coord(q: dict, layout: dict) -> tuple[str, str | None, str]:
    pts = {it["gid"].split(":", 1)[1]: it["xy"] for it in _by_prefix(layout, "point:")}
    t, text = q["type"], q["q"]
    names = re.findall(r"point (\w+)", text)
    if t == "exists":
        return _exists(q, layout)
    if t == "count":
        return _status(q["answer"], str(len(pts))), str(len(pts)), ""
    if any(n not in pts for n in names) or not names:
        return "non_verificabile", None, f"punti {names} non disegnati"
    x, y = pts[names[0]]
    if t in ("quadrant", "side_x", "side_y"):
        need = {"quadrant": min(abs(x), abs(y)), "side_x": abs(y), "side_y": abs(x)}[t]
        if need < SIDE:
            return "non_ammissibile", None, f"punto a {need:.2f} da un asse"
        expected = {"quadrant": {(True, True): "first", (False, True): "second", (False, False): "third",
                                 (True, False): "fourth"}[(x > 0, y > 0)],
                    "side_x": "above" if y > 0 else "below", "side_y": "left" if x < 0 else "right"}[t]
        return _status(q["answer"], expected), expected, ""
    if t == "coords":
        ticks = {(it["gid"].split(":")[1], it["text"]) for it in _by_prefix(layout, "tick_label:")}
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return "non_ammissibile", None, "coordinate non intere"
        if ("x", str(round(x))) not in ticks or ("y", str(round(y))) not in ticks:
            return "non_ammissibile", None, "manca l'etichetta di tacca per leggere le coordinate"
        expected = f"({round(x)}, {round(y)})"
        return _status(q["answer"], expected), expected, ""
    if t in ("pair_lr", "pair_ud"):
        x2, y2 = pts[names[1]]
        d = x - x2 if t == "pair_lr" else y - y2
        if abs(d) < SIDE:
            return "non_ammissibile", None, f"scarto {d:.2f}"
        expected = ("left" if d < 0 else "right") if t == "pair_lr" else ("above" if d > 0 else "below")
        return _status(q["answer"], expected), expected, ""
    if t == "line_on":
        lines = _by_prefix(layout, "line")
        m = re.search(r"through (\w+) and (\w+) pass through point (\w+)", text)
        if not lines or not m or any(n not in pts for n in m.groups()):
            return "non_verificabile", None, "retta o punti mancanti"
        a, b = lines[0]["path"][0], lines[0]["path"][-1]
        if max(_dist_line(pts[m.group(1)], a, b), _dist_line(pts[m.group(2)], a, b)) > ON_LINE:
            return "errata", None, "la retta disegnata non passa per i due punti nominati"
        d = _dist_line(pts[m.group(3)], a, b)
        if ON_LINE < d < OFF_LINE:
            return "non_ammissibile", None, f"punto a {d:.2f} dalla retta"
        expected = "yes" if d <= ON_LINE else "no"
        return _status(q["answer"], expected), expected, ""
    return "non_verificabile", None, f"tipo {t} non previsto"


def _polygon_vertices(layout: dict):
    poly = _by_prefix(layout, "polygon")
    if not poly:
        return None, None
    path = [tuple(p) for p in poly[0]["path"]]
    if len(path) > 1 and path[0] == path[-1]:
        path = path[:-1]
    names = {}
    for it in _by_prefix(layout, "vertex_label:"):  # ogni etichetta al vertice più vicino
        c = _to_data(layout, _center(it["bbox"]))
        names[it["text"]] = min(range(len(path)), key=lambda i: math.dist(c, path[i]))
    if sorted(names.values()) != list(range(len(path))):
        return path, None
    return path, names


def _polygon(q: dict, layout: dict) -> tuple[str, str | None, str]:
    t, text = q["type"], q["q"]
    if t == "exists":
        return _exists(q, layout)
    path, names = _polygon_vertices(layout)
    if path is None or names is None:
        return "non_verificabile", None, "poligono o etichette dei vertici non riconosciuti"
    label = {i: n for n, i in names.items()}
    n = len(path)
    sides = [(path[i], path[(i + 1) % n], {label[i], label[(i + 1) % n]}) for i in range(n)]
    numbers = [it for it in layout["items"] if it["kind"] == "text" and NUMBER.match(it["text"])]
    if t == "count":
        return _status(q["answer"], str(len(numbers))), str(len(numbers)), ""
    if t == "side_len":
        m = re.search(r"side (\w)(\w)\?", text)
        want = {m.group(1), m.group(2)} if m else None
        nearest = [(min(range(n), key=lambda i: _dist_segment(_to_data(layout, _center(it["bbox"])), sides[i][0], sides[i][1])), it)
                   for it in numbers]
        found = [it["text"] for i, it in nearest if sides[i][2] == want]
        if len(found) != 1:
            return "non_ammissibile", None, f"{len(found)} misure vicine al lato"
        return _status(q["answer"], found[0]), found[0], ""
    if t == "angle_val":
        arcs, texts = _by_prefix(layout, "angle_arc"), _by_prefix(layout, "angle_label")
        m = re.search(r"vertex (\w+)\?", text)
        if not arcs or not texts or not m or m.group(1) not in names:
            return "non_verificabile", None, "arco o valore dell'angolo mancanti"
        at = min(range(n), key=lambda i: math.dist(arcs[0]["center"], path[i]))
        if at != names[m.group(1)]:
            return "errata", None, f"l'arco è al vertice {label[at]}"
        return _status(q["answer"], texts[0]["text"]), texts[0]["text"], ""
    if t == "right_vertex":
        marks = _by_prefix(layout, "right_marker")
        if not marks:
            return "non_verificabile", None, "segno dell'angolo retto mancante"
        corner = marks[0]["path"][len(marks[0]["path"]) // 2]
        at = min(range(n), key=lambda i: math.dist(corner, path[i]))
        if abs(_angle(path[at - 1], path[at], path[(at + 1) % n]) - 90) > 0.5:
            return "errata", None, "il segno dell'angolo retto è su un angolo non retto"
        return _status(q["answer"], label[at]), label[at], ""
    if t in ("highest", "lowest", "leftmost", "rightmost"):
        k, sign = {"highest": (1, 1), "lowest": (1, -1), "leftmost": (0, -1), "rightmost": (0, 1)}[t]
        order = sorted(range(n), key=lambda i: sign * path[i][k], reverse=True)
        size = max(max(p[0] for p in path) - min(p[0] for p in path), max(p[1] for p in path) - min(p[1] for p in path))
        if sign * (path[order[0]][k] - path[order[1]][k]) < EXTREME * size:
            return "non_ammissibile", None, "due vertici quasi alla stessa altezza o ascissa"
        return _status(q["answer"], label[order[0]]), label[order[0]], ""
    if t == "m_side":
        m = re.search(r"point (\w+) lie", text)
        pts = {it["gid"].split(":", 1)[1]: it["xy"] for it in _by_prefix(layout, "point:")}
        if not m or m.group(1) not in pts:
            return "non_verificabile", None, "punto mancante"
        on = [s for s in sides if _dist_segment(pts[m.group(1)], s[0], s[1]) <= ON_LINE]
        if len(on) != 1:
            return "non_ammissibile", None, f"il punto sta su {len(on)} lati"
        expected = "".join(sorted(on[0][2]))
        return ("ok" if set(q["answer"]) == on[0][2] else "errata"), expected, ""
    if t == "shape_type":
        true = triangle_classes(path) if n == 3 else quad_classes(path)
        return _classify(text, q["answer"], true, TRIANGLE_VOCAB if n == 3 else QUAD_VOCAB)
    return "non_verificabile", None, f"tipo {t} non previsto"


def _curve_value(c: dict, x: float) -> float:
    return c["a"] * x + c["b"] if c["kind"] == "line" else c["a"] * (x - c["h"]) ** 2 + c["k"]


def _visible_at(item: dict, x: float) -> bool:
    return any(a <= x <= b for a, b in item["visible"])


def _difference(f: dict, g: dict) -> tuple[float, float, float]:
    """Coefficienti di f - g, ricavati qui dalle formule dei grafici."""
    def poly(c):
        return (0.0, c["a"], c["b"]) if c["kind"] == "line" else (c["a"], -2 * c["a"] * c["h"], c["a"] * c["h"] ** 2 + c["k"])
    (p2, p1, p0), (q2, q1, q0) = poly(f), poly(g)
    return p2 - q2, p1 - q1, p0 - q0


def visible_intersections(f: dict, g: dict, items: dict) -> tuple[int | None, list[str]]:
    """(conteggio geometrico delle intersezioni disegnate, motivi di non ammissibilità della domanda).

    Un'intersezione è disegnata se cade dentro il dominio x del tratto disegnato e sotto il livello |y| oltre cui il
    renderer toglie i grafici (`x_range`, `clip_y` del layout), cioè dove entrambi i grafici sono disegnati. Il conteggio
    è None se f - g è costante (grafici paralleli o coincidenti). Motivi di non ammissibilità: intersezione a meno di
    EDGE dal bordo del tratto disegnato, tangenza, due intersezioni a meno di CLOSE_ROOTS, grafici che si avvicinano
    sotto TANGENT senza incrociarsi."""
    a, b, c = _difference(f, g)
    fi, gi = items[f["name"]], items[g["name"]]
    lo, hi = max(fi["x_range"][0], gi["x_range"][0]), min(fi["x_range"][1], gi["x_range"][1])
    clips = [v for v in (fi.get("clip_y"), gi.get("clip_y")) if v is not None]
    if not clips:
        return None, ["livello di taglio dei grafici non registrato nel layout"]
    clip = min(clips)
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None, ["grafici paralleli o coincidenti"]
        xs, tangent = [-c / b], False
    else:
        d = b * b - 4 * a * c
        xs = [] if d < 0 else sorted({(-b - math.sqrt(d)) / (2 * a), (-b + math.sqrt(d)) / (2 * a)})
        tangent = d == 0
    why, count = [], 0
    for x in xs:
        y = _curve_value(f, x)
        count += lo <= x <= hi and abs(y) <= clip
        edge = min(abs(x - lo), abs(x - hi), abs(abs(y) - clip))
        if edge < EDGE:
            why.append(f"intersezione ({x:.3f}, {y:.3f}) a {edge:.3f} dal bordo del tratto disegnato")
    if tangent:
        why.append("tangenza")
    if len(xs) == 2 and xs[1] - xs[0] < CLOSE_ROOTS:
        why.append(f"due intersezioni a {xs[1] - xs[0]:.3f} in x")
    if not xs:  # nessun incrocio: nel tratto disegnato i grafici non devono sembrare toccarsi
        grid = [lo + (hi - lo) * i / 2000 for i in range(2001)]
        gaps = [abs(_curve_value(f, x) - _curve_value(g, x)) for x in grid
                if abs(_curve_value(f, x)) <= clip and abs(_curve_value(g, x)) <= clip]
        if gaps and min(gaps) < TANGENT:
            why.append(f"grafici a {min(gaps):.3f} senza incrociarsi")
    return count, why


def _function(q: dict, layout: dict, spec: dict) -> tuple[str, str | None, str]:
    t, text = q["type"], q["q"]
    if t == "exists":
        return _exists(q, layout)
    items = {it["gid"].split(":", 1)[1]: it for it in _by_prefix(layout, "curve:")}
    curves = {c["name"]: c for c in spec["curves"]}  # dalla specifica solo le formule
    if any(name not in items for name in curves):
        return "non_verificabile", None, "grafico non disegnato"
    if t == "intersections":
        if len(curves) != 2:
            return "non_verificabile", None, "servono due grafici"
        count, why = visible_intersections(*curves.values(), items)
        if why:
            return "non_ammissibile", None if count is None else str(count), "; ".join(why)
        return _status(q["answer"], str(count)), str(count), ""
    m = re.search(r"graph of (\w+)", text)
    if not m or m.group(1) not in curves:
        return "non_verificabile", None, "grafico non riconosciuto"
    c, item = curves[m.group(1)], items[m.group(1)]
    if t == "intercept":
        if not _visible_at(item, 0.0):
            return "non_ammissibile", None, "il grafico non incontra l'asse y nel tratto disegnato"
        y0 = _curve_value(c, 0.0)
        if abs(y0) < SIDE:
            return "non_ammissibile", None, f"intercetta {y0:.2f} vicina all'origine"
        expected = "above" if y0 > 0 else "below"
        return _status(q["answer"], expected), expected, ""
    if t == "slope":
        a, b = max(item["visible"], key=lambda r: r[1] - r[0])
        expected = "increasing" if _curve_value(c, b) > _curve_value(c, a) else "decreasing"
        return _status(q["answer"], expected), expected, ""
    if t in ("opens", "vertex_ud", "vertex_lr"):
        if c["kind"] != "parabola":
            return "non_verificabile", None, "non è una parabola"
        h = c["h"]
        if not _visible_at(item, h):
            return "non_ammissibile", None, "vertice non disegnato"
        k = _curve_value(c, h)
        if t == "opens":
            expected = "upward" if _curve_value(c, h + 1) > k else "downward"
        elif t == "vertex_ud":
            if abs(k) < SIDE:
                return "non_ammissibile", None, "vertice sull'asse x"
            expected = "above" if k > 0 else "below"
        else:
            if abs(h) < SIDE:
                return "non_ammissibile", None, "vertice sull'asse y"
            expected = "left" if h < 0 else "right"
        return _status(q["answer"], expected), expected, ""
    return "non_verificabile", None, f"tipo {t} non previsto"


def _shapes(q: dict, layout: dict) -> tuple[str, str | None, str]:
    t, text = q["type"], q["q"]
    if t == "exists":
        return _exists(q, layout)
    shapes = []
    for it in _by_prefix(layout, "shape:"):
        if it["shape"] == "Circle":
            kind = "circle"
        elif it["shape"] == "Rectangle":
            kind = "square" if _close(it["width"], it["height"]) else "rectangle"
        elif it["shape"] == "Polygon" and len(it["path"]) - (it["path"][0] == it["path"][-1]) == 3:
            kind = "triangle"
        else:
            return "non_verificabile", None, f"figura {it['shape']} non prevista"
        shapes.append((kind, it["bbox"]))
    kinds = [k for k, _ in shapes]
    if t == "count":
        return _status(q["answer"], str(len(shapes))), str(len(shapes)), ""
    if t in ("has_type", "count_type"):
        m = re.search(r"(?:Is there an? |How many )(\w+?)s? (?:in|are)", text)
        if not m or m.group(1) not in SHAPE_VOCAB:
            return "non_verificabile", None, "tipo di figura non riconosciuto"
        want = m.group(1)
        if t == "has_type":
            present = want in kinds or (want == "rectangle" and "square" in kinds)  # un quadrato è un rettangolo
            expected = "yes" if present else "no"
            return _status(q["answer"], expected), expected, ""
        if want == "rectangle" and "square" in kinds:
            return "non_ammissibile", None, "contare i rettangoli con un quadrato presente è ambiguo"
        expected = str(kinds.count(want))
        return _status(q["answer"], expected), expected, ""
    if t == "label_type":
        m = re.search(r"labeled (\w+)", text)
        labels = [it for it in _by_prefix(layout, "label:") if m and it["text"] == m.group(1)]
        if not labels:
            return "non_verificabile", None, "etichetta non disegnata"
        cx, cy = _center(labels[0]["bbox"])
        inside = [k for k, b in shapes if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
        if len(inside) != 1:
            return "non_ammissibile", None, f"l'etichetta è dentro {len(inside)} figure"
        true = {inside[0]} | ({"rectangle"} if inside[0] == "square" else set())
        return _classify(text, q["answer"], true, SHAPE_VOCAB)
    return "non_verificabile", None, f"tipo {t} non previsto"


def check_questions(spec: dict, questions: list[dict], layout: dict) -> list[dict]:
    """Esito di ogni domanda {"type", "q", "answer"} della scena contro il suo layout."""
    out = []
    for q in questions:
        fam = spec["family"]
        if fam == "coord":
            status, expected, why = _coord(q, layout)
        elif fam == "polygon":
            status, expected, why = _polygon(q, layout)
        elif fam == "function":
            status, expected, why = _function(q, layout, spec)
        elif fam == "shapes":
            status, expected, why = _shapes(q, layout)
        else:
            status, expected, why = "non_verificabile", None, f"famiglia {fam}"
        out.append({"type": q["type"], "q": q["q"], "answer": q["answer"], "expected": expected, "status": status,
                    "why": why})
    return out


# ---------------------------------------------------------------- invarianti del renderer

def _overlap(a, b) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > OVERLAP_PX and min(a[3], b[3]) - max(a[1], b[1]) > OVERLAP_PX


def renderer_violations(spec: dict, layout: dict) -> list[dict]:
    """Violazioni delle invarianti del renderer (S2 §2.4), con i gid coinvolti."""
    W, H = layout["size"]
    texts = [it for it in layout["items"] if it["kind"] == "text"]
    markers = [it for it in layout["items"] if it["kind"] == "marker"]
    out = []
    for it in texts:
        x0, y0, x1, y1 = it["bbox"]
        if x0 < -0.5 or y0 < -0.5 or x1 > W + 0.5 or y1 > H + 0.5:
            out.append({"kind": "testo_fuori", "gids": [it["gid"]], "detail": it["bbox"]})
        if y1 - y0 < MIN_TEXT_PX:
            out.append({"kind": "testo_piccolo", "gids": [it["gid"]], "detail": round(y1 - y0, 1)})
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if _overlap(a["bbox"], b["bbox"]):
                out.append({"kind": "testi_sovrapposti", "gids": [a["gid"], b["gid"]], "detail": None})
        own = a["gid"].split(":", 1)[-1]
        for mk in markers:
            if mk["gid"].split(":", 1)[-1] != own and _overlap(a["bbox"], mk["bbox"]):
                out.append({"kind": "testo_su_marcatore", "gids": [a["gid"], mk["gid"]], "detail": None})
    shapes = _by_prefix(layout, "shape:")
    for i, a in enumerate(shapes):
        for b in shapes[i + 1:]:
            if _overlap(a["bbox"], b["bbox"]):
                out.append({"kind": "figure_sovrapposte", "gids": [a["gid"], b["gid"]], "detail": None})
    if spec["family"] == "polygon":
        path, names = _polygon_vertices(layout)
        if path is not None and names is None:
            out.append({"kind": "etichette_dei_vertici", "gids": ["polygon"], "detail": "non una per vertice"})
        elif path is not None:
            n = len(path)
            for it in texts:
                if it["gid"].startswith("length:") and NUMBER.match(it["text"]):
                    c = _to_data(layout, _center(it["bbox"]))
                    i = min(range(n), key=lambda j: _dist_segment(c, path[j], path[(j + 1) % n]))
                    drawn = math.dist(path[i], path[(i + 1) % n])
                    if abs(float(it["text"]) - drawn) > LENGTH_TOL * drawn + 0.05:
                        out.append({"kind": "lunghezza_incoerente", "gids": [it["gid"]],
                                    "detail": f"scritto {it['text']}, disegnato {drawn:.2f}"})
            for it in texts:
                if it["gid"] == "angle_label":
                    arc = _by_prefix(layout, "angle_arc")
                    if arc:
                        at = min(range(n), key=lambda j: math.dist(arc[0]["center"], path[j]))
                        drawn = _angle(path[at - 1], path[at], path[(at + 1) % n])
                        if abs(float(it["text"].rstrip("°")) - drawn) > ANGLE_TOL:
                            out.append({"kind": "angolo_incoerente", "gids": [it["gid"]],
                                        "detail": f"scritto {it['text']}, disegnato {drawn:.2f}"})
    return out


def check_sample(meta: dict, layout: dict) -> dict:
    """Esiti di un campione degli shard: domande (`qa`) e domanda della coppia controfattuale (`pair`)."""
    qs = list(meta.get("qa", []))
    if "pair" in meta:
        qs.append({k: meta["pair"][k] for k in ("type", "q", "answer")})
    return {"questions": check_questions(meta["spec"], qs, layout), "violations": renderer_violations(meta["spec"], layout)}
