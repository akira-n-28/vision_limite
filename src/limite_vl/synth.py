"""Diagrammi sintetici con fatti esatti per F1c: domande visive verificabili e coppie controfattuali.

Ogni scena è una specifica di soli dati (dict JSON) generata da un Random con il seme della chiave, poi disegnata con
matplotlib. Le domande vengono dalla specifica, quindi la risposta è vera per costruzione; le opzioni servono alla
valutazione a scelta forzata. Famiglie:
- coord: piano cartesiano con punti etichettati (quadrante, sopra/sotto l'asse x, sinistra/destra dell'asse y,
  posizione relativa di due punti, conteggio, coordinate se ci sono le tacche, appartenenza a una retta);
- polygon: triangoli e quadrilateri con lati, un angolo e un angolo retto segnati (lunghezze e angoli scritti,
  vertice più alto o più a sinistra, lato su cui sta un punto, tipo di figura, numero di lati con la misura);
- function: grafici di rette e parabole (verso, crescenza, segno dell'intercetta, posizione del vertice, numero di
  intersezioni fra due grafici);
- shapes: da 1 a 4 figure separate (conteggio, presenza di un tipo, figura con una data etichetta).
Le domande "Is there a point/graph/shape labeled X?" con risposta No allenano a non nominare cose che non ci sono.
`counterfactual` cambia un solo dettaglio decisivo (un punto riflesso, il verso di una parabola, un valore scritto,
una figura in più) e restituisce la domanda la cui risposta cambia.

Versioni del generatore (`version`; campo `generator_version` della specifica, assente in v3):
- 3: le domande degli shard data/shards/synth/v3 (job 29756), riprodotte esattamente con version=3;
- 4 (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §2): le intersezioni si contano nel dominio disegnato
  (`visible_bound`), le domande di classificazione mostrano le opzioni o chiedono la classe più specifica, le shapes
  non hanno figure sovrapposte. Le scene e le domande degli altri tipi restano quelle di v3.
`render(spec, layout=True)` restituisce anche gli oggetti disegnati, per le verifiche di limite_vl.synth_check.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from copy import deepcopy

FAMILIES = ("coord", "polygon", "function", "shapes")
LETTERS = "ABCDEFGHKLMNPQRSTUVW"  # senza I, J, O (origine), X, Y, Z
YES_NO = ["yes", "no"]
VERSIONS = (3, 4)
GENERATOR_VERSION = 4
SHAPE_GAP = 0.2  # da v4: distanza minima fra i riquadri di due figure delle shapes, in unità del disegno


def spec_version(spec: dict) -> int:
    """Versione del generatore di una specifica: quelle degli shard v3 non hanno il campo."""
    version = spec.get("generator_version", 3)
    if version not in VERSIONS:
        raise ValueError(f"versione del generatore {version!r}: attesa una di {VERSIONS}")
    return version


def rng_for(key: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def _qa(qtype: str, q: str, options: list[str], answer: str) -> dict:
    if answer not in options:
        raise ValueError(f"{q}: risposta {answer!r} fuori dalle opzioni {options}")
    return {"type": qtype, "q": q, "options": options, "answer": answer}


def _style(rng: random.Random) -> dict:
    return {"lw": round(rng.uniform(1.0, 2.2), 2), "font": rng.randint(12, 16),
            "font_family": rng.choice(["DejaVu Sans", "DejaVu Serif"]),
            "width": 32 * rng.randint(14, 20), "height": 32 * rng.randint(14, 20)}  # griglie di patch pari (pixel shuffle 2x2)


def _absent_label(rng: random.Random, used: set[str]) -> str:
    return rng.choice([c for c in LETTERS if c not in used])


# ---------------------------------------------------------------- coord

def _coord_spec(rng: random.Random) -> dict:
    R = rng.choice([5, 6, 8])
    ticks = rng.random() < 0.6
    step = 1 if R <= 6 else 2
    labels = rng.sample(LETTERS, 6)
    points: list[dict] = []
    n = rng.randint(2, 5)
    tries = 0
    while len(points) < n and tries < 500:
        tries += 1
        x = step * rng.randint(-R // step, R // step)
        y = step * rng.randint(-R // step, R // step)
        if abs(x) < 2 or abs(y) < 2 or any(max(abs(x - p["x"]), abs(y - p["y"])) < 2 for p in points):  # lontano dagli assi
            continue
        points.append({"label": labels[len(points)], "x": x, "y": y})
    spec = {"family": "coord", "R": R, "ticks": ticks, "step": step, "points": points, "line": None, **_style(rng)}
    if len(points) >= 3 and rng.random() < 0.4:
        a, b, c = points[0], points[1], points[2]
        if a["x"] != b["x"]:
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            g = math.gcd(dx, dy)
            on = [(a["x"] + k * dx // g, a["y"] + k * dy // g) for k in range(-12, 13)]
            on = [(x, y) for x, y in on if -R <= x <= R and -R <= y <= R and abs(x) >= 2 and abs(y) >= 2
                  and all(max(abs(x - p["x"]), abs(y - p["y"])) >= 2 for p in points if p is not c)]
            if on and rng.random() < 0.5:
                c["x"], c["y"] = rng.choice(on)
                spec["line"] = {"through": [a["label"], b["label"]], "test": c["label"], "on": True}
            else:
                norm = math.hypot(dx, dy)
                dist = abs(dy * (c["x"] - a["x"]) - dx * (c["y"] - a["y"])) / norm
                if dist >= 1.0:
                    spec["line"] = {"through": [a["label"], b["label"]], "test": c["label"], "on": False}
    return spec


def _quadrant(x: float, y: float) -> str:
    return {(True, True): "first", (False, True): "second", (False, False): "third", (True, False): "fourth"}[(x > 0, y > 0)]


def _coord_qa(spec: dict, rng: random.Random, version: int) -> list[dict]:
    pts = spec["points"]
    out = []
    for p in pts:
        L = p["label"]
        out.append(_qa("quadrant", f"In which quadrant is point {L}?", ["first", "second", "third", "fourth"],
                       _quadrant(p["x"], p["y"])))
        out.append(_qa("side_x", f"Is point {L} above or below the x-axis?", ["above", "below"],
                       "above" if p["y"] > 0 else "below"))
        out.append(_qa("side_y", f"Is point {L} to the left or to the right of the y-axis?", ["left", "right"],
                       "left" if p["x"] < 0 else "right"))
        if spec["ticks"]:
            cands = [(p["x"], p["y"]), (-p["x"], p["y"]), (p["x"], -p["y"]), (p["y"], p["x"]),
                     (p["x"] + spec["step"], p["y"]), (p["x"], p["y"] - spec["step"])]
            opts = list(dict.fromkeys(f"({x}, {y})" for x, y in cands))[:4]
            rng.shuffle(opts)
            out.append(_qa("coords", f"What are the coordinates of point {L}?", opts, f"({p['x']}, {p['y']})"))
    for a in pts:
        for b in pts:
            if a is b:
                continue
            if abs(a["x"] - b["x"]) >= 2:
                out.append(_qa("pair_lr", f"Is point {a['label']} to the left or to the right of point {b['label']}?",
                               ["left", "right"], "left" if a["x"] < b["x"] else "right"))
            if abs(a["y"] - b["y"]) >= 2:
                out.append(_qa("pair_ud", f"Is point {a['label']} above or below point {b['label']}?",
                               ["above", "below"], "above" if a["y"] > b["y"] else "below"))
    out.append(_qa("count", "How many labeled points are marked in the figure, not counting the origin O?",
                   [str(k) for k in range(1, 7)], str(len(pts))))
    used = {p["label"] for p in pts}
    out.append(_qa("exists", f"Is there a point labeled {rng.choice(sorted(used))}?", YES_NO, "yes"))
    out.append(_qa("exists", f"Is there a point labeled {_absent_label(rng, used)}?", YES_NO, "no"))
    if spec["line"]:
        a, b = spec["line"]["through"]
        out.append(_qa("line_on", f"Does the line through {a} and {b} pass through point {spec['line']['test']}?",
                       YES_NO, "yes" if spec["line"]["on"] else "no"))
    return out


def _draw_axes(ax, R: float, spec: dict, ticks: bool, step: int, labels: bool) -> None:
    # gid: solo per riconoscere gli oggetti in render(..., layout=True); non cambia i pixel
    lw, fs = spec["lw"], spec["font"]
    ax.annotate("", xy=(R + 1, 0), xytext=(-R - 1, 0), arrowprops={"arrowstyle": "->", "lw": lw * 0.8},
                annotation_clip=False, gid="axis:x")
    ax.annotate("", xy=(0, R + 1), xytext=(0, -R - 1), arrowprops={"arrowstyle": "->", "lw": lw * 0.8},
                annotation_clip=False, gid="axis:y")
    ax.text(R + 0.9, -0.7, "x", fontsize=fs, family=spec["font_family"], style="italic", gid="axis_label:x")
    ax.text(0.3, R + 0.7, "y", fontsize=fs, family=spec["font_family"], style="italic", gid="axis_label:y")
    ax.text(-0.15, -0.15, "O", fontsize=fs, family=spec["font_family"], ha="right", va="top", gid="axis_label:O")
    if ticks:
        for t in range(-R, R + 1):
            if t == 0:
                continue
            ax.plot([t, t], [-0.12, 0.12], color="black", lw=lw * 0.6, gid="tick")
            ax.plot([-0.12, 0.12], [t, t], color="black", lw=lw * 0.6, gid="tick")
            if labels and t % step == 0:
                ax.text(t, -0.55, str(t), fontsize=fs - 4, ha="center", va="top", family=spec["font_family"],
                        gid=f"tick_label:x:{t}")
                ax.text(-0.3, t, str(t), fontsize=fs - 4, ha="right", va="center", family=spec["font_family"],
                        gid=f"tick_label:y:{t}")


def _draw_coord(ax, spec: dict) -> None:
    R = spec["R"]
    _draw_axes(ax, R, spec, spec["ticks"], spec["step"], labels=True)
    by = {p["label"]: p for p in spec["points"]}
    if spec["line"]:
        a, b = (by[k] for k in spec["line"]["through"])
        slope = (b["y"] - a["y"]) / (b["x"] - a["x"])
        xs = [-R - 1, R + 1]
        ys = [a["y"] + slope * (x - a["x"]) for x in xs]
        ax.plot(xs, ys, color="black", lw=spec["lw"], gid="line")
    for p in spec["points"]:
        ax.plot(p["x"], p["y"], "o", color="black", ms=4 + spec["lw"], gid=f"point:{p['label']}")
        dx = 0.3 if p["x"] < R - 0.5 else -0.8
        ax.text(p["x"] + dx, p["y"] + 0.3, p["label"], fontsize=spec["font"], family=spec["font_family"],
                gid=f"label:{p['label']}")
    ax.set_xlim(-R - 1.5, R + 1.5)
    ax.set_ylim(-R - 1.5, R + 1.5)


# ---------------------------------------------------------------- polygon

TRIPLES = [(3, 4, 5), (5, 12, 13), (6, 8, 10), (8, 15, 17), (7, 24, 25), (9, 12, 15)]
POLY_TYPES = ("right triangle", "isosceles triangle", "scalene triangle", "square", "rectangle", "parallelogram",
              "trapezoid")


def _fmt_len(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def _rotate(pts: list[tuple[float, float]], theta: float) -> list[tuple[float, float]]:
    c, s = math.cos(theta), math.sin(theta)
    return [(x * c - y * s, x * s + y * c) for x, y in pts]


def _angle_at(pts, i) -> float:
    (ax_, ay), (bx, by), (cx, cy) = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
    v1, v2 = (ax_ - bx, ay - by), (cx - bx, cy - by)
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _polygon_spec(rng: random.Random) -> dict:
    kind = rng.choice(POLY_TYPES)
    if kind == "right triangle":
        a, b, _ = rng.choice(TRIPLES)
        k = rng.choice([1, 1, 2])
        pts, right = [(0.0, 0.0), (a * k, 0.0), (0.0, b * k)], 0
    elif kind == "isosceles triangle":
        c = rng.randint(3, 7)
        # h = c darebbe un isoscele rettangolo; lati obliqui entro il 12% della base, quasi equilatero (audit di F1c)
        h = rng.choice([v for v in range(5, 13) if v != c and abs(math.hypot(c, v) - 2 * c) > 0.12 * 2 * c])
        pts, right = [(-c, 0.0), (c, 0.0), (0.0, h)], None
    elif kind == "scalene triangle":
        while True:
            pts = [(0.0, 0.0), (rng.uniform(6, 12), 0.0), (rng.uniform(-2, 10), rng.uniform(4, 10))]
            angles = [_angle_at(pts, i) for i in range(3)]
            sides = sorted(math.dist(pts[i], pts[(i + 1) % 3]) for i in range(3))
            if (min(angles) > 25 and max(angles) < 105 and all(abs(a - 90) > 10 for a in angles)
                    and all(sides[i + 1] / sides[i] > 1.12 for i in range(2))):
                break
        right = None
    elif kind == "square":
        s = rng.randint(3, 9)
        pts, right = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)], None
    elif kind == "rectangle":
        w = rng.randint(4, 12)
        h = rng.choice([v for v in range(2, 10) if max(w, v) / min(w, v) >= 1.3])
        pts, right = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)], None
    elif kind == "parallelogram":
        b, h = rng.randint(5, 10), rng.randint(3, 7)
        d = rng.choice([-1, 1]) * rng.uniform(0.4, 0.8) * h
        pts, right = [(0.0, 0.0), (b, 0.0), (b + d, h), (d, h)], None
    else:  # trapezoid
        b1 = rng.randint(7, 12)
        b2 = rng.randint(3, int(b1 * 0.7))
        h = rng.randint(3, 7)
        off = rng.uniform(0, b1 - b2)
        pts, right = [(0.0, 0.0), (b1, 0.0), (off + b2, h), (off, h)], None
    theta = math.radians(rng.uniform(-25, 25)) if rng.random() < 0.7 else 0.0
    pts = _rotate(pts, theta)
    if rng.random() < 0.5:  # specchio orizzontale (ordine invertito per restare antiorario): stessi fatti metrici
        pts = [(-x, y) for x, y in pts][::-1]
        right = None if right is None else len(pts) - 1 - right
    lengths = [math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    start = rng.randint(0, len(LETTERS) - len(pts))
    labels = list(LETTERS[start:start + len(pts)])
    n = len(pts)
    sides = [labels[i] + labels[(i + 1) % n] for i in range(n)]
    shown = sorted(rng.sample(range(n), rng.randint(1, min(3, n))))
    spec = {"family": "polygon", "kind": kind, "points": [list(p) for p in pts], "labels": labels, "sides": sides,
            "lengths": [_fmt_len(v) for v in lengths], "shown": shown, "right": right, "angle": None, "m": None,
            **_style(rng)}
    if n == 3 and right is None and rng.random() < 0.7:
        i = rng.randrange(3)
        spec["angle"] = {"vertex": i, "value": int(round(_angle_at(pts, i)))}
    if rng.random() < 0.5:
        side = rng.randrange(n)
        spec["m"] = {"label": _absent_label(rng, set(labels)), "side": side, "t": rng.choice([0.5, rng.uniform(0.3, 0.7)])}
    return spec


def _poly_geometry(spec: dict):
    pts = [tuple(p) for p in spec["points"]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    size = max(max(xs) - min(xs), max(ys) - min(ys))
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    return pts, size, (cx, cy)


TRIANGLE_TYPES = ["right triangle", "isosceles triangle", "equilateral triangle"]
QUAD_TYPES = ["square", "rectangle", "parallelogram", "trapezoid"]  # catena: ognuna è un caso della successiva


def _options_text(options: list[str]) -> str:
    return ", ".join(options[:-1]) + " or " + options[-1]


def _polygon_qa(spec: dict, rng: random.Random, version: int) -> list[dict]:
    pts, size, _ = _poly_geometry(spec)
    labels, sides, n = spec["labels"], spec["sides"], len(spec["labels"])
    out = []
    shown_vals = [spec["lengths"][i] for i in spec["shown"]]
    for i in spec["shown"]:
        true = spec["lengths"][i]
        v = float(true)
        cands = [true] + [s for s in shown_vals if s != true] + [_fmt_len(v + 1), _fmt_len(v * 2), _fmt_len(max(1.0, v - 2)),
                                                                _fmt_len(v + 3)]
        opts = list(dict.fromkeys(cands))[:4]
        rng.shuffle(opts)
        out.append(_qa("side_len", f"What is the length of side {sides[i]}?", opts, true))
    if spec["angle"]:
        d = spec["angle"]["value"]
        opts = list(dict.fromkeys(f"{x}°" for x in (d, d + 10, d - 10, 180 - d, d + 20) if 0 < x < 180))[:4]
        rng.shuffle(opts)
        out.append(_qa("angle_val", f"What is the measure of the marked angle at vertex {labels[spec['angle']['vertex']]}?",
                       opts, f"{d}°"))
    if spec["right"] is not None:
        out.append(_qa("right_vertex", "At which vertex is the right angle?", list(labels), labels[spec["right"]]))
    for qtype, q, key, sign in (("highest", "Which vertex is the highest?", 1, 1),
                                ("lowest", "Which vertex is the lowest?", 1, -1),
                                ("leftmost", "Which vertex is farthest to the left?", 0, -1),
                                ("rightmost", "Which vertex is farthest to the right?", 0, 1)):
        order = sorted(range(n), key=lambda i: sign * pts[i][key], reverse=True)
        if sign * (pts[order[0]][key] - pts[order[1]][key]) >= 0.15 * size:
            out.append(_qa(qtype, q, list(labels), labels[order[0]]))
    if spec["m"]:
        out.append(_qa("m_side", f"On which side does point {spec['m']['label']} lie?", list(sides), sides[spec["m"]["side"]]))
    name = "".join(labels)
    # v3: domande aperte, con più risposte vere (un triangolo rettangolo è anche scaleno, un quadrato è anche un
    # rettangolo). v4: le opzioni nel testo; per i quadrilateri la classe più specifica della catena (S2 §2.2).
    if n == 3:  # un rettangolo dei nostri è anche scaleno: lo scaleno non ha una risposta unica, niente domanda
        if spec["kind"] != "scalene triangle":
            q = (f"What type of triangle is {name}?" if version < 4 else
                 f"Which of these describes triangle {name}: {_options_text(TRIANGLE_TYPES)}?")
            out.append(_qa("shape_type", q, list(TRIANGLE_TYPES), spec["kind"]))
    else:  # il nome più specifico, come nelle didascalie di MAVIS ("ABCD is a square")
        q = (f"What type of quadrilateral is {name}?" if version < 4 else
             f"What is the most specific type of quadrilateral {name}: {_options_text(QUAD_TYPES)}?")
        out.append(_qa("shape_type", q, list(QUAD_TYPES), spec["kind"]))
    out.append(_qa("count", "How many sides have their length written in the figure?",
                   [str(k) for k in range(1, n + 1)], str(len(spec["shown"]))))
    used = set(labels) | ({spec["m"]["label"]} if spec["m"] else set())
    out.append(_qa("exists", f"Is there a point labeled {rng.choice(sorted(used))}?", YES_NO, "yes"))
    out.append(_qa("exists", f"Is there a point labeled {_absent_label(rng, used)}?", YES_NO, "no"))
    return out


def _draw_polygon(ax, spec: dict) -> None:
    import matplotlib.patches as mpatches

    pts, size, (cx, cy) = _poly_geometry(spec)
    n, lw, fs, fam = len(pts), spec["lw"], spec["font"], spec["font_family"]
    ax.add_patch(mpatches.Polygon(pts, closed=True, fill=False, lw=lw, edgecolor="black", gid="polygon"))
    off = 0.07 * size
    for (x, y), lab in zip(pts, spec["labels"]):
        d = math.hypot(x - cx, y - cy) or 1.0
        ax.text(x + off * (x - cx) / d, y + off * (y - cy) / d, lab, fontsize=fs, family=fam, ha="center", va="center",
                gid=f"vertex_label:{lab}")
    for i in spec["shown"]:
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        nx, ny = y2 - y1, -(x2 - x1)
        d = math.hypot(nx, ny) or 1.0
        if (mx - cx) * nx + (my - cy) * ny < 0:
            nx, ny = -nx, -ny
        ax.text(mx + 0.06 * size * nx / d, my + 0.06 * size * ny / d, spec["lengths"][i], fontsize=fs - 1, family=fam,
                ha="center", va="center", gid=f"length:{spec['sides'][i]}")
    if spec["right"] is not None:
        i = spec["right"]
        v = pts[i]
        u1 = [(pts[i - 1][k] - v[k]) for k in range(2)]
        u2 = [(pts[(i + 1) % n][k] - v[k]) for k in range(2)]
        s = 0.07 * size
        u1 = [c / math.hypot(*u1) * s for c in u1]
        u2 = [c / math.hypot(*u2) * s for c in u2]
        ax.plot([v[0] + u1[0], v[0] + u1[0] + u2[0], v[0] + u2[0]], [v[1] + u1[1], v[1] + u1[1] + u2[1], v[1] + u2[1]],
                color="black", lw=lw * 0.7, gid="right_marker")
    if spec["angle"]:
        i = spec["angle"]["vertex"]
        v = pts[i]
        a1 = math.degrees(math.atan2(pts[i - 1][1] - v[1], pts[i - 1][0] - v[0]))
        a2 = math.degrees(math.atan2(pts[(i + 1) % n][1] - v[1], pts[(i + 1) % n][0] - v[0]))
        lo, hi = sorted((a1, a2))
        if hi - lo > 180:
            lo, hi = hi, lo + 360
        r = 0.12 * size
        ax.add_patch(mpatches.Arc(v, 2 * r, 2 * r, theta1=lo, theta2=hi, lw=lw * 0.7, gid="angle_arc"))
        mid = math.radians((lo + hi) / 2)
        ax.text(v[0] + 2.1 * r * math.cos(mid), v[1] + 2.1 * r * math.sin(mid), f"{spec['angle']['value']}°",
                fontsize=fs - 2, family=fam, ha="center", va="center", gid="angle_label")
    if spec["m"]:
        i, t = spec["m"]["side"], spec["m"]["t"]
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        mx, my = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        ax.plot(mx, my, "o", color="black", ms=3 + lw, gid=f"point:{spec['m']['label']}")
        d = math.hypot(mx - cx, my - cy) or 1.0
        ax.text(mx + off * (mx - cx) / d, my + off * (my - cy) / d, spec["m"]["label"], fontsize=fs, family=fam,
                ha="center", va="center", gid=f"label:{spec['m']['label']}")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 0.25 * size
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


# ---------------------------------------------------------------- function

def _curve_value(c: dict, x: float) -> float:
    return c["a"] * x + c["b"] if c["kind"] == "line" else c["a"] * (x - c["h"]) ** 2 + c["k"]


def _label_pos(c: dict, others: list[dict], R: int) -> list[float] | None:
    for x0 in (4.0, -4.0, 3.0, -3.0, 2.0, -2.0, 5.0, -5.0):
        y0 = _curve_value(c, x0)
        if 1.5 <= abs(y0) <= R - 0.8 and all(abs(_curve_value(o, x0) - y0) >= 1.5 for o in others):
            return [x0 + 0.3, y0 + 0.3]
    return None


def _new_curve(rng: random.Random, name: str, R: int) -> dict:
    while True:
        if rng.random() < 0.5:
            c = {"kind": "line", "name": name, "a": rng.choice([-2, -1, -0.5, 0.5, 1, 2]),
                 "b": rng.choice([v for v in range(-4, 5) if v])}
        else:
            c = {"kind": "parabola", "name": name, "a": rng.choice([-1, -0.5, -0.25, 0.25, 0.5, 1]),
                 "h": rng.randint(-3, 3), "k": rng.choice([v for v in range(-4, 5) if v])}
        y0 = _curve_value(c, 0)
        if abs(y0) >= 1 and abs(y0) <= R - 0.5:
            return c


def visible_bound(spec: dict) -> float:
    """Metà del lato del quadrato in cui _draw_function disegna i grafici: x in [-V, V], tratti con |y| > V tolti."""
    return spec["R"] + 1


def _difference(f: dict, g: dict) -> tuple[float, float, float]:
    """Coefficienti (a2, a1, a0) di f - g."""
    def coeffs(c):  # a2 x^2 + a1 x + a0
        if c["kind"] == "line":
            return 0.0, c["a"], c["b"]
        return c["a"], -2 * c["a"] * c["h"], c["a"] * c["h"] ** 2 + c["k"]
    return tuple(p - q for p, q in zip(coeffs(f), coeffs(g)))


def intersection_points(f: dict, g: dict) -> list[tuple[float, float]] | None:
    """Conteggio geometrico: tutte le intersezioni reali (x, y) dei due grafici, senza margini né domini; None se f - g
    è costante (grafici paralleli o coincidenti). L'ammissibilità della domanda è un'altra cosa (_intersections)."""
    a2, a1, a0 = _difference(f, g)
    if abs(a2) < 1e-9:
        if abs(a1) < 1e-9:
            return None
        xs = [-a0 / a1]
    else:
        disc = a1 * a1 - 4 * a2 * a0
        xs = [] if disc < 0 else sorted({(-a1 - math.sqrt(disc)) / (2 * a2), (-a1 + math.sqrt(disc)) / (2 * a2)})
    return [(x, _curve_value(f, x)) for x in xs]


def _intersections(f: dict, g: dict, R: float) -> int | None:
    """Risposta della domanda: intersezioni con |x|, |y| <= R; None se la domanda non è ammissibile a vista
    (intersezione a meno di 0.4 dal bordo, tangenza, due radici vicine, grafici paralleli). v3 la chiamava con R =
    spec["R"], ma _draw_function disegna fino a R + 1: da v4 R = visible_bound(spec) (S2 §2.1)."""
    a2, a1, a0 = _difference(f, g)
    if abs(a2) < 1e-9:
        if abs(a1) < 1e-9:
            return None
        roots = [-a0 / a1]
    else:
        disc = a1 * a1 - 4 * a2 * a0
        if abs(disc) < 0.5:
            return None
        roots = [] if disc < 0 else [(-a1 - math.sqrt(disc)) / (2 * a2), (-a1 + math.sqrt(disc)) / (2 * a2)]
    count = 0
    for x in roots:
        y = _curve_value(f, x)
        inside = abs(x) <= R and abs(y) <= R
        near_edge = abs(abs(x) - R) < 0.4 or abs(abs(y) - R) < 0.4
        if near_edge:
            return None
        count += inside
    if len(roots) == 2 and abs(roots[0] - roots[1]) < 0.6:
        return None
    return count


NEAR_MISS = 0.3  # da v4: senza intersezioni, due grafici più vicini di così nel tratto disegnato sembrano toccarsi


def _near_miss(f: dict, g: dict, bound: float) -> bool:
    """Nessuna intersezione reale, ma dove sono disegnati entrambi (|x| <= bound, |y| <= bound) i due grafici arrivano a
    meno di NEAR_MISS: la risposta "0" non si vede con certezza (S2 §2.1)."""
    if intersection_points(f, g) != []:  # None (paralleli) o intersezioni reali: valgono le altre regole
        return False
    grid = [-bound + 2 * bound * k / 2000 for k in range(2001)]
    gaps = [abs(_curve_value(f, x) - _curve_value(g, x)) for x in grid
            if abs(_curve_value(f, x)) <= bound and abs(_curve_value(g, x)) <= bound]
    return bool(gaps) and min(gaps) < NEAR_MISS


def _function_spec(rng: random.Random) -> dict:
    R = 6
    two = rng.random() < 0.5
    while True:
        curves = [_new_curve(rng, "f", R)] + ([_new_curve(rng, "g", R)] if two else [])
        for c in curves:
            c["label_at"] = _label_pos(c, [o for o in curves if o is not c], R)
        if all(c["label_at"] for c in curves):  # ogni grafico nominato nelle domande ha la sua etichetta
            break
    return {"family": "function", "R": R, "ticks": rng.random() < 0.5, "curves": curves, **_style(rng)}


def _function_qa(spec: dict, rng: random.Random, version: int) -> list[dict]:
    out = []
    for c in spec["curves"]:
        f = c["name"]
        if c["kind"] == "parabola":
            out.append(_qa("opens", f"Does the graph of {f} open upward or downward?", ["upward", "downward"],
                           "upward" if c["a"] > 0 else "downward"))
            out.append(_qa("vertex_ud", f"Is the vertex of the graph of {f} above or below the x-axis?", ["above", "below"],
                           "above" if c["k"] > 0 else "below"))
            if abs(c["h"]) >= 1:
                out.append(_qa("vertex_lr", f"Is the vertex of the graph of {f} to the left or to the right of the y-axis?",
                               ["left", "right"], "left" if c["h"] < 0 else "right"))
        else:
            out.append(_qa("slope", f"Is the graph of {f} increasing or decreasing?", ["increasing", "decreasing"],
                           "increasing" if c["a"] > 0 else "decreasing"))
        out.append(_qa("intercept", f"Does the graph of {f} cross the y-axis above or below the origin?", ["above", "below"],
                       "above" if _curve_value(c, 0) > 0 else "below"))
    if len(spec["curves"]) == 2:
        if version < 4:
            n = _intersections(*spec["curves"], spec["R"])
        else:  # dominio disegnato e grafici che si sfiorano senza incrociarsi
            V = visible_bound(spec)
            n = _intersections(*spec["curves"], V)
            if n is not None and _near_miss(*spec["curves"], V):
                n = None
        if n is not None:
            out.append(_qa("intersections", "How many times do the graphs of f and g intersect in the figure?",
                           ["0", "1", "2"], str(n)))
    used = {c["name"] for c in spec["curves"]}
    out.append(_qa("exists", f"Is there a graph labeled {rng.choice(sorted(used))}?", YES_NO, "yes"))
    out.append(_qa("exists", f"Is there a graph labeled {rng.choice([n for n in 'hpq' if n not in used])}?", YES_NO, "no"))
    return out


def _draw_function(ax, spec: dict) -> None:
    import numpy as np

    R, V = spec["R"], visible_bound(spec)
    _draw_axes(ax, R, spec, spec["ticks"], 1, labels=spec["ticks"])
    xs = np.linspace(-V, V, 600)
    colors = ["black", "black"] if spec["lw"] < 1.5 else ["black", "#1f4e9c"]
    for c, color in zip(spec["curves"], colors):
        ys = np.array([_curve_value(c, float(x)) for x in xs])
        ys[np.abs(ys) > V] = np.nan
        (line,) = ax.plot(xs, ys, color=color, lw=spec["lw"], gid=f"curve:{c['name']}")
        line.drawn_ylim = float(V)  # per _layout: oltre questo |y| il tratto non è disegnato
        ax.text(*c["label_at"], c["name"], fontsize=spec["font"], family=spec["font_family"], style="italic", color=color,
                gid=f"curve_label:{c['name']}")
    ax.set_xlim(-R - 1.5, R + 1.5)
    ax.set_ylim(-R - 1.5, R + 1.5)


# ---------------------------------------------------------------- shapes

SHAPE_TYPES = ("circle", "triangle", "square", "rectangle")


def _shapes_spec(rng: random.Random) -> dict:
    n = rng.randint(1, 4)
    cells = rng.sample([(c, r) for c in range(3) for r in range(3)], n)
    labels = rng.sample(LETTERS, n)
    shapes = [{"type": rng.choice(SHAPE_TYPES), "label": lab, "cx": c * 3 + 1.5 + rng.uniform(-0.3, 0.3),
               "cy": r * 3 + 1.5 + rng.uniform(-0.3, 0.3), "size": rng.uniform(0.8, 1.2)}
              for (c, r), lab in zip(cells, labels)]
    return {"family": "shapes", "shapes": shapes, **_style(rng)}


def _shape_box(s: dict) -> tuple[float, float, float, float]:
    """Riquadro (x0, x1, y0, y1) della figura come la disegna _draw_shapes."""
    x, y, r = s["cx"], s["cy"], s["size"]
    w, lo, hi = {"circle": (r, -r, r), "square": (r, -r, r), "rectangle": (1.2 * r, -0.6 * r, 0.6 * r),
                 "triangle": (r, -0.8 * r, r)}[s["type"]]
    return x - w, x + w, y + lo, y + hi


def _shapes_overlap(shapes: list[dict], gap: float = SHAPE_GAP) -> bool:
    """Due figure con i riquadri a meno di `gap` (anche sovrapposti)."""
    boxes = [_shape_box(s) for s in shapes]
    return any(a[0] < b[1] + gap and b[0] < a[1] + gap and a[2] < b[3] + gap and b[2] < a[3] + gap
               for i, a in enumerate(boxes) for b in boxes[i + 1:])


def _shapes_qa(spec: dict, rng: random.Random, version: int) -> list[dict]:
    shapes = spec["shapes"]
    types = [s["type"] for s in shapes]
    out = [_qa("count", "How many shapes are in the figure?", [str(k) for k in range(1, 6)], str(len(shapes)))]
    present = sorted(set(types))
    # un quadrato è anche un rettangolo: con un quadrato non si chiede del rettangolo (audit di F1c)
    absent = [t for t in SHAPE_TYPES if t not in present and not (t == "rectangle" and "square" in present)]
    out.append(_qa("has_type", f"Is there a {rng.choice(present)} in the figure?", YES_NO, "yes"))
    if absent:
        out.append(_qa("has_type", f"Is there a {rng.choice(absent)} in the figure?", YES_NO, "no"))
    t = rng.choice([x for x in SHAPE_TYPES if not (x == "rectangle" and "square" in present)])
    out.append(_qa("count_type", f"How many {t}s are in the figure?", [str(k) for k in range(0, 5)], str(types.count(t))))
    for s in shapes:  # v3: aperta (un quadrato è anche un rettangolo); v4: il nome più specifico fra le opzioni
        q = (f"What shape is labeled {s['label']}?" if version < 4 else
             f"What is the most specific name of the shape labeled {s['label']}: {_options_text(list(SHAPE_TYPES))}?")
        out.append(_qa("label_type", q, list(SHAPE_TYPES), s["type"]))
    used = {s["label"] for s in shapes}
    out.append(_qa("exists", f"Is there a shape labeled {rng.choice(sorted(used))}?", YES_NO, "yes"))
    out.append(_qa("exists", f"Is there a shape labeled {_absent_label(rng, used)}?", YES_NO, "no"))
    return out


def _draw_shapes(ax, spec: dict) -> None:
    import matplotlib.patches as mpatches

    lw, fs, fam = spec["lw"], spec["font"], spec["font_family"]
    for s in spec["shapes"]:
        x, y, r = s["cx"], s["cy"], s["size"]
        if s["type"] == "circle":
            patch = mpatches.Circle((x, y), r, fill=False, lw=lw)
        elif s["type"] == "square":
            patch = mpatches.Rectangle((x - r, y - r), 2 * r, 2 * r, fill=False, lw=lw)
        elif s["type"] == "rectangle":
            patch = mpatches.Rectangle((x - 1.2 * r, y - 0.6 * r), 2.4 * r, 1.2 * r, fill=False, lw=lw)
        else:
            patch = mpatches.Polygon([(x - r, y - 0.8 * r), (x + r, y - 0.8 * r), (x, y + r)], closed=True, fill=False, lw=lw)
        patch.set_gid(f"shape:{s['label']}")
        ax.add_patch(patch)
        ax.text(x, y - 0.1 * r, s["label"], fontsize=fs, family=fam, ha="center", va="center", gid=f"label:{s['label']}")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)


# ---------------------------------------------------------------- interfaccia

SPECS = {"coord": _coord_spec, "polygon": _polygon_spec, "function": _function_spec, "shapes": _shapes_spec}
QAS = {"coord": _coord_qa, "polygon": _polygon_qa, "function": _function_qa, "shapes": _shapes_qa}
DRAW = {"coord": _draw_coord, "polygon": _draw_polygon, "function": _draw_function, "shapes": _draw_shapes}


def make_spec(key: str, family: str | None = None, version: int = GENERATOR_VERSION) -> dict:
    """Specifica della scena. Con version=3 è quella degli shard v3; da v4 le shapes con due figure troppo vicine si
    estraggono di nuovo dallo stesso generatore, quindi le altre scene restano identiche a v3."""
    if version not in VERSIONS:
        raise ValueError(f"versione del generatore {version!r}: attesa una di {VERSIONS}")
    rng = rng_for(key)
    family = family or FAMILIES[rng.randrange(len(FAMILIES))]
    spec = SPECS[family](rng)
    if version >= 4 and family == "shapes":
        while _shapes_overlap(spec["shapes"]):
            spec = SPECS[family](rng)
    spec["key"] = key
    if version >= 4:
        spec["generator_version"] = version
    return spec


def questions(spec: dict) -> list[dict]:
    """Tutte le domande della scena (deterministiche dato spec["key"]), nella versione del generatore della specifica."""
    return QAS[spec["family"]](spec, rng_for(spec["key"] + "/qa"), spec_version(spec))


def render(spec: dict, layout: bool = False):
    """PIL.Image RGB di spec["width"] x spec["height"] (multipli di 32); con layout=True la coppia (immagine,
    oggetti disegnati: vedi _layout). I pixel non dipendono da layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    w, h = spec["width"], spec["height"]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    try:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        DRAW[spec["family"]](ax, spec)
        # limiti fissi e riquadro ridotto: con adjustable="datalim" matplotlib spostava i limiti e tagliava la scena
        ax.set_aspect("equal", adjustable="box", anchor="C")
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        items = _layout(fig, ax, (w, h)) if layout else None
    finally:
        plt.close(fig)
    image = Image.fromarray(rgb)
    if image.size != (w, h):
        image = image.resize((w, h), Image.BICUBIC)
    return (image, items) if layout else image


def _layout(fig, ax, size: tuple[int, int]) -> dict:
    """Oggetti con un gid, come sono stati disegnati: riquadri in pixel dell'immagine (origine in alto a sinistra) e dati
    nelle unità del disegno. Testi: testo e riquadro; marcatori: centro; spezzate: vertici, oppure per i grafici gli
    intervalli di x in cui la curva è disegnata; figure: geometria. `transform` porta (x, y) del disegno in pixel."""
    renderer = fig.canvas.get_renderer()
    cw, ch = fig.canvas.get_width_height()
    sx, sy = size[0] / cw, size[1] / ch  # 1 se il canvas ha già la dimensione dell'immagine

    def box(artist) -> list[float]:
        b = artist.get_window_extent(renderer)
        return [round(b.x0 * sx, 1), round((ch - b.y1) * sy, 1), round(b.x1 * sx, 1), round((ch - b.y0) * sy, 1)]

    def num(v) -> float:
        return round(float(v), 6)

    items = []
    for t in ax.texts:
        gid = t.get_gid()
        if gid and t.get_text():
            items.append({"gid": gid, "kind": "text", "text": t.get_text(), "bbox": box(t)})
    for line in ax.lines:
        gid = line.get_gid()
        if not gid or gid == "tick":
            continue
        xs, ys = [float(v) for v in line.get_xdata()], [float(v) for v in line.get_ydata()]
        if gid.startswith("curve:"):
            runs, cur = [], None
            for x, y in zip(xs, ys):
                if y == y:  # NaN: tratto non disegnato
                    cur = [x, x] if cur is None else [cur[0], x]
                elif cur is not None:
                    runs.append(cur)
                    cur = None
            if cur is not None:
                runs.append(cur)
            items.append({"gid": gid, "kind": "curve", "visible": [[num(a), num(b)] for a, b in runs],
                          "x_range": [num(min(xs)), num(max(xs))], "clip_y": getattr(line, "drawn_ylim", None)})
        elif line.get_marker() not in (None, "None", "", " "):
            items.append({"gid": gid, "kind": "marker", "xy": [num(xs[0]), num(ys[0])], "bbox": box(line)})
        else:
            items.append({"gid": gid, "kind": "line", "path": [[num(x), num(y)] for x, y in zip(xs, ys)], "bbox": box(line)})
    for p in ax.patches:
        gid = p.get_gid()
        if not gid:
            continue
        item = {"gid": gid, "kind": "patch", "shape": type(p).__name__, "bbox": box(p)}
        if hasattr(p, "get_xy") and type(p).__name__ == "Polygon":
            item["path"] = [[num(x), num(y)] for x, y in p.get_xy()]
        elif type(p).__name__ == "Rectangle":
            item.update(xy=[num(v) for v in p.get_xy()], width=num(p.get_width()), height=num(p.get_height()))
        elif type(p).__name__ == "Circle":
            item.update(center=[num(v) for v in p.center], radius=num(p.radius))
        elif type(p).__name__ == "Arc":
            item.update(center=[num(v) for v in p.center])
        items.append(item)
    ox, oy = ax.transData.transform((0.0, 0.0))
    ux, uy = ax.transData.transform((1.0, 1.0)) - (ox, oy)
    return {"size": list(size), "xlim": [num(v) for v in ax.get_xlim()], "ylim": [num(v) for v in ax.get_ylim()],
            "transform": {"x0": num(ox * sx), "y0": num((ch - oy) * sy), "ux": num(ux * sx), "uy": num(-uy * sy)},
            "items": items}


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def counterfactual(spec: dict) -> tuple[dict, dict, dict, str, str] | None:
    """(scena originale, scena modificata, domanda, risposta sull'originale, risposta sulla modificata), oppure
    None. Si cambia un solo dettaglio decisivo e si sceglie una domanda la cui risposta cambia. L'originale
    restituito può differire da `spec` (nel piano cartesiano si toglie la retta), `spec` non si tocca."""
    rng = rng_for(spec["key"] + "/cf")
    spec = deepcopy(spec)
    fam = spec["family"]
    if fam == "coord":
        spec["line"] = None  # niente retta: riflettere un punto ne cambierebbe i fatti
    new = deepcopy(spec)
    if fam == "coord":
        p = rng.choice(range(len(spec["points"])))
        a, b = spec["points"][p], new["points"][p]
        axis = rng.choice(["x", "y"])
        b[axis] = -b[axis]
        if any(max(abs(b["x"] - o["x"]), abs(b["y"] - o["y"])) < 2 for i, o in enumerate(new["points"]) if i != p):
            return None
        if axis == "x":
            q = _qa("side_y", f"Is point {a['label']} to the left or to the right of the y-axis?", ["left", "right"], "left")
            return spec, new, q, "left" if a["x"] < 0 else "right", "left" if b["x"] < 0 else "right"
        q = _qa("side_x", f"Is point {a['label']} above or below the x-axis?", ["above", "below"], "above")
        return spec, new, q, "above" if a["y"] > 0 else "below", "above" if b["y"] > 0 else "below"
    if fam == "function":
        c = rng.randrange(len(spec["curves"]))
        old, cur = spec["curves"][c], new["curves"][c]
        cur["a"] = -cur["a"]
        for o in new["curves"]:  # l'etichetta deve restare vicina al suo grafico e lontana dall'altro
            o["label_at"] = _label_pos(o, [x for x in new["curves"] if x is not o], new["R"])
        if not all(o["label_at"] for o in new["curves"]) or abs(_curve_value(cur, 0)) > new["R"] - 0.5:
            return None
        if cur["kind"] == "parabola":
            q = _qa("opens", f"Does the graph of {old['name']} open upward or downward?", ["upward", "downward"], "upward")
            return spec, new, q, "upward" if old["a"] > 0 else "downward", "upward" if cur["a"] > 0 else "downward"
        q = _qa("slope", f"Is the graph of {old['name']} increasing or decreasing?", ["increasing", "decreasing"], "increasing")
        return spec, new, q, "increasing" if old["a"] > 0 else "decreasing", "increasing" if cur["a"] > 0 else "decreasing"
    if fam == "polygon":
        i = rng.choice(spec["shown"])
        v = float(spec["lengths"][i])
        alt = _fmt_len(v + rng.choice([2, 3, 4]))
        new["lengths"][i] = alt
        opts = [spec["lengths"][i], alt]
        q = _qa("side_len", f"What is the length of side {spec['sides'][i]}?", opts, spec["lengths"][i])
        return spec, new, q, spec["lengths"][i], alt
    # shapes: una figura in più o in meno
    shapes = new["shapes"]
    if len(shapes) > 1 and rng.random() < 0.5:
        shapes.pop(rng.randrange(len(shapes)))
    elif len(shapes) < 4:
        used = {(round((s["cx"] - 1.5) / 3), round((s["cy"] - 1.5) / 3)) for s in shapes}
        free = [(c, r) for c in range(3) for r in range(3) if (c, r) not in used]
        c, r = rng.choice(free)
        shapes.append({"type": rng.choice(SHAPE_TYPES), "label": _absent_label(rng, {s["label"] for s in shapes}),
                       "cx": c * 3 + 1.5, "cy": r * 3 + 1.5, "size": 1.0})
    else:
        return None
    q = _qa("count", "How many shapes are in the figure?", [str(k) for k in range(1, 6)], str(len(spec["shapes"])))
    return spec, new, q, str(len(spec["shapes"])), str(len(shapes))
