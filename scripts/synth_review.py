#!/usr/bin/env python3
"""Analisi degli scarti del controllo dei sintetici (job 30393, outputs/synth_check/c1c2-run1) in un job SLURM CPU, per
distinguere errori reali, ambiguità e controlli troppo restrittivi prima del render di v4. Non scrive shard e non
modifica né il generatore né il checker: legge problems.jsonl e report.json del controllo e lo shard delle coppie v3,
ridisegna in memoria le scene indicate (le chiavi sono deterministiche) e salva tabelle e immagini nella cartella
nuova --out.

Tre analisi:
- coppie controfattuali polygon di v3. Per ogni coppia: la risposta contro la misura stampata; la fattibilità di un
  triangolo con le misure stampate e i vincoli dichiarati nel disegno (angolo retto marcato, angolo scritto), con gli
  intervalli d'arrotondamento delle misure ("larga": intero ±0.5, decimale ±0.05, angolo ±0.5°; "stretta": intero
  esatto); la compatibilità con una riscalatura uniforme del disegno; le sovrapposizioni della misura cambiata. Le
  proporzioni del disegno non contano come vincolo;
- riquadri di testo sovrapposti (testi_sovrapposti, testo_su_marcatore) nelle scene v4 non approvate dal checker.
  Per ogni violazione: categoria, intersezione dei riquadri, inchiostro dei due oggetti disegnati ciascuno da solo
  (pixel in comune, distanza fra i tratti pieni). Poi un campione stratificato per categoria e per sovrapposizione
  lieve o marcata dei riquadri, più scene approvate di controllo, con ritagli ingranditi e le domande che dipendono
  da quei testi;
- domande coords non ammissibili: etichetta mancante, etichette vicine, tacca disegnata e sua sporgenza dall'asse.

Uscite in --out: summary.json, pairs.jsonl, overlaps.jsonl, sample.jsonl, coords.jsonl, img/ (fogli PNG e scene del
campione).

    srun $RUN_PYTHON -u -B scripts/synth_review.py --out DIR [--check outputs/synth_check/c1c2-run1]
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import synth, synth_check  # noqa: E402
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.provenance import code_identity, sha256_file, write_new_json  # noqa: E402

OVERLAP_KINDS = ("testi_sovrapposti", "testo_su_marcatore")
ORIGIN = {"axis_label:O", "tick_label:x:-1", "tick_label:y:-1"}
NAMES = {"length": "misura", "label": "etichetta", "point": "punto", "tick_label": "tacca", "axis_label": "asse",
         "vertex_label": "vertice", "curve_label": "grafico", "angle_label": "angolo"}
LIGHT_PX = 3.0   # sovrapposizione lieve: il lato minore dell'intersezione dei due riquadri è al più 3 px
INK = 160        # luminanza (0-255) sotto la quale un pixel è inchiostro
CORE = 96        # tratto pieno, per la distanza fra i glifi
TOUCH_PX = 1.5   # glifi a contatto: tratti pieni a meno di 1.5 px


# ---------------------------------------------------------------- disegno oggetto per oggetto

def _scene(spec: dict):
    """Figura e assi costruiti come in synth.render, per poter disegnare gli oggetti uno alla volta."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(spec["width"] / 100, spec["height"] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    synth.DRAW[spec["family"]](ax, spec)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    return fig, ax


def _close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _pixels(fig):
    import numpy as np

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def _artists(ax) -> list:
    return list(ax.texts) + list(ax.lines) + list(ax.patches) + list(ax.collections)


def _by_gid(ax, gid: str):
    found = [a for a in _artists(ax) if a.get_gid() == gid]
    if len(found) != 1:
        raise ValueError(f"gid {gid}: {len(found)} oggetti")
    return found[0]


def _alone(fig, ax, target):
    """Pixel della figura con il solo `target` visibile, su sfondo bianco."""
    arts = _artists(ax)
    visible = [a.get_visible() for a in arts]
    for a in arts:
        a.set_visible(a is target)
    try:
        return _pixels(fig)
    finally:
        for a, v in zip(arts, visible):
            a.set_visible(v)


def _lum(rgb):
    return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


def _ink(la, lb) -> dict:
    """Pixel d'inchiostro in comune fra due oggetti disegnati da soli e distanza minima fra i loro tratti pieni."""
    import numpy as np

    shared = int(np.count_nonzero((la < INK) & (lb < INK)))
    ca, cb = np.argwhere(la < CORE), np.argwhere(lb < CORE)
    gap = None
    if len(ca) and len(cb):
        d = np.sqrt(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1))
        gap = round(float(d.min()), 2)
    if shared:
        ink = "inchiostro_in_comune"
    elif gap is None:
        ink = "non_misurabile"
    else:
        ink = "a_contatto" if gap <= TOUCH_PX else "separati"
    return {"ink_shared_px": shared, "core_gap_px": gap, "ink": ink}


def _items(layout: dict) -> dict:
    return {it["gid"]: it for it in layout["items"] if "bbox" in it}


def _box_overlap(a, b) -> tuple[float, float]:
    return round(min(a[2], b[2]) - max(a[0], b[0]), 1), round(min(a[3], b[3]) - max(a[1], b[1]), 1)


def _category(kind: str, gids: list[str]) -> str:
    if set(gids) <= ORIGIN:
        return "origine"
    a, b = (NAMES.get(g.split(":")[0], g.split(":")[0]) for g in gids)
    if kind == "testo_su_marcatore":
        return f"{a}_su_{b}"
    return "_".join(sorted((a, b)))


def _stratum(family: str, category: str) -> str:
    if category == "origine":
        return f"origine/{family}"
    if family == "polygon" and category in ("etichetta_misura", "misura_su_punto"):
        return f"{category}/polygon"
    return "altro"


# ---------------------------------------------------------------- censimento delle sovrapposizioni v4

def _overlap_row(key: str, family: str, v: dict, items: dict, lums: dict) -> dict:
    ga, gb = v["gids"]
    ba, bb = items[ga]["bbox"], items[gb]["bbox"]
    ox, oy = _box_overlap(ba, bb)
    category = _category(v["kind"], v["gids"])
    return {"key": key, "family": family, "kind": v["kind"], "gids": [ga, gb], "category": category,
            "stratum": _stratum(family, category), "texts": [items[ga].get("text"), items[gb].get("text")],
            "bbox": [ba, bb], "box_overlap_px": [ox, oy], "severity": "lieve" if min(ox, oy) <= LIGHT_PX else "marcata",
            **_ink(lums[ga], lums[gb])}


def census_task(task: tuple[str, str]) -> list[dict]:
    key, family = task
    spec = synth.make_spec(key, family, 4)
    _, layout = synth.render(spec, layout=True)
    viols = [v for v in synth_check.renderer_violations(spec, layout) if v["kind"] in OVERLAP_KINDS]
    if not viols:
        return []
    items = _items(layout)
    fig, ax = _scene(spec)
    try:
        lums = {}
        for v in viols:
            for g in v["gids"]:
                if g not in lums:
                    lums[g] = _lum(_alone(fig, ax, _by_gid(ax, g)))
    finally:
        _close(fig)
    return [_overlap_row(key, family, v, items, lums) for v in viols]


def scan_task(task: tuple[str, str]) -> dict | None:
    """Scena v4 approvata: la coppia di testi coi riquadri più vicini (separazione negativa = sovrapposti)."""
    key, family = task
    spec = synth.make_spec(key, family, 4)
    _, layout = synth.render(spec, layout=True)
    texts = [it for it in layout["items"] if it["kind"] == "text"]
    best = None
    for a, b in itertools.combinations(texts, 2):
        ba, bb = a["bbox"], b["bbox"]
        sep = max(max(ba[0], bb[0]) - min(ba[2], bb[2]), max(ba[1], bb[1]) - min(ba[3], bb[3]))
        if best is None or sep < best[0]:
            best = (sep, [a["gid"], b["gid"]])
    if best is None:
        return None
    return {"key": key, "family": family, "separation_px": round(best[0], 1), "gids": best[1]}


def choose_sample(rows: list[dict], per_cell: int, seed: int) -> list[tuple[str, str, dict]]:
    """Fino a per_cell violazioni per strato e gravità, in scene diverse, con un'estrazione deterministica."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["stratum"], r["severity"])].append(r)
    chosen, used = [], set()
    for stratum, severity in sorted(cells):
        cand = sorted(cells[(stratum, severity)], key=lambda r: (r["key"], r["gids"]))
        random.Random(f"{seed}/{stratum}/{severity}").shuffle(cand)
        taken = 0
        for r in cand:
            if taken == per_cell:
                break
            if r["key"] in used:
                continue
            chosen.append((stratum, severity, r))
            used.add(r["key"])
            taken += 1
    return chosen


# ---------------------------------------------------------------- casi del campione

def _needs(spec: dict, q: dict) -> set[str]:
    """gid dei testi e dei marcatori da cui dipende la lettura della risposta (approssimazione dichiarata: nomi
    citati, misure ed etichette di tacca necessarie)."""
    fam, t, text = spec["family"], q["type"], q["q"]
    if fam == "coord":
        pts = {p["label"]: p for p in spec["points"]}
        names = [n for n in re.findall(r"point (\w+)", text) if n in pts]
        need = {f"label:{n}" for n in names} | {f"point:{n}" for n in names}
        if t == "coords":
            for n in names:
                need |= {f"tick_label:x:{pts[n]['x']}", f"tick_label:y:{pts[n]['y']}"}
        elif t in ("count", "exists"):
            need |= {f"label:{n}" for n in pts} | {f"point:{n}" for n in pts}
            if t == "count":
                need.add("axis_label:O")
        return need
    if fam == "polygon":
        vertices = {f"vertex_label:{c}" for c in spec["labels"]}
        m = spec.get("m")
        mark = {f"label:{m['label']}", f"point:{m['label']}"} if m else set()
        if t == "side_len":
            side = re.search(r"side (\w+)\?", text).group(1)
            return {f"length:{side}"} | {f"vertex_label:{c}" for c in side}
        if t == "m_side":
            return mark | vertices
        if t == "angle_val":
            return {"angle_label"} | vertices
        if t == "count":
            return {f"length:{spec['sides'][i]}" for i in spec["shown"]}
        if t == "exists":
            return vertices | mark
        return vertices  # right_vertex, highest, lowest, leftmost, rightmost, shape_type
    if fam == "function":
        if t in ("exists", "intersections"):
            return {f"curve_label:{c['name']}" for c in spec["curves"]}
        return {f"curve_label:{n}" for n in re.findall(r"graph of (\w+)", text)}
    return set()


def _crops(image, boxes, la, lb, margin: int = 10, zoom: int = 4) -> list:
    """Ritaglio ingrandito: com'è, coi due riquadri (rosso, blu), e con l'inchiostro dei due oggetti disegnati da soli
    (rosso, blu, viola dove coincidono)."""
    import numpy as np
    from PIL import Image, ImageDraw

    W, H = image.size
    x0 = max(0, int(math.floor(min(b[0] for b in boxes))) - margin)
    y0 = max(0, int(math.floor(min(b[1] for b in boxes))) - margin)
    x1 = min(W, int(math.ceil(max(b[2] for b in boxes))) + margin)
    y1 = min(H, int(math.ceil(max(b[3] for b in boxes))) + margin)
    size = ((x1 - x0) * zoom, (y1 - y0) * zoom)
    raw = image.crop((x0, y0, x1, y1)).resize(size, Image.NEAREST)
    boxed = raw.copy()
    draw = ImageDraw.Draw(boxed)
    for b, color in zip(boxes, ((220, 30, 30), (30, 60, 220))):
        draw.rectangle([(b[0] - x0) * zoom, (b[1] - y0) * zoom, (b[2] - x0) * zoom, (b[3] - y0) * zoom],
                       outline=color, width=2)
    ink = np.full(la.shape + (3,), 255, dtype=np.uint8)
    ma, mb = la < INK, lb < INK
    ink[ma] = (230, 70, 70)
    ink[mb] = (70, 100, 230)
    ink[ma & mb] = (150, 0, 150)
    inked = Image.fromarray(ink).crop((x0, y0, x1, y1)).resize(size, Image.NEAREST)
    return [raw, boxed, inked]


def _case(key: str, family: str, version: int, ga: str, gb: str, stratum: str, severity: str,
          img: Path) -> tuple[dict, list]:
    import numpy as np

    spec = synth.make_spec(key, family, version)
    image, layout = synth.render(spec, layout=True)
    items = _items(layout)
    fig, ax = _scene(spec)
    try:
        full = _pixels(fig)
        pa, pb = _alone(fig, ax, _by_gid(ax, ga)), _alone(fig, ax, _by_gid(ax, gb))
    finally:
        _close(fig)
    la, lb = _lum(pa), _lum(pb)
    ba, bb = items[ga]["bbox"], items[gb]["bbox"]
    qs = synth.questions(spec)
    results = synth_check.check_questions(spec, qs, layout)
    dep = [{"type": q["type"], "q": q["q"], "answer": q["answer"], "status": r["status"]}
           for q, r in zip(qs, results) if _needs(spec, q) & {ga, gb}]
    viols = synth_check.renderer_violations(spec, layout)
    png = f"img/scenes/{key}-v{version}.png"
    path = img.parent / png
    if path.exists():
        sys.exit(f"{path} esiste già")
    path.write_bytes(synth.png_bytes(image))
    row = {"key": key, "family": family, "version": version, "stratum": stratum, "severity": severity,
           "gids": [ga, gb], "texts": [items[ga].get("text"), items[gb].get("text")], "bbox": [ba, bb],
           "box_overlap_px": list(_box_overlap(ba, bb)), **_ink(la, lb), "dependent_questions": dep,
           "overlap_violations": [[v["kind"], v["gids"]] for v in viols if v["kind"] in OVERLAP_KINDS],
           "render_identical": bool(np.array_equal(full, np.asarray(image))), "png": png}
    return row, _crops(image, [ba, bb], la, lb)


def _caption(n: int, row: dict) -> str:
    ox, oy = row["box_overlap_px"]
    dep = ", ".join(f"{q['type']}:{q['status']}" for q in row["dependent_questions"]) or "nessuna"
    return (f"{n}. {row['key']} {row['family']} [{row['stratum']}, {row['severity']}] {row['gids'][0]} x {row['gids'][1]}"
            f" ({row['texts'][0]!r} / {row['texts'][1]!r})\n"
            f"   riquadri {ox} x {oy} px | inchiostro in comune {row['ink_shared_px']} px | distanza dei tratti"
            f" {row['core_gap_px']} px | {row['ink']} | domande: {dep}")


# ---------------------------------------------------------------- fogli di immagini

def _save_new(image, path: Path) -> None:
    if path.exists():
        sys.exit(f"{path} esiste già")
    image.save(path, optimize=True)


def _sheet(rows: list[tuple[list, str]], path: Path, min_width: int = 1100) -> None:
    """Una riga per caso: didascalia (anche su più righe) e immagini affiancate."""
    from PIL import Image, ImageDraw, ImageFont

    pad, line = 8, 13
    font = ImageFont.load_default()
    caps = [text.count("\n") + 1 for _, text in rows]
    width = max([min_width] + [sum(im.width for im in ims) + pad * (len(ims) + 1) for ims, _ in rows])
    heights = [max(im.height for im in ims) + line * c + 2 * pad for (ims, _), c in zip(rows, caps)]
    sheet = Image.new("RGB", (width, sum(heights)), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for (ims, text), c, h in zip(rows, caps, heights):
        draw.multiline_text((pad, y + 3), text, fill=(0, 0, 0), font=font, spacing=2)
        x = pad
        for im in ims:
            sheet.paste(im, (x, y + line * c + pad))
            x += im.width + pad
        draw.line([(0, y + h - 1), (width, y + h - 1)], fill=(190, 190, 190))
        y += h
    _save_new(sheet, path)


def _grid_sheet(cells: list[tuple[list, str]], cols: int, path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    pad, cap = 6, 14
    font = ImageFont.load_default()
    cw = max(sum(im.width for im in ims) + pad * (len(ims) - 1) for ims, _ in cells) + 2 * pad
    cw = max(cw, 150)
    ch = max(max(im.height for im in ims) for ims, _ in cells) + cap + 2 * pad
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cw * cols, ch * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for n, (ims, text) in enumerate(cells):
        x0, y0 = (n % cols) * cw, (n // cols) * ch
        draw.text((x0 + pad, y0 + 2), text, fill=(0, 0, 0), font=font)
        x = x0 + pad
        for im in ims:
            sheet.paste(im, (x, y0 + cap + pad))
            x += im.width + pad
        draw.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline=(200, 200, 200))
    _save_new(sheet, path)


def _scaled(image, factor: float):
    from PIL import Image

    return image.resize((round(image.width * factor), round(image.height * factor)), Image.LANCZOS)


# ---------------------------------------------------------------- coppie controfattuali polygon

def _interval(text: str, integers_exact: bool) -> tuple[float, float]:
    v = float(text)
    if "." in text:
        return v - 0.05, v + 0.05
    return (v, v) if integers_exact else (v - 0.5, v + 0.5)


def _grid(lo: float, hi: float, n: int = 9) -> list[float]:
    return [lo] if hi <= lo else [lo + (hi - lo) * k / (n - 1) for k in range(n)]


def triangle_feasible(printed: dict[int, tuple[float, float]], angle: tuple[int, float, float] | None) -> bool:
    """Esiste un triangolo con i lati stampati (intervalli; il lato i va dal vertice i al vertice i+1) e con l'angolo
    dichiarato (vertice, intervallo in gradi), se c'è? I lati non stampati sono liberi."""
    if angle is None:
        if len(printed) < 3:
            return True
        return any(s[0] < s[1] + s[2] and s[1] < s[0] + s[2] and s[2] < s[0] + s[1]
                   for s in itertools.product(*(_grid(*printed[i]) for i in range(3))))
    k, lo, hi = angle
    p, q, o = (k - 1) % 3, k, (k + 1) % 3  # i due lati del vertice k e il lato opposto
    known = set(printed)
    if len(known) <= 1 or known == {p, q}:
        return True
    thetas = [math.radians(t) for t in _grid(lo, hi)]
    if known == {p, q, o}:  # teorema del coseno: serve uno zero di f nella scatola degli intervalli
        vals = [a * a + b * b - 2 * a * b * math.cos(t) - c * c for a in _grid(*printed[p]) for b in _grid(*printed[q])
                for c in _grid(*printed[o]) for t in thetas]
        return min(vals) <= 0 <= max(vals)
    adj = p if p in known else q  # un lato del vertice e l'opposto: l'altro lato del vertice è libero
    for a in _grid(*printed[adj]):
        for c in _grid(*printed[o]):
            for t in thetas:
                disc = c * c - (a * math.sin(t)) ** 2
                if disc >= 0 and a * math.cos(t) + math.sqrt(disc) > 0:
                    return True
    return False


def uniform_scale_ok(printed: dict[int, tuple[float, float]], drawn: list[float]) -> bool:
    """Le misure stampate sono quelle disegnate per un solo fattore di scala, entro gli intervalli?"""
    lo = max(printed[i][0] / drawn[i] for i in printed)
    hi = min(printed[i][1] / drawn[i] for i in printed)
    return lo <= hi


def _constraints(spec: dict) -> str:
    labels = spec["labels"]
    parts = [f"{spec['sides'][i]}={spec['lengths'][i]}" for i in spec["shown"]]
    if spec["right"] is not None:
        parts.append(f"angolo retto in {labels[spec['right']]}")
    if spec["angle"]:
        parts.append(f"angolo {spec['angle']['value']} in {labels[spec['angle']['vertex']]}")
    return ", ".join(parts)


def _png(data: bytes):
    from PIL import Image

    image = Image.open(io.BytesIO(data), formats=("PNG",))
    image.load()
    return image.convert("RGB")


SHORT = {"contraddizione_dimostrabile": "contraddizione", "contraddizione_con_interi_esatti": "interi esatti",
         "riscalatura_uniforme": "riscalatura", "solo_scala_non_simile": "non simile"}


def analyze_pairs(path: Path) -> tuple[list[dict], list, dict]:
    from PIL import Image

    groups = defaultdict(dict)
    for key, files in iter_shard(path):
        meta = json.loads(files["json"])
        if meta["family"] == "polygon":
            groups[meta["pair"]["id"]][meta["pair"]["variant"]] = (key, meta, files.get("png"))
    rows, label_cells, examples = [], [], defaultdict(list)
    for pid in sorted(groups):
        (ka, ma, data_a), (kb, mb, data_b) = groups[pid]["a"], groups[pid]["b"]
        sa, sb = ma["spec"], mb["spec"]
        changed = [i for i, (x, y) in enumerate(zip(sa["lengths"], sb["lengths"])) if x != y]
        i = changed[0]
        side, n = sa["sides"][i], len(sa["points"])
        ia, layout_a = synth.render(sa, layout=True)
        ib, layout_b = synth.render(sb, layout=True)
        printed_a = {g[len("length:"):]: it["text"] for g, it in _items(layout_a).items() if g.startswith("length:")}
        printed_b = {g[len("length:"):]: it["text"] for g, it in _items(layout_b).items() if g.startswith("length:")}
        angle = None
        if sb["right"] is not None:
            angle = (sb["right"], 90.0, 90.0)
        elif sb["angle"]:
            angle = (sb["angle"]["vertex"], sb["angle"]["value"] - 0.5, sb["angle"]["value"] + 0.5)
        feasible = {}
        for mode, exact in (("larga", False), ("stretta", True)):
            for variant, s in (("a", sa), ("b", sb)):
                printed = {j: _interval(s["lengths"][j], exact) for j in s["shown"]}
                feasible[f"{variant}_{mode}"] = triangle_feasible(printed, angle) if n == 3 else True
        pts = [tuple(p) for p in sb["points"]]
        drawn = [math.dist(pts[j], pts[(j + 1) % n]) for j in range(n)]
        uniform = uniform_scale_ok({j: _interval(sb["lengths"][j], False) for j in sb["shown"]}, drawn)
        if not feasible["b_larga"]:
            cls = "contraddizione_dimostrabile"
        elif not feasible["b_stretta"]:
            cls = "contraddizione_con_interi_esatti"
        elif uniform:
            cls = "riscalatura_uniforme"
        else:
            cls = "solo_scala_non_simile"
        viols = synth_check.renderer_violations(sb, layout_b)
        gid = f"length:{side}"
        row = {"pair": pid, "keys": [ka, kb], "kind": sa["kind"], "n": n, "changed": changed, "side": side,
               "value_a": sa["lengths"][i], "value_b": sb["lengths"][i], "drawn": round(drawn[i], 3),
               "question": mb["pair"]["q"], "answer_a": ma["pair"]["answer"], "answer_b": mb["pair"]["answer"],
               "answer_matches_printed": printed_a.get(side) == ma["pair"]["answer"]
               and printed_b.get(side) == mb["pair"]["answer"],
               "printed_sides": len(sb["shown"]), "constraints_b": _constraints(sb),
               "right_vertex": None if sb["right"] is None else sb["labels"][sb["right"]],
               "angle": None if not sb["angle"] else {"vertex": sb["labels"][sb["angle"]["vertex"]],
                                                      "value": sb["angle"]["value"]},
               "feasible": feasible, "uniform_scale": uniform, "class": cls,
               "only_lengths_differ": {k: v for k, v in sa.items() if k != "lengths"}
               == {k: v for k, v in sb.items() if k != "lengths"},
               "png_identical": synth.png_bytes(ia) == data_a and synth.png_bytes(ib) == data_b,
               "length_incoherent": [v["detail"] for v in viols if v["kind"] == "lunghezza_incoerente"],
               "changed_label_overlaps": [[v["kind"], v["gids"]] for v in viols
                                          if v["kind"] in OVERLAP_KINDS and gid in v["gids"]]}
        rows.append(row)
        img_a, img_b = _png(data_a), _png(data_b)
        boxes = [_items(layout_a)[gid]["bbox"], _items(layout_b)[gid]["bbox"]]
        x0 = max(0, int(min(b[0] for b in boxes)) - 6)
        y0 = max(0, int(min(b[1] for b in boxes)) - 6)
        x1 = min(img_b.width, int(max(b[2] for b in boxes)) + 7)
        y1 = min(img_b.height, int(max(b[3] for b in boxes)) + 7)
        crops =[im.crop((x0, y0, x1, y1)).resize(((x1 - x0) * 3, (y1 - y0) * 3), Image.NEAREST) for im in (img_a, img_b)]
        flag = " *" if row["changed_label_overlaps"] else ""
        label_cells.append((crops, f"{pid[-5:]} {side} {row['value_a']}->{row['value_b']} {SHORT[cls]}{flag}"))
        if len(examples[cls]) < 3:
            examples[cls].append(([_scaled(img_a, 0.6), _scaled(img_b, 0.6)],
                                  f"{pid}: {sa['kind']}; vincoli in b: {row['constraints_b']}\n"
                                  f"   {side} {row['value_a']} -> {row['value_b']} (disegnato {row['drawn']}); classe {cls}"))
    return rows, label_cells, examples


# ---------------------------------------------------------------- coordinate

def _tick_line(ax, axis: str, v: int):
    for line in ax.lines:
        if line.get_gid() != "tick":
            continue
        xs, ys = list(line.get_xdata()), list(line.get_ydata())
        if (axis == "x" and xs == [v, v]) or (axis == "y" and ys == [v, v]):
            return line
    return None


def _extent(lum, axis: str, cx: int, cy: int, half: int = 1, reach: int = 12) -> int:
    """Estensione dell'inchiostro attraverso l'asse, in pixel, nella colonna (asse x) o nella riga (asse y) della tacca."""
    import numpy as np

    if axis == "x":
        band = lum[:, max(0, cx - half): cx + half + 1]
        hits = np.where((band < INK).any(axis=1))[0]
        near = hits[np.abs(hits - cy) <= reach]
    else:
        band = lum[max(0, cy - half): cy + half + 1, :]
        hits = np.where((band < INK).any(axis=0))[0]
        near = hits[np.abs(hits - cx) <= reach]
    return int(near.max() - near.min() + 1) if len(near) else 0


def _band(image, t: dict, axis: str, v: int, zoom: int = 4):
    from PIL import Image

    ux, uy = t["ux"], -t["uy"]  # pixel per unità
    if axis == "x":
        cx, cy = t["x0"] + t["ux"] * v, t["y0"]
        box = (cx - 1.6 * ux, cy - 0.9 * uy, cx + 1.6 * ux, cy + 1.6 * uy)
    else:
        cx, cy = t["x0"], t["y0"] + t["uy"] * v
        box = (cx - 1.8 * ux, cy - 1.6 * uy, cx + 0.9 * ux, cy + 1.6 * uy)
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(image.width, int(box[2]) + 1), min(image.height, int(box[3]) + 1)
    return image.crop((x0, y0, x1, y1)).resize(((x1 - x0) * zoom, (y1 - y0) * zoom), Image.NEAREST)


def analyze_coords(questions: list[dict], shown: int = 4) -> tuple[list[dict], list]:
    rows, cells = [], []
    for r in questions:
        version = 4 if r["origin"] == "generato" else 3
        spec = synth.make_spec(r["key"], "coord", version)
        image, layout = synth.render(spec, layout=True)
        name = re.search(r"point (\w+)\?", r["q"]).group(1)
        p = next(p for p in spec["points"] if p["label"] == name)
        labels = {(it["gid"].split(":")[1], it["text"]) for it in layout["items"] if it["gid"].startswith("tick_label:")}
        t = layout["transform"]
        axes = {}
        fig, ax = _scene(spec)
        try:
            full = _lum(_pixels(fig))
            for axis, v in (("x", p["x"]), ("y", p["y"])):
                info = {"value": v, "label": (axis, str(v)) in labels,
                        "neighbors_labeled": [(axis, str(v - 1)) in labels, (axis, str(v + 1)) in labels]}
                if not info["label"]:
                    cx = round(t["x0"] + t["ux"] * v) if axis == "x" else round(t["x0"])
                    cy = round(t["y0"]) if axis == "x" else round(t["y0"] + t["uy"] * v)
                    tick = _tick_line(ax, axis, v)
                    info["tick_drawn"] = tick is not None
                    info["tick_px"] = _extent(_lum(_alone(fig, ax, tick)), axis, cx, cy) if tick is not None else 0
                    info["axis_px"] = _extent(_lum(_alone(fig, ax, _by_gid(ax, f"axis:{axis}"))), axis, cx, cy)
                    info["full_px"] = _extent(full, axis, cx, cy)
                    info["protrusion_px"] = round((info["full_px"] - info["axis_px"]) / 2, 1)
                axes[axis] = info
        finally:
            _close(fig)
        missing = [a for a in ("x", "y") if not axes[a]["label"]]
        drawn = all(axes[a]["tick_drawn"] and all(axes[a]["neighbors_labeled"]) for a in missing)
        visible = all(axes[a]["protrusion_px"] >= 1 for a in missing)
        if not missing:  # il checker dice che manca un'etichetta, il layout le ha entrambe
            cls = "etichette_presenti"
        elif not drawn:
            cls = "informazione_mancante"
        else:
            cls = "leggibile_da_scala" if visible else "tacca_poco_visibile"
        row = {"key": r["key"], "origin": r["origin"], "version": version, "q": r["q"], "answer": r["answer"],
               "why": r["why"], "R": spec["R"], "step": spec["step"], "ticks": spec["ticks"], "point": name,
               "x": p["x"], "y": p["y"], "line_test": bool(spec["line"]) and spec["line"]["test"] == name,
               "missing": missing, "axes": axes, "class": cls}
        rows.append(row)
        if version == 4 and len(cells) < shown and missing:
            a = missing[0]
            cells.append(([_band(image, t, a, axes[a]["value"]), _scaled(image, 0.5)],
                          f"{r['key']}: punto {name} ({p['x']}, {p['y']}), R={spec['R']}, etichette ogni {spec['step']};"
                          f" asse {a}: {axes[a]['value']} senza etichetta, vicine etichettate {axes[a]['neighbors_labeled']},"
                          f"\n   tacca {axes[a]['tick_px']} px, asse {axes[a]['axis_px']} px, sporgenza"
                          f" {axes[a]['protrusion_px']} px per lato; classe {cls}"))
    return rows, cells


# ---------------------------------------------------------------- main

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _flat(counter: Counter) -> dict:
    return {"/".join(map(str, k)) if isinstance(k, tuple) else str(k): v for k, v in sorted(counter.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--check", type=Path, default=ROOT / "outputs/synth_check/c1c2-run1")
    ap.add_argument("--pairs-shard", type=Path, default=ROOT / "data/shards/synth/v3/pairs/synth-pairs.tar")
    ap.add_argument("--per-cell", type=int, default=2, help="violazioni per strato e gravità nel campione")
    ap.add_argument("--controls-scan", type=int, default=300, help="chiavi v4 per famiglia scandite per i controlli")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("synth_review.py va eseguito in un job SLURM")
    problems_path, report_path = args.check / "problems.jsonl", args.check / "report.json"
    for p in (problems_path, report_path, args.pairs_shard):
        if not p.is_file():
            sys.exit(f"manca {p}")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: scegliere un'altra cartella")
    img = args.out / "img"
    (img / "scenes").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    nonverified, flagged, coord_questions = set(), [], []
    with open(problems_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["what"] == "sample" and r["origin"] == "generato":
                nonverified.add(r["key"])
                if r["family"] != "shapes" and any(x.startswith("altre violazioni") for x in r["reasons"]):
                    flagged.append((r["key"], r["family"]))
            elif r["what"] == "question" and r.get("type") == "coords" and r.get("status") == "non_ammissibile":
                coord_questions.append(r)
    flagged.sort()
    n_generated = report["samples"]["generato"]
    scan = [(f"synth-check-{i:07d}", synth.FAMILIES[i % len(synth.FAMILIES)])
            for i in range(min(n_generated, len(synth.FAMILIES) * args.controls_scan))]
    scan = [(k, fam) for k, fam in scan if fam != "shapes" and k not in nonverified]
    with Pool(args.workers) as pool:
        census = [row for rows in pool.imap(census_task, flagged, chunksize=8) for row in rows]
        scanned = [r for r in pool.imap(scan_task, scan, chunksize=8) if r]
    print(f"censimento: {len(flagged)} scene, {len(census)} violazioni ({time.time() - t0:.0f} s)", flush=True)
    got = Counter((r["family"], r["kind"]) for r in census)
    want = Counter({(fam, kind): n for fam, kinds in report["violations"]["generato"].items()
                    for kind, n in kinds.items() if kind in OVERLAP_KINDS})

    sample_rows, sheets = [], defaultdict(list)
    for stratum, severity, r in choose_sample(census, args.per_cell, args.seed):
        row, crops = _case(r["key"], r["family"], 4, *r["gids"], stratum, severity, img)
        sample_rows.append(row)
        sheets[stratum].append((crops, _caption(len(sample_rows), row)))
    controls = []
    for fam in ("coord", "function", "polygon"):
        cand = [r for r in scanned if r["family"] == fam]
        if cand:
            controls.append(("vicini", min(cand, key=lambda r: (r["separation_px"], r["key"]))))
    taken = {r["key"] for _, r in controls}
    rng = random.Random(f"{args.seed}/controlli")
    for fam in ("coord", "polygon"):
        cand = sorted((r for r in scanned if r["family"] == fam and r["key"] not in taken), key=lambda r: r["key"])
        if cand:
            controls.append(("a_caso", rng.choice(cand)))
    for how, r in controls:
        row, crops = _case(r["key"], r["family"], 4, *r["gids"], "controllo", how, img)
        row["separation_px"] = r["separation_px"]
        sample_rows.append(row)
        sheets["controllo"].append((crops, _caption(len(sample_rows), row)))
    for stratum, rows in sorted(sheets.items()):
        _sheet(rows, img / f"campione-{stratum.replace('/', '-')}.png")
    print(f"campione: {len(sample_rows)} casi ({time.time() - t0:.0f} s)", flush=True)

    pairs, label_cells, examples = analyze_pairs(args.pairs_shard)
    for k in range(0, len(label_cells), 25):
        _grid_sheet(label_cells[k:k + 25], 5, img / f"coppie-misure-{k // 25 + 1}.png")
    ex_rows = [cell for cls in sorted(examples) for cell in examples[cls]]
    for k in range(0, len(ex_rows), 6):
        _sheet(ex_rows[k:k + 6], img / f"coppie-esempi-{k // 6 + 1}.png")
    print(f"coppie: {len(pairs)} ({time.time() - t0:.0f} s)", flush=True)

    coords, coord_cells = analyze_coords(sorted(coord_questions, key=lambda r: (r["origin"] != "generato", r["key"], r["q"])))
    if coord_cells:
        _sheet(coord_cells, img / "coords.png")
    print(f"coordinate: {len(coords)} domande ({time.time() - t0:.0f} s)", flush=True)

    _write_jsonl(args.out / "overlaps.jsonl", census)
    _write_jsonl(args.out / "sample.jsonl", sample_rows)
    _write_jsonl(args.out / "pairs.jsonl", pairs)
    _write_jsonl(args.out / "coords.jsonl", coords)
    summary = {
        "schema": "synth-review/1", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(ROOT),
        "args": {k: str(v) for k, v in vars(args).items()},
        "inputs": [{"path": str(p), "sha256": sha256_file(p)} for p in (problems_path, report_path, args.pairs_shard)],
        "thresholds": {"LIGHT_PX": LIGHT_PX, "INK": INK, "CORE": CORE, "TOUCH_PX": TOUCH_PX, "ORIGIN": sorted(ORIGIN)},
        "census": {"scenes": len(flagged), "violations": len(census), "matches_report": got == want,
                   "by_family_kind": _flat(got), "by_category": _flat(Counter((r["family"], r["category"]) for r in census)),
                   "by_category_ink": _flat(Counter((r["family"], r["category"], r["ink"]) for r in census)),
                   "by_stratum_severity_ink": _flat(Counter((r["stratum"], r["severity"], r["ink"]) for r in census))},
        "sample": {"cases": len(sample_rows), "by_stratum_severity": _flat(Counter((r["stratum"], r["severity"])
                                                                                   for r in sample_rows)),
                   "render_identical": sum(r["render_identical"] for r in sample_rows)},
        "pairs": {"n": len(pairs), "by_class": _flat(Counter(r["class"] for r in pairs)),
                  "by_kind_class": _flat(Counter((r["kind"], r["class"]) for r in pairs)),
                  "by_printed_class": _flat(Counter((r["printed_sides"], r["class"]) for r in pairs)),
                  "answer_matches_printed": sum(r["answer_matches_printed"] for r in pairs),
                  "only_lengths_differ": sum(r["only_lengths_differ"] for r in pairs),
                  "png_identical": sum(r["png_identical"] for r in pairs),
                  "a_feasible": sum(r["feasible"]["a_larga"] and r["feasible"]["a_stretta"] for r in pairs),
                  "length_incoherent": sum(bool(r["length_incoherent"]) for r in pairs),
                  "changed_label_overlaps": sum(bool(r["changed_label_overlaps"]) for r in pairs)},
        "coords": {"by_version_class": _flat(Counter((r["version"], r["class"]) for r in coords)),
                   "line_test": _flat(Counter((r["version"], r["line_test"]) for r in coords)),
                   "by_version_R_step": _flat(Counter((r["version"], r["R"], r["step"]) for r in coords))},
        "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("census", "sample", "pairs", "coords")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
