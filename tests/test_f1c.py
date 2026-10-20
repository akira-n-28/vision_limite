#!/usr/bin/env python3
"""Test dei dati e della valutazione di F1c: generatore sintetico (limite_vl.synth), domande MAVIS e testi visibili
(limite_vl.f1c), nuovi loader di data.py con il processor di Violetto, padding a sinistra (limite_vl.generation)."""
from __future__ import annotations

import io
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import data, synth  # noqa: E402
from limite_vl.f1c import (array_rows, mavis_count_caption, mavis_questions, pick_questions, qa_messages,  # noqa: E402
                           split_lines, visible_caption)
from limite_vl.generation import left_pad  # noqa: E402
from limite_vl.processing import EOS_ID, IGNORE, Example, LimiteVLProcessor  # noqa: E402

VIOLETTO = ROOT / "hf/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_synth():
    for fam in synth.FAMILIES:
        specs = [synth.make_spec(f"t-{fam}-{i}", fam) for i in range(40)]
        qs = [synth.questions(s) for s in specs]
        types = {q["type"] for qq in qs for q in qq}
        check(f"sintetici {fam}: domande con risposta fra le opzioni", all(q["answer"] in q["options"] for qq in qs for q in qq))
        check(f"sintetici {fam}: deterministici", synth.make_spec(f"t-{fam}-3", fam) == specs[3]
              and synth.questions(specs[3]) == qs[3])
        check(f"sintetici {fam}: tipi di domanda", len(types) >= 4, sorted(types))
        exists = [q["answer"] for qq in qs for q in qq if q["type"] == "exists"]
        check(f"sintetici {fam}: esistenza equilibrata", exists.count("yes") == exists.count("no"))
        image = synth.render(specs[0])
        check(f"sintetici {fam}: dimensione esatta", image.size == (specs[0]["width"], specs[0]["height"])
              and specs[0]["width"] % 32 == 0 and specs[0]["height"] % 32 == 0, image.size)
        arr = np.asarray(image)
        check(f"sintetici {fam}: c'è qualcosa disegnato", (arr < 100).mean() > 0.002, round(float((arr < 100).mean()), 4))
        before = deepcopy(specs)
        cfs = [synth.counterfactual(s) for s in specs]
        ok = [c for c in cfs if c is not None and c[3] != c[4]]
        check(f"sintetici {fam}: controfattuali con risposta diversa", len(ok) >= 20, len(ok))
        check(f"sintetici {fam}: controfattuale non tocca la scena", specs == before)
    bad = [q for i in range(300) for s in [synth.make_spec(f"t-shp-{i}", "shapes")]
           if any(x["type"] == "square" for x in s["shapes"]) for q in synth.questions(s)
           if "rectangle" in q["q"] and (q["type"] == "count_type" or q["answer"] == "no")]
    check("shapes: niente domande sul rettangolo se c'è un quadrato", not bad, bad[:2])
    iso = [s for i in range(300) if (s := synth.make_spec(f"t-iso-{i}", "polygon"))["kind"] == "isosceles triangle"]
    sides = [sorted(float(x) for x in s["lengths"]) for s in iso]
    check("polygon: isosceli non quasi equilateri", sides and all(abs(v[1] - v[0]) / v[0] > 0.1 or abs(v[2] - v[1]) / v[1] > 0.1
                                                                   for v in sides), len(iso))
    s = synth.make_spec("t-coord-quadrant", "coord")
    for p in s["points"]:
        q = next(q for q in synth.questions(s) if q["q"] == f"In which quadrant is point {p['label']}?")
        exp = {(True, True): "first", (False, True): "second", (False, False): "third", (True, False): "fourth"}[(p["x"] > 0, p["y"] > 0)]
        check(f"coord: quadrante di {p['label']} ({p['x']}, {p['y']})", q["answer"] == exp)
    sq = next(s for i in range(200) if (s := synth.make_spec(f"t-sq-{i}", "polygon"))["kind"] == "square")
    check("polygon: lati del quadrato uguali e interi", len(set(sq["lengths"])) == 1 and sq["lengths"][0].isdigit(), sq["lengths"])
    rt = next(s for i in range(200) if (s := synth.make_spec(f"t-rt-{i}", "polygon"))["kind"] == "right triangle")
    i = rt["right"]
    pts = rt["points"]
    v1 = np.subtract(pts[i - 1], pts[i])
    v2 = np.subtract(pts[(i + 1) % 3], pts[i])
    check("polygon: angolo retto nel vertice dichiarato", abs(float(np.dot(v1, v2))) < 1e-6)


REF = ("ABCED is a rectangle, but the side CD is extended into an equilateral triangle outside. Besides, ECF takes on a "
       "sector shape. Angle BAD has a measure of 90 degrees. Angle FCE is measured at 60 degrees. Side AB is 4 units "
       "long. Side AD measures 7 units in length. The segment EC spans 4 units.")


def test_mavis():
    qs = {q["type"] + ":" + q["q"]: q for q in mavis_questions(REF, 1, "k")}
    by_type = {}
    for q in qs.values():
        by_type.setdefault(q["type"], []).append(q)
    check("MAVIS: niente conteggio con un lato prolungato", "count" not in by_type)
    plain = REF.replace("ABCED is a rectangle, but the side CD is extended into an equilateral triangle outside.", "ABCED is a rectangle.")
    counts = [q for q in mavis_questions(plain, 1, "k") if q["type"] == "count"]
    check("MAVIS: conteggio 2 senza prolungamenti", counts and counts[0]["answer"] == "2", counts)
    types = {q["answer"] for q in mavis_questions(plain, 1, "k") if q["type"] == "shape_type"}
    check("MAVIS: tipi senza prolungamenti", types == {"rectangle", "sector"}, types)
    rad = mavis_questions("ABC is a right triangle. Angle ACB is 60 degrees. The length of segment AC is 10*√(3)/3.", 0, "k")
    side = [q for q in rad if q["type"] == "side_len"]
    check("MAVIS: radice con denominatore", side and side[0]["answer"] == "10*√(3)/3", side)
    check("MAVIS: niente tipo per la figura con un lato prolungato", {q["answer"] for q in by_type["shape_type"]} == {"sector"},
          by_type["shape_type"])
    check("MAVIS: lato condiviso", by_type["shared_side"][0]["answer"] == "CE", by_type.get("shared_side"))
    check("MAVIS: angoli in gradi", {q["answer"] for q in by_type["angle_val"]} == {"90°", "60°"})
    fake = [q for q in by_type["exists"] if q["answer"] == "no"]
    check("MAVIS: figura inventata nello stile di MAVIS", fake and fake[0]["q"].endswith("FCGH?"), fake)
    check("MAVIS: niente conteggio se la profondità non torna", not any(q["type"] == "count" for q in mavis_questions(REF, 2, "k")))
    check("MAVIS: didascalia con conteggio", mavis_count_caption(plain, 1).startswith(
        "The figure contains 2 shapes: ABCED (rectangle) and ECF (sector). ABCED")
          and mavis_count_caption(REF, 1) == REF)
    p, a = qa_messages(pick_questions("k3", list(qs.values())))
    check("domande: formato", p.startswith("Answer") and a, (p, a))


STEM = """### Image 1: Geometric Diagram
**Scene Summary:** A geometry problem diagram featuring a circle with a tangent line and a secant line, accompanied by a fill-in-the-blank question to determine the unknown length $x$.

**Explicit Component Breakdown:**
* **Circle (`None`):** A circular shape with no visible label.
* **Instruction Text (`Find the value of x.`):** The instruction to solve for $x$.
* **Secant External Segment (`13`):** The portion of the secant line outside the circle, labeled $13$.

**Interactions and Relationships:**
* The secant line passes through the circle. The tangent forms a right angle with the radius (implied by geometric conventions).

**Implicit and Inferred Properties:**
* **Geometric Configuration:** The diagram represents a tangent-secant theorem scenario.

**Identified Ambiguities:**
* None."""


def test_stem():
    v = visible_caption(STEM)
    check("STEM: niente titoli, domanda, deduzioni", not any(w in v for w in ("###", "**Scene", "question", "solve",
                                                                         "theorem", "implied", "Ambigu", "Inferred")), v)
    check("STEM: resta il contenuto visibile", all(w in v for w in ("circle with a tangent", "Circle", "labeled $13$",
                                                                  "passes through the circle")), v)


def test_lines():
    rows = array_rows(r"\begin{array}{rl}{a+b}&{=c}\\{d}&{=\frac{1}{2}}\end{array}")
    check("array: righe", rows == [r"{a+b}{=c}", r"{d}{=\frac{1}{2}}"], rows)
    check("array: una riga sola -> None", array_rows(r"\begin{array}{l}x\end{array}") is None)
    check("array: niente array -> None", array_rows(r"x^2") is None)
    # riga 1 con frazione (numeratore, linea, denominatore separati da 2 righe bianche), poi 15 bianche, riga 2
    ink = [False] * 3 + [True] * 8 + [False] * 2 + [True] * 2 + [False] * 2 + [True] * 8 + [False] * 15 + [True] * 18 + [False] * 3
    check("righe di testo: frazione unita, due righe", split_lines(ink, 2) == [[3, 25], [40, 58]], split_lines(ink, 2))
    check("righe di testo: tre righe chieste, spazi non netti -> None", split_lines(ink, 3) is None)
    flat = [True] * 10 + [False] * 3 + [True] * 10 + [False] * 3 + [True] * 10
    check("righe di testo: spazi uguali, due righe chieste -> None", split_lines(flat, 2) is None)


def test_loaders(proc):
    spec = synth.make_spec("t-loader", "coord")
    meta = {"id": "t-loader", "family": "coord", "width": spec["width"], "height": spec["height"],
            "max_patches": spec["width"] * spec["height"] // 256, "qa": synth.questions(spec)}
    files = {"png": synth.png_bytes(synth.render(spec)), "json": json.dumps(meta).encode()}
    ex = data.build_example(proc, "synth_qa", "t-loader", files)
    labelled = proc.tokenizer.decode(ex.labels[ex.labels != IGNORE].tolist())
    sample = data.synth_qa_sample("t-loader", files)
    check("synth_qa: label = risposta + EOS", labelled == sample.messages[1]["content"] + "<|endoftext|>", labelled)
    check("synth_qa: patch esatte", sum(p.shape[0] for p in ex.patches) == meta["max_patches"])
    mav = {"variant": "caption", "depth": 1, "max_patches": 2304, "text_en": {"caption": REF}}
    files = {"png": png(Image.new("RGB", (768, 768), "white")), "json": json.dumps(mav).encode()}
    for kind in ("mavis_qa", "mavis_count_caption"):
        ex = data.build_example(proc, kind, "d1-7", files)
        check(f"{kind}: esempio con label", int((ex.labels != IGNORE).sum()) > 1)
    st = {"caption": STEM}
    files = {"png": png(Image.new("RGB", (300, 200), "white")), "json": json.dumps(st).encode()}
    ex = data.build_example(proc, "stem_visible", "sft-x-1", files)
    text = proc.tokenizer.decode(ex.labels[ex.labels != IGNORE].tolist())
    check("stem_visible: target pulito", "theorem" not in text and "Circle" in text, text[:120])
    img = Image.new("L", (640, 192), 255)
    um = {"rows": ["{a+b}{=c}", "{d}{=e}"], "lines": [[20, 80], [110, 170]], "max_patches": 480, "height": 192}
    files = {"png": png(img), "json": json.dumps(um).encode()}
    s = data.unimer_line_sample("train-0000001", files)
    check("unimer_line: ritaglio di una riga", s.image.height < 192 and s.messages[1]["content"] in um["rows"], s.image.size)
    s = data.unimer_line_select_sample("train-0000001", files)
    check("unimer_line_select: richiesta della riga", "line" in s.messages[0]["content"][1]["text"]
          and s.messages[1]["content"] in um["rows"])


def test_left_pad():
    def ex(n, n_img, seed):
        rng = np.random.default_rng(seed)
        patches = [rng.integers(0, 256, (16, 768), dtype=np.uint8) for _ in range(n_img)]
        ids = rng.integers(0, 1000, n).astype(np.int32)
        return Example(ids, ids.copy(), patches, [(4, 4)] * n_img)
    a, b = ex(5, 1, 0), ex(9, 2, 1)
    batch = left_pad([a, b])
    check("padding a sinistra: forme", tuple(batch["input_ids"].shape) == (2, 9) and batch["attention_mask"][0].tolist()
          == [0, 0, 0, 0, 1, 1, 1, 1, 1])
    check("padding a sinistra: pad = EOS e token in coda", batch["input_ids"][0, 0].item() == EOS_ID
          and batch["input_ids"][0, 4:].tolist() == a.input_ids.tolist())
    check("padding a sinistra: immagini nell'ordine", tuple(batch["pixel_values"].shape) == (3, 16, 768)
          and bool((batch["pixel_values"][0].numpy() == a.patches[0]).all())
          and bool((batch["pixel_values"][2].numpy() == b.patches[1]).all()))


if __name__ == "__main__":
    test_synth()
    test_mavis()
    test_stem()
    test_lines()
    test_loaders(LimiteVLProcessor.from_pretrained(str(VIOLETTO)))
    test_left_pad()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
