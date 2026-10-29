#!/usr/bin/env python3
"""Test del generatore sintetico v4 e della sua verifica indipendente (limite_vl.synth, limite_vl.synth_check), mirati
ai difetti trovati in v3 (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §§2, 5).

Le parti che disegnano usano matplotlib e leggono gli shard v3: si eseguono in un job (slurm/tests_cpu.sbatch,
slurm/synth_semantics_check.sbatch), mai sul frontend."""
from __future__ import annotations

import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import synth, synth_check  # noqa: E402
from limite_vl.data import iter_shard  # noqa: E402

V3 = ROOT / "data/shards/synth/v3"
CHANGED = {"intersections", "shape_type", "label_type"}  # tipi di domanda che cambiano da v3 a v4
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def curve(kind: str, name: str, **kw) -> dict:
    return {"kind": kind, "name": name, **kw}


def fake_items(spec: dict) -> dict:
    """Layout dei grafici come lo produrrebbe il disegno (600 campioni in [-V, V], tratti con |y| > V tolti), per
    collaudare synth_check senza matplotlib. Nel job il layout vero viene da synth.render."""
    V = spec["R"] + 1
    xs = [-V + 2 * V * i / 599 for i in range(600)]
    items = {}
    for c in spec["curves"]:
        runs, cur = [], None
        for x in xs:
            if abs(synth._curve_value(c, x)) <= V:
                cur = [x, x] if cur is None else [cur[0], x]
            elif cur is not None:
                runs.append(cur)
                cur = None
        if cur is not None:
            runs.append(cur)
        items[c["name"]] = {"visible": runs, "x_range": [-V, V], "clip_y": float(V), "samples": len(xs)}
    return items


def first(keys, predicate):
    """Il primo valore che soddisfa predicate (ciclo esplicito: niente := negli iterabili delle comprehension)."""
    for value in keys:
        if predicate(value):
            return value
    raise AssertionError("nessun valore trovato")


def test_v3_reproduction():
    man = json.loads((V3 / "manifest.json").read_text())
    counts = Counter()
    for i in range(int(man["args"]["train"])):
        key, fam = f"synth-{i:07d}", synth.FAMILIES[i % len(synth.FAMILIES)]
        counts[f"family/{fam}"] += 1
        for q in synth.questions(synth.make_spec(key, fam, 3)):
            counts[f"qa/{fam}/{q['type']}"] += 1
            counts[f"answer/{q['type']}/{q['answer']}"] += 1
    check("v3: istogrammi delle 60.000 scene identici al manifest del job 29756", counts == Counter(man["counts"]["train"]))
    for k, (key, files) in enumerate(iter_shard(V3 / "train" / "synth-train-00000.tar")):
        if k >= 8:
            break
        meta = json.loads(files["json"])
        spec = synth.make_spec(key, meta["family"], 3)
        check(f"v3: specifica e domande di {key} identiche allo shard", spec == meta["spec"] and synth.questions(spec) == meta["qa"])
        check(f"v3: PNG di {key} identico allo shard (i gid non cambiano i pixel)",
              synth.png_bytes(synth.render(spec)) == files["png"])


def test_intersections():
    # audit-68: conteggio geometrico 1, ma a 1/3 dal bordo V = 7, dentro il margine: nessuna domanda in v4
    s = synth.make_spec("audit-68", "function", 4)
    V = synth.visible_bound(s)
    drawn = [p for p in synth.intersection_points(*s["curves"]) if abs(p[0]) <= V and abs(p[1]) <= V]
    check("audit-68: conteggio geometrico nel tratto disegnato = 1", len(drawn) == 1, drawn)
    check("audit-68: v4 non pone la domanda (a 1/3 dal bordo)",
          not [q for q in synth.questions(s) if q["type"] == "intersections"])
    count, why = synth_check.visible_intersections(*s["curves"], fake_items(s))
    check("audit-68: la verifica dà conteggio 1 e non ammissibile", count == 1 and why, why)
    q3 = [q["answer"] for q in synth.questions(synth.make_spec("audit-68", "function", 3)) if q["type"] == "intersections"]
    check("audit-68: v3 riprodotta, domanda con target 0", q3 == ["0"], q3)
    # caso reale lontano dal bordo: domanda ammessa con target 1 in v4, 0 in v3
    s4, s3 = synth.make_spec("synth-0049798", "function", 4), synth.make_spec("synth-0049798", "function", 3)
    a4 = [q["answer"] for q in synth.questions(s4) if q["type"] == "intersections"]
    a3 = [q["answer"] for q in synth.questions(s3) if q["type"] == "intersections"]
    check("synth-0049798: target 1 in v4, 0 in v3", a4 == ["1"] and a3 == ["0"], (a4, a3))
    count, why = synth_check.visible_intersections(*s4["curves"], fake_items(s4))
    check("synth-0049798: la verifica dà conteggio 1, ammissibile", count == 1 and not why, why)
    # casi costruiti
    f, g = curve("line", "f", a=1, b=-6), curve("line", "g", a=-1, b=7)  # (6.5, 0.5)
    check("y = x - 6, y = -x + 7: v3 (bordo 6) conta 0, v4 (bordo 7) conta 1",
          synth._intersections(f, g, 6) == 0 and synth._intersections(f, g, 7) == 1)
    f, g = curve("line", "f", a=1, b=7.45), curve("line", "g", a=-1, b=7.45)  # (0, 7.45)
    check("intersezione con |y| = 7.45: non disegnata, conteggio 0", synth._intersections(f, g, 7) == 0)
    f, g = curve("line", "f", a=1, b=6.8), curve("line", "g", a=-1, b=6.8)  # (0, 6.8)
    check("intersezione a 0.2 dal bordo: nessuna domanda", synth._intersections(f, g, 7) is None)
    tangent = (curve("parabola", "f", a=1, h=0, k=0), curve("line", "g", a=0, b=0))
    close = (curve("parabola", "f", a=1, h=0, k=0), curve("line", "g", a=0, b=0.04))
    parallel = (curve("line", "f", a=1, b=1), curve("line", "g", a=1, b=3))
    check("tangenza, radici vicine, rette parallele: nessuna domanda",
          all(synth._intersections(*p, 7) is None for p in (tangent, close, parallel)))
    check("tangenza: conteggio geometrico 1 (punto doppio)", len(synth.intersection_points(*tangent)) == 1)
    near = (curve("parabola", "f", a=1, h=0, k=0.2), curve("line", "g", a=0, b=0))  # minimo 0.2, nessun incrocio
    far = (curve("parabola", "f", a=1, h=0, k=1), curve("line", "g", a=0, b=0))  # minimo 1
    check("grafici a 0.2 senza incrociarsi: v4 non pone la domanda, v3 sì (target 0)",
          synth._near_miss(*near, 7) and synth._intersections(*near, 6) == 0 and not synth._near_miss(*far, 7))
    check("grafici a 0.2 senza incrociarsi: la verifica li segnala", bool(synth_check.visible_intersections(
        *near, fake_items({"R": 6, "curves": list(near)}))[1]))
    # tratti visibili (revisione di 38af82a): l'incrocio in un buco del disegno non si conta
    f, g = curve("line", "f", a=1, b=1), curve("line", "g", a=-1, b=1)  # incrocio in (0, 1)
    gap = {"x_range": [-7, 7], "clip_y": 7, "visible": [[-7, -2], [2, 7]]}
    check("incrocio in x = 0 dentro un buco dei tratti visibili: conteggio 0, ammissibile (prima 1)",
          synth_check.visible_intersections(f, g, {"f": dict(gap), "g": dict(gap)}) == (0, []))
    small = {"x_range": [-7, 7], "clip_y": 7, "visible": [[-7, -0.2], [0.2, 7]]}
    count, why = synth_check.visible_intersections(f, g, {"f": dict(small), "g": dict(small)})
    check("incrocio a 0.2 da un buco del disegno: conteggio 0, non ammissibile", count == 0 and bool(why), why)
    whole = {"x_range": [-7, 7], "clip_y": 7, "visible": [[-7, 7]]}
    check("stessi grafici disegnati per intero: conteggio 1, ammissibile",
          synth_check.visible_intersections(f, g, {"f": dict(whole), "g": dict(whole)}) == (1, []))
    edge = {"x_range": [-7, 7], "clip_y": 7, "samples": 600, "visible": [[-7, 0.01], [0.6, 7]]}
    count, why = synth_check.visible_intersections(f, g, {"f": dict(edge), "g": dict(edge)})
    check("incrocio a 0.01 da un buco del disegno: conteggio 1 ma non ammissibile", count == 1 and bool(why), why)
    steep = (curve("line", "f", a=100, b=1), curve("line", "g", a=0, b=6.5))  # incrocio (0.055, 6.5); f esce a x = 0.06
    count, why = synth_check.visible_intersections(*steep, fake_items({"R": 6, "curves": list(steep)}))
    check("incrocio entro un passo di campionamento dal taglio di un grafico ripido: non ammissibile",
          count == 1 and any("campionamento" in w for w in why), why)
    # su 4.000 scene v4 la verifica indipendente conferma ogni domanda posta
    bad, asked = [], 0
    for i in range(4000):
        s = synth.make_spec(f"t-int-{i}", "function", 4)
        q = [x for x in synth.questions(s) if x["type"] == "intersections"]
        if not q:
            continue
        asked += 1
        count, why = synth_check.visible_intersections(*s["curves"], fake_items(s))
        if why or str(count) != q[0]["answer"]:
            bad.append((s["key"], q[0]["answer"], count, why))
    check("v4: ogni domanda intersections confermata dalla verifica indipendente", asked > 500 and not bad, (asked, bad[:3]))


def test_classification():
    bad = []
    for i in range(4000):
        for version in (3, 4):
            s = synth.make_spec(f"t-cls-{i}", "polygon", version)
            for q in synth.questions(s):
                if q["type"] != "shape_type":
                    continue
                pts = [tuple(p) for p in s["points"]]
                tri = len(pts) == 3
                true = synth_check.triangle_classes(pts) if tri else synth_check.quad_classes(pts)
                status, expected, why = synth_check._classify(q["q"], q["answer"], true,
                                                               synth_check.TRIANGLE_VOCAB if tri else synth_check.QUAD_VOCAB)
                ok = status == "ok" if version == 4 else (status == "non_ammissibile" or s["kind"] == "trapezoid")
                if not ok:
                    bad.append((version, s["kind"], q["q"], status, why))
    check("shape_type: v4 una sola risposta vera; v3 aperta con più risposte vere (tranne il trapezio)", not bad, bad[:3])
    sq = next(s for i in range(400) if (s := synth.make_spec(f"t-sq-{i}", "polygon", 4))["kind"] == "square")
    q = next(q for q in synth.questions(sq) if q["type"] == "shape_type")
    true = synth_check.quad_classes([tuple(p) for p in sq["points"]])
    check("quadrato: la domanda mostra le opzioni e chiede la classe più specifica",
          "most specific" in q["q"] and all(o in q["q"] for o in q["options"]), q["q"])
    check("quadrato: solo 'square' è giusta, 'rectangle' è errata",
          synth_check._classify(q["q"], "square", true, synth_check.QUAD_VOCAB)[0] == "ok"
          and synth_check._classify(q["q"], "rectangle", true, synth_check.QUAD_VOCAB)[0] == "errata")
    rt = next(s for i in range(400) if (s := synth.make_spec(f"t-rt-{i}", "polygon", 4))["kind"] == "right triangle")
    q = next(q for q in synth.questions(rt) if q["type"] == "shape_type")
    check("triangolo rettangolo: opzioni nel testo, è anche scaleno ma 'scalene' non è fra le opzioni",
          all(o in q["q"] for o in q["options"]) and "scalene triangle" in synth_check.triangle_classes(
              [tuple(p) for p in rt["points"]]) and "scalene" not in q["q"])
    labels = [q for i in range(400) for s in [synth.make_spec(f"t-lt-{i}", "shapes", 4)] for q in synth.questions(s)
              if q["type"] == "label_type"]
    check("label_type v4: opzioni e classe più specifica nel testo", labels and all(
        "most specific" in q["q"] and all(o in q["q"] for o in q["options"]) for q in labels), len(labels))
    check("label_type v4: un quadrato chiede 'square', 'rectangle' è errata",
          synth_check._classify(next(q["q"] for q in labels if q["answer"] == "square"), "rectangle",
                                {"square", "rectangle"}, synth_check.SHAPE_VOCAB)[0] == "errata")


def test_v3_v4_regression():
    same_specs = differ_qa = resampled = 0
    for i in range(8000):
        key, fam = f"synth-{i:07d}", synth.FAMILIES[i % len(synth.FAMILIES)]
        s3, s4 = synth.make_spec(key, fam, 3), synth.make_spec(key, fam, 4)
        base = {k: v for k, v in s4.items() if k != "generator_version"}
        if base != s3:
            resampled += 1
            if fam != "shapes" or not (synth._shapes_overlap(s3["shapes"]) or not synth._shapes_fit(s3)):
                differ_qa += 1  # una scena cambiata senza motivo
            continue
        same_specs += 1
        q3, q4 = synth.questions(s3), synth.questions(s4)
        keep3 = [q for q in q3 if q["type"] not in CHANGED]
        keep4 = [q for q in q4 if q["type"] not in CHANGED]
        c3 = [(q["type"], q["answer"], q["options"]) for q in q3 if q["type"] in CHANGED - {"intersections"}]
        c4 = [(q["type"], q["answer"], q["options"]) for q in q4 if q["type"] in CHANGED - {"intersections"}]
        if keep3 != keep4 or c3 != c4:
            differ_qa += 1
    expected = sum(1 for i in range(3, 8000, 4) for s3 in [synth.make_spec(f"synth-{i:07d}", "shapes", 3)]
                   if synth._shapes_overlap(s3["shapes"]) or not synth._shapes_fit(s3))
    check("v3 -> v4: cambiano solo le scene shapes con figure vicine o fuori dagli assi", differ_qa == 0,
          (same_specs, resampled, differ_qa))
    check("v3 -> v4: riestratte esattamente le scene shapes da correggere, le altre identiche",
          resampled == expected and same_specs + resampled == 8000, (same_specs, resampled, expected))
    bad = Counter()
    for i in range(3, 60000, 4):
        s4 = synth.make_spec(f"synth-{i:07d}", "shapes", 4)
        bad["vicine"] += synth._shapes_overlap(s4["shapes"])
        bad["fuori"] += not synth._shapes_fit(s4)
    check("shapes v4: nessuna figura vicina a un'altra o fuori dagli assi nelle 15.000 scene", not +bad, dict(bad))


def test_clipping_generator():
    """synth-0000043: in v3 (e in v4 prima della correzione) il rettangolo L esce a sinistra dagli assi [0, 9]."""
    s3 = synth.make_spec("synth-0000043", "shapes", 3)
    rect = next(x for x in s3["shapes"] if x["label"] == "L")
    x0 = synth._shape_box(rect)[0]
    check("synth-0000043 v3: rettangolo L con x minimo -0.119, fuori dagli assi", abs(x0 + 0.118847) < 1e-5
          and not synth._shapes_fit(s3), round(x0, 6))
    s4 = synth.make_spec("synth-0000043", "shapes", 4)
    check("synth-0000043 v4: scena riestratta, tutte le figure dentro gli assi col tratto",
          {k: v for k, v in s4.items() if k != "generator_version"} != s3 and synth._shapes_fit(s4)
          and not synth._shapes_overlap(s4["shapes"]))
    m = synth._stroke_margin(s3)
    check("margine del tratto: mezzo tratto più 1 px in unità del disegno", abs(m - (1.4 * 100 / 72 / 2 + 1) * 9 / 480) < 1e-12, m)


def rendered(key: str, family: str, version: int):
    spec = synth.make_spec(key, family, version)
    image, layout = synth.render(spec, layout=True)
    return spec, image, layout


def test_rendered_checks():
    """Con il layout vero: le domande v4 risultano ok, le risposte alterate no, i difetti noti di v3 si vedono."""
    for fam in synth.FAMILIES:
        statuses, mutated = Counter(), Counter()
        for i in range(40):
            spec, _, layout = rendered(f"t-render-{fam}-{i}", fam, 4)
            qs = synth.questions(spec)
            for r in synth_check.check_questions(spec, qs, layout):
                statuses[r["status"]] += 1
            wrong = [{**q, "answer": next(o for o in q["options"] if o != q["answer"])} for q in qs if len(q["options"]) > 1]
            for r in synth_check.check_questions(spec, wrong, layout):
                mutated[r["status"]] += 1
        check(f"{fam} v4: nessuna domanda errata o non verificabile", not statuses["errata"] and not statuses["non_verificabile"],
              dict(statuses))
        check(f"{fam} v4: le risposte alterate non risultano mai ok", not mutated["ok"], dict(mutated))
    spec, _, layout = rendered("audit-68", "function", 3)
    r = synth_check.check_questions(spec, [q for q in synth.questions(spec) if q["type"] == "intersections"], layout)
    check("audit-68 v3 disegnato: target 0 non ammissibile, conteggio 1", [(x["status"], x["expected"]) for x in r]
          == [("non_ammissibile", "1")], r)
    spec, _, layout = rendered("synth-0049798", "function", 3)
    r = synth_check.check_questions(spec, [q for q in synth.questions(spec) if q["type"] == "intersections"], layout)
    check("synth-0049798 v3 disegnato: target 0 errato, si vede 1", [(x["status"], x["expected"]) for x in r]
          == [("errata", "1")], r)
    sq = first((synth.make_spec(f"t-sql-{i}", "shapes", 3) for i in range(200)),
               lambda s: any(x["type"] == "square" for x in s["shapes"]))
    _, layout = synth.render(sq, layout=True)
    r = [x for x in synth_check.check_questions(sq, synth.questions(sq), layout) if x["type"] == "label_type" and x["answer"] == "square"]
    check("label_type v3 su un quadrato: domanda aperta non ammissibile", r and all(x["status"] == "non_ammissibile" for x in r), r[:1])


def test_clipping_rendered():
    """Il verificatore trova il taglio rispetto all'area effettiva degli assi e non promuove la scena; la scena v4 dopo
    la correzione è pulita."""
    s3 = synth.make_spec("synth-0000043", "shapes", 3)
    _, layout = synth.render(s3, layout=True)
    cut = [v for v in synth_check.renderer_violations(s3, layout) if v["kind"] == "figura_tagliata"]
    check("synth-0000043 v3 disegnato: rettangolo L tagliato di più di 1 px dagli assi",
          [v["gids"] for v in cut] == [["shape:L"]] and cut[0]["detail"] > 5, cut)
    meta = {"spec": s3, "qa": synth.questions(s3)}
    res = synth_check.check_sample(meta, layout)
    check("synth-0000043 v3: campione non verificato, nessuna domanda ok", not res["verified"]
          and res["blocking"] == ["figura_tagliata"] and all(r["status"] != "ok" for r in res["questions"]),
          (res["blocking"], Counter(r["status"] for r in res["questions"])))
    s4 = synth.make_spec("synth-0000043", "shapes", 4)
    _, layout = synth.render(s4, layout=True)
    res = synth_check.check_sample({"spec": s4, "qa": synth.questions(s4)}, layout)
    check("synth-0000043 v4 disegnato: nessuna figura tagliata, domande ok", not [
        v for v in res["violations"] if v["kind"] == "figura_tagliata"] and not res["blocking"]
        and all(r["status"] == "ok" for r in res["questions"]), (res["violations"], res["questions"][:2]))
    check("area degli assi registrata nel layout", len(layout.get("axes_px", [])) == 4, layout.get("axes_px"))


def test_renderer_invariants():
    spec, _, layout = rendered("t-inv-coord", "coord", 4)
    base = synth_check.renderer_violations(spec, layout)
    kinds = Counter(v["kind"] for v in base)
    check("coord v4: nessun testo fuori dall'immagine", not kinds["testo_fuori"], dict(kinds))
    lay = deepcopy(layout)
    text = next(it for it in lay["items"] if it["kind"] == "text" and it["gid"].startswith("label:"))
    text["bbox"] = [-30.0, 5.0, -10.0, 25.0]
    check("mutazione: un'etichetta fuori dall'immagine viene trovata",
          any(v["kind"] == "testo_fuori" and text["gid"] in v["gids"] for v in synth_check.renderer_violations(spec, lay)))
    lay = deepcopy(layout)
    texts = [it for it in lay["items"] if it["kind"] == "text"]
    texts[1]["bbox"] = list(texts[0]["bbox"])
    check("mutazione: due testi sovrapposti vengono trovati",
          any(v["kind"] == "testi_sovrapposti" for v in synth_check.renderer_violations(spec, lay)))
    lay = deepcopy(layout)
    t = next(it for it in lay["items"] if it["kind"] == "text")
    t["bbox"] = [t["bbox"][0], t["bbox"][1], t["bbox"][2], t["bbox"][1] + 6.0]
    check("mutazione: un testo alto 6 px viene trovato",
          any(v["kind"] == "testo_piccolo" and t["gid"] in v["gids"] for v in synth_check.renderer_violations(spec, lay)))
    over = next(s for i in range(200) if len((s := synth.make_spec(f"t-ov-{i}", "shapes", 4))["shapes"]) >= 2)
    clean = synth_check.renderer_violations(over, synth.render(over, layout=True)[1])
    check("shapes v4: nessuna figura sovrapposta", not [v for v in clean if v["kind"] == "figure_sovrapposte"], clean)
    over["shapes"][1].update(cx=over["shapes"][0]["cx"] + 0.5, cy=over["shapes"][0]["cy"] + 0.3)  # costruita apposta
    _, layout = synth.render(over, layout=True)
    check("mutazione: due figure sovrapposte vengono trovate",
          any(v["kind"] == "figure_sovrapposte" for v in synth_check.renderer_violations(over, layout)))
    spec, _, layout = rendered("t-inv-poly", "polygon", 4)
    lay = deepcopy(layout)
    length = next((it for it in lay["items"] if it["gid"].startswith("length:")), None)
    if length:
        length["text"] = str(float(length["text"]) + 3)
        check("mutazione: una lunghezza scritta diversa dal disegno viene trovata",
              any(v["kind"] == "lunghezza_incoerente" for v in synth_check.renderer_violations(spec, lay)))


if __name__ == "__main__":
    test_v3_reproduction()
    test_intersections()
    test_classification()
    test_v3_v4_regression()
    test_clipping_generator()
    test_rendered_checks()
    test_clipping_rendered()
    test_renderer_invariants()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
