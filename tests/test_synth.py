#!/usr/bin/env python3
"""Test del generatore sintetico v4, della sua verifica indipendente (limite_vl.synth, limite_vl.synth_check) e degli
esiti per campione di scripts/synth_semantics_check.py, mirati ai difetti trovati in v3 e nelle revisioni
(docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §§2, 5). Comprendono le correzioni dopo l'analisi degli scarti (§2.5):
coppie polygon per riscalatura, punto segnato lontano dalle misure, tacche -1 senza numero, coordinate lette dalle
tacche vicine, limiti esatti delle contraddizioni di scripts/synth_review.py.

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


def same_questions(s3: dict, s4: dict, skip: set[str] = frozenset()) -> bool:
    """Le domande dei tipi non toccati da v4 sono identiche; quelle di classificazione hanno le stesse risposte."""
    q3, q4 = synth.questions(s3), synth.questions(s4)
    keep3 = [q for q in q3 if q["type"] not in CHANGED | skip]
    keep4 = [q for q in q4 if q["type"] not in CHANGED | skip]
    c3 = [(q["type"], q["answer"], q["options"]) for q in q3 if q["type"] in CHANGED - {"intersections"}]
    c4 = [(q["type"], q["answer"], q["options"]) for q in q4 if q["type"] in CHANGED - {"intersections"}]
    return keep3 == keep4 and c3 == c4


def test_v3_v4_regression():
    same_specs = differ_qa = resampled = moved = expected_moved = 0
    for i in range(8000):
        key, fam = f"synth-{i:07d}", synth.FAMILIES[i % len(synth.FAMILIES)]
        s3, s4 = synth.make_spec(key, fam, 3), synth.make_spec(key, fam, 4)
        base = {k: v for k, v in s4.items() if k != "generator_version"}
        if fam == "polygon" and s3["m"] is not None and s3["m"]["side"] in s3["shown"]:
            # da v4 il punto segnato lascia i lati con la misura scritta (S2 §2.5): cambia solo il suo lato, o si toglie
            expected_moved += 1
            moved += base != s3
            free = [j for j in range(len(s3["points"])) if j not in s3["shown"]]
            m3, m4 = s3["m"], s4["m"]
            ok_m = (m4 is None and not free) or (m4 is not None and m4["side"] in free
                                                  and {k: v for k, v in m4.items() if k != "side"}
                                                  == {k: v for k, v in m3.items() if k != "side"})
            rest_same = {k: v for k, v in base.items() if k != "m"} == {k: v for k, v in s3.items() if k != "m"}
            if not (ok_m and rest_same and same_questions(s3, s4, {"m_side", "exists"})):
                differ_qa += 1
            continue
        if base != s3:
            resampled += 1
            if fam != "shapes" or not (synth._shapes_overlap(s3["shapes"]) or not synth._shapes_fit(s3)):
                differ_qa += 1  # una scena cambiata senza motivo
            continue
        same_specs += 1
        if not same_questions(s3, s4):
            differ_qa += 1
    expected = sum(1 for i in range(3, 8000, 4) for s3 in [synth.make_spec(f"synth-{i:07d}", "shapes", 3)]
                   if synth._shapes_overlap(s3["shapes"]) or not synth._shapes_fit(s3))
    check("v3 -> v4: cambiano solo le shapes con figure vicine o fuori dagli assi e il punto segnato dei polygon",
          differ_qa == 0, (same_specs, resampled, moved, differ_qa))
    check("v3 -> v4: riestratte esattamente le shapes da correggere, spostati solo i punti sui lati con la misura,"
          " le altre scene identiche", resampled == expected and moved == expected_moved
          and same_specs + resampled + moved == 8000, (same_specs, resampled, expected, moved, expected_moved))
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


def load_script(name: str):
    """scripts/<name>.py come modulo, senza eseguirne main (scripts/ in coda al percorso per i moduli fratelli)."""
    import importlib.util

    if str(ROOT / "scripts") not in sys.path:
        sys.path.append(str(ROOT / "scripts"))
    found = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(found)
    found.loader.exec_module(mod)
    return mod


def raises(fn, exc=ValueError) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def test_sample_evidence():
    """scripts/synth_semantics_check.py (revisione di f7d32d3): per un record di shard la semantica della scena, la
    corrispondenza fra il rendering e il PNG dello shard e l'esito del campione restano distinti; il layout salvato
    conta solo col digest del PNG; una scena generata in memoria ha un'altra origine."""
    import hashlib
    import io
    import tarfile
    import tempfile

    sc, sr = load_script("synth_semantics_check"), load_script("synth_render")

    def scene(i):
        spec, image, layout = rendered(f"synth-{i:07d}", "shapes", 4)
        return spec, image, layout, {"id": spec["key"], "family": "shapes", "qa": synth.questions(spec), "spec": spec}

    # record come quelli di v3: id, famiglia, domande e specifica, senza layout
    spec, image, layout, meta = first((scene(i) for i in range(43, 400, 4)),
                                      lambda c: synth_check.check_sample(c[3], c[2])["verified"])
    key, png = spec["key"], synth.png_bytes(image)
    other = synth.png_bytes(synth.render(synth.make_spec(key + "-altra", "shapes", 4)))
    linked = {**deepcopy(layout), "png_sha256": hashlib.sha256(png).hexdigest()}

    def state(m, data):
        return sc._check(key, m, "shard", png=data)[2]

    def parts(s, *names):
        return tuple(s[n] for n in names)

    # PNG corrispondente
    c, p, s = sc._check(key, meta, "shard", png=png)
    check(f"{key}: record senza layout, PNG corrispondente: byte identici, semantica e campione verificati",
          parts(s, "image", "layout", "semantic", "outcome", "reasons") ==
          ("byte_identici", "ridisegnato_layout_assente", "verificata", "verificato", [])
          and not [x for x in p if x["what"] == "sample"], s)
    # PNG mancante
    _, p, s = sc._check(key, meta, "shard", png=None)
    rows = [x for x in p if x["what"] == "sample"]
    check("PNG mancante: semantica verificata ma campione non verificato, motivo nella riga del campione",
          parts(s, "image", "semantic", "outcome") == ("png_mancante", "verificata", "non_verificato")
          and len(rows) == 1 and rows[0]["reasons"] == ["immagine: png_mancante"], (s, rows))
    s = state({**meta, "layout": linked}, None)
    check("PNG mancante con layout e digest nel record: campione non verificato",
          parts(s, "image", "layout", "outcome") == ("png_mancante", "salvato_png_mancante", "non_verificato"), s)
    # PNG sostituito con un'altra immagine, o con un pixel cambiato
    s = state(meta, other)
    check("PNG di un'altra scena: immagine da verificare, campione non verificato con semantica verificata",
          other != png and parts(s, "image", "semantic", "outcome", "reasons") ==
          ("diversa_da_verificare", "verificata", "non_verificato", ["immagine: diversa_da_verificare"]), s)
    s = state({**meta, "layout": linked}, other)
    check("PNG sostituito, layout col digest dell'originale: digest diverso, ridisegno, campione non verificato",
          parts(s, "image", "layout", "outcome") == ("diversa_da_verificare", "ridisegnato_digest_diverso",
                                                     "non_verificato"), s)
    touched = image.copy()
    r, g, b = touched.getpixel((0, 0))
    touched.putpixel((0, 0), (255 - r, g, b))
    s = state(meta, synth.png_bytes(touched))
    check("PNG con un pixel cambiato: pixel diversi, campione non verificato",
          parts(s, "image", "outcome") == ("diversa_da_verificare", "non_verificato"), s)
    s = state(meta, b"non un PNG")
    check("file che non si decodifica come PNG: da verificare, campione non verificato",
          parts(s, "image", "outcome") == ("diversa_da_verificare", "non_verificato"), s)
    # layout nel record senza digest (record precedenti al digest): non prova il legame col PNG
    s = state({**meta, "layout": deepcopy(layout)}, png)
    check("layout senza digest, PNG corrispondente: si ridisegna, byte identici, campione verificato",
          parts(s, "image", "layout", "outcome") == ("byte_identici", "ridisegnato_senza_digest", "verificato"), s)
    s = state({**meta, "layout": deepcopy(layout)}, other)
    check("layout senza digest, PNG sostituito: il layout salvato da solo non promuove, campione non verificato",
          parts(s, "image", "layout", "outcome") == ("diversa_da_verificare", "ridisegnato_senza_digest",
                                                     "non_verificato"), s)
    s = state({**meta, "layout": {**deepcopy(layout), "items": []}}, png)
    check("layout senza digest e svuotato: non si usa, la semantica viene dal ridisegno",
          parts(s, "semantic", "outcome") == ("verificata", "verificato"), s)
    # layout con digest uguale al PNG: si usa quello del record
    s = state({**meta, "layout": linked}, png)
    check("layout con digest corrispondente: campione verificato col layout salvato",
          parts(s, "image", "layout", "outcome") == ("digest_corrispondente", "salvato_con_digest", "verificato"), s)
    s = state({**meta, "layout": {k: v for k, v in deepcopy(linked).items() if k != "axes_px"}}, png)
    check("digest corrispondente ma layout senza area degli assi: semantica sul layout salvato, non verificato",
          parts(s, "image", "semantic", "outcome") == ("digest_corrispondente", "non_verificata", "non_verificato")
          and any(x.startswith("violazioni bloccanti:") and "layout_senza_assi" in x for x in s["reasons"]), s)
    # byte diversi, pixel identici: confronto dei pixel decodificati, distinto da quello dei byte
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)
    recoded = buf.getvalue()
    s = state(meta, recoded)
    check("PNG ricodificato (byte diversi, pixel identici): pixel identici, campione verificato",
          recoded != png and parts(s, "image", "outcome") == ("pixel_identici", "verificato"), s)
    rgba, buf = image.convert("RGBA"), io.BytesIO()
    rgba.putpixel((0, 0), rgba.getpixel((0, 0))[:3] + (0,))
    rgba.save(buf, format="PNG")
    s = state(meta, buf.getvalue())
    check("stessi valori RGB ma un pixel trasparente: pixel diversi, campione non verificato",
          parts(s, "image", "outcome") == ("diversa_da_verificare", "non_verificato"), s)
    # scena generata in memoria: un'altra origine, nessun esito di campione degli shard
    c, p, s = sc._check(key, meta, "generato", layout=layout)
    check("scena in memoria: immagine in_memoria, esito di scena, nessun conteggio degli shard",
          parts(s, "image", "layout", "outcome") == ("in_memoria", "generato", "scena_verificata")
          and all(k.split("/")[1] == "generato" for k in c), s)
    check("scena in memoria senza layout o con un PNG, record di shard con un layout dato dal chiamante: rifiutati",
          raises(lambda: sc._check(key, meta, "generato")) and raises(lambda: sc._check(key, meta, "generato", png=png,
                                                                                        layout=layout))
          and raises(lambda: sc._check(key, meta, "shard", png=png, layout=layout)))
    summary = sc.summarize(sc.generate_task((0, 8, 4))["counts"])
    check("--generate: 8 scene contate solo come generate, conteggi coerenti",
          summary["total"] == {"shard": 0, "generato": 8} and not summary["incoherent"] and "shard" not in summary["image"],
          (summary["total"], summary["incoherent"]))
    # record scritti con le funzioni di scripts/synth_render.py, letti da shard_task
    [(_, png_r, meta_r)] = sr.scene((key, "shapes", 4))
    check("synth_render v4: layout con lo sha256 dei byte del PNG scritto, stesso PNG del rendering",
          meta_r["layout"]["png_sha256"] == hashlib.sha256(png_r).hexdigest() and png_r == png)
    check("synth_render v3: nessun layout nel JSON, come negli shard v3", sr._draw(synth.make_spec(key, "shapes", 3))[1] is None)
    records = [("caso-a", png, meta),       # senza layout, PNG corrispondente: verificato
               ("caso-b", None, meta),      # PNG mancante: non verificato
               ("caso-c", other, meta_r),   # PNG sostituito, digest diverso: non verificato
               ("caso-d", recoded, meta_r),  # digest diverso, pixel identici: verificato
               ("caso-e", png_r, meta_r)]   # come scritto da synth_render: digest corrispondente, verificato
    with tempfile.TemporaryDirectory(prefix="test_synth_") as d:
        path = Path(d) / "evidence.tar"
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
            for name, data, m in records:
                if data is not None:
                    sr.add_member(tar, f"{name}.png", data)
                sr.add_member(tar, f"{name}.json", json.dumps(m, ensure_ascii=False).encode())
        res = sc.shard_task(str(path))
        tar_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        case_rows = sc.save_cases([f"{key}:shapes:4"], Path(d) / "cases")
        saved = json.loads((Path(d) / "cases" / f"{key}-v4.json").read_text())
        saved_png = (Path(d) / "cases" / f"{key}-v4.png").read_bytes()
    summary = sc.summarize(res["counts"])
    rows = sorted((x["key"], x["reasons"]) for x in res["problems"] if x["what"] == "sample")
    check("shard: semantica verificata in 5 campioni su 5, campioni verificati 3",
          summary["semantic"]["shard"] == {"verificata": 5}
          and summary["sample"]["shard"] == {"verificato": 3, "non_verificato": 2}, summary["sample"])
    check("shard: stato dell'immagine e provenienza del layout per campione",
          summary["image"]["shard"] == {"byte_identici": 1, "png_mancante": 1, "diversa_da_verificare": 1,
                                        "pixel_identici": 1, "digest_corrispondente": 1}
          and summary["layout"]["shard"] == {"ridisegnato_layout_assente": 2, "ridisegnato_digest_diverso": 2,
                                             "salvato_con_digest": 1}, (summary["image"], summary["layout"]))
    check("shard: conteggi coerenti, una riga sample col motivo per ogni campione non verificato",
          not summary["incoherent"] and summary["total"] == {"shard": 5, "generato": 0}
          and rows == [("caso-b", ["immagine: png_mancante"]), ("caso-c", ["immagine: diversa_da_verificare"])],
          (summary["incoherent"], rows))
    check("shard: ingresso registrato con lo sha256 del file", res["input"]["sha256"] == tar_sha)
    bad = Counter(res["counts"])
    bad["sample/shard/verificato"] += 1
    bad["sample/shard/non_verificato"] -= 1
    check("coerenza: un verificato senza semantica verificata e immagine corrispondente viene segnalato",
          any(x.startswith("shard: verificati") for x in sc.summarize(bad)["incoherent"]))
    bad = Counter(res["counts"])
    bad["image/shard/byte_identici"] += 1
    check("coerenza: totali diversi fra i gruppi vengono segnalati",
          any("totali diversi" in x for x in sc.summarize(bad)["incoherent"]))
    check("--cases: il layout salvato porta lo sha256 del PNG salvato, esito di scena",
          saved["layout"]["png_sha256"] == hashlib.sha256(saved_png).hexdigest() == case_rows[0]["png_sha256"]
          and case_rows[0]["outcome"] == "scena_verificata", case_rows)


def test_pairs_v4():
    """Coppie controfattuali polygon (S2 §2.5). Da v4 resta scritta una sola misura, quella della domanda, e la variante
    è la figura riscalata: fattore dalla lunghezza esatta, misure ricalcolate dalla geometria. Fra a e b cambia solo il
    numero scritto. In v3 lo shard delle coppie si riproduce."""
    import io
    import math

    import numpy as np
    from PIL import Image

    sr, sc = load_script("synth_render"), load_script("synth_semantics_check")
    bad, outside, stats = [], [], Counter()
    for i in range(1, 4 * 60, 4):  # famiglie a rotazione come in synth_render: i % 4 == 1 è polygon
        (ka, pa, ma), (kb, pb, mb) = sr.scene((f"synth-pair-{i:05d}", "polygon", 4))
        sa, sb = ma["spec"], mb["spec"]
        n, j = len(sa["points"]), sa["shown"][0]
        k = float(mb["pair"]["answer"]) / math.dist(sa["points"][j], sa["points"][(j + 1) % n])
        why = []
        if not sa["shown"] == sb["shown"] == [j]:
            why.append("misure scritte")
        if any(abs(xb - k * xa) > 1e-9 or abs(yb - k * ya) > 1e-9 for (xa, ya), (xb, yb) in zip(sa["points"], sb["points"])):
            why.append("riscalatura")
        if sb["lengths"] != [synth._fmt_len(math.dist(sb["points"][t], sb["points"][(t + 1) % n])) for t in range(n)]:
            why.append("misure dalla geometria")
        if {x: v for x, v in sa.items() if x not in ("points", "lengths")} != {x: v for x, v in sb.items()
                                                                               if x not in ("points", "lengths")}:
            why.append("angolo, punto segnato o altri campi diversi")
        if not ma["pair"]["answer"] == sa["lengths"][j] != sb["lengths"][j] == mb["pair"]["answer"]:
            why.append("risposte")
        printed = [{it["gid"]: it["text"] for it in m["layout"]["items"] if it["gid"].startswith("length:")} for m in (ma, mb)]
        if printed != [{f"length:{sa['sides'][j]}": ma["pair"]["answer"]}, {f"length:{sb['sides'][j]}": mb["pair"]["answer"]}]:
            why.append("misura disegnata")
        for m in (ma, mb):
            res = synth_check.check_sample(m, m["layout"])
            wrong = [v["kind"] for v in res["violations"] if v["kind"] in ("lunghezza_incoerente", "angolo_incoerente")]
            if wrong or [r["status"] for r in res["questions"]] != ["ok"]:
                why.append(f"{m['id']}: {wrong}, {[r['status'] for r in res['questions']]}")
            stats["altre violazioni"] += len(res["violations"]) - len(wrong)
            if m["spec"]["m"]:
                side = m["spec"]["sides"][m["spec"]["m"]["side"]]
                q = {"type": "m_side", "q": f"On which side does point {m['spec']['m']['label']} lie?", "answer": side}
                if synth_check.check_questions(m["spec"], [q], m["layout"])[0]["status"] != "ok":
                    why.append(f"{m['id']}: punto segnato fuori dal suo lato")
        ia, ib = (np.asarray(Image.open(io.BytesIO(p)).convert("RGB")).astype(int) for p in (pa, pb))
        diff = np.abs(ia - ib).max(axis=2) > 0
        mask = np.zeros_like(diff)
        for m in (ma, mb):
            x0, y0, x1, y1 = next(it["bbox"] for it in m["layout"]["items"] if it["gid"].startswith("length:"))
            mask[max(0, int(y0) - 3): int(y1) + 4, max(0, int(x0) - 3): int(x1) + 4] = True
        if np.count_nonzero(diff & ~mask):
            outside.append((ka, int(np.count_nonzero(diff & ~mask)), int(np.abs(ia - ib)[~mask].max())))
        stats["pixel diversi nel riquadro della misura"] += int(np.count_nonzero(diff & mask))
        stats["coppie"] += 1
        stats["con punto segnato"] += sa["m"] is not None
        stats["con angolo scritto"] += sa["angle"] is not None
        stats["con angolo retto"] += sa["right"] is not None
        if why:
            bad.append((ka, why))
    check("coppie polygon v4: una misura scritta, b è a riscalata dal fattore sulla lunghezza esatta, misure ricalcolate,"
          " angolo e punto segnato uguali, risposte e checker ok", not bad, (dict(stats), bad[:3]))
    check("coppie polygon v4: fra a e b cambiano solo i pixel del numero scritto",
          not outside and stats["pixel diversi nel riquadro della misura"] > 0, outside[:5])
    summary = sc.summarize(sc.pairs_task((0, 4, 4))["counts"])
    check("--pairs: 4 coppie, 8 record contati come generati, conteggi coerenti",
          summary["total"] == {"shard": 0, "generato": 8} and not summary["incoherent"], summary["total"])
    records = {}
    for key, files in iter_shard(V3 / "pairs" / "synth-pairs.tar"):
        records[key] = files
        if len(records) == 8:
            break
    same = [key in records and records[key]["png"] == png and json.loads(records[key]["json"]) == meta
            for i in range(4) for key, png, meta in sr.scene((f"synth-pair-{i:05d}", synth.FAMILIES[i % 4], 3))]
    check("coppie v3: le prime quattro dello shard si riproducono identiche, PNG e JSON", len(same) == 8 and all(same), same)


def test_marked_point_v4():
    """Da v4 il punto segnato dei polygon sta su un lato senza misura scritta. Se non ce n'è si toglie, e con lui solo la
    domanda m_side (S2 §2.5). Sul disegno non restano sovrapposizioni fra misure, punti segnati ed etichette."""
    counts, bad = Counter(), []
    for i in range(1, 4 * 2500, 4):
        key = f"synth-{i:07d}"
        s3, s4 = synth.make_spec(key, "polygon", 3), synth.make_spec(key, "polygon", 4)
        m3, m4 = s3["m"], s4["m"]
        free = [j for j in range(len(s4["points"])) if j not in s4["shown"]]
        counts["scene"] += 1
        counts["con punto in v3"] += m3 is not None
        counts["con punto in v4"] += m4 is not None
        counts["spostati"] += m3 is not None and m4 is not None and m4["side"] != m3["side"]
        counts["tolti"] += m3 is not None and m4 is None
        if m4 is not None and m4["side"] in s4["shown"]:
            bad.append((key, "punto su un lato con la misura"))
        if m3 is not None and m4 is None and free:
            bad.append((key, "tolto con un lato libero"))
        if m3 is not None and m3["side"] not in s3["shown"] and m4 != m3:
            bad.append((key, "spostato senza motivo"))
        if any(q["type"] == "m_side" for q in synth.questions(s4)) != (m4 is not None):
            bad.append((key, "domanda m_side"))
    check("punto segnato v4: mai su un lato con la misura, tolto solo senza lati liberi, domanda m_side solo col punto",
          not bad, (dict(counts), bad[:3]))
    check("punto segnato v4: scelta deterministica", synth.make_spec("synth-0000005", "polygon", 4)
          == synth.make_spec("synth-0000005", "polygon", 4))
    check("m_side v4: la domanda resta in più di un terzo delle scene polygon", counts["con punto in v4"] * 3 > counts["scene"],
          dict(counts))
    marked = ("length", "label", "point")
    over, statuses = Counter(), Counter()
    for i in range(1, 4 * 300, 4):
        spec, _, layout = rendered(f"synth-{i:07d}", "polygon", 4)
        for v in synth_check.renderer_violations(spec, layout):
            if v["kind"] in ("testi_sovrapposti", "testo_su_marcatore") and all(g.split(":")[0] in marked for g in v["gids"]):
                over["/".join(sorted(g.split(":")[0] for g in v["gids"]))] += 1
        qs = [q for q in synth.questions(spec) if q["type"] in ("m_side", "side_len")]
        for r in synth_check.check_questions(spec, qs, layout):
            statuses[r["type"] + "/" + r["status"]] += 1
    check("polygon v4 disegnati (300): nessuna sovrapposizione fra misure, punti segnati e loro etichette", not over, dict(over))
    check("polygon v4 disegnati: domande m_side e side_len ok", statuses and all(k.endswith("/ok") for k in statuses),
          dict(statuses))


def test_origin_labels_v4():
    """Da v4 le tacche -1 di coord e function restano senza numero (S2 §2.5): niente sovrapposizioni all'origine,
    domande ancora risolvibili. In v3 i numeri restano."""
    origin = {"axis_label:O", "tick_label:x:-1", "tick_label:y:-1"}
    counts, statuses = Counter(), Counter()
    for fam in ("coord", "function"):
        for i in range(150):
            spec, _, layout = rendered(f"t-origin-{fam}-{i}", fam, 4)
            if not spec["ticks"]:
                continue
            t = layout["transform"]
            counts["scene con tacche"] += 1
            counts["numeri -1"] += sum(it["kind"] == "text" and it["text"] == "-1" for it in layout["items"])
            counts["tacche -1 su entrambi gli assi"] += (
                any(abs(c - (t["x0"] - t["ux"])) <= 2 for c in synth_check._axis_ticks(layout, "x"))
                and any(abs(c - (t["y0"] - t["uy"])) <= 2 for c in synth_check._axis_ticks(layout, "y")))
            counts["sovrapposizioni all'origine"] += sum(v["kind"] == "testi_sovrapposti" and set(v["gids"]) <= origin
                                                        for v in synth_check.renderer_violations(spec, layout))
            for r in synth_check.check_questions(spec, synth.questions(spec), layout):
                statuses[f"{fam}/{r['status']}"] += 1
    check("origine v4: nessun numero sulle tacche -1, che restano disegnate; nessuna sovrapposizione all'origine",
          counts["numeri -1"] == 0 and counts["sovrapposizioni all'origine"] == 0
          and counts["tacche -1 su entrambi gli assi"] == counts["scene con tacche"] > 0, dict(counts))
    check("origine v4: domande coord e function risolvibili, nessuna errata o non verificabile",
          not any(k.endswith(("/errata", "/non_verificabile")) for k in statuses), dict(statuses))
    s3 = first((synth.make_spec(f"t-origin-v3-{i}", "coord", 3) for i in range(200)), lambda s: s["ticks"] and s["step"] == 1)
    layout = synth.render(s3, layout=True)[1]
    check("origine v3: i numeri delle tacche -1 restano su entrambi gli assi",
          sum(it["kind"] == "text" and it["text"] == "-1" for it in layout["items"]) == 2)


def test_coords_from_ticks():
    """Coordinate senza numero sulla loro tacca (S2 §2.5). Il checker le accetta solo se la tacca è disegnata e le due
    vicine hanno un numero, alla stessa distanza e su una scala lineare. Le posizioni vengono dai riquadri disegnati e
    i valori dai testi, non dai gid."""
    def find(both: bool):
        for i in range(0, 4 * 5000, 4):
            spec = synth.make_spec(f"synth-check-{i:07d}", "coord", 4)
            if not spec["ticks"] or spec["step"] != 2 or not spec["line"]:
                continue
            p = next(p for p in spec["points"] if p["label"] == spec["line"]["test"])
            odd = (p["x"] % 2 != 0, p["y"] % 2 != 0)
            if (all(odd) if both else sum(odd) == 1):
                q = next(q for q in synth.questions(spec) if q["type"] == "coords" and q["q"].endswith(f"point {p['label']}?"))
                return spec, p, q
        raise AssertionError("nessun caso")

    spec, p, q = find(False)
    _, layout = synth.render(spec, layout=True)

    def status(lay, answer=None):
        r = synth_check.check_questions(spec, [{**q, "answer": answer or q["answer"]}], lay)[0]
        return r["status"], r["why"]

    check(f"{spec['key']}: punto {p['label']} ({p['x']}, {p['y']}), una coordinata senza numero: ok dalle tacche vicine",
          status(layout) == ("ok", ""), status(layout))
    axis = "x" if p["x"] % 2 else "y"
    a = 0 if axis == "x" else 1
    t = layout["transform"]
    pos = t["x0"] + t["ux"] * p["x"] if axis == "x" else t["y0"] + t["uy"] * p["y"]
    step = abs(t["ux"] if axis == "x" else t["uy"])

    def tick_at(it, at):
        x0, y0, x1, y1 = it["bbox"]
        return (it["kind"] == "tick" and abs(synth_check._center(it["bbox"])[a] - at) <= 2
                and ((y1 - y0 > x1 - x0) if axis == "x" else (x1 - x0 > y1 - y0)))

    def label_at(it, at):
        return (it["kind"] == "text" and it["gid"].startswith(f"tick_label:{axis}:")
                and abs(synth_check._center(it["bbox"])[a] - at) <= 2)

    def shifted(it, d):
        b = list(it["bbox"])
        b[a] += d
        b[a + 2] += d
        return {**it, "bbox": b}

    def mutate(fn):
        lay = deepcopy(layout)
        lay["items"] = [x for it in lay["items"] for x in fn(it)]
        return lay

    cases = {
        "tacca mancante": mutate(lambda it: [] if tick_at(it, pos) else [it]),
        "tacca spostata di 0.4 unità": mutate(lambda it: [shifted(it, 0.4 * step)] if tick_at(it, pos) else [it]),
        "tacca vicina senza numero": mutate(lambda it: [] if label_at(it, pos + step) else [it]),
        "un solo numero sull'asse": mutate(lambda it: [it] if it["kind"] != "text" or not it["gid"].startswith(
            f"tick_label:{axis}:") or label_at(it, pos + step) else []),
        "numero vicino sbagliato": mutate(lambda it: [{**it, "text": str(int(it["text"]) + 1)}] if label_at(it, pos + step)
                                          else [it]),
        "tacca vicina e suo numero spostati di 0.3 unità": mutate(
            lambda it: [shifted(it, 0.3 * step)] if tick_at(it, pos + step) or label_at(it, pos + step) else [it]),
    }
    got = {name: status(lay) for name, lay in cases.items()}
    check("coordinate da tacche: tacca mancante o spostata, scala insufficiente o incoerente danno non ammissibile",
          all(s == "non_ammissibile" for s, _ in got.values()), got)
    relabeled = mutate(lambda it: [{**it, "gid": f"tick_label:{axis}:99"}] if label_at(it, pos + step) else [it])
    check("coordinate da tacche: il gid non conta, un gid sbagliato col numero giusto resta ok", status(relabeled)[0] == "ok",
          status(relabeled))
    wrong = f"({p['x'] + 2}, {p['y']})" if axis == "x" else f"({p['x']}, {p['y'] - 2})"
    check("coordinate da tacche: una risposta alterata sulla coordinata senza numero è errata",
          status(layout, wrong)[0] == "errata", status(layout, wrong))
    spec, p, q = find(True)
    _, layout = synth.render(spec, layout=True)
    check(f"{spec['key']}: punto {p['label']} ({p['x']}, {p['y']}), nessuna delle due coordinate col numero: ok",
          status(layout) == ("ok", ""), status(layout))
    spec = synth.make_spec("synth-0010596", "coord", 3)
    q = next(q for q in synth.questions(spec) if q["type"] == "coords" and q["q"].endswith("point W?"))
    check("synth-0010596 v3: W (-3, 8), prima non ammissibile, ora ok dalle tacche vicine",
          synth_check.check_questions(spec, [q], synth.render(spec, layout=True)[1])[0]["status"] == "ok")


def test_review_bounds():
    """scripts/synth_review.py: i limiti esatti del lato opposto (opposite_range) contengono ogni valore campionato e la
    griglia li raggiunge; le contraddizioni delle coppie si dimostrano con un margine, non con la griglia."""
    import math
    import random

    rv = load_script("synth_review")
    rng, bad, worst = random.Random(0), [], 0.0
    for _ in range(300):
        p0, q0, t0 = rng.uniform(1, 15), rng.uniform(1, 15), rng.uniform(10, 168)
        P, Q, T = (p0, p0 + rng.uniform(0, 1)), (q0, q0 + rng.uniform(0, 1)), (t0, t0 + rng.uniform(0, 2))
        lo, hi = rv.opposite_range(P, Q, T)
        vals = [math.sqrt(a * a + b * b - 2 * a * b * math.cos(math.radians(c))) for a in rv._grid(*P, 21)
                for b in rv._grid(*Q, 21) for c in rv._grid(*T, 5)]
        if min(vals) < lo or max(vals) > hi:
            bad.append((P, Q, T, lo, hi, min(vals), max(vals)))
        worst = max(worst, min(vals) - lo, hi - max(vals))
    check("opposite_range: i valori campionati stanno nei limiti esatti, raggiunti dalla griglia a meno di 0.01",
          not bad and worst < 0.01, (round(worst, 5), bad[:2]))
    tri = {"points": [[0, 0]] * 3, "shown": [0, 1, 2], "labels": ["S", "T", "U"], "sides": ["ST", "TU", "US"],
           "lengths": ["5.9", "11.7", "12.8"], "angle": {"vertex": 1, "value": 47}}
    proof = rv.contradiction_proof(tri, (1, 46.5, 47.5), False)
    check("coppia 00033: con 5.9, 11.7 e 47° in T, US sta sotto 9 ed è scritto 12.8: impossibile, margine oltre 3",
          proof["proven"] and proof["margin"] > 3 and proof["opposite_range"][1] < 9, proof["constraint"])
    tri = {"points": [[0, 0]] * 3, "shown": [0, 1, 2], "labels": ["G", "H", "K"], "sides": ["GH", "HK", "KG"],
           "lengths": ["50", "17", "48"], "angle": None}
    loose, strict = (rv.contradiction_proof(tri, (2, 90.0, 90.0), e) for e in (False, True))
    check("coppia 00285: angolo retto in K, 17 e 48 danno 50.9: compatibile con interi arrotondati, impossibile se esatti",
          not loose["proven"] and strict["proven"], (loose["margin"], strict["margin"]))


if __name__ == "__main__":
    test_v3_reproduction()
    test_intersections()
    test_classification()
    test_v3_v4_regression()
    test_clipping_generator()
    test_rendered_checks()
    test_clipping_rendered()
    test_renderer_invariants()
    test_sample_evidence()
    test_pairs_v4()
    test_marked_point_v4()
    test_origin_labels_v4()
    test_coords_from_ticks()
    test_review_bounds()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
