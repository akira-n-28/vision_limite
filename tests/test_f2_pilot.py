#!/usr/bin/env python3
"""Test CPU del pilota dell'audit F2 (schema f2-pilot/2): limite_vl.f2_claims (misure della didascalia e della soluzione,
frasi asserite, alterazioni con verità certa e non garantita, controlli sui sistemi di riferimento, frasi candidate, fatti
e categorie) e un giro completo di scripts/f2_pilot.py su fixture in stile MAVIS (export, risposte simulate di E, VA, VB e
VC, metriche, lotti incompleti, codice congelato). Nessun agente, nessun dato vero."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2_claims import (asserted_at, caption_measures, caption_negative, classify_sentences,  # noqa: E402
                                 code_only_measures, context_sentence, cot_negative, fact_category, fact_spans, frame_control,
                                 inject_configuration, inject_construction, inject_measure, inject_orientation,
                                 quote_spans, resolve_claim, resolve_page, sentence_spans, shared_edges, solution_measures)
from limite_vl.provenance import sha256_text  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


CAPTION = "ABCD is a square. CBEF is a square. Side AB is 5 units long. Angle ABC measures 90°."
CAP_SEMI = ("ABCD is the rectangle ABCD with side CD removed and replaced by a semicircle with diameter CD drawn outside the "
            "rectangle. CBEF is a square. Side AB is 5 units long. Side AD is 3 units long. Angle ABC measures 90°.")


def test_measures():
    caps = ("The segment DC has a length of 1. The distance between points D and F is 4*√(2). The angle CDF measures 90°. "
            "Side CB is measured at 70*√(3) units. The length from Point E to Point D is 96. The degree size of angle HEG is 45. "
            "The measurement from F to G is 9. The side is shown.")
    f = caption_measures(caps)
    check("didascalia: segmenti, angoli, radici, punti", f.get(("seg", "CD")) == 1 and abs(f[("seg", "DF")] - 4 * 2 ** 0.5) < 1e-9
          and f.get(("ang", "D", "CF")) == 90 and abs(f[("seg", "BC")] - 70 * 3 ** 0.5) < 1e-6 and f.get(("seg", "DE")) == 96
          and f.get(("ang", "E", "GH")) == 45 and f.get(("seg", "FG")) == 9 and len(f) == 7, f)
    cap = "Side AB is 16 units long. The distance between points D and F is 4*√(2). Angle ACB measures 30°."
    sol = ("<think>So AB = 16. Then BC = 2*AB = 32. Also \\(DF = 4\\sqrt{2}\\), i.e. DF = 5.66 roughly. "
           "\\angle ACB = 30^\\circ. Wait, AB = 18? Suppose AB = 20, then no.</think>")
    got = [(d["key"], d["verdict"]) for d in solution_measures(sol, cap)]
    check("soluzione: misura a fine frase, espressioni escluse, radici e decimali, ipotesi",
          got == [("seg-AB", "match"), ("seg-DF", "match"), ("seg-DF", "match"), ("ang-C-AB", "match"),
                  ("seg-AB", "contradiction"), ("seg-AB", "hypothetical")], got)
    cap2 = "The distance of side FE is 10*√(3)/3. Side DC is 12 units long. Side AB is 7 units long. Side BD is 7 units long."
    check("didascalia: frazione con radice", abs(caption_measures(cap2)[("seg", "EF")] - 10 * 3 ** 0.5 / 3) < 1e-9)
    sol2 = ("<think>So the angle between FE and FH is 90°, and angle between side BC and side BA is 45 degrees. "
            "FE = 10√3/3. Here r = half of EF = half of DC = 6. So AD = AB + BD = 14. Also DC = 12.</think>")
    got = [(d["key"], d["verdict"]) for d in solution_measures(sol2, cap2)]
    check("soluzione: angoli fra due lati non sono lunghezze, frazioni con radice, valori in catena non contano",
          got == [("seg-EF", "match"), ("seg-CD", "chained"), ("seg-CD", "match")], got)
    cap3 = "Side AB is 1 units long. Side BC is 2 units long. Side FG is 8 units long. Side EC is 84*√(3). Angle ABC measures 45°."
    sol3 = ("<think>AB * BC = 1 * √2 = √2. So (√3/2) * FG = 4. Then 180 - angle ABC = 135°. Vector EC = (84, +84√2). "
            "Also FC is long and EC = 84\\sqrt3. Given AB = 1, BC = 2.</think>")
    got = [(d["key"], d["verdict"]) for d in solution_measures(sol3, cap3)]
    check("soluzione: prodotti, differenze e vettori non sono misure; \\sqrt3 senza graffe",
          got == [("seg-CE", "match"), ("seg-AB", "match"), ("seg-BC", "match")], got)


def test_asserted():
    def at(text, needle):
        return asserted_at(text, text.index(needle))
    check("frase asserita", at("<think>So AB = 5. Next step.</think>", "AB"))
    check("ipotesi, condizioni, domande, negazioni, casi e verifiche non asseriscono",
          not at("<think>If AB = 5 then x.</think>", "AB") and not at("<think>Maybe AB = 5.</think>", "AB")
          and not at("<think>AB = 5? Yes.</think>", "AB") and not at("<think>AB is not 5.</think>", "AB")
          and not at("<think>Case 1: AB = 5.</think>", "AB") and not at("<think>We need to confirm AB = 5.</think>", "AB"))
    check("frase corretta subito dopo non asserisce", not at("<think>AB = 5. Wait, that is off.</think>", "AB")
          and not at("<think>AB = 5. No, AB = 6.</think>", "AB = 5"))


def test_injections():
    rng = random.Random(1)
    sol = ('<think>It says: "Figure: Side AB is 5 units long." So AB = 5 and the square CBEF is attached to side CB of ABCD. '
           "The side AB is horizontal. Angle ABC = 90°.</think> \\boxed{25}")
    t, info = inject_measure(sol, CAPTION, rng)
    ms = solution_measures(t, CAPTION)
    check("misura cambiata: fuori dalla citazione del prompt, contraddizione per il codice",
          info["start"] > sol.index('long."') and any(d["verdict"] == "contradiction" and d["start"] == info["start"] for d in ms)
          and t[info["start"]:info["end"]] == info["changed"], info)
    check("misura cambiata: nessun punto valido", inject_measure("<think>AB is long.</think>", CAPTION, rng) is None)
    check("misura: mai in una frase ipotetica", inject_measure("<think>Maybe AB = 5 here.</think>", CAPTION, rng) is None)
    shared, edges = shared_edges(CAPTION)
    check("lati in comune fra le forme", shared == {"BC"} and sorted(edges) == ["ABCD", "CBEF"], (shared, edges))
    t, info = inject_construction(sol, CAPTION, rng)
    new = t[info["start"]:info["end"]]
    check("costruzione cambiata: un lato non in comune, verità nota", info["original"] == "CB" and "".join(sorted(new)) not in shared
          and "".join(sorted(new)) in {e for es in edges.values() for e in es}, info)
    check("costruzione: senza parola di attacco nessuna alterazione", inject_construction("<think>side CB = 5</think>", CAPTION, rng) is None)
    check("costruzione: 'adjacent side' non è un attacco, niente parentesi né domande",
          inject_construction("<think>The adjacent side CB is 5.</think>", CAPTION, rng) is None
          and inject_construction("<think>Square CBEF is next to ABCD (sharing side CB).</think>", CAPTION, rng) is None
          and inject_construction("<think>Square CBEF attached to side CB? Actually yes.</think>", CAPTION, rng) is None)
    cap4 = "ABCD is a square. DCEF is a rectangle. Side AB is 5 units long."
    t4, i4 = inject_construction("<think>The rectangle is attached along side DC to rectangle DCEF.</think>", cap4, random.Random(0))
    check("costruzione: il lato nuovo non appartiene alla forma nominata subito dopo",
          "".join(sorted(t4[i4["start"]:i4["end"]])) in {"AB", "BC", "AD"}, i4)
    t, info = inject_orientation(sol, random.Random(0))
    check("orientamento scambiato, verità dichiarata non garantita", t[info["start"]:info["end"]] == "vertical"
          and info["truth"] == "not guaranteed")
    check("orientamento: mai dentro una citazione, mai 'right angle', 'right at', 'isosceles right', dentro/fuori",
          inject_orientation('<think>It says: "Figure: AB is horizontal."</think>', rng) is None
          and inject_orientation("<think>There is a right angle at A. The perimeter is as below.</think>", rng) is None
          and inject_orientation("<think>So triangle ABC is right at B.</think>", rng) is None
          and inject_orientation("<think>Triangle ABC is isosceles right.</think>", rng) is None
          and inject_orientation("<think>The semicircle on CD is outside.</think>", rng) is None
          and inject_orientation("<think>Here AB is horizontal.</think>", random.Random(0)) is not None)
    neg = caption_negative(CAPTION, "A")
    check("controllo negativo dalla didascalia: tutte le misure uguali", all(d["verdict"] == "match" for d in solution_measures(neg, CAPTION))
          and sum(d["verdict"] == "match" for d in solution_measures(neg, CAPTION)) == 2)
    check("controllo negativo dal CoT: ammesso solo se coerente con la didascalia",
          cot_negative("Angle ABC = 90° and AB = 5, so the area is 25.", CAPTION, "A") is not None
          and cot_negative("AB = 6, so the area is 36. Angle ABC = 90°.", CAPTION, "A") is None
          and cot_negative("The area is 25.", CAPTION, "A") is None)


def test_configuration():
    t, i = inject_configuration("<think>The semicircle is drawn outside the rectangle. So the area adds.</think>", CAP_SEMI, random.Random(0))
    check("configurazione dentro/fuori: dalla didascalia, verità certa", i["mode"] == "inside/outside" and t[i["start"]:i["end"]] == "inside"
          and i["certain"] is True, i)
    check("dentro/fuori: originale già diverso dalla didascalia, nessuna alterazione",
          inject_configuration("<think>The semicircle is drawn inside the rectangle.</think>", CAP_SEMI, random.Random(0)) is None)
    sq = "ABCD is a square. Side AB is 5 units long."
    t, i = inject_configuration("<think>So AB and CD are opposite sides.</think>", sq, random.Random(0))
    check("lati opposti -> adiacenti", i["mode"] == "opposite/adjacent" and t[i["start"]:i["end"]] == "adjacent", i)
    t, i = inject_configuration("<think>So A and C are opposite vertices.</think>", sq, random.Random(0))
    check("vertici opposti -> adiacenti", t[i["start"]:i["end"]] == "adjacent", i)
    check("opposti/adiacenti: originale sbagliato, lato rimosso, ipotesi: nessuna alterazione",
          inject_configuration("<think>So AB and CD are adjacent sides.</think>", sq, random.Random(0)) is None
          and inject_configuration("<think>So AB and CD are opposite sides.</think>", CAP_SEMI, random.Random(0)) is None
          and inject_configuration("<think>Maybe AB and CD are opposite sides.</think>", sq, random.Random(0)) is None)
    tri = "ABC is a right triangle. Angle ABC measures 90°. Side AB is 5 units long."
    t, i = inject_configuration("<think>In triangle ABC the right angle is at B.</think>", tri, random.Random(0))
    check("angolo retto spostato su un altro vertice dello stesso triangolo", i["mode"] == "right angle"
          and t[i["start"]:i["end"]] in {"A", "C"}, i)
    check("angolo retto: senza il 90° nella didascalia nessuna alterazione", inject_configuration(
        "<think>In triangle ABC the right angle is at B.</think>", "ABC is a right triangle. Side AB is 5 units long.", random.Random(0)) is None)


def _coords(st: str) -> dict:
    val = lambda s: (-1 if s.startswith("-") else 1) * float(s.lstrip("-") or 0) if s != "0" else 0.0  # noqa: E731
    return {p: (val(x), val(y)) for p, x, y in re.findall(r"([A-D]) = \((-?\w+), (-?\w+)\)", st)}


def test_frame_controls():
    rect = "ABCD is a rectangle. Side AB is 5 units long. Side AD is 3 units long."
    ok_legit = ok_broken = True
    turns = set()
    for seed in range(24):
        c = _coords(frame_control(rect, random.Random(seed), False)["statement"])
        d = lambda p, q: math.dist(c[p], c[q])  # noqa: E731
        ab = (c["B"][0] - c["A"][0], c["B"][1] - c["A"][1])
        ad = (c["D"][0] - c["A"][0], c["D"][1] - c["A"][1])
        ok_legit &= (d("A", "B") == 5 and d("A", "D") == 3 and ab[0] * ad[0] + ab[1] * ad[1] == 0
                     and all(abs(c["C"][k] - (c["B"][k] + c["D"][k] - c["A"][k])) < 1e-9 for k in (0, 1)))
        turns.add((ab, ad))
        b = _coords(frame_control(rect, random.Random(seed), True)["statement"])
        ok_broken &= math.dist(b["A"], b["B"]) == 3
    check("assi leciti: rettangolo ABCD con AB = 5, AD = 3, in ordine, girato o specchiato", ok_legit and len(turns) == 8, len(turns))
    check("assi rotti nel rettangolo: lunghezze scambiate", ok_broken)
    sq = "ABCD is a square. Side AB is 4 units long."
    b = _coords(frame_control(sq, random.Random(3), True)["statement"])
    g = _coords(frame_control(sq, random.Random(3), False)["statement"])
    check("assi rotti nel quadrato: A e C vicini (ordine dei vertici rotto)", math.dist(b["A"], b["C"]) == 4
          and abs(math.dist(g["A"], g["C"]) - 4 * 2 ** 0.5) < 1e-9)
    c = frame_control(rect, random.Random(0), True)
    check("controllo: base chosen, verità attesa", c["basis"] == "chosen" and c["expected"] == "E" and c["family"] == "frame_broken")
    check("senza lati noti nessun controllo", frame_control("ABCD is a rectangle.", random.Random(0), False) is None)


def test_sentences():
    text = "<think>So x = 3.5 here. Then e.g. the side. Next line\nAnother one.</think> Answer."
    got = [text[a:b] for a, b in sentence_spans(text)]
    check("frasi: decimali e 'e.g.' interi, a capo e tag separano", got == ["So x = 3.5 here.", "Then e.g. the side.", "Next line",
                                                                         "Another one.", "Answer."], got)
    t2 = '<think>It says: "Figure: CBEF is a square." So AB = 5.</think>'
    got = [t2[a:b] for a, b in sentence_spans(t2)]
    check("frasi: la virgoletta di chiusura chiude la frase", got == ['It says: "Figure: CBEF is a square."', "So AB = 5."], got)
    sol = ('<think>It says: "Figure: CBEF is a square." So AB = 5. The semicircle is drawn outside the rectangle. '
           "Then 5 * 3 = 15. So AB = 5. AB is 7 here.</think> The answer is \\boxed{A}.")
    gs = {g["text"]: g for g in classify_sentences(sol, CAP_SEMI)}
    check("categorie: citazione della didascalia, solo misure uguali, candidata, scartata",
          gs['It says: "Figure: CBEF is a square."']["category"] == "caption_verbatim"
          and gs["So AB = 5."]["category"] == "measure_only" and len(gs["So AB = 5."]["spans"]) == 2
          and gs["The semicircle is drawn outside the rectangle."]["category"] == "candidate"
          and gs["Then 5 * 3 = 15."]["category"] == "other" and gs["The answer is \\boxed{A}."]["category"] == "other"
          and gs["AB is 7 here."]["category"] == "candidate", {k: v["category"] for k, v in gs.items()})
    check("categorie: relazione senza valori e valore di una grandezza senza nome sono candidate",
          classify_sentences("<think>D and C are adjacent.</think>", CAP_SEMI)[0]["category"] == "candidate"
          and classify_sentences("<think>Total = 9 + 2π.</think>", CAP_SEMI)[0]["category"] == "candidate"
          and classify_sentences("<think>So h = 3/2.</think>", CAP_SEMI)[0]["category"] == "candidate")
    check("fatto di sole misure uguali alla didascalia: scorciatoia del codice",
          code_only_measures("AB has length 5", CAP_SEMI) == [("seg-AB", 5.0)]
          and code_only_measures("Side AB is 5 units long", CAP_SEMI) == [("seg-AB", 5.0)]
          and code_only_measures("AB = 7", CAP_SEMI) is None and code_only_measures("CE = 4", CAP_SEMI) is None
          and code_only_measures("square ABCD has AB = 5", CAP_SEMI) is None)


def test_facts_resolution():
    text = "AB = 5. So AB = 5 again. CD = 3."
    a = text.index("So")
    check("fatto: span solo dentro la sua frase", fact_spans(text, [(a, a + 16)], "AB = 5") == [(a + 3, a + 9)])
    text2 = "<think>Place \\(D\\) at the origin. The side \\(DF\\) is vertical, so DF = 12.\nNext line.</think>"
    sp = quote_spans(text2, "the side DF is vertical")
    check("citazione: span nel testo originale", len(sp) == 1 and text2[sp[0][0]:sp[0][1]].lower().startswith("the side"), sp)
    check("frase di contesto", context_sentence(text2, sp[0]).endswith("DF = 12."))
    E = {"status": "resolved", "value": "E"}
    K = {"status": "resolved", "value": "K"}
    C = {"status": "resolved", "value": "C"}
    M = {"status": "resolved", "value": "M"}
    Q = {"status": "quarantine", "value": None}
    check("categorie: errore visivo solo se asserito", fact_category("asserted", "figure", True, E, M) == "visual_error"
          and fact_category("rejected", "figure", False, E, M) == "discarded_hypothesis"
          and fact_category("hypothesis", "figure", True, E, M) == "reasoning_flag"
          and fact_category("rejected", "figure", True, E, M) == "reasoning_flag"
          and fact_category("hypothesis", "figure", False, E, M) == "open_hypothesis")
    check("categorie: pagina contraddetta conta solo per proprietà visibili asserite",
          fact_category("asserted", "figure", True, K, C) == "orientation_error"
          and fact_category("asserted", "given", True, K, C) == "orientation_error"
          and fact_category("asserted", "chosen", True, K, C) == "frame_change"
          and fact_category("asserted", "computed", True, K, C) == "frame_change"
          and fact_category("asserted", "figure", True, K, M) == "ok")
    check("categorie: quarantene", fact_category("asserted", "figure", True, Q, M) == "quarantine"
          and fact_category("asserted", "figure", True, K, Q) == "quarantine"
          and fact_category("asserted", "computed", True, K, Q) == "ok")
    check("risoluzione: consenso, terzo giudizio, maggioranza, quarantena",
          resolve_claim(["false", "not_visible"])["value"] == "E" and resolve_claim(["true", "false"])["status"] == "need_blind"
          and resolve_claim(["true", "false", "deduction"])["value"] == "K"
          and resolve_claim(["true", "false", "unclear"])["status"] == "quarantine")
    check("risoluzione della pagina", resolve_page(["contradicts", "contradicts"])["value"] == "C"
          and resolve_page(["matches", "none"])["value"] == "M" and resolve_page(["contradicts", "none"])["status"] == "need_blind"
          and resolve_page(["contradicts", "none", "contradicts"])["value"] == "C")


# ---------------------------------------------------------------- giro completo

def run(*args) -> subprocess.CompletedProcess:
    env = dict(os.environ, SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID", "test"))
    return subprocess.run([sys.executable, "-B", *map(str, args)], env=env, capture_output=True, text=True)


def png(i: int) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64 + i, 48), "white").save(buf, format="PNG")
    return buf.getvalue()


def meta(i: int) -> dict:
    return {"id": f"d{i % 4}-{i}", "depth": i % 4, "choices": ["15", "20", "30", "35"], "answer_index": 0,
            "text_en": {"caption": CAP_SEMI, "diagram_generation": "", "vision_dominant_question": "What is the area of ABCD?",
                        "CoT_reasoning": "Angle ABC = 90° and side AB is 5, so the area of ABCD is 15."}}


SOLUTION = ("<think>The figure shows rectangle ABCD with AB = 5 and AD = 3. Square CBEF is attached to side CB of ABCD. The side AB is "
            "horizontal. AB and BC are adjacent sides. The semicircle is drawn outside the rectangle. Angle ABC = 90°. Maybe the "
            "semicircle is inside. Suppose CBEF is attached to side AD, but that is impossible. So the area is 15. Variante {c}."
            "</think> The area is \\boxed{{15}}.")


def test_pipeline(tmp: Path):
    from limite_vl.mavis import f2_problem

    metas = [meta(i) for i in range(1, 21)]
    sdir = tmp / "shards"
    sdir.mkdir()
    with tarfile.open(sdir / "caption-00000.tar", "w", format=tarfile.USTAR_FORMAT) as tar:
        for i, m in enumerate(metas, 1):
            for name, data in ((f"{m['id']}.json", json.dumps(m).encode()), (f"{m['id']}.png", png(i))):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    gen = tmp / "gen-pilot"
    gen.mkdir()
    rows = []
    for m in metas:
        p = f2_problem(m)
        rows.append({"key": p["key"], "source_dataset": f"MAVIS-Geometry d{m['depth']}", "answer": p["answer_text"],
                     "answer_letter": p["answer_letter"], "qwen4b_pass_rate": None, "prompt_tokens": 40, "question": p["question"],
                     "question_sha256": sha256_text(p["question"]), "prompt_sha256": "0" * 64,
                     "completions": [{"text": SOLUTION.format(c=c), "finish_reason": "stop", "tokens": 30} for c in "abc"]})
    (gen / "part-00000.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sel = tmp / "sel"
    r = run(ROOT / "scripts" / "f2_select.py", "--gen", gen, "--out", sel, "--keep", "2", "--workers", "1")
    recs = [json.loads(x) for x in (sel / "selected.v2.jsonl").read_text().splitlines()] if r.returncode == 0 else []
    check("fixture: selezione v2 con le domande", len(recs) == 40, r.stderr[-400:])
    amap = tmp / "audit_map.jsonl"
    audited = [x for x in recs if int(x["key"].rsplit("-", 1)[1]) <= 8]
    amap.write_text("".join(json.dumps({"fid": f"f{i:03d}", "audit": "finto", "sid": x["sid"], "status": "exact", "calibration": False,
                                        "verdict": "include" if i % 2 else "exclude"}) + "\n" for i, x in enumerate(audited, 1)))
    # campione di sviluppo finto: una soluzione auditata (esclusa dal campione nuovo e usata come regressione)
    dev = tmp / "dev"
    (dev / "_gold").mkdir(parents=True)
    d0 = next(x for x in recs if x["key"] == "mavis-d1-1")
    dpng = png(1)
    (dev / "p001.png").write_bytes(dpng)
    (dev / "_gold" / "items.json").write_text(json.dumps([{
        "pid": "p001", "kind": "real", "family": None, "key": d0["key"], "depth": 1, "sid": d0["sid"], "source_sid": None,
        "v1_verdict": "include", "question": d0["question"], "solution": d0["solution"], "caption": CAP_SEMI,
        "image": str(dev / "p001.png"), "image_sha256": hashlib.sha256(dpng).hexdigest()}]))
    out = tmp / "pilot"
    r = run(ROOT / "scripts" / "f2_pilot.py", "export", "--selection", sel / "selected.v2.jsonl", "--mavis-shards", sdir / "*.tar",
            "--audit-map", amap, "--out", out, "--seed", "5", "--exclude-dirs", dev, "--regression-dir", dev,
            "--regression-pids", "p001", "--real-per-depth", "1", "--measure", "2", "--construction", "2", "--configuration", "2",
            "--orientation", "2", "--caption-negatives", "1", "--cot-negatives", "1", "--frame-controls", "2")
    check("export del pilota", r.returncode == 0, r.stderr[-800:])
    if r.returncode:
        return
    man = json.loads((out / "manifest.json").read_text())
    items = {it["pid"]: it for it in json.loads((out / "_gold" / "items.json").read_text())}
    truth = json.loads((out / "_gold" / "truth.json").read_text())
    measures = json.loads((out / "_gold" / "measures.json").read_text())
    check("campione: 4 vere (una per profondità), alterazioni per famiglia, negativi, regressione, controlli sugli assi",
          man["design"]["obtained"] == {"real": 4, "measure": 2, "construction": 2, "configuration": 2, "orientation": 2,
                                        "caption_negative": 1, "cot_negative": 1, "regression": 1}
          and man["design"]["frame_controls"] == {"frame_legit": 2, "frame_broken": 2}
          and sorted(it["depth"] for it in items.values() if it["kind"] == "real") == [0, 1, 2, 3], man["design"])
    check("campione indipendente: nessuna chiave del campione di sviluppo fra le nuove",
          "mavis-d1-1" not in {it["key"] for it in items.values() if it["kind"] != "regression"}
          and [it["key"] for it in items.values() if it["kind"] == "regression"] == ["mavis-d1-1"])
    check("verità: certa per misura, costruzione e configurazione, non garantita per l'orientamento",
          all(t["certain"] == (t["family"] != "orientation") for t in truth.values() if "start" in t))
    check("congelati: codice, istruzioni, criteri", set(man["frozen"]) >= {"scripts/f2_pilot.py", "src/limite_vl/f2_claims.py",
                                                                          "config/audit/f2_classify.md", "config/audit/f2_claim_verify2.md"}
          and "criteria" in man)
    m_pids = [p for p, t in truth.items() if t["family"] == "measure"]
    check("canale deterministico: ogni misura cambiata è una contraddizione per il codice",
          all(any(d["verdict"] == "contradiction" and d["start"] == truth[p]["start"] for d in measures[p]) for p in m_pids))
    e_tasks = [t for b in sorted((out / "batches").glob("E-*.json")) for t in json.loads(b.read_text())]
    check("lotti E alla cieca: domanda, soluzione con tutti i marcatori, lista; niente categorie",
          e_tasks and all(set(t) == {"id", "question", "solution", "list"} for t in e_tasks)
          and all(t["solution"].count("⟦") >= len(t["list"]) for t in e_tasks))
    again = run(ROOT / "scripts" / "f2_pilot.py", "export", "--selection", sel / "selected.v2.jsonl", "--mavis-shards", sdir / "*.tar",
                "--audit-map", amap, "--out", out, "--seed", "5")
    check("export: campione esistente non riscritto", again.returncode != 0 and "non è vuota" in again.stderr)

    def answer(prefix: str, judge) -> None:
        for b in sorted((out / "batches").glob(f"{prefix}-*.json")):
            (out / "answers" / b.name).write_text(json.dumps([judge(t) for t in json.loads(b.read_text())]))

    first_comp = []

    def classify(t: dict) -> dict:  # E finto: un fatto per frase, la frase stessa; ipotesi e scarti dalle prime parole
        parts = dict(re.findall(r"⟦(s\d{3})⟧ (.*?)(?=⟦s\d{3}⟧|$)", t["solution"], re.S))
        gold = {g["g"]: g for g in json.loads((out / "_gold" / "sentences.json").read_text())[t["id"].rsplit("-", 1)[0]]}
        sents = []
        for s in t["list"]:
            text = re.sub(r"</?think>", "", parts[s]).strip()
            if gold[s]["complement"]:
                facts = []
                if not first_comp:  # una frase scartata dal generatore con un fatto: la copertura lo deve contare
                    first_comp.append(s)
                    facts = [{"quote": text[:20], "statement": text, "kind": "existence", "basis": "given", "stance": "asserted"}]
            else:
                stance = "hypothesis" if text.startswith("Maybe") else "rejected" if text.startswith("Suppose") else "asserted"
                facts = [{"quote": text[:60].rstrip("."), "statement": text, "kind": "measure" if re.search(r"\d", text) else "construction",
                          "basis": "given", "stance": stance, "used_in_answer": False}]
                for q, st in (("AB = 5", "AB has length 5"), ("AD = 3", "Side AD is 3 units long")):
                    if q in text:  # misure come fatti a sé: uguali alla didascalia, vanno alla scorciatoia del codice
                        facts.append({"quote": q, "statement": st, "kind": "measure", "basis": "given", "stance": "asserted"})
            sents.append({"s": s, "facts": facts})
        return {"id": t["id"], "sentences": sents}

    # prima un lotto E incompleto: va rifatto e nessun lotto VA parte
    answer("E", classify)
    b0 = sorted((out / "batches").glob("E-*.json"))[0]
    broken = json.loads((out / "answers" / b0.name).read_text())
    broken[0]["sentences"] = broken[0]["sentences"][1:]
    (out / "answers" / b0.name).write_text(json.dumps(broken))
    r = run(ROOT / "scripts" / "f2_pilot.py", "round", "--dir", out)
    check("turno: lotto E con un marcatore mancante da rifare, niente VA", r.returncode == 0 and "DA RIFARE E-01" in r.stdout
          and not list((out / "batches").glob("VA-*.json")), r.stdout[-300:] + r.stderr[-300:])
    first_comp.clear()
    answer("E", classify)
    r = run(ROOT / "scripts" / "f2_pilot.py", "round", "--dir", out)
    va_tasks = [t for b in (out / "batches").glob("VA-*.json") for t in json.loads(b.read_text())]
    check("turno: lotti VA e VB alla cieca (niente soluzione, famiglia, attese o fonte)", r.returncode == 0 and va_tasks
          and all(set(t) == {"id", "image", "claims"} for t in va_tasks)
          and all(set(c) == {"cid", "statement", "quote", "context", "basis"} for t in va_tasks for c in t["claims"]), r.stderr[-500:])
    units = json.loads((out / "_gold" / "units.json").read_text())
    sentences = json.loads((out / "_gold" / "sentences.json").read_text())
    check("affermazioni: fatti uguali uniti, scorciatoie (frasi e fatti) e controlli sugli assi mescolati; i fatti del "
          "codice fuori dal campione non hanno cid",
          any(u["source"] == "shortcut" for us in units.values() for u in us)
          and any(u["source"] == "shortcut_fact" for us in units.values() for u in us)
          and all(u["cid"] is None for us in units.values() for u in us if u["source"] == "code")
          and sum(u["source"] == "frame" for us in units.values() for u in us) == 4)
    fact_spans_of = {}
    for pid, us in units.items():
        for u in us:
            if not u["cid"]:
                continue
            spans = []
            for fid in u.get("facts", []):
                g = next(g for g in sentences[pid] if g["g"] == fid.split(".")[0])
                spans += [tuple(x) for x in g["spans"]]
            fact_spans_of[(pid, u["cid"])] = spans
    first_real = sorted(p for p, it in items.items() if it["kind"] == "real")[0]

    def verify(who: str):
        def judge(t: dict) -> dict:
            pid = t["id"].rsplit("-", 1)[0]
            tr = truth.get(pid, {})
            out_claims = []
            for c in t["claims"]:
                u = next(u for u in units[pid] if u["cid"] and u["cid"] == c["cid"])
                hit = "start" in tr and any(a < tr["end"] and b > tr["start"] for a, b in fact_spans_of[(pid, c["cid"])])
                cfg, page = "true", "matches" if "horizontal" in c["statement"] else "none"
                if hit and tr["family"] == "orientation":
                    page = "contradicts"
                elif hit or u.get("family") == "frame_broken" or c["statement"].startswith(("Maybe", "Suppose")):
                    cfg = "false"
                if who == "VB" and pid == first_real and c["statement"].startswith("So the area"):
                    cfg = "false"  # un disaccordo, risolto da VC
                out_claims.append({"cid": c["cid"], "config": cfg, "page": page, "image_shows": ""})
            return {"id": t["id"], "claims": out_claims}
        return judge

    answer("VA", verify("VA"))
    answer("VB", verify("VB"))
    r = run(ROOT / "scripts" / "f2_pilot.py", "round", "--dir", out)
    vc = [t for b in (out / "batches").glob("VC-*.json") for t in json.loads(b.read_text())]
    check("turno: VC con il disaccordo e il ricontrollo", r.returncode == 0 and any(
        t["id"].startswith(first_real) and any(c["statement"].startswith("So the area") for c in t["claims"]) for t in vc), r.stderr[-300:])
    answer("VC", verify("VA"))
    r = run(ROOT / "scripts" / "f2_pilot.py", "final", "--dir", out)
    m = json.loads((out / "pilot_metrics.json").read_text()) if r.returncode == 0 else {}
    cal = m.get("calibration", {})
    check("metriche: recall dei candidati, della classificazione e sensibilità per le famiglie certe",
          all(cal[f]["candidate_recall"]["k"] == 2 and cal[f]["classification_recall"]["k"] == 2
              and cal[f]["verifier_sensitivity"]["k"] == 2 for f in ("measure", "construction", "configuration")),
          (r.stderr[-600:], {f: (cal.get(f, {}).get("candidate_recall"), cal.get(f, {}).get("verifier_sensitivity")) for f in cal}))
    check("metriche: orientamento (verità non garantita) trovato come orientamento contraddetto, a parte",
          cal["orientation"]["end_to_end"]["k"] == 2 and cal["orientation"]["certain"] is False)
    check("metriche: controlli sugli assi e scorciatoia", m["frame_controls"]["legit"] == {"n": 2, "false_alarms": 0, "quarantined": 0}
          and m["frame_controls"]["broken"]["sensitivity"]["k"] == 2 and m["shortcut"]["checks"] >= 1
          and m["shortcut"]["disagreement"]["k"] == 0 and m["shortcut"]["facts_by_code_only"] > 0
          and m["shortcut"]["facts_checked"] > 0, (m["frame_controls"], m["shortcut"]))
    rc = m["real"]["categories"]
    check("metriche: ipotesi aperte e scartate non sono errori visivi; nessun falso allarme sui negativi",
          m["real"]["with_visual_error"] == 0 and rc.get("open_hypothesis") == 4 and rc.get("discarded_hypothesis") == 4
          and m["negatives"]["items_with_false_alarm"] == 0, (rc, m["negatives"]))
    check("metriche: copertura del generatore dal campione alla cieca, disaccordo, regressione a parte",
          m["generator"]["complement_with_asserted_fact"]["k"] == 1 and m["agreement"]["va_vb_disagreement"]["k"] >= 1
          and m["regression"]["items"] == 1, (m["generator"]["complement_with_asserted_fact"], m["agreement"]))
    check("criteri: manca solo la copertura (una frase scartata con un fatto)",
          [k for k, v in m["criteria"]["passed"].items() if not v] == ["complement_fact_rate"]
          and "non certificazione" in (out / "pilot_report.md").read_text(), m["criteria"]["passed"])
    saved = (out / "manifest.json").read_text()
    man["frozen"]["src/limite_vl/f2_claims.py"] = "0" * 64
    (out / "manifest.json").write_text(json.dumps(man))
    r = run(ROOT / "scripts" / "f2_pilot.py", "final", "--dir", out)
    check("codice cambiato dopo l'export: round e final si fermano", r.returncode != 0 and "congelati" in r.stderr, r.stderr[-200:])
    (out / "manifest.json").write_text(saved)


if __name__ == "__main__":
    test_measures()
    test_asserted()
    test_injections()
    test_configuration()
    test_frame_controls()
    test_sentences()
    test_facts_resolution()
    tmp = ROOT / "outputs" / "tmp" / f"test-f2-pilot-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        test_pipeline(tmp)
    finally:
        shutil.rmtree(tmp)
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
