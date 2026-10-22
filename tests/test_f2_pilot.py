#!/usr/bin/env python3
"""Test CPU del pilota dell'audit F2 v2: limite_vl.f2_claims (misure della didascalia e della soluzione, alterazioni con
verità nota, controlli negativi, citazioni, risoluzione per affermazione) e un giro completo di scripts/f2_pilot.py su
fixture in stile MAVIS (export, risposte simulate di E, VA, VB e VC, metriche). Nessun agente, nessun dato vero."""
from __future__ import annotations

import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image  # noqa: E402

from limite_vl.f2_claims import (caption_measures, caption_negative, context_sentence, cot_negative,  # noqa: E402
                                 inject_construction, inject_measure, inject_orientation, quote_spans, resolve_claim,
                                 shared_edges, solution_measures)
from limite_vl.mavis import f2_problem  # noqa: E402
from limite_vl.provenance import sha256_text  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


CAPTION = "ABCD is a square. CBEF is a square. Side AB is 5 units long. Angle ABC measures 90°."


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
    # falsi positivi visti sul campione vero (job 29861): angoli letti come lati, frazioni con radice, catene
    cap2 = "The distance of side FE is 10*√(3)/3. Side DC is 12 units long. Side AB is 7 units long. Side BD is 7 units long."
    check("didascalia: frazione con radice", abs(caption_measures(cap2)[("seg", "EF")] - 10 * 3 ** 0.5 / 3) < 1e-9)
    sol2 = ("<think>So the angle between FE and FH is 90°, and angle between side BC and side BA is 45 degrees. "
            "FE = 10√3/3. Here r = half of EF = half of DC = 6. So AD = AB + BD = 14. Also DC = 12.</think>")
    got = [(d["key"], d["verdict"]) for d in solution_measures(sol2, cap2)]
    check("soluzione: angoli fra due lati non sono lunghezze, frazioni con radice, valori in catena non contano",
          got == [("seg-EF", "match"), ("seg-CD", "chained"), ("seg-CD", "match")], got)  # BD dopo "+": espressione, saltata
    cap3 = "Side AB is 1 units long. Side BC is 2 units long. Side FG is 8 units long. Side EC is 84*√(3). Angle ABC measures 45°."
    sol3 = ("<think>AB * BC = 1 * √2 = √2. So (√3/2) * FG = 4. Then 180 - angle ABC = 135°. Vector EC = (84, +84√2). "
            "Also FC is long and EC = 84\\sqrt3. Given AB = 1, BC = 2.</think>")
    got = [(d["key"], d["verdict"]) for d in solution_measures(sol3, cap3)]
    check("soluzione: prodotti, differenze e vettori non sono misure; \\sqrt3 senza graffe",
          got == [("seg-CE", "match"), ("seg-AB", "match"), ("seg-BC", "match")], got)
    check("orientamento: mai 'right angle' né lontano da un elemento della figura",
          inject_orientation("<think>There is a right angle at A. The perimeter is as below.</think>", random.Random(0)) is None
          and inject_orientation("<think>Here AB is horizontal.</think>", random.Random(0)) is not None)
    cap4 = "ABCD is a square. DCEF is a rectangle. Side AB is 5 units long."
    t4, i4 = inject_construction("<think>Since they are adjacent, side DC is a side of rectangle DCEF.</think>", cap4, random.Random(0))
    check("costruzione: il lato nuovo non appartiene alla forma nominata subito dopo",
          "".join(sorted(t4[i4["start"]:i4["end"]])) in {"AB", "BC", "AD"}, i4)
    check("costruzione: mai su una frase interrogativa", inject_construction(
        "<think>Square CBEF attached to side CB? Actually yes.</think>", CAPTION, random.Random(0)) is None)


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
    shared, edges = shared_edges(CAPTION)
    check("lati in comune fra le forme", shared == {"BC"} and sorted(edges) == ["ABCD", "CBEF"], (shared, edges))
    t, info = inject_construction(sol, CAPTION, rng)
    new = t[info["start"]:info["end"]]
    check("costruzione cambiata: un lato non in comune, verità nota", info["original"] == "CB" and "".join(sorted(new)) not in shared
          and "".join(sorted(new)) in {e for es in edges.values() for e in es}, info)
    check("costruzione: senza parola di attacco nessuna alterazione", inject_construction("<think>side CB = 5</think>", CAPTION, rng) is None)
    t, info = inject_orientation(sol, random.Random(0))
    check("orientamento scambiato, verità dichiarata non garantita", t[info["start"]:info["end"]] == "vertical"
          and info["truth"] == "not guaranteed")
    quoted = '<think>It says: "Figure: AB is horizontal."</think>'
    check("orientamento: mai dentro la citazione del prompt", inject_orientation(quoted, rng) is None)
    neg = caption_negative(CAPTION, "A")
    check("controllo negativo dalla didascalia: tutte le misure uguali", all(d["verdict"] == "match" for d in solution_measures(neg, CAPTION))
          and sum(d["verdict"] == "match" for d in solution_measures(neg, CAPTION)) == 2)
    check("controllo negativo dal CoT: ammesso solo se coerente con la didascalia",
          cot_negative("Angle ABC = 90° and AB = 5, so the area is 25.", CAPTION, "A") is not None
          and cot_negative("AB = 6, so the area is 36. Angle ABC = 90°.", CAPTION, "A") is None
          and cot_negative("The area is 25.", CAPTION, "A") is None)


def test_quotes_resolution():
    text = "<think>Place \\(D\\) at the origin. The side \\(DF\\) is vertical, so DF = 12.\nNext line.</think>"
    sp = quote_spans(text, "the side DF is vertical")
    check("citazione: span nel testo originale", len(sp) == 1 and text[sp[0][0]:sp[0][1]].lower().startswith("the side"), sp)
    check("citazione che non c'è: nessuno span", quote_spans(text, "DF is horizontal") == [])
    ctx = context_sentence(text, sp[0])
    check("frase di contesto", ctx.startswith("The side") and ctx.endswith("DF = 12."), ctx)
    check("risoluzione: consenso", resolve_claim(["false", "not_visible"]) == {"status": "resolved", "value": "E", "how": "consenso"})
    check("risoluzione: disaccordo chiede il terzo", resolve_claim(["true", "false"])["status"] == "need_blind")
    check("risoluzione: maggioranza di tre", resolve_claim(["true", "false", "deduction"])["value"] == "K")
    check("risoluzione: tre senza maggioranza in quarantena", resolve_claim(["true", "false", "unclear"])["status"] == "quarantine")


def run(*args) -> subprocess.CompletedProcess:
    env = dict(os.environ, SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID", "test"))
    return subprocess.run([sys.executable, "-B", *map(str, args)], env=env, capture_output=True, text=True)


def png(i: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64 + i, 48), "white").save(buf, format="PNG")
    return buf.getvalue()


def meta(i: int) -> dict:
    return {"id": f"d{i % 4}-{i}", "depth": i % 4, "choices": ["25", "20", "30", "35"], "answer_index": 0,
            "text_en": {"caption": CAPTION, "diagram_generation": "", "vision_dominant_question": "What is the area of CBEF?",
                        "CoT_reasoning": "Angle ABC = 90° and side AB is 5, so the area of CBEF is 25."}}


def test_pipeline(tmp: Path):
    metas = [meta(i) for i in range(1, 17)]
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
        sols = [f"<think>The figure shows square ABCD with AB = 5. Square CBEF is attached to side CB of ABCD. The side AB is "
                f"horizontal. Angle ABC = 90°. So the area is 25. Variante {c}.</think> The area is \\boxed{{25}}." for c in "abc"]
        rows.append({"key": p["key"], "source_dataset": f"MAVIS-Geometry d{m['depth']}", "answer": p["answer_text"],
                     "answer_letter": p["answer_letter"], "qwen4b_pass_rate": None, "prompt_tokens": 40, "question": p["question"],
                     "question_sha256": sha256_text(p["question"]), "prompt_sha256": "0" * 64,
                     "completions": [{"text": s, "finish_reason": "stop", "tokens": 30} for s in sols]})
    (gen / "part-00000.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sel = tmp / "sel"
    r = run(ROOT / "scripts" / "f2_select.py", "--gen", gen, "--out", sel, "--keep", "2", "--workers", "1")
    recs = [json.loads(x) for x in (sel / "selected.v2.jsonl").read_text().splitlines()] if r.returncode == 0 else []
    check("fixture: selezione v2 con le domande", len(recs) == 32, r.stderr[-400:])
    amap = tmp / "audit_map.jsonl"
    audited = [x for x in recs if int(x["key"].rsplit("-", 1)[1]) <= 8]
    amap.write_text("".join(json.dumps({"fid": f"f{i:03d}", "audit": "finto", "sid": x["sid"], "status": "exact",
                                        "verdict": "include" if i % 2 else "exclude"}) + "\n" for i, x in enumerate(audited, 1)))
    out = tmp / "pilot"
    r = run(ROOT / "scripts" / "f2_pilot.py", "export", "--selection", sel / "selected.v2.jsonl", "--mavis-shards", sdir / "*.tar",
            "--audit-map", amap, "--out", out, "--seed", "5", "--real-per-depth", "1", "--measure", "2", "--construction", "2",
            "--orientation", "2", "--caption-negatives", "1", "--cot-negatives", "1")
    check("export del pilota", r.returncode == 0, r.stderr[-800:])
    if r.returncode:
        return
    man = json.loads((out / "manifest.json").read_text())
    items = {it["pid"]: it for it in json.loads((out / "_gold" / "items.json").read_text())}
    truth = json.loads((out / "_gold" / "truth.json").read_text())
    measures = json.loads((out / "_gold" / "measures.json").read_text())
    check("campione congelato: 4 vere (una per profondità, fra le auditate), 6 alterazioni, 2 negativi",
          man["design"]["obtained"] == {"real": 4, "measure": 2, "construction": 2, "orientation": 2, "caption_negative": 1,
                                        "cot_negative": 1}
          and sorted(it["depth"] for it in items.values() if it["kind"] == "real") == [0, 1, 2, 3]
          and all(it["v1_verdict"] for it in items.values() if it["kind"] == "real"), man["design"]["obtained"])
    check("campione: basi della calibrazione e negativi fuori dalle auditate e da problemi distinti",
          len({it["key"] for it in items.values()}) == len(items)
          and not {it["source_sid"] for it in items.values() if it["source_sid"]} & {x["sid"] for x in audited})
    check("campione: criteri di riuscita fissati prima delle risposte e istruzioni con sha256",
          "criteria" in man and man["instructions"]["E"]["path"].endswith("f2_extract.md"))
    m_pids = [p for p, t in truth.items() if t["family"] == "measure"]
    check("canale deterministico: ogni misura cambiata è una contraddizione per il codice",
          all(any(d["verdict"] == "contradiction" and d["start"] == truth[p]["start"] for d in measures[p]) for p in m_pids)
          and all(d["verdict"] != "contradiction" for p, it in items.items() if it["kind"] != "calibration" for d in measures[p]))
    again = run(ROOT / "scripts" / "f2_pilot.py", "export", "--selection", sel / "selected.v2.jsonl", "--mavis-shards", sdir / "*.tar",
                "--audit-map", amap, "--out", out, "--seed", "5")
    check("export: campione esistente non riscritto", again.returncode != 0 and "non è vuota" in again.stderr)

    def answer(prefix: str, judge) -> None:
        for b in sorted((out / "batches").glob(f"{prefix}-*.json")):
            (out / "answers" / b.name).write_text(json.dumps([judge(t) for t in json.loads(b.read_text())]))

    def extract(t: dict) -> dict:  # E finto: ogni frase del ragionamento è un'affermazione, citata alla lettera
        think = t["solution"].split("<think>", 1)[1].split("</think>")[0]
        sents = [s.strip() for s in think.replace("From the figure:", "").split(". ") if s.strip()]
        claims = [{"quote": s.rstrip("."), "statement": s, "kind": "measure" if any(ch.isdigit() for ch in s) else "construction"}
                  for s in sents]
        if t["id"] == sorted(items)[0]:
            claims.append({"quote": "una frase che non c'è", "statement": "x", "kind": "measure"})
        return {"id": t["id"], "claims": claims}

    answer("E", extract)
    r = run(ROOT / "scripts" / "f2_pilot.py", "round", "--dir", out)
    check("turno: lotti VA e VB dalle affermazioni valide, senza la soluzione", r.returncode == 0 and list((out / "batches").glob("VA-*.json"))
          and all("solution" not in t for b in (out / "batches").glob("VA-*.json") for t in json.loads(b.read_text())), r.stderr[-500:])
    sys.path.insert(0, str(ROOT / "scripts"))
    from f2_pilot import Pilot

    claims = Pilot(out).claims()
    bad = {(p, c["cid"]) for p, t in truth.items() if "start" in t for c in claims[p] if c["valid"]
           and any(s < t["end"] and e > t["start"] for s, e in c["spans"])}
    first_real = sorted(p for p, it in items.items() if it["kind"] == "real")[0]

    def verify(who: str):
        def judge(t: dict) -> dict:
            out_claims = []
            for c in t["claims"]:
                v = "false" if (t["id"], c["cid"]) in bad else "true"
                if who == "VB" and t["id"] == first_real and c["cid"] == "c01":
                    v = "false"  # un disaccordo, risolto da VC
                out_claims.append({"cid": c["cid"], "verdict": v, "image_shows": ""})
            return {"id": t["id"], "claims": out_claims}
        return judge

    answer("VA", verify("VA"))
    answer("VB", verify("VB"))
    r = run(ROOT / "scripts" / "f2_pilot.py", "round", "--dir", out)
    vc = [t for b in (out / "batches").glob("VC-*.json") for t in json.loads(b.read_text())]
    check("turno: VC con il disaccordo e il ricontrollo delle vere", r.returncode == 0 and any(
        t["id"] == first_real and any(c["cid"] == "c01" for c in t["claims"]) for t in vc) and sum(len(t["claims"]) for t in vc) >= 2,
        r.stderr[-300:])
    answer("VC", verify("VA"))
    r = run(ROOT / "scripts" / "f2_pilot.py", "final", "--dir", out)
    m = json.loads((out / "pilot_metrics.json").read_text()) if r.returncode == 0 else {}
    cal = m.get("calibration", {})
    check("metriche: recall dell'estrazione e sensibilità attribuibile per famiglia",
          all(cal[f]["extraction_recall"]["k"] == 2 and cal[f]["verifier_sensitivity"]["k"] == 2 for f in ("measure", "construction", "orientation")),
          (r.stderr[-500:], {f: (cal.get(f, {}).get("extraction_recall"), cal.get(f, {}).get("verifier_sensitivity")) for f in cal}))
    check("metriche: nessun falso allarme sui controlli negativi", m["negatives"]["items_with_false_alarm"] == 0
          and m["negatives"]["code_contradictions"] == 0)
    check("metriche: disaccordo, VC, nessuna quarantena, citazione non ritrovata contata",
          m["agreement"]["va_vb_disagreement"]["k"] == 1 and m["agreement"]["claim_quarantine"]["k"] == 0
          and m["extraction"]["invalid_quotes"] == 1, m.get("agreement"))
    check("metriche: costo per soluzione e criteri", m["cost"]["calls"]["E"] >= 1 and m["cost"]["calls_per_solution"] > 0
          and m["criteria"]["all_passed"] is True and "non certificazione" in (out / "pilot_report.md").read_text(), m.get("criteria"))


if __name__ == "__main__":
    test_measures()
    test_injections()
    test_quotes_resolution()
    tmp = ROOT / "outputs" / "tmp" / f"test-f2-pilot-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        test_pipeline(tmp)
    finally:
        shutil.rmtree(tmp)
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
