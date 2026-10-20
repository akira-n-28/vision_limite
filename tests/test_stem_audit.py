#!/usr/bin/env python3
"""Test dell'audit automatico STEM: regole di limite_vl.stem_audit (affermazioni, campionamento, intervalli, errori
introdotti, controllo dei programmi, risoluzione delle divergenze) e un giro completo di scripts/stem_audit_round.py
su una cartella finta (lotti, risposte simulate, terzi controlli, revisioni, arbitri, risultati finali)."""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from limite_vl import stem_audit as sa  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


CAPTION = """### Image 1: Line graph
**Scene Summary:** A line graph of distance over time.

**Explicit Component Breakdown:**
* **X-axis (`t`):** Horizontal axis with ticks at 8, 9, 10; the label is t.
* **Segment 1:** Rises from (8,0) to (10,8).

**Implicit and Inferred Properties:**
* The return trip is slower than the outbound trip.

**Identified Ambiguities:**
* None."""


def test_rules():
    u = sa.claim_units(CAPTION)
    check("affermazioni: titolo dell'immagine e 'None' tolti", all("Image 1" not in x["text"] and x["text"] != "None." for x in u), u)
    check("affermazioni: punto e virgola divide", any(x["text"] == "the label is t." for x in u), [x["text"] for x in u])
    check("affermazioni: sezioni e visibilità", [x["section"] for x in u] == ["summary", "components", "components", "components", "inferred"]
          and [x["visible"] for x in u] == [True, True, True, True, False], [(x["section"], x["visible"]) for x in u])
    al = sa.allocate({"a": 5, "b": 100, "c": 100}, 30)
    check("allocazione: sorgente piccola intera, resto diviso", al["a"] == 5 and sum(al.values()) == 30 and abs(al["b"] - al["c"]) <= 1, al)
    lo, hi = sa.clopper_pearson(0, 368)
    check("Clopper-Pearson 0/368 ha limite superiore circa 1%", lo == 0 and 0.0095 < hi < 0.0101, hi)
    lo, hi = sa.clopper_pearson(5, 100)
    check("Clopper-Pearson 5/100", abs(lo - 0.0164) < 0.002 and abs(hi - 0.1128) < 0.002, (lo, hi))
    cr = sa.clustered_rate([(0, 20)] * 10)
    check("tasso raggruppato senza errori: caso peggiore, una osservazione per problema", cr["deff_rule"] == "caso peggiore"
          and abs(cr["ci"][1] - sa.clopper_pearson(0, 10)[1]) < 1e-9, cr)
    cr = sa.clustered_rate([(2, 20)] * 5 + [(0, 20)] * 5)
    check("tasso raggruppato con errori: deff stimato", cr["deff_rule"] == "stimato" and cr["deff"] >= 1 and cr["rate"] == 0.05, cr)
    check("correzione per la sensibilità", sa.corrected_upper(0.01, 0.5) == 0.02 and sa.corrected_upper(0.01, 0) == 1.0)
    rf = sa.resolve_field
    check("campo: consenso", rf(["E", "E"], [])["how"] == "consenso")
    check("campo: disaccordo chiede un terzo giudizio", rf(["E", "K"], [])["status"] == "need_blind")
    check("campo: poi un arbitro", rf(["E", "K", "E"], [])["status"] == "need_arbiter")
    check("campo: arbitro che conferma la maggioranza", rf(["E", "K", "E"], ["E"])["value"] == "E")
    check("campo: arbitro contro la maggioranza chiede R2", rf(["E", "K", "E"], ["K"])["status"] == "need_arbiter")
    check("campo: due arbitri concordi", rf(["E", "K", "E"], ["K", "K"])["value"] == "K")
    check("campo: arbitri discordi in quarantena", rf(["E", "K", "E"], ["K", "E"])["status"] == "quarantine")
    check("campo: controllo del 10% concorde", rf(["K", "K", "K"], [])["how"] == "consenso+controllo")
    check("campo: incerti chiedono altri giudizi", rf([None, None], [])["status"] == "need_blind")
    rg = sa.resolve_gold
    check("risposta: ritrovata da due solutori", rg(["match", "match"], [])["value"] == "correct")
    check("risposta: un solutore diverso chiede revisioni", rg(["match", "mismatch"], [])["status"] == "need_review")
    check("risposta: duplicato in conflitto chiede revisioni", rg(["match", "match"], [], force_review=True)["status"] == "need_review")
    check("risposta: revisioni concordi", rg(["mismatch", "unsolved"], ["wrong", "wrong"])["value"] == "wrong")
    check("risposta: revisioni discordi chiedono l'arbitro", rg(["mismatch", "match"], ["wrong", "correct"])["status"] == "need_arbiter")
    check("risposta: arbitro", rg(["mismatch", "match"], ["wrong", "correct", "correct"])["value"] == "correct")
    check("risposta: tre solutori e revisioni discordi in quarantena",
          rg(["match", "match", "mismatch"], ["wrong", "correct"])["status"] == "quarantine")
    rng = random.Random(1)
    m = sa.mutate_text("Side AB is 7 units long.", rng)
    check("errore introdotto: numero cambiato", m and m[0] != "Side AB is 7 units long." and "7" not in m[0].split()[3], m)
    m = sa.mutate_text("The arrow points to the left.", random.Random(2))
    check("errore introdotto: direzione opposta", m and "right" in m[0], m)
    check("errore introdotto: niente da cambiare", sa.mutate_text("A circle with center O.", rng) is None)
    check("risposta sbagliata: altra lettera", sa.wrong_answer("B", ["A", "B", "C"], rng) in ("A", "C"))
    w = sa.wrong_answer("12", [], rng)
    check("risposta sbagliata: altro numero", w is not None and w != "12", w)
    check("risposta sbagliata: testo libero non si cambia", sa.wrong_answer("isosceles", [], rng) is None)
    check("scelta multipla: opzioni e lettera", sa.mc_options("Find x.\nA. 1\nB. 2\nC. 3") == ["A", "B", "C"]
          and sa.answer_letter("\\text{(B)}") == "B" and sa.mc_options("Find x if x + 2 = 5.") == [])
    check("programma ammesso", sa.program_problems("import sympy\nresult = sympy.sqrt(a**2 + b**2)") == [])
    check("programma: import vietato", "import os" in sa.program_problems("import os\nresult = 1"))
    check("programma: attributo privato", any("attributo" in p for p in sa.program_problems("result = ().__class__")))
    check("programma: manca result", "manca result" in sa.program_problems("x = 1"))
    check("numeri non dichiarati", sa.undeclared_literals("result = a * 7 + 2") == [7])
    check("input dalla domanda", sa.input_in_question("3/4", "A ratio of 3 to 4") and not sa.input_in_question("5", "a = 3"))
    check("classi di un'affermazione", sa.unit_classes([{"kind": "visible", "label": "verified"}, {"kind": "deduction", "label": "false"}])
          == {"vis": "K", "ded": "E"})
    eq = lambda x, y: x.replace(" ", "") == y.replace(" ", "")  # noqa: E731
    q = "Which is correct?\nA. 2\nB. 1\nC. 3"
    check("confronto: lettera minuscola e con contenuto", sa.match_answer("d", "D", "", eq) and sa.match_answer("c (cornea)", "C", "", eq))
    check("confronto: contenuto dell'opzione al posto della lettera", sa.match_answer("1", "B", q, eq) and not sa.match_answer("2", "B", q, eq))
    check("confronto: numero con unità", sa.match_answer("14 mi/h", "14", "", eq) and not sa.match_answer("(1) S=2", "2", "", eq))
    check("confronto: lettere diverse restano diverse", not sa.match_answer("A", "B", "", eq) and not sa.match_answer("a (x)", "D", "", eq))
    check("testo dell'opzione", sa.option_text(q, "C") == "3" and sa.option_text("no options", "A") is None)
    s1 = sa.recheck_sample({"a": [f"k{i}" for i in range(25)], "b": ["x"]}, 0.1, 3)
    check("controllo del 10%: per gruppo, almeno uno, ripetibile", len(s1) == 4 and "x" in s1
          and s1 == sa.recheck_sample({"a": [f"k{i}" for i in range(25)], "b": ["x"]}, 0.1, 3), s1)


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))


def facts(*labels):
    return [{"fact": "f", "kind": k, "label": lab, "reason": "r"} for k, lab in labels]


def test_round():
    import stem_audit_round as sr

    tmp = Path(tempfile.mkdtemp(prefix="test_stem_audit_", dir=os.environ.get("TMPDIR")))
    units = [{"cid": "c01", "section": "summary", "visible": True, "text": "A triangle."},
             {"cid": "c02", "section": "components", "visible": True, "text": "Side AB is 5."}]
    items = [{"pid": "p001", "key": "sft-S1-1", "source": "S1", "question": "What is 3 times 4?", "units": units, "image": "img/p001.png",
              "calibration": False, "tags": []},
             {"pid": "p002", "key": "sft-S1-2", "source": "S1", "question": "Find x.", "units": units, "image": "img/p002.png",
              "calibration": False, "tags": []},
             {"pid": "p003", "key": "cal-mavis-d0-1", "source": "_cal_mavis", "question": "Pick.\nA. 1\nB. 2", "units": units,
              "image": "img/p003.png", "calibration": True, "tags": []}]
    write(tmp / "_gold/items.json", items)
    write(tmp / "_gold/gold.json", {"p001": "12", "p002": "5", "p003": "B"})
    write(tmp / "_gold/calibration.json", {"p003": {"kind": "mavis", "gold_true": "B", "units_true": ["c01"],
                                                     "units_injected": {"c02": {"original": "Side AB is 4.", "changed": "Side AB is 5."}},
                                                     "legible": True, "coherent": True}})
    det = {p: {"image": {"ok": True, "flags": []}, "text": {"flags": [], "gold_wrong_by_code": False}, "contamination_candidate": False,
               "calibration": p == "p003"} for p in ("p001", "p002", "p003")}
    write(tmp / "deterministic.json", det)
    write(tmp / "manifest.json", {"args": {"seed": "5", "sources": "None"}, "population": 100, "units_injected": 1, "gold_injected": 0,
                                  "deterministic_flags": {}, "duplicate_conflicts": 0, "contamination_candidates": 0})
    write(tmp / "source_sizes.json", {"S1": 100})
    write(tmp / "population_checks.json", {"S1": {"righe": 100}})
    solve = lambda p: {"id": p, "image": "x", "question": "q", "check": ["solve", "legibility", "coherence"]}  # noqa: E731
    desc = lambda p: {"id": p, "image": "x", "units": [{"cid": "c01", "section": "summary", "text": "t"},  # noqa: E731
                                                      {"cid": "c02", "section": "components", "text": "t"}]}
    for r in ("SA", "SB"):
        write(tmp / f"batches/{r}-01.json", [solve(p) for p in ("p001", "p002", "p003")])
    for r in ("DA", "DB"):
        write(tmp / f"batches/{r}-01.json", [desc(p) for p in ("p001", "p002", "p003")])
    ok = {"legibility": "ok", "coherence": "ok"}
    prog = {"inputs": {"a": {"value": "3", "source": "question"}, "b": {"value": "4", "source": "question"}}, "code": "result = a * b"}
    write(tmp / "answers/SA-01.json", [{"id": "p001", "own_answer": "12", "program": prog, **ok},
                                       {"id": "p002", "own_answer": "5", **ok}, {"id": "p003", "own_answer": "B", **ok}])
    write(tmp / "answers/SB-01.json", [{"id": "p001", "own_answer": "12", **ok},
                                       {"id": "p002", "own_answer": "7", "legibility": "illegible", "coherence": "ok"},
                                       {"id": "p003", "own_answer": "B", **ok}])
    good = [{"cid": "c01", "facts": facts(("visible", "verified"))}, {"cid": "c02", "facts": facts(("visible", "verified"))}]
    write(tmp / "answers/DA-01.json", [{"id": "p001", "units": good}, {"id": "p002", "units": good},
                                       {"id": "p003", "units": [good[0], {"cid": "c02", "facts": facts(("visible", "false"))}]}])
    write(tmp / "answers/DB-01.json", [{"id": "p001", "units": good}, {"id": "p002", "units": good}, {"id": "p003", "units": good}])

    a = sr.Audit(tmp, 0.1, 15)
    t = a.next_tasks()
    sc = {x["id"]: x["check"] for x in t["SC"]}
    check("turno 2: terzo controllo della leggibilità contestata", sc.get("p002") == ["legibility"], sc)
    check("turno 2: controllo completo del 10% degli approvati", any(c == ["solve", "legibility", "coherence"] for c in sc.values()), sc)
    check("turno 2: revisioni della risposta non ritrovata", [x["id"] for x in t["G1"]] == ["p002"] and t["G1"][0]["other_answers"] == ["7"], t["G1"])
    dc = {x["id"]: [u["cid"] for u in x["units"]] for x in t["DC"]}
    check("turno 2: terzo controllo dell'affermazione contestata", dc.get("p003") == ["c02"], dc)
    g1 = a.gold_state("p001")
    check("codice: programma con input dalla domanda verifica la risposta", g1["evidence"] == "codice (solo testo della domanda)", g1["programs"])
    a.write_tasks(t)
    a.save_code()
    sc_ans = []
    for x in t["SC"]:
        if x["check"] == ["legibility"]:
            sc_ans.append({"id": x["id"], "legibility": "illegible"})
        else:
            sc_ans.append({"id": x["id"], "own_answer": {"p001": "12", "p003": "B"}[x["id"]], **ok})
    write(tmp / "answers/SC-01.json", sc_ans)
    write(tmp / "answers/G1-01.json", [{"id": "p002", "verdict": "correct", "answer_coherent": "yes", "note": "n"}])
    write(tmp / "answers/G2-01.json", [{"id": "p002", "verdict": "correct", "answer_coherent": "yes", "note": "n"}])
    write(tmp / "answers/DC-01.json", [{"id": x["id"], "units": [{"cid": u["cid"], "facts": facts(("visible", "false" if x["id"] == "p003" else "verified"))}
                                                                for u in x["units"]]} for x in t["DC"]])

    a = sr.Audit(tmp, 0.1, 15)
    t = a.next_tasks()
    r1 = {x["id"]: [d["field"] for d in x["disputes"]] for x in t["R1"]}
    check("turno 3: arbitri sulle maggioranze", r1.get("p002") == ["legibility"] and r1.get("p003") == ["c02:vis"], r1)
    check("turno 3: niente terzi controlli ripetuti", not t["SC"] and not t["DC"], {k: len(v) for k, v in t.items()})
    a.write_tasks(t)
    write(tmp / "answers/R1-01.json", [{"id": "p002", "disputes": [{"field": "legibility", "verdict": "illegible", "note": "n"}]},
                                       {"id": "p003", "disputes": [{"field": "c02:vis", "verdict": "false", "note": "n"}]}])

    a = sr.Audit(tmp, 0.1, 15)
    t = a.next_tasks()
    check("turno 4: niente più da fare", not any(t.values()), {k: len(v) for k, v in t.items()})
    sr.final(a, tmp)
    rows = {json.loads(line)["pid"]: json.loads(line) for line in (tmp / "audit.jsonl").open()}
    check("conclusione: problema valido per RL e F2", rows["p001"]["rl"] == "include" and rows["p001"]["f2"] == "include", rows["p001"]["reasons"])
    check("conclusione: figura illeggibile esclusa", rows["p002"]["rl"] == "exclude", rows["p002"]["reasons"])
    check("conclusione: descrizione falsa esclude solo F2", rows["p003"]["f2"] == "exclude", rows["p003"]["reasons"])
    cal = json.loads((tmp / "calibration_summary.json").read_text())
    check("calibrazione: errore introdotto trovato dalla procedura", cal["affermazione alterata"].get("procedura_trovata") == 1, cal)
    check("calibrazione: affermazione vera senza falso allarme", cal["affermazione vera (MAVIS)"].get("procedura_giusta") == 1, cal)
    report = (tmp / "audit_report.md").read_text()
    check("report con affidabilità degli agenti e livelli di evidenza", "Affidabilità degli agenti" in report and "**codice**" in report)
    csv_text = (tmp / "audit_by_source.csv").read_text()
    check("statistiche per sorgente senza la calibrazione", "S1" in csv_text and "_cal_mavis" not in csv_text)
    check("liste proposte", (tmp / "include_rl.txt").read_text() == "sft-S1-1\n" and "sft-S1-2" in (tmp / "exclude.txt").read_text())


def test_program_sandbox():
    import stem_audit_round as sr

    r = sr.run_program({"inputs": {"r": {"value": "2", "source": "image"}}, "code": "import sympy\nresult = sympy.pi * r**2"})
    check("programma eseguito con sympy", r.get("ok") and r["result"] == "4*pi", r)
    r = sr.run_program({"inputs": {}, "code": "import os\nresult = 1"})
    check("programma con import vietato non eseguito", not r.get("ok") and "import os" in r["error"], r)
    r = sr.run_program({"inputs": {"x": {"value": "open(1)", "source": "image"}}, "code": "result = x"})
    check("input con nomi vietati rifiutato", not r.get("ok"), r)
    r = sr.run_program({"inputs": {}, "code": "while True:\n    pass\nresult = 1"})
    check("programma che non termina fermato", not r.get("ok"), r)


if __name__ == "__main__":
    test_rules()
    test_round()
    test_program_sandbox()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
