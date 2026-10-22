#!/usr/bin/env python3
"""Test CPU dei record F2 v2: src/limite_vl/f2_records.py, src/limite_vl/provenance.py, scripts/f2_backfill.py e la
selezione v2 di scripts/f2_select.py. Solo libreria standard e fixture minuscole (risposte confrontabili come testo,
senza math_verify): gira in slurm/tests_cpu.sbatch."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from f2_backfill import is_injection, match_key  # noqa: E402

from limite_vl.f2_records import (CERT_SCHEMA, admit, association_ok, check_failures, compare_selections,  # noqa: E402
                                  duplicate_sids, eligible, make_sid, parse_sid, policy_open, read_selected, record_policy,
                                  record_version, solution_checks, solution_record, validate_certificate, validate_record)
from limite_vl.provenance import HashingTextWriter, sha256_file, sha256_text, write_new_json  # noqa: E402
from limite_vl.stem_audit import clopper_pearson, corrected_upper  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def record(key="p1", completion=0, solution="<think>2+2</think>\\boxed{4}", gen_run="gen-a", **kw) -> dict:
    args = dict(gen_run=gen_run, key=key, completion=completion, solution=solution, raw_text=solution,
                finish_reason="stop", tokens=10, source_dataset="S", answer="4", rewritten=False,
                violetto_pass_rate=0.5, origin="select", question="What is 2+2?")
    args.update(kw)
    return solution_record(**args)


TEST_POLICY = {"schema": "f2-policy/1", "thresholds": {"solution_visual": 0.05}, "frame_counts_as_error": True,
               "flag_policy": {"prompt_quote": "keep", "text_before_think": "keep"}, "calibration": {"required_kinds": ["measure"]}}


def certificate(selection_sha: str, decision: str = "certified", audited=(), excluded=(), quarantined=(),
                flags=None) -> dict:
    """Certificato di prova coerente anche nei numeri, solo in memoria (quelli veri li scrive scripts/f2_certify.py):
    0 errori su 300 soluzioni, sensibilità 60/60 sulle misure, limite corretto sotto la soglia del 5%."""
    policy = {**TEST_POLICY, "flag_policy": flags or TEST_POLICY["flag_policy"]}
    entries = [list(a) if len(a) > 2 else [*a, "include"] for a in audited]
    entries += [[*e[:2], "exclude"] for e in excluded] + [[*q[:2], "quarantine"] for q in quarantined]
    s_lo = clopper_pearson(60, 60)[0]
    lo, hi = clopper_pearson(0, 300)
    verdicts = {v: sum(e[2] == v for e in entries) for v in ("include", "exclude", "quarantine")}
    return {"schema": CERT_SCHEMA, "population": {"selection_sha256": selection_sha, "solutions": 3000},
            "decision": decision, "reasons": [], "stale": [], "retroactive": False,
            "audited": entries, "excluded": [list(e[:2]) for e in excluded], "quarantined": [list(q[:2]) for q in quarantined],
            "audit": {"name": "prova", "manifest_sha256": "0" * 64, "version": 2},
            "policy": {"content": policy, "open": policy_open(policy)}, "thresholds": dict(policy["thresholds"]),
            "record_policy": record_policy(policy),
            "rates": {"solution_visual": {"k": 0, "n": 300, "rate": 0.0, "lo": lo, "hi": hi},
                      "verdicts": {k: v for k, v in verdicts.items() if v}, "sample_in_population": len(entries)},
            "sensitivity": {"measure": {"k": 60, "n": 60, "rate": 1.0, "lo": s_lo, "hi": 1.0}},
            "corrected_upper": {"solution_visual": corrected_upper(hi, s_lo), "sensitivity_lower": s_lo}}


def test_ids():
    check("sid: andata e ritorno", parse_sid(make_sid("f2-mavis2k", "mavis-d1-37894", 2)) == ("f2-mavis2k", "mavis-d1-37894", 2))
    check("sid: chiave con '/' ricostruita", parse_sid(make_sid("g", "a/b/c", 0)) == ("g", "a/b/c", 0))
    check("sid: valori non validi rifiutati",
          raises(lambda: make_sid("a/b", "k", 0), ValueError) and raises(lambda: make_sid("g", "k", -1), ValueError)
          and raises(lambda: make_sid("g", "k", True), ValueError) and raises(lambda: make_sid("g", "", 0), ValueError)
          and raises(lambda: parse_sid("g/k/x"), ValueError) and raises(lambda: parse_sid("solo"), ValueError))
    a, b = record(completion=0), record(completion=1)  # stesso problema, stesso testo
    check("due soluzioni dello stesso problema: sid diversi anche con testo identico",
          a["sid"] != b["sid"] and a["text_sha256"] == b["text_sha256"])
    check("un esito dato a una soluzione non vale per l'altra dello stesso problema",
          association_ok(a, a["sid"], a["text_sha256"]) and not association_ok(b, a["sid"], a["text_sha256"]))
    check("sid ripetuti trovati", duplicate_sids([a, b, a]) == [a["sid"]] and duplicate_sids([a, b]) == [])
    check("record coerente", validate_record(a) == [] and record_version(a) == 2)


def test_hash_invalidation():
    r = record()
    sid, sha = r["sid"], r["text_sha256"]
    edited = {**r, "solution": r["solution"].replace("2+2", "2+2 ")}
    check("testo modificato: hash diverso, associazione non valida",
          sha256_text(edited["solution"]) != sha and not association_ok(edited, sid, sha))
    check("testo modificato senza aggiornare l'hash: record incoerente", "testo diverso dal suo text_sha256" in validate_record(edited))
    redone = record(solution=edited["solution"])  # stesso sid, nuovo hash
    sel = "a" * 64
    cert = certificate(sel, audited=[(sid, sha, "include")])
    ok_old, _ = admit(r, cert, sel)
    ok_new, why = admit(redone, cert, sel)
    check("testo cambiato dopo l'audit: non ammesso", ok_old and not ok_new and why == "testo cambiato dopo l'audit", why)
    tampered = {**r, "checks": {**r["checks"], "think_close": 1, "think_open": 1}, "solution": "nessun tag \\boxed{4}"}
    check("controlli manomessi: ricalcolati e segnalati", any("controlli" in p for p in validate_record(tampered)))


def test_checks():
    ok = solution_checks("<think>a</think>b \\boxed{1}")
    check("controlli: soluzione normale", check_failures(ok) == [] and ok["prompt_quotes"] == 0 and not ok["text_before_think"])
    for name, text in [("vuota", ""), ("senza tag", "solo testo \\boxed{1}"), ("due </think>", "<think>a</think>b</think>\\boxed{1}"),
                       ("</think> prima di <think>", "</think>x<think>y"), ("due <think>", "<think>a<think>b</think>c"),
                       ("solo </think>", "a</think>\\boxed{1}")]:
        check(f"controlli: {name} scartata (think_tags)", check_failures(solution_checks(text)) == ["think_tags"])
    before = solution_checks("Sure.<think>a</think>b")
    check("controlli: testo prima di <think> solo segnalato", before["text_before_think"] and check_failures(before) == [])
    check("controlli: spazi prima di <think> non contano", not solution_checks("\n <think>a</think>b")["text_before_think"])
    quotes = [('It says: "Figure: ABC is a square."', 1), ("Figure:\nABC", 1), ("(Figure: x) e Figure: y", 2),
              ("Figure 1: Geometric diagram", 0), ("the figure: ABC", 0), ("SubFigure: x", 0), ("Figure:", 0),
              ("非 Figure: 汉字", 1)]
    bad = [(t, n, solution_checks(t)["prompt_quotes"]) for t, n in quotes if solution_checks(t)["prompt_quotes"] != n]
    check("controlli: citazioni del prompt (casi limite)", not bad, str(bad))
    q = record(solution='<think>It says: "Figure: AB = 3".</think>\\boxed{4}')
    check("controlli: citazione del prompt segnalata, non scartata", q["flags"] == ["prompt_quote"] and q["failed"] == [])


def test_no_certification():
    sel = "b" * 64
    r = record()
    check("risposta giusta ma nessun certificato: non ammessa", admit(r, None, sel) == (False, "nessun certificato"))
    check("nessun campo che certifichi nel record", not any("certif" in k for k in r))
    check("certificato 'rejected': non ammessa", admit(r, certificate(sel, "rejected"), sel)[0] is False)
    no_thr = {**certificate(sel), "thresholds": {}}
    check("certificato senza soglie: non valido", "certificato senza soglie" in validate_certificate(no_thr)
          and admit(r, no_thr, sel)[0] is False)
    no_rates = {**certificate(sel), "rates": {}}
    check("certificato senza i tassi: non valido", admit(r, no_rates, sel)[0] is False)
    check("certificato di un'altra selezione: non ammessa", admit(r, certificate("c" * 64), sel)[1] == "certificato di un'altra selezione")
    check("esclusa dall'audit: non ammessa", admit(r, certificate(sel, excluded=[(r["sid"], r["text_sha256"])]), sel)[1] == "esclusa dall'audit")
    check("in quarantena: non ammessa", admit(r, certificate(sel, quarantined=[(r["sid"], r["text_sha256"])]), sel)[1] == "in quarantena")
    wrong = {**r, "answer_correct": False}
    check("certificato valido ma risposta non verificata: non ammessa", admit(wrong, certificate(sel), sel)[0] is False)
    two = record(solution="<think>a</think>b</think>\\boxed{4}")
    check("certificato valido ma controlli non superati: non ammessa", admit(two, certificate(sel), sel)[0] is False)
    v1 = {"key": "p1", "solution": r["solution"], "answer": "4"}
    check("record v1: non ammissibile", admit(v1, certificate(sel), sel)[0] is False)
    check("solo record v2 coerente con certificato valido: ammessa", admit(r, certificate(sel), sel) == (True, "ammessa"))
    check("certificato schema sbagliato", validate_certificate({**certificate(sel), "schema": "x"}) != [])
    no_q = record(question=None)
    check("domanda non salvata nella generazione: non ammessa", admit(no_q, certificate(sel), sel)
          == (False, "domanda non salvata nella generazione") and validate_record(no_q) == [])
    check("domanda diversa dal suo hash nella generazione: errore",
          raises(lambda: record(question="Q", question_sha256="0" * 64), ValueError))
    check("domanda cambiata dopo la selezione: record incoerente",
          "domanda diversa dal suo question_sha256" in validate_record({**r, "question": "altro"}))
    q = record(solution='<think>It says: "Figure: AB = 3".</think>\\boxed{4}')
    check("citazione del prompt: scartata se la politica dice reject",
          admit(q, certificate(sel, flags={"prompt_quote": "reject", "text_before_think": "keep"}), sel)[1]
          == "segnalazione scartata dalla politica: prompt_quote")
    check("citazione del prompt: ammessa se la politica dice keep", admit(q, certificate(sel), sel)[0] is True)
    undecided = certificate(sel, flags={"prompt_quote": None, "text_before_think": "keep"})
    check("certificato con segnalazioni non decise: non valido", "record_policy con segnalazioni non decise"
          in validate_certificate(undecided) and admit(q, undecided, sel)[0] is False)
    check("certificato senza record_policy: non valido", admit(r, {k: v for k, v in certificate(sel).items()
                                                                  if k != "record_policy"}, sel)[0] is False)


def test_certificate_consistency():
    """Certificati `certified` alterati: ognuno deve risultare non valido dal solo certificato."""
    import copy

    sel = "d" * 64
    base = certificate(sel, audited=[("g/k/0", "a" * 64, "include"), ("g/k/1", "b" * 64)], excluded=[("g/k/2", "c" * 64)])
    check("certificato coerente: valido", validate_certificate(base) == [], validate_certificate(base))

    def bad(name, change, expect):
        c = copy.deepcopy(base)
        change(c)
        problems = validate_certificate(c)
        check(f"certificato alterato rifiutato: {name}", any(expect in p for p in problems), problems)

    def over_threshold(c):  # soglia più severa del limite, ma coerente fra politica e certificato
        c["policy"]["content"]["thresholds"]["solution_visual"] = 0.001
        c["thresholds"]["solution_visual"] = 0.001
    bad("limite corretto oltre la soglia", over_threshold, "oltre la soglia")
    bad("metrica obbligatoria mancante", lambda c: c["rates"].pop("solution_visual"), "tasso mancante")
    bad("metrica non valida (k > n)", lambda c: c["rates"]["solution_visual"].update(k=400), "conteggi non validi")
    bad("rate incoerente con k/n", lambda c: c["rates"]["solution_visual"].update(rate=0.2), "incoerenti")
    bad("calibrazione richiesta non eseguita", lambda c: c["sensitivity"].pop("measure"), "sensitivity.measure")
    bad("calibrazione vuota", lambda c: c["sensitivity"]["measure"].update(k=0, n=0, rate=None), "sensitivity.measure")
    bad("scelte della politica aperte", lambda c: c["policy"]["content"].update(frame_counts_as_error=None), "aperte")
    bad("soglie diverse dalla politica", lambda c: c["thresholds"].update(solution_visual=0.5), "soglie diverse")
    bad("regole per record diverse dalla politica", lambda c: c["record_policy"]["flags"].update(prompt_quote="reject"),
        "record_policy diversa")
    bad("limite corretto non ricalcolabile", lambda c: c["corrected_upper"].update(solution_visual=0.001), "ricalcolato")
    bad("sensibilità usata diversa dalla calibrazione", lambda c: c["corrected_upper"].update(sensitivity_lower=0.99),
        "sensitivity_lower")
    bad("certified con motivi di rifiuto", lambda c: c["reasons"].append("x"), "motivi di rifiuto")
    bad("audit retroattivo", lambda c: c.update(retroactive=True), "audit v2 non retroattivo")
    bad("audit v1", lambda c: c["audit"].update(version=1), "audit v2 non retroattivo")
    bad("esclusa non fra gli auditati", lambda c: c["excluded"].append(["g/k/9", "e" * 64]), "excluded diverso")
    bad("verdetto cambiato senza aggiornare gli esclusi", lambda c: c["audited"][0].__setitem__(2, "exclude"), "excluded diverso")
    bad("conteggio dei verdetti incoerente", lambda c: c["rates"]["verdicts"].update(include=7), "rates.verdicts")
    bad("sid ripetuti negli esiti", lambda c: c["audited"].append(list(c["audited"][0])), "sid ripetuti")
    bad("hash non valido negli esiti", lambda c: c["audited"][0].__setitem__(1, "zz"), "voce non valida")
    bad("riferimento alla selezione incoerente", lambda c: c["population"].update(selection={"sha256": "e" * 64}),
        "population.selection.sha256")
    rejected = {**copy.deepcopy(base), "decision": "rejected", "reasons": ["x"]}
    check("certificato rejected: basta la struttura (non ammette nulla)", validate_certificate(rejected) == [])


def test_policy():
    repo = json.loads((ROOT / "config" / "audit" / "f2_policy.json").read_text())
    open_ = policy_open(repo)
    check("politica del repository: tutte le scelte aperte, nessuna decisa dal codice",
          {"thresholds.solution_visual", "thresholds.claim_visual", "frame_counts_as_error", "flag_policy.prompt_quote",
           "flag_policy.text_before_think"} <= set(open_), open_)
    decided = {**repo, "thresholds": {k: 0.05 for k in repo["thresholds"]}, "frame_counts_as_error": True,
               "flag_policy": {"prompt_quote": "reject", "text_before_think": "keep"}}
    check("politica decisa: nessuna scelta aperta", policy_open(decided) == [])
    check("politica: schema sbagliato", policy_open({"schema": "x"}) != [])
    rules = record_policy(repo)
    check("regole per record: segnalazioni non decise restano None", rules["flags"] == {"prompt_quote": None, "text_before_think": None}
          and rules["rejecting"] == ["think_tags"])
    q = record(solution='<think>It says: "Figure: AB = 3".</think>\\boxed{4}')
    check("popolazione: segnalazione non decisa fuori", eligible(q, rules) == (False, "segnalazione senza decisione: prompt_quote")
          and eligible(record(), rules) == (True, "ok"))


def test_v1_readable(tmp: Path):
    v1 = {"key": "p1", "source_dataset": "S", "answer": "4", "solution": "<think>a</think>\\boxed{4}", "rewritten": False,
          "tokens": 5, "violetto_pass_rate": 0.75}
    path = tmp / "mixed.jsonl"
    path.write_text(json.dumps(v1) + "\n\n" + json.dumps(record()) + "\n")
    got = list(read_selected(path))
    check("selezione v1 e v2 leggibili", [record_version(r) for r in got] == [1, 2])
    bad = tmp / "bad.jsonl"
    bad.write_text(json.dumps({"schema": "f2-sol/9", "key": "k"}) + "\n")
    check("schema sconosciuto rifiutato", raises(lambda: list(read_selected(bad)), ValueError))
    hist = ROOT / "outputs" / "f2_select" / "f2-mavis2k" / "selected.jsonl"
    if hist.is_file():  # solo le prime righe: il file storico non si legge tutto
        with open(hist) as fh:
            first = [json.loads(next(fh)) for _ in range(3)]
        check("selezione storica f2-mavis2k leggibile come v1", all(record_version(r) == 1 for r in first))


def test_match_key():
    def comp(i, raw, cleaned=None, fr="stop"):
        return {"i": i, "raw": raw, "cleaned": cleaned or raw, "finish_reason": fr}

    m = match_key([{"sha": "B", "rewritten": False}], [comp(0, "A"), comp(1, "B"), comp(2, "C")])
    check("corrispondenza: testo identico alla generazione", m["status"] == "matched" and m["assign"] == [(1, "raw")])
    m = match_key([{"sha": "Bc", "rewritten": True}], [comp(0, "A"), comp(1, "B", "Bc")])
    check("corrispondenza: testo ripulito", m["status"] == "matched" and m["assign"] == [(1, "cleaned")])
    m = match_key([{"sha": "Bc", "rewritten": False}], [comp(1, "B", "Bc")])
    check("rewritten incoerente: mancante, non indovinata", m["status"] == "missing")
    m = match_key([{"sha": "B", "rewritten": False}], [comp(1, "B", fr="length")])
    check("generazione troncata esclusa", m["status"] == "missing")
    m = match_key([{"sha": "X", "rewritten": False}], [comp(0, "X"), comp(1, "X")])
    check("testi identici, una riga: ambigua", m["status"] == "ambiguous" and m["candidates"] == [[0, 1]])
    m = match_key([{"sha": "X", "rewritten": False}] * 2, [comp(0, "X"), comp(1, "X")])
    check("testi identici, due righe: l'ordine crescente decide", m["status"] == "matched"
          and m["assign"] == [(0, "raw+order"), (1, "raw+order")])
    m = match_key([{"sha": "X", "rewritten": False}] * 2, [comp(0, "X"), comp(1, "X"), comp(2, "X")])
    check("testi identici, due righe su tre generazioni: ambigua", m["status"] == "ambiguous")
    m = match_key([{"sha": "C", "rewritten": False}, {"sha": "B", "rewritten": False}], [comp(1, "B"), comp(2, "C")])
    check("ordine incompatibile con la selezione: mancante", m["status"] == "missing")
    m = match_key([{"sha": "Z", "rewritten": None}], [comp(0, "A", "Z")])
    check("selezione senza rewritten: si prova anche il testo ripulito", m["assign"] == [(0, "cleaned")])
    check("errore introdotto ricostruito",
          is_injection("AB = 12, BC = 12.", "AB = 12, BC = 16.", "12", "16")
          and not is_injection("AB = 12.", "AB = 13.", "12", "16") and not is_injection("AB = 12.", "AB = 16 .", "12", "16"))


def test_provenance(tmp: Path):
    check("sha256 del testo", sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    path = tmp / "out.jsonl"
    w = HashingTextWriter(path)
    w.write_json_line({"a": "é"})
    entry = w.close()
    check("scrittura con hash", entry["sha256"] == sha256_file(path) and entry["lines"] == 1)
    check("file esistente non sovrascritto", raises(lambda: HashingTextWriter(path), FileExistsError)
          and raises(lambda: write_new_json(path, {}), FileExistsError))
    w = HashingTextWriter(tmp / "aborted.jsonl")
    w.abort()
    check("scrittura interrotta: nessun file", not (tmp / "aborted.jsonl").exists() and not (tmp / "aborted.jsonl.tmp").exists())


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID", "test"))
    return subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script), *args], env=env, capture_output=True, text=True)


def test_pipeline(tmp: Path):
    """Generazione finta -> selezione v2 -> selezione v1 equivalente -> backfill -> confronto -> mappa di un audit finto."""
    gen = tmp / "gen-e2e"
    gen.mkdir()
    good = [f"<think>easy {c}</think>\\boxed{{4}}" for c in "ABC"]
    two_close = "<think>a</think>b</think>\\boxed{4}"
    quote = '<think>It says: "Figure: Side AB is 12 units long." The description gives AB = 12.</think>\\boxed{4}'
    same = "<think>twin</think>\\boxed{9}"
    wrong = "<think>no</think>\\boxed{5}"
    c = lambda t, fr="stop": {"text": t, "finish_reason": fr, "tokens": 5}  # noqa: E731
    rows = [{"key": "p1", "source_dataset": "S", "answer": "4", "completions": [c(good[0]), c(good[1]), c(good[2]), c(wrong)]},
            {"key": "p2", "source_dataset": "S", "answer": "7", "completions": [c(wrong), c(wrong, "length")]},
            {"key": "p3", "source_dataset": "S", "answer": "4", "completions": [c(two_close), c(good[0])]},
            {"key": "p4", "source_dataset": "T", "answer": "4", "completions": [c(quote), c(wrong)]},
            {"key": "p6", "source_dataset": "T", "answer": "9", "completions": [c(same), c(same)]}]
    (gen / "part-00000.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sel = tmp / "sel-v2"
    r = run("f2_select.py", "--gen", str(gen), "--out", str(sel), "--keep", "2", "--workers", "2")
    check("f2_select v2 termina", r.returncode == 0, r.stderr[-800:])
    if r.returncode:
        return
    new = list(read_selected(sel / "selected.v2.jsonl"))
    rej = list(read_selected(sel / "rejected.v2.jsonl"))
    stats = json.loads((sel / "stats.json").read_text())
    check("selezione v2: record coerenti e sid unici", all(validate_record(x) == [] for x in new + rej)
          and not duplicate_sids(new + rej))
    check("selezione v2: soluzione con due </think> scartata e registrata a parte",
          [x["sid"] for x in rej] == ["gen-e2e/p3/0"] and rej[0]["rejected_by"] == ["think_tags"]
          and stats["new_check_rejects"] == {"think_tags": 1} and stats["chosen_before_new_checks"] == len(new) + 1)
    check("selezione v2: la soluzione scartata non viene sostituita", [x["sid"] for x in new if x["key"] == "p3"] == ["gen-e2e/p3/1"])
    p4 = [x for x in new if x["key"] == "p4"]
    check("selezione v2: citazione del prompt solo segnalata", len(p4) == 1 and p4[0]["flags"] == ["prompt_quote"]
          and p4[0]["rewritten"] and stats["flags_on_selected"] == {"prompt_quote": 1})
    check("selezione v2: due soluzioni identiche dello stesso problema distinte",
          sorted(x["sid"] for x in new if x["key"] == "p6") == ["gen-e2e/p6/0", "gen-e2e/p6/1"])
    manifest = json.loads((sel / "selection.json").read_text())
    check("selezione v2: manifest con hash di parti e uscite", manifest["gen"]["parts"][0]["sha256"] == sha256_file(gen / "part-00000.jsonl")
          and manifest["outputs"]["selected_v2"]["sha256"] == sha256_file(sel / "selected.v2.jsonl"))
    again = run("f2_select.py", "--gen", str(gen), "--out", str(sel), "--keep", "2", "--workers", "2")
    check("f2_select: cartella non vuota rifiutata", again.returncode != 0 and "non è vuota" in again.stderr)

    # selezione v1 equivalente (quella che avrebbe scritto f2_select.py prima dei controlli nuovi)
    chosen = sorted(new + rej, key=lambda x: ([row["key"] for row in rows].index(x["key"]), x["completion"]))
    v1_path = tmp / "v1" / "selected.jsonl"
    v1_path.parent.mkdir()
    v1_path.write_text("".join(json.dumps({k: x[k] for k in ("key", "source_dataset", "answer", "solution", "rewritten", "tokens",
                                                          "violetto_pass_rate")}, ensure_ascii=False) + "\n" for x in chosen))
    audit = tmp / "audit-finto"
    (audit / "_gold").mkdir(parents=True)
    p1_first = [x for x in chosen if x["key"] == "p1"][0]
    p4_first = [x for x in chosen if x["key"] == "p4"][0]
    items = [{"fid": "f001", "key": "p1", "solution": p1_first["solution"], "calibration": False},
             {"fid": "f002", "key": "p4", "solution": p4_first["solution"].replace("12 units", "16 units"), "calibration": True},
             {"fid": "f003", "key": "p6", "solution": same + " ", "calibration": False}]
    (audit / "_gold" / "items_f2.json").write_text(json.dumps(items))
    (audit / "_gold" / "calibration_f2.json").write_text(json.dumps({"f002": {"kind": "f2_injected", "key": "p4", "original": "12",
                                                                             "changed": "16", "context": ""}}))
    (audit / "audit_f2.jsonl").write_text("".join(json.dumps({"fid": f, "verdict": v}) + "\n"
                                                  for f, v in (("f001", "include"), ("f002", "exclude"), ("f003", "include"))))
    (audit / "manifest.json").write_text("{}")
    bf = run("f2_backfill.py", "--gen", str(gen), "--selected", str(v1_path), "--audit", str(audit), "--workers", "2")
    check("f2_backfill termina", bf.returncode == 0, bf.stderr[-800:])
    if bf.returncode:
        return
    out = v1_path.parent
    back = list(read_selected(out / "selected.v2.jsonl"))
    info = json.loads((out / "backfill.json").read_text())
    check("backfill: tutte le righe ritrovate, nessun problema", len(back) == len(chosen) and info["counts"]["rows_by_status"] == {"matched": len(chosen)}
          and (out / "backfill_issues.jsonl").read_text() == "")
    check("backfill: stessi sid e hash della selezione v2 (scarto compreso)",
          {x["sid"]: x["text_sha256"] for x in back} == {x["sid"]: x["text_sha256"] for x in new + rej})
    check("backfill: due righe identiche risolte dall'ordine", sorted(x["match"] for x in back if x["key"] == "p6") == ["raw+order", "raw+order"])
    check("backfill: la soluzione con due </think> resta ma con il controllo fallito",
          [x["failed"] for x in back if x["sid"] == "gen-e2e/p3/0"] == [["think_tags"]])
    check("backfill: niente certificati e nessun record certificato", not list(out.glob("*cert*"))
          and all(x["origin"] == "backfill" and not any("certif" in k for k in x) for x in back))
    amap = {row["fid"]: row for row in map(json.loads, (out / "audit_map-audit-finto.jsonl").read_text().splitlines())}
    check("mappa dell'audit: item reale collegato al sid della prima soluzione",
          amap["f001"]["status"] == "exact" and amap["f001"]["sid"] == p1_first["sid"] and amap["f001"]["position"] == 0)
    check("mappa dell'audit: calibrazione verificata ricostruendo la modifica",
          amap["f002"]["status"] == "injected-verified" and amap["f002"]["source_sid"] == p4_first["sid"]
          and amap["f002"]["audited_text_sha256"] != amap["f002"]["source_text_sha256"])
    check("mappa dell'audit: testo auditato diverso da ogni soluzione, nessuna associazione",
          amap["f003"]["status"] == "missing" and amap["f003"]["sid"] is None)
    rerun = run("f2_backfill.py", "--gen", str(gen), "--selected", str(v1_path), "--workers", "1")
    check("backfill: uscite esistenti non sovrascritte", rerun.returncode != 0 and "non li sovrascrivo" in rerun.stderr)

    cmp_dir = tmp / "sel-cmp"
    r = run("f2_select.py", "--gen", str(gen), "--out", str(cmp_dir), "--keep", "2", "--workers", "2",
            "--compare", str(out / "selected.v2.jsonl"))
    cmp = json.loads((cmp_dir / "compare.json").read_text()) if r.returncode == 0 else {}
    check("confronto con il backfill: uguali a meno dei controlli nuovi", cmp.get("identical_except_new_checks") is True
          and cmp.get("only_reference_rejected_by_new_checks") == 1 and cmp.get("reference_solutions") == len(chosen)
          and cmp.get("reference_file", {}).get("sha256") == sha256_file(out / "selected.v2.jsonl"), r.stderr[-500:] or str(cmp))
    c2 = compare_selections({"g/k/0": "x"}, {"g/k/0": "y", "g/k/1": "z"}, set())
    check("confronto: testo cambiato e soluzione mancante non spiegata", c2["same_sid_other_text"] == 1
          and c2["only_reference_unexplained"] == 1 and not c2["identical_except_new_checks"])


if __name__ == "__main__":
    test_ids()
    test_hash_invalidation()
    test_checks()
    test_no_certification()
    test_certificate_consistency()
    test_policy()
    test_match_key()
    tmp = ROOT / "outputs" / "tmp" / f"test-f2-records-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        test_v1_readable(tmp)
        test_provenance(tmp)
        test_pipeline(tmp)
    finally:
        shutil.rmtree(tmp)
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
