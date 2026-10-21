#!/usr/bin/env python3
"""Audit automatico STEM: stato dei giudizi, lotti del turno successivo e risultati finali. Job CPU (math_verify e
sympy per confrontare le risposte e verificare i calcoli).

Legge OUT/_gold (problemi, risposte, calibrazione), OUT/deterministic.json, OUT/batches/*.json (compiti) e
OUT/answers/*.json (risposte degli agenti, stesso nome del lotto). Turni (regole in limite_vl.stem_audit):
- SA, SB, SC: risolvere dalla sola immagine e domanda, leggibilità, coerenza (SC: campi contestati, o tutto per il
  10% dei problemi approvati); DA, DB, DC: fatti di ogni affermazione della descrizione;
- G1, G2: revisione della risposta di riferimento con la risposta in vista; R1, R2: arbitri;
- FA, FB, FC, FR1, FR2: soluzioni F2 di Violetto.
Verifica con codice: i programmi che gli agenti scrivono per il calcolo (input dichiarati come presi dalla domanda o
letti nell'immagine) si eseguono in processi separati con limiti di tempo e memoria (code_results.json); un risultato
uguale alla risposta di riferimento con soli input presenti nel testo della domanda è una verifica con codice, con
input letti nell'immagine verifica solo il calcolo.
Senza --final scrive i lotti che servono ora e non sono già assegnati; i lotti con risposte mancanti o non valide
vanno rifatti (stesso file). Con --final scrive audit.jsonl, audit_f2.jsonl, calibration.jsonl, disagreements.jsonl,
quarantine.jsonl, audit_by_source.csv, audit_report.md e le liste proposte.

    srun $RUN_PYTHON -u -B scripts/stem_audit_round.py --dir OUT [--final]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import answers_match  # noqa: E402
from limite_vl.stem_audit import (ALLOWED_IMPORTS, FORBIDDEN_NAMES, UNSURE, clopper_pearson, clustered_rate,  # noqa: E402
                                  coherence_class, corrected_upper, match_answer, input_in_question, legibility_class, program_problems,
                                  recheck_sample, resolve_field, resolve_gold, undeclared_literals, unit_classes)

S_BLIND, D_BLIND, ARBITERS = ("SA", "SB", "SC"), ("DA", "DB", "DC"), ("R1", "R2")
F2_BLIND, F2_ARBITERS = ("FA", "FB", "FC"), ("FR1", "FR2")
F2_FIELDS = ("visual_ok", "reasoning_valid", "relies_on_unseen")
GOLD_THRESHOLD, VISUAL_THRESHOLD = 0.01, 0.02
# Regola fissata prima dello screening per scegliere le sorgenti da certificare (audit_report.md la riporta).
PROMISING = {"min_population": 1000, "max_gold_quarantine": 2, "max_bad_items": 1, "max_claim_rate": VISUAL_THRESHOLD,
             "max_captions_with_errors": 1}
BATCH = {"SC": 12, "DC": 6, "G1": 12, "G2": 12, "R1": 8, "R2": 8, "FC": 6, "FR1": 6, "FR2": 6}


def round_of(name: str) -> str:
    return name.rsplit("-", 1)[0]


def low(v):
    return v.strip().lower() if isinstance(v, str) else v


def as_bool(v):
    if isinstance(v, bool):
        return v
    return {"true": True, "yes": True, "false": False, "no": False}.get(low(v))


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


# ---------------------------------------------------------------- esecuzione dei programmi

_VALUE = re.compile(r"^[\w\s+\-*/^().,√π]{1,120}$")
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in (
    "abs", "min", "max", "round", "range", "len", "sum", "int", "float", "str", "list", "tuple", "dict", "set",
    "sorted", "enumerate", "zip", "pow", "divmod", "all", "any", "map", "filter", "bool", "reversed", "isinstance")}


def _child(code: str, inputs: dict, conn) -> None:
    import resource

    vm = next(int(line.split()[1]) * 1024 for line in open("/proc/self/status") if line.startswith("VmSize:"))
    limit = vm + (1536 << 20)  # memoria già mappata dal processo padre + 1.5 GiB
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    try:
        import fractions
        import itertools

        import sympy

        mods = {"math": math, "sympy": sympy, "fractions": fractions, "itertools": itertools}

        def guarded_import(name, *a, **k):
            if name.split(".")[0] not in ALLOWED_IMPORTS:
                raise ImportError(name)
            return __import__(name, *a, **k)

        env = {"__builtins__": SAFE_BUILTINS | {"__import__": guarded_import, "print": lambda *a, **k: None},
               **mods, "sqrt": sympy.sqrt, "pi": sympy.pi, "Rational": sympy.Rational}
        for name, spec in inputs.items():
            env[name] = sympy.sympify(str(spec["value"]).replace("√", "sqrt").replace("π", "pi").replace("^", "**"))
        exec(compile(code, "<agente>", "exec"), env)  # noqa: S102 - codice controllato da program_problems, processo isolato
        res = env["result"]
        try:
            res = sympy.nsimplify(res) if isinstance(res, float) else sympy.simplify(res)
        except Exception:  # noqa: BLE001
            pass
        conn.send({"ok": True, "result": str(res)})
    except Exception as exc:  # noqa: BLE001
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})


def run_program(program: dict) -> dict:
    import multiprocessing as mp

    code, inputs = program.get("code") or "", program.get("inputs") or {}
    if not isinstance(inputs, dict) or not all(isinstance(v, dict) and "value" in v for v in inputs.values()):
        return {"ok": False, "error": "input non nel formato {nome: {value, source}}"}
    bad = [n for n, v in inputs.items() if not n.isidentifier() or n.startswith("_") or "__" in str(v["value"])
           or not _VALUE.match(str(v["value"])) or set(re.findall(r"[A-Za-z_]+", str(v["value"]))) & FORBIDDEN_NAMES]
    problems = program_problems(code) + [f"input {n}" for n in bad]
    if problems:
        return {"ok": False, "error": "; ".join(problems)}
    ctx = mp.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(code, inputs, send))
    p.start()
    p.join(15)
    if p.is_alive():
        p.kill()
        p.join()
        return {"ok": False, "error": "tempo scaduto"}
    return recv.recv() if recv.poll() else {"ok": False, "error": f"uscito con codice {p.exitcode}"}


# ---------------------------------------------------------------- stato dell'audit

class Audit:
    def __init__(self, root: Path, frac: float, seed: int):
        self.root, self.frac, self.seed = root, frac, seed
        g = root / "_gold"
        self.items = {it["pid"]: it for it in json.loads((g / "items.json").read_text())}
        self.gold = json.loads((g / "gold.json").read_text())
        self.truth = json.loads((g / "calibration.json").read_text())
        self.det = json.loads((root / "deterministic.json").read_text())
        f2 = g / "items_f2.json"
        self.f2 = {it["fid"]: it for it in json.loads(f2.read_text())} if f2.exists() else {}
        self.batches, self.answers, self.invalid = {}, defaultdict(dict), {}
        for path in sorted((root / "batches").glob("*.json")):
            self.batches[path.stem] = json.loads(path.read_text())
        for name, batch in self.batches.items():
            path = root / "answers" / f"{name}.json"
            if not path.exists():
                continue
            try:
                got = json.loads(path.read_text())
                if isinstance(got, dict):
                    got = got.get("items") or got.get("results") or list(got.values())
                got = {str(a["id"]): a for a in got if isinstance(a, dict) and "id" in a}
            except Exception as exc:  # noqa: BLE001 - risposta non leggibile: lotto da rifare
                self.invalid[name] = f"JSON non valido: {exc}"[:200]
                continue
            missing = [t["id"] for t in batch if t["id"] not in got]
            if missing:
                self.invalid[name] = f"{len(missing)} voci mancanti (es. {missing[0]})"
                continue
            r = round_of(name)
            for t in batch:
                self.answers[r][t["id"]] = self.answers[r].get(t["id"], {}) | got[t["id"]] | {"_task": t}
        self.assigned = defaultdict(set)
        for name, batch in self.batches.items():
            for t in batch:
                for f in self.task_fields(round_of(name), t):
                    self.assigned[round_of(name)].add((t["id"], f))
        self.code_path = root / "code_results.json"
        self.code = json.loads(self.code_path.read_text()) if self.code_path.exists() else {}
        self._gold_cache = {}
        self.complete = {r: all((root / "answers" / f"{n}.json").exists() and n not in self.invalid
                                for n in self.batches if round_of(n) == r) and any(round_of(n) == r for n in self.batches)
                         for r in ("SA", "SB", "DA", "DB", "FA", "FB")}

    @staticmethod
    def task_fields(r: str, t: dict) -> list[str]:
        if r in S_BLIND:
            return list(t.get("check", []))
        if r in D_BLIND:
            return [f"{u['cid']}:{k}" for u in t["units"] for k in ("vis", "ded")]
        if r in ("G1", "G2"):
            return ["gold"]
        if r in F2_BLIND:
            return list(F2_FIELDS)
        return [d["field"] for d in t.get("disputes", [])]

    def done(self, r: str, pid: str, field: str) -> bool:
        return (pid, field) in self.assigned[r] and pid in self.answers[r]

    def pending(self, r: str, pid: str, field: str) -> bool:
        return (pid, field) in self.assigned[r] and pid not in self.answers[r]

    def unit_fields(self, pid: str) -> list[str]:
        return [f"{u['cid']}:{k}" for u in self.items[pid]["units"] for k in ("vis", "ded")]

    def unit_text(self, pid: str, cid: str) -> dict:
        return next(u for u in self.items[pid]["units"] if u["cid"] == cid)

    # ------------------------------------------------------------ giudizi alla cieca

    def blind_value(self, r: str, pid: str, field: str):
        """(classe, etichetta da mostrare, nota) del giudizio r sul campo, o None se r non l'ha giudicato."""
        if not self.done(r, pid, field):
            return None
        a = self.answers[r][pid]
        if field == "legibility":
            v = low(a.get("legibility"))
            return legibility_class(v), v, a.get("legibility_note")
        if field == "coherence":
            v = low(a.get("coherence"))
            return coherence_class(v), v, a.get("coherence_note")
        cid, kind = field.split(":")
        for u in a.get("units") or []:
            if isinstance(u, dict) and u.get("cid") == cid:
                facts = [f for f in u.get("facts") or [] if isinstance(f, dict)]
                if not facts:
                    return None, None, "nessun fatto elencato"
                mine = [f for f in facts if ("ded" if str(f.get("kind", "")).lower().startswith("ded") else "vis") == kind]
                return unit_classes(facts)[kind], mine, None
        return None, None, "affermazione mancante"

    def arbiter_value(self, r: str, pid: str, field: str):
        if not self.done(r, pid, field):
            return None
        for d in self.answers[r][pid].get("disputes") or []:
            if isinstance(d, dict) and d.get("field") == field:
                v = low(d.get("verdict"))
                if field == "gold":
                    cls = v if v in ("correct", "wrong") else None
                elif field in F2_FIELDS:
                    cls = as_bool(v)
                elif field == "legibility":
                    cls = legibility_class(v)
                elif field == "coherence":
                    cls = coherence_class(v)
                else:
                    cls = {"false": "E", "not_false": "K"}.get(v)
                return cls, v, d.get("note"), as_bool(d.get("shared_error"))
        return None, None, "verdetto mancante", None

    def f2_value(self, r: str, pid: str, field: str):
        if not self.done(r, pid, field):
            return None
        a = self.answers[r][pid]
        v = as_bool(a.get(field))
        note = a.get("reasoning_note") if field == "reasoning_valid" else json.dumps(a.get("issues") or [], ensure_ascii=False)[:800]
        return v, v, note

    def field_state(self, pid: str, field: str) -> dict:
        if field in F2_FIELDS:
            blind = [x for r in F2_BLIND if (x := self.f2_value(r, pid, field)) is not None]
            arbs = [x for r in F2_ARBITERS if (x := self.arbiter_value(r, pid, field)) is not None]
        else:
            rounds = S_BLIND if field in ("legibility", "coherence") else D_BLIND
            blind = [x for r in rounds if (x := self.blind_value(r, pid, field)) is not None]
            arbs = [x for r in ARBITERS if (x := self.arbiter_value(r, pid, field)) is not None]
        st = resolve_field([x[0] for x in blind], [x[0] for x in arbs])
        return st | {"blind": [{"value": x[1], "class": x[0], "note": x[2]} for x in blind],
                     "arbiters": [{"value": x[1], "class": x[0], "note": x[2], "shared_error": x[3]} for x in arbs]}

    # ------------------------------------------------------------ risposta di riferimento e codice

    def programs(self, pid: str) -> list[dict]:
        out = []
        for r in S_BLIND:
            if not self.done(r, pid, "solve"):
                continue
            prog = self.answers[r][pid].get("program")
            if not isinstance(prog, dict) or not prog.get("code"):
                continue
            key = hashlib.sha256(json.dumps(prog, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
            if key not in self.code:
                self.code[key] = run_program(prog)
            res = self.code[key]
            q = self.items[pid]["question"]
            inputs = prog.get("inputs") or {}
            src = {n: low(v.get("source")) for n, v in inputs.items() if isinstance(v, dict)}
            text_ok = all(s == "question" and input_in_question(inputs[n]["value"], q) for n, s in src.items())
            try:
                literals = undeclared_literals(prog["code"])
            except SyntaxError:
                literals = []
            level = "testo" if text_ok and not literals else "letture"
            match = bool(res.get("ok")) and match_answer(res["result"], self.gold[pid], q, answers_match)
            out.append({"round": r, "key": key, "ok": res.get("ok", False), "result": res.get("result"),
                        "error": res.get("error"), "level": level, "matches_gold": match,
                        "image_inputs": [n for n, s in src.items() if s != "question"], "undeclared_literals": literals[:5]})
        return out

    def solves(self, pid: str) -> list[tuple[str, str, str | None]]:
        out = []
        for r in S_BLIND:
            if not self.done(r, pid, "solve"):
                continue
            own = self.answers[r][pid].get("own_answer")
            if own is None or (isinstance(own, str) and (not own.strip() or low(own) in ("null", "none", "unknown"))):
                out.append((r, "unsolved", None))
            else:
                out.append((r, "match" if match_answer(str(own), self.gold[pid], self.items[pid]["question"], answers_match)
                            else "mismatch", str(own)))
        return out

    def reviews(self, pid: str) -> list[tuple[str, str | None, dict]]:
        out = []
        for r in ("G1", "G2"):
            if self.done(r, pid, "gold"):
                a = self.answers[r][pid]
                v = low(a.get("verdict"))
                out.append((r, v if v in ("correct", "wrong", "uncertain") else None, a))
        arb = self.arbiter_value("R1", pid, "gold")
        if arb is not None:
            out.append(("R1", arb[0] or "uncertain", {"verdict": arb[1], "note": arb[2]}))
        return out

    def gold_state(self, pid: str) -> dict:
        if pid not in self._gold_cache:
            self._gold_cache[pid] = self._gold_state(pid)
        return self._gold_cache[pid]

    def _gold_state(self, pid: str) -> dict:
        det = self.det[pid]
        s, rv, progs = self.solves(pid), self.reviews(pid), self.programs(pid)
        info = {"solves": [{"round": r, "result": res, "answer": ans} for r, res, ans in s],
                "reviews": [{"round": r, "verdict": v, "note": a.get("note"), "corrected_answer": a.get("corrected_answer"),
                             "answer_coherent": low(a.get("answer_coherent"))} for r, v, a in rv],
                "programs": progs}
        if det["text"].get("gold_wrong_by_code"):
            return {"status": "resolved", "value": "wrong", "how": "codice: lettera fuori dalle opzioni", "evidence": "codice"} | info
        conflict = bool(det.get("duplicates", {}).get("conflict"))
        st = resolve_gold([x[1] for x in s], [x[1] for x in rv], force_review=conflict)
        if st["status"] != "resolved":
            evidence = "incerto"
        elif st["value"] == "correct" and any(p["matches_gold"] and p["level"] == "testo" for p in progs):
            evidence = "codice (solo testo della domanda)"
        elif st["value"] == "correct" and any(p["matches_gold"] for p in progs):
            evidence = "codice con letture dell'agente"
        else:
            evidence = "agenti"
        return st | {"evidence": evidence, "duplicate_conflict": conflict} | info

    # ------------------------------------------------------------ approvati e controllo del 10%

    def s_approved(self, pid: str) -> bool:
        if not all(pid in self.answers[r] for r in ("SA", "SB")):
            return False
        for f in ("legibility", "coherence"):
            vals = [self.blind_value(r, pid, f) for r in ("SA", "SB")]
            if any(v is None or v[0] in UNSURE for v in vals) or vals[0][0] != vals[1][0]:
                return False
        return [x[1] for x in self.solves(pid) if x[0] in ("SA", "SB")] == ["match", "match"]

    def d_approved(self, pid: str) -> bool:
        if not all(pid in self.answers[r] for r in ("DA", "DB")):
            return False
        for f in self.unit_fields(pid):
            vals = [self.blind_value(r, pid, f) for r in ("DA", "DB")]
            if any(v is None or v[0] in UNSURE for v in vals) or vals[0][0] != vals[1][0]:
                return False
        return True

    def rechecks(self, kind: str) -> set[str]:
        by = defaultdict(list)
        for pid, it in self.items.items():
            if it["image"] and (self.s_approved(pid) if kind == "S" else self.d_approved(pid)):
                by[it["source"]].append(pid)
        return recheck_sample(by, self.frac, self.seed + (0 if kind == "S" else 1))

    # ------------------------------------------------------------ prossimi compiti

    def next_tasks(self) -> dict[str, list[dict]]:
        tasks = defaultdict(list)
        # il 10% di controllo si estrae solo a passaggi A e B completi: dopo, l'insieme degli approvati non cambia più
        s_all, d_all = self.complete["SA"] and self.complete["SB"], self.complete["DA"] and self.complete["DB"]
        s_re = self.rechecks("S") if s_all else set()
        d_re = self.rechecks("D") if d_all else set()
        for pid, it in self.items.items():
            if not it["image"]:
                continue
            base = {"id": pid, "image": it["image"]}
            s_ready = s_all and all(pid in self.answers[r] for r in ("SA", "SB"))
            d_ready = d_all and bool(it["units"]) and all(pid in self.answers[r] for r in ("DA", "DB"))
            if s_ready and not any((pid, f) in self.assigned["SC"] for f in ("solve", "legibility", "coherence")):
                if pid in s_re:
                    tasks["SC"].append(base | {"question": it["question"], "check": ["solve", "legibility", "coherence"]})
                else:
                    need = [f for f in ("legibility", "coherence") if self.field_state(pid, f)["status"] == "need_blind"]
                    if need:
                        tasks["SC"].append(base | {"question": it["question"], "check": need})
            if d_ready and not any((pid, f) in self.assigned["DC"] for f in self.unit_fields(pid)):
                units = it["units"] if pid in d_re else [
                    u for u in it["units"] if any(self.field_state(pid, f"{u['cid']}:{k}")["status"] == "need_blind" for k in ("vis", "ded"))]
                if units:
                    tasks["DC"].append(base | {"units": [{k: u[k] for k in ("cid", "section", "text")} for u in units]})
            g = self.gold_state(pid) if s_ready else None
            if g and g["status"] == "need_review" and not self.pending("SC", pid, "solve"):
                others = sorted({x["answer"] for x in g["solves"] if x["result"] == "mismatch"} |
                                {p["result"] for p in g["programs"] if p["ok"] and not p["matches_gold"]})
                for r in ("G1", "G2"):
                    if (pid, "gold") not in self.assigned[r]:
                        tasks[r].append(base | {"question": it["question"], "reference_answer": self.gold[pid],
                                                "other_answers": others[:4]})
            for r in ARBITERS:
                disputes = []
                fields = (["legibility", "coherence"] if s_ready else []) + (self.unit_fields(pid) if d_ready else [])
                for f in fields:
                    st = self.field_state(pid, f)
                    rounds = S_BLIND if f in ("legibility", "coherence") else D_BLIND
                    if st["status"] != "need_arbiter" or (pid, f) in self.assigned[r] or any(self.pending(x, pid, f) for x in rounds):
                        continue
                    if r == "R2" and not self.done("R1", pid, f):
                        continue
                    d = {"field": f, "judgments": [{"value": b["value"], "note": b["note"]} for b in st["blind"]]}
                    if ":" in f:
                        u = self.unit_text(pid, f.split(":")[0])
                        d |= {"statement": u["text"], "section": u["section"],
                              "kind": "visible facts" if f.endswith(":vis") else "deductions"}
                    disputes.append(d)
                if r == "R1" and g and g["status"] == "need_arbiter" and (pid, "gold") not in self.assigned["R1"]:
                    disputes.append({"field": "gold", "reference_answer": self.gold[pid],
                                     "judgments": [{"value": x["verdict"], "note": x["note"], "corrected_answer": x["corrected_answer"]}
                                                   for x in g["reviews"]]})
                if disputes:
                    tasks[r].append(base | {"question": it["question"], "disputes": disputes})
        for fid, it in self.f2.items():
            if not (self.complete["FA"] and self.complete["FB"]) or not all(fid in self.answers[r] for r in ("FA", "FB")):
                continue
            base = {"id": fid, "image": it["image"], "question": it["question"], "solution": it["solution"]}
            states = {f: self.field_state(fid, f) for f in F2_FIELDS}
            if any(s["status"] == "need_blind" for s in states.values()) and (fid, F2_FIELDS[0]) not in self.assigned["FC"]:
                tasks["FC"].append(base)
            for r in F2_ARBITERS:
                disputes = [{"field": f, "judgments": [{"value": b["value"], "note": b["note"]} for b in s["blind"]]}
                            for f, s in states.items() if s["status"] == "need_arbiter" and (fid, f) not in self.assigned[r]
                            and (r == "FR1" or self.done("FR1", fid, f)) and not self.pending("FC", fid, f)]
                if disputes:
                    tasks[r].append(base | {"disputes": disputes})
        return tasks

    def write_tasks(self, tasks: dict[str, list[dict]]) -> list[str]:
        names = []
        for r, items in tasks.items():
            start = max((int(n.rsplit("-", 1)[1]) for n in self.batches if round_of(n) == r), default=0)
            size = BATCH.get(r, 10)
            for b in range(0, len(items), size):
                name = f"{r}-{start + b // size + 1:02d}"
                (self.root / "batches" / f"{name}.json").write_text(json.dumps(items[b:b + size], indent=1, ensure_ascii=False))
                names.append(name)
        return names

    def save_code(self) -> None:
        self.code_path.write_text(json.dumps(self.code, indent=1, ensure_ascii=False))

    # ------------------------------------------------------------ conclusioni per problema

    def conclude(self, pid: str) -> dict:
        it, det = self.items[pid], self.det[pid]
        base = {"pid": pid, "key": it["key"], "source": it["source"], "calibration": it["calibration"], "tags": it.get("tags", [])}
        if not it["image"]:
            why = f"immagine illeggibile (codice: {det['image'].get('error')})"
            return base | {"rl": "exclude", "f2": "exclude", "reasons": [why], "unreadable": True, "evidence": {"immagine": "codice"}}
        g = self.gold_state(pid)
        leg, coh = self.field_state(pid, "legibility"), self.field_state(pid, "coherence")
        units = {f: self.field_state(pid, f) for f in self.unit_fields(pid)}
        g_rev = [x for x in g["reviews"] if x["round"] in ("G1", "G2")]
        answer_incoherent = len(g_rev) == 2 and all(x["answer_coherent"] == "no" for x in g_rev)
        vis_err = [f for f, s in units.items() if f.endswith(":vis") and s["status"] == "resolved" and s["value"] == "E"]
        ded_err = [f for f, s in units.items() if f.endswith(":ded") and s["status"] == "resolved" and s["value"] == "E"]
        open_units = [f for f, s in units.items() if s["status"] != "resolved"]

        def listed(kind):  # affermazioni in cui A o B elencano almeno un fatto di quel tipo
            out = set()
            for f in units:
                if f.endswith(f":{kind}"):
                    for r in ("DA", "DB"):
                        x = self.blind_value(r, pid, f)
                        if x and x[1]:
                            out.add(f)
            return out

        vis_listed, ded_listed = listed("vis"), listed("ded")
        reasons, evidence = [], {"risposta": g["evidence"]}
        if g["status"] == "resolved" and g["value"] == "wrong":
            reasons.append(f"risposta di riferimento sbagliata ({g['how']})")
        if leg["status"] == "resolved" and leg["value"] == "illegible":
            reasons.append(f"figura illeggibile alla risoluzione di training ({leg['how']})")
        if (coh["status"] == "resolved" and coh["value"] == "problem") or answer_incoherent:
            reasons.append("domanda o risposta incoerente con la figura" + (f" ({coh['how']})" if coh["status"] == "resolved" else " (revisioni)"))
        evidence["leggibilità"] = "agenti" if leg["status"] == "resolved" else "incerto"
        evidence["coerenza"] = "agenti" if coh["status"] == "resolved" else "incerto"
        if reasons:
            rl = "exclude"
        elif g["status"] == "resolved" and leg["status"] == "resolved" and coh["status"] == "resolved":
            rl = "include"
            reasons.append("risposta valida (" + g["evidence"] + "), figura leggibile e domanda coerente (agenti)")
        else:
            rl = "quarantine"
            reasons += [f"{name} non risolta: evidenza insufficiente" for name, s in (("risposta", g), ("leggibilità", leg), ("coerenza", coh))
                        if s["status"] != "resolved"]
        if rl == "exclude":
            f2 = "exclude"
        elif vis_err or ded_err:
            f2 = "exclude"
            reasons.append(f"descrizione con {len(vis_err)} affermazioni con fatti visibili falsi e {len(ded_err)} con deduzioni false: "
                           "ponte F2 non affidabile")
        elif not units and not it["calibration"]:
            f2 = "quarantine"
            reasons.append("descrizione senza affermazioni da verificare")
        elif rl == "include" and not open_units:
            f2 = "include"
            reasons.append("nessun fatto falso nella descrizione (agenti)")
        else:
            f2 = "quarantine"
            if open_units:
                reasons.append(f"{len(open_units)} giudizi sulla descrizione non risolti")
        flags = det["image"].get("flags", []) + det["text"]["flags"]
        disagree = {}
        for name, fs in (("leggibilità", ["legibility"]), ("coerenza", ["coherence"]), ("descrizione", list(units))):
            disagree[name] = any(len({b["class"] for b in self.field_state(pid, f)["blind"][:2]}) > 1 for f in fs)
        disagree["risposta"] = len({x["result"] for x in g["solves"] if x["round"] in ("SA", "SB")}) > 1
        facts = Counter()
        for r in ("DA", "DB"):
            for u in (self.answers[r].get(pid) or {}).get("units") or []:
                if not isinstance(u, dict):
                    continue
                for f in u.get("facts") or []:
                    if isinstance(f, dict):
                        kind = "deduzione" if str(f.get("kind", "")).lower().startswith("ded") else "visibile"
                        facts[f"{r}/{kind}/{low(f.get('label'))}"] += 1
        rech_s = pid in self.answers["SC"] and "solve" in self.answers["SC"][pid]["_task"].get("check", [])
        rech_d = pid in self.answers["DC"] and len(self.answers["DC"][pid]["_task"]["units"]) == len(it["units"])

        def changed(states):  # il controllo in più ha cambiato o lasciato aperto l'esito di A e B
            return any(s["status"] != "resolved" or (s["blind"] and s["value"] != s["blind"][0]["class"]) for s in states)

        recheck_changed = (rech_s and (changed([leg, coh]) or g["value"] != "correct")) or (rech_d and changed(list(units.values())))
        return base | {
            "rl": rl, "f2": f2, "reasons": reasons, "unreadable": False, "evidence": evidence, "deterministic_flags": flags,
            "contamination_candidate": det["contamination_candidate"], "duplicates": det.get("duplicates"),
            "gold": {k: g[k] for k in ("status", "value", "how", "evidence", "solves", "reviews", "programs")},
            "legibility": {k: leg[k] for k in ("status", "value", "how")},
            "coherence": {k: coh[k] for k in ("status", "value", "how")} | {"answer_incoherent_by_reviews": answer_incoherent},
            "claims": {"units": len(it["units"]), "vis_errors": len(vis_err), "ded_errors": len(ded_err),
                       "vis_listed": len(vis_listed), "ded_listed": len(ded_listed),
                       "vis_resolved": len([f for f in vis_listed if units[f]["status"] == "resolved"]),
                       "ded_resolved": len([f for f in ded_listed if units[f]["status"] == "resolved"]),
                       "open": len(open_units), "error_cids": sorted({f.split(":")[0] for f in vis_err + ded_err})},
            "facts_by_agent": dict(facts), "disagreement": disagree, "rechecked_s": rech_s, "rechecked_d": rech_d,
            "recheck_changed": recheck_changed,
            "_units": units, "_leg": leg, "_coh": coh, "_gold": g}


# ---------------------------------------------------------------- calibrazione

def rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": (k / n if n else None), "lo": lo, "hi": hi}


def calibration_stats(audit: Audit, rows: dict[str, dict]) -> tuple[dict, list[dict]]:
    """Quanti errori noti trovano i singoli agenti e la procedura intera, quanti ne perdono, e i falsi allarmi."""
    per_item, c = [], defaultdict(Counter)
    for pid, t in audit.truth.items():
        r = rows.get(pid)
        if r is None or r["unreadable"]:
            continue
        units, g, leg, coh = r["_units"], r["_gold"], r["_leg"], r["_coh"]
        rec = {"pid": pid, "kind": t["kind"], "source": r["source"]}
        for cid in t.get("units_injected", {}):
            states = [units.get(f"{cid}:vis"), units.get(f"{cid}:ded")]
            for rr in ("DA", "DB"):
                vals = [audit.blind_value(rr, pid, f"{cid}:{k}") for k in ("vis", "ded")]
                if any(v is not None for v in vals):
                    c["affermazione alterata"][f"agente_{'trovata' if any(v and v[0] == 'E' for v in vals) else 'persa'}"] += 1
            if any(s and s["status"] == "resolved" and s["value"] == "E" for s in states):
                out = "trovata"
            elif all(s and s["status"] == "resolved" for s in states):
                out = "persa"
            else:
                out = "quarantena"
            c["affermazione alterata"][f"procedura_{out}"] += 1
            rec.setdefault("units_injected", {})[cid] = out
        for cid in t.get("units_true", []):
            for rr in ("DA", "DB"):
                v = audit.blind_value(rr, pid, f"{cid}:vis")
                if v is not None:
                    c["affermazione vera (MAVIS)"][f"agente_{'falso_allarme' if v[0] == 'E' else 'giusta'}"] += 1
            s = units.get(f"{cid}:vis")
            if s and s["status"] == "resolved":
                c["affermazione vera (MAVIS)"][f"procedura_{'falso_allarme' if s['value'] == 'E' else 'giusta'}"] += 1
            else:
                c["affermazione vera (MAVIS)"]["procedura_quarantena"] += 1
        if "gold_injected" in t:
            for x in g["solves"]:
                if x["round"] in ("SA", "SB"):
                    c["risposta sbagliata introdotta"][f"solutore_{'trova_altro' if x['result'] == 'mismatch' else 'conferma' if x['result'] == 'match' else 'non_risolve'}"] += 1
            out = "trovata" if g["status"] == "resolved" and g["value"] == "wrong" else "persa" if g["status"] == "resolved" else "quarantena"
            c["risposta sbagliata introdotta"][f"procedura_{out}"] += 1
            c["risposta sbagliata introdotta"]["programma_conferma_la_sbagliata"] += any(p["matches_gold"] for p in g["programs"])
            rec["gold"] = out
        elif t.get("gold_true") is not None:
            for x in g["solves"]:
                if x["round"] in ("SA", "SB"):
                    c["risposta vera (MAVIS, sintetici)"][f"solutore_{x['result']}"] += 1
            out = "falso_allarme" if g["status"] == "resolved" and g["value"] == "wrong" else "giusta" if g["status"] == "resolved" else "quarantena"
            c["risposta vera (MAVIS, sintetici)"][f"procedura_{out}"] += 1
            rec["gold"] = out
        if t.get("legible") is False:
            out = "trovata" if leg["status"] == "resolved" and leg["value"] == "illegible" else "persa" if leg["status"] == "resolved" else "quarantena"
            c["figura illeggibile introdotta"][f"procedura_{out}"] += 1
            for b in leg["blind"][:2]:
                c["figura illeggibile introdotta"][f"agente_{'trovata' if b['class'] == 'illegible' else 'persa'}"] += 1
            rec["legibility"] = out
        elif t.get("legible") is True:
            out = "falso_allarme" if leg["status"] == "resolved" and leg["value"] == "illegible" else "giusta" if leg["status"] == "resolved" else "quarantena"
            c["figura leggibile (MAVIS, sintetici)"][f"procedura_{out}"] += 1
        if t.get("coherent") is False:
            out = "trovata" if (coh["status"] == "resolved" and coh["value"] == "problem") or r["coherence"]["answer_incoherent_by_reviews"] \
                else "persa" if coh["status"] == "resolved" else "quarantena"
            c["domanda scambiata"][f"procedura_{out}"] += 1
            for b in coh["blind"][:2]:
                c["domanda scambiata"][f"agente_{'trovata' if b['class'] == 'problem' else 'persa'}"] += 1
            rec["coherence"] = out
        elif t.get("coherent") is True:
            out = "falso_allarme" if coh["status"] == "resolved" and coh["value"] == "problem" else "giusta" if coh["status"] == "resolved" else "quarantena"
            c["domanda coerente (MAVIS, sintetici)"][f"procedura_{out}"] += 1
        per_item.append(rec)
    summary = {k: dict(v) for k, v in c.items()}

    def sens(name, prefix="procedura"):
        v = c[name]
        found, lost = v[f"{prefix}_trovata"], v[f"{prefix}_persa"]
        return rate(found, found + lost)

    summary["_sensibilità"] = {
        "affermazioni (procedura)": sens("affermazione alterata"), "affermazioni (agente singolo)": sens("affermazione alterata", "agente"),
        "risposte (procedura)": sens("risposta sbagliata introdotta"), "illeggibili (procedura)": sens("figura illeggibile introdotta"),
        "domande scambiate (procedura)": sens("domanda scambiata")}
    fa = c["affermazione vera (MAVIS)"]
    summary["_falsi_allarmi"] = {
        "affermazioni vere (agente singolo)": rate(fa["agente_falso_allarme"], fa["agente_falso_allarme"] + fa["agente_giusta"]),
        "affermazioni vere (procedura)": rate(fa["procedura_falso_allarme"], fa["procedura_falso_allarme"] + fa["procedura_giusta"]),
        "risposte vere (procedura)": rate(c["risposta vera (MAVIS, sintetici)"]["procedura_falso_allarme"],
                                          c["risposta vera (MAVIS, sintetici)"]["procedura_falso_allarme"] + c["risposta vera (MAVIS, sintetici)"]["procedura_giusta"])}
    return summary, per_item


# ---------------------------------------------------------------- risultati finali

def pooled(sample: list[dict], cal: dict, others: list[Path]) -> dict | None:
    """Risposte e fatti visibili dei problemi MAVIS per profondità di questo audit e degli audit in --pool (stessa procedura),
    con la sensibilità sommata delle calibrazioni. Un problema presente in più audit conta una volta (prima questo audit)."""
    if not others:
        return None
    real = lambda r: not r["calibration"] and r["source"].startswith("MAVIS-Geometry d")  # noqa: E731
    rows = {r["key"]: r for r in sample if real(r)}
    keys = ("risposte (procedura)", "affermazioni (procedura)")
    sens = {k: [cal["_sensibilità"][k]["k"], cal["_sensibilità"][k]["n"]] for k in keys}
    for d in others:
        with open(d / "audit.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                if real(r):
                    rows.setdefault(r["key"], r)
        oc = json.loads((d / "calibration_summary.json").read_text())["_sensibilità"]
        for k in keys:
            sens[k][0] += oc[k]["k"]
            sens[k][1] += oc[k]["n"]
    live = [r for r in rows.values() if not r["unreadable"]]
    gres = [r for r in live if r["gold"]["status"] == "resolved"]
    g = rate(sum(r["gold"]["value"] == "wrong" for r in gres), len(gres))
    vis = clustered_rate([(r["claims"]["vis_errors"], r["claims"]["vis_resolved"]) for r in live])
    s_gold, s_claim = rate(*sens[keys[0]]), rate(*sens[keys[1]])
    res = {"audits": [str(d) for d in others], "problems": len(rows), "gold": g, "gold_by_code": sum(r["gold"]["evidence"].startswith("codice") for r in gres),
           "gold_sensitivity": s_gold, "gold_ci_hi_corrected": corrected_upper(g["hi"], s_gold["lo"]),
           "vis": vis, "vis_sensitivity": s_claim, "vis_ci_hi_corrected": corrected_upper(vis["ci"][1], s_claim["lo"])}
    res["gold_certified"] = g["n"] > 0 and res["gold_ci_hi_corrected"] <= GOLD_THRESHOLD
    res["visual_certified"] = vis["total"] > 0 and res["vis_ci_hi_corrected"] <= VISUAL_THRESHOLD
    return res


def final_f2_only(audit: Audit, out: Path) -> None:
    """Audit delle sole soluzioni F2 (stem_audit_export.py --f2-mavis): esiti per soluzione, sensibilità sulle soluzioni
    con una misura della figura cambiata apposta (_gold/calibration_f2.json), soluzioni valide in include_f2_solutions.txt."""
    rows = final_f2(audit, out)
    truth_path = out / "_gold" / "calibration_f2.json"
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else {}
    real, cal_rows = [r for r in rows if r["fid"] not in truth], [r for r in rows if r["fid"] in truth]
    # una soluzione con una misura cambiata è trovata se la procedura la esclude (fatto sulla figura falso o
    # ragionamento non valido: una misura cambiata in un calcolo può risultare in tutti e due)
    proc, agent = Counter(), Counter()
    for r in cal_rows:
        proc[{"exclude": "trovata", "include": "persa"}.get(r["verdict"], "quarantena")] += 1
        for rr in F2_BLIND:
            vis, rea = audit.f2_value(rr, r["fid"], "visual_ok"), audit.f2_value(rr, r["fid"], "reasoning_valid")
            if vis is not None or rea is not None:
                agent["trovata" if (vis and vis[0] is False) or (rea and rea[0] is False) else "persa"] += 1
    sens = rate(proc["trovata"], proc["trovata"] + proc["persa"])
    sens_agent = rate(agent["trovata"], agent["trovata"] + agent["persa"])
    c = Counter(r["verdict"] for r in real)
    vis = [r for r in real if r["visual_ok"]["status"] == "resolved"]
    vis_bad = rate(sum(r["visual_ok"]["value"] is False for r in vis), len(vis))
    rea = [r for r in real if r["reasoning_valid"]["status"] == "resolved"]
    rea_bad = rate(sum(r["reasoning_valid"]["value"] is False for r in rea), len(rea))
    uns = [r for r in real if r["relies_on_unseen"]["status"] == "resolved"]
    uns_bad = rate(sum(r["relies_on_unseen"]["value"] is True for r in uns), len(uns))
    good = rate(c["include"], len(real))
    (out / "include_f2_solutions.txt").write_text("".join(r["key"] + "\n" for r in real if r["verdict"] == "include"))
    by_src = defaultdict(Counter)
    for r in real:
        by_src[r["source"]][r["verdict"]] += 1
    summary = {"solutions": len(real), "verdicts": dict(c), "by_source": {k: dict(v) for k, v in sorted(by_src.items())},
               "valid": good, "visual_not_ok": vis_bad, "reasoning_invalid": rea_bad, "relies_on_unseen": uns_bad,
               "calibration": {"procedura": dict(proc), "agente_singolo": dict(agent), "sensitivity": sens,
                               "sensitivity_agent": sens_agent},
               "visual_not_ok_ci_hi_corrected": corrected_upper(vis_bad["hi"], sens["lo"]) if vis_bad["n"] else None}
    (out / "f2_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    man = json.loads((out / "manifest.json").read_text())
    md = ["# Audit automatico delle soluzioni F2 di Violetto su MAVIS-Geometry", "",
          f"{len(real)} soluzioni selezionate da f2_select.py ({man['args'].get('f2_selected')}), una per problema, in parti uguali "
          f"fra le profondità (seed {man['args']['seed']}); {len(cal_rows)} soluzioni di calibrazione con una misura della figura "
          "cambiata apposta in un punto del ragionamento, mescolate alla cieca. Agenti Haiku: due controlli indipendenti alla cieca "
          "(config/audit/f2_solution.md: fatti sulla figura falsi o non visibili, ragionamento valido, informazioni né nella figura "
          "né nella domanda), poi terzo controllo e arbitri dove serve.", "",
          "## Affidabilità degli agenti (calibrazione)", "",
          f"- Soluzioni con una misura cambiata escluse dalla procedura: {proc['trovata']}/{sens['n']} ({pct(sens['rate'])}, IC 95% {ci(sens)}); "
          f"in quarantena {proc['quarantena']}. Agente singolo: {agent['trovata']}/{sens_agent['n']} ({pct(sens_agent['rate'])}).", "",
          "## Esito", "",
          "| Controllo | Soluzioni | IC 95% | Limite sup. corretto |", "|---|---|---|---|",
          f"| Valide (fatti visibili, ragionamento, nessuna informazione esterna) | {c['include']}/{len(real)} ({pct(good['rate'])}) | {ci(good)} | — |",
          f"| Con fatti sulla figura falsi o non visibili | {vis_bad['k']}/{vis_bad['n']} ({pct(vis_bad['rate'])}) | {ci(vis_bad)} | "
          f"{pct(summary['visual_not_ok_ci_hi_corrected'])} |",
          f"| Ragionamento non valido | {rea_bad['k']}/{rea_bad['n']} ({pct(rea_bad['rate'])}) | {ci(rea_bad)} | — |",
          f"| Informazioni né nella figura né nella domanda | {uns_bad['k']}/{uns_bad['n']} ({pct(uns_bad['rate'])}) | {ci(uns_bad)} | — |",
          "", "Per profondità: " + "; ".join(f"{k} {v.get('include', 0)} valide, {v.get('exclude', 0)} escluse, "
                                             f"{v.get('quarantine', 0)} in quarantena" for k, v in sorted(by_src.items())) + ".",
          "", "## Limiti", "",
          "- Tutti i giudizi vengono da Haiku; la calibrazione misura solo le misure cambiate, non i passaggi di ragionamento "
          "sbagliati, che possono sfuggire di più.",
          "- Le soluzioni sono state scelte fra quelle con la risposta finale corretta (f2_select.py): il tasso di soluzioni valide "
          "vale per i dati che entrerebbero nel training, non per tutte le generazioni.",
          "- Nessun controllo umano durante l'audit; nessuna modifica ai dataset, ai filtri o ai training.",
          "", "Dettaglio per soluzione in audit_f2.jsonl; riepilogo in f2_summary.json; soluzioni valide in include_f2_solutions.txt."]
    if audit.invalid:
        md += ["", f"Lotti con risposte non valide (esclusi): {', '.join(sorted(audit.invalid))}."]
    (out / "audit_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def final(audit: Audit, out: Path, pool: list[Path] = ()) -> None:
    if not audit.items:  # audit delle sole soluzioni F2
        final_f2_only(audit, out)
        return
    rows = {pid: audit.conclude(pid) for pid in sorted(audit.items)}
    audit.save_code()
    cal, cal_items = calibration_stats(audit, rows)
    with open(out / "calibration.jsonl", "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in cal_items)
    dis, quar = [], []
    with open(out / "audit.jsonl", "w") as fh:
        for r in rows.values():
            units, leg, coh, g = r.pop("_units", {}), r.pop("_leg", None), r.pop("_coh", None), r.pop("_gold", None)
            fields = dict(units) | ({"legibility": leg, "coherence": coh} if leg else {})
            r["judgments"] = {f: {k: s[k] for k in ("blind", "arbiters", "status", "value", "how")}
                              for f, s in fields.items() if s["how"] != "consenso"}
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            for f, s in fields.items():
                if s["how"] != "consenso" and (s["arbiters"] or len(s["blind"]) > 2 or s["status"] != "resolved"):
                    dis.append({"pid": r["pid"], "source": r["source"], "calibration": r["calibration"], "field": f,
                                **{k: s[k] for k in ("blind", "arbiters", "status", "value", "how")}})
            if g and g["reviews"]:
                dis.append({"pid": r["pid"], "source": r["source"], "calibration": r["calibration"], "field": "gold",
                            "reference_answer": audit.gold[r["pid"]], **{k: g[k] for k in ("solves", "reviews", "programs", "status", "value", "how")}})
            if "quarantine" in (r["rl"], r["f2"]) and not r["calibration"]:
                quar.append({k: r[k] for k in ("pid", "key", "source", "rl", "f2", "reasons")})
    with open(out / "disagreements.jsonl", "w") as fh:
        fh.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in dis)
    with open(out / "quarantine.jsonl", "w") as fh:
        fh.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in quar)
    sample = [r for r in rows.values() if not r["calibration"]]
    for name, cond in (("include_rl.txt", lambda r: r["rl"] == "include"), ("include_f2.txt", lambda r: r["f2"] == "include"),
                       ("exclude.txt", lambda r: r["rl"] == "exclude"), ("exclude_f2_only.txt", lambda r: r["f2"] == "exclude" and r["rl"] != "exclude"),
                       ("quarantine.txt", lambda r: "quarantine" in (r["rl"], r["f2"]))):
        (out / name).write_text("".join(r["key"] + "\n" for r in sample if cond(r)))

    sens = cal["_sensibilità"]
    s_claim, s_gold = sens["affermazioni (procedura)"]["lo"], sens["risposte (procedura)"]["lo"]
    sizes = json.loads((out / "source_sizes.json").read_text())
    pop = json.loads((out / "population_checks.json").read_text())
    by_src = defaultdict(list)
    for r in sample:
        by_src[r["source"]].append(r)
    table = []
    for src, rs in by_src.items():
        live = [r for r in rs if not r["unreadable"]]
        gres = [r for r in live if r["gold"]["status"] == "resolved"]
        g = rate(sum(r["gold"]["value"] == "wrong" for r in gres), len(gres))
        g_code = sum(r["gold"]["evidence"].startswith("codice") for r in gres)
        g_q = len(live) - len(gres)
        ln = [r for r in rs if r["unreadable"] or r["legibility"]["status"] == "resolved"]
        il = rate(sum(r["unreadable"] or r["legibility"]["value"] == "illegible" for r in ln), len(ln))
        cn = [r for r in live if r["coherence"]["status"] == "resolved" or r["coherence"]["answer_incoherent_by_reviews"]]
        co = rate(sum(r["coherence"]["value"] == "problem" or r["coherence"]["answer_incoherent_by_reviews"] for r in cn), len(cn))
        vis = clustered_rate([(r["claims"]["vis_errors"], r["claims"]["vis_resolved"]) for r in live])
        ded = clustered_rate([(r["claims"]["ded_errors"], r["claims"]["ded_resolved"]) for r in live])
        capn = [r for r in live if r["claims"]["units"] and (r["claims"]["vis_errors"] or r["claims"]["ded_errors"] or not r["claims"]["open"])]
        ca = rate(sum(r["claims"]["vis_errors"] > 0 for r in capn), len(capn))
        facts = Counter()
        for r in live:
            facts.update(r["facts_by_agent"])
        fv = sum(v for k, v in facts.items() if "/visibile/" in k)
        fd = sum(v for k, v in facts.items() if "/deduzione/" in k)
        dis_any = rate(sum(any(r["disagreement"].values()) for r in live), len(live))
        rech = [r for r in live if r["rechecked_s"] or r["rechecked_d"]]
        changed = sum(r["recheck_changed"] for r in rech)
        row = {"source": src, "population": sizes.get(src), "sampled": len(rs), "unreadable_by_code": sum(r["unreadable"] for r in rs),
               "flags_by_code": sum(bool(r.get("deterministic_flags")) for r in live),
               "duplicate_conflicts_by_code": sum(bool((r.get("duplicates") or {}).get("conflict")) for r in live),
               "contamination_candidates": sum(bool(r.get("contamination_candidate")) for r in live),
               "pop_letter_outside_options": pop.get(src, {}).get("lettera_fuori_dalle_opzioni", 0),
               "pop_empty_fields": pop.get(src, {}).get("campi_vuoti", 0), "pop_duplicate_questions": pop.get(src, {}).get("domanda_duplicata", 0),
               "gold_resolved": g["n"], "gold_wrong": g["k"], "gold_wrong_rate": g["rate"], "gold_ci_lo": g["lo"], "gold_ci_hi": g["hi"],
               "gold_ci_hi_corrected": corrected_upper(g["hi"], s_gold), "gold_by_code": g_code, "gold_quarantine": g_q,
               "illegible": il["k"], "illegible_n": il["n"], "illegible_ci_hi": il["hi"],
               "incoherent": co["k"], "incoherent_n": co["n"], "incoherent_ci_hi": co["hi"],
               "vis_units": vis["total"], "vis_false": vis["errors"], "vis_false_rate": vis["rate"], "vis_ci_lo": vis["ci"][0],
               "vis_ci_hi": vis["ci"][1], "vis_ci_hi_corrected": corrected_upper(vis["ci"][1], s_claim), "vis_deff": vis["deff"],
               "vis_deff_rule": vis["deff_rule"], "ded_units": ded["total"], "ded_false": ded["errors"], "ded_false_rate": ded["rate"],
               "ded_ci_hi": ded["ci"][1], "units_open": sum(r["claims"]["open"] for r in live),
               "captions_with_false_visible": ca["k"], "captions_n": ca["n"], "captions_ci_hi": ca["hi"],
               "facts_visible_by_agents": fv, "facts_visible_false_share": (sum(v for k, v in facts.items() if "/visibile/false" in k) / fv) if fv else None,
               "facts_visible_unverifiable_share": (sum(v for k, v in facts.items() if "/visibile/unverifiable" in k) / fv) if fv else None,
               "facts_deductions_by_agents": fd, "facts_deduction_false_share": (sum(v for k, v in facts.items() if "/deduzione/false" in k) / fd) if fd else None,
               "ab_disagreement_rate": dis_any["rate"], "rechecked": len(rech), "recheck_changed": changed,
               "rl_include": sum(r["rl"] == "include" for r in rs), "rl_exclude": sum(r["rl"] == "exclude" for r in rs),
               "rl_quarantine": sum(r["rl"] == "quarantine" for r in rs), "f2_include": sum(r["f2"] == "include" for r in rs),
               "f2_exclude": sum(r["f2"] == "exclude" for r in rs), "f2_quarantine": sum(r["f2"] == "quarantine" for r in rs)}
        row["gold_certified"] = g["n"] > 0 and row["gold_ci_hi_corrected"] <= GOLD_THRESHOLD
        row["visual_certified"] = vis["total"] > 0 and row["vis_ci_hi_corrected"] <= VISUAL_THRESHOLD
        row["rl_promising"] = (row["population"] or 0) >= PROMISING["min_population"] and g["k"] == 0 and g["n"] > 0 \
            and g_q <= PROMISING["max_gold_quarantine"] and il["k"] + co["k"] <= PROMISING["max_bad_items"]
        row["f2_promising"] = row["rl_promising"] and vis["rate"] is not None and vis["rate"] <= PROMISING["max_claim_rate"] \
            and ca["k"] <= PROMISING["max_captions_with_errors"]
        table.append(row)
    table.sort(key=lambda r: (not r["rl_promising"], -(r["population"] or 0)))
    with open(out / "audit_by_source.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        for row in table:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    f2_rows = final_f2(audit, out)
    (out / "calibration_summary.json").write_text(json.dumps(cal, indent=1, ensure_ascii=False))
    pool_res = pooled(sample, cal, list(pool))
    if pool_res:
        (out / "pooled.json").write_text(json.dumps(pool_res, indent=1, ensure_ascii=False))
    report(audit, out, sample, table, f2_rows, cal, pop, pool_res)


def final_f2(audit: Audit, out: Path) -> list[dict]:
    rows = []
    for fid, it in sorted(audit.f2.items()):
        fs = {f: audit.field_state(fid, f) for f in F2_FIELDS}
        ok = all(s["status"] == "resolved" for s in fs.values())
        good = ok and fs["visual_ok"]["value"] is True and fs["reasoning_valid"]["value"] is True and fs["relies_on_unseen"]["value"] is False
        rows.append({"fid": fid, "key": it["key"], "source": it["source"], "verdict": "include" if good else ("exclude" if ok else "quarantine"),
                     **{f: {k: s[k] for k in ("status", "value", "how", "blind", "arbiters")} for f, s in fs.items()}})
    with open(out / "audit_f2.jsonl", "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return rows


def ci(r: dict) -> str:
    return f"{pct(r['lo'])}-{pct(r['hi'])}"


def report(audit: Audit, out: Path, sample: list[dict], table: list[dict], f2_rows: list[dict], cal: dict, pop: dict,
           pool: dict | None = None) -> None:
    man = json.loads((out / "manifest.json").read_text())
    mavis = man["args"].get("mavis_sample") not in (None, "None", "0")
    stage = "MAVIS-Geometry come dati F2" if mavis else (
        "certificazione" if man["args"].get("sources") not in (None, "None") else "screening")
    sens, fa = cal["_sensibilità"], cal["_falsi_allarmi"]
    n_pop = man["population"] or sum(json.loads((out / "source_sizes.json").read_text()).values())
    mavis_q = man.get("mavis_question") or "domanda vision_dominant"  # campioni precedenti a structure_question
    population = (f"popolazione {n_pop} problemi MAVIS-Geometry (caption.jpg, {mavis_q}, ponte con la "
                  "didascalia riscritta da limite_vl.mavis)" if mavis else
                  f"popolazione {man['population']} problemi sft senza decontaminati e domande in giapponese o coreano")
    md = [f"# Audit automatico STEM{'' if mavis else ' di MMFineReason'}: {stage}", "",
          f"Campione: {len(sample)} problemi da {len(table)} sorgenti (seed {man['args']['seed']}); {population}. "
          f"Calibrazione: {len(audit.truth)} elementi con verità nota "
          f"o errori introdotti ({man['units_injected']} affermazioni alterate, {man['gold_injected']} risposte sbagliate apposta), mescolati "
          "alla cieca nei lotti. Agenti Haiku: due controlli indipendenti alla cieca per compito (risolvere senza descrizione; fatti della "
          "descrizione), poi terzo controllo, revisioni della risposta e arbitri dove serve, al più cinque giudizi per campo.", "",
          "Tre livelli di evidenza, sempre separati: **codice** (controlli deterministici, risposta verificata da un programma con input presi "
          "dal testo della domanda), **agenti** (giudizi concordi o risolti da arbitri: non sono una verifica oggettiva), **incerto** "
          "(quarantena).", "",
          "## Affidabilità degli agenti (calibrazione)", "",
          "| Controllo | Trovati dalla procedura | IC 95% | Agente singolo | Persi |", "|---|---|---|---|---|"]
    for label, key, agent in (("Affermazioni alterate", "affermazioni (procedura)", "affermazioni (agente singolo)"),
                              ("Risposte sbagliate introdotte", "risposte (procedura)", None),
                              ("Figure rimpicciolite fino a illeggibili", "illeggibili (procedura)", None),
                              ("Domande scambiate", "domande scambiate (procedura)", None)):
        s = sens[key]
        a = sens[agent] if agent else None
        agent_cell = "—" if a is None else f"{a['k']}/{a['n']} ({pct(a['rate'])})"
        md.append(f"| {label} | {s['k']}/{s['n']} ({pct(s['rate'])}) | {ci(s)} | {agent_cell} | {s['n'] - s['k']} |")
    md += ["", f"Falsi allarmi su verità nota: affermazioni vere di MAVIS segnate false da un agente {fa['affermazioni vere (agente singolo)']['k']}/"
           f"{fa['affermazioni vere (agente singolo)']['n']} ({pct(fa['affermazioni vere (agente singolo)']['rate'])}), dalla procedura "
           f"{fa['affermazioni vere (procedura)']['k']}/{fa['affermazioni vere (procedura)']['n']}; risposte vere giudicate sbagliate "
           f"{fa['risposte vere (procedura)']['k']}/{fa['risposte vere (procedura)']['n']}. Dettaglio in calibration_summary.json e calibration.jsonl.", "",
           f"Correzione per la certificazione: i limiti superiori si dividono per il limite inferiore della sensibilità della procedura "
           f"(affermazioni {pct(sens['affermazioni (procedura)']['lo'])}, risposte {pct(sens['risposte (procedura)']['lo'])}): un errore che gli "
           "agenti non vedono non compare nei conteggi.", "",
           "## Controlli deterministici", "",
           "Su tutta la popolazione (dal testo, per sorgente, in population_checks.json): campi vuoti "
           f"{sum(v.get('campi_vuoti', 0) for v in pop.values())}, risposte lettera fuori dalle opzioni {sum(v.get('lettera_fuori_dalle_opzioni', 0) for v in pop.values())}, "
           f"domande duplicate {sum(v.get('domanda_duplicata', 0) for v in pop.values())}, descrizioni di più immagini "
           f"{sum(v.get('descrizione_di_piu_immagini', 0) for v in pop.values())}. Integrità delle immagini su tutta la popolazione: già verificata "
           "dalla conversione in shard (job 29460, nessuna illeggibile).",
           f"Sul campione: segnalazioni {json.dumps(man['deterministic_flags'], ensure_ascii=False)}; duplicati con risposta diversa "
           f"{man['duplicate_conflicts']}; candidati alla contaminazione p6 non confermati {man['contamination_candidates']}.", "",
           "## Esito per sorgente", "",
           "| Sorgente | Popolazione | Campione | Risposte sbagliate / risolte (risolte con codice) | Limite sup. corretto | Illeggibili | Incoerenti | Fatti visibili falsi | Limite sup. corretto | Deduzioni false | Disaccordo A/B | RL | F2 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        rl = "certificata" if r["gold_certified"] else ("promettente" if r["rl_promising"] else "no")
        f2 = "certificata" if r["gold_certified"] and r["visual_certified"] else ("promettente" if r["f2_promising"] else "no")
        md.append(f"| {r['source']} | {r['population']} | {r['sampled']} | {r['gold_wrong']}/{r['gold_resolved']} ({r['gold_by_code']} da codice) | "
                  f"{pct(r['gold_ci_hi_corrected'])} | {r['illegible']}/{r['illegible_n']} | {r['incoherent']}/{r['incoherent_n']} | "
                  f"{r['vis_false']}/{r['vis_units']} ({pct(r['vis_false_rate'])}) | {pct(r['vis_ci_hi_corrected'])} | "
                  f"{r['ded_false']}/{r['ded_units']} | {pct(r['ab_disagreement_rate'])} | {rl} | {f2} |")
    if pool:
        g, v = pool["gold"], pool["vis"]
        names = ", ".join(Path(d).name for d in pool["audits"])
        md += ["", f"## Problemi MAVIS per profondità insieme (questo audit e {names})", "",
               f"Stessa procedura e stesse regole; un problema presente in più audit conta una volta. {pool['problems']} problemi.", "",
               "| Controllo | Errori / giudicati | IC 95% sup. | Sensibilità della procedura (lim. inf., calibrazioni sommate) | "
               "Limite sup. corretto | Soglia | Certificato |", "|---|---|---|---|---|---|---|",
               f"| Risposte di riferimento ({pool['gold_by_code']} da codice) | {g['k']}/{g['n']} | {pct(g['hi'])} | "
               f"{pool['gold_sensitivity']['k']}/{pool['gold_sensitivity']['n']} ({pct(pool['gold_sensitivity']['lo'])}) | "
               f"{pct(pool['gold_ci_hi_corrected'])} | {GOLD_THRESHOLD:.0%} | {'sì' if pool['gold_certified'] else 'no'} |",
               f"| Fatti visibili della didascalia | {v['errors']}/{v['total']} | {pct(v['ci'][1])} | "
               f"{pool['vis_sensitivity']['k']}/{pool['vis_sensitivity']['n']} ({pct(pool['vis_sensitivity']['lo'])}) | "
               f"{pct(pool['vis_ci_hi_corrected'])} | {VISUAL_THRESHOLD:.0%} | {'sì' if pool['visual_certified'] else 'no'} |",
               "", "Dettaglio in pooled.json."]
    n_rl = Counter(r["rl"] for r in sample)
    n_f2 = Counter(r["f2"] for r in sample)
    ev = Counter(r["gold"]["evidence"] for r in sample if not r["unreadable"])
    md += ["", "## Decisioni sui problemi del campione", "",
           f"- RL: {n_rl['include']} ammessi, {n_rl['exclude']} esclusi, {n_rl['quarantine']} in quarantena (include_rl.txt, exclude.txt, quarantine.txt).",
           f"- F2: {n_f2['include']} ammessi, {n_f2['exclude']} esclusi (di cui {sum(r['f2'] == 'exclude' and r['rl'] != 'exclude' for r in sample)} "
           f"solo per la descrizione, validi per RL: exclude_f2_only.txt), {n_f2['quarantine']} in quarantena.",
           f"- Evidenza sulla risposta di riferimento: {json.dumps(dict(ev), ensure_ascii=False)}.",
           f"- Controllo del 10% dei casi approvati: {sum(r['rechecked'] for r in table)} ricontrollati, {sum(r['recheck_changed'] for r in table)} "
           "con un esito cambiato o rimasto aperto (stima degli errori condivisi da A e B).", "",
           "Motivazioni per problema in audit.jsonl (campo reasons) e quarantine.jsonl."]
    if f2_rows:
        c = Counter(r["verdict"] for r in f2_rows)
        vis = [r for r in f2_rows if r["visual_ok"]["status"] == "resolved"]
        bad = sum(r["visual_ok"]["value"] is False for r in vis)
        b = rate(bad, len(vis))
        md += ["", "## Soluzioni F2 di Violetto (primo lotto, f2-sft120k-v4)", "",
               f"- {len(f2_rows)} soluzioni, una per problema, stratificate per sorgente: {c['include']} valide, {c['exclude']} da escludere, "
               f"{c['quarantine']} in quarantena.",
               f"- Con almeno un'affermazione sulla figura falsa o non verificabile: {bad}/{len(vis)} ({pct(b['rate'])}, IC {ci(b)}; agenti).",
               f"- Ragionamento non valido: {sum(r['reasoning_valid']['value'] is False for r in f2_rows)}; informazioni né nella figura né nella "
               f"domanda: {sum(r['relies_on_unseen']['value'] is True for r in f2_rows)}. Una risposta finale corretta non vuol dire ragionamento visivo valido."]
    md += ["", "## Regole", "",
           f"- Certificazione solo se il limite superiore dell'IC al 95%, diviso per il limite inferiore della sensibilità degli agenti misurata "
           f"sulla calibrazione, sta sotto la soglia: risposte di riferimento {GOLD_THRESHOLD:.0%}, fatti visibili {VISUAL_THRESHOLD:.0%}. Per i fatti "
           "si conta per affermazione (una frase può contenere più fatti: è prudente) e si tiene conto della correlazione dentro il problema "
           "(dimensione efficace n/deff; caso peggiore con meno di 5 errori). I problemi in quarantena non entrano nei denominatori e sono contati a parte.",
           f"- Promettente (regola fissata prima dello screening, per scegliere le sorgenti da certificare): popolazione almeno "
           f"{PROMISING['min_population']}, nessuna risposta sbagliata confermata, al più {PROMISING['max_gold_quarantine']} risposte non risolte, al più "
           f"{PROMISING['max_bad_items']} problema illeggibile o incoerente; per F2 anche fatti visibili falsi al più {PROMISING['max_claim_rate']:.0%} e al più "
           f"{PROMISING['max_captions_with_errors']} descrizione con errori.",
           "- RL: problema ammesso con risposta valida, figura leggibile e domanda coerente, anche se la descrizione è sbagliata (non si usa). "
           "F2: in più nessun fatto falso (visibile o dedotto) nella descrizione e tutti i giudizi risolti. Evidenza insufficiente: quarantena.",
           "", "## Limiti", "",
           "- L'accordo fra modelli non garantisce l'assenza di errori condivisi: tutti i giudizi vengono da Haiku"
           + ("." if mavis else ", che legge le figure peggio di Qwen3-VL-235B.")
           + " La calibrazione misura gli errori che Haiku trova fra quelli introdotti (cambi di numeri e di direzioni); errori di altro "
           "tipo possono sfuggire di più.",
           "- La verifica con codice controlla il calcolo, non la lettura della figura né l'impostazione del problema scritta dall'agente; è "
           "indipendente dall'immagine solo quando tutti gli input stanno nel testo della domanda.",
           "- Le revisioni della risposta vedono la risposta di riferimento e possono esserne influenzate."]
    if mavis and man.get("mavis_question"):
        md += ["- Le domande riscritte (limite_vl.mavis.structure_question) dicono nel testo forme, costruzioni e direzioni prese dalla "
               "didascalia riscritta; le misure restano solo nella figura. I problemi esclusi o in quarantena in un audit precedente "
               "formano la sorgente \"MAVIS-Geometry ricontrollo\", fuori dai conteggi per profondità e dal conteggio comune."]
    elif mavis:
        md += ["- Le domande vision_dominant di MAVIS nominano lati che la figura non disegna (\"Side CD extends into an equilateral "
               "triangle\": il lato CD è tolto) e non ripetono le premesse sulle forme (\"ECFG is a rectangle\"): gli agenti le giudicano "
               "spesso incoerenti anche quando la risposta si ricava dalla figura, e le soluzioni alla cieca discordano di più. Le "
               "esclusioni per coerenza misurano la formulazione delle domande, non errori nelle risposte."]
    else:
        md += ["- Le descrizioni di Qwen scrivono (`None`) per un componente senza etichetta: nel secondo passaggio alla cieca 7 fatti su 702 "
               "unità lo hanno letto come un'etichetta \"None\" e segnato falso; il chiarimento nelle istruzioni vale solo da DC-13 e per gli "
               "arbitri.",
               "- Gli agenti a volte scambiano le immagini di problemi diversi quando ne aprono più d'una insieme (verificato su p299/p301 in "
               "R1-10 con le dimensioni dei PNG); la regola di aprire un'immagine alla volta vale solo dai lotti DC-27, R1-12 e R2 in poi."]
    md += ["- Le figure non sono sempre in scala: alcuni arbitri hanno giudicato false deduzioni che usano dati del testo (rapporti, angoli) "
           "perché il disegno non li rispetta, quindi l'errore sulle deduzioni può essere sovrastimato.",
           "- Un verdetto fuori dai valori ammessi (es. \"unverifiable\" invece di \"not_false\") conta come incerto e manda il campo in quarantena.",
           "- Nessun controllo umano durante l'audit; nessuna modifica ai dataset, ai filtri o ai training."]
    if audit.invalid:
        md += ["", f"Lotti con risposte non valide (esclusi): {', '.join(sorted(audit.invalid))}."]
    (out / "audit_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--recheck-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=None, help="default: seed del campione + 10")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--pool", nargs="*", default=[], type=Path,
                    help="con --final: audit MAVIS precedenti (cartelle) da contare insieme a questo per la certificazione")
    args = ap.parse_args()
    man = json.loads((args.dir / "manifest.json").read_text())
    seed = args.seed if args.seed is not None else int(man["args"]["seed"]) + 10
    audit = Audit(args.dir, args.recheck_frac, seed)
    for name, why in sorted(audit.invalid.items()):
        print(f"DA RIFARE {name}: {why}")
    missing = [n for n in audit.batches if not (args.dir / "answers" / f"{n}.json").exists()]
    if args.final:
        open_tasks = audit.next_tasks()
        if missing or audit.invalid or any(open_tasks.values()):
            print(f"attenzione: lotti senza risposta {missing}, non validi {sorted(audit.invalid)}, compiti ancora da assegnare "
                  f"{({r: len(v) for r, v in open_tasks.items() if v})}")
        final(audit, args.dir, args.pool)
        return
    tasks = audit.next_tasks()
    names = audit.write_tasks(tasks)
    audit.save_code()
    print(json.dumps({"lotti": len(audit.batches), "senza risposta": missing, "nuovi lotti": names,
                      "compiti": {r: len(v) for r, v in tasks.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
