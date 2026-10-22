#!/usr/bin/env python3
"""Pilota metodologico dell'audit F2 su MAVIS, seconda versione (schema f2-pilot/2): candidati deterministici, fatti per
frase classificati dagli agenti, due giudizi per affermazione (configurazione e pagina), categorie che separano gli errori
veri dai cambi di coordinate leciti e dalle ipotesi scartate. È un test del metodo, non una certificazione: non scrive
certificati e gli intervalli su campioni così piccoli non provano niente sulla qualità dei dati. La prima versione
(schema f2-pilot/1, estrazione libera, campione mavis-v2c) è nel commit 86a94e9.

Tre comandi, ognuno in un job CPU (slurm/f2_pilot.sbatch); gli agenti leggono i lotti e scrivono le risposte in mezzo:
- export: campione congelato da una selezione v2 con le domande (selected.v2q.jsonl): soluzioni vere (--real-per-depth
  per profondità fra quelle auditate da v1, mai fra le chiavi dei campioni di sviluppo --exclude-dirs), alterazioni con
  verità certa (misura, costruzione, configurazione: f2_claims.inject_*), di orientamento con verità non garantita
  (categoria a parte, mai usata per dire certificabile una sensibilità), controlli negativi verificabili, controlli sui
  sistemi di riferimento per i verificatori (f2_claims.frame_control: coordinate lecite e rotte, verità certa), e
  facoltativamente item di regressione da un campione di sviluppo (--regression-dir, --regression-pids: solo debug,
  fuori dai criteri). Per ogni item: frasi in categorie (f2_claims.classify_sentences), campione alla cieca delle frasi
  scartate dal generatore (copertura), campione delle frasi verificate dal codice (affidabilità della scorciatoia).
  Scrive _gold/ (mai nei lotti), img/, batches/E-NN.json e manifest.json con codice, istruzioni e criteri congelati;
- round: dopo E (classificazione) costruisce le affermazioni da verificare (fatti uguali uniti, controlli e scorciatoie
  mescolati alla cieca, cid opachi) e scrive i lotti VA e VB; dopo VA e VB scrive i lotti VC (disaccordi su configurazione
  o, se conta, pagina, incerti e il --recheck-frac delle configurazioni giudicate giuste da entrambi);
- final: metriche separate (recall dei candidati, recall della classificazione, sensibilità finale dei verificatori,
  falsi allarmi, copertura del generatore, scorciatoia, controlli sugli assi, disaccordi, quarantene, costo) in
  pilot_metrics.json, facts_resolved.jsonl e pilot_report.md.
round e final si rifiutano di lavorare se codice o istruzioni sono cambiati dopo l'export.

    srun $RUN_PYTHON -u -B scripts/f2_pilot.py export --selection outputs/f2_questions/f2-mavis2k/selected.v2q.jsonl \\
        --mavis-shards 'data/shards/mavis-geometry/9c80fff3-caption/*.tar' \\
        --audit-map outputs/f2_backfill/f2-mavis2k/audit_map-mavis-f2sol.jsonl --out outputs/f2_pilot/mavis-v3a \\
        --exclude-dirs outputs/f2_pilot/mavis-v2a outputs/f2_pilot/mavis-v2b outputs/f2_pilot/mavis-v2c --seed 20261101
    srun $RUN_PYTHON -u -B scripts/f2_pilot.py round --dir outputs/f2_pilot/mavis-v3a
    srun $RUN_PYTHON -u -B scripts/f2_pilot.py final --dir outputs/f2_pilot/mavis-v3a
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2_claims import (BASES, CATEGORIES, KINDS, STANCES, caption_negative, classify_sentences,  # noqa: E402
                                 code_only_measures, cot_negative, fact_category, fact_spans, frame_control, inject_configuration,
                                 inject_construction, inject_measure, inject_orientation, page_class, resolve_claim,
                                 resolve_page, solution_measures, verdict_class)
from limite_vl.provenance import code_identity, file_entry, sha256_file, sha256_text  # noqa: E402
from limite_vl.stem_audit import _norm_map, clopper_pearson, recheck_sample  # noqa: E402

SCHEMA = "f2-pilot/2"
INSTRUCTIONS = {"E": "f2_classify.md", "VA": "f2_claim_verify2.md", "VB": "f2_claim_verify2.md", "VC": "f2_claim_verify2.md",
                "rules": "_rules.md"}
FROZEN_CODE = ("scripts/f2_pilot.py", "src/limite_vl/f2_claims.py", "src/limite_vl/stem_audit.py")
CERTAIN = ("measure", "construction", "configuration")  # verità certa: le sole famiglie per una sensibilità certificabile
CAL_FAMILIES = CERTAIN + ("orientation",)  # orientation: verità non garantita, riportata a parte
MARK = "⟦{}⟧ "
# criteri di accettazione, congelati nell'export prima di vedere le risposte (proposta; si cambiano solo rifacendo
# l'export). Il pilota passa se il generatore copre i punti alterati e quasi tutte le frasi scartate sono vuote, la
# classificazione trova i punti alterati come asserzioni, i verificatori li riconoscono, i controlli negativi, gli assi
# leciti e la scorciatoia del codice non danno falsi allarmi, e quarantene e disaccordi restano pochi.
CRITERIA = {"candidate_recall_min": 1.0,
            "classification_recall_min": {"measure": 11 / 12, "construction": 11 / 12, "configuration": 11 / 12},
            "verifier_sensitivity_min": {"measure": 0.75, "construction": 0.75, "configuration": 0.75},
            "complement_fact_rate_max": 0.03,
            "frame_legit_false_alarm_max": 1, "frame_broken_sensitivity_min": 0.75,
            "shortcut_disagreement_max": 0.02,
            "negative_items_with_false_alarm_max": 1, "negative_fact_false_alarm_max": 0.02,
            "unit_quarantine_max": 0.10, "va_vb_disagreement_max": 0.20}


def rank(s: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}/{s}".encode()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def pack(tasks: list[dict], size, limit: int) -> list[list[dict]]:
    """Task in chiamate, nell'ordine dato, finché la somma di size resta entro limit (un task da solo può superarlo)."""
    calls, cur, n = [], [], 0
    for t in tasks:
        s = size(t)
        if cur and n + s > limit:
            calls.append(cur)
            cur, n = [], 0
        cur.append(t)
        n += s
    if cur:
        calls.append(cur)
    return calls


def write_batches(out: Path, prefix: str, calls: list[list[dict]]) -> list[str]:
    names = []
    for i, call in enumerate(calls, 1):
        name = f"{prefix}-{i:02d}"
        write_json(out / "batches" / f"{name}.json", call)
        names.append(name)
    return names


def _key(statement: str) -> str:
    return _norm_map(statement or "")[0].strip(" .")


# ---------------------------------------------------------------- export

def mavis_problems(shards: str, keys: set[str]) -> dict[str, dict]:
    """Per le chiavi chieste: domanda, didascalia riscritta, CoT (solo se la didascalia non ha frasi riscritte), lettera
    giusta e PNG di caption.jpg dagli shard (tar letti in sequenza, nessun file per campione)."""
    from limite_vl.mavis import f2_problem

    out, pending = {}, {}
    for shard in sorted(glob.glob(shards)):
        with tarfile.open(shard, "r|") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                stem, ext = m.name.rsplit(".", 1)
                if ext not in ("json", "png"):
                    continue
                pending.setdefault(stem, {})[ext] = tar.extractfile(m).read()
                got = pending[stem]
                if "json" in got and "png" in got:
                    meta = json.loads(got["json"])
                    key = f"mavis-{meta['id']}"
                    if key in keys and (p := f2_problem(meta)):
                        out[key] = {"question": p["question"], "caption": p["caption"], "letter": p["answer_letter"],
                                    "cot": p["cot"] if p["rewritten"] == 0 else "", "png": got["png"], "depth": meta["depth"]}
                    del pending[stem]
    return out


def marked_solution(text: str, groups: list[dict]) -> str:
    """La soluzione con il marcatore del gruppo davanti a ogni occorrenza di ogni frase (tutte, non solo quelle in lista:
    l'agente non sa quali frasi il generatore ha scartato)."""
    starts = sorted((a, g["g"]) for g in groups for a, _ in g["spans"])
    out, last = [], 0
    for a, gid in starts:
        out.append(text[last:a])
        out.append(MARK.format(gid))
        last = a
    out.append(text[last:])
    return "".join(out)


def shortcut_claims(it: dict, groups: list[dict], measures: list[dict], frac: float, seed: int) -> list[dict]:
    """Campione delle frasi che saltano gli agenti (measure_only, caption_verbatim): il frac, almeno una se ce ne sono.
    Ogni misura uguale alla didascalia diventa un'affermazione da verificare alla cieca (attesa: giusta)."""
    pool = [g for g in groups if g["category"] in ("measure_only", "caption_verbatim")]
    if not pool:
        return []
    rng = random.Random(f"{seed}/{it['pid']}/shortcut")
    out = []
    for g in rng.sample(pool, max(1, round(frac * len(pool)))):
        a, b = g["spans"][0]
        if g["category"] == "caption_verbatim":
            text = re.sub(r'^"?\s*Figure:\s*', "", g["text"]).strip(' "')
            out.append({"statement": text, "quote": text, "context": g["text"][:320], "basis": "given", "group": g["g"],
                        "category": g["category"]})
            continue
        for d in measures:
            if a <= d["start"] < b and d["verdict"] == "match":
                out.append({"statement": d["text"].strip(), "quote": d["text"].strip(), "context": g["text"][:320],
                            "basis": "given", "group": g["g"], "category": g["category"], "key": d["key"]})
    return out


def export(args) -> None:
    out = args.out
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} non è vuota: un campione congelato non si riscrive")
    t0 = time.time()
    h, recs = hashlib.sha256(), []
    with open(args.selection, "rb") as fh:
        for line in fh:
            h.update(line)
            recs.append(json.loads(line))
    sel_entry = {"path": str(args.selection), "bytes": args.selection.stat().st_size, "sha256": h.hexdigest()}
    eligible = [r for r in recs if not r["failed"] and r.get("question_origin") in ("generation", "reconstructed")]
    audited = {}
    for line in open(args.audit_map):
        r = json.loads(line)
        if r.get("status") == "exact" and not r.get("calibration"):
            audited[r["sid"]] = r.get("verdict")
    excluded, dev = set(), []
    for d in list(args.exclude_dirs or []) + ([args.regression_dir] if args.regression_dir else []):
        p = Path(d) / "_gold" / "items.json"
        if not p.exists():
            sys.exit(f"{p} non esiste: le chiavi dei campioni di sviluppo vanno escluse")
        excluded |= {it["key"] for it in json.loads(p.read_text())}
        dev.append(file_entry(p))
    depth = lambda r: int(r["source_dataset"].rsplit("d", 1)[1])  # noqa: E731  "MAVIS-Geometry d2"
    real = []
    for d in range(4):
        cand = sorted((r for r in eligible if r["sid"] in audited and r["key"] not in excluded and depth(r) == d),
                      key=lambda r: rank(r["sid"], args.seed))
        if len(cand) < args.real_per_depth:
            sys.exit(f"profondità {d}: {len(cand)} soluzioni auditate fuori dallo sviluppo, ne servono {args.real_per_depth}")
        real += cand[:args.real_per_depth]
    used_keys = {r["key"] for r in real} | excluded
    pool = sorted((r for r in eligible if r["sid"] not in audited and r["key"] not in used_keys),
                  key=lambda r: rank(r["sid"], args.seed + 3))
    probs = mavis_problems(args.mavis_shards, {r["key"] for r in real + pool})
    print(f"{len(recs)} record, {len(eligible)} ammissibili, {len(excluded)} chiavi di sviluppo escluse, {len(real)} vere, "
          f"{len(pool)} per la calibrazione, {len(probs)} problemi letti ({time.time() - t0:.0f} s)", flush=True)
    rng = random.Random(args.seed + 30)
    entries = [{"kind": "real", "family": None, "rec": r, "prob": probs[r["key"]], "text": r["solution"], "truth": None}
               for r in real]
    want = {"measure": args.measure, "construction": args.construction, "configuration": args.configuration,
            "orientation": args.orientation}
    inject = {"measure": lambda s, p: inject_measure(s, p["caption"], rng),
              "construction": lambda s, p: inject_construction(s, p["caption"], rng),
              "configuration": lambda s, p: inject_configuration(s, p["caption"], rng),
              "orientation": lambda s, p: inject_orientation(s, rng)}
    for fam in CAL_FAMILIES:
        n = 0
        for r in pool:
            if n >= want[fam]:
                break
            p = probs.get(r["key"])
            if r["key"] in used_keys or p is None:
                continue
            got = inject[fam](r["solution"], p)
            if got:
                truth = {**got[1], "certain": fam in CERTAIN}
                entries.append({"kind": "calibration", "family": fam, "rec": r, "prob": p, "text": got[0], "truth": truth})
                used_keys.add(r["key"])
                n += 1
    for fam, n_want in (("caption_negative", args.caption_negatives), ("cot_negative", args.cot_negatives)):
        n = 0
        for r in pool:
            if n >= n_want:
                break
            p = probs.get(r["key"])
            if r["key"] in used_keys or p is None:
                continue
            text = caption_negative(p["caption"], p["letter"]) if fam == "caption_negative" else cot_negative(p["cot"], p["caption"], p["letter"])
            if text:
                entries.append({"kind": "negative", "family": fam, "rec": r, "prob": p, "text": text,
                                "truth": {"family": fam, "truth": "every figure statement is a caption sentence" if fam == "caption_negative"
                                          else "every measure written equals the caption"}})
                used_keys.add(r["key"])
                n += 1
    if args.regression_dir and args.regression_pids:
        rdir = Path(args.regression_dir)
        old = {it["pid"]: it for it in json.loads((rdir / "_gold" / "items.json").read_text())}
        old_err = Counter()
        if (rdir / "claims_resolved.jsonl").exists():
            for line in open(rdir / "claims_resolved.jsonl"):
                x = json.loads(line)
                old_err[x["pid"]] += x.get("value") == "E"
        for pid in args.regression_pids.split(","):
            it = old[pid]
            png = Path(it["image"]).read_bytes()
            if hashlib.sha256(png).hexdigest() != it["image_sha256"]:
                sys.exit(f"regressione {pid}: immagine diversa da quella del campione di sviluppo")
            rec = {"key": it["key"], "sid": it["sid"] or it["source_sid"], "question": it["question"]}
            entries.append({"kind": "regression", "family": it["family"] or it["kind"], "rec": rec, "text": it["solution"],
                            "truth": None, "prob": {"caption": it["caption"], "png": png, "depth": it["depth"]},
                            "regression_of": {"dir": str(rdir), "pid": pid, "v1_verdict": it["v1_verdict"],
                                              "errors_before": old_err[pid]}})
    entries.sort(key=lambda e: rank(f"{e['kind']}:{e['family']}:{e['rec']['sid']}", args.seed + 1))
    items, truth, measures, sentences, shortcuts, controls = [], {}, {}, {}, {}, {}
    (out / "img").mkdir(parents=True)
    for i, e in enumerate(entries, 1):
        pid, r, p = f"p{i:03d}", e["rec"], e["prob"]
        img = (out / "img" / f"{pid}.png").resolve()  # percorso assoluto: gli agenti lo aprono con Read
        img.write_bytes(p["png"])
        it = {"pid": pid, "kind": e["kind"], "family": e["family"], "key": r["key"], "depth": p["depth"],
              "sid": r["sid"] if e["kind"] in ("real", "regression") else None,
              "source_sid": r["sid"] if e["kind"] in ("calibration", "negative") else None,
              "v1_verdict": audited.get(r["sid"]) if e["kind"] != "regression" else e["regression_of"]["v1_verdict"],
              "question": r["question"], "solution": e["text"], "text_sha256": sha256_text(e["text"]), "image": str(img),
              "image_sha256": hashlib.sha256(p["png"]).hexdigest(), "caption": p["caption"]}
        if e.get("regression_of"):
            it["regression_of"] = e["regression_of"]
        items.append(it)
        measures[pid] = solution_measures(e["text"], p["caption"])
        groups = classify_sentences(e["text"], p["caption"], measures[pid])
        other = [g["g"] for g in groups if g["category"] == "other"]
        comp = set(random.Random(f"{args.seed}/{pid}/complement").sample(other, min(len(other), args.complement_per_item)))
        for g in groups:
            g["listed"] = g["category"] == "candidate" or g["g"] in comp
            g["complement"] = g["g"] in comp
        sentences[pid] = groups
        shortcuts[pid] = shortcut_claims(it, groups, measures[pid], args.shortcut_frac, args.seed)
        if e["truth"]:
            truth[pid] = e["truth"]
    # controlli sugli assi: un item per controllo, alternando lecito e rotto, finché ne servono
    order = sorted(items, key=lambda it: rank(it["pid"], args.seed + 8))
    need = {"frame_legit": args.frame_controls, "frame_broken": args.frame_controls}
    for it in order:
        fam = "frame_legit" if need["frame_legit"] >= need["frame_broken"] else "frame_broken"
        if not need[fam]:
            break
        c = frame_control(it["caption"], random.Random(f"{args.seed}/{it['pid']}/frame"), fam == "frame_broken")
        if c:
            controls[it["pid"]] = c
            need[fam] -= 1
    # lotti E: per item i marcatori in lista (candidati e campione alla cieca delle scartate), a pezzi di --e-ids
    tasks = []
    for it in items:
        groups = sentences[it["pid"]]
        listed = [g["g"] for g in groups if g["listed"]]
        if not listed:
            continue
        text = marked_solution(it["solution"], groups)
        for j in range(0, len(listed), args.e_ids):
            part = listed[j:j + args.e_ids]
            tid = f"{it['pid']}-{j // args.e_ids + 1}"
            tasks.append({"id": tid, "question": it["question"], "solution": text, "list": part})
            for g in groups:
                if g["g"] in part:
                    g["task"] = tid
    tasks.sort(key=lambda t: rank(t["id"], args.seed + 2))
    names = write_batches(out, "E", pack(tasks, lambda t: len(t["list"]), args.e_ids))
    write_json(out / "_gold" / "items.json", items)
    write_json(out / "_gold" / "truth.json", truth)
    write_json(out / "_gold" / "measures.json", measures)
    write_json(out / "_gold" / "sentences.json", sentences)
    write_json(out / "_gold" / "shortcuts.json", shortcuts)
    write_json(out / "_gold" / "controls.json", controls)
    (out / "answers").mkdir()
    obtained = Counter(it["family"] if it["kind"] in ("calibration", "negative") else it["kind"] for it in items)
    cats = Counter(g["category"] for gs in sentences.values() for g in gs)
    frozen = {p: sha256_file(ROOT / p) for p in FROZEN_CODE}
    frozen.update({f"config/audit/{f}": sha256_file(ROOT / "config" / "audit" / f) for f in set(INSTRUCTIONS.values())})
    manifest = {"schema": SCHEMA, "job_id": os.environ.get("SLURM_JOB_ID"), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "seed": args.seed, "selection": sel_entry, "audit_map": file_entry(args.audit_map),
                "development_samples_excluded": dev, "excluded_keys": len(excluded),
                "population": {"records": len(recs), "eligible": len(eligible), "audited_v1": len(audited)},
                "design": {"real_per_depth": args.real_per_depth,
                           "requested": {**want, "caption_negative": args.caption_negatives, "cot_negative": args.cot_negatives,
                                         "frame_controls_each": args.frame_controls,
                                         "regression": args.regression_pids.split(",") if args.regression_pids else []},
                           "obtained": dict(obtained), "frame_controls": dict(Counter(c["family"] for c in controls.values())),
                           "sentence_categories": dict(cats), "listed": sum(g["listed"] for gs in sentences.values() for g in gs),
                           "complement_sampled": sum(g["complement"] for gs in sentences.values() for g in gs),
                           "shortcut_checks": sum(len(v) for v in shortcuts.values()),
                           "complement_per_item": args.complement_per_item, "shortcut_frac": args.shortcut_frac,
                           "e_ids": args.e_ids, "v_claims": args.v_claims, "recheck_frac": args.recheck_frac},
                "criteria": CRITERIA, "batches": {"E": names}, "frozen": frozen,
                "instructions": {k: f"config/audit/{f}" for k, f in INSTRUCTIONS.items()}, "code": code_identity(),
                "categories": CATEGORIES,
                "notes": ["test del metodo, non certificazione: nessun certificato, nessun uso dei tassi per i dati",
                          "orientation: verità non garantita, fuori dai criteri e da ogni sensibilità certificabile",
                          "regressione: item del campione di sviluppo, solo per debug, fuori dai criteri",
                          "measure_only e caption_verbatim saltano gli agenti: il valore uguale alla didascalia non prova il "
                          "ragionamento; un campione (--shortcut-frac) va comunque ai verificatori"]}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"obtained": dict(obtained), "frame_controls": manifest["design"]["frame_controls"],
                      "listed": manifest["design"]["listed"], "E": len(names), "seconds": round(time.time() - t0, 1)},
                     ensure_ascii=False))


# ---------------------------------------------------------------- stato

class Pilot:
    def __init__(self, root: Path):
        self.root = root
        self.man = json.loads((root / "manifest.json").read_text())
        if self.man.get("schema") != SCHEMA:
            sys.exit(f"{root}: schema {self.man.get('schema')}, questo script legge {SCHEMA} (f2-pilot/1: commit 86a94e9)")
        changed = [p for p, h in self.man["frozen"].items() if sha256_file(ROOT / p) != h]
        if changed:
            sys.exit(f"codice o istruzioni cambiati dopo l'export (congelati): {', '.join(changed)}")
        gold = root / "_gold"
        self.items = {it["pid"]: it for it in json.loads((gold / "items.json").read_text())}
        self.truth = json.loads((gold / "truth.json").read_text())
        self.measures = json.loads((gold / "measures.json").read_text())
        self.sentences = {pid: {g["g"]: g for g in gs} for pid, gs in json.loads((gold / "sentences.json").read_text()).items()}
        self.shortcuts = json.loads((gold / "shortcuts.json").read_text())
        self.controls = json.loads((gold / "controls.json").read_text())
        self.units = json.loads((gold / "units.json").read_text()) if (gold / "units.json").exists() else None
        self.batches = {p.stem: json.loads(p.read_text()) for p in sorted((root / "batches").glob("*.json"))}
        self.answers, self.invalid = defaultdict(dict), {}
        for name, batch in self.batches.items():
            path = root / "answers" / f"{name}.json"
            if not path.exists():
                continue
            try:
                got = {str(a["id"]): a for a in json.loads(path.read_text()) if isinstance(a, dict) and "id" in a}
            except Exception as exc:  # noqa: BLE001 - lotto da rifare
                self.invalid[name] = f"JSON non valido: {exc}"[:200]
                continue
            missing = [t["id"] for t in batch if t["id"] not in got]
            if not missing:
                missing = [f"{t['id']}/{x}" for t in batch for x in self._missing(name, t, got[t["id"]])]
            if missing:
                self.invalid[name] = f"{len(missing)} mancanti: {', '.join(missing[:5])}"
                continue
            r = name.rsplit("-", 1)[0]
            for t in batch:
                self.answers[r][t["id"]] = got[t["id"]]

    @staticmethod
    def _missing(name: str, task: dict, ans: dict) -> list[str]:
        """Marcatori (E) o cid (V*) del task senza risposta: il lotto va rifatto."""
        if name.startswith("E-"):
            have = {str(s.get("s")) for s in ans.get("sentences") or [] if isinstance(s, dict)}
            return [x for x in task["list"] if x not in have]
        have = {str(c.get("cid")) for c in ans.get("claims") or [] if isinstance(c, dict)}
        return [c["cid"] for c in task["claims"] if c["cid"] not in have]

    def complete(self, r: str) -> bool:
        names = [n for n in self.batches if n.rsplit("-", 1)[0] == r]
        return bool(names) and all((self.root / "answers" / f"{n}.json").exists() and n not in self.invalid for n in names)

    def facts(self) -> dict[str, list[dict]]:
        """Fatti classificati, per item: la citazione deve ritrovarsi nella frase indicata (altrimenti il fatto non è
        valido e conta come errore della classificazione); campi fuori dagli elenchi presi nel modo più prudente
        (stance asserted, basis figure: contano come errori). fid = <gruppo>.<n>."""
        out = defaultdict(list)
        for tid, a in self.answers["E"].items():
            pid = tid.rsplit("-", 1)[0]
            text, groups = self.items[pid]["solution"], self.sentences[pid]
            for s in a.get("sentences") or []:
                if not isinstance(s, dict) or str(s.get("s")) not in groups or not groups[str(s.get("s"))]["listed"]:
                    continue
                g = groups[str(s["s"])]
                for k, f in enumerate([f for f in s.get("facts") or [] if isinstance(f, dict)][:8], 1):
                    spans = fact_spans(text, [tuple(x) for x in g["spans"]], str(f.get("quote", "")))
                    stance = str(f.get("stance", "")).lower()
                    basis = str(f.get("basis", "")).lower()
                    out[pid].append({"fid": f"{g['g']}.{k}", "group": g["g"], "quote": f.get("quote"),
                                     "statement": str(f.get("statement") or ""), "kind": str(f.get("kind", "")).lower(),
                                     "basis": basis if basis in BASES else "figure",
                                     "stance": stance if stance in STANCES else "asserted",
                                     "used": bool(f.get("used_in_answer")) if stance in ("hypothesis", "rejected") else True,
                                     "fields_ok": stance in STANCES and basis in BASES and str(f.get("kind", "")).lower() in KINDS,
                                     "spans": spans, "valid": bool(spans) and bool(str(f.get("statement") or "").strip()),
                                     "complement": g["complement"]})
        return out

    def judgments(self, r: str) -> dict[tuple[str, str], dict]:
        """(pid, cid) -> giudizio di quel verificatore."""
        out = {}
        for tid, a in self.answers[r].items():
            pid = tid.rsplit("-", 1)[0]
            for c in a.get("claims") or []:
                if isinstance(c, dict):
                    out[(pid, str(c.get("cid")))] = c
        return out


def build_units(pl: Pilot, seed: int, frac: float) -> dict[str, list[dict]]:
    """Affermazioni da verificare per item: i fatti validi uniti per (affermazione normalizzata, base), così ogni
    occorrenza tiene i suoi span e la sua stance; più scorciatoie e controlli sugli assi, mescolati alla cieca, con cid
    opachi in ordine casuale. Un fatto la cui affermazione e la cui citazione dicono solo misure uguali alla didascalia
    (code_only_measures, le stesse in entrambe) non va agli agenti (source code, categoria code_match), salvo il frac
    delle affermazioni di questo tipo, che va ai verificatori come controllo della scorciatoia (shortcut_fact)."""
    facts = pl.facts()
    units = {}
    for pid, it in pl.items.items():
        merged: dict[tuple, dict] = {}
        for f in facts.get(pid, []):
            if not f["valid"]:
                continue
            k = (_key(f["statement"]), f["basis"])
            u = merged.setdefault(k, {"source": "solution", "statement": f["statement"], "basis": f["basis"],
                                      "quote": f["quote"], "group": f["group"], "facts": []})
            u["facts"].append(f["fid"])
            cs, cq = code_only_measures(f["statement"], it["caption"]), code_only_measures(str(f["quote"]), it["caption"])
            u.setdefault("code", True)
            u["code"] &= bool(cs) and bool(cq) and set(cs) <= set(cq)
        code, us = [], []
        for u in merged.values():
            (code if u.pop("code") else us).append(u)
        if code:
            rng = random.Random(f"{seed}/{pid}/code")
            picked = rng.sample(range(len(code)), max(1, round(frac * len(code))))
            for i, u in enumerate(code):
                u["source"] = "shortcut_fact" if i in picked else "code"
            us += [u for u in code if u["source"] == "shortcut_fact"]
        us += [{"source": "shortcut", **c} for c in pl.shortcuts.get(pid, [])]
        if pid in pl.controls:
            us.append({"source": "frame", **pl.controls[pid]})
        random.Random(f"{seed}/{pid}/units").shuffle(us)
        for i, u in enumerate(us, 1):
            u["cid"] = f"c{i:03d}"
            if u["source"] in ("solution", "shortcut_fact"):
                u["context"] = pl.sentences[pid][u["group"]]["text"][:320]
        us += [{**u, "cid": None} for u in code if u["source"] == "code"]  # senza cid: nessun agente le vede
        if us:
            units[pid] = us
    return units


def verify_tasks(pl: Pilot, units: dict[str, list[dict]], size: int, cids: dict[str, set[str]] | None = None) -> list[dict]:
    tasks = []
    for pid, us in sorted(units.items()):
        cl = [{"cid": u["cid"], "statement": u["statement"], "quote": u["quote"], "context": u.get("context", ""),
               "basis": u["basis"]} for u in us if u["cid"] and (cids is None or u["cid"] in cids.get(pid, ()))]
        for j in range(0, len(cl), size):
            tasks.append({"id": f"{pid}-{j // size + 1}", "image": pl.items[pid]["image"], "claims": cl[j:j + size]})
    return tasks


def counts_page(pl: Pilot, pid: str, u: dict, facts_by_id: dict) -> bool:
    """La pagina conta per l'affermazione se almeno un'occorrenza la presenta come proprietà visibile asserita."""
    if u["source"] not in ("solution", "shortcut_fact"):
        return u["basis"] in ("given", "figure")
    return any(facts_by_id[(pid, f)]["basis"] in ("given", "figure") and facts_by_id[(pid, f)]["stance"] == "asserted"
               for f in u["facts"])


def round_(args) -> None:
    pl = Pilot(args.dir)
    seed, design = pl.man["seed"], pl.man["design"]
    for name, why in sorted(pl.invalid.items()):
        print(f"DA RIFARE {name}: {why}")
    has = lambda r: any(n.rsplit("-", 1)[0] == r for n in pl.batches)  # noqa: E731
    new = []
    if pl.complete("E") and not has("VA"):
        units = build_units(pl, seed, design["shortcut_frac"])
        write_json(args.dir / "_gold" / "units.json", units)
        pl.units = units
        tasks = verify_tasks(pl, units, design["v_claims"])
        for r, s in (("VA", seed + 4), ("VB", seed + 5)):
            new += write_batches(args.dir, r, pack(sorted(tasks, key=lambda t: rank(t["id"], s)),
                                                   lambda t: len(t["claims"]), design["v_claims"]))
    elif pl.units and pl.complete("VA") and pl.complete("VB") and not has("VC"):
        va, vb = pl.judgments("VA"), pl.judgments("VB")
        fb = {(pid, f["fid"]): f for pid, fs in pl.facts().items() for f in fs}
        need, agreed = defaultdict(set), []
        for pid, us in pl.units.items():
            for u in us:
                if not u["cid"]:
                    continue
                a, b = va.get((pid, u["cid"]), {}), vb.get((pid, u["cid"]), {})
                st = resolve_claim([a.get("config"), b.get("config")])
                pg = resolve_page([a.get("page"), b.get("page")])
                if st["status"] == "need_blind" or (pg["status"] == "need_blind" and counts_page(pl, pid, u, fb)):
                    need[pid].add(u["cid"])
                elif st["value"] == "K":
                    agreed.append(f"{pid}/{u['cid']}")
        for x in recheck_sample({"units": agreed}, design["recheck_frac"], seed + 6) if agreed else set():
            pid, cid = x.split("/")
            need[pid].add(cid)
        tasks = verify_tasks(pl, pl.units, design["v_claims"], need)
        new += write_batches(args.dir, "VC", pack(sorted(tasks, key=lambda t: rank(t["id"], seed + 7)),
                                                  lambda t: len(t["claims"]), design["v_claims"]))
    print(json.dumps({"nuovi lotti": new, "lotti": len(pl.batches) + len(new), "non validi": sorted(pl.invalid)}, ensure_ascii=False))


# ---------------------------------------------------------------- metriche

def _rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "lo": round(lo, 3), "hi": round(hi, 3)}


def _overlaps(spans, a: int, b: int) -> bool:
    return any(s < b and e > a for s, e in spans)


def final(args) -> None:
    pl = Pilot(args.dir)
    if not pl.units:
        sys.exit("final: mancano le affermazioni da verificare (round dopo E)")
    facts = pl.facts()
    fb = {(pid, f["fid"]): f for pid, fs in facts.items() for f in fs}
    js = {r: pl.judgments(r) for r in ("VA", "VB", "VC")}
    unit_state, per_fact, per_unit = {}, [], []
    for pid, us in pl.units.items():
        for u in us:
            if not u["cid"]:  # scorciatoia del codice, fuori dal campione di controllo
                for fid in u["facts"]:
                    f = fb[(pid, fid)]
                    per_fact.append({"pid": pid, "fid": fid, "cid": None, "kind": f["kind"], "basis": f["basis"],
                                     "stance": f["stance"], "used": f["used"], "complement": f["complement"],
                                     "item_kind": pl.items[pid]["kind"], "family": pl.items[pid]["family"],
                                     "statement": f["statement"], "spans": f["spans"], "category": "code_match"})
                continue
            got = [js[r].get((pid, u["cid"])) for r in ("VA", "VB", "VC")]
            vs = [g for g in got[:2]] + ([got[2]] if got[2] else [])
            cfg = resolve_claim([(g or {}).get("config") for g in vs])
            pg = resolve_page([(g or {}).get("page") for g in vs])
            if cfg["status"] != "resolved":  # dopo l'ultimo giro quel che non è risolto resta incerto
                cfg = {"status": "quarantine", "value": None, "how": cfg.get("how") or "giudizi mancanti o incerti"}
            if pg["status"] != "resolved":
                pg = {"status": "quarantine", "value": None, "how": pg.get("how") or "giudizi mancanti o incerti"}
            unit_state[(pid, u["cid"])] = (cfg, pg)
            per_unit.append({"pid": pid, "cid": u["cid"], "source": u["source"], "family": u.get("family"),
                             "statement": u["statement"], "basis": u["basis"],
                             "config": [(g or {}).get("config") for g in vs], "page": [(g or {}).get("page") for g in vs],
                             "config_state": cfg, "page_state": pg})
            for fid in u.get("facts", []):
                f = fb[(pid, fid)]
                cat = fact_category(f["stance"], f["basis"], f["used"], cfg, pg)
                per_fact.append({"pid": pid, "fid": fid, "cid": u["cid"], "kind": f["kind"], "basis": f["basis"],
                                 "stance": f["stance"], "used": f["used"], "complement": f["complement"],
                                 "item_kind": pl.items[pid]["kind"], "family": pl.items[pid]["family"],
                                 "statement": f["statement"], "spans": f["spans"], "category": cat})
    with open(args.dir / "facts_resolved.jsonl", "w") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in per_fact)
    with open(args.dir / "units_resolved.jsonl", "w") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in per_unit)
    cat_of = {(x["pid"], x["fid"]): x["category"] for x in per_fact}
    m: dict = {"note": "test del metodo, non certificazione: gli intervalli su campioni così piccoli non provano la qualità dei dati"}
    all_facts = [f for fs in facts.values() for f in fs]
    listed = sum(g["listed"] for gs in pl.sentences.values() for g in gs.values())
    m["classification"] = {"listed_sentences": listed, "facts": len(all_facts), "valid": sum(f["valid"] for f in all_facts),
                           "invalid_quotes": sum(not f["valid"] for f in all_facts),
                           "fields_out_of_list": sum(not f["fields_ok"] for f in all_facts),
                           "facts_per_listed_sentence": round(len(all_facts) / max(1, listed), 2),
                           "by_kind": dict(Counter(f["kind"] for f in all_facts)), "by_basis": dict(Counter(f["basis"] for f in all_facts)),
                           "by_stance": dict(Counter(f["stance"] for f in all_facts))}
    # generatore: categorie delle frasi e copertura stimata sul campione alla cieca delle frasi scartate
    cats = Counter((pl.items[pid]["kind"], g["category"]) for pid, gs in pl.sentences.items() for g in gs.values())
    comp = [(pid, g["g"]) for pid, gs in pl.sentences.items() for g in gs.values() if g["complement"]]
    with_fact = {(pid, f["group"]) for pid, fs in facts.items() for f in fs if f["valid"] and f["complement"] and f["stance"] == "asserted"}
    m["generator"] = {"sentence_groups": {f"{k}/{c}": n for (k, c), n in sorted(cats.items())},
                      "complement_sampled": len(comp),
                      "complement_with_asserted_fact": _rate(sum(x in with_fact for x in comp), len(comp)),
                      "complement_examples": [f"{pid}/{g}: {pl.sentences[pid][g]['text'][:160]}" for pid, g in comp if (pid, g) in with_fact][:20]}
    # alterazioni: recall dei candidati, della classificazione, sensibilità finale dei verificatori
    cal = {}
    for fam in CAL_FAMILIES:
        rows = []
        for pid, t in pl.truth.items():
            if t.get("family") != fam:
                continue
            gs = [g for g in pl.sentences[pid].values() if _overlaps(g["spans"], t["start"], t["end"])]
            candidate = any(g["category"] == "candidate" for g in gs)
            fs = [f for f in facts.get(pid, []) if f["valid"] and _overlaps(f["spans"], t["start"], t["end"])]
            asserted = [f for f in fs if f["stance"] == "asserted"]
            cats_here = [cat_of.get((pid, f["fid"])) for f in asserted]
            found = "visual_error" in cats_here or (fam == "orientation" and "orientation_error" in cats_here)
            unit_of = {fid: u["cid"] for u in pl.units.get(pid, []) for fid in u.get("facts", [])}
            agent = {r: any(verdict_class((js[r].get((pid, unit_of.get(f["fid"], ""))) or {}).get("config")) == "E"
                            or (fam == "orientation" and page_class((js[r].get((pid, unit_of.get(f["fid"], ""))) or {}).get("page")) == "C")
                            for f in asserted) for r in ("VA", "VB")}
            code = fam == "measure" and any(d["verdict"] == "contradiction" and d["start"] == t["start"] for d in pl.measures.get(pid, []))
            # l'affermazione riporta ciò che il testo dice dopo l'alterazione (non il valore della didascalia)
            ch = str(t.get("changed", ""))
            tokens = {ch, ch[::-1]} if len(ch) == 2 and ch.isupper() else {ch}
            faithful = None if len(ch) < 2 else any(re.search(rf"(?<![A-Za-z0-9]){re.escape(x)}(?![A-Za-z0-9])", f["statement"], re.I)
                                                    for f in asserted for x in tokens)
            rows.append({"pid": pid, "mode": t.get("mode"), "candidate": candidate, "classified": bool(fs), "faithful": faithful,
                         "classified_asserted": bool(asserted), "stances": sorted({f["stance"] for f in fs}), "found": found,
                         "quarantined": not found and "quarantine" in cats_here, "code": code, **agent})
        cl = [r for r in rows if r["classified_asserted"]]
        cal[fam] = {"n": len(rows), "certain": fam in CERTAIN,
                    "candidate_recall": _rate(sum(r["candidate"] for r in rows), len(rows)),
                    "classification_recall": _rate(len(cl), len(rows)),
                    "classification_any_stance": _rate(sum(r["classified"] for r in rows), len(rows)),
                    "statement_faithful": _rate(sum(bool(r["faithful"]) for r in cl if r["faithful"] is not None),
                                                sum(r["faithful"] is not None for r in cl)),
                    "verifier_sensitivity": _rate(sum(r["found"] for r in cl), len(cl)),
                    "end_to_end": _rate(sum(r["found"] for r in rows), len(rows)),
                    "quarantined": sum(r["quarantined"] for r in rows),
                    "agent_sensitivity": {a: _rate(sum(r[a] for r in cl), len(cl)) for a in ("VA", "VB")},
                    "code_or_agents": _rate(sum(r["found"] or r["code"] for r in rows), len(rows)) if fam == "measure" else None,
                    "items": rows}
    m["calibration"] = cal
    # controlli sugli assi e scorciatoia del codice
    fr = defaultdict(list)
    for x in per_unit:
        if x["source"] == "frame":
            fr[x["family"]].append(x["config_state"])
    m["frame_controls"] = {"legit": {"n": len(fr["frame_legit"]), "false_alarms": sum(s.get("value") == "E" for s in fr["frame_legit"]),
                                     "quarantined": sum(s["status"] == "quarantine" for s in fr["frame_legit"])},
                           "broken": {"n": len(fr["frame_broken"]),
                                      "sensitivity": _rate(sum(s.get("value") == "E" for s in fr["frame_broken"]), len(fr["frame_broken"])),
                                      "quarantined": sum(s["status"] == "quarantine" for s in fr["frame_broken"])}}
    sc = [x for x in per_unit if x["source"] in ("shortcut", "shortcut_fact")]
    m["shortcut"] = {"checks": len(sc), "sentences": sum(x["source"] == "shortcut" for x in sc),
                     "facts_checked": sum(x["source"] == "shortcut_fact" for x in sc),
                     "facts_by_code_only": sum(x["category"] == "code_match" for x in per_fact), "disagreement": _rate(sum(x["config_state"].get("value") == "E" for x in sc), len(sc)),
                     "quarantined": sum(x["config_state"]["status"] == "quarantine" for x in sc),
                     "examples": [f"{x['pid']}/{x['cid']}: {x['statement'][:120]}" for x in sc if x["config_state"].get("value") == "E"][:10],
                     "note": "un valore uguale alla didascalia non prova il ragionamento: misura solo l'affidabilità della scorciatoia"}
    # controlli negativi: nessun errore visivo contato atteso
    neg = [pid for pid, it in pl.items.items() if it["kind"] == "negative"]
    nf = [x for x in per_fact if x["pid"] in neg]
    fa = sorted({x["pid"] for x in nf if x["category"] in ("visual_error",)})
    m["negatives"] = {"items": len(neg), "items_with_false_alarm": len(fa), "false_alarm_items": fa, "facts": len(nf),
                      "fact_false_alarm": _rate(sum(x["category"] == "visual_error" for x in nf), len(nf)),
                      "orientation_errors": sum(x["category"] == "orientation_error" for x in nf),
                      "code_contradictions": sum(d["verdict"] == "contradiction" for pid in neg for d in pl.measures.get(pid, []))}
    # disaccordi e quarantene sulle affermazioni verificate
    both = [x for x in per_unit if all(verdict_class(v) for v in x["config"][:2])]
    m["agreement"] = {"units": len(per_unit), "both_certain": len(both),
                      "va_vb_disagreement": _rate(sum(verdict_class(x["config"][0]) != verdict_class(x["config"][1]) for x in both), len(both)),
                      "page_disagreement": sum(page_class(x["page"][0]) != page_class(x["page"][1]) for x in per_unit if len(x["page"]) >= 2),
                      "sent_to_vc": sum(len(x["config"]) == 3 for x in per_unit),
                      "unit_quarantine": _rate(sum(x["config_state"]["status"] == "quarantine" for x in per_unit), len(per_unit)),
                      "items_with_quarantine": len({x["pid"] for x in per_unit if x["config_state"]["status"] == "quarantine"})}
    # soluzioni vere e regressione: categorie per item, confronto con v1 e con il campione di sviluppo
    def by_item(kind: str) -> dict:
        pids = sorted(pid for pid, it in pl.items.items() if it["kind"] == kind)
        rows = {}
        for pid in pids:
            c = Counter(x["category"] for x in per_fact if x["pid"] == pid)
            rows[pid] = {"v1": pl.items[pid]["v1_verdict"], **{k: c.get(k, 0) for k in CATEGORIES},
                         "code_contradictions": sum(d["verdict"] == "contradiction" for d in pl.measures.get(pid, []))}
            if pl.items[pid].get("regression_of"):
                rows[pid]["before"] = pl.items[pid]["regression_of"]
        return {"items": len(pids), "with_visual_error": sum(r["visual_error"] > 0 for r in rows.values()),
                "with_orientation_error": sum(r["orientation_error"] > 0 for r in rows.values()),
                "with_reasoning_flag": sum(r["reasoning_flag"] > 0 for r in rows.values()),
                "categories": dict(sum((Counter({k: r[k] for k in CATEGORIES}) for r in rows.values()), Counter())),
                "v1_vs_pilot": dict(Counter(f"v1 {r['v1']} / pilota {'errore' if r['visual_error'] else 'nessun errore'}" for r in rows.values())),
                "per_item": rows}
    m["real"] = by_item("real")
    m["regression"] = by_item("regression")
    # costo
    calls = Counter(n.rsplit("-", 1)[0] for n in pl.batches)
    n_sol = sum(it["kind"] != "negative" for it in pl.items.values())
    m["cost"] = {"calls": dict(calls), "total_calls": sum(calls.values()),
                 "calls_per_solution_item": round(sum(calls.values()) / max(1, n_sol), 2),
                 "batch_bytes": {r: sum(len(json.dumps(b, ensure_ascii=False).encode()) for n, b in pl.batches.items()
                                        if n.rsplit("-", 1)[0] == r) for r in calls},
                 "invalid_batches": sorted(pl.invalid)}
    usage = args.dir / "answers" / "_usage.json"
    if usage.exists():
        m["cost"]["usage"] = json.loads(usage.read_text())
    # criteri congelati nell'export
    cr, checks = pl.man["criteria"], {}
    for fam in CERTAIN:
        if cal[fam]["n"]:
            checks[f"candidate_recall.{fam}"] = cal[fam]["candidate_recall"]["rate"] >= cr["candidate_recall_min"]
            checks[f"classification_recall.{fam}"] = cal[fam]["classification_recall"]["rate"] >= cr["classification_recall_min"][fam] - 1e-9
            checks[f"verifier_sensitivity.{fam}"] = (cal[fam]["verifier_sensitivity"]["rate"] or 0) >= cr["verifier_sensitivity_min"][fam]
    checks["complement_fact_rate"] = (m["generator"]["complement_with_asserted_fact"]["rate"] or 0) <= cr["complement_fact_rate_max"]
    if m["frame_controls"]["legit"]["n"]:
        checks["frame_legit_false_alarms"] = m["frame_controls"]["legit"]["false_alarms"] <= cr["frame_legit_false_alarm_max"]
    if m["frame_controls"]["broken"]["n"]:
        checks["frame_broken_sensitivity"] = (m["frame_controls"]["broken"]["sensitivity"]["rate"] or 0) >= cr["frame_broken_sensitivity_min"]
    if sc:
        checks["shortcut_disagreement"] = (m["shortcut"]["disagreement"]["rate"] or 0) <= cr["shortcut_disagreement_max"]
    checks["negative_items"] = m["negatives"]["items_with_false_alarm"] <= cr["negative_items_with_false_alarm_max"]
    checks["negative_facts"] = (m["negatives"]["fact_false_alarm"]["rate"] or 0) <= cr["negative_fact_false_alarm_max"]
    checks["unit_quarantine"] = (m["agreement"]["unit_quarantine"]["rate"] or 0) <= cr["unit_quarantine_max"]
    checks["va_vb_disagreement"] = (m["agreement"]["va_vb_disagreement"]["rate"] or 0) <= cr["va_vb_disagreement_max"]
    m["criteria"] = {"thresholds": cr, "passed": checks, "all_passed": all(checks.values())}
    write_json(args.dir / "pilot_metrics.json", m)
    write_report(args.dir, pl, m, checks)


def write_report(out: Path, pl: Pilot, m: dict, checks: dict) -> None:
    pct = lambda r: "—" if r.get("rate") is None else f"{100 * r['rate']:.0f}%"  # noqa: E731
    frac = lambda r: f"{r['k']}/{r['n']}"  # noqa: E731
    cal, g = m["calibration"], m["generator"]
    obtained = pl.man["design"]["obtained"]
    md = ["# Pilota dell'audit F2, schema f2-pilot/2 (test del metodo, non certificazione)", "",
          f"Item: {', '.join(f'{v} {k}' for k, v in sorted(obtained.items()))}. Frasi in lista agli agenti "
          f"{m['classification']['listed_sentences']}, fatti {m['classification']['facts']} "
          f"({m['classification']['facts_per_listed_sentence']} per frase), citazioni non ritrovate "
          f"{m['classification']['invalid_quotes']}, campi fuori elenco {m['classification']['fields_out_of_list']}.", "",
          "| Famiglia | Verità | Candidati | Classificazione (asserita) | Sensibilità dei verificatori | Da capo a fondo | Quarantene |",
          "|---|---|---|---|---|---|---|"]
    for f in CAL_FAMILIES:
        c = cal[f]
        md.append(f"| {f} | {'certa' if c['certain'] else 'non garantita'} | {frac(c['candidate_recall'])} | "
                  f"{frac(c['classification_recall'])} | {frac(c['verifier_sensitivity'])} ({pct(c['verifier_sensitivity'])}) | "
                  f"{frac(c['end_to_end'])} | {c['quarantined']} |")
    if cal["measure"]["code_or_agents"]:
        md.append(f"\nMisure: codice o agenti {frac(cal['measure']['code_or_agents'])}.")
    fc, sc, ng, ag = m["frame_controls"], m["shortcut"], m["negatives"], m["agreement"]
    md += ["", f"- Copertura del generatore: {frac(g['complement_with_asserted_fact'])} frasi scartate (campione alla cieca) "
               f"con un fatto asserito ({pct(g['complement_with_asserted_fact'])}).",
           f"- Controlli sugli assi: leciti {fc['legit']['false_alarms']}/{fc['legit']['n']} falsi allarmi; rotti "
           f"{frac(fc['broken']['sensitivity'])} trovati.",
           f"- Scorciatoia del codice: {frac(sc['disagreement'])} misure uguali alla didascalia giudicate sbagliate.",
           f"- Controlli negativi: {ng['items_with_false_alarm']}/{ng['items']} item con un errore visivo contato; "
           f"{frac(ng['fact_false_alarm'])} fatti ({pct(ng['fact_false_alarm'])}); orientamenti contraddetti {ng['orientation_errors']}.",
           f"- Disaccordo VA/VB sulla configurazione {pct(ag['va_vb_disagreement'])}; mandate a VC {ag['sent_to_vc']}; "
           f"quarantene {frac(ag['unit_quarantine'])} ({pct(ag['unit_quarantine'])}).",
           f"- Soluzioni vere: {m['real']['with_visual_error']}/{m['real']['items']} con un errore visivo contato, "
           f"{m['real']['with_orientation_error']} con un orientamento contraddetto, {m['real']['with_reasoning_flag']} con "
           f"un'ipotesi falsa usata per la risposta; categorie {m['real']['categories']}; confronto con v1 {m['real']['v1_vs_pilot']}.",
           f"- Costo: {m['cost']['total_calls']} chiamate ({m['cost']['calls']}), {m['cost']['calls_per_solution_item']} per "
           f"soluzione.", "",
           f"Criteri congelati nell'export: {'tutti soddisfatti' if m['criteria']['all_passed'] else 'non tutti soddisfatti'} "
           f"({', '.join(k for k, v in checks.items() if not v) or 'nessuno mancato'}).", "",
           "Gli intervalli su campioni così piccoli non provano nulla sulla qualità dei dati: il pilota decide solo se il metodo "
           "regge. L'orientamento ha verità non garantita ed è fuori dai criteri; gli item di regressione vengono dal campione "
           "di sviluppo e sono fuori dai criteri."]
    (out / "pilot_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--selection", required=True, type=Path)
    e.add_argument("--mavis-shards", required=True)
    e.add_argument("--audit-map", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--exclude-dirs", nargs="*", default=[], help="campioni di sviluppo: le loro chiavi non entrano")
    e.add_argument("--real-per-depth", type=int, default=6)
    e.add_argument("--measure", type=int, default=12)
    e.add_argument("--construction", type=int, default=12)
    e.add_argument("--configuration", type=int, default=12)
    e.add_argument("--orientation", type=int, default=6)
    e.add_argument("--caption-negatives", type=int, default=6)
    e.add_argument("--cot-negatives", type=int, default=6)
    e.add_argument("--frame-controls", type=int, default=12, help="controlli sugli assi per tipo (leciti e rotti)")
    e.add_argument("--regression-dir", default=None)
    e.add_argument("--regression-pids", default="", help="pid del campione di sviluppo, separati da virgole")
    e.add_argument("--complement-per-item", type=int, default=8, help="frasi scartate dal generatore mandate alla cieca")
    e.add_argument("--shortcut-frac", type=float, default=0.2, help="frazione delle frasi verificate dal codice ricontrollate")
    e.add_argument("--e-ids", type=int, default=140, help="frasi in lista per task e per chiamata di classificazione")
    e.add_argument("--v-claims", type=int, default=60, help="affermazioni per chiamata di verifica")
    e.add_argument("--recheck-frac", type=float, default=0.10)
    for name in ("round", "final"):
        sub.add_parser(name).add_argument("--dir", required=True, type=Path)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_pilot.py va eseguito in un job SLURM")
    {"export": export, "round": round_, "final": final}[args.cmd](args)


if __name__ == "__main__":
    main()
