#!/usr/bin/env python3
"""Pilota metodologico dell'audit F2 v2 su MAVIS: estrazione delle affermazioni, confronto deterministico delle misure,
verifica indipendente per affermazione. È un test del metodo, non una certificazione: non scrive certificati e gli
intervalli su campioni così piccoli non provano niente sulla qualità dei dati.

Tre comandi, ognuno in un job CPU (slurm/f2_pilot.sbatch); gli agenti leggono i lotti e scrivono le risposte in mezzo:
- export: campione congelato da una selezione v2 con le domande (selected.v2q.jsonl di f2_questions.py): soluzioni vere
  (--real-per-depth per profondità, scelte fra quelle auditate da mavis-f2sol per avere un confronto), alterazioni di
  misura con verità nota (f2_claims.inject_measure), di costruzione con verità nota (inject_construction), di
  orientamento con verità non garantita (inject_orientation), controlli negativi verificabili (frasi della didascalia
  certificata, CoT del motore con le misure uguali alla didascalia). Per ogni item il confronto deterministico delle misure
  (f2_claims.solution_measures). Scrive _gold/ (mai nei lotti), img/, batches/E-NN.json e manifest.json con i criteri
  di riuscita fissati prima di vedere le risposte;
- round: dopo le risposte E valida le citazioni (devono ritrovarsi nel testo) e scrive i lotti VA e VB (immagine e
  affermazioni, senza la soluzione); dopo VA e VB scrive i lotti VC (disaccordi, incerti e il 15% delle affermazioni
  giudicate vere da entrambi);
- final: metriche separate (recall dell'estrazione sul punto alterato, sensibilità attribuibile dei verificatori, falsi
  allarmi sui controlli negativi, disaccordi, quarantene, accordo con il confronto deterministico, costo per soluzione)
  in pilot_metrics.json e pilot_report.md.

    srun $RUN_PYTHON -u -B scripts/f2_pilot.py export --selection outputs/f2_questions/f2-mavis2k/selected.v2q.jsonl \
        --mavis-shards 'data/shards/mavis-geometry/9c80fff3-caption/*.tar' \
        --audit-map outputs/f2_backfill/f2-mavis2k/audit_map-mavis-f2sol.jsonl --out outputs/f2_pilot/mavis-v2a --seed 20261023
    srun $RUN_PYTHON -u -B scripts/f2_pilot.py round --dir outputs/f2_pilot/mavis-v2a
    srun $RUN_PYTHON -u -B scripts/f2_pilot.py final --dir outputs/f2_pilot/mavis-v2a
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import statistics
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2_claims import (caption_negative, context_sentence, cot_negative, inject_construction,  # noqa: E402
                                 inject_measure, inject_orientation, quote_spans, resolve_claim, solution_measures,
                                 verdict_class)
from limite_vl.provenance import code_identity, file_entry, sha256_file, sha256_text  # noqa: E402
from limite_vl.stem_audit import clopper_pearson, recheck_sample  # noqa: E402

INSTRUCTIONS = {"E": "f2_extract.md", "VA": "f2_claim_verify.md", "VB": "f2_claim_verify.md", "VC": "f2_claim_verify.md",
                "rules": "_rules.md"}
CAL_FAMILIES = ("measure", "construction", "orientation")
# criteri di riuscita del pilota, fissati nell'export prima di vedere le risposte (proposta, modificabile solo rifacendo
# l'export): il pilota "funziona" se l'estrazione trova i punti alterati, i verificatori li riconoscono, i controlli
# negativi non danno falsi allarmi e le quarantene restano poche
CRITERIA = {"extraction_recall_min": {"measure": 11 / 12, "construction": 5 / 6, "orientation": 5 / 6},
            "verifier_sensitivity_min": {"measure": 0.75, "construction": 4 / 6, "orientation": 0.5},
            "negative_items_with_false_alarm_max": 1, "negative_claim_false_alarm_max": 0.02,
            "claim_quarantine_max": 0.10, "va_vb_disagreement_max": 0.20, "code_contradictions_flagged_min": 0.75}


def rank(s: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}/{s}".encode()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def batches(out: Path, prefix: str, tasks: list[dict], size: int) -> list[str]:
    names = []
    for b in range(0, len(tasks), size):
        name = f"{prefix}-{b // size + 1:02d}"
        write_json(out / "batches" / f"{name}.json", tasks[b:b + size])
        names.append(name)
    return names


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
        if r.get("status") == "exact":
            audited[r["sid"]] = r.get("verdict")
    depth = lambda r: int(r["source_dataset"].rsplit("d", 1)[1])  # noqa: E731  "MAVIS-Geometry d2"
    real = []
    for d in range(4):
        cand = sorted((r for r in eligible if r["sid"] in audited and depth(r) == d), key=lambda r: rank(r["sid"], args.seed))
        real += cand[:args.real_per_depth]
    used = {r["sid"] for r in real}
    used_keys = {r["key"] for r in real}
    pool = sorted((r for r in eligible if r["sid"] not in audited and r["sid"] not in used), key=lambda r: rank(r["sid"], args.seed + 3))
    probs = mavis_problems(args.mavis_shards, {r["key"] for r in real + pool})
    print(f"{len(recs)} record, {len(eligible)} ammissibili, {len(real)} veri, {len(pool)} per la calibrazione, "
          f"{len(probs)} problemi letti ({time.time() - t0:.0f} s)", flush=True)
    rng = random.Random(args.seed + 30)
    entries = [{"kind": "real", "family": None, "rec": r, "text": r["solution"], "truth": None} for r in real]
    want = {"measure": args.measure, "construction": args.construction, "orientation": args.orientation}
    inject = {"measure": lambda s, p: inject_measure(s, p["caption"], rng),
              "construction": lambda s, p: inject_construction(s, p["caption"], rng),
              "orientation": lambda s, p: inject_orientation(s, rng)}
    for fam in CAL_FAMILIES:
        n = 0
        for r in pool:
            if n >= want[fam]:
                break
            p = probs.get(r["key"])
            if r["sid"] in used or r["key"] in used_keys or p is None:
                continue
            got = inject[fam](r["solution"], p)
            if got:
                entries.append({"kind": "calibration", "family": fam, "rec": r, "text": got[0], "truth": got[1]})
                used.add(r["sid"])
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
                entries.append({"kind": "negative", "family": fam, "rec": r, "text": text,
                                "truth": {"family": fam, "truth": "every figure statement is a caption sentence" if fam == "caption_negative"
                                          else "every measure written equals the caption"}})
                used_keys.add(r["key"])
                n += 1
    entries.sort(key=lambda e: rank(f"{e['kind']}:{e['family']}:{e['rec']['sid']}", args.seed + 1))
    items, truth, measures = [], {}, {}
    (out / "img").mkdir(parents=True)
    for i, e in enumerate(entries, 1):
        pid, r = f"p{i:03d}", e["rec"]
        p = probs[r["key"]]
        img = (out / "img" / f"{pid}.png").resolve()  # percorso assoluto: gli agenti lo aprono con Read
        img.write_bytes(p["png"])
        items.append({"pid": pid, "kind": e["kind"], "family": e["family"], "key": r["key"], "depth": p["depth"],
                      "sid": r["sid"] if e["kind"] == "real" else None, "source_sid": r["sid"] if e["kind"] != "real" else None,
                      "v1_verdict": audited.get(r["sid"]), "question": r["question"], "solution": e["text"],
                      "text_sha256": sha256_text(e["text"]), "image": str(img), "image_sha256": hashlib.sha256(p["png"]).hexdigest(),
                      "caption": p["caption"]})
        measures[pid] = solution_measures(e["text"], p["caption"])
        if e["truth"]:
            truth[pid] = e["truth"]
    write_json(out / "_gold" / "items.json", items)
    write_json(out / "_gold" / "truth.json", truth)
    write_json(out / "_gold" / "measures.json", measures)
    order = sorted(items, key=lambda it: rank(it["pid"], args.seed + 2))
    names = batches(out, "E", [{"id": it["pid"], "question": it["question"], "solution": it["solution"]} for it in order], args.e_batch)
    (out / "answers").mkdir()
    obtained = Counter(it["family"] or "real" for it in items)
    manifest = {"schema": "f2-pilot/1", "job_id": os.environ.get("SLURM_JOB_ID"), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "seed": args.seed, "selection": sel_entry, "audit_map": file_entry(args.audit_map),
                "population": {"records": len(recs), "eligible": len(eligible), "audited_v1": len(audited)},
                "design": {"real_per_depth": args.real_per_depth, "requested": {**want, "caption_negative": args.caption_negatives,
                                                                              "cot_negative": args.cot_negatives},
                           "obtained": dict(obtained), "e_batch": args.e_batch, "v_batch": args.v_batch,
                           "recheck_frac": args.recheck_frac},
                "criteria": CRITERIA, "batches": {"E": names},
                "instructions": {k: {"path": f"config/audit/{f}", "sha256": sha256_file(ROOT / "config" / "audit" / f)}
                                 for k, f in INSTRUCTIONS.items()},
                "code": code_identity(),
                "notes": ["test del metodo, non certificazione: nessun certificato, nessun uso dei tassi per i dati",
                          "orientation: verità non garantita (stima prudente della sensibilità)",
                          "le soluzioni vere sono fra le 150 di mavis-f2sol: v1_verdict permette il confronto con l'audit v1"]}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"obtained": dict(obtained), "E": len(names), "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))


# ---------------------------------------------------------------- stato

class Pilot:
    def __init__(self, root: Path):
        self.root = root
        self.man = json.loads((root / "manifest.json").read_text())
        self.items = {it["pid"]: it for it in json.loads((root / "_gold" / "items.json").read_text())}
        self.truth = json.loads((root / "_gold" / "truth.json").read_text())
        self.measures = json.loads((root / "_gold" / "measures.json").read_text())
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
            if missing:
                self.invalid[name] = f"{len(missing)} item mancanti"
                continue
            r = name.rsplit("-", 1)[0]
            for t in batch:
                self.answers[r][t["id"]] = got[t["id"]]

    def complete(self, r: str) -> bool:
        names = [n for n in self.batches if n.rsplit("-", 1)[0] == r]
        return bool(names) and all((self.root / "answers" / f"{n}.json").exists() and n not in self.invalid for n in names)

    def claims(self) -> dict[str, list[dict]]:
        """Affermazioni estratte e validate: una citazione che non si ritrova nel testo è scartata (conta come errore
        dell'estrazione); cid in ordine, spans = tutte le occorrenze."""
        out = {}
        for pid, a in self.answers["E"].items():
            text, kept = self.items[pid]["solution"], []
            for c in a.get("claims") or []:
                if not isinstance(c, dict):
                    continue
                spans = quote_spans(text, str(c.get("quote", "")))
                kept.append({"cid": f"c{len(kept) + 1:02d}", "quote": c.get("quote"), "statement": c.get("statement"),
                             "kind": str(c.get("kind", "")).lower(), "spans": spans, "valid": bool(spans)})
            out[pid] = kept
        return out

    def verdicts(self, r: str, pid: str) -> dict[str, str | None]:
        a = self.answers[r].get(pid) or {}
        return {str(c.get("cid")): c.get("verdict") for c in a.get("claims") or [] if isinstance(c, dict)}


def verify_task(it: dict, claims: list[dict], cids: set[str] | None = None) -> dict:
    return {"id": it["pid"], "image": it["image"],
            "claims": [{"cid": c["cid"], "statement": c["statement"], "quote": c["quote"],
                        "context": context_sentence(it["solution"], c["spans"][0])}
                       for c in claims if c["valid"] and (cids is None or c["cid"] in cids)]}


def round_(args) -> None:
    pl = Pilot(args.dir)
    seed = pl.man["seed"]
    for name, why in sorted(pl.invalid.items()):
        print(f"DA RIFARE {name}: {why}")
    has = lambda r: any(n.rsplit("-", 1)[0] == r for n in pl.batches)  # noqa: E731
    new = []
    if pl.complete("E") and not has("VA"):
        claims = pl.claims()
        tasks = [verify_task(pl.items[pid], cs) for pid, cs in sorted(claims.items())]
        tasks = [t for t in tasks if t["claims"]]
        for r, s in (("VA", seed + 4), ("VB", seed + 5)):
            new += batches(args.dir, r, sorted(tasks, key=lambda t: rank(t["id"], s)), pl.man["design"]["v_batch"])
    elif pl.complete("VA") and pl.complete("VB") and not has("VC"):
        claims = pl.claims()
        need, agreed = defaultdict(set), []
        for pid, cs in claims.items():
            va, vb = pl.verdicts("VA", pid), pl.verdicts("VB", pid)
            for c in cs:
                if not c["valid"]:
                    continue
                st = resolve_claim([va.get(c["cid"]), vb.get(c["cid"])])
                if st["status"] == "need_blind":
                    need[pid].add(c["cid"])
                elif st["value"] == "K":
                    agreed.append(f"{pid}/{c['cid']}")
        for x in recheck_sample({"claims": agreed}, pl.man["design"]["recheck_frac"], seed + 6) if agreed else set():
            pid, cid = x.split("/")
            need[pid].add(cid)
        tasks = [verify_task(pl.items[pid], claims[pid], cids) for pid, cids in sorted(need.items())]
        new += batches(args.dir, "VC", sorted(tasks, key=lambda t: rank(t["id"], seed + 7)), pl.man["design"]["v_batch"])
    print(json.dumps({"nuovi lotti": new, "lotti": len(pl.batches) + len(new), "non validi": sorted(pl.invalid)}, ensure_ascii=False))


# ---------------------------------------------------------------- metriche

def _rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "lo": round(lo, 3), "hi": round(hi, 3)}


def _overlaps(spans, a: int, b: int) -> bool:
    return any(s < b and e > a for s, e in spans)


def final(args) -> None:
    pl = Pilot(args.dir)
    claims = pl.claims()
    resolved, per_claim = {}, []
    for pid, cs in claims.items():
        js = {r: pl.verdicts(r, pid) for r in ("VA", "VB", "VC")}
        for c in cs:
            if not c["valid"]:
                continue
            vs = [js["VA"].get(c["cid"]), js["VB"].get(c["cid"])] + ([js["VC"][c["cid"]]] if c["cid"] in js["VC"] else [])
            st = resolve_claim(vs)
            if st["status"] != "resolved":  # dopo l'ultimo giro quel che non è risolto resta incerto
                st = {"status": "quarantine", "value": None, "how": st.get("how") or "giudizi mancanti o incerti"}
            resolved[(pid, c["cid"])] = st
            per_claim.append({"pid": pid, "cid": c["cid"], "kind": c["kind"], "family": pl.items[pid]["family"] or "real",
                              "statement": c["statement"], "verdicts": vs, **st})
    with open(args.dir / "claims_resolved.jsonl", "w") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in per_claim)
    m: dict = {"note": "test del metodo, non certificazione: gli intervalli su campioni così piccoli non provano la qualità dei dati"}
    n_items = len(pl.items)
    all_claims = [c for cs in claims.values() for c in cs]
    m["extraction"] = {"items_extracted": len(claims), "items": n_items,
                       "claims_per_item": {"mean": round(len(all_claims) / max(1, len(claims)), 1),
                                           "median": statistics.median([len(cs) for cs in claims.values()]) if claims else None,
                                           "max": max((len(cs) for cs in claims.values()), default=0)},
                       "invalid_quotes": sum(not c["valid"] for c in all_claims), "by_kind": dict(Counter(c["kind"] for c in all_claims))}
    # alterazioni: recall dell'estrazione e sensibilità attribuibile
    cal = {}
    for fam in CAL_FAMILIES:
        rows = []
        for pid, t in pl.truth.items():
            if t.get("family") != fam:
                continue
            cs = [c for c in claims.get(pid, []) if c["valid"] and _overlaps(c["spans"], t["start"], t["end"])]
            states = [resolved.get((pid, c["cid"])) for c in cs]
            found = any(s and s["status"] == "resolved" and s["value"] == "E" for s in states)
            quarantined = not found and any(s and s["status"] == "quarantine" for s in states)
            agent = {r: any(verdict_class(pl.verdicts(r, pid).get(c["cid"])) == "E" for c in cs) for r in ("VA", "VB")}
            rows.append({"pid": pid, "extracted": bool(cs), "found": found, "quarantined": quarantined, **agent})
        ext = [r for r in rows if r["extracted"]]
        cal[fam] = {"n": len(rows), "extraction_recall": _rate(len(ext), len(rows)),
                    "verifier_sensitivity": _rate(sum(r["found"] for r in ext), len(ext)),
                    "end_to_end": _rate(sum(r["found"] for r in rows), len(rows)),
                    "quarantined": sum(r["quarantined"] for r in rows),
                    "agent_sensitivity": {a: _rate(sum(r[a] for r in ext), len(ext)) for a in ("VA", "VB")}, "items": rows}
    m["calibration"] = cal
    # controlli negativi
    neg = [pid for pid, it in pl.items.items() if it["kind"] == "negative"]
    neg_claims = [x for x in per_claim if x["pid"] in neg]
    fa_items = sorted({x["pid"] for x in neg_claims if x["status"] == "resolved" and x["value"] == "E"})
    m["negatives"] = {"items": len(neg), "items_with_false_alarm": len(fa_items), "false_alarm_items": fa_items,
                      "claims": len(neg_claims),
                      "claim_false_alarm": _rate(sum(x["status"] == "resolved" and x["value"] == "E" for x in neg_claims), len(neg_claims)),
                      "code_contradictions": sum(d["verdict"] == "contradiction" for pid in neg for d in pl.measures.get(pid, [])),
                      "by_family": {f: sum(pl.items[p]["family"] == f for p in fa_items) for f in ("caption_negative", "cot_negative")}}
    # disaccordi e quarantene
    both = [x for x in per_claim if all(verdict_class(v) for v in x["verdicts"][:2])]
    m["agreement"] = {"claims": len(per_claim), "both_certain": len(both),
                      "va_vb_disagreement": _rate(sum(verdict_class(x["verdicts"][0]) != verdict_class(x["verdicts"][1]) for x in both), len(both)),
                      "unclear": {a: sum(verdict_class(x["verdicts"][i]) is None for x in per_claim) for i, a in enumerate(("VA", "VB"))},
                      "sent_to_vc": sum(len(x["verdicts"]) == 3 for x in per_claim),
                      "claim_quarantine": _rate(sum(x["status"] == "quarantine" for x in per_claim), len(per_claim)),
                      "items_with_quarantine": len({x["pid"] for x in per_claim if x["status"] == "quarantine"})}
    # soluzioni vere: confronto con il canale deterministico e con l'audit v1
    real = [pid for pid, it in pl.items.items() if it["kind"] == "real"]
    contra = [(pid, d) for pid in real for d in pl.measures.get(pid, []) if d["verdict"] == "contradiction"]
    flagged = [any(_overlaps([s], d["start"], d["end"]) and resolved.get((pid, c["cid"]), {}).get("value") == "E"
                   for c in claims.get(pid, []) if c["valid"] for s in c["spans"]) for pid, d in contra]
    vis_err = {pid: any(x["status"] == "resolved" and x["value"] == "E" and x["kind"] != "frame" for x in per_claim if x["pid"] == pid)
               for pid in real}
    m["real"] = {"items": len(real), "with_visual_error": sum(vis_err.values()),
                 "with_frame_error": sum(any(x["pid"] == pid and x["kind"] == "frame" and x["value"] == "E" for x in per_claim) for pid in real),
                 "code_contradictions": len(contra), "code_contradictions_flagged": _rate(sum(flagged), len(flagged)),
                 "v1_vs_pilot": dict(Counter(f"v1 {pl.items[p]['v1_verdict']} / pilota {'errore' if vis_err[p] else 'nessun errore'}" for p in real))}
    # costo
    calls = Counter(n.rsplit("-", 1)[0] for n in pl.batches)
    chars = {r: sum(len(json.dumps(b, ensure_ascii=False)) for n, b in pl.batches.items() if n.rsplit("-", 1)[0] == r) for r in calls}
    m["cost"] = {"calls": dict(calls), "total_calls": sum(calls.values()), "calls_per_solution": round(sum(calls.values()) / n_items, 2),
                 "batch_chars": chars, "invalid_batches": sorted(pl.invalid)}
    usage = args.dir / "answers" / "_usage.json"
    if usage.exists():
        m["cost"]["usage"] = json.loads(usage.read_text())
    # criteri fissati nell'export
    cr, checks = pl.man["criteria"], {}
    for fam in CAL_FAMILIES:
        if cal[fam]["n"]:
            checks[f"extraction_recall.{fam}"] = cal[fam]["extraction_recall"]["rate"] >= cr["extraction_recall_min"][fam]
            checks[f"verifier_sensitivity.{fam}"] = (cal[fam]["verifier_sensitivity"]["rate"] or 0) >= cr["verifier_sensitivity_min"][fam]
    checks["negative_items"] = m["negatives"]["items_with_false_alarm"] <= cr["negative_items_with_false_alarm_max"]
    checks["negative_claims"] = (m["negatives"]["claim_false_alarm"]["rate"] or 0) <= cr["negative_claim_false_alarm_max"]
    checks["claim_quarantine"] = (m["agreement"]["claim_quarantine"]["rate"] or 0) <= cr["claim_quarantine_max"]
    checks["va_vb_disagreement"] = (m["agreement"]["va_vb_disagreement"]["rate"] or 0) <= cr["va_vb_disagreement_max"]
    if contra:
        checks["code_contradictions_flagged"] = m["real"]["code_contradictions_flagged"]["rate"] >= cr["code_contradictions_flagged_min"]
    m["criteria"] = {"thresholds": cr, "passed": checks, "all_passed": all(checks.values())}
    write_json(args.dir / "pilot_metrics.json", m)
    pct = lambda r: "—" if r.get("rate") is None else f"{100 * r['rate']:.0f}%"  # noqa: E731
    md = ["# Pilota dell'audit F2 v2 (test del metodo, non certificazione)", "",
          f"{n_items} item: {m['real']['items']} soluzioni vere, " + ", ".join(f"{cal[f]['n']} {f}" for f in CAL_FAMILIES) +
          f", {m['negatives']['items']} controlli negativi. Affermazioni estratte: {len(all_claims)} "
          f"({m['extraction']['claims_per_item']['mean']} per item), citazioni non ritrovate {m['extraction']['invalid_quotes']}.", "",
          "| Famiglia | Recall dell'estrazione | Sensibilità dei verificatori | Da capo a fondo | Quarantene |", "|---|---|---|---|---|"]
    for f in CAL_FAMILIES:
        c = cal[f]
        md.append(f"| {f} | {c['extraction_recall']['k']}/{c['n']} | {c['verifier_sensitivity']['k']}/{c['verifier_sensitivity']['n']} "
                  f"({pct(c['verifier_sensitivity'])}) | {c['end_to_end']['k']}/{c['n']} | {c['quarantined']} |")
    ng, ag = m["negatives"], m["agreement"]
    md += ["", f"- Controlli negativi: {ng['items_with_false_alarm']}/{ng['items']} item con un falso allarme; "
               f"{ng['claim_false_alarm']['k']}/{ng['claim_false_alarm']['n']} affermazioni ({pct(ng['claim_false_alarm'])}); "
               f"contraddizioni trovate dal codice {ng['code_contradictions']}.",
           f"- Disaccordo VA/VB {pct(ag['va_vb_disagreement'])} delle affermazioni giudicate da entrambi; mandate a VC {ag['sent_to_vc']}; "
           f"quarantene {ag['claim_quarantine']['k']} ({pct(ag['claim_quarantine'])}) in {ag['items_with_quarantine']} item.",
           f"- Soluzioni vere con un errore visivo: {m['real']['with_visual_error']}/{m['real']['items']}; contraddizioni di misura trovate "
           f"dal codice {m['real']['code_contradictions']}, segnalate dai verificatori {m['real']['code_contradictions_flagged']['k']}; "
           f"confronto con l'audit v1: {m['real']['v1_vs_pilot']}.",
           f"- Costo: {m['cost']['total_calls']} chiamate ({m['cost']['calls']}), {m['cost']['calls_per_solution']} per soluzione.",
           "", f"Criteri fissati nell'export: {'tutti soddisfatti' if m['criteria']['all_passed'] else 'non tutti soddisfatti'} "
               f"({', '.join(k for k, v in checks.items() if not v) or 'nessuno mancato'}).",
           "", "Gli intervalli su campioni così piccoli non provano nulla sulla qualità dei dati: il pilota decide solo se il metodo regge."]
    (args.dir / "pilot_report.md").write_text("\n".join(md) + "\n")
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
    e.add_argument("--real-per-depth", type=int, default=6)
    e.add_argument("--measure", type=int, default=12)
    e.add_argument("--construction", type=int, default=6)
    e.add_argument("--orientation", type=int, default=6)
    e.add_argument("--caption-negatives", type=int, default=6)
    e.add_argument("--cot-negatives", type=int, default=6)
    e.add_argument("--e-batch", type=int, default=2, help="soluzioni per lotto di estrazione (quelle MAVIS sono lunghe)")
    e.add_argument("--v-batch", type=int, default=2)
    e.add_argument("--recheck-frac", type=float, default=0.15)
    for name in ("round", "final"):
        sub.add_parser(name).add_argument("--dir", required=True, type=Path)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_pilot.py va eseguito in un job SLURM")
    {"export": export, "round": round_, "final": final}[args.cmd](args)


if __name__ == "__main__":
    main()
