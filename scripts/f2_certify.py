#!/usr/bin/env python3
"""F2: certificato di un lotto di soluzioni (schema f2-cert/1) da un audit delle soluzioni e da una politica dichiarata.

Da eseguire in un job CPU (slurm/f2_certify.sbatch). Legge, senza modificarli:
- l'audit (cartella di stem_audit_export.py e stem_audit_round.py): giudizi degli agenti e verità della calibrazione;
- la selezione v2 su cui l'audit ha campionato (--selection, selected.v2.jsonl): è la popolazione;
- la politica (--policy, config/audit/f2_policy.json): soglie, sistemi di riferimento, segnalazioni;
- per gli audit v1 (stem_audit_export.py --f2-mavis) la mappa fid -> sid di f2_backfill.py (--audit-map).
Scrive in --out, che deve essere nuova o vuota: certificate.json, certify_items.jsonl (esito e attribuzione per item) e
certify_report.md.

Regole (revisione tecnica, sezione 6):
- un esito vale per (sid, text_sha256): un item il cui testo non è più quello della selezione è "stale" e non conta;
- popolazione = record della selezione che superano i controlli che scartano e le decisioni della politica sulle
  segnalazioni; i tassi si stimano sugli item del campione che ne fanno parte;
- tassi per soluzione con Clopper-Pearson; per affermazione (solo audit v2: figure_claims degli agenti come
  denominatore) con stem_audit.clustered_rate;
- calibrazione attribuibile: un errore introdotto è trovato solo se la procedura esclude la soluzione per un fatto
  visivo e la citazione di un agente copre il punto alterato (stem_audit.quote_covers); un tipo richiesto dalla
  politica e non calibrato vale sensibilità zero;
- limite corretto = limite superiore del tasso / limite inferiore della sensibilità più bassa fra i tipi richiesti
  (fatti visivi); ragionamento e informazioni esterne non sono calibrati e usano il limite non corretto;
- decision "certified" solo se la politica non ha scelte aperte, l'audit è completo, nessun item è stale e ogni limite
  sta sotto la sua soglia. Incerto e quarantena restano esclusi dal training.
Un audit v1 dà al più un certificato retroattivo (retroactive: true): non separa i sistemi di riferimento (frame_ok)
e non conta le affermazioni.

    srun $RUN_PYTHON -u -B scripts/f2_certify.py --audit outputs/stem_audit/NOME --selection SEL/selected.v2.jsonl \
        --policy config/audit/f2_policy.json --out outputs/f2_cert/NOME [--audit-map BACKFILL/audit_map-NOME.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stem_audit_round import F2_BLIND, F2_GOOD, Audit, as_bool, f2_outcomes, pct, rate  # noqa: E402

from limite_vl.f2_records import (CERT_SCHEMA, eligible, policy_open, record_policy, record_version,  # noqa: E402
                                  validate_certificate)
from limite_vl.provenance import HashingTextWriter, code_identity, file_entry, sha256_file, write_new_json  # noqa: E402
from limite_vl.stem_audit import clustered_rate, corrected_upper, note_covers, quote_covers  # noqa: E402

VISUAL = ("visual_ok", "frame_ok")


def read_selection(path: Path) -> tuple[dict[str, dict], dict]:
    sel, h = {}, hashlib.sha256()
    with open(path, "rb") as fh:
        for n, line in enumerate(fh, 1):
            h.update(line)
            rec = json.loads(line)
            if record_version(rec) != 2:
                sys.exit(f"{path}:{n}: serve una selezione v2")
            if rec["sid"] in sel:
                sys.exit(f"{path}:{n}: sid ripetuto {rec['sid']}")
            sel[rec["sid"]] = rec
    return sel, {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def injected_span(solution: str, truth: dict) -> tuple[int, int] | None:
    """Punto alterato nel testo dell'item: dalla verità v2 (start, end) o, per gli audit v1, ricostruito dal contesto
    salvato (80 caratteri per lato attorno al valore cambiato); None se non è univoco."""
    if "start" in truth:
        return truth["start"], truth["end"]
    changed, context = truth.get("changed"), truth.get("context")
    if not (changed and context):
        return None
    hits = [p for p in range(len(solution)) if solution.startswith(changed, p) and solution[max(0, p - 80):p + 80] == context]
    return (hits[0], hits[0] + len(changed)) if len(hits) == 1 else None


def verdict(row: dict, fields: list[str]) -> str:
    states = [row[f] for f in fields if f in row]
    if any(s["status"] != "resolved" for s in states):
        return "quarantine"
    return "include" if all(row[f]["value"] is F2_GOOD[f] for f in fields if f in row) else "exclude"


def agent_quotes(audit: Audit, fid: str, field: str) -> dict[str, list[str] | None]:
    """Per ogni agente alla cieca che ha giudicato l'item: le citazioni dei suoi problemi su `field` se l'ha giudicato
    falso, None se l'ha giudicato vero o non l'ha giudicato."""
    out = {}
    for r in F2_BLIND:
        a = audit.answers[r].get(fid)
        if a is None:
            continue
        if as_bool(a.get(field)) is not False:
            out[r] = None
            continue
        issues = [i for i in a.get("issues") or [] if isinstance(i, dict)]
        if audit.version == 2:
            issues = [i for i in issues if (str(i.get("kind", "")).lower() == "frame") == (field == "frame_ok")]
        out[r] = [str(i.get("quote", "")) for i in issues]
    return out


def claim_cluster(audit: Audit, fid: str, row: dict) -> tuple[int, int] | None:
    """(affermazioni false, affermazioni sulla figura) di una soluzione per il tasso per affermazione (audit v2): errori
    = il massimo fra gli agenti che l'hanno giudicata falsa, affermazioni = la mediana dei figure_claims degli agenti."""
    s = row.get("visual_ok")
    if s is None or s["status"] != "resolved":
        return None
    claims = [a["figure_claims"] for r in F2_BLIND if (a := audit.answers[r].get(fid)) and isinstance(a.get("figure_claims"), int)]
    if not claims:
        return None
    flagged = [q for q in agent_quotes(audit, fid, "visual_ok").values() if q is not None]
    errors = 0 if s["value"] is True else max(1, max((len(q) for q in flagged), default=1))
    return errors, max(errors, round(statistics.median(claims)))


def certify(audit_dir: Path, selection: Path, policy_path: Path, audit_map: Path | None) -> tuple[dict, list[dict], list[str]]:
    """Certificato (in memoria), esiti per item e righe del report: lo stesso calcolo serve a scriverlo e a --verify."""
    man = json.loads((audit_dir / "manifest.json").read_text())
    audit = Audit(audit_dir, 0.10, int(man["args"]["seed"]) + 10)
    if not audit.f2:
        sys.exit(f"{audit_dir}: nessuna soluzione F2 (_gold/items_f2.json)")
    policy = json.loads(policy_path.read_text())
    open_choices, rules = policy_open(policy), record_policy(policy)
    sel, sel_entry = read_selection(selection)
    reasons = []
    if open_choices:
        reasons.append("scelte della politica ancora aperte: " + ", ".join(open_choices))

    # legame item -> (sid, testo auditato)
    if audit.version == 2:
        expected = man["f2_v2"]["selection"]["sha256"]
        if expected != sel_entry["sha256"]:
            sys.exit(f"--selection non è quella dell'audit (sha256 {sel_entry['sha256'][:12]} invece di {expected[:12]})")
        binding = {fid: {"sid": it.get("sid"), "text_sha256": it["text_sha256"], "source_sid": it.get("source_sid")}
                   for fid, it in audit.f2.items()}
    else:
        if not audit_map:
            sys.exit("audit v1: serve --audit-map (f2_backfill.py --audit)")
        binding = {}
        for line in open(audit_map):
            r = json.loads(line)
            if r["audit"] != audit_dir.name:
                sys.exit(f"{audit_map}: mappa dell'audit {r['audit']}, non di {audit_dir.name}")
            binding[r["fid"]] = {"sid": r.get("sid"), "text_sha256": r["audited_text_sha256"], "source_sid": r.get("source_sid")}
        missing = sorted(set(audit.f2) - set(binding))
        if missing:
            sys.exit(f"{audit_map}: mancano gli item {missing[:5]}")
        reasons.append("audit v1: certificato solo retroattivo (nessun frame_ok, nessun conteggio delle affermazioni)")
    pending = [n for n in audit.batches if not (audit_dir / "answers" / f"{n}.json").exists()]
    open_tasks = {r: len(v) for r, v in audit.next_tasks().items() if v}
    if pending or audit.invalid or open_tasks:
        reasons.append(f"audit incompleto: lotti senza risposta {len(pending)}, non validi {len(audit.invalid)}, compiti da "
                       f"assegnare {open_tasks}")

    # popolazione e campione
    flags_decided = all(v is not None for v in rules["flags"].values())
    in_pop = (lambda r: eligible(r, rules)[0]) if flags_decided else (lambda r: not r["failed"])
    pop = [r for r in sel.values() if in_pop(r)]
    frame_rule = policy.get("frame_counts_as_error")
    has_frame = audit.version == 2
    fields = ["visual_ok", "reasoning_valid", "relies_on_unseen"] + (["frame_ok"] if has_frame and frame_rule is not False else [])
    truth = json.loads((audit_dir / "_gold" / "calibration_f2.json").read_text()) \
        if (audit_dir / "_gold" / "calibration_f2.json").exists() else {}
    details, real, stale = [], [], []
    cal = {k: Counter() for k in ("measure", "layout", "negative")}
    agents = {k: Counter() for k in ("measure", "layout")}
    for row in f2_outcomes(audit):
        fid, b = row["fid"], binding[row["fid"]]
        item = audit.f2[fid]
        v = verdict(row, fields)
        d = {"fid": fid, "sid": b["sid"], "text_sha256": b["text_sha256"], "verdict": v}
        if fid not in truth and not item.get("calibration"):
            rec = sel.get(b["sid"]) if b["sid"] else None
            if rec is None or rec["text_sha256"] != b["text_sha256"]:
                d["status"] = "stale"
                stale.append([fid, b["sid"]])
            elif not in_pop(rec):
                d["status"] = "outside_population"
            else:
                d["status"] = "ok"
                real.append((row, d))
        else:
            t = truth.get(fid, {})
            kind = {"f2_injected": "measure"}.get(t.get("kind"), t.get("kind"))
            d |= {"status": "calibration", "kind": kind, "source_sid": b.get("source_sid") or t.get("source_sid")}
            if kind == "negative":
                vis = [row[f] for f in VISUAL if f in fields and f in row]
                bad = any(s["status"] == "resolved" and s["value"] is False for s in vis)
                d["outcome"] = "falso_allarme" if bad else "giusta" if all(s["status"] == "resolved" for s in vis) else "quarantena"
                cal[kind][d["outcome"]] += 1
            elif kind in ("measure", "layout"):
                span = injected_span(item["solution"], t)
                vis_fields = [f for f in VISUAL if f in row and (f == "visual_ok" or f in fields)]
                excluded = any(row[f]["status"] == "resolved" and row[f]["value"] is False for f in vis_fields)
                hits: dict[str, bool] = {}  # agente -> ha indicato il punto alterato
                for f in vis_fields:
                    for r, qs in agent_quotes(audit, fid, f).items():
                        hits[r] = hits.get(r, False) or bool(span and qs and any(quote_covers(item["solution"], q, *span) for q in qs))
                attributable = any(hits.values())
                # qualunque campo: anche una nota (es. sul ragionamento) che riproduce il punto alterato
                any_field = attributable or bool(span and any(
                    note_covers(item["solution"], str(a.get("reasoning_note", "")), *span)
                    or any(quote_covers(item["solution"], str(i.get("quote", "")), *span)
                           for i in a.get("issues") or [] if isinstance(i, dict))
                    for r in F2_BLIND if (a := audit.answers[r].get(fid))))
                if v == "quarantine":
                    d["outcome"] = "quarantena"
                else:
                    d["outcome"] = "trovata" if excluded and attributable else "persa"
                d |= {"span": span, "excluded_visual": excluded, "excluded_any": v == "exclude", "attributable": attributable,
                      "attributable_any_field": any_field}
                cal[kind][d["outcome"]] += 1
                cal[kind]["esclusa_per_qualunque_motivo"] += v == "exclude"
                cal[kind]["esclusa_e_attribuibile_qualunque_campo"] += v == "exclude" and any_field
                for h in hits.values():
                    agents[kind]["trovata" if h else "persa"] += 1
                if span is None:
                    cal[kind]["punto_non_ricostruito"] += 1
        details.append(d)
    if stale:
        reasons.append(f"{len(stale)} item auditati con un testo diverso da quello della selezione (stale)")

    # tassi sugli item del campione nella popolazione
    def field_rate(name: str, bad) -> dict:
        res = [row for row, _ in real if all(row[f]["status"] == "resolved" for f in name.split("+") if f in row)]
        return rate(sum(bad(row) for row in res), len(res))

    visual_fields = [f for f in VISUAL if f in fields]
    rates = {
        "solution_visual": field_rate("+".join(visual_fields), lambda r: any(r[f]["value"] is False for f in visual_fields if f in r)),
        "solution_frame": field_rate("frame_ok", lambda r: r["frame_ok"]["value"] is False) if has_frame else None,
        "reasoning_invalid": field_rate("reasoning_valid", lambda r: r["reasoning_valid"]["value"] is False),
        "relies_on_unseen": field_rate("relies_on_unseen", lambda r: r["relies_on_unseen"]["value"] is True),
        "claim_visual": None, "verdicts": dict(Counter(d["verdict"] for _, d in real)), "sample_in_population": len(real)}
    if has_frame:
        clusters = [c for row, d in real if (c := claim_cluster(audit, d["fid"], row))]
        cr = clustered_rate(clusters)
        rates["claim_visual"] = {"errors": cr["errors"], "claims": cr["total"], "rate": cr["rate"], "lo": cr["ci"][0],
                                 "hi": cr["ci"][1], "deff": cr["deff"], "deff_rule": cr["deff_rule"]} if cr["total"] else None
    sensitivity = {k: rate(c["trovata"], c["trovata"] + c["persa"]) | {
        "quarantena": c["quarantena"], "esclusa_per_qualunque_motivo": c["esclusa_per_qualunque_motivo"],
        "esclusa_e_attribuibile_qualunque_campo": c["esclusa_e_attribuibile_qualunque_campo"],
        "punto_non_ricostruito": c["punto_non_ricostruito"]} for k, c in cal.items() if k != "negative"}
    sensitivity_agent = {k: rate(c["trovata"], c["trovata"] + c["persa"]) for k, c in agents.items()}
    neg = cal["negative"]
    false_alarms = rate(neg["falso_allarme"], neg["falso_allarme"] + neg["giusta"]) | {"quarantena": neg["quarantena"]}
    required = (policy.get("calibration") or {}).get("required_kinds") or []
    uncal = [k for k in required if sensitivity.get(k, {}).get("n", 0) == 0]
    if uncal:
        reasons.append("tipi di errore richiesti senza calibrazione: " + ", ".join(uncal))
    s_lo = min((0.0 if k in uncal else sensitivity[k]["lo"]) for k in required) if required else 0.0
    corrected = {"solution_visual": corrected_upper(rates["solution_visual"]["hi"], s_lo) if rates["solution_visual"]["n"] else 1.0,
                 "claim_visual": corrected_upper(rates["claim_visual"]["hi"], s_lo) if rates["claim_visual"] else None,
                 "reasoning_invalid": rates["reasoning_invalid"]["hi"] if rates["reasoning_invalid"]["n"] else 1.0,
                 "relies_on_unseen": rates["relies_on_unseen"]["hi"] if rates["relies_on_unseen"]["n"] else 1.0,
                 "sensitivity_lower": s_lo}
    thresholds = policy.get("thresholds") or {}
    for k, thr in thresholds.items():
        if thr is None:
            continue
        if corrected.get(k) is None:
            reasons.append(f"{k}: tasso non misurato da questo audit")
        elif corrected[k] > thr:
            reasons.append(f"{k}: limite {pct(corrected[k])} oltre la soglia {pct(thr)}")
    decision = "certified" if not reasons else "rejected"

    entry = lambda d: [d["sid"], d["text_sha256"]]  # noqa: E731
    cert = {"schema": CERT_SCHEMA, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job_id": os.environ.get("SLURM_JOB_ID"),
            "retroactive": audit.version == 1, "decision": decision, "reasons": reasons,
            "population": {"selection_sha256": sel_entry["sha256"], "selection": sel_entry, "solutions": len(pop),
                           "problems": len({r["key"] for r in pop}), "by_source": dict(Counter(r["source_dataset"] for r in pop)),
                           "flags_decided": flags_decided},
            "audit": {"name": audit_dir.name, "dir": str(audit_dir), "version": audit.version,
                      "manifest_sha256": sha256_file(audit_dir / "manifest.json"),
                      "audit_map": file_entry(audit_map) if audit_map else None,
                      "design": (man.get("f2_v2") or {}).get("design") or {"unit": "una soluzione per problema (v1)",
                                                                            "seed": man["args"].get("seed")}},
            "policy": {"file": file_entry(policy_path), "content": policy, "open": open_choices},
            "record_policy": rules, "fields": fields, "rates": rates, "sensitivity": sensitivity,
            "sensitivity_agent": sensitivity_agent, "false_alarms": false_alarms, "corrected_upper": corrected,
            "thresholds": thresholds,
            "audited": [entry(d) + [d["verdict"]] for _, d in real],
            "excluded": [entry(d) for _, d in real if d["verdict"] == "exclude"],
            "quarantined": [entry(d) for _, d in real if d["verdict"] == "quarantine"],
            "stale": stale, "code": code_identity(),
            "notes": ["incerto e quarantena restano esclusi dal training",
                      "un audit a campione certifica il tasso della popolazione, non le singole soluzioni non estratte",
                      "ragionamento e informazioni esterne: limite non corretto (errori non calibrati)"]}
    problems = validate_certificate(cert)
    if problems:
        raise RuntimeError(f"certificato non valido: {problems}")

    def line(name: str, r: dict | None) -> str:
        return "—" if not r or not r.get("n", r.get("claims")) else \
            f"{r.get('k', r.get('errors'))}/{r.get('n', r.get('claims'))} ({pct(r['rate'])}, IC {pct(r['lo'])}-{pct(r['hi'])})"

    md = [f"# Certificato F2 di {audit_dir.name}: {decision}", "",
          f"Selezione {sel_entry['path']} (sha256 {sel_entry['sha256'][:12]}), popolazione {len(pop)} soluzioni; audit v{audit.version}"
          f"{' (retroattivo)' if audit.version == 1 else ''}; politica {policy_path} (scelte aperte: {', '.join(open_choices) or 'nessuna'}).",
          "", "## Tassi sul campione", "",
          f"- Item del campione nella popolazione: {len(real)}; stale {len(stale)}; verdetti {rates['verdicts']}.",
          f"- Soluzioni con fatti visivi falsi o non visibili ({'+'.join(visual_fields)}): {line('v', rates['solution_visual'])}; "
          f"limite corretto {pct(corrected['solution_visual'])}.",
          f"- Sistemi di riferimento in contrasto con la figura: {line('f', rates['solution_frame'])}.",
          f"- Per affermazione: {line('c', rates['claim_visual'])}; limite corretto {pct(corrected['claim_visual'])}.",
          f"- Ragionamento non valido: {line('r', rates['reasoning_invalid'])}; informazioni esterne: {line('u', rates['relies_on_unseen'])}.",
          "", "## Calibrazione attribuibile", ""]
    for k, s in sensitivity.items():
        md.append(f"- {k}: trovate {s['k']}/{s['n']} ({pct(s['rate'])}, IC {pct(s['lo'])}-{pct(s['hi'])}: esclusione per un fatto "
                  f"visivo con una citazione sul punto alterato, la misura usata per correggere); escluse e attribuibili per "
                  f"qualunque campo {s['esclusa_e_attribuibile_qualunque_campo']}, escluse per qualunque motivo "
                  f"{s['esclusa_per_qualunque_motivo']} (criterio degli audit v1); quarantena {s['quarantena']}; punto non "
                  f"ricostruito {s['punto_non_ricostruito']}; agente singolo {sensitivity_agent[k]['k']}/{sensitivity_agent[k]['n']}.")
    md += [f"- Controlli negativi: falsi allarmi {line('n', false_alarms)}.",
           f"- Limite inferiore della sensibilità usato (tipi {', '.join(required) or 'nessuno'}): {pct(s_lo)}.", "",
           "## Decisione", "", f"**{decision}**." + ("" if not reasons else " Motivi:"), *[f"- {r}" for r in reasons]]
    return cert, details, md


# campi che decidono il certificato: --verify li ricalcola dagli artefatti e li confronta
DECIDING = ("decision", "reasons", "retroactive", "population", "fields", "rates", "sensitivity", "false_alarms",
            "corrected_upper", "thresholds", "record_policy", "audited", "excluded", "quarantined", "stale")


def _path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def verify(cert: dict) -> list[str]:
    """Verifica che richiede gli artefatti originali (validate_certificate copre ciò che si controlla dal solo
    certificato): sha256 di selezione, manifest dell'audit, politica e mappa uguali a quelli registrati, politica
    incorporata uguale al file, e certificato ricalcolato da quegli artefatti uguale in ogni campo di DECIDING."""
    problems = [f"certificato: {p}" for p in validate_certificate(cert)]
    a, pop, pol = cert.get("audit") or {}, cert.get("population") or {}, cert.get("policy") or {}
    audit_dir, selection = _path(a.get("dir", "")), _path((pop.get("selection") or {}).get("path", ""))
    policy_path = _path((pol.get("file") or {}).get("path", ""))
    amap = _path(a["audit_map"]["path"]) if a.get("audit_map") else None
    checks = [(selection, pop.get("selection_sha256"), "selezione"), (audit_dir / "manifest.json", a.get("manifest_sha256"), "manifest dell'audit"),
              (policy_path, (pol.get("file") or {}).get("sha256"), "politica")] + ([(amap, a["audit_map"]["sha256"], "mappa dell'audit")] if amap else [])
    for path, sha, what in checks:
        if not path.is_file():
            problems.append(f"{what} mancante: {path}")
        elif sha256_file(path) != sha:
            problems.append(f"{what} cambiata dopo il certificato: {path}")
    if problems:
        return problems
    if json.loads(policy_path.read_text()) != pol.get("content"):
        problems.append("politica incorporata diversa dal file")
    new, _, _ = certify(audit_dir, selection, policy_path, amap)

    def deciding(c: dict) -> dict:  # il percorso della selezione può essere scritto in modi diversi: conta lo sha256
        c = json.loads(json.dumps({k: c.get(k) for k in DECIDING}))
        ((c.get("population") or {}).get("selection") or {}).pop("path", None)
        return c

    old, now = deciding(cert), deciding(new)
    problems += [f"{k}: ricalcolato diverso da quello del certificato" for k in DECIDING if old[k] != now[k]]
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--selection", type=Path, help="selected.v2.jsonl su cui l'audit ha campionato")
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--out", type=Path, help="cartella nuova o vuota")
    ap.add_argument("--audit-map", type=Path, help="audit v1: audit_map-<audit>.jsonl di f2_backfill.py")
    ap.add_argument("--verify", type=Path, help="certificate.json da verificare rileggendo gli artefatti (non scrive nulla)")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_certify.py va eseguito in un job SLURM")
    t0 = time.time()
    if args.verify:
        problems = verify(json.loads(args.verify.read_text()))
        print(json.dumps({"certificate": str(args.verify), "ok": not problems, "problems": problems}, indent=1, ensure_ascii=False))
        sys.exit(1 if problems else 0)
    if not (args.audit and args.selection and args.policy and args.out):
        sys.exit("servono --audit, --selection, --policy e --out (oppure --verify)")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: un certificato non si riscrive")
    cert, details, md = certify(args.audit, args.selection, args.policy, args.audit_map)
    args.out.mkdir(parents=True, exist_ok=True)
    w = HashingTextWriter(args.out / "certify_items.jsonl")
    for d in details:
        w.write_json_line(d)
    cert["items_file"] = w.close()
    write_new_json(args.out / "certificate.json", cert)
    (args.out / "certify_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"fine in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
