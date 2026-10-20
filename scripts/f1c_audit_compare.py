#!/usr/bin/env python3
"""Confronto delle risposte degli agenti dell'audit di F1c con le risposte di riferimento. Job CPU (usa math_verify).

- Domande a scelta (stem_facts-1..4, synth, mavis): accordo con il riferimento, risposte AMBIGUOUS, elenco delle
  divergenze con immagine, domanda, opzioni, riferimento e risposta dell'agente (da decidere guardando l'immagine).
- Problemi (mmf_answers, replay): risposta dell'agente contro quella del dataset con limite_vl.f2.answers_match
  (lettere per le scelte multiple, poi math_verify, come la selezione delle soluzioni).
- Controlli descrittivi (stem_captions, mmf_solutions, unimer_lines): conteggi dai file degli agenti.
Scrive OUT/summary.json e OUT/divergenze.md.

    srun $RUN_PYTHON -u -B scripts/f1c_audit_compare.py --audit outputs/f1c/audit --out outputs/f1c/audit/compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import answers_match  # noqa: E402

CHOICE = ["stem_facts-1", "stem_facts-2", "stem_facts-3", "stem_facts-4", "synth", "mavis"]


def load_answers(task_dir: Path) -> dict:
    out = {}
    for path in sorted(task_dir.glob("answers*.json")):
        for row in json.loads(path.read_text()):
            out[row["id"]] = row
    return out


def load_items(task_dir: Path) -> dict:
    return {it["id"]: it for it in json.loads((task_dir / "items.json").read_text())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    A, summary, md = args.audit, {}, ["# Divergenze fra agenti e riferimenti\n"]
    for task in CHOICE:
        items, gold, ans = load_items(A / task), json.loads((A / "_gold" / f"{task}.json").read_text()), load_answers(A / task)
        agree = amb = missing = 0
        rows = []
        for iid, it in items.items():
            a = ans.get(iid)
            if a is None:
                missing += 1
                continue
            if a["answer"] == "AMBIGUOUS":
                amb += 1
                rows.append((iid, it, gold[iid], a))
            elif a["answer"] == gold[iid]:
                agree += 1
            else:
                rows.append((iid, it, gold[iid], a))
        n = len(items) - missing
        summary[task] = {"items": len(items), "answered": n, "agree": agree, "ambiguous": amb,
                         "disagree": n - agree - amb, "agree_rate": agree / max(1, n)}
        md.append(f"\n## {task}: accordo {agree}/{n}, AMBIGUOUS {amb}, divergenze {n - agree - amb}, senza risposta {missing}\n")
        for iid, it, g, a in rows:
            md.append(f"- `{iid}` {it['q']} | opzioni {it['options']} | riferimento **{g}** | agente **{a['answer']}** "
                      f"({a.get('confidence')}) {a.get('note', '')} | {it['image']}")
    for task in ("mmf_answers", "replay"):
        items, gold, ans = load_items(A / task), json.loads((A / "_gold" / f"{task}.json").read_text()), load_answers(A / task)
        match = unsolved = 0
        rows = []
        for iid in items:
            a = ans.get(iid)
            if a is None:
                continue
            if a["answer"] in ("UNSOLVABLE", "UNSURE"):
                unsolved += 1
                rows.append((iid, gold[iid], a))
                continue
            try:
                ok = answers_match(str(a["answer"]), str(gold[iid]))
            except Exception:  # noqa: BLE001
                ok = False
            match += ok
            if not ok:
                rows.append((iid, gold[iid], a))
        summary[task] = {"items": len(items), "answered": len(ans), "match": match, "unsolved": unsolved,
                         "mismatch": len(ans) - match - unsolved}
        md.append(f"\n## {task}: coincidono {match}/{len(ans)}, non risolti {unsolved}\n")
        for iid, g, a in rows:
            md.append(f"- `{iid}` riferimento **{g}** | agente **{a['answer']}** ({a.get('confidence')}) {a.get('note', '')}")
    caps = load_answers(A / "stem_captions")
    if caps:
        fc = [len(r.get("false_claims", [])) for r in caps.values()]
        summary["stem_captions"] = {"items": len(caps), "claims_checked": sum(int(r.get("claims_checked") or 0) for r in caps.values()),
                                    "false_claims": sum(fc), "items_with_errors": sum(x > 0 for x in fc),
                                    "mismatch": sum(bool(r.get("image_caption_mismatch")) for r in caps.values()),
                                    "removed_false_claims": sum(int(r.get("removed_false_claims") or 0) for r in caps.values())}
        md.append(f"\n## stem_captions: {summary['stem_captions']}\n")
        for iid, r in caps.items():
            for c in r.get("false_claims", []):
                md.append(f"- `{iid}` {c.get('claim')} — {c.get('why')}")
    sols = load_answers(A / "mmf_solutions")
    if sols:
        summary["mmf_solutions"] = {"items": len(sols),
                                    "with_false_visual_claims": sum(bool(r.get("false_visual_claims")) for r in sols.values()),
                                    "reasoning_invalid": sum(r.get("reasoning_valid") is False for r in sols.values()),
                                    "reference_wrong": sum(r.get("reference_answer_correct") == "no" for r in sols.values()),
                                    "reference_unsure": sum(r.get("reference_answer_correct") == "unsure" for r in sols.values())}
        md.append(f"\n## mmf_solutions: {summary['mmf_solutions']}\n")
        for iid, r in sols.items():
            if r.get("false_visual_claims") or r.get("reasoning_valid") is False or r.get("reference_answer_correct") != "yes":
                md.append(f"- `{iid}` falsi visivi {r.get('false_visual_claims')} | ragionamento valido {r.get('reasoning_valid')}: "
                          f"{r.get('reasoning_issue')} | riferimento giusto {r.get('reference_answer_correct')} | {r.get('note', '')}")
    lines = load_answers(A / "unimer_lines")
    if lines:
        summary["unimer_lines"] = {"items": len(lines), **{k: sum(r.get("matches") == k for r in lines.values()) for k in ("yes", "partly", "no")},
                                   "crop_issues": sum(bool((r.get("crop_issue") or "").strip() and (r.get("crop_issue") or "").lower()
                                                           not in ("none", "no", "-")) for r in lines.values())}
        md.append(f"\n## unimer_lines: {summary['unimer_lines']}\n")
        for iid, r in lines.items():
            if r.get("matches") != "yes" or (r.get("crop_issue") or "").lower() not in ("", "none", "no", "-"):
                md.append(f"- `{iid}` {r.get('matches')} | {r.get('differences')} | ritaglio: {r.get('crop_issue')}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (args.out / "divergenze.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
