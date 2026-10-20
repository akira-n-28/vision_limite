#!/usr/bin/env python3
"""Prova solo domanda: Violetto senza descrizione contro il ponte testuale, sugli stessi problemi. Job CPU.

Per i problemi della generazione --qonly (f2_generate.py --no-caption) si valutano con limite_vl.f2.grade le soluzioni
della prova e quelle del ponte (--bridge, il primo lotto) e si riportano: problemi risolti (almeno una corretta) e
con almeno una soluzione utilizzabile (corretta e non scartata da clean_solution: niente immagine, CJK, struttura della
descrizione), pass rate, motivi di scarto delle corrette, tabella risolti dall'una/dall'altra/da entrambe, per sorgente,
e le frasi con cui le soluzioni utilizzabili presumono com'è la figura ("assuming the figure shows", "the figure
probably"). Scrive OUT/stats.json, OUT/usable.jsonl (soluzioni utilizzabili della prova, per l'audit) e OUT/examples.md.

    srun $RUN_PYTHON -u -B scripts/f2_qonly_compare.py --qonly outputs/f2_gen/f2-qonly5k --bridge outputs/f2_gen/f2-sft120k --out DIR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import _KEY_PREFIX, grade  # noqa: E402

ASSUME = re.compile(r"\bassum(?:e|ing) (?:that )?the (?:figure|diagram|picture|image)\b|\bthe (?:figure|diagram) "
                    r"(?:probably|likely|presumably|might|may) (?:show|be|depict)|\b(?:typical|standard|usual) "
                    r"(?:figure|diagram|configuration)\b|\bwithout (?:seeing )?the (?:figure|diagram)\b", re.IGNORECASE)


def graded(run_dir: Path, keys: set[str]) -> dict[str, dict]:
    out = {}
    for part in sorted(run_dir.glob("part-*.jsonl")):
        with open(part, "rb") as fh:
            for line in fh:
                m = _KEY_PREFIX.match(line)
                if not m or json.loads(m.group(1)) not in keys:
                    continue
                rec = json.loads(line)
                gs = [grade(c["text"], rec["answer"], c["finish_reason"]) for c in rec["completions"]]
                out[rec["key"]] = {"source": rec.get("source_dataset") or rec.get("source"), "answer": rec["answer"],
                                   "grades": gs, "texts": [c["text"] for c in rec["completions"]]}
    return out


def summary(res: dict) -> dict:
    n = len(res)
    gens = [g for r in res.values() for g in r["grades"]]
    rejects = Counter(g["reject"] for g in gens if g["correct"] and g["reject"])
    return {"problems": n, "solved": sum(any(g["correct"] for g in r["grades"]) for r in res.values()),
            "usable_problems": sum(any(g["usable"] for g in r["grades"]) for r in res.values()),
            "pass_rate": sum(g["correct"] for g in gens) / max(1, len(gens)),
            "usable_rate": sum(g["usable"] for g in gens) / max(1, len(gens)),
            "truncated": sum(not g["stopped"] for g in gens) / max(1, len(gens)), "rejects_of_correct": dict(rejects)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qonly", required=True, type=Path)
    ap.add_argument("--bridge", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    t0 = time.time()
    keys = {line.strip() for line in (args.qonly / "keys.txt").open() if line.strip()}
    q = graded(args.qonly, keys)
    b = graded(args.bridge, set(q))
    common = sorted(set(q) & set(b))
    q, b = {k: q[k] for k in common}, {k: b[k] for k in common}
    print(f"problemi in comune: {len(common)} ({time.time() - t0:.0f} s)", flush=True)
    table = Counter()
    by_src = defaultdict(lambda: Counter())
    for k in common:
        qs, bs = any(g["usable"] for g in q[k]["grades"]), any(g["usable"] for g in b[k]["grades"])
        table[("entrambi" if qs and bs else "solo_domanda" if qs else "solo_ponte" if bs else "nessuno")] += 1
        src = q[k]["source"]
        by_src[src]["problemi"] += 1
        by_src[src]["utilizzabili_domanda"] += qs
        by_src[src]["utilizzabili_ponte"] += bs
    assume, examples = 0, defaultdict(list)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "usable.jsonl", "w") as fh:
        for k in common:
            for g, text in zip(q[k]["grades"], q[k]["texts"]):
                if not g["usable"]:
                    continue
                m = ASSUME.search(g["cleaned"] or text)
                assume += bool(m)
                if m and len(examples["presume_la_figura"]) < 10:
                    s = g["cleaned"] or text
                    examples["presume_la_figura"].append(f"`{k}` …{s[max(0, m.start() - 150): m.end() + 150]}…")
                fh.write(json.dumps({"key": k, "source": q[k]["source"], "answer": q[k]["answer"],
                                     "solution": g["cleaned"], "assume_figure": bool(m)}, ensure_ascii=False) + "\n")
                break
    for k in common:
        for g, text in zip(q[k]["grades"], q[k]["texts"]):
            if g["correct"] and g["reject"] and len(examples[f"scarto:{g['reject']}"]) < 5:
                examples[f"scarto:{g['reject']}"].append(f"`{k}` {text[:300]!r}")
    usable_q = sum(any(g["usable"] for g in r["grades"]) for r in q.values())
    stats = {"common_problems": len(common), "solo_domanda": summary(q), "ponte": summary(b), "tabella_utilizzabili": dict(table),
             "presume_la_figura": {"soluzioni_utilizzabili_controllate": usable_q, "con_frase": assume},
             "per_sorgente": {s: dict(c) for s, c in sorted(by_src.items(), key=lambda kv: -kv[1]["problemi"])},
             "seconds": round(time.time() - t0, 1)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    md = ["# Prova solo domanda: esempi\n"]
    for k, items in examples.items():
        md.append(f"\n## {k}\n")
        md.extend(f"- {e}" for e in items)
    (args.out / "examples.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: stats[k] for k in ("common_problems", "solo_domanda", "ponte", "tabella_utilizzabili",
                                            "presume_la_figura")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
