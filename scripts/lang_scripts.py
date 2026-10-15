#!/usr/bin/env python3
"""Testi non in inglese di MMFineReason separati per scrittura: cinese (solo ideogrammi), giapponese (kana),
coreano (hangul), e come ragiona Violetto sulle domande in cinese.

Serve a decidere se il cinese va escluso come giapponese e coreano: scripts/lang_survey.py (job 29592) contava
insieme ideogrammi, kana e hangul. Da eseguire in un job CPU (slurm/lang_scripts.sbatch). Legge le colonne di
testo dei parquet e, con --gen, le generazioni di F2. Scrive in --out:
- question_ja_ko.txt, caption_ja_ko.txt: chiavi con kana o hangul nella domanda o nella descrizione;
- scripts.json: conteggi per scrittura di domande, descrizioni e soluzioni corrette; per le domande in cinese,
  pass rate e lingua del ragionamento (quota di ideogrammi fra ideogrammi e lettere latine) di tutte le
  generazioni e di quelle corrette.
Stampa alcuni esempi: inizio della domanda in cinese e del ragionamento di Violetto.

    srun $RUN_PYTHON -u -B scripts/lang_scripts.py --parquet FILE... --out DIR [--gen outputs/f2_gen/RUN]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import grade, split_completion  # noqa: E402
from limite_vl.mmfine import mmfine_key, split_of  # noqa: E402

HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
KANA = re.compile(r"[぀-ヿ]")
HANGUL = re.compile(r"[ᄀ-ᇿ㄰-㆏가-힯]")
LATIN = re.compile(r"[A-Za-z]")


def script_of(text: str | None) -> str | None:
    """"ja" con kana, "ko" con hangul, "zh" con soli ideogrammi, None senza nessuno dei tre."""
    text = text or ""
    if KANA.search(text):
        return "ja"
    if HANGUL.search(text):
        return "ko"
    if HAN.search(text):
        return "zh"
    return None


def reasoning_language(text: str) -> str:
    """Lingua del ragionamento dalla quota di ideogrammi fra ideogrammi e lettere latine."""
    reasoning, _ = split_completion(text)
    body = reasoning if reasoning is not None else text
    han, latin = len(HAN.findall(body)), len(LATIN.findall(body))
    frac = han / max(1, han + latin)
    return "zh" if frac > 0.5 else "mixed" if frac > 0.05 else "en"


def grade_line(line: str) -> dict:
    rec = json.loads(line)
    out = []
    for c in rec["completions"]:
        g = grade(c["text"], rec["answer"], c["finish_reason"])
        out.append({"correct": g["correct"], "usable": g["usable"], "reject": g["reject"],
                    "script": script_of(c["text"]), "lang": reasoning_language(c["text"]),
                    "head": (split_completion(c["text"])[0] or c["text"])[:500]})
    return {"key": rec["key"], "completions": out}


def main() -> None:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", nargs="+", type=Path, required=True, help="parquet dello snapshot (sft e rl)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gen", type=Path, help="cartella di f2_generate.py")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("lang_scripts.py va eseguito in un job SLURM")
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    q_script, questions = {}, {}
    q_counts, c_counts = Counter(), Counter()
    per_source = defaultdict(Counter)
    q_ja_ko, c_ja_ko = [], []
    rows = 0
    for path in sorted(args.parquet):
        split = split_of(path)
        table = pq.read_table(path, columns=["id", "source", "question", "caption"])
        for r in table.to_pylist():
            rows += 1
            key = mmfine_key(split, r["source"], r["id"])
            qs, cs = script_of(r["question"]), script_of(r["caption"])
            q_counts[qs or "none"] += 1
            c_counts[cs or "none"] += 1
            if qs:
                q_script[key] = qs
                per_source[r["source"]][f"question_{qs}"] += 1
                if qs == "zh":
                    questions[key] = r["question"]
                else:
                    q_ja_ko.append(key)
            if cs:
                per_source[r["source"]][f"caption_{cs}"] += 1
                if cs != "zh":
                    c_ja_ko.append(key)
        del table
    (args.out / "question_ja_ko.txt").write_text("".join(f"{k}\n" for k in sorted(q_ja_ko)))
    (args.out / "caption_ja_ko.txt").write_text("".join(f"{k}\n" for k in sorted(c_ja_ko)))
    stats = {"rows": rows, "questions": dict(q_counts), "captions": dict(c_counts),
             "per_source": {k: dict(v) for k, v in sorted(per_source.items(), key=lambda x: -sum(x[1].values()))}}
    print(f"{rows} righe ({time.time() - t0:.0f} s); domande per scrittura {dict(q_counts)}; "
          f"descrizioni {dict(c_counts)}", flush=True)
    for name, c in list(stats["per_source"].items())[:10]:
        print(f"  {name:40s} {dict(c)}", flush=True)

    if args.gen:
        acc = defaultdict(Counter)  # per scrittura della domanda
        sol_script = Counter()      # scrittura delle soluzioni corrette, domande senza CJK
        examples = []
        parts = sorted(args.gen.glob("part-*.jsonl"))
        with Pool(args.workers) as pool:
            for part in parts:
                with open(part) as fh:
                    for item in pool.imap(grade_line, fh, chunksize=16):
                        qs = q_script.get(item["key"], "en")
                        a = acc[qs]
                        a["problems"] += 1
                        a["solved"] += any(c["correct"] for c in item["completions"])
                        for c in item["completions"]:
                            a["completions"] += 1
                            a["correct"] += c["correct"]
                            a[f"lang_{c['lang']}"] += 1
                            if c["correct"]:
                                a[f"correct_lang_{c['lang']}"] += 1
                                if qs == "en":
                                    sol_script[c["script"] or "none"] += 1
                        if qs == "zh" and len(examples) < args.examples:
                            best = next((c for c in item["completions"] if c["correct"]), item["completions"][0])
                            examples.append({"key": item["key"], "question": questions.get(item["key"], "")[:200],
                                             "correct": best["correct"], "lang": best["lang"], "head": best["head"]})
        stats["f2_generation"] = {"parts": len(parts), **{k: dict(v) for k, v in acc.items()},
                                  "correct_solution_script_english_questions": dict(sol_script)}
        print(f"\ngenerazione F2 ({len(parts)} parti):", flush=True)
        for qs, a in acc.items():
            rate = a["correct"] / max(1, a["completions"])
            langs = {k: a[k] for k in sorted(a) if "lang" in k}
            print(f"  domande {qs}: {a['problems']} problemi, risolti {a['solved'] / max(1, a['problems']):.3f}, "
                  f"pass rate {rate:.3f}; lingua del ragionamento {langs}", flush=True)
        print(f"  soluzioni corrette su domande senza CJK, per scrittura: {dict(sol_script)}", flush=True)
        print("\nesempi di domande in cinese (inizio della domanda | inizio del ragionamento):", flush=True)
        for ex in examples:
            q = ex["question"].replace("\n", " ")
            h = ex["head"].replace("\n", " ")[:400]
            print(f"\n[{ex['key']}] corretta={ex['correct']} lingua={ex['lang']}\n  D: {q}\n  R: {h}", flush=True)
        stats["examples"] = examples
    stats["seconds"] = round(time.time() - t0, 1)
    (args.out / "scripts.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"\nfine in {stats['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
