#!/usr/bin/env python3
"""F2, ponte testuale: sceglie le soluzioni di Violetto da usare come target (dopo scripts/f2_generate.py).

Da eseguire in un job CPU (slurm/f2_select.sbatch). Per ogni generazione: limite_vl.f2.grade (finita con
stop, con </think>, \\boxed{} corretto rispetto alla risposta verificata; nelle corrette i riferimenti alla
descrizione diventano riferimenti alla figura, e si scartano quelle che dicono di non vedere l'immagine, quelle
con caratteri CJK e quelle che citano la struttura della descrizione, come "Explicit Component Breakdown"). Per
ogni problema tiene fino a --keep soluzioni utilizzabili (scelta deterministica per chiave, la stessa delle
versioni precedenti), e scrive in --out, che deve essere una cartella nuova o vuota:
- selected.v2.jsonl: le soluzioni scelte come record f2-sol/2 (limite_vl.f2_records): sid
  "<generazione>/<chiave>/<indice>", sha256 del testo generato e di quello scelto, controlli deterministici, più i
  campi delle selezioni storiche (chiave, sorgente, risposta, soluzione da `<think>` in poi, rewritten, token, pass
  rate di Violetto con la descrizione), quindi leggibile dagli script che leggevano selected.jsonl;
- rejected.v2.jsonl: le soluzioni scelte ma scartate dai controlli nuovi di f2_records.REJECTING (oggi think_tags:
  più di un <think> o </think>), con `rejected_by`. Non vengono sostituite con altre soluzioni dello stesso problema:
  così la selezione resta confrontabile con quella storica (sostituirle è un'alternativa da approvare);
- rl_pool.jsonl: problemi con nessuna soluzione corretta ma almeno una finita (al pool di RL di F4);
- stats.json: conteggi, pass rate per sorgente, lunghezze, motivi di scarto, scarti dei controlli nuovi e
  segnalazioni di f2_records.FLAGGED (solo rilevate, nessuno scarto: es. citazioni del prompt "Figure:");
- solved_unselected.txt: chiavi dei problemi con almeno una soluzione corretta ma nessuna utilizzabile (tutte
  scartate dai filtri), da rigenerare con f2_generate.py --include. Con --keys-only si scrivono solo questo file,
  stats.json e selection.json;
- selection.json: argomenti, sha256 delle parti e dei file di esclusione, sha256 delle uscite, identità del codice;
- compare.json (con --compare): confronto per sid e hash con un'altra selezione v2, di solito il backfill della
  selezione storica (scripts/f2_backfill.py).
Con --exclude salta i problemi delle liste di decontaminazione (data/decontam/p6/<sorgente>/exclude_final.txt):
non entrano né nelle soluzioni né nel pool di RL. Nessuna soluzione è certificata da questa selezione: la risposta
giusta è necessaria, non sufficiente (il certificato viene dall'audit, f2_records.admit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import grade, no_image_phrase  # noqa: E402
from limite_vl.f2_records import FLAGGED, REJECTING, compare_selections, gen_run_name, read_selected, solution_record  # noqa: E402
from limite_vl.provenance import HashingTextWriter, code_identity, file_entry, write_new_json  # noqa: E402


def grade_problem(line: bytes) -> dict:
    rec = json.loads(line)
    grades = []
    for c in rec["completions"]:
        g = grade(c["text"], rec["answer"], c["finish_reason"])
        if not g["correct"] and rec.get("answer_letter"):  # MAVIS: vale anche la lettera dell'opzione giusta
            by_letter = grade(c["text"], rec["answer_letter"], c["finish_reason"])
            g = by_letter if by_letter["correct"] else g
        grades.append(g)
    return {"rec": rec, "grades": grades}


def hashed_lines(parts: list[Path], digests: dict):
    """Righe delle parti (in byte), con lo sha256 di ogni parte calcolato durante la lettura."""
    for part in parts:
        h = hashlib.sha256()
        with open(part, "rb") as fh:
            for line in fh:
                h.update(line)
                yield line
        digests[part.name] = h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, type=Path, help="cartella di f2_generate.py")
    ap.add_argument("--out", required=True, type=Path, help="cartella nuova o vuota")
    ap.add_argument("--keep", type=int, default=2, help="soluzioni tenute per problema")
    ap.add_argument("--exclude", nargs="*", default=[], help="file di chiavi da saltare (decontaminazione)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--keys-only", action="store_true",
                    help="solo stats.json, solved_unselected.txt e selection.json, senza soluzioni e pool di RL")
    ap.add_argument("--compare", type=Path, help="selezione v2 di riferimento (es. selected.v2.jsonl di f2_backfill.py)")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_select.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: una selezione non si sovrascrive, usare una cartella nuova")
    if args.compare and args.keys_only:
        sys.exit("--compare richiede le soluzioni (senza --keys-only)")
    t0 = time.time()
    parts = sorted(args.gen.glob("part-*.jsonl"))
    if not parts:
        sys.exit(f"nessuna parte in {args.gen}")
    gen_run = gen_run_name(args.gen)
    exclude: set[str] = set()
    for path in args.exclude:
        exclude.update(line.strip() for line in open(path) if line.strip())
    args.out.mkdir(parents=True, exist_ok=True)
    reasons, per_source, lengths, pass_rates = Counter(), defaultdict(lambda: [0, 0]), [], []
    no_image_phrases, no_image_examples = Counter(), defaultdict(list)
    new_rejects, flags = Counter(), Counter()
    n_problems = n_selected = n_rl = n_excluded = n_with_selection = n_with_selection_v2 = n_chosen = 0
    solved_unselected = []
    part_digests: dict[str, str] = {}
    sel = rej = rl = None
    if not args.keys_only:
        sel = HashingTextWriter(args.out / "selected.v2.jsonl")
        rej = HashingTextWriter(args.out / "rejected.v2.jsonl")
        rl = HashingTextWriter(args.out / "rl_pool.jsonl")
    with Pool(args.workers) as pool:
        for item in pool.imap(grade_problem, hashed_lines(parts, part_digests), chunksize=16):
            rec, grades = item["rec"], item["grades"]
            if rec["key"] in exclude:
                n_excluded += 1
                continue
            n_problems += 1
            usable = [i for i, g in enumerate(grades) if g["usable"]]
            correct = sum(g["correct"] for g in grades)
            for i, g in enumerate(grades):
                if not g["stopped"]:
                    reasons["truncated"] += 1
                elif not g["has_think"]:
                    reasons["no_think"] += 1
                elif g["answer"] is None:
                    reasons["no_boxed"] += 1
                elif not g["correct"]:
                    reasons["wrong"] += 1
                elif g["reject"]:
                    reasons[g["reject"]] += 1
                    if g["reject"] == "no_image":  # frase che ha fatto scartare, con un po' di contesto
                        text = rec["completions"][i]["text"]
                        phrase = no_image_phrase(text)
                        no_image_phrases[phrase.lower()] += 1
                        if len(no_image_examples[phrase.lower()]) < 3:
                            at = text.find(phrase)
                            no_image_examples[phrase.lower()].append(
                                {"key": rec["key"], "context": text[max(0, at - 200):at + len(phrase) + 100]})
                else:
                    reasons["usable"] += 1
                    reasons["rewritten"] += g["cleaned"] != rec["completions"][i]["text"]
            rate = correct / max(1, len(grades))
            pass_rates.append(rate)
            src = per_source[rec.get("source_dataset") or "?"]
            src[0] += correct
            src[1] += len(grades)
            if usable:
                n_with_selection += 1
                kept_here = 0
                rng = random.Random(int(hashlib.blake2b(rec["key"].encode(), digest_size=8).hexdigest(), 16))
                for i in sorted(rng.sample(usable, min(args.keep, len(usable)))):
                    c = rec["completions"][i]
                    n_chosen += 1
                    v2 = solution_record(gen_run=gen_run, key=rec["key"], completion=i, solution=grades[i]["cleaned"],
                                         raw_text=c["text"], finish_reason=c["finish_reason"], tokens=c["tokens"],
                                         source_dataset=rec.get("source_dataset"), answer=rec["answer"],
                                         rewritten=grades[i]["cleaned"] != c["text"], violetto_pass_rate=rate,
                                         origin="select", answer_letter=rec.get("answer_letter"))
                    if v2["failed"]:
                        new_rejects.update(v2["failed"])
                        if rej is not None:
                            rej.write_json_line({**v2, "rejected_by": v2["failed"]})
                        continue
                    flags.update(v2["flags"])
                    lengths.append(c["tokens"])
                    if sel is not None:
                        sel.write_json_line(v2)
                    n_selected += 1
                    kept_here += 1
                n_with_selection_v2 += kept_here > 0
            elif correct:
                solved_unselected.append(rec["key"])
            elif any(g["stopped"] for g in grades):
                n_rl += 1
                if rl is not None:
                    rl.write_json_line({"key": rec["key"], "source_dataset": rec.get("source_dataset"),
                                        "answer": rec["answer"], "qwen4b_pass_rate": rec.get("qwen4b_pass_rate")})
    outputs = {} if sel is None else {"selected_v2": sel.close(), "rejected_v2": rej.close(), "rl_pool": rl.close()}
    stats = {"schema": "f2-select-stats/2", "excluded_contaminated": n_excluded, "exclude_files": args.exclude,
             "problems": n_problems, "selected_solutions": n_selected, "problems_with_selection": n_with_selection,
             "problems_with_selection_v2": n_with_selection_v2, "chosen_before_new_checks": n_chosen,
             "new_check_rejects": dict(new_rejects), "rejecting_checks": list(REJECTING),
             "flags_on_selected": dict(flags), "flagged_checks": list(FLAGGED),
             "problems_solved": sum(1 for r in pass_rates if r > 0), "solved_unselected": len(solved_unselected),
             "rl_pool": n_rl,
             "completion_outcomes": dict(reasons), "no_image_phrases": dict(no_image_phrases.most_common()),
             "no_image_examples": no_image_examples,
             "violetto_pass_rate_mean": statistics.mean(pass_rates) if pass_rates else None,
             "pass_rate_by_source": {k: round(v[0] / max(1, v[1]), 3) for k, v in sorted(per_source.items())},
             "selected_tokens": {"mean": statistics.mean(lengths), "median": statistics.median(lengths),
                                 "max": max(lengths)} if lengths else None,
             "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "stats.json", stats)
    unselected = args.out / "solved_unselected.txt"
    unselected.write_text("".join(f"{k}\n" for k in sorted(solved_unselected)))
    outputs["stats"] = file_entry(args.out / "stats.json")
    outputs["solved_unselected"] = file_entry(unselected)
    if args.compare:
        new = {r["sid"]: r["text_sha256"] for r in read_selected(args.out / "selected.v2.jsonl")}
        ref = {r["sid"]: r["text_sha256"] for r in read_selected(args.compare)}
        rejected = {r["sid"] for r in read_selected(args.out / "rejected.v2.jsonl")}
        cmp = {"reference_file": file_entry(args.compare), **compare_selections(new, ref, rejected)}
        write_new_json(args.out / "compare.json", cmp)
        outputs["compare"] = file_entry(args.out / "compare.json")
        print("confronto:", json.dumps({k: v for k, v in cmp.items() if k not in ("examples", "reference_file")}), flush=True)
    write_new_json(args.out / "selection.json", {
        "schema": "f2-selection/2", "job_id": os.environ.get("SLURM_JOB_ID"), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gen_run": gen_run, "gen": {"dir": str(args.gen), "parts": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": part_digests.get(p.name)} for p in parts]},
        "args": {"keep": args.keep, "keys_only": args.keys_only, "workers": args.workers},
        "exclude_files": [file_entry(p) for p in args.exclude], "outputs": outputs,
        "rejecting_checks": list(REJECTING), "flagged_checks": list(FLAGGED), "code": code_identity(),
        "notes": ["nessuna soluzione è certificata da questa selezione",
                  "le soluzioni scartate dai controlli nuovi non sono sostituite (rejected.v2.jsonl)"]})
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
