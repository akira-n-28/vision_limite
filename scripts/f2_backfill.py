#!/usr/bin/env python3
"""F2: backfill non distruttivo di una selezione storica nel formato v2 (limite_vl.f2_records).

Da eseguire in un job CPU (slurm/f2_backfill.sbatch). Le righe di selected.jsonl delle selezioni v1 non dicono da
quale generazione vengono. Per ognuna lo script ritrova la generazione nelle parti di f2_generate.py e scrive,
accanto alla selezione e senza toccarla:
- selected.v2.jsonl: le righe ritrovate come record f2-sol/2 (sid, raw_sha256, text_sha256, controlli), nello stesso
  ordine di selected.jsonl;
- backfill_issues.jsonl: le righe non ritrovate o ambigue, con i candidati, senza nessuna associazione inventata;
- audit_map-<audit>.jsonl (con --audit): per ogni item di un audit delle soluzioni F2 (stem_audit_export.py --f2-mavis)
  il sid della soluzione auditata e l'hash del testo visto dagli agenti;
- backfill.json: conteggi, hash degli input e delle uscite, identità del codice.
Rifiuta di partire se una di queste uscite esiste già.

Corrispondenza esatta, mai per somiglianza. Una riga v1 corrisponde alla generazione i della stessa chiave se
quella generazione è finita da sé (finish_reason stop, come richiedeva la selezione) e se:
- la riga non è riscritta (`rewritten` falso) e il suo testo ha lo stesso sha256 della generazione (`raw`), oppure
- la riga è riscritta e il suo testo ha lo stesso sha256 di f2.clean_solution(generazione) con il codice attuale
  (`cleaned`).
f2_select.py ha sempre scritto le soluzioni di un problema in ordine crescente di indice (sorted(rng.sample(...)),
in tutte le versioni del file nel repository): un'assegnazione delle righe della chiave alle generazioni deve quindi
essere crescente. Se ne esiste una sola, le righe sono ritrovate (`order` segnala che senza l'ordine c'erano più
candidati, per esempio generazioni con testo identico); se ne esistono più di una, la chiave è ambigua; se nessuna,
le righe sono mancanti. Una riga riscritta con regole di pulizia diverse da quelle attuali risulta mancante.

Il backfill non certifica nulla e non scrive certificati: `answer_correct` è ereditato dalla selezione v1 (che
verificava la risposta) e non viene ricalcolato.

    srun $RUN_PYTHON -u -B scripts/f2_backfill.py --gen outputs/f2_gen/RUN --selected outputs/f2_select/SEL/selected.jsonl \
        [--audit outputs/stem_audit/NOME ...]
"""
from __future__ import annotations

import argparse
import hashlib
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
from limite_vl.f2 import clean_solution  # noqa: E402
from limite_vl.f2_records import FLAGGED, REJECTING, gen_run_name, solution_record  # noqa: E402
from limite_vl.provenance import (HashingTextWriter, code_identity, file_entry, refuse_existing, sha256_text,  # noqa: E402
                                  write_new_json)

_KEY_PREFIX = re.compile(rb'^\{"key": ("(?:[^"\\]|\\.)*")')


def completion_digests(line: bytes) -> dict:
    """Per una riga delle parti: metadati del problema e, per ogni generazione, sha256 del testo e del testo ripulito."""
    row = json.loads(line)
    comps = []
    for i, c in enumerate(row["completions"]):
        text = c["text"]
        cleaned, _ = clean_solution(text)
        comps.append({"i": i, "raw": sha256_text(text), "cleaned": sha256_text(cleaned),
                      "finish_reason": c["finish_reason"], "tokens": c.get("tokens")})
    return {"key": row["key"], "source_dataset": row.get("source_dataset"), "answer": row.get("answer"),
            "answer_letter": row.get("answer_letter"), "completions": comps}


def match_key(old: list[dict], comps: list[dict]) -> dict:
    """Assegnazione delle righe v1 di una chiave (nell'ordine del file: {"sha", "rewritten"}) alle generazioni
    ({"i", "raw", "cleaned", "finish_reason"}). Ritorna {"status": matched|ambiguous|missing, "assign": [(i, metodo)]
    per riga se matched, "candidates": candidati per riga}."""
    candidates = []
    for r in old:
        fields = {True: ("cleaned",), False: ("raw",), None: ("raw", "cleaned")}[r["rewritten"]]
        candidates.append([(c["i"], next(f for f in fields if c[f] == r["sha"])) for c in comps
                           if c["finish_reason"] == "stop" and any(c[f] == r["sha"] for f in fields)])
    found = []

    def search(j: int, last: int, acc: list) -> None:
        if len(found) > 1:
            return
        if j == len(old):
            found.append(list(acc))
            return
        for i, method in candidates[j]:
            if i > last:
                search(j + 1, i, acc + [(i, method)])

    search(0, -1, [])
    cand = [[i for i, _ in c] for c in candidates]
    if len(found) == 1:
        assign = [(i, m if len(candidates[j]) == 1 else f"{m}+order") for j, (i, m) in enumerate(found[0])]
        return {"status": "matched", "assign": assign, "candidates": cand}
    return {"status": "ambiguous" if found else "missing", "assign": None, "candidates": cand}


def is_injection(source: str, altered: str, original: str, changed: str) -> bool:
    """True se `altered` è `source` con una sola occorrenza di `original` sostituita da `changed` (la modifica di
    stem_audit_export.inject_solution)."""
    pos = source.find(original)
    while pos >= 0:
        if source[:pos] + changed + source[pos + len(original):] == altered:
            return True
        pos = source.find(original, pos + 1)
    return False


def audit_map(audit_dir: Path, kept: dict[str, list[dict]]) -> tuple[list[dict], Counter]:
    """Item dell'audit -> sid. Item reali: la soluzione con lo stesso text_sha256 fra quelle ritrovate della chiave.
    Item di calibrazione: la prima soluzione della chiave (quella che stem_audit_export.py ha alterato), verificata
    ricostruendo la modifica (un solo tratto: misura originale -> misura cambiata di _gold/calibration_f2.json)."""
    items = json.loads((audit_dir / "_gold" / "items_f2.json").read_text())
    truth_path = audit_dir / "_gold" / "calibration_f2.json"
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else {}
    verdicts = {}
    if (audit_dir / "audit_f2.jsonl").exists():
        for line in open(audit_dir / "audit_f2.jsonl"):
            r = json.loads(line)
            verdicts[r["fid"]] = r["verdict"]
    rows, counts = [], Counter()
    for it in items:
        sha = sha256_text(it["solution"])
        row = {"fid": it["fid"], "audit": audit_dir.name, "key": it["key"], "calibration": bool(it.get("calibration")),
               "audited_text_sha256": sha, "verdict": verdicts.get(it["fid"])}
        recs = kept.get(it["key"], [])
        if not row["calibration"]:
            hits = [r for r in recs if r["text_sha256"] == sha]
            if len(hits) == 1:
                row.update({"sid": hits[0]["sid"], "position": hits[0]["_position"], "status": "exact"})
            else:
                row.update({"sid": None, "status": "missing" if not hits else "ambiguous",
                            "candidates": [r["sid"] for r in hits]})
        else:
            t = truth.get(it["fid"])
            first = recs[0] if recs and recs[0]["_position"] == 0 else None
            ok = bool(first and t and t.get("key") == it["key"]
                      and is_injection(first["solution"], it["solution"], t["original"], t["changed"]))
            row.update({"source_sid": first["sid"] if first else None,
                        "source_text_sha256": first["text_sha256"] if first else None,
                        "status": "injected-verified" if ok else "unverified"})
            if not ok:
                row["reason"] = ("prima soluzione della chiave non ritrovata" if not first else
                                 "verità di calibrazione mancante" if not t else "modifica diversa da quella registrata")
        counts[f"{'calibrazione' if row['calibration'] else 'reali'}:{row['status']}"] += 1
        rows.append(row)
    return rows, counts


def iter_part_lines(parts: list[Path], wanted: set[str], digests: dict, seen: Counter):
    """Righe delle parti con chiave in `wanted` (riconosciuta senza decodificare la riga); calcola intanto lo sha256
    di ogni parte e conta le chiavi viste (per le chiavi ripetute)."""
    for part in parts:
        h = hashlib.sha256()
        with open(part, "rb") as fh:
            for line in fh:
                h.update(line)
                m = _KEY_PREFIX.match(line)
                if not m:
                    raise ValueError(f"{part}: riga senza la chiave come prima voce")
                key = json.loads(m.group(1))
                seen[key] += 1
                if key in wanted:
                    yield line
        digests[part.name] = h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, type=Path, help="cartella di f2_generate.py")
    ap.add_argument("--selected", required=True, type=Path, help="selected.jsonl v1 di f2_select.py")
    ap.add_argument("--out-dir", type=Path, help="cartella delle uscite (default: quella di --selected)")
    ap.add_argument("--audit", nargs="*", type=Path, default=[], help="cartelle di audit delle soluzioni F2 da collegare")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_backfill.py va eseguito in un job SLURM")
    t0 = time.time()
    out = args.out_dir or args.selected.parent
    names = ["selected.v2.jsonl", "backfill_issues.jsonl", "backfill.json"] + [f"audit_map-{a.name}.jsonl" for a in args.audit]
    refuse_existing(*(out / n for n in names))
    parts = sorted(args.gen.glob("part-*.jsonl"))
    if not parts:
        sys.exit(f"nessuna parte in {args.gen}")
    for a in args.audit:
        if not (a / "_gold" / "items_f2.json").is_file():
            sys.exit(f"{a}: manca _gold/items_f2.json (non è un audit delle soluzioni F2)")
    gen_run = gen_run_name(args.gen)
    out.mkdir(parents=True, exist_ok=True)

    # 1. righe v1 per chiave: sha256 del testo e flag rewritten, con lo sha256 del file
    by_key: dict[str, list[dict]] = defaultdict(list)
    h, n_v1 = hashlib.sha256(), 0
    with open(args.selected, "rb") as fh:
        for n, line in enumerate(fh, 1):
            h.update(line)
            rec = json.loads(line)
            if "schema" in rec:
                sys.exit(f"{args.selected}:{n}: record con schema {rec['schema']!r}: il backfill legge solo selezioni v1")
            rewritten = rec.get("rewritten")
            by_key[rec["key"]].append({"line": n, "sha": sha256_text(rec["solution"]),
                                       "rewritten": None if rewritten is None else bool(rewritten)})
            n_v1 += 1
    selected_entry = {"path": str(args.selected), "bytes": args.selected.stat().st_size, "sha256": h.hexdigest()}
    print(f"selezione v1: {n_v1} righe, {len(by_key)} chiavi ({time.time() - t0:.0f} s)", flush=True)

    # 2. generazioni delle chiavi della selezione
    comps, meta, part_digests, seen = {}, {}, {}, Counter()
    with Pool(args.workers) as pool:
        for d in pool.imap(completion_digests, iter_part_lines(parts, set(by_key), part_digests, seen), chunksize=8):
            comps[d["key"]] = d["completions"]
            meta[d["key"]] = d
    print(f"generazioni lette per {len(comps)} chiavi ({time.time() - t0:.0f} s)", flush=True)

    # 3. assegnazioni per chiave
    results, issues_by_line, status = {}, {}, Counter()
    repeated = {k for k in by_key if seen[k] > 1}
    for key, old in by_key.items():
        if key in repeated:
            res = {"status": "duplicate_key_in_parts", "assign": None, "candidates": None}
        elif key not in comps:
            res = {"status": "key_not_in_parts", "assign": None, "candidates": None}
        else:
            res = match_key(old, comps[key])
        results[key] = res
        for j, r in enumerate(old):
            if res["status"] != "matched":
                issues_by_line[r["line"]] = {"line": r["line"], "key": key, "position": j, "status": res["status"],
                                             "rewritten": r["rewritten"], "text_sha256": r["sha"],
                                             "candidates": res["candidates"][j] if res["candidates"] else None}
        status[res["status"]] += len(old)

    # 4. scrittura: record v2 nell'ordine di selected.jsonl, problemi a parte
    audit_keys = set()
    for a in args.audit:
        audit_keys |= {it["key"] for it in json.loads((a / "_gold" / "items_f2.json").read_text())}
    kept: dict[str, list[dict]] = defaultdict(list)
    methods, failed, flags = Counter(), Counter(), Counter()
    position = Counter()
    sel = HashingTextWriter(out / "selected.v2.jsonl")
    iss = HashingTextWriter(out / "backfill_issues.jsonl")
    with open(args.selected, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            rec = json.loads(line)
            key = rec["key"]
            j = position[key]
            position[key] += 1
            if n in issues_by_line:
                iss.write_json_line(issues_by_line[n])
                continue
            i, method = results[key]["assign"][j]
            c = comps[key][i]
            m = meta[key]
            v2 = solution_record(gen_run=gen_run, key=key, completion=i, solution=rec["solution"], raw_sha256=c["raw"],
                                 finish_reason=c["finish_reason"], tokens=rec.get("tokens", c["tokens"]),
                                 source_dataset=rec.get("source_dataset"), answer=rec["answer"],
                                 rewritten=method.startswith("cleaned"), violetto_pass_rate=rec.get("violetto_pass_rate"),
                                 origin="backfill", answer_letter=m.get("answer_letter"), match=method)
            sel.write_json_line(v2)
            methods[method] += 1
            failed.update(v2["failed"])
            flags.update(v2["flags"])
            if key in audit_keys:
                kept[key].append({**v2, "_position": j})
    outputs = {"selected_v2": sel.close(), "issues": iss.close()}

    # 5. item degli audit
    maps = {}
    for a in args.audit:
        rows, counts = audit_map(a, kept)
        w = HashingTextWriter(out / f"audit_map-{a.name}.jsonl")
        for row in rows:
            w.write_json_line(row)
        outputs[f"audit_map-{a.name}"] = w.close()
        maps[a.name] = {"items": len(rows), **dict(sorted(counts.items())),
                        "audit_manifest_sha256": hashlib.sha256((a / "manifest.json").read_bytes()).hexdigest()}

    plan = args.gen / "plan.json"
    manifest = {"schema": "f2-backfill/1", "job_id": os.environ.get("SLURM_JOB_ID"),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gen_run": gen_run,
                "gen": {"dir": str(args.gen), "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest() if plan.exists() else None,
                        "parts": [{"name": p.name, "bytes": p.stat().st_size, "sha256": part_digests.get(p.name)} for p in parts]},
                "selected_v1": selected_entry, "outputs": outputs,
                "counts": {"v1_rows": n_v1, "v1_keys": len(by_key), "v2_rows": outputs["selected_v2"]["lines"],
                           "rows_by_status": dict(status), "matched_by_method": dict(methods),
                           "keys_by_status": dict(Counter(r["status"] for r in results.values()))},
                "checks": {"rejecting": list(REJECTING), "flagged": list(FLAGGED),
                           "failed": dict(failed), "flags": dict(flags)},
                "audit_maps": maps, "code": code_identity(),
                "notes": ["answer_correct ereditato dalla selezione v1 (che verificava la risposta), non ricalcolato",
                          "i record di backfill non sono certificati e il backfill non scrive certificati",
                          "una riga riscritta con regole di clean_solution diverse da quelle attuali risulta mancante"]}
    if manifest["counts"]["v2_rows"] + outputs["issues"]["lines"] != n_v1:
        raise RuntimeError("righe v2 + problemi diverse dalle righe v1")
    write_new_json(out / "backfill.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("counts", "checks", "audit_maps")}, indent=1, ensure_ascii=False))
    print(f"fine in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
