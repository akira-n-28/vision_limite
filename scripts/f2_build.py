#!/usr/bin/env python3
"""F2: shard di training dalle sole soluzioni ammesse dal gate (limite_vl.f2_records.admit con il certificato del lotto).

Da eseguire in un job CPU (slurm/f2_build.sbatch). Vuole una selezione v2 (selected.v2.jsonl di f2_select.py) e il
certificato del suo lotto (scripts/f2_certify.py): se il certificato non è ben formato, non è `certified` o è di
un'altra selezione (sha256 del file), si ferma prima di scrivere. Ogni record passa da admit(): controlli che scartano,
risposta verificata, domanda salvata nella generazione, segnalazioni secondo le regole del certificato, esclusioni e
quarantene dell'audit, testo uguale a quello auditato. --exclude toglie problemi (decontaminazione, esiti di un audit
dei problemi). Gli shard costruiti prima del gate (f2-sft120k-v4) restano come sono; per usarli train_f2.py vuole
un uncertified_reason esplicito nella config.

Per ogni record ammesso cerca il problema negli shard (--shards: MMFineReason, stessa chiave; --mavis-shards: shard
caption di MAVIS-Geometry, chiave `mavis-<meta["id"]>`) e scrive un campione `<chiave>-g<indice della generazione>` con
l'immagine (byte invariati: l'originale di MMFineReason, caption.jpg di MAVIS già a 2304 patch) e un JSON con la domanda
salvata nella generazione o ricostruita esattamente (non quella dello shard), la soluzione, sid e gli sha256 di testo,
domanda e immagine (data.f2_solution_sample li riverifica). Uscite in --out, nuova o vuota: tar da --per-shard campioni
(scritti in `.tmp`, riletti verificando chiavi e hash di testo, domanda e immagine, poi rinominati), samples.jsonl (sample_id, sid,
text_sha256, question_sha256, image_sha256, tar) e manifest.json (schema f2-build/2, con gli sha256 di certificato, selezione,
samples.jsonl e tar, i non ammessi per motivo e l'identità del codice). Chiavi ripetute negli shard, sid ripetuti nella
selezione o problemi ammessi non trovati: errore, e al posto di manifest.json resta build_failed.json.

    srun $RUN_PYTHON -u -B scripts/f2_build.py --selected SEL/selected.v2.jsonl --certificate CERT/certificate.json \
        --shards 'data/shards/mmfinereason/REV-sft/*.tar' --out data/shards/f2-mmfinereason/NOME [--exclude LISTA ...]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shard_llava_pretrain import HashingWriter, add_member  # noqa: E402  (stessa cartella scripts/)

from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.f2_records import admit, eligible, record_version, validate_certificate  # noqa: E402
from limite_vl.provenance import HashingTextWriter, code_identity, file_entry, sha256_text, write_new_json  # noqa: E402

BUILD_SCHEMA = "f2-build/2"


def read_selection(path: Path) -> tuple[list[dict], dict]:
    recs, h, sids = [], hashlib.sha256(), set()
    with open(path, "rb") as fh:
        for n, line in enumerate(fh, 1):
            h.update(line)
            rec = json.loads(line)
            if record_version(rec) != 2:
                sys.exit(f"{path}:{n}: il build vuole una selezione v2 (selected.v2.jsonl)")
            if rec["sid"] in sids:
                sys.exit(f"{path}:{n}: sid ripetuto {rec['sid']}")
            sids.add(rec["sid"])
            recs.append(rec)
    return recs, {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def problem_key(shard_key: str, meta: dict, mavis: bool) -> str:
    return f"mavis-{meta['id']}" if mavis else shard_key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", required=True, type=Path, help="selected.v2.jsonl di f2_select.py")
    ap.add_argument("--certificate", required=True, type=Path, help="certificate.json di f2_certify.py per quella selezione")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shards", help="glob degli shard di MMFineReason")
    src.add_argument("--mavis-shards", help="glob degli shard caption di MAVIS-Geometry")
    ap.add_argument("--out", required=True, type=Path, help="cartella nuova o vuota")
    ap.add_argument("--exclude", nargs="*", default=[], type=Path, help="file di chiavi di problema da escludere")
    ap.add_argument("--prefix", help="nome dei tar (default f2-mmfinereason o f2-mavis)")
    ap.add_argument("--per-shard", type=int, default=50_000)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_build.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: gli shard non si riscrivono, usare una cartella nuova")
    t0 = time.time()
    mavis = bool(args.mavis_shards)
    prefix = args.prefix or ("f2-mavis" if mavis else "f2-mmfinereason")

    # ---- gate: certificato e selezione ----
    cert = json.loads(args.certificate.read_text())
    problems = validate_certificate(cert)
    if problems:
        sys.exit(f"{args.certificate}: certificato non valido: {problems}")
    if cert["decision"] != "certified":
        sys.exit(f"{args.certificate}: lotto non certificato ({'; '.join(cert.get('reasons') or [])[:500]})")
    recs, sel_entry = read_selection(args.selected)
    if cert["population"]["selection_sha256"] != sel_entry["sha256"]:
        sys.exit(f"{args.certificate}: certificato di un'altra selezione (sha256 {cert['population']['selection_sha256'][:12]}, "
                 f"{args.selected} ha {sel_entry['sha256'][:12]})")
    by_sid = {r["sid"]: r["text_sha256"] for r in recs}
    stale = [e[0] for f in ("audited", "excluded", "quarantined") for e in cert[f] if by_sid.get(e[0]) != e[1]]
    if stale:
        sys.exit(f"{args.certificate}: esiti per soluzione che non corrispondono alla selezione (sid o text_sha256): {stale[:5]}")
    pop = sum(eligible(r, cert["record_policy"])[0] for r in recs)
    if pop != cert["population"]["solutions"]:
        sys.exit(f"{args.certificate}: popolazione di {cert['population']['solutions']} soluzioni, la selezione ne ha {pop} ammissibili")
    exclude: set[str] = set()
    for path in args.exclude:
        exclude.update(line.strip() for line in open(path) if line.strip())
    admitted: dict[str, list[dict]] = defaultdict(list)
    refused = Counter()
    for rec in recs:
        if rec["key"] in exclude:
            refused["problema escluso (--exclude)"] += 1
            continue
        ok, why = admit(rec, cert, sel_entry["sha256"])
        if ok:
            admitted[rec["key"]].append(rec)
        else:
            refused[why] += 1
    n_admitted = sum(len(v) for v in admitted.values())
    print(f"{len(recs)} record, ammessi {n_admitted} di {len(admitted)} problemi; non ammessi {dict(refused)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    if not n_admitted:
        sys.exit("nessuna soluzione ammessa dal gate")

    # ---- shard ----
    args.out.mkdir(parents=True, exist_ok=True)
    samples = HashingTextWriter(args.out / "samples.jsonl")
    shard_idx, count, rows = 0, 0, []
    found, seen = set(), set()
    tar = fh = hw = None
    keys: list[str] = []
    expect: dict[str, str] = {}

    def close():
        nonlocal tar, fh, hw, keys, expect
        if tar is None:
            return
        tar.close()
        fh.flush()
        os.fsync(fh.fileno())
        fh.close()
        path = args.out / f"{prefix}-{shard_idx:05d}.tar"
        tmp = path.with_name(path.name + ".tmp")
        got = []
        for k, files in iter_shard(tmp):
            meta = json.loads(files["json"])
            if sha256_text(meta["solution"]) != expect.get(k) or meta["text_sha256"] != expect.get(k):
                raise RuntimeError(f"{tmp.name}: {k} riletto con un testo diverso da quello ammesso")
            image = files.get("png") or files.get("jpg") or b""
            if hashlib.sha256(image).hexdigest() != meta["image_sha256"] or sha256_text(meta["question"]) != meta["question_sha256"]:
                raise RuntimeError(f"{tmp.name}: {k} riletto con immagine o domanda diverse da quelle scritte")
            got.append(k)
        if got != keys:
            raise RuntimeError(f"{path.name}: {len(got)} campioni riletti invece di {len(keys)}")
        os.replace(tmp, path)
        rows.append({"name": path.name, "samples": len(keys), "bytes": hw.n, "sha256": hw.sha.hexdigest()})
        print(f"{path.name}: {len(keys)} campioni, {hw.n / 1e9:.2f} GB ({time.time() - t0:.0f} s)", flush=True)
        tar = fh = hw = None
        keys, expect = [], {}

    try:
        for shard in sorted(glob.glob(args.mavis_shards or args.shards)):
            for shard_key, files in iter_shard(shard):
                meta = json.loads(files["json"])
                key = problem_key(shard_key, meta, mavis)
                if key in seen:
                    raise RuntimeError(f"chiave {key} ripetuta negli shard ({shard}): campioni duplicati")
                seen.add(key)
                if key not in admitted:
                    continue
                found.add(key)
                ext = "png" if "png" in files else "jpg"
                for rec in sorted(admitted[key], key=lambda r: r["completion"]):
                    if tar is None:
                        path = args.out / f"{prefix}-{shard_idx:05d}.tar"
                        fh = open(path.with_name(path.name + ".tmp"), "wb")
                        hw = HashingWriter(fh)
                        tar = tarfile.open(fileobj=hw, mode="w", format=tarfile.USTAR_FORMAT)
                    sample = f"{key}-g{rec['completion']}"
                    image_sha = hashlib.sha256(files[ext]).hexdigest()
                    out = {"id": sample, "problem": key, "sid": rec["sid"], "text_sha256": rec["text_sha256"],
                           "question_sha256": rec["question_sha256"], "image_sha256": image_sha,
                           "question_origin": rec["question_origin"], "source_dataset": rec.get("source_dataset"),
                           "question": rec["question"], "answer": rec["answer"], "solution": rec["solution"],
                           "solution_tokens": rec.get("tokens"), "violetto_pass_rate": rec.get("violetto_pass_rate"),
                           "qwen4b_pass_rate": meta.get("pass_rate"), "max_patches": meta.get("max_patches"),
                           "source": meta.get("source")}
                    add_member(tar, f"{sample}.{ext}", files[ext])
                    add_member(tar, f"{sample}.json", json.dumps(out, ensure_ascii=False).encode())
                    samples.write_json_line({"sample_id": sample, "sid": rec["sid"], "text_sha256": rec["text_sha256"],
                                             "question_sha256": rec["question_sha256"], "image_sha256": image_sha,
                                             "tar": f"{prefix}-{shard_idx:05d}.tar"})
                    keys.append(sample)
                    expect[sample] = rec["text_sha256"]
                    count += 1
                    if len(keys) >= args.per_shard:
                        close()
                        shard_idx += 1
        close()
    except Exception:
        samples.abort()
        raise
    samples_entry = samples.close()
    missing = sorted(set(admitted) - found)
    manifest = {"schema": BUILD_SCHEMA, "job_id": os.environ.get("SLURM_JOB_ID"), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "selected": sel_entry, "certificate": file_entry(args.certificate), "selection_sha256": sel_entry["sha256"],
                "certificate_decision": cert["decision"], "record_policy": cert["record_policy"],
                "policy_file": cert.get("policy", {}).get("file"), "shards_glob": args.mavis_shards or args.shards,
                "source": "mavis-geometry" if mavis else "mmfinereason", "exclude_files": [file_entry(p) for p in args.exclude],
                "records": len(recs), "admitted": n_admitted, "not_admitted": dict(refused), "samples": count,
                "problems": len(found), "missing_problems": len(missing), "missing_examples": missing[:20],
                "samples_file": samples_entry, "shards": rows, "code": code_identity(), "seconds": round(time.time() - t0, 1)}
    if missing:
        write_new_json(args.out / "build_failed.json", manifest)
        sys.exit(f"{len(missing)} problemi ammessi non trovati negli shard: build non valido (build_failed.json)")
    write_new_json(args.out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("records", "admitted", "not_admitted", "samples", "problems", "seconds")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
