#!/usr/bin/env python3
"""F2: shard di training dalle soluzioni scelte (scripts/f2_select.py) e dalle immagini di MMFineReason.

Da eseguire in un job CPU (slurm/f2_build.sbatch). Per ogni soluzione in selected.jsonl cerca il problema negli
shard di MMFineReason (stessa chiave) e scrive un campione `<chiave>-s<k>` con l'immagine originale (byte
invariati) e un JSON con domanda, soluzione di Violetto (da `<think>` in poi), risposta, sorgente, pass rate di
Violetto con la descrizione e max_patches della regola STEM (data.stem_image). La descrizione non entra
nell'esempio: il modello deve ricavare quei fatti dall'immagine. Shard da --per-shard campioni, nell'ordine
degli shard di MMFineReason; ogni tar si scrive in `.tmp`, si rilegge e si rinomina.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shard_llava_pretrain import HashingWriter, add_member  # noqa: E402  (stessa cartella scripts/)

from limite_vl.data import iter_shard  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", required=True, type=Path, help="selected.jsonl di f2_select.py")
    ap.add_argument("--shards", required=True, help="glob degli shard di MMFineReason")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prefix", default="f2-mmfinereason")
    ap.add_argument("--per-shard", type=int, default=50_000)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f2_build.py va eseguito in un job SLURM")
    t0 = time.time()
    solutions: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.selected):
        rec = json.loads(line)
        solutions[rec["key"]].append(rec)
    args.out.mkdir(parents=True, exist_ok=True)
    shard_idx, count, rows = 0, 0, []
    found: set[str] = set()
    tar = fh = hw = None
    keys: list[str] = []

    def close():
        nonlocal tar, fh, hw, keys
        if tar is None:
            return
        tar.close()
        fh.flush()
        os.fsync(fh.fileno())
        fh.close()
        path = args.out / f"{args.prefix}-{shard_idx:05d}.tar"
        tmp = path.with_name(path.name + ".tmp")
        got = [k for k, _ in iter_shard(tmp)]
        if got != keys:
            raise RuntimeError(f"{path.name}: {len(got)} campioni riletti invece di {len(keys)}")
        os.replace(tmp, path)
        rows.append({"name": path.name, "samples": len(keys), "bytes": hw.n, "sha256": hw.sha.hexdigest()})
        print(f"{path.name}: {len(keys)} campioni, {hw.n / 1e9:.2f} GB ({time.time() - t0:.0f} s)", flush=True)
        tar = fh = hw = None
        keys = []

    for shard in sorted(glob.glob(args.shards)):
        for key, files in iter_shard(shard):
            if key not in solutions:
                continue
            found.add(key)
            meta = json.loads(files["json"])
            ext = "png" if "png" in files else "jpg"
            for k, sol in enumerate(solutions[key]):
                if tar is None:
                    path = args.out / f"{args.prefix}-{shard_idx:05d}.tar"
                    fh = open(path.with_name(path.name + ".tmp"), "wb")
                    hw = HashingWriter(fh)
                    tar = tarfile.open(fileobj=hw, mode="w", format=tarfile.USTAR_FORMAT)
                sample = f"{key}-s{k}"
                out = {"id": sample, "problem": key, "source_dataset": meta.get("source_dataset"),
                       "question": meta["question"], "answer": sol["answer"], "solution": sol["solution"],
                       "solution_tokens": sol.get("tokens"), "violetto_pass_rate": sol.get("violetto_pass_rate"),
                       "qwen4b_pass_rate": meta.get("pass_rate"), "max_patches": meta.get("max_patches"),
                       "source": meta.get("source")}
                add_member(tar, f"{sample}.{ext}", files[ext])
                add_member(tar, f"{sample}.json", json.dumps(out, ensure_ascii=False).encode())
                keys.append(sample)
                count += 1
                if len(keys) >= args.per_shard:
                    close()
                    shard_idx += 1
    close()
    missing = sorted(set(solutions) - found)
    manifest = {"selected": str(args.selected), "shards_glob": args.shards, "samples": count,
                "problems": len(found), "missing_problems": len(missing), "missing_examples": missing[:20],
                "shards": rows, "seconds": round(time.time() - t0, 1)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("samples", "problems", "missing_problems", "seconds")}, indent=1))
    if missing:
        sys.exit(f"{len(missing)} problemi di selected.jsonl non trovati negli shard")


if __name__ == "__main__":
    main()
