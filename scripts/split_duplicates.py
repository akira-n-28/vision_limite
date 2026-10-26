#!/usr/bin/env python3
"""Duplicati fra validazione e training, per sorgente (F1-clean).

La validazione di train_f1 sono i primi `val_count` campioni dell'ultimo shard di ogni sorgente (stream.Source.val_shard,
stream.validation_samples); il training salta quelli e usa tutto il resto. Un campione di training con la stessa immagine
(sha256 dei byte) o con lo stesso target di un campione di validazione rende la validazione ottimistica.

Per ogni sorgente si prendono i primi --val-count campioni dell'ultimo shard (512 copre ogni val_count usato) e si
cercano nel resto: immagine identica, target identico (didascalia, formula visibile; per i sintetici solo l'immagine).
Si contano anche i duplicati interni al training (immagini identiche), solo come informazione.

Uscite in --out (cartella nuova): <sorgente>/train_duplicates_of_val.txt (chiavi di training da togliere perché ripetono
un campione di validazione), <sorgente>/val_with_duplicates.jsonl e report.json. Job CPU: slurm/split_duplicates.sbatch.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.latex import lone_index, visible_tokens  # noqa: E402
from limite_vl.provenance import code_identity  # noqa: E402

IMAGE_EXT = ("png", "jpg", "jpeg")


def target(kind: str, meta: dict) -> str | None:
    if kind == "mavis":
        return (meta.get("text_en") or {}).get("caption")
    if kind == "unimer":
        label = meta.get("label") or ""
        new = lone_index(label)
        return " ".join(visible_tokens(new if new is not None else label))
    if kind == "pixmo":
        return meta.get("caption")
    return None


def digest(data: bytes | str | None) -> bytes | None:
    if data is None:
        return None
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).digest()


def describe(task: tuple[str, str, int, int]) -> list[tuple[str, bytes | None, bytes | None]]:
    """(chiave, sha dell'immagine, sha del target) dei campioni [start, stop) dello shard."""
    path, kind, start, stop = task
    out = []
    for i, (key, files) in enumerate(iter_shard(path)):
        if i < start:
            continue
        if i >= stop:
            break
        img = next((files[e] for e in IMAGE_EXT if e in files), None)
        meta = json.loads(files["json"]) if "json" in files else {}
        out.append((key, digest(img), digest(target(kind, meta))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, help="nome:tipo=glob degli shard (tipo: mavis, unimer, pixmo, synth)")
    ap.add_argument("--val-count", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("split_duplicates.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} esiste e non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    big = 10**12
    report = {"schema": "split-duplicates/1", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
              "val_count": args.val_count, "sources": {}}
    for spec in args.source:
        head, _, pattern = spec.partition("=")
        name, _, kind = head.partition(":")
        shards = sorted(glob.glob(pattern))
        if not shards:
            sys.exit(f"{name}: nessuno shard per {pattern}")
        val_shard = shards[-1]
        with Pool(workers) as pool:
            (val,) = pool.map(describe, [(val_shard, kind, 0, args.val_count)])
            tasks = [(p, kind, 0, big) for p in shards[:-1]] + [(val_shard, kind, args.val_count, big)]
            train = [r for rows in pool.map(describe, tasks) for r in rows]
        val_img = {img: key for key, img, _ in val if img}
        val_tgt = {tgt: key for key, _, tgt in val if tgt}
        dup_train, val_hits = [], Counter()
        per_val: dict[str, Counter] = {}
        images = Counter(img for _, img, _ in train if img)
        for key, img, tgt in train:
            by_img = img is not None and img in val_img
            by_tgt = tgt is not None and tgt in val_tgt
            if by_img or by_tgt:
                dup_train.append(key)
                for how, hit in (("immagine", by_img), ("target", by_tgt)):
                    if hit:
                        vk = val_img[img] if how == "immagine" else val_tgt[tgt]
                        per_val.setdefault(vk, Counter())[how] += 1
                        val_hits[how] += 1
        out = args.out / name
        out.mkdir()
        (out / "train_duplicates_of_val.txt").write_text("".join(k + "\n" for k in sorted(set(dup_train))))
        with open(out / "val_with_duplicates.jsonl", "w") as fh:
            for vk, c in sorted(per_val.items()):
                fh.write(json.dumps({"val_key": vk, **c}) + "\n")
        report["sources"][name] = {
            "kind": kind, "shards": len(shards), "val_shard": Path(val_shard).name, "val_samples": len(val),
            "train_samples": len(train), "val_with_duplicates": len(per_val),
            "val_with_image_duplicates": sum(1 for c in per_val.values() if c["immagine"]),
            "val_with_target_duplicates": sum(1 for c in per_val.values() if c["target"]),
            "train_duplicates_of_val": len(set(dup_train)),
            "train_internal_image_duplicates": sum(n - 1 for n in images.values() if n > 1),
        }
        print(name, json.dumps(report["sources"][name]), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
