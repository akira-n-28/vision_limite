#!/usr/bin/env python3
"""UniMER: sovrapposizioni fra il training e UniMER-Test (spe, cpe, sce, hwe), che è la metrica delle formule di F1.

Per la trascrizione conta la formula, non l'immagine: un campione di training con la stessa formula visibile di un
campione di test (latex.visible_tokens, dopo la rietichettatura degli indici isolati) insegna proprio quella risposta.
Si cercano:
- formula visibile identica (la sovrapposizione che conta per la metrica);
- immagine identica (sha256 dei byte);
- immagine quasi identica (pHash entro --radius bit): solo contata, perché sulle formule brevi il pHash confonde
  immagini diverse.
Le formule brevi si ripetono per forza (`x^2`): il report divide i conteggi per lunghezza, così si può scegliere se
togliere dal training le formule del test o valutare a parte i casi mai visti.

Uscite in --out (cartella nuova): report.json, test_seen.jsonl (per ogni campione di test visto nel training: come e
quante volte) e train_matches_test.txt (chiavi di training con formula o immagine di un campione di test).
Job CPU: slurm/unimer_test_overlap.sbatch.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.decontam import HashIndex, phash  # noqa: E402
from limite_vl.latex import lone_index, visible_tokens  # noqa: E402
from limite_vl.provenance import code_identity, file_entry  # noqa: E402

BUCKETS = ((1, 5), (6, 20), (21, 50), (51, 10**9))
_TEST: dict = {}


def iter_pairs(path: Path):
    pending: dict[str, dict] = {}
    with tarfile.open(path, "r|") as tar:
        for m in tar:
            if not m.isfile():
                continue
            key, ext = m.name.rsplit("/", 1)[-1].split(".", 1)
            pending.setdefault(key, {})[ext] = tar.extractfile(m).read()
            if {"json", "png"} <= pending[key].keys():
                files = pending.pop(key)
                yield key, json.loads(files["json"]), files["png"]


def label_of(meta: dict) -> str:
    """Formula del campione: `label` per le formule intere, le righe (`rows`) per gli shard delle righe."""
    return meta["label"]


def vis_string(label: str) -> str:
    new = lone_index(label)
    return " ".join(visible_tokens(new if new is not None else label))


def describe(path: Path) -> list[tuple]:
    out = []
    for key, meta, png in iter_pairs(path):
        with Image.open(io.BytesIO(png)) as im:
            h = phash(im)
        out.append((key, path.name, vis_string(label_of(meta)), hashlib.sha256(png).digest(), h))
    return out


def init(test: dict) -> None:
    _TEST.update(test)
    _TEST["hidx"] = HashIndex([h for h in test["phash_list"]], radius=test["radius"])


def scan(path: Path) -> list[tuple]:
    hits = []
    for key, meta, png in iter_pairs(path):
        vis = vis_string(label_of(meta))
        sha = hashlib.sha256(png).digest()
        formula = vis in _TEST["by_vis"]
        image = sha in _TEST["by_sha"]
        with Image.open(io.BytesIO(png)) as im:
            near = bool(_TEST["hidx"].query(phash(im)))
        if formula or image or near:
            hits.append((key, path.name, vis, formula, image, near, sha if image else None))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-shards", nargs="+", type=Path, required=True)
    ap.add_argument("--train-shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius", type=int, default=2)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("unimer_test_overlap.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} esiste e non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

    with Pool(workers) as pool:
        test_rows = [r for rows in pool.map(describe, args.test_shards) for r in rows]
    subset = {r[0]: r[1].replace("unimer-test-", "").replace(".tar", "") for r in test_rows}
    by_vis, by_sha = defaultdict(list), defaultdict(list)
    for key, _, vis, sha, _ in test_rows:
        by_vis[vis].append(key)
        by_sha[sha].append(key)
    test = {"by_vis": dict(by_vis), "by_sha": dict(by_sha), "phash_list": [r[4] for r in test_rows], "radius": args.radius}
    print(f"test: {len(test_rows)} campioni, {len(by_vis)} formule visibili distinte", flush=True)

    hits = []
    with Pool(workers, initializer=init, initargs=(test,)) as pool:
        for path, res in zip(args.train_shards, pool.imap(scan, args.train_shards)):
            hits.extend(res)
            print(f"{path.name}: {len(res)} campioni di training con una corrispondenza", flush=True)

    seen = defaultdict(Counter)
    train_keys = []
    near_only = 0
    for key, shard, vis, formula, image, near, sha in hits:
        if formula:
            for t in by_vis[vis]:
                seen[t]["formula"] += 1
        if image:
            for t in by_sha[sha]:
                seen[t]["image"] += 1
        if formula or image:
            train_keys.append(key)
        elif near:
            near_only += 1
    test_len = {r[0]: len(r[2].split()) for r in test_rows}
    per_subset = defaultdict(Counter)
    for key in subset:
        sub = subset[key]
        n = test_len[key]
        bucket = next(f"{a}-{b}" if b < 10**9 else f"{a}+" for a, b in BUCKETS if a <= n <= b)
        per_subset[sub]["test"] += 1
        per_subset[sub][f"test_{bucket}"] += 1
        if seen[key]["formula"]:
            per_subset[sub]["formula_seen"] += 1
            per_subset[sub][f"formula_seen_{bucket}"] += 1
        if seen[key]["image"]:
            per_subset[sub]["image_seen"] += 1
    with open(args.out / "test_seen.jsonl", "w") as fh:
        for key in sorted(k for k in seen if seen[k]):
            fh.write(json.dumps({"key": key, "subset": subset[key], "visible_tokens": test_len[key], **seen[key]}) + "\n")
    (args.out / "train_matches_test.txt").write_text("".join(k + "\n" for k in sorted(set(train_keys))))
    report = {
        "schema": "unimer-test-overlap/1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code": code_identity(),
        "test_shards": [file_entry(p) for p in args.test_shards],
        "train_shards": [file_entry(p) for p in args.train_shards],
        "test_samples": len(test_rows),
        "train_samples_matching": len(set(train_keys)),
        "train_samples_near_image_only": near_only,
        "per_subset": {k: dict(v) for k, v in sorted(per_subset.items())},
        "note": "formula_seen: la formula visibile del test compare nel training; image_seen: stessa immagine",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("test_samples", "train_samples_matching", "train_samples_near_image_only",
                                             "per_subset")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
