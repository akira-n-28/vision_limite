#!/usr/bin/env python3
"""Controlli di UniMER senza LLM su tutti gli shard di train (F1-clean).

Per ogni campione (immagine PNG e JSON con la formula):
- struttura del LaTeX (latex.structure_problems: graffe, \\left/\\right, ambienti);
- formula senza token visibili;
- stessa immagine (sha256 dei byte) con formule visibili diverse: etichette in conflitto. Con le stesse formule
  visibili è solo un duplicato;
- immagine decodificata: vuota (nessun pixel scuro o contrasto quasi nullo) o con dimensioni diverse da quelle del JSON;
- area originale per token visibile fuori proporzione: z robusto (mediana e MAD del logaritmo) oltre --z. Sono solo
  candidati da guardare, non errori.

Le etichette vengono dal LaTeX che ha generato l'immagine (salvo CROHME, scritta a mano, < 1%): i controlli cercano
disallineamenti, rese fallite ed etichette malformate.

Uscite in --out (cartella nuova): report.json (conteggi, esempi, soglie, identità di codice e shard) e flagged.jsonl
(un record per campione segnalato). Job CPU: slurm/unimer_check.sbatch.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.latex import structure_problems, visible_tokens  # noqa: E402
from limite_vl.provenance import code_identity, file_entry  # noqa: E402

EXAMPLES = 20


def iter_pairs(path: Path):
    """(chiave, json, png) dagli shard: i due file di un campione sono consecutivi."""
    pending: dict[str, dict] = {}
    with tarfile.open(path, "r|") as tar:
        for m in tar:
            if not m.isfile():
                continue
            key, ext = m.name.rsplit(".", 1)
            pending.setdefault(key, {})[ext] = tar.extractfile(m).read()
            if {"json", "png"} <= pending[key].keys():
                files = pending.pop(key)
                yield key, json.loads(files["json"]), files["png"]
    if pending:
        raise RuntimeError(f"{path.name}: campioni incompleti {sorted(pending)[:5]}")


def check_shard(path: Path) -> list[tuple]:
    out = []
    for key, meta, png in iter_pairs(path):
        label = meta["label"]
        vis = visible_tokens(label)
        problems = structure_problems(label)
        if not vis:
            problems.append("formula_vuota")
        with Image.open(io.BytesIO(png)) as im:
            gray = im.convert("L")
            lo, hi = gray.getextrema()
            hist = gray.histogram()
            dark = sum(hist[:128]) / max(1, gray.width * gray.height)
            if (gray.width, gray.height) != (meta["width"], meta["height"]):
                problems.append("dimensioni_diverse_dal_json")
        if hi - lo < 16 or dark == 0:
            problems.append("immagine_vuota")
        area = meta["orig_width"] * meta["orig_height"]
        ratio = math.log(area / len(vis)) if vis and area else None
        out.append((key, hashlib.sha256(png).digest(), hashlib.sha1(" ".join(vis).encode()).digest(), label,
                    meta["orig_width"], meta["orig_height"], len(vis), ratio, problems))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--z", type=float, default=5.0)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("unimer_check.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} esiste e non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    rows = []
    with Pool(workers) as pool:
        for path, res in zip(args.shards, pool.imap(check_shard, args.shards)):
            rows.extend(res)
            print(f"{path.name}: {len(res)} campioni", flush=True)

    flags: dict[str, list[str]] = defaultdict(list)
    for key, *_, problems in rows:
        flags[key].extend(problems)
    # stessa immagine, formule visibili diverse
    by_image: dict[bytes, tuple] = {}
    conflicts, duplicates = [], 0
    for key, img, vis, label, *_ in rows:
        if img in by_image:
            k0, vis0, label0 = by_image[img]
            if vis0 != vis:
                conflicts.append({"ids": [k0, key], "labels": [label0, label]})
                flags[k0].append("stessa_immagine_formula_diversa")
                flags[key].append("stessa_immagine_formula_diversa")
            else:
                duplicates += 1
        else:
            by_image[img] = (key, vis, label)
    # proporzioni: z robusto del logaritmo dell'area per token visibile
    ratios = [r[7] for r in rows if r[7] is not None]
    med = statistics.median(ratios)
    mad = statistics.median(abs(x - med) for x in ratios) * 1.4826
    outliers = []
    for key, _, _, label, w, h, n, ratio, _ in rows:
        if ratio is not None and mad and abs(ratio - med) / mad > args.z:
            z = (ratio - med) / mad
            flags[key].append("proporzioni_alte" if z > 0 else "proporzioni_basse")
            outliers.append({"id": key, "z": round(z, 2), "orig": [w, h], "visible_tokens": n, "label": label[:300]})

    info = {r[0]: r for r in rows}
    counts, examples = Counter(), defaultdict(list)
    with open(args.out / "flagged.jsonl", "w") as fh:
        for key, cats in flags.items():
            if not cats:
                continue
            cats = sorted(set(cats))
            counts.update(cats)
            r = info[key]
            rec = {"id": key, "issues": cats, "label": r[3], "orig": [r[4], r[5]], "visible_tokens": r[6]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for c in cats:
                if len(examples[c]) < EXAMPLES:
                    examples[c].append(rec)
    outliers.sort(key=lambda o: -abs(o["z"]))
    report = {
        "schema": "unimer-check/1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code": code_identity(),
        "shards": [file_entry(s) for s in args.shards],
        "samples": len(rows),
        "flagged_samples": sum(1 for c in flags.values() if c),
        "issue_samples": dict(counts.most_common()),
        "image_duplicates_same_formula": duplicates,
        "conflicts": conflicts[:200],
        "ratio_log_median": med, "ratio_log_mad": mad, "z": args.z,
        "outliers_top": outliers[:60],
        "examples": examples,
        "note": "proporzioni_* sono candidati da guardare, non errori; gli altri sono difetti dell'etichetta o dell'immagine",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("samples", "flagged_samples", "issue_samples", "image_duplicates_same_formula",
                                             "ratio_log_median", "ratio_log_mad")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
