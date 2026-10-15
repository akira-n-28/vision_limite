#!/usr/bin/env python3
"""Quante immagini sono frenate dal tetto di ingrandimento di processing.capped_size: per le immagini STEM di
MMFineReason (gli shard di F2) e per i benchmark visivi, patch e token con la regola attuale (data.stem_image:
al più 2x e 2304 patch) e con un tetto di ingrandimento più alto. La diagnosi delle formule (job 29647) dice che
con più pixel il proiettore di F1 legge meglio le immagini piccole.

Da eseguire in un job CPU (slurm/image_sizes.sbatch). Legge solo le dimensioni (intestazione dell'immagine).

    srun $RUN_PYTHON -u -B scripts/image_sizes.py --shards GLOB [--per-shard N] --benchmark NOME=CARTELLA ...
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from decontam import IMAGE_COLUMNS, LooseFiles, image_from  # noqa: E402

from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.processing import DIAGRAM_MAX_PATCHES, capped_size, exact_patches  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SCALES = (2.0, 3.0, 4.0)


def size_of(data: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.height, im.width
    except Exception:  # noqa: BLE001 - immagine illeggibile
        return None


def report(name: str, sizes: list[tuple[int, int]]) -> None:
    if not sizes:
        print(f"\n== {name}: nessuna immagine", flush=True)
        return
    tokens = defaultdict(list)
    for h, w in sizes:
        for s in SCALES:
            hh, ww = capped_size(h, w, s, DIAGRAM_MAX_PATCHES)
            tokens[s].append(exact_patches(hh, ww) // 4)
    base = tokens[2.0]
    limited = sum(t < DIAGRAM_MAX_PATCHES // 4 for t in base)  # sotto il tetto di patch: frenate dal 2x
    print(f"\n== {name}: {len(sizes)} immagini, lato corto mediano {statistics.median(min(s) for s in sizes):.0f} px, "
          f"frenate dal tetto 2x {limited} ({limited / len(sizes):.1%})", flush=True)
    for s in SCALES:
        t = sorted(tokens[s])
        q = [t[int(p * (len(t) - 1))] for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
        print(f"  tetto {s:.0f}x: token immagine medi {statistics.mean(t):.0f}, quantili 10/25/50/75/90 {q}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", help="glob degli shard di MMFineReason")
    ap.add_argument("--per-shard", type=int, default=2000)
    ap.add_argument("--benchmark", action="append", default=[], help="NOME=CARTELLA dello snapshot")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("image_sizes.py va eseguito in un job SLURM")
    if args.shards:
        sizes = []
        for shard in sorted(glob.glob(args.shards)):
            for i, (_, files) in enumerate(iter_shard(shard)):
                if i >= args.per_shard:
                    break
                data = files.get("png") or files.get("jpg")
                if data and (s := size_of(data)):
                    sizes.append(s)
        report("STEM (shard MMFineReason)", sizes)
    for spec in args.benchmark:
        name, _, folder = spec.partition("=")
        folder = Path(folder)
        loose = LooseFiles(folder)
        sizes = []
        for f in sorted(folder.rglob("*.parquet")):
            table = pq.read_table(f)
            cols = [c for c in IMAGE_COLUMNS if c in table.column_names]
            for row in table.select(cols).to_pylist():
                data = next((d for d in (image_from(row[c], loose) for c in cols) if d), None)
                if data and (s := size_of(data)):
                    sizes.append(s)
            del table
        report(name, sizes)


if __name__ == "__main__":
    main()
