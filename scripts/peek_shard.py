#!/usr/bin/env python3
"""Primi campioni di uno shard tar: JSON troncato e immagini salvate in --out (al massimo --n).

Da eseguire in un job (es. slurm/shard_mavis_geometry.sbatch con LIMIT), mai sul frontend.

    srun $RUN_PYTHON -B scripts/peek_shard.py SHARD.tar --out DIR --n 3
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.data import iter_shard  # noqa: E402

IMAGE_EXTS = ("png", "jpg", "jpeg", "webp")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shard", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=1200)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for i, (key, files) in enumerate(iter_shard(args.shard)):
        if i >= args.n:
            break
        text = files.get("json", b"").decode()
        print(f"\n-- {key}: {sorted(files)}\n{text[: args.max_chars]}{' …' if len(text) > args.max_chars else ''}")
        for ext in IMAGE_EXTS:
            if ext in files:
                with Image.open(io.BytesIO(files[ext])) as im:
                    print(f"[{ext}] {im.width}x{im.height} {im.mode}, {len(files[ext])} byte")
                (args.out / f"{key}.{ext}").write_bytes(files[ext])


if __name__ == "__main__":
    main()
