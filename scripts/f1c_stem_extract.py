#!/usr/bin/env python3
"""Immagini STEM reali per i fatti verificati a mano di F1c. Solo in un job SLURM CPU.

Prende i campioni di validazione della sorgente STEM della config (i primi val_count dell'ultimo shard, mai in
training, già senza i contaminati), salta i primi --skip (usati nell'analisi degli errori di F1) e salva i successivi
--n come PNG alla dimensione che vede il modello (data.stem_image), con un indice JSON (chiave, file, max_patches,
domanda e descrizione di Qwen3-VL come aiuto alla lettura). I fatti si scrivono poi a mano in facts.json guardando
le immagini.

    srun $RUN_PYTHON -u -B scripts/f1c_stem_extract.py --config CONFIG --out DIR [--skip 20] [--n 100]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from train_f1 import load_config  # noqa: E402

from limite_vl.data import stem_image  # noqa: E402
from limite_vl.stream import make_sources, validation_samples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip", type=int, default=20)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f1c_stem_extract.py va eseguito in un job SLURM")
    cfg = load_config(args.config)
    src = next(s for s in make_sources(cfg["sources"]) if s.name == "stem")
    raw = validation_samples(src, cfg["val_count"])[args.skip:args.skip + args.n]
    (args.out / "img").mkdir(parents=True, exist_ok=True)
    index = []
    for key, files in raw:
        meta = json.loads(files["json"])
        image, max_patches = stem_image(files)
        path = args.out / "img" / f"{key}.png"
        image.convert("RGB").save(path)
        index.append({"key": key, "image": str(path), "max_patches": max_patches, "size": list(image.size),
                      "source_dataset": meta.get("source_dataset"), "question": meta.get("question"),
                      "caption": meta.get("caption")})
    (args.out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    print(f"{len(index)} immagini in {args.out}")


if __name__ == "__main__":
    main()
