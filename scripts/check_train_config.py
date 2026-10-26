#!/usr/bin/env python3
"""Controllo CPU di una config di training F1 prima di occupare la GPU: manifest e hash degli shard di ogni sorgente
(train_f1.shard_hashes), insiemi di validazione con immagine vera e bianca e prompt dei campioni
(train_f1.validation_sets), passi totali, liste di decontaminazione obbligatorie (data_lists.missing_required) e
provenienza di tutte le liste. Da eseguire in un job CPU (slurm/check_train_config.sbatch).

    srun $RUN_PYTHON -u -B scripts/check_train_config.py --config CONFIG
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_f1 import load_config, shard_hashes, validation_sets  # noqa: E402  (stessa cartella scripts/)

from limite_vl.data_lists import missing_required, provenance  # noqa: E402
from limite_vl.processing import LimiteVLProcessor  # noqa: E402
from limite_vl.stream import make_sources  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    missing = missing_required(cfg["sources"])
    if missing:
        sys.exit("liste di decontaminazione mancanti nella config:\n" + "\n".join(missing))
    for name, lists in provenance(cfg["sources"]).items():
        print(f"{name} ({lists['family']}): " + "; ".join(f"{k} {Path(e['path']).name} {e['lines']} righe {e['sha256'][:12]}"
                                                         for k in ("exclude", "relabel") for e in lists[k]), flush=True)
    sources = make_sources(cfg["sources"])
    for s in sources:
        h = shard_hashes(s)
        print(f"{s.name}: {s.kind}, {len(h['shards'])} shard, peso {s.weight}, validazione {h['val_shard']}", flush=True)
    processor = LimiteVLProcessor.from_pretrained(cfg["decoder"])
    batches, prompts = validation_sets(sources, processor, cfg)
    for (name, mode), items in sorted(batches.items()):
        print(f"validazione {name}/{mode}: {len(items)} micro-batch", flush=True)
    print("prompt dei campioni:", {k: len(v) for k, v in prompts.items()})
    steps = cfg["total_tokens"] // (cfg["rows_per_step"] * cfg["max_len"])
    print(f"passi: {steps} ({cfg['total_tokens'] / 1e6:.0f}M token)")


if __name__ == "__main__":
    main()
