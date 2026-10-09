#!/usr/bin/env python3
"""Statistiche del flusso di training senza GPU: esempi per sorgente, errori con il loro tipo, riempimento
delle righe, velocità di un worker. Da eseguire in un job CPU (slurm/stream_stats.sbatch).

    srun $RUN_PYTHON -B scripts/stream_stats.py --config config/runs/f1-smoke.json --batches 40
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from train_f1 import load_config  # noqa: E402

from limite_vl.data import build_example  # noqa: E402
from limite_vl.processing import IGNORE, LimiteVLProcessor  # noqa: E402
from limite_vl.stream import MixedRowStream, SourceReader, make_sources  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--batches", type=int, default=40, help="micro-batch da un worker")
    ap.add_argument("--direct", type=int, default=200, help="esempi costruiti direttamente per sorgente")
    args = ap.parse_args()
    cfg = load_config(args.config)
    processor = LimiteVLProcessor.from_pretrained(cfg["decoder"])
    sources = make_sources(cfg["sources"])

    for s in sources:  # costruzione diretta: errori con traccia
        reader = SourceReader(s, cfg["seed"], 0, 1, cfg["val_count"])
        errors, lengths, t0 = Counter(), [], time.time()
        first_trace = None
        for _ in range(args.direct):
            key, files = reader.next()
            try:
                ex = build_example(processor, s.kind, key, files)
                lengths.append(len(ex))
            except Exception as exc:  # noqa: BLE001
                errors[f"{type(exc).__name__}: {str(exc)[:120]}"] += 1
                first_trace = first_trace or (key, traceback.format_exc())
        dt = time.time() - t0
        print(f"\n-- {s.name}: {len(lengths)}/{args.direct} esempi, {1000 * dt / args.direct:.1f} ms/esempio, "
              f"token medi {sum(lengths) / max(1, len(lengths)):.0f}; errori {dict(errors)}", flush=True)
        if first_trace:
            print(f"primo errore ({first_trace[0]}):\n{first_trace[1]}", flush=True)

    stream = MixedRowStream(sources, processor, cfg["max_len"], cfg["micro_rows"], seed=cfg["seed"],
                            shuffle_buffer=cfg["shuffle_buffer"], pack_buffer=cfg["pack_buffer"],
                            val_count=cfg["val_count"])
    it = stream.iter_worker(0, cfg["num_workers"])
    sources_count, t0, fill, labelled = Counter(), time.time(), 0, 0
    for i in range(args.batches):
        batch = next(it)
        sources_count.update(batch["meta"]["sources"])
        fill += int((batch["position_ids"] > 0).sum() + (batch["position_ids"] == 0).sum())  # righe intere
        labelled += int((batch["labels"] != IGNORE).sum())
        if i == 0:
            print(f"\nprimo micro-batch dopo {time.time() - t0:.1f} s (riempimento del buffer)", flush=True)
            t0 = time.time()
    dt = time.time() - t0
    meta = batch["meta"]
    print(f"\n-- flusso, worker 0 di {cfg['num_workers']}: {args.batches} micro-batch, "
          f"{dt / max(1, args.batches - 1):.2f} s/micro-batch dopo il primo; esempi consegnati {dict(sources_count)}; "
          f"estrazioni {meta['state']['mix_draws']}; stats {meta['stats']}; token con label {labelled}", flush=True)


if __name__ == "__main__":
    main()
