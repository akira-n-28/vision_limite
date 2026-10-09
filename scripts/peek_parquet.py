#!/usr/bin/env python3
"""Ispezione di file Parquet di dataset: schema, righe, lunghezze dei testi, dimensioni delle immagini, esempi.

Da eseguire in un job CPU (slurm/peek_parquet.sbatch), mai sul frontend. Per ogni colonna di testo
(anche dentro struct) stampa la lunghezza in caratteri e, con --tokenizer, in token di Violetto; per
le colonne immagine (struct con `bytes`) le dimensioni di alcune immagini; poi --examples righe con i
testi troncati a --max-chars.

    srun $RUN_PYTHON -I scripts/peek_parquet.py FILE.parquet [...] --tokenizer <snapshot di Violetto>
"""
from __future__ import annotations

import argparse
import io
import statistics
from pathlib import Path

import pyarrow.parquet as pq


def flatten(row: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and "bytes" not in value:
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def describe(values: list[float]) -> str:
    if not values:
        return "-"
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return (f"media {statistics.mean(values):.0f}, mediana {pick(0.5):.0f}, p90 {pick(0.9):.0f}, "
            f"min {values[0]:.0f}, max {values[-1]:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--tokenizer", help="cartella con tokenizer.json (Qwen2Tokenizer)")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--images", type=int, default=20, help="immagini da aprire per colonna")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import Qwen2Tokenizer
        tok = Qwen2Tokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    for path in args.files:
        table = pq.read_table(path)
        print(f"\n######## {path} — {table.num_rows} righe, {path.stat().st_size / 2**20:.1f} MiB")
        print(table.schema.to_string(show_schema_metadata=False))
        rows = [flatten(r) for r in table.to_pylist()]
        if not rows:
            continue
        print("\n-- colonne")
        for col in rows[0]:
            values = [r[col] for r in rows]
            sample = next((v for v in values if v is not None), None)
            if isinstance(sample, str):
                texts = [v for v in values if isinstance(v, str)]
                line = f"{col}: testo, {len(texts)}/{len(values)} non nulli; caratteri {describe([len(t) for t in texts])}"
                if tok is not None:
                    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
                    line += f"; token {describe(lengths)}"
                print(line)
            elif isinstance(sample, dict) and "bytes" in sample:
                from PIL import Image
                sizes = []
                for v in values[: args.images]:
                    if v and v.get("bytes"):
                        with Image.open(io.BytesIO(v["bytes"])) as im:
                            sizes.append(f"{im.width}x{im.height} {im.mode} {im.format}")
                print(f"{col}: immagine; prime {len(sizes)}: {', '.join(sizes[:8])}")
            elif isinstance(sample, list):
                print(f"{col}: lista, lunghezze {describe([len(v) for v in values if v is not None])}; "
                      f"esempio {str(sample)[:200]}")
            else:
                distinct = sorted({str(v) for v in values})
                print(f"{col}: {type(sample).__name__}, {len(distinct)} valori distinti: {distinct[:10]}")
        for i, row in enumerate(rows[: args.examples]):
            print(f"\n-- esempio {i}")
            for col, value in row.items():
                if isinstance(value, dict) and "bytes" in value:
                    continue
                text = str(value)
                print(f"[{col}] {text[: args.max_chars]}{' …' if len(text) > args.max_chars else ''}")


if __name__ == "__main__":
    main()
