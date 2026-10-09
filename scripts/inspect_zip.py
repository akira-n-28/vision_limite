#!/usr/bin/env python3
"""Contenuto di un archivio zip senza estrarlo: cartelle, estensioni, dimensioni, file di testo e campioni.

Da eseguire in un job CPU (slurm/inspect_zip.sbatch), mai sul frontend. Stampa il conteggio dei file per
cartella (fino a --depth livelli) ed estensione, gli schemi dei nomi (cifre sostituite da #), i primi
nomi di ogni cartella, l'inizio dei file di testo (json, jsonl, txt, csv, md) e la struttura del primo
record dei JSON piccoli; copia in --out alcune immagini, per guardarle. Tutte le uscite hanno un tetto
globale (--max-folders, --texts, --samples): archivi con una cartella per campione non devono produrre
log enormi né migliaia di file (job 29414).

    srun $RUN_PYTHON -I scripts/inspect_zip.py ARCHIVIO.zip --out DIR
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

TEXT = {".json", ".jsonl", ".txt", ".csv", ".md", ".yaml", ".yml"}
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MAX_JSON_PARSE = 256 * 2**20  # oltre, solo l'inizio del file


def shape(value, depth: int = 0) -> str:
    """Struttura sintetica di un valore JSON: chiavi, tipi e lunghezze."""
    if depth > 3:
        return "…"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {shape(v, depth + 1)}" for k, v in list(value.items())[:20]) + "}"
    if isinstance(value, list):
        return f"[{len(value)} × {shape(value[0], depth + 1) if value else '-'}]"
    if isinstance(value, str):
        return f"str({len(value)})"
    return type(value).__name__


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--names", type=int, default=5, help="nomi di esempio per cartella")
    ap.add_argument("--max-folders", type=int, default=40, help="cartelle da elencare")
    ap.add_argument("--texts", type=int, default=6, help="file di testo da mostrare in tutto")
    ap.add_argument("--samples", type=int, default=6, help="immagini da copiare in tutto")
    ap.add_argument("--head", type=int, default=1500, help="caratteri da stampare per file di testo")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        total_c = sum(i.compress_size for i in infos)
        total_u = sum(i.file_size for i in infos)
        methods = Counter(i.compress_type for i in infos)
        print(f"{args.archive}: {len(infos)} file, compressi {total_c / 2**30:.2f} GiB, "
              f"estratti {total_u / 2**30:.2f} GiB, metodi {dict(methods)}")

        by_dir: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        for info in infos:
            parts = PurePosixPath(info.filename).parts[:-1]
            by_dir["/".join(parts[: args.depth]) or "."].append(info)

        folders = sorted(by_dir)
        print(f"\n-- {len(folders)} cartelle a profondità {args.depth}; le prime {min(len(folders), args.max_folders)}")
        for folder in folders[: args.max_folders]:
            files = by_dir[folder]
            exts = Counter(PurePosixPath(i.filename).suffix.lower() for i in files)
            patterns = Counter(re.sub(r"\d+", "#", PurePosixPath(i.filename).name) for i in files)
            size = sum(i.file_size for i in files)
            print(f"{folder}: {len(files)} file, {size / 2**20:.0f} MiB, {dict(exts.most_common(6))}")
            print(f"    schemi dei nomi: {dict(patterns.most_common(8))}")
            for info in files[: args.names]:
                print(f"    {info.filename} ({info.file_size} B)")

        print(f"\n-- file di testo (i primi {args.texts})")
        texts = [i for i in infos if PurePosixPath(i.filename).suffix.lower() in TEXT]
        for info in texts[: args.texts]:
            suffix = PurePosixPath(info.filename).suffix.lower()
            print(f"\n## {info.filename} ({info.file_size / 2**20:.1f} MiB)")
            with zf.open(info) as f:
                head = f.read(args.head).decode("utf-8", errors="replace")
            print(head + (" …" if info.file_size > args.head else ""))
            if suffix == ".json" and info.file_size <= MAX_JSON_PARSE:
                data = json.loads(zf.read(info))
                n = len(data) if isinstance(data, (list, dict)) else 1
                first = data[0] if isinstance(data, list) and data else (
                    next(iter(data.values())) if isinstance(data, dict) and data else data)
                print(f"-> {type(data).__name__} con {n} elementi; primo: {shape(first)}")
            elif suffix == ".jsonl":
                with zf.open(info) as f:
                    lines = sum(1 for _ in io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
                print(f"-> {lines} righe")

        print(f"\n-- campioni di immagini (i primi {args.samples})")
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        images = [i for i in infos if PurePosixPath(i.filename).suffix.lower() in IMAGES]
        for info in images[: args.samples]:
            data = zf.read(info)
            with Image.open(io.BytesIO(data)) as im:
                print(f"{info.filename}: {im.width}x{im.height} {im.mode} {im.format}")
            (args.out / info.filename.replace("/", "__")).write_bytes(data)


if __name__ == "__main__":
    main()
