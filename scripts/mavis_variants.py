#!/usr/bin/env python3
"""Confronto delle tre varianti d'immagine di MAVIS-Geometry (text_dominant, text_lite, vision_dominant).

Da eseguire in un job CPU (slurm/mavis_variants.sbatch), mai sul frontend. Per le righe scelte salva
ogni variante alla risoluzione che vede l'encoder (target_size con DIAGRAM_MAX_PATCHES e con
DEFAULT_MAX_PATCHES) e un confronto affiancato, e stampa didascalia e domande. Su tutte le righe
stampa dimensioni originali, token immagine e la frazione di pixel che cambia fra le varianti.

    srun $RUN_PYTHON -I scripts/mavis_variants.py FILE.parquet --out DIR --rows 0 1 2
"""
from __future__ import annotations

import argparse
import io
import statistics
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.processing import DEFAULT_MAX_PATCHES, DIAGRAM_MAX_PATCHES, PATCH_SIZE, target_size, to_rgb  # noqa: E402

VARIANTS = ("text_dominant", "text_lite", "vision_dominant")
Image.MAX_IMAGE_PIXELS = None  # diagrammi fino a ~2500x8800 px, file fidati del dataset


def decode(cell: dict) -> Image.Image:
    with Image.open(io.BytesIO(cell["bytes"])) as image:
        return to_rgb(image)


def encoder_view(image: Image.Image, max_patches: int) -> Image.Image:
    height, width = target_size(image.height, image.width, max_patches)
    return image.resize((width, height), Image.BILINEAR)


def tokens(image: Image.Image, max_patches: int) -> int:
    height, width = target_size(image.height, image.width, max_patches)
    return (height // PATCH_SIZE // 2) * (width // PATCH_SIZE // 2)


def changed_fraction(a: Image.Image, b: Image.Image) -> float:
    """Frazione di pixel che differiscono di più di 32 livelli di grigio, a 512 px di lato lungo."""
    scale = 512 / max(a.size)
    size = (max(1, round(a.width * scale)), max(1, round(a.height * scale)))
    x = np.asarray(a.convert("L").resize(size, Image.BILINEAR), dtype=np.int16)
    y = np.asarray(b.convert("L").resize(size, Image.BILINEAR), dtype=np.int16)
    return float((np.abs(x - y) > 32).mean())


def side_by_side(images: list[Image.Image], labels: list[str]) -> Image.Image:
    height = max(im.height for im in images)
    canvas = Image.new("RGB", (sum(im.width for im in images) + 8 * (len(images) - 1), height + 16), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, label in zip(images, labels):
        canvas.paste(im, (x, 16))
        draw.text((x + 2, 2), label, fill="red")
        x += im.width + 8
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.parquet)
    print(f"{args.parquet}: {table.num_rows} righe")
    texts = table.column("text_en").to_pylist()
    choices = table.column("choices").to_pylist()
    answers = table.column("answer_index").to_pylist()

    stats = {"tok_default": [], "tok_diagram": [], "aspect": []}
    diffs = {"text_dominant~text_lite": [], "text_dominant~vision_dominant": [], "text_lite~vision_dominant": []}
    print("\n-- tutte le righe: dimensioni (td / tl / vd), token a 1024 e 2304 patch, pixel cambiati")
    for i in range(table.num_rows):
        images = {v: decode(table.column(f"image_{v}")[i].as_py()) for v in VARIANTS}
        td, tl, vd = (images[v] for v in VARIANTS)
        stats["tok_default"].append(tokens(td, DEFAULT_MAX_PATCHES))
        stats["tok_diagram"].append(tokens(td, DIAGRAM_MAX_PATCHES))
        stats["aspect"].append(max(td.size) / min(td.size))
        pairs = {"text_dominant~text_lite": (td, tl), "text_dominant~vision_dominant": (td, vd),
                 "text_lite~vision_dominant": (tl, vd)}
        row_diffs = {k: changed_fraction(a, b) for k, (a, b) in pairs.items()}
        for k, d in row_diffs.items():
            diffs[k].append(d)
        sizes = " / ".join(f"{im.width}x{im.height}" for im in images.values())
        print(f"{i:3d} {sizes}  tok {stats['tok_default'][-1]}/{stats['tok_diagram'][-1]}  "
              + "  ".join(f"{k.replace('text_dominant', 'td').replace('text_lite', 'tl').replace('vision_dominant', 'vd')} "
                          f"{100 * d:.2f}%" for k, d in row_diffs.items()))

        if i in args.rows:
            views = []
            for v, im in images.items():
                diagram = encoder_view(im, DIAGRAM_MAX_PATCHES)
                default = encoder_view(im, DEFAULT_MAX_PATCHES)
                diagram.save(args.out / f"row{i:02d}_{v}_2304.png")
                default.save(args.out / f"row{i:02d}_{v}_1024.png")
                views.append(default)
            side_by_side(views, list(VARIANTS)).save(args.out / f"row{i:02d}_compare_1024.png")

    print("\n-- riassunto")
    for k, values in stats.items():
        print(f"{k}: media {statistics.mean(values):.2f}, min {min(values):.2f}, max {max(values):.2f}")
    for k, values in diffs.items():
        same = sum(d < 1e-3 for d in values)
        print(f"pixel cambiati {k}: media {100 * statistics.mean(values):.2f}%, max {100 * max(values):.2f}%, "
              f"righe identiche {same}/{len(values)}")

    for i in args.rows:
        t = texts[i]
        print(f"\n-- riga {i}")
        print(f"[caption] {t['caption']}")
        print(f"[diagram_generation] {t['diagram_generation']}")
        for v in VARIANTS:
            print(f"[{v}_question] {t[f'{v}_question']}")
        print(f"[choices] {choices[i]}  [answer_index] {answers[i]}")
    print(f"\nimmagini in {args.out}")


if __name__ == "__main__":
    main()
