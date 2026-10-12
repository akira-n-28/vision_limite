#!/usr/bin/env python3
"""Analisi di UniMER-1M.zip e UniMER-Test.zip (wanderkid/UniMER_Dataset) senza estrarli, per gli shard di F1.

Da eseguire in un job CPU (slurm/unimer_survey.sbatch), mai sul frontend. Uscite a tetto fisso: al massimo
--save immagini in --out.

1. train: righe di train.txt, indici delle immagini, intervalli di righe senza immagine;
2. confini presunti fra le parti (Pix2tex 158.303, arXiv 820.152, CROHME 8.834, HME100K 74.502 righe, in
   quest'ordine secondo la scheda): immagini e formule ai bordi, salvate per guardarle;
3. dimensioni e modo delle immagini su un campione, per parte;
4. formule: caratteri e token di Violetto nel formato spaziato e dopo despace, controllo che despace
   conservi i token LaTeX, esempi;
5. test: righe, immagini e indici per sottoinsieme;
6. costo in token immagine con l'ingrandimento proposto e controllo di idempotenza della dimensione finale.

    srun $RUN_PYTHON -B scripts/unimer_survey.py --train-zip Z --test-zip T --tokenizer VIOLETTO --out DIR
"""
from __future__ import annotations

import argparse
import io
import math
import random
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.latex import despace, latex_tokens  # noqa: E402
from limite_vl.processing import PATCH_SIZE, UNIT, target_size  # noqa: E402

PARTS = (("pix2tex", 158_303), ("arxiv", 820_152), ("crohme", 8_834), ("hme100k", 74_502))


def describe(values, fmt=".0f") -> str:
    if not values:
        return "-"
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * len(v)))]  # noqa: E731
    return (f"media {statistics.mean(v):{fmt}}, mediana {pick(0.5):{fmt}}, p10 {pick(0.1):{fmt}}, "
            f"p90 {pick(0.9):{fmt}}, p99 {pick(0.99):{fmt}}, min {v[0]:{fmt}}, max {v[-1]:{fmt}}")


def part_of(index: int) -> str:
    start = 0
    for name, n in PARTS:
        if index < start + n:
            return name
        start += n
    return "oltre"


def ranges(sorted_missing: list[int]) -> list[tuple[int, int]]:
    out = []
    for i in sorted_missing:
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def final_size(h: int, w: int, scale: float, max_patches: int) -> tuple[int, int]:
    """Dimensione proposta: ingrandimento fino a `scale`, multipli di 32, al più max_patches patch."""
    s = scale
    while True:
        hh = max(UNIT, math.ceil(h * s / UNIT) * UNIT)
        ww = max(UNIT, math.ceil(w * s / UNIT) * UNIT)
        if (hh // PATCH_SIZE) * (ww // PATCH_SIZE) <= max_patches or s < 0.05:
            return hh, ww
        s *= 0.95


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-zip", required=True, type=Path)
    ap.add_argument("--test-zip", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--save", type=int, default=16)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--max-patches", type=int, default=1024)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    from transformers import Qwen2Tokenizer
    tok = Qwen2Tokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    # 1. train
    zf = zipfile.ZipFile(args.train_zip)
    names = {i.filename: i for i in zf.infolist() if not i.is_dir()}
    labels = zf.read("UniMER-1M/train.txt").decode("utf-8").split("\n")
    if labels and labels[-1] == "":
        labels.pop()
    images = {}
    for name in names:
        p = PurePosixPath(name)
        if p.parent.as_posix() == "UniMER-1M/images" and p.suffix == ".png" and p.stem.isdigit():
            images[int(p.stem)] = name
    print(f"train.txt: {len(labels)} righe; immagini {len(images)}; indice minimo {min(images)}, massimo {max(images)}; "
          f"indici oltre le righe {sum(i >= len(labels) for i in images)}; righe vuote {sum(not l.strip() for l in labels)}",
          flush=True)
    missing = sorted(set(range(len(labels))) - set(images))
    rs = ranges(missing)
    print(f"righe senza immagine: {len(missing)} in {len(rs)} intervalli; più lunghi "
          f"{sorted(rs, key=lambda r: r[0] - r[1])[:8]}; per parte {dict(Counter(part_of(i) for i in missing))}")
    print(f"immagini per parte presunta: {dict(Counter(part_of(i) for i in images))}", flush=True)

    # 2. bordi fra le parti
    saved = 0
    start = 0
    picks = []
    for name, n in PARTS:
        for i in (start, start + n // 2, start + n - 1):
            picks.append(i)
        start += n
    print("\n-- bordi delle parti presunte (indice, parte, immagine, formula)")
    for i in picks:
        has = i in images
        size = ""
        if has:
            with Image.open(io.BytesIO(zf.read(images[i]))) as im:
                size = f"{im.width}x{im.height} {im.mode}"
                if saved < args.save:
                    im.save(args.out / f"train-{i:07d}-{part_of(i)}.png")
                    saved += 1
        print(f"{i:7d} {part_of(i):8s} {size or 'nessuna':14s} {labels[i][:140]!r}")

    # 3. dimensioni
    idx = rng.sample(sorted(images), min(args.sample, len(images)))
    by_part: dict[str, list] = {}
    modes = Counter()
    for i in idx:
        with Image.open(io.BytesIO(zf.read(images[i]))) as im:
            by_part.setdefault(part_of(i), []).append((im.width, im.height))
            modes[im.mode] += 1
    print(f"\n-- dimensioni su {len(idx)} immagini; modi {dict(modes)}")
    for part, sizes in sorted(by_part.items()):
        print(f"{part}: {len(sizes)}; larghezza {describe([w for w, _ in sizes])}; altezza {describe([h for _, h in sizes])}")
        print(f"   altezze più frequenti {Counter(h for _, h in sizes).most_common(6)}")

    # 4. formule
    sample_labels = [labels[i] for i in idx]
    chars_raw = [len(s) for s in sample_labels]
    natural = [despace(s) for s in sample_labels]
    tok_raw = [len(tok(s, add_special_tokens=False)["input_ids"]) for s in sample_labels[:1500]]
    tok_nat = [len(tok(s, add_special_tokens=False)["input_ids"]) for s in natural[:1500]]
    kept = sum(latex_tokens(a) == latex_tokens(b) for a, b in zip(sample_labels, natural))
    print(f"\n-- formule: caratteri spaziati {describe(chars_raw)}; caratteri dopo despace {describe([len(s) for s in natural])}")
    print(f"token di Violetto: spaziati {describe(tok_raw)}; dopo despace {describe(tok_nat)}")
    print(f"despace conserva i token LaTeX: {kept}/{len(natural)}")
    envs = Counter(e for s in sample_labels for e in set(s.split()) if e.startswith("\\begin{") or e.startswith("\\begin"))
    print(f"ambienti \\begin più frequenti: {envs.most_common(6)}; con array {sum('array' in s for s in sample_labels)}")
    for i in idx[:8]:
        print(f"[{i} {part_of(i)}] {labels[i][:160]!r}\n    -> {despace(labels[i])[:160]!r}")

    # 5. test
    tz = zipfile.ZipFile(args.test_zip)
    print("\n-- test")
    for subset in ("spe", "cpe", "sce", "hwe"):
        lines = tz.read(f"UniMER-Test/{subset}.txt").decode("utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        imgs = sorted(int(PurePosixPath(i.filename).stem) for i in tz.infolist()
                      if PurePosixPath(i.filename).parent.as_posix() == f"UniMER-Test/{subset}"
                      and i.filename.endswith(".png"))
        sizes = []
        for j in imgs[:300]:
            with Image.open(io.BytesIO(tz.read(f"UniMER-Test/{subset}/{j:07d}.png"))) as im:
                sizes.append((im.width, im.height))
        for j in imgs[:2]:
            if saved < args.save:
                with Image.open(io.BytesIO(tz.read(f"UniMER-Test/{subset}/{j:07d}.png"))) as im:
                    im.save(args.out / f"test-{subset}-{j:07d}.png")
                saved += 1
        print(f"{subset}: {len(lines)} righe, {len(imgs)} immagini, indici {imgs[0]}-{imgs[-1]}, oltre le righe "
              f"{sum(j >= len(lines) for j in imgs)}; larghezza {describe([w for w, _ in sizes])}; "
              f"altezza {describe([h for _, h in sizes])}; esempio [{imgs[0]}] {lines[imgs[0]][:100]!r}")

    # 6. token immagine con l'ingrandimento proposto
    tokens, not_idem = [], 0
    for sizes in by_part.values():
        for w, h in sizes:
            hh, ww = final_size(h, w, args.scale, args.max_patches)
            exact = (hh // PATCH_SIZE) * (ww // PATCH_SIZE)
            if target_size(hh, ww, exact) != (hh, ww):
                not_idem += 1
            tokens.append(exact // 4)
    print(f"\n-- ingrandimento fino a {args.scale}x, al più {args.max_patches} patch: token immagine {describe(tokens)}; "
          f"dimensioni non idempotenti {not_idem}/{len(tokens)}")
    print(f"\nimmagini salvate: {saved} in {args.out}")


if __name__ == "__main__":
    main()
