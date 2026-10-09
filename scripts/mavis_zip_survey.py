#!/usr/bin/env python3
"""Analisi di RuleBaseGeo.zip (CaraJ/MAVIS-Geometry) senza estrarlo, per progettare gli shard di F1.

Da eseguire in un job CPU (slurm/mavis_zip_survey.sbatch), mai sul frontend. Ogni campione è una cartella
RuleBaseGeo/depthK/<i>/ con <n>_annotation.json, <n>_caption.jpg, <n>_text.jpg, <n>_vision_dominant.jpg.

1. conteggio dei campioni per depthK e dei campioni incompleti o con file inattesi;
2. immagini identiche fra le varianti (CRC32 e dimensione dal catalogo dello zip, senza leggere i dati);
3. tutte le annotazioni: chiavi, lunghezza delle didascalie, didascalie vuote o duplicate, scelte;
4. corrispondenza con le varianti del parquet di esempio (stessi byte, cercando la riga per didascalia);
5. su un campione di immagini caption: dimensioni, token a 1024 e 2304 patch, e tempo e byte della
   conversione per gli shard (decodifica, ridimensionamento a 2304 patch, PNG e JPEG).

Uscite piccole e a tetto fisso: al massimo 3 immagini in --out.

    srun $RUN_PYTHON -B scripts/mavis_zip_survey.py RuleBaseGeo.zip --parquet visualization_small.parquet --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.processing import DEFAULT_MAX_PATCHES, DIAGRAM_MAX_PATCHES, PATCH_SIZE, target_size, to_rgb  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
KINDS = ("annotation.json", "caption.jpg", "text.jpg", "vision_dominant.jpg")
IMAGE_KINDS = KINDS[1:]
NAME_RE = re.compile(r"^(\d+)_(" + "|".join(re.escape(k) for k in KINDS) + r")$")
PARQUET_VARIANTS = ("text_dominant", "text_lite", "vision_dominant")


def describe(values: list[float], fmt: str = ".0f") -> str:
    if not values:
        return "-"
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return (f"media {statistics.mean(values):{fmt}}, mediana {pick(0.5):{fmt}}, p10 {pick(0.1):{fmt}}, "
            f"p90 {pick(0.9):{fmt}}, p99 {pick(0.99):{fmt}}, min {values[0]:{fmt}}, max {values[-1]:{fmt}}")


def tokens(height: int, width: int, max_patches: int) -> int:
    h, w = target_size(height, width, max_patches)
    return (h // PATCH_SIZE // 2) * (w // PATCH_SIZE // 2)


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--parquet", type=Path, required=True, help="visualization_small.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size-sample", type=int, default=2000, help="immagini caption da misurare")
    ap.add_argument("--convert-sample", type=int, default=200, help="immagini da convertire per la stima")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    zf = zipfile.ZipFile(args.archive)
    infos = [i for i in zf.infolist() if not i.is_dir()]
    log(f"{args.archive}: {len(infos)} file, catalogo letto in {time.time() - t0:.0f} s")

    # 1. campioni
    samples: dict[str, dict[str, zipfile.ZipInfo]] = defaultdict(dict)
    unexpected: Counter = Counter()
    for info in infos:
        path = PurePosixPath(info.filename)
        m = NAME_RE.match(path.name)
        if not m:
            unexpected[re.sub(r"\d+", "#", info.filename)] += 1
            continue
        samples[str(path.parent)][m.group(2)] = info
    by_depth = Counter(PurePosixPath(k).parts[1] if len(PurePosixPath(k).parts) > 1 else "." for k in samples)
    incomplete = {k: sorted(set(KINDS) - set(v)) for k, v in samples.items() if set(v) != set(KINDS)}
    log(f"\n-- 1. campioni: {len(samples)}; per cartella {dict(sorted(by_depth.items()))}")
    log(f"campioni incompleti: {len(incomplete)}; esempi {list(incomplete.items())[:5]}")
    log(f"file con nomi inattesi: {sum(unexpected.values())}; schemi {dict(unexpected.most_common(10))}")

    # 2. varianti identiche (CRC32 + dimensione)
    def same(a: zipfile.ZipInfo, b: zipfile.ZipInfo) -> bool:
        return a.CRC == b.CRC and a.file_size == b.file_size

    pairs = Counter()
    complete = [v for v in samples.values() if set(v) == set(KINDS)]
    for v in complete:
        c, t, d = v["caption.jpg"], v["text.jpg"], v["vision_dominant.jpg"]
        pairs["caption=vision_dominant"] += same(c, d)
        pairs["caption=text"] += same(c, t)
        pairs["text=vision_dominant"] += same(t, d)
        pairs["tutte e tre uguali"] += same(c, d) and same(c, t)
    log(f"\n-- 2. immagini identiche su {len(complete)} campioni completi: {dict(pairs)}")
    sizes = {k: [v[k].file_size / 1024 for v in complete] for k in IMAGE_KINDS}
    for k, values in sizes.items():
        log(f"{k}: KiB {describe(values)}")

    # 3. annotazioni, in ordine di posizione nello zip (lettura sequenziale)
    parquet = pq.read_table(args.parquet)
    p_text = parquet.column("text_en").to_pylist()
    wanted = {row["caption"]: i for i, row in enumerate(p_text)}
    found: dict[int, list[str]] = defaultdict(list)
    top_keys, en_keys, n_choices, answer = Counter(), Counter(), Counter(), Counter()
    cap_chars, cap_hashes, empty, bad = [], Counter(), 0, 0
    examples = []
    ann = sorted(((v["annotation.json"], k) for k, v in samples.items() if "annotation.json" in v),
                 key=lambda x: x[0].header_offset)
    t1 = time.time()
    for n, (info, key) in enumerate(ann):
        try:
            data = json.loads(zf.read(info))
        except (ValueError, UnicodeDecodeError):
            bad += 1
            continue
        top_keys.update(data.keys())
        en = data.get("text_en") or {}
        en_keys.update(en.keys())
        caption = en.get("caption") or ""
        if not caption.strip():
            empty += 1
        cap_chars.append(len(caption))
        cap_hashes[hashlib.blake2b(caption.encode(), digest_size=8).digest()] += 1
        n_choices[len(data.get("choices") or [])] += 1
        answer[data.get("answer_index")] += 1
        if caption in wanted:
            found[wanted[caption]].append(key)
        if len(examples) < 2 and n % 50000 == 0:
            examples.append((key, data))
        if n and n % 50000 == 0:
            log(f"   {n} annotazioni in {time.time() - t1:.0f} s")
    dup = sum(c - 1 for c in cap_hashes.values() if c > 1)
    log(f"\n-- 3. annotazioni lette: {len(ann) - bad} in {time.time() - t1:.0f} s (illeggibili {bad})")
    log(f"chiavi: {dict(top_keys)}; text_en: {dict(en_keys)}")
    log(f"didascalie: caratteri {describe(cap_chars)}; vuote {empty}; distinte {len(cap_hashes)}, "
        f"duplicati {dup}")
    log(f"numero di scelte {dict(n_choices)}; answer_index {dict(answer)}")
    for key, data in examples:
        text = json.dumps({k: v for k, v in data.items() if k != "text_zh"}, ensure_ascii=False)
        log(f"[{key}] {text[:1500]}")

    # 4. corrispondenza con il parquet (byte identici)
    log(f"\n-- 4. righe del parquet ritrovate per didascalia: {len(found)}/{parquet.num_rows}")
    match = Counter()
    for i, keys in sorted(found.items()):
        v = samples[keys[0]]
        z = {k: hashlib.sha256(zf.read(v[k])).hexdigest() for k in IMAGE_KINDS}
        for pv in PARQUET_VARIANTS:
            digest = hashlib.sha256(parquet.column(f"image_{pv}")[i].as_py()["bytes"]).hexdigest()
            hits = [k for k in IMAGE_KINDS if z[k] == digest] or ["nessuna"]
            match[(pv, "|".join(hits))] += 1
        if i < 3:
            log(f"riga {i} -> {keys}")
    for (pv, hits), count in sorted(match.items()):
        log(f"parquet {pv} = zip {hits}: {count} righe")
    if found:  # una riga per guardarla a occhio
        i, keys = min(found.items())
        for k in IMAGE_KINDS:
            with Image.open(io.BytesIO(zf.read(samples[keys[0]][k]))) as im:
                im = to_rgb(im)
                h, w = target_size(im.height, im.width, DEFAULT_MAX_PATCHES)
                im.resize((w, h), Image.BILINEAR).save(args.out / f"riga{i:02d}_{k.replace('.jpg', '')}_1024.png")

    # 5. dimensioni e costo di conversione sulle immagini caption
    step = max(1, len(complete) // args.size_sample)
    chosen = complete[::step][: args.size_sample]
    dims, tok_d, tok_g, aspect = [], [], [], []
    conv_t, png_b, jpg_b = [], [], []
    for n, v in enumerate(chosen):
        raw = zf.read(v["caption.jpg"])
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
            dims.append(max(w, h))
            aspect.append(max(w, h) / min(w, h))
            tok_d.append(tokens(h, w, DEFAULT_MAX_PATCHES))
            tok_g.append(tokens(h, w, DIAGRAM_MAX_PATCHES))
            if n < args.convert_sample:
                start = time.perf_counter()
                th, tw = target_size(h, w, DIAGRAM_MAX_PATCHES)
                im.draft("RGB", (tw, th))  # decodifica JPEG ridotta per DCT, almeno alla dimensione finale
                small = to_rgb(im).resize((tw, th), Image.BILINEAR)
                buf = io.BytesIO()
                small.save(buf, format="PNG")
                conv_t.append(time.perf_counter() - start)
                png_b.append(buf.tell() / 1024)
                buf = io.BytesIO()
                small.save(buf, format="JPEG", quality=95)
                jpg_b.append(buf.tell() / 1024)
    log(f"\n-- 5. {len(chosen)} immagini caption (una ogni {step})")
    log(f"lato lungo px: {describe(dims)}")
    log(f"rapporto d'aspetto: {describe(aspect, '.2f')}; oltre 3: {sum(a > 3 for a in aspect)}, "
        f"oltre 6: {sum(a > 6 for a in aspect)}")
    log(f"token a 1024 patch: {describe(tok_d)}")
    log(f"token a 2304 patch: {describe(tok_g)}")
    log(f"conversione su {len(conv_t)}: s per immagine {describe(conv_t, '.3f')}")
    log(f"PNG a 2304 patch, KiB: {describe(png_b)}; JPEG q95, KiB: {describe(jpg_b)}")
    if conv_t:
        n_img = len(complete)
        log(f"stima per {n_img} immagini caption: {statistics.mean(conv_t) * n_img / 3600:.1f} ore-CPU, "
            f"PNG {statistics.mean(png_b) * n_img / 2**20:.1f} GiB, JPEG {statistics.mean(jpg_b) * n_img / 2**20:.1f} GiB")
    log(f"\nfine in {time.time() - t0:.0f} s; immagini in {args.out}")


if __name__ == "__main__":
    main()
