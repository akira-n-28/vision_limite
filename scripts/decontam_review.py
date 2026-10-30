#!/usr/bin/env python3
"""Revisione delle esclusioni di scripts/decontam.py: quante corrispondenze sono duplicati veri.

Da eseguire in un job CPU (slurm/decontam_review.sbatch), mai sul frontend. Non cambia le liste di esclusione:
misura e scrive esempi, per scegliere le soglie.

- Immagini: per ogni campione escluso per immagine ricalcola il pHash, ritrova le immagini dei benchmark entro il
  raggio e ne misura la somiglianza dei pixel: correlazione su 64x64 in scala di grigi (trasparenza su bianco) e
  rapporto d'aspetto. Il pHash da solo confonde le immagini quasi vuote (formule, grafici su fondo bianco).
- Testi: per le esclusioni solo testo, la copertura degli n-gram (senza le formule di rito) della domanda del
  benchmark e di quella di training; vale la minore, cioè le due domande devono coincidere in entrambe le
  direzioni ("What is the area of the shaded part?" dentro un problema lungo non basta).
- Uscita in --out/<sorgente>: review.jsonl (un campione per riga), stats.json (fasce e conteggi per soglia),
  pairs_<fascia>.png (coppie training | benchmark), text_examples.txt ed exclude_proposed.txt (le chiavi che la
  regola --propose-* escluderebbe; le liste di decontam.py restano com'erano).
- Convenzione delle immagini: quella dichiarata dall'indice, oppure --convention per un indice che non la dichiara.
  Ogni scansione deve avere la stessa, altrimenti la revisione si ferma. pHash e somiglianza normalizzano le immagini
  con processing.to_rgb (limite_vl.decontam), come l'indice e la scansione.
- Immagini dei benchmark:
  - con un indice di scripts/decontam.py index, dalle cartelle --benchmark nello stesso ordine dell'indice;
  - con un indice del pannello (voci con `item` e `pos`, scripts/panel_index.py), dagli snapshot fissati in
    panel_index.py. Ogni voce ritrova il proprio elemento e la propria posizione con lo stesso sha256 dei pixel; un
    mancato abbinamento ferma la revisione, senza ripieghi.

    srun $RUN_PYTHON -u -B scripts/decontam_review.py --index INDEX.json --benchmark mathvista=DIR ... \\
        --scan data/decontam/p6/mmfinereason ... --out outputs/decontam_review [--convention grigio-diretto/0]
    srun $RUN_PYTHON -u -B scripts/decontam_review.py --index data/decontam/pannello-v3/index.json \\
        --scan DIR ... --out DIR
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from decontam import ID_COLUMNS, IMAGE_COLUMNS, LooseFiles, image_from, texts_of  # noqa: E402
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.decontam import (  # noqa: E402
    CONVENTIONS, HashIndex, TextIndex, image_phash, index_convention, normalize_words, word_ngrams)
from limite_vl.processing import to_rgb  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SIM_BINS = [(0.95, 1.01), (0.9, 0.95), (0.8, 0.9), (0.5, 0.8), (-1.01, 0.5)]
COV_BINS = [(0.8, 1.01), (0.5, 0.8), (0.3, 0.5), (0.0, 0.3)]


def on_white(im: Image.Image) -> Image.Image:
    """Scala di grigi dopo processing.to_rgb (trasparenze su bianco), la stessa normalizzazione dell'indice."""
    return to_rgb(im).convert("L")


def small(data: bytes) -> tuple[np.ndarray, float] | None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            g = on_white(im)
            aspect = g.width / max(1, g.height)
            return np.asarray(g.resize((64, 64), Image.BILINEAR), dtype=np.float32), aspect
    except Exception:  # noqa: BLE001 - immagine illeggibile
        return None


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if d == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float((a * b).sum() / d)


def train_task(task: tuple[str, bytes, str]):
    key, data, convention = task
    try:
        with Image.open(io.BytesIO(data)) as im:
            h = image_phash(im, convention)
    except Exception:  # noqa: BLE001
        return key, None, None
    return key, h, small(data)


def is_panel_index(index: list[dict]) -> bool:
    """Indice di scripts/panel_index.py: voci con `item` e `pos` (una per immagine)."""
    return bool(index) and all("item" in it and "pos" in it for it in index)


def bench_images(specs: list[str], index: list[dict], convention: str, workers: int = 1) -> list[bytes | None]:
    """Byte delle immagini dei benchmark nello stesso ordine dell'indice. Indice del pannello: ogni voce si ritrova
    con i lettori di panel_index.py (elemento e posizione esatti, stesso sha256 dei pixel); un errore ferma tutto.
    Indice di decontam.py index: stesso ciclo di decontam.cmd_index sulle cartelle --benchmark."""
    if is_panel_index(index):
        import panel_index

        out, errors = panel_index.resolve_entries(index, convention, workers)
        if errors:
            sys.exit(f"{len(errors)} voci dell'indice non ritrovate esattamente, per esempio: {errors[:5]}")
        return out
    if not specs:
        sys.exit("indice di decontam.py index: servono le cartelle --benchmark")
    out: list[bytes | None] = []
    for spec in specs:
        name, _, folder = spec.partition("=")
        folder = Path(folder)
        loose = LooseFiles(folder)
        for f in sorted(folder.rglob("*.parquet")):
            table = pq.read_table(f)
            cols = table.column_names
            image_cols = [c for c in IMAGE_COLUMNS if c in cols]
            id_col = next((c for c in ID_COLUMNS if c in cols), None)
            for i, row in enumerate(table.to_pylist()):
                it = index[len(out)]
                rid = str(row[id_col]) if id_col else f"{f.name}:{i}"
                if (it["benchmark"], it["file"], it["id"]) != (name, f.name, rid):
                    sys.exit(f"ordine diverso dall'indice in {name}/{f.name} riga {i}: {it}")
                out.append(next((d for d in (image_from(row[c], loose) for c in image_cols) if d), None))
    if len(out) != len(index):
        sys.exit(f"benchmark: {len(out)} righe, indice {len(index)}")
    return out


def thumb(data: bytes, h: int = 120, w: int = 240) -> Image.Image:
    with Image.open(io.BytesIO(data)) as im:
        g = on_white(im).convert("RGB")
    g.thumbnail((w, h))
    return g


def pairs_sheet(rows: list[tuple[str, bytes, bytes]], path: Path) -> None:
    sheet = Image.new("RGB", (500, 150 * max(1, len(rows))), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (label, a, b) in enumerate(rows):
        y = 150 * r
        for x, data in ((0, a), (250, b)):
            try:
                sheet.paste(thumb(data), (x, y + 22))
            except Exception:  # noqa: BLE001
                draw.text((x + 4, y + 60), "illeggibile", fill="red")
        draw.text((4, y + 4), label[:80], fill="black")
        draw.line([(0, y + 149), (500, y + 149)], fill="gray")
    sheet.save(path)


def proposed_rule(r: dict, args) -> bool:
    im, tx = r.get("image") or {}, r.get("text") or {}
    img_ok = im.get("sim") is not None and im["sim"] >= args.propose_sim and (im.get("aspect_ratio") or 9) <= 1.2
    return img_ok or (tx.get("coverage", 0) >= args.propose_cov and tx.get("words", 0) >= args.propose_words)


def bins_of(value: float, bins) -> str:
    return next(f"{lo:.2f}-{hi:.2f}" for lo, hi in bins if lo <= value < hi)


def review_scan(scan: Path, index, with_hash, hidx, tidx, bench_small, bench_bytes, bench_grams, args, pool, out: Path,
                convention: str):
    t0 = time.time()
    summary = json.loads((scan / "summary.json").read_text())
    scan_convention = summary.get("convenzione_immagini") or args.convention
    if scan_convention != convention:
        sys.exit(f"{scan}: convenzione della scansione {scan_convention!r} diversa da quella dell'indice {convention!r}")
    kind = summary["kind"]
    image_keys, text_keys = set(), set()
    for line in open(scan / "matches.jsonl"):
        m = json.loads(line)
        (image_keys if m["match"] == "image" else text_keys).add(m["key"])
    text_keys -= image_keys
    out.mkdir(parents=True, exist_ok=True)
    rows, train_bytes = {}, {}
    tasks = []

    def flush():
        for key, h, sm in pool.imap(train_task, tasks, chunksize=32):
            best = {"sim": None, "distance": None, "benchmark": None, "bench_id": None, "aspect_ratio": None}
            if h is not None and sm is not None:
                for j, d in hidx.query(h)[: args.max_candidates]:
                    pos = with_hash[j]
                    bs = bench_small[pos]
                    if bs is None:
                        continue
                    s = corr(sm[0], bs[0])
                    ratio = max(sm[1], bs[1]) / max(1e-6, min(sm[1], bs[1]))
                    if best["sim"] is None or s > best["sim"]:
                        best = {"sim": round(s, 4), "distance": d, "benchmark": index[pos]["benchmark"],
                                "bench_id": index[pos]["id"], "aspect_ratio": round(ratio, 3), "pos": pos}
            rows[key]["image"] = best
        tasks.clear()

    for shard in summary["shards"]:
        for key, files in iter_shard(shard):
            if key not in image_keys and key not in text_keys:
                continue
            meta = json.loads(files["json"]) if "json" in files else {}
            rows[key] = {"key": key, "match": "image" if key in image_keys else "text_only"}
            texts = [t for t in texts_of(kind, meta) if t]
            if texts:
                best = {"coverage": 0.0}
                for t in texts:
                    words = normalize_words(t)
                    tg = word_ngrams(words, tidx.n)
                    for j in tidx.query(t):
                        bg = bench_grams[j]
                        shared = len(tg & bg)
                        cov_b, cov_t = shared / max(1, len(bg)), shared / max(1, len(tg))
                        cov = min(cov_b, cov_t)  # in entrambe le direzioni: una domanda corta dentro una lunga non basta
                        if cov > best["coverage"]:
                            best = {"coverage": round(cov, 3), "cov_bench": round(cov_b, 3), "cov_train": round(cov_t, 3),
                                    "words": len(words), "benchmark": index[j]["benchmark"],
                                    "bench_id": index[j]["id"], "train_text": t[:400], "bench_text": index[j]["text"][:400]}
                rows[key]["text"] = best
            if key in image_keys:
                ext = next((e for e in ("png", "jpg", "jpeg", "webp") if e in files), None)
                if ext:
                    train_bytes[key] = files[ext]
                    tasks.append((key, files[ext], convention))
                    if len(tasks) >= 2048:
                        flush()
        flush()
    # statistiche
    stats = {"kind": kind, "image_keys": len(image_keys), "text_only_keys": len(text_keys), "found": len(rows),
             "image_bins": Counter(), "image_bins_by_benchmark": defaultdict(Counter), "text_bins": Counter(),
             "text_bins_by_benchmark": defaultdict(Counter), "by_threshold": {}}
    for r in rows.values():
        im = r.get("image")
        if r["match"] == "image" and im:
            b = bins_of(im["sim"], SIM_BINS) if im["sim"] is not None else "nessun candidato"
            stats["image_bins"][b] += 1
            stats["image_bins_by_benchmark"][im["benchmark"] or "-"][b] += 1
        tx = r.get("text")
        if r["match"] == "text_only" and tx:
            b = bins_of(tx["coverage"], COV_BINS)
            stats["text_bins"][b] += 1
            stats["text_bins_by_benchmark"][tx.get("benchmark") or "-"][b] += 1
    for sim_t in (0.9, 0.95):
        for cov_t, words_t in ((0.5, 0), (0.8, 0), (0.8, 15), (0.8, 25), (1.01, 0)):
            n = n_img = 0
            for r in rows.values():
                im, tx = r.get("image") or {}, r.get("text") or {}
                img_ok = im.get("sim") is not None and im["sim"] >= sim_t and (im.get("aspect_ratio") or 9) <= 1.2
                txt_ok = tx.get("coverage", 0) >= cov_t and tx.get("words", 0) >= words_t
                n += img_ok or txt_ok
                n_img += img_ok
            stats["by_threshold"][f"sim>={sim_t},cov>={cov_t},words>={words_t}"] = {"total": n, "image": n_img}
    lengths = Counter()
    for r in rows.values():
        tx = r.get("text") or {}
        if r["match"] == "text_only" and tx.get("coverage", 0) >= 0.8:
            lengths["<15" if tx["words"] < 15 else "15-24" if tx["words"] < 25 else ">=25"] += 1
    stats["text_cov08_by_words"] = lengths
    proposed = sorted(r["key"] for r in rows.values() if proposed_rule(r, args))
    (out / "exclude_proposed.txt").write_text("".join(f"{k}\n" for k in proposed))
    stats["proposed"] = {"rule": f"sim>={args.propose_sim} e aspetto<=1.2, oppure copertura>={args.propose_cov} in "
                                 f"entrambe le direzioni con almeno {args.propose_words} parole", "excluded": len(proposed)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=dict))
    with open(out / "review.jsonl", "w") as fh:
        for r in rows.values():
            r2 = dict(r)
            if r2.get("text"):
                r2["text"] = {k: v for k, v in r2["text"].items() if k not in ("train_text", "bench_text")}
            if r2.get("image"):
                r2["image"] = {k: v for k, v in r2["image"].items() if k != "pos"}
            fh.write(json.dumps(r2, ensure_ascii=False) + "\n")
    # esempi
    rng = random.Random(0)
    by_bin = defaultdict(list)
    for r in rows.values():
        im = r.get("image")
        if r["match"] == "image" and im and im["sim"] is not None:
            by_bin[bins_of(im["sim"], SIM_BINS)].append(r)
    for lo, hi in SIM_BINS:
        b = f"{lo:.2f}-{hi:.2f}"
        sheet = []
        for r in rng.sample(by_bin[b], min(args.examples, len(by_bin[b]))):
            im = r["image"]
            label = (f"sim {im['sim']:.3f} d={im['distance']} asp {im['aspect_ratio']} {im['benchmark']}/{im['bench_id']} "
                     f"{r['key']}")
            bb = bench_bytes[im["pos"]]
            if bb:
                sheet.append((label, train_bytes[r["key"]], bb))
        if sheet:
            pairs_sheet(sheet, out / f"pairs_{b}.png")
    with open(out / "text_examples.txt", "w") as fh:
        tb = defaultdict(list)
        for r in rows.values():
            if r["match"] == "text_only" and r.get("text") and "train_text" in r["text"]:
                tb[bins_of(r["text"]["coverage"], COV_BINS)].append(r)
        for lo, hi in COV_BINS:
            b = f"{lo:.2f}-{hi:.2f}"
            fh.write(f"===== copertura {b}: {len(tb[b])} campioni\n")
            for r in rng.sample(tb[b], min(args.examples, len(tb[b]))):
                t = r["text"]
                fh.write(f"--- {r['key']} -> {t['benchmark']}/{t['bench_id']} copertura {t['coverage']} (benchmark "
                         f"{t['cov_bench']}, training {t['cov_train']}) parole {t['words']}\n"
                         f"TRAIN: {t['train_text']}\nBENCH: {t['bench_text']}\n")
    print(f"{scan.name}: {json.dumps({k: stats[k] for k in ('image_keys', 'text_only_keys', 'image_bins', 'text_bins')}, default=dict)}"
          f" ({time.time() - t0:.0f} s)", flush=True)
    print(f"  per soglia: {stats['by_threshold']}", flush=True)
    print(f"  proposta: {stats['proposed']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--benchmark", action="append", default=[],
                    help="nome=cartella, come per decontam.py index (non serve con un indice del pannello)")
    ap.add_argument("--convention", choices=CONVENTIONS, default=None,
                    help="convenzione delle immagini di un indice che non la dichiara (p6, f1eval: grigio-diretto/0)")
    ap.add_argument("--scan", action="append", required=True, type=Path, help="cartella di uscita di decontam.py scan")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--ngram", type=int, default=10)
    ap.add_argument("--max-df", type=int, default=25)
    ap.add_argument("--max-candidates", type=int, default=50)
    ap.add_argument("--examples", type=int, default=10)
    ap.add_argument("--propose-sim", type=float, default=0.9)
    ap.add_argument("--propose-cov", type=float, default=0.8)
    ap.add_argument("--propose-words", type=int, default=15)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("decontam_review.py va eseguito in un job SLURM")
    t0 = time.time()
    loaded = json.loads(Path(args.index).read_text())
    try:
        convention = index_convention(loaded, args.convention)
    except ValueError as e:
        sys.exit(f"{args.index}: {e}")
    index = loaded["items"]
    with_hash = [i for i, it in enumerate(index) if it["phash"]]
    hidx = HashIndex([int(index[i]["phash"], 16) for i in with_hash], radius=args.radius)
    tidx = TextIndex.build([it["text"] for it in index], n=args.ngram, max_df=args.max_df)
    bench_grams = [word_ngrams(normalize_words(it["text"]), args.ngram) & tidx.grams.keys() for it in index]
    bench_bytes = bench_images(args.benchmark, index, convention, args.workers)
    with Pool(args.workers) as pool:
        bench_small = pool.map(small, [b or b"" for b in bench_bytes], chunksize=64)
        print(f"benchmark: {len(index)} voci, convenzione {convention}, {sum(s is not None for s in bench_small)} "
              f"immagini ridotte ({time.time() - t0:.0f} s)", flush=True)
        for scan in args.scan:
            review_scan(scan, index, with_hash, hidx, tidx, bench_small, bench_bytes, bench_grams, args, pool,
                        args.out / scan.name, convention)


if __name__ == "__main__":
    main()
