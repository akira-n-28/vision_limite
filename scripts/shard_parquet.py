#!/usr/bin/env python3
"""Converte dataset in parquet con immagini incluse (MMFineReason, PixMo-Cap) in shard tar per il training.

Da eseguire SOLO in un job SLURM CPU (slurm/shard_parquet.sbatch). Legge i parquet a blocchi di righe
(memoria limitata), ridimensiona le immagini una sola volta nel pool di processi e scrive tar in stile
webdataset con `<id>.<png|jpg>` e `<id>.json`. Nel JSON max_patches = patch esatte dell'immagine salvata:
il processor non la ridimensiona più.

Tipi (--kind):
- mmfinereason (OpenDataArena/MMFineReason-1.8M, job 29460): immagini STEM piccole (100-800 px) salvate
  com'erano (PNG o JPEG, byte originali): ingrandite già salvate occupavano circa 108 KB a campione, cioè
  circa 195 GB per tutto il dataset (job 29465). Il caricatore (data.stem_caption_sample) le porta a
  capped_size (al più 2x e al più 2304 patch); nel JSON max_patches è già quello finale. JSON: id,
  sorgente, split, domanda (senza il segnaposto <image>), risposta verificata, descrizione di
  Qwen3-VL-235B (caption), pass_rate di Qwen3-VL-4B. Non si tengono il ragionamento di Qwen né la
  risposta originale: le soluzioni di F2 le scrive Violetto. Esclusi solo righe senza immagine o senza
  descrizione; domanda e risposta mancanti restano (servono solo a F2, che le filtra).
- pixmo (anthracite-org/pixmo-cap-images): foto, portate a target_size con 1024 patch come in NaFlex,
  JPEG qualità 95. JSON: id (indice globale), didascalia, URL d'origine. Esclusi: righe senza immagine o
  senza didascalia.

Ordine: quello dei file e delle righe. Ripresa: piano deterministico per file; uno shard registrato in
shards.jsonl e presente con la dimensione registrata viene saltato; ogni tar si scrive in `.tmp`, si
rilegge e si rinomina. Uno shard corrisponde a --files-per-shard file parquet, così il piano non richiede
di leggere le righe in anticipo.

    srun $RUN_PYTHON -u -B scripts/shard_parquet.py --kind mmfinereason --files P1 P2 ... --out DIR --source R@C
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shard_llava_pretrain import HashingWriter, add_member  # noqa: E402  (stessa cartella scripts/)

from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.processing import (DEFAULT_MAX_PATCHES, DIAGRAM_MAX_PATCHES, capped_size, exact_patches,  # noqa: E402
                                  target_size, to_rgb)

Image.MAX_IMAGE_PIXELS = None
BATCH_ROWS = 512
KINDS = {
    "mmfinereason": {"columns": ["id", "source", "question", "answer", "caption", "pass_rate", "image"],
                     "ext": None},  # estensione dal formato originale (png o jpg)
    "pixmo": {"columns": ["image", "caption", "image_url"], "ext": "jpg"},
}


def convert(task: tuple[str, str, bytes]):
    """Immagine originale -> immagine finale nel formato del tipo. Gira nei processi del pool."""
    kind, key, raw = task
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()  # decodifica completa: un'immagine troncata finisce fra le illeggibili
            width, height, fmt = im.width, im.height, im.format
            image = to_rgb(im) if kind == "pixmo" else None
        if kind == "mmfinereason":
            if fmt not in ("PNG", "JPEG"):
                buf = io.BytesIO()
                with Image.open(io.BytesIO(raw)) as im:
                    to_rgb(im).save(buf, format="PNG")
                data, ext = buf.getvalue(), "png"
            else:
                data, ext = raw, "png" if fmt == "PNG" else "jpg"
            h, w = capped_size(height, width, 2.0, DIAGRAM_MAX_PATCHES)
        else:
            h, w = target_size(height, width, DEFAULT_MAX_PATCHES)
            if (w, h) != image.size:
                image = image.resize((w, h), Image.BILINEAR)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=95)
            data, ext = buf.getvalue(), "jpg"
    except Exception as exc:  # noqa: BLE001 - immagine illeggibile: fuori e registrata
        return key, None, f"{type(exc).__name__}: {exc}"[:200], None
    patches = exact_patches(h, w)
    return key, (data, ext), None, (width, height, w, h, patches, target_size(h, w, patches) == (h, w))


def rows_of(kind: str, path: Path, file_index: int):
    """(chiave, byte dell'immagine, metadati) per ogni riga utile del file; esclusi con il motivo."""
    pf = pq.ParquetFile(path)
    split = path.name.split("-")[0]
    row = 0
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=KINDS[kind]["columns"]):
        for rec in batch.to_pylist():
            index = row
            row += 1
            image = (rec.get("image") or {}).get("bytes")
            if kind == "mmfinereason":
                # gli id si ripetono fra sorgenti (7963 distinti su 8000 righe in rl-00000, job 29460)
                source_tag = re.sub(r"[^A-Za-z0-9]+", "_", rec.get("source") or "nosource").strip("_")
                key = f"{split}-{source_tag}-{rec['id']}"
                question = (rec.get("question") or "").replace("<image>", "").strip()
                missing = [n for n, v in (("image", image), ("caption", rec.get("caption"))) if not v]
                meta = {"id": key, "source_dataset": rec.get("source"), "split": split, "question": question,
                        "answer": (rec.get("answer") or "").strip(), "caption": (rec.get("caption") or "").strip(),
                        "pass_rate": rec.get("pass_rate")}
            else:
                key = f"pixmo-{file_index:05d}-{index:06d}"
                missing = [n for n, v in (("image", image), ("caption", rec.get("caption"))) if not v]
                meta = {"id": key, "caption": (rec.get("caption") or "").strip(), "image_url": rec.get("image_url")}
            yield key, image, meta, missing


def check_shard(path: Path, keys: list[str]) -> None:
    got = []
    for key, files in iter_shard(path):
        if len(files) != 2 or "json" not in files or not ({"png", "jpg"} & set(files)):
            raise RuntimeError(f"{path.name}: {key} ha {sorted(files)}")
        got.append(key)
    if got != keys:
        raise RuntimeError(f"{path.name}: {len(got)} campioni riletti invece di {len(keys)}")


def write_shard(path: Path, kind: str, files: list[tuple[int, Path]], pool: Pool, source: str) -> dict:
    t0 = time.time()
    tmp = path.with_name(path.name + ".tmp")
    written, bad, excluded, not_idem, tokens, seen = [], [], {}, [], [], set()
    with open(tmp, "wb") as fh:
        hw = HashingWriter(fh)
        with tarfile.open(fileobj=hw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for file_index, fpath in files:
                pending, metas = [], {}

                def flush():
                    for key, image_out, err, dims in pool.map(convert, pending, chunksize=16):
                        if err:
                            bad.append(f"{key}: {err}")
                            continue
                        width, height, w, h, patches, idem = dims
                        if not idem:
                            not_idem.append(key)
                        data, ext = image_out
                        meta = {**metas[key], "orig_width": width, "orig_height": height, "width": w, "height": h,
                                "max_patches": patches, "format": "PNG" if ext == "png" else "JPEG",
                                "stored_resized": kind == "pixmo", "source": source}
                        add_member(tar, f"{key}.{ext}", data)
                        add_member(tar, f"{key}.json", json.dumps(meta, ensure_ascii=False).encode())
                        written.append(key)
                        tokens.append(patches // 4)
                    pending.clear()
                    metas.clear()

                for key, image, meta, missing in rows_of(kind, fpath, file_index):
                    if missing:
                        excluded.setdefault("missing_" + "_".join(missing), []).append(key)
                        continue
                    if key in seen:
                        excluded.setdefault("duplicate_id", []).append(key)
                        continue
                    seen.add(key)
                    pending.append((kind, key, image))
                    metas[key] = meta
                    if len(pending) >= BATCH_ROWS * 2:
                        flush()
                flush()
        fh.flush()
        os.fsync(fh.fileno())
    check_shard(tmp, written)
    os.replace(tmp, path)
    return {"name": path.name, "files": [str(p) for _, p in files], "samples": len(written), "bytes": hw.n,
            "sha256": hw.sha.hexdigest(), "first": written[0] if written else None,
            "last": written[-1] if written else None, "image_tokens": sum(tokens), "unreadable": bad,
            "not_idempotent": not_idem, "excluded": {k: len(v) for k, v in excluded.items()},
            "excluded_examples": {k: v[:10] for k, v in excluded.items()}, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--files", required=True, nargs="+", type=Path, help="parquet, in ordine")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", required=True, help="repo@commit, scritto in ogni campione")
    ap.add_argument("--prefix", required=True, help="nome degli shard, es. mmfinereason-sft")
    ap.add_argument("--files-per-shard", type=int, default=4)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("shard_parquet.py va eseguito in un job SLURM")
    missing = [str(p) for p in args.files if not p.is_file()]
    if missing:
        sys.exit(f"file mancanti: {missing[:5]}")
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    indexed = list(enumerate(args.files))
    plan = [indexed[i:i + args.files_per_shard] for i in range(0, len(indexed), args.files_per_shard)]
    plan_doc = {"kind": args.kind, "source": args.source, "prefix": args.prefix, "files_per_shard": args.files_per_shard,
                "files": [p.resolve().name for p in args.files]}
    digest = hashlib.sha256(json.dumps(plan_doc, sort_keys=True).encode()).hexdigest()
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()).get("digest") != digest:
            sys.exit(f"piano diverso da quello in {plan_path}: cartella di un'altra esecuzione")
    else:
        plan_path.write_text(json.dumps({"digest": digest, "shards": len(plan), **plan_doc}, indent=1))
    log_path = args.out / "shards.jsonl"
    done = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            done[row["name"]] = row
    rows = []
    with Pool(args.workers) as pool:
        for idx, files in enumerate(plan):
            name = f"{args.prefix}-{idx:05d}.tar"
            path = args.out / name
            prev = done.get(name)
            if prev and path.is_file() and path.stat().st_size == prev["bytes"]:
                rows.append(prev)
                continue
            row = write_shard(path, args.kind, files, pool, args.source)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            rows.append(row)
            print(f"[{idx + 1}/{len(plan)}] {name}: {row['samples']} campioni, {row['bytes'] / 1e9:.2f} GB, token immagine "
                  f"medi {row['image_tokens'] / max(1, row['samples']):.0f}, esclusi {row['excluded']}, illeggibili "
                  f"{len(row['unreadable'])}, non idempotenti {len(row['not_idempotent'])}, {row['seconds']:.0f} s", flush=True)
    written = sum(r["samples"] for r in rows)
    excluded: dict = {}
    for r in rows:
        for k, v in r["excluded"].items():
            excluded[k] = excluded.get(k, 0) + v
    manifest = {**plan_doc, "job_id": os.environ.get("SLURM_JOB_ID"), "plan_digest": digest,
                "counts": {"written": written, "excluded": excluded,
                           "unreadable": sum(len(r["unreadable"]) for r in rows),
                           "not_idempotent": sum(len(r["not_idempotent"]) for r in rows),
                           "image_tokens_mean": sum(r["image_tokens"] for r in rows) / max(1, written)},
                "shards": [{k: r[k] for k in ("name", "samples", "bytes", "sha256", "first", "last", "image_tokens")}
                           for r in rows],
                "seconds": round(time.time() - t0, 1)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("counts", "seconds")}, indent=1), flush=True)
    if written == 0:
        sys.exit("nessun campione scritto")


if __name__ == "__main__":
    main()
