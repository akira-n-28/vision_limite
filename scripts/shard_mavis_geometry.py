#!/usr/bin/env python3
"""Converte RuleBaseGeo.zip di CaraJ/MAVIS-Geometry in shard tar da circa 1 GB.

Da eseguire SOLO in un job SLURM CPU (slurm/shard_mavis_geometry.sbatch): legge lo zip e scrive i tar
direttamente, senza estrarre gli ~800 mila file piccoli su GlusterFS.

Ogni campione dello zip è una cartella RuleBaseGeo/depthK/<i>/ con <n>_annotation.json e tre immagini:
<n>_caption.jpg, <n>_text.jpg e <n>_vision_dominant.jpg (job 29415). Per le didascalie di F1 serve
caption.jpg, la variante con tutte le misure citate nella didascalia (job 29410); --variant sceglie
un'altra variante per F2.

Shard in stile webdataset, per ogni campione `<id>.png` e `<id>.json` (id = d<K>-<i>):
- l'immagine è ridimensionata una sola volta alla dimensione esatta di target_size(..., --max-patches),
  multipla di 32, e salvata in PNG; con lo stesso max_patches il processor non la ridimensiona più
  (controllato su ogni immagine);
- il JSON ha id, cartella nello zip, profondità, variante, text_en completo, scelte, indice della
  risposta, dimensioni originali e salvate, max_patches e sorgente.
Esclusi ed elencati nel manifest: campioni senza tutti i file, didascalie duplicate (resta la prima
nell'ordine dello zip), immagini illeggibili.

Ordine: quello dello zip (il mescolamento è compito del loader). Il piano dipende solo dallo zip e dagli
argomenti: shard da --per-shard campioni. Ripresa come in shard_llava_pretrain.py: uno shard già
registrato in shards.jsonl e presente con la dimensione registrata viene saltato; ogni tar si scrive in
`.tmp` e si rinomina a fine shard, dopo averlo riletto. --limit N converte solo i primi N campioni del
piano (prova).

Uscita in --out: mavis-geometry-<variante>-NNNNN.tar, plan.json, shards.jsonl e manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import statistics
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path, PurePosixPath

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shard_llava_pretrain import HashingWriter, add_member  # noqa: E402  (stessa cartella scripts/)

from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.processing import DIAGRAM_MAX_PATCHES, UNIT, target_size, to_rgb  # noqa: E402

VARIANTS = ("caption", "text", "vision_dominant")
REQUIRED = {"annotation.json", "caption.jpg", "text.jpg", "vision_dominant.jpg"}
NAME_RE = re.compile(r"^(\d+)_(annotation\.json|caption\.jpg|text\.jpg|vision_dominant\.jpg)$")
BATCH = 256  # campioni letti dallo zip per volta: la memoria resta limitata
READ_BUFFER = 16 * 2**20  # lettura dello zip in blocchi sequenziali (job 29416: 40 ms/campione a salti)
Image.MAX_IMAGE_PIXELS = None  # diagrammi fino a ~12k px di lato, file del dataset verificato


def collect(zf: zipfile.ZipFile) -> tuple[dict[str, dict[str, zipfile.ZipInfo]], list[str]]:
    """id -> {tipo: ZipInfo}, più i file con percorsi inattesi."""
    samples: dict[str, dict[str, zipfile.ZipInfo]] = defaultdict(dict)
    unexpected = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename).parts
        m = NAME_RE.match(parts[-1])
        if not m or len(parts) != 4 or not re.fullmatch(r"depth\d+", parts[1]) or not parts[2].isdigit():
            unexpected.append(info.filename)
            continue
        samples[f"d{parts[1][5:]}-{parts[2]}"][m.group(2)] = info
    return samples, unexpected


def build_plan(zf, samples, limit: int | None) -> tuple[list[str], list[str], list[list[str]]]:
    """Campioni da scrivere nell'ordine dello zip; incompleti; duplicati [id, id tenuto]."""
    order = sorted(samples, key=lambda k: min(i.header_offset for i in samples[k].values()))
    kept, incomplete, duplicates = [], [], []
    seen: dict[bytes, str] = {}
    for key in order:
        files = samples[key]
        if set(files) != REQUIRED:
            incomplete.append(key)
            continue
        caption = json.loads(zf.read(files["annotation.json"]))["text_en"]["caption"]
        digest = hashlib.sha256(caption.encode()).digest()
        if digest in seen:
            duplicates.append([key, seen[digest]])
            continue
        seen[digest] = key
        kept.append(key)
        if limit and len(kept) >= limit:
            break
    return kept, incomplete, duplicates


def convert(task: tuple[str, bytes, int]) -> tuple[str, bytes | None, str | None, tuple | None]:
    """JPEG originale -> PNG alla dimensione esatta di target_size. Gira nei processi del pool."""
    key, raw, max_patches = task
    try:
        with Image.open(io.BytesIO(raw)) as im:
            width, height = im.size
            th, tw = target_size(height, width, max_patches)
            im.draft("RGB", (tw, th))  # decodifica JPEG ridotta per DCT, mai sotto la dimensione finale
            small = to_rgb(im)
        if small.size != (tw, th):
            small = small.resize((tw, th), Image.BILINEAR)  # resample=2 di preprocessor_config.json
        buf = io.BytesIO()
        small.save(buf, format="PNG")
    except Exception as exc:  # noqa: BLE001 - immagine illeggibile: fuori e registrata
        return key, None, f"{type(exc).__name__}: {exc}"[:200], None
    idempotent = target_size(th, tw, max_patches) == (th, tw)
    return key, buf.getvalue(), None, (width, height, tw, th, idempotent)


def check_shard(path: Path, keys: list[str]) -> None:
    """Rilegge il tar: stessi id nello stesso ordine, ognuno con png e json."""
    got = []
    for key, files in iter_shard(path):
        if set(files) != {"png", "json"}:
            raise RuntimeError(f"{path.name}: {key} ha {sorted(files)}")
        got.append(key)
    if got != keys:
        raise RuntimeError(f"{path.name}: {len(got)} campioni riletti invece di {len(keys)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", required=True, help="repo@commit, scritto in ogni campione")
    ap.add_argument("--variant", choices=VARIANTS, default="caption")
    ap.add_argument("--max-patches", type=int, default=DIAGRAM_MAX_PATCHES)
    ap.add_argument("--per-shard", type=int, default=19000, help="campioni per shard (~52 KB l'uno)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--limit", type=int, help="solo i primi N campioni del piano (prova)")
    args = ap.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("shard_mavis_geometry.py va eseguito in un job SLURM")
    if not args.zip.is_file():
        sys.exit(f"file mancante: {args.zip}")

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    # Con un buffer grande i seek in avanti nell'ordine dello zip restano dentro il buffer: su GlusterFS
    # ogni seek fuori buffer costa un giro sulla rete.
    zf = zipfile.ZipFile(open(args.zip, "rb", buffering=READ_BUFFER))
    samples, unexpected = collect(zf)
    kept, incomplete, duplicates = build_plan(zf, samples, args.limit)
    plan = [kept[i:i + args.per_shard] for i in range(0, len(kept), args.per_shard)]
    print(f"zip: {len(samples)} campioni, file inattesi {len(unexpected)}; incompleti {len(incomplete)}, "
          f"duplicati {len(duplicates)}; da scrivere {len(kept)} in {len(plan)} shard ({time.time() - t0:.0f} s)",
          flush=True)

    plan_doc = {"source": args.source, "variant": args.variant, "max_patches": args.max_patches,
                "per_shard": args.per_shard, "limit": args.limit, "shards": plan}
    plan_digest = hashlib.sha256(json.dumps(plan_doc, sort_keys=True).encode()).hexdigest()
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()).get("digest") != plan_digest:
            sys.exit(f"piano diverso da quello in {plan_path}: cartella di un'altra esecuzione")
    else:
        plan_path.write_text(json.dumps({"digest": plan_digest, "shards": len(plan), "samples": len(kept),
                                         **{k: v for k, v in plan_doc.items() if k != "shards"}}, indent=1))

    log_path = args.out / "shards.jsonl"
    done: dict[str, dict] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            done[row["name"]] = row

    prefix = f"mavis-geometry-{args.variant}"
    with Pool(args.workers) as pool:
        for idx, keys in enumerate(plan):
            name = f"{prefix}-{idx:05d}.tar"
            final = args.out / name
            prev = done.get(name)
            if prev and final.is_file() and final.stat().st_size == prev["bytes"]:
                continue
            t_shard = time.time()
            tmp = args.out / (name + ".tmp")
            written, bad, tokens, png_bytes, not_idempotent = [], [], [], 0, []
            timing = {"read": 0.0, "wait": 0.0, "write": 0.0}

            def read(batch: list[str]) -> tuple[list[tuple], dict[str, bytes]]:
                start = time.perf_counter()
                tasks, anns = [], {}
                for k in batch:
                    anns[k] = zf.read(samples[k]["annotation.json"])
                    tasks.append((k, zf.read(samples[k][f"{args.variant}.jpg"]), args.max_patches))
                timing["read"] += time.perf_counter() - start
                return tasks, anns

            # Doppio buffer: il processo principale legge il blocco successivo e scrive il precedente
            # mentre il pool converte quello corrente.
            batches = [keys[i:i + BATCH] for i in range(0, len(keys), BATCH)]
            tasks, anns = read(batches[0])
            pending = pool.map_async(convert, tasks, chunksize=8)
            with open(tmp, "wb") as fh:
                hw = HashingWriter(fh)
                with tarfile.open(fileobj=hw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                    for j in range(len(batches)):
                        nxt = read(batches[j + 1]) if j + 1 < len(batches) else None
                        start = time.perf_counter()
                        results = pending.get()
                        timing["wait"] += time.perf_counter() - start
                        cur_anns = anns
                        if nxt is not None:
                            tasks, anns = nxt
                            pending = pool.map_async(convert, tasks, chunksize=8)
                        start = time.perf_counter()
                        for key, png, err, dims in results:
                            if err:
                                bad.append(f"{key}: {err}")
                                continue
                            width, height, tw, th, idempotent = dims
                            if not idempotent:
                                not_idempotent.append(key)
                            files = samples[key]
                            ann = json.loads(cur_anns[key])
                            meta = {"id": key, "zip_dir": str(PurePosixPath(files["annotation.json"].filename).parent),
                                    "depth": int(key[1:key.index("-")]), "variant": args.variant,
                                    "text_en": ann["text_en"], "choices": ann["choices"],
                                    "answer_index": ann["answer_index"],
                                    "orig_width": width, "orig_height": height, "width": tw, "height": th,
                                    "max_patches": args.max_patches, "format": "PNG", "source": args.source}
                            add_member(tar, f"{key}.png", png)
                            add_member(tar, f"{key}.json", json.dumps(meta, ensure_ascii=False).encode())
                            written.append(key)
                            tokens.append((th // UNIT) * (tw // UNIT))
                            png_bytes += len(png)
                        timing["write"] += time.perf_counter() - start
                fh.flush()
                os.fsync(fh.fileno())
            check_shard(tmp, written)
            os.replace(tmp, final)
            row = {"name": name, "samples": len(written), "bytes": hw.n, "sha256": hw.sha.hexdigest(),
                   "first": written[0] if written else None, "last": written[-1] if written else None,
                   "image_tokens": sum(tokens), "png_bytes": png_bytes, "unreadable": bad,
                   "not_idempotent": not_idempotent, "seconds": round(time.time() - t_shard, 1),
                   "timing": {k: round(v, 1) for k, v in timing.items()}}
            with open(log_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            done[name] = row
            print(f"[{idx + 1}/{len(plan)}] {name}: {len(written)} campioni, {hw.n / 1e9:.2f} GB, "
                  f"token immagine medi {statistics.mean(tokens) if tokens else 0:.0f}, illeggibili {len(bad)}, "
                  f"non idempotenti {len(not_idempotent)}, {row['seconds']:.0f} s "
                  f"({row['seconds'] / max(1, len(written)) * 1000:.1f} ms/campione; s di lettura, attesa del pool, "
                  f"scrittura: {row['timing']})", flush=True)

    rows = [done[f"{prefix}-{i:05d}.tar"] for i in range(len(plan))]
    n_written = sum(r["samples"] for r in rows)
    manifest = {
        "source": args.source, "job_id": os.environ.get("SLURM_JOB_ID"), "zip": str(args.zip),
        "zip_sha256_expected": args.zip.resolve().name, "variant": args.variant,
        "max_patches": args.max_patches, "per_shard": args.per_shard, "limit": args.limit,
        "plan_digest": plan_digest,
        "counts": {"zip_samples": len(samples), "unexpected_files": len(unexpected),
                   "incomplete": len(incomplete), "duplicate_captions": len(duplicates),
                   "planned": len(kept), "written": n_written,
                   "unreadable": sum(len(r["unreadable"]) for r in rows),
                   "not_idempotent": sum(len(r["not_idempotent"]) for r in rows)},
        "image_tokens": {"total": sum(r["image_tokens"] for r in rows),
                         "mean": sum(r["image_tokens"] for r in rows) / max(1, n_written)},
        "png_bytes_mean": sum(r["png_bytes"] for r in rows) / max(1, n_written),
        "shards": [{k: r[k] for k in ("name", "samples", "bytes", "sha256", "first", "last")} for r in rows],
        "excluded": {"incomplete": incomplete, "duplicate_captions": duplicates,
                     "unreadable": [b for r in rows for b in r["unreadable"]],
                     "unexpected_files": unexpected[:50]},
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("counts", "image_tokens", "png_bytes_mean", "seconds")}, indent=1),
          flush=True)
    if n_written == 0:
        sys.exit("nessun campione scritto")


if __name__ == "__main__":
    main()
