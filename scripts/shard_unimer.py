#!/usr/bin/env python3
"""Converte UniMER-1M.zip e UniMER-Test.zip (wanderkid/UniMER_Dataset) in shard tar per F1.

Da eseguire SOLO in un job SLURM CPU (slurm/shard_unimer.sbatch): legge gli zip e scrive i tar
direttamente, senza estrarre il milione di file piccoli su GlusterFS.

Corrispondenza (job 29436): l'immagine `UniMER-1M/images/<indice>.png` ha come formula la riga <indice>
di `train.txt`; nel test, `UniMER-Test/<sub>/<indice>.png` la riga <indice> di `<sub>.txt`. Il dataset è
mescolato: le righe vuote sono HME100K (tolto per copyright, nessuna immagine) e non si possono separare
per indice le formule scritte a mano di CROHME (meno dell'1%), che restano.

Per ogni campione, `<id>.png` e `<id>.json`:
- immagine ridimensionata una sola volta a processing.capped_size (ingrandimento al più --scale, al più
  --max-patches patch), BICUBIC, nel modo originale (L o RGB); nel JSON max_patches = patch esatte, così
  il processor non la ridimensiona;
- JSON con id, sottoinsieme, indice, formula senza spazi fra i token (latex.despace) e formula originale,
  dimensioni, max_patches, sorgente.
Esclusi ed elencati nel manifest: righe vuote, immagini senza riga, formule che despace non conserva
token per token, immagini illeggibili.

Train in --out/unimer-train-NNNNN.tar (--per-shard campioni, nell'ordine dello zip); test in
--out/test/unimer-test-<sub>.tar. Ripresa come negli altri script di sharding: piano deterministico,
shard registrati in shards.jsonl e presenti con la dimensione registrata vengono saltati, ogni tar si
scrive in `.tmp`, si rilegge e si rinomina. --limit N: solo i primi N campioni del train (prova).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path, PurePosixPath

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shard_llava_pretrain import HashingWriter, add_member  # noqa: E402  (stessa cartella scripts/)
from shard_mavis_geometry import check_shard  # noqa: E402

from limite_vl.latex import despace, latex_tokens  # noqa: E402
from limite_vl.processing import capped_size, exact_patches, target_size  # noqa: E402

BATCH = 2048  # campioni letti dallo zip per volta (immagini piccole)
READ_BUFFER = 16 * 2**20
TEST_SUBSETS = ("spe", "cpe", "sce", "hwe")


def read_lines(zf: zipfile.ZipFile, name: str) -> list[str]:
    lines = zf.read(name).decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def plan_items(zf: zipfile.ZipFile, folder: str, lines: list[str], subset: str) -> tuple[list[dict], dict]:
    """Campioni (in ordine di posizione nello zip) ed esclusi per motivo."""
    excluded = {"blank_label": [], "no_label": [], "despace_changes_tokens": []}
    items = []
    infos = [i for i in zf.infolist() if PurePosixPath(i.filename).parent.as_posix() == folder
             and i.filename.endswith(".png") and PurePosixPath(i.filename).stem.isdigit()]
    for info in sorted(infos, key=lambda i: i.header_offset):
        index = int(PurePosixPath(info.filename).stem)
        key = f"{subset}-{index:07d}"
        if index >= len(lines):
            excluded["no_label"].append(key)
            continue
        spaced = lines[index].strip()
        if not spaced:
            excluded["blank_label"].append(key)
            continue
        label = despace(spaced)
        if latex_tokens(label) != latex_tokens(spaced):
            excluded["despace_changes_tokens"].append(key)
            continue
        items.append({"key": key, "member": info.filename, "index": index, "subset": subset,
                      "label": label, "label_spaced": spaced})
    return items, excluded


def convert(task: tuple[str, bytes, float, int]):
    """PNG originale -> PNG a capped_size. Gira nei processi del pool."""
    key, raw, scale, max_patches = task
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            width, height = im.size
            image = im if im.mode in ("L", "RGB") else im.convert("RGB")
            h, w = capped_size(height, width, scale, max_patches)
            if (w, h) != image.size:
                image = image.resize((w, h), Image.BICUBIC)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
    except Exception as exc:  # noqa: BLE001 - immagine illeggibile: fuori e registrata
        return key, None, f"{type(exc).__name__}: {exc}"[:200], None
    patches = exact_patches(h, w)
    return key, buf.getvalue(), None, (width, height, w, h, patches, target_size(h, w, patches) == (h, w))


def write_shard(path: Path, zf: zipfile.ZipFile, items: list[dict], pool: Pool, args, source: str) -> dict:
    """Un tar con gli item dati; conversione nel pool con doppio buffer. Ritorna la riga di shards.jsonl."""
    if not items:
        raise ValueError(f"{path.name}: nessun campione")
    t0 = time.time()
    tmp = path.with_name(path.name + ".tmp")
    written, bad, not_idempotent, tokens = [], [], [], []
    timing = {"read": 0.0, "wait": 0.0, "write": 0.0}

    def read(batch):
        start = time.perf_counter()
        out = [(it["key"], zf.read(it["member"]), args.scale, args.max_patches) for it in batch]
        timing["read"] += time.perf_counter() - start
        return out

    by_key = {it["key"]: it for it in items}
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    pending = pool.map_async(convert, read(batches[0]), chunksize=64)
    with open(tmp, "wb") as fh:
        hw = HashingWriter(fh)
        with tarfile.open(fileobj=hw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for j in range(len(batches)):
                nxt = read(batches[j + 1]) if j + 1 < len(batches) else None
                start = time.perf_counter()
                results = pending.get()
                timing["wait"] += time.perf_counter() - start
                if nxt is not None:
                    pending = pool.map_async(convert, nxt, chunksize=64)
                start = time.perf_counter()
                for key, png, err, dims in results:
                    if err:
                        bad.append(f"{key}: {err}")
                        continue
                    width, height, w, h, patches, idempotent = dims
                    if not idempotent:
                        not_idempotent.append(key)
                    it = by_key[key]
                    meta = {"id": key, "subset": it["subset"], "index": it["index"], "label": it["label"],
                            "label_spaced": it["label_spaced"], "orig_width": width, "orig_height": height,
                            "width": w, "height": h, "max_patches": patches, "format": "PNG", "source": source}
                    add_member(tar, f"{key}.png", png)
                    add_member(tar, f"{key}.json", json.dumps(meta, ensure_ascii=False).encode())
                    written.append(key)
                    tokens.append(patches // 4)
                timing["write"] += time.perf_counter() - start
        fh.flush()
        os.fsync(fh.fileno())
    check_shard(tmp, written)
    os.replace(tmp, path)
    return {"name": path.name, "samples": len(written), "bytes": hw.n, "sha256": hw.sha.hexdigest(),
            "first": written[0] if written else None, "last": written[-1] if written else None,
            "image_tokens": sum(tokens), "unreadable": bad, "not_idempotent": not_idempotent,
            "seconds": round(time.time() - t0, 1), "timing": {k: round(v, 1) for k, v in timing.items()}}


def write_set(out: Path, zf, shards: list[list[dict]], pool, args, source: str, plan_doc: dict) -> list[dict]:
    """Scrive (o riprende) una serie di shard con il loro plan.json e shards.jsonl."""
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(plan_doc, sort_keys=True).encode()).hexdigest()
    plan_path = out / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()).get("digest") != digest:
            sys.exit(f"piano diverso da quello in {plan_path}: cartella di un'altra esecuzione")
    else:
        plan_path.write_text(json.dumps({"digest": digest, "shards": len(shards),
                                         "samples": sum(map(len, shards)),
                                         **{k: v for k, v in plan_doc.items() if k != "keys"}}, indent=1))
    log_path = out / "shards.jsonl"
    done = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            done[row["name"]] = row
    rows = []
    for name, items in shards:
        path = out / name
        prev = done.get(name)
        if prev and path.is_file() and path.stat().st_size == prev["bytes"]:
            rows.append(prev)
            continue
        row = write_shard(path, zf, items, pool, args, source)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        rows.append(row)
        print(f"{name}: {row['samples']} campioni, {row['bytes'] / 1e9:.2f} GB, token immagine medi "
              f"{row['image_tokens'] / max(1, row['samples']):.0f}, illeggibili {len(row['unreadable'])}, "
              f"non idempotenti {len(row['not_idempotent'])}, {row['seconds']:.0f} s (lettura, attesa del pool, scrittura: "
              f"{row.get('timing')})", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, type=Path, help="UniMER-1M.zip")
    ap.add_argument("--test-zip", required=True, type=Path, help="UniMER-Test.zip")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", required=True, help="repo@commit, scritto in ogni campione")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--max-patches", type=int, default=1024)
    ap.add_argument("--per-shard", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--limit", type=int, help="solo i primi N campioni del train (prova), niente test")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("shard_unimer.py va eseguito in un job SLURM")
    for p in (args.zip, args.test_zip):
        if not p.is_file():
            sys.exit(f"file mancante: {p}")

    t0 = time.time()
    common = {"source": args.source, "scale": args.scale, "max_patches": args.max_patches}
    manifest = {**common, "job_id": os.environ.get("SLURM_JOB_ID"), "zip": str(args.zip),
                "zip_sha256_expected": args.zip.resolve().name, "test_zip": str(args.test_zip),
                "test_zip_sha256_expected": args.test_zip.resolve().name, "limit": args.limit, "sets": {}}
    with Pool(args.workers) as pool:
        zf = zipfile.ZipFile(open(args.zip, "rb", buffering=READ_BUFFER))
        lines = read_lines(zf, "UniMER-1M/train.txt")
        items, excluded = plan_items(zf, "UniMER-1M/images", lines, "train")
        if args.limit:
            items = items[: args.limit]
        shards = [(f"unimer-train-{i // args.per_shard:05d}.tar", items[i:i + args.per_shard])
                  for i in range(0, len(items), args.per_shard)]
        print(f"train: {len(lines)} righe, {len(items)} campioni in {len(shards)} shard; esclusi "
              f"{ {k: len(v) for k, v in excluded.items()} } ({time.time() - t0:.0f} s)", flush=True)
        plan_doc = {**common, "per_shard": args.per_shard, "limit": args.limit,
                    "keys": [[it["key"] for it in s] for _, s in shards]}
        rows = write_set(args.out, zf, shards, pool, args, args.source, plan_doc)
        manifest["sets"]["train"] = {"lines": len(lines), "planned": len(items),
                                     "excluded": {k: len(v) for k, v in excluded.items()},
                                     "excluded_examples": {k: v[:20] for k, v in excluded.items()}, "shards": rows}
        if not args.limit:
            tz = zipfile.ZipFile(open(args.test_zip, "rb", buffering=READ_BUFFER))
            test_shards, test_info = [], {}
            for sub in TEST_SUBSETS:
                sub_lines = read_lines(tz, f"UniMER-Test/{sub}.txt")
                sub_items, sub_excluded = plan_items(tz, f"UniMER-Test/{sub}", sub_lines, sub)
                test_shards.append((f"unimer-test-{sub}.tar", sub_items))
                test_info[sub] = {"lines": len(sub_lines), "planned": len(sub_items),
                                  "excluded": {k: len(v) for k, v in sub_excluded.items()},
                                  "excluded_examples": {k: v[:20] for k, v in sub_excluded.items()}}
                print(f"test {sub}: {len(sub_items)} campioni; esclusi { test_info[sub]['excluded'] }", flush=True)
            test_doc = {**common, "keys": [[it["key"] for it in s] for _, s in test_shards]}
            test_rows = write_set(args.out / "test", tz, test_shards, pool, args, args.source, test_doc)
            for sub, row in zip(TEST_SUBSETS, test_rows):
                test_info[sub]["shard"] = row
            manifest["sets"]["test"] = test_info

    def strip(row):
        return {k: row[k] for k in ("name", "samples", "bytes", "sha256", "first", "last", "image_tokens")}

    all_rows = list(manifest["sets"]["train"]["shards"])
    manifest["sets"]["train"]["shards"] = [strip(r) for r in all_rows]
    manifest["shards"] = manifest["sets"]["train"]["shards"]  # per train_f1.shard_hashes
    written = sum(r["samples"] for r in all_rows)
    manifest["counts"] = {"written": written, "unreadable": sum(len(r["unreadable"]) for r in all_rows),
                          "not_idempotent": sum(len(r["not_idempotent"]) for r in all_rows),
                          "image_tokens_mean": sum(r["image_tokens"] for r in all_rows) / max(1, written)}
    if "test" in manifest["sets"]:
        for sub in TEST_SUBSETS:
            manifest["sets"]["test"][sub]["shard"] = strip(manifest["sets"]["test"][sub]["shard"])
        (args.out / "test" / "manifest.json").write_text(json.dumps(
            {**common, "shards": [manifest["sets"]["test"][s]["shard"] for s in TEST_SUBSETS]}, indent=1))
    manifest["seconds"] = round(time.time() - t0, 1)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("counts", "seconds")}, indent=1), flush=True)
    if written == 0:
        sys.exit("nessun campione scritto")


if __name__ == "__main__":
    main()
