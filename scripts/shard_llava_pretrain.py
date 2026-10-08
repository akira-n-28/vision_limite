#!/usr/bin/env python3
"""Converte LLaVA-Pretrain (images.zip + blip_laion_cc_sbu_558k.json) in shard tar da circa 1 GB.

Da eseguire SOLO in un job SLURM CPU (slurm/shard_llava_pretrain.sbatch): legge lo zip in ordine
e scrive i tar direttamente, senza estrarre i ~558 mila file piccoli su GlusterFS.

Shard in stile webdataset: per ogni campione `<id>.jpg` (byte originali, nessuna ricodifica) e
`<id>.json` (id, percorso nello zip, prompt, didascalia, larghezza, altezza, formato, sorgente).
Ordine: quello dello zip (il mescolamento è compito del loader). Le immagini che Pillow non
riesce ad aprire (intestazione illeggibile) e quelle senza didascalia restano fuori e sono
elencate nel manifest. Non si decodificano i pixel: un'immagine con l'intestazione valida ma
dati troncati va gestita dal loader del training.

Ripresa: i confini degli shard si calcolano dai soli metadati dello zip (dimensioni non
compresse), quindi sono identici a ogni esecuzione; uno shard già registrato in shards.jsonl e
presente con la dimensione registrata viene saltato. Ogni tar si scrive in `.tmp` e si rinomina
solo a fine shard.

Uscita in --out: llava-pretrain-NNNNN.tar, plan.json (piano degli shard), shards.jsonl (una riga
per shard: nome, campioni, byte, sha256, primo e ultimo id) e manifest.json (sorgente, conteggi,
campioni esclusi, statistiche delle dimensioni).
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
from pathlib import Path

KEY_RE = re.compile(r"^[0-9A-Za-z_-]+$")
PER_SAMPLE_OVERHEAD = 2048  # intestazioni tar e JSON del campione, stima per il piano


class HashingWriter:
    """File in scrittura che calcola SHA-256 e byte scritti mentre tarfile scrive."""

    def __init__(self, fh):
        self.fh, self.sha, self.n = fh, hashlib.sha256(), 0

    def write(self, data: bytes) -> int:
        self.fh.write(data)
        self.sha.update(data)
        self.n += len(data)
        return len(data)

    def tell(self) -> int:
        return self.n

    def flush(self) -> None:
        self.fh.flush()


def add_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    tar.addfile(info, io.BytesIO(data))


def load_captions(path: Path) -> tuple[dict[str, dict], list[str]]:
    """image (percorso nello zip) -> {id, prompt, caption}; più i problemi trovati."""
    problems: list[str] = []
    out: dict[str, dict] = {}
    for row in json.loads(path.read_text()):
        conv = row.get("conversations") or []
        human = next((c["value"] for c in conv if c.get("from") == "human"), None)
        gpt = next((c["value"] for c in conv if c.get("from") == "gpt"), None)
        if not row.get("image") or human is None or gpt is None:
            problems.append(f"voce incompleta: {row.get('id')}")
            continue
        if row["image"] in out:
            problems.append(f"immagine ripetuta nel JSON: {row['image']}")
            continue
        if not KEY_RE.match(str(row["id"])):
            problems.append(f"id non usabile come chiave: {row['id']!r}")
            continue
        out[row["image"]] = {"id": str(row["id"]), "prompt": human, "caption": gpt}
    return out, problems


def make_plan(infos: list[zipfile.ZipInfo], captions: dict[str, dict], target: int) -> list[list[str]]:
    shards: list[list[str]] = [[]]
    size = 0
    for info in infos:
        if info.filename not in captions:
            continue
        cost = info.file_size + PER_SAMPLE_OVERHEAD
        if shards[-1] and size + cost > target:
            shards.append([])
            size = 0
        shards[-1].append(info.filename)
        size += cost
    return shards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--captions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", required=True, help="repo@commit, scritto in ogni campione")
    ap.add_argument("--target-bytes", type=int, default=1_000_000_000)
    args = ap.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("shard_llava_pretrain.py va eseguito in un job SLURM")
    for p in (args.zip, args.captions):
        if not p.is_file():
            sys.exit(f"file mancante: {p}")
    from PIL import Image  # dopo i controlli: errore chiaro se manca l'ambiente

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    captions, caption_problems = load_captions(args.captions)
    print(f"didascalie: {len(captions)} (problemi: {len(caption_problems)})", flush=True)

    zf = zipfile.ZipFile(args.zip)
    infos = [i for i in zf.infolist() if not i.is_dir()]
    in_zip = {i.filename for i in infos}
    no_caption = [i.filename for i in infos if i.filename not in captions]
    no_image = [k for k in captions if k not in in_zip]
    plan = make_plan(infos, captions, args.target_bytes)
    by_name = {i.filename: i for i in infos}
    print(f"zip: {len(infos)} file; senza didascalia {len(no_caption)}; didascalie senza immagine "
          f"{len(no_image)}; shard previsti {len(plan)}", flush=True)

    # Il piano dipende solo dai metadati: se esiste già deve coincidere (ripresa sicura).
    plan_doc = {"source": args.source, "target_bytes": args.target_bytes,
                "shards": [[(n, by_name[n].file_size) for n in names] for names in plan]}
    plan_digest = hashlib.sha256(json.dumps(plan_doc, sort_keys=True).encode()).hexdigest()
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old.get("digest") != plan_digest:
            sys.exit(f"piano diverso da quello in {plan_path}: cartella di un'altra esecuzione")
    else:
        plan_path.write_text(json.dumps({"digest": plan_digest, "shards": len(plan),
                                         "samples": sum(map(len, plan))}, indent=1))

    log_path = args.out / "shards.jsonl"
    done: dict[str, dict] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            done[row["name"]] = row

    unreadable: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    written = 0
    for idx, names in enumerate(plan):
        name = f"llava-pretrain-{idx:05d}.tar"
        final = args.out / name
        prev = done.get(name)
        if prev and final.is_file() and final.stat().st_size == prev["bytes"]:
            widths.extend(prev["widths"])
            heights.extend(prev["heights"])
            unreadable.extend(prev["unreadable"])
            written += prev["samples"]
            continue
        tmp = args.out / (name + ".tmp")
        w_shard, h_shard, bad_shard, keys = [], [], [], []
        with open(tmp, "wb") as fh:
            hw = HashingWriter(fh)
            with tarfile.open(fileobj=hw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for member in names:
                    data = zf.read(by_name[member])
                    try:
                        with Image.open(io.BytesIO(data)) as im:
                            width, height, fmt, mode = im.width, im.height, im.format, im.mode
                    except Exception as exc:  # noqa: BLE001 - immagine illeggibile: fuori e registrata
                        bad_shard.append(f"{member}: {type(exc).__name__}")
                        continue
                    cap = captions[member]
                    ext = "png" if fmt == "PNG" else "jpg"
                    meta = {"id": cap["id"], "image": member, "prompt": cap["prompt"], "caption": cap["caption"],
                            "width": width, "height": height, "format": fmt, "mode": mode, "source": args.source}
                    add_member(tar, f"{cap['id']}.{ext}", data)
                    add_member(tar, f"{cap['id']}.json", json.dumps(meta, ensure_ascii=False).encode())
                    keys.append(cap["id"])
                    w_shard.append(width)
                    h_shard.append(height)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)
        row = {"name": name, "samples": len(keys), "bytes": hw.n, "sha256": hw.sha.hexdigest(),
               "first": keys[0] if keys else None, "last": keys[-1] if keys else None,
               "unreadable": bad_shard, "widths": w_shard, "heights": h_shard}
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        done[name] = row
        widths.extend(w_shard)
        heights.extend(h_shard)
        unreadable.extend(bad_shard)
        written += len(keys)
        print(f"[{idx + 1}/{len(plan)}] {name}: {len(keys)} campioni, {hw.n / 1e9:.2f} GB, "
              f"illeggibili {len(bad_shard)}, {time.time() - t0:.0f} s", flush=True)

    def stats(xs: list[int]) -> dict:
        if not xs:
            return {}
        q = statistics.quantiles(xs, n=20)
        return {"min": min(xs), "p05": q[0], "median": statistics.median(xs), "p95": q[-1], "max": max(xs)}

    # Ultima riga di ogni shard, nell'ordine del piano (una ripresa può aver riscritto uno shard).
    shard_rows = [done[f"llava-pretrain-{i:05d}.tar"] for i in range(len(plan))]
    manifest = {
        "source": args.source, "job_id": os.environ.get("SLURM_JOB_ID"),
        "zip": str(args.zip), "captions": str(args.captions),
        "zip_sha256_expected": args.zip.resolve().name, "captions_sha256_expected": args.captions.resolve().name,
        "target_bytes": args.target_bytes, "plan_digest": plan_digest,
        "counts": {"zip_files": len(infos), "captions": len(captions), "written": written,
                   "zip_without_caption": len(no_caption), "caption_without_image": len(no_image),
                   "unreadable": len(unreadable), "caption_problems": len(caption_problems),
                   "min_side_below_64": sum(1 for w, h in zip(widths, heights) if min(w, h) < 64)},
        "width": stats(widths), "height": stats(heights),
        "shards": [{k: r[k] for k in ("name", "samples", "bytes", "sha256", "first", "last")} for r in shard_rows],
        "examples": {"zip_without_caption": no_caption[:20], "caption_without_image": no_image[:20],
                     "unreadable": unreadable[:50], "caption_problems": caption_problems[:20]},
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("counts", "width", "height", "seconds")}, indent=1), flush=True)
    if written == 0:
        sys.exit("nessun campione scritto")


if __name__ == "__main__":
    main()
