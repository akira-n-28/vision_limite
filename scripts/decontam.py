#!/usr/bin/env python3
"""Decontaminazione contro i benchmark visivi (src/limite_vl/decontam.py). Da eseguire in job CPU
(slurm/decontam.sbatch), mai sul frontend.

  index  Legge i benchmark dalle cartelle degli snapshot (parquet, con le immagini incluse come struct
         {bytes} oppure come percorsi a file sciolti o dentro zip della stessa cartella) e scrive un indice
         JSON: per ogni elemento benchmark, id, pHash dell'immagine e testo della domanda.
  scan   Confronta gli shard di training di un tipo con l'indice e scrive in --out:
         exclude.txt (chiavi da escludere, una per riga), matches.jsonl (ogni corrispondenza con benchmark,
         id, tipo e distanza) e summary.json. Le chiavi in exclude.txt sono quelle del flusso di training
         (stream.Source.exclude le salta).

    srun $RUN_PYTHON -u -B scripts/decontam.py index --benchmark mathvista=DIR ... --out INDEX.json
    srun $RUN_PYTHON -u -B scripts/decontam.py scan --index INDEX.json --kind mmfinereason --shards GLOB --out DIR
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.decontam import HashIndex, TextIndex, phash  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
IMAGE_COLUMNS = ("decoded_image", "image", "image_path", "images")  # in ordine di preferenza
TEXT_COLUMNS = ("question", "query", "query_wo", "question_for_eval", "problem", "text")
ID_COLUMNS = ("pid", "id", "ID", "sample_index", "problem_index", "index")


def hash_bytes(data: bytes) -> int | None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return phash(im)
    except Exception:  # noqa: BLE001 - immagine illeggibile: nessun hash
        return None


# ---------------------------------------------------------------- index
class LooseFiles:
    """Immagini referenziate per percorso: file sciolti nella cartella dello snapshot o membri di zip."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.zips = {}
        for z in folder.rglob("*.zip"):
            zf = zipfile.ZipFile(z)
            for name in zf.namelist():
                if not name.endswith("/"):
                    self.zips.setdefault(name, (zf, name))
                    self.zips.setdefault(PurePosixPath(name).name, (zf, name))

    def read(self, rel: str) -> bytes | None:
        path = self.folder / rel
        if path.is_file():
            return path.read_bytes()
        for key in (rel, PurePosixPath(rel).name):
            if key in self.zips:
                zf, name = self.zips[key]
                return zf.read(name)
        return None


def image_from(value, loose: LooseFiles) -> bytes | None:
    if isinstance(value, (bytes, bytearray)):  # colonna binaria semplice (non lo struct {bytes, path} di datasets)
        return bytes(value) or None
    if isinstance(value, dict):
        if value.get("bytes"):
            return value["bytes"]
        if value.get("path"):
            return loose.read(value["path"])
    if isinstance(value, list) and value:
        return image_from(value[0], loose)
    if isinstance(value, str) and value:
        return loose.read(value)
    return None


def cmd_index(args) -> None:
    items, report = [], {}
    for spec in args.benchmark:
        name, _, folder = spec.partition("=")
        folder = Path(folder)
        files = sorted(p for p in folder.rglob("*.parquet"))
        if not files:
            sys.exit(f"{name}: nessun parquet in {folder}")
        loose = LooseFiles(folder)
        counts = Counter()
        for f in files:
            table = pq.read_table(f)
            cols = table.column_names
            image_cols = [c for c in IMAGE_COLUMNS if c in cols]
            text_cols = [c for c in TEXT_COLUMNS if c in cols]
            id_col = next((c for c in ID_COLUMNS if c in cols), None)
            for i, row in enumerate(table.to_pylist()):
                counts["rows"] += 1
                data = next((d for d in (image_from(row[c], loose) for c in image_cols) if d), None)
                h = hash_bytes(data) if data else None
                text = " ".join(str(row[c]) for c in text_cols if row.get(c))
                counts["with_image"] += h is not None
                if data and h is None:
                    counts["unreadable"] += 1
                    if "first_error" not in report.setdefault(name, {}):
                        try:
                            with Image.open(io.BytesIO(data)) as im:
                                phash(im)
                        except Exception as e:  # noqa: BLE001 - solo per il messaggio
                            report[name]["first_error"] = repr(e)[:300]
                counts["with_text"] += bool(text)
                items.append({"benchmark": name, "file": f.name, "id": str(row[id_col]) if id_col else f"{f.name}:{i}",
                              "phash": f"{h:016x}" if h is not None else None, "text": text})
        report[name] = {**report.get(name, {}), **counts}
        print(f"{name}: {dict(counts)} da {len(files)} parquet", flush=True)
        if counts["with_image"] == 0:
            first = pq.read_table(files[0]).slice(0, 1).to_pylist()[0]
            seen = {c: (type(first[c]).__name__, repr(first[c])[:120]) for c in first if c in IMAGE_COLUMNS}
            sys.exit(f"{name}: nessuna immagine letta ({report[name]}): colonne {pq.read_schema(files[0])}, prima riga {seen}, "
                     f"zip {[str(z) for z in folder.rglob('*.zip')]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"items": items, "report": report}, ensure_ascii=False))
    print(f"indice: {len(items)} elementi -> {args.out}")


# ---------------------------------------------------------------- scan
def texts_of(kind: str, meta: dict) -> list[str]:
    if kind == "mmfinereason":
        return [meta.get("question") or ""]
    if kind == "mavis_caption":
        en = meta.get("text_en") or {}
        return [en.get(k) or "" for k in ("text_dominant_question", "text_lite_question", "vision_dominant_question")]
    if kind in ("deepvision", "virl"):
        return [meta.get("question") or ""]
    return []  # didascalie e formule: solo immagini


def scan_sample(task: tuple[str, str, bytes | None, list[str]]):
    key, ext, data, texts = task
    return key, hash_bytes(data) if data else None, texts


def cmd_scan(args) -> None:
    t0 = time.time()
    index = json.loads(Path(args.index).read_text())["items"]
    with_hash = [(i, int(it["phash"], 16)) for i, it in enumerate(index) if it["phash"]]
    hidx = HashIndex([h for _, h in with_hash], radius=args.radius)
    tidx = TextIndex.build([it["text"] for it in index], n=args.ngram, max_df=args.max_df)
    print(f"indice: {len(with_hash)} immagini, {len(tidx.grams)} n-gram ({tidx.dropped} esclusi perché in più di "
          f"{args.max_df} testi diversi)", flush=True)
    shards = sorted(glob.glob(args.shards))
    if not shards:
        sys.exit(f"nessuno shard per {args.shards}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    excluded: set[str] = set()
    per_bench, by_kind = Counter(), Counter()
    n = 0
    with Pool(args.workers) as pool, open(out / "matches.jsonl", "w") as matches:
        for shard in shards:
            tasks = []

            def flush():
                nonlocal n
                for key, h, texts in pool.imap(scan_sample, tasks, chunksize=64):
                    n += 1
                    found = []
                    if h is not None:
                        for j, d in hidx.query(h):
                            found.append(("image", with_hash[j][0], d))
                    for t in texts:
                        if t:
                            found.extend(("text", j, None) for j in tidx.query(t))
                    for m, (kind, j, d) in enumerate(found):
                        it = index[j]
                        per_bench[f"{it['benchmark']}/{kind}"] += 1
                        if m < args.max_lines:  # un'immagine quasi vuota può somigliare a centinaia di altre
                            matches.write(json.dumps({"key": key, "match": kind, "distance": d, "benchmark": it["benchmark"],
                                                      "bench_id": it["id"], "n_matches": len(found)},
                                                     ensure_ascii=False) + "\n")
                    if found:
                        excluded.add(key)
                        by_kind["image" if any(k == "image" for k, _, _ in found) else "text_only"] += 1
                tasks.clear()

            for key, files in iter_shard(shard):
                ext = next((e for e in ("png", "jpg", "jpeg", "webp") if e in files), None)
                meta = json.loads(files["json"]) if "json" in files else {}
                tasks.append((key, ext, files.get(ext) if ext else None, texts_of(args.kind, meta)))
                if len(tasks) >= 4096:
                    flush()
            flush()
            print(f"{Path(shard).name}: {n} campioni finora, {len(excluded)} da escludere ({time.time() - t0:.0f} s)",
                  flush=True)
    (out / "exclude.txt").write_text("".join(f"{k}\n" for k in sorted(excluded)))
    summary = {"kind": args.kind, "shards": shards, "index": args.index, "radius": args.radius, "ngram": args.ngram,
               "max_df": args.max_df, "ngrams_dropped": tidx.dropped, "samples": n, "excluded": len(excluded),
               "excluded_by": dict(by_kind), "matches_per_benchmark": dict(per_bench), "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("samples", "excluded", "excluded_by", "matches_per_benchmark", "seconds")},
                     indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--benchmark", action="append", required=True, help="nome=cartella dello snapshot")
    a.add_argument("--out", required=True)
    b = sub.add_parser("scan")
    b.add_argument("--index", required=True)
    b.add_argument("--kind", required=True)
    b.add_argument("--shards", required=True, help="glob degli shard tar")
    b.add_argument("--out", required=True)
    b.add_argument("--radius", type=int, default=4, help="distanza massima fra pHash, in bit")
    b.add_argument("--ngram", type=int, default=10)
    b.add_argument("--max-df", type=int, default=25, help="n-gram in più di tanti testi diversi dei benchmark: ignorati")
    b.add_argument("--max-lines", type=int, default=20, help="righe di matches.jsonl al più per campione")
    b.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("decontam.py va eseguito in un job SLURM")
    cmd_index(args) if args.cmd == "index" else cmd_scan(args)


if __name__ == "__main__":
    main()
