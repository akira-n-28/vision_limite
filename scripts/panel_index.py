#!/usr/bin/env python3
"""Indice del pannello di valutazione (docs/PANNELLO_VALUTAZIONE.md): una voce per immagine, con pHash, sha256 dei
pixel normalizzati e testo, più le voci di solo testo. Solo in un job SLURM CPU; snapshot in sola lettura.

Riusa la lettura delle immagini di scripts/decontam.py (struct {bytes, path}, colonne binarie, percorsi in cartelle o
zip) e limite_vl.decontam.phash. Ogni voce ha i campi che legge `decontam.py scan` (benchmark, file, id, phash, text) e:
- item: l'elemento (domanda o problema) a cui appartiene l'immagine, unico nel benchmark; pos: posizione
  dell'immagine nell'elemento (gli elementi con più immagini, come MUIRBench, MV-MATH e OlympiadBench, hanno una voce
  per immagine; quelli di solo testo una voce senza immagine);
- pixel_sha256: sha256 di dimensioni e valori RGB dopo processing.to_rgb (trasparenze su bianco). Due immagini con
  lo stesso pixel_sha256 sono identiche nei pixel decodificati. Un pHash uguale indica solo un candidato duplicato;
- hints: i campi che servono a gruppi, ruoli e copertura (problem_index, docId e documento UCSF, imgname,
  counterpart_idx con opzioni e risposta, meta_info, lingua, contesto e compito di MathVista, sottoinsieme, formula).
I benchmark e le revisioni sono quelli di BENCHMARKS, cioè le sorgenti riservate del catalogo (downloads/catalog_status.tsv).

Uscita nella cartella nuova --out: index.json ({"items": [...], "report": {...}}) e report.json (voci per benchmark e
file, immagini illeggibili, identità del codice).

    srun $RUN_PYTHON -u -B scripts/panel_index.py --out DIR [--workers 4]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

from decontam import LooseFiles, image_from  # noqa: E402  (scripts/decontam.py: lettura delle immagini)
from limite_vl.decontam import phash  # noqa: E402
from limite_vl.processing import to_rgb  # noqa: E402
from limite_vl.provenance import code_identity, write_new_json  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
HF = ROOT / "hf/hub"


def snap(repo: str, rev: str) -> Path:
    return HF / f"datasets--{repo.replace('/', '--')}" / "snapshots" / rev


TEXT = ("question", "query", "problem", "prompt")
# nome -> snapshot, file, colonna id (None = posizione), colonne di testo, colonne d'immagine, campi per i gruppi
BENCHMARKS = {
    "mathvista": dict(repo="AI4Math/MathVista", rev="2b6ad69445fbb5695c9b165475e8decdbeb97747", files="data/*.parquet",
                      id="pid", images=("decoded_image",),
                      hints={"image": "image", "language": "metadata.language", "source": "metadata.source",
                             "context": "metadata.context", "task": "metadata.task", "category": "metadata.category"}),
    "mathverse": dict(repo="AI4Math/MathVerse", rev="3bc86196678bad115a923d2851c6821dbe235939", files="testmini*.parquet",
                      id="sample_index", images=("image",),
                      hints={"problem_index": "problem_index", "version": "problem_version"}),
    "mathvision": dict(repo="MathLLMs/MathVision", rev="2837ddb3f13abaf6b3997c12d80753e5470bd46a", files="data/*.parquet",
                       id="id", images=("decoded_image",), hints={}),
    "wemath": dict(repo="We-Math/We-Math", rev="527c44d4d94c4e3c7c98157460146a8b18c8420a", files="data/*.parquet",
                   id="question number", images=("image_path",), hints={"ID": "ID"}),
    "dynamath": dict(repo="DynaMath/DynaMath_Sample", rev="bc7fcd1ae58caa445d61f6dc0c241b0bb8a7f4c6",
                     files="data/*.parquet", id="id", images=("image", "decoded_image"), hints={"seed": "id"}),
    "geoperception": dict(repo="euclid-multimodal/Geoperception", rev="c34c595649b36c51789f1fab94b2bccd71cb8d80",
                          files="data/*.parquet", id="id", images=("image",), hints={"predicate": "predicate"}),
    "visonlyqa_real": dict(repo="ryokamoi/VisOnlyQA_Eval_Real_v1.1", rev="3747d1370919a6ed0b7ee09896ef6dbffc8aa6d9",
                           files="data/*.parquet", id=None, images=("decoded_image", "image"), hints={}),
    "visonlyqa_synth": dict(repo="ryokamoi/VisOnlyQA_Eval_Synthetic", rev="ef0b5fbcc40a35fdd2947f9b65bed8a8ee99e86d",
                            files="data/*.parquet", id=None, images=("decoded_image", "image"), hints={}),
    "olympiadbench": dict(repo="Hothan/OlympiadBench", rev="91184b52131e7fc9455fef848035173aea8cc01a",
                          files="OlympiadBench/*/*.parquet", id="id", images=tuple(f"image_{i}" for i in range(1, 10)), multi=True,
                          hints={}),
    "ocrbench": dict(repo="echo840/OCRBench", rev="92a54bd1384387c178d5a07140a2d85e0a3d12e1", files="data/*.parquet",
                     id=None, images=("image",), hints={"dataset": "dataset", "question_type": "question_type"}),
    "docvqa": dict(repo="lmms-lab-encoder/DocVQA", rev="539088ef8a8ada01ac8e2e6d4e372586748a265e",
                   files="DocVQA/validation-*.parquet", id="questionId", images=("image",),
                   hints={"docId": "docId", "question_types": "question_types", "ucsf": "ucsf_document_id",
                          "page": "ucsf_document_page_no"}),
    "chartqa": dict(repo="ahmed-masry/ChartQA", rev="af8b6f5c08c95085271561c2a3f9d15f2b5a9031", files="data/*.parquet",
                    id=None, images=("image",), hints={"imgname": "imgname", "type": "type"}),
    "muirbench": dict(repo="MUIRBENCH/MUIRBENCH", rev="4c393cffc985c77d28de3b9045e2e5186920df80", files="data/*.parquet",
                      id="idx", images=("image_list",), multi=True,
                      hints={"counterpart_idx": "counterpart_idx", "task": "task", "image_relation": "image_relation",
                             "image_type": "image_type", "options": "options", "answer": "answer"}),
    "mmstar": dict(repo="Lin-Chen/MMStar", rev="bc98d668301da7b14f648724866e57302778ab27", files="mmstar.parquet",
                   id="index", images=("image",),
                   hints={"category": "category", "l2_category": "l2_category", "source": "meta_info.source",
                          "source_split": "meta_info.split", "image_path": "meta_info.image_path"}),
    "gsm8k": dict(repo="openai/gsm8k", rev="740312add88f781978c0658806c59bc2815b9866", files="main/test-*.parquet",
                  id=None, images=(), hints={}),
    "aime24": dict(repo="HuggingFaceH4/aime_2024", rev="2fe88a2f1091d5048c0f36abc874fb997b3dd99a", files="data/*.parquet",
                   id="id", images=(), hints={}),
    "aime26": dict(repo="MathArena/aime_2026", rev="10b4e45b7a503075d4da8a0d57916a4f06ce6bd2", files="data/*.parquet",
                   id="problem_idx", images=(), hints={}),
    "hmmt25": dict(repo="MathArena/hmmt_feb_2025", rev="6fdc4277120810ff75aa22d2d5489b91f7a262a1",
                   files="data/*.parquet", id="problem_idx", images=(), hints={}),
    "hmmt26": dict(repo="MathArena/hmmt_feb_2026", rev="02fba4f74d8e68e73e66a02d540fd979c05c274c",
                   files="data/*.parquet", id="problem_idx", images=(), hints={}),
    "apex": dict(repo="MathArena/apex-shortlist", rev="f3efdf224ef665f129ddaae37699f6098c65781b", files="data/*.parquet",
                 id="problem_idx", images=(), hints={}),
    "beyondaime": dict(repo="ByteDance-Seed/BeyondAIME", rev="c705198ae1043810b1e1693bd879250b51a7a523",
                       files="data/*.parquet", id=None, images=(), hints={}),
}
JSONL = {  # benchmark di solo testo in JSONL: file, colonna id (None = posizione)
    "aime25": dict(repo="opencompass/AIME2025", rev="a6ad95f611d72cf628a80b58bd0432ef6638f958", files="*.jsonl", id=None),
    "math500": dict(repo="HuggingFaceH4/MATH-500", rev="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", files="*.jsonl",
                    id="unique_id"),
    "ifeval": dict(repo="google/IFEval", rev="966cd89545d6b6acfd7638bc708b98261ca58e84", files="*.jsonl", id="key"),
}
MVMATH = dict(repo="PeijieWang/MV-MATH", rev="b22144b7d9315ce30c7ee8ce1bbaaa7b52fa8674")
UNIMER = dict(repo="wanderkid/UniMER_Dataset", rev="2343ddd963290469da36ca83e3a56c66e068add9", zip="UniMER-Test.zip")
CACHE: dict[str, tuple] = {}


def hash_image(data: bytes) -> tuple[str | None, str | None, list[int] | None]:
    """(pHash esadecimale, sha256 dei pixel normalizzati, dimensioni) di un'immagine; None se non si decodifica."""
    key = hashlib.sha256(data).hexdigest()
    if key not in CACHE:
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                rgb = to_rgb(im)
                pixels = hashlib.sha256(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes()).hexdigest()
                CACHE[key] = (f"{phash(rgb):016x}", pixels, list(rgb.size))
        except Exception:  # noqa: BLE001 - immagine illeggibile: contata nel report
            CACHE[key] = (None, None, None)
    return CACHE[key]


def field(row: dict, path: str):
    value = row
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def images_of(row: dict, columns: tuple, loose: LooseFiles, multi: bool) -> list[bytes]:
    """Immagini dell'elemento. Con multi tutte quelle delle colonne (liste comprese, come `image_list` o `image_1`-
    `image_9`); altrimenti la prima colonna che ne ha una: le colonne alternative (percorso e struct decodificato)
    sono la stessa immagine e non vanno contate due volte."""
    out = []
    for c in columns:
        value = row.get(c)
        if isinstance(value, list):
            found = [d for d in (image_from(v, loose) for v in value) if d]
        else:
            data = image_from(value, loose)
            found = [data] if data else []
        if found and not multi:
            return found[:1]
        out += found
    return out


def entries(name: str, file: str, item: str, text: str, images: list[bytes], hints: dict, counts: Counter) -> list[dict]:
    counts["elementi"] += 1
    base = {"benchmark": name, "file": file, "item": item, "text": text, "hints": hints}
    if not images:
        counts["solo_testo"] += 1
        return [{**base, "id": item, "pos": None, "phash": None, "pixel_sha256": None, "size": None}]
    out = []
    for pos, data in enumerate(images):
        h, pixels, size = hash_image(data)
        counts["immagini"] += 1
        counts["illeggibili"] += h is None
        out.append({**base, "id": item if len(images) == 1 else f"{item}#{pos}", "pos": pos, "phash": h,
                    "pixel_sha256": pixels, "size": size})
    return out


def parquet_task(task: tuple[str, str]) -> dict:
    name, rel = task
    spec = BENCHMARKS[name]
    folder = snap(spec["repo"], spec["rev"])
    loose = LooseFiles(folder)
    counts, items = Counter(), []
    path = folder / rel
    table = pq.read_table(path)
    cols = table.column_names
    text_cols = [c for c in TEXT if c in cols]
    for i, row in enumerate(table.to_pylist()):
        ident = row.get(spec["id"]) if spec["id"] else None
        # unico nel benchmark: lo stesso id può ripetersi fra file (MathVision testmini e test, varianti di DynaMath)
        item = f"{Path(rel).stem}:{ident if ident is not None else i}"
        hints = {"id": None if ident is None else str(ident), **{k: field(row, v) for k, v in spec["hints"].items()}}
        if name == "olympiadbench":
            hints["subset"] = Path(rel).parent.name
        text = " ".join(str(row[c]) for c in text_cols if row.get(c))
        images = images_of(row, spec["images"], loose, spec.get("multi", False))
        items += entries(name, rel, item, text, images, hints, counts)
    return {"benchmark": name, "file": rel, "items": items, "counts": dict(counts)}


def jsonl_task(task: tuple[str, str]) -> dict:
    name, rel = task
    spec = JSONL[name]
    counts, items = Counter(), []
    path = snap(spec["repo"], spec["rev"]) / rel
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        ident = row.get(spec["id"]) if spec["id"] else None
        item = f"{Path(rel).stem}:{ident if ident is not None else i}"
        text = " ".join(str(row[c]) for c in TEXT if row.get(c))
        items += entries(name, rel, item, text, [], {"id": None if ident is None else str(ident)}, counts)
    return {"benchmark": name, "file": rel, "items": items, "counts": dict(counts)}


def mvmath_task(_: tuple) -> dict:
    folder = snap(MVMATH["repo"], MVMATH["rev"])
    loose = LooseFiles(folder)
    counts, items = Counter(), []
    for i, row in enumerate(json.loads((folder / "MV-MATH.json").read_text(encoding="utf-8"))):
        paths = row.get("input_image") or []
        images = [d for d in (loose.read(p) for p in paths) if d]
        counts["percorsi_non_trovati"] += len(paths) - len(images)
        item = f"MV-MATH:{row.get('problem_id', i)}"
        hints = {k: row.get(k) for k in ("answer_type", "subject", "image_relavance", "is_multi_img", "difficulty")}
        items += entries("mvmath", "MV-MATH.json", item, str(row.get("question") or ""), images, hints, counts)
    return {"benchmark": "mvmath", "file": "MV-MATH.json", "items": items, "counts": dict(counts)}


def unimer_task(task: tuple) -> dict:
    """UniMER-Test: `UniMER-Test/<sub>/<indice>.png` ha la formula della riga <indice> di `<sub>.txt` (shard_unimer.py).
    La chiave `<sub>-<indice a 7 cifre>` è quella degli shard di test."""
    sub = task[1]
    path = snap(UNIMER["repo"], UNIMER["rev"]) / UNIMER["zip"]
    counts, items = Counter(), []
    with zipfile.ZipFile(path) as zf:
        lines = zf.read(f"UniMER-Test/{sub}.txt").decode("utf-8").split("\n")
        infos = [i for i in zf.infolist() if i.filename.startswith(f"UniMER-Test/{sub}/") and i.filename.endswith(".png")]
        for info in sorted(infos, key=lambda i: i.header_offset):
            stem = Path(info.filename).stem
            if not stem.isdigit():
                continue
            index = int(stem)
            label = lines[index].strip() if index < len(lines) else ""
            items += entries("unimer_test", f"{sub}", f"{sub}-{index:07d}", label, [zf.read(info)],
                             {"subset": sub, "formula": label}, counts)
    return {"benchmark": "unimer_test", "file": sub, "items": items, "counts": dict(counts)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("panel_index.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota")
    tasks, missing = [], []
    for name, spec in BENCHMARKS.items():
        folder = snap(spec["repo"], spec["rev"])
        files = sorted(str(p.relative_to(folder)) for p in folder.glob(spec["files"]))
        if not files:
            missing.append(f"{name}: nessun file {spec['files']} in {folder}")
        tasks += [(parquet_task, (name, f)) for f in files]
    for name, spec in JSONL.items():
        folder = snap(spec["repo"], spec["rev"])
        files = sorted(str(p.relative_to(folder)) for p in folder.glob(spec["files"]))
        if not files:
            missing.append(f"{name}: nessun file {spec['files']} in {folder}")
        tasks += [(jsonl_task, (name, f)) for f in files]
    if not (snap(MVMATH["repo"], MVMATH["rev"]) / "MV-MATH.json").is_file():
        missing.append("mvmath: MV-MATH.json mancante")
    tasks.append((mvmath_task, ("mvmath",)))
    tasks += [(unimer_task, ("unimer_test", sub)) for sub in ("spe", "cpe", "sce", "hwe")]
    if missing:
        sys.exit("snapshot mancanti, nessun download nei job: " + "; ".join(missing))
    args.out.mkdir(parents=True, exist_ok=True)
    items, report = [], {}
    with Pool(args.workers) as pool:
        results = [pool.apply_async(fn, (t,)) for fn, t in tasks]
        for r in results:
            res = r.get()
            items += res["items"]
            report.setdefault(res["benchmark"], {})[res["file"]] = res["counts"]
            print(f"{res['benchmark']} / {res['file']}: {res['counts']}", flush=True)
    totals = {b: dict(sum((Counter(c) for c in files.values()), Counter())) for b, files in report.items()}
    write_new_json(args.out / "index.json", {"items": items, "report": totals})
    write_new_json(args.out / "report.json", {"schema": "panel-index/1", "job_id": os.environ.get("SLURM_JOB_ID"),
                                              "code": code_identity(), "per_benchmark": totals, "per_file": report,
                                              "voci": len(items)})
    print(json.dumps(totals, ensure_ascii=False))


if __name__ == "__main__":
    main()
