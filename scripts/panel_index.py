#!/usr/bin/env python3
"""Indice del pannello di valutazione (docs/PANNELLO_VALUTAZIONE.md): una voce per immagine, con pHash, sha256 dei
pixel normalizzati e testo, più le voci di solo testo. Solo in un job SLURM CPU; snapshot in sola lettura.

Riusa la lettura delle immagini di scripts/decontam.py (struct {bytes, path}, colonne binarie, percorsi in cartelle o
zip) e la convenzione delle immagini di limite_vl.decontam (IMAGE_CONVENTION: processing.to_rgb, trasparenze su
bianco), dichiarata in index.json e in report.json. Ogni voce ha i campi che legge `decontam.py scan` (benchmark,
file, id, phash, text) e:
- item: l'elemento (domanda o problema) a cui appartiene l'immagine, unico nel benchmark; pos: posizione
  dell'immagine nell'elemento (gli elementi con più immagini, come MUIRBench, MV-MATH e OlympiadBench, hanno una voce
  per immagine; quelli di solo testo una voce senza immagine);
- pixel_sha256: sha256 di dimensioni e valori RGB dopo la normalizzazione. Due immagini con lo stesso pixel_sha256
  sono identiche nei pixel decodificati. Un pHash uguale indica solo un candidato duplicato;
- hints: i campi che servono a gruppi, ruoli e copertura (problem_index, docId e documento UCSF, imgname,
  counterpart_idx con opzioni e risposta, meta_info, lingua, contesto e compito di MathVista, sottoinsieme, formula).
I benchmark e le revisioni sono quelli di BENCHMARKS, cioè le sorgenti riservate del catalogo (downloads/catalog_status.tsv).
I file di ChartQA sono elencati uno per uno (val e test): un train scaricato in futuro nella stessa cache non entra.

Gli stessi lettori (source_rows) costruiscono l'indice e ritrovano l'immagine di una voce (resolve_entries,
--verify): ogni voce deve ritrovare esattamente il proprio elemento e la propria posizione, con lo stesso
sha256 dei pixel; un elemento assente o ripetuto, una posizione mancante o pixel diversi sono errori, senza
alternative.

Uscite:
- costruzione, nella cartella nuova --out: index.json ({"convenzione_immagini", "schema", "items", "report"}) e
  report.json (voci per benchmark e file, immagini illeggibili, identità del codice);
- verifica, --verify DIR: rilegge ogni immagine dell'indice in DIR e scrive DIR/verifica.json (file nuovo).

    srun $RUN_PYTHON -u -B scripts/panel_index.py --out DIR [--workers 4]
    srun $RUN_PYTHON -u -B scripts/panel_index.py --verify DIR [--workers 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

from decontam import LooseFiles, image_from  # noqa: E402  (scripts/decontam.py: lettura delle immagini)
from limite_vl.decontam import IMAGE_CONVENTION, hash_image_bytes, index_convention  # noqa: E402
from limite_vl.provenance import code_identity, write_new_json  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
HF = ROOT / "hf/hub"
SCHEMA = "panel-index/2"


def snap(repo: str, rev: str) -> Path:
    return HF / f"datasets--{repo.replace('/', '--')}" / "snapshots" / rev


TEXT = ("question", "query", "problem", "prompt")
# nome -> snapshot, file (glob, o tupla di file esatti), colonna id (None = posizione), colonne d'immagine, campi per
# i gruppi
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
    # solo i due split autorizzati, uno per uno: il train (candidato di training) non deve entrare nel pannello
    "chartqa": dict(repo="ahmed-masry/ChartQA", rev="af8b6f5c08c95085271561c2a3f9d15f2b5a9031",
                    files=("data/val-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"),
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
MVMATH = dict(repo="PeijieWang/MV-MATH", rev="b22144b7d9315ce30c7ee8ce1bbaaa7b52fa8674", file="MV-MATH.json")
UNIMER = dict(repo="wanderkid/UniMER_Dataset", rev="2343ddd963290469da36ca83e3a56c66e068add9", zip="UniMER-Test.zip",
              subsets=("spe", "cpe", "sce", "hwe"))
CACHE: dict[str, tuple] = {}


def hash_image(data: bytes) -> tuple[str | None, str | None, list[int] | None]:
    """(pHash esadecimale, sha256 dei pixel, dimensioni) nella convenzione dell'indice; None se non si decodifica."""
    key = hashlib.sha256(data).hexdigest()
    if key not in CACHE:
        h, pixels, size = hash_image_bytes(data, IMAGE_CONVENTION)
        CACHE[key] = (None if h is None else f"{h:016x}", pixels, size)
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


def benchmark_files(name: str) -> tuple[list[str], list[str]]:
    """(file dell'indice, file mancanti) di un benchmark. Una tupla di file esatti non cambia se nella stessa cache
    compaiono altri split; un glob vale solo dove il repository non ha file di training."""
    if name == "mvmath":
        present = (snap(MVMATH["repo"], MVMATH["rev"]) / MVMATH["file"]).is_file()
        return ([MVMATH["file"]], []) if present else ([], [MVMATH["file"]])
    if name == "unimer_test":
        present = (snap(UNIMER["repo"], UNIMER["rev"]) / UNIMER["zip"]).is_file()
        return (list(UNIMER["subsets"]), []) if present else ([], [UNIMER["zip"]])
    spec = BENCHMARKS.get(name) or JSONL[name]
    folder = snap(spec["repo"], spec["rev"])
    if isinstance(spec["files"], tuple):
        return ([f for f in spec["files"] if (folder / f).is_file()],
                [f for f in spec["files"] if not (folder / f).is_file()])
    files = sorted(str(p.relative_to(folder)) for p in folder.glob(spec["files"]))
    return files, ([] if files else [spec["files"]])


# Lettori: per ogni elemento di un file (item, testo, immagini in ordine, hints, riferimenti d'immagine non trovati).
# Costruzione dell'indice e risoluzione delle voci usano gli stessi.
def parquet_rows(name: str, rel: str):
    spec = BENCHMARKS[name]
    folder = snap(spec["repo"], spec["rev"])
    loose = LooseFiles(folder)
    table = pq.read_table(folder / rel)
    text_cols = [c for c in TEXT if c in table.column_names]
    for i, row in enumerate(table.to_pylist()):
        ident = row.get(spec["id"]) if spec["id"] else None
        # unico nel benchmark: lo stesso id può ripetersi fra file (MathVision testmini e test, varianti di DynaMath)
        item = f"{Path(rel).stem}:{ident if ident is not None else i}"
        hints = {"id": None if ident is None else str(ident), **{k: field(row, v) for k, v in spec["hints"].items()}}
        if name == "olympiadbench":
            hints["subset"] = Path(rel).parent.name
        text = " ".join(str(row[c]) for c in text_cols if row.get(c))
        yield item, text, images_of(row, spec["images"], loose, spec.get("multi", False)), hints, 0


def jsonl_rows(name: str, rel: str):
    spec = JSONL[name]
    path = snap(spec["repo"], spec["rev"]) / rel
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        ident = row.get(spec["id"]) if spec["id"] else None
        item = f"{Path(rel).stem}:{ident if ident is not None else i}"
        text = " ".join(str(row[c]) for c in TEXT if row.get(c))
        yield item, text, [], {"id": None if ident is None else str(ident)}, 0


def mvmath_rows(rel: str = MVMATH["file"]):
    folder = snap(MVMATH["repo"], MVMATH["rev"])
    loose = LooseFiles(folder)
    for i, row in enumerate(json.loads((folder / rel).read_text(encoding="utf-8"))):
        paths = row.get("input_image") or []
        images = [d for d in (loose.read(p) for p in paths) if d]
        item = f"MV-MATH:{row.get('problem_id', i)}"
        hints = {k: row.get(k) for k in ("answer_type", "subject", "image_relavance", "is_multi_img", "difficulty")}
        yield item, str(row.get("question") or ""), images, hints, len(paths) - len(images)


def unimer_rows(sub: str):
    """UniMER-Test: `UniMER-Test/<sub>/<indice>.png` ha la formula della riga <indice> di `<sub>.txt` (shard_unimer.py).
    La chiave `<sub>-<indice a 7 cifre>` è quella degli shard di test."""
    path = snap(UNIMER["repo"], UNIMER["rev"]) / UNIMER["zip"]
    with zipfile.ZipFile(path) as zf:
        lines = zf.read(f"UniMER-Test/{sub}.txt").decode("utf-8").split("\n")
        infos = [i for i in zf.infolist() if i.filename.startswith(f"UniMER-Test/{sub}/") and i.filename.endswith(".png")]
        for info in sorted(infos, key=lambda i: i.header_offset):
            stem = Path(info.filename).stem
            if not stem.isdigit():
                continue
            index = int(stem)
            label = lines[index].strip() if index < len(lines) else ""
            yield f"{sub}-{index:07d}", label, [zf.read(info)], {"subset": sub, "formula": label}, 0


def source_rows(benchmark: str, file: str):
    if benchmark == "mvmath":
        return mvmath_rows(file)
    if benchmark == "unimer_test":
        return unimer_rows(file)
    if benchmark in JSONL:
        return jsonl_rows(benchmark, file)
    if benchmark in BENCHMARKS:
        return parquet_rows(benchmark, file)
    raise ValueError(f"benchmark sconosciuto: {benchmark}")


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


def file_task(task: tuple[str, str]) -> dict:
    name, rel = task
    counts, items = Counter(), []
    for item, text, images, hints, missing in source_rows(name, rel):
        if name == "mvmath":
            counts["percorsi_non_trovati"] += missing
        items += entries(name, rel, item, text, images, hints, counts)
    return {"benchmark": name, "file": rel, "items": items, "counts": dict(counts)}


# ---------------------------------------------------------------- ritrovare le immagini delle voci
def resolve_file(task: tuple) -> dict:
    """Rilegge un file e ritrova le immagini delle voci richieste. task = (benchmark, file, {item: {pos: (posizione
    nella lista delle voci, phash, pixel_sha256)}}, convenzione, tenere i byte). Esito per voce: ok, pixel diversi,
    posizione mancante, elemento assente o ripetuto; nessuna alternativa."""
    benchmark, file, wanted, convention, keep = task
    found, errors, counts = {}, [], Counter()
    seen = Counter()
    for item, _, images, _, _ in source_rows(benchmark, file):
        if item not in wanted:
            continue
        seen[item] += 1
        if seen[item] > 1:
            errors.append(f"{benchmark}/{file}: elemento {item} ripetuto nella sorgente")
            continue
        positions = wanted[item]
        if len(images) != len(positions):
            errors.append(f"{benchmark}/{file}: {item} ha {len(images)} immagini nella sorgente e {len(positions)} voci")
        for pos, (where, ph, px) in positions.items():
            if pos >= len(images):
                counts["posizione_mancante"] += 1
                errors.append(f"{benchmark}/{file}: {item} posizione {pos} mancante")
                continue
            h, pixels, _ = hash_image_bytes(images[pos], convention)
            same = h is not None and f"{h:016x}" == ph and (px is None or pixels == px)
            if same:
                counts["ok"] += 1
                if keep:
                    found[where] = images[pos]
            else:
                counts["pixel_diversi"] += 1
                errors.append(f"{benchmark}/{file}: {item} posizione {pos}: immagine diversa dalla voce")
    for item, positions in wanted.items():
        if not seen[item]:
            counts["elemento_assente"] += len(positions)
            errors.append(f"{benchmark}/{file}: elemento {item} assente dalla sorgente")
    return {"benchmark": benchmark, "file": file, "found": found, "errors": errors, "counts": dict(counts)}


def _wanted(items: list[dict], convention: str, keep: bool) -> list[tuple]:
    by_file: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for where, it in enumerate(items):
        if it.get("pos") is None:
            continue
        slot = by_file[(it["benchmark"], it["file"])][it["item"]]
        if it["pos"] in slot:
            raise ValueError(f"voce ripetuta nell'indice: {it['benchmark']}/{it['item']} posizione {it['pos']}")
        slot[it["pos"]] = (where, it["phash"], it.get("pixel_sha256"))
    return [(b, f, dict(w), convention, keep) for (b, f), w in sorted(by_file.items())]


def resolve_entries(items: list[dict], convention: str, workers: int = 1) -> tuple[list[bytes | None], list[str]]:
    """Byte dell'immagine di ogni voce, nello stesso ordine (None per le voci di solo testo), ed errori. Con errori
    il chiamante si ferma: non c'è un ripiego."""
    tasks = _wanted(items, convention, keep=True)
    out: list[bytes | None] = [None] * len(items)
    errors: list[str] = []
    if workers <= 1:
        results = [resolve_file(t) for t in tasks]
    else:
        with Pool(workers) as pool:
            results = list(pool.imap_unordered(resolve_file, tasks))
    for res in results:
        errors += res["errors"]
        for where, data in res["found"].items():
            out[where] = data
    return out, errors


def verify_index(folder: Path, workers: int) -> dict:
    loaded = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    convention = index_convention(loaded)
    items = loaded["items"]
    tasks = _wanted(items, convention, keep=False)
    counts, errors, per_file = Counter(), [], {}
    with Pool(max(1, workers)) as pool:
        for res in pool.imap_unordered(resolve_file, tasks):
            counts.update(res["counts"])
            errors += res["errors"]
            per_file[f"{res['benchmark']}/{res['file']}"] = res["counts"]
            print(f"{res['benchmark']} / {res['file']}: {res['counts']}", flush=True)
    voices = sum(1 for it in items if it.get("pos") is not None)
    return {"schema": "panel-index-verify/1", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
            "convenzione_immagini": convention, "voci_con_immagine": voices, "esiti": dict(counts),
            "errori": len(errors), "primi_errori": errors[:100], "per_file": dict(sorted(per_file.items())),
            "verificato": counts["ok"] == voices and not errors}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="cartella nuova dell'indice")
    ap.add_argument("--verify", type=Path, help="cartella di un indice da verificare rileggendo ogni immagine")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("panel_index.py va eseguito in un job SLURM")
    if (args.out is None) == (args.verify is None):
        sys.exit("indicare --out oppure --verify")
    if args.verify:
        report = verify_index(args.verify, args.workers)
        write_new_json(args.verify / "verifica.json", report)
        print(json.dumps({k: report[k] for k in ("voci_con_immagine", "esiti", "errori", "verificato")}))
        sys.exit(0 if report["verificato"] else 1)
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota")
    tasks, missing = [], []
    for name in [*BENCHMARKS, *JSONL, "mvmath", "unimer_test"]:
        files, absent = benchmark_files(name)
        missing += [f"{name}: {f}" for f in absent]
        tasks += [(name, f) for f in files]
    if missing:
        sys.exit("snapshot o file mancanti, nessun download nei job: " + "; ".join(missing))
    args.out.mkdir(parents=True, exist_ok=True)
    items, report = [], {}
    with Pool(args.workers) as pool:
        for res in pool.imap(file_task, tasks):
            items += res["items"]
            report.setdefault(res["benchmark"], {})[res["file"]] = res["counts"]
            print(f"{res['benchmark']} / {res['file']}: {res['counts']}", flush=True)
    totals = {b: dict(sum((Counter(c) for c in files.values()), Counter())) for b, files in report.items()}
    write_new_json(args.out / "index.json", {"schema": SCHEMA, "convenzione_immagini": IMAGE_CONVENTION,
                                             "items": items, "report": totals})
    write_new_json(args.out / "report.json", {"schema": SCHEMA, "convenzione_immagini": IMAGE_CONVENTION,
                                              "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
                                              "per_benchmark": totals, "per_file": report, "voci": len(items)})
    print(json.dumps(totals, ensure_ascii=False))


if __name__ == "__main__":
    main()
