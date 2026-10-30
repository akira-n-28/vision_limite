#!/usr/bin/env python3
"""Inventario dei benchmark locali per il pannello di valutazione (docs/PANNELLO_VALUTAZIONE.md). Solo in un job SLURM
CPU, in sola lettura: non estrae archivi e non scrive fuori da --out.

Per ogni sorgente `riservata` di downloads/catalog_status.tsv, più UniMER (UniMER-Test.zip), legge lo snapshot nella
cache HF alla revisione del catalogo:
- file presenti contro il manifest del download (downloads/manifests/*.tsv) e esito della verifica degli hash
  (outputs/verify/report-<job>.tsv);
- parquet: righe, colonne con tipo, valori nulli o vuoti, valori distinti; per le colonne con pochi valori (split,
  versione, tipo, sorgente) i conteggi; per le colonne d'immagine (struct con bytes, binary) quante righe ne hanno una;
  per le struct (es. `metadata` di MathVista) gli stessi riassunti per campo;
- JSON e JSONL: numero di record e riassunto dei campi; zip: numero di membri per estensione e per cartella, letti
  dall'indice dello zip senza estrarre;
- data/eval-datasets: le copie usate dal valutatore testuale.
Per gli indici di decontaminazione (data/decontam/p6/index.json, data/decontam/f1eval/index.json): elementi per
benchmark e per file, con o senza pHash e testo, e il report dell'indice.

Uscita: --out/inventory.json (nuovo); a schermo un riassunto per sorgente.

    srun $RUN_PYTHON -u -B scripts/panel_inventory.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.provenance import code_identity, write_new_json  # noqa: E402

LOW = 40  # sotto questo numero di valori distinti si riportano i conteggi
EXTRA = {"wanderkid/UniMER_Dataset": "UniMER-Test.zip"}  # sorgente candidata con la parte di test riservata


def summarize_values(values: list) -> dict:
    """Riassunto di una lista di valori scalari: nulli, vuoti, distinti, conteggi se pochi."""
    out = {"n": len(values), "nulli": sum(v is None for v in values),
           "vuoti": sum(v in ("", [], {}) for v in values)}
    hashable = [json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for v in values]
    distinct = Counter(hashable)
    out["distinti"] = len(distinct)
    if len(distinct) <= LOW:
        out["conteggi"] = {str(k): c for k, c in distinct.most_common()}
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if nums and len(nums) == len(values) - out["nulli"]:
        out["min"], out["max"] = min(nums), max(nums)
    return out


def is_image_type(t: pa.DataType) -> bool:
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return True
    return pa.types.is_struct(t) and any(f.name == "bytes" for f in t)


def summarize_column(arr: pa.ChunkedArray | pa.Array) -> dict:
    t = arr.type
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    out: dict = {"tipo": str(t)}
    if is_image_type(t):
        if pa.types.is_struct(t):
            data = arr.field("bytes")
            out["immagini"] = sum(1 for v in data.to_pylist() if v)
        else:
            out["immagini"] = sum(1 for v in arr.to_pylist() if v)
        out["nulli"] = arr.null_count
        return out
    if pa.types.is_struct(t):
        out["campi"] = {f.name: summarize_column(arr.field(f.name)) for f in t}
        return out
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        values = arr.to_pylist()
        lengths = [len(v) for v in values if v is not None]
        out.update(nulli=sum(v is None for v in values), lunghezza_min=min(lengths, default=0),
                   lunghezza_max=max(lengths, default=0))
        if pa.types.is_struct(t.value_type) and any(f.name == "bytes" for f in t.value_type):
            out["immagini"] = sum(len(v) for v in values if v)
        return out
    out.update(summarize_values(arr.to_pylist()))
    return out


def summarize_parquet(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    out = {"righe": pf.metadata.num_rows, "colonne": {}}
    for name in pf.schema_arrow.names:
        try:
            out["colonne"][name] = summarize_column(pf.read(columns=[name]).column(name))
        except Exception as exc:  # noqa: BLE001 - colonna illeggibile: lo si registra
            out["colonne"][name] = {"errore": f"{type(exc).__name__}: {exc}"[:200]}
    return out


def summarize_records(records: list) -> dict:
    out = {"record": len(records)}
    if records and isinstance(records[0], dict):
        keys = Counter(k for r in records if isinstance(r, dict) for k in r)
        out["campi"] = {}
        for k, n in keys.most_common():
            values = [r.get(k) for r in records if isinstance(r, dict)]
            scalars = [v for v in values if not isinstance(v, (list, dict))]
            entry = {"presente": n}
            if len(scalars) == len(values):
                entry.update(summarize_values(values))
            elif all(isinstance(v, dict) for v in values if v is not None):
                sub = {}
                for kk in sorted({kk for v in values if v for kk in v})[:30]:
                    vv = [v.get(kk) if v else None for v in values]
                    if all(not isinstance(x, (list, dict)) for x in vv):
                        sub[kk] = summarize_values(vv)
                entry["sottocampi"] = sub
            out["campi"][k] = entry
    return out


def summarize_json(path: Path) -> dict:
    if path.stat().st_size > 50 * 2 ** 20:
        return {"saltato": "più di 50 MiB"}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return summarize_records(data)
    if isinstance(data, dict):
        values = list(data.values())
        if values and all(isinstance(v, dict) for v in values):
            return {"chiavi": len(data), **summarize_records(values)}
        return {"chiavi": len(data), "esempio_chiavi": list(data)[:20]}
    return {"tipo": type(data).__name__}


def summarize_jsonl(path: Path) -> dict:
    if path.stat().st_size > 50 * 2 ** 20:
        with open(path, encoding="utf-8") as fh:
            return {"righe": sum(1 for _ in fh), "nota": "più di 50 MiB: solo righe contate"}
    with open(path, encoding="utf-8") as fh:
        return summarize_records([json.loads(line) for line in fh if line.strip()])


def summarize_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
    by_ext = Counter(Path(i.filename).suffix.lower() for i in infos)
    by_dir = Counter("/".join(i.filename.split("/")[:2]) if i.filename.count("/") >= 2 else i.filename.split("/")[0]
                     for i in infos)
    return {"membri": len(infos), "byte_non_compressi": sum(i.file_size for i in infos),
            "per_estensione": dict(by_ext.most_common()),
            "per_cartella": dict(by_dir.most_common(60)) if len(by_dir) <= 60 else
            {"cartelle": len(by_dir), "prime": dict(by_dir.most_common(20))}}


def summarize_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return summarize_parquet(path)
        if suffix == ".json":
            return summarize_json(path)
        if suffix == ".jsonl":
            return summarize_jsonl(path)
        if suffix == ".zip":
            return summarize_zip(path)
    except Exception as exc:  # noqa: BLE001 - file illeggibile: lo si registra, l'inventario continua
        return {"errore": f"{type(exc).__name__}: {exc}"[:300]}
    return {"byte": path.stat().st_size}


def catalog_rows() -> list[dict]:
    rows = []
    for line in (ROOT / "downloads/catalog_status.tsv").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 9 or f[0] == "gruppo":
            continue
        rows.append({"gruppo": f[0], "repo": f[1], "revisione": f[2], "disponibilita": f[3], "verifica": f[4],
                     "derivati": f[5], "ruolo": f[6], "ammissione": f[7], "note": f[8]})
    return rows


def manifest_files(repo: str, revision: str) -> dict[str, int]:
    files = {}
    prefix = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
    for m in sorted((ROOT / "downloads/manifests").glob("*.tsv")):
        for line in m.read_text(encoding="utf-8").splitlines():
            f = line.split("\t")
            if f and f[0].startswith(prefix):
                files[f[0][len(prefix):]] = int(f[2])
    return files


def verify_status(job: str, repo: str) -> dict:
    out = Counter()
    for j in [x.strip() for x in job.split(",") if x.strip().isdigit()]:
        report = ROOT / f"outputs/verify/report-{j}.tsv"
        if not report.is_file():
            out[f"report-{j} mancante"] += 1
            continue
        for line in report.read_text(encoding="utf-8").splitlines()[1:]:
            f = line.split("\t")
            if len(f) >= 5 and f[4].startswith(repo + "@"):  # etichetta repo@revisione:file del manifest
                out[f[0]] += 1
    return dict(out)


def index_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    per = Counter()
    for it in data["items"]:
        per[(it["benchmark"], it.get("file"), bool(it.get("phash")), bool(it.get("text")))] += 1
    rows = [{"benchmark": b, "file": f, "phash": h, "testo": t, "elementi": n} for (b, f, h, t), n in sorted(per.items())]
    fields = sorted({k for it in data["items"][:1000] for k in it})
    return {"elementi": len(data["items"]), "campi": fields, "report": data.get("report"), "per_file": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("panel_inventory.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)
    sources = []
    for row in catalog_rows():
        if "riservata" not in row["ammissione"] or "/" not in row["repo"]:
            continue
        repo, rev = row["repo"], row["revisione"]
        snap = ROOT / "hf/hub" / f"datasets--{repo.replace('/', '--')}" / "snapshots" / rev
        expected = manifest_files(repo, rev)
        present = {str(p.relative_to(snap)): p for p in sorted(snap.rglob("*")) if p.is_file()} if snap.is_dir() else {}
        only = EXTRA.get(repo)
        entry = {**row, "snapshot": str(snap.relative_to(ROOT)), "snapshot_presente": snap.is_dir(),
                 "file_del_manifest": len(expected), "file_presenti": len(present),
                 "mancanti_rispetto_al_manifest": sorted(set(expected) - set(present)),
                 "verifica_hash": verify_status(row["verifica"], repo), "file": {}}
        for rel, p in present.items():
            if only and rel != only:
                continue
            entry["file"][rel] = {"byte": p.stat().st_size, **summarize_file(p)}
        sources.append(entry)
        print(f"{repo}@{rev[:8]}: {len(present)} file presenti su {len(expected)} del manifest, verifica "
              f"{entry['verifica_hash']}", flush=True)
    eval_copies = {}
    base = ROOT / "data/eval-datasets"
    for p in sorted(base.rglob("*")) if base.is_dir() else []:
        if p.is_file():
            eval_copies[str(p.relative_to(ROOT))] = {"byte": p.stat().st_size, **summarize_file(p)}
    indexes = {}
    for rel in ("data/decontam/p6/index.json", "data/decontam/f1eval/index.json"):
        p = ROOT / rel
        if p.is_file():
            indexes[rel] = index_summary(p)
            print(f"{rel}: {indexes[rel]['elementi']} elementi", flush=True)
    write_new_json(args.out / "inventory.json", {"schema": "panel-inventory/1", "job_id": os.environ.get("SLURM_JOB_ID"),
                                                 "code": code_identity(), "sorgenti": sources,
                                                 "copie_del_valutatore": eval_copies, "indici": indexes})
    for s in sources:
        for rel, f in s["file"].items():
            if "righe" in f or "record" in f or "membri" in f:
                print(f"  {s['repo']}: {rel}: righe {f.get('righe', f.get('record', f.get('membri')))}", flush=True)


if __name__ == "__main__":
    main()
