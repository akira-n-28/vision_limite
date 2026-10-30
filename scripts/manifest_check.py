#!/usr/bin/env python3
"""Verifiche CPU del manifest delle ammissioni dei sintetici v4 e dei loader che lo applicano (S1 §§6, 8-9; S2 §2.7).
Solo in un job SLURM CPU; manifest e shard in sola lettura, scrive solo nella cartella nuova --out.

1. Integrità: manifest.load (sha256 di records.jsonl.gz, record_id ricalcolati, un solo split per gruppo, conteggi per
   shard); shard su disco con sha256 e dimensione del manifest; file derivati (keys/) con i loro sha256 e uguali alle
   chiavi del manifest.
2. Split: ricalcolato due volte dai record con la regola, la quota e il seme del manifest (manifest.assign_splits) e
   uguale a quello scritto; con un altro seme è diverso. I legami esatti ricalcolati dagli hash del manifest (pixel,
   stato della scena, coppia) danno gli stessi gruppi; nessun gruppo è in due split; dev per famiglia rispetto alla
   quota.
3. Loader: sorgenti di stream.make_sources con il manifest per train e dev (shard train) e per test e coppie (split
   test). Ogni sorgente si legge per un'epoca con --workers worker di SourceReader: le chiavi consegnate devono essere
   esattamente quelle ammesse dello split, ciascuna una volta, nessuna esclusa o in quarantena, e i gruppi letti non
   devono comparire in altri split. Le domande costruite (f1c.pick_questions, come data.synth_qa_sample) hanno gli
   stessi digest e gli stessi conteggi per famiglia e compito attesi dal manifest. validation_samples restituisce lo
   split dev. Un breve MixedRowStream col processor di Violetto consegna solo chiavi di train e conta gli esempi.
4. Arresti espliciti: manifest mancante, records.jsonl.gz alterato, sha256 di uno shard diverso, gruppo spostato in un
   altro split, record_id alterato, shard v3 col manifest v4, shard v4 senza manifest, split senza record ammessi,
   split sconosciuto, manifest senza split. Ognuno deve dare ManifestError; le copie alterate stanno in una cartella
   temporanea del job.

Uscite: --out/report.json con esiti e conteggi; il job esce con errore se un controllo fallisce.

    srun $RUN_PYTHON -u -B scripts/manifest_check.py --manifest DIR --shards DIR --out DIR [--workers 2]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.f1c import pick_questions  # noqa: E402
from limite_vl.provenance import code_identity, sha256_file, write_new_json  # noqa: E402
from limite_vl.stream import MixedRowStream, SourceReader, make_sources, validation_samples  # noqa: E402

VIOLETTO = ROOT / "hf/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
SOURCES: dict = {}  # nome -> Source, ereditate dai processi del pool
RESULTS: list[dict] = []


def check(name: str, ok: bool, detail=None) -> bool:
    RESULTS.append({"controllo": name, "esito": bool(ok), "dettaglio": detail})
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail is not None else ""), flush=True)
    return ok


def specs(manifest: Path, shards: Path) -> dict[str, dict]:
    train = str(shards / "train" / "*.tar")
    return {"train": {"name": "synth-train", "kind": "synth_qa", "shards": train, "weight": 1.0,
                      "manifest": str(manifest), "split": "train"},
            "dev": {"name": "synth-dev", "kind": "synth_qa", "shards": train, "weight": 1.0,
                    "manifest": str(manifest), "split": "dev"},
            "test": {"name": "synth-test", "kind": "synth_qa", "shards": str(shards / "test" / "*.tar"), "weight": 1.0,
                     "manifest": str(manifest), "split": "test"},
            "coppie": {"name": "synth-coppie", "kind": "synth_qa", "shards": str(shards / "pairs" / "*.tar"),
                       "weight": 1.0, "manifest": str(manifest), "split": "test"}}


def read_epoch(task: tuple[str, int, int]) -> dict:
    """Un'epoca di un worker di SourceReader: chiavi consegnate, contatori del reader, QA costruite."""
    name, worker, n_workers = task
    source = SOURCES[name]
    reader = SourceReader(source, seed=0, worker=worker, n_workers=n_workers, val_count=0)
    keys, picked = [], {}
    digests = mf.Digests()
    while True:
        key, files = reader.next()
        if reader.epoch > 0:  # primo campione dell'epoca successiva: l'epoca è finita
            break
        keys.append(key)
        meta = json.loads(files["json"])
        items = list(meta["qa"]) if meta["qa"] else [meta["pair"]]
        picked[key] = [[q["type"], mf.qa_digest(digests, q)] for q in pick_questions(key, items)]
        problem = source.admission.check(key, files)
        if problem:
            picked[key].append(["problema", problem])
    return {"name": name, "worker": worker, "keys": keys, "picked": picked, "counts": dict(reader.counts)}


def expect_error(name: str, fn) -> None:
    try:
        fn()
    except mf.ManifestError as exc:
        check(f"arresto: {name}", True, str(exc)[:160])
        return
    except Exception as exc:  # noqa: BLE001 - un errore di altro tipo non è l'arresto esplicito atteso
        check(f"arresto: {name}", False, f"{type(exc).__name__}: {exc}"[:200])
        return
    check(f"arresto: {name}", False, "nessun errore")


def rewrite_records(folder: Path, edit) -> None:
    """Riscrive records.jsonl.gz di una copia del manifest con edit(lista di record) e aggiorna sha256 e righe in
    manifest.json, così l'unica incoerenza è quella introdotta da edit."""
    meta = json.loads((folder / "manifest.json").read_text())
    lines = gzip.decompress((folder / "records.jsonl.gz").read_bytes()).decode().splitlines()
    rows = [json.loads(x) for x in lines]
    edit(rows)
    data = gzip.compress("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode(),
                         mtime=0)
    (folder / "records.jsonl.gz").write_bytes(data)
    meta["records"].update(sha256=hashlib.sha256(data).hexdigest(), righe=len(rows))
    (folder / "manifest.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--v3-shards", type=Path, default=ROOT / "data/shards/synth/v3")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=2, help="worker di SourceReader per ogni sorgente")
    ap.add_argument("--stream-batches", type=int, default=30)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("manifest_check.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: scegliere un'altra cartella")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 1. integrità
    m = mf.load(args.manifest)
    meta = m.meta
    records = [r for shard in m.by_shard.values() for r in shard.values()]
    check("manifest letto e verificato (sha256 dei record, record_id, un solo split per gruppo, conteggi per shard)",
          len(records) == meta["records"]["righe"], len(records))
    disk = [(s["nome"], sha256_file(mf.shard_path(s["percorso"])) == s["sha256"]) for s in meta["shards"]]
    check("shard su disco con gli sha256 del manifest", all(ok for _, ok in disk), [n for n, ok in disk if not ok])
    expected_keys = {
        "train": sorted(r["record"] for r in records if r["split"] == "train" and r["stato"] == "ammesso"),
        "dev": sorted(r["record"] for r in records if r["split"] == "dev" and r["stato"] == "ammesso"),
        "test": sorted(r["record"] for r in records if r["insieme"] == "test" and r["stato"] == "ammesso"),
        "coppie": sorted(r["record"] for r in records if r["insieme"] == "coppie" and r["stato"] == "ammesso"),
        "non_ammessi": sorted(r["record"] for r in records if r["stato"] != "ammesso")}
    derived_ok = []
    for name, ks in expected_keys.items():
        entry = meta["derivati"][f"keys/{name}.txt"]
        path = args.manifest / "keys" / f"{name}.txt"
        derived_ok.append(path.read_text().splitlines() == ks and sha256_file(path) == entry["sha256"]
                          and entry["righe"] == len(ks))
    check("file derivati (keys/) uguali alle chiavi del manifest e ai loro sha256", all(derived_ok), derived_ok)

    # ---- 2. split e gruppi
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[r["gruppo"]].append(r)
    info = {g: {"famiglia": rs[0]["famiglia"], "valutazione": any(r["insieme"] != "train" for r in rs),
                "ammessi": sum(r["stato"] == "ammesso" for r in rs)} for g, rs in by_group.items()}
    rule = meta["split"]
    again = [mf.assign_splits(info, rule["quota_dev"], rule["seme"]) for _ in range(2)]
    written = {g: rs[0]["split"] for g, rs in by_group.items()}
    check("split ricalcolato due volte con quota e seme del manifest: identico a quello scritto",
          again[0] == again[1] == written, sum(again[0][g] != written[g] for g in written))
    other = mf.assign_splits(info, rule["quota_dev"], rule["seme"] + 1)
    moved = sum(other[g] != written[g] for g in written)
    check("con un altro seme lo split di dev cambia (la scelta dipende dal seme)", moved > 0, moved)
    check("ogni gruppo ha una sola famiglia", all(len({r["famiglia"] for r in rs}) == 1 for rs in by_group.values()))
    parent: dict[str, str] = {r["record"]: r["record"] for r in records}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    first: dict[tuple, str] = {}
    for r in records:
        for link in (("pixel", r["immagini"][0]["pixel_sha256"]), ("stato", r["stato_scena_sha256"]),
                     ("coppia", r["coppia"])):
            if link[1] is None:
                continue
            if link in first:
                a, b = find(first[link]), find(r["record"])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[link] = r["record"]
    ours = defaultdict(set)
    for r in records:
        ours[find(r["record"])].add(r["record"])
    check("legami esatti ricalcolati dagli hash del manifest: stessi gruppi",
          sorted(map(sorted, ours.values())) == sorted(sorted(r["record"] for r in rs) for rs in by_group.values()),
          len(ours))
    crossing = [g for g, rs in by_group.items() if len({r["split"] for r in rs}) > 1]
    check("nessun gruppo in due split (train, dev, test)", not crossing, len(crossing))
    eval_train = [r["record"] for r in records if r["split"] in ("train", "dev") and info[r["gruppo"]]["valutazione"]]
    check("nessun gruppo legato a test o coppie in train o dev", not eval_train, len(eval_train))
    dev_info = {}
    for fam, d in rule["dev_per_famiglia"].items():
        biggest = max((i["ammessi"] for g, i in info.items() if i["famiglia"] == fam and written[g] == "dev"),
                      default=0)
        dev_info[fam] = {**d, "gruppo_dev_piu_grande": biggest,
                         "entro_la_quota": d["obiettivo"] <= d["dev"] < d["obiettivo"] + max(biggest, 1)}
    check("dev per famiglia: almeno l'obiettivo, superato al più dall'ultimo gruppo",
          all(v["entro_la_quota"] for v in dev_info.values()), dev_info)

    # ---- 3. loader
    sp = specs(args.manifest, args.shards)
    sources = make_sources(list(sp.values()))
    for key_name, source in zip(sp, sources):
        SOURCES[key_name] = source
    with Pool(min(len(SOURCES) * args.workers, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))) as pool:
        epochs = pool.map(read_epoch, [(n, w, args.workers) for n in SOURCES for w in range(args.workers)])
    loaded: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, Counter] = defaultdict(Counter)
    picked: dict[str, dict] = defaultdict(dict)
    for e in epochs:
        loaded[e["name"]].extend(e["keys"])
        counts[e["name"]].update(e["counts"])
        picked[e["name"]].update(e["picked"])
    rec_by_key = {r["record"]: r for r in records}
    group_split = {g: written[g] for g in by_group}
    load_report = {}
    for name in SOURCES:
        ks = loaded[name]
        ok_once = len(ks) == len(set(ks))
        ok_set = sorted(ks) == expected_keys[name]
        not_admitted = [k for k in ks if rec_by_key[k]["stato"] != "ammesso"]
        wrong_split = [k for k in ks if group_split[rec_by_key[k]["gruppo"]] != sp[name]["split"]]
        problems = [k for k, p in picked[name].items() if p and p[-1][0] == "problema"]
        check(f"{name}: un'epoca con {args.workers} worker consegna esattamente le chiavi ammesse dello split, una volta",
              ok_once and ok_set and not not_admitted and not wrong_split and not problems,
              {"consegnati": len(ks), "attesi": len(expected_keys[name]), "doppi": len(ks) - len(set(ks)),
               "non_ammessi": len(not_admitted), "altro_split": len(wrong_split), "problemi": len(problems)})
        fam_loaded, task_loaded, task_expected, digest_bad = Counter(), Counter(), Counter(), 0
        for k in ks:
            r = rec_by_key[k]
            fam_loaded[r["famiglia"]] += 1
            exp = pick_questions(k, r["qa"])  # la stessa scelta del loader: dipende solo da chiave e numero di QA
            got = [p for p in picked[name][k] if p[0] != "problema"]
            digest_bad += [[q["tipo"], q["sha256"]] for q in exp] != got
            for q in exp:
                task_expected[f"{r['famiglia']}/{q['tipo']}"] += 1
            for t, _ in got:
                task_loaded[f"{r['famiglia']}/{t}"] += 1
        check(f"{name}: domande costruite con gli stessi digest delle QA ammesse e gli stessi conteggi per compito",
              not digest_bad and task_loaded == task_expected, digest_bad)
        load_report[name] = {"split": sp[name]["split"], "consegnati": len(ks), "contatori_reader": dict(counts[name]),
                             "per_famiglia": dict(sorted(fam_loaded.items())),
                             "domande_costruite_per_famiglia_e_compito": dict(sorted(task_loaded.items())),
                             "atteso_dal_manifest": SOURCES[name].admission.describe()}
    seen = {name: {rec_by_key[k]["gruppo"] for k in loaded[name]} for name in SOURCES}
    overlap = {f"{a}+{b}": len(seen[a] & seen[b]) for a in ("train", "dev") for b in SOURCES if b != a and
               (a, b) != ("dev", "train")}
    check("gruppi letti: nessuno in comune fra train, dev e valutazione", not any(overlap.values()), overlap)
    val = [k for k, _ in validation_samples(SOURCES["train"], 10 ** 9)]
    check("validation_samples della sorgente di train: tutto lo split dev del manifest, nell'ordine degli shard",
          sorted(val) == expected_keys["dev"] and len(val) == len(set(val)), len(val))
    from limite_vl.processing import LimiteVLProcessor

    processor = LimiteVLProcessor.from_pretrained(str(VIOLETTO))
    stream = MixedRowStream([SOURCES["train"]], processor, max_len=4096, micro_rows=2, seed=0, shuffle_buffer=64,
                            pack_buffer=32)
    keys, last = [], None
    for n, batch in enumerate(stream.iter_worker(0, 1)):
        keys += [k.split("/", 1)[1] for k in batch["meta"]["keys"]]
        last = batch["meta"]
        if n + 1 >= args.stream_batches:
            break
    train_set = set(expected_keys["train"])
    check("MixedRowStream col processor vero: solo chiavi di train ammesse, esempi contati, nessun errore del manifest",
          keys and all(k in train_set for k in keys) and last["stats"].get("loaded/synth-train", 0) > 0
          and not last["stats"].get("manifest_errors/synth-train") and not last["stats"].get("errors/synth-train"),
          {"esempi": len(keys), "stats": last["stats"], "reader": last["readers"]})

    # ---- 4. arresti espliciti
    with tempfile.TemporaryDirectory(prefix="manifest_check_") as tmp:
        tmp = Path(tmp)

        def copy(name: str) -> Path:
            dst = tmp / name
            shutil.copytree(args.manifest, dst)
            return dst

        base = dict(sp["train"])
        same = copy("uguale")
        check("controllo dei casi: la copia non alterata si carica", bool(make_sources([{**base, "manifest": str(same)}])))
        expect_error("manifest mancante", lambda: make_sources([{**base, "manifest": str(tmp / "inesistente")}]))
        broken = copy("records-alterati")
        data = bytearray((broken / "records.jsonl.gz").read_bytes())
        data[len(data) // 2] ^= 0xFF
        (broken / "records.jsonl.gz").write_bytes(bytes(data))
        expect_error("records.jsonl.gz alterato", lambda: make_sources([{**base, "manifest": str(broken)}]))
        other_sha = copy("sha-shard")
        meta2 = json.loads((other_sha / "manifest.json").read_text())
        meta2["shards"][0]["sha256"] = "0" * 64
        (other_sha / "manifest.json").write_text(json.dumps(meta2, indent=1, ensure_ascii=False))
        expect_error("sha256 di uno shard diverso", lambda: make_sources([{**base, "manifest": str(other_sha)}]))
        moved_dir = copy("gruppo-spostato")
        multi = next(g for g, rs in by_group.items() if len(rs) > 1 and rs[0]["split"] == "train")

        def move(rows):
            target = next(r for r in rows if r["gruppo"] == multi)
            target["split"] = "dev"

        rewrite_records(moved_dir, move)
        expect_error("gruppo in due split", lambda: make_sources([{**base, "manifest": str(moved_dir)}]))
        renamed = copy("record-id")

        def rename(rows):
            rows[0]["record"] = rows[0]["record"] + "x"

        rewrite_records(renamed, rename)
        expect_error("record_id non ricalcolabile", lambda: make_sources([{**base, "manifest": str(renamed)}]))
        expect_error("shard v3 col manifest v4", lambda: make_sources(
            [{**base, "shards": str(args.v3_shards / "train" / "*.tar")}]))
        expect_error("shard v4 senza manifest", lambda: make_sources(
            [{k: v for k, v in base.items() if k not in ("manifest", "split")}]))
        expect_error("split senza record ammessi (test sugli shard di train)", lambda: make_sources(
            [{**base, "split": "test"}]))
        expect_error("split sconosciuto", lambda: make_sources([{**base, "split": "val"}]))
        expect_error("manifest senza split", lambda: make_sources([{k: v for k, v in base.items() if k != "split"}]))

    failed = [r["controllo"] for r in RESULTS if not r["esito"]]
    coverage = {}
    for key, n in meta["conteggi"]["qa_per_split_famiglia_tipo"].items():
        split, fam, qtype = key.split("/")
        coverage.setdefault(f"{fam}/{qtype}", {})[split] = n
    report = {"schema": "manifest-check/1", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
              "args": {k: str(v) for k, v in vars(args).items()},
              "manifest": {"path": str(args.manifest), "manifest_sha256": m.manifest_sha256,
                           "records_sha256": meta["records"]["sha256"]},
              "controlli": RESULTS, "falliti": failed, "dev": dev_info, "loader": load_report,
              "gruppi_letti_in_comune": overlap, "qa_ammesse_per_compito_e_split": coverage,
              "conteggi_manifest": meta["conteggi"], "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "report.json", report)
    print(f"{len(RESULTS) - len(failed)} controlli superati su {len(RESULTS)} ({report['seconds']} s)")
    if failed:
        sys.exit("controlli falliti: " + "; ".join(failed))


if __name__ == "__main__":
    main()
