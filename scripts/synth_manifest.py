#!/usr/bin/env python3
"""Manifest delle ammissioni del candidato dei sintetici v4 (docs/specs/S1_RECORD_MANIFEST.md §9, sottoinsieme di S1;
docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §2.7). Solo in un job SLURM CPU; legge shard e uscite del candidato in sola
lettura.

Ingressi: --candidate (uscita di scripts/synth_candidate.py: report.json, records.jsonl) e --shards (la cartella del
render controllata dal candidato). Prima di scrivere si controlla che:
- il report del candidato non abbia errori dell'insieme, e che records.jsonl e il manifest del render abbiano gli
  sha256 registrati nel report;
- ogni shard abbia lo sha256 e la dimensione registrati dal candidato: sono gli shard già controllati, che non si
  ricontrollano (nessun ridisegno);
- ogni record del candidato stia nello shard e nella posizione indicati, con gli stessi sha256 di JSON e PNG;
- i gruppi ricalcolati dai legami esatti (pixel, stato della scena, coppia) coincidano con quelli del candidato, e
  ogni gruppo abbia una sola famiglia.

Split (manifest.assign_splits): `test` per i gruppi con record di test o delle coppie, che restano riservati (le
scene di train di questi gruppi sono già escluse per contaminazione); fra gli altri, dev per circa --dev-fraction
delle scene ammesse di ogni famiglia, con gruppi indivisibili e ordine fissato da --seed; il resto train. Stato:
`ammesso` per le scene `candidato`, `quarantena` ed `escluso` come nel candidato, con i motivi.

Uscite nella cartella nuova --out (data/manifests/<nome>-<versione>, S1 §7):
- records.jsonl.gz: una riga per record (scena o variante di coppia): identità, insieme d'origine, shard e posizione,
  famiglia, gruppo, split, stato, motivi, hash dell'immagine, del JSON, dello stato della scena e di ogni QA, evidenze;
- manifest.json: shard con sha256, ingressi con sha256, regola degli split, conteggi per insieme, split, stato,
  famiglia e compito, sha256 di records.jsonl.gz e dei file derivati, identità del codice, parti di S1 non coperte;
- keys/: file derivati dal manifest, una chiave per riga, ordinati: train.txt, dev.txt, test.txt (insieme di test),
  coppie.txt (insieme delle coppie, split test), non_ammessi.txt (quarantena ed esclusi).

    srun $RUN_PYTHON -u -B scripts/synth_manifest.py --candidate DIR --shards DIR --out DIR --name NOME \\
        [--dev-fraction 0.05] [--seed 0]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, refuse_existing, sha256_file, write_new_json  # noqa: E402

SOURCE = "synth"
SETS = ("train", "test", "pairs")  # cartelle del render
SET_NAME = {"train": "train", "test": "test", "pairs": "coppie"}
STATE = {"candidato": "ammesso", "quarantena": "quarantena", "escluso": "escluso"}
NOT_COVERED = [
    "variant_id e uid (S1 §1): dipendono dal contratto di risposta, non ancora definito (S3, D3). L'ammissione è per "
    "record; ogni QA di un record ammesso è approvata dal checker, e il loader (f1c.pick_questions) costruisce solo QA "
    "del record, il cui JSON è verificato col suo sha256",
    "input_tokens_sha256, target_tokens_sha256, risposta_sha256 e l'identità di tokenizer e template (S1 §2, §7): "
    "dipendono dal contratto",
    "indice globale fra sorgenti (S1 §4) e import delle esclusioni storiche (S1 §5): i gruppi sono quelli dei legami "
    "esatti dentro i sintetici v4 (pixel, stato della scena, coppia); nessun confronto con altre sorgenti o con i "
    "benchmark riservati",
    "ammissione della singola QA dentro un record ammesso (S1 §6, 'una QA non ammessa non viene costruita'): non serve "
    "a questo candidato, dove un record è ammesso solo se tutte le sue QA sono approvate",
    "contatori per compito e per target-token nel flusso (S1 §6): il flusso conta record letti, saltati, consegnati e "
    "gli esempi costruiti per sorgente; i conteggi per compito sono nel controllo (scripts/manifest_check.py)",
    "campo gold (S2 §1): non assegnato, nessuna stima campionaria (S4)",
]


def union_groups(rows: list[dict]) -> dict[str, str]:
    """Chiave -> rappresentante (la chiave minima) del gruppo per legami esatti: stessi pixel, stesso stato della
    scena, stessa coppia. È la regola di scripts/synth_candidate.link_groups, ricalcolata qui per confronto."""
    parent = {r["key"]: r["key"] for r in rows}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    first: dict[tuple, str] = {}
    for r in rows:
        for link in (("pixels", r["pixels_sha256"]), ("stato", r["state_sha256"]), ("coppia", r["pair"])):
            if link[1] is None:
                continue
            if link in first:
                a, b = find(first[link]), find(r["key"])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[link] = r["key"]
    return {k: find(k) for k in parent}


def gzip_lines(lines: list[str]) -> bytes:
    """records.jsonl.gz riproducibile: righe in ordine fisso, gzip senza data nell'intestazione."""
    return gzip.compress("".join(line + "\n" for line in lines).encode("utf-8"), compresslevel=6, mtime=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", required=True, help="nome del manifest, es. synth-v4-candidato")
    ap.add_argument("--version", default="1")
    ap.add_argument("--dev-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("synth_manifest.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: una versione del manifest si scrive una volta sola")
    t0 = time.time()
    problems: list[str] = []

    # ---- candidato e render: gli sha256 registrati dal candidato
    report_path = args.candidate / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("fatal"):
        sys.exit(f"{report_path}: il candidato ha errori dell'insieme: {report['fatal'][:3]}")
    records_path = args.candidate / "records.jsonl"
    if sha256_file(records_path) != report["files"]["records"]["sha256"]:
        sys.exit(f"{records_path}: sha256 diverso da quello del report del candidato")
    render_path = args.shards / "manifest.json"
    if sha256_file(render_path) != report["render"]["manifest_sha256"]:
        sys.exit(f"{render_path}: sha256 diverso da quello registrato dal candidato")
    render = json.loads(render_path.read_text())
    revision = f"v{render['generator_version']}@{render['code']['commit']}"
    shards = []
    for set_dir in SETS:
        for row in render["sets"][set_dir]:
            path = args.shards / set_dir / row["name"]
            sha = sha256_file(path)
            if sha != report["render"]["shards"].get(row["name"]) or sha != row["sha256"] \
                    or path.stat().st_size != row["bytes"]:
                sys.exit(f"{path}: sha256 o dimensione diversi da quelli del candidato e del render")
            shards.append({"insieme": SET_NAME[set_dir], "nome": row["name"],
                           "percorso": str(path.resolve().relative_to(mf.ROOT)), "sha256": sha, "bytes": row["bytes"],
                           "record": row["samples"]})
    rows = {}
    with open(records_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["key"]] = r
    if len(rows) != sum(s["record"] for s in shards):
        sys.exit(f"{len(rows)} record nel candidato, {sum(s['record'] for s in shards)} negli shard")

    # ---- record letti dagli shard: posizione, sha256 di JSON e PNG, QA
    digests = mf.Digests()
    found: dict[str, dict] = {}
    for s in shards:
        for index, (key, files) in enumerate(iter_shard(mf.ROOT / s["percorso"])):
            r = rows.get(key)
            if r is None:
                problems.append(f"{key}: nello shard {s['nome']} ma non nel candidato")
                continue
            if r["shard"] != s["nome"] or SET_NAME[r["split"]] != s["insieme"]:
                problems.append(f"{key}: shard o insieme diversi da quelli del candidato")
            if hashlib.sha256(files["json"]).hexdigest() != r["json_sha256"] \
                    or hashlib.sha256(files["png"]).hexdigest() != r["png_sha256"]:
                problems.append(f"{key}: JSON o PNG diversi da quelli del candidato")
            meta = json.loads(files["json"])
            qas = list(meta["qa"]) + ([meta["pair"]] if "pair" in meta else [])
            found[key] = {"shard": s, "indice": index, "w": meta["width"], "h": meta["height"],
                          "qa": [{"tipo": q["type"], "sha256": mf.qa_digest(digests, q)} for q in qas]}
    missing = sorted(set(rows) - set(found))
    if missing:
        problems.append(f"{len(missing)} record del candidato non trovati negli shard, es. {missing[:3]}")

    # ---- gruppi: ricalcolati e confrontati con quelli del candidato
    order = sorted(rows.values(), key=lambda r: (SETS.index(r["split"]), r["key"]))
    rep = union_groups(order)
    ours, theirs = defaultdict(set), defaultdict(set)
    for r in order:
        ours[rep[r["key"]]].add(r["key"])
        theirs[r["group"]].add(r["key"])
    if sorted(map(sorted, ours.values())) != sorted(map(sorted, theirs.values())):
        problems.append("gruppi ricalcolati diversi da quelli del candidato")
    rid = {r["key"]: mf.record_id(digests, SOURCE, revision, r["key"]) for r in order}
    index_name = {"nome": "legami esatti dei sintetici v4 (pixel, stato della scena, coppia)",
                  "candidato_report_sha256": sha256_file(report_path)}
    gid, info = {}, {}
    for members in ours.values():
        families = {rows[k]["family"] for k in members}
        if len(families) != 1:
            problems.append(f"gruppo con più famiglie: {sorted(members)[:3]}")
        g = mf.group_id(digests, index_name, [rid[k] for k in members])
        for k in members:
            gid[k] = g
        info[g] = {"famiglia": min(families), "valutazione": any(rows[k]["split"] != "train" for k in members),
                   "ammessi": sum(rows[k]["state"] == "candidato" for k in members)}
    if problems:
        sys.exit("controlli prima della scrittura falliti: " + "; ".join(problems[:10]))
    split_of = mf.assign_splits(info, args.dev_fraction, args.seed)

    # ---- record del manifest
    check_report = Path(report["args"]["check"]) / "report.json"
    evidence_candidate = {"controllo": "synth_candidate", "versione": report["schema"], "job": report["job_id"],
                          "rif": {"file": str(records_path.resolve().relative_to(mf.ROOT)),
                                  "sha256": report["files"]["records"]["sha256"]}}
    evidence_check = {"controllo": "synth_semantics_check", "versione": "synth-check/3",
                      "job": report["check"]["job_id"],
                      "rif": {"file": str(check_report.resolve().relative_to(mf.ROOT)),
                              "sha256": report["check"]["report_sha256"]}}
    lines, records = [], []
    for r in order:
        f = found[r["key"]]
        rec = {"record_id": rid[r["key"]], "sorgente": SOURCE, "revisione": revision, "record": r["key"],
               "insieme": SET_NAME[r["split"]], "shard": f["shard"]["percorso"], "indice": f["indice"],
               "famiglia": r["family"], "coppia": r["pair"], "gruppo": gid[r["key"]], "split": split_of[gid[r["key"]]],
               "stato": STATE[r["state"]], "motivo": r["categories"], "gold": None,
               "immagini": [{"sha256": r["png_sha256"], "pixel_sha256": r["pixels_sha256"], "ordine": 0,
                             "griglia": [f["h"] // 16, f["w"] // 16]}],
               "json_sha256": r["json_sha256"], "stato_scena_sha256": r["state_sha256"], "qa": f["qa"],
               "evidenze": [{**evidence_candidate, "esito": r["state"]},
                            {**evidence_check, "esito": "verificato" if r["checker_verified"] else "non_verificato"}],
               "trasformazione": None, "genitori": []}
        records.append(rec)
        lines.append(json.dumps(rec, sort_keys=True, ensure_ascii=False))

    # ---- coerenza prima di scrivere
    bad = [x["record"] for x in records if x["split"] in ("train", "dev") and x["insieme"] != "train"]
    bad += [x["record"] for x in records if x["insieme"] != "train" and x["split"] != "test"]
    bad += [x["record"] for x in records if x["split"] == "test" and x["insieme"] == "train" and x["stato"] == "ammesso"]
    if bad:
        sys.exit(f"split incoerenti con gli insiemi o con lo stato: {bad[:5]}")

    # ---- scrittura: records.jsonl.gz, file derivati, manifest.json
    args.out.mkdir(parents=True, exist_ok=True)
    data = gzip_lines(lines)
    data_path = args.out / "records.jsonl.gz"
    refuse_existing(data_path)
    data_path.write_bytes(data)
    derived = {}
    keys = {"train": [x["record"] for x in records if x["split"] == "train" and x["stato"] == "ammesso"],
            "dev": [x["record"] for x in records if x["split"] == "dev" and x["stato"] == "ammesso"],
            "test": [x["record"] for x in records if x["insieme"] == "test" and x["stato"] == "ammesso"],
            "coppie": [x["record"] for x in records if x["insieme"] == "coppie" and x["stato"] == "ammesso"],
            "non_ammessi": [x["record"] for x in records if x["stato"] != "ammesso"]}
    (args.out / "keys").mkdir()
    for name, ks in keys.items():
        path = args.out / "keys" / f"{name}.txt"
        text = "".join(k + "\n" for k in sorted(ks))
        path.write_text(text, encoding="utf-8")
        derived[f"keys/{name}.txt"] = {**file_entry(path, mf.sha256_text(text)), "righe": len(ks),
                                       "contenuto": "chiavi ammesse" if name != "non_ammessi" else "quarantena ed esclusi"}

    counts = {"per_insieme_split_stato": Counter(), "per_split_famiglia_stato": Counter(),
              "qa_per_split_famiglia_tipo": Counter(), "gruppi_per_split": Counter()}
    for x in records:
        counts["per_insieme_split_stato"][f"{x['insieme']}/{x['split']}/{x['stato']}"] += 1
        counts["per_split_famiglia_stato"][f"{x['split']}/{x['famiglia']}/{x['stato']}"] += 1
        if x["stato"] == "ammesso":
            for q in x["qa"]:
                counts["qa_per_split_famiglia_tipo"][f"{x['split']}/{x['famiglia']}/{q['tipo']}"] += 1
    for g in info:
        counts["gruppi_per_split"][split_of[g]] += 1
    eval_sets = defaultdict(set)
    for x in records:
        eval_sets[x["gruppo"]].add(x["insieme"])
    shared = sum(1 for sets in eval_sets.values() if {"test", "coppie"} <= sets)
    dev_target = {}
    for fam in sorted({i["famiglia"] for i in info.values()}):
        admitted = sum(i["ammessi"] for g, i in info.items() if i["famiglia"] == fam and not i["valutazione"])
        got = sum(i["ammessi"] for g, i in info.items() if i["famiglia"] == fam and split_of[g] == "dev")
        dev_target[fam] = {"ammesse_fuori_valutazione": admitted, "obiettivo": round(args.dev_fraction * admitted),
                           "dev": got, "quota": round(got / admitted, 5) if admitted else None}
    meta = {
        "schema": mf.SCHEMA, "nome": args.name, "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"),
        "codice": code_identity(), "sorgente": SOURCE, "revisione": revision,
        "indice": index_name,
        "ingressi": {"candidato_report": file_entry(report_path), "candidato_records": file_entry(records_path),
                     "render_manifest": file_entry(render_path), "controllo_report": file_entry(check_report),
                     "candidato_job": report["job_id"], "render_job": render["job_id"],
                     "controllo_job": report["check"]["job_id"]},
        "shards": shards,
        "liste_importate": [],
        "tokenizer": None, "template": None, "contratti": None,
        "split": {"valori": list(mf.SPLITS), "unita": "gruppo",
                  "regola": "test per i gruppi con record di test o delle coppie; fra gli altri, per famiglia e in "
                            "ordine di sha256({seme, gruppo}), dev finché le scene ammesse di dev sono meno di "
                            "round(quota * ammesse della famiglia); il resto train (manifest.assign_splits)",
                  "quota_dev": args.dev_fraction, "seme": args.seed, "strati": "famiglia", "dev_per_famiglia": dev_target,
                  "gruppi_condivisi_test_coppie": shared},
        "stati": {"ammesso": "scena candidato del controllo (S2 §2.6): ammessa nel manifest, non certificata",
                  "quarantena": "testi sovrapposti, scena da rivedere", "escluso": "contaminazione con la valutazione"},
        "conteggi": {k: dict(sorted(v.items())) for k, v in counts.items()},
        "records": {"file": "records.jsonl.gz", "sha256": hashlib.sha256(data).hexdigest(), "righe": len(lines)},
        "derivati": derived,
        "fuori_dal_sottoinsieme": NOT_COVERED,
        "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "manifest.json", meta)
    loaded = mf.load(args.out)  # rilettura con le verifiche dei loader
    print(json.dumps({"record": len(lines), "split": meta["conteggi"]["per_insieme_split_stato"],
                      "dev_per_famiglia": dev_target, "gruppi": dict(counts["gruppi_per_split"]),
                      "gruppi_condivisi_test_coppie": shared, "riletto": len(loaded.by_shard),
                      "seconds": meta["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
