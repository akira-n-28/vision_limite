#!/usr/bin/env python3
"""Sovrapposizioni fra i benchmark del pannello già indicizzati (data/decontam/p6/index.json,
data/decontam/f1eval/index.json), per raggruppare problemi e immagini comuni prima di assegnare i ruoli di sviluppo e
di test (docs/PANNELLO_VALUTAZIONE.md). Solo in un job SLURM CPU, in sola lettura.

Per ogni coppia di insiemi diversi (benchmark e split, dal nome del file dell'indice):
- immagini con pHash identico e con pHash entro --radius bit (decontam.HashIndex). Sono candidati: un pHash uguale
  non prova la stessa immagine (S1 §4);
- domande con lo stesso testo normalizzato (decontam.normalize_words) e almeno --min-words parole;
- per MathVision e MathVista, gli ID del testmini presenti anche nel test.
Uscita: --out/overlaps.json (nuovo), con i conteggi per coppia e fino a 5 esempi; a schermo le coppie più frequenti.

    srun $RUN_PYTHON -u -B scripts/panel_overlaps.py --out DIR [--radius 4] [--min-words 8]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.decontam import HashIndex, normalize_words  # noqa: E402
from limite_vl.provenance import code_identity, sha256_file, write_new_json  # noqa: E402

INDEXES = ("data/decontam/p6/index.json", "data/decontam/f1eval/index.json")


def split_of(benchmark: str, file: str) -> str:
    stem = Path(file).name
    for name in ("testmini", "test", "train", "val"):
        if stem.startswith(name):
            return name
    if re.match(r"sample\d+", stem):
        return "sample"
    return stem.split("-")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--min-words", type=int, default=8)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("panel_overlaps.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)
    items = []
    for rel in INDEXES:
        for it in json.loads((ROOT / rel).read_text(encoding="utf-8"))["items"]:
            items.append({"set": f"{it['benchmark']}:{split_of(it['benchmark'], it['file'])}", "id": str(it["id"]),
                          "phash": int(it["phash"], 16) if it.get("phash") else None,
                          "words": normalize_words(it.get("text") or "")})
    with_hash = [i for i, it in enumerate(items) if it["phash"] is not None]
    index = HashIndex([items[i]["phash"] for i in with_hash], radius=args.radius)
    exact, near = Counter(), Counter()
    examples: dict[str, list] = defaultdict(list)
    items_with_cross = Counter()
    for a, i in enumerate(with_hash):
        crossed = set()
        for b, d in index.query(items[i]["phash"]):
            j = with_hash[b]
            if j <= i or items[i]["set"] == items[j]["set"]:
                continue
            pair = " + ".join(sorted((items[i]["set"], items[j]["set"])))
            (exact if d == 0 else near)[pair] += 1
            crossed.add(items[j]["set"])
            if len(examples[pair]) < 5:
                examples[pair].append([items[i]["set"], items[i]["id"], items[j]["set"], items[j]["id"], d])
        for s in crossed:
            items_with_cross[f"{items[i]['set']} -> {s}"] += 1
    by_text: dict[str, set] = defaultdict(set)
    for it in items:
        if len(it["words"]) >= args.min_words:
            by_text[" ".join(it["words"])].add(it["set"])
    text_pairs = Counter()
    for sets in by_text.values():
        sets = sorted(sets)
        for x in range(len(sets)):
            for y in range(x + 1, len(sets)):
                text_pairs[f"{sets[x]} + {sets[y]}"] += 1
    ids = defaultdict(set)
    for it in items:
        ids[it["set"]].add(it["id"])
    id_overlap = {f"{b}: testmini nel test": len(ids[f"{b}:testmini"] & ids[f"{b}:test"])
                  for b in ("mathvision", "mathvista")}
    sizes = Counter(it["set"] for it in items)
    report = {"schema": "panel-overlaps/1", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
              "indici": {rel: sha256_file(ROOT / rel) for rel in INDEXES}, "radius": args.radius,
              "min_words": args.min_words, "elementi_per_insieme": dict(sorted(sizes.items())),
              "phash_identico": dict(exact.most_common()), "phash_entro_radius": dict(near.most_common()),
              "elementi_con_un_vicino_in_un_altro_insieme": dict(items_with_cross.most_common()),
              "testo_identico": dict(text_pairs.most_common()), "id_condivisi": id_overlap,
              "esempi": {k: v for k, v in examples.items()}}
    write_new_json(args.out / "overlaps.json", report)
    print(json.dumps({k: report[k] for k in ("elementi_per_insieme", "id_condivisi")}, ensure_ascii=False))
    for name in ("phash_identico", "phash_entro_radius", "testo_identico"):
        print(name, json.dumps(dict(list(report[name].items())[:25]), ensure_ascii=False))


if __name__ == "__main__":
    main()
