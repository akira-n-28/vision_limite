#!/usr/bin/env python3
"""MAVIS: 40 immagini da guardare (10 per profondità) con la descrizione canonica di F1, per verificare la fedeltà
dell'immagine (punti etichettati, valori scritti come nella descrizione, lato sostituito e direzione).

La scelta è deterministica: per ogni profondità i 10 campioni con l'hash (blake2b di --seed e id) più basso, quindi un
campione casuale e riproducibile. Uscite in --out (cartella nuova): <id>.png (l'immagine dello shard, quella che vede il
modello), items.jsonl (id, profondità, descrizione, punti, misure, figure) e checklist.md (le domande da chiudere per
ogni immagine). Job CPU: slurm/mavis_contact_sheet.sbatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.f1_mavis import mavis_truth  # noqa: E402
from limite_vl.f1_schema import render_geometry  # noqa: E402

PER_DEPTH = 10


def iter_pairs(path: Path):
    pending: dict[str, dict] = {}
    with tarfile.open(path, "r|") as tar:
        for m in tar:
            if not m.isfile():
                continue
            key, ext = m.name.rsplit("/", 1)[-1].split(".", 1)
            pending.setdefault(key, {})[ext] = tar.extractfile(m).read()
            if "json" in pending[key] and ("png" in pending[key] or "jpg" in pending[key]):
                yield key, pending.pop(key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", default="20261026")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("mavis_contact_sheet.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} esiste e non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)
    best: dict[int, list[tuple[str, str]]] = {}
    for shard in args.shards:  # primo passaggio: solo i JSON, per scegliere
        for key, files in iter_pairs(shard):
            meta = json.loads(files["json"])
            rank = hashlib.blake2b(f"{args.seed}/{meta['id']}".encode(), digest_size=8).hexdigest()
            lst = best.setdefault(meta["depth"], [])
            lst.append((rank, key))
            lst.sort()
            del lst[PER_DEPTH:]
    chosen = {key for lst in best.values() for _, key in lst}
    items = []
    for shard in args.shards:
        for key, files in iter_pairs(shard):
            if key not in chosen:
                continue
            meta = json.loads(files["json"])
            te = meta["text_en"]
            truth, why = mavis_truth(te["caption"], te["diagram_generation"])
            ext = "png" if "png" in files else "jpg"
            (args.out / f"{meta['id']}.{ext}").write_bytes(files[ext])
            items.append({"id": meta["id"], "depth": meta["depth"], "image": f"{meta['id']}.{ext}",
                          "width": meta.get("width"), "height": meta.get("height"),
                          "description": render_geometry(truth) if truth else None, "truth_status": why})
    items.sort(key=lambda x: (x["depth"], x["id"]))
    with open(args.out / "items.jsonl", "w") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in items)
    lines = ["# Verifica delle immagini MAVIS (40, 10 per profondità)", "",
             "Per ogni immagine, guardando solo l'immagine e la descrizione: (1) tutti i punti della riga Points sono "
             "etichettati e non ce ne sono altri; (2) ogni valore di Segments e Angles è scritto nella figura, uguale; "
             "(3) le figure di Shapes ci sono, del tipo detto, attaccate al lato detto; (4) il lato sostituito e la "
             "direzione (inside/outside) corrispondono; (5) etichette e valori sono leggibili a questa risoluzione.", ""]
    for x in items:
        lines += [f"## {x['id']} (profondità {x['depth']}, {x['width']}x{x['height']})", "", "```", x["description"] or "-",
                  "```", "", "- [ ] punti  - [ ] valori  - [ ] figure  - [ ] lato e direzione  - [ ] leggibile", ""]
    (args.out / "checklist.md").write_text("\n".join(lines))
    print(f"{len(items)} immagini in {args.out}", flush=True)


if __name__ == "__main__":
    main()
