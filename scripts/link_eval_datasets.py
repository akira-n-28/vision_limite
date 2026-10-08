#!/usr/bin/env python3
"""Cartella di lavoro del valutatore con i dataset dei tasksets puntati agli snapshot verificati.

Perché: `datasets` 5 in modalità offline non costruisce un dataset dalla sola cache Hub (vuole la
propria cache già elaborata da un caricamento online; job 29391). In `load_dataset` una cartella
locale con lo stesso nome del repo vince sul ramo Hub (datasets/load.py, dataset_module_factory),
quindi il valutatore, lanciato da questa cartella, legge gli stessi byte delle revisioni fissate.

Dataset e revisioni si leggono da limite-evals/src/limite_evals/suites.py (solo analisi del
sorgente, nessun import): un taskset senza revisione fissata o senza snapshot completo fa fallire
lo script. Solo link simbolici: va bene sul frontend.

    scripts/link_eval_datasets.py --tasksets aime25,aime26,hmmt26 --root data/eval-datasets
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "third_party/limite-violetto/limite-evals/src/limite_evals/suites.py"
HF_HOME = Path(os.environ.get("HF_HOME", ROOT / "hf"))
SPECS_DIR = Path(os.environ.get("FETCH_STATE_DIR", ROOT / "data")) / "snapshot-specs"


def taskset_specs() -> dict[str, dict[str, str]]:
    """name -> {dataset, revision} dalle chiamate con argomenti letterali in suites.py."""
    out: dict[str, dict[str, str]] = {}
    for node in ast.walk(ast.parse(SUITES.read_text())):
        if not isinstance(node, ast.Call):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg in ("name", "dataset", "revision")}
        if set(kw) != {"name", "dataset", "revision"}:
            continue
        if all(isinstance(v, ast.Constant) and isinstance(v.value, str) for v in kw.values()):
            out[kw["name"].value] = {"dataset": kw["dataset"].value, "revision": kw["revision"].value}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasksets", required=True, help="nomi separati da virgola (come --tasksets di limite-eval)")
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()

    specs = taskset_specs()
    args.root.mkdir(parents=True, exist_ok=True)
    errors = []
    for name in args.tasksets.split(","):
        spec = specs.get(name)
        if spec is None:
            errors.append(f"{name}: non trovato in {SUITES}")
            continue
        repo, rev = spec["dataset"], spec["revision"]
        if not rev:
            errors.append(f"{name}: {repo} senza revisione fissata nel valutatore")
            continue
        snap = HF_HOME / "hub" / f"datasets--{repo.replace('/', '--')}" / "snapshots" / rev
        spec_file = SPECS_DIR / f"datasets--{repo.replace('/', '--')}@{rev}.json"
        if not snap.is_dir() or not spec_file.is_file():
            errors.append(f"{name}: snapshot {repo}@{rev[:8]} assente (manifest p1 e link_hf_snapshot.py)")
            continue
        missing = [p for p in json.loads(spec_file.read_text())["files"] if not (snap / p).is_file()]
        if missing:
            errors.append(f"{name}: snapshot incompleto, mancano {missing[:3]}")
            continue
        link = args.root / repo
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            if link.is_symlink() and Path(os.readlink(link)) == snap:
                print(f"{name}: {repo} -> {rev[:8]} (già presente)")
                continue
            if not link.is_symlink():
                errors.append(f"{name}: {link} esiste e non è un link")
                continue
            link.unlink()
        link.symlink_to(snap)
        print(f"{name}: {repo} -> {rev[:8]}")
    if errors:
        sys.exit("\n".join(errors))


if __name__ == "__main__":
    main()
