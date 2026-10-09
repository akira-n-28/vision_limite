#!/usr/bin/env python3
"""Copia il codice Transformers di Limite dallo snapshot verificato in third_party/limite-modeling/.

Copia solo i file elencati in src/limite_vl/limite_code.py (PINNED_SHA256) e fallisce se uno SHA-256
non coincide. Scrive SOURCE.json con repository, revisione e hash. Solo file piccoli e libreria
standard: va bene sul frontend.

    scripts/vendor_limite_modeling.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import limite_code as lc  # noqa: E402  (solo libreria standard a livello di modulo)


def main() -> None:
    hf_home = Path(os.environ.get("HF_HOME", ROOT / "hf"))
    snapshot = (hf_home / "hub" / f"models--{lc.SOURCE_REPO.replace('/', '--')}" / "snapshots"
                / lc.SOURCE_REVISION)
    if not snapshot.is_dir():
        sys.exit(f"snapshot assente: {snapshot}")
    for name, expected in lc.PINNED_SHA256.items():
        if (actual := lc.sha256(snapshot / name)) != expected:
            sys.exit(f"{snapshot / name}: sha256 {actual}, atteso {expected}")
    lc.VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    for name in lc.PINNED_SHA256:
        shutil.copyfile(snapshot / name, lc.VENDOR_DIR / name)  # copia del contenuto, non del link
    lc.verify()
    source = {"repo": lc.SOURCE_REPO, "revision": lc.SOURCE_REVISION, "sha256": lc.PINNED_SHA256}
    (lc.VENDOR_DIR / "SOURCE.json").write_text(json.dumps(source, indent=1) + "\n")
    print(f"{len(lc.PINNED_SHA256)} file in {lc.VENDOR_DIR} da {lc.SOURCE_REPO}@{lc.SOURCE_REVISION[:8]}")


if __name__ == "__main__":
    main()
