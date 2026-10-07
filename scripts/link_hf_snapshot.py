#!/usr/bin/env python3
"""Crea snapshots/<commit>/<path> (link simbolici ai blob) per le specifiche di hf_manifest.py.

Solo operazioni sui metadati del filesystem (stat, symlink): va bene sul frontend.
Collega soltanto i file il cui blob esiste con la dimensione attesa; elenca i mancanti.

    scripts/link_hf_snapshot.py                       # tutte le specifiche
    scripts/link_hf_snapshot.py SPEC.json [SPEC.json]
Esce con codice 1 se qualche file manca.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

STATE_DIR = Path(os.environ.get("FETCH_STATE_DIR", "/home/prignano/vision_limite/data"))
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def link_one(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    repo_dir = Path(spec["cache_repo"])
    commit = spec["commit"]
    snap = repo_dir / "snapshots" / commit
    missing = 0
    for rel, meta in sorted(spec["files"].items()):
        blob = repo_dir / "blobs" / meta["etag"]
        if not blob.is_file() or blob.stat().st_size != meta["size"]:
            print(f"  manca: {rel}")
            missing += 1
            continue
        link = snap / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        target = os.path.relpath(blob, link.parent)
        if link.is_symlink():
            if os.readlink(link) == target:
                continue
            link.unlink()
        elif link.exists():
            print(f"  esiste un file non-link, lo lascio: {link}")
            missing += 1
            continue
        link.symlink_to(target)
    req = spec.get("requested_revision", "")
    if req and not SHA_RE.match(req):
        ref = repo_dir / "refs" / req
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(commit)
    status = "completo" if missing == 0 else f"{missing} file mancanti"
    print(f"{spec['repo']}@{commit[:8]}: {status} -> {snap}")
    return missing


def main() -> None:
    specs = [Path(p) for p in sys.argv[1:]] or sorted((STATE_DIR / "snapshot-specs").glob("*.json"))
    if not specs:
        sys.exit("nessuna specifica trovata")
    missing = sum(link_one(p) for p in specs)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
