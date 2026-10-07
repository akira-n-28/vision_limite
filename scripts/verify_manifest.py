#!/usr/bin/env python3
"""Verifica dimensione e hash dei file di uno o più manifest TSV.

Calcolo pesante su disco: va eseguito SOLO dentro un job SLURM CPU
(slurm/verify_hashes.sbatch), mai sul frontend.

Hash supportati nella colonna 4: "sha256:<hex>", "gitsha1:<hex>" (hash git del blob,
usato da Hugging Face per i file non-LFS), "-" (solo dimensione).
Scrive un report TSV e esce con codice 1 se qualcosa non torna.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

CHUNK = 8 * 1024 * 1024


def digest(path: Path, algo: str, size: int) -> str:
    h = hashlib.sha256() if algo == "sha256" else hashlib.sha1()
    if algo == "gitsha1":
        h.update(f"blob {size}\0".encode())
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("verify_manifest.py va eseguito in un job SLURM")

    bad = 0
    seen: set[str] = set()
    with open(args.report, "w") as out:
        out.write("status\tdest\tsize\texpected\tlabel\n")
        for manifest in args.manifests:
            for line in Path(manifest).read_text().splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                _url, dest, size_s, expected, label = line.split("\t")[:5]
                if dest in seen:
                    continue
                seen.add(dest)
                p = Path(dest)
                if not p.is_file():
                    status = "missing"
                elif size_s not in ("", "-") and p.stat().st_size != int(size_s):
                    status = "bad_size"
                elif expected in ("", "-"):
                    status = "ok_size_only"
                else:
                    algo, want = expected.split(":", 1)
                    got = digest(p, algo, p.stat().st_size)
                    status = "ok" if got == want else "bad_hash"
                if not status.startswith("ok"):
                    bad += 1
                out.write(f"{status}\t{dest}\t{size_s}\t{expected}\t{label}\n")
                out.flush()
                print(status, label, flush=True)
    print(f"file con problemi: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
