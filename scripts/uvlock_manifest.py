#!/usr/bin/env python3
"""Estrae da un uv.lock le wheel per CPython 3.12 su Linux x86_64 in un manifest TSV.

Solo libreria standard (tomllib): si può eseguire sul frontend. Non risolve
dipendenze e non usa la rete: legge URL, dimensioni e SHA-256 già fissati nel lock.
Per ogni pacchetto sceglie una sola wheel compatibile (cp312 > abi3 > py3-none-any).
I pacchetti senza wheel compatibile (sorgenti git, solo sdist) vengono elencati a parte:
vanno compilati in un job SLURM o sostituiti.

    scripts/uvlock_manifest.py path/uv.lock --wheelhouse /home/prignano/vision_limite/wheels \\
        --out downloads/manifests/p1-wheels.tsv
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
import urllib.parse
from pathlib import Path

WHEEL_RE = re.compile(
    r"^(?P<name>.+?)-(?P<ver>[^-]+)(?:-(?P<build>\d[^-]*))?-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$"
)
MANYLINUX_RE = re.compile(r"^manylinux_(\d+)_(\d+)_x86_64$")
LEGACY = {"manylinux1_x86_64": (2, 5), "manylinux2010_x86_64": (2, 12), "manylinux2014_x86_64": (2, 17)}


def platform_ok(plat: str, max_glibc: tuple[int, int]) -> int:
    """0 = incompatibile; più alto = preferito."""
    best = 0
    for p in plat.split("."):
        if p == "any":
            best = max(best, 1)
        elif p == "linux_x86_64":
            best = max(best, 2)
        elif p in LEGACY and LEGACY[p] <= max_glibc:
            best = max(best, 3)
        elif (m := MANYLINUX_RE.match(p)) and (int(m[1]), int(m[2])) <= max_glibc:
            best = max(best, 3)
    return best


def python_ok(py: str, abi: str) -> int:
    pys = py.split(".")
    if "cp312" in pys and abi in ("cp312", "abi3", "none"):
        return 3
    if abi == "abi3" and any(p.startswith("cp3") and 2 <= len(p) and int(p[3:] or 0) <= 12 for p in pys):
        return 2
    if abi == "none" and any(p in ("py3", "py312", "py2.py3") or p.startswith("py3") for p in pys):
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lock")
    ap.add_argument("--wheelhouse", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-glibc", default="2.35", help="glibc del nodo di calcolo (verificarla con `ldd --version`)")
    args = ap.parse_args()
    max_glibc = tuple(int(x) for x in args.max_glibc.split("."))

    lock = tomllib.loads(Path(args.lock).read_text())
    rows, unresolved = [], []
    for pkg in lock.get("package", []):
        source = pkg.get("source", {})
        if "editable" in source or "virtual" in source or "directory" in source:
            continue
        best = None
        for w in pkg.get("wheels", []):
            fname = urllib.parse.unquote(w["url"].rsplit("/", 1)[-1])
            m = WHEEL_RE.match(fname)
            if not m:
                continue
            score_py = python_ok(m["py"], m["abi"])
            score_plat = platform_ok(m["plat"], max_glibc)
            if not score_py or not score_plat:
                continue
            key = (score_py, score_plat)
            if best is None or key > best[0]:
                best = (key, fname, w)
        if best is None:
            unresolved.append(f"{pkg['name']}=={pkg['version']} ({', '.join(source) or 'nessuna sorgente'})")
            continue
        _, fname, w = best
        sha = w.get("hash", "-")
        size = w.get("size", "-")
        rows.append((w["url"], str(Path(args.wheelhouse) / fname), str(size), sha, f"{pkg['name']}=={pkg['version']}"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("# url\tdest\tsize_bytes\thash\tlabel\n")
        for r in sorted(rows, key=lambda r: r[4]):
            fh.write("\t".join(r) + "\n")
    total = sum(int(r[2]) for r in rows if r[2].isdigit())
    print(f"{len(rows)} wheel, {total / 2**30:.2f} GiB -> {out}")
    if unresolved:
        print(f"{len(unresolved)} pacchetti senza wheel compatibile:")
        for u in unresolved:
            print("  ", u)


if __name__ == "__main__":
    sys.exit(main())
