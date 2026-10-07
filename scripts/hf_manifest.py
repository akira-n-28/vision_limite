#!/usr/bin/env python3
"""Crea un manifest di download per un repository Hugging Face a una revisione fissata.

Solo libreria standard: si può eseguire sul frontend (`/usr/bin/python3 -I`).
Interroga soltanto l'API JSON di Hugging Face (piccole risposte, una richiesta alla
volta, con lo stesso lock di fetch.sh e il controllo MemAvailable >= 1 GiB).
Non scarica i file: scrive righe TSV per fetch_manifest.sh.

Le destinazioni seguono il layout della cache Hugging Face, così gli stessi file
servono sia come percorsi espliciti sia al valutatore ufficiale in modalità offline:
    $HF_HOME/hub/<models|datasets>--<org>--<name>/blobs/<etag>
Dopo il download, `link_hf_snapshot.py` crea snapshots/<commit>/<path> e refs/.

Esempio:
    scripts/hf_manifest.py --repo paradigma-inc/limite-1b-violetto \\
        --revision b1f3d572ccacb6919f4d64c321b70ba034ddaef2 \\
        --out downloads/manifests/p1-modelli.tsv
    scripts/hf_manifest.py --repo MathArena/aime_2026 --type dataset \\
        --revision 10b4e45b7a503075d4da8a0d57916a4f06ce6bd2 --out ... --append
"""
from __future__ import annotations

import argparse
import fcntl
import fnmatch
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
MIN_START_KB = 1024 * 1024  # 1 GiB, come fetch.sh
STATE_DIR = Path(os.environ.get("FETCH_STATE_DIR", "/home/prignano/vision_limite/data"))
DEFAULT_HF_HOME = os.environ.get("HF_HOME", "/home/prignano/vision_limite/hf")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def mem_available_kb() -> int:
    with open(os.environ.get("FETCH_MEMINFO", "/proc/meminfo")) as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0


def token() -> str | None:
    path = os.environ.get("HF_TOKEN_FILE")
    if not path or not os.access(path, os.R_OK):
        return None
    return Path(path).read_text().strip() or None


def get_json(url: str) -> tuple[object, str | None]:
    """Una richiesta GET; restituisce (json, url della pagina successiva)."""
    if mem_available_kb() < MIN_START_KB:
        sys.exit("MemAvailable sotto 1 GiB: non interrogo l'API")
    req = urllib.request.Request(url, headers={"User-Agent": "limite-vl-hf-manifest"})
    tok = token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read(64 * 1024 * 1024)  # le pagine dell'API sono piccole
        link = resp.headers.get("Link", "")
    nxt = None
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    if m:
        nxt = m.group(1)
    return json.loads(body), nxt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="org/nome")
    ap.add_argument("--type", choices=["model", "dataset"], default="model")
    ap.add_argument("--revision", required=True, help="commit (40 hex, consigliato) o nome di branch")
    ap.add_argument("--include", action="append", default=[], help="glob dei percorsi da includere (ripetibile)")
    ap.add_argument("--exclude", action="append", default=[], help="glob dei percorsi da escludere (ripetibile)")
    ap.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    ap.add_argument("--out", required=True, help="manifest TSV")
    ap.add_argument("--append", action="store_true", help="aggiunge righe a un manifest esistente")
    args = ap.parse_args()

    kind = f"{args.type}s"
    repo_q = urllib.parse.quote(args.repo, safe="/")
    rev_q = urllib.parse.quote(args.revision, safe="")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / ".fetch.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("un trasferimento è in corso (lock occupato): riprovare quando finisce")

        info, _ = get_json(f"{API}/api/{kind}/{repo_q}/revision/{rev_q}")
        commit = info["sha"]
        if SHA_RE.match(args.revision) and commit != args.revision:
            sys.exit(f"revisione risolta {commit} diversa da quella richiesta {args.revision}")

        entries: list[dict] = []
        url: str | None = f"{API}/api/{kind}/{repo_q}/tree/{commit}?recursive=true&expand=false"
        while url:
            page, url = get_json(url)
            entries.extend(page)  # type: ignore[arg-type]

    org_name = args.repo.replace("/", "--")
    cache_repo = Path(args.hf_home) / "hub" / f"{kind}--{org_name}"
    prefix = "" if args.type == "model" else "datasets/"

    files = []
    for e in entries:
        if e.get("type") != "file":
            continue
        path = e["path"]
        if args.include and not any(fnmatch.fnmatch(path, g) for g in args.include):
            continue
        if any(fnmatch.fnmatch(path, g) for g in args.exclude):
            continue
        lfs = e.get("lfs")
        if lfs:
            etag, size, digest = lfs["oid"], int(lfs["size"]), f"sha256:{lfs['oid']}"
        else:
            etag, size, digest = e["oid"], int(e["size"]), f"gitsha1:{e['oid']}"
        files.append((path, etag, size, digest))

    if not files:
        sys.exit("nessun file corrisponde ai filtri")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a" if args.append else "w") as fh:
        if not args.append or out.stat().st_size == 0:
            fh.write("# url\tdest\tsize_bytes\thash\tlabel\n")
        for path, etag, size, digest in sorted(files):
            url = f"{API}/{prefix}{args.repo}/resolve/{commit}/{urllib.parse.quote(path)}"
            dest = cache_repo / "blobs" / etag
            fh.write(f"{url}\t{dest}\t{size}\t{digest}\t{args.repo}@{commit[:8]}:{path}\n")

    # Specifica dello snapshot, usata da link_hf_snapshot.py dopo il download.
    spec_dir = STATE_DIR / "snapshot-specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{kind}--{org_name}@{commit}.json"
    previous = json.loads(spec_path.read_text())["files"] if spec_path.exists() else {}
    previous.update({p: {"etag": t, "size": s} for p, t, s, _ in files})
    spec = {
        "repo": args.repo,
        "type": args.type,
        "commit": commit,
        "requested_revision": args.revision,
        "cache_repo": str(cache_repo),
        "files": previous,
    }
    spec_path.write_text(json.dumps(spec, indent=1, sort_keys=True))

    total = sum(size for _, _, size, _ in files)
    print(f"{args.repo}@{commit}: {len(files)} file, {total / 2**30:.2f} GiB -> {out}")
    print(f"specifica snapshot: {spec_path}")


if __name__ == "__main__":
    main()
