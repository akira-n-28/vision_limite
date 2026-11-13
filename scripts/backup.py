#!/usr/bin/env python3
"""backup.py — inventario, piano dei volumi, creazione e verifica degli archivi del backup (docs/BACKUP_RIPRISTINO.md).

Solo libreria standard: gira nei job CPU (slurm/backup_*.sbatch) con /usr/bin/python3 -I -B, mai sul frontend.
Gli archivi sono tar POSIX (PAX) standard, con gzip dove conviene: il ripristino usa solo cat, tar e sha256sum.

    inventario --root R --out DIR     visita tutta la cartella senza seguire i link (file nascosti, ignorati e non
                                      tracciati compresi), con git, link, hard link, nomi problematici e segreti
    piano --inventario DIR --out P    categorie (partizione dell'albero, ripristinabili da sole), volumi da circa
                                      5 GiB nell'ordine di priorità, nucleo, compressione, firme; esclusioni solo
                                      dichiarate (sottoinsiemi, tipi non archiviabili); rifiuta inventari con errori
                                      e il percorso incrementale (--precedente): un aggiornamento è un supplemento
                                      autonomo (--dipende-da, --prefisso-volumi)
    crea --piano P --staging S --stato T
                                      scrive i volumi leggendo ogni file una volta (sha256 per file, per parte e per
                                      flusso), li rilegge per intero e li fa elencare a GNU tar; staging limitato;
                                      un membro mancante o un file non volatile cambiato blocca il volume; un
                                      marcatore pronto di un altro piano ferma tutto
    ripristino --pronto ...           prova con gli strumenti standard: cat | tar -xp, confronto con l'indice,
                                      sha256sum -c; rimuove la copia di prova (--esistente: ripristino vero)
    catalogo, completezza, firme, rimossi
                                      catalogo di recupero; verifica finale con esito e copertura dalle voci
                                      archiviate; confronto di un inventario con un piano di base; elenco dei
                                      percorsi da togliere dopo il ripristino di un supplemento

La visita non entra in `trasferimento/` (materiale creato per il trasferimento: staging, stato, credenziali del
client), non attraversa altri filesystem e non legge file fuori dalla radice: per i link registra solo la
destinazione e se esiste. La ricerca dei segreti non scrive mai il testo trovato: solo percorso, regola e righe.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Materiale del trasferimento: escluso dalla visita e dal backup (documentato in docs/BACKUP_RIPRISTINO.md).
EXCLUDED_TOP = ("trasferimento",)
# Aree che la ricerca dei segreti non legge: dati, pesi, ambienti e cache di terzi (test e fixture dei pacchetti
# contengono chiavi finte) e oggetti git compressi (letti a parte con git cat-file).
SECRET_SKIP = ("venvs/", "cache/", "wheels/", "hf/", "data/shards/", "tools/")
SECRET_MAX_BYTES = 16 * 1024 * 1024
SECRET_RULES = {  # id: (espressione sui byte, affidabilità)
    "huggingface": (rb"\bhf_[A-Za-z0-9]{30,}", "alta"),
    "github": (rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})", "alta"),
    "anthropic": (rb"\bsk-ant-[A-Za-z0-9_-]{20,}", "alta"),
    "openai": (rb"\bsk-(?:proj-)?[A-Za-z0-9]{32,}", "media"),
    "aws": (rb"\bAKIA[0-9A-Z]{16}\b", "alta"),
    "google-oauth": (rb"\bya29\.[0-9A-Za-z_-]{20,}|\b1//0[0-9A-Za-z_-]{20,}", "alta"),
    "google-service-account": (rb"\"type\"\s*:\s*\"service_account\"", "alta"),
    "chiave-privata": (rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----", "alta"),
    "slack": (rb"\bxox[baprs]-[0-9A-Za-z-]{10,}", "alta"),
    "url-con-credenziali": (rb"\b[a-z][a-z0-9+.-]*://[^/\s:@'\"<>]+:[^/\s@'\"<>]{3,}@[A-Za-z0-9.-]+", "media"),
    "assegnazione": (rb"(?i)\b(?:password|passwd|secret|api_?key|access_?token|auth_?token)\b\s*[:=]\s*"
                     rb"['\"][^'\"\s]{8,}['\"]", "bassa"),
}


def need_slurm() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("backup.py va eseguito in un job SLURM (slurm/backup_*.sbatch), non sul frontend")


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True)


def area_of(rel: str, is_dir: bool) -> str:
    """Area di riepilogo: cartella di primo livello, o più profonda dove il progetto tiene raccolte distinte."""
    parts = rel.split("/")
    if len(parts) == 1 and not is_dir:
        return "(radice)"
    top = parts[0]
    if top == "data" and len(parts) > 1 and parts[1] == "shards":
        depth = 4
    elif top == "hf" and len(parts) > 1 and parts[1] == "hub":
        depth = 3
    elif top in ("data", "outputs", "cache", "venvs", "third_party", "hf"):
        depth = 2
    else:
        depth = 1
    return "/".join(parts[:depth])


def walk(root: Path, errors: list):
    """Visita in profondità, figli in ordine di nome, senza seguire i link e senza cambiare filesystem.
    Restituisce (rel, os.stat_result) per ogni voce; i genitori vengono prima dei figli."""
    root_dev = os.lstat(root).st_dev
    stack = [""]
    while stack:
        rel = stack.pop()
        absdir = os.path.join(root, rel) if rel else str(root)
        try:
            with os.scandir(absdir) as it:
                entries = sorted(it, key=lambda e: os.fsencode(e.name))
        except OSError as exc:
            errors.append({"p": rel, "errore": f"scandir: {exc.__class__.__name__}: {exc.strerror}"})
            continue
        subdirs = []
        for e in entries:
            child = f"{rel}/{e.name}" if rel else e.name
            if not rel and e.name in EXCLUDED_TOP:
                continue
            try:
                st = e.stat(follow_symlinks=False)
            except OSError as exc:
                errors.append({"p": child, "errore": f"lstat: {exc.__class__.__name__}: {exc.strerror}"})
                continue
            yield child, st
            if stat.S_ISDIR(st.st_mode):
                if st.st_dev != root_dev:
                    errors.append({"p": child, "errore": "altro filesystem: non visitato"})
                    continue
                subdirs.append(child)
        stack.extend(reversed(subdirs))


def kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "f"
    if stat.S_ISDIR(mode):
        return "d"
    if stat.S_ISLNK(mode):
        return "l"
    return {stat.S_IFIFO: "fifo", stat.S_IFSOCK: "socket", stat.S_IFCHR: "chr", stat.S_IFBLK: "blk"}.get(
        stat.S_IFMT(mode), "altro")


def git_lines(repo: Path, *args: str, z: bool = False) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout
    sep = b"\0" if z else b"\n"
    return [os.fsdecode(x) for x in out.split(sep) if x]


def git_info(repo: Path) -> dict:
    def one(*args: str) -> str:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else f"(errore {r.returncode}: {r.stderr.strip()[:200]})"
    info = {
        "head": one("rev-parse", "HEAD"),
        "branch": one("rev-parse", "--abbrev-ref", "HEAD"),
        "status": one("status", "--porcelain=v2", "--branch"),
        "stash": one("stash", "list"),
        "worktree": one("worktree", "list", "--porcelain"),
        "refs": one("for-each-ref", "--format=%(refname) %(objectname) %(objecttype)"),
        "remotes": one("remote", "-v"),
        "shallow": one("rev-parse", "--is-shallow-repository"),
        "commit_non_pubblicati": one("log", "--branches", "--tags", "--not", "--remotes", "--format=%H %s"),
        "alternates": (repo / ".git/objects/info/alternates").read_text().strip()
        if (repo / ".git/objects/info/alternates").is_file() else "",
        "gitmodules": (repo / ".gitmodules").is_file(),
        "lfs": any("filter=lfs" in p.read_text(errors="replace") for p in [repo / ".gitattributes"] if p.is_file()),
        "hooks_attivi": sorted(p.name for p in (repo / ".git/hooks").iterdir() if not p.name.endswith(".sample"))
        if (repo / ".git/hooks").is_dir() else [],
        "count_objects": one("count-objects", "-v"),
    }
    return info


def scan_bytes(data: bytes) -> dict[str, list[int]]:
    hits: dict[str, list[int]] = {}
    for rid, (rx, _) in RULES_C.items():
        lines = []
        for m in rx.finditer(data):
            lines.append(data.count(b"\n", 0, m.start()) + 1)
            if len(lines) >= 20:
                break
        if lines:
            hits[rid] = lines
    return hits


RULES_C = {rid: (re.compile(rx), conf) for rid, (rx, conf) in SECRET_RULES.items()}


def scan_git_objects(repo: Path) -> list[dict]:
    """Segreti negli oggetti git (anche quelli già pubblicati): solo id del blob, regola e numero di righe."""
    out = []
    proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch-all-objects", "--batch"],
                            stdout=subprocess.PIPE)
    assert proc.stdout is not None
    while True:
        header = proc.stdout.readline()
        if not header:
            break
        oid, otype, size = header.split()
        data = proc.stdout.read(int(size))
        proc.stdout.read(1)
        if otype == b"blob" and b"\0" not in data[:8192]:
            hits = scan_bytes(data)
            if hits:
                out.append({"oggetto": oid.decode(), "regole": {k: len(v) for k, v in hits.items()}})
    proc.wait()
    return out


def cmd_inventario(args) -> None:
    need_slurm()
    root = Path(args.root).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    start_ns = time.time_ns()
    uid = os.getuid()
    head0 = git_lines(root, "rev-parse", "HEAD")[0]

    tracked = set(git_lines(root, "ls-files", "-z", "--cached", z=True))
    untracked = set(git_lines(root, "ls-files", "-z", "--others", "--exclude-standard", z=True))

    errors: list = []
    areas: dict[str, Counter] = defaultdict(Counter)
    seen_ino: dict[int, str] = {}
    links: dict[int, dict] = {}
    names_by_dir: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    largest: list[tuple[int, str]] = []
    special, non_utf8, long_paths, sparse, recent, unreadable, foreign = [], [], [], [], [], [], []
    git_state = Counter()
    secret_hits, secret_skipped_big = [], 0
    counts = Counter()
    sym_out = gzip.open(out / "collegamenti.jsonl.gz", "wt", encoding="ascii")
    inv = gzip.open(out / "inventario.jsonl.gz", "wt", encoding="ascii")
    for rel, st in walk(root, errors):
        t = kind(st.st_mode)
        rec = {"p": rel, "t": t, "s": st.st_size, "b": st.st_blocks, "n": st.st_nlink, "i": st.st_ino,
               "m": stat.S_IMODE(st.st_mode), "u": st.st_uid, "g": st.st_gid, "mt": st.st_mtime_ns}
        a = area_of(rel, t == "d")
        counts[t] += 1
        areas[a]["voci"] += 1
        areas[a][f"n_{t}"] += 1
        parent, _, name = rel.rpartition("/")
        key = unicodedata.normalize("NFC", name).casefold()
        names_by_dir[parent][key].append(name)
        try:
            rel.encode("utf-8")
        except UnicodeEncodeError:
            non_utf8.append(rel.encode("utf-8", "surrogateescape").hex())
        if len(os.fsencode(rel)) + len(os.fsencode(str(root))) >= 1000:
            long_paths.append(rel)
        if st.st_uid != uid:
            foreign.append({"p": rel, "uid": st.st_uid})
        if st.st_mtime_ns > start_ns:
            recent.append(rel)
        if t == "f":
            if rel in tracked:
                git_state["tracciati"] += 1
            elif rel in untracked:
                git_state["non_tracciati"] += 1
            else:
                git_state["ignorati_o_fuori_da_git"] += 1
            first = st.st_nlink == 1 or st.st_ino not in seen_ino
            if st.st_nlink > 1:
                g = links.setdefault(st.st_ino, {"nlink": st.st_nlink, "size": st.st_size, "paths": []})
                g["paths"].append(rel)
                seen_ino.setdefault(st.st_ino, rel)
            if first:
                areas[a]["byte"] += st.st_size
                areas[a]["byte_allocati"] += st.st_blocks * 512
                counts["byte"] += st.st_size
                counts["byte_allocati"] += st.st_blocks * 512
                largest.append((st.st_size, rel))
                if len(largest) > 400:
                    largest = sorted(largest, reverse=True)[:100]
            else:
                areas[a]["byte_hardlink_ripetuti"] += st.st_size
            if st.st_size > 1 << 20 and st.st_blocks * 512 < st.st_size // 2:
                sparse.append({"p": rel, "s": st.st_size, "allocati": st.st_blocks * 512})
            readable = (st.st_mode & stat.S_IRUSR) if st.st_uid == uid else os.access(root / rel, os.R_OK)
            if not readable:
                unreadable.append(rel)
            elif args.segreti and not rel.startswith(SECRET_SKIP) and "/.git/objects/" not in f"/{rel}":
                if st.st_size > SECRET_MAX_BYTES:
                    secret_skipped_big += 1
                elif st.st_size:
                    try:
                        with open(root / rel, "rb") as fh:
                            data = fh.read(SECRET_MAX_BYTES + 1)
                    except OSError as exc:
                        errors.append({"p": rel, "errore": f"lettura per i segreti: {exc.strerror}"})
                        data = b""
                    if data and b"\0" not in data[:8192]:
                        hits = scan_bytes(data)
                        if hits:
                            secret_hits.append({"p": rel, "regole": hits})
        elif t == "l":
            target = os.readlink(root / rel)
            rec["l"] = target
            lexical = os.path.normpath(os.path.join(str(root), parent, target))
            inside = lexical == str(root) or lexical.startswith(str(root) + os.sep)
            real = os.path.realpath(root / rel)
            real_inside = real == str(root) or real.startswith(str(root) + os.sep)
            exists = os.path.exists(root / rel)
            sym_out.write(dump({"p": rel, "l": target, "assoluto": os.path.isabs(target), "dentro": inside,
                                "reale_dentro": real_inside, "esiste": exists,
                                "verso_escluso": real_inside and os.path.relpath(real, root).split(os.sep)[0]
                                in EXCLUDED_TOP}) + "\n")
            c = "assoluto" if os.path.isabs(target) else "relativo"
            counts[f"link_{c}"] += 1
            if not exists:
                counts["link_interrotti"] += 1
            if not real_inside:
                counts["link_fuori"] += 1
        elif t not in ("d",):
            special.append({"p": rel, "t": t})
        inv.write(dump(rec) + "\n")
    inv.close()
    sym_out.close()

    collisions = []
    for parent, keys in names_by_dir.items():
        for _, names in keys.items():
            if len(names) > 1:
                collisions.append({"cartella": parent, "nomi": sorted(names)})
    groups = []
    with gzip.open(out / "hardlink.jsonl.gz", "wt", encoding="ascii") as fh:
        for ino, g in links.items():
            g["inode"] = ino
            g["aree"] = sorted({area_of(p, False) for p in g["paths"]})
            g["esterni"] = g["nlink"] - len(g["paths"])
            fh.write(dump(g) + "\n")
            groups.append(g)
    hl = {
        "gruppi": len(groups),
        "byte_una_volta": sum(g["size"] for g in groups),
        "gruppi_con_nomi_fuori_dalla_visita": sum(1 for g in groups if g["esterni"] > 0),
        "byte_con_nomi_fuori_dalla_visita": sum(g["size"] for g in groups if g["esterni"] > 0),
        "gruppi_fra_aree": [{"aree": g["aree"], "esempio": g["paths"][:2], "size": g["size"]}
                            for g in groups if len(g["aree"]) > 1][:50],
        "coppie_di_aree": Counter(" + ".join(g["aree"]) for g in groups).most_common(40),
    }

    git = {"radice": git_info(root), "stato_file": dict(git_state), "head_inizio": head0}
    tp = root / "third_party"
    git["third_party"] = {p.name: git_info(p) for p in sorted(tp.iterdir()) if (p / ".git").exists()} if tp.is_dir() else {}
    git["head_fine"] = git_lines(root, "rev-parse", "HEAD")[0]
    (out / "git.json").write_text(json.dumps(git, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    secrets = {"regole": {k: {"espressione": v[0].decode(), "affidabilita": v[1]} for k, v in SECRET_RULES.items()},
               "esclusi_per_area": list(SECRET_SKIP), "soglia_byte": SECRET_MAX_BYTES,
               "file_testo_saltati_perche_grandi": secret_skipped_big, "file": secret_hits}
    if args.segreti:
        secrets["oggetti_git"] = {"radice": scan_git_objects(root)}
        for p in (sorted(tp.iterdir()) if tp.is_dir() else []):
            if (p / ".git").exists():
                secrets["oggetti_git"][f"third_party/{p.name}"] = scan_git_objects(p)
    (out / "segreti.json").write_text(json.dumps(secrets, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "radice": str(root), "esclusi": list(EXCLUDED_TOP), "job": os.environ.get("SLURM_JOB_ID"),
        "inizio": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)), "secondi": round(time.time() - t0, 1),
        "totali": dict(counts), "aree": {a: dict(c) for a, c in sorted(areas.items())},
        "hardlink": hl, "speciali": special, "nomi_non_utf8_hex": non_utf8[:100], "n_nomi_non_utf8": len(non_utf8),
        "percorsi_lunghi": long_paths[:100], "collisioni_maiuscole_o_unicode": collisions[:200],
        "n_collisioni": len(collisions), "sparsi": sparse[:100], "modificati_durante_la_visita": recent[:200],
        "n_modificati_durante_la_visita": len(recent), "illeggibili": unreadable[:200],
        "n_illeggibili": len(unreadable), "di_altri_utenti": foreign[:100], "n_di_altri_utenti": len(foreign),
        "errori": errors[:200], "n_errori": len(errors), "piu_grandi": [{"p": p, "s": s} for s, p in
                                                                        sorted(largest, reverse=True)[:60]],
        "segreti_file": len(secret_hits), "git_head": git["head_fine"],
    }
    (out / "riepilogo.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_text_summary(out, summary)
    with open(out / "SHA256SUMS", "w", encoding="ascii") as fh:
        for name in sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
            h = hashlib.sha256((out / name).read_bytes()).hexdigest()
            fh.write(f"{h}  {name}\n")
    print(f"inventario: {counts['f']} file, {counts['d']} cartelle, {counts['l']} link, "
          f"{counts['byte'] / 2**30:.1f} GiB in {time.time() - t0:.0f} s -> {out}")


def gib(n: int) -> str:
    return f"{n / 2**30:9.2f}"


def write_text_summary(out: Path, s: dict) -> None:
    lines = [f"inventario di {s['radice']} (job {s['job']}, {s['inizio']}, {s['secondi']} s)",
             f"totali: {dump(s['totali'])}", "",
             f"{'area':60} {'GiB':>9} {'file':>9} {'cartelle':>8} {'link':>7} {'GiB hl rip.':>11}"]
    for a, c in s["aree"].items():
        if c.get("byte", 0) < 1 << 20 and c.get("n_f", 0) < 1000:
            continue
        lines.append(f"{a[:60]:60} {gib(c.get('byte', 0))} {c.get('n_f', 0):9} {c.get('n_d', 0):8} "
                     f"{c.get('n_l', 0):7} {gib(c.get('byte_hardlink_ripetuti', 0)):>11}")
    small = [a for a, c in s["aree"].items() if c.get("byte", 0) < 1 << 20 and c.get("n_f", 0) < 1000]
    lines += [f"(+ {len(small)} aree sotto 1 MiB e 1000 file)", "",
              f"hard link: {dump({k: v for k, v in s['hardlink'].items() if k != 'gruppi_fra_aree'})}",
              f"gruppi fra aree (primi): {dump(s['hardlink']['gruppi_fra_aree'][:10])}",
              f"speciali: {len(s['speciali'])}; nomi non UTF-8: {s['n_nomi_non_utf8']}; "
              f"collisioni per maiuscole/Unicode: {s['n_collisioni']}; sparsi: {len(s['sparsi'])}; "
              f"modificati durante la visita: {s['n_modificati_durante_la_visita']}; illeggibili: {s['n_illeggibili']}; "
              f"di altri utenti: {s['n_di_altri_utenti']}; errori: {s['n_errori']}; file con segreti: {s['segreti_file']}",
              "", "più grandi:"]
    lines += [f"  {gib(x['s'])} GiB  {x['p']}" for x in s["piu_grandi"][:40]]
    (out / "riepilogo.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- piano dei volumi

GIB = 1 << 30
MIB = 1 << 20
VOLUME_BYTES = 5 * GIB      # dati per volume, indicativo (consegna); un file più grande ha un volume a sé, in parti
PART_BYTES = 5 * GIB        # dimensione massima di ogni file caricato
ENTRY_OVERHEAD = 2048       # al più per voce: intestazione ustar, intestazione PAX e riempimento a 512 byte
CODE_TOPS = frozenset({".git", "archive", "config", "docs", "downloads", "env", "env-evals", "revisioni", "runs",
                       "scripts", "slurm", "src", "tests", "third_party", "secrets"})
STATE_FILES = frozenset({"data/download-log.tsv", "data/.fetch.lock"})  # cambiano durante il trasferimento
# File che cambiano per costruzione mentre il backup lavora (registri dei job e del trasferimento, indice di git
# riscritto da `git status`): un loro cambiamento dopo l'inventario è un avviso motivato. Per ogni altro file è un
# errore bloccante, perché lo stato raccolto non sarebbe più coerente con l'inventario.
VOLATILE_FILES = STATE_FILES | frozenset({"runs/ledger.tsv", ".git/index"})
ARCHIVABLE = frozenset({"f", "d", "l", "fifo", "chr", "blk"})  # i socket e i tipi ignoti si escludono nel piano
PLAN_SCHEMA = "limite-vl-backup-piano/2"
NUCLEO_FIXED = frozenset({"codice", "data-metadati", "strumenti", "wheels", "hf-piccoli"})
NUCLEO_SHARDS = frozenset({"data-shards-unimer-2343ddd9-dense", "data-shards-mavis-geometry-9c80fff3-caption",
                           "data-shards-pixmo-cap-3e42775e", "data-shards-synth-v2", "data-shards-synth-v3",
                           "data-shards-synth-v4", "data-shards-replay-replay-slice1-p0",
                           "data-shards-f2-mmfinereason-f2-sft120k-v4"})
NUCLEO_HF = frozenset({"hf-datasets--AI4Math--MathVista"})  # benchmark riservato sopra 1 GiB; gli altri in hf-piccoli
INCOMPRESSIBLE = frozenset({".gz", ".tgz", ".zip", ".parquet", ".safetensors", ".pt", ".pth", ".bin", ".whl",
                            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".zst", ".xz", ".bz2", ".7z", ".tar",
                            ".arrow", ".npy", ".npz", ".pdf"})


def category(rel: str, t: str, big_hf: frozenset, big_out: frozenset) -> str:
    """Categoria di archiviazione: le categorie sono una partizione dell'albero e si ripristinano indipendentemente."""
    parts = rel.split("/")
    top, n = parts[0], len(parts)
    if rel in STATE_FILES or top in CODE_TOPS or (n == 1 and t != "d"):
        return "codice"
    if top == "data":
        if n >= 4 and parts[1] == "shards" and (n > 4 or t == "d"):
            return f"data-shards-{parts[2]}-{parts[3]}"
        return "data-metadati"
    if top == "hf":
        if n >= 3 and parts[1] == "hub" and (n > 3 or t == "d") and parts[2] in big_hf:
            return f"hf-{parts[2]}"
        return "hf-piccoli"
    if top == "outputs":
        if n >= 2 and (n > 2 or t == "d") and parts[1] in big_out:
            return f"outputs-{parts[1]}"
        return "outputs-resto"
    if top == "venvs" or (top == "cache" and n >= 2 and parts[1] == "uv"):
        return "ambienti"  # uv installa i venv con hard link alla sua cache: stessi volumi, link conservati
    if top == "cache":
        return "cache-altre"
    if top == "wheels":
        return "wheels"
    if top == "tools":
        return "strumenti"
    return f"altro-{top}"


def is_nucleo(cat: str) -> bool:
    return (cat in NUCLEO_FIXED or cat in NUCLEO_SHARDS or cat in NUCLEO_HF or cat.startswith("outputs-")
            or cat.startswith("hf-models--"))


def cat_rank(cat: str) -> tuple:
    """Ordine di creazione e caricamento: prima il nucleo (i più piccoli e insostituibili), poi il resto."""
    order = [("codice", 0), ("data-metadati", 1), ("strumenti", 2), ("hf-piccoli", 3), ("hf-models--", 4),
             ("hf-datasets--", 5), ("outputs-resto", 6), ("outputs-", 7), ("data-shards-", 8), ("wheels", 9),
             ("ambienti", 10), ("cache-altre", 11), ("altro-", 12)]
    rank = next(r for prefix, r in order + [("", 99)] if cat.startswith(prefix))
    return (0 if is_nucleo(cat) else 1, rank, cat)


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="ascii") as fh:
        for line in fh:
            yield json.loads(line)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(4 * MIB):
            h.update(chunk)
    return h.hexdigest()


def write_new(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def signature(recs: list[dict]) -> str:
    h = hashlib.sha256()
    for r in recs:
        h.update(dump([r["p"], r["t"], r.get("s"), r["m"], r["mt"], r.get("l"), r["n"]]).encode() + b"\n")
    return h.hexdigest()


def assign_categories(records: list[dict], big_hf: frozenset, big_out: frozenset) -> list[dict]:
    """Categoria di ogni voce (campo `c`), con i nomi di un gruppo di hard link nella categoria del primo nome.
    La usano sia il piano sia la completezza, perché le voci attese siano calcolate nello stesso modo."""
    for r in records:
        r["c"] = category(r["p"], r["t"], big_hf, big_out)
    inode_cat: dict[int, str] = {}
    moved = []
    for r in records:
        if r["t"] == "f" and r["n"] > 1:
            first = inode_cat.setdefault(r["i"], r["c"])
            if first != r["c"]:
                moved.append({"p": r["p"], "da": r["c"], "a": first})
                r["c"] = first
    return moved


def inventory_problems(summary: dict) -> list[str]:
    """Errori di visita o di lettura dell'inventario: finché ci sono, l'inventario non descrive tutta la cartella."""
    out = []
    if summary.get("n_errori", 0):
        out.append(f"{summary['n_errori']} errori di visita o di stat (primi: {summary.get('errori', [])[:3]})")
    if summary.get("n_illeggibili", 0):
        out.append(f"{summary['n_illeggibili']} file illeggibili (primi: {summary.get('illeggibili', [])[:3]})")
    return out


def plan_exclusions(pdir: Path, plan: dict) -> dict[str, str]:
    """Esclusioni dichiarate nel piano (percorso -> motivo), verificate con lo sha256 registrato. I piani dello
    schema 1 non ne hanno."""
    meta = plan.get("esclusioni")
    if not meta:
        return {}
    path = pdir / meta["file"]
    if file_sha256(path) != meta["sha256"]:
        raise SystemExit(f"{path}: sha256 diverso da quello registrato nel piano")
    return {e["p"]: e["motivo"] for e in read_jsonl_gz(path)}


def compressible_share(recs: list[dict]) -> float:
    tot = inc = 0
    seen = set()
    for r in recs:
        if r["t"] != "f" or (r["n"] > 1 and r["i"] in seen):
            continue
        seen.add(r["i"])
        ext = os.path.splitext(r["p"])[1].lower()
        tot += r["s"]
        if ext in INCOMPRESSIBLE or (ext == "" and r["s"] > MIB):
            inc += r["s"]
    return 1.0 - inc / tot if tot else 1.0


def pack(cat: str, recs: list[dict], groups: dict[int, list[int]], budget: int) -> list[dict]:
    """Volumi di una categoria, nell'ordine dell'inventario. I nomi di un hard link stanno nel volume del primo nome
    (dati una volta sola, link conservati dall'estrazione); un file sopra il limite ha un volume a sé."""
    vols, cur, cur_cost, cur_data, emitted = [], [], 0, 0, set()

    def close():
        nonlocal cur, cur_cost, cur_data
        if cur:
            vols.append({"membri": cur, "dati": cur_data, "grande": False})
        cur, cur_cost, cur_data = [], 0, 0

    for k, r in enumerate(recs):
        if k in emitted:
            continue
        unit = groups.get(r["i"], [k]) if r["t"] == "f" and r["n"] > 1 else [k]
        data = r["s"] if r["t"] == "f" else 0  # i dati di un gruppo di hard link si scrivono una volta
        cost = data + ENTRY_OVERHEAD * len(unit) + len(os.fsencode(r.get("l", "")))
        emitted.update(unit)
        if data > budget - 64 * 1024:
            close()
            vols.append({"membri": unit, "dati": data, "grande": True})
            continue
        if cur and cur_cost + cost > budget - 64 * 1024:
            close()
        cur.extend(unit)
        cur_cost += cost
        cur_data += data
    close()
    return vols


def link_report(inv: Path) -> list[str]:
    lines = []
    for x in read_jsonl_gz(inv / "collegamenti.jsonl.gz"):
        flags = [f for f, bad in (("assoluto", x["assoluto"]), ("fuori dalla radice", not x["reale_dentro"]),
                                  ("interrotto", not x["esiste"]), ("verso escluso", x["verso_escluso"])) if bad]
        if flags:
            lines.append(f"  {x['p']} -> {x['l']}  [{', '.join(flags)}]")
    return lines


def checkpoint_report(records: list[dict]) -> list[str]:
    rx = re.compile(r"^outputs/[^/]+/(?:.*/)?(?:checkpoints|snapshots)/step-\d+$|^outputs/[^/]+/(?:.*/)?final$")
    dirs = {r["p"] for r in records if r["t"] == "d" and rx.match(r["p"])}
    files: dict[str, list] = defaultdict(list)
    for r in records:
        parent = r["p"].rpartition("/")[0]
        if parent in dirs and r["t"] == "f":
            files[parent].append((r["p"].rpartition("/")[2], r["s"]))
    out = ["cartella\tpesi\tottimizzatore\tstato_trainer\tfile"]
    for d in sorted(dirs):
        names = {n for n, _ in files[d]}
        weights = any(n.endswith((".safetensors", ".bin")) or (n.endswith(".pt") and n != "optimizer.pt")
                      for n in names)
        out.append(f"{d}\t{'sì' if weights else 'no'}\t{'sì' if 'optimizer.pt' in names else 'no'}\t"
                   f"{'sì' if 'trainer_state.json' in names else 'no'}\t"
                   + ";".join(f"{n}={s}" for n, s in sorted(files[d])))
    return out


def cmd_piano(args) -> None:
    need_slurm()
    if args.precedente:
        sys.exit("aggiornamento incrementale (--precedente) non supportato: la completezza non risolve le categorie "
                 "riusate da un'altra generazione. Creare una generazione autonoma (per esempio un supplemento con "
                 "--dipende-da) e ripristinarla dopo la base (docs/BACKUP_RIPRISTINO.md)")
    inv = Path(args.inventario)
    out = Path(args.out)
    summary = json.loads((inv / "riepilogo.json").read_text(encoding="utf-8"))
    secrets = json.loads((inv / "segreti.json").read_text(encoding="utf-8"))
    bad = inventory_problems(summary)
    if bad:
        sys.exit(f"inventario {inv} non accettabile come completo: {'; '.join(bad)}. Risolvere e rifare l'inventario")
    records = list(read_jsonl_gz(inv / "inventario.jsonl.gz"))

    present = [r["p"] for r in records if r["p"].startswith("secrets/")]
    if present:
        sys.exit(f"secrets/ contiene {len(present)} voci: vanno nell'archivio cifrato separato, non nel piano")
    strong = [h["p"] for h in secrets["file"]
              if any(secrets["regole"][rid]["affidabilita"] == "alta" for rid in h["regole"])]
    if strong:
        sys.exit(f"possibili segreti ad alta affidabilità in {strong[:5]}: separarli prima del piano")

    big_hf = frozenset(a.split("/")[2] for a, c in summary["aree"].items()
                       if a.startswith("hf/hub/") and a.count("/") == 2 and c.get("byte", 0) >= args.soglia_grandi)
    big_out = frozenset(a.split("/")[1] for a, c in summary["aree"].items()
                        if a.startswith("outputs/") and a.count("/") == 1 and c.get("byte", 0) >= args.soglia_grandi)
    moved = assign_categories(records, big_hf, big_out)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["c"]].append(r)
    cats = sorted(by_cat, key=cat_rank)
    if args.solo_categorie:
        cats = [c for c in cats if c in set(args.solo_categorie.split(","))]
    if args.escludi_categorie:
        cats = [c for c in cats if c not in set(args.escludi_categorie.split(","))]

    # Esclusioni dichiarate: tipi non archiviabili (socket) e voci fuori da un sottoinsieme dichiarato di una
    # categoria (con le cartelle che contengono i percorsi scelti). Nient'altro resta fuori da una categoria del piano.
    subsets: dict[str, set[str]] = {}
    for spec in args.sottoinsieme:
        cat, _, listfile = spec.partition("=")
        if cat not in cats or not listfile:
            sys.exit(f"--sottoinsieme {spec}: la categoria deve essere nel piano (CATEGORIA=FILE)")
        wanted = {x.strip() for x in Path(listfile).read_text(encoding="utf-8").splitlines() if x.strip()}
        known = {r["p"] for r in by_cat[cat]}
        if not wanted or not wanted <= known:
            sys.exit(f"--sottoinsieme {cat}: percorsi assenti dalla categoria: {sorted(wanted - known)[:5]}")
        keep = set(wanted)
        for p in wanted:  # le cartelle antenate dentro la categoria
            parts = p.split("/")
            keep.update("/".join(parts[:k]) for k in range(1, len(parts)))
        subsets[cat] = keep
    exclusions = []
    for c in cats:
        kept = []
        for r in by_cat[c]:
            if r["t"] not in ARCHIVABLE:
                exclusions.append({"p": r["p"], "categoria": c, "motivo": f"tipo {r['t']} non archiviabile"})
            elif c in subsets and r["p"] not in subsets[c]:
                exclusions.append({"p": r["p"], "categoria": c, "motivo": "fuori dal sottoinsieme dichiarato"})
            else:
                kept.append(r)
        by_cat[c] = kept
    out.mkdir(parents=True, exist_ok=False)
    with gzip.open(out / "esclusioni.jsonl.gz", "wt", encoding="ascii") as fh:
        for e in exclusions:
            fh.write(dump(e) + "\n")

    vol_rows, cat_info, members_out, order = [], {}, [], 0
    for c in cats:
        recs = by_cat[c]
        sig = signature(recs)
        share = compressible_share(recs)
        seen_ino: set[int] = set()
        nbytes = 0
        for r in recs:  # byte dei dati, una volta per inode
            if r["t"] == "f" and not (r["n"] > 1 and r["i"] in seen_ino):
                seen_ino.add(r["i"])
                nbytes += r["s"]
        info = {"nucleo": is_nucleo(c), "voci": len(recs), "firma": sig, "quota_comprimibile": round(share, 3),
                "compressione": "gzip" if share >= 0.5 else "nessuna", "byte": nbytes, "volumi": [],
                "sottoinsieme": c in subsets}
        groups: dict[int, list[int]] = defaultdict(list)
        for k, r in enumerate(recs):
            if r["t"] == "f" and r["n"] > 1:
                groups[r["i"]].append(k)
        for j, v in enumerate(pack(c, recs, groups, args.byte_volume), start=1):
            order += 1
            vid = f"{args.prefisso_volumi}{c}-{j:03d}"
            info["volumi"].append(vid)
            vol_rows.append({"ordine": order, "id": vid, "categoria": c, "nucleo": info["nucleo"],
                             "compressione": info["compressione"], "voci": len(v["membri"]), "byte": v["dati"],
                             "grande": v["grande"]})
            for k in v["membri"]:
                members_out.append({**recs[k], "v": vid})
        cat_info[c] = info

    plan = {"schema": PLAN_SCHEMA, "generazione": args.gen, "radice": summary["radice"],
            "dipende_da": args.dipende_da or None, "prefisso_volumi": args.prefisso_volumi,
            "inventario": {"cartella": str(inv.resolve()), "job": summary["job"], "inizio": summary["inizio"],
                           "sha256": file_sha256(inv / "inventario.jsonl.gz"), "git_head": summary["git_head"]},
            "byte_volume": args.byte_volume, "dimensione_parte": args.dimensione_parte,
            "hf_grandi": sorted(big_hf), "outputs_grandi": sorted(big_out), "esclusi": list(EXCLUDED_TOP),
            "esclusioni": {"file": "esclusioni.jsonl.gz", "sha256": file_sha256(out / "esclusioni.jsonl.gz"),
                           "voci": len(exclusions), "motivi": dict(Counter(e["motivo"] for e in exclusions))},
            "volatili": sorted(VOLATILE_FILES),
            "hardlink_spostati": moved, "categorie": cat_info, "volumi": vol_rows, "job": os.environ["SLURM_JOB_ID"],
            "creato": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_new(out / "piano.json", json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
    with gzip.open(out / "membri.jsonl.gz", "wt", encoding="ascii") as fh:
        for m in members_out:
            fh.write(dump(m) + "\n")
    tsv = ["ordine\tvolume\tcategoria\tnucleo\tcompressione\tvoci\tbyte_dati\tgrande"]
    tsv += [f"{v['ordine']}\t{v['id']}\t{v['categoria']}\t{int(v['nucleo'])}\t{v['compressione']}\t{v['voci']}\t"
            f"{v['byte']}\t{int(v['grande'])}" for v in vol_rows]
    write_new(out / "volumi.tsv", "\n".join(tsv) + "\n")
    ctsv = ["categoria\tnucleo\tcompressione\tquota_comprimibile\tvoci\tbyte\tvolumi\tsottoinsieme\tfirma"]
    ctsv += [f"{c}\t{int(i['nucleo'])}\t{i['compressione']}\t{i['quota_comprimibile']}\t{i['voci']}\t{i['byte']}\t"
             f"{len(i['volumi'])}\t{int(i['sottoinsieme'])}\t{i['firma']}" for c, i in cat_info.items()]
    write_new(out / "categorie.tsv", "\n".join(ctsv) + "\n")
    write_new(out / "checkpoint.tsv", "\n".join(checkpoint_report(records)) + "\n")

    nuc = sum(i["byte"] for i in cat_info.values() if i["nucleo"])
    tot = sum(i["byte"] for i in cat_info.values())
    rep = [f"piano {args.gen}: {len(cat_info)} categorie, {len(vol_rows)} volumi; dati {tot / GIB:.2f} GiB "
           f"(nucleo {nuc / GIB:.2f} GiB, resto {(tot - nuc) / GIB:.2f} GiB); esclusioni dichiarate "
           f"{len(exclusions)}" + (f"; dipende da {args.dipende_da}" if args.dipende_da else ""),
           f"inventario {inv} (job {summary['job']}, git {summary['git_head'][:12]})", "",
           f"{'categoria':58} {'nucl':4} {'compr':7} {'quota':5} {'GiB':>8} {'vol':>4}"]
    rep += [f"{c[:58]:58} {('sì' if i['nucleo'] else 'no'):4} {i['compressione']:7} {i['quota_comprimibile']:5} "
            f"{i['byte'] / GIB:8.2f} {len(i['volumi']):4}" + ("  sottoinsieme dichiarato" if i["sottoinsieme"] else "")
            for c, i in cat_info.items()]
    rep += ["", "esclusioni dichiarate:", *[f"  {e['p']}  [{e['motivo']}]" for e in exclusions[:40]], "",
            f"hard link spostati di categoria: {len(moved)}", *[f"  {m['p']}: {m['da']} -> {m['a']}" for m in moved[:20]],
            "",
            "collegamenti assoluti, fuori dalla radice, interrotti o verso materiale escluso:", *link_report(inv), "",
            "segnalazioni della ricerca dei segreti (nessuna ad alta affidabilità; da riesaminare):"]
    rep += [f"  {h['p']}: {', '.join(f'{k} ({secrets['regole'][k]['affidabilita']}) righe {v}' for k, v in h['regole'].items())}"
            for h in secrets["file"]]
    rep += [f"  oggetti git {repo}: {len(objs)}" for repo, objs in secrets.get("oggetti_git", {}).items()]
    write_new(out / "report.txt", "\n".join(rep) + "\n")
    with open(out / "SHA256SUMS", "w", encoding="ascii") as fh:
        for name in sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
            fh.write(f"{file_sha256(out / name)}  {name}\n")
    print("\n".join(rep[:4]))


# ---------------------------------------------------------------- creazione e verifica dei volumi

class PartWriter:
    """Flusso scritto in parti di al più `part_size` byte; sha256 e md5 per parte e per tutto il flusso. Le parti
    restano `.tmp` fino a close(), che le rinomina: un solo file si chiama come il volume, altrimenti `.partNNN`."""

    def __init__(self, base: Path, part_size: int):
        self.base, self.part_size = base, part_size
        self.parts: list[dict] = []
        self.sha, self.md5, self.pos, self.fh = hashlib.sha256(), hashlib.md5(), 0, None

    def _finish(self) -> None:
        if self.fh is not None:
            self.fh.flush()
            os.fsync(self.fh.fileno())
            self.fh.close()
            self.fh = None

    def _new(self) -> None:
        tmp = self.base.parent / f"{self.base.name}.part{len(self.parts):03d}.tmp"
        self.fh = open(tmp, "xb")
        self.parts.append({"tmp": tmp, "byte": 0, "sha": hashlib.sha256(), "md5": hashlib.md5()})

    def write(self, data) -> int:
        mv = memoryview(data).cast("B")
        n = len(mv)
        while mv:
            if self.fh is None or self.parts[-1]["byte"] == self.part_size:
                self._finish()
                self._new()
            cur = self.parts[-1]
            take = mv[: self.part_size - cur["byte"]]
            self.fh.write(take)
            for h in (cur["sha"], cur["md5"], self.sha, self.md5):
                h.update(take)
            cur["byte"] += len(take)
            self.pos += len(take)
            mv = mv[len(take):]
        return n

    def tell(self) -> int:
        return self.pos

    def flush(self) -> None:
        if self.fh is not None:
            self.fh.flush()

    def close(self) -> list[dict]:
        if not self.parts:
            self._new()
        self._finish()
        single = len(self.parts) == 1
        out = []
        for k, p in enumerate(self.parts):
            name = self.base.name if single else f"{self.base.name}.part{k:03d}"
            os.replace(p["tmp"], self.base.parent / name)
            out.append({"nome": name, "byte": p["byte"], "sha256": p["sha"].hexdigest(), "md5": p["md5"].hexdigest()})
        return out

    def abort(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None
        for p in self.parts:
            p["tmp"].unlink(missing_ok=True)


class HashReader:
    def __init__(self, fh):
        self.fh, self.sha, self.n = fh, hashlib.sha256(), 0

    def read(self, size: int = -1) -> bytes:
        b = self.fh.read(size)
        self.sha.update(b)
        self.n += len(b)
        return b


class ConcatReader:
    """Legge le parti di un volume in sequenza, con sha256 e md5 per parte e per tutto il flusso."""

    def __init__(self, paths: list[Path]):
        self.paths, self.i, self.fh = list(paths), 0, None
        self.sha, self.md5, self.parts, self.pos = hashlib.sha256(), hashlib.md5(), [], 0

    def read(self, size: int = -1) -> bytes:
        out, need = [], (size if size is not None and size >= 0 else 1 << 62)
        while need > 0:
            if self.fh is None:
                if self.i >= len(self.paths):
                    break
                self.fh = open(self.paths[self.i], "rb")
                self.cur = {"byte": 0, "sha": hashlib.sha256(), "md5": hashlib.md5()}
            b = self.fh.read(min(need, 4 * MIB))
            if not b:
                self.fh.close()
                self.fh = None
                self.parts.append({"byte": self.cur["byte"], "sha256": self.cur["sha"].hexdigest(),
                                   "md5": self.cur["md5"].hexdigest()})
                self.i += 1
                continue
            for h in (self.cur["sha"], self.cur["md5"], self.sha, self.md5):
                h.update(b)
            self.cur["byte"] += len(b)
            self.pos += len(b)
            out.append(b)
            need -= len(b)
        return b"".join(out)

    def drain(self) -> None:
        while self.read(4 * MIB):
            pass


def sha256sum_line(sha: str, path: str) -> str:
    """Riga nel formato di GNU sha256sum (con il prefisso \\ se il nome contiene \\ o a capo)."""
    if "\\" in path or "\n" in path or "\r" in path:
        esc = path.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
        return f"\\{sha}  {esc}"
    return f"{sha}  {path}"


def build_volume(root: Path, vid: str, members: list[dict], compress: bool, dest: Path, part_size: int):
    """Scrive il volume (tar PAX, gzip facoltativo) leggendo ogni file una volta: sha256 per file e per parte.
    Ogni avviso ha `bloccante`: un membro mancante, non archiviabile o di tipo cambiato, o un file non volatile
    cambiato dopo l'inventario, impedisce di dichiarare il volume pronto; il cambiamento di un file volatile
    (VOLATILE_FILES, per esempio il ledger) resta un avviso motivato."""
    ext = ".tar.gz" if compress else ".tar"
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob(f"{vid}{ext}*"):  # resti di un tentativo interrotto di questo stesso volume
        stale.unlink()
    pw = PartWriter(dest / f"{vid}{ext}", part_size)
    gz = gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=pw, mtime=0) if compress else None
    tf = tarfile.TarFile(fileobj=gz or pw, mode="w", format=tarfile.PAX_FORMAT, copybufsize=4 * MIB)
    index, warnings, hashes = [], [], {}
    try:
        for r in members:
            rel, ap = r["p"], root / r["p"]
            volatile = rel in VOLATILE_FILES
            try:
                st = os.lstat(ap)
            except FileNotFoundError:
                warnings.append({"p": rel, "avviso": "mancante al momento dell'archiviazione", "bloccante": True})
                continue
            t = kind(st.st_mode)
            if t != r["t"]:
                warnings.append({"p": rel, "avviso": f"tipo {t} invece di {r['t']}", "bloccante": True})
                continue
            if (t == "f" and (st.st_size != r["s"] or st.st_mtime_ns != r["mt"])) or (
                    t == "l" and os.readlink(ap) != r.get("l")):
                warnings.append({"p": rel, "avviso": "cambiato dopo l'inventario", "bloccante": not volatile})
            ti = tf.gettarinfo(str(ap), arcname=rel)
            if ti is None:
                warnings.append({"p": rel, "avviso": f"tipo {t} non archiviabile", "bloccante": True})
                continue
            ti.mtime = st.st_mtime_ns // 1_000_000_000  # secondi interi: nessuna intestazione PAX per la data
            ti.mode = stat.S_IMODE(ti.mode)  # gettarinfo dà st_mode intero; il tar conserva solo i permessi
            e = {"p": rel, "m": ti.mode, "mt": ti.mtime, "u": ti.uid, "g": ti.gid}
            if ti.isreg():
                with open(ap, "rb") as fh:
                    hr = HashReader(fh)
                    tf.addfile(ti, hr)
                st2 = os.lstat(ap)
                if st2.st_size != ti.size or st2.st_mtime_ns != st.st_mtime_ns:
                    warnings.append({"p": rel, "avviso": "modificato durante la lettura", "bloccante": not volatile})
                e.update(t="f", s=ti.size, sha256=hr.sha.hexdigest())
                hashes[rel] = e["sha256"]
            elif ti.islnk():
                tf.addfile(ti)
                e.update(t="h", l=ti.linkname, sha256=hashes.get(ti.linkname))
            elif ti.issym():
                tf.addfile(ti)
                e.update(t="l", l=ti.linkname)
            elif ti.isdir():
                tf.addfile(ti)
                e.update(t="d")
            else:
                tf.addfile(ti)
                e.update(t=t)
            index.append(e)
        tf.close()
        if gz is not None:
            gz.close()
        parts = pw.close()
    except BaseException:
        pw.abort()
        raise
    stream = {"byte": pw.pos, "sha256": pw.sha.hexdigest(), "md5": pw.md5.hexdigest()}
    return parts, stream, index, warnings


def gnu_tar_names(paths: list[Path], compress: bool) -> list[str]:
    """Elenco dei nomi letto da GNU tar (lo strumento standard del ripristino) dalle parti concatenate."""
    cmd = 'cat -- "$@" | tar -t' + ("z" if compress else "") + "f - --quoting-style=literal"
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd, "_", *map(str, paths)], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"GNU tar non legge il volume: {r.stderr.decode(errors='replace')[:300]}")
    return [os.fsdecode(x).rstrip("/") for x in r.stdout.split(b"\n") if x]


def verify_volume(paths: list[Path], compress: bool, index: list[dict]) -> tuple[list[str], ConcatReader]:
    """Rilettura completa: ogni voce del tar coincide con l'indice (nome, tipo, permessi, data, link, sha256)."""
    cr = ConcatReader(paths)
    src = gzip.GzipFile(fileobj=cr, mode="rb") if compress else cr
    tf = tarfile.open(fileobj=src, mode="r|", bufsize=MIB)
    problems, k = [], 0
    for m in tf:
        if k >= len(index):
            problems.append(f"voce in più: {m.name}")
            break
        e = index[k]
        k += 1
        want = {"f": m.isreg(), "h": m.islnk(), "l": m.issym(), "d": m.isdir()}.get(e["t"], True)
        if m.name != e["p"] or not want:
            problems.append(f"{e['p']}: nel tar {m.name} di tipo {m.type!r}")
            continue
        if m.mode != e["m"] or m.mtime != e["mt"]:
            problems.append(f"{e['p']}: permessi o data diversi")
        if e["t"] in ("h", "l") and m.linkname != e["l"]:
            problems.append(f"{e['p']}: destinazione {m.linkname} invece di {e['l']}")
        if e["t"] == "f":
            fh = tf.extractfile(m)
            h, n = hashlib.sha256(), 0
            while chunk := fh.read(4 * MIB):
                h.update(chunk)
                n += len(chunk)
            if n != e["s"] or h.hexdigest() != e["sha256"]:
                problems.append(f"{e['p']}: contenuto diverso dall'indice")
    if k != len(index):
        problems.append(f"voci dell'indice non trovate nel tar: {len(index) - k}")
    while src.read(4 * MIB):  # blocchi finali e riempimento: chiude gli hash del flusso e il CRC di gzip
        pass
    cr.drain()
    return problems, cr


def staged_bytes(staging: Path) -> int:
    return sum(p.stat().st_size for p in staging.rglob("*") if p.is_file()) if staging.exists() else 0


def cmd_crea(args) -> None:
    need_slurm()
    pdir = Path(args.piano)
    plan = json.loads((pdir / "piano.json").read_text(encoding="utf-8"))
    root, gen = Path(plan["radice"]), plan["generazione"]
    staging, state = Path(args.staging) / gen, Path(args.stato) / gen
    for d in (state / "vol", state / "indici", staging):
        d.mkdir(parents=True, exist_ok=True)
    members: dict[str, list[dict]] = defaultdict(list)
    for m in read_jsonl_gz(pdir / "membri.jsonl.gz"):
        members[m["v"]].append(m)
    wanted = set(args.volumi.split(",")) if args.volumi else None
    limit = int(args.limite_staging_gib * GIB)
    plan_sha = file_sha256(pdir / "piano.json")
    done = 0
    for v in plan["volumi"]:
        vid, cat = v["id"], v["categoria"]
        if wanted is not None and vid not in wanted:
            continue
        if args.solo_nucleo and not v["nucleo"]:  # backup selezionato: solo le categorie del nucleo
            continue
        marker = state / "vol" / f"{vid}.pronto.json"
        if marker.exists():  # si riusa solo un volume di questa generazione, di questo piano e di questo volume
            old = json.loads(marker.read_text(encoding="utf-8"))
            mismatch = [k for k, want in (("generazione", gen), ("volume", vid), ("categoria", cat),
                                          ("piano_sha256", plan_sha)) if old.get(k) != want]
            if mismatch:
                sys.exit(f"{marker}: marcatore di un altro piano o volume (campi diversi: {mismatch}; piano del "
                         f"marcatore {str(old.get('piano_sha256'))[:12]}, piano corrente {plan_sha[:12]}). Non lo "
                         "sovrascrivo: usare una generazione nuova o una cartella di stato separata")
            continue
        if args.massimo is not None and done >= args.massimo:
            break
        while True:  # staging limitato: si aspetta che il caricamento verificato liberi spazio
            used = staged_bytes(staging)
            if used == 0 or used + v["byte"] <= limit:
                break
            if not args.attendi:
                print(f"staging pieno ({used / GIB:.1f} GiB): mi fermo prima di {vid}")
                return
            time.sleep(60)
        if (Path(args.stato) / "STOP").exists():
            print(f"trovato {Path(args.stato) / 'STOP'}: mi fermo prima di {vid}")
            return
        t0 = time.time()
        compress = plan["categorie"][cat]["compressione"] == "gzip"
        parts, stream, index, warnings = build_volume(root, vid, members[vid], compress, staging / cat,
                                                      plan["dimensione_parte"])
        t1 = time.time()
        paths = [staging / cat / p["nome"] for p in parts]
        problems, cr = verify_volume(paths, compress, index)
        if cr.sha.hexdigest() != stream["sha256"] or [p["sha256"] for p in cr.parts] != [p["sha256"] for p in parts]:
            problems.append("hash delle parti diversi tra scrittura e rilettura")
        names = gnu_tar_names(paths, compress)
        if names != [e["p"] for e in index]:
            problems.append(f"GNU tar elenca {len(names)} nomi diversi dall'indice ({len(index)})")
        blocking = [w for w in warnings if w["bloccante"]]
        problems += [f"{w['p']}: {w['avviso']}" for w in blocking]
        if [e["p"] for e in index] != [m["p"] for m in members[vid]]:
            problems.append("le voci archiviate non sono tutte e sole quelle del piano")
        if problems:
            for p in paths:
                p.unlink(missing_ok=True)
            (state / "vol" / f"{vid}.errore.json").write_text(json.dumps(
                {"volume": vid, "piano_sha256": plan_sha, "problemi": problems[:200], "n_problemi": len(problems)},
                indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            sys.exit(f"{vid}: volume non dichiarato pronto ({len(problems)} problemi, primi: {problems[:5]}); parti "
                     "rimosse dallo staging. Se l'albero è cambiato, rifare inventario e piano")
        idx_gz = state / "indici" / f"{vid}.jsonl.gz"
        tmp_gz = idx_gz.with_name(idx_gz.name + ".tmp")
        with open(tmp_gz, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as g:
            for e in index:
                g.write((dump(e) + "\n").encode("ascii"))
        os.replace(tmp_gz, idx_gz)
        sums = state / "indici" / f"{vid}.sha256"
        write_new(sums, "".join(sha256sum_line(e["sha256"], e["p"]) + "\n" for e in index if e.get("sha256")))
        extra = []
        for f in (idx_gz, sums):
            data = f.read_bytes()
            extra.append({"nome": f.name, "byte": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                          "md5": hashlib.md5(data).hexdigest()})
        rows = ["tipo\tfile\tbyte\tsha256\tmd5\tlocale\tremoto"]
        rows += [f"volume\t{p['nome']}\t{p['byte']}\t{p['sha256']}\t{p['md5']}\t{staging / cat / p['nome']}\t"
                 f"volumi/{cat}/{p['nome']}" for p in parts]
        rows += [f"indice\t{x['nome']}\t{x['byte']}\t{x['sha256']}\t{x['md5']}\t{state / 'indici' / x['nome']}\t"
                 f"indice/volumi/{x['nome']}" for x in extra]
        write_new(state / "vol" / f"{vid}.parti.tsv", "\n".join(rows) + "\n")
        info = {"schema": "limite-vl-backup-volume/2", "volume": vid, "categoria": cat, "generazione": gen,
                "ordine": v["ordine"], "nucleo": v["nucleo"], "compressione": "gzip" if compress else "nessuna",
                "voci": len(index), "file": sum(1 for e in index if e["t"] == "f"),
                "byte_dati": sum(e["s"] for e in index if e["t"] == "f"), "flusso": stream, "parti": parts,
                "indice": extra, "avvisi": warnings, "job": os.environ.get("SLURM_JOB_ID"),
                "piano_sha256": plan_sha,
                "creato": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "secondi_scrittura": round(t1 - t0, 1),
                "secondi_verifica": round(time.time() - t1, 1),
                "verifica": "rilettura completa (nomi, tipi, permessi, date, link, sha256 per file e per parte) e "
                            "elenco di GNU tar uguale all'indice"}
        write_new(marker, json.dumps(info, indent=1, ensure_ascii=False) + "\n")
        with open(state / "registro.tsv", "a", encoding="utf-8") as fh:
            fh.write(f"{info['creato']}\tpronto\t{vid}\t{len(parts)} parti\t{stream['byte']}\t{stream['sha256']}\t"
                     f"{len(warnings)} avvisi\tjob {info['job']}\n")
        done += 1
        print(f"{vid}: {len(index)} voci, {info['byte_dati'] / GIB:.2f} GiB di dati, {stream['byte'] / GIB:.2f} GiB "
              f"in {len(parts)} parti, {info['secondi_scrittura']} + {info['secondi_verifica']} s, "
              f"{len(warnings)} avvisi", flush=True)


def cmd_ripristino(args) -> None:
    """Prova di ripristino con gli strumenti standard: hash delle parti, cat | tar -xp, confronto con l'indice e
    sha256sum -c; la copia di prova (creata qui) si rimuove alla fine se richiesto."""
    need_slurm()
    if args.esistente and args.rimuovi:
        sys.exit("--esistente e --rimuovi insieme: una radice esistente non si rimuove mai")
    info = json.loads(Path(args.pronto).read_text(encoding="utf-8"))
    folder = Path(args.cartella)
    paths = [folder / p["nome"] for p in info["parti"]]
    problems = []
    for p, want in zip(paths, info["parti"]):
        if not p.is_file() or p.stat().st_size != want["byte"] or file_sha256(p) != want["sha256"]:
            problems.append(f"{p.name}: dimensione o sha256 diversi dal volume pronto")
    index = list(read_jsonl_gz(Path(args.indice)))
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=args.esistente)  # --esistente: ripristino vero, volume dopo volume
    flags = "-xp" + ("z" if info["compressione"] == "gzip" else "") + "f"
    r = subprocess.run(["bash", "-o", "pipefail", "-c", f'cat -- "${{@:2}}" | tar {flags} - -C "$1"', "_", str(dest),
                        *map(str, paths)], capture_output=True)
    if r.returncode != 0:
        problems.append(f"estrazione con GNU tar fallita: {r.stderr.decode(errors='replace')[:300]}")
    checked = Counter()
    for e in index:
        p = dest / e["p"]
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            problems.append(f"{e['p']}: mancante dopo l'estrazione")
            continue
        t = kind(st.st_mode)
        if e["t"] == "h":
            if t != "f" or st.st_ino != os.lstat(dest / e["l"]).st_ino:
                problems.append(f"{e['p']}: hard link non ricostruito")
        elif t != e["t"]:
            problems.append(f"{e['p']}: tipo {t} invece di {e['t']}")
            continue
        if e["t"] == "l":
            if os.readlink(p) != e["l"]:
                problems.append(f"{e['p']}: destinazione del link diversa")
        else:
            if stat.S_IMODE(st.st_mode) != e["m"]:
                problems.append(f"{e['p']}: permessi {oct(stat.S_IMODE(st.st_mode))} invece di {oct(e['m'])}")
            if e["t"] != "d" and st.st_mtime_ns // 1_000_000_000 != e["mt"]:
                problems.append(f"{e['p']}: data diversa")
        if e["t"] == "f" and file_sha256(p) != e["sha256"]:
            problems.append(f"{e['p']}: sha256 diverso")
        checked[e["t"]] += 1
    sums = Path(args.sha256sum)
    if sums.stat().st_size:  # un volume di sole cartelle e link ha il file delle somme vuoto
        s = subprocess.run(["sha256sum", "--quiet", "--strict", "-c", str(sums)], cwd=dest, capture_output=True)
        if s.returncode != 0:
            problems.append(f"sha256sum -c: {(s.stdout + s.stderr).decode(errors='replace')[:300]}")
        checked["sha256sum_c"] = sum(1 for _ in sums.open(encoding="utf-8"))
    report = {"volume": info["volume"], "parti": len(paths), "voci_controllate": dict(checked),
              "problemi": problems[:100], "n_problemi": len(problems), "estrazione": f"cat parti | tar {flags} -",
              "job": os.environ.get("SLURM_JOB_ID"), "data": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.rimuovi:
        shutil.rmtree(dest)
        report["copia_di_prova"] = "rimossa"
    Path(args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if problems:
        sys.exit(1)


# ---------------------------------------------------------------- catalogo di recupero e completezza

def read_tsv(path: Path) -> list[dict]:
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x and not x.startswith("#")]
    head = lines[0].split("\t")
    return [dict(zip(head, x.split("\t"))) for x in lines[1:]]


def write_tsv(path: Path, head: list[str], rows: list[list]) -> None:
    def clean(v) -> str:
        return str(v).replace("\t", " ").replace("\n", " ")
    write_new(path, "\n".join(["\t".join(head)] + ["\t".join(clean(v) for v in r) for r in rows]) + "\n")


def volume_hashes(state: Path) -> dict[str, tuple[str, str]]:
    """sha256 e volume di ogni file archiviato, dagli indici dei volumi pronti."""
    out: dict[str, tuple[str, str]] = {}
    for idx in sorted((state / "indici").glob("*.jsonl.gz")):
        vid = idx.name[: -len(".jsonl.gz")]
        for e in read_jsonl_gz(idx):
            if e.get("sha256"):
                out[e["p"]] = (e["sha256"], vid)
    return out


def cmd_catalogo(args) -> None:
    """Catalogo di recupero (repository, commit, file esatti, dimensioni, hash, presenza locale, ruolo), derivati,
    riferimento degli hash per il confronto con le copie sul Mac, ambienti, dipendenze esterne e percorsi assoluti."""
    need_slurm()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inv = Path(args.inventario)
    records = {r["p"]: r for r in read_jsonl_gz(inv / "inventario.jsonl.gz")}
    hashes = volume_hashes(Path(args.stato))
    status = {r["repo"]: r for r in read_tsv(root / "downloads/catalog_status.tsv")}

    rows = []
    for spec_path in sorted((root / "data/snapshot-specs").glob("*.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        cache = os.path.relpath(spec["cache_repo"], root) if spec["cache_repo"].startswith(str(root)) else spec["cache_repo"]
        st = status.get(spec["repo"], {})
        for name, f in sorted(spec["files"].items()):
            etag = f.get("etag", "")
            kind_h = "sha256" if re.fullmatch(r"[0-9a-f]{64}", etag) else "gitsha1" if re.fullmatch(r"[0-9a-f]{40}", etag) else "?"
            blob = f"{cache}/blobs/{etag}"
            present = blob in records and records[blob]["s"] == f.get("size")
            local = hashes.get(blob, ("", ""))
            rows.append([spec["type"], spec["repo"], spec["commit"], spec.get("requested_revision", ""), name,
                         f.get("size", ""), kind_h, etag, "sì" if present else "no", local[0], local[1],
                         st.get("gruppo", "-"), st.get("ruolo_pipeline", "-"), st.get("ammissione", "-"),
                         "pubblico: scaricato senza token (secrets/ vuota)" if present else "da verificare sul sito"])
    write_tsv(out / "catalogo_recupero.tsv",
              ["tipo", "repository", "commit", "revisione_richiesta", "file", "byte", "tipo_hash", "hash_hf",
               "presente", "sha256_locale", "volume", "gruppo", "ruolo", "ammissione", "accesso"], rows)

    ref = [[h, records[p]["s"], p, v] for p, (h, v) in sorted(hashes.items())
           if p.startswith(("hf/hub/", "data/shards/")) and p in records]
    write_tsv(out / "hash_riferimento_dati.tsv", ["sha256", "byte", "percorso", "volume"], ref)

    ledger = {}
    for line in (root / "runs/ledger.tsv").read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 5 and p[3] == "submitted":
            ledger.setdefault(p[1], (p[2], p[4]))
    der = []
    for man in sorted(root.glob("data/shards/*/*/manifest.json")):
        m = json.loads(man.read_text(encoding="utf-8"))
        d = os.path.relpath(man.parent, root)
        files = [r for p, r in records.items() if p.startswith(d + "/") and r["t"] == "f"]
        job = str(m.get("job_id", m.get("job", "")))
        name, note = ledger.get(job, ("-", "-"))
        der.append([d, m.get("source", "-"), job or "-", m.get("commit", "-"), name, note[:300], len(files),
                    sum(r["s"] for r in files)])
    write_tsv(out / "derivati.tsv", ["cartella", "sorgente", "job", "commit", "launcher", "nota_del_ledger",
                                      "file", "byte"], der)

    env = []
    for venv in sorted((root / "venvs").glob("*build-*")):
        cfg = (venv / "pyvenv.cfg").read_text(encoding="utf-8") if (venv / "pyvenv.cfg").is_file() else ""
        job = venv.name.rsplit("-", 1)[1]
        freeze = root / f"outputs/env/freeze-{job}.txt"
        keys = []
        if freeze.is_file():
            for line in freeze.read_text(encoding="utf-8").splitlines():
                if re.match(r"(torch|torchvision|torchaudio|transformers|vllm|numpy|verifiers|limite[-_a-z]*)\b", line, re.I):
                    keys.append(line.strip())
        target = os.readlink(venv / "bin/python") if (venv / "bin/python").is_symlink() else "-"
        env.append([os.path.relpath(venv, root), target, " ".join(cfg.split("\n")).strip(),
                    os.path.relpath(freeze, root) if freeze.is_file() else "-", "; ".join(keys)])
    for lock in ("env/uv.lock", "env-evals/uv.lock", "env/pyproject.toml", "env-evals/pyproject.toml"):
        if (root / lock).is_file():
            env.append([lock, "-", f"sha256 {file_sha256(root / lock)}", "-", "-"])
    for tool in ("tools/uv", "tools/rclone/rclone"):
        if (root / tool).is_file():
            env.append([tool, "-", f"sha256 {file_sha256(root / tool)}", "-", "-"])
    git = json.loads((inv / "git.json").read_text(encoding="utf-8"))
    for name, g in git.get("third_party", {}).items():
        env.append([f"third_party/{name}", "-", f"commit {g['head']} shallow={g['shallow']}", "-",
                    " ".join(g["remotes"].split()[:2])])
    write_tsv(out / "ambienti.tsv", ["voce", "python_di_base", "identita", "freeze", "pacchetti_chiave"], env)

    ext = []
    for x in read_jsonl_gz(inv / "collegamenti.jsonl.gz"):
        if x["assoluto"] or not x["reale_dentro"]:
            ext.append([x["p"], x["l"], "assoluto" if x["assoluto"] else "relativo",
                        "dentro la radice" if x["reale_dentro"] else "fuori dalla radice", "sì" if x["esiste"] else "no"])
    write_tsv(out / "collegamenti_assoluti_o_esterni.tsv", ["link", "destinazione", "tipo", "risolto", "esiste"], ext)

    needle = str(root).encode()
    counts: Counter = Counter()
    listed = []
    for p, r in records.items():
        if r["t"] != "f" or r["s"] > SECRET_MAX_BYTES or p.startswith(SECRET_SKIP) or "/.git/" in f"/{p}":
            continue
        try:
            data = (root / p).read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192] or needle not in data:
            continue
        a = area_of(p, False)
        counts[a] += 1
        if p.split("/")[0] in ("config", "scripts", "slurm", "src", "tests", "downloads", "env", "env-evals") \
                or p.startswith("data/manifests/") or p.startswith("data/snapshot-specs/"):
            listed.append([p, data.count(needle)])
    write_tsv(out / "percorsi_assoluti.tsv", ["file", "occorrenze"], sorted(listed))
    write_tsv(out / "percorsi_assoluti_per_area.tsv", ["area", "file_di_testo_con_la_radice"],
              sorted(counts.items(), key=lambda x: -x[1]))
    print(f"catalogo: {len(rows)} file di {len({r[1] for r in rows})} repository, {len(ref)} hash di riferimento, "
          f"{len(der)} derivati, {len(env)} voci d'ambiente, {len(ext)} link, {sum(counts.values())} file con la radice "
          f"-> {out}")


def cmd_firme(args) -> None:
    """Confronto in sola lettura fra un inventario nuovo e un piano di base: per ogni categoria del piano o
    dell'inventario dice se la firma (percorsi, tipi, dimensioni, date, permessi, link) è uguale, cambiata, nuova o
    scomparsa. Serve a decidere cosa mettere in un supplemento autonomo; non riusa volumi."""
    need_slurm()
    base = json.loads(Path(args.piano_base).read_text(encoding="utf-8"))
    inv = Path(args.inventario)
    summary = json.loads((inv / "riepilogo.json").read_text(encoding="utf-8"))
    records = list(read_jsonl_gz(inv / "inventario.jsonl.gz"))
    assign_categories(records, frozenset(base["hf_grandi"]), frozenset(base["outputs_grandi"]))
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["c"]].append(r)
    out = {}
    for c in sorted(set(by_cat) | set(base["categorie"]), key=cat_rank):
        if c not in base["categorie"]:
            out[c] = "nuova"
        elif c not in by_cat:
            out[c] = "scomparsa"
        else:
            out[c] = "uguale" if signature(by_cat[c]) == base["categorie"][c]["firma"] else "cambiata"
    rep = {"piano_base": args.piano_base, "generazione_base": base["generazione"], "inventario": str(inv),
           "errori_inventario": inventory_problems(summary), "categorie": out,
           "conteggi": dict(Counter(out.values())), "job": os.environ.get("SLURM_JOB_ID")}
    Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("conteggi", "errori_inventario")}, ensure_ascii=False),
          [c for c, s in out.items() if s != "uguale"])


def cmd_rimossi(args) -> None:
    """Percorsi presenti negli indici della base e assenti in quelli nuovi (per esempio file cancellati o rinominati
    fra il codice di g1 e quello del supplemento): elenco per `ripristina_volumi.sh --rimuovi-elenco`, prima i file e
    poi le cartelle dalla più profonda."""
    need_slurm()
    old = {e["p"]: e["t"] for f in args.base for e in read_jsonl_gz(Path(f))}
    new = {e["p"] for f in args.nuovo for e in read_jsonl_gz(Path(f))}
    gone = [p for p in old if p not in new]
    files = sorted(p for p in gone if old[p] != "d")
    dirs = sorted((p for p in gone if old[p] == "d"), key=lambda p: (-p.count("/"), p))
    Path(args.out).write_text("".join(f"{p}\n" for p in files + dirs), encoding="utf-8")
    print(f"rimossi: {len(files)} file e link, {len(dirs)} cartelle -> {args.out}")


def git_run(root: Path, *a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={root}", "-C", str(root), *a], capture_output=True)


def cmd_composizione(args) -> None:
    """Verifica in sola lettura di un ripristino in più passaggi (la base, poi i supplementi). La radice deve contenere
    esattamente lo stato finale descritto dagli indici applicati in ordine (per un percorso vale l'ultimo indice che
    lo contiene), meno i percorsi rimossi: niente file mancanti, niente file vecchi della base rimasti, stessi tipi,
    permessi, date, link e sha256. A richiesta controlla anche il commit di Git (stato pulito salvo i file volatili e
    connettività degli oggetti), confronta alcune cartelle con il progetto e con il commit, e cerca file attesi per
    sha256 seguendo i link dentro la radice."""
    need_slurm()
    root = Path(args.radice).resolve()
    final: dict[str, dict] = {}
    origin: dict[str, str] = {}
    for f in args.indici:
        for e in read_jsonl_gz(Path(f)):
            final[e["p"]] = e
            origin[e["p"]] = Path(f).name
    removed = [x for x in Path(args.rimossi).read_text(encoding="utf-8").splitlines() if x] if args.rimossi else []
    for p in removed:
        final.pop(p, None)
    implicit = set()
    for p in final:
        d = os.path.dirname(p)
        while d and d not in final:
            implicit.add(d)
            d = os.path.dirname(d)
    present: dict[str, os.stat_result] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for n in dirnames + filenames:
            full = os.path.join(dirpath, n)
            present[os.path.relpath(full, root)] = os.lstat(full)
    problems: list[str] = []
    missing = sorted(p for p in final if p not in present)
    extra = sorted(p for p in present if p not in final and p not in implicit)
    problems += [f"{p}: mancante (atteso da {origin[p]})" for p in missing]
    problems += [f"{p}: presente ma assente dallo stato finale (file vecchio della base?)" for p in extra]
    problems += [f"{p}: dovrebbe essere una cartella" for p in sorted(implicit)
                 if p in present and kind(present[p].st_mode) != "d"]
    checked = Counter()
    for p, e in final.items():
        st = present.get(p)
        if st is None:
            continue
        t = kind(st.st_mode)
        if e["t"] == "h":
            target = present.get(e["l"])
            if t != "f" or target is None or st.st_ino != target.st_ino:
                problems.append(f"{p}: hard link non ricostruito")
        elif t != e["t"]:
            problems.append(f"{p}: tipo {t} invece di {e['t']} ({origin[p]})")
            continue
        if e["t"] == "l":
            if os.readlink(root / p) != e["l"]:
                problems.append(f"{p}: destinazione del link diversa")
        elif e["t"] != "h":
            if stat.S_IMODE(st.st_mode) != e["m"]:
                problems.append(f"{p}: permessi {oct(stat.S_IMODE(st.st_mode))} invece di {oct(e['m'])}")
            if e["t"] != "d" and st.st_mtime_ns // 1_000_000_000 != e["mt"]:
                problems.append(f"{p}: data diversa da {origin[p]}")
        if e["t"] == "f" and (st.st_size != e["s"] or file_sha256(root / p) != e["sha256"]):
            problems.append(f"{p}: dimensione o sha256 diversi da {origin[p]}")
        checked[e["t"]] += 1
    git: dict = {}
    if args.git_commit:
        want = git_run(root, "rev-parse", "--verify", "--quiet", f"{args.git_commit}^{{commit}}")
        head = git_run(root, "rev-parse", "--verify", "--quiet", "HEAD")
        git["commit_atteso"] = want.stdout.decode().strip()
        git["head"] = head.stdout.decode().strip()
        if want.returncode or head.returncode or git["head"] != git["commit_atteso"]:
            problems.append(f"git: HEAD {git['head'] or '?'} invece di {args.git_commit}")
        s = git_run(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        if s.returncode:
            problems.append(f"git status: {s.stderr.decode(errors='replace')[:300]}")
        fields = [x for x in s.stdout.decode(errors="replace").split("\0") if x]
        changed, i = [], 0
        while i < len(fields):
            code, path = fields[i][:2], fields[i][3:]
            i += 2 if code[0] in "RC" else 1
            changed.append(f"{code} {path}")
            if path not in VOLATILE_FILES:
                problems.append(f"git: {code.strip()} {path} diverso dal commit e non volatile")
        git["stato"] = changed
        fsck = git_run(root, "fsck", "--no-progress", "--connectivity-only", "--no-dangling")
        git["fsck"] = "ok" if fsck.returncode == 0 else (fsck.stdout + fsck.stderr).decode(errors="replace")[:300]
        if fsck.returncode:
            problems.append(f"git fsck: {git['fsck']}")
    compared: dict[str, dict] = {}
    if args.confronta:
        if not args.progetto or not args.git_commit:
            sys.exit("--confronta richiede --progetto e --git-commit")
        proj = Path(args.progetto)
        for prefix in args.confronta:
            pre = prefix.rstrip("/") + "/"
            mine = sorted(p for p, e in final.items() if p.startswith(pre) and e["t"] == "f")
            theirs = sorted(os.path.relpath(os.path.join(d, n), proj)
                            for d, _, fs in os.walk(proj / prefix) for n in fs)
            if not mine or mine != theirs:
                problems.append(f"{prefix}: file diversi fra ripristino ({len(mine)}) e progetto ({len(theirs)})")
            for p in mine:
                blob = git_run(root, "cat-file", "blob", f"{args.git_commit}:{p}")
                h = {"ripristino": file_sha256(root / p),
                     "progetto": file_sha256(proj / p) if (proj / p).is_file() else None,
                     "commit": hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None}
                compared[p] = h
                if not h["ripristino"] == h["progetto"] == h["commit"]:
                    problems.append(f"{p}: ripristino, progetto e commit non coincidono")
    expected: dict[str, dict] = {}
    for item in args.atteso:
        p, want_sha = item.split("=", 1)
        real = os.path.realpath(root / p)
        res = {"risolto": os.path.relpath(real, root), "sha256_atteso": want_sha}
        if not real.startswith(str(root) + os.sep) or not os.path.isfile(real):
            problems.append(f"{p}: non risolve a un file dentro la radice")
        else:
            res |= {"byte": os.path.getsize(real), "sha256": file_sha256(Path(real))}
            if res["sha256"] != want_sha:
                problems.append(f"{p}: sha256 {res['sha256']} invece di {want_sha}")
        expected[p] = res
    report = {"esito": "conforme" if not problems else "non conforme", "n_problemi": len(problems),
              "problemi": problems[:200], "radice": str(root),
              "indici": [{"file": Path(f).name, "sha256": file_sha256(Path(f))} for f in args.indici],
              "rimossi": {"file": args.rimossi, "percorsi": len(removed)},
              "totali": {"voci_finali": len(final), "presenti": len(present), "cartelle_implicite": len(implicit),
                         "mancanti": len(missing), "in_piu": len(extra), "controllate": dict(checked)},
              "git": git, "confronti": compared, "attesi": expected,
              "job": os.environ.get("SLURM_JOB_ID"), "data": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    Path(args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("esito", "n_problemi", "totali")}, ensure_ascii=False), problems[:5])
    if problems:
        sys.exit(1)


def warning_blocking(w: dict) -> bool:
    """Un avviso blocca la completezza, salvo il cambiamento di un file volatile. I volumi dello schema 1 non hanno
    il campo `bloccante`: lo si ricava con la stessa regola."""
    if "bloccante" in w:
        return bool(w["bloccante"])
    if w["avviso"].startswith(("mancante", "tipo")):
        return True
    return w["p"] not in VOLATILE_FILES


def cmd_completezza(args) -> None:
    """Verifica finale di una o più generazioni, con esito e codice d'uscita:
    - ogni piano è integro (SHA256SUMS) e il suo inventario è quello registrato, senza errori di visita o lettura;
    - ogni volume selezionato ha il marcatore pronto di questa generazione, di questo piano e di questo volume, indici
      con gli hash registrati, parti e indici verificati su Drive, e il suo indice contiene tutti e soli i membri;
    - gli avvisi bloccanti (membri mancanti, file non volatili cambiati) rendono il volume non verificato;
    - la copertura si calcola dalle voci davvero archiviate nei volumi verificati: ogni voce attesa (inventario, categorie
      del piano, meno le esclusioni dichiarate) deve esserci, e nessuna voce in più."""
    need_slurm()
    state = Path(args.stato)
    problems, notes, rows, parts_rows = [], [], [], []
    archived: dict[str, str] = {}
    expected: set[str] = set()
    idx_ok = {r["remoto"]: r for r in read_tsv(state / "indici-verificati.tsv")} \
        if (state / "indici-verificati.tsv").is_file() else {}
    total = Counter()
    for pdir in map(Path, args.piani):
        plan = json.loads((pdir / "piano.json").read_text(encoding="utf-8"))
        plan_sha, gen = file_sha256(pdir / "piano.json"), plan["generazione"]
        for line in (pdir / "SHA256SUMS").read_text(encoding="ascii").splitlines():
            want, name = line.split("  ", 1)
            if file_sha256(pdir / name) != want:
                problems.append(f"piano {pdir.name}: {name} diverso da SHA256SUMS")
        inv = Path(plan["inventario"]["cartella"])
        if not (inv / "inventario.jsonl.gz").is_file() or \
                file_sha256(inv / "inventario.jsonl.gz") != plan["inventario"]["sha256"]:
            problems.append(f"piano {pdir.name}: inventario {inv} assente o diverso da quello registrato")
            continue
        summary = json.loads((inv / "riepilogo.json").read_text(encoding="utf-8"))
        problems += [f"inventario {inv.name}: {x}" for x in inventory_problems(summary)]
        excluded = plan_exclusions(pdir, plan)
        records = list(read_jsonl_gz(inv / "inventario.jsonl.gz"))
        assign_categories(records, frozenset(plan["hf_grandi"]), frozenset(plan["outputs_grandi"]))
        cats = {c for c, i in plan["categorie"].items() if i["nucleo"] or not args.solo_nucleo}
        exp_plan = {r["p"] for r in records if r["c"] in cats} - set(excluded)
        if expected & exp_plan:
            problems.append(f"piano {pdir.name}: {len(expected & exp_plan)} voci attese anche da un altro piano")
        expected |= exp_plan
        total["esclusioni_dichiarate"] += len(excluded)
        members: dict[str, list[str]] = defaultdict(list)
        for m in read_jsonl_gz(pdir / "membri.jsonl.gz"):
            members[m["v"]].append(m["p"])
        for v in plan["volumi"]:
            vid = v["id"]
            if v["categoria"] not in cats:  # fuori dalla selezione dichiarata (--solo-nucleo)
                continue
            vp: list[str] = []
            pronto, parti, ver = (state / "vol" / f"{vid}{s}" for s in (".pronto.json", ".parti.tsv", ".verificato.tsv"))
            info = json.loads(pronto.read_text(encoding="utf-8")) if pronto.is_file() else None
            if info is None:
                vp.append("non pronto")
            else:
                vp += [f"marcatore con {k} diverso" for k, want in (("generazione", gen), ("volume", vid),
                       ("categoria", v["categoria"]), ("piano_sha256", plan_sha)) if info.get(k) != want]
            if not vp and not (parti.is_file() and ver.is_file()):
                vp.append("non verificato su Drive" if parti.is_file() else "manca parti.tsv")
            idx_paths: list[str] = []
            if not vp:
                for x in info["indice"]:
                    if file_sha256(state / "indici" / x["nome"]) != x["sha256"]:
                        vp.append(f"indice {x['nome']} diverso da quello registrato nel marcatore")
                plist = read_tsv(parti)
                if [(p["file"], int(p["byte"]), p["sha256"], p["md5"]) for p in plist if p["tipo"] == "volume"] != \
                        [(p["nome"], p["byte"], p["sha256"], p["md5"]) for p in info["parti"]]:
                    vp.append("parti.tsv diverso dalle parti del marcatore")
                vrows = {r["remoto"]: r for r in read_tsv(ver)}
                for p in plist:
                    rem = f"LimiteVL-backup/{gen}/{p['remoto']}"
                    got = vrows.get(rem) if p["tipo"] == "volume" else idx_ok.get(rem)
                    if got is None:
                        vp.append(f"{p['file']} non verificato su Drive")
                    elif (got["byte"], got["md5_drive"]) != (p["byte"], p["md5"]) or got["sha256_drive"] not in (
                            p["sha256"], "(non fornito da Drive)"):
                        vp.append(f"{p['file']} diverso su Drive")
                    if p["tipo"] == "volume":
                        parts_rows.append([vid, p["file"], p["byte"], p["sha256"], p["md5"], rem,
                                           got["id_drive"] if got else "-"])
                idx_paths = [e["p"] for e in read_jsonl_gz(state / "indici" / f"{vid}.jsonl.gz")]
                if idx_paths != members[vid]:
                    vp.append(f"indice con {len(idx_paths)} voci, membri del piano {len(members[vid])}: non coincidono")
                for w in info["avvisi"]:
                    if warning_blocking(w):
                        vp.append(f"avviso bloccante: {w['p']}: {w['avviso']}")
                    else:
                        notes.append(f"{vid}: {w['p']}: {w['avviso']} (file volatile, avviso motivato)")
            if not vp:
                for p in idx_paths:
                    if p in archived:
                        vp.append(f"{p} già archiviato in {archived[p]}")
                    archived.setdefault(p, vid)
            problems += [f"{vid}: {x}" for x in vp]
            total["volumi"] += 1
            total["volumi_verificati"] += not vp
            if info:
                total["parti"] += len(info["parti"])
                total["byte_caricati"] += info["flusso"]["byte"]
                total["byte_dati"] += info["byte_dati"]
            rows.append([v["ordine"], vid, v["categoria"], int(v["nucleo"]), info["compressione"] if info else "-",
                         info["voci"] if info else 0, info["byte_dati"] if info else 0,
                         len(info["parti"]) if info else 0, info["flusso"]["byte"] if info else 0,
                         info["flusso"]["sha256"] if info else "-",
                         sum(1 for w in (info or {}).get("avvisi", []) if not warning_blocking(w)),
                         sum(1 for w in (info or {}).get("avvisi", []) if warning_blocking(w)),
                         "verificato" if not vp else f"non verificato: {vp[0]}"])
    missing = sorted(expected - archived.keys())
    extra = sorted(archived.keys() - expected)
    if missing:
        problems.append(f"{len(missing)} voci attese non archiviate in un volume verificato (prime: {missing[:5]})")
    if extra:
        problems.append(f"{len(extra)} voci archiviate ma non attese (prime: {extra[:5]})")
    total["voci_attese"], total["voci_archiviate"] = len(expected), len(archived)
    total["voci_mancanti"], total["voci_in_piu"] = len(missing), len(extra)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / "volumi.tsv", ["ordine", "volume", "categoria", "nucleo", "compressione", "voci", "byte_dati",
                                   "parti", "byte_caricati", "sha256_flusso", "avvisi_motivati", "avvisi_bloccanti",
                                   "stato"],
              sorted(rows, key=lambda r: (r[0] if isinstance(r[0], int) else 0, r[1])))
    write_tsv(out / "parti.tsv", ["volume", "file", "byte", "sha256", "md5", "remoto", "id_drive"], parts_rows)
    rep = {"esito": "completo" if not problems else "incompleto", "problemi": problems[:500],
           "n_problemi": len(problems), "avvisi_motivati": notes[:200], "n_avvisi_motivati": len(notes),
           "totali": dict(total), "piani": [str(p) for p in args.piani], "solo_nucleo": args.solo_nucleo,
           "voci_mancanti_elenco": missing[:1000], "job": os.environ.get("SLURM_JOB_ID"),
           "data": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_new(out / "completezza.json", json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k in ("esito", "n_problemi", "n_avvisi_motivati", "totali")},
                     ensure_ascii=False), problems[:5])
    if problems:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventario")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--segreti", action="store_true", help="cerca anche segreti nei file di testo e negli oggetti git")
    p.set_defaults(func=cmd_inventario)
    p = sub.add_parser("piano")
    p.add_argument("--inventario", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gen", required=True, help="generazione del backup, es. g1-20261112")
    p.add_argument("--solo-categorie", default="")
    p.add_argument("--escludi-categorie", default="")
    p.add_argument("--precedente", default="", help="non supportato: rifiutato con una diagnosi (usare --dipende-da)")
    p.add_argument("--sottoinsieme", action="append", default=[], metavar="CATEGORIA=FILE",
                   help="solo i percorsi elencati della categoria; il resto è un'esclusione dichiarata")
    p.add_argument("--prefisso-volumi", default="", help="prefisso dei nomi dei volumi (es. s1-), contro le omonimie")
    p.add_argument("--dipende-da", default="", help="generazione da ripristinare prima di questa (supplemento)")
    p.add_argument("--byte-volume", type=int, default=VOLUME_BYTES)
    p.add_argument("--dimensione-parte", type=int, default=PART_BYTES)
    p.add_argument("--soglia-grandi", type=int, default=GIB, help="repository HF e cartelle di outputs con categoria propria")
    p.set_defaults(func=cmd_piano)
    p = sub.add_parser("crea")
    p.add_argument("--piano", required=True)
    p.add_argument("--staging", required=True)
    p.add_argument("--stato", required=True)
    p.add_argument("--volumi", default="", help="solo questi volumi (separati da virgole)")
    p.add_argument("--massimo", type=int, default=None, help="al più N volumi in questa esecuzione")
    p.add_argument("--limite-staging-gib", type=float, default=30.0)
    p.add_argument("--attendi", action="store_true", help="con lo staging pieno aspetta invece di uscire")
    p.add_argument("--solo-nucleo", action="store_true", help="backup selezionato: solo i volumi del nucleo")
    p.set_defaults(func=cmd_crea)
    p = sub.add_parser("ripristino")
    p.add_argument("--pronto", required=True)
    p.add_argument("--cartella", required=True, help="cartella con le parti (scaricate o in staging)")
    p.add_argument("--indice", required=True)
    p.add_argument("--sha256sum", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--rimuovi", action="store_true")
    p.add_argument("--esistente", action="store_true", help="estrae in una radice esistente (ripristino vero)")
    p.set_defaults(func=cmd_ripristino)
    p = sub.add_parser("catalogo")
    p.add_argument("--root", required=True)
    p.add_argument("--inventario", required=True)
    p.add_argument("--stato", required=True, help="trasferimento/stato/<generazione> (indici dei volumi)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_catalogo)
    p = sub.add_parser("completezza")
    p.add_argument("--stato", required=True, help="trasferimento/stato/<generazione>")
    p.add_argument("--piani", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--solo-nucleo", action="store_true", help="backup selezionato: solo i volumi e le categorie del nucleo")
    p.set_defaults(func=cmd_completezza)
    p = sub.add_parser("firme")
    p.add_argument("--piano-base", required=True, help="piano.json della generazione di base")
    p.add_argument("--inventario", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_firme)
    p = sub.add_parser("rimossi")
    p.add_argument("--base", nargs="+", required=True, help="indici (jsonl.gz) dei volumi della base")
    p.add_argument("--nuovo", nargs="+", required=True, help="indici (jsonl.gz) dei volumi nuovi")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_rimossi)
    p = sub.add_parser("composizione")
    p.add_argument("--radice", required=True, help="cartella ripristinata (base e supplementi)")
    p.add_argument("--indici", nargs="+", required=True, help="indici jsonl.gz in ordine: base, poi supplementi")
    p.add_argument("--rimossi", help="elenco dei percorsi tolti dopo il supplemento (<supplemento>-rimossi.txt)")
    p.add_argument("--git-commit", help="commit atteso nella radice ripristinata")
    p.add_argument("--progetto", help="cartella del progetto per --confronta")
    p.add_argument("--confronta", action="append", default=[], help="cartella relativa da confrontare (ripetibile)")
    p.add_argument("--atteso", action="append", default=[], help="PERCORSO=SHA256 di un file atteso (ripetibile)")
    p.add_argument("--report", required=True)
    p.set_defaults(func=cmd_composizione)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
