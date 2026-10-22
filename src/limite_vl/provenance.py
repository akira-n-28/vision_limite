"""Provenienza degli artefatti: hash di testi e file, identità del codice, scrittura di file nuovi senza sovrascrivere.

Solo libreria standard: si importa anche dagli script di sola lettura dei dati. Gli hash di file grandi vanno
calcolati nei job SLURM (regola del frontend in CLAUDE.md), non qui sul frontend.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_text(text: str) -> str:
    """sha256 esadecimale del testo in UTF-8 (stesso valore per stesso testo, qualunque sia il file che lo contiene)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_entry(path: Path | str, digest: str | None = None) -> dict:
    """Percorso, dimensione e sha256 di un file (calcolato se non è già noto, es. da HashingTextWriter)."""
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest or sha256_file(path)}


def code_identity(root: Path = ROOT) -> dict:
    """Commit, stato del working tree e sha256 di `git diff HEAD` (modifiche dei file tracciati); i file non
    tracciati si elencano per nome. Con un albero sporco il commit da solo non basta (LAGRANGE.md, sezione 7)."""
    def git(*cmd: str) -> str:
        return subprocess.run(["git", "-C", str(root), *cmd], capture_output=True, text=True, check=True).stdout

    diff = git("diff", "HEAD")
    status = [line for line in git("status", "--porcelain").splitlines() if line.strip()]
    return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(status),
            "diff_sha256": sha256_text(diff) if diff else None,
            "untracked": sorted(line[3:] for line in status if line.startswith("??"))}


class HashingTextWriter:
    """File di testo nuovo scritto riga per riga, con sha256 e conteggio delle righe calcolati durante la scrittura
    (nessuna rilettura). Scrive in `<nome>.tmp` e rinomina solo con close() senza errori; rifiuta un file esistente."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        refuse_existing(self.path)
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self._fh = open(self._tmp, "w", encoding="utf-8")
        self._sha = hashlib.sha256()
        self.lines = 0

    def write_json_line(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self._fh.write(line)
        self._sha.update(line.encode("utf-8"))
        self.lines += 1

    def close(self) -> dict:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        refuse_existing(self.path)
        os.replace(self._tmp, self.path)
        return {**file_entry(self.path, self._sha.hexdigest()), "lines": self.lines}

    def abort(self) -> None:
        self._fh.close()
        self._tmp.unlink(missing_ok=True)

    def __enter__(self) -> "HashingTextWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()


def refuse_existing(*paths: Path | str) -> None:
    """Errore se uno dei file esiste già: gli artefatti non si sovrascrivono (CLAUDE.md, nessuna perdita di dati)."""
    found = [str(p) for p in paths if Path(p).exists()]
    if found:
        raise FileExistsError(f"file già presenti, non li sovrascrivo: {found}")


def write_new_json(path: Path | str, obj: dict) -> None:
    """JSON in un file che non deve esistere: scritto in `.tmp`, poi rinominato."""
    path = Path(path)
    refuse_existing(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    refuse_existing(path)
    os.replace(tmp, path)
