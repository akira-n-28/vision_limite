"""MMFineReason (OpenDataArena/MMFineReason-1.8M): chiavi dei campioni e lettura dei problemi dai parquet.

La chiave `<split>-<sorgente>-<id>` è la stessa negli shard (scripts/shard_parquet.py) e nella generazione
di F2 (scripts/f2_generate.py), così le soluzioni si possono unire alle immagini più tardi. Gli id si ripetono
fra sorgenti (job 29460), per questo la sorgente fa parte della chiave.

La lettura per la generazione usa solo le colonne di testo (niente immagini) e procede in due passate:
prima chiave, sorgente, pass_rate e presenza dei campi (leggero anche su un milione di righe), poi il testo
completo solo dei problemi scelti.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def mmfine_key(split: str, source: str | None, id_: object) -> str:
    source_tag = re.sub(r"[^A-Za-z0-9]+", "_", source or "nosource").strip("_")
    return f"{split}-{source_tag}-{id_}"


def split_of(path: Path) -> str:
    """`sft` o `rl` dal nome originale del parquet (percorsi dello snapshot, non dei blob)."""
    name = Path(path).name
    if "-" not in name or len(name.split("-")[0]) > 8:
        raise ValueError(f"{name}: serve il percorso dello snapshot (es. data/sft-00000-of-00193.parquet)")
    return name.split("-")[0]


def scan_problems(paths: list[Path]) -> list[dict]:
    """Prima passata: chiave, sorgente, pass_rate di Qwen3-VL-4B, file e riga, per i problemi con domanda,
    risposta e descrizione non vuote."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    out = []
    for path in paths:
        split = split_of(path)
        table = pq.read_table(path, columns=["id", "source", "pass_rate", "question", "answer", "caption"])
        ok = None
        for col in ("question", "answer", "caption"):
            filled = pc.fill_null(pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(table[col])), 0), False)
            ok = filled if ok is None else pc.and_(ok, filled)
        ids, sources, rates = table["id"].to_pylist(), table["source"].to_pylist(), table["pass_rate"].to_pylist()
        for row, good in enumerate(ok.to_pylist()):
            if good:
                out.append({"key": mmfine_key(split, sources[row], ids[row]), "source_dataset": sources[row],
                            "qwen4b_pass_rate": rates[row], "file": str(path), "row": row})
        del table
    return out


def rank(key: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}/{key}".encode(), digest_size=8).digest(), "little")


def choose(problems: list[dict], n: int, easy_frac: float, seed: int, exclude: set[str] = frozenset(),
           exclude_sources: set[str] = frozenset()) -> list[dict]:
    """n problemi: easy_frac fra quelli che Qwen3-VL-4B risolve sempre (pass_rate 1.0), il resto fra gli altri,
    in ordine pseudo-casuale fissato dal seed; se una fascia non basta, si completa con l'altra. Salta le chiavi
    di `exclude` e i problemi delle sorgenti (colonna `source`) di `exclude_sources`."""
    pool = [p for p in problems if p["key"] not in exclude and p["source_dataset"] not in exclude_sources]
    easy = sorted((p for p in pool if p["qwen4b_pass_rate"] == 1.0), key=lambda p: rank(p["key"], seed))
    hard = sorted((p for p in pool if p["qwen4b_pass_rate"] != 1.0), key=lambda p: rank(p["key"], seed))
    n_easy = min(len(easy), round(n * easy_frac))
    n_hard = min(len(hard), n - n_easy)
    n_easy = min(len(easy), n - n_hard)
    chosen = easy[:n_easy] + hard[:n_hard]
    return sorted(chosen, key=lambda p: (p["file"], p["row"]))


def load_texts(chosen: list[dict]) -> list[dict]:
    """Seconda passata: domanda (senza il segnaposto <image>), risposta e descrizione dei problemi scelti."""
    import pyarrow.parquet as pq

    by_file: dict[str, list[dict]] = {}
    for p in chosen:
        by_file.setdefault(p["file"], []).append(p)
    out = []
    for path, items in by_file.items():
        table = pq.read_table(path, columns=["question", "answer", "caption"])
        rows = table.take([p["row"] for p in items]).to_pylist()
        for p, r in zip(items, rows):
            question = r["question"].replace("<image>", "").strip()
            if not question:  # la domanda era solo il segnaposto dell'immagine
                continue
            out.append({**{k: p[k] for k in ("key", "source_dataset", "qwen4b_pass_rate")},
                        "question": question, "answer": r["answer"].strip(), "caption": r["caption"].strip()})
        del table
    return out
