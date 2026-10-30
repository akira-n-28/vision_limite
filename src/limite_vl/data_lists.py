"""Liste di dati di una config di training (F1-clean): quali liste di decontaminazione sono obbligatorie per ogni
famiglia di shard, e la provenienza di tutte le liste (sha256 e righe) per il manifest del run.

Le liste di decontaminazione approvate (data/decontam/*/<sorgente>/exclude_final.txt) devono comparire nel campo
`exclude` di ogni sorgente della loro famiglia: train_f1.py e check_train_config.py rifiutano una config che ne
dimentica una. Le altre liste (qualità, incerti, duplicati della validazione, contaminazione di UniMER-Test,
rietichettature) sono scelte della config: non obbligatorie, ma registrate nel manifest con hash e righe, così un run
dice esattamente quali chiavi ha saltato o cambiato. Solo libreria standard.
"""
from __future__ import annotations

from pathlib import Path

from .provenance import file_entry

ROOT = Path(__file__).resolve().parents[2]
# famiglia (cartella sotto data/shards) -> liste di decontaminazione obbligatorie
REQUIRED = {
    "mavis-geometry": ("data/decontam/p6/mavis/exclude_final.txt", "data/decontam/f1eval/mavis/exclude_final.txt"),
    "unimer": ("data/decontam/p6/unimer/exclude_final.txt", "data/decontam/f1eval/unimer/exclude_final.txt"),
    "pixmo-cap": ("data/decontam/p6/pixmo/exclude_final.txt", "data/decontam/f1eval/pixmo/exclude_final.txt"),
    "synth": ("data/decontam/f1eval/synth/exclude_final.txt",),
    "mmfinereason": ("data/decontam/p6/mmfinereason/exclude_final.txt",),
    "llava-pretrain": ("data/decontam/p6/llava/exclude_final.txt",),
}


def family(shards: str) -> str | None:
    """Famiglia di una sorgente dal glob degli shard: la cartella dopo `shards/`."""
    parts = Path(shards).parts
    return parts[parts.index("shards") + 1] if "shards" in parts and parts.index("shards") + 1 < len(parts) else None


def missing_required(specs: list[dict], root: Path = ROOT) -> list[str]:
    """Errori: liste di decontaminazione obbligatorie assenti dal campo `exclude` di una sorgente."""
    errors = []
    for spec in specs:
        fam = family(spec["shards"])
        have = {Path(p).resolve() for p in spec.get("exclude", [])}
        for req in REQUIRED.get(fam, ()):
            if (root / req).resolve() not in have:
                errors.append(f"sorgente {spec['name']} ({fam}): manca la lista di decontaminazione {req}")
    return errors


def _entry(path: str) -> dict:
    with open(path) as fh:
        lines = sum(1 for line in fh if line.strip())
    return {**file_entry(path), "lines": lines}


def provenance(specs: list[dict]) -> dict:
    """{sorgente: {"exclude": [...], "relabel": [...], "manifest": {...}}} con percorso, byte, sha256 e righe di ogni
    lista; per una sorgente con manifest delle ammissioni (limite_vl.manifest), manifest.json con lo split letto."""
    out = {}
    for spec in specs:
        entry = {"family": family(spec["shards"]),
                 "exclude": [_entry(p) for p in spec.get("exclude", [])],
                 "relabel": [_entry(p) for p in spec.get("relabel", [])]}
        if spec.get("manifest"):
            entry["manifest"] = {**file_entry(Path(spec["manifest"]) / "manifest.json"), "split": spec.get("split")}
        out[spec["name"]] = entry
    return out
