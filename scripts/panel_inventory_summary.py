#!/usr/bin/env python3
"""Riassunto leggibile dell'inventario dei benchmark (scripts/panel_inventory.py): una riga per file e per colonna, con
righe, nulli, vuoti, valori distinti, immagini e i conteggi delle colonne con pochi valori. Solo in un job SLURM CPU.

    srun $RUN_PYTHON -u -B scripts/panel_inventory_summary.py --inventory FILE --out FILE.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def short(counts: dict, n: int = 14) -> str:
    items = list(counts.items())
    text = ", ".join(f"{k[:40]}: {v}" for k, v in items[:n])
    return text + (f", … ({len(items)} valori)" if len(items) > n else "")


def column_line(name: str, c: dict, indent: str) -> list[str]:
    if "errore" in c:
        return [f"{indent}- `{name}`: errore {c['errore']}"]
    parts = [c.get("tipo", "")[:40]]
    for k in ("nulli", "vuoti", "distinti", "immagini", "lunghezza_min", "lunghezza_max", "min", "max"):
        if k in c:
            parts.append(f"{k} {c[k]}")
    if "conteggi" in c:
        parts.append("valori {" + short(c["conteggi"]) + "}")
    lines = [f"{indent}- `{name}`: " + "; ".join(str(p) for p in parts if p != "")]
    for sub, cc in (c.get("campi") or {}).items():
        lines += column_line(f"{name}.{sub}", cc, indent + "  ")
    for sub, cc in (c.get("sottocampi") or {}).items():
        lines += column_line(f"{name}.{sub}", cc, indent + "  ")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("panel_inventory_summary.py va eseguito in un job SLURM")
    if args.out.exists():
        sys.exit(f"{args.out} esiste già")
    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    out = [f"# Inventario dei benchmark locali (job {inv['job_id']}, codice {inv['code']['commit'][:8]})", ""]
    for s in inv["sorgenti"]:
        out += [f"## {s['repo']}@{s['revisione'][:8]} ({s['gruppo']}, {s['ammissione']})", "",
                f"snapshot `{s['snapshot']}`: {s['file_presenti']} file su {s['file_del_manifest']} del manifest; "
                f"mancanti {s['mancanti_rispetto_al_manifest']}; verifica degli hash {s['verifica_hash']}", ""]
        for rel, f in s["file"].items():
            head = f"- **{rel}** ({f.get('byte', 0)} byte)"
            for k in ("righe", "record", "chiavi", "membri"):
                if k in f:
                    head += f", {k} {f[k]}"
            out.append(head)
            for name, c in (f.get("colonne") or {}).items():
                out += column_line(name, c, "  ")
            for name, c in (f.get("campi") or {}).items():
                out += column_line(name, c, "  ")
            if "per_estensione" in f:
                out.append("  - per estensione: " + short(f["per_estensione"]))
                out.append("  - per cartella: " + short(f["per_cartella"], 30))
            if "errore" in f:
                out.append(f"  - errore: {f['errore']}")
        out.append("")
    out += ["## Copie del valutatore testuale (data/eval-datasets)", ""]
    for rel, f in inv["copie_del_valutatore"].items():
        out.append(f"- **{rel}**: " + ", ".join(f"{k} {f[k]}" for k in ("righe", "record") if k in f))
    out += ["", "## Indici di decontaminazione", ""]
    for rel, idx in inv["indici"].items():
        out.append(f"- **{rel}**: {idx['elementi']} elementi, campi {idx['campi']}, report {json.dumps(idx['report'])}")
        for row in idx["per_file"]:
            out.append(f"  - {row['benchmark']} / {row['file']}: {row['elementi']} (pHash {row['phash']}, testo {row['testo']})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(out)} righe -> {args.out}")


if __name__ == "__main__":
    main()
