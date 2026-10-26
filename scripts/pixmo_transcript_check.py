#!/usr/bin/env python3
"""PixMo-Cap: descrizioni contro trascrizioni degli annotatori su tutte le righe degli shard (limite_vl.transcripts).

Legge dai parquet originali solo le colonne `caption` e `transcripts` (le trascrizioni non sono negli shard). La chiave
`pixmo-<file>-<riga>` è quella di scripts/shard_parquet.py, quindi gli esiti si applicano agli shard. Numeri, colori e
testi citati che nessuna trascrizione contiene sono candidati aggiunti dal riscrittore: da confermare, non errori.

Quarta versione: sono segnali di rischio, non esclusioni (l'assenza di un dettaglio nelle trascrizioni non prova che
sia falso nell'immagine). Ogni segnale ha un tipo: numero (orario, anno, conteggio 2-12, altro), colore, testo citato.
Uscite in --out (cartella nuova): report.json (tassi per tipo, esempi, identità di codice e file), flagged.jsonl (righe
con almeno un segnale, con i tipi) e risk_<tipo>.txt (chiavi per tipo, per decidere dopo una verifica sulle immagini).
Nessuna lista di esclusione. Job CPU: slurm/pixmo_transcript_check.sbatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.provenance import code_identity  # noqa: E402
from limite_vl.transcripts import compare, number_kind  # noqa: E402

EXAMPLES = 30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards-jsonl", type=Path, required=True, help="shards.jsonl degli shard PixMo (file parquet per shard)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("pixmo_transcript_check.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} esiste e non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)

    shards = [json.loads(line) for line in args.shards_jsonl.read_text().splitlines() if line.strip()]
    rows = flagged = no_transcript = 0
    with_flag, checked, missing_items = Counter(), Counter(), Counter()
    flags_per_row = Counter()
    examples = defaultdict(list)
    values = defaultdict(Counter)
    files = []
    risk_keys: dict[str, list[str]] = defaultdict(list)
    risk_rows = Counter()
    with open(args.out / "flagged.jsonl", "w") as fh:
        for shard in shards:
            file_index = int(shard["name"].rsplit("-", 1)[1].split(".")[0])
            (path,) = shard["files"]
            files.append({"path": path, "bytes": Path(path).stat().st_size})
            pf = pq.ParquetFile(path)
            row = 0
            for batch in pf.iter_batches(batch_size=4096, columns=["caption", "transcripts"]):
                for rec in batch.to_pylist():
                    key = f"pixmo-{file_index:05d}-{row:06d}"
                    row += 1
                    rows += 1
                    caption, transcripts = rec.get("caption") or "", rec.get("transcripts") or []
                    if not any(t and t.strip() for t in transcripts):
                        no_transcript += 1
                        continue
                    r = compare(caption, transcripts)
                    checked.update(r["checked"])
                    n = 0
                    for kind in ("numbers", "colors", "quotes"):
                        if r[kind]:
                            with_flag[kind] += 1
                            missing_items[kind] += len(r[kind])
                            values[kind].update(str(v) for v in r[kind])
                            n += len(r[kind])
                            if len(examples[kind]) < EXAMPLES:
                                examples[kind].append({"id": key, "missing": r[kind], "caption": caption[:1200],
                                                       "transcripts": [t[:1200] for t in transcripts]})
                    flags_per_row[min(n, 5)] += 1
                    if n:
                        flagged += 1
                        kinds = sorted({f"numero_{number_kind(v)}" for v in r["numbers"]}
                                       | ({"colore"} if r["colors"] else set()) | ({"testo_citato"} if r["quotes"] else set()))
                        for k in kinds:
                            risk_keys[k].append(key)
                            risk_rows[k] += 1
                        fh.write(json.dumps({"id": key, "risks": kinds, "numbers": r["numbers"], "colors": r["colors"],
                                             "quotes": r["quotes"]}, ensure_ascii=False) + "\n")
            print(f"{Path(path).name}: {rows} righe, {flagged} con candidati", flush=True)

    for k, keys in risk_keys.items():
        (args.out / f"risk_{k}.txt").write_text("".join(x + "\n" for x in keys))
    compared = rows - no_transcript
    report = {
        "schema": "pixmo-transcript-check/4",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code": code_identity(),
        "files": files,
        "rows": rows,
        "rows_without_transcript": no_transcript,
        "rows_compared": compared,
        "rows_with_candidates": flagged,
        "risk_rows_by_type": dict(risk_rows.most_common()),
        "risk_rate_by_type": {k: round(v / max(1, compared), 4) for k, v in risk_rows.items()},
        "rows_with_candidates_by_kind": dict(with_flag),
        "items_checked": dict(checked),
        "items_missing": dict(missing_items),
        "rate_rows": {k: round(v / max(1, compared), 4) for k, v in with_flag.items()},
        "rate_items": {k: round(missing_items[k] / max(1, checked[k]), 4) for k in checked},
        "candidates_per_row": {str(k): v for k, v in sorted(flags_per_row.items())},
        "top_missing_values": {k: v.most_common(30) for k, v in values.items()},
        "examples": examples,
        "note": "segnali di rischio per tipo, non esclusioni: da verificare sulle immagini prima di qualunque decisione",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("rows", "rows_without_transcript", "rows_with_candidates",
                                             "rows_with_candidates_by_kind", "risk_rows_by_type", "rate_items")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
