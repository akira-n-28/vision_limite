#!/usr/bin/env python3
"""Controllo dei dati di F1c prima del training. Solo in un job SLURM CPU.

- mavis: su tutti gli shard, quante didascalie danno tante figure principali quante dice la profondità (le sole
  su cui si fanno conteggio e figura inventata), domande per campione e per tipo, esempi;
- stem: su --stem-rows descrizioni, quanta parte del testo resta con f1c.visible_caption, quante restano vuote,
  esempi prima e dopo;
- unimer-lines: righe accoppiate dello shard --lines, con alcune righe ritagliate salvate come PNG da guardare;
- synth: alcune scene del test con le loro domande, salvate come PNG.
Scrive OUT/report.json, OUT/examples.txt e le immagini in OUT/.

    srun $RUN_PYTHON -u -B scripts/f1c_data_check.py --out DIR --mavis GLOB --stem GLOB [--lines GLOB] [--synth TAR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import data  # noqa: E402
from limite_vl.f1c import mavis_count_caption, mavis_questions, visible_caption  # noqa: E402
from limite_vl.facts import mavis_facts  # noqa: E402


def check_mavis(paths: list[str], ex) -> dict:
    by_depth, qtypes, nq, errors = Counter(), Counter(), Counter(), Counter()
    consistent = Counter()
    shown = 0
    for path in paths:
        for key, files in data.iter_shard(path):
            meta = json.loads(files["json"])
            cap, depth = meta["text_en"]["caption"].strip(), meta["depth"]
            by_depth[depth] += 1
            f = mavis_facts(cap)
            main = [k for k in f["shape_order"] if not k.startswith("ext:")]
            consistent[depth] += len(main) == depth + 1
            try:
                qs = mavis_questions(cap, depth, key)
            except Exception as exc:  # noqa: BLE001
                errors[type(exc).__name__] += 1
                continue
            nq[min(len(qs), 12)] += 1
            qtypes.update(q["type"] for q in qs)
            if shown < 6 and depth == shown % 4:
                shown += 1
                sample = data.mavis_qa_sample(key, files)
                ex.write(f"=== MAVIS {key} (profondità {depth})\nDIDASCALIA: {cap}\nCON CONTEGGIO: "
                         f"{mavis_count_caption(cap, depth)}\nRICHIESTA: {sample.messages[0]['content'][1]['text']}\n"
                         f"RISPOSTA: {sample.messages[1]['content']}\nTUTTE: "
                         + json.dumps([(q['q'], q['answer']) for q in qs], ensure_ascii=False) + "\n\n")
    total = sum(by_depth.values())
    return {"samples": total, "by_depth": dict(by_depth),
            "consistent_by_depth": {d: round(consistent[d] / by_depth[d], 4) for d in sorted(by_depth)},
            "consistent": round(sum(consistent.values()) / max(1, total), 4),
            "questions_per_sample": dict(sorted(nq.items())), "question_types": dict(qtypes.most_common()),
            "errors": dict(errors)}


def check_stem(paths: list[str], rows: int, ex) -> dict:
    ratios, empty, n, shown = [], 0, 0, 0
    for path in paths:
        for key, files in data.iter_shard(path):
            if n >= rows:
                break
            meta = json.loads(files["json"])
            vis = visible_caption(meta["caption"])
            n += 1
            ratios.append(len(vis) / max(1, len(meta["caption"])))
            empty += len(vis) < 40
            if shown < 4:
                shown += 1
                ex.write(f"=== STEM {key}\nORIGINALE:\n{meta['caption']}\n\nVISIBILE:\n{vis}\n\n")
        if n >= rows:
            break
    q = statistics.quantiles(ratios, n=10) if len(ratios) > 10 else ratios
    return {"captions": n, "kept_chars_mean": round(sum(ratios) / max(1, n), 3),
            "kept_chars_deciles": [round(x, 3) for x in q], "empty": empty}


def check_lines(paths: list[str], out: Path, ex) -> dict:
    n, rows_hist, saved = 0, Counter(), 0
    for path in paths:
        for key, files in data.iter_shard(path):
            meta = json.loads(files["json"])
            n += 1
            rows_hist[min(len(meta["rows"]), 8)] += 1
            if saved < 8 and n % 997 == 1:
                sample = data.unimer_line_sample(key, files)
                sample.image.save(out / f"line-{key}.png")
                full = data.open_image(files["png"])
                full.save(out / f"line-{key}-full.png")
                ex.write(f"=== RIGA {key}: {sample.messages[1]['content']}\nRIGHE: {json.dumps(meta['rows'])}\n"
                         f"FASCE: {meta['lines']} (altezza {meta['height']})\n\n")
                saved += 1
        if n > 50000:
            break
    return {"read": n, "rows": dict(sorted(rows_hist.items()))}


def check_synth(path: str, out: Path, ex) -> dict:
    fams = Counter()
    for i, (key, files) in enumerate(data.iter_shard(path)):
        meta = json.loads(files["json"])
        fams[meta["family"]] += 1
        if i < 12:
            (out / f"synth-{key}.png").write_bytes(files["png"])
            ex.write(f"=== SINTETICO {key} ({meta['family']}, {meta['width']}x{meta['height']})\n"
                     + "\n".join(f"  {q['q']} -> {q['answer']}   {q['options']}" for q in meta["qa"])
                     + ("\n  COPPIA: " + json.dumps(meta["pair"]) if "pair" in meta else "") + "\n\n")
    return {"families": dict(fams)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mavis")
    ap.add_argument("--stem")
    ap.add_argument("--stem-rows", type=int, default=3000)
    ap.add_argument("--lines")
    ap.add_argument("--synth", nargs="*", default=[])
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f1c_data_check.py va eseguito in un job SLURM")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    with open(args.out / "examples.txt", "w") as ex:
        if args.mavis:
            report["mavis"] = check_mavis(sorted(glob.glob(args.mavis)), ex)
            print("mavis:", json.dumps(report["mavis"]), flush=True)
        if args.stem:
            report["stem"] = check_stem(sorted(glob.glob(args.stem)), args.stem_rows, ex)
            print("stem:", json.dumps(report["stem"]), flush=True)
        if args.lines:
            report["lines"] = check_lines(sorted(glob.glob(args.lines)), args.out, ex)
            print("righe:", json.dumps(report["lines"]), flush=True)
        for path in args.synth:
            report.setdefault("synth", {})[Path(path).name] = check_synth(path, args.out, ex)
            print("sintetici:", json.dumps(report["synth"]), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
