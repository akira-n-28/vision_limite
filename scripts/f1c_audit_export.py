#!/usr/bin/env python3
"""Materiale per il controllo delle risposte di riferimento e dei dati (audit di F1c). Solo in un job SLURM CPU.

Per ogni controllo scrive in OUT/<controllo>/ le immagini (PNG, come le vede il modello) e items.json senza le
risposte, e in OUT/_gold/<controllo>.json le risposte di riferimento (gli agenti lavorano alla cieca):
- stem_facts-1..4: i 391 fatti verificati a mano sulle 100 immagini STEM reali, in 4 gruppi da 25 immagini;
- synth: 20 scene di test per famiglia (fino a 5 domande ciascuna) e 40 coppie controfattuali;
- mavis: tutte le domande verificabili dei primi 40 diagrammi di validazione;
- stem_captions: 50 descrizioni di Qwen3-VL di MMFineReason (originale e versione visibile di F1c, target di F1c);
- mmf_answers e mmf_solutions: 50 problemi del primo lotto F2 (f2-sft120k-v4): domanda alla cieca, e soluzione di
  Violetto scelta (target di F2) da verificare;
- replay: 30 problemi di testo della fetta di replay con soluzione tenuta (domanda alla cieca);
- unimer_lines: 30 righe ritagliate di formule su più righe con la loro trascrizione.

    srun $RUN_PYTHON -u -B scripts/f1c_audit_export.py --out DIR --config config/runs/f1b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from train_f1 import load_config  # noqa: E402

from limite_vl import data  # noqa: E402
from limite_vl.f1c import key_rng, mavis_questions, visible_caption  # noqa: E402
from limite_vl.stream import make_sources, validation_samples  # noqa: E402

DATA = ROOT / "data"


def save(image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return str(path)


def write(out: Path, task: str, items: list[dict], gold: dict | None) -> None:
    (out / task).mkdir(parents=True, exist_ok=True)
    (out / task / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    if gold is not None:
        (out / "_gold").mkdir(parents=True, exist_ok=True)
        (out / "_gold" / f"{task}.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False))
    print(f"{task}: {len(items)} voci", flush=True)


def stem_facts(out: Path) -> None:
    facts = json.loads((ROOT / "outputs/f1c/stem_real/facts.json").read_text())
    keys = list(dict.fromkeys(f["key"] for f in facts))
    for g in range(4):
        group = set(keys[g * 25:(g + 1) * 25])
        items, gold = [], {}
        for i, f in enumerate(facts):
            if f["key"] in group:
                fid = f"sf-{i:03d}"
                items.append({"id": fid, "image": f["image"], "q": f["q"], "options": f["options"]})
                gold[fid] = f["answer"]
        write(out, f"stem_facts-{g + 1}", items, gold)


def synth(out: Path) -> None:
    items, gold, per_family = [], {}, defaultdict(int)
    for key, files in data.iter_shard(DATA / "shards/synth/v2/test/synth-test.tar"):
        meta = json.loads(files["json"])
        if per_family[meta["family"]] >= 20:
            continue
        per_family[meta["family"]] += 1
        img = save(data.open_image(files["png"]), out / "synth/img" / f"{key}.png")
        qs = meta["qa"]
        for j in sorted(key_rng(key + "/audit").sample(range(len(qs)), min(5, len(qs)))):
            qid = f"{key}-q{j}"
            items.append({"id": qid, "image": img, "q": qs[j]["q"], "options": qs[j]["options"]})
            gold[qid] = qs[j]["answer"]
    for n, (key, files) in enumerate(data.iter_shard(DATA / "shards/synth/v2/pairs/synth-pairs.tar")):
        if n >= 80:
            break
        meta = json.loads(files["json"])
        img = save(data.open_image(files["png"]), out / "synth/img" / f"{key}.png")
        p = meta["pair"]
        items.append({"id": key, "image": img, "q": p["q"], "options": p["options"]})
        gold[key] = p["answer"]
    write(out, "synth", items, gold)


def mavis(out: Path, cfg: dict) -> None:
    src = next(s for s in make_sources(cfg["sources"]) if s.name == "mavis")
    items, gold = [], {}
    for key, files in validation_samples(src, cfg["val_count"])[:40]:
        meta = json.loads(files["json"])
        img = save(data.open_image(files["png"]), out / "mavis/img" / f"{key}.png")
        for j, q in enumerate(mavis_questions(meta["text_en"]["caption"].strip(), meta["depth"], key)):
            qid = f"{key}-q{j}"
            items.append({"id": qid, "image": img, "q": q["q"], "options": q["options"]})
            gold[qid] = q["answer"]
    write(out, "mavis", items, gold)


def stem_captions(out: Path) -> None:
    items = []
    for n, (key, files) in enumerate(data.iter_shard(DATA / "shards/mmfinereason/7c93f724-sft/mmfinereason-sft-00003.tar")):
        if len(items) >= 50:
            break
        if n % 40:
            continue
        meta = json.loads(files["json"])
        image, _ = data.stem_image(files)
        img = save(image, out / "stem_captions/img" / f"{key}.png")
        items.append({"id": key, "image": img, "source_dataset": meta.get("source_dataset"),
                      "caption_visible": visible_caption(meta["caption"]), "caption_original": meta["caption"]})
    write(out, "stem_captions", items, None)


def mmf(out: Path) -> None:
    answers, solutions, gold, seen = [], [], {}, set()
    for n, (key, files) in enumerate(data.iter_shard(DATA / "shards/f2-mmfinereason/f2-sft120k-v4/f2-mmfinereason-00000.tar")):
        if len(answers) >= 50:
            break
        meta = json.loads(files["json"])
        if n % 97 or meta["problem"] in seen:
            continue
        seen.add(meta["problem"])
        ext = "png" if "png" in files else "jpg"
        image = data.open_image(files[ext])
        img = save(image, out / "mmf/img" / f"{meta['problem']}.png")
        answers.append({"id": meta["problem"], "image": img, "question": meta["question"],
                        "source_dataset": meta.get("source_dataset")})
        solutions.append({"id": meta["problem"], "image": img, "question": meta["question"],
                          "reference_answer": meta["answer"], "solution": meta["solution"]})
        gold[meta["problem"]] = meta["answer"]
    write(out, "mmf_answers", answers, gold)
    write(out, "mmf_solutions", solutions, None)


def replay(out: Path) -> None:
    items, gold = [], {}
    for n, (key, files) in enumerate(data.iter_shard(DATA / "shards/replay/replay-slice1-p0/replay-00000.tar")):
        if len(items) >= 30:
            break
        if n % 21:
            continue
        meta = json.loads(files["json"])
        items.append({"id": meta["key"], "problem": meta["prompt"], "difficulty": meta.get("difficulty")})
        gold[meta["key"]] = meta["answer"]
    write(out, "replay", items, gold)


def unimer_lines(out: Path) -> None:
    items = []
    for n, (key, files) in enumerate(data.iter_shard(DATA / "shards/unimer/2343ddd9-dense/lines/unimer-lines-00000.tar")):
        if len(items) >= 30:
            break
        if n % 211:
            continue
        sample = data.unimer_line_sample(key, files)
        img = save(sample.image, out / "unimer_lines/img" / f"{key}.png")
        items.append({"id": key, "image": img, "transcription": sample.messages[1]["content"]})
    write(out, "unimer_lines", items, None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f1c_audit_export.py va eseguito in un job SLURM")
    cfg = load_config(args.config)
    stem_facts(args.out)
    synth(args.out)
    mavis(args.out, cfg)
    stem_captions(args.out)
    mmf(args.out)
    replay(args.out)
    unimer_lines(args.out)


if __name__ == "__main__":
    main()
