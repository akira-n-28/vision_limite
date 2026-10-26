#!/usr/bin/env python3
"""Controlli di UniMER senza LLM su tutti gli shard di train (F1-clean), quarta versione: categorie separate.

Per ogni campione (immagine PNG e JSON con la formula):
- errori certi (quality_excluded): struttura del LaTeX rotta (latex.structure_problems), formula senza token visibili,
  immagine vuota o con dimensioni diverse dal JSON, comandi che non disegnano il loro testo (\\footnote, \\label, \\cite,
  \\ref: l'etichetta contiene testo che nell'immagine non c'è);
- rietichettati (relabel.jsonl, latex.effective_label): graffe scritte Ḋ e Ḍ (`^Ḋ74Ḍ` = `^{74}`); formula fatta solo
  di un indice senza base (`^{26}`, anche `{_4}`): ritagliata, l'immagine mostra solo il contenuto, che diventa
  l'etichetta. `^2S_{1/2}` e `^{14}C` restano come sono;
- stessa immagine (sha256 dei byte) con etichette visibili diverse dopo la rietichettatura e la normalizzazione di
  latex.visible_tokens: si disegnano le etichette candidate con matplotlib (latex_render) e si tiene quella che
  corrisponde ai pixel. Le altre del gruppo vanno fra gli errori certi (stessa immagine, etichetta sbagliata). La scelta
  si usa solo se i controlli con verità nota (etichetta vera contro etichetta alterata su formule brevi non segnalate)
  danno almeno --min-control-accuracy; altrimenti, e quando non c'è un vincitore netto, il gruppo va fra gli incerti;
- incerti (uncertain): gruppi irrisolti, proporzioni fuori norma (z robusto oltre --z), marcatori Ḋ e Ḍ che non si
  riparano (graffe sbilanciate dopo la sostituzione).
Le stesse immagini con la stessa etichetta sono duplicati, contati qui e trattati dal controllo dei duplicati (test E).

Uscite in --out (cartella nuova): quality_excluded.txt e uncertain.txt (una chiave per riga, per `exclude` della config),
relabel.jsonl (chiave, etichetta originale, nuova, regola), reasons.jsonl (chiave e categorie) e report.json.
Job CPU: slurm/unimer_check.sbatch.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.latex import effective_label, structure_problems, unrendered_commands, visible_tokens  # noqa: E402
from limite_vl.latex_render import best_label, to_gray  # noqa: E402
from limite_vl.provenance import code_identity, file_entry  # noqa: E402

EXAMPLES = 20
CONTROLS = 400
CERTAIN = {"graffe", "left_right", "ambiente", "formula_vuota", "immagine_vuota", "dimensioni_diverse_dal_json",
           "comando_non_reso", "etichetta_diversa_dall_immagine"}
UNCERTAIN = {"conflitto_irrisolto", "proporzioni_fuori_norma", "marcatori_non_riparabili"}


def iter_pairs(path: Path):
    """(chiave, json, png) dagli shard: i due file di un campione sono consecutivi."""
    pending: dict[str, dict] = {}
    with tarfile.open(path, "r|") as tar:
        for m in tar:
            if not m.isfile():
                continue
            key, ext = m.name.rsplit("/", 1)[-1].split(".", 1)
            pending.setdefault(key, {})[ext] = tar.extractfile(m).read()
            if {"json", "png"} <= pending[key].keys():
                files = pending.pop(key)
                yield key, json.loads(files["json"]), files["png"]
    if pending:
        raise RuntimeError(f"{path.name}: campioni incompleti {sorted(pending)[:5]}")


def check_shard(path: Path) -> list[tuple]:
    out = []
    for key, meta, png in iter_pairs(path):
        label = meta["label"]
        effective, rules = effective_label(label)
        new = effective if rules else None
        vis = visible_tokens(effective)
        problems = list(structure_problems(effective))
        if unrendered_commands(effective):
            problems.append("comando_non_reso")
        if not vis:
            problems.append("formula_vuota")
        with Image.open(io.BytesIO(png)) as im:
            gray = im.convert("L")
            lo, hi = gray.getextrema()
            dark = sum(gray.histogram()[:128]) / max(1, gray.width * gray.height)
            if (gray.width, gray.height) != (meta["width"], meta["height"]):
                problems.append("dimensioni_diverse_dal_json")
        if hi - lo < 16 or dark == 0:
            problems.append("immagine_vuota")
        if "Ḋ" in effective or "Ḍ" in effective:
            problems.append("marcatori_non_riparabili")
        problems += [f"rietichettato_{r}" for r in rules]
        area = meta["orig_width"] * meta["orig_height"]
        ratio = math.log(area / len(vis)) if vis and area else None
        out.append((key, str(path), hashlib.sha256(png).digest(), " ".join(vis), label, new, ratio, problems))
    return out


def read_images(path: str, keys: set[str]) -> dict[str, bytes]:
    return {k: png for k, _, png in iter_pairs(Path(path)) if k in keys}


def perturb(label: str, rng: random.Random) -> str | None:
    """Etichetta alterata per i controlli: una lettera cambiata, oppure una barra sulla lettera."""
    letters = list(re.finditer(r"(?<![\\A-Za-z])[A-Za-z](?![A-Za-z])", label))
    if not letters:
        return None
    m = rng.choice(letters)
    if rng.random() < 0.5:
        repl = "b" if m.group() != "b" else "d"
        return label[:m.start()] + repl + label[m.end():]
    return label[:m.start()] + "\\bar{" + m.group() + "}" + label[m.end():]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--z", type=float, default=5.0)
    ap.add_argument("--min-control-accuracy", type=float, default=0.95)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("unimer_check.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} esiste e non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    rows = []
    with Pool(workers) as pool:
        for path, res in zip(args.shards, pool.imap(check_shard, args.shards)):
            rows.extend(res)
            print(f"{path.name}: {len(res)} campioni", flush=True)
    info = {r[0]: r for r in rows}
    cats: dict[str, list[str]] = defaultdict(list)
    for key, _, _, _, _, new, _, problems in rows:
        cats[key].extend(problems)

    # gruppi con la stessa immagine ed etichette visibili diverse (dopo rietichettatura e normalizzazione)
    groups: dict[bytes, list[str]] = defaultdict(list)
    for key, _, img, *_ in rows:
        groups[img].append(key)
    conflict_groups = [ks for ks in groups.values() if len({info[k][3] for k in ks}) > 1]
    duplicates = sum(len(ks) - 1 for ks in groups.values() if len({info[k][3] for k in ks}) == 1)

    # controlli con verità nota per il confronto con il disegno di matplotlib
    rng = random.Random(20261026)
    clean = [r for r in rows if not cats[r[0]] and 2 <= len(r[3].split()) <= 6]
    controls = rng.sample(clean, min(CONTROLS, len(clean)))
    need: dict[str, set[str]] = defaultdict(set)
    for ks in conflict_groups:
        for k in ks:
            need[info[k][1]].add(k)
    for r in controls:
        need[r[1]].add(r[0])
    images: dict[str, bytes] = {}
    with Pool(workers) as pool:
        for got in pool.starmap(read_images, [(p, ks) for p, ks in need.items()]):
            images.update(got)

    ctrl, ctrl_examples = Counter(), []
    for r in controls:
        wrong = perturb(r[4], rng)
        if wrong is None:
            continue
        with Image.open(io.BytesIO(images[r[0]])) as im:
            pick, scores = best_label(to_gray(im), [r[4], wrong])
        ctrl["controlli"] += 1
        outcome = "giusti" if pick == r[4] else "sbagliati" if pick == wrong else "senza_vincitore"
        ctrl[outcome] += 1
        if outcome != "giusti" and len(ctrl_examples) < 30:
            ctrl_examples.append({"key": r[0], "label": r[4], "wrong": wrong, "outcome": outcome,
                                  "scores": {k: round(v, 3) for k, v in scores.items()}})
    decided = ctrl["giusti"] + ctrl["sbagliati"]
    accuracy = ctrl["giusti"] / decided if decided else 0.0
    use_render = accuracy >= args.min_control_accuracy and decided >= 100

    resolved = unresolved = 0
    loser_commands = Counter()
    resolutions = []
    for ks in conflict_groups:
        effective = {k: (info[k][5] if info[k][5] is not None else info[k][4]) for k in ks}
        pick, scores = None, {}
        if use_render:
            with Image.open(io.BytesIO(images[ks[0]])) as im:
                pick, scores = best_label(to_gray(im), sorted(set(effective.values())))
        if pick is None:
            unresolved += 1
            for k in ks:
                cats[k].append("conflitto_irrisolto")
        else:
            resolved += 1
            win = " ".join(visible_tokens(pick))
            for k in ks:
                if info[k][3] != win:
                    cats[k].append("etichetta_diversa_dall_immagine")
                    loser_commands.update(t for t in info[k][3].split() if t.startswith("\\") and t not in win.split())
        if len(resolutions) < 60:
            resolutions.append({"keys": ks, "labels": sorted(set(effective.values())), "pick": pick,
                                "scores": {k: round(v, 3) for k, v in scores.items()}})

    ratios = [r[6] for r in rows if r[6] is not None]
    med = statistics.median(ratios)
    mad = statistics.median(abs(x - med) for x in ratios) * 1.4826
    for key, *_, ratio, _ in rows:
        if ratio is not None and mad and abs(ratio - med) / mad > args.z:
            cats[key].append("proporzioni_fuori_norma")

    excluded, uncertain, relabels = [], [], []
    counts, states, examples = Counter(), Counter(), defaultdict(list)
    with open(args.out / "reasons.jsonl", "w") as fh:
        for key in sorted(cats):
            cs = sorted(set(cats[key]))
            if not cs:
                continue
            counts.update(cs)
            r = info[key]
            if any(c in CERTAIN for c in cs):
                state = "excluded"
                excluded.append(key)
            elif any(c in UNCERTAIN for c in cs):
                state = "uncertain"
                uncertain.append(key)
            else:
                state = "accepted"
            states[state] += 1
            if state == "accepted" and r[5] is not None:
                rules = [c.removeprefix("rietichettato_") for c in cs if c.startswith("rietichettato_")]
                relabels.append({"key": key, "label": r[4], "new_label": r[5], "rule": "+".join(rules)})
            rec = {"key": key, "state": state, "categories": cs, "label": r[4]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for c in cs:
                if len(examples[c]) < EXAMPLES:
                    examples[c].append(rec)
    (args.out / "quality_excluded.txt").write_text("".join(k + "\n" for k in excluded))
    (args.out / "uncertain.txt").write_text("".join(k + "\n" for k in uncertain))
    with open(args.out / "relabel.jsonl", "w") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in relabels)

    report = {
        "schema": "unimer-check/4",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code": code_identity(),
        "shards": [file_entry(s) for s in args.shards],
        "samples": len(rows),
        "quality_excluded": len(excluded),
        "uncertain": len(uncertain),
        "relabeled": len(relabels),
        "accepted_with_flags": states["accepted"],
        "accepted": len(rows) - len(excluded) - len(uncertain),
        "categories": dict(counts.most_common()),
        "conflict_groups": len(conflict_groups),
        "conflict_samples": sum(len(ks) for ks in conflict_groups),
        "conflicts_resolved_by_render": resolved,
        "conflicts_unresolved": unresolved,
        "render_controls": dict(ctrl), "render_control_accuracy": round(accuracy, 4), "render_used": use_render,
        "render_control_failures": ctrl_examples,
        "commands_only_in_wrong_labels": loser_commands.most_common(40),
        "duplicates_same_image_same_label": duplicates,
        "ratio_log_median": med, "ratio_log_mad": mad, "z": args.z,
        "resolutions": resolutions,
        "examples": examples,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("samples", "quality_excluded", "uncertain", "relabeled", "accepted",
                                             "categories", "conflict_groups", "conflicts_resolved_by_render",
                                             "conflicts_unresolved", "render_controls", "render_control_accuracy",
                                             "render_used")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
