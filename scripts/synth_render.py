#!/usr/bin/env python3
"""Disegna i diagrammi sintetici di F1c (limite_vl.synth) in shard tar. Solo in un job SLURM CPU.

Uscite in --out:
- train/synth-train-NNNNN.tar: --train scene, famiglie a rotazione (chiavi synth-NNNNNNN);
- test/synth-test.tar: --test scene di test (chiavi synth-test-NNNNN), con semi diversi dal training;
- pairs/synth-pairs.tar: --pairs coppie controfattuali (chiavi synth-pair-NNNNN-a e -b): la stessa scena con un
  solo dettaglio decisivo cambiato e la domanda la cui risposta cambia (nel JSON: pair = id, variante, domanda,
  opzioni, risposta per quella variante);
- manifest.json: conteggi per famiglia, tipo di domanda e risposta (per controllare l'equilibrio).
Ogni campione: <chiave>.png alla dimensione esatta (lati multipli di 32) e <chiave>.json con id, famiglia,
dimensioni, max_patches, domande (qa) e specifica della scena.

    srun $RUN_PYTHON -u -B scripts/synth_render.py --out DIR [--train 60000] [--test 300] [--pairs 300]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shard_llava_pretrain import HashingWriter, add_member  # noqa: E402  (stessa cartella scripts/)
from shard_mavis_geometry import check_shard  # noqa: E402

from limite_vl import synth  # noqa: E402


def scene(task: tuple[str, str]) -> list[tuple[str, bytes, dict]]:
    """Una scena (o una coppia controfattuale) -> [(chiave, png, meta)]. Gira nei processi del pool."""
    key, family = task
    if not key.startswith("synth-pair-"):
        spec = synth.make_spec(key, family)
        return [(key, synth.png_bytes(synth.render(spec)), _meta(key, spec, synth.questions(spec)))]
    for attempt in range(20):
        spec = synth.make_spec(f"{key}/{attempt}", family)
        cf = synth.counterfactual(spec)
        if cf is None or cf[3] == cf[4]:
            continue
        orig, new, q, ans_a, ans_b = cf
        out = []
        for variant, s, ans in (("a", orig, ans_a), ("b", new, ans_b)):
            k = f"{key}-{variant}"
            pair = {"id": key, "variant": variant, "q": q["q"], "type": q["type"], "options": q["options"], "answer": ans}
            if ans not in q["options"]:
                raise ValueError(f"{k}: risposta {ans!r} fuori dalle opzioni")
            out.append((k, synth.png_bytes(synth.render(s)), _meta(k, s, [], pair)))
        return out
    raise RuntimeError(f"{key}: nessuna coppia controfattuale in 20 tentativi")


def _meta(key: str, spec: dict, qa: list[dict], pair: dict | None = None) -> dict:
    meta = {"id": key, "family": spec["family"], "width": spec["width"], "height": spec["height"],
            "max_patches": (spec["width"] // 16) * (spec["height"] // 16), "qa": qa, "spec": spec}
    if pair:
        meta["pair"] = pair
    return meta


def write_tar(path: Path, tasks: list[tuple[str, str]], pool: Pool, counts: Counter) -> dict:
    t0 = time.time()
    tmp = path.with_name(path.name + ".tmp")
    written = []
    with open(tmp, "wb") as fh:
        hw = HashingWriter(fh)
        with tarfile.open(fileobj=hw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for items in pool.imap(scene, tasks, chunksize=16):
                for key, png, meta in items:
                    add_member(tar, f"{key}.png", png)
                    add_member(tar, f"{key}.json", json.dumps(meta, ensure_ascii=False).encode())
                    written.append(key)
                    counts[f"family/{meta['family']}"] += 1
                    for q in meta["qa"]:
                        counts[f"qa/{meta['family']}/{q['type']}"] += 1
                        counts[f"answer/{q['type']}/{q['answer']}"] += 1
                    if "pair" in meta:
                        counts[f"pair/{meta['family']}/{meta['pair']['type']}"] += 1
        fh.flush()
        os.fsync(fh.fileno())
    check_shard(tmp, written)
    os.replace(tmp, path)
    row = {"name": path.name, "samples": len(written), "bytes": hw.n, "sha256": hw.sha.hexdigest(),
           "seconds": round(time.time() - t0, 1)}
    print(f"{path.name}: {row['samples']} campioni, {row['bytes'] / 1e6:.0f} MB, {row['seconds']:.0f} s", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--per-shard", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("synth_render.py va eseguito in un job SLURM")
    if (args.out / "manifest.json").exists():
        sys.exit(f"{args.out} contiene già un manifest: scegliere un'altra cartella")
    fam = synth.FAMILIES
    sets = {
        "train": [(f"synth-{i:07d}", fam[i % len(fam)]) for i in range(args.train)],
        "test": [(f"synth-test-{i:05d}", fam[i % len(fam)]) for i in range(args.test)],
        "pairs": [(f"synth-pair-{i:05d}", fam[i % len(fam)]) for i in range(args.pairs)],
    }
    manifest = {"job_id": os.environ.get("SLURM_JOB_ID"), "args": {k: str(v) for k, v in vars(args).items()},
                "sets": {}, "counts": {}}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for name, tasks in sets.items():
            counts: Counter = Counter()
            (args.out / name).mkdir(parents=True, exist_ok=True)
            if name == "train":
                chunks = [tasks[i:i + args.per_shard] for i in range(0, len(tasks), args.per_shard)]
                rows = [write_tar(args.out / name / f"synth-train-{j:05d}.tar", c, pool, counts)
                        for j, c in enumerate(chunks)]
            else:
                rows = [write_tar(args.out / name / f"synth-{name}.tar", tasks, pool, counts)]
            manifest["sets"][name] = rows
            manifest["counts"][name] = dict(sorted(counts.items()))
    manifest["shards"] = manifest["sets"]["train"]
    (args.out / "train" / "manifest.json").write_text(json.dumps(  # train_f1.shard_hashes lo cerca accanto agli shard
        {"source": "limite_vl.synth", "job_id": manifest["job_id"], "shards": manifest["sets"]["train"]}, indent=1))
    manifest["seconds"] = round(time.time() - t0, 1)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    for name, c in manifest["counts"].items():
        print(name, json.dumps({k: v for k, v in c.items() if not k.startswith("answer/")}), flush=True)
    print(f"fatto in {manifest['seconds']:.0f} s")


if __name__ == "__main__":
    main()
