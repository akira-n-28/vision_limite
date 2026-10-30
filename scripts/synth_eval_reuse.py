#!/usr/bin/env python3
"""Quante scene di test e coppie dei sintetici v4 hanno lo stesso stato delle scene v3 con la stessa chiave
(docs/PANNELLO_VALUTAZIONE.md). Test e coppie v3 sono stati consultati nelle valutazioni F1c (job 29741-29761): una
scena v4 con lo stesso stato è la stessa istanza, quindi non è un test indipendente. Solo specifiche, nessun disegno;
solo in un job SLURM CPU.

Lo stato è quello di scripts/synth_candidate.scene_state (specifica senza chiave, versione e stile). Per le coppie si
confrontano gli stati delle varianti a e b costruite come in synth_render.scene, senza disegnarle.

    srun $RUN_PYTHON -u -B scripts/synth_eval_reuse.py --out DIR [--test 300] [--pairs 300]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from limite_vl import synth  # noqa: E402
from limite_vl.provenance import code_identity, write_new_json  # noqa: E402
from synth_candidate import digest, scene_state  # noqa: E402


def pair_states(key: str, family: str, version: int) -> tuple[str, str] | None:
    """Stati delle varianti a e b della coppia, con la stessa ricerca dei tentativi di synth_render.scene."""
    for attempt in range(20):
        spec = synth.make_spec(f"{key}/{attempt}", family, version)
        cf = synth.counterfactual(spec)
        if cf is None or cf[3] == cf[4]:
            continue
        return digest(scene_state(cf[0])), digest(scene_state(cf[1]))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--pairs", type=int, default=300)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("synth_eval_reuse.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota")
    fam = synth.FAMILIES
    test = Counter()
    for i in range(args.test):
        key, family = f"synth-test-{i:05d}", fam[i % len(fam)]
        same = digest(scene_state(synth.make_spec(key, family, 3))) == digest(scene_state(synth.make_spec(key, family, 4)))
        test[f"{family}/{'stesso_stato' if same else 'stato_diverso'}"] += 1
    pairs = Counter()
    for i in range(args.pairs):
        key, family = f"synth-pair-{i:05d}", fam[i % len(fam)]
        s3, s4 = pair_states(key, family, 3), pair_states(key, family, 4)
        if s3 is None or s4 is None:
            pairs[f"{family}/coppia_mancante"] += 1
            continue
        pairs[f"{family}/a_{'uguale' if s3[0] == s4[0] else 'diversa'}_b_{'uguale' if s3[1] == s4[1] else 'diversa'}"] += 1
    report = {"schema": "synth-eval-reuse/1", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
              "test": dict(sorted(test.items())), "coppie": dict(sorted(pairs.items())),
              "test_stesso_stato": sum(v for k, v in test.items() if k.endswith("stesso_stato")),
              "coppie_a_e_b_uguali": sum(v for k, v in pairs.items() if k.endswith("a_uguale_b_uguale"))}
    args.out.mkdir(parents=True, exist_ok=True)
    write_new_json(args.out / "reuse.json", report)
    print(report)


if __name__ == "__main__":
    main()
