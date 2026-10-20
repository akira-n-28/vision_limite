"""Replay testuale di F2: problemi di testo con risposta verificabile, risolti da Violetto con il budget pieno.

Sorgenti (gruppo p5-replay di downloads/sources.tsv, scelta dell'utente del 2026-10-16):
- agentica-org/DeepScaleR-Preview-Dataset (deepscaler.json): AIME 1984-2023, AMC, Omni-MATH, STILL, con
  `problem` e `answer`;
- zwhe99/DeepMath-103K (data/train-*.parquet): `question`, `final_answer`, `difficulty` (livelli circa 5-9),
  `topic`; le soluzioni di R1 non si leggono.

Prompt: le due istruzioni del valutatore (limite-evals, suites.py), scelte per problema in modo deterministico
dalla chiave, così il replay copre i due formati della suite: quella di AIME prima del problema, quella di
MATH-500 e BeyondAIME dopo.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

# Testuali da third_party/limite-violetto/limite-evals/src/limite_evals/suites.py (AIME_INSTRUCTION,
# MATH500_INSTRUCTION).
AIME_INSTRUCTION = (
    "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
)
MATH500_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
STYLES = ("aime", "math500")


def prompt_style(key: str) -> str:
    return STYLES[int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2]


def replay_prompt(question: str, style: str) -> str:
    if style == "aime":
        return AIME_INSTRUCTION + question.strip()
    if style == "math500":
        return question.strip() + MATH500_INSTRUCTION
    raise ValueError(f"stile sconosciuto: {style}")


def load_deepscaler(path: Path) -> list[dict]:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows or not {"problem", "answer"} <= set(rows[0]):
        raise ValueError(f"{path}: attesa una lista di oggetti con problem e answer")
    return [{"key": f"dsr-{i:06d}", "source": "deepscaler", "question": r["problem"], "answer": str(r["answer"]),
             "difficulty": None, "topic": None}
            for i, r in enumerate(rows)]


def load_deepmath(paths: list[Path]) -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    for path in sorted(paths):
        table = pq.read_table(path, columns=["question", "final_answer", "difficulty", "topic"])
        stem = Path(path).name.split("-of-")[0].removeprefix("train-")  # train-00003-of-00010 -> 00003
        for i, r in enumerate(table.to_pylist()):
            out.append({"key": f"dm-{stem}-{i:05d}", "source": "deepmath", "question": r["question"],
                        "answer": str(r["final_answer"]), "difficulty": r["difficulty"], "topic": r["topic"]})
    return out


def load_problems(deepscaler: Path | None, deepmath: list[Path]) -> list[dict]:
    problems = (load_deepscaler(deepscaler) if deepscaler else []) + (load_deepmath(deepmath) if deepmath else [])
    keys = [p["key"] for p in problems]
    if len(set(keys)) != len(keys):
        raise ValueError("chiavi duplicate fra le sorgenti del replay")
    return problems


def choose(problems: list[dict], n: int, dsr_frac: float, min_difficulty: float, seed: int,
           exclude: set[str]) -> list[dict]:
    """n problemi: dsr_frac da DeepScaleR, il resto da DeepMath con difficulty >= min_difficulty, in ordine
    pseudo-casuale fissato dal seed; senza domanda o risposta e chiavi di exclude saltati."""
    ok = [p for p in problems if p["key"] not in exclude and (p["question"] or "").strip() and (p["answer"] or "").strip()]
    dsr = [p for p in ok if p["source"] == "deepscaler"]
    dm = [p for p in ok if p["source"] == "deepmath" and (p["difficulty"] or 0) >= min_difficulty]
    rng = random.Random(seed)
    rng.shuffle(dsr)
    rng.shuffle(dm)
    n_dsr = min(len(dsr), round(n * dsr_frac))
    n_dm = min(len(dm), n - n_dsr)
    return sorted(dsr[:n_dsr] + dm[:n_dm], key=lambda p: p["key"])
