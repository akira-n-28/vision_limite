#!/usr/bin/env python3
"""Test CPU del replay testuale di F2 (src/limite_vl/replay.py, data.text_replay_sample, scripts/replay_select.py).
Da eseguire in slurm/tests_cpu.sbatch."""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.replay import (AIME_INSTRUCTION, MATH500_INSTRUCTION, choose, load_problems, prompt_style,  # noqa: E402
                              replay_prompt)

VIOLETTO = ROOT / "hf/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
SUITES = ROOT / "third_party/limite-violetto/limite-evals/src/limite_evals/suites.py"
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_prompts():
    consts = {}
    for node in ast.parse(SUITES.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    check("istruzione AIME uguale a quella del valutatore", consts.get("AIME_INSTRUCTION") == AIME_INSTRUCTION)
    check("istruzione MATH-500 uguale a quella del valutatore", consts.get("MATH500_INSTRUCTION") == MATH500_INSTRUCTION)
    check("prompt aime: istruzione prima", replay_prompt(" Find x. ", "aime") == AIME_INSTRUCTION + "Find x.")
    check("prompt math500: istruzione dopo", replay_prompt("Find x.", "math500") == "Find x." + MATH500_INSTRUCTION)
    styles = [prompt_style(f"dsr-{i:06d}") for i in range(1000)]
    check("stile deterministico", styles == [prompt_style(f"dsr-{i:06d}") for i in range(1000)])
    check("stili circa metà e metà", 400 < styles.count("aime") < 600, str(styles.count("aime")))


def make_sources(tmp: Path) -> tuple[Path, list[Path]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    dsr = tmp / "deepscaler.json"
    dsr.write_text(json.dumps([{"problem": f"Problem {i}: compute {i}+1.", "solution": "", "answer": str(i + 1)}
                               for i in range(10)]))
    dm = tmp / "train-00000-of-00002.parquet"
    pq.write_table(pa.table({"question": [f"Hard {i}: compute {i}*2." for i in range(10)],
                             "final_answer": [str(2 * i) for i in range(10)],
                             "difficulty": [5.0 + 0.5 * i for i in range(10)], "topic": ["algebra"] * 10,
                             "r1_solution_1": ["x"] * 10}), dm)
    return dsr, [dm]


def test_choose(tmp: Path):
    dsr, dm = make_sources(tmp)
    problems = load_problems(dsr, dm)
    check("caricamento: 20 problemi con chiavi distinte", len({p["key"] for p in problems}) == 20)
    check("chiavi di DeepMath dal nome del parquet", any(p["key"] == "dm-00000-00003" for p in problems))
    chosen = choose(problems, 8, 0.5, 6.0, 0, exclude={"dsr-000000"})
    srcs = [p["source"] for p in chosen]
    check("choose: metà per sorgente", srcs.count("deepscaler") == 4 and srcs.count("deepmath") == 4, str(srcs))
    check("choose: difficoltà minima", all(p["difficulty"] >= 6.0 for p in chosen if p["source"] == "deepmath"))
    check("choose: esclusioni", all(p["key"] != "dsr-000000" for p in chosen))
    check("choose: deterministico", [p["key"] for p in chosen] == [p["key"] for p in choose(problems, 8, 0.5, 6.0, 0, {"dsr-000000"})])


def test_example():
    from limite_vl.data import build_example, generation_prompt
    from limite_vl.processing import IGNORE, LimiteVLProcessor

    proc = LimiteVLProcessor.from_pretrained(str(VIOLETTO))
    tok = proc.tokenizer
    prompt = replay_prompt("What is 2+2?", "aime")
    files = {"json": json.dumps({"prompt": prompt, "solution": "<think>2+2=4</think>The answer is \\boxed{4}."}).encode()}
    ex = build_example(proc, "text_replay", "dsr-000001-s0", files)
    white = build_example(proc, "text_replay", "dsr-000001-s0", files, image_mode="white")
    check("solo testo: nessuna patch", not ex.patches and not ex.grids)
    check("modo white uguale a real", ex.input_ids.tolist() == white.input_ids.tolist())
    # come replay_generate.py: testo del template, poi tokenizzato da vLLM (default add_special_tokens=True)
    gen_text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    gen_ids = list(tok(gen_text)["input_ids"])
    check("prompt identico a quello della generazione", ex.input_ids[: len(gen_ids)].tolist() == gen_ids,
          tok.decode(ex.input_ids[: len(gen_ids)].tolist()) + " | " + gen_text)
    target = tok.decode(ex.input_ids[ex.labels != IGNORE].tolist(), skip_special_tokens=False)
    check("label: soluzione e <|endoftext|>", target == "<think>2+2=4</think>The answer is \\boxed{4}.<|endoftext|>", target)
    check("label assenti sul prompt", bool((ex.labels[: len(gen_ids)] == IGNORE).all()))
    gp, ref = generation_prompt(proc, "text_replay", "dsr-000001-s0", files)
    check("generation_prompt: prompt della generazione", gp.input_ids.tolist() == gen_ids and ref.startswith("<think>"),
          str(len(gp.input_ids)) + " contro " + str(len(gen_ids)))


def test_select(tmp: Path):
    dsr, dm = make_sources(tmp)
    gen = tmp / "gen"
    gen.mkdir()
    rows = [
        ("dsr-000001", "deepscaler", None, "2", [("<think>ok</think>\\boxed{2}", "stop", 100)]),
        ("dsr-000002", "deepscaler", None, "3", [("<think>no</think>\\boxed{5}", "stop", 100)]),
        ("dm-00000-00004", "deepmath", 7.0, "8", [("<think>" + "lungo " * 10, "length", 126976)]),
        ("dm-00000-00005", "deepmath", 7.5, "10", [("<think>ok</think>\\boxed{10}", "stop", 40000)]),
        ("dm-00000-00006", "deepmath", 8.0, "12", [("<think>ok</think>\\boxed{12}", "stop", 900)]),
    ]
    with open(gen / "part-00000.jsonl", "w") as fh:
        for key, src, diff, ans, comps in rows:
            fh.write(json.dumps({"key": key, "source": src, "answer": ans, "difficulty": diff, "topic": None,
                                 "style": prompt_style(key), "prompt_tokens": 50,
                                 "completions": [{"text": t, "finish_reason": f, "tokens": n} for t, f, n in comps]}) + "\n")
    out, shards = tmp / "sel", tmp / "shards"
    env = {**os.environ, "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "test")}
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/replay_select.py"), "--gen", str(gen),
                        "--deepscaler", str(dsr), "--deepmath", *map(str, dm), "--out", str(out), "--shards", str(shards),
                        "--workers", "1"], capture_output=True, text=True, env=env)
    check("replay_select termina", r.returncode == 0, r.stderr[-500:])
    if r.returncode:
        return
    stats = json.loads((out / "stats.json").read_text())
    check("esiti", stats["outcomes"] == {"correct": 3, "wrong": 1, "truncated": 1}, str(stats["outcomes"]))
    check("giuste troppo lunghe contate a parte", stats["kept"] == 2 and stats["correct_too_long"] == 1)
    with tarfile.open(shards / "replay-00000.tar") as tar:
        names = sorted(m.name for m in tar)
        meta = json.loads(tar.extractfile("dsr-000001-s0.json").read())
    check("shard: un campione per soluzione tenuta", names == ["dm-00000-00006-s0.json", "dsr-000001-s0.json"], str(names))
    check("shard: prompt ricostruito come in generazione",
          meta["prompt"] == replay_prompt("Problem 1: compute 1+1.", prompt_style("dsr-000001")))
    check("shard: soluzione intera", meta["solution"] == "<think>ok</think>\\boxed{2}")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="test_replay_", dir=os.environ.get("TMPDIR")))
    try:
        test_prompts()
        test_choose(tmp)
        test_example()
        test_select(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(FAILS)} falliti: {FAILS}" if FAILS else "\ntutti ok")
    sys.exit(1 if FAILS else 0)
