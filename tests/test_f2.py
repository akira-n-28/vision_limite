#!/usr/bin/env python3
"""Test CPU del ponte testuale di F2 (src/limite_vl/f2.py, scripts/f2_select.py). Da eseguire in slurm/tests_cpu.sbatch."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.f2 import (answers_match, bridge_prompt, clean_solution, grade, last_boxed, split_completion,  # noqa: E402
                          training_question)

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_parsing():
    check("last_boxed: graffe annidate e ultimo boxed", last_boxed("x \\boxed{1} y \\boxed{\\frac{1}{2}} z") == "\\frac{1}{2}")
    check("last_boxed: assente", last_boxed("no answer") is None)
    reasoning, final = split_completion("<think>calcolo</think>Quindi \\boxed{3}.")
    check("split_completion: ragionamento e risposta", reasoning == "calcolo" and final == "Quindi \\boxed{3}.")
    check("split_completion: senza </think>", split_completion("<think>troncato")[0] is None)
    p = bridge_prompt("Find x.", "A triangle with sides 3, 4, 5.")
    check("bridge_prompt: figura nel testo del problema, senza la parola description",
          p == "Figure:\nA triangle with sides 3, 4, 5.\n\nFind x.\nPlease reason step by step, and put your final answer "
               "within \\boxed{}." and "description" not in p.lower())
    check("training_question: domanda e istruzione, senza descrizione", training_question("Find x.") ==
          "Find x.\nPlease reason step by step, and put your final answer within \\boxed{}.")


def test_answers():
    cases = [("8", "8", True), ("8.0", "8", True), ("\\frac{1}{2}", "0.5", True), ("2\\sqrt{3}", "\\sqrt{12}", True),
             ("(A)", "A", True), ("\\text{B}", "B", True), ("A", "B", False), ("7", "8", False), ("x", "A", False)]
    bad = [(p, g) for p, g, want in cases if answers_match(p, g) != want]
    check("answers_match: numeri, frazioni, radici, lettere", not bad, str(bad))


def test_grade():
    ok = grade("<think>3+5=8</think>The answer is \\boxed{8}.", "8", "stop")
    check("grade: corretta e utilizzabile", ok["correct"] and ok["usable"])
    leak = grade("<think>The user describes a figure. From the description, the side is 8.</think>\\boxed{8}", "8", "stop")
    check("grade: corretta che cita la descrizione, riscritta e utilizzabile", leak["usable"] and leak["cleaned"] ==
          "<think>The user shows a figure. From the figure, the side is 8.</think>\\boxed{8}", str(leak["cleaned"]))
    blind = grade("<think>The image is not provided, so assume the side is 8.</think>\\boxed{8}", "8", "stop")
    check("grade: corretta ma dice di non avere l'immagine, scartata", blind["correct"] and not blind["usable"]
          and blind["reject"] == "no_image")
    for phrase in ("The user has provided a figure, but it's not an actual image.", "We don't have the image, so assume",
                   "The figure is given as text instead of an image."):
        g = grade(f"<think>{phrase} The side is 8.</think>\\boxed{{8}}", "8", "stop")
        check(f"grade: scartata come no_image: {phrase!r}", g["correct"] and g["reject"] == "no_image")
    for phrase in ("The image shows a triangle with the actual side lengths.", "We have the figure: the side is 8."):
        g = grade(f"<think>{phrase}</think>\\boxed{{8}}", "8", "stop")
        check(f"grade: non scartata: {phrase!r}", g["usable"])
    for phrase in ('The "Explicit Component Breakdown" lists the vertices.', "Identified Ambiguities: None.",
                   'The figure says "Figure 1: Geometric Diagram" with a triangle.', "The summary says the area is 8.",
                   "We have a description of a figure with a square."):  # -> "a figure of a figure"
        g = grade(f"<think>{phrase} The side is 8.</think>\\boxed{{8}}", "8", "stop")
        check(f"grade: scartata come mentions_caption: {phrase!r}", g["correct"] and g["reject"] == "mentions_caption",
              str(g["cleaned"]))
    for phrase in ("The summary statistics give a mean of 8.", "In Figure 1 the triangle is isosceles."):
        g = grade(f"<think>{phrase}</think>\\boxed{{8}}", "8", "stop")
        check(f"grade: non scartata come mentions_caption: {phrase!r}", g["usable"], str(g["reject"]))
    zh = grade("<think>The figure labels point 甲 on the circle, so the side is 8.</think>\\boxed{8}", "8", "stop")
    check("grade: corretta ma con caratteri CJK, scartata", zh["correct"] and not zh["usable"] and zh["reject"] == "non_english")
    from limite_vl.f2 import bridge_prompt, clean_caption
    caption = ("### Image 1: Geometric Diagram\n\n**Scene Summary:** Three adjacent squares with a shaded triangle.\n\n"
               "**Explicit Component Breakdown:**  \n* **Square (`ABCD`):** A square with vertices $A$, $B$, $C$, $D$.  \n\n"
               "**Interactions and Relationships:**  \n* Square $ABCD$ is to the right of square $BEFG$.  \n\n"
               "**Implicit and Inferred Properties:**  \n* **Square $BEFG$:** Side length is 3.  \n\n"
               "**Identified Ambiguities:**  \n* The label near $K$ is partially obscured.\n\n"
               "### Image 2: Table / Matrix\n\n**Scene Summary:** A table of values.\n\n**Identified Ambiguities:** None.")
    want = ("Three adjacent squares with a shaded triangle.\n\n* **Square (`ABCD`):** A square with vertices $A$, $B$, $C$, $D$.\n\n"
            "* Square $ABCD$ is to the right of square $BEFG$.\n\n* **Square $BEFG$:** Side length is 3.\n\n"
            "A table of values.")
    got = clean_caption(caption)
    check("clean_caption: niente titoli delle sezioni né del tipo di immagine, niente ambiguità, contenuto intatto",
          got == want, repr(got))
    check("bridge_prompt: descrizione pulita solo con clean=True", "Scene Summary" in bridge_prompt("Q?", caption)
          and "Scene Summary" not in bridge_prompt("Q?", caption, clean=True))
    check("clean_solution: maiuscole e figure description", clean_solution("Description: the figure description shows")[0]
          == "Figure: the figure shows")
    openings = {
        "We need to parse the problem. The user gave a description of a geometric diagram with triangle ABC.":
            "We need to parse the problem. The user shows a geometric diagram with triangle ABC.",
        "We have a problem description with a figure described in text.": "We have the problem with a figure shown.",
        "There's a description of a downward-opening parabola.": "There's a downward-opening parabola.",
        "The user has provided a description of a figure with two hyperbolas.": "The user shows a figure with two hyperbolas.",
    }
    got = {k: clean_solution(k)[0] for k in openings}
    check("clean_solution: aperture frequenti riscritte in modo leggibile", got == openings,
          str({k: v for k, v in got.items() if v != openings[k]}))
    check("grade: troncata non conta", not grade("<think>3+5=8</think>\\boxed{8}", "8", "length")["correct"])
    check("grade: senza </think> non conta", not grade("\\boxed{8}", "8", "stop")["correct"])
    check("grade: risposta sbagliata", not grade("<think>x</think>\\boxed{9}", "8", "stop")["correct"])


def test_mmfine():
    from limite_vl.mmfine import choose, load_texts, mmfine_key, scan_problems
    check("mmfine_key: sorgente normalizzata nella chiave", mmfine_key("sft", "We-Math 2.0/pro", 12) == "sft-We_Math_2_0_pro-12")
    probs = [{"key": f"k{i}", "source_dataset": "S" if i % 2 else "T", "qwen4b_pass_rate": 1.0 if i % 3 == 0 else 0.25,
              "file": "f", "row": i} for i in range(300)]
    pick = choose(probs, 60, 0.5, seed=0)
    again = choose(probs, 60, 0.5, seed=0)
    check("choose: metà facili e metà difficili, deterministico", len(pick) == 60
          and sum(p["qwen4b_pass_rate"] == 1.0 for p in pick) == 30 and pick == again)
    check("choose: esclusioni rispettate", not any(p["key"] == "k0" for p in choose(probs, 300, 0.5, 0, exclude={"k0"})))
    no_t = choose(probs, 300, 0.5, 0, exclude_sources={"T"})
    check("choose: sorgenti escluse", len(no_t) == 150 and all(p["source_dataset"] == "S" for p in no_t))
    snap = ROOT / "hf/hub/datasets--OpenDataArena--MMFineReason-1.8M/snapshots/7c93f724791d01246f5425e1182b14ec8d013847/data"
    shard = ROOT / "data/shards/mmfinereason/7c93f724-rl/mmfinereason-rl-00000.tar"
    parquet = snap / "rl-00000-of-00005.parquet"
    if not (parquet.is_file() and shard.is_file()):
        check("parquet e shard rl presenti", False, f"{parquet} {shard}")
        return
    from limite_vl.data import iter_shard
    shard_keys = {k for k, _ in iter_shard(shard)}
    scanned = scan_problems([parquet])
    inside = sum(p["key"] in shard_keys for p in scanned)
    check("scan_problems: le chiavi dei parquet sono quelle degli shard", inside >= 0.99 * len(scanned),
          f"{inside}/{len(scanned)}")
    texts = load_texts(scanned[:5])
    check("load_texts: domanda senza <image>, risposta e descrizione", len(texts) == 5
          and all("<image>" not in t["question"] and t["answer"] and t["caption"] for t in texts))


def test_generated_keys(tmp: Path):
    from limite_vl.f2 import generated_keys
    run = tmp / "run"
    run.mkdir()
    rows = [{"key": "sft-A-1", "completions": [{"text": '"key": "falsa"', "finish_reason": "stop", "tokens": 3}]},
            {"key": 'sft-B-"2"', "completions": []}]
    (run / "part-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "plan.json").write_text(json.dumps({"parts": 2}))
    check("generated_keys: chiavi dalla prima voce delle righe, anche con caratteri di escape",
          generated_keys(run) == {"sft-A-1", 'sft-B-"2"'})
    check("generated_keys: niente keys.txt se mancano parti del piano", not (run / "keys.txt").exists())
    (run / "part-00001.jsonl").write_text(json.dumps({"key": "sft-C-3", "completions": []}) + "\n")
    keys = generated_keys(run)
    check("generated_keys: keys.txt scritto a piano completo e poi usato", (run / "keys.txt").is_file()
          and keys == {"sft-A-1", 'sft-B-"2"', "sft-C-3"} and generated_keys(run) == keys)


def test_select(tmp: Path):
    gen = tmp / "gen"
    gen.mkdir()
    good = "<think>easy</think>\\boxed{4}"
    rows = [{"key": "p1", "source_dataset": "S", "answer": "4", "qwen4b_pass_rate": 1.0, "prompt_tokens": 10,
             "completions": [{"text": good, "finish_reason": "stop", "tokens": 5}] * 3
             + [{"text": "<think>no</think>\\boxed{5}", "finish_reason": "stop", "tokens": 5}]},
            {"key": "p2", "source_dataset": "S", "answer": "7", "qwen4b_pass_rate": 0.0, "prompt_tokens": 10,
             "completions": [{"text": "<think>no</think>\\boxed{5}", "finish_reason": "stop", "tokens": 5}] * 4}]
    (gen / "part-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    env = dict(os.environ, SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID", "test"))
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/f2_select.py"), "--gen", str(gen), "--out", str(tmp / "sel"),
                    "--keep", "2", "--workers", "2"], check=True, env=env)
    selected = [json.loads(line) for line in (tmp / "sel" / "selected.jsonl").read_text().splitlines()]
    rl = [json.loads(line) for line in (tmp / "sel" / "rl_pool.jsonl").read_text().splitlines()]
    check("f2_select: due soluzioni per il problema risolto, pass rate 3/4",
          len(selected) == 2 and all(s["key"] == "p1" and s["violetto_pass_rate"] == 0.75 for s in selected))
    check("f2_select: problema mai risolto nel pool di RL", [r["key"] for r in rl] == ["p2"])
    (tmp / "exclude.txt").write_text("p1\n")
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/f2_select.py"), "--gen", str(gen), "--out", str(tmp / "sel2"),
                    "--keep", "2", "--workers", "2", "--exclude", str(tmp / "exclude.txt")], check=True, env=env)
    stats = json.loads((tmp / "sel2" / "stats.json").read_text())
    check("f2_select: problemi contaminati saltati", (tmp / "sel2" / "selected.jsonl").read_text() == ""
          and stats["excluded_contaminated"] == 1 and stats["problems"] == 1)


if __name__ == "__main__":
    test_parsing()
    test_answers()
    test_grade()
    test_mmfine()
    tmp = ROOT / "outputs" / "tmp" / f"test-f2-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        test_generated_keys(tmp)
        test_select(tmp)
    finally:
        shutil.rmtree(tmp)
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
