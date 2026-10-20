#!/usr/bin/env python3
"""Test della misura delle fughe dal ponte testuale (scripts/f2_leak.py): sezioni della descrizione, copie dalle parti
non visibili e visibili al netto della domanda, risposta nella descrizione, tracce del ponte."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import f2_leak as fl  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


CAPTION = """### Image 1: Flowchart
**Scene Summary:** A flowchart with a decision diamond and three output boxes for a piecewise function.

**Explicit Component Breakdown:**
* **Decision diamond (`x > 1`):** checks whether the input exceeds one before branching to the left output box.

**Implicit and Inferred Properties:**
* **Calculation Requirement:** The sum a + b + c corresponds to 1 + 1 + 4 = 6 for the three given inputs.

**Identified Ambiguities:**
* None."""
QUESTION = "Given the flowchart, if the inputs x are 0, 1 and 2 the outputs are a, b and c. Then a+b+c = ____."


def test_sections():
    vis, hid = fl.sections(CAPTION)
    check("sezioni: visibile senza le proprietà dedotte", "decision diamond" in vis and "Calculation" not in vis)
    check("sezioni: nascosta con le proprietà dedotte", "corresponds to 1 + 1 + 4 = 6" in hid and "diamond" not in hid)


def test_analyze():
    leak = ("We compute each output. The problem also mentions \"Calculation Requirement: The sum a + b + c corresponds to "
            "1 + 1 + 4 = 6 for the three given inputs\". So the answer is 6.")
    r = fl.analyze(leak, QUESTION, CAPTION, "6", 8)
    check("copia dalla parte nascosta", r["copia_nascosta"] and r["copertura"] > 0.2, r)
    clean = "For x = 0 we use y = 4^x = 1, for x = 1 y = 1 and for x = 2 y = 4, so a + b + c = 6."
    r = fl.analyze(clean, QUESTION, CAPTION, "6", 8)
    check("soluzione pulita: niente copia", not r["copia_nascosta"] and not r["copia_visibile"], r)
    vis = "Look: a decision diamond checks whether the input exceeds one before branching to the left output box."
    r = fl.analyze(vis, QUESTION, CAPTION, "6", 8)
    check("copia dalla parte visibile", r["copia_visibile"] and not r["copia_nascosta"], r)
    restate = "Given the flowchart, if the inputs x are 0, 1 and 2 the outputs are a, b and c, we sum them."
    r = fl.analyze(restate, QUESTION, CAPTION, "6", 8)
    check("ripetere la domanda non è una fuga", not r["copia_nascosta"] and not r["copia_visibile"], r)
    check("risposta nella parte nascosta (non stretta per un numero di una cifra)",
          r["risposta_nascosta"] and not r["risposta_nascosta_stretta"], r)
    r = fl.analyze("The figure says the sum is shown but not given here.", QUESTION, CAPTION, "6", 8)
    check("tracce del ponte", set(r["tracce"]) >= {"the_figure_says", "not_given"}, r["tracce"])


if __name__ == "__main__":
    test_sections()
    test_analyze()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
