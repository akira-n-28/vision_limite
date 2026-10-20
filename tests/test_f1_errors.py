#!/usr/bin/env python3
"""Test dell'analisi degli errori di F1 (scripts/f1_error_report.py): token visibili delle formule, allineamento,
cicli, tipi di sostituzione, fatti delle didascalie MAVIS, elementi delle descrizioni STEM, tipi di token."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import f1_error_report as rep  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_visible_tokens():
    v = rep.visible_tokens
    check("visibili: \\cal e \\mathcal uguali", v(r"{\cal L}^{x}") == v(r"\mathcal{L}^x"), v(r"{\cal L}^{x}"))
    check("visibili: x^{\\prime} = x'", v(r"x^{\prime}") == v("x'") == ["x", "'"], v(r"x^{\prime}"))
    check("visibili: \\left( e \\big( come (", v(r"\left(a\right)") == v(r"\big(a\big)") == ["(", "a", ")"])
    check("visibili: spazi e array tolti", v(r"\begin{array}{rl}a&=b\,.\end{array}") == ["a", "=", "b", "."],
          v(r"\begin{array}{rl}a&=b\,.\end{array}"))
    check("visibili: sinonimi", v(r"a\ge b") == v(r"a\geq b"))


def test_align():
    ops = rep.align(list("abxd"), list("abcde"))
    kinds = [o[0] for o in ops]
    check("allineamento: una sostituzione e un'omissione", kinds.count("sub") == 1 and kinds.count("del") == 1
          and kinds.count("eq") == 3, kinds)
    check("allineamento: accuratezza", abs(rep.accuracy(ops, 5) - 0.6) < 1e-9)
    ops = rep.align(list("abcc"), list("abc"))
    check("allineamento: aggiunta", [o[0] for o in ops].count("ins") == 1)


def test_loop():
    toks = list("xyz") + list("ab") * 15
    check("ciclo: coda periodica", rep.loop_start(toks) == 3, rep.loop_start(toks))
    check("ciclo: nessuna coda", rep.loop_start(list("abcdefghijklmnopqrstuvwxyz")) is None)
    check("ciclo: parole di una didascalia",
          rep.is_loop("This detailed illustration is a " + "detailed, and " * 10, eos=False)
          and not rep.is_loop("A cat sits on a mat.", eos=True))


def test_sub_kind():
    check("sostituzione: a/\\alpha glifo simile", rep.sub_kind("a", r"\alpha") == "glifo simile")
    check("sostituzione: K/\\kappa glifo simile", rep.sub_kind("K", r"\kappa") == "glifo simile")
    check("sostituzione: P/p maiuscola", rep.sub_kind("P", "p") == "maiuscola/minuscola")
    check("sostituzione: stessa classe", rep.sub_kind("3", "8") == "stessa classe: cifra")
    check("sostituzione: classe diversa", rep.sub_kind("+", "x") == "classe diversa")


REF = ("ABCED is a rectangle, but the side CD is extended into an equilateral triangle outside. Besides, ECF takes on a "
       "sector shape. Angle BAD has a measure of 90 degrees. Angle FCE is measured at 60 degrees. Side AB is 4 units "
       "long. Side AD measures 7 units in length. The segment EC spans 4 units.")
GEN = ("ABCED is a rectangle, but the side CD is extended into an equilateral triangle outside the rectangle. ECF is "
       "identified as a sector. FCG is geometrically a sector. The measure of angle BAD is 90 degrees. The angle FCE "
       "is 60 degrees. The measure of angle GCF is 60°. The length of side AB is 4. The length of side AD is 7. The "
       "length of side EC is 4.")


def test_mavis():
    ref = rep.mavis_facts(REF)
    check("MAVIS: figure", ref["shape"] == {"ABCDE": "rectangle", "ext:CD": "equilateral triangle", "CEF": "sector"},
          ref["shape"])
    check("MAVIS: angoli", ref["angle"] == {"A|BD": "90", "C|EF": "60"}, ref["angle"])
    check("MAVIS: lati", ref["side"] == {"AB": "4", "AD": "7", "CE": "4"}, ref["side"])
    c = rep.compare_facts(ref, rep.mavis_facts(GEN))
    check("MAVIS: figura inventata", c["shape"]["ok"] == 3 and c["shape"]["extra"] == 1, c["shape"])
    check("MAVIS: angolo inventato", c["angle"]["ok"] == 2 and c["angle"]["extra"] == 1, c["angle"])
    check("MAVIS: lati giusti", c["side"]["ok"] == 3 and c["side"]["extra"] == 0, c["side"])
    f = rep.mavis_facts("ABCD is a square. The distance between points A and B is 80. Triangle EFG is isosceles.")
    check("MAVIS: punti A e B, forma prima del nome", f["side"] == {"AB": "80"} and f["shape"]["EFG"] == "triangle"
          and f["shape"]["ABCD"] == "square", f)
    w = rep.compare_facts(ref, rep.mavis_facts(REF.replace("7 units", "9 units").replace("a rectangle", "a square")))
    check("MAVIS: valore e tipo sbagliati", w["side"]["wrong"] == 1 and w["shape"]["wrong"] == 1, w)


def test_stem():
    text = ("### Image 1: Geometric Diagram\n\n**Scene Summary:** Square ABCD with a point E.\n* **Vertex A (`A`):** "
            "top-left.\n* **Side (`AB`):** length `6 cm`, angle of 45°.")
    items = rep.stem_items(text)
    check("STEM: punti", items["points"] == {"A", "B", "C", "D", "E"}, items["points"])
    check("STEM: numeri senza l'intestazione", items["numbers"] == {"6", "45"}, items["numbers"])
    check("STEM: tipo e geometria", items["type"] == "geometric diagram" and {"square", "point"} <= items["geometry"],
          items)


def test_token_category():
    cases = {" 45": "numero", " ABC": "maiuscole", " square": "forma", " red": "colore", "\\frac": "latex",
             "**": "formato", ".": "punteggiatura", " the": "funzionale", " mushroom": "contenuto", "\n\n": "spazi",
             "<|endoftext|>": "eos"}
    got = {k: rep.token_category(k) for k in cases}
    check("tipi di token", got == cases, {k: v for k, v in got.items() if cases[k] != v})


if __name__ == "__main__":
    test_visible_tokens()
    test_align()
    test_loop()
    test_sub_kind()
    test_mavis()
    test_stem()
    test_token_category()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
