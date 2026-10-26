"""Struttura del LaTeX per il controllo di UniMER (latex.structure_problems): formule valide senza segnalazioni (falsi
allarmi) e alterazioni note trovate (mutazioni). Solo libreria standard."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.latex import structure_problems  # noqa: E402

VALID = [
    r"\frac{a+b}{2}\geq\sqrt{ab}",
    r"\left(\frac{1}{2}\right)^{n}",
    r"\left.\frac{\partial f}{\partial x}\right|_{x=0}",
    r"\begin{array}{cc}a&b\\c&d\end{array}",
    r"\begin{array}{c}\begin{array}{c}x\\y\end{array}\\z\end{array}",
    r"\{x\in\mathbb{R}:x>0\}",
    r"f ( x ) = \sum _ { i = 1 } ^ { n } x _ { i }",
    r"\mathrm{d}x\,\mathrm{d}y",
]
MUTATIONS = [
    (r"\frac{a+b}{2\geq\sqrt{ab}", "graffe"),
    (r"\frac{a+b}}{2}", "graffe"),
    (r"\left(\frac{1}{2}^{n}", "left_right"),
    (r"\begin{array}{cc}a&b\\c&d\end{matrix}", "ambiente"),
    (r"\begin{array}{cc}a&b\\c&d", "ambiente"),
    (r"a&b\\c&d\end{array}", "ambiente"),
]


def check(name, cond, detail=None):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or detail is None else f"  -> {detail}"))
    return cond


def main() -> int:
    ok = True
    for f in VALID:
        got = structure_problems(f)
        ok &= check(f"valida: {f}", got == [], got)
    for f, expected in MUTATIONS:
        got = structure_problems(f)
        ok &= check(f"mutazione {expected}: {f}", expected in got, got)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
