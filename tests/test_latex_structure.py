"""Struttura del LaTeX per il controllo di UniMER (latex.structure_problems): formule valide senza segnalazioni (falsi
allarmi) e alterazioni note trovate (mutazioni). Solo libreria standard."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.latex import bare_fragment, effective_label, lone_index, structure_problems, unrendered_commands, visible_tokens  # noqa: E402

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
    for f in ("^2", "_{3}", " ^{-1}", "{}_{2}", "{ }^{n}"):
        ok &= check(f"frammento senza base: {f!r}", bare_fragment(f))
    for f in ("x^2", "a_{3}", r"\frac{1}{2}^{n}", "{x}_{2}", r"\left(a\right)^2"):
        ok &= check(f"con base: {f!r}", not bare_fragment(f))
    ok &= check("comando non reso: \\footnote", unrendered_commands(r"x^{1}\footnote{Corresponding author}") == [r"\footnote"])
    ok &= check("comandi non resi: \\label e \\cite", unrendered_commands(r"E=mc^2\label{eq:1}\cite{x}") == [r"\label", r"\cite"])
    ok &= check("nessun comando non reso", unrendered_commands(r"\frac{a}{b}\mathrm{d}x\text{if }x>0") == [])
    # indici isolati: si rietichettano col contenuto; con altro dopo restano come sono (terza versione, job 30029)
    for f, content in (("^{26}", "26"), ("_2", "2"), ("^\\circ", "\\circ"), ("_{4\\sigma}", "4\\sigma"), ("{}^{-1}", "-1"),
                       ("^\\mathrm{16}", "\\mathrm{16}"), ("^{\\mathrm{c}}", "\\mathrm{c}"), ("^{-}", "-")):
        ok &= check(f"indice isolato {f!r} -> {content!r}", lone_index(f) == content, lone_index(f))
    for f in ("^2S_{1/2}", "^{14}C", "x^2", "^{+1.6}_{-0.5}", "^{}", r"\frac{1}{2}"):
        ok &= check(f"non è un indice isolato: {f!r}", lone_index(f) is None, lone_index(f))
    # normalizzazione: stessi pixel, stessi token visibili
    for a, b in ((r"\ensuremath{\delta}", r"\delta"), (r"\protect\phi", r"\phi"), ("h_{}", "h"), (r"\textsf{R}", "R"),
                 (r"x^{ }+1", "x+1")):
        ok &= check(f"stessi token visibili: {a!r} e {b!r}", visible_tokens(a) == visible_tokens(b),
                    (visible_tokens(a), visible_tokens(b)))
    ok &= check("token visibili diversi: \\lambda e \\bar{\\lambda}", visible_tokens(r"\lambda") != visible_tokens(r"\bar{\lambda}"))
    # quarta versione (job 30046): graffe codificate, accenti maiuscoli, indici isolati fra graffe
    for label, expected in (("^Ḋ74Ḍ", ("74", ["graffe_codificate", "indice_isolato"])),
                            (r"\ensuremath Ḋ\mathbf ḊyḌḌ(0)", (r"\ensuremath {\mathbf {y}}(0)", ["graffe_codificate"])),
                            ("ḊaḌ_{2}Ḍ", ("ḊaḌ_{2}Ḍ", [])),
                            ("{_4}", ("4", ["indice_isolato"])),
                            ("x^2", ("x^2", []))):
        ok &= check(f"etichetta effettiva di {label!r}", effective_label(label) == expected, effective_label(label))
    for a, b in ((r"\Vec{V}", r"\vec{V}"), (r"\Bar{\tau}", r"\bar{\tau}"), (r"\Hat{x}", r"\hat{x}")):
        ok &= check(f"accento maiuscolo: {a!r} e {b!r}", visible_tokens(a) == visible_tokens(b))
    ok &= check("graffe esterne che non racchiudono tutto: non è un indice isolato", lone_index("{_4}x{}") is None)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
