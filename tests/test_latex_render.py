"""Scelta dell'etichetta confrontando l'immagine con le candidate disegnate da matplotlib (limite_vl.latex_render).
Verifica il meccanismo su immagini disegnate dallo stesso matplotlib; sulle immagini di UniMER l'accuratezza la misurano
i controlli di scripts/unimer_check.py. Richiede matplotlib, numpy e PIL (job SLURM)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.latex_render import best_label, render, similarity  # noqa: E402

PAIRS = [
    (r"\bar{\lambda}", r"\lambda"),
    (r"x^{2}", r"x_{2}"),
    (r"a+b", r"a-b"),
    (r"\alpha\beta", r"\alpha\gamma"),
    (r"\frac{1}{2}", r"\frac{1}{3}"),
]


def check(name, cond, detail=None):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or detail is None else f"  -> {detail}"))
    return cond


def main() -> int:
    ok = True
    img = render(r"\frac{a}{b}")
    ok &= check("disegno di una formula", img is not None and img.min() < 0.5 < img.max())
    ok &= check("somiglianza con se stessa", img is not None and similarity(img, img) > 0.99)
    ok &= check("comando non supportato: None", render(r"\notacommand{x}") is None)
    for true, wrong in PAIRS:
        image = render(true)
        pick, scores = best_label(image, [wrong, true])
        ok &= check(f"sceglie {true} contro {wrong}", pick == true, scores)
    pick, scores = best_label(render("x"), ["x", "{x}"])
    ok &= check("etichette diverse con lo stesso disegno: nessun vincitore netto", pick is None, scores)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
