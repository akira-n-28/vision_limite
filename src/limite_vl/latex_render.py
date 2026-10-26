"""Scelta dell'etichetta di una formula confrontando l'immagine con le etichette candidate disegnate da matplotlib
(mathtext, font Computer Modern). Serve nei gruppi di UniMER in cui la stessa immagine ha formule diverse: al più una
etichetta corrisponde ai pixel.

Funziona per formule brevi che mathtext sa disegnare; il renderizzatore di UniMER è un altro, quindi l'accuratezza va
misurata ogni volta su controlli con verità nota (scripts/unimer_check.py: etichetta vera contro etichetta alterata) e
la scelta si usa solo se i controlli la confermano. Senza un vincitore netto la risposta è None.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

MIN_SCORE = 0.75
MIN_MARGIN = 0.05


def render(label: str, dpi: int = 300) -> np.ndarray | None:
    """Etichetta disegnata da mathtext (font cm), in grigi 0-1 con il bianco a 1; None se mathtext non la sa disegnare."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import mathtext, rcParams

    rcParams["mathtext.fontset"] = "cm"
    buf = io.BytesIO()
    try:
        mathtext.math_to_image(f"${label}$", buf, dpi=dpi, format="png")
    except Exception:  # noqa: BLE001 - comando non supportato o formula non valida per mathtext
        return None
    buf.seek(0)
    with Image.open(buf) as im:
        return to_gray(im)


def to_gray(im: Image.Image) -> np.ndarray:
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def ink_crop(gray: np.ndarray, threshold: float = 0.5) -> np.ndarray | None:
    ys, xs = np.nonzero(gray < threshold)
    if len(ys) == 0:
        return None
    return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def similarity(image: np.ndarray, candidate: np.ndarray) -> float:
    """Correlazione dei pixel dopo il ritaglio sull'inchiostro e il ridimensionamento della candidata alla stessa
    dimensione; -1 se una delle due è vuota o i rapporti d'aspetto differiscono di oltre 1,5 volte."""
    a, b = ink_crop(image), ink_crop(candidate)
    if a is None or b is None:
        return -1.0
    ra, rb = a.shape[1] / a.shape[0], b.shape[1] / b.shape[0]
    if max(ra, rb) / min(ra, rb) > 1.5:
        return -1.0
    b = np.asarray(Image.fromarray((b * 255).astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.BILINEAR),
                   dtype=np.float32) / 255.0
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den else -1.0


def best_label(image: np.ndarray, labels: list[str]) -> tuple[str | None, dict[str, float]]:
    """(etichetta scelta o None, punteggi): vince se ha almeno MIN_SCORE e supera la seconda di MIN_MARGIN."""
    scores = {}
    for label in labels:
        r = render(label)
        scores[label] = similarity(image, r) if r is not None else -1.0
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < MIN_SCORE:
        return None, scores
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < MIN_MARGIN:
        return None, scores
    return ranked[0][0], scores
