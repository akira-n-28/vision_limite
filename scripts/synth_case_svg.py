#!/usr/bin/env python3
"""Schema SVG di una scena shapes dei sintetici, ricostruito dalla specifica senza matplotlib (solo libreria standard):
serve a documentare un caso, per esempio una figura tagliata dagli assi. Non è il PNG del renderer, che si ottiene in
un job con scripts/synth_semantics_check.py --cases CHIAVE:shapes:VERSIONE.

Lo schema riproduce la geometria di synth._draw_shapes e di synth.render: immagine larghezza x altezza, assi quadrati
di lato min(larghezza, altezza) al centro (set_aspect equal, anchor C) che mostrano [0, 9]^2, tratto di lw punti a 100
dpi. Il disegno è tagliato dall'area degli assi come in matplotlib. Il contorno completo delle figure che escono è
ripassato in rosso tratteggiato, e il bordo dell'area degli assi in grigio, che nell'immagine vera non compare.

    python3 -I -S scripts/synth_case_svg.py CHIAVE VERSIONE OUT.svg
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]


def load_synth():
    """limite_vl.synth da solo (a livello di modulo usa solo la libreria standard)."""
    spec = importlib.util.spec_from_file_location("synth", ROOT / "src" / "limite_vl" / "synth.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def svg(synth, spec: dict, caption: str) -> str:
    W, H = spec["width"], spec["height"]
    side = min(W, H)
    x0, y0 = (W - side) / 2, (H - side) / 2
    unit = side / synth.SHAPES_EXTENT

    def px(x: float, y: float) -> tuple[float, float]:
        return x0 + x * unit, y0 + side - y * unit

    stroke = spec["lw"] * 100 / 72
    font = spec["font"] * 100 / 72
    shapes, outlines, labels = [], [], []
    for s in spec["shapes"]:
        x, y, r = s["cx"], s["cy"], s["size"]
        if s["type"] == "circle":
            cx, cy = px(x, y)
            el = f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r * unit:.2f}"'
        elif s["type"] in ("square", "rectangle"):
            w, h = (2 * r, 2 * r) if s["type"] == "square" else (2.4 * r, 1.2 * r)
            left, top = px(x - w / 2, y + h / 2)
            el = f'<rect x="{left:.2f}" y="{top:.2f}" width="{w * unit:.2f}" height="{h * unit:.2f}"'
        else:
            pts = " ".join("{:.2f},{:.2f}".format(*px(a, b)) for a, b in
                           ((x - r, y - 0.8 * r), (x + r, y - 0.8 * r), (x, y + r)))
            el = f'<polygon points="{pts}"'
        shapes.append(el + "/>")
        b = synth._shape_box(s)
        if b[0] < 0 or b[1] > synth.SHAPES_EXTENT or b[2] < 0 or b[3] > synth.SHAPES_EXTENT:
            outlines.append(el + ' stroke="#d62728" stroke-dasharray="6 4" stroke-width="1.5"/>')
        lx, ly = px(x, y - 0.1 * r)
        labels.append(f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle" dominant-baseline="central" '
                      f'font-size="{font:.1f}" font-family="{escape(spec["font_family"])}">{escape(s["label"])}</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H + 44}" viewBox="0 0 {W} {H + 44}">
  <defs><clipPath id="axes"><rect x="{x0:.2f}" y="{y0:.2f}" width="{side:.2f}" height="{side:.2f}"/></clipPath></defs>
  <rect x="0" y="0" width="{W}" height="{H}" fill="white" stroke="#bbbbbb"/>
  <rect x="{x0:.2f}" y="{y0:.2f}" width="{side:.2f}" height="{side:.2f}" fill="none" stroke="#999999" stroke-dasharray="3 3"/>
  <g fill="none" stroke="black" stroke-width="{stroke:.2f}" clip-path="url(#axes)">
    {chr(10).join("    " + e for e in shapes).strip()}
  </g>
  <g fill="none">
    {chr(10).join("    " + e for e in outlines).strip()}
  </g>
  {chr(10).join("  " + e for e in labels).strip()}
  <text x="6" y="{H + 18}" font-size="13" font-family="DejaVu Sans" fill="#333333">{escape(caption)}</text>
  <text x="6" y="{H + 36}" font-size="11" font-family="DejaVu Sans" fill="#777777">schema dalla specifica, non il PNG del renderer; grigio: area degli assi; rosso: contorno tagliato</text>
</svg>
"""


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("uso: synth_case_svg.py CHIAVE VERSIONE OUT.svg")
    key, version, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    if out.exists():
        sys.exit(f"{out} esiste già")
    synth = load_synth()
    spec = synth.make_spec(key, "shapes", version)
    fits = synth._shapes_fit(spec)
    worst = min(min(b[0], b[2], synth.SHAPES_EXTENT - b[1], synth.SHAPES_EXTENT - b[3])
                for b in (synth._shape_box(s) for s in spec["shapes"]))
    caption = (f"{key}, generatore v{version}: " + ("tutte le figure dentro gli assi col tratto" if fits else
               f"figura fuori dagli assi di {-worst:.3f} unità") + f" (margine del tratto {synth._stroke_margin(spec):.3f})")
    out.write_text(svg(synth, spec, caption))
    print(out, caption)


if __name__ == "__main__":
    main()
