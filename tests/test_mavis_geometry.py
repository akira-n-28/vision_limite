"""Coerenza geometrica delle costruzioni MAVIS (scripts/mavis_geometry_check.py): campioni veri senza violazioni e
costruzioni alterate trovate. Solo libreria standard."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from limite_vl.f1_mavis import mavis_truth  # noqa: E402
from mavis_geometry_check import Checker, value  # noqa: E402

# campione reale d2-35352 (sample.jsonl del job 30033) e d0-16855 (job 29798)
GEN_RIGHT = ("Firstly, draw a triangle with a right angle abc. secondly, aligned with edge cb, sketch a rectangle cbde. "
             "thirdly, joined to edge ed, design a rectangle edfg, inscribe a semicircle using the base fg as the diameter "
             "and then remove the base fg to finalize the shape. the size of angle acb is 30 degrees. angle bce is measured "
             "at 90 degrees. the angle deg measures 90°. the measurement of side ab is 9. side cb stretches for 9*√(3) units. "
             "side ce has a length of 15*√(3) units. side ed measures 9*√(3) units in length. side eg stretches for 28 units.")
CAP_RIGHT = ("ABC is a right-angled triangle. CBDE takes the form of a rectangle. EDFG is a rectangle, and the side FG extends "
             "into an internal semi-circle. The size of angle ACB is 30 degrees. Angle BCE is measured at 90 degrees. The angle "
             "DEG measures 90°. The measurement of side AB is 9. Side CB stretches for 9*√(3) units. Side CE has a length of "
             "15*√(3) units. Side ED measures 9*√(3) units in length. Side EG stretches for 28 units. ")
GEN_ISO = ("Firstly, draw a sector abc. secondly, bordering edge cb, draw a rectangle cbde. thirdly, attached to edge ed, "
           "create a triangle edf with two equal-length sides ed and df. angle cba has a size of 120°. angle bce is 90 degrees. "
           "angle fde is 30°. the segment ab has a length of 6. the distance between points c and b is 6. the length of side "
           "bd is 5*√(3). the distance between points e and d is 6.")
CAP_ISO = ("ABC is identified as a sector. And then, CBDE takes the form of a rectangle. Furthermore, EDF is an isosceles "
           "triangle, with ED equalling DF. Angle CBA has a size of 120°. Angle BCE is 90 degrees. Angle FDE is 30°. The segment "
           "AB has a length of 6. The distance between points C and B is 6. The length of side BD is 5*√(3). The distance "
           "between points E and D is 6. ")


def check(name, cond, detail=None):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or detail is None else f"  -> {detail}"))
    return cond


def results(cap, gen):
    truth, why = mavis_truth(cap, gen)
    assert truth is not None, why
    return Checker(truth, gen).run()


def main() -> int:
    ok = True
    ok &= check("valori", [value(v) for v in ("42", "3/2", "9\\sqrt{3}", "\\sqrt{2}", "10\\sqrt{3}/3", "x")] ==
                [42.0, 1.5, 9 * 3 ** 0.5, 2 ** 0.5, 10 * 3 ** 0.5 / 3, None])
    r = results(CAP_RIGHT, GEN_RIGHT)
    ok &= check("triangolo rettangolo vero: tangente controllata e coerente",
                any(c == "rettangolo_tangente" and good for c, good, _ in r) and all(good for _, good, _ in r), r)
    r = results(CAP_ISO, GEN_ISO)
    ok &= check("settore e rettangolo veri: raggi e lati opposti coerenti",
                {c for c, _, _ in r} >= {"settore_raggi", "rectangle_lati_opposti"} and all(good for _, good, _ in r), r)
    bad = results(CAP_RIGHT.replace("side AB is 9", "side AB is 10"), GEN_RIGHT.replace("side ab is 9", "side ab is 10"))
    ok &= check("mutazione: cateto cambiato viola la tangente", any(c == "rettangolo_tangente" and not g for c, g, _ in bad), bad)
    bad = results(CAP_ISO.replace("points C and B is 6", "points C and B is 7"),
                  GEN_ISO.replace("points c and b is 6", "points c and b is 7"))
    ok &= check("mutazione: raggio cambiato viola il settore", any(c == "settore_raggi" and not g for c, g, _ in bad), bad)
    bad = results(CAP_ISO.replace("points E and D is 6", "points E and D is 5"),
                  GEN_ISO.replace("points e and d is 6", "points e and d is 5"))
    ok &= check("mutazione: lato opposto cambiato viola il rettangolo",
                any(c == "rectangle_lati_opposti" and not g for c, g, _ in bad), bad)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
