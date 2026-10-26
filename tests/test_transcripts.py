"""Descrizioni di PixMo-Cap contro le trascrizioni (limite_vl.transcripts): formulazioni equivalenti senza segnalazioni
(falsi allarmi) e aggiunte note trovate (mutazioni). Solo libreria standard."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.transcripts import compare, numbers  # noqa: E402

TRANSCRIPT = ["okay so this is a street at night, there are three cars parked, one of them is grey and the other two are "
              "red, and there's a sign that says open twenty four hours, and maybe twenty-five people walking"]
SAME = [
    "The image shows a street at night with 3 parked cars: one gray and two red. A sign reads \"OPEN 24 HOURS\". "
    "About 25 people are walking.",
    "Three cars are parked on a street at night. One car is grey, the others are red. Twenty-five pedestrians walk by.",
]
ADDED = [
    ("numero inventato", "There are 3 parked cars and 7 bicycles on the street at night.", "numbers"),
    ("colore inventato", "Three cars are parked: one gray, two red, and a blue truck behind them.", "colors"),
    ("testo inventato", "A sign reads \"FREE PARKING TODAY\" above three cars.", "quotes"),
    ("conteggio cambiato", "Four cars are parked on the street at night.", "numbers"),
]


def check(name, cond, detail=None):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or detail is None else f"  -> {detail}"))
    return cond


def main() -> int:
    ok = True
    ok &= check("numeri: cifre, parole, decine con unità", numbers("3 dogs, twenty four cats, twenty-five birds, 1,000 "
                                                                     "ants") == {3.0, 24.0, 25.0, 1000.0},
                numbers("3 dogs, twenty four cats, twenty-five birds, 1,000 ants"))
    ok &= check("numeri: 'one of the' non è un conteggio", numbers("one of the cars") == set(), numbers("one of the cars"))
    for i, cap in enumerate(SAME):
        r = compare(cap, TRANSCRIPT)
        ok &= check(f"formulazione equivalente {i} senza segnalazioni", not (r["numbers"] or r["colors"] or r["quotes"]), r)
    for name, cap, kind in ADDED:
        r = compare(cap, TRANSCRIPT)
        ok &= check(f"aggiunta trovata: {name}", bool(r[kind]), r)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
