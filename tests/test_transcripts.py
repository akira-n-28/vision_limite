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
# forme equivalenti viste negli esempi del job 30028 (falsi allarmi della prima versione), con la trascrizione vera
EQUIVALENT = [
    ("orario 16:19 contro 1619", "Across the top, the screen shows the time \"16:19\" and the network status \"4G\". The clock "
     "reads 16:19.", ["There is text across the top in white that reads 1619 for the time, 4G."]),
    ("cifre dette una a una", "The tail shows the serial number \"29000.\" and the number 29000 again.",
     ["there's a number it says two nine zero zero zero"]),
    ("lettere sillabate", "A receipt from the online retailer \"FRUUGO,\" on a white background.",
     ["the background is all white, at the top it says F R U U G O underneath that"]),
    ("parola storpiata dalla trascrizione", "The text indicates it is from \"Peggy's Supermarkt in Messingladde.\"",
     ["it says Peggy's and then there's some smaller lettering that says supermarket in Messingladde"]),
    ("colore con suffisso", "It features a gray plastic handle and a pink interior.",
     ["It's got kind of a grayish plastic handle there, showing the pinkness of the steak"]),
    ("numeri degli elenchi", "Four screwdrivers are shown:\n1. **First:** small.\n2. **Second:** larger.\n3. Third one.",
     ["In this photo we see four small hand tools"]),
    ("virgolette accoppiate male", "The label reads \"OK\" followed by \"Close\" on a 5\" screen.",
     ["a button that says OK and another that says close on a five inch screen"]),
    # terza versione (esempi del job 30031)
    ("creamy e golden hour non sono colori", "A creamy sauce is lit by the golden hour sun.",
     ["there is some sauce and it's outside in the evening sun"]),
    ("orario detto a parole", "The clock at the top shows 8:27.", ["at the top the clock says eight twenty seven"]),
    ("citazioni numeriche corte", "It shows the time \"8:43\" and the number \"7\" in a box.",
     ["it's 843 and there is a seven in a box"]),
    ("zero letto anche come parola", "White text saying \"IsNotZero\" on the right.",
     ["In white text it says is not zero on the right"]),
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
    for name, cap, tr in EQUIVALENT:
        r = compare(cap, tr)
        ok &= check(f"forma equivalente senza segnalazioni: {name}", not (r["numbers"] or r["colors"] or r["quotes"]), r)
    r = compare("The movie was released in 2011 and the clock shows 9:41.",
                ["a soccer punch Like 2010 11 12 something like that"])
    ok &= check("anno e orario aggiunti dal riscrittore", r["numbers"] == ["9:41", "2011"], r)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
