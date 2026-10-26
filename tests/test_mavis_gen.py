"""Confronto fra diagram_generation e didascalia di MAVIS (limite_vl.mavis_gen): coppie vere senza segnalazioni (falsi
allarmi) e alterazioni note trovate tutte (test di mutazione). Solo libreria standard."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.mavis_gen import compare, figures, measures  # noqa: E402

# coppie vere dallo shard caption 00003 (job 29798): didascalia e diagram_generation come nei dati
REAL = {
    "d0-41141": ("ABCED is a rectangle, although its side CD extends into an equilateral triangle beyond the rectangle. "
                 "Angle BAD equals 90 degrees. The distance between points A and B is 30. The length measured for side AD is 42. ",
                 "Draft a rectangle abcd, extending side cd to create an external equilateral triangle, with side cd then "
                 "removed. angle bad equals 90 degrees. the distance between points a and b is 30. the length measured for "
                 "side ad is 42."),
    "d0-16855": ("ABCD conforms a parallelogram. The angle CBA is 45 degrees. The distance of side AB is 1. Side BC measures "
                 "√(2) units in length. ",
                 "Create a parallelogram abcd. the angle cba is 45 degrees. the distance of side ab is 1. side bc measures "
                 "√(2) units in length."),
    "d0-9323": ("ABCD takes on a rectangular shape. Angle BAD measures 90 degrees. Side AB is 3 units long. Side AD measures "
                "5 units in length. ",
                "Design a rectangle abcd. angle bad measures 90 degrees. side ab is 3 units long. side ad measures 5 units "
                "in length."),
    "d0-48357": ("ABCD is defined as a rectangle. The measurement of angle BAD is 90°. The length from Point A to Point B is 2. "
                 "Side AD measures 4 units in length. ",
                 "Design a rectangle abcd. the measurement of angle bad is 90°. the length from point a to point b is 2. side "
                 "ad measures 4 units in length."),
    "d0-21715": ("ABCED is a rectangle, with the side CD extended into an equilateral triangle inside the rectangle. The angle "
                 "BAD measures 90°. Side AB is 13 units long. Side AD is measured at 26 units. ",
                 "Draft a modified rectangle abcd, where side cd is expanded inwards to shape an equilateral triangle, with "
                 "side cd being removed afterwards. the angle bad measures 90°. side ab is 13 units long. side ad is measured "
                 "at 26 units."),
    "d0-13996": ("ABCD takes on a rectangular shape. The measure of angle BAD is 90 degrees. The length measured for side AB is "
                 "1. Side AD stretches for 3/2 units. ",
                 "Trace a rectangle abcd. the measure of angle bad is 90 degrees. the length measured for side ab is 1. side ad "
                 "stretches for 3/2 units."),
}
# coppia a più figure scritta con i modelli di frase dei dati (diagram_generation dal sondaggio, didascalia dagli schemi più
# frequenti di mavis_caption_survey)
MULTI_GEN = ("Firstly, render a sector abc. secondly, linking to edge cb, draw a rectangle cbde, extending side de to create an "
             "external equilateral triangle, with side de then removed. thirdly, meeting at edge fd, draft a right triangle "
             "fdg. the size of angle cba is 60 degrees. the angle bce is 90 degrees. side ab measures 42 units in length. the "
             "measurement from c to b is 42. the distance of side ce is 34. the measure of angle fgd is 45°.")
MULTI_CAP = ("ABC takes the form of a sector. CBDFE is a rectangle, although its side DE extends into an equilateral triangle "
             "beyond the rectangle. FDG is a right triangle. The size of angle CBA is 60 degrees. The angle BCE is 90 degrees. "
             "Side AB measures 42 units in length. The measurement from C to B is 42. The distance of side CE is 34. The "
             "measure of angle FGD is 45°.")
SEMI_GEN = ("Firstly, render a sector abc. secondly, linking to edge cb, draw a rectangle cbde, extending side de to create an "
            "external equilateral triangle, with side de then removed. thirdly, meeting at edge fd, draft a right triangle fdg. "
            "fourthly, joined to edge hg, illustrate a rectangle hgij, creating a semicircle inside with ij as the diameter, "
            "and then eliminate the side ij. angle bad equals 90 degrees.")


def check(name, cond, detail=None):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or detail is None else f"  -> {detail}"))
    return cond


def cats(caption, generation):
    return {c for c, _ in compare(caption, generation)["issues"]}


def main() -> int:
    ok = True
    # falsi allarmi: nessuna segnalazione sulle coppie vere
    for key, (cap, gen) in REAL.items():
        r = compare(cap, gen)
        ok &= check(f"coppia vera {key} senza segnalazioni", not r["issues"], r["issues"])
    r = compare(MULTI_CAP, MULTI_GEN)
    ok &= check("coppia a più figure senza segnalazioni", not r["issues"], r["issues"])
    ok &= check("coppia a più figure: tre figure e sei misure", r["figures"] == 3 and r["measures"] == 6, r)

    # lettura delle figure
    figs, unread = figures(SEMI_GEN)
    got = [(f.kind, f.name, f.attach, f.modification, f.removed, f.direction) for f in figs]
    ok &= check("figure di diagram_generation", got == [
        ("sector", "abc", None, None, None, None),
        ("rectangle", "cbde", "cb", "equilateral triangle", "de", "outside"),
        ("right triangle", "fdg", "fd", None, None, None),
        ("rectangle", "hgij", "hg", "semicircle", "ij", "inside")] and not unread, (got, unread))
    figs, _ = figures("Firstly, draw a quadrilateral with opposite sides parallel and equal in length abcd. secondly, aligned "
                      "with edge dc, create a triangle dce with two congruent sides dc and ce. thirdly, attached to edge ec, "
                      "outline a circle and its circumscribing square ecfg. fourthly, adjacent to edge gf, draw a isosceles "
                      "triangle gfh where gf = fh.")
    got = [(f.kind, f.name, f.inscribed_circle) for f in figs]
    ok &= check("parallelogramma, isosceli e quadrato con cerchio", got == [
        ("parallelogram", "abcd", False), ("isosceles triangle", "dce", False), ("square", "ecfg", True),
        ("isosceles triangle", "gfh", False)], got)
    m = measures("The measurement from C to B is 42. The length of side DE equals 2*√(2). Angle FGD is 45°.")
    ok &= check("misure: da-a, radice, angolo", m["side"] == {"bc": "42", "de": "2√2"} and m["angle"] == {"g|df": "45"}
                and not m["unread"], m)

    # test di mutazione: ogni alterazione nota deve finire nella sua categoria
    mutations = [
        ("valore cambiato nella didascalia", MULTI_CAP.replace("Side AB measures 42", "Side AB measures 43"), MULTI_GEN,
         "misura_diversa"),
        ("valore cambiato in diagram_generation", MULTI_CAP, MULTI_GEN.replace("side ce is 34", "side ce is 35"),
         "misura_diversa"),
        ("tipo di figura cambiato", MULTI_CAP.replace("form of a sector", "form of a square"), MULTI_GEN,
         "figura_tipo_diverso"),
        ("direzione invertita", MULTI_CAP.replace("beyond the rectangle", "inside the rectangle"), MULTI_GEN,
         "direzione_opposta"),
        ("misura inventata nella didascalia", MULTI_CAP + " Side FG is 7 units long.", MULTI_GEN, "misura_solo_didascalia"),
        ("misura tolta dalla didascalia", MULTI_CAP.replace(" The distance of side CE is 34.", ""), MULTI_GEN,
         "misura_solo_generazione"),
        ("nome di figura cambiato", MULTI_CAP.replace("FDG is a right triangle", "FDH is a right triangle"), MULTI_GEN,
         "figura_non_nominata"),
        ("punto in più nella didascalia", MULTI_CAP.replace("FDG is a right triangle", "FDH is a right triangle"), MULTI_GEN,
         "punti_solo_didascalia"),
        ("modifica tolta", MULTI_CAP.replace(", although its side DE extends into an equilateral triangle beyond the "
                                             "rectangle", ""), MULTI_GEN, "modifica_assente"),
        ("angolo spostato su un altro vertice", MULTI_CAP.replace("angle BCE is 90", "angle CBE is 90"), MULTI_GEN,
         "misura_solo_didascalia"),
        ("lato tolto diverso", MULTI_CAP.replace("its side DE extends", "its side CE extends"), MULTI_GEN,
         "lato_tolto_diverso"),
    ]
    for name, cap, gen, expected in mutations:
        got = cats(cap, gen)
        ok &= check(f"mutazione trovata: {name}", expected in got, got)
    for key, (cap, gen) in REAL.items():  # anche sulle coppie vere: l'ultimo numero cambiato viene trovato
        got = cats(bump_last_number(cap), gen)
        ok &= check(f"mutazione trovata: ultimo valore cambiato in {key}", "misura_diversa" in got, got)
        got = cats(cap, bump_last_number(gen))
        ok &= check(f"mutazione trovata: ultimo valore cambiato in diagram_generation di {key}", "misura_diversa" in got, got)
    return 0 if ok else 1


def bump_last_number(text: str) -> str:
    m = list(re.finditer(r"\d+", text))[-1]
    return text[:m.start()] + str(int(m.group()) + 1) + text[m.end():]


if __name__ == "__main__":
    sys.exit(main())
