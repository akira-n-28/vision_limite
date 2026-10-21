"""Riscrittura esatta delle didascalie MAVIS per il ponte di F2 (limite_vl.mavis)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.mavis import arc_ask, bridge_caption, f2_problem, rewrite_sentence, structure_question  # noqa: E402

GEN_OUT = ("Draft a rectangle abcd, extending side cd to create an external equilateral triangle, with side cd then "
           "removed. angle bad equals 90 degrees.")
GEN_IN = ("Draft a modified rectangle abcd, where side cd is expanded inwards to shape an equilateral triangle, with "
          "side cd being removed afterwards. the angle bad measures 90°.")
GEN_MULTI = ("Firstly, render a sector abc. secondly, linking to edge cb, draw a rectangle cbde, extending side de to "
             "create an external equilateral triangle, with side de then removed. thirdly, meeting at edge fd, draft a "
             "right triangle fdg. fourthly, joined to edge hg, illustrate a rectangle hgij, creating a semicircle inside "
             "with ij as the diameter, and then eliminate the side ij. angle bad equals 90 degrees.")


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    return cond


def main() -> int:
    ok = True
    s, what = rewrite_sentence("ABCED is a rectangle, although its side CD extends into an equilateral triangle beyond the rectangle.", GEN_OUT)
    ok &= check("triangolo fuori", what == "rewritten" and s == "ABCED is a pentagon: the rectangle ABCD with side CD removed and "
                "replaced by an equilateral triangle CED drawn outside the rectangle.")
    s, what = rewrite_sentence("ABCED is a rectangle, with the side CD extended into an equilateral triangle inside the rectangle.", GEN_IN)
    ok &= check("triangolo dentro", what == "rewritten" and "triangle CED drawn inside" in s)
    s, what = rewrite_sentence("ABCED is a rectangle, where the side CD is prolonged into an equilateral triangle.", GEN_IN)
    ok &= check("direzione da diagram_generation (dentro)", what == "rewritten" and "drawn inside" in s)
    s, what = rewrite_sentence("CBDFE is a rectangle, with the side DE extending into an equilateral triangle.", GEN_MULTI)
    ok &= check("figura successiva, direzione dalla sua parte", what == "rewritten" and s.startswith("CBDFE is a pentagon: the rectangle CBDE")
                and "triangle DFE drawn outside" in s)
    s, what = rewrite_sentence("HGIJ is a rectangle, featuring an extension of the side IJ into a semi-circle.", GEN_MULTI)
    ok &= check("semicerchio, direzione da diagram_generation", what == "rewritten" and s == "HGIJ is the rectangle HGIJ with side IJ "
                "removed and replaced by a semicircle with diameter IJ drawn inside the rectangle.")
    s, what = rewrite_sentence("Furthermore, ABCED is a rectangle, but includes an equilateral triangle extension on the side CD "
                               "beyond the rectangle.", GEN_OUT)
    ok &= check("forma prima del lato, con prefisso", what == "rewritten" and s.startswith("Furthermore, ABCED is a pentagon"))
    s, what = rewrite_sentence("On top of that, ABCD is a rectangle, however, the side CD extends into a semi-circular shape inside.", GEN_IN)
    ok &= check("semicircolare dentro", what == "rewritten" and s.startswith("On top of that, ABCD is the rectangle ABCD") and "inside" in s)
    s, what = rewrite_sentence("What's more, DEFHG is a rectangle, but includes an equilateral triangle extension on the side FG.",
                               "Firstly, draw a sector abc. secondly, adjacent to edge cb, trace a rectangle defg modified by "
                               "extending the side fg to create an external equilateral triangle, followed by the removal of fg.")
    ok &= check("prefisso con apostrofo", what == "rewritten" and s.startswith("What's more, DEFHG is a pentagon: the rectangle DEFG")
                and "triangle FHG drawn outside" in s)
    s, what = rewrite_sentence("FEGH is a rectangle, featuring an extension of the side GH into a semi-circle.",
                               "Firstly, draw a sector abc. fourthly, bordering edge fe, outline a rectangle fegh, inscribe a "
                               "semicircle using the base gh as the diameter and then remove the base gh to finalize the shape.")
    ok &= check("semicerchio inscritto = dentro", what == "rewritten" and s.endswith("diameter GH drawn inside the rectangle."))
    s, what = rewrite_sentence("ABCD is a rectangle.", GEN_OUT)
    ok &= check("rettangolo semplice invariato", what == "same" and s == "ABCD is a rectangle.")
    s, what = rewrite_sentence("Side AB extends for 5 units.", GEN_OUT)
    ok &= check("misura invariata", what == "same")
    s, what = rewrite_sentence("ABCED is a rectangle, with the side CD extending into an equilateral triangle.", "Draw a rectangle abcd.")
    ok &= check("senza direzione: fallisce e lascia la frase", what == "failed" and s.startswith("ABCED is a rectangle"))
    cap, counts = bridge_caption("ABCED is a rectangle, although its side CD extends into an equilateral triangle beyond the rectangle. "
                                 "Angle BAD equals 90 degrees. The distance between points A and B is 30. ", GEN_OUT)
    ok &= check("didascalia intera", counts == {"rewritten": 1, "failed": 0} and cap.endswith("between points A and B is 30.")
                and "is a pentagon" in cap)
    meta = {"id": "d0-41141", "depth": 0, "choices": ["1", "2", "225*√(3) + 1260", "4"], "answer_index": 2,
            "text_en": {"caption": "ABCED is a rectangle, although its side CD extends into an equilateral triangle beyond the "
                                   "rectangle. Side AB is 30. ", "diagram_generation": GEN_OUT,
                        "vision_dominant_question": "Side CD materializes as an equilateral triangle. What is the area of ABCED?",
                        "CoT_reasoning": "..."}}
    p = f2_problem(meta)
    ok &= check("problema F2", p is not None and p["answer_letter"] == "C" and p["answer_text"] == "225*√(3) + 1260"
                and p["question"].endswith("C. 225*√(3) + 1260\nD. 4") and "pentagon" in p["caption"] and p["rewritten"] == 1)
    ok &= check("domanda F2 con la struttura e senza misure", p is not None and p["question"].startswith(
        "ABCED is a pentagon: the rectangle ABCD with side CD removed and replaced by an equilateral triangle CED drawn outside "
        "the rectangle. What is the area of ABCED?\nA. 1") and "30" not in p["question"] and "materializes" not in p["question"])
    q, qc = structure_question("Angle BAD is 90°. Besides, CBDFE is a pentagon: the rectangle CBDE with side DE removed. "
                               "Side AB is 5. Square ECFG has a circle inscribed.", "Side DE extends outward. Find the area of CBDFE.")
    ok &= check("connettivo iniziale tolto, misure tolte", q == "CBDFE is a pentagon: the rectangle CBDE with side DE removed. "
                "Square ECFG has a circle inscribed. Find the area of CBDFE." and qc == {"structure": 2, "ask_digits": False, "connector": True, "arc": False})
    q, qc = structure_question("ABCD is a square, and CBE is a sector. The radius CB is 5.", "What is the perimeter of CBE?")
    ok &= check("nome di punti non preso per connettivo", q.startswith("ABCD is a square, and CBE") and not qc["connector"])
    ok &= check("arco del semicerchio nominato", arc_ask("In shape GHIJ, what is the length of IJ?", ["IJ"])
                == ("In shape GHIJ, what is the length of the semicircular arc IJ?", True)
                and arc_ask("How long is JI in shape HGIJ?", ["IJ"])[0] == "How long is the semicircular arc JI in shape HGIJ?")
    ok &= check("arco già nominato o lato non chiesto: invariato", arc_ask("What is the arc length CD in the shape ABCD?", ["CD"])[1] is False
                and arc_ask("What is the total area of the shape ABCD?", ["CD"]) == ("What is the total area of the shape ABCD?", False))
    semi = {"id": "d0-1", "depth": 0, "choices": ["3*π", "6", "6*π", "12"], "answer_index": 0,
            "text_en": {"caption": "ABCD is a rectangle, featuring an extension of the side CD into a semi-circle outside. Side AB is 6. ",
                        "diagram_generation": "Draw a rectangle abcd, creating a semicircle outside with cd as the diameter.",
                        "vision_dominant_question": "Side CD extends into a semi-circle. In shape ABCD, how long is CD?"}}
    p = f2_problem(semi)
    ok &= check("problema F2 con l'arco", p is not None and p["question"].startswith("ABCD is the rectangle ABCD with side CD removed and "
                "replaced by a semicircle with diameter CD drawn outside the rectangle. In shape ABCD, how long is the semicircular arc CD?"))
    semi["answer_index"] = 1
    ok &= check("arco chiesto ma risposta senza π: problema scartato", f2_problem(semi) is None)
    meta["text_en"]["vision_dominant_question"] = "Side CD materializes as an equilateral triangle. What is the area of ABCED if AB = 30?"
    ok &= check("richiesta con cifre: problema scartato", f2_problem(meta) is None)
    meta["text_en"]["vision_dominant_question"] = "Side CD materializes as an equilateral triangle. What is the area of ABCED?"
    meta["text_en"]["diagram_generation"] = "Draw a rectangle abcd."
    meta["text_en"]["caption"] = "ABCED is a rectangle, with the side CD extending into an equilateral triangle. "
    ok &= check("problema scartato se la riscrittura fallisce", f2_problem(meta) is None)
    print("tutti ok" if ok else "ci sono errori")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
