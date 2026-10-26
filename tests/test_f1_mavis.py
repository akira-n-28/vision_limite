"""Schema delle uscite di F1-clean (limite_vl.f1_schema) e dati MAVIS (limite_vl.f1_mavis): verità dalle coppie vere,
descrizione canonica, andata e ritorno senza perdite, metriche che vedono ogni alterazione nota. Solo libreria standard."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from limite_vl.f1_mavis import describe, mavis_questions, mavis_truth, questions_example  # noqa: E402
from limite_vl.f1_schema import geometry_scores, latex_value, parse_geometry, render_geometry  # noqa: E402
from test_mavis_gen import MULTI_CAP, MULTI_GEN, REAL  # noqa: E402

D41141 = ("Figure: geometry diagram.\nPoints: A, B, C, D, E.\nShapes: rectangle ABCD, side CD replaced by equilateral "
          "triangle CED outside the rectangle.\nSegments: AB = 30; AD = 42.\nAngles: ∠BAD = 90°.")


def check(name, cond, detail=None):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond or detail is None else f"  -> {detail}"))
    return cond


def perfect(s: dict) -> bool:
    return (s["parsed"] and s["points"]["precision"] == s["points"]["recall"] == 1 and s["shapes"]["precision"] ==
            s["shapes"]["recall"] == 1 and s["shape_kind_ok"] == s["shapes_matched"] == s["shape_modification_ok"]
            and all(s[k]["pairs_ok"] == s[k]["true"] == s[k]["pred"] for k in ("sides", "angles")))


def main() -> int:
    ok = True
    ok &= check("valori in LaTeX", [latex_value(v) for v in ("2√3", "10√3/3", "√2", "3/2", "42")] ==
                ["2\\sqrt{3}", "10\\sqrt{3}/3", "\\sqrt{2}", "3/2", "42"])
    truth, why = mavis_truth(*REAL["d0-41141"])
    ok &= check("verità di d0-41141", why == "ok" and truth == {
        "points": ["A", "B", "C", "D", "E"],
        "shapes": [{"kind": "rectangle", "name": "ABCD", "modification": "equilateral triangle", "removed": "CD",
                    "direction": "outside", "triangle": "CED"}],
        "sides": {"AB": "30", "AD": "42"}, "angles": {"BAD": "90"}}, truth)
    ok &= check("descrizione canonica di d0-41141", render_geometry(truth) == D41141, render_geometry(truth))
    for key, (cap, gen) in REAL.items():
        t, why = mavis_truth(cap, gen)
        ok &= check(f"andata e ritorno senza perdite: {key}", t is not None and perfect(geometry_scores(render_geometry(t), t)),
                    (why, t))
    multi, why = mavis_truth(MULTI_CAP, MULTI_GEN)
    ok &= check("verità a più figure", why == "ok" and multi["points"] == list("ABCDEFG") and
                [s["name"] for s in multi["shapes"]] == ["ABC", "CBDE", "FDG"] and multi["shapes"][1]["triangle"] == "DFE"
                and multi["shapes"][2]["attach"] == "FD" and multi["angles"] == {"ABC": "60", "BCE": "90", "DGF": "45"}
                and multi["sides"] == {"AB": "42", "BC": "42", "CE": "34"}, multi)
    ok &= check("andata e ritorno a più figure", perfect(geometry_scores(render_geometry(multi), multi)))
    qs = {q["q"]: q["answer"] for q in mavis_questions("k", multi)}
    ok &= check("domande: lato in comune col triangolo aggiunto", qs.get("Which side do DFE and FDG share?") == "FD"
                and qs.get("Which side do ABC and CBDE share?") == "CB", qs)
    ok &= check("domande: lato sostituito e direzione", qs.get("Which side of rectangle CBDE is replaced by an equilateral "
                                                               "triangle?") == "DE" and qs.get(
        "Is the equilateral triangle DFE drawn inside or outside rectangle CBDE?") == "outside", qs)
    ok &= check("domande: presenze con sì e no in egual numero",
                sum(a == "yes" for a in qs.values()) == sum(a == "no" for a in qs.values()) == 2, qs)
    ok &= check("domande: punto inesistente", qs.get("Is there a point labeled H?") == "no", qs)
    prompt, answer, picked = questions_example("k", multi)
    ok &= check("esempio di domande nel formato f1c", 1 <= len(picked) <= 5 and answer.count("\n") == len(picked) - 1)
    p, a = describe("k", multi)
    ok &= check("esempio di descrizione", a == render_geometry(multi) and p.endswith("."))

    # metriche: ogni alterazione nota si vede nel punteggio
    t = mavis_truth(*REAL["d0-9323"])[0]
    base = render_geometry(t)
    cases = [
        ("valore cambiato", base.replace("AD = 5", "AD = 6"), lambda s: s["sides"]["wrong_value"] == 1),
        ("punto inventato", base.replace("Points: A, B, C, D.", "Points: A, B, C, D, E."), lambda s: s["points"]["precision"] < 1),
        ("angoli mancanti", base.replace("\nAngles: ∠BAD = 90°.", ""), lambda s: s["angles"]["pairs_ok"] == 0),
        ("segmento inventato", base.replace("AD = 5", "AD = 5; BC = 7"), lambda s: s["sides"]["invented"] == 1),
        ("tipo di figura sbagliato", base.replace("rectangle ABCD", "square ABCD"), lambda s: s["shape_kind_ok"] == 0),
        ("riga illeggibile", base + "\nThe figure is nice.", lambda s: not s["parsed"]),
        ("figura inventata", base.replace("rectangle ABCD.", "rectangle ABCD; square DCEF on side DC."),
         lambda s: s["shapes"]["precision"] < 1),
    ]
    for name, text, cond in cases:
        s = geometry_scores(text, t)
        ok &= check(f"metrica vede: {name}", cond(s), s)
    t41 = mavis_truth(*REAL["d0-41141"])[0]
    s = geometry_scores(D41141.replace("outside the rectangle", "inside the rectangle"), t41)
    ok &= check("metrica vede: direzione invertita", s["shape_modification_ok"] == 0, s)
    ok &= check("parser: ordine diverso e senza punto finale", perfect(geometry_scores(
        "Figure: geometry diagram\nAngles: ∠DAB = 90°\nSegments: AD = 42; AB = 30\nPoints: E, D, C, B, A\nShapes: rectangle "
        "ABCD, side CD replaced by equilateral triangle CED outside the rectangle", t41)))
    ok &= check("parser: righe lette", parse_geometry(D41141)["unparsed"] == [])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
