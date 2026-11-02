#!/usr/bin/env python3
"""Test CPU della preparazione della verifica delle golden answer dei sintetici v4 (scripts/synth_gold_sample.py), su
record e layout piccoli costruiti qui. Solo in un job (slurm/tests_cpu.sbatch o slurm/synth_gold_sample.sbatch).

Coprono: strati e unità del campione (QA costruite dal loader per train e dev, tutte le QA del test, coppie intere),
ordine per rango riproducibile e annidato, esclusione dei record non ammessi; limite esatto con zero errori uguale a
quello di S2 §2.8; margini calcolati dal layout salvato (testo più basso, riquadri vicini, vertice estremo,
coordinate senza etichetta, intersezioni e grafici vicini); scelta dei k casi con la misura più piccola."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import synth_gold_sample as gs  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.f1c import pick_questions  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def record(key, famiglia="coord", insieme="train", split="train", stato="ammesso", n_qa=6, coppia=None):
    d = mf.Digests()
    return {"record": key, "record_id": mf.record_id(d, "synth", "v4@" + "0" * 40, key), "famiglia": famiglia,
            "insieme": insieme, "split": split, "stato": stato, "coppia": coppia,
            "qa": [{"tipo": "exists", "sha256": mf.sha256_text(f"{key}/{i}")} for i in range(n_qa)]}


def test_populations():
    records = [record(f"synth-{i:07d}") for i in range(6)]
    records += [record("synth-0000006", stato="quarantena"), record("synth-0000007", split="dev"),
                record("synth-0000008", famiglia="polygon"), record("synth-0000009", split="test", stato="escluso")]
    records += [record("synth-test-00000", insieme="test", split="test", n_qa=4)]
    records += [record("synth-pair-00000-a", insieme="coppie", split="test", n_qa=1, coppia="synth-pair-00000"),
                record("synth-pair-00000-b", insieme="coppie", split="test", n_qa=1, coppia="synth-pair-00000"),
                record("synth-pair-00001-a", insieme="coppie", split="test", n_qa=1, coppia="synth-pair-00001"),
                record("synth-pair-00001-b", insieme="coppie", split="test", n_qa=1, coppia="synth-pair-00001",
                       stato="quarantena")]
    pops = gs.populations(records, seed=0)
    expected_coord = sum(len(pick_questions(r["record"], list(range(6)))) for r in records[:6] + [records[7]])
    check("campione: QA costruite dal loader per le scene ammesse di train e dev",
          len(pops["coord"]) == expected_coord and all(u["records"][0]["stato"] == "ammesso" for u in pops["coord"]),
          f"{len(pops['coord'])} contro {expected_coord}")
    check("campione: escluse le scene in quarantena o escluse", all(
        u["records"][0]["record"] not in ("synth-0000006", "synth-0000009") for s in pops.values() for u in s))
    check("campione: il test con tutte le sue QA", len(pops["test"]) == 4)
    check("campione: una coppia è un'unità, solo con entrambe le varianti ammesse",
          len(pops["coppie"]) == 1 and len(pops["coppie"][0]["records"]) == 2)
    check("campione: strati per famiglia", len(pops["polygon"]) == len(pick_questions("synth-0000008", list(range(6)))))
    again = gs.populations(list(reversed(records)), seed=0)
    check("campione: ordine per rango riproducibile e indipendente dall'ordine dei record",
          [u["rango"] for u in again["coord"]] == [u["rango"] for u in pops["coord"]]
          and [u["rango"] for u in pops["coord"]] == sorted(u["rango"] for u in pops["coord"]))
    other = gs.populations(records, seed=1)
    check("campione: con un altro seme l'ordine cambia",
          [u["rango"] for u in other["coord"]] != [u["rango"] for u in pops["coord"]])
    small, large = pops["coord"][:3], pops["coord"][:5]
    check("campione: un n più piccolo è il prefisso di quello più grande", large[:3] == small)


def test_bound():
    for N, n, want in ((45061, 75, 0.039), (4038, 50, 0.058), (300, 30, 0.090)):
        got = gs.upper_zero(N, n)
        check(f"limite con zero errori su {n} di {N}: circa {want:.1%} come S2 §2.8", abs(got - want) < 0.0015,
              f"{got:.4f}")
    check("limite: più estrazioni danno un limite più basso", gs.upper_zero(1000, 100) < gs.upper_zero(1000, 50))


def text(gid, bbox, value="A"):
    return {"gid": gid, "kind": "text", "text": value, "bbox": bbox}


def test_margins():
    layout = {"items": [text("label:A", [10, 10, 20, 22]), text("label:B", [19.5, 15, 30, 27]),
                        text("tick_label:x:2", [50, 50, 56, 61], "2"),
                        {"gid": "point:A", "kind": "marker", "bbox": [12, 12, 16, 16]}]}
    low, close = gs.text_margins(layout)
    check("margini: altezza del testo più basso", abs(low - 11.0) < 1e-9, str(low))
    check("margini: avvicinamento fra riquadri (minore delle due sovrapposizioni)", abs(close - 0.5) < 1e-9, str(close))
    check("coordinate: senza l'etichetta di y si contano le tacche", gs.coords_from_ticks(layout, "(2, 5)")
          and not gs.coords_from_ticks({"items": [text("tick_label:x:2", [0, 0, 1, 1]),
                                                  text("tick_label:y:5", [0, 0, 1, 1])]}, "(2, 5)"))
    # le etichette cadono sui vertici (0, 0), (4, 0) e (2, 3), nelle unità del disegno (trasformazione identica)
    poly = {"items": [{"gid": "polygon", "kind": "patch", "path": [[0, 0], [4, 0], [2, 3], [0, 0]]},
                      text("vertex_label:A", [0, 0, 0, 0], "A"), text("vertex_label:B", [4, 0, 4, 0], "B"),
                      text("vertex_label:C", [2, 3, 2, 3], "C")],
            "transform": {"x0": 0, "y0": 0, "ux": 1, "uy": 1}}
    m = gs.extreme_margins(poly)
    check("margini: vertice più alto staccato di 3/4 del lato del riquadro", abs(m.get("highest", 0) - 0.75) < 1e-9,
          str(m))
    check("margini: vertici più bassi alla pari, stacco zero", abs(m.get("lowest", 1)) < 1e-9, str(m))
    curve = {"x_range": [-7, 7], "clip_y": 7, "visible": [[-7, 7]], "samples": 701}
    meta = {"spec": {"curves": [{"name": "f", "kind": "line", "a": 1, "b": 0}, {"name": "g", "kind": "line", "a": -1, "b": 2}]},
            "layout": {"items": [{"gid": "curve:f", "kind": "curve", **curve}, {"gid": "curve:g", "kind": "curve", **curve}]}}
    edge, gap = gs.intersection_margins(meta)
    check("intersezioni: distanza dal bordo del tratto disegnato", abs(edge - 6) < 1e-9 and gap is None, f"{edge}, {gap}")
    meta["spec"]["curves"][1] = {"name": "g", "kind": "line", "a": 1, "b": 0.5}
    edge, gap = gs.intersection_margins(meta)
    check("intersezioni: rette parallele, distanza minima fra i grafici", edge is None and abs(gap - 0.5) < 1e-9,
          f"{edge}, {gap}")


def test_best():
    best = gs.Best(2)
    for value, tie, item in ((3.0, "c", "x3"), (1.0, "b", "x1b"), (2.0, "a", "x2"), (1.0, "a", "x1a")):
        best.add(value, tie, item)
    check("casi mirati: i k con la misura più piccola, a parità di misura il rango più piccolo",
          best.items() == ["x1a", "x1b"], str(best.items()))


if __name__ == "__main__":
    test_populations()
    test_bound()
    test_margins()
    test_best()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
