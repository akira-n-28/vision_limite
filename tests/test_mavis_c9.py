#!/usr/bin/env python3
"""Test CPU di C9 per MAVIS-Geometry (src/limite_vl/mavis_c9.py, scripts/mavis_c9.py; S2 §§3, 5). Solo in un job
(slurm/tests_cpu.sbatch o slurm/mavis_c9.sbatch): usa git e processi.

Le aspettative sono scritte a mano per ogni caso, senza chiamare la funzione che genera la risposta:
- valori: esatto, visualizzato previsto e risposta di lettura separati; 3/2 = 1.5, 44/3 ≠ 14.67; l'esatto non si
  ricostruisce dal decimale; confronti che valgono per ogni esatto compatibile con la scrittura;
- shape_type: triangolo rettangolo isoscele, quadrato, rettangolo, rombo e inclusioni; misure insufficienti (verità
  ignota, mai trattata come falsa) o contraddittorie; uguaglianze apparenti dovute all'arrotondamento; settori;
- controllo indipendente: accetta ogni domanda emessa e trova risposte, opzioni, ordine, testo e fatti alterati;
- riproducibilità, posizione della risposta, compatibilità con i manifest correnti (costruzione storica predefinita,
  f1_mavis.py identico a 68daf5a, decisione allegata senza la costruzione per i record storici, loader che rifiuta pool
  e costruzioni non coerenti);
- derivazione di un candidato C9 da un manifest costruito su shard fittizi, con un testo del pannello in comune che
  manda il gruppo in quarantena, verifica che trova le alterazioni, lettura col reader (stream.make_sources) che dà gli
  stessi esempi del genitore;
- valori del candidato contro i JSON originali degli shard (verify_c9): validi con un decimale esatto e uno
  arrotondato; con gli hash interni aggiornati, un esatto alterato, una rappresentazione alterata, un esatto alternativo
  con lo stesso visualizzato e la voce coerente, un elemento mancante, ripetuto o in più fanno fallire la verifica dei
  valori, e i primi quattro solo quella."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import mavis_c9 as job  # noqa: E402  (scripts/mavis_c9.py)
import mavis_manifest as mm  # noqa: E402
import test_mavis_manifest as tm  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl import data as dt  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl import mavis_c9 as c9  # noqa: E402
from limite_vl.f1_mavis import describe, mavis_questions, mavis_truth  # noqa: E402
from limite_vl.mavis_c9 import ShapeFacts  # noqa: E402
from limite_vl.provenance import sha256_file  # noqa: E402
from test_mavis_gen import MULTI_CAP, MULTI_GEN, REAL  # noqa: E402

FAILS: list[str] = []
TRI = ["acute triangle", "right triangle", "obtuse triangle"]
QUAD = ["rectangle", "rhombus", "square"]


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def quad(kind, sides=None, angles=None, name="ABCD"):
    return ShapeFacts(name, kind, "quadrilateral", "", sides=dict(sides or {}), angles=dict(angles or {}))


def tri(kind, sides=None, angles=None, apex=None, name="ABC"):
    return ShapeFacts(name, kind, "triangle", "", sides=dict(sides or {}), angles=dict(angles or {}), apex=apex)


def outcome(f, key="k"):
    return c9.shape_outcome(key, f)


# ---------------------------------------------------------------- valori

def test_values() -> None:
    cases = {"42": ("42", "intero"), "3/2": ("1.5", "decimale_esatto"), "44/3": ("14.67", "decimale_arrotondato"),
             "352/3": ("117.33", "decimale_arrotondato"), "2√3": ("2\\sqrt{3}", "radice"),
             "10√3/3": ("10\\sqrt{3}/3", "radice"), "√2": ("\\sqrt{2}", "radice"), "12.5": ("12.5", "decimale_esatto"),
             "√2.5": ("\\sqrt{2.5}", "non_analizzabile")}
    got = {v: (c9.value_record(v)["visualizzato_previsto"], c9.value_record(v)["rappresentazione"]) for v in cases}
    check("valori: visualizzato previsto e rappresentazione per intero, decimale esatto e arrotondato, radici",
          got == cases, got)
    check("valori: risposta di lettura uguale al visualizzato, con il simbolo dei gradi per gli angoli",
          c9.value_record("44/3")["risposta_lettura"] == "14.67" and c9.value_record("90", "°")["risposta_lettura"]
          == "90°")
    check("valori: √12 si normalizza in 2√3, √8/2 in √2", c9.parse_exact("√12") == c9.Exact(Fraction(2), 3)
          and c9.parse_exact("√8/2") == c9.Exact(Fraction(1), 2))
    three_half, one_five = c9.parse_exact("3/2"), c9.parse_exact("1.5")
    check("valori: 3/2 e 1.5 sono lo stesso numero", three_half == one_five)
    check("valori: 44/3 e 14.67 non sono lo stesso numero", c9.parse_exact("44/3") != c9.parse_exact("14.67"))
    s = c9.shown_of("44/3")
    check("valori: 44/3 stampato 14.67 è un intervallo che contiene sia 44/3 sia 14.67, non un punto esatto",
          not s.exact and s.lo <= Fraction(44, 3) <= s.hi and s.lo <= Fraction(1467, 100) <= s.hi and s.text == "14.67")
    check("valori: 3/2 stampato 1.5 è un punto esatto", c9.shown_of("3/2").exact)
    sh = c9.shown_of
    check("confronti: due 14.67 (44/3) non dimostrano né uguaglianza né differenza",
          not c9.provably_equal(sh("44/3"), sh("44/3")) and not c9.provably_different(sh("44/3"), sh("44/3")))
    check("confronti: 1.5 e 1.5 sono dimostrati uguali, 3 e 3 anche, 2√3 e √12 anche",
          c9.provably_equal(sh("3/2"), sh("3/2")) and c9.provably_equal(sh("3"), sh("3"))
          and c9.provably_equal(sh("2√3"), sh("√12")))
    check("confronti: 14.67 contro 15, contro 14.5 e contro 3√2 (4.24) sono differenze dimostrate",
          c9.provably_different(sh("44/3"), sh("15")) and c9.provably_different(sh("44/3"), sh("29/2"))
          and c9.provably_different(sh("44/3"), sh("3√2")))
    check("confronti: 10/3 (3.33) contro √11 (3.317) è una differenza dimostrata, contro 7/2 anche",
          c9.provably_different(sh("10/3"), sh("√11")) and c9.provably_different(sh("10/3"), sh("7/2")))
    check("confronti: 2√3 (3.464) contro 7/2 (3.5) differenza, contro 2√3 nessuna",
          c9.provably_different(sh("2√3"), sh("7/2")) and not c9.provably_different(sh("2√3"), sh("2√3")))
    # misure di un record vero (d0-13996: AD = 3/2) e di uno costruito con 44/3
    cap, gen = REAL["d0-13996"]
    vals = c9.measure_values(gen)
    check("valori del record d0-13996: AD esatto 3/2, visualizzato 1.5, decimale esatto; angolo BAD 90°",
          vals["side:AD"] == {"esatto": "3/2", "visualizzato_previsto": "1.5", "rappresentazione": "decimale_esatto",
                              "risposta_lettura": "1.5"}
          and vals["angle:BAD"]["risposta_lettura"] == "90°", vals)
    cap44, gen44 = cap.replace("3/2", "44/3"), gen.replace("3/2", "44/3")
    truth = mavis_truth(cap44, gen44)[0]
    hist = {q["q"]: q["answer"] for q in mavis_questions("d0-44", truth)}
    vals = c9.measure_values(gen44)
    pool, _, _ = c9.questions("d0-44", truth, gen44)
    side = next(q for q in pool if q["q"] == "What is the length of AD?")
    check("44/3: il target di lettura storico resta 14.67, l'esatto 44/3 è conservato e non ricostruito dal decimale",
          hist["What is the length of AD?"] == "14.67" and side["answer"] == "14.67" and side["elemento"] == "side:AD"
          and vals["side:AD"]["esatto"] == "44/3" and vals["side:AD"]["rappresentazione"] == "decimale_arrotondato"
          and c9.parse_exact(vals["side:AD"]["esatto"]) != c9.parse_exact(side["answer"]), (side, vals["side:AD"]))
    check("44/3: la descrizione canonica scrive il visualizzato (14.67), come prima",
          "AD = 14.67" in describe("d0-44", truth)[1])


# ---------------------------------------------------------------- shape_type: casi con aspettative scritte a mano

def recovered(o, answer, refuted=None) -> bool:
    return (o["esito"] == "recuperabile" and o["risposta"] == answer and o["domanda"]["answer"] == answer
            and (refuted is None or sorted(o["confutazioni"]) == sorted(refuted)))


def omitted(o, reason) -> bool:
    return o["esito"] == "omessa" and o["motivo"] == reason


def test_shapes() -> list[tuple[ShapeFacts, dict]]:
    emitted = []

    def keep(f, o):
        if o["esito"] == "recuperabile":
            emitted.append((f, o))
        return o

    # triangoli
    f = tri("right triangle", {"AB": "5", "BC": "5"}, {"C": "45"})
    o = keep(f, outcome(f))
    check("rettangolo isoscele (cateti stampati uguali, 45°): right, unica vera fra acute/right/obtuse; isosceles non "
          "è fra le opzioni", recovered(o, "right triangle", ["acute triangle", "obtuse triangle"])
          and sorted(o["opzioni"]) == sorted(TRI) and "isosceles triangle" not in o["opzioni"], o)
    f = tri("isosceles triangle", {"CB": "7", "BD": "7"}, {"B": "90"}, apex="B", name="CBD")
    check("isoscele con angolo al vertice di 90° (rettangolo isoscele): right", recovered(keep(f, outcome(f)),
                                                                                          "right triangle"))
    f = tri("isosceles triangle", {"AB": "4"}, {"B": "90"}, apex="B")
    check("isoscele con un solo lato e 90° stampato: right", recovered(keep(f, outcome(f)), "right triangle"))
    o = outcome(tri("right triangle", {"AB": "7"}, {"C": "45"}))
    check("rettangolo con un lato e 45°: l'angolo retto non è una misura stampata, omessa (richiede il segno)",
          omitted(o, "richiede_segno_angolo_retto"), o)
    o = outcome(tri("right triangle", {"AB": "35", "CA": "70"}, {"C": "30"}))
    check("rettangolo con cateto, ipotenusa e 30°: senza trigonometria non si decide, omessa (richiede il segno)",
          omitted(o, "richiede_segno_angolo_retto"), o)
    f = tri("isosceles triangle", {"AB": "4"}, {"B": "120"}, apex="B")
    check("isoscele con 120° stampato: obtuse, confutate acute e right",
          recovered(keep(f, outcome(f)), "obtuse triangle", ["acute triangle", "right triangle"]))
    o = outcome(tri("isosceles triangle", {"AB": "4"}, {"B": "45"}, apex="B"))
    check("isoscele con un solo lato e 45°: l'angolo stampato può essere alla base, omessa con right e obtuse non "
          "confutate", omitted(o, "non_deciso_dalle_misure_stampate")
          and o["non_confutate"] == ["right triangle", "obtuse triangle"], o)
    f = tri("isosceles triangle", {"AB": "4", "BC": "4"}, {"B": "45"}, apex="B")
    check("isoscele coi due lati uguali stampati e 45° al vertice: acute (angoli 45, 67.5, 67.5)",
          recovered(keep(f, outcome(f)), "acute triangle", ["obtuse triangle", "right triangle"]))
    f = tri("isosceles triangle", {"AB": "4", "BC": "4"}, {"B": "60"}, apex="B")
    check("isoscele equilatero (lati uguali, 60°): acute; equilateral non è fra le opzioni",
          recovered(keep(f, outcome(f)), "acute triangle"))
    o = outcome(tri("isosceles triangle", {"AB": "13/3", "BC": "13/3"}, {"B": "45"}, apex="B"))
    check("uguaglianza apparente: lati 13/3 stampati 4.33 e 4.33 non dimostrano l'isoscele, con 45° omessa",
          omitted(o, "non_deciso_dalle_misure_stampate"), o)
    f = tri("isosceles triangle", {"AB": "13/3", "BC": "13/3"}, {"B": "120"}, apex="B")
    check("uguaglianza apparente con 120° stampato: obtuse dalla sola misura dell'angolo",
          recovered(keep(f, outcome(f)), "obtuse triangle"))
    o = outcome(tri("right triangle", {"AB": "13/3", "BC": "13/3"}, {"C": "45"}))
    check("rettangolo isoscele coi cateti 4.33 e 4.33 (13/3): l'arrotondamento non prova l'uguaglianza, omessa",
          omitted(o, "richiede_segno_angolo_retto"), o)
    o = outcome(tri("isosceles triangle", {"AB": "3", "BC": "4"}, {"B": "45"}, apex="B"))
    check("contraddizione: isoscele coi lati uguali stampati 3 e 4", omitted(o, "misure_contraddittorie")
          and o["dettaglio"] == "lati_uguali_diversi", o)
    o = outcome(tri("right triangle", {"AB": "3"}, {"C": "120"}))
    check("contraddizione: triangolo rettangolo con un angolo di 120°", omitted(o, "misure_contraddittorie"), o)
    o = outcome(tri("isosceles triangle", {"AB": "3"}, {"B": "45"}, apex=None))
    check("isoscele senza vertice nelle annotazioni: omessa", omitted(o, "vertice_dell_isoscele_ignoto"), o)
    # quadrilateri
    o = outcome(quad("square", {"AB": "30"}, {"A": "90"}))
    check("quadrato: rettangolo, rombo e quadrato tutti veri, omessa", omitted(o, "piu_opzioni_vere")
          and o["vere"] == QUAD, o)
    o = outcome(quad("square", {"AB": "30", "DA": "30"}, {"A": "90"}))
    check("quadrato coi lati adiacenti stampati: omessa", omitted(o, "piu_opzioni_vere"), o)
    f = quad("rectangle", {"AB": "3", "DA": "5"}, {"A": "90"})
    check("rettangolo 3 x 5: rectangle, rombo e quadrato confutati dai lati stampati diversi",
          recovered(keep(f, outcome(f)), "rectangle", ["rhombus", "square"]))
    o = outcome(quad("rectangle", {"AB": "5", "DA": "5"}, {"A": "90"}))
    check("rettangolo coi lati adiacenti uguali (un quadrato): omessa", omitted(o, "piu_opzioni_vere"), o)
    o = outcome(quad("parallelogram", {"AB": "65", "BC": "88"}, {"B": "45"}))
    check("parallelogramma generico: nessuna opzione vera (parallelogram esclusa), omessa",
          omitted(o, "nessuna_opzione_vera"), o)
    f = quad("parallelogram", {"AB": "12", "BC": "12"}, {"B": "60"})
    o = keep(f, outcome(f))
    check("parallelogramma coi lati adiacenti uguali (un rombo): rhombus, rettangolo e quadrato confutati dall'angolo",
          recovered(o, "rhombus", ["rectangle", "square"]) and "60" in o["confutazioni"]["rectangle"], o)
    o = outcome(quad("parallelogram", {"AB": "12", "BC": "12"}, {"B": "90"}))
    check("rombo che dalle misure è un quadrato: omessa (S2 §5)", omitted(o, "piu_opzioni_vere"), o)
    f = quad("parallelogram", {"AB": "4/3", "BC": "4/3"}, {"B": "45"})
    o = keep(f, outcome(f))
    check("rombo coi lati 4/3 stampati 1.33 e 1.33: rhombus dall'esatto, confutazioni dal solo angolo",
          recovered(o, "rhombus") and all("45" in v for v in o["confutazioni"].values()), o)
    o = outcome(quad("rectangle", {"AB": "4/3", "DA": "4/3"}, {"A": "90"}))
    check("rettangolo coi lati 4/3 e 4/3 (quadrato per l'esatto): omessa", omitted(o, "piu_opzioni_vere"), o)
    for other in ("15", "29/2"):
        f = quad("rectangle", {"AB": "44/3", "DA": other}, {"A": "90"})
        check(f"rettangolo 44/3 (14.67) x {other}: rectangle, lati stampati diversi in modo dimostrato",
              recovered(keep(f, outcome(f)), "rectangle", ["rhombus", "square"]))
    o = outcome(quad("parallelogram", {"AB": "5"}, {"B": "45"}))
    check("misure insufficienti: parallelogramma con un solo lato, il rombo è ignoto e non falso: omessa",
          omitted(o, "verita_ignota") and o["ignote"] == ["rhombus"], o)
    for name, f, detail in (
            ("rettangolo con un angolo di 60°", quad("rectangle", {"AB": "3"}, {"A": "60"}),
             "angolo_non_retto_in_un_rettangolo"),
            ("quadrato coi lati 3 e 4", quad("square", {"AB": "3", "BC": "4"}, {"A": "90"}), "lati_diversi_in_un_quadrato"),
            ("parallelogramma coi lati opposti 3 e 4", quad("parallelogram", {"AB": "3", "CD": "4"}, {"B": "45"}),
             "lati_opposti_diversi")):
        o = outcome(f)
        check(f"contraddizione: {name}", omitted(o, "misure_contraddittorie") and o["dettaglio"] == detail, o)
    o = outcome(quad("parallelogram", {"AB": "99", "BC": "71", "CD": "99"}, {"B": "60"}))
    check("parallelogramma con tre lati stampati, generico: omessa", omitted(o, "nessuna_opzione_vera"), o)
    f = quad("rectangle", {"AB": "99", "DA": "71", "CD": "99"}, {"A": "90"})
    check("rettangolo con tre lati stampati: rectangle", recovered(keep(f, outcome(f)), "rectangle"))
    o = outcome(ShapeFacts("ABC", "sector", "sector", "", sides={"AB": "1", "BC": "1"}, angles={"B": "60"}))
    check("settore: nessun insieme informativo, omessa", omitted(o, "settore_nessun_insieme_informativo"), o)
    return emitted


# ---------------------------------------------------------------- controllo indipendente e mutazioni

def test_checker(emitted) -> None:
    alarms = {f"{f.name} {o['risposta']}": c9.check_outcome("k", f, o) for f, o in emitted}
    check(f"controllo indipendente: tutte le {len(emitted)} domande emesse accettate",
          len(emitted) == 13 and not any(alarms.values()), {k: v for k, v in alarms.items() if v} or len(emitted))
    rect = quad("rectangle", {"AB": "3", "DA": "5"}, {"A": "90"})
    o = outcome(rect)
    opts, ans = o["opzioni"], o["risposta"]

    def flagged(f, options, answer, text=None):
        return bool(c9.check_question("k", f, options, answer, text or c9.question_text(f.family, f.name, options)))

    check("mutazione trovata: risposta cambiata in un'opzione sbagliata (square)", flagged(rect, opts, "square"))
    swapped = ["parallelogram" if x == "square" else x for x in opts]
    check("mutazione trovata: opzione vera di inclusione aggiunta (parallelogram al posto di square)",
          flagged(rect, swapped, ans))
    rev = list(reversed(opts))
    check("mutazione trovata: ordine delle opzioni diverso da quello per chiave", flagged(rect, rev, ans))
    check("mutazione trovata: testo della domanda alterato",
          flagged(rect, opts, ans, o["domanda"]["q"].replace(" or ", ", ")))
    check("mutazione trovata: risposta assente o ripetuta fra le opzioni",
          flagged(rect, [x for x in opts if x != ans] + ["square"], ans) and flagged(rect, opts + [ans], ans))
    check("mutazione trovata: fatti alterati, lati uguali (un quadrato): più opzioni vere",
          flagged(quad("rectangle", {"AB": "3", "DA": "3"}, {"A": "90"}), opts, ans))
    check("mutazione trovata: fatti alterati, lato adiacente tolto: rombo e quadrato ignoti",
          flagged(quad("rectangle", {"AB": "3"}, {"A": "90"}), opts, ans))
    obt = tri("isosceles triangle", {"AB": "4"}, {"B": "120"}, apex="B")
    oo = outcome(obt)
    check("mutazione trovata: angolo al vertice cambiato in 45° (la risposta obtuse diventa falsa)",
          flagged(tri("isosceles triangle", {"AB": "4"}, {"B": "45"}, apex="B"), oo["opzioni"], oo["risposta"]))
    check("mutazione trovata: risposta cambiata in acute", flagged(obt, oo["opzioni"], "acute triangle"))
    check("mutazione trovata: risposta con verità ignota (rombo di un parallelogramma con un solo lato)",
          flagged(quad("parallelogram", {"AB": "5"}, {"B": "45"}), c9.key_shuffle("k", "ABCD", tuple(QUAD)),
                  "rhombus"))
    check("mutazione trovata: rettangolo isoscele coi cateti arrotondati 13/3 posto come domanda right",
          flagged(tri("right triangle", {"AB": "13/3", "BC": "13/3"}, {"C": "45"}),
                  c9.key_shuffle("k", "ABC", tuple(TRI)), "right triangle"))
    forged = dict(o, confutazioni={"rhombus": "x"})
    check("mutazione trovata: esito con confutazioni diverse dalle opzioni sbagliate",
          bool(c9.check_outcome("k", rect, forged)))


def test_determinism() -> None:
    f = quad("rectangle", {"AB": "3", "DA": "5"}, {"A": "90"})
    check("riproducibile: stesso esito a ogni chiamata, ordine che dipende da chiave e figura",
          outcome(f, "d1-7") == outcome(f, "d1-7")
          and len({tuple(c9.key_shuffle(f"d1-{i}", "ABCD", tuple(QUAD))) for i in range(30)}) > 1)
    pos = Counter(outcome(f, f"d{i % 4}-{i}")["opzioni"].index("rectangle") for i in range(3000))
    check("posizione della risposta senza scorciatoia: ogni posizione fra il 28% e il 39% su 3000 chiavi",
          all(0.28 <= pos[i] / 3000 <= 0.39 for i in range(3)), dict(pos))


# ---------------------------------------------------------------- pool delle domande da testi veri

RIGHT_ISO_CAP = ("ABC is a right triangle. Angle ACB is 45 degrees. Side AB is 5 units long. Side BC is 5 units long. ")
RIGHT_ISO_GEN = "Draw a right triangle abc. angle acb is 45 degrees. side ab is 5 units long. side bc is 5 units long."


def test_pool() -> None:
    pairs = dict(REAL, **{"d2-90010": (MULTI_CAP, MULTI_GEN), "d0-77001": (RIGHT_ISO_CAP, RIGHT_ISO_GEN)})
    bad = []
    for key, (cap, gen) in pairs.items():
        truth = mavis_truth(cap, gen)[0]
        hist = mavis_questions(key, truth)
        pool, outs, vals = c9.questions(key, truth, gen)
        keep_h = [(q["type"], q["q"], q["answer"]) for q in hist if q["type"] != "shape_type"]
        keep_c = [(q["type"], q["q"], q["answer"]) for q in pool if q["type"] != "shape_type"]
        reading = [q for q in pool if q["type"] in ("side_len", "angle_val")]
        if keep_h != keep_c or len(outs) != sum(q["type"] == "shape_type" for q in hist) \
                or any(vals[q["elemento"]]["risposta_lettura"] != q["answer"] for q in reading) \
                or sum(q["type"] == "shape_type" for q in pool) != sum(o["esito"] == "recuperabile" for o in outs):
            bad.append(key)
    check("pool C9: domande diverse da shape_type identiche e nello stesso ordine, lettura legata al valore "
          "visualizzato, una figura per domanda shape_type storica", not bad, bad)
    by = {k: c9.questions(k, mavis_truth(*v)[0], v[1])[1] for k, v in pairs.items()}
    check("pool C9 sui testi: d0-9323 (rettangolo 3 x 5) rectangle; d0-16855 (parallelogramma 1, √2, 45°) omessa",
          recovered(by["d0-9323"][0], "rectangle") and omitted(by["d0-16855"][0], "nessuna_opzione_vera"))
    check("pool C9 sui testi: settore omesso, triangolo FDG senza lati stampati omesso (richiede il segno)",
          [o["motivo"] for o in by["d2-90010"]] == ["settore_nessun_insieme_informativo", "richiede_segno_angolo_retto"],
          by["d2-90010"])
    check("pool C9 sui testi: rettangolo isoscele coi cateti 5 e 5 e 45°: right", recovered(by["d0-77001"][0],
                                                                                         "right triangle"))


# ---------------------------------------------------------------- compatibilità con i manifest correnti

def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 32), "white").save(buf, format="PNG")
    return buf.getvalue()


def files_for(key, cap, gen, adm: dict | None) -> dict:
    meta = {"id": key, "variant": "caption", "depth": 0, "text_en": {"caption": cap, "diagram_generation": gen},
            "max_patches": 2304}
    out = {"json": json.dumps(meta).encode(), "png": png_bytes()}
    if adm is not None:
        out[mf.ADMISSION_FILE] = mf.canonical(adm).encode()
    return out


def test_compatibility() -> None:
    old = subprocess.run(["git", "-C", str(ROOT), "show", "68daf5a:src/limite_vl/f1_mavis.py"], capture_output=True,
                         check=True).stdout
    check("compatibilità: f1_mavis.py (costruzione storica) identico al commit 68daf5a revisionato per C7",
          old == (ROOT / "src/limite_vl/f1_mavis.py").read_bytes())
    rec = {"split": "train", "stato": "ammesso", "qa": [{"tipo": "side_len", "sha256": "0" * 64}],
           "descrizione_sha256": "1" * 64, "compiti": [{"compito": "describe", "stato": "ammesso"}]}
    _, _, extra_old = mf._record_tasks("k", rec, None, Counter())
    _, _, extra_new = mf._record_tasks("k", {**rec, "costruzione_qa": c9.CONSTRUCTION}, None, Counter())
    check("compatibilità: la decisione allegata di un record storico è invariata (nessuna costruzione), quella C9 la "
          "dichiara", "costruzione_qa" not in extra_old and extra_new.get("costruzione_qa") == c9.CONSTRUCTION
          and {k: v for k, v in extra_new.items() if k != "costruzione_qa"} == extra_old)
    cap, gen = REAL["d0-9323"]
    key = "d0-9323"
    truth = mavis_truth(cap, gen)[0]
    hist = mavis_questions(key, truth)
    pool = c9.questions(key, truth, gen)[0]
    h_sha = mf.qa_list_sha256((q["type"], mf.qa_item_sha256(q)) for q in hist)
    c_sha = mf.qa_list_sha256((q["type"], mf.qa_item_sha256(q)) for q in pool)
    base = {"compiti_ammessi": ["count_points", "describe", "side_len"], "compiti_bloccati": ["shape_type"],
            "descrizione_sha256": mf.sha256_text(describe(key, truth)[1])}

    def load(adm):
        try:
            return dt.SAMPLE_LOADERS["mavis_f1"](key, files_for(key, cap, gen, adm))
        except mf.ManifestError as exc:
            return exc

    s_old = load({**base, "qa_sha256": h_sha})
    s_new = load({**base, "qa_sha256": c_sha, "costruzione_qa": c9.CONSTRUCTION})
    check("loader: costruzione storica senza dichiarazione e C9 dichiarata, stessi messaggi con shape_type bloccata",
          not isinstance(s_old, Exception) and not isinstance(s_new, Exception) and s_old.messages == s_new.messages
          and "shape_type" not in s_new.tasks, (s_old, s_new))
    check("loader: pool C9 senza la costruzione dichiarata rifiutato; pool storico con la costruzione C9 rifiutato; "
          "costruzione sconosciuta rifiutata",
          isinstance(load({**base, "qa_sha256": c_sha}), mf.ManifestError)
          and isinstance(load({**base, "qa_sha256": h_sha, "costruzione_qa": c9.CONSTRUCTION}), mf.ManifestError)
          and isinstance(load({**base, "qa_sha256": c_sha, "costruzione_qa": "mavis-c9/0"}), mf.ManifestError))
    raw = dt.SAMPLE_LOADERS["mavis_f1"](key, files_for(key, cap, gen, None))
    check("loader senza manifest: la scelta storica, invariata", raw.blocked == {} and (
        raw.tasks == ("describe",) or set(raw.tasks) <= {q["type"] for q in hist}))


# ---------------------------------------------------------------- derivazione, verifica e reader su shard fittizi

def rewrite(folder: Path, dst: Path, edit) -> Path:
    shutil.copytree(folder, dst)
    path = dst / "records.jsonl.gz"
    recs = [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()]
    for r in recs:
        edit(r)
    path.unlink()
    sha, _ = um.write_records_gz(path, (json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                                        for r in recs))
    meta = json.loads((dst / "manifest.json").read_text())
    meta["records"]["sha256"] = sha
    meta["conteggi"] = mm.summarize(recs)
    (dst / "manifest.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    return dst


VALUE_CHECKS = ("valori uguali a quelli ricavati dalle annotazioni",
                "valori coerenti con la regola di rappresentazione")


def entry(values: list[dict], elem: str) -> dict:
    return next(v for v in values if v["elemento"] == elem)


def test_values_against_source(tmp: Path, args, parent: Path, parent_sha: str, ver: dict, C: dict) -> None:
    """Valori del candidato contro i JSON originali (verify_c9): casi validi, poi mutazioni dei valori con gli hash
    interni aggiornati (rewrite), che devono fallire per incoerenza o per differenza dalla fonte."""
    v = ver["valori"]
    check("valori contro la fonte, candidato valido: ogni misura registrata ritrovata nei JSON originali, nessuna "
          "discrepanza né incoerenza", ver["esito"] and v["record_letti"] == len(C) and v["misure_confrontate"]
          == v["misure_attese"] == v["misure_registrate"] > 0 and not v["discrepanze"] and v["incoerenze"] == 0, v)
    exact = entry(C["d0-13996"]["valori"], "side:AD")
    rounded = entry(C["d0-97044"]["valori"], "side:AD")
    check("valori validi verificati contro la fonte: decimale esatto (d0-13996, 3/2 -> 1.5) e decimale arrotondato "
          "(d0-97044, 44/3 -> 14.67)",
          exact == {"elemento": "side:AD", "esatto": "3/2", "visualizzato_previsto": "1.5",
                    "rappresentazione": "decimale_esatto", "risposta_lettura": "1.5"}
          and rounded == {"elemento": "side:AD", "esatto": "44/3", "visualizzato_previsto": "14.67",
                          "rappresentazione": "decimale_arrotondato", "risposta_lettura": "14.67"}
          and {"decimale_esatto", "decimale_arrotondato"} <= set(v["rappresentazioni_attese"]), (exact, rounded))

    def side_ad(**new):
        return lambda vals: entry(vals, "side:AD").update(new)

    # (nome, cartella, record, alterazione, discrepanze attese, voce coerente, falliscono solo i controlli dei valori)
    cases = (
        ("esatto alterato (3/2 -> 7/4), target e visualizzato invariati", "esatto", "d0-13996",
         side_ad(esatto="7/4"), {"esatto"}, False, True),
        ("rappresentazione alterata (decimale esatto -> decimale arrotondato)", "rappresentazione", "d0-13996",
         side_ad(rappresentazione="decimale_arrotondato"), {"rappresentazione"}, False, True),
        ("esatto alternativo con lo stesso visualizzato, voce coerente (44/3 -> 4402/300, sempre 14.67)",
         "alternativo-44", "d0-97044", side_ad(esatto="4402/300"), {"esatto"}, True, True),
        ("esatto alternativo con lo stesso visualizzato e la rappresentazione aggiornata (3/2 -> 301/200, sempre 1.5)",
         "alternativo-3", "d0-13996", side_ad(esatto="301/200", rappresentazione="decimale_arrotondato"),
         {"esatto", "rappresentazione"}, True, True),
        ("elemento mancante", "mancante", "d0-13996", lambda vals: vals.remove(entry(vals, "side:AD")),
         {"mancante"}, True, False),
        ("elemento ripetuto", "ripetuto", "d0-13996", lambda vals: vals.append(dict(entry(vals, "side:AD"))),
         {"ripetuto"}, True, False),
        ("elemento in più, voce coerente", "in-piu", "d0-13996",
         lambda vals: vals.append({"elemento": "side:ZZ", "esatto": "5", "visualizzato_previsto": "5",
                                   "rappresentazione": "intero", "risposta_lettura": "5"}), {"in_piu"}, True, False),
    )
    for name, slug, key, change, kinds, coherent, only_values in cases:
        def edit(r, key=key, change=change):
            if r["record"] == key:
                change(r["valori"])

        res = job.verify_c9(rewrite(args.new, tmp / f"valori-{slug}", edit), parent, parent_sha,
                            args.out / "report.json", workers=2)
        fails = [c["controllo"] for c in res["controlli"] if not c["esito"]]
        val = res["valori"]
        cond = (not res["esito"] and set(val["discrepanze"]) == kinds and (val["incoerenze"] == 0) == coherent
                and val["record_con_discrepanze"] == 1)
        if only_values:  # hash interni aggiornati: nessun altro controllo se ne accorge
            cond = cond and bool(fails) and all(f.startswith(VALUE_CHECKS) for f in fails)
        check(f"verifica dei valori: trova {name}", cond, (fails, val["discrepanze"], val["incoerenze"]))


def test_pipeline(tmp: Path) -> None:
    from limite_vl import stream as st

    s = tm.spec()
    cap, gen = REAL["d0-13996"]  # AD = 3/2 (decimale esatto); la copia con 44/3 dà un decimale arrotondato
    s["shards"][1][1].append(tm.sample("d0-97044", cap.replace("3/2", "44/3"), gen.replace("3/2", "44/3"), tm.png(77)))
    fargs = tm.write_fixture(tmp, s)
    mm.run(fargs)
    parent, parent_sha = fargs.out, sha256_file(fargs.out / "manifest.json")
    texts = {}
    for _, items in s["shards"]:
        for key, _, meta in items:
            truth = mavis_truth(meta["text_en"]["caption"], meta["text_en"]["diagram_generation"])[0]
            if truth:
                for o in c9.questions(key, truth, meta["text_en"]["diagram_generation"])[1]:
                    if o["esito"] == "recuperabile":
                        texts[key] = o["domanda"]["q"]
    target = "d0-91006"
    index = tmp / "indice"
    index.mkdir()
    (index / "index.json").write_text(json.dumps({"schema": "panel-index/2", "items": [
        {"benchmark": "prova", "item": "1", "text": "Look at the figure. " + texts[target] + " Explain why."},
        {"benchmark": "prova", "item": "2", "text": "What is the area of the shaded region in the figure below?"}]}))
    args = argparse.Namespace(manifest=parent, manifest_sha256=parent_sha, new=tmp / "c9", out=tmp / "scan",
                              report=tmp / "c9-verifica", index=index, index_sha256=None, summary=tmp / "c9.json",
                              version="2", workers=2)
    try:
        code = job.cmd_costruzione(args)
    except job.Failed as exc:
        code = str(exc)
    ver = json.loads((args.report / "verifica.json").read_text()) if (args.report / "verifica.json").exists() \
        else {"esito": False, "controlli": []}
    check("derivazione C9 su shard fittizi: scritta e verificata", code == 0 and ver["esito"],
          (code, [c for c in ver["controlli"] if not c["esito"]]))
    if code != 0:
        return
    child = mf.load(args.new)
    P = {k: r for rows in mf.load(parent).by_shard.values() for k, r in rows.items()}
    C = {k: r for rows in child.by_shard.values() for k, r in rows.items()}
    moved = sorted(k for k in C if C[k]["stato"] != P[k]["stato"])
    # lo stesso testo (stessa figura ABCD, stesso ordine delle opzioni) può comparire in altri record: tutti i loro gruppi
    with_text = {k for k, r in C.items() for e in r["shape_type_c9"] if e["esito"] == "recuperabile"
                 and c9.question_text(e["fatti"]["famiglia"], e["figura"], e["opzioni"]) == texts[target]}
    groups = {C[k]["gruppo"] for k in with_text}
    check("controllo testuale: la domanda nuova in comune con un testo del pannello manda in quarantena i gruppi dei "
          "record che la hanno, e solo quelli", moved == sorted(k for k in C if C[k]["gruppo"] in groups
                                                                and P[k]["stato"] == "ammesso")
          and target in moved and job.HIT_REASON in C[target]["motivo"]
          and C[target]["evidenze"][-1]["testo_nuovo"] == "candidato"
          and all(C[k]["evidenze"][-1]["testo_nuovo"] == "nessuna_corrispondenza" for k in C if k not in with_text),
          (moved, sorted(with_text)))
    same = all(C[k][f] == P[k][f] for k in C for f in ("gruppo", "split", "record_id", "json_sha256", "immagini"))
    check("derivazione: gruppi, split e identità ereditati; costruzione dichiarata; shape_type mai ammessa",
          same and all(r["costruzione_qa"] == c9.CONSTRUCTION for r in C.values())
          and not any(t["compito"] == "shape_type" and t["stato"] == "ammesso" for r in C.values() for t in r["compiti"]))
    check("derivazione: valori registrati per ogni record con verità; AD = 3/2 di d0-13996 come decimale esatto",
          all(r["valori"] for r in C.values() if r["verita_sha256"])
          and next(v for v in C["d0-13996"]["valori"] if v["elemento"] == "side:AD")
          == {"elemento": "side:AD", "esatto": "3/2", "visualizzato_previsto": "1.5",
              "rappresentazione": "decimale_esatto", "risposta_lettura": "1.5"})
    for name, edit in (
            ("stato riammesso", lambda r: r.update(stato="ammesso") if r["record"] == target else None),
            ("domanda di lettura alterata", lambda r: r["qa"][0].update(sha256="0" * 64)
             if r["record"] == "d0-9323" else None),
            ("valore visualizzato alterato", lambda r: r["valori"][0].update(risposta_lettura="99")
             if r["record"] == "d0-9323" else None),
            ("risposta shape_type alterata", lambda r: [e.update(risposta="square") for e in r["shape_type_c9"]
                                                        if e["esito"] == "recuperabile"] if r["record"] == "d0-9323"
             else None),
            ("evidenza del genitore tolta", lambda r: r["evidenze"].pop(0) if r["record"] == "d0-9323" else None),
            ("shape_type ammessa", lambda r: [t.update(stato="ammesso") for t in r["compiti"]
                                              if t["compito"] == "shape_type"] if r["record"] == "d0-91007" else None)):
        bad = rewrite(args.new, tmp / f"alterato-{len(FAILS)}-{name.replace(' ', '_')}", edit)
        res = job.verify_c9(bad, parent, parent_sha, args.out / "report.json", workers=2)
        check(f"verifica: trova l'alterazione ({name})", not res["esito"],
              [c["controllo"] for c in res["controlli"] if not c["esito"]])
    test_values_against_source(tmp, args, parent, parent_sha, ver, C)
    try:
        job.cmd_costruzione(argparse.Namespace(**{**vars(args), "manifest": args.new,
                                                  "manifest_sha256": sha256_file(args.new / "manifest.json"),
                                                  "new": tmp / "c9bis", "out": tmp / "scanbis", "report": tmp / "vbis",
                                                  "summary": tmp / "bis.json", "version": "3"}))
        twice = False
    except job.Failed:
        twice = True
    check("derivazione: rifiutata da un genitore che dichiara già la costruzione", twice)
    glob_ = str(fargs.shards / "*.tar")

    def spec(name, folder, sha, split):
        return {"name": name, "kind": "mavis_f1", "shards": glob_, "weight": 1.0, "manifest": str(folder),
                "manifest_sha256": sha, "split": split}

    c9_sha = sha256_file(args.new / "manifest.json")
    sources = st.make_sources([spec("c9", args.new, c9_sha, "train"), spec("c7", parent, parent_sha, "train")])
    got = {}
    for src in sources:
        reader = st.SourceReader(src, seed=3, worker=0, n_workers=1, val_count=0)
        got[src.name] = {}
        while True:
            key, files = reader.next()
            if reader.epoch > 0:
                break
            got[src.name][key] = files
    bad = []
    for key, files in got["c9"].items():
        smp = dt.SAMPLE_LOADERS["mavis_f1"](key, files)
        ref = dt.SAMPLE_LOADERS["mavis_f1"](key, got["c7"][key]) if key in got["c7"] else None
        n_shape = sum(q["tipo"] == "shape_type" for q in C[key]["qa"])
        expect = 0 if smp.tasks == ("describe",) else n_shape
        if "shape_type" in smp.tasks or ref is None or smp.messages != ref.messages \
                or (smp.blocked or {}).get("shape_type", 0) != expect:
            bad.append(key)
    check("reader col candidato C9: consegna i record ammessi di train, digest del pool C9 verificato, shape_type mai, "
          "stessi messaggi del genitore", got["c9"] and set(got["c9"]) == set(got["c7"]) - set(moved) and not bad,
          (len(got["c9"]), len(got["c7"]), bad[:5]))


def main() -> int:
    test_values()
    emitted = test_shapes()
    test_checker(emitted)
    test_determinism()
    test_pool()
    test_compatibility()
    with tempfile.TemporaryDirectory() as tmp:
        test_pipeline(Path(tmp))
    print(f"{'FALLITI: ' + str(len(FAILS)) if FAILS else 'tutti superati'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
