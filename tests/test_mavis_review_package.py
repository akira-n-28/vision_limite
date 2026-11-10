"""Test CPU della selezione del pacchetto di revisione visiva di MAVIS (scripts/mavis_review_package.py): celle a
giro, gruppi distinti, deficit registrati senza compensazioni, ordine di elaborazione, condizioni degli strati e
protezioni del primo passaggio. Dati finti, nessun file reale. Solo in un job (slurm/tests_cpu.sbatch)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mavis_review_package as mrp  # noqa: E402

PASSED: list[str] = []


def ok(cond, name: str) -> None:
    if not cond:
        raise AssertionError(name)
    PASSED.append(name)
    print("ok  ", name)


def gid(n: int) -> str:
    return f"{n:064x}"


def fake_cands(spec: dict) -> dict:
    """{strato: {cella: [gruppi]}} -> candidati nel formato di candidates_by_stratum."""
    out = {}
    for s, cells in spec.items():
        out[s] = {}
        for cell, groups in cells.items():
            out[s][cell] = {g: (mrp.rank(strato=s, unita=f"r{g[-4:]}|u"), f"r{g[-4:]}|u", f"r{g[-4:]}", {"tipo": "x"})
                            for g in groups}
    return out


def test_select() -> None:
    a, b = "lettura_valore/intero", "classificazione_omessa/piu_opzioni_vere"
    plan = (("controllo_noto", "radice_di_quadrato_perfetto", 1), ("lettura_valore", "intero", 4),
            ("classificazione_omessa", "piu_opzioni_vere", 4))
    groups = [gid(i) for i in range(1, 30)]
    cands = fake_cands({
        a: {("d3", "alto"): groups[0:5], ("d2", "basso"): groups[5:10], ("d1", "alto"): groups[10:12],
            ("d0", "basso"): groups[12:14]},
        b: {("d3",): [groups[0], groups[20]], ("d1",): [groups[21]]},  # groups[0] anche in a
    })
    picks, report, order = mrp.select(plan, cands, {gid(99)})
    ok(order == [b, a], "strati elaborati dal meno disponibile (3 gruppi) al più disponibile (14)")
    by = {r["strato"]: r for r in report}
    pa = [p for p in picks if p["strato"] == a]
    pb = [p for p in picks if p["strato"] == b]
    ok(len(pa) == 4 and [p["cella"] for p in pa] == [("d3", "alto"), ("d2", "basso"), ("d1", "alto"), ("d0", "basso")],
       "celle a giro: d3, d2, d1, d0 alternando affollamento")
    ok(len(pb) == 3 and by[b]["deficit"] == 1 and len([c for c in by[b]["celle"] if "esaurita" in c]) == 4
       and [p["cella"] for p in pb] == [("d3",), ("d1",), ("d3",)],
       "deficit registrato quando tutte le celle dello strato sono esaurite, senza compensare con un altro strato")
    ok(len({p["gruppo"] for p in picks}) == len(picks) and gid(99) not in {p["gruppo"] for p in picks},
       "gruppi distinti e gruppo del controllo riservato")
    first_d3 = sorted(cands[b][("d3",)], key=lambda g: mrp.rank(strato=b, gruppo=g))[0]
    ok(pb[0]["gruppo"] == first_d3, "nella cella il gruppo col rango minore")
    ok(groups[0] in {p["gruppo"] for p in pb} and groups[0] not in {p["gruppo"] for p in pa},
       "un gruppo usato da uno strato non torna in un altro")
    again = mrp.select(plan, cands, {gid(99)})
    ok(again[0] == picks and again[1] == report, "selezione deterministica")
    empty = mrp.select(plan, {}, set())
    ok(all(r["deficit"] == r["quota"] for r in empty[1]) and not empty[0], "nessun candidato: deficit pieni")


def test_strata() -> None:
    ok(mrp.apparent_rounding({"AB": "44/3", "BC": "44/3"}), "uguaglianza apparente: due lati 14.67 da 44/3")
    ok(not mrp.apparent_rounding({"AB": "12", "BC": "12"}), "lati interi uguali: uguaglianza dimostrata, non apparente")
    ok(not mrp.apparent_rounding({"AB": "44/3", "BC": "12"}), "un solo decimale arrotondato: nessuna uguaglianza")
    base = {"esito": "omessa", "fatti": {"lati": {"AB": "12", "BC": "7"}}}
    ok(mrp.omitted_class({**base, "motivo": "piu_opzioni_vere"}) == "piu_opzioni_vere"
       and mrp.omitted_class({**base, "motivo": "richiede_segno_angolo_retto"}) == "segno_angolo_retto"
       and mrp.omitted_class({**base, "motivo": "non_deciso_dalle_misure_stampate"}) == "misure_insufficienti"
       and mrp.omitted_class({"esito": "omessa", "motivo": "non_deciso_dalle_misure_stampate",
                              "fatti": {"lati": {"AB": "44/3", "AC": "44/3"}}}) == "uguaglianza_apparente_arrotondamento"
       and mrp.omitted_class({**base, "motivo": "verita_ignota"}) is None
       and mrp.omitted_class({"esito": "recuperabile"}) is None, "sottocategorie delle omesse")
    rec = {"record": "d1-7", "split": "train", "stato": "ammesso", "descrizione_sha256": "x",
           "compiti": [{"compito": "describe", "stato": "ammesso"},
                       {"compito": "side_len", "stato": "ammesso", "domande": 2},
                       {"compito": "angle_val", "stato": "ammesso", "domande": 1},
                       {"compito": "shape_type", "stato": "quarantena", "domande": 1},
                       {"compito": "count_points", "stato": "ammesso", "domande": 1}],
           "qa": [{"tipo": "side_len", "sha256": "1", "elemento": "side:AB"},
                  {"tipo": "side_len", "sha256": "2", "elemento": "side:BC"},
                  {"tipo": "angle_val", "sha256": "3", "elemento": "angle:ABC"},
                  {"tipo": "shape_type", "sha256": "4"}, {"tipo": "count_points", "sha256": "5"}],
           "valori": [{"elemento": "side:AB", "rappresentazione": "decimale_arrotondato"},
                      {"elemento": "side:BC", "rappresentazione": "radice"},
                      {"elemento": "angle:ABC", "rappresentazione": "intero"}],
           "shape_type_c9": [{"figura": "ABC", "esito": "recuperabile", "risposta": "acute triangle"}]}
    got = {s for s, _, _ in mrp.units_of(rec)}
    ok(got == {"lettura_valore/decimale_arrotondato", "lettura_valore/radice", "lettura_valore/angolo",
               "classificazione_recuperabile/acute triangle", "altro_compito/describe", "altro_compito/count_points"},
       "unità candidate di un record di train ammesso")
    ok(not mrp.units_of({**rec, "split": "dev"}) and not mrp.units_of({**rec, "stato": "quarantena"}),
       "nessuna unità fuori dai record di train ammessi")
    blocked = {**rec, "compiti": [{**t, "stato": "quarantena"} if t["compito"] == "side_len" else t
                                  for t in rec["compiti"]]}
    ok(not {s for s, _, _ in mrp.units_of(blocked)} & {"lettura_valore/decimale_arrotondato", "lettura_valore/radice"},
       "lettura esclusa se il suo compito non è ammesso")
    ctrl = {**rec, "record": mrp.CONTROL[0], "valori": [{"elemento": mrp.CONTROL[1], "rappresentazione": "radice"}],
            "qa": [{"tipo": "side_len", "sha256": "1", "elemento": mrp.CONTROL[1]}]}
    ok("lettura_valore/radice" not in {s for s, _, _ in mrp.units_of(ctrl)}, "il controllo noto non entra negli strati")
    s_val = {"elemento": "side:BC", "compito": "side_len"}
    ok(mrp.condition_ok("lettura_valore", "radice", rec, s_val)
       and not mrp.condition_ok("lettura_valore", "intero", rec, s_val)
       and not mrp.condition_ok("lettura_valore", "radice", {**rec, "split": "dev"}, s_val),
       "condizione della lettura riverificata a parte")
    rec2 = {**rec, "shape_type_c9": [{"figura": "ABC", "esito": "omessa", "motivo": "non_deciso_dalle_misure_stampate",
                                      "fatti": {"lati": {"AB": "44/3", "AC": "44/3"}}}]}
    s_fig = {"figura": "ABC", "compito": "shape_type"}
    ok(mrp.condition_ok("classificazione_omessa", "uguaglianza_apparente_arrotondamento", rec2, s_fig)
       and not mrp.condition_ok("classificazione_omessa", "misure_insufficienti", rec2, s_fig)
       and mrp.condition_ok("classificazione_recuperabile", "acute triangle", rec, s_fig)
       and not mrp.condition_ok("classificazione_recuperabile", "acute triangle",
                                {**rec, "compiti": [{"compito": "shape_type", "stato": "ammesso"}]}, s_fig),
       "condizione delle classificazioni riverificata a parte, shape_type ammessa rifiutata")


def test_first_pass() -> None:
    card = {"id": "M01", "record": "d0-1", "domanda": "Is the semicircle on AB drawn inside or outside rectangle ABCD?",
            "consegna": "c", "opzioni": []}
    sol = {"target": "inside", "sottocategoria": "direction", "categoria": "altro_compito",
           "strato": "altro_compito/direction", "motivo_selezione": "altro_compito/direction, cella d0: 1° gruppo",
           "annotazioni": {"didascalia": "a long caption with words"}, "valore": None, "c9": None}
    sec = mrp.secrets_card(card, sol)
    ok("inside" not in sec and "a long caption with words" in sec and "direction" in sec,
       "segreti: il target scritto nella domanda non è un segreto, didascalia e categoria sì")
    ok(mrp.strings_of({"a": ["x", {"b": "y"}], "c": True, "d": None}) == "x\ny\n\n",
       "testi di un oggetto senza chiavi né booleani")
    ok(not mrp.HIDDEN_KEYS & {"id", "posizione", "record", "gruppo", "immagine", "immagine_sha256", "larghezza", "altezza",
                              "richiesta", "controllo_noto", "domanda", "opzioni", "consegna"},
       "le chiavi del primo passaggio non sono fra quelle nascoste")
    words = ("arrotond", "esatt", "radice", "intero", "recuperabil", "omess", "controllo", "piu_opzioni", "quadrato")
    ok(not any(w in t.lower() for t in mrp.CONSEGNE.values() for w in words), "consegne senza parole delle categorie")
    text = mrp.tsv_text([{k: ("a\tb\nc\\" if k == "domanda" else "") for k in mrp.TSV_FIELDS}])
    ok(text.splitlines()[1].split("\t")[mrp.TSV_FIELDS.index("domanda")] == "a\\tb\\nc\\\\",
       "inventario: tab, a capo e barra rovesciata protetti")
    ok(sum(q for _, _, q in mrp.PLAN) == mrp.TOTAL == 80, "quote del piano: 80 schede")


def main() -> None:
    test_select()
    test_strata()
    test_first_pass()
    print(f"{len(PASSED)} controlli superati")


if __name__ == "__main__":
    main()
