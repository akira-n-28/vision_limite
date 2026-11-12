"""Test CPU dell'adattatore che integra gli esiti della revisione dei sintetici (scripts/review_results.py): letture
rigide dei TSV, sha256 e byte esatti dei file caricati, indipendenza ricalcolata dalle esposizioni (anche per scena),
righe TSV ricostruite dal JSON con prima risposta e giudizio finale conservati, regole della decontaminazione,
equivalenze di scrittura delle risposte, proposta per record e gruppo, riesame tecnico separato. Dati finti.

    srun $RUN_PYTHON -B tests/test_review_results.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import review_results as rr  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, exc=rr.Failed) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def test_tsv():
    cols = ("a", "b")
    rows = rr.read_tsv("a\tb\nx\\ny\tz\\tw\\\\v\n", cols)
    check("TSV: a capo, tabulazioni e barre rovesciate ripristinati", rows == [{"a": "x\ny", "b": "z\tw\\v"}], rows)
    check("TSV: intestazione diversa dal modulo rifiutata", raises(lambda: rr.read_tsv("a\tc\nx\ty\n", cols)))
    check("TSV: riga con un campo in più rifiutata", raises(lambda: rr.read_tsv("a\tb\nx\ty\tz\n", cols)))
    check("TSV: file senza a capo finale rifiutato", raises(lambda: rr.read_tsv("a\tb\nx\ty", cols)))


def test_read_results():
    with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
        paths = {}
        for name, data in (("avanzamento", b'{"x": 1}\n'), ("esiti", b"a\tb\r\nx\ty\r\n"), ("dec", b"u\n")):
            p = Path(tmp) / name
            p.write_bytes(data)
            paths[name] = (str(p), hashlib.sha256(data).hexdigest())
        args = argparse.Namespace(avanzamento=paths["avanzamento"], esiti=paths["esiti"],
                                  esiti_decontaminazione=paths["dec"])
        res = rr.read_results(args, rr.Checks())
        check("risultati: testo uguale ai byte del file, \\r\\n compreso (nessuna conversione)",
              res["esiti"]["text"].encode() == b"a\tb\r\nx\ty\r\n" and res["avanzamento"]["bytes"] == 9)
        bad = argparse.Namespace(**{**vars(args), "esiti": (paths["esiti"][0], "0" * 64)})
        check("risultati: sha256 diverso da quello atteso rifiutato", raises(lambda: rr.read_results(bad, rr.Checks())))
        before = {n: hashlib.sha256(Path(p).read_bytes()).hexdigest() for n, (p, _) in paths.items()}
        check("risultati: file letti e non modificati", before == {n: s for n, (_, s) in paths.items()})


def unit(uid, records, level="qa", pair=False, variants=("",)):
    return {"id": uid, "livello": level, "coppia": pair, "varianti": list(variants),
            "record_per_variante": dict(zip(variants, records)),
            "immagini": [{"record": r} for r in records]}


def voce(saved, shown, answer="3", fixed=None, noted=None, variants=("",), pair=False, non_riesco=False):
    first = {k: {"risposta": answer, "non_riesco": non_riesco, "leggibilita": "si", "note": ""} for k in variants}
    dopo = {"varianti": {k: {"corretta": "si", "coerente_con_annotazioni": "si", "rispondibile_da_immagine": "si",
                             "gravita": "nessun_difetto", "tipo_difetto": "", "note": "ok\nriga", "risposta_dopo_soluzione": ""}
                         for k in variants}}
    if pair:
        dopo["coppia"] = {"contratto_coppia": "si", "gravita": "nessun_difetto", "tipo_difetto": "", "note": ""}
    return {"prima": {"varianti": first, "salvata_il": saved, "salvataggi": 1}, "soluzione_mostrata_il": shown,
            "dopo": dopo, "tempo_ms": 12500, "prima_fissata": fixed, "esposta_senza_primo_giudizio": noted,
            "varianti": {k: {} for k in variants}, "bozza": first, "eventi": []}


T1, T2, T3, T4, T5 = (f"2026-11-05T10:0{i}:00.000Z" for i in range(1, 6))


def test_independence():
    units = [unit("A", ["r1"]), unit("B", ["r2"]), unit("C", ["r3"])]
    scene = {"r1": "s", "r2": "s", "r3": "t"}  # r1 e r2 hanno la stessa scena
    state = {"unita": {"A": voce(T1, T2, fixed={"quando": T2, "motivo": "soluzione_mostrata", "tramite": "A"}),
                       "B": voce(T3, T4, fixed={"quando": T3, "motivo": "record_gia_esposto", "tramite": "A"},
                                 noted={"quando": T2, "tramite": "A", "record": ["r1"], "collegamento": "stessa_scena"}),
                       "C": voce(T1, T5, fixed={"quando": T5, "motivo": "soluzione_mostrata", "tramite": "C"})},
             "esposizioni": [{"record": "r1", "quando": T2, "tramite": "A"}, {"record": "r2", "quando": T4, "tramite": "B"},
                             {"record": "r3", "quando": T5, "tramite": "C"}]}
    ind = rr.independence(units, state, scene)
    check("indipendenza: primo giudizio salvato prima della propria soluzione = indipendente",
          ind["A"]["indipendente"] and ind["C"]["indipendente"])
    check("indipendenza: record con la stessa scena esposto prima del primo giudizio = non indipendente",
          not ind["B"]["indipendente"] and ind["B"]["prima_esposizione"]["tramite"] == "A"
          and ind["B"]["prima_esposizione"]["collegamento"] == "stessa_scena", ind["B"])
    check("indipendenza: annotazioni e provenienza del blocco coerenti col ricalcolo",
          all(ind[k]["annotazione_coerente"] and ind[k]["blocco_coerente"] for k in "ABC"))
    state["unita"]["B"]["esposta_senza_primo_giudizio"] = None  # file che dichiara indipendente un'unità esposta
    check("indipendenza: un'annotazione di indipendenza smentita dalle date è vista",
          not rr.independence(units, state, scene)["B"]["annotazione_coerente"])


def test_rows():
    qa, scena = unit("Q", ["r1"]), unit("S", ["r2"], level="scena")
    pair = unit("P", ["p-a", "p-b"], pair=True, variants=("a", "b"))
    pkg = {"campione": [qa, pair], "mirati": [scena],
           "righe_esiti": [{"unita": "Q", "variante": "", "record": "r1"}, {"unita": "S", "variante": "", "record": "r2"},
                           {"unita": "P", "variante": "a", "record": "p-a"}, {"unita": "P", "variante": "b", "record": "p-b"},
                           {"unita": "P", "variante": "coppia", "record": "p-a|p-b"}]}
    state = {"unita": {"Q": voce(T1, T2, answer="(5, -5)"), "S": voce(T1, T2, answer=""),
                       "P": voce(T1, T2, variants=("a", "b"), pair=True, non_riesco=True)}}
    rows = rr.expected_rows(pkg, state)
    check("righe: prima risposta e giudizio finale conservati", rows[0] == [
        "Q", "", "r1", "(5, -5)", "si", "si", "si", "non_applicabile", "non_applicabile", "nessun_difetto", "", "13",
        "ok\nriga"], rows[0])
    check("righe: caso sulla scena senza risposta e risposta non data nelle coppie",
          rows[1][3] == "nessuna_risposta_caso_sulla_scena" and rows[2][3] == rows[3][3] == "non_riesco_a_rispondere")
    check("righe: contratto della coppia nella sua riga", rows[4][3:9] == ["non_applicabile"] * 5 + ["si"], rows[4])
    state["unita"]["Q"]["esposta_senza_primo_giudizio"] = {"quando": T1, "tramite": "S", "record": ["r1"],
                                                            "collegamento": "stessa_scena"}
    row = rr.expected_rows(pkg, state)[0]
    check("righe: risposta non indipendente spostata nelle note, col motivo",
          row[3] == "non_indipendente" and row[12].startswith("[primo giudizio non indipendente: r1 esposto il")
          and "risposta: (5, -5)" in row[12], row[12][:80])
    check("rr.js_round come Math.round di JavaScript", rr.js_round(12.5) == 13 and rr.js_round(2.4999) == 2)


def test_decontam():
    d = {"id": "D-1", "record": "x", "materiale": "completo",
         "elenco": [{"id": "D-1-1", "immagine": "a.png", "problemi": ""}, {"id": "D-1-2", "immagine": "b.png", "problemi": ""}]}
    v = {"esito": "no", "decisiva": "", "note": "", "corrispondenze": {"D-1-1": {"esame": "contenuto_diverso", "note": ""},
                                                                    "D-1-2": {"esame": "incerto", "note": "dubbio"}}}
    check("decontaminazione: no con una corrispondenza incerta è un errore", rr.decontam_errors(d, v) != [])
    v["corrispondenze"]["D-1-2"]["esame"] = "contenuto_diverso"
    check("decontaminazione: no con tutte le corrispondenze diverse e materiale completo", rr.decontam_errors(d, v) == [])
    v2 = {**v, "esito": "si", "decisiva": ""}
    check("decontaminazione: sì senza corrispondenza decisiva è un errore", rr.decontam_errors(d, v2) != [])
    pkg = {"decontaminazione": [d], "righe_esiti_decontaminazione": [{"unita": "D-1", "record": "x"}]}
    row = rr.expected_decontam_rows(pkg, {"decontaminazione": {"D-1": v}})[0]
    check("decontaminazione: riga con esami e note per corrispondenza",
          row == ["D-1", "x", "no", "2/2: D-1-1=contenuto_diverso; D-1-2=contenuto_diverso", "", "D-1-2: dubbio"], row)


def test_answers():
    same = [("(5, -5)", "(5,-5)"), ("12", "12.0"), ("Above", "above"), ("60°", "60"), ("yes.", "yes")]
    check("risposte: equivalenze di scrittura riconosciute",
          all(rr.compare_answer({"risposta": a, "non_riesco": False}, b) == "equivalente_nella_scrittura" for a, b in same))
    check("risposte: uguali, diverse e non date distinte",
          rr.compare_answer({"risposta": "left", "non_riesco": False}, "left") == "uguale"
          and rr.compare_answer({"risposta": "left", "non_riesco": False}, "right") == "diversa"
          and rr.compare_answer({"risposta": "", "non_riesco": True}, "3") == "non_riesco")


def rec(key, group, state="quarantena", motives=None, outcome="nessuna_corrispondenza", split="train", insieme="train"):
    return {"record": key, "record_id": "id-" + key, "gruppo": group, "split": split, "insieme": insieme,
            "famiglia": "function", "coppia": None, "stato": state, "motivo": motives or {},
            "evidenze": [{"controllo": "decontaminazione_pannello", "esito": outcome}]}


def test_propose():
    D = {"decontaminazione_pannello": ["gruppo con un candidato"]}
    V = {"incerto": ["testi_sovrapposti g1 g2"]}
    records = [rec("c1", "G1", motives=D, outcome="candidato"), rec("c2", "G1", motives=D, outcome="candidato"),
               rec("m3", "G1", motives=D),  # quarantena ereditata dal gruppo
               rec("c4", "G2", motives=D, outcome="candidato"), rec("c5", "G2", motives=D, outcome="candidato"),
               rec("q9", "G2", motives={**D, **V}),
               rec("c6", "G3", "escluso", {**D, "contaminazione": ["stato del test"]}, "candidato"),
               rec("q7", "G5", motives=V), rec("q8", "G6", motives=V),
               rec("q12", "G8", motives={"incerto": ["testi_sovrapposti g1 g2", "testo_piccolo g3"]}),
               rec("c10", "G4", motives=D, outcome="candidato"), rec("a11", "G7", "ammesso")]
    human = {"c1": "no", "c2": "no", "c4": "no", "c5": "incerto", "c6": "no", "c10": "si"}
    visual = {"q7": (True, {}, "M-1"), "q8": (False, {}, "M-2"), "q9": (True, {}, "M-3"), "q12": (True, {}, "M-4")}
    rows, groups = rr.propose(records, human, visual, {})
    by = {r["record"]: r for r in rows}
    check("proposta: gruppo con tutti i candidati diversi, anche il membro con la quarantena ereditata",
          all(by[k]["categoria"] == "riammissione_proposta" and by[k]["motivi_rimovibili"] == ["decontaminazione_pannello"]
              for k in ("c1", "c2", "m3")))
    check("proposta: un candidato incerto blocca tutto il gruppo, anche il record con la quarantena visiva approvata",
          all(by[k]["categoria"] == "irrisolto" for k in ("c4", "c5", "q9"))
          and by["q9"]["motivi"]["incerto"] == "riesaminabile_dai_giudizi", by["q9"])
    check("proposta: l'esclusione per contaminazione resta anche col candidato risolto",
          by["c6"]["categoria"] == "resta_bloccato" and by["c6"]["blocchi_rimanenti"] == ["contaminazione"]
          and by["c6"]["stato_proposto"] == "escluso")
    check("proposta: quarantena visiva tolta solo coi giudizi positivi",
          by["q7"]["categoria"] == "riammissione_proposta" and by["q8"]["categoria"] == "resta_bloccato")
    check("proposta: quarantena con un dettaglio non esaminato dai casi mirati (testo piccolo) resta",
          by["q12"]["categoria"] == "resta_bloccato" and by["q12"]["motivi"]["incerto"] == "valido")
    check("proposta: contaminazione confermata proposta come esclusione del gruppo",
          by["c10"]["categoria"] == "esclusione_proposta" and by["c10"]["stato_proposto"] == "escluso")
    check("proposta: record ammessi senza motivi fuori dall'elenco; split e gruppi invariati",
          "a11" not in by and all((r["split"], r["gruppo"]) == (x["split"], x["gruppo"])
                                  for r in rows for x in records if x["record"] == r["record"]))
    rows, groups = rr.propose(records, human, visual, {"c5": "contenuto_diverso"})
    by = {r["record"]: r for r in rows}
    check("proposta: il riesame tecnico resta separato e chiede approvazione",
          by["c5"]["categoria"] == "riammissione_dopo_riesame_tecnico" and by["c5"]["esito_umano"] == "incerto"
          and by["c5"]["stato_proposto"] == "quarantena"
          and {g["gruppo"]: g["stato_decontaminazione"] for g in groups}["G2"] == "risolvibile_col_riesame_tecnico")


def test_technical():
    incerti = {"D-1": "x", "D-2": "y"}
    with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
        p = Path(tmp) / "r.tsv"
        head = "unita\trecord\tconclusione\tmotivazione\n"
        p.write_text(head + "D-1\tx\tcontenuto_diverso\tdomande diverse\n", encoding="utf-8")
        check("riesame: conclusione letta per record", rr.read_technical(p, incerti) == {"x": "contenuto_diverso"})
        for name, body in (("unità non incerta", "D-9\tz\tirrisolto\tm\n"), ("record diverso", "D-1\ty\tirrisolto\tm\n"),
                           ("conclusione sconosciuta", "D-1\tx\tforse\tm\n"), ("motivazione vuota", "D-1\tx\tirrisolto\t \n"),
                           ("riga ripetuta", "D-1\tx\tirrisolto\tm\nD-1\tx\tirrisolto\tm\n")):
            p.write_text(head + body, encoding="utf-8")
            check(f"riesame: {name} rifiutata", raises(lambda: rr.read_technical(p, incerti)))


if __name__ == "__main__":
    test_tsv()
    test_read_results()
    test_independence()
    test_rows()
    test_decontam()
    test_answers()
    test_propose()
    test_technical()
    print(f"\n{'falliti: ' + ', '.join(FAILS) if FAILS else 'falliti: nessuno'}")
    sys.exit(1 if FAILS else 0)
