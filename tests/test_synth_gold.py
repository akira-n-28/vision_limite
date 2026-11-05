#!/usr/bin/env python3
"""Test CPU della preparazione della verifica delle golden answer dei sintetici v4 (scripts/synth_gold_sample.py), su
record e layout piccoli costruiti qui. Solo in un job (slurm/tests_cpu.sbatch o slurm/synth_gold_sample.sbatch).

Coprono: strati e unità del campione (QA costruite dal loader per train e dev, tutte le QA del test, coppie intere),
ordine per rango riproducibile e annidato, esclusione dei record non ammessi; limite esatto con zero errori uguale a
quello di S2 §2.8; margini calcolati dal layout salvato (testo più basso, riquadri vicini, vertice estremo,
coordinate senza etichetta, intersezioni e grafici vicini); scelta dei k casi con la misura più piccola.

Dal 2026-11-04 anche: prompt esportato uguale a quello di data.synth_qa_sample su scene vere del generatore, con
opzioni incorporate nella domanda (label_type, shape_type) e senza (count), domanda singola per test e coppie e loader
che rifiuta le coppie; soluzioni, opzioni e modulo con una riga per variante in una coppia con risposte diverse;
corrispondenze della decontaminazione prese tutte dalla scansione (12, anche l'undicesima oltre l'anteprima di 10),
riconciliate con summary.json e matches.jsonl, con materiale incompleto segnalato e identificativi incoerenti fermati.

Dal 2026-11-05 anche compare_previous su fogli fittizi piccoli, scritti come li scrive il job: dal formato 1 di
campione-1 (risposte aggregate separate per variante, varianti b assenti dichiarate senza confronto), formato attuale
identico, sola variante b cambiata in una coppia casuale e in una mirata, varianti mancanti, aggiunte, ripetute o col
record sbagliato, opzioni della sola variante b, e i controlli già presenti su ranghi, ordine, manifest e immagini."""
from __future__ import annotations

import io
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import synth_gold_sample as gs  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl import synth  # noqa: E402
from limite_vl.data import synth_qa_sample  # noqa: E402
from limite_vl.f1c import QA_ONE, pick_questions  # noqa: E402

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


def tiny_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, format="PNG")
    return buf.getvalue()


def scene_files(key: str, family: str) -> tuple[dict, dict, dict]:
    """(meta, file del campione, record del manifest) di una scena vera del generatore v4, senza disegno."""
    qa = synth.questions(synth.make_spec(key, family))
    meta = {"id": key, "family": family, "qa": qa, "max_patches": 784}
    d = mf.Digests()
    rec = {"record": key, "insieme": "train", "split": "train", "stato": "ammesso",
           "qa": [{"tipo": q["type"], "sha256": mf.qa_digest(d, q)} for q in qa]}
    return meta, {"json": json.dumps(meta).encode(), "png": tiny_png()}, rec


def loader_text(key: str, files: dict) -> str:
    return synth_qa_sample(key, files).messages[0]["content"][1]["text"]


def test_prompts():
    embedded = plain = None
    for i in range(400):  # scene vere: una con label_type fra più domande, una con una sola domanda count
        key = f"synth-{i:07d}"
        meta, files, rec = scene_files(key, "shapes")
        picked = pick_questions(key, list(range(len(meta["qa"]))))
        types = [meta["qa"][p]["type"] for p in picked]
        if embedded is None and "label_type" in types and len(picked) > 1:
            embedded = (key, meta, files, rec, picked[types.index("label_type")], picked)
        if plain is None and types == ["count"]:
            plain = (key, meta, files, rec, picked[0], picked)
        if embedded and plain:
            break
    check("prompt: trovate una scena con label_type fra più domande e una con la sola count", embedded and plain)
    if not (embedded and plain):
        return
    key, meta, files, rec, pos, picked = embedded
    p = gs.prompt_for(rec, meta, files, pos)
    q = meta["qa"][pos]
    n = picked.index(pos) + 1
    check("prompt con opzioni incorporate: uguale a quello del loader (data.synth_qa_sample)",
          p["prompt"] == loader_text(key, files) and p["origine_prompt"] == "loader", p["prompt"][:120])
    check("prompt con opzioni incorporate: la QA campionata è la domanda numero_nel_prompt",
          p["numero_nel_prompt"] == n and f"\n{n}. {q['q']}" in p["prompt"]
          and len(p["domande_nel_prompt"]) == len(picked) and p["domande_nel_prompt"][n - 1]["posizione"] == pos)
    check("prompt con opzioni incorporate: le opzioni sono nel testo della domanda, nessun target aggiunto",
          gs.options_in_question(q) and all(o in p["prompt"] for o in q["options"]) and " / " not in p["prompt"]
          and "answer" not in p["prompt"].lower().replace("answer each question with a short answer", ""))
    check("prompt: nella cella TSV gli a capo sono la sequenza \\n, il testo esatto resta in prompts.jsonl",
          "\n" not in gs.tsv_text(p["prompt"]) and gs.tsv_text(p["prompt"]).replace("\\n", "\n") == p["prompt"])
    key, meta, files, rec, pos, _ = plain
    p = gs.prompt_for(rec, meta, files, pos)
    q = meta["qa"][pos]
    check("prompt senza opzioni incorporate: uguale a quello del loader, domanda singola senza le opzioni",
          p["prompt"] == loader_text(key, files) == QA_ONE + q["q"] and p["numero_nel_prompt"] == 1
          and not gs.options_in_question(q) and " / ".join(q["options"]) not in p["prompt"], p["prompt"])
    other = next(i for i in range(len(meta["qa"])) if i != pos)
    p = gs.prompt_for(rec, meta, files, other)
    check("prompt: una QA che il loader non sceglie è segnalata e mostrata da sola",
          p["origine_prompt"] == "domanda_singola_qa_non_scelta" and not p["consegnato_dal_loader"]
          and p["prompt"] == QA_ONE + meta["qa"][other]["q"])
    p = gs.prompt_for({**rec, "stato": "quarantena"}, meta, files, None)
    check("prompt: caso sulla scena intera con le domande del loader, non consegnato se la scena non è ammessa",
          p["origine_prompt"] == "loader_scena" and not p["consegnato_dal_loader"] and p["prompt"] == loader_text(key, files))
    test_rec = {**rec, "insieme": "test", "split": "test"}
    p = gs.prompt_for(test_rec, meta, files, 2)
    check("prompt del test: la QA da sola col prefisso di qa_messages",
          p["prompt"] == QA_ONE + meta["qa"][2]["q"] and p["origine_prompt"] == "domanda_singola_test")
    pair_q = {"type": "side_x", "q": "Is point A above or below the x-axis?", "options": ["above", "below"],
              "answer": "above"}
    pair_meta = {"id": "synth-pair-00000-a", "family": "coord", "qa": [], "max_patches": 784,
                 "pair": {**pair_q, "id": "synth-pair-00000", "variant": "a"}}
    pair_files = {"json": json.dumps(pair_meta).encode(), "png": tiny_png()}
    pair_rec = {"record": "synth-pair-00000-a", "insieme": "coppie", "split": "test", "stato": "ammesso",
                "qa": [{"tipo": "side_x", "sha256": "0" * 64}]}
    p = gs.prompt_for(pair_rec, pair_meta, pair_files, 0)
    try:
        gs.loader_prompt("synth-pair-00000-a", pair_files)
        refused = False
    except ValueError:
        refused = True
    check("prompt delle coppie: domanda singola, e il loader di training non costruisce il prompt di una coppia",
          p["prompt"] == QA_ONE + pair_q["q"] and p["origine_prompt"] == "domanda_singola_coppia" and refused)


def test_pairs():
    def pair_record(v):
        return {"record": f"synth-pair-00007-{v}", "record_id": f"{v * 64}", "insieme": "coppie", "split": "test",
                "stato": "ammesso", "stato_scena_sha256": f"{'1' if v == 'a' else '2'}" * 64}

    rs = [pair_record("a"), pair_record("b")]
    metas = {r["record"]: {"pair": {"variant": r["record"][-1]}} for r in rs}
    qs = [{"type": "side_len", "q": "What is the length of side AB?", "options": ["9", "11"], "answer": a}
          for a in ("9", "11")]
    sols, opts, form = gs.unit_rows("M-coppie_per_famiglia-001", rs, metas, qs, 0, lambda r, pos: "ok")
    check("coppie: una soluzione per variante, con etichette a e b, record e risposte diverse",
          [(s["variante"], s["record"], s["risposta"]) for s in sols]
          == [("a", "synth-pair-00007-a", "9"), ("b", "synth-pair-00007-b", "11")]
          and sols[0]["record_id"] != sols[1]["record_id"] and sols[0]["stato_scena_sha256"] != sols[1]["stato_scena_sha256"],
          str(sols))
    check("coppie: opzioni per variante e modulo con a, b e la riga della coppia per il contratto",
          [o["variante"] for o in opts] == ["a", "b"] and [f["variante"] for f in form] == ["a", "b", "coppia"]
          and form[2]["record"] == "synth-pair-00007-a|synth-pair-00007-b" and "contratto_coppia" not in form[2]
          and form[0]["contratto_coppia"] == "non_applicabile")
    bad = {**metas, "synth-pair-00007-a": {"pair": {"variant": "b"}}}
    try:
        gs.unit_rows("x", rs, bad, qs, 0, lambda r, pos: "ok")
        refused = False
    except ValueError:
        refused = True
    check("coppie: una variante dichiarata diversa dal suffisso della chiave è rifiutata", refused)
    one = gs.unit_rows("C-coord-001", [{**pair_record("a"), "insieme": "train", "split": "train"}],
                       {"synth-pair-00007-a": {}}, [qs[0]], 3, lambda r, pos: "ok")
    check("unità non di coppia: una riga, senza variante", [len(x) for x in one] == [1, 1, 1]
          and one[0][0]["variante"] == "" and one[2][0]["variante"] == "")


def test_decontam_full():
    key = "synth-0000123"
    r = {"record": key, "record_id": "c" * 64, "insieme": "train", "split": "train", "stato": "quarantena",
         "famiglia": "function", "shard": "data/shards/synth/v4/train/synth-train-00001.tar", "indice": 5,
         "immagini": [{"sha256": "ab" * 32}]}
    details = [{"match": "image", "distance": 2, "benchmark": "dynamath", "bench_id": f"id{j}",
                "item": f"sample1:{j}", "pos": 0, "elemento": f"dynamath/sample1:{j}"} for j in range(12)]
    preview = [{"tipo": d["match"], "elemento": d["elemento"], "posizione": d["pos"], "distanza": d["distance"],
                "ruolo": "test", "gruppo_pannello": "g", "metrica_primaria": True} for d in details[:10]]
    ev = {"controllo": gs.CHECK, "esito": "candidato", "esito_gruppo": "candidato", "corrispondenze": {"image": 12},
          "elementi": preview, "rif": {"file": "scan/records.jsonl.gz", "sha256": "f" * 64}}
    row = {"key": key, "shard": r["shard"], "posizione": 5, "byte_sha256": "ab" * 32, "corrispondenze": {"image": 12},
           "dettagli": details, "dettagli_completi": True}
    entries = {(d["elemento"], 0): {"id": d["bench_id"], "pixel_sha256": f"p{min(j, 10)}"}  # 10 e 11: stessi pixel
               for j, d in enumerate(details)}
    elements = {d["elemento"]: {"ruolo": "test", "gruppo": "g", "metrica_primaria": True} for d in details}
    files = {(d["elemento"], 0): {"file": f"immagini/pannello/{j}.png", "sha256": f"s{j}"} for j, d in enumerate(details)}
    unit = {"unita": "D-001", "rango": "0" * 64, "records": [r], "evidenza": ev}
    recs, matches, fatal = gs.decontam_tables([unit], {key: row}, entries, elements, files)
    check("decontaminazione: tutte le 12 corrispondenze della scansione, non le 10 dell'anteprima",
          len(matches) == 12 and recs[0]["corrispondenze_esportate"] == 12 and recs[0]["nell_anteprima_del_manifest"] == 10
          and not fatal, f"{len(matches)} {fatal}")
    eleventh = matches[10]
    check("decontaminazione: l'undicesima corrispondenza esportata, con record, elemento, posizione ed evidenza",
          eleventh["corrispondenza"] == "D-001-011" and eleventh["record"] == key and eleventh["record_id"] == "c" * 64
          and eleventh["elemento"] == "dynamath/sample1:10" and eleventh["posizione_immagine"] == 0
          and eleventh["voce_indice"] == "id10" and eleventh["nell_anteprima_del_manifest"] == "no"
          and eleventh["immagine_pannello"] == "immagini/pannello/10.png" and matches[9]["nell_anteprima_del_manifest"] == "sì"
          and "dynamath/sample1:10" in recs[0]["elementi"] and recs[0]["materiale"] == "completo", str(eleventh))
    other = {"key": "synth-0000124", "corrispondenze": {}, "dettagli": [], "dettagli_completi": True}
    lines = [{"key": key, "n_matches": 12, **d} for d in details]
    checks, counts = gs.reconcile({key: row, "synth-0000124": other}, {"matches_per_benchmark": {"dynamath/image": 12}},
                                  lines, {key}, matches)
    check("decontaminazione: conteggi riconciliati con summary.json e matches.jsonl",
          all(c["esito"] for c in checks), [c["controllo"] for c in checks if not c["esito"]])
    check("decontaminazione: corrispondenze, voci e immagini distinte (due voci con gli stessi pixel)",
          counts["corrispondenze_esportate"] == 12 and counts["duplicati_di_evidenza"] == 0
          and counts["voci_del_pannello_distinte"] == 12 and counts["immagini_del_pannello_distinte_per_pixel"] == 11
          and counts["corrispondenze_oltre_l_anteprima"] == 2, str(counts))
    checks, _ = gs.reconcile({key: row}, {"matches_per_benchmark": {"dynamath/image": 12}}, lines[:10], {key},
                             matches[:10])
    check("decontaminazione: un foglio con le sole 10 dell'anteprima non si riconcilia",
          not all(c["esito"] for c in checks))
    cut = {**row, "dettagli": details[:7], "dettagli_completi": False}  # scansione con --max-lines 7
    cut_unit = {**unit, "evidenza": {**ev, "elementi": preview[:7]}}
    recs, matches, fatal = gs.decontam_tables([cut_unit], {key: cut}, entries, elements,
                                              {k: v for k, v in files.items() if k[0] != "dynamath/sample1:3"})
    check("decontaminazione: dettagli tagliati e immagine non ritrovata rendono il record incompleto, senza fermare",
          not fatal and recs[0]["materiale"].startswith("incompleto") and "7 su 12" in recs[0]["materiale"]
          and "non ritrovate 1 su 7" in recs[0]["materiale"] and matches[3]["problemi"], recs[0]["materiale"])
    recs, _, fatal = gs.decontam_tables([unit], {}, entries, elements, files)
    check("decontaminazione: riga della scansione assente segnalata", recs[0]["materiale"].startswith("incompleto")
          and "assente" in recs[0]["materiale"] and recs[0]["corrispondenze_esportate"] == 0)
    _, _, fatal = gs.decontam_tables([unit], {key: {**row, "posizione": 6}}, entries, elements, files)
    _, _, fatal2 = gs.decontam_tables([{**unit, "evidenza": {**ev, "elementi": preview[1:]}}], {key: row}, entries,
                                      elements, files)
    check("decontaminazione: posizione o anteprima diverse dall'evidenza sono incoerenze che fermano il foglio",
          any("posizione" in f for f in fatal) and any("anteprima" in f for f in fatal2))


# ---------------------------------------------------------------- confronto con un campione precedente (fittizio)

MSHA = "e" * 64
DESIGN = {"coord": {"N": 100, "n": 1, "p_inclusione": 0.01, "unita": "QA"},
          "coppie": {"N": 10, "n": 1, "p_inclusione": 0.1, "unita": "coppia"}}


def fx_record(key, insieme="train", split="train", stato="ammesso"):
    return {"record": key, "record_id": mf.sha256_text("id/" + key), "insieme": insieme, "split": split,
            "stato": stato, "stato_scena_sha256": mf.sha256_text("scena/" + key)}


def fx_qa(tipo, text, options, answer):
    return {"type": tipo, "q": text, "options": options, "answer": answer}


def gold_tables() -> dict:
    """Tabelle nuove fittizie nel formato attuale, con soluzioni e opzioni da unit_rows come in main(): un'unità
    casuale semplice e una coppia casuale; una coppia mirata, una regola nuova e un caso sulla scena intera; un
    candidato della decontaminazione."""
    def pair(p):
        return [fx_record(f"synth-pair-{p:05d}-{v}", "coppie", "test") for v in "ab"]

    side_x = "Is point A above or below the x-axis?"
    units = [
        ("C-coord-001", "campione", [fx_record("synth-0000001")], 2,
         [fx_qa("quadrant", "In which quadrant is point A?", ["I", "II", "III", "IV"], "II")]),
        ("C-coppie-001", "campione", pair(1), 0,
         [fx_qa("side_x", side_x, ["above", "below"], a) for a in ("above", "below")]),
        ("M-coppie_per_famiglia-001", "mirati", pair(2), 0,
         [fx_qa("side_len", "What is the length of side AB?", ["9", "11"], a) for a in ("9", "11")]),
        ("M-regole_nuove_v4-001", "mirati", [fx_record("synth-0000005")], 1,
         [fx_qa("label_type", "What shape is labeled B: circle, square or triangle?", ["circle", "square", "triangle"],
                "square")]),
        ("M-quarantena-001", "mirati", [fx_record("synth-0000004", stato="quarantena")], None, None),
    ]
    metas = {r["record"]: {"pair": {"variant": r["record"][-1]}} for _, _, rs, _, _ in units for r in rs
             if r["insieme"] == "coppie"}
    tables = {"campione": [], "mirati": [], "candidati": [], "soluzioni": [], "opzioni": []}
    for uid, table, rs, pos, qs in units:
        row = {"unita": uid, "record": "|".join(r["record"] for r in rs),
               "record_id": "|".join(r["record_id"] for r in rs),
               "posizione_qa": "" if pos is None else pos, "tipo": qs[0]["type"] if qs else "scena",
               "domanda": qs[0]["q"] if qs else "scena intera", "famiglia": "coord",
               "immagini": "|".join(f"immagini/{r['record']}.png" for r in rs)}
        if table == "campione":
            row.update({"strato": "coppie" if len(rs) == 2 else "coord", "rango": mf.sha256_text(uid), "N": 100,
                        "n": 1, "p_inclusione": "1/100", "insieme": rs[0]["insieme"], "split": rs[0]["split"]})
        else:
            row.update({"categoria": uid[2:-4], "motivo": "caso fittizio", "misura": "",
                        "insieme": "|".join(r["insieme"] for r in rs), "split": "|".join(r["split"] for r in rs),
                        "stato": "|".join(r["stato"] for r in rs)})
        tables[table].append(row)
        sols, opts, _ = gs.unit_rows(uid, rs, metas, qs or [None] * len(rs), pos, lambda r, p: "ok")
        tables["soluzioni"] += sols
        tables["opzioni"] += opts
    tables["candidati"].append({"unita": "D-001", "rango": mf.sha256_text("D-001"), "record": "synth-0000009",
                                "record_id": mf.sha256_text("id/synth-0000009"), "insieme": "train", "split": "train",
                                "stato": "quarantena", "famiglia": "coord", "esito": "candidato",
                                "esito_gruppo": "candidato", "corrispondenze": '{"image": 1}',
                                "elementi": "image dynamath/x:1 pos 0 d=2 test",
                                "immagini": "immagini/synth-0000009.png|immagini/pannello/ab.png"})
    return tables


def v1_solutions(tables: dict) -> list[dict]:
    """soluzioni.tsv del formato 1 (campione-1, codice 87c4108): una riga per unità; le coppie casuali con le risposte
    aggregate "a: x | b: y", le coppie mirate con la sola variante a."""
    by_unit = defaultdict(list)
    for s in tables["soluzioni"]:
        by_unit[s["unita"]].append(s)
    rows = []
    for uid, ss in by_unit.items():
        if uid.startswith("C-") and len(ss) == 2:
            rows.append({"unita": uid, "risposta": " | ".join(f"{s['variante']}: {s['risposta']}" for s in ss),
                         "esito_checker": "|".join(s["esito_checker"] for s in ss),
                         "stato_scena_sha256": "|".join(s["stato_scena_sha256"] for s in ss)})
        else:
            rows.append({k: ss[0][k] for k in gs.SOLUTION_COLUMNS[gs.FORMAT_V1]})
    return rows


def sheet_images(tables: dict) -> list[str]:
    return [p for name in ("campione", "mirati", "candidati") for row in tables[name] for p in row["immagini"].split("|")
            if p.startswith("immagini/synth-")]


def write_images(where: Path, tables: dict, changed: dict | None = None) -> None:
    (where / "immagini").mkdir(parents=True, exist_ok=True)
    for p in sheet_images(tables):
        (where / p).write_bytes((changed or {}).get(p, f"png fittizio {p}".encode()))


def write_sheet(where: Path, tables: dict, fmt: str = gs.SCHEMA, files_fmt: str | None = None,
                manifest_sha: str = MSHA) -> None:
    """Foglio precedente su disco: report.json con lo schema fmt e i file nel formato files_fmt (di solito lo stesso)."""
    files_fmt = files_fmt or fmt
    where.mkdir(parents=True)
    report = {"schema": fmt, "seme": 0, "audit": gs.AUDIT, "manifest": {"manifest_sha256": manifest_sha},
              "campione": {"strati": DESIGN}}
    (where / "report.json").write_text(json.dumps(report), encoding="utf-8")

    def columns(rows):
        return list(dict.fromkeys(k for r in rows for k in r)) or ["unita"]

    c, m = tables["campione"], tables["mirati"]
    if files_fmt == gs.FORMAT_V1:  # opzioni del primo record nella riga dell'unità
        first = {}
        for o in tables["opzioni"]:
            first.setdefault(o["unita"], o["opzioni"])
        c = [{**r, "opzioni": first.get(r["unita"], "")} for r in c]
        m = [{**r, "opzioni": first.get(r["unita"], "")} for r in m]
        gs.write_table(where / "soluzioni.tsv", v1_solutions(tables), list(gs.SOLUTION_COLUMNS[gs.FORMAT_V1]))
    else:
        gs.write_table(where / "soluzioni.tsv", tables["soluzioni"], list(gs.SOLUTION_COLUMNS[gs.SCHEMA]))
        gs.write_table(where / "opzioni.tsv", tables["opzioni"], list(gs.OPTION_COLUMNS))
    gs.write_table(where / "campione.tsv", c, columns(c))
    gs.write_table(where / "mirati.tsv", m, columns(m))
    gs.write_table(where / "candidati_decontaminazione.tsv", tables["candidati"], columns(tables["candidati"]))
    write_images(where, tables)


def compare(prev_tables: dict, new_tables: dict | None = None, *, changed_images: dict | None = None, **sheet) -> dict:
    new_tables = new_tables or gold_tables()
    with tempfile.TemporaryDirectory() as tmp:
        prev, out = Path(tmp) / "precedente", Path(tmp) / "nuovo"
        write_sheet(prev, prev_tables, **sheet)
        write_images(out, new_tables, changed_images)
        return gs.compare_previous(prev, new_tables, DESIGN, 0, MSHA, out)


def row_of(tables: dict, kind: str, uid: str, variant: str) -> dict:
    return next(r for r in tables[kind] if r["unita"] == uid and r["variante"] == variant)


def drop(tables: dict, kind: str, uid: str, variant: str) -> None:
    tables[kind] = [r for r in tables[kind] if not (r["unita"] == uid and r["variante"] == variant)]


def changed_keys(result: dict, kind: str) -> list:
    return [(d["unita"], d["variante"], d["campi"]) for d in result["differenze"].get(kind, [])]


def problems(result: dict, which: str) -> list[str]:
    return result["differenze"].get(f"struttura_{which}", [])


def test_compare_previous():
    check("confronto: risposta aggregata del formato 1 separata per variante, rifiutata se ambigua",
          gs.split_aggregated({"risposta": "a: 9 | b: 11", "esito_checker": "ok|ok", "stato_scena_sha256": "h|k"})
          == {"a": {"risposta": "9", "esito_checker": "ok", "stato_scena_sha256": "h"},
              "b": {"risposta": "11", "esito_checker": "ok", "stato_scena_sha256": "k"}}
          and gs.split_aggregated({"risposta": "a: 9 | 10 | b: 11", "esito_checker": "ok|ok",
                                   "stato_scena_sha256": "h|k"}) is None
          and gs.split_aggregated({"risposta": "b: 11 | a: 9", "esito_checker": "ok|ok",
                                   "stato_scena_sha256": "h|k"}) is None)

    # vecchio -> nuovo, con le trasformazioni previste
    r = compare(gold_tables(), fmt=gs.FORMAT_V1)
    tr = r.get("trasformazioni") or {}
    check("confronto dal formato 1: selezione invariata con le sole trasformazioni previste",
          r["selezione_invariata"] and r["formati"] == {"precedente": gs.FORMAT_V1, "nuovo": gs.SCHEMA},
          json.dumps(r["differenze"], ensure_ascii=False)[:300])
    check("confronto dal formato 1: risposte aggregate della coppia casuale separate e confrontate per variante",
          tr.get("risposte_aggregate_separate") == ["C-coppie-001"] and r.get("soluzioni_confrontate") == 5
          and r.get("opzioni_confrontate") == 4, str(tr))
    check("confronto dal formato 1: varianti b assenti dichiarate aggiunte senza confronto, confronto non completo",
          tr.get("varianti_aggiunte_senza_confronto") == {"soluzioni": ["M-coppie_per_famiglia-001/b"],
                                                          "opzioni": ["C-coppie-001/b", "M-coppie_per_famiglia-001/b"]}
          and r["confronto_completo"] is False and r["seconde_varianti_aggiunte"] == 1, str(tr))
    old = gold_tables()
    row_of(old, "soluzioni", "C-coppie-001", "b")["risposta"] = "left"
    r = compare(old, fmt=gs.FORMAT_V1)
    check("confronto dal formato 1: una risposta b diversa nella riga aggregata è rilevata sulla variante b",
          not r["selezione_invariata"] and changed_keys(r, "soluzioni") == [("C-coppie-001", "b", ["risposta"])],
          str(r["differenze"]))

    # nuovo -> nuovo
    r = compare(gold_tables())
    tr = r.get("trasformazioni") or {}
    check("confronto nel formato attuale: fogli identici danno selezione invariata e confronto completo",
          r["selezione_invariata"] and r["confronto_completo"] is True and r.get("soluzioni_confrontate") == 6
          and r.get("opzioni_confrontate") == 6 and tr.get("risposte_aggregate_separate") == []
          and tr.get("varianti_aggiunte_senza_confronto") == {"soluzioni": [], "opzioni": []},
          json.dumps(r["differenze"], ensure_ascii=False)[:300])
    for uid, field, value in (("C-coppie-001", "risposta", "left"), ("M-coppie_per_famiglia-001", "esito_checker",
                                                                     "non_ammissibile"),
                              ("M-coppie_per_famiglia-001", "stato_scena_sha256", "0" * 64)):
        new = gold_tables()
        row_of(new, "soluzioni", uid, "b")[field] = value
        r = compare(gold_tables(), new)
        check(f"confronto nel formato attuale: solo {field} della variante b di {uid} cambiato, rilevato su (unita, b)",
              not r["selezione_invariata"] and changed_keys(r, "soluzioni") == [(uid, "b", [field])]
              and set(r["differenze"]) == {"soluzioni"}, str(r["differenze"]))
    new = gold_tables()
    drop(new, "soluzioni", "C-coppie-001", "b")
    r = compare(gold_tables(), new)
    check("confronto: variante mancante nel nuovo foglio rilevata",
          not r["selezione_invariata"] and r["differenze"].get("soluzioni_mancanti_nel_nuovo") == ["C-coppie-001/b"]
          and "soluzioni C-coppie-001/b: variante mancante" in problems(r, "nuova"), str(r["differenze"]))
    new = gold_tables()
    new["soluzioni"].append({**row_of(new, "soluzioni", "C-coord-001", ""), "variante": "a"})
    r = compare(gold_tables(), new)
    check("confronto: variante aggiunta a un'unità che non è una coppia rilevata come inattesa",
          not r["selezione_invariata"] and "soluzioni C-coord-001/a: variante inattesa" in problems(r, "nuova"),
          str(r["differenze"]))
    old = gold_tables()
    drop(old, "soluzioni", "M-coppie_per_famiglia-001", "b")
    r = compare(old)
    check("confronto nel formato attuale: una variante b assente nel precedente non è un'aggiunta prevista",
          not r["selezione_invariata"]
          and r["differenze"].get("soluzioni_aggiunte_non_previste") == ["M-coppie_per_famiglia-001/b"]
          and "soluzioni M-coppie_per_famiglia-001/b: variante mancante" in problems(r, "precedente"),
          str(r["differenze"]))
    new = gold_tables()
    new["soluzioni"].append(dict(row_of(new, "soluzioni", "C-coppie-001", "b")))
    r = compare(gold_tables(), new)
    check("confronto: variante ripetuta rilevata",
          not r["selezione_invariata"] and "soluzioni C-coppie-001/b: riga ripetuta" in problems(r, "nuova"),
          str(r["differenze"]))
    new = gold_tables()
    a, b = row_of(new, "soluzioni", "C-coppie-001", "a"), row_of(new, "soluzioni", "C-coppie-001", "b")
    b.update(record=a["record"], record_id=a["record_id"])
    r = compare(gold_tables(), new)
    check("confronto: variante b associata al record della variante a rilevata",
          not r["selezione_invariata"] and any(p.startswith("soluzioni C-coppie-001/b: associata al record")
                                               for p in problems(r, "nuova"))
          and changed_keys(r, "soluzioni") == [("C-coppie-001", "b", ["record", "record_id"])], str(r["differenze"]))
    for uid in ("C-coppie-001", "M-coppie_per_famiglia-001"):
        new = gold_tables()
        row_of(new, "opzioni", uid, "b")["opzioni"] = "above / below / on"
        r = compare(gold_tables(), new)
        check(f"confronto: opzioni della sola variante b di {uid} cambiate, rilevate su (unita, b)",
              not r["selezione_invariata"] and changed_keys(r, "opzioni") == [(uid, "b", ["opzioni"])]
              and set(r["differenze"]) == {"opzioni"}, str(r["differenze"]))

    # controlli già presenti: rango, ordine, manifest, immagini; formato
    new = gold_tables()
    new["campione"][0]["rango"] = "0" * 64
    r1 = compare(gold_tables(), new)
    new = gold_tables()
    new["campione"].reverse()
    r2 = compare(gold_tables(), new)
    r3 = compare(gold_tables(), manifest_sha="f" * 64)
    r4 = compare(gold_tables(), changed_images={"immagini/synth-pair-00002-b.png": b"altri byte"})
    check("confronto: restano i controlli su rango, ordine, manifest e byte delle immagini v4",
          set(r1["differenze"]) == {"campione"} and set(r2["differenze"]) == {"campione"}
          and set(r3["differenze"]) == {"manifest"} and set(r4["differenze"]) == {"immagini_v4"},
          str([sorted(x["differenze"]) for x in (r1, r2, r3, r4)]))
    r1 = compare(gold_tables(), fmt="synth-gold-sample/9")
    r2 = compare(gold_tables(), fmt=gs.FORMAT_V1, files_fmt=gs.SCHEMA)
    check("confronto: schema sconosciuto, o schema del formato 1 con i file del formato attuale, rifiutati",
          set(r1["differenze"]) == {"formato"} and set(r2["differenze"]) == {"formato"}
          and r1["confronto_completo"] is False and not r2["selezione_invariata"],
          str([r1["differenze"], r2["differenze"]])[:300])


if __name__ == "__main__":
    test_populations()
    test_bound()
    test_margins()
    test_best()
    test_prompts()
    test_pairs()
    test_decontam_full()
    test_compare_previous()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
