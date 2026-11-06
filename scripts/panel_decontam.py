#!/usr/bin/env python3
"""Decontaminazione esterna di UniMER-1M e MAVIS-Geometry contro pannello-v3 e versione nuova dei loro manifest
candidati (docs/PANNELLO_VALUTAZIONE.md §§3.1, 5-6; S1 §§3-5, 9; S2 §§3-4). Solo in un job SLURM CPU: manifest, shard,
snapshot, indice, pannello ed evidenze si leggono soltanto; si scrive solo in cartelle nuove.

Riusa le funzioni condivise (limite_vl.decontam, limite_vl.manifest, scripts/unimer_manifest.py,
scripts/mavis_manifest.py, scripts/synth_decontam.py, scripts/manifest_recheck.py, scripts/shard_mavis_geometry.py) e le
regole documentate: identità dei pixel nella convenzione dell'indice (rgb-su-bianco/1), pHash entro 4 bit, 10-gram di
normalize_words con max_df 25, formula visibile di run2 (latex.effective_label e visible_tokens) con la soglia dei 20
token. Gli shard contengono immagini ridimensionate (UniMER fino a 4x, MAVIS a 2304 patch), che non possono avere gli
stessi pixel di un originale: l'identità dei pixel si controlla sull'immagine originale dello snapshot (lo stesso tipo
di immagine indicizzato dal pannello), e anche sul PNG dello shard e sui ritagli, che sono le immagini viste dal modello.

Controlli per sorgente (ogni campione contro tutte le voci dell'indice, in ogni ruolo del pannello):
- unimer, formule (insieme `train`): pixel dell'originale `UniMER-1M/images/<indice>.png` e del PNG dello shard;
  formula visibile dell'etichetta contro le formule di UniMER-Test (dal pannello) e di OCRBench (risposte delle domande
  HME100K, dal parquet della revisione del pannello). Formula uguale con più di 20 token visibili: candidato; più
  corta: evidenza senza effetto sullo stato (una formula comune non identifica un problema). Il pHash non si usa: sulle
  formule, quasi bianche, non è informativo (regola del pannello, job 30039).
- unimer, righe ritagliate (insieme `righe`, stessa chiave e stesso record_id della formula): l'immagine intera (stesso
  PNG della formula) e il suo originale; i ritagli di tutte le fasce costruiti come data.unimer_line_sample, contro i
  pixel del pannello; le righe del LaTeX (i target di transcribe_line e transcribe_line_select) contro le formule. Il
  ritaglio si confronta solo per identità: lo stesso contenuto tagliato con margini diversi non si riconosce. Le righe
  sono identificate da (insieme, shard, posizione, chiave), mai dalla sola chiave.
- mavis: pixel e pHash dell'originale `<cartella>/<n>_caption.jpg` e del PNG dello shard; 10-gram dei testi di
  text_en (caption, diagram_generation, text_dominant_question, text_lite_question, vision_dominant_question,
  CoT_reasoning) e dei testi costruiti dal loader (descrizione canonica e domande di f1_mavis) contro i testi
  dell'indice. text_zh, text.jpg e vision_dominant.jpg non sono negli shard e non si confrontano. La frequenza di un
  10-gram nella sorgente non decide nulla (serve solo a ordinare i casi): una corrispondenza testuale si ignora solo se
  tutti i suoi 10-gram sono coperti da una regola esplicita di TEXT_RULES, ristretta al suo campo e verificata sul
  testo del record. Oggi c'è una sola regola: l'enumerazione delle etichette dei punti nella riga `Points:` della
  descrizione canonica del loader. La regola di panel-decontam/1 (10-gram in più di 25 record ignorati in ogni campo,
  job 30652) è abolita: `riesame` ricalcola le decisioni di una scansione già fatta e ne scrive la versione nuova.

Esiti per record: `corrispondenza_confermata` (pixel identici, in una delle immagini), `candidato` (pHash, testo o
formula lunga uguale: somiglianze da verificare, non prove), `non_verificabile` (un controllo richiesto non eseguito:
immagine assente o illeggibile, formula vuota), `nessuna_corrispondenza` (nei controlli eseguiti). Lo stato vale per il
gruppo del manifest (synth_decontam.decide, regola del 2026-11-02): confermata -> escluso, candidato o non verificabile
-> quarantena; la gravità non diminuisce mai, split e gruppi restano quelli del genitore. MAVIS: i compiti seguono lo
stato del record e shape_type resta almeno in quarantena (mavis_manifest.tasks_of).

Comandi:
  scan              shard del manifest letti e hashati in un passaggio, immagini originali dallo zip dello snapshot
                    (CRC dei membri controllato; sha256 dello zip calcolato nella scansione completa) -> --out con
                    records.jsonl.gz (una riga per campione, tutte le corrispondenze, nessun taglio) e summary.json.
                    --limit N: solo i primi N campioni di ogni shard (prova limitata).
  verifica          compatibilità e copertura (indice, pannello con tutti i ruoli, manifest, shard, zip, identificativi,
                    corrispondenze ricondotte all'elemento, riconciliazione con run2 per UniMER) -> --report/report.json;
                    con --partial su una prova limitata.
  manifest          la verifica completa, poi la versione nuova del manifest (--new, scritta una volta) e la sua verifica.
  verifica-manifest rilegge una versione derivata: il genitore con la sua verifica (unimer_manifest.verify o
                    mavis_manifest.verify, evidenze obbligatorie vincolate) e le transizioni dovute alla
                    decontaminazione, ricalcolate dalla scansione (per candidato-2 con la sua regola storica).
  riesame           MAVIS: decisioni testuali ricalcolate con TEXT_RULES su una scansione già fatta, senza rileggere
                    immagini: legge soltanto i JSON degli shard dei record con corrispondenze testuali (descrizione del
                    loader legata al manifest dal suo sha256, testi di contesto), scrive --out (regole, decisioni per
                    record e riepilogo, con la decisione precedente accanto alla nuova), la versione nuova del manifest
                    derivata dal precedente (--precedente) e la sua verifica (verifica-riesame).
  verifica-riesame  rilegge la versione del riesame: genitore e nonno con le loro verifiche, decisioni ricalcolate dalla
                    scansione e dalle descrizioni del riesame, transizioni, compiti, nessuna riammissione.

    srun $RUN_PYTHON -u -B scripts/panel_decontam.py scan unimer|mavis --manifest DIR --manifest-sha256 HEX \\
        --index DIR --index-sha256 HEX --out DIR [--limit N] [--workers N]
    srun $RUN_PYTHON -u -B scripts/panel_decontam.py verifica unimer|mavis --manifest DIR --manifest-sha256 HEX \\
        --index DIR --index-sha256 HEX --panel DIR --scan DIR --report DIR [--partial] [--overlap DIR] [--historical DIR]
    srun $RUN_PYTHON -u -B scripts/panel_decontam.py manifest ... --new DIR --version 2 --summary FILE \\
        (--check DIR --overlap DIR | --evidence DIR)
    srun $RUN_PYTHON -u -B scripts/panel_decontam.py riesame mavis --origine DIR --origine-sha256 HEX \\
        --manifest DIR --manifest-sha256 HEX --index DIR --index-sha256 HEX --panel DIR --scan DIR --evidence DIR \\
        --out DIR --new DIR --version 3 --report DIR --summary FILE
"""
from __future__ import annotations

import argparse
import difflib
import gc
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import re
import sys
import tarfile
import threading
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import manifest_recheck as mr  # noqa: E402
import mavis_manifest as mm  # noqa: E402
import synth_decontam as sd  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.data import open_image  # noqa: E402
from limite_vl.decontam import (IMAGE_CONVENTION, HashIndex, TextIndex, hash_image_bytes, index_convention,  # noqa: E402
                                normalize_words, pixel_sha256, word_ngrams)
from limite_vl.f1_mavis import describe, mavis_questions, mavis_truth  # noqa: E402
from limite_vl.latex import despace, effective_label, visible_tokens  # noqa: E402
from limite_vl.processing import capped_size, to_rgb  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, sha256_file, write_new_json  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # diagrammi di MAVIS fino a ~12k px di lato, file del dataset verificati

CHECK = sd.CHECK  # "decontaminazione_pannello", come per i sintetici v4
SCHEMA = "panel-decontam/2"  # /2: decisioni testuali con regole esplicite, senza la frequenza nella sorgente
OLD_SCHEMA = "panel-decontam/1"  # candidato-2 (job 30652): 10-gram in più di 25 record ignorati in ogni campo
OLD_MAX_DF = 25  # soglia della regola storica, solo per riverificare candidato-2: non decide più nulla
SCAN_SCHEMA = "panel-decontam-scan/2"
SEVERITY, OUTCOMES, GROUP_STATE = sd.SEVERITY, sd.OUTCOMES, sd.GROUP_STATE
RADIUS, NGRAM, MAX_DF = 4, 10, 25  # valori documentati di decontam.py scan (PANNELLO_VALUTAZIONE §§3.1, 6.2)
# Regole esplicite per le corrispondenze testuali non informative: ristrette a un campo, motivate dal contenuto e
# verificate sul testo del record. Un 10-gram coperto non conta; ogni altro 10-gram della corrispondenza sì.
TEXT_RULES = {
    "enumerazione_punti_loader": {
        "versione": 1,
        "campo": "descrizione_del_loader",
        "condizione": "il 10-gram è una sequenza contigua di 10 etichette della riga `Points:` della descrizione "
                      "canonica del record (f1_schema.render_geometry: prima riga `Figure: geometry diagram.`, seconda "
                      "`Points: ` seguito dalle etichette ordinate dei punti della verità, ciascuna una sola lettera "
                      "maiuscola, separate da virgole). La descrizione è ricalcolata dal JSON del record e legata al "
                      "manifest dal suo sha256 (descrizione_sha256)",
        "motivo": "la riga è scritta dal loader e contiene solo le lettere dei punti in ordine alfabetico, nulla del "
                  "problema; una sequenza di lettere in comune con un benchmark (le dodici lettere A-L del problema 6 "
                  "di AIME 2025 I) non indica contaminazione",
        "non_si_applica": "agli altri campi: la stessa sequenza in una didascalia, una domanda, un CoT o una domanda "
                          "del loader resta un candidato"},
}
TEXT_RULES_SHA256 = mf.sha256_text(mf.canonical(TEXT_RULES))
FREQUENT = 30  # 10-gram più frequenti riportati per ordinare i casi da esaminare
LONG = um.LONG  # 20 token visibili: soglia di run2 per le formule lunghe
MAX_EVIDENCE = 10  # corrispondenze riportate nell'evidenza di un record; l'elenco completo sta nella scansione
BATCH = {"unimer": 2048, "mavis": 192}  # campioni per blocco del pool: immagini piccole o grandi
READ_BUFFER = 16 * 2**20  # lettura dello zip in blocchi sequenziali (scripts/shard_mavis_geometry.py)
SOURCES = {"unimer": "unimer", "mavis": "mavis-geometry"}
PANEL_SCHEMA = "panel-index/2"
MAVIS_TEXT_FIELDS = ("caption", "diagram_generation", "text_dominant_question", "text_lite_question",
                     "vision_dominant_question", "CoT_reasoning")
UNIMER_ZIP_FOLDER = "UniMER-1M/images"
HF = ROOT / "hf/hub"
OCRBENCH = HF / "datasets--echo840--OCRBench/snapshots/92a54bd1384387c178d5a07140a2d85e0a3d12e1/data/test-00000-of-00001.parquet"
MATHVERSE = (HF / "datasets--AI4Math--MathVerse/snapshots/3bc86196678bad115a923d2851c6821dbe235939/testmini.parquet",
             HF / "datasets--AI4Math--MathVerse/snapshots/3bc86196678bad115a923d2851c6821dbe235939/testmini_text_only.parquet")
# controlli che devono essere eseguiti perché un record non sia `non_verificabile`
REQUIRED = {("unimer", "train"): ("pixel_originale", "pixel_png", "formula"),
            ("unimer", "righe"): ("pixel_originale", "pixel_png", "pixel_ritagli", "formula"),
            ("mavis", "train"): ("pixel_originale", "pixel_png", "phash_originale", "phash_png", "testo")}
CANDIDATE_TYPES = ("phash", "testo", "formula_lunga")
TYPE_ORDER = ("pixel", "phash", "testo", "formula_lunga", "formula_breve")  # per l'anteprima nelle evidenze
REASON = {"unimer": {"corrispondenza_confermata": "decontaminazione_pannello: pixel identici a un'immagine di "
                                                  "pannello-v3 (gruppo escluso)",
                     "candidato": "decontaminazione_pannello: formula con più di 20 token visibili uguale a una di "
                                  "UniMER-Test o di OCRBench (gruppo in quarantena, da verificare)",
                     "non_verificabile": "decontaminazione_pannello: controllo incompleto (gruppo in quarantena)"},
          "mavis": {"corrispondenza_confermata": "decontaminazione_pannello: pixel identici a un'immagine di "
                                                 "pannello-v3 (gruppo escluso)",
                    "candidato": "decontaminazione_pannello: candidato da verificare (pHash entro 4 bit o 10-gram in "
                                 "comune; gruppo in quarantena)",
                    "non_verificabile": "decontaminazione_pannello: controllo incompleto (gruppo in quarantena)"}}
RULES = {
    "confermata": "pixel identici dopo la normalizzazione rgb-su-bianco/1 (sha256 dei pixel uguale a quello di una voce "
                  "dell'indice) per l'immagine originale dello snapshot, il PNG dello shard o un ritaglio delle righe",
    "candidato_unimer": "formula visibile (latex.effective_label, visible_tokens) uguale a una di UniMER-Test o di "
                        "OCRBench, con più di 20 token visibili (soglia di run2). Il pHash delle formule non si usa",
    "candidato_mavis": "pHash entro 4 bit dell'originale o del PNG dello shard, oppure un 10-gram di normalize_words "
                       "in comune fra un testo del record e un testo dell'indice (dal lato dell'indice senza gli n-gram "
                       "in più di 25 testi diversi del pannello) che non sia coperto da una regola esplicita del suo "
                       "campo. Una somiglianza non dimostra contaminazione",
    "evidenza_senza_effetto": "formula visibile uguale con al più 20 token visibili: registrata, stato invariato (una "
                              "formula breve o comune non identifica un problema; run2 e S1 §5)",
    "testo_coperto": "corrispondenza testuale i cui 10-gram sono tutti coperti da una regola esplicita di TEXT_RULES "
                     "(oggi solo enumerazione_punti_loader, nel solo campo descrizione_del_loader, verificata sul testo "
                     "del record): registrata, stato invariato. Basta un 10-gram non coperto, o un'altra corrispondenza "
                     "del record, perché il record resti candidato. La frequenza nella sorgente non decide",
    "non_verificabile": "un controllo richiesto non eseguito: immagine assente o illeggibile, formula vuota, testi della "
                        "verità mancanti",
    "nessuna_corrispondenza": "nessuna corrispondenza nei controlli eseguiti; non certifica la risposta né l'assenza di "
                              "ogni contaminazione",
    "gruppo": "lo stato vale per il gruppo del manifest (formula e righe ritagliate; stessa immagine o figura): "
              "confermata -> escluso, candidato o non verificabile -> quarantena; la gravità non diminuisce mai (nessuna "
              "riammissione) e split e gruppi restano quelli del genitore",
    "compiti_mavis": "i compiti seguono lo stato del record; shape_type resta almeno in quarantena",
}
NOT_COVERED = {
    "unimer": "decontaminazione esterna contro pannello-v3 (evidenza decontaminazione_pannello): pixel delle immagini "
              "originali, dei PNG degli shard e dei ritagli delle righe contro tutte le immagini del pannello; formule "
              "visibili contro UniMER-Test e OCRBench (HME100K). Non coperti: pHash delle formule (non informativo), "
              "formule dentro i testi dei problemi degli altri benchmark, stessa riga ritagliata con margini diversi",
    "mavis": "decontaminazione esterna contro pannello-v3 (evidenza decontaminazione_pannello): pixel e pHash di "
             "caption.jpg originale e del PNG dello shard, 10-gram dei testi di text_en e del loader contro tutti i testi "
             "dell'indice. Non coperti: text_zh, text.jpg e vision_dominant.jpg (non negli shard), la stessa figura "
             "disegnata in un altro stile o con un'altra impaginazione",
}
MISSING = {
    "unimer": "verifica dei candidati della decontaminazione esterna (gruppi in quarantena, formule lunghe uguali a "
              "UniMER-Test o OCRBench): nessun candidato è risolto",
    "mavis": "verifica visiva dei candidati della decontaminazione esterna (gruppi in quarantena per pHash o testo): "
             "nessun candidato è risolto",
}

G: dict = {}  # strutture dell'indice, condivise coi processi del pool (fork)


class Failed(Exception):
    """Un ingresso o un controllo non superato: niente manifest."""


def rel(path) -> str:
    return um.rel(Path(path))


def ident(r: dict) -> tuple[str, str]:
    """Identità di un record dentro il manifest: (insieme, chiave). In UniMER formula e riga ritagliata hanno la stessa
    chiave e lo stesso record_id."""
    return r["insieme"], r["record"]


def label_of(i: tuple[str, str]) -> str:
    return f"{i[0]}/{i[1]}"


def formula_string(label: str) -> str:
    """Formula visibile come la confronta run2 (unimer_test_overlap.vis_string, unimer_manifest.vis_digest)."""
    return " ".join(visible_tokens(effective_label(label)[0]))


# ---------------------------------------------------------------- indice, formule dei benchmark, provenienze

def read_index(index_dir: Path) -> tuple[dict, str]:
    path = index_dir / "index.json"
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path)


def element_of(it: dict) -> str:
    return f"{it['benchmark']}/{it['item']}"


def is_hme(row: dict) -> bool:
    """Domande di OCRBench con le formule manoscritte di HME100K (Handwritten Mathematical Expression Recognition)."""
    return "mathematical expression" in str(row.get("question_type") or "").lower() \
        or str(row.get("dataset") or "").lower().startswith("hme")


def ocrbench_formulas(path: Path, items: list[dict]) -> tuple[list[tuple[int, str]], dict]:
    """(voce dell'indice, formula) per le risposte delle domande HME100K di OCRBench, lette dal parquet della revisione
    del pannello; ogni riga scelta deve ritrovare la sua voce (`test-00000-of-00001:<posizione>`) con gli stessi
    dataset e tipo di domanda. Restituisce anche la copertura."""
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    missing = [c for c in ("dataset", "question_type", "answer") if c not in names]
    if missing:
        raise Failed(f"{path}: colonne mancanti {missing}")
    rows = pq.read_table(path, columns=["dataset", "question_type", "answer"]).to_pylist()
    stem = Path(path).stem
    by_item = {it["item"]: j for j, it in enumerate(items) if it["benchmark"] == "ocrbench"}
    out, bad, kinds, empty = [], [], Counter(), 0
    for n, row in enumerate(rows):
        if not is_hme(row):
            continue
        kinds[f"{row.get('dataset')} / {row.get('question_type')}"] += 1
        j = by_item.get(f"{stem}:{n}")
        hints = items[j].get("hints") or {} if j is not None else {}
        if j is None or hints.get("dataset") != row.get("dataset") or hints.get("question_type") != row.get("question_type"):
            bad.append(f"{stem}:{n}")
            continue
        answers = row.get("answer")
        answers = [answers] if isinstance(answers, str) else [a for a in (answers or []) if isinstance(a, str)]
        answers = [a.strip() for a in answers if a and a.strip()]
        empty += not answers
        out += [(j, a) for a in answers]
    if bad:
        raise Failed(f"OCRBench: {len(bad)} domande HME senza la loro voce nell'indice, es. {bad[:3]}")
    elements = {j for j, _ in out}
    return out, {"file": file_entry(path), "righe": len(rows), "domande_hme": sum(kinds.values()),
                 "tipi": dict(kinds), "con_risposta": len(elements), "senza_risposta": empty, "formule": len(out)}


def benchmark_formulas(items: list[dict], ocr: list[tuple[int, str]]) -> tuple[dict, dict]:
    """Formula visibile -> [(voce, token visibili)] per UniMER-Test (la formula del pannello, con gli spazi del formato
    im2latex tolti come negli shard: latex.despace) e per le risposte HME di OCRBench."""
    formulas: dict[str, list[tuple[int, int]]] = defaultdict(list)
    cov = {"unimer_test": Counter(), "ocrbench": Counter()}
    for j, it in enumerate(items):
        if it["benchmark"] != "unimer_test":
            continue
        raw = ((it.get("hints") or {}).get("formula") or "").strip()
        vis = formula_string(despace(raw)) if raw else ""
        cov["unimer_test"]["elementi"] += 1
        if not vis:
            cov["unimer_test"]["senza_formula"] += 1
            continue
        cov["unimer_test"]["con_formula"] += 1
        formulas[vis].append((j, len(vis.split())))
    for j, answer in ocr:
        vis = formula_string(despace(answer))
        cov["ocrbench"]["risposte"] += 1
        if not vis:
            cov["ocrbench"]["risposte_senza_formula"] += 1
            continue
        if (j, len(vis.split())) not in formulas[vis]:
            formulas[vis].append((j, len(vis.split())))
    return dict(formulas), {k: dict(v) for k, v in cov.items()}


def mathverse_sources(paths) -> dict:
    """Provenienze dichiarate dagli elementi di MathVerse (colonna metadata, se c'è): cerca un legame documentabile con
    MAVIS o RuleBaseGeo. Nessun legame dichiarato non prova l'assenza di contaminazione."""
    import pyarrow.parquet as pq

    out = {"file": [], "fonti": Counter(), "con_mavis_o_rulebasegeo": [], "senza_metadata": 0}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise Failed(f"MathVerse: file mancante {path}")
        names = pq.read_schema(path).names
        cols = [c for c in ("sample_index", "problem_index", "metadata") if c in names]
        out["file"].append({**file_entry(path), "colonne_lette": cols})
        for row in pq.read_table(path, columns=cols).to_pylist():
            md = row.get("metadata")
            if not isinstance(md, dict):
                out["senza_metadata"] += 1
                continue
            out["fonti"][str(md.get("source"))] += 1
            if re.search(r"mavis|rulebase", json.dumps(md, ensure_ascii=False).lower()):
                out["con_mavis_o_rulebasegeo"].append(str(row.get("sample_index")))
    out["fonti"] = dict(out["fonti"].most_common())
    return out


def structures(items: list[dict], source: str, formulas: dict | None) -> dict:
    """Strutture di confronto per i processi del pool: voce -> (elemento, benchmark, posizione); pixel -> voci; per MAVIS
    pHash e n-gram, per UniMER le formule dei benchmark."""
    out = {"entries": [(element_of(it), it["benchmark"], it.get("pos")) for it in items]}
    by_pixels: dict[str, list[int]] = defaultdict(list)
    for j, it in enumerate(items):
        if it.get("pixel_sha256"):
            by_pixels[it["pixel_sha256"]].append(j)
    out["by_pixels"] = dict(by_pixels)
    if source == "mavis":
        with_hash = [(j, int(it["phash"], 16)) for j, it in enumerate(items) if it.get("phash")]
        out["hash_entry"] = [j for j, _ in with_hash]
        out["hidx"] = HashIndex([h for _, h in with_hash], radius=RADIUS)
        out["tidx"] = TextIndex.build([it.get("text") or "" for it in items], n=NGRAM, max_df=MAX_DF)
    else:
        out["formulas"] = formulas or {}
    return out


# ---------------------------------------------------------------- esame di un campione (processi del pool)

def image_check(data: bytes | None, convention: str, phash: bool) -> dict:
    if data is None:
        return {"lettura": "assente"}
    h, pixels, size = hash_image_bytes(data, convention)
    out = {"lettura": "letta" if h is not None else "illeggibile", "byte_sha256": hashlib.sha256(data).hexdigest()}
    if h is not None:
        out.update(pixel_sha256=pixels, dimensioni=size)
        if phash:
            out["phash"] = f"{h:016x}"
    return out


def line_crop(image: Image.Image, lines: list, k: int) -> Image.Image:
    """Il ritaglio della riga k come lo costruisce data.unimer_line_sample (stessi margini, stesso ingrandimento al più
    2x entro 1024 patch); tests/test_panel_decontam.py controlla che coincida per la riga scelta dal loader."""
    y0, y1 = lines[k]
    pad = max(2, int(0.15 * (y1 - y0)))
    top = max(0, y0 - min(pad, (y0 - lines[k - 1][1]) // 2 if k else pad))
    bottom = min(image.height, y1 + min(pad, (lines[k + 1][0] - y1) // 2 if k + 1 < len(lines) else pad))
    crop = image.crop((0, top, image.width, bottom))
    h, w = capped_size(crop.height, crop.width, 2.0, 1024)
    if (w, h) != crop.size:
        crop = crop.resize((w, h), Image.BICUBIC)
    return crop


def entry_ref(j: int) -> dict:
    element, bench, pos = G["entries"][j]
    return {"elemento": element, "benchmark": bench, "posizione": pos}


def text_hits(text: str) -> list[tuple[int, list[str]]]:
    """(voce, 10-gram in comune nell'ordine del testo) per un testo, senza gli n-gram di rito del pannello (in più di
    max_df testi dell'indice, già tolti da TextIndex)."""
    tidx = G["tidx"]
    words = normalize_words(text)
    hits: dict[int, list[str]] = defaultdict(list)
    for i in range(len(words) - tidx.n + 1):
        g = tuple(words[i:i + tidx.n])
        for j in tidx.grams.get(g, ()):
            s = " ".join(g)
            if s not in hits[j]:
                hits[j].append(s)
    return sorted(hits.items())


# ---------------------------------------------------------------- regole esplicite per le corrispondenze testuali

DESCRIPTION_HEAD, POINTS_PREFIX = "Figure: geometry diagram.", "Points: "  # f1_schema.render_geometry
RULE_FIELDS = frozenset(r["campo"] for r in TEXT_RULES.values())


def points_span(description: str) -> tuple[list[str], int, int] | None:
    """Parole normalizzate della descrizione canonica e intervallo [inizio, fine) delle etichette della sua riga
    `Points:`, se la descrizione ha il formato di f1_schema.render_geometry (prima riga fissa; seconda riga `Points: `,
    etichette ordinate, distinte, ciascuna una sola lettera maiuscola, separate da `, `, punto finale). None se il
    formato è un altro: la regola non si applica e le corrispondenze restano candidati."""
    lines = description.split("\n")
    if len(lines) < 2 or lines[0] != DESCRIPTION_HEAD or not lines[1].startswith(POINTS_PREFIX) \
            or not lines[1].endswith("."):
        return None
    labels = lines[1][len(POINTS_PREFIX):-1].split(", ")
    if any(not re.fullmatch(r"[A-Z]", x) for x in labels) or labels != sorted(set(labels)):
        return None
    words = normalize_words(description)
    start = len(normalize_words(DESCRIPTION_HEAD + "\n" + POINTS_PREFIX))
    end = start + len(labels)
    if words[start:end] != [x.lower() for x in labels]:
        return None
    return words, start, end


def rule_coverage(field: str, text) -> dict[str, str]:
    """{10-gram: regola} dei 10-gram di un testo coperti da TEXT_RULES nel suo campo. enumerazione_punti_loader: nella
    descrizione del loader, un 10-gram è coperto se ogni sua occorrenza fra le parole normalizzate cade dentro le
    etichette della riga `Points:`. In ogni altro campo nessun 10-gram è coperto."""
    if field != TEXT_RULES["enumerazione_punti_loader"]["campo"] or not isinstance(text, str):
        return {}
    span = points_span(text)
    if span is None:
        return {}
    words, start, end = span
    where: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(words) - NGRAM + 1):
        where[tuple(words[i:i + NGRAM])].append(i)
    return {" ".join(g): "enumerazione_punti_loader" for g, pos in where.items()
            if all(start <= i and i + NGRAM <= end for i in pos)}


def coverage_of(texts: dict[str, str] | None) -> dict[str, dict[str, str]]:
    """{campo: {10-gram: regola}} dai testi dei campi con una regola (la descrizione del loader di un record)."""
    return {f: c for f, t in (texts or {}).items() if f in RULE_FIELDS and (c := rule_coverage(f, t))}


def informative(d: dict, coverage: dict | None = None) -> bool:
    """Una corrispondenza testuale conta come candidato se almeno uno dei suoi 10-gram non è coperto da una regola
    esplicita del suo campo. La frequenza del 10-gram nella sorgente non conta."""
    if d["tipo"] != "testo":
        return True
    covered = (coverage or {}).get(d["campo"]) or {}
    return any(g not in covered for g in d.get("ngram_tutti") or [d.get("ngram")])


RULE_BINDING = {"descrizione_del_loader": "descrizione_sha256"}  # campo -> sha256 del suo testo nel record del manifest


def bind_rule_texts(rows: dict, records: dict, texts: dict | None = None) -> tuple[dict, list[str]]:
    """Coperture per record dalle corrispondenze testuali che cadono in un campo con una regola. Il testo del campo
    (dalla riga della scansione panel-decontam-scan/2, oppure da `texts` nel riesame di una scansione /1) deve esserci,
    avere lo sha256 registrato nel record del manifest (RULE_BINDING) e contenere ogni 10-gram delle corrispondenze del
    campo. Restituisce {identità: {campo: {10-gram: regola}}} e i problemi; un testo non legato non copre nulla."""
    coverage, problems = {}, []
    for i, row in rows.items():
        details = [d for d in row.get("dettagli") or [] if d["tipo"] == "testo" and d["campo"] in RULE_FIELDS]
        if not details:
            continue
        given = (texts.get(i) if texts is not None else row.get("testi_delle_regole")) or {}
        bound = {}
        for f in sorted({d["campo"] for d in details}):
            t, digest = given.get(f), (records.get(i) or {}).get(RULE_BINDING.get(f, ""))
            if not isinstance(t, str):
                problems.append(f"{label_of(i)}: testo di {f} mancante")
            elif digest is None or mf.sha256_text(t) != digest:
                problems.append(f"{label_of(i)}: {f} senza lo sha256 registrato nel manifest ({RULE_BINDING.get(f)})")
            else:
                grams = {" ".join(g) for g in word_ngrams(normalize_words(t), NGRAM)}
                absent = [g for d in details if d["campo"] == f for g in d["ngram_tutti"] if g not in grams]
                if absent:
                    problems.append(f"{label_of(i)}: 10-gram della corrispondenza assenti da {f}: {absent[:2]}")
                else:
                    bound[f] = t
        if cov := coverage_of(bound):
            coverage[i] = cov
    return coverage, problems


def covered_rules(d: dict, coverage: dict | None) -> list[str]:
    covered = (coverage or {}).get(d.get("campo")) or {}
    return sorted({covered[g] for g in d.get("ngram_tutti") or [] if g in covered})


def frequency_rule_v1(rows) -> frozenset:
    """SOLO STORICO: la regola abolita di panel-decontam/1 (candidato-2, job 30652), che ignorava in ogni campo i 10-gram
    in comune presenti in più di OLD_MAX_DF record della sorgente. Serve soltanto a riverificare le decisioni di
    candidato-2; non decide nulla di nuovo."""
    df: Counter = Counter()
    for row in rows:
        df.update(set(row.get("ngram_in_comune") or ()))
    return frozenset(g for g, n in df.items() if n > OLD_MAX_DF)


def frequency_coverage_v1(row: dict, frequent: frozenset) -> dict[str, dict[str, str]]:
    """SOLO STORICO: la regola di panel-decontam/1 nella forma di una copertura: ogni 10-gram frequente, in ogni campo
    della riga (la regola non guardava il campo)."""
    fields = {d["campo"] for d in row.get("dettagli") or [] if d["tipo"] == "testo"}
    return {f: dict.fromkeys(frequent, "frequenza_v1") for f in fields}


def examine(task: tuple) -> dict:
    """Una riga della scansione: digest letti, controlli eseguiti, tutte le corrispondenze."""
    source, insieme, shard, pos, key, raw_json, png, orig, member, read_problem = task
    conv = G["convention"]
    row = {"insieme": insieme, "shard": shard, "indice": pos, "record": key,
           "json_sha256": hashlib.sha256(raw_json).hexdigest() if raw_json is not None else None,
           "png_sha256": hashlib.sha256(png).hexdigest() if png is not None else None,
           "problemi": [], "controlli": {}, "dettagli": []}
    try:
        meta = json.loads(raw_json)
    except (TypeError, ValueError):
        meta = {}
        row["problemi"].append("JSON del campione mancante o non valido")
    if meta.get("id") != key:
        row["problemi"].append(f"id del JSON {meta.get('id')!r} diverso dalla chiave")
    mavis = source == "mavis"
    if member is not None:  # il membro dello zip è quello del campione: cartella (MAVIS) o indice (UniMER) del JSON
        where = PurePosixPath(member)
        if mavis and where.parent.as_posix() != meta.get("zip_dir"):
            row["problemi"].append(f"membro {member} fuori dalla cartella {meta.get('zip_dir')!r} del JSON")
        if not mavis and (meta.get("subset") != "train" or not where.stem.isdigit()
                          or int(where.stem) != meta.get("index")):
            row["problemi"].append(f"membro {member} diverso dall'indice {meta.get('index')!r} del JSON")
    o = image_check(orig, conv, mavis)
    o["membro"] = member
    if read_problem:
        o["errore_di_lettura"] = read_problem
    if o["lettura"] == "letta" and list(o["dimensioni"]) != [meta.get("orig_width"), meta.get("orig_height")]:
        row["problemi"].append(f"originale {o['dimensioni']} diverso dalle dimensioni originali del JSON "
                               f"{[meta.get('orig_width'), meta.get('orig_height')]}: non è l'immagine del campione")
    p = image_check(png, conv, mavis)
    row["originale"], row["png"] = o, p
    checks = row["controlli"]
    checks["pixel_originale"] = o["lettura"] == "letta"
    checks["pixel_png"] = p["lettura"] == "letta"
    found = row["dettagli"]
    for where, im in (("originale", o), ("png", p)):
        if im.get("pixel_sha256"):
            found += [{"tipo": "pixel", "immagine": where, **entry_ref(j)} for j in G["by_pixels"].get(im["pixel_sha256"], ())]
    if mavis:
        for where, im in (("originale", o), ("png", p)):
            checks[f"phash_{where}"] = "phash" in im
            if "phash" in im:
                for jj, d in G["hidx"].query(int(im["phash"], 16)):
                    found.append({"tipo": "phash", "immagine": where, "distanza": d, **entry_ref(G["hash_entry"][jj])})
        te = meta.get("text_en") or {}
        texts = [(f, te.get(f)) for f in MAVIS_TEXT_FIELDS]
        caption, generation = te.get("caption"), te.get("diagram_generation")
        truth = mavis_truth(caption, generation)[0] if caption and generation else None
        if truth is not None:
            texts.append(("descrizione_del_loader", describe(key, truth)[1]))
            texts += [("domande_del_loader", q["q"]) for q in mavis_questions(key, truth)]
        stats = {}
        for field, text in texts:
            s = stats.setdefault(field, {"testi": 0, "confrontabili": 0})
            if not isinstance(text, str) or not text.strip():
                continue
            s["testi"] += 1
            s["confrontabili"] += len(normalize_words(text)) >= NGRAM
            found += [{"tipo": "testo", "campo": field, "ngram": gs[0], "ngram_tutti": gs, **entry_ref(j)}
                      for j, gs in text_hits(text)]
        row["testi"] = stats
        row["ngram_in_comune"] = sorted({g for d in found if d["tipo"] == "testo" for g in d["ngram_tutti"]})
        # i testi dei campi con una regola esplicita, se vi cade una corrispondenza: la decisione li rilegge
        matched = {d["campo"] for d in found if d["tipo"] == "testo"} & RULE_FIELDS
        row["testi_delle_regole"] = {f: t for f, t in texts if f in matched and isinstance(t, str)}
        checks["testo"] = bool(caption and generation)
    else:
        formulas = [("etichetta", meta.get("label"))] if insieme == "train" else \
            [(f"riga {k}", r) for k, r in enumerate(meta.get("rows") or [])]
        compared = []
        for field, label in formulas:
            vis = formula_string(label) if isinstance(label, str) else ""
            n = len(vis.split())
            compared.append({"campo": field, "token_visibili": n})
            for j, _ in G["formulas"].get(vis, ()) if vis else ():
                found.append({"tipo": "formula_lunga" if n > LONG else "formula_breve", "campo": field,
                              "token_visibili": n, **entry_ref(j)})
        row["formule"] = compared
        checks["formula"] = bool(compared) and all(isinstance(label, str) for _, label in formulas)
        if insieme == "righe":
            crops, ok = [], p["lettura"] == "letta"
            lines = meta.get("lines") or []
            if ok and lines:
                try:
                    image = open_image(png)
                    for k in range(len(lines)):
                        rgb = to_rgb(line_crop(image, lines, k))
                        px = pixel_sha256(rgb)
                        crops.append({"riga": k, "fascia": lines[k], "pixel_sha256": px, "dimensioni": list(rgb.size)})
                        found += [{"tipo": "pixel", "immagine": f"ritaglio {k}", **entry_ref(j)}
                                  for j in G["by_pixels"].get(px, ())]
                except Exception as exc:  # noqa: BLE001 - ritaglio non costruibile: controllo incompleto
                    ok = False
                    row["ritagli_errore"] = f"{type(exc).__name__}: {exc}"[:200]
            row["ritagli"] = crops
            checks["pixel_ritagli"] = ok and bool(lines) and len(crops) == len(lines)
    row["corrispondenze"] = dict(Counter(d["tipo"] for d in found))
    return row


def record_outcome(row: dict, source: str, coverage: dict | None = None) -> str:
    """Esito della decontaminazione di un record dalla sua riga della scansione; `coverage`: {campo: {10-gram: regola}}
    dei 10-gram coperti da una regola esplicita (coverage_of), le cui corrispondenze restano evidenze senza effetto."""
    found = row.get("corrispondenze") or {}
    if found.get("pixel"):
        return "corrispondenza_confermata"
    if found.get("phash") or found.get("formula_lunga") or any(
            d["tipo"] == "testo" and informative(d, coverage) for d in row.get("dettagli") or []):
        return "candidato"
    checks = row.get("controlli") or {}
    if row.get("problemi") or not all(checks.get(c) for c in REQUIRED[(source, row["insieme"])]):
        return "non_verificabile"
    return "nessuna_corrispondenza"


def decide(records: list[dict], outcome: dict[tuple, str]) -> dict[tuple, dict]:
    """Come synth_decontam.decide, con l'identità (insieme, chiave): per ogni record esito proprio, esito e stato del
    gruppo, stato nuovo e membri del gruppo che lo causano. Nessuna riammissione."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[r["gruppo"]].append(r)
    out = {}
    for rs in by_group.values():
        own = {ident(r): outcome[ident(r)] for r in rs}
        worst = min(own.values(), key=OUTCOMES.index)
        group_state = GROUP_STATE.get(worst)
        causes = sorted(label_of(i) for i, o in own.items() if o == worst) if group_state else []
        for r in rs:
            state = r["stato"]
            if group_state and SEVERITY[group_state] > SEVERITY[state]:
                state = group_state
            out[ident(r)] = {"esito": own[ident(r)], "esito_gruppo": worst, "stato_gruppo": group_state,
                             "stato": state, "cause": causes}
    return out


def task_states(compiti: list[dict], state: str) -> list[dict]:
    """MAVIS: i compiti seguono lo stato del record, shape_type resta almeno in quarantena (mavis_manifest.tasks_of)."""
    return [{**t, "stato": um.worst(state, "quarantena") if t["compito"] in mm.QUARANTINED_TASKS else state}
            for t in compiti]


# ---------------------------------------------------------------- scansione

def zip_members(source: str, zf: zipfile.ZipFile) -> dict:
    """Chiave del campione -> membro dello zip con l'immagine originale."""
    if source == "unimer":
        out = {}
        for info in zf.infolist():
            path = PurePosixPath(info.filename)
            if path.parent.as_posix() == UNIMER_ZIP_FOLDER and path.suffix == ".png" and path.stem.isdigit():
                out[f"train-{int(path.stem):07d}"] = info
        return out
    import shard_mavis_geometry as smg

    samples, _ = smg.collect(zf)  # lo stesso lettore dello sharding: d<K>-<i> -> {tipo: ZipInfo}
    return {k: files["caption.jpg"] for k, files in samples.items() if "caption.jpg" in files}


def manifest_meta(folder: Path, expected: str | None) -> tuple[dict, str]:
    path = folder / "manifest.json"
    sha = sha256_file(path)
    if expected and sha != expected:
        raise Failed(f"{path}: sha256 {sha}, atteso {expected}: non è il manifest da decontaminare")
    return json.loads(path.read_text(encoding="utf-8")), sha


def cmd_scan(args) -> dict:
    t0 = time.time()
    if args.out.exists() and any(args.out.iterdir()):
        raise Failed(f"{args.out} non è vuota: una scansione si scrive in una cartella nuova")
    meta, msha = manifest_meta(args.manifest, args.manifest_sha256)
    if meta.get("sorgente") != SOURCES[args.source]:
        raise Failed(f"manifest della sorgente {meta.get('sorgente')!r}, atteso {SOURCES[args.source]!r}")
    loaded, isha = read_index(args.index)
    if args.index_sha256 and isha != args.index_sha256:
        raise Failed(f"indice con sha256 {isha}, atteso {args.index_sha256}")
    try:
        convention = index_convention(loaded)
    except ValueError as exc:
        raise Failed(f"indice: {exc}") from None
    if convention != IMAGE_CONVENTION or loaded.get("schema") != PANEL_SCHEMA:
        raise Failed(f"indice nella convenzione {convention!r} con schema {loaded.get('schema')!r}: servono "
                     f"{IMAGE_CONVENTION} e {PANEL_SCHEMA} (sha256 dei pixel)")
    items = loaded["items"]
    extra, formulas = {}, None
    if args.source == "unimer":
        ocr, extra["ocrbench"] = ocrbench_formulas(args.ocrbench, items)
        formulas, extra["formule_dei_benchmark"] = benchmark_formulas(items, ocr)
        extra["formule_distinte"] = len(formulas)
    else:
        extra["mathverse_provenienze"] = mathverse_sources(args.mathverse)
    G.clear()
    G.update(structures(items, args.source, formulas), convention=convention)
    index_info = {**file_entry(args.index / "index.json", isha), "schema": loaded.get("schema"),
                  "convenzione_immagini": convention, "voci": len(items),
                  "voci_con_pixel_sha256": sum(map(len, G["by_pixels"].values())),
                  "voci_con_phash": sum(1 for it in items if it.get("phash"))}
    if args.source == "mavis":
        index_info["ngram_di_rito_esclusi"] = G["tidx"].dropped
    del loaded, items
    gc.collect()
    snap = meta["identita"]["snapshot"]["zip"]
    zip_path = Path(snap["percorso"])
    zinfo = {"percorso": str(zip_path), "blob": zip_path.resolve().name, "sha256_atteso": snap["sha256_atteso"],
             "byte": zip_path.stat().st_size if zip_path.is_file() else None}
    zinfo["blob_coincide"] = zinfo["blob"] == zinfo["sha256_atteso"]
    if not zip_path.is_file() or not zinfo["blob_coincide"]:
        raise Failed(f"zip dello snapshot assente o diverso da quello del manifest: {zinfo}")
    # il pool nasce prima del thread che hasha lo zip e prima dell'indice dello zip: i processi ereditano solo le
    # strutture dell'indice del pannello
    pool = multiprocessing.get_context("fork").Pool(args.workers)
    zip_hash: dict = {}
    hasher = None
    if not args.limit:  # lo sha256 dello zip intero si calcola mentre si leggono i membri
        hasher = threading.Thread(target=lambda: zip_hash.update(sha256=sha256_file(zip_path)), daemon=True)
        hasher.start()
    zf = zipfile.ZipFile(open(zip_path, "rb", buffering=READ_BUFFER))
    members = zip_members(args.source, zf)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {"righe": Counter(), "esiti_propri": Counter(), "controlli_eseguiti": Counter(),
             "controlli_non_eseguiti": Counter(), "lettura_originale": Counter(), "lettura_png": Counter(),
             "corrispondenze_per_benchmark_tipo": Counter(), "record_con_corrispondenze_per_benchmark_tipo": Counter(),
             "problemi": Counter(), "testi_per_campo": defaultdict(Counter), "formule": Counter()}
    examples: dict[str, list] = defaultdict(list)
    # MAVIS: frequenze dei 10-gram in comune, solo per ordinare i casi da esaminare (non decidono nulla)
    df: Counter = Counter()
    ngram_elements: dict[str, set] = defaultdict(set)
    ngram_fields: dict[str, Counter] = defaultdict(Counter)
    ngram_covered: dict[str, Counter] = defaultdict(Counter)
    only_covered = [0]  # record con corrispondenze testuali tutte coperte da una regola
    shards_out = []
    raw_records = gzip.GzipFile(args.out / "records.jsonl.gz", "wb", mtime=0)
    writer = io.TextIOWrapper(raw_records, encoding="utf-8")

    def handle(results: list[dict]) -> None:
        for row in results:
            writer.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["righe"][row["insieme"]] += 1
            coverage = coverage_of(row.get("testi_delle_regole"))
            if args.source == "mavis":
                texts = [d for d in row["dettagli"] if d["tipo"] == "testo"]
                df.update(set(row["ngram_in_comune"]))
                for d in texts:
                    cov = (coverage.get(d["campo"]) or {})
                    for g in d["ngram_tutti"]:
                        ngram_elements[g].add(d["elemento"])
                        ngram_fields[g][d["campo"]] += 1
                        ngram_covered[g][cov.get(g, "non_coperta")] += 1
                if texts and not any(informative(d, coverage) for d in texts):
                    only_covered[0] += 1
            stats["esiti_propri"][f"{row['insieme']}/{record_outcome(row, args.source, coverage)}"] += 1
            for c, v in row["controlli"].items():
                stats["controlli_eseguiti" if v else "controlli_non_eseguiti"][f"{row['insieme']}/{c}"] += 1
            stats["lettura_originale"][row["originale"]["lettura"]] += 1
            stats["lettura_png"][row["png"]["lettura"]] += 1
            for p in row["problemi"]:
                stats["problemi"][mr.cause(p)] += 1
            seen = set()
            for d in row["dettagli"]:
                k = f"{d['benchmark']}/{d['tipo']}"
                stats["corrispondenze_per_benchmark_tipo"][k] += 1
                seen.add(k)
            for k in seen:
                stats["record_con_corrispondenze_per_benchmark_tipo"][k] += 1
                if len(examples[k]) < 10:
                    examples[k].append(f"{row['insieme']}/{row['record']}")
            for field, s in (row.get("testi") or {}).items():
                stats["testi_per_campo"][field]["testi"] += s["testi"]
                stats["testi_per_campo"][field]["confrontabili"] += s["confrontabili"]
            for c in row.get("formule") or []:
                stats["formule"][f"{row['insieme']}/confrontate"] += c["token_visibili"] > 0
                stats["formule"][f"{row['insieme']}/vuote"] += c["token_visibili"] == 0

    pending = None
    for s in meta["shards"]:
        path = mf.shard_path(s["percorso"])
        raw = um.HashingReader(path)
        n, done = 0, True
        batch: list[tuple] = []
        with tarfile.open(fileobj=io.BufferedReader(raw, buffer_size=1 << 20), mode="r|") as tar:
            for pos, (key, files) in enumerate(um.iter_samples(tar)):
                if args.limit and pos >= args.limit:
                    done = False
                    break
                n += 1
                raw_json, png = files.get("json"), files.get("png")
                orig, member, problem = None, None, None
                info = members.get(key)
                if info is not None:
                    member = info.filename
                    try:
                        orig = zf.read(info)  # zipfile controlla il CRC-32 del membro
                    except (zipfile.BadZipFile, OSError) as exc:
                        problem = f"{type(exc).__name__}: {exc}"[:200]
                batch.append((args.source, s["insieme"], s["percorso"], pos, key, raw_json, png, orig, member,
                              problem))
                if len(batch) >= BATCH[args.source]:
                    cur = pool.map_async(examine, batch, chunksize=16)
                    if pending is not None:
                        handle(pending.get())
                    pending, batch = cur, []
        if batch:
            cur = pool.map_async(examine, batch, chunksize=16)
            if pending is not None:
                handle(pending.get())
            pending = cur
        sha, size = raw.finish() if done else (None, None)
        shards_out.append({"insieme": s["insieme"], "percorso": s["percorso"], "sha256_atteso": s["sha256"],
                           "byte_attesi": s["bytes"], "record_attesi": s["record"], "campioni_letti": n,
                           "completo": done, "sha256_letto": sha, "byte_letti": size,
                           "uguale": done and sha == s["sha256"] and size == s["bytes"] and n == s["record"]})
        print(f"{Path(s['percorso']).name}: {n} campioni ({time.time() - t0:.0f} s)", flush=True)
    if pending is not None:
        handle(pending.get())
    pool.close()
    pool.join()
    writer.close()
    if hasher is not None:
        hasher.join()
        zinfo["sha256_calcolato"] = zip_hash.get("sha256")
        zinfo["sha256_uguale"] = zinfo["sha256_calcolato"] == zinfo["sha256_atteso"]
    else:
        zinfo["sha256_calcolato"], zinfo["sha256_uguale"] = None, None
    zinfo["membri_con_immagine"] = len(members)
    texts_summary = None
    if args.source == "mavis":
        def ngram_row(g: str) -> dict:
            return {"ngram": g, "record": df[g], "elementi": sorted(ngram_elements[g])[:5],
                    "elementi_totali": len(ngram_elements[g]), "campi": dict(ngram_fields[g].most_common()),
                    "corrispondenze_per_regola": dict(ngram_covered[g].most_common())}

        texts_summary = {
            "regole_testuali": TEXT_RULES, "regole_testuali_sha256": TEXT_RULES_SHA256,
            "ngram_in_comune_distinti": len(df),
            "ngram_piu_frequenti": [ngram_row(g) for g, _ in df.most_common(FREQUENT)],
            "record_con_sole_corrispondenze_testuali_coperte": only_covered[0],
            "nota": "la frequenza (record per 10-gram) serve solo a ordinare i casi da esaminare e non decide nulla: una "
                    "corrispondenza è senza effetto solo se tutti i suoi 10-gram sono coperti da una regola esplicita "
                    "del suo campo"}
    records_file = file_entry(args.out / "records.jsonl.gz")
    summary = {
        "schema": SCAN_SCHEMA, "sorgente": args.source, "job_id": os.environ.get("SLURM_JOB_ID"),
        "codice": code_identity(), "args": {k: str(v) for k, v in vars(args).items() if k != "func"},
        "limite": args.limit or None, "parziale": bool(args.limit),
        "manifest": {"percorso": rel(args.manifest), "manifest_sha256": msha,
                     "records_sha256": meta["records"]["sha256"], "righe": meta["records"]["righe"]},
        "indice": index_info, "convenzione_immagini": convention, "zip": zinfo, "shard": shards_out,
        "soglie": {"phash_bit": RADIUS if args.source == "mavis" else None, "ngram": NGRAM if args.source == "mavis" else None,
                   "max_df": MAX_DF if args.source == "mavis" else None,
                   "formula_lunga_oltre_token": LONG if args.source == "unimer" else None},
        **extra, "testi_in_comune": texts_summary,
        "statistiche": {k: (dict(sorted(v.items())) if not isinstance(v, defaultdict) else
                            {f: dict(c) for f, c in sorted(v.items())}) for k, v in stats.items()},
        "esempi_per_benchmark_tipo": dict(sorted(examples.items())),
        "files": {"records.jsonl.gz": records_file},
        "secondi": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib(),
        "workers": args.workers}
    write_new_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary["statistiche"][k] for k in ("righe", "esiti_propri", "record_con_corrispondenze_per_"
                                                                     "benchmark_tipo", "problemi")},
                     ensure_ascii=False)[:3000])
    print(f"scansione: {sum(stats['righe'].values())} campioni in {summary['secondi']} s, RSS {summary['rss_max_mib']} MiB")
    return summary


# ---------------------------------------------------------------- verifica di compatibilità

def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_and_bind(args, partial: bool, checks: list[dict]) -> dict:
    """Ingressi, compatibilità e copertura (docstring). Ogni controllo va in `checks`; il chiamante decide."""
    def check(name: str, ok: bool, detail=None) -> bool:
        checks.append({"controllo": name, "esito": bool(ok), "dettaglio": detail})
        print(("ok   " if ok else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail is not None else ""), flush=True)
        return bool(ok)

    # ---- indice
    loaded, isha = read_index(args.index)
    try:
        convention = index_convention(loaded)
    except ValueError as exc:
        convention = str(exc)
    check("indice: convenzione dichiarata rgb-su-bianco/1 e schema panel-index/2",
          convention == IMAGE_CONVENTION and loaded.get("schema") == PANEL_SCHEMA, [convention, loaded.get("schema")])
    check("indice: sha256 uguale a quello atteso", isha == args.index_sha256, isha)
    items = loaded["items"]
    verifica = json.loads((args.index / "verifica.json").read_text(encoding="utf-8"))
    with_image = sum(1 for it in items if it.get("pos") is not None)
    check("indice: verifica.json, ogni voce con immagine ritrova la propria immagine nella sorgente",
          verifica.get("verificato") is True and verifica.get("errori") == 0
          and verifica.get("voci_con_immagine") == with_image and verifica.get("convenzione_immagini") == convention,
          {k: verifica.get(k) for k in ("verificato", "errori", "voci_con_immagine")})
    check("indice: sha256 dei pixel per ogni voce con un'immagine leggibile",
          sum(1 for it in items if it.get("pixel_sha256")) == sum(1 for it in items if it.get("phash")
                                                                   and it.get("pos") is not None))
    # ---- pannello e perimetro
    panel_summary = json.loads((args.panel / "summary.json").read_text(encoding="utf-8"))
    check("pannello costruito da questo indice (sha256 registrato in summary.json)",
          (panel_summary.get("indice") or {}).get("sha256") == isha)
    elements = {e["chiave"]: e for e in sd.read_jsonl_gz(args.panel / "elements.jsonl.gz")}
    index_elements = {element_of(it) for it in items}
    check("perimetro: ogni elemento del pannello ha voci nell'indice e ogni voce un elemento",
          index_elements == set(elements), {"solo_indice": len(index_elements - set(elements)),
                                           "solo_pannello": len(set(elements) - index_elements)})
    perimeter = Counter(role_of(e) for e in elements.values())
    check("perimetro: tutti i ruoli presenti, compresi test non ammesso, diagnostica e solo_decontaminazione",
          all(perimeter[r] for r in ("sviluppo", "test_ammesso", "test_non_ammesso", "diagnostica",
                                     "solo_decontaminazione")), dict(perimeter))
    check("pannello: numero di elementi uguale a summary.json", len(elements) == panel_summary.get("elementi"),
          len(elements))
    # ---- manifest
    try:
        m = mr.load_expected(args.manifest, args.manifest_sha256)
    except mr.Problem as exc:
        check("manifest: sha256 atteso, letto e verificato da manifest.load", False, str(exc))
        raise Failed(str(exc)) from None
    check("manifest: sha256 atteso, letto e verificato da manifest.load (record_id, un solo split per gruppo, "
          "conteggi per shard)", True, m.manifest_sha256)
    check("manifest: sorgente", m.meta.get("sorgente") == SOURCES[args.source], m.meta.get("sorgente"))
    try:
        classes = mr.shard_classes(m)
        check("manifest: classificazione degli shard uguale all'insieme dei loro record", True, classes)
    except mr.Problem as exc:
        check("manifest: classificazione degli shard uguale all'insieme dei loro record", False, str(exc))
    records = [r for rows in m.by_shard.values() for r in rows.values()]
    ids = Counter(ident(r) for r in records)
    check("manifest: identità (insieme, chiave) uniche", max(ids.values()) == 1, len(ids))
    # ---- scansione
    scan_summary = json.loads((args.scan / "summary.json").read_text(encoding="utf-8"))
    check("scansione: stessa sorgente, stesso indice (sha256) e stessa convenzione",
          scan_summary.get("sorgente") == args.source and (scan_summary.get("indice") or {}).get("sha256") == isha
          and scan_summary.get("convenzione_immagini") == convention == IMAGE_CONVENTION)
    check("scansione: dello stesso manifest (sha256 di manifest.json e dei record)",
          (scan_summary.get("manifest") or {}).get("manifest_sha256") == m.manifest_sha256
          and (scan_summary.get("manifest") or {}).get("records_sha256") == m.meta["records"]["sha256"])
    rec_file = (scan_summary.get("files") or {}).get("records.jsonl.gz") or {}
    scan_sha = sha256_file(args.scan / "records.jsonl.gz")
    check("scansione: records.jsonl.gz con lo sha256 di summary.json", scan_sha == rec_file.get("sha256"), scan_sha)
    check("scansione: completa (nessun limite)" if not partial else "scansione: prova limitata dichiarata",
          (not scan_summary.get("parziale")) if not partial else bool(scan_summary.get("parziale")),
          scan_summary.get("limite"))
    known = {s["percorso"]: s for s in m.meta["shards"]}
    shard_rows = scan_summary.get("shard") or []
    check("scansione: solo shard del manifest, ciascuno una volta",
          [s["percorso"] for s in shard_rows] == [s["percorso"] for s in m.meta["shards"]]
          and all(s["sha256_atteso"] == known[s["percorso"]]["sha256"] for s in shard_rows))
    if not partial:
        check("shard: letti per intero, con sha256, byte e campioni del manifest", all(s["uguale"] for s in shard_rows),
              [Path(s["percorso"]).name for s in shard_rows if not s["uguale"]])
    zinfo = scan_summary.get("zip") or {}
    snap = m.meta["identita"]["snapshot"]["zip"]
    check("snapshot: zip del manifest, blob col nome dello sha256 atteso"
          + ("" if partial else ", sha256 ricalcolato uguale"),
          zinfo.get("percorso") == snap["percorso"] and zinfo.get("sha256_atteso") == snap["sha256_atteso"]
          and zinfo.get("blob_coincide") is True and (partial or zinfo.get("sha256_uguale") is True),
          {k: zinfo.get(k) for k in ("blob_coincide", "sha256_uguale")})
    rows = read_rows(args.scan / "records.jsonl.gz")
    check("scansione: una riga per campione letto", len(rows) == sum(s["campioni_letti"] for s in shard_rows), len(rows))
    # ---- identificativi
    by_shard = m.by_shard
    base = {r["record"]: r for r in records if r["insieme"] == "train"}
    problems, examples, scan = Counter(), [], {}
    for row in rows:
        r = by_shard.get(row["shard"], {}).get(row["record"])
        why = None
        if ident(row) in scan:
            why = "riga ripetuta"
        elif r is None:
            why = "campione senza record nello shard"
        elif r["insieme"] != row["insieme"]:
            why = "insieme diverso"
        elif r["indice"] != row["indice"]:
            why = "posizione diversa"
        elif r["json_sha256"] != row["json_sha256"]:
            why = "JSON diverso da quello del manifest"
        elif r["immagini"][0]["sha256"] != row["png_sha256"]:
            why = "byte del PNG diversi da quelli del manifest"
        elif row["insieme"] == "righe" and row["png_sha256"] != (r["genitori"] or [{}])[0].get("png_sha256"):
            why = "riga ritagliata con un PNG diverso da quello della formula"
        elif row["problemi"]:
            why = "problemi della lettura: " + "; ".join(row["problemi"])[:200]
        if why:
            problems[mr.cause(why)] += 1
            if len(examples) < 5:
                examples.append(f"{label_of(ident(row))}: {why}")
        scan.setdefault(ident(row), row)
    check("identificativi: ogni riga ritrova un solo record per insieme, shard, posizione e chiave, coi digest di JSON "
          "e PNG del manifest (le righe ritagliate col PNG della formula)", not problems,
          {"problemi": dict(problems), "esempi": examples})
    if partial:
        scanned = {s["percorso"]: s["campioni_letti"] for s in shard_rows}
        expected = {ident(r) for r in records if r["shard"] in scanned and r["indice"] < scanned[r["shard"]]}
    else:
        expected = set(ids)
    missing, extra = expected - set(scan), set(scan) - expected
    check("copertura: tutti e soli i record " + ("delle parti lette degli shard" if partial else "del manifest")
          + ", una volta ciascuno", not missing and not extra,
          {"mancanti": len(missing), "in_piu": len(extra), "esempi": [label_of(i) for i in sorted(missing)[:3]]})
    unknown = Counter(d.get("benchmark") for row in scan.values() for d in row["dettagli"]
                      if d.get("elemento") not in elements)
    check("corrispondenze: ogni elemento citato esiste nel pannello", not unknown, dict(unknown))
    coverage = {}
    if args.source == "mavis":
        check("scansione: schema panel-decontam-scan/2, coi testi dei campi con una regola esplicita (una scansione "
              "/1 di MAVIS si ridecide con `riesame`)", scan_summary.get("schema") == SCAN_SCHEMA,
              scan_summary.get("schema"))
        coverage, bad = bind_rule_texts(scan, {ident(r): r for r in records})
        check("testo: per ogni corrispondenza in un campo con una regola, il testo del campo è nella scansione, ha lo "
              "sha256 del manifest e contiene i 10-gram della corrispondenza", not bad, {"casi": len(bad),
                                                                                      "esempi": bad[:3]})
    n_items = len(items)
    del loaded, items
    inp = {"index_sha": isha, "index_voci": n_items, "convention": convention, "verifica": verifica,
           "panel_summary": panel_summary, "elements": elements, "perimeter": perimeter, "manifest": m,
           "records": records, "base": base, "scan_summary": scan_summary, "scan": scan, "scan_sha": scan_sha,
           "check": check, "partial": partial, "coverage": coverage}
    if args.source == "unimer":
        reconcile_run2(args, inp, check)
    else:
        inp["storiche"] = historical(args, scan, coverage)
    return inp


def role_of(e: dict | None) -> str | None:
    if not e:
        return None
    role = e.get("ruolo")
    if role == "test":
        return "test_ammesso" if e.get("metrica_primaria") is True else "test_non_ammesso"
    return role


def reconcile_run2(args, inp: dict, check) -> None:
    """UniMER: run2 (UniMER-Test contro il training, già nel manifest genitore) riconciliato con la scansione nuova.
    - copertura: le chiavi di UniMER-Test di run2 (test_seen.jsonl e test_unseen_*.txt) sono gli elementi di
      UniMER-Test del pannello;
    - formule, per record: la formula visibile ritrovata uguale a una di UniMER-Test del pannello se e solo se run2 la
      registrava (formula breve o lunga), e lunga se e solo se run2 la contava fra le contaminazioni lunghe; il record
      con la stessa immagine di run2 (byte del PNG ridimensionato) resta con l'esito di run2;
    - immagini: i record con l'immagine del test di run2 ritrovati dall'identità dei pixel degli originali (riportato,
      non bloccante: lo stato di run2 resta)."""
    overlap = Path(args.overlap)
    run2_keys = set()
    seen = um.read_jsonl(overlap / "test_seen.jsonl")
    run2_keys |= {r["key"] for r in seen}
    unseen_files = sorted(overlap.glob("test_unseen_*.txt"))
    for p in unseen_files:
        run2_keys |= set(um.read_keys(p))
    panel_keys = {e.split("/", 1)[1] for e, x in inp["elements"].items() if x.get("benchmark") == "unimer_test"}
    check("riconciliazione run2: UniMER-Test di run2 uguale a quello del pannello (chiavi)", run2_keys == panel_keys,
          {"run2": len(run2_keys), "pannello": len(panel_keys), "solo_run2": len(run2_keys - panel_keys),
           "solo_pannello": len(panel_keys - run2_keys)})
    inp["file_run2_letti"] = [file_entry(overlap / "test_seen.jsonl")] + [file_entry(p) for p in unseen_files]
    bad, image_found, image_missing, counts = [], 0, [], Counter()
    for (insieme, key), row in inp["scan"].items():
        if insieme != "train":
            continue
        r = inp["base"].get(key)
        e2 = [e["esito"] for e in r["evidenze"] if e.get("lista") == "unimer_overlap_run2"]
        e2 = e2[0] if len(e2) == 1 else None
        test = [d for d in row["dettagli"] if d["benchmark"] == "unimer_test"]
        formula = {d["tipo"] for d in test if d["tipo"] in ("formula_breve", "formula_lunga")}
        pixel = any(d["tipo"] == "pixel" for d in test)
        counts[f"run2 {e2} / nuova {'+'.join(sorted(formula)) or 'nessuna'}{' / pixel' if pixel else ''}"] += 1
        if e2 == "immagine_del_test":
            if pixel:
                image_found += 1
            else:
                image_missing.append(key)
            continue
        expected = {"formula_lunga_del_test": {"formula_lunga"}, "formula_breve_del_test": {"formula_breve"},
                    "nessuna_corrispondenza": set()}.get(e2)
        if expected is None or formula != expected:
            bad.append(f"{key}: run2 {e2}, nuova {sorted(formula)}")
    check("riconciliazione run2: formule uguali a UniMER-Test ritrovate dal pannello come in run2, per record (breve, "
          "lunga oltre 20 token, nessuna)", not bad, {"diversi": len(bad), "esempi": bad[:5]})
    inp["riconciliazione_run2"] = {"per_esito": dict(sorted(counts.items())),
                                   "immagine_del_test_ritrovata_dai_pixel": image_found,
                                   "immagine_del_test_non_ritrovata": image_missing[:20],
                                   "nota": "l'identità di run2 era sui byte del PNG ridimensionato; quella nuova sui pixel "
                                           "degli originali. Lo stato dato da run2 resta in ogni caso"}
    check("riconciliazione run2: immagini del test di run2 ritrovate dall'identità dei pixel degli originali "
          "(riportato, non bloccante)", True, {"ritrovate": image_found, "non_ritrovate": len(image_missing)})


def historical(args, scan: dict, coverage: dict) -> list[dict]:
    """MAVIS: candidati storici (data/decontam/p6/mavis e f1eval/mavis, convenzione grigio-diretto/0 sugli shard): solo
    un confronto diagnostico coi candidati nuovi. Le liste finali sono vuote e non si importano."""
    out = []
    new = {k for (s, k), row in scan.items()
           if record_outcome(row, args.source, coverage.get((s, k))) != "nessuna_corrispondenza"}
    for folder in args.historical or []:
        folder = Path(folder)
        if not (folder / "exclude.txt").is_file() or not (folder / "exclude_final.txt").is_file():
            raise Failed(f"scansione storica incompleta: {folder} senza exclude.txt o exclude_final.txt")
        raw, final = um.read_keys(folder / "exclude.txt"), um.read_keys(folder / "exclude_final.txt")
        out.append({"cartella": rel(folder), "exclude": file_entry(folder / "exclude.txt"),
                    "exclude_final": file_entry(folder / "exclude_final.txt"), "candidati_storici": len(raw),
                    "lista_finale": len(final), "storici_fra_i_record_letti": len({k for k in raw if ("train", k) in scan}),
                    "storici_anche_candidati_nuovi": len(set(raw) & new),
                    "storici_non_candidati_nuovi_esempi": sorted({k for k in raw if ("train", k) in scan} - new)[:10],
                    "nota": "convenzione grigio-diretto/0 sugli shard ridimensionati; diagnostico, non importato"})
    return out


def binding_report(inp: dict, args) -> dict:
    """Conteggi della verifica: esiti propri e di gruppo, corrispondenze per benchmark, ruolo e tipo."""
    scan, elements, coverage = inp["scan"], inp["elements"], inp["coverage"]
    outcomes = {i: record_outcome(row, args.source, coverage.get(i)) for i, row in scan.items()}
    recs = [r for r in inp["records"] if ident(r) in scan]
    decisions = decide(recs, outcomes)
    per_bench, records_per_bench, by_kind = Counter(), Counter(), Counter()
    details_by_kind = Counter()
    only_covered = 0
    for i, row in scan.items():
        seen = set()
        texts = [d for d in row["dettagli"] if d["tipo"] == "testo"]
        only_covered += bool(texts) and not any(informative(d, coverage.get(i)) for d in texts)
        for d in row["dettagli"]:
            e = elements.get(d["elemento"])
            kind = d["tipo"] if informative(d, coverage.get(i)) else "testo_coperto"
            k = f"{d['benchmark']}/{role_of(e)}/{kind}"
            per_bench[k] += 1
            seen.add(f"{d['benchmark']}/{kind}")
            details_by_kind[kind] += 1
        for k in seen:
            records_per_bench[k] += 1
        by_kind[f"{i[0]}/{outcomes[i]}"] += 1
    changes = Counter(f"{r['insieme']}/{r['split']}: {r['stato']} -> {decisions[ident(r)]['stato']}" for r in recs
                      if decisions[ident(r)]["stato"] != r["stato"])
    groups = Counter()
    for g in {r["gruppo"]: decisions[ident(r)]["esito_gruppo"] for r in recs}.values():
        groups[g] += 1
    return {"esiti_propri": dict(sorted(by_kind.items())), "esiti_dei_gruppi": dict(sorted(groups.items())),
            "cambi_di_stato": dict(sorted(changes.items())),
            "corrispondenze_per_benchmark_ruolo_tipo": dict(sorted(per_bench.items())),
            "record_con_corrispondenze_per_benchmark_tipo": dict(sorted(records_per_bench.items())),
            "corrispondenze_per_tipo": dict(sorted(details_by_kind.items())),
            "record_con_sole_corrispondenze_testuali_coperte": only_covered,
            "gruppi_toccati": sum(n for o, n in groups.items() if o in GROUP_STATE)}, outcomes, decisions


def cmd_verifica(args) -> int:
    t0 = time.time()
    if args.report.exists() and any(args.report.iterdir()):
        raise Failed(f"{args.report} non è vuota")
    checks: list[dict] = []
    inp = load_and_bind(args, args.partial, checks)
    stats, _, _ = binding_report(inp, args)
    report = {"schema": SCHEMA + "-verifica", "sorgente": args.source, "job_id": os.environ.get("SLURM_JOB_ID"),
              "codice": code_identity(), "parziale": args.partial, "controlli": checks,
              "falliti": [c["controllo"] for c in checks if not c["esito"]],
              "perimetro_pannello": dict(inp["perimeter"]), "esiti": stats,
              "riconciliazione_run2": inp.get("riconciliazione_run2"), "candidati_storici": inp.get("storiche"),
              "scansione": {k: inp["scan_summary"].get(k) for k in ("limite", "secondi", "rss_max_mib", "workers",
                                                                      "statistiche", "zip", "shard")},
              "nota": "con --partial gli esiti per gruppo valgono solo fra i record letti",
              "secondi": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()}
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "report.json", report)
    print(f"{len(checks) - len(report['falliti'])} controlli superati su {len(checks)}")
    print(json.dumps(stats, ensure_ascii=False)[:4000])
    return 1 if report["falliti"] else 0


# ---------------------------------------------------------------- versione nuova del manifest

def evidence_of(row: dict, decision: dict, elements: dict, ref: dict, job, coverage: dict | None,
                extra: dict | None = None) -> dict:
    """Evidenza `decontaminazione_pannello` di un record: esiti, controlli, conteggi e un'anteprima di al più
    MAX_EVIDENCE corrispondenze (prima le più gravi e le testuali informative); l'elenco completo sta nella scansione.
    Le corrispondenze testuali coperte da una regola esplicita compaiono come `testo_coperto`, con la regola."""
    found = sorted(row["dettagli"], key=lambda d: (TYPE_ORDER.index(d["tipo"]), not informative(d, coverage)))
    preview = []
    for d in found[:MAX_EVIDENCE]:
        e = elements.get(d["elemento"]) or {}
        item = {"tipo": d["tipo"] if informative(d, coverage) else "testo_coperto", "elemento": d["elemento"],
                "posizione": d.get("posizione"), "immagine": d.get("immagine"), "campo": d.get("campo"),
                "distanza": d.get("distanza"), "ngram": d.get("ngram"), "ruolo": role_of(e),
                "gruppo_pannello": e.get("gruppo"), "metrica_primaria": e.get("metrica_primaria")}
        if item["tipo"] == "testo_coperto":
            item["regole"] = covered_rules(d, coverage)
        preview.append(item)
    return {"controllo": CHECK, "versione": SCHEMA, "job": job, "esito": decision["esito"],
            "esito_gruppo": decision["esito_gruppo"], "cause_nel_gruppo": decision["cause"],
            "controlli": row["controlli"], "corrispondenze": row["corrispondenze"],
            "testo_coperto": sum(d["tipo"] == "testo" and not informative(d, coverage) for d in found),
            "regole_testuali_sha256": TEXT_RULES_SHA256, "elementi": preview,
            "corrispondenze_nella_scansione": len(found), "rif": ref, **(extra or {})}


def derived_keys(source: str, recs: list[dict]) -> dict[str, list[str]]:
    """keys/ come nella costruzione del genitore (unimer_manifest.run, mavis_manifest.run)."""
    if source == "unimer":
        def pick(insieme, split=None, admitted=True):
            return [r["record"] for r in recs if r["insieme"] == insieme and (split is None or r["split"] == split)
                    and (r["stato"] == "ammesso") == admitted]
        return {"train.txt": pick("train", "train"), "dev.txt": pick("train", "dev"),
                "non_ammessi.txt": pick("train", admitted=False), "righe_train.txt": pick("righe", "train"),
                "righe_dev.txt": pick("righe", "dev"), "righe_non_ammesse.txt": pick("righe", admitted=False)}
    return {"train.txt": [r["record"] for r in recs if r["split"] == "train" and r["stato"] == "ammesso"],
            "dev.txt": [r["record"] for r in recs if r["split"] == "dev" and r["stato"] == "ammesso"],
            "non_ammessi.txt": [r["record"] for r in recs if r["stato"] != "ammesso"]}


def summarize(source: str, recs: list[dict]) -> dict:
    return um.summarize(recs) if source == "unimer" else mm.summarize(recs)


def parent_lines(m: mf.Manifest) -> list[dict]:
    """Record del genitore nell'ordine di records.jsonl.gz (già verificato da manifest.load)."""
    raw = (m.folder / m.meta["records"]["file"]).read_bytes()
    return [json.loads(x) for x in gzip.decompress(raw).decode("utf-8").splitlines()]


def cmd_manifest(args) -> int:
    t0 = time.time()
    for p in (args.new, args.report):
        if p.exists() and any(p.iterdir()):
            raise Failed(f"{p} non è vuota: una versione del manifest si scrive una volta sola")
    checks: list[dict] = []
    inp = load_and_bind(args, False, checks)
    failed = [c["controllo"] for c in checks if not c["esito"]]
    if failed:
        raise Failed("controlli di compatibilità falliti, manifest non scritto: " + "; ".join(failed))
    stats, outcomes, decisions = binding_report(inp, args)
    m = inp["manifest"]
    elements, scan = inp["elements"], inp["scan"]
    ref = {"file": rel(args.scan / "records.jsonl.gz"), "sha256": inp["scan_sha"]}
    job = inp["scan_summary"].get("job_id")
    reasons = REASON[args.source]
    new, lines = [], []
    for r in parent_lines(m):
        d = decisions[ident(r)]
        rec = dict(r)
        rec["evidenze"] = list(r["evidenze"] or []) + [evidence_of(scan[ident(r)], d, elements, ref, job,
                                                                   inp["coverage"].get(ident(r)))]
        if d["stato_gruppo"]:
            rec["motivo"] = list(r["motivo"] or []) + [reasons[d["esito_gruppo"]]]
        rec["stato"] = d["stato"]
        if args.source == "mavis":
            rec["compiti"] = task_states(r["compiti"], d["stato"])
        new.append({k: rec[k] for k in ("record", "insieme", "split", "stato", "gruppo", "compiti", "motivo",
                                        "trasformazione", "famiglia", "misure_razionali_come_decimali") if k in rec})
        lines.append(json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    args.new.mkdir(parents=True, exist_ok=True)
    data_sha, n_lines = um.write_records_gz(args.new / "records.jsonl.gz", iter(lines))
    del lines
    (args.new / "keys").mkdir()
    derived = {}
    for name, ks in derived_keys(args.source, new).items():
        text = "".join(k + "\n" for k in sorted(ks))
        (args.new / "keys" / name).write_text(text, encoding="utf-8")
        derived[f"keys/{name}"] = {**file_entry(args.new / "keys" / name, mf.sha256_text(text)), "righe": len(ks)}
    meta0 = m.meta
    split = dict(meta0["split"])
    split["ereditato_da"] = {"percorso": rel(args.manifest), "versione": meta0["versione"],
                             "manifest_sha256": m.manifest_sha256}
    split["regola_versione"] = ("split e gruppi della versione precedente, invariati: la decontaminazione cambia solo "
                                "stati, motivi, evidenze e (MAVIS) stati dei compiti")
    admitted = Counter(f"{x['split']}/{x.get('famiglia') or x['insieme']}" for x in new if x["stato"] == "ammesso")
    split["ammessi_dopo_la_decontaminazione"] = dict(sorted(admitted.items()))
    scan_summary = inp["scan_summary"]
    sources = {k: scan_summary.get(k) for k in ("ocrbench", "formule_dei_benchmark", "formule_distinte",
                                                "mathverse_provenienze") if k in scan_summary}
    stati = dict(meta0["stati"])
    stati["quarantena"] += ("; oppure gruppo con un candidato della decontaminazione esterna contro pannello-v3 o un "
                            "controllo incompleto: da verificare")
    stati["escluso"] = (stati["escluso"] + "; oppure " if not stati["escluso"].startswith("nessuna") else "") \
        + "gruppo con pixel identici a un'immagine di pannello-v3"
    meta = {**meta0,
            "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
            "precedente": {"percorso": rel(args.manifest), "versione": meta0["versione"],
                           "manifest_sha256": m.manifest_sha256, "records_sha256": meta0["records"]["sha256"]},
            "split": split, "stati": stati,
            "decontaminazione": {
                "controllo": CHECK, "versione": SCHEMA, "regole": RULES,
                "indice": {**file_entry(args.index / "index.json", inp["index_sha"]),
                           "schema": PANEL_SCHEMA, "convenzione_immagini": inp["convention"],
                           "voci": inp["index_voci"], "verifica": file_entry(args.index / "verifica.json")},
                "pannello": {"elements": file_entry(args.panel / "elements.jsonl.gz"),
                             "summary": file_entry(args.panel / "summary.json"),
                             "job": inp["panel_summary"].get("job_id"), "codice": inp["panel_summary"].get("code"),
                             "perimetro": dict(sorted(inp["perimeter"].items()))},
                "scansione": {"summary": file_entry(args.scan / "summary.json"),
                              "records": file_entry(args.scan / "records.jsonl.gz", inp["scan_sha"]),
                              "job": job, "codice": scan_summary.get("codice"), "soglie": scan_summary.get("soglie"),
                              "zip": scan_summary.get("zip"), "shard": scan_summary.get("shard"), **sources},
                "riconciliazione_run2": inp.get("riconciliazione_run2"), "candidati_storici": inp.get("storiche"),
                "regole_testuali": {"regole": TEXT_RULES, "sha256": TEXT_RULES_SHA256}
                if args.source == "mavis" else None,
                "controlli_di_compatibilita": checks, **stats},
            "conteggi": summarize(args.source, new),
            "records": {"file": "records.jsonl.gz", "sha256": data_sha, "righe": n_lines},
            "derivati": derived,
            "fuori_dal_sottoinsieme": [NOT_COVERED[args.source] if x.startswith("decontaminazione esterna") else x
                                       for x in meta0["fuori_dal_sottoinsieme"]],
            "verifiche_mancanti": [MISSING[args.source] if x.startswith("decontaminazione esterna") else x
                                   for x in meta0["verifiche_mancanti"]],
            "secondi_costruzione": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()}
    write_new_json(args.new / "manifest.json", meta)
    del new, inp, decisions, outcomes, scan, m
    gc.collect()
    result = verify_manifest(args)
    result.update({"job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
                   "secondi_totali": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()})
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "verifica.json", result)
    summary = {"schema": "limite-vl-manifest-riepilogo/1", "manifest": rel(args.new), "nome": meta["nome"],
               "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"),
               "manifest_sha256": sha256_file(args.new / "manifest.json"), "records_sha256": data_sha, "righe": n_lines,
               "sorgente": meta["sorgente"], "revisione": meta["revisione"], "precedente": meta["precedente"],
               "decontaminazione": {k: meta["decontaminazione"].get(k) for k in (
                   "esiti_propri", "esiti_dei_gruppi", "cambi_di_stato", "record_con_corrispondenze_per_benchmark_tipo",
                   "corrispondenze_per_benchmark_ruolo_tipo", "gruppi_toccati", "riconciliazione_run2",
                   "candidati_storici", "record_con_sole_corrispondenze_testuali_coperte", "regole_testuali")}
               | {"indice_sha256": meta["decontaminazione"]["indice"]["sha256"],
                  "scansione_records_sha256": meta["decontaminazione"]["scansione"]["records"]["sha256"]},
               "conteggi": meta["conteggi"], "split_ammessi": split["ammessi_dopo_la_decontaminazione"],
               "controlli": [{k: c[k] for k in ("controllo", "esito")} for c in result["controlli"]],
               "esito_verifica": result["esito"], "fuori_dal_sottoinsieme": meta["fuori_dal_sottoinsieme"],
               "verifiche_mancanti": meta["verifiche_mancanti"],
               "nota": "decontaminazione esterna nei controlli dichiarati: non certifica le risposte, non stima la "
                       "qualità e non autorizza il training"}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.summary, summary)
    print(json.dumps({"righe": n_lines, "cambi_di_stato": stats["cambi_di_stato"], "esiti": stats["esiti_propri"],
                      "verifica": f"{result['superati']}/{result['totale']}", "secondi": result["secondi_totali"],
                      "rss_max_mib": result["rss_max_mib"]}, ensure_ascii=False))
    if not result["esito"]:
        raise Failed("verifica della versione nuova fallita: vedi " + str(args.report / "verifica.json"))
    return 0


# ---------------------------------------------------------------- verifica della versione derivata

STATE_FIELDS = ("stato", "motivo", "evidenze", "compiti")


def old_or_new_coverage(ok, source: str, version, dec: dict, rows: dict, records: dict, scan_dir: Path):
    """Coperture con cui una versione derivata è stata decisa, e il campo dell'evidenza che conta le corrispondenze
    testuali senza effetto. panel-decontam/1 (mavis-geometry-candidato-2, unimer-candidato-2): la regola storica di
    frequenza, ricalcolata dalle righe e confrontata con quella registrata, solo per riverificare quelle versioni.
    panel-decontam/2: le regole esplicite, coi testi della scansione legati al manifest."""
    if version == OLD_SCHEMA:
        frequent = frequency_rule_v1(rows.values()) if source == "mavis" else frozenset()
        if source == "mavis":
            declared = dec.get("testo_di_rito") or {}
            ok(f"testo, regola storica di {OLD_SCHEMA} (abolita, ricalcolata solo per riverificare): 10-gram in più di "
               f"{OLD_MAX_DF} record ricalcolati dalle righe, uguali a quelli registrati nel manifest",
               set(declared.get("ngram") or []) == set(frequent) and declared.get("max_df_sorgente") == OLD_MAX_DF,
               sorted(frequent))
        coverage = {i: frequency_coverage_v1(row, frequent) for i, row in rows.items()} if frequent else {}
        return coverage, "testo_di_rito"
    coverage = {}
    if source == "mavis":
        summary = json.loads((scan_dir / "summary.json").read_text(encoding="utf-8"))
        ok("scansione: schema panel-decontam-scan/2, coi testi dei campi con una regola esplicita",
           summary.get("schema") == SCAN_SCHEMA, summary.get("schema"))
        coverage, bad = bind_rule_texts(rows, records)
        ok("testo: per ogni corrispondenza in un campo con una regola, il testo del campo è nella scansione, ha lo sha256 "
           "del manifest e contiene i 10-gram della corrispondenza", not bad, {"casi": len(bad), "esempi": bad[:3]})
        ok("regole testuali registrate nel manifest uguali a quelle del codice (TEXT_RULES e sha256)",
           dec.get("regole_testuali") == {"regole": TEXT_RULES, "sha256": TEXT_RULES_SHA256})
    return coverage, "testo_coperto"


def verify_manifest(args) -> dict:
    """Rilegge la versione derivata (--new) e il genitore (--manifest, --manifest-sha256): verifica del genitore con il
    suo verificatore e le sue evidenze obbligatorie; transizioni dovute alla decontaminazione ricalcolate dalla
    scansione, senza fidarsi delle evidenze scritte."""
    checks = []

    def ok(name: str, cond: bool, detail=None) -> None:
        checks.append({"controllo": name, "esito": bool(cond), "dettaglio": detail})
        print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail is not None else ""),
              flush=True)

    source = args.source
    try:
        parent = mr.load_expected(args.manifest, args.manifest_sha256)
        ok("genitore: manifest.json con lo sha256 atteso, letto e verificato da manifest.load", True,
           parent.manifest_sha256)
    except mr.Problem as exc:
        ok("genitore: manifest.json con lo sha256 atteso, letto e verificato da manifest.load", False, str(exc))
        return finish(checks, args)
    # il verificatore della costruzione vale solo per il candidato iniziale: un genitore già derivato (stati dovuti a
    # una decontaminazione) si verifica con la sua catena (verifica-riesame), non con le aspettative del candidato
    ok("genitore: candidato iniziale (senza split.ereditato_da), a cui si applica il verificatore della costruzione",
       not (parent.meta.get("split") or {}).get("ereditato_da"),
       ((parent.meta.get("split") or {}).get("ereditato_da") or {}).get("percorso"))
    if (parent.meta.get("split") or {}).get("ereditato_da"):
        return finish(checks, args)
    try:
        child = mf.load(args.new)
        ok("versione nuova: letta e verificata da manifest.load (sha256 dei record, record_id ricalcolati, un solo "
           "split per gruppo, conteggi per shard)", True, child.manifest_sha256)
    except mf.ManifestError as exc:
        ok("versione nuova: letta e verificata da manifest.load", False, str(exc))
        return finish(checks, args)
    meta = child.meta
    prev, inherited = meta.get("precedente") or {}, (meta.get("split") or {}).get("ereditato_da") or {}
    ok("versione nuova: precedente ed ereditato_da indicano il genitore, con gli sha256 registrati",
       prev.get("manifest_sha256") == inherited.get("manifest_sha256") == parent.manifest_sha256
       and prev.get("records_sha256") == parent.meta["records"]["sha256"] and meta.get("versione") != parent.meta.get(
           "versione") and meta.get("nome") == parent.meta.get("nome"))
    # ---- genitore: evidenze obbligatorie e verificatore della costruzione
    ns = argparse.Namespace(source=source, check=args.check, overlap=args.overlap, evidence=args.evidence)
    try:
        required = mr.required_evidence(ns)
        mr.evidence_identity(parent.meta.get("liste_importate"), required)
        mr.evidence_identity(meta.get("liste_importate"), required)
        ok("evidenze obbligatorie del genitore registrate con sha256 uguali su disco, nel genitore e nella versione "
           "nuova", meta.get("liste_importate") == parent.meta.get("liste_importate"))
    except mr.Problem as exc:
        ok("evidenze obbligatorie del genitore registrate con sha256 uguali su disco", False, str(exc))
        return finish(checks, args)
    if source == "unimer":
        try:
            mr.shard_classes(parent)
            mr.shard_classes(child)
            ok("classificazione degli shard uguale all'insieme dei loro record, nel genitore e nella versione nuova",
               True)
        except mr.Problem as exc:
            ok("classificazione degli shard uguale all'insieme dei loro record", False, str(exc))
        res = um.verify(parent.folder, Path(args.check), Path(args.overlap), m=parent)
    else:
        res = mm.verify(parent.folder, required, m=parent)
    ok(f"genitore: verifica della costruzione ({'unimer' if source == 'unimer' else 'mavis'}_manifest.verify), "
       f"{res['superati']} su {res['totale']}", res["esito"],
       [c["controllo"] for c in res["controlli"] if not c["esito"]])
    old = {ident(r): r for rows in parent.by_shard.values() for r in rows.values()}
    recs = [r for rows in child.by_shard.values() for r in rows.values()]
    new = {ident(r): r for r in recs}
    # ---- identità
    same_keys = set(old) == set(new) and len(new) == len(recs)
    differ = [label_of(i) for i, r in new.items() if i in old and any(
        r.get(k) != old[i].get(k) for k in set(r) | set(old[i]) if k not in STATE_FIELDS)]
    if source == "mavis":
        differ += [label_of(i) for i, r in new.items() if i in old and [{k: v for k, v in t.items() if k != "stato"}
                                                                        for t in r["compiti"]]
                   != [{k: v for k, v in t.items() if k != "stato"} for t in old[i]["compiti"]]]
    else:
        differ += [label_of(i) for i, r in new.items() if i in old and r["compiti"] != old[i]["compiti"]]
    ok("identità: stessi record (insieme, chiave), e ogni campo uguale al genitore salvo stato, motivi, evidenze e "
       "stati dei compiti (gruppi, split, posizioni, digest, trasformazioni e genitori invariati)",
       same_keys and not differ, {"diversi": len(differ), "esempi": differ[:5]})
    if not same_keys:
        return finish(checks, args)
    # ---- scansione: righe ricalcolate in esiti, senza fidarsi delle evidenze scritte
    dec = meta.get("decontaminazione") or {}
    scan_dir = Path(args.scan)
    scan_sha = sha256_file(scan_dir / "records.jsonl.gz")
    ok("scansione: records.jsonl.gz con lo sha256 registrato nel manifest",
       scan_sha == ((dec.get("scansione") or {}).get("records") or {}).get("sha256"), scan_sha)
    rows = {ident(r): r for r in read_rows(scan_dir / "records.jsonl.gz")}
    ok("scansione: una riga per record, con shard, posizione e digest del manifest",
       set(rows) == set(new) and all(
           (rows[i]["shard"], rows[i]["indice"], rows[i]["json_sha256"], rows[i]["png_sha256"])
           == (r["shard"], r["indice"], r["json_sha256"], r["immagini"][0]["sha256"]) for i, r in new.items()
           if i in rows))
    if set(rows) != set(new):
        return finish(checks, args)
    version = dec.get("versione")
    ok("decontaminazione: versione delle decisioni riconosciuta (panel-decontam/1 storica, panel-decontam/2)",
       version in (OLD_SCHEMA, SCHEMA), version)
    coverage, counted = old_or_new_coverage(ok, source, version, dec, rows, new, scan_dir)
    outcomes = {i: record_outcome(row, source, coverage.get(i)) for i, row in rows.items()}
    decisions = decide(list(old.values()), outcomes)
    # ---- evidenze e motivi
    kept, one_new, wrong_ev, kept_reasons, bad_reason, unexplained = [], [], [], [], [], []
    reasons = set(REASON[source].values())
    for i, r in new.items():
        p, d = old[i], decisions[i]
        ev, pev = r["evidenze"] or [], p["evidenze"] or []
        if ev[:len(pev)] != pev:
            kept.append(label_of(i))
        added = ev[len(pev):]
        if len(added) != 1 or added[0].get("controllo") != CHECK:
            one_new.append(label_of(i))
            continue
        e = added[0]
        if (e.get("esito"), e.get("esito_gruppo"), e.get("cause_nel_gruppo")) != (d["esito"], d["esito_gruppo"],
                                                                                 d["cause"]) \
                or e.get("versione") != version \
                or e.get("rif", {}).get("sha256") != scan_sha or e.get("corrispondenze") != rows[i]["corrispondenze"] \
                or e.get("corrispondenze_nella_scansione") != len(rows[i]["dettagli"]) \
                or e.get(counted) != sum(x["tipo"] == "testo" and not informative(x, coverage.get(i))
                                         for x in rows[i]["dettagli"]):
            wrong_ev.append(label_of(i))
        mo, pmo = r["motivo"] or [], p["motivo"] or []
        if mo[:len(pmo)] != pmo:
            kept_reasons.append(label_of(i))
        extra = mo[len(pmo):]
        expected_extra = [REASON[source][d["esito_gruppo"]]] if d["stato_gruppo"] else []
        if extra != expected_extra or any(x not in reasons for x in extra):
            bad_reason.append(label_of(i))
        if r["stato"] != p["stato"] and not extra:
            unexplained.append(label_of(i))
    ok(f"{CHECK}: evidenze del genitore conservate", not kept, len(kept))
    ok(f"{CHECK}: una e una sola evidenza nuova per record", not one_new, len(one_new))
    ok(f"{CHECK}: esito proprio, esito del gruppo, cause e corrispondenze dell'evidenza uguali a quelli ricalcolati "
       "dalla scansione", not wrong_ev, {"casi": len(wrong_ev), "esempi": wrong_ev[:3]})
    ok(f"{CHECK}: motivi del genitore conservati", not kept_reasons, len(kept_reasons))
    ok(f"{CHECK}: un motivo nuovo per gruppo toccato, quello della regola, e nessun altro", not bad_reason,
       {"casi": len(bad_reason), "esempi": bad_reason[:3]})
    ok(f"{CHECK}: ogni cambio di stato ha il suo motivo nuovo", not unexplained, len(unexplained))
    # ---- stati e compiti
    wrong_state = [label_of(i) for i, r in new.items() if r["stato"] != decisions[i]["stato"]]
    better = [label_of(i) for i, r in new.items() if SEVERITY[r["stato"]] < SEVERITY[old[i]["stato"]]]
    ok(f"{CHECK}: stato = il più grave fra quello del genitore e quello dell'esito del gruppo (ricalcolato dalla "
       "scansione)", not wrong_state, {"casi": len(wrong_state), "esempi": wrong_state[:3]})
    ok("nessun record torna ammesso o passa a uno stato meno grave", not better, len(better))
    group_outcome = defaultdict(set)
    for i, r in new.items():
        mine = [e for e in r["evidenze"] or [] if e.get("controllo") == CHECK]
        group_outcome[r["gruppo"]].add(mine[-1].get("esito_gruppo") if mine else None)
    ok("un solo esito per gruppo", all(len(v) == 1 for v in group_outcome.values()))
    if source == "mavis":
        bad_tasks = [label_of(i) for i, r in new.items() if r["compiti"] != task_states(old[i]["compiti"], r["stato"])
                     or any(SEVERITY[t["stato"]] < SEVERITY[pt["stato"]] for t, pt in zip(r["compiti"], old[i]["compiti"]))
                     or any(t["compito"] in mm.QUARANTINED_TASKS and t["stato"] == "ammesso" for t in r["compiti"])]
        ok("compiti: seguono lo stato del record, shape_type mai ammessa, nessun compito meno grave del genitore",
           not bad_tasks, bad_tasks[:5])
    # ---- conteggi e file derivati
    ok("conteggi uguali a manifest.json (ricalcolati dai record)", summarize(source, recs) == meta.get("conteggi"))
    keys_ok = []
    for name, ks in derived_keys(source, recs).items():
        entry = (meta.get("derivati") or {}).get(f"keys/{name}") or {}
        path = Path(args.new) / "keys" / name
        text = path.read_text(encoding="utf-8") if path.is_file() else None
        keys_ok.append(text == "".join(k + "\n" for k in sorted(ks)) and entry.get("righe") == len(ks)
                       and mf.sha256_text(text or "") == entry.get("sha256"))
    ok("file derivati (keys/) uguali alle chiavi ricalcolate e ai loro sha256", all(keys_ok) and len(keys_ok) == len(
        meta.get("derivati") or {}), keys_ok)
    return finish(checks, args, child)


def finish(checks: list[dict], args, child: mf.Manifest | None = None) -> dict:
    return {"manifest": rel(args.new), "manifest_sha256": child.manifest_sha256 if child else None,
            "genitore": rel(args.manifest), "genitore_sha256": args.manifest_sha256, "controlli": checks,
            "superati": sum(c["esito"] for c in checks), "totale": len(checks),
            "esito": bool(checks) and all(c["esito"] for c in checks)}


def cmd_verifica_manifest(args) -> int:
    if args.report.exists() and any(args.report.iterdir()):
        raise Failed(f"{args.report} non è vuota")
    t0 = time.time()
    result = verify_manifest(args)
    result.update({"job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
                   "secondi": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()})
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "verifica.json", result)
    print(f"{result['superati']} controlli superati su {result['totale']}")
    return 0 if result["esito"] else 1


# ---------------------------------------------------------------- riesame delle decisioni testuali (MAVIS)

RIESAME_SCHEMA = "panel-decontam-riesame/1"
REASON_RIESAME = {
    "piu_grave": "decontaminazione_pannello, riesame panel-decontam/2: corrispondenza testuale da verificare, non coperta "
                 "da una regola esplicita, che la regola di frequenza abolita ignorava (gruppo in quarantena)",
    "meno_grave": "decontaminazione_pannello, riesame panel-decontam/2: le corrispondenze testuali che rendevano il "
                  "gruppo candidato sono coperte da una regola esplicita; resta lo stato della versione precedente "
                  "(nessuna riammissione senza una decisione)"}
CONTEXT_WORDS, MAX_TEXT = 15, 2000  # parole attorno al primo 10-gram e caratteri dei testi riportati nei casi


def transition_reason(before: str | None, after: str | None) -> str | None:
    """Motivo nuovo del riesame quando l'esito del gruppo cambia (più grave o meno grave); None se non cambia."""
    if before == after:
        return None
    if before not in OUTCOMES or after not in OUTCOMES:
        return f"decontaminazione_pannello, riesame: esito del gruppo non riconosciuto ({before} -> {after})"
    worse = OUTCOMES.index(after) < OUTCOMES.index(before)
    return REASON_RIESAME["piu_grave" if worse else "meno_grave"]


def last_check(r: dict) -> dict:
    """L'ultima evidenza della decontaminazione esterna di un record: quella della versione che lo ha deciso."""
    mine = [e for e in r.get("evidenze") or [] if e.get("controllo") == CHECK]
    return mine[-1] if mine else {}


class CountingFile(io.FileIO):
    """File senza buffer che conta i byte letti: il riesame dichiara quanto ha letto degli shard."""
    letti = 0

    def read(self, size=-1):
        data = super().read(size)
        self.letti += len(data or b"")
        return data

    def readinto(self, b):
        n = super().readinto(b)
        self.letti += n or 0
        return n


def record_texts(m: mf.Manifest, wanted: dict) -> tuple[dict, dict, list[str]]:
    """Soli JSON dei record di `wanted` ({identità: campi}) dagli shard del manifest, senza leggere immagini: le
    intestazioni del tar si leggono in sequenza saltando i dati (512 byte per membro, file senza buffer) e ogni JSON
    scelto deve avere lo json_sha256 e la posizione del record. Per ogni record la descrizione canonica del loader è
    ricalcolata come nella scansione (mavis_truth, describe con la chiave del tar), le domande del loader con
    mavis_questions, gli altri testi sono quelli di text_en. Restituisce testi, conteggi della lettura e problemi."""
    recs = {ident(r): r for rows in m.by_shard.values() for r in rows.values()}
    by_shard: dict[str, dict[str, tuple]] = defaultdict(dict)
    for i in wanted:
        by_shard[recs[i]["shard"]][recs[i]["record"]] = i
    out, problems, stats = {}, [], Counter()
    t0 = time.time()
    for shard in sorted(by_shard):
        keys = by_shard[shard]
        pos, last, found = -1, None, 0
        with CountingFile(mf.shard_path(shard), "rb") as raw, tarfile.open(fileobj=raw, mode="r:") as tar:
            for member in tar:
                stats["intestazioni_lette"] += 1
                if not member.isfile():
                    continue
                stem, _, ext = member.name.rpartition("/")[2].partition(".")
                if stem != last:
                    pos, last = pos + 1, stem
                if ext != "json" or stem not in keys:
                    continue
                i = keys[stem]
                data = tar.extractfile(member).read()
                stats["json_letti"] += 1
                r = recs[i]
                if i in out:
                    problems.append(f"{label_of(i)}: JSON ripetuto nello shard")
                    continue
                if hashlib.sha256(data).hexdigest() != r["json_sha256"] or pos != r["indice"]:
                    problems.append(f"{label_of(i)}: JSON con sha256 o posizione diversi dal manifest")
                    continue
                te = json.loads(data).get("text_en") or {}
                caption, generation = te.get("caption"), te.get("diagram_generation")
                truth = mavis_truth(caption, generation)[0] if caption and generation else None
                texts = {}
                for f in wanted[i]:
                    if f == "descrizione_del_loader":
                        texts[f] = describe(stem, truth)[1] if truth is not None else None
                    elif f == "domande_del_loader":
                        texts[f] = [q["q"] for q in mavis_questions(stem, truth)] if truth is not None else []
                    else:
                        texts[f] = te.get(f)
                out[i] = {"json_sha256": r["json_sha256"], "testi": texts}
                found += 1
            stats["byte_letti"] += raw.letti
        stats["shard_aperti"] += 1
        print(f"{Path(shard).name}: {found} JSON su {len(keys)} cercati ({time.time() - t0:.0f} s)", flush=True)
    missing = [label_of(i) for i in wanted if i not in out]
    if missing:
        problems.append(f"{len(missing)} record senza JSON letto, es. {missing[:3]}")
    stats["byte_degli_shard_aperti"] = sum(mf.shard_path(s).stat().st_size for s in by_shard)
    stats["secondi"] = round(time.time() - t0, 1)
    return out, dict(stats), problems


def element_texts(index_dir: Path, wanted: set[str]) -> dict[str, list[str]]:
    """Testi dell'indice degli elementi richiesti (ogni voce distinta una volta)."""
    loaded, _ = read_index(index_dir)
    out: dict[str, list[str]] = {}
    for it in loaded["items"]:
        e, text = element_of(it), (it.get("text") or "").strip()
        if e in wanted and text and text not in out.setdefault(e, []):
            out[e].append(text)
    del loaded
    gc.collect()
    return out


def word_position(words: list[str], gram: str) -> int | None:
    g = gram.split()
    for k in range(len(words) - len(g) + 1):
        if words[k:k + len(g)] == g:
            return k
    return None


def case_context(field_text, bench_texts: list[str], grams: list[str]) -> dict:
    """Contesto di una corrispondenza testuale informativa per chi la esamina: il testo del record (fra le domande del
    loader quella che contiene il 10-gram), il testo del benchmark che lo contiene, la sequenza comune più lunga in
    parole normalizzate e le parole attorno al primo 10-gram. Non decide nulla."""
    if isinstance(field_text, list):
        field_text = next((q for q in field_text if word_position(normalize_words(q), grams[0]) is not None), "")
    bench = next((t for t in bench_texts if word_position(normalize_words(t), grams[0]) is not None),
                 bench_texts[0] if bench_texts else "")
    a, b = normalize_words(field_text or ""), normalize_words(bench)
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    ka, kb = word_position(a, grams[0]), word_position(b, grams[0])

    def around(ws, k):
        return None if k is None else " ".join(ws[max(0, k - CONTEXT_WORDS):k + NGRAM + CONTEXT_WORDS])

    return {"testo_del_record": (field_text or "")[:MAX_TEXT], "testo_del_benchmark": bench[:MAX_TEXT],
            "parole_del_record": len(a), "parole_del_benchmark": len(b), "parole_comuni_consecutive": m.size,
            "sequenza_comune_piu_lunga": " ".join(a[m.a:m.a + m.size]),
            "contesto_nel_record": around(a, ka), "contesto_nel_benchmark": around(b, kb)}


def riesame_inputs(args, check) -> dict:
    """Ingressi del riesame: il genitore (versione decisa con panel-decontam/1) derivato dal candidato iniziale; la sua
    scansione, l'indice e il pannello invariati (sha256 registrati nel genitore); la derivazione riverificata con la
    regola storica (verify_manifest, che verifica anche il candidato iniziale con mavis_manifest.verify)."""
    parent_meta, _ = manifest_meta(args.manifest, args.manifest_sha256)
    origin_meta, _ = manifest_meta(args.origine, args.origine_sha256)
    pdec = parent_meta.get("decontaminazione") or {}
    prev = parent_meta.get("precedente") or {}
    inherited = (parent_meta.get("split") or {}).get("ereditato_da") or {}
    check("genitore: derivato dal candidato iniziale indicato (precedente ed ereditato_da con lo sha256 di --origine), "
          "che non eredita da altre versioni",
          prev.get("manifest_sha256") == inherited.get("manifest_sha256") == args.origine_sha256
          and prev.get("percorso") == inherited.get("percorso") == rel(args.origine)
          and not (origin_meta.get("split") or {}).get("ereditato_da"))
    check("genitore: MAVIS, decisioni panel-decontam/1 con la regola di frequenza da riesaminare",
          parent_meta.get("sorgente") == SOURCES["mavis"] and pdec.get("controllo") == CHECK
          and pdec.get("versione") == OLD_SCHEMA and bool(pdec.get("testo_di_rito")), pdec.get("versione"))
    scan_ref, panel_ref = pdec.get("scansione") or {}, pdec.get("pannello") or {}
    scan_rec, scan_sum = sha256_file(args.scan / "records.jsonl.gz"), sha256_file(args.scan / "summary.json")
    check("scansione originale invariata: records.jsonl.gz e summary.json con gli sha256 registrati nel genitore",
          scan_rec == (scan_ref.get("records") or {}).get("sha256")
          and scan_sum == (scan_ref.get("summary") or {}).get("sha256"), scan_rec)
    isha = sha256_file(args.index / "index.json")
    check("indice invariato: index.json con lo sha256 atteso, lo stesso registrato nel genitore",
          isha == args.index_sha256 == (pdec.get("indice") or {}).get("sha256"), isha)
    check("pannello invariato: elements.jsonl.gz e summary.json con gli sha256 registrati nel genitore",
          sha256_file(args.panel / "elements.jsonl.gz") == (panel_ref.get("elements") or {}).get("sha256")
          and sha256_file(args.panel / "summary.json") == (panel_ref.get("summary") or {}).get("sha256"))
    if not all(c["esito"] for c in check.done):
        return {}
    ns = argparse.Namespace(source="mavis", manifest=args.origine, manifest_sha256=args.origine_sha256,
                            new=args.manifest, check=None, overlap=None, evidence=args.evidence, scan=args.scan)
    res = verify_manifest(ns)
    check(f"genitore: derivazione dal candidato iniziale riverificata con la sua regola storica (verifica-manifest, "
          f"{res['superati']} su {res['totale']}, compreso mavis_manifest.verify del candidato iniziale)", res["esito"],
          [c["controllo"] for c in res["controlli"] if not c["esito"]])
    gc.collect()
    return {"pdec": pdec, "scan_rec": scan_rec, "scan_sum": scan_sum, "isha": isha,
            "verifica_genitore": {k: res[k] for k in ("superati", "totale", "esito")}}


def cmd_riesame(args) -> int:
    t0 = time.time()
    if args.source != "mavis":
        raise Failed("riesame: solo MAVIS ha decisioni testuali da ricalcolare")
    for p in (args.out, args.new, args.report):
        if p.exists() and any(p.iterdir()):
            raise Failed(f"{p} non è vuota: il riesame si scrive in cartelle nuove")
    if args.summary.exists():
        raise Failed(f"{args.summary} esiste già")
    checks: list[dict] = []

    def check(name: str, cond, detail=None) -> bool:
        checks.append({"controllo": name, "esito": bool(cond), "dettaglio": detail})
        print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail is not None else ""),
              flush=True)
        return bool(cond)

    check.done = checks

    def stop_if_failed(what: str) -> None:
        failed = [c["controllo"] for c in checks if not c["esito"]]
        if failed:
            raise Failed(f"riesame: {what}, nessuna uscita scritta: " + "; ".join(failed[:5]))

    inp = riesame_inputs(args, check)
    stop_if_failed("ingressi non verificati")
    pdec = inp["pdec"]
    origin = mr.load_expected(args.origine, args.origine_sha256)
    parent = mr.load_expected(args.manifest, args.manifest_sha256)
    v1 = {ident(r): r for rs in origin.by_shard.values() for r in rs.values()}
    v2 = {ident(r): r for rs in parent.by_shard.values() for r in rs.values()}
    rows = {ident(r): r for r in read_rows(args.scan / "records.jsonl.gz")}
    check("scansione, genitore e candidato iniziale: gli stessi record", set(rows) == set(v2) == set(v1), len(rows))
    stop_if_failed("record diversi")
    # ---- testi dei record con corrispondenze testuali: solo JSON, nessuna immagine
    wanted = {i: sorted({d["campo"] for d in row["dettagli"] if d["tipo"] == "testo"}) for i, row in rows.items()
              if any(d["tipo"] == "testo" for d in row["dettagli"])}
    texts, reading, bad = record_texts(parent, wanted)
    check("JSON dei record con corrispondenze testuali letti dagli shard, una volta ciascuno, con lo sha256 e la "
          "posizione del manifest (immagini saltate)", not bad, {"record": len(wanted), "problemi": bad[:3], **reading})
    descriptions = {i: {f: t["testi"][f] for f in RULE_FIELDS if isinstance(t["testi"].get(f), str)}
                    for i, t in texts.items()}
    descriptions = {i: d for i, d in descriptions.items() if d}
    coverage, bad = bind_rule_texts(rows, v2, descriptions)
    check("testo: per ogni corrispondenza nella descrizione del loader, la descrizione ricalcolata dal JSON ha lo sha256 "
          "del manifest (descrizione_sha256) e contiene i 10-gram della corrispondenza", not bad,
          {"casi": len(bad), "esempi": bad[:3]})
    # ---- decisione precedente (regola storica) e decisione nuova (regole esplicite)
    frequent = frequency_rule_v1(rows.values())
    check("regola storica ricalcolata dalle righe, solo per il confronto: gli stessi 10-gram registrati nel genitore",
          set(frequent) == set((pdec.get("testo_di_rito") or {}).get("ngram") or []), sorted(frequent))
    old_cov = {i: frequency_coverage_v1(row, frequent) for i, row in rows.items() if row["dettagli"]}
    old_out = {i: record_outcome(row, "mavis", old_cov.get(i)) for i, row in rows.items()}
    new_out = {i: record_outcome(row, "mavis", coverage.get(i)) for i, row in rows.items()}
    before = {i: last_check(r) for i, r in v2.items()}
    check("decisione precedente: esiti propri ricalcolati con la regola storica uguali a quelli delle evidenze "
          "panel-decontam/1 del genitore", all(before[i].get("esito") == old_out[i]
                                               and before[i].get("versione") == OLD_SCHEMA for i in v2))
    stop_if_failed("decisioni non ricalcolabili")
    d3 = decide(list(v2.values()), new_out)
    d1 = decide(list(v1.values()), new_out)
    kept_by_old = sorted(label_of(i) for i in v2 if d3[i]["stato"] != d1[i]["stato"])
    del origin, v1, d1
    gc.collect()

    def flipped(i) -> list[dict]:
        return [{"campo": x["campo"], "elemento": x["elemento"], "ngram_tutti": x["ngram_tutti"],
                 "prima": "candidato" if informative(x, old_cov.get(i)) else "ignorata_per_frequenza",
                 "dopo": "candidato" if informative(x, coverage.get(i))
                 else "coperta: " + ", ".join(covered_rules(x, coverage.get(i)))}
                for x in rows[i]["dettagli"]
                if x["tipo"] == "testo" and informative(x, old_cov.get(i)) != informative(x, coverage.get(i))]

    changes = []
    for i, r in v2.items():
        b, d = before[i], d3[i]
        if (b.get("esito"), b.get("esito_gruppo"), r["stato"]) == (d["esito"], d["esito_gruppo"], d["stato"]):
            continue
        if b.get("esito") != d["esito"]:
            worse = OUTCOMES.index(d["esito"]) < OUTCOMES.index(b.get("esito"))
            cause = ("esito proprio: corrispondenza testuale non coperta da una regola esplicita, prima ignorata per la "
                     "sola frequenza" if worse else "esito proprio: corrispondenze testuali coperte da una regola "
                                                         "esplicita")
        else:
            cause = "esito del gruppo, per " + ", ".join(d["cause"][:5]) + (" e altri" if len(d["cause"]) > 5 else "")
        changes.append({"insieme": r["insieme"], "record": r["record"], "gruppo": r["gruppo"], "split": r["split"],
                        "esito": [b.get("esito"), d["esito"]], "esito_gruppo": [b.get("esito_gruppo"), d["esito_gruppo"]],
                        "stato": [r["stato"], d["stato"]], "causa": cause, "corrispondenze_cambiate": flipped(i),
                        "cause_nel_gruppo": d["cause"], "motivo_nuovo": transition_reason(b.get("esito_gruppo"),
                                                                                          d["esito_gruppo"])})
    changes.sort(key=lambda c: (c["gruppo"], c["insieme"], c["record"]))
    changed = {(c["insieme"], c["record"]) for c in changes}
    # ---- frequenze dei 10-gram: solo per ordinare i casi
    table: dict[str, dict] = {}
    for i in wanted:
        for d in rows[i]["dettagli"]:
            if d["tipo"] != "testo":
                continue
            src = texts[i]["testi"].get(d["campo"])
            src_id = mf.sha256_text(" ".join(normalize_words(src if isinstance(src, str) else " ".join(src or []))))
            for g in d["ngram_tutti"]:
                t = table.setdefault(g, {"record": set(), "gruppi": set(), "testi": set(), "campi": Counter(),
                                         "elementi": set(), "per_regola": Counter()})
                t["record"].add(i)
                t["gruppi"].add(v2[i]["gruppo"])
                t["testi"].add(src_id)
                t["campi"][d["campo"]] += 1
                t["elementi"].add(d["elemento"])
                t["per_regola"][((coverage.get(i) or {}).get(d["campo"]) or {}).get(g, "non_coperta")] += 1
    ngrams = [{"ngram": g, "record": len(t["record"]), "gruppi": len(t["gruppi"]), "testi_distinti": len(t["testi"]),
               "campi": dict(t["campi"].most_common()), "elementi": sorted(t["elementi"])[:5],
               "elementi_totali": len(t["elementi"]), "corrispondenze_per_regola": dict(t["per_regola"].most_common()),
               "ignorato_dalla_regola_precedente": g in frequent}
              for g, t in sorted(table.items(), key=lambda kv: (-len(kv[1]["record"]), kv[0]))]
    del table
    # ---- casi testuali informativi, coi contesti (record, benchmark)
    cases = [(i, d) for i in sorted(wanted) for d in rows[i]["dettagli"]
             if d["tipo"] == "testo" and informative(d, coverage.get(i))]
    bench = element_texts(args.index, {d["elemento"] for _, d in cases})
    elements = {e["chiave"]: e for e in sd.read_jsonl_gz(args.panel / "elements.jsonl.gz")}
    by_case: dict[tuple, list] = defaultdict(list)
    for i, d in cases:
        ctx = case_context(texts[i]["testi"].get(d["campo"]), bench.get(d["elemento"]) or [], d["ngram_tutti"])
        by_case[(d["elemento"], d["campo"])].append({
            "record": i[1], "gruppo": v2[i]["gruppo"], "split": v2[i]["split"], "stato": [v2[i]["stato"], d3[i]["stato"]],
            "esito": [old_out[i], new_out[i]], "ngram_tutti": d["ngram_tutti"],
            "ignorata_dalla_regola_precedente": not informative(d, old_cov.get(i)),
            **{k: v for k, v in ctx.items() if k != "testo_del_benchmark"}})
    case_rows = [{"elemento": e, "campo": f, "record": len(rs), "gruppi": len({x["gruppo"] for x in rs}),
                  "ignorati_dalla_regola_precedente": sum(x["ignorata_dalla_regola_precedente"] for x in rs),
                  "parole_comuni_consecutive_max": max(x["parole_comuni_consecutive"] for x in rs), "casi": rs}
                 for (e, f), rs in sorted(by_case.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    case_elements = {e: {"benchmark": (elements.get(e) or {}).get("benchmark"), "ruolo": role_of(elements.get(e)),
                         "gruppo_pannello": (elements.get(e) or {}).get("gruppo"),
                         "metrica_primaria": (elements.get(e) or {}).get("metrica_primaria"),
                         "testi": [t[:MAX_TEXT] for t in bench.get(e) or []]} for e in sorted({e for e, _ in by_case})}
    # ---- conteggi prima e dopo, per record e per gruppo

    def tally(values) -> dict:
        return dict(sorted(Counter(values).items()))

    g_before = {r["gruppo"]: before[i].get("esito_gruppo") for i, r in v2.items()}
    g_after = {r["gruppo"]: d3[i]["esito_gruppo"] for i, r in v2.items()}
    counts = {
        "record": {"esiti_propri_prima": tally(old_out.values()), "esiti_propri_dopo": tally(new_out.values()),
                   "transizioni_degli_esiti_propri": tally(f"{old_out[i]} -> {new_out[i]}" for i in v2
                                                           if old_out[i] != new_out[i]),
                   "stati_prima": tally(r["stato"] for r in v2.values()), "stati_dopo": tally(d3[i]["stato"] for i in v2),
                   "cambi_di_stato": tally(f"{r['split']}: {r['stato']} -> {d3[i]['stato']}" for i, r in v2.items()
                                           if r["stato"] != d3[i]["stato"]),
                   "ammessi_per_split_prima": tally(r["split"] for r in v2.values() if r["stato"] == "ammesso"),
                   "ammessi_per_split_dopo": tally(r["split"] for i, r in v2.items() if d3[i]["stato"] == "ammesso"),
                   "con_corrispondenze_testuali": len(wanted), "cambiati": len(changes)},
        "gruppi": {"totali": len(g_before), "esiti_prima": tally(g_before.values()), "esiti_dopo": tally(g_after.values()),
                   "transizioni": tally(f"{g_before[g]} -> {g_after[g]}" for g in g_before if g_before[g] != g_after[g]),
                   "toccati_prima": sum(o in GROUP_STATE for o in g_before.values()),
                   "toccati_dopo": sum(o in GROUP_STATE for o in g_after.values()),
                   "con_un_cambio_di_stato": len({v2[i]["gruppo"] for i in v2 if v2[i]["stato"] != d3[i]["stato"]})}}
    # ---- uscite del riesame: regole, decisioni per record, riepilogo (scritte una volta)
    args.out.mkdir(parents=True, exist_ok=True)
    rules_doc = {"schema": RIESAME_SCHEMA + "-regole", "versione_decisioni": SCHEMA, "regole_testuali": TEXT_RULES,
                 "sha256": TEXT_RULES_SHA256, "regole": RULES,
                 "regola_precedente": {"versione": OLD_SCHEMA, "registrata_nel_genitore": pdec.get("testo_di_rito"),
                                       "ricalcolata": sorted(frequent),
                                       "stato": "abolita: la frequenza di un 10-gram nella sorgente non è un motivo "
                                                "sufficiente per ignorare una corrispondenza"},
                 "frequenza": "solo per raggruppare e ordinare i casi da esaminare (summary.json, ngram)",
                 "codice": code_identity()}
    write_new_json(args.out / "regole.json", rules_doc)

    def decision_line(i) -> str:
        r, b, d = v2[i], before[i], d3[i]
        row = rows[i]
        return json.dumps({
            "insieme": i[0], "record": i[1], "gruppo": r["gruppo"], "split": r["split"], "json_sha256": r["json_sha256"],
            "descrizione_sha256": r.get("descrizione_sha256"),
            "descrizione_del_loader": (descriptions.get(i) or {}).get("descrizione_del_loader"),
            "dettagli_testuali": [{"campo": x["campo"], "elemento": x["elemento"], "ngram_tutti": x["ngram_tutti"],
                                   "regole": covered_rules(x, coverage.get(i)), "informativa": informative(x, coverage.get(i)),
                                   "ignorata_dalla_regola_precedente": not informative(x, old_cov.get(i))}
                                  for x in row["dettagli"] if x["tipo"] == "testo"],
            "esito": [b.get("esito"), d["esito"]], "esito_gruppo": [b.get("esito_gruppo"), d["esito_gruppo"]],
            "stato": [r["stato"], d["stato"]]}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    dsha, n_dec = um.write_records_gz(args.out / "decisioni.jsonl.gz",
                                      (decision_line(i) for i in sorted(set(wanted) | changed)))
    summary = {
        "schema": RIESAME_SCHEMA, "sorgente": "mavis", "job_id": os.environ.get("SLURM_JOB_ID"),
        "codice": code_identity(), "args": {k: str(v) for k, v in vars(args).items() if k != "func"},
        "ingressi": {"origine": {"percorso": rel(args.origine), "manifest_sha256": args.origine_sha256},
                     "genitore": {"percorso": rel(args.manifest), "manifest_sha256": args.manifest_sha256,
                                  "versione_decisioni": OLD_SCHEMA},
                     "scansione": {"records": file_entry(args.scan / "records.jsonl.gz", inp["scan_rec"]),
                                   "summary": file_entry(args.scan / "summary.json", inp["scan_sum"]),
                                   "job": (pdec.get("scansione") or {}).get("job")},
                     "indice": file_entry(args.index / "index.json", inp["isha"]),
                     "pannello": {"elements": file_entry(args.panel / "elements.jsonl.gz"),
                                  "summary": file_entry(args.panel / "summary.json")}},
        "verifica_del_genitore": inp["verifica_genitore"],
        "regole": {"file": file_entry(args.out / "regole.json"), "regole_testuali_sha256": TEXT_RULES_SHA256,
                   "regola_precedente_ricalcolata": sorted(frequent)},
        "letture": reading, "decisioni": {"file": file_entry(args.out / "decisioni.jsonl.gz", dsha), "righe": n_dec},
        "conteggi": counts, "ngram": ngrams, "casi_testuali_informativi": case_rows, "elementi_dei_casi": case_elements,
        "cambi": changes, "stati_mantenuti_dalla_sola_decisione_precedente": kept_by_old,
        "nota": "la decisione nuova non usa la frequenza: le tabelle ngram servono solo a ordinare i casi. Nessun "
                "candidato è risolto; nessuna riammissione",
        "controlli": checks, "secondi": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()}
    write_new_json(args.out / "summary.json", summary)
    print(json.dumps({"cambi": len(changes), "conteggi": counts}, ensure_ascii=False)[:4000], flush=True)
    # ---- versione nuova del manifest, derivata dal genitore
    job = os.environ.get("SLURM_JOB_ID")
    ref = {"file": rel(args.scan / "records.jsonl.gz"), "sha256": inp["scan_rec"]}
    dref = {"file": rel(args.out / "decisioni.jsonl.gz"), "sha256": dsha}
    stats, _, _ = binding_report({"scan": rows, "elements": elements, "coverage": coverage,
                                  "records": list(v2.values())}, args)
    new, lines = [], []
    for r in parent_lines(parent):
        i = ident(r)
        b, d = before[i], d3[i]
        rec = dict(r)
        extra = {"esito_precedente": b.get("esito"), "esito_gruppo_precedente": b.get("esito_gruppo"),
                 "versione_precedente": b.get("versione"), "job_della_scansione": b.get("job"), "riesame": dref}
        rec["evidenze"] = list(r["evidenze"] or []) + [evidence_of(rows[i], d, elements, ref, job, coverage.get(i),
                                                                   extra)]
        reason = transition_reason(b.get("esito_gruppo"), d["esito_gruppo"])
        if reason:
            rec["motivo"] = list(r["motivo"] or []) + [reason]
        rec["stato"] = d["stato"]
        rec["compiti"] = task_states(r["compiti"], d["stato"])
        new.append({k: rec[k] for k in ("record", "insieme", "split", "stato", "gruppo", "compiti", "motivo",
                                        "trasformazione", "famiglia", "misure_razionali_come_decimali") if k in rec})
        lines.append(json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    args.new.mkdir(parents=True, exist_ok=True)
    data_sha, n_lines = um.write_records_gz(args.new / "records.jsonl.gz", iter(lines))
    del lines
    (args.new / "keys").mkdir()
    derived = {}
    for name, ks in derived_keys("mavis", new).items():
        text = "".join(k + "\n" for k in sorted(ks))
        (args.new / "keys" / name).write_text(text, encoding="utf-8")
        derived[f"keys/{name}"] = {**file_entry(args.new / "keys" / name, mf.sha256_text(text)), "righe": len(ks)}
    meta0 = parent.meta
    split = dict(meta0["split"])
    split["ereditato_da"] = {"percorso": rel(args.manifest), "versione": meta0["versione"],
                             "manifest_sha256": parent.manifest_sha256}
    admitted = Counter(f"{x['split']}/{x.get('famiglia') or x['insieme']}" for x in new if x["stato"] == "ammesso")
    split["ammessi_dopo_la_decontaminazione"] = dict(sorted(admitted.items()))
    chain = [{"versione": (meta0.get("precedente") or {}).get("versione"), "percorso": rel(args.origine),
              "manifest_sha256": args.origine_sha256, "decisione": "candidato iniziale (mavis_manifest.py): assegna "
                                                                   "gruppi e split"},
             {"versione": meta0["versione"], "percorso": rel(args.manifest), "manifest_sha256": parent.manifest_sha256,
              "job": meta0.get("job_id"), "decisione": "decontaminazione esterna contro pannello-v3 con "
                                                       f"{OLD_SCHEMA} (regola di frequenza nella sorgente, abolita)"},
             {"versione": args.version, "percorso": rel(args.new), "manifest_sha256": None, "job": job,
              "decisione": f"riesame delle decisioni testuali con {SCHEMA} (regole esplicite), sulla stessa scansione"}]
    meta = {**meta0,
            "versione": args.version, "job_id": job, "codice": code_identity(),
            "precedente": {"percorso": rel(args.manifest), "versione": meta0["versione"],
                           "manifest_sha256": parent.manifest_sha256, "records_sha256": meta0["records"]["sha256"]},
            "split": split, "catena_delle_versioni": chain,
            "decontaminazione": {
                "controllo": CHECK, "versione": SCHEMA, "regole": RULES,
                "regole_testuali": {"regole": TEXT_RULES, "sha256": TEXT_RULES_SHA256},
                "indice": pdec["indice"], "pannello": pdec["pannello"], "scansione": pdec["scansione"],
                "riesame": {"regole": file_entry(args.out / "regole.json"),
                            "decisioni": file_entry(args.out / "decisioni.jsonl.gz", dsha),
                            "summary": file_entry(args.out / "summary.json"), "job": job, "letture": reading},
                "decisione_precedente": {"versione": OLD_SCHEMA, "manifest": rel(args.manifest),
                                         "manifest_sha256": parent.manifest_sha256, "regola": pdec.get("testo_di_rito"),
                                         "stato": "abolita; i suoi esiti restano nelle evidenze panel-decontam/1"},
                "candidati_storici": pdec.get("candidati_storici"),
                "controlli_di_compatibilita": checks, **stats},
            "conteggi": summarize("mavis", new),
            "records": {"file": "records.jsonl.gz", "sha256": data_sha, "righe": n_lines},
            "derivati": derived,
            "secondi_costruzione": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()}
    write_new_json(args.new / "manifest.json", meta)
    del new, rows, v2, parent, d3, before, texts, descriptions, coverage, old_cov
    gc.collect()
    args.riesame = args.out
    result = verify_riesame(args)
    result.update({"job_id": job, "codice": code_identity(), "secondi_totali": round(time.time() - t0, 1),
                   "rss_max_mib": um.rss_mib()})
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "verifica.json", result)
    git_summary = {
        "schema": "limite-vl-manifest-riepilogo/1", "manifest": rel(args.new), "nome": meta["nome"],
        "versione": args.version, "job_id": job, "manifest_sha256": sha256_file(args.new / "manifest.json"),
        "records_sha256": data_sha, "righe": n_lines, "sorgente": meta["sorgente"], "revisione": meta["revisione"],
        "precedente": meta["precedente"], "catena_delle_versioni": chain,
        # in git niente testi interi di benchmark o sorgente (non si ridistribuiscono): stanno in --out/summary.json
        "riesame": {"regole_testuali": TEXT_RULES, "regole_testuali_sha256": TEXT_RULES_SHA256,
                    "regola_precedente": rules_doc["regola_precedente"], "conteggi": counts, "ngram": ngrams,
                    "casi_testuali_informativi": [
                        {**{k: v for k, v in c.items() if k != "casi"},
                         "casi": [{k: x[k] for k in ("record", "gruppo", "split", "stato", "esito",
                                                     "ignorata_dalla_regola_precedente", "parole_comuni_consecutive",
                                                     "sequenza_comune_piu_lunga")} for x in c["casi"]]}
                        for c in case_rows],
                    "elementi_dei_casi": {e: {k: v for k, v in x.items() if k != "testi"}
                                          for e, x in case_elements.items()},
                    "cambi": changes,
                    "stati_mantenuti_dalla_sola_decisione_precedente": kept_by_old, "letture": reading,
                    "file": {k: meta["decontaminazione"]["riesame"][k] for k in ("regole", "decisioni", "summary")}},
        "decontaminazione": {k: meta["decontaminazione"].get(k) for k in (
            "esiti_propri", "esiti_dei_gruppi", "cambi_di_stato", "record_con_corrispondenze_per_benchmark_tipo",
            "corrispondenze_per_benchmark_ruolo_tipo", "gruppi_toccati",
            "record_con_sole_corrispondenze_testuali_coperte")}
        | {"indice_sha256": pdec["indice"]["sha256"], "scansione_records_sha256": inp["scan_rec"]},
        "conteggi": meta["conteggi"], "split_ammessi": split["ammessi_dopo_la_decontaminazione"],
        "controlli": [{k: c[k] for k in ("controllo", "esito")} for c in result["controlli"]],
        "esito_verifica": result["esito"], "fuori_dal_sottoinsieme": meta["fuori_dal_sottoinsieme"],
        "verifiche_mancanti": meta["verifiche_mancanti"],
        "nota": "riesame delle decisioni testuali sulla scansione esistente: nessuna immagine riletta, nessun candidato "
                "risolto, nessuna riammissione; non certifica le risposte e non autorizza il training"}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.summary, git_summary)
    print(json.dumps({"righe": n_lines, "cambi": len(changes), "verifica": f"{result['superati']}/{result['totale']}",
                      "secondi": result["secondi_totali"], "rss_max_mib": result["rss_max_mib"]}, ensure_ascii=False))
    if not result["esito"]:
        raise Failed("verifica della versione del riesame fallita: vedi " + str(args.report / "verifica.json"))
    return 0


def verify_riesame(args) -> dict:
    """Rilegge la versione del riesame (--new) senza fidarsi di quanto il riesame ha scritto: il genitore (--manifest)
    con la sua regola storica e il candidato iniziale (--origine) con mavis_manifest.verify; la catena delle versioni;
    scansione, indice e pannello invariati; le descrizioni del riesame legate al manifest; le decisioni ricalcolate
    con TEXT_RULES dalla scansione; evidenze, motivi, stati e compiti; il confronto completo col genitore e l'elenco dei
    cambi del riepilogo."""
    checks = []

    def ok(name: str, cond, detail=None) -> None:
        checks.append({"controllo": name, "esito": bool(cond), "dettaglio": detail})
        print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail is not None else ""),
              flush=True)

    ns = argparse.Namespace(source="mavis", manifest=args.origine, manifest_sha256=args.origine_sha256,
                            new=args.manifest, check=None, overlap=None, evidence=args.evidence, scan=args.scan)
    res = verify_manifest(ns)
    ok(f"genitore: derivazione dal candidato iniziale con la sua regola storica, {res['superati']} su {res['totale']} "
       "(compreso mavis_manifest.verify del candidato iniziale)", res["esito"],
       [c["controllo"] for c in res["controlli"] if not c["esito"]])
    gc.collect()
    try:
        parent = mr.load_expected(args.manifest, args.manifest_sha256)
        child = mf.load(args.new)
        ok("versione nuova: letta e verificata da manifest.load (sha256 dei record, record_id ricalcolati, un solo "
           "split per gruppo, conteggi per shard)", True, child.manifest_sha256)
    except (mr.Problem, mf.ManifestError) as exc:
        ok("versione nuova e genitore letti e verificati da manifest.load", False, str(exc))
        return finish(checks, args)
    meta, pm = child.meta, parent.meta
    dec, pdec = meta.get("decontaminazione") or {}, pm.get("decontaminazione") or {}
    prev, inherited = meta.get("precedente") or {}, (meta.get("split") or {}).get("ereditato_da") or {}
    pinh = (pm.get("split") or {}).get("ereditato_da") or {}
    ok("catena: la versione nuova eredita dal genitore e il genitore dal candidato iniziale (precedente ed ereditato_da "
       "con gli sha256); stesso nome, versione diversa",
       prev.get("manifest_sha256") == inherited.get("manifest_sha256") == parent.manifest_sha256
       and prev.get("percorso") == inherited.get("percorso") == rel(args.manifest)
       and prev.get("records_sha256") == pm["records"]["sha256"]
       and pinh.get("manifest_sha256") == args.origine_sha256 and pinh.get("percorso") == rel(args.origine)
       and meta.get("nome") == pm.get("nome") and meta.get("versione") != pm.get("versione"))
    chain = meta.get("catena_delle_versioni") or []
    ok("parentela documentata (catena_delle_versioni) uguale alla catena effettiva",
       [(c.get("percorso"), c.get("manifest_sha256")) for c in chain]
       == [(rel(args.origine), args.origine_sha256), (rel(args.manifest), parent.manifest_sha256), (rel(args.new), None)],
       [c.get("percorso") for c in chain])
    ok("decisioni: panel-decontam/2 con le regole testuali del codice (TEXT_RULES e il loro sha256)",
       dec.get("controllo") == CHECK and dec.get("versione") == SCHEMA
       and dec.get("regole_testuali") == {"regole": TEXT_RULES, "sha256": TEXT_RULES_SHA256}
       and mf.sha256_text(mf.canonical(TEXT_RULES)) == TEXT_RULES_SHA256)
    old_rule = dec.get("decisione_precedente") or {}
    ok("decisione precedente citata: panel-decontam/1, manifest del genitore e regola storica come registrata",
       old_rule.get("versione") == OLD_SCHEMA == pdec.get("versione")
       and old_rule.get("manifest_sha256") == parent.manifest_sha256 and old_rule.get("regola") == pdec.get("testo_di_rito"))
    scan_dir = Path(args.scan)
    scan_rec = sha256_file(scan_dir / "records.jsonl.gz")
    ok("scansione, indice e pannello: gli stessi del genitore, invariati su disco",
       dec.get("scansione") == pdec.get("scansione") and dec.get("indice") == pdec.get("indice")
       and dec.get("pannello") == pdec.get("pannello") and scan_rec == pdec["scansione"]["records"]["sha256"]
       and sha256_file(scan_dir / "summary.json") == pdec["scansione"]["summary"]["sha256"]
       and sha256_file(Path(args.index) / "index.json") == pdec["indice"]["sha256"] == args.index_sha256
       and sha256_file(Path(args.panel) / "elements.jsonl.gz") == pdec["pannello"]["elements"]["sha256"]
       and sha256_file(Path(args.panel) / "summary.json") == pdec["pannello"]["summary"]["sha256"])
    rdir, rref = Path(args.riesame), dec.get("riesame") or {}
    names = {"regole": "regole.json", "decisioni": "decisioni.jsonl.gz", "summary": "summary.json"}
    files_ok = all((rdir / n).is_file() and sha256_file(rdir / n) == (rref.get(k) or {}).get("sha256")
                   for k, n in names.items())
    ok("riesame: regole, decisioni e riepilogo con gli sha256 registrati nel manifest", files_ok)
    if not files_ok:
        return finish(checks, args, child)
    rules_doc = json.loads((rdir / "regole.json").read_text(encoding="utf-8"))
    ok("riesame: regole.json con le regole del codice, il loro sha256 e la regola precedente come registrata nel "
       "genitore", rules_doc.get("regole_testuali") == TEXT_RULES and rules_doc.get("sha256") == TEXT_RULES_SHA256
       and (rules_doc.get("regola_precedente") or {}).get("registrata_nel_genitore") == pdec.get("testo_di_rito"))
    old = {ident(r): r for rs in parent.by_shard.values() for r in rs.values()}
    recs = [r for rs in child.by_shard.values() for r in rs.values()]
    new = {ident(r): r for r in recs}
    same_keys = set(old) == set(new) and len(new) == len(recs)
    differ = [label_of(i) for i, r in new.items() if i in old and any(
        r.get(k) != old[i].get(k) for k in set(r) | set(old[i]) if k not in STATE_FIELDS)]
    differ += [label_of(i) for i, r in new.items() if i in old and [{k: v for k, v in t.items() if k != "stato"}
                                                                    for t in r["compiti"]]
               != [{k: v for k, v in t.items() if k != "stato"} for t in old[i]["compiti"]]]
    ok("confronto col genitore: stessi record, gruppi, split, posizioni e digest; cambiano solo stato, motivi, "
       "evidenze e stati dei compiti", same_keys and not differ, {"diversi": len(differ), "esempi": differ[:5]})
    if not same_keys:
        return finish(checks, args, child)
    rows = {ident(r): r for r in read_rows(scan_dir / "records.jsonl.gz")}
    ok("scansione: una riga per record, con shard, posizione e digest del manifest",
       set(rows) == set(new) and all(
           (rows[i]["shard"], rows[i]["indice"], rows[i]["json_sha256"], rows[i]["png_sha256"])
           == (r["shard"], r["indice"], r["json_sha256"], r["immagini"][0]["sha256"]) for i, r in new.items()
           if i in rows))
    if set(rows) != set(new):
        return finish(checks, args, child)
    drows = {(x["insieme"], x["record"]): x for x in read_rows(rdir / "decisioni.jsonl.gz")}
    descriptions = {i: {"descrizione_del_loader": x["descrizione_del_loader"]} for i, x in drows.items()
                    if isinstance(x.get("descrizione_del_loader"), str)}
    del drows
    coverage, bad = bind_rule_texts(rows, old, descriptions)
    ok("testo: una descrizione del riesame per ogni corrispondenza nella descrizione del loader, con lo sha256 del "
       "manifest (descrizione_sha256) e i 10-gram della corrispondenza", not bad, {"casi": len(bad), "esempi": bad[:3]})
    outcomes = {i: record_outcome(row, "mavis", coverage.get(i)) for i, row in rows.items()}
    decisions = decide(list(old.values()), outcomes)
    kept, one_new, wrong_ev, kept_reasons, bad_reason, unexplained = [], [], [], [], [], []
    for i, r in new.items():
        p, d, b = old[i], decisions[i], last_check(old[i])
        ev, pev = r["evidenze"] or [], p["evidenze"] or []
        if ev[:len(pev)] != pev:
            kept.append(label_of(i))
        added = ev[len(pev):]
        if len(added) != 1 or added[0].get("controllo") != CHECK:
            one_new.append(label_of(i))
            continue
        e = added[0]
        if (e.get("esito"), e.get("esito_gruppo"), e.get("cause_nel_gruppo")) != (d["esito"], d["esito_gruppo"],
                                                                                 d["cause"]) \
                or e.get("versione") != SCHEMA or e.get("regole_testuali_sha256") != TEXT_RULES_SHA256 \
                or (e.get("esito_precedente"), e.get("esito_gruppo_precedente"), e.get("versione_precedente")) \
                != (b.get("esito"), b.get("esito_gruppo"), b.get("versione")) \
                or (e.get("rif") or {}).get("sha256") != scan_rec \
                or (e.get("riesame") or {}).get("sha256") != rref["decisioni"]["sha256"] \
                or e.get("corrispondenze") != rows[i]["corrispondenze"] \
                or e.get("corrispondenze_nella_scansione") != len(rows[i]["dettagli"]) \
                or e.get("testo_coperto") != sum(x["tipo"] == "testo" and not informative(x, coverage.get(i))
                                                 for x in rows[i]["dettagli"]):
            wrong_ev.append(label_of(i))
        mo, pmo = r["motivo"] or [], p["motivo"] or []
        if mo[:len(pmo)] != pmo:
            kept_reasons.append(label_of(i))
        extra = mo[len(pmo):]
        reason = transition_reason(b.get("esito_gruppo"), d["esito_gruppo"])
        if extra != ([reason] if reason else []):
            bad_reason.append(label_of(i))
        if r["stato"] != p["stato"] and not extra:
            unexplained.append(label_of(i))
    ok(f"{CHECK}: evidenze del genitore conservate (compresa la decisione panel-decontam/1)", not kept, len(kept))
    ok(f"{CHECK}: una e una sola evidenza nuova per record", not one_new, len(one_new))
    ok(f"{CHECK}: evidenza nuova uguale alla decisione ricalcolata con le regole esplicite, con la decisione precedente "
       "accanto e i riferimenti a scansione e riesame", not wrong_ev, {"casi": len(wrong_ev), "esempi": wrong_ev[:3]})
    ok(f"{CHECK}: motivi del genitore conservati", not kept_reasons, len(kept_reasons))
    ok(f"{CHECK}: un motivo nuovo solo dove l'esito del gruppo cambia, quello della transizione", not bad_reason,
       {"casi": len(bad_reason), "esempi": bad_reason[:3]})
    ok(f"{CHECK}: ogni cambio di stato ha il suo motivo nuovo", not unexplained, len(unexplained))
    wrong_state = [label_of(i) for i, r in new.items() if r["stato"] != decisions[i]["stato"]]
    better = [label_of(i) for i, r in new.items() if SEVERITY[r["stato"]] < SEVERITY[old[i]["stato"]]]
    ok(f"{CHECK}: stato = il più grave fra quello del genitore e quello dell'esito del gruppo ricalcolato con le regole "
       "esplicite", not wrong_state, {"casi": len(wrong_state), "esempi": wrong_state[:3]})
    ok("nessuna riammissione: nessun record passa a uno stato meno grave del genitore", not better, len(better))
    group_outcome = defaultdict(set)
    for r in recs:
        group_outcome[r["gruppo"]].add(last_check(r).get("esito_gruppo"))
    ok("un solo esito per gruppo", all(len(v) == 1 for v in group_outcome.values()))
    bad_tasks = [label_of(i) for i, r in new.items() if r["compiti"] != task_states(old[i]["compiti"], r["stato"])
                 or any(SEVERITY[t["stato"]] < SEVERITY[pt["stato"]] for t, pt in zip(r["compiti"], old[i]["compiti"]))
                 or any(t["compito"] in mm.QUARANTINED_TASKS and t["stato"] == "ammesso" for t in r["compiti"])]
    ok("compiti: seguono lo stato del record, shape_type mai ammessa, nessun compito meno grave del genitore",
       not bad_tasks, bad_tasks[:5])
    ok("conteggi uguali a manifest.json (ricalcolati dai record)", summarize("mavis", recs) == meta.get("conteggi"))
    keys_ok = []
    for name, ks in derived_keys("mavis", recs).items():
        entry = (meta.get("derivati") or {}).get(f"keys/{name}") or {}
        path = Path(args.new) / "keys" / name
        text = path.read_text(encoding="utf-8") if path.is_file() else None
        keys_ok.append(text == "".join(k + "\n" for k in sorted(ks)) and entry.get("righe") == len(ks)
                       and mf.sha256_text(text or "") == entry.get("sha256"))
    ok("file derivati (keys/) uguali alle chiavi ricalcolate e ai loro sha256", all(keys_ok) and len(keys_ok) == len(
        meta.get("derivati") or {}), keys_ok)
    summary = json.loads((rdir / "summary.json").read_text(encoding="utf-8"))
    recomputed = sorted((i[0], i[1], last_check(old[i]).get("esito"), decisions[i]["esito"],
                         last_check(old[i]).get("esito_gruppo"), decisions[i]["esito_gruppo"], old[i]["stato"],
                         new[i]["stato"]) for i in new
                        if (last_check(old[i]).get("esito"), last_check(old[i]).get("esito_gruppo"), old[i]["stato"])
                        != (decisions[i]["esito"], decisions[i]["esito_gruppo"], new[i]["stato"]))
    listed = sorted((c["insieme"], c["record"], *c["esito"], *c["esito_gruppo"], *c["stato"])
                    for c in summary.get("cambi") or [])
    ok("confronto col genitore: l'elenco dei cambi del riesame è completo e uguale a quello ricalcolato (esiti propri, "
       "esiti dei gruppi, stati)", recomputed == listed, {"ricalcolati": len(recomputed), "elencati": len(listed)})
    return finish(checks, args, child)


def cmd_verifica_riesame(args) -> int:
    if args.report.exists() and any(args.report.iterdir()):
        raise Failed(f"{args.report} non è vuota")
    t0 = time.time()
    result = verify_riesame(args)
    result.update({"job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
                   "secondi": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()})
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "verifica.json", result)
    print(f"{result['superati']} controlli superati su {result['totale']}")
    return 0 if result["esito"] else 1


# ---------------------------------------------------------------- argomenti

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "verifica", "manifest", "verifica-manifest"):
        p = sub.add_parser(name)
        p.add_argument("source", choices=sorted(SOURCES))
        p.add_argument("--manifest", required=True, type=Path, help="manifest da decontaminare (il genitore)")
        p.add_argument("--manifest-sha256", required=True, help="sha256 atteso del suo manifest.json")
        p.add_argument("--index", type=Path, help="cartella dell'indice del pannello (index.json, verifica.json)")
        p.add_argument("--index-sha256", help="sha256 atteso di index.json")
        if name == "scan":
            p.add_argument("--out", required=True, type=Path)
            p.add_argument("--limit", type=int, default=0, help="solo i primi N campioni di ogni shard (prova)")
            p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            p.add_argument("--ocrbench", type=Path, default=OCRBENCH, help="parquet di OCRBench (formule HME100K)")
            p.add_argument("--mathverse", type=Path, nargs="+", default=list(MATHVERSE),
                           help="parquet di MathVerse (provenienze dichiarate)")
            continue
        p.add_argument("--scan", required=True, type=Path, help="uscita di `scan`")
        p.add_argument("--report", required=True, type=Path, help="cartella nuova dei rapporti")
        p.add_argument("--panel", type=Path, help="cartella dei gruppi del pannello (elements.jsonl.gz, summary.json)")
        p.add_argument("--overlap", type=Path, help="unimer: outputs/unimer_overlap/run2")
        p.add_argument("--historical", type=Path, nargs="*", default=[],
                       help="mavis: cartelle delle scansioni storiche (p6, f1eval), solo per il confronto")
        if name == "verifica":
            p.add_argument("--partial", action="store_true", help="scansione limitata (prova)")
        if name in ("manifest", "verifica-manifest"):
            p.add_argument("--new", required=True, type=Path, help="cartella della versione nuova")
            p.add_argument("--check", type=Path, help="unimer: outputs/unimer_check/run4")
            p.add_argument("--evidence", type=Path, help="mavis: cartella outputs con le evidenze storiche")
        if name == "manifest":
            p.add_argument("--version", required=True)
            p.add_argument("--summary", required=True, type=Path, help="rapporto sintetico per git")
    for name in ("riesame", "verifica-riesame"):
        p = sub.add_parser(name)
        p.add_argument("source", choices=["mavis"])
        p.add_argument("--origine", required=True, type=Path, help="candidato iniziale (assegna gruppi e split)")
        p.add_argument("--origine-sha256", required=True, help="sha256 atteso del suo manifest.json")
        p.add_argument("--manifest", required=True, type=Path, help="versione da riesaminare, decisa con panel-decontam/1")
        p.add_argument("--manifest-sha256", required=True, help="sha256 atteso del suo manifest.json")
        p.add_argument("--index", required=True, type=Path)
        p.add_argument("--index-sha256", required=True)
        p.add_argument("--panel", required=True, type=Path)
        p.add_argument("--scan", required=True, type=Path, help="la scansione da cui il genitore è stato deciso")
        p.add_argument("--evidence", required=True, type=Path, help="cartella outputs con le evidenze del candidato")
        p.add_argument("--new", required=True, type=Path, help="cartella della versione nuova")
        p.add_argument("--report", required=True, type=Path, help="cartella nuova della verifica")
        if name == "riesame":
            p.add_argument("--out", required=True, type=Path, help="cartella nuova del riesame (regole, decisioni, "
                                                                   "riepilogo)")
            p.add_argument("--version", required=True)
            p.add_argument("--summary", required=True, type=Path, help="riepilogo per git")
        else:
            p.add_argument("--riesame", required=True, type=Path, help="cartella del riesame")
    return ap


def main() -> None:
    args = parser().parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("panel_decontam.py va eseguito in un job SLURM")
    need = {"scan": ("index", "index_sha256"), "verifica": ("index", "index_sha256", "panel"),
            "manifest": ("index", "index_sha256", "panel")}.get(args.cmd, ())
    if args.cmd != "scan" and args.source == "unimer" and args.cmd != "verifica-manifest":
        need += ("overlap",)
    if args.cmd in ("manifest", "verifica-manifest"):
        need += ("check", "overlap") if args.source == "unimer" else ("evidence",)
    missing = [n for n in need if getattr(args, n, None) in (None, "")]
    if missing:
        sys.exit(f"{args.cmd} {args.source}: argomenti mancanti {missing}")
    try:
        if args.cmd == "scan":
            cmd_scan(args)
            code = 0
        elif args.cmd == "verifica":
            code = cmd_verifica(args)
        elif args.cmd == "manifest":
            code = cmd_manifest(args)
        elif args.cmd == "riesame":
            code = cmd_riesame(args)
        elif args.cmd == "verifica-riesame":
            code = cmd_verifica_riesame(args)
        else:
            code = cmd_verifica_manifest(args)
    except (Failed, mr.Problem) as exc:
        sys.exit(str(exc))
    sys.exit(code)


if __name__ == "__main__":
    main()
