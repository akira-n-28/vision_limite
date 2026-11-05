#!/usr/bin/env python3
"""Preparazione della verifica delle golden answer dei sintetici v4: inventario dei controlli disponibili e campione
riproducibile per la revisione visiva (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §§1, 2.8, 2.10-2.11; S4). Solo in un
job SLURM CPU. Manifest, shard, scansione e uscite dei controlli si leggono soltanto: nessun dato o spiegazione
rigenerati, nessuna revisione avviata, nessun agente.

Quattro proprietà, tenute separate in tutti i file:
1. correttezza matematica della risposta: il target è vero per la scena descritta dalla specifica;
2. coerenza fra annotazioni e immagine renderizzata: il PNG mostra ciò che dicono specifica e layout;
3. rispondibilità con la sola immagine e la domanda: l'informazione serve visibile, leggibile e univoca;
4. validità delle spiegazioni: i record v4 non ne hanno (risposte brevi), quindi non applicabile.

**Inventario** (inventario.tsv, e nel report): per famiglia e tipo di domanda, le QA del manifest, quelle delle scene
ammesse, gli esiti del checker (candidato, job 30448-30449) e i controlli deterministici che coprono ciascuna proprietà,
con il loro grado di indipendenza. Generatore e checker sono dello stesso progetto, e il checker legge il layout, non i
pixel: una risposta che coincide con quella del checker non diventa `gold_verificato` (S2 §1). Il campo gold resta
vuoto.

**Campione casuale** (campione.tsv), per strato come S2 §2.8, sugli stati del manifest letto:
- coord, function, polygon, shapes: le QA che il loader costruisce (f1c.pick_questions) per le scene ammesse di train
  e dev;
- test: tutte le QA delle scene di test ammesse;
- coppie: le coppie con entrambe le varianti ammesse; l'unità è la coppia.
In ogni strato le n unità col digest più piccolo di {"audit", "seme", "record_id", "posizione", "qa"} (per le coppie
{"audit", "seme", "coppia"}): estrazione senza reinserimento con probabilità uguali n/N. Una numerosità più piccola,
fissata prima di guardare, è il prefisso nell'ordine del rango. Il report riporta N, n, la probabilità d'inclusione e,
solo come informazione, il limite superiore esatto al 95% (ipergeometrico) con zero errori osservati.

**Casi mirati** (mirati.tsv), separati: non stimano un tasso. Tutte le scene in quarantena; scene ammesse vicino alle
soglie del checker (testi più bassi, riquadri più vicini alla sovrapposizione, intersezioni più vicine al bordo del
tratto, grafici più vicini senza incrociarsi, vertici estremi col margine più piccolo, coordinate lette dalle tacche
vicine); coppie per famiglia; esclusioni per contaminazione accanto al record di valutazione dello stesso gruppo; QA
delle regole nuove di v4 (intersections, shape_type e label_type con le opzioni).

**Prompt del primo passaggio** (PROTOCOL). Il revisore vede l'immagine e il prompt, senza target e senza opzioni
aggiunte:
- train e dev: il testo dell'utente che costruisce il loader di training (data.synth_qa_sample: da 1 a 5 domande
  numerate, f1c.pick_questions e f1c.qa_messages), ricostruito anche dalle posizioni del manifest e confrontato; la QA
  campionata è la domanda numero_nel_prompt. Le opzioni compaiono solo se fanno parte della domanda (shape_type,
  label_type). Una QA di una scena di train o dev che il loader non sceglie (solo fra i casi mirati delle soglie) non è
  mai mostrata al modello: si vede come domanda singola, ed è segnalata;
- test e coppie: ogni QA da sola, col prefisso di f1c.qa_messages per una domanda. È il protocollo del foglio: quello
  di valutazione di F0 non è fissato. Il loader di training non costruisce il prompt delle coppie (le loro QA sono solo
  nella coppia), e il job lo verifica.
Il testo esatto, con il suo sha256 e le domande che contiene, sta in prompts.jsonl; nelle celle TSV gli a capo sono la
sequenza \\n. Le opzioni sono in opzioni.tsv e i target in soluzioni.tsv, da consultare dopo il primo giudizio.

**Coppie:** soluzioni, opzioni ed esiti hanno una riga per variante (a e b, dalla variante dichiarata nel JSON e
uguale al suffisso della chiave), anche nei casi mirati; il modulo ha in più una riga `coppia` per il contratto.

**Candidati della decontaminazione esterna** (candidati_decontaminazione.tsv, una riga per record, e
corrispondenze_decontaminazione.tsv, una riga per corrispondenza). L'evidenza del manifest ha solo un'anteprima di al
più MAX_EVIDENCE elementi: il foglio segue il suo riferimento alla scansione completa (records.jsonl.gz, con lo
sha256 dell'evidenza e del manifest; summary.json e matches.jsonl con quelli del manifest) e ne prende tutte le
corrispondenze. Per ogni record verifica chiave, shard, posizione e sha256 del PNG, conteggi e anteprima uguali a quelli
dell'evidenza; per ogni corrispondenza la voce dell'indice (elemento, posizione dell'immagine e id) e l'elemento del
pannello (elements.jsonl.gz con lo sha256 del manifest). Riconcilia i conteggi con summary.json e matches.jsonl e
distingue corrispondenze, coppie record-voce, voci, elementi e immagini distinte. Un dettaglio mancante nella
scansione o un'immagine del pannello non ritrovata rende il record `incompleto`; un identificativo incoerente ferma il
job prima di scrivere. Ogni immagine del pannello si copia una volta in immagini/pannello/<sha256>.<estensione>.

**Fogli:** ogni unità ha le immagini copiate byte per byte dagli shard (immagini/, sha256 controllato). esiti.tsv è il
modulo vuoto, con una colonna per proprietà. Con --previous il job confronta selezione, ranghi, target, opzioni e
immagini con un campione precedente (confronto.json) ed esce con errore se la selezione è cambiata. Soluzioni e opzioni
si confrontano per (unita, variante) nel formato dichiarato dal report precedente (compare_previous): dal formato 1 di
campione-1 solo con le trasformazioni previste, e le varianti b che quel formato non aveva restano non certificate.

    srun $RUN_PYTHON -u -B scripts/synth_gold_sample.py --manifest DIR --out DIR [--index DIR] [--review DIR] \\
        [--previous DIR] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from limite_vl import manifest as mf  # noqa: E402
from limite_vl import synth_check as sc  # noqa: E402
from limite_vl.data import SAMPLE_LOADERS, iter_shard  # noqa: E402
from limite_vl.f1c import pick_questions, qa_messages  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, sha256_file, write_new_json  # noqa: E402
from synth_decontam import CHECK, MAX_EVIDENCE  # noqa: E402

SCHEMA = "synth-gold-sample/2"
AUDIT = "synth-v4-1"  # etichetta dell'estrazione (S2 §2.8)
FAMILIES = ("coord", "function", "polygon", "shapes")
SIZES = {"coord": 75, "function": 75, "polygon": 75, "shapes": 75, "test": 50, "coppie": 30}  # proposta S2 §2.8
EXTREME_TYPES = {"highest": (1, 1), "lowest": (1, -1), "leftmost": (0, -1), "rightmost": (0, 1)}
NEW_RULES = ("intersections", "shape_type", "label_type")
COORDS = re.compile(r"^\((-?\d+), (-?\d+)\)$")
REVIEW_COLUMNS = ("unita", "variante", "record", "risposta_revisore", "corretta", "coerente_con_annotazioni",
                  "rispondibile_da_immagine", "spiegazione", "contratto_coppia", "gravita", "tipo_difetto", "tempo_s",
                  "note")
DECONTAM_REVIEW_COLUMNS = ("unita", "record", "stesso_contenuto", "corrispondenze_guardate", "corrispondenza_decisiva",
                           "note")
PROMPT_ORIGINS = {
    "loader": "train o dev: prompt di data.synth_qa_sample per la scena; la QA campionata è la domanda "
              "numero_nel_prompt",
    "loader_scena": "caso sulla scena intera di train o dev: le domande che il loader costruisce per la scena (il modello "
                    "le vede solo se la scena è ammessa)",
    "domanda_singola_qa_non_scelta": "QA di una scena di train o dev che il loader non sceglie: il modello non la vede "
                                     "mai in training; qui come domanda singola di qa_messages",
    "domanda_singola_test": "test: la QA da sola, col prefisso di qa_messages per una domanda",
    "domanda_singola_coppia": "coppie: la domanda della coppia da sola, uguale per le due varianti, col prefisso di "
                              "qa_messages per una domanda",
    "nessuno": "caso sulla scena intera di un record che il loader di training non legge",
}
PROTOCOL = {
    "primo_passaggio": "immagine e prompt (colonna prompt, testo esatto in prompts.jsonl), senza target e senza opzioni "
                       "aggiunte; si risponde alla QA campionata (numero_nel_prompt)",
    "train_dev": "il prompt del loader di training (data.synth_qa_sample), con da 1 a 5 domande numerate; le opzioni si "
                 "vedono solo se sono nel testo della domanda (shape_type, label_type). Ogni prompt esportato è uguale "
                 "a qa_messages(pick_questions) ricostruito dalle posizioni del manifest",
    "test": "ogni QA da sola, qa_messages([domanda]): protocollo del foglio, perché quello di valutazione di F0 non è "
            "fissato (contratto D3). f1c_eval.py, storico, usava lo stesso prefisso e dava un punteggio alle opzioni per "
            "verosimiglianza, senza mostrarle",
    "coppie": "la domanda della coppia da sola per ciascuna variante, come nel test; il loader di training non "
              "costruisce questo prompt, perché le QA delle coppie stanno solo nel campo pair",
    "dopo_il_primo_giudizio": "opzioni.tsv (opzioni ausiliarie, con opzioni_nel_testo_della_domanda) e soluzioni.tsv "
                              "(target, esito del checker e stato della scena, per variante)",
}
SAME_SAMPLE = ("unita", "strato", "rango", "N", "n", "p_inclusione", "record", "record_id", "insieme", "split",
               "famiglia", "shard", "indice", "posizione_qa", "ordine_nel_loader", "tipo", "domanda", "immagini_sha256",
               "qa_sha256")
SAME_TARGETED = ("unita", "categoria", "motivo", "misura", "record", "record_id", "insieme", "split", "stato",
                 "famiglia", "posizione_qa", "tipo", "domanda")
SAME_DECONTAM = ("unita", "rango", "record", "record_id", "insieme", "split", "stato", "famiglia", "esito",
                 "esito_gruppo", "corrispondenze")

# Controlli deterministici per proprietà e tipo di domanda (S2 §§2.3-2.6). R: per record, I: invariante del renderer
# sul layout, C: solo un campione; "parziale" segnala che il controllo non è indipendente dal generatore.
COMMON = {
    "coerenza": "R parziale: il layout porta lo sha256 del PNG dello stesso render (digest_corrispondente su 60.900 "
                "record) e la scena si riottiene dalla chiave con byte identici; i pixel non sono letti, quindi che il "
                "PNG mostri ciò che dice il layout dipende dal renderer (C: revisione visiva)",
    "spiegazione": "non applicabile: i record v4 hanno risposte brevi, senza spiegazioni",
}
CHECKS = {
    "coord": {
        "quadrant": ("segni ricalcolati dai centri dei marcatori disegnati", "I: distanza dagli assi >= SIDE 0.5; testi "
                     "non sovrapposti e alti >= 10 px"),
        "side_x": ("segno ricalcolato dal centro del marcatore", "I: distanza dall'asse >= 0.5; testi leggibili"),
        "side_y": ("segno ricalcolato dal centro del marcatore", "I: distanza dall'asse >= 0.5; testi leggibili"),
        "coords": ("valore letto dalle etichette delle tacche o dalle tacche vicine con scala univoca (axis_value), "
                   "confrontato con la posizione del marcatore", "I: etichetta o tacche vicine leggibili; testi non "
                   "sovrapposti"),
        "pair_lr": ("confronto dei centri disegnati", "I: scarto >= 0.5"),
        "pair_ud": ("confronto dei centri disegnati", "I: scarto >= 0.5"),
        "count": ("etichette dei punti nel layout", "I: testi non sovrapposti, alti >= 10 px"),
        "exists": ("testi delle etichette nel layout", "I: testi non sovrapposti, alti >= 10 px"),
        "line_on": ("distanza fra marcatore e retta disegnata (sotto 0.05 sì, oltre 0.5 no)", "I: casi fra 0.05 e 0.5 "
                    "non ammessi"),
    },
    "polygon": {
        "side_len": ("misura scritta più vicina al lato; coerente con la lunghezza disegnata entro il 2%",
                     "I: una sola misura vicina al lato; testi non sovrapposti"),
        "angle_val": ("arco al vertice nominato; valore scritto coerente con l'angolo disegnato entro 0.6 gradi",
                      "I: arco e valore presenti, non sovrapposti"),
        "right_vertex": ("segno dell'angolo retto su un angolo disegnato di 90 +- 0.5 gradi", "I: segno presente"),
        "highest": ("ordine dei vertici disegnati", "I: margine >= EXTREME 0.15 del lato del riquadro"),
        "lowest": ("ordine dei vertici disegnati", "I: margine >= 0.15"),
        "leftmost": ("ordine dei vertici disegnati", "I: margine >= 0.15"),
        "rightmost": ("ordine dei vertici disegnati", "I: margine >= 0.15"),
        "m_side": ("il punto sta su un solo lato disegnato (entro 0.05)", "I: etichetta del punto non sovrapposta"),
        "shape_type": ("classi da predicati sui vertici disegnati (triangle_classes, quad_classes); esattamente "
                       "un'opzione vera", "I: opzioni nel testo della domanda"),
        "count": ("numero delle misure scritte nel layout", "I: testi non sovrapposti"),
        "exists": ("etichette dei vertici e del punto nel layout", "I: testi non sovrapposti"),
    },
    "function": {
        "opens": ("dalle formule della specifica e dal tratto disegnato (parziale: formule dalla specifica)",
                  "I: vertice nel tratto disegnato"),
        "vertex_ud": ("formule della specifica, vertice nel tratto disegnato (parziale)", "I: |k| >= 0.5"),
        "vertex_lr": ("formule della specifica, vertice nel tratto disegnato (parziale)", "I: |h| >= 0.5"),
        "slope": ("valori agli estremi del tratto disegnato più lungo (parziale)", "I: tratto disegnato"),
        "intercept": ("valore in x = 0 se il grafico incontra l'asse y nel tratto disegnato (parziale)",
                      "I: |f(0)| >= 0.5"),
        "intersections": ("conteggio geometrico delle intersezioni nei tratti disegnati di entrambi i grafici "
                          "(visible_intersections; parziale: formule dalla specifica)",
                          "I: nessuna intersezione a meno di EDGE 0.4 dal bordo del tratto, niente tangenze o radici "
                          "vicine, grafici non più vicini di 0.25 senza incrociarsi"),
        "exists": ("nomi dei grafici nel layout", "I: testi non sovrapposti"),
    },
    "shapes": {
        "count": ("figure riconosciute dalle patch del layout", "I: figure non sovrapposte e dentro gli assi "
                  "(violazioni bloccanti)"),
        "has_type": ("tipi dalle patch (quadrato dalle misure; un quadrato è anche un rettangolo)", "I: come count"),
        "count_type": ("conteggio per tipo dalle patch; 'rettangoli' con un quadrato presente non ammesso",
                       "I: come count"),
        "label_type": ("figura che contiene l'etichetta, classe più specifica; esattamente un'opzione vera",
                       "I: etichetta dentro una sola figura"),
        "exists": ("etichette delle figure nel layout", "I: testi non sovrapposti"),
    },
}
PAIR_CHECK = ("come il tipo di domanda, su entrambe le immagini; contratto della coppia (synth_candidate): stessa "
              "domanda e opzioni, risposte diverse, un solo dettaglio cambiato", "I: come il tipo di domanda")
LIMITS = [
    "il checker legge il layout, non i pixel: non vede un grafico che attraversa un'etichetta di tacca né vicinanze "
    "ambigue senza sovrapposizione dei riquadri (S2 §2.6)",
    "indipendenza parziale: per function le formule vengono dalla specifica; predicati, contratto delle coppie e "
    "definizione di stato sono scritti dallo stesso progetto",
    "la soglia di 10 px per l'altezza dei testi è una proposta, misurata sul layout alla risoluzione del PNG",
    "la riproducibilità dalla chiave usa lo stesso codice del generatore: prova l'integrità, non la correttezza",
    "nessun record è gold_verificato o target_controllato: S2 è una bozza, i tipi non sono promossi e manca la stima "
    "campionaria (S4)",
    "il prompt di test e coppie è quello del foglio (domanda singola): il protocollo di valutazione di F0 non è fissato",
]


def rank(obj: dict) -> str:
    return mf.sha256_text(mf.canonical(obj))


def upper_zero(N: int, n: int, alpha: float = 0.05) -> float | None:
    """Limite superiore esatto (ipergeometrico, unilaterale) della quota di errori con 0 errori in n estratti da N:
    il K più grande con P(X = 0 | K) >= alpha, diviso per N."""
    if n <= 0 or N <= 0:
        return None

    def p_zero(K: int) -> float:
        if K > N - n:
            return 0.0
        return math.exp(sum(math.log((N - K - i) / (N - i)) for i in range(n)))

    lo, hi = 0, N
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if p_zero(mid) >= alpha:
            lo = mid
        else:
            hi = mid - 1
    return lo / N


# ---------------------------------------------------------------- margini dal layout salvato (nessun ridisegno)

def text_margins(layout: dict) -> tuple[float | None, float | None]:
    """(altezza del testo più basso in px, avvicinamento massimo fra riquadri: il minore dei due sovrapposizioni in x e
    in y; sopra OVERLAP_PX in entrambe è una violazione, sotto zero c'è spazio fra i riquadri)."""
    texts = [it for it in layout["items"] if it["kind"] == "text"]
    markers = [it for it in layout["items"] if it["kind"] == "marker"]
    low = min((it["bbox"][3] - it["bbox"][1] for it in texts), default=None)
    close = None
    for i, a in enumerate(texts):
        own = a["gid"].split(":", 1)[-1]
        for b in texts[i + 1:] + [mk for mk in markers if mk["gid"].split(":", 1)[-1] != own]:
            v = min(min(a["bbox"][2], b["bbox"][2]) - max(a["bbox"][0], b["bbox"][0]),
                    min(a["bbox"][3], b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1]))
            close = v if close is None else max(close, v)
    return low, close


def intersection_margins(meta: dict) -> tuple[float | None, float | None]:
    """(distanza minima delle intersezioni dal bordo del tratto disegnato, come in synth_check.visible_intersections;
    distanza minima fra i grafici dove sono disegnati entrambi, se non si incrociano)."""
    curves = list(meta["spec"].get("curves") or [])
    if len(curves) != 2:
        return None, None
    items = {it["gid"].split(":", 1)[1]: it for it in meta["layout"]["items"] if it["gid"].startswith("curve:")}
    f, g = curves
    fi, gi = items.get(f["name"]), items.get(g["name"])
    clips = [v for v in ((fi or {}).get("clip_y"), (gi or {}).get("clip_y")) if v is not None]
    if fi is None or gi is None or not clips:
        return None, None
    clip = min(clips)
    a, b, c = sc._difference(f, g)
    if abs(a) < 1e-12:
        xs = [] if abs(b) < 1e-12 else [-c / b]
    else:
        d = b * b - 4 * a * c
        xs = [] if d < 0 else sorted({(-b - math.sqrt(d)) / (2 * a), (-b + math.sqrt(d)) / (2 * a)})
    lo, hi = max(fi["x_range"][0], gi["x_range"][0]), min(fi["x_range"][1], gi["x_range"][1])
    if xs:
        holes = [e for e, cause in sc._run_ends(f, fi) + sc._run_ends(g, gi) if cause == "buco"]
        return min(min([abs(x - lo), abs(x - hi), abs(abs(sc._curve_value(f, x)) - clip)] + [abs(x - e) for e in holes])
                   for x in xs), None
    grid = [lo + (hi - lo) * i / 2000 for i in range(2001)]
    near = [abs(sc._curve_value(f, x) - sc._curve_value(g, x)) for x in grid
            if sc._visible_at(fi, x) and sc._visible_at(gi, x)]
    return None, (min(near) if near else None)


def extreme_margins(layout: dict) -> dict[str, float]:
    """Per ogni tipo di vertice estremo, lo stacco fra il primo e il secondo vertice in frazione del lato del riquadro
    (il checker chiede almeno EXTREME)."""
    path, names = sc._polygon_vertices(layout)
    if not path or names is None:
        return {}
    size = max(max(p[0] for p in path) - min(p[0] for p in path), max(p[1] for p in path) - min(p[1] for p in path))
    out = {}
    for t, (k, sign) in EXTREME_TYPES.items():
        order = sorted(range(len(path)), key=lambda i: sign * path[i][k], reverse=True)
        out[t] = sign * (path[order[0]][k] - path[order[1]][k]) / size if size else 0.0
    return out


def coords_from_ticks(layout: dict, answer: str) -> bool:
    """La coordinata si legge contando le tacche: manca l'etichetta della tacca di x o di y."""
    m = COORDS.match(answer)
    if not m:
        return False
    labels = {it["gid"] for it in layout["items"] if it["kind"] == "text" and it["gid"].startswith("tick_label:")}
    return f"tick_label:x:{int(m.group(1))}" not in labels or f"tick_label:y:{int(m.group(2))}" not in labels


class Best:
    """I k elementi con la chiave più piccola, a parità di chiave in ordine di rango."""

    def __init__(self, k: int):
        self.k, self.heap = k, []

    def add(self, value: float, tie: str, item) -> None:
        entry = (-value, _Rev(tie), item)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
        elif entry > self.heap[0]:
            heapq.heapreplace(self.heap, entry)

    def items(self) -> list:
        return [it for _, _, it in sorted(self.heap, key=lambda e: (-e[0], e[1].s))]


class _Rev:
    """Ordine inverso delle stringhe, per tenere il rango più piccolo a parità di valore nel mucchio."""

    def __init__(self, s: str):
        self.s = s

    def __lt__(self, other) -> bool:
        return self.s > other.s

    def __eq__(self, other) -> bool:
        return self.s == other.s


# ---------------------------------------------------------------- campione

def populations(records: list[dict], seed: int) -> dict[str, list[dict]]:
    """Unità di ogni strato con il loro rango (S2 §2.8)."""
    pops: dict[str, list[dict]] = defaultdict(list)
    pairs: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["stato"] != "ammesso":
            continue
        if r["insieme"] == "coppie":
            pairs[r["coppia"]].append(r)
            continue
        if r["insieme"] == "test":
            positions, stratum = list(range(len(r["qa"]))), "test"
        elif r["split"] in ("train", "dev"):
            positions, stratum = pick_questions(r["record"], list(range(len(r["qa"])))), r["famiglia"]
        else:
            continue
        for order, pos in enumerate(positions):
            q = r["qa"][pos]
            pops[stratum].append({"rango": rank({"audit": AUDIT, "seme": seed, "record_id": r["record_id"],
                                                 "posizione": pos, "qa": q["sha256"]}),
                                  "records": [r], "posizione": pos, "ordine_nel_loader": order, "tipo": q["tipo"],
                                  "qa_sha256": q["sha256"]})
    for pid, rs in sorted(pairs.items()):
        if len(rs) == 2:
            rs = sorted(rs, key=lambda r: r["record"])
            pops["coppie"].append({"rango": rank({"audit": AUDIT, "seme": seed, "coppia": pid}), "records": rs,
                                   "posizione": 0, "ordine_nel_loader": 0, "tipo": rs[0]["qa"][0]["tipo"],
                                   "qa_sha256": "|".join(r["qa"][0]["sha256"] for r in rs), "coppia": pid})
    for units in pops.values():
        units.sort(key=lambda u: u["rango"])
    return dict(pops)


# ---------------------------------------------------------------- prompt, varianti e celle (pure, collaudate in tests/)

def cell(value) -> str:
    return str(value).replace("\t", " ").replace("\n", " ")


def tsv_text(text: str) -> str:
    """Testo in una cella TSV: gli a capo come la sequenza \\n (il testo esatto sta in prompts.jsonl)."""
    return text.replace("\n", "\\n")


def options_in_question(q: dict) -> bool:
    """Tutte le opzioni compaiono, come parole intere, nel testo della domanda (shape_type e label_type di v4; anche
    domande come "above or below")."""
    return all(re.search(rf"(?<!\w){re.escape(o)}(?!\w)", q["q"]) for o in q["options"])


def loader_prompt(key: str, files: dict[str, bytes]) -> str:
    """Testo dell'utente che il loader di training costruisce per una scena (data.synth_qa_sample), dopo l'immagine."""
    sample = SAMPLE_LOADERS["synth_qa"](key, files)
    user = sample.messages[0]
    if user["role"] != "user" or [c["type"] for c in user["content"]] != ["image", "text"]:
        raise ValueError(f"{key}: messaggio dell'utente inatteso dal loader")
    return user["content"][1]["text"]


def prompt_for(r: dict, meta: dict, files: dict[str, bytes] | None, pos) -> dict:
    """Prompt del primo passaggio per la QA in posizione pos del record (None: la scena intera), con il collegamento
    fra la QA e il prompt completo (PROMPT_ORIGINS)."""
    qa_sha = [q["sha256"] for q in r["qa"]]
    train_like = r["insieme"] == "train" and r["split"] in ("train", "dev")
    if train_like and meta["qa"]:
        positions = pick_questions(r["record"], list(range(len(meta["qa"]))))
        text = loader_prompt(r["record"], files)
        rebuilt = qa_messages([meta["qa"][p] for p in positions])[0]
        if text != rebuilt:
            raise ValueError(f"{r['record']}: prompt del loader diverso da quello ricostruito dalle posizioni")
        listed = [{"numero": i, "posizione": p, "tipo": meta["qa"][p]["type"], "qa_sha256": qa_sha[p]}
                  for i, p in enumerate(positions, 1)]
        delivered = r["stato"] == "ammesso"
        if pos is None:
            return {"prompt": text, "origine_prompt": "loader_scena", "numero_nel_prompt": None,
                    "domande_nel_prompt": listed, "consegnato_dal_loader": delivered}
        if pos in positions:
            return {"prompt": text, "origine_prompt": "loader", "numero_nel_prompt": positions.index(pos) + 1,
                    "domande_nel_prompt": listed, "consegnato_dal_loader": delivered}
        q = meta["qa"][pos]
        return {"prompt": qa_messages([q])[0], "origine_prompt": "domanda_singola_qa_non_scelta",
                "numero_nel_prompt": 1, "domande_nel_prompt": [{"numero": 1, "posizione": pos, "tipo": q["type"],
                                                                "qa_sha256": qa_sha[pos]}],
                "consegnato_dal_loader": False}
    if pos is None:
        return {"prompt": "", "origine_prompt": "nessuno", "numero_nel_prompt": None, "domande_nel_prompt": [],
                "consegnato_dal_loader": False}
    items = list(meta["qa"]) + ([meta["pair"]] if "pair" in meta else [])
    q = items[pos]
    return {"prompt": qa_messages([q])[0],
            "origine_prompt": "domanda_singola_coppia" if r["insieme"] == "coppie" else "domanda_singola_test",
            "numero_nel_prompt": 1, "domande_nel_prompt": [{"numero": 1, "posizione": pos, "tipo": q["type"],
                                                            "qa_sha256": qa_sha[pos]}],
            "consegnato_dal_loader": False}


def variant_label(r: dict, meta: dict) -> str:
    """Variante dichiarata nel JSON della coppia, uguale al suffisso della chiave; "" fuori dalle coppie."""
    if r["insieme"] != "coppie":
        return ""
    v = (meta.get("pair") or {}).get("variant")
    if v not in ("a", "b") or not r["record"].endswith(f"-{v}"):
        raise ValueError(f"{r['record']}: variante {v!r} incoerente con la chiave")
    return v


def unit_rows(uid: str, rs: list[dict], metas: dict, qs: list, pos, status_of) -> tuple[list, list, list]:
    """(soluzioni, opzioni, righe del modulo) di un'unità. Una coppia ha una riga per variante, con etichetta a o b e
    identificativi propri, e nel modulo una riga `coppia` in più per il contratto; le altre unità una riga."""
    pair = len(rs) == 2 and all(r["insieme"] == "coppie" for r in rs)
    labels = [variant_label(r, metas[r["record"]]) for r in rs] if pair else [""] * len(rs)
    if pair and labels != ["a", "b"]:
        raise ValueError(f"{uid}: varianti {labels}, attese a e b")
    sols, opts, form = [], [], []
    for r, q, v in zip(rs, qs, labels):
        if q is not None:
            sols.append({"unita": uid, "variante": v, "record": r["record"], "record_id": r["record_id"],
                         "posizione_qa": pos, "tipo": q["type"], "risposta": q["answer"],
                         "esito_checker": status_of(r, pos), "stato": r["stato"],
                         "stato_scena_sha256": r["stato_scena_sha256"]})
            opts.append({"unita": uid, "variante": v, "record": r["record"], "posizione_qa": pos, "tipo": q["type"],
                         "opzioni": " / ".join(q["options"]),
                         "opzioni_nel_testo_della_domanda": "sì" if options_in_question(q) else "no"})
        if pair:
            form.append({"unita": uid, "variante": v, "record": r["record"], "spiegazione": "non_applicabile",
                         "contratto_coppia": "non_applicabile"})
    if pair:
        form.append({"unita": uid, "variante": "coppia", "record": "|".join(r["record"] for r in rs),
                     "risposta_revisore": "non_applicabile", "spiegazione": "non_applicabile"})
    else:
        form.append({"unita": uid, "variante": "", "record": "|".join(r["record"] for r in rs),
                     "spiegazione": "non_applicabile", "contratto_coppia": "non_applicabile"})
    return sols, opts, form


# ---------------------------------------------------------------- corrispondenze complete della decontaminazione

def last_evidence(r: dict) -> dict | None:
    """L'evidenza più recente della decontaminazione esterna (una per derivazione)."""
    found = [e for e in r.get("evidenze") or [] if e.get("controllo") == CHECK]
    return found[-1] if found else None


def element_text(m: dict) -> str:
    return (f"{m['tipo']} {m['elemento']} pos {m['posizione_immagine']} d={m['distanza']} {m['ruolo']}"
            + ("" if m["metrica_primaria"] is None else f" metrica_primaria={m['metrica_primaria']}"))


def decontam_tables(units: list[dict], scan_rows: dict[str, dict], entries: dict, elements: dict,
                    panel_files: dict) -> tuple[list[dict], list[dict], list[str]]:
    """Righe per record e per corrispondenza dei candidati, con tutte le corrispondenze della riga della scansione.

    units: [{"unita", "rango", "records": [r], "evidenza": e}]; scan_rows: chiave -> riga di records.jsonl.gz;
    entries: (elemento, posizione) -> voce dell'indice; elements: chiave -> elemento del pannello; panel_files:
    (elemento, posizione) -> {"file", "sha256"} delle immagini ritrovate. Restituisce anche le incoerenze (chiave,
    shard, posizione, sha256 del PNG, conteggi o anteprima diversi dall'evidenza, voce con un altro id): con una sola il
    foglio non si scrive. Una riga o un dettaglio mancante, un elemento o una voce assenti, un'immagine non ritrovata
    rendono il record `incompleto`, senza fermare il job."""
    rec_rows, match_rows, fatal = [], [], []
    resolved: dict[str, str] = {}

    def where(path: str) -> str:
        if path not in resolved:
            resolved[path] = str(mf.shard_path(path))
        return resolved[path]

    for u in units:
        uid, r, e = u["unita"], u["records"][0], u["evidenza"]
        key, preview = r["record"], e.get("elementi") or []
        row, missing, details = scan_rows.get(key), [], []
        total = None
        if row is None:
            missing.append("riga della scansione assente")
        else:
            why = []
            if where(row.get("shard", "")) != where(r["shard"]):
                why.append("shard")
            if row.get("posizione") != r["indice"]:
                why.append("posizione")
            if row.get("byte_sha256") != r["immagini"][0]["sha256"]:
                why.append("sha256 del PNG")
            if (row.get("corrispondenze") or {}) != (e.get("corrispondenze") or {}):
                why.append("conteggi diversi dall'evidenza")
            details = row.get("dettagli") or []
            head = [{"tipo": d.get("match"), "elemento": d.get("elemento"), "posizione": d.get("pos"),
                     "distanza": d.get("distance")} for d in details[:MAX_EVIDENCE]]
            if [{k: x.get(k) for k in ("tipo", "elemento", "posizione", "distanza")} for x in preview] != head:
                why.append("anteprima dell'evidenza diversa dalla scansione")
            if why:
                fatal.append(f"{key}: {', '.join(why)}")
            total = sum((row.get("corrispondenze") or {}).values())
            if row.get("dettagli_completi") is not True or len(details) != total:
                missing.append(f"dettagli nella scansione {len(details)} su {total}")
        rows_here, not_found, absent = [], 0, 0
        for j, d in enumerate(details, 1):
            el, pos = d.get("elemento"), d.get("pos")
            entry, info, problems = entries.get((el, pos)), elements.get(el), []
            absent += info is None or entry is None
            if info is None:
                problems.append("elemento assente dal pannello")
            if entry is None:
                problems.append("voce dell'indice assente")
            elif entry.get("id") != d.get("bench_id"):
                fatal.append(f"{key}: corrispondenza {j} con la voce {entry.get('id')} invece di {d.get('bench_id')}")
            image = panel_files.get((el, pos))
            if image is None:
                problems.append("immagine del pannello non ritrovata")
                not_found += 1
            info = info or {}
            rows_here.append({
                "corrispondenza": f"{uid}-{j:03d}", "unita": uid, "record": key, "record_id": r["record_id"], "n": j,
                "di": len(details), "tipo": d.get("match"), "distanza": d.get("distance"),
                "benchmark": d.get("benchmark"), "elemento": el, "posizione_immagine": pos,
                "voce_indice": d.get("bench_id"), "ruolo": info.get("ruolo"), "gruppo_pannello": info.get("gruppo"),
                "metrica_primaria": info.get("metrica_primaria"),
                "pixel_sha256_pannello": (entry or {}).get("pixel_sha256"),
                "nell_anteprima_del_manifest": "sì" if j <= len(preview) else "no",
                "immagine_pannello": (image or {}).get("file", ""),
                "immagine_pannello_sha256": (image or {}).get("sha256", ""), "problemi": "; ".join(problems)})
        if not_found:
            missing.append(f"immagini del pannello non ritrovate {not_found} su {len(details)}")
        if absent:
            missing.append(f"voci o elementi del pannello assenti {absent} su {len(details)}")
        match_rows += rows_here
        roles = Counter("test_ammesso" if m["ruolo"] == "test" and m["metrica_primaria"] is True else
                        "test_non_ammesso" if m["ruolo"] == "test" else str(m["ruolo"]) for m in rows_here)
        files = list(dict.fromkeys(m["immagine_pannello"] for m in rows_here if m["immagine_pannello"]))
        rec_rows.append({
            "unita": uid, "rango": u["rango"], "record": key, "record_id": r["record_id"], "insieme": r["insieme"],
            "split": r["split"], "stato": r["stato"], "famiglia": r["famiglia"], "esito": e["esito"],
            "esito_gruppo": e["esito_gruppo"], "corrispondenze": json.dumps(e.get("corrispondenze") or {},
                                                                            sort_keys=True),
            "corrispondenze_nella_scansione": "" if total is None else total,
            "corrispondenze_esportate": len(rows_here), "nell_anteprima_del_manifest": len(preview),
            "voci_del_pannello": len({(m["elemento"], m["posizione_immagine"]) for m in rows_here}),
            "elementi_del_pannello": len({m["elemento"] for m in rows_here}),
            "immagini_del_pannello_distinte": len({m["pixel_sha256_pannello"] for m in rows_here
                                                   if m["pixel_sha256_pannello"]}),
            "ruoli": ", ".join(f"{k} {v}" for k, v in sorted(roles.items())),
            "materiale": "completo" if not missing else "incompleto: " + "; ".join(missing),
            "elementi": "; ".join(element_text(m) for m in rows_here), "immagini_pannello": "|".join(files)})
    return rec_rows, match_rows, fatal


def reconcile(scan_rows: dict[str, dict], summary: dict, matches_lines: list[dict], candidates: set[str],
              match_rows: list[dict]) -> tuple[list[dict], dict]:
    """Conteggi della scansione (righe, summary.json, matches.jsonl) contro quelli esportati: (controlli, conteggi)."""
    checks = []

    def check(name, ok, detail=None):
        checks.append({"controllo": name, "esito": bool(ok), "dettaglio": detail})

    def key6(k, d):
        return (k, d.get("match"), d.get("distance"), d.get("benchmark"), d.get("bench_id"), d.get("elemento"),
                d.get("pos"))

    found = {k for k, row in scan_rows.items() if sum((row.get("corrispondenze") or {}).values())}
    check("record con corrispondenze nella scansione = candidati del manifest", found == candidates,
          {"solo_scansione": len(found - candidates), "solo_manifest": len(candidates - found)})
    total = sum(sum((row.get("corrispondenze") or {}).values()) for row in scan_rows.values())
    per_bench = Counter(summary.get("matches_per_benchmark") or {})
    check("corrispondenze della scansione = summary.json", total == sum(per_bench.values()),
          {"righe": total, "summary": sum(per_bench.values())})
    details = Counter(key6(k, d) for k, row in scan_rows.items() for d in row.get("dettagli") or [])
    lines = Counter(key6(m["key"], m) for m in matches_lines)
    check("matches.jsonl = dettagli di records.jsonl.gz", lines == details,
          {"matches.jsonl": sum(lines.values()), "dettagli": sum(details.values())})
    exported = Counter((m["record"], m["tipo"], m["distanza"], m["benchmark"], m["voce_indice"], m["elemento"],
                        m["posizione_immagine"]) for m in match_rows)
    wanted = Counter({k: v for k, v in details.items() if k[0] in candidates})
    check("corrispondenze esportate = dettagli della scansione dei candidati, nessuna omessa", exported == wanted,
          {"esportate": len(match_rows), "attese": sum(wanted.values())})
    check("per benchmark e tipo: esportate = summary.json",
          Counter(f"{m['benchmark']}/{m['tipo']}" for m in match_rows) == per_bench)
    pairs = {(m["record"], m["elemento"], m["posizione_immagine"]) for m in match_rows}
    counts = {"record_candidati": len(candidates), "corrispondenze_scansione": total,
              "corrispondenze_esportate": len(match_rows),
              "per_tipo": dict(Counter(m["tipo"] for m in match_rows)),
              "coppie_record_voce_distinte": len(pairs),
              "duplicati_di_evidenza": len(match_rows) - len(pairs),
              "voci_del_pannello_distinte": len({(m["elemento"], m["posizione_immagine"]) for m in match_rows}),
              "elementi_del_pannello_distinti": len({m["elemento"] for m in match_rows}),
              "immagini_del_pannello_distinte_per_pixel": len({m["pixel_sha256_pannello"] for m in match_rows
                                                               if m["pixel_sha256_pannello"]}),
              "file_del_pannello_distinti": len({m["immagine_pannello_sha256"] for m in match_rows
                                                 if m["immagine_pannello_sha256"]}),
              "record_oltre_l_anteprima": len({m["record"] for m in match_rows
                                               if m["nell_anteprima_del_manifest"] == "no"}),
              "corrispondenze_oltre_l_anteprima": sum(m["nell_anteprima_del_manifest"] == "no" for m in match_rows)}
    return checks, counts


# ---------------------------------------------------------------- confronto con un campione precedente

FORMAT_V1 = "synth-gold-sample/1"  # campione-1 (job 30522): soluzioni una per unità, opzioni nel foglio dell'unità
SOLUTION_COLUMNS = {
    FORMAT_V1: ("unita", "risposta", "esito_checker", "stato_scena_sha256"),
    SCHEMA: ("unita", "variante", "record", "record_id", "posizione_qa", "tipo", "risposta", "esito_checker", "stato",
             "stato_scena_sha256"),
}
OPTION_COLUMNS = ("unita", "variante", "record", "posizione_qa", "tipo", "opzioni", "opzioni_nel_testo_della_domanda")
# colonne canoniche dei fogli col formato fisso: le scrive main (write_sheet) anche quando la tabella è vuota, e le
# controlla compare_previous (format_problems); gli altri fogli prendono le colonne dalla prima riga
SHEET_COLUMNS = {"soluzioni.tsv": SOLUTION_COLUMNS[SCHEMA], "opzioni.tsv": OPTION_COLUMNS, "esiti.tsv": REVIEW_COLUMNS,
                 "esiti_decontaminazione.tsv": DECONTAM_REVIEW_COLUMNS}
LEGACY_EMPTY = ["unita"]  # intestazione di una tabella vuota scritta senza colonne, prima del 2026-11-05
# campi confrontati per (unita, variante); nel formato 1 il record è quello dell'unità nella posizione della variante
SAME_SOLUTION = {FORMAT_V1: ("record", "risposta", "esito_checker", "stato_scena_sha256"),
                 SCHEMA: SOLUTION_COLUMNS[SCHEMA][2:]}
SAME_OPTION = {FORMAT_V1: ("record", "opzioni"), SCHEMA: OPTION_COLUMNS[2:]}


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def tsv_columns(path: Path) -> list[str]:
    """Intestazione di un TSV; vuota se il file non c'è."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh, delimiter="\t"), [])


def write_table(path: Path, table: list[dict], columns=None) -> dict:
    """TSV con le celle di cell(). Senza colonne date valgono quelle della prima riga, e una tabella vuota ha la sola
    intestazione "unita". Un campo di una riga fuori dalle colonne è un errore, non un dato perso in silenzio."""
    columns = list(columns or (list(table[0]) if table else LEGACY_EMPTY))
    extra = sorted({k for row in table for k in row} - set(columns))
    if extra:
        raise ValueError(f"{Path(path).name}: campi fuori dalle colonne {extra[:5]}")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in table:
            w.writerow({c: cell(row.get(c, "")) for c in columns})
    return {**file_entry(path), "righe": len(table)}


def write_sheet(out_dir: Path, name: str, table: list[dict]) -> dict:
    """Un foglio del campione come lo scrive main: colonne canoniche (SHEET_COLUMNS) per i fogli col formato fisso,
    anche vuoti; per gli altri, quelle della prima riga."""
    return write_table(Path(out_dir) / name, table, SHEET_COLUMNS.get(name))


def legacy_empty(path: Path) -> bool:
    """Tabella vuota scritta senza colonne prima del 2026-11-05: la sola intestazione "unita" e nessuna riga."""
    return tsv_columns(path) == LEGACY_EMPTY and not read_tsv(path)


def sheet_units(campione: list[dict], mirati: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """Unità di un foglio: tabella d'origine, record e record_id nell'ordine dell'unità, coppia o no, con o senza QA, e
    le opzioni del formato 1. Problemi: unità ripetute."""
    units, problems = {}, []
    for name, rows in (("campione", campione), ("mirati", mirati)):
        for row in rows:
            uid = cell(row.get("unita", ""))
            if uid in units:
                problems.append(f"{uid}: unità ripetuta")
                continue
            records = cell(row.get("record", "")).split("|")
            units[uid] = {"tabella": name, "records": records, "record_ids": cell(row.get("record_id", "")).split("|"),
                          "coppia": len(records) == 2 and set(cell(row.get("insieme", "")).split("|")) == {"coppie"},
                          "con_qa": cell(row.get("posizione_qa", "")) != "", "opzioni": row.get("opzioni")}
    return units, problems


def expected_variants(u: dict, fmt: str, kind: str) -> dict[str, int]:
    """Varianti attese con una riga di `kind` (soluzioni o opzioni) e posizione del loro record nell'unità. Formato
    attuale: a e b nelle coppie, "" nelle altre unità con una QA. Formato 1: la sola a nelle soluzioni delle coppie
    mirate e nelle opzioni di tutte le coppie (la riga dell'unità aveva le opzioni del primo record)."""
    if not u["con_qa"]:
        return {}
    if not u["coppia"]:
        return {"": 0} if len(u["records"]) == 1 else {}
    if fmt == FORMAT_V1 and (kind == "opzioni" or u["tabella"] == "mirati"):
        return {"a": 0}
    return {"a": 0, "b": 1}


def split_aggregated(row: dict) -> dict[str, dict] | None:
    """Soluzione del formato 1 di una coppia casuale ("a: x | b: y", esiti e stati della scena separati da "|")
    separata per variante; None se non si separa in modo univoco."""
    answers = row.get("risposta", "").split(" | ")
    status = row.get("esito_checker", "").split("|")
    scene = row.get("stato_scena_sha256", "").split("|")
    if not len(answers) == len(status) == len(scene) == 2:
        return None
    out = {}
    for v, a, s, h in zip("ab", answers, status, scene):
        if not a.startswith(f"{v}: "):
            return None
        out[v] = {"risposta": a[len(v) + 2:], "esito_checker": s, "stato_scena_sha256": h}
    return out


def sheet_index(units: dict[str, dict], solutions: list[dict], options: list[dict] | None,
                fmt: str) -> tuple[dict, dict, list[str], list[str]]:
    """Soluzioni e opzioni di un foglio per (unita, variante), con i problemi di struttura: righe ripetute, unità
    assenti, varianti inattese o mancanti, varianti associate a un record diverso da quello dell'unità. Nel formato 1
    le risposte aggregate delle coppie casuali si separano per variante (restituisce anche quelle unità) e le opzioni
    vengono dal foglio dell'unità (options None)."""
    problems, split = [], []
    indexes: dict[str, dict] = {"soluzioni": {}, "opzioni": {}}

    def add(kind: str, uid: str, v: str, fields: dict) -> None:
        index, where = indexes[kind], f"{kind} {uid}/{v or '-'}"
        u = units.get(uid)
        if u is None:
            problems.append(f"{where}: unità assente da campione e mirati")
            return
        if (uid, v) in index:
            problems.append(f"{where}: riga ripetuta")
            return
        want = expected_variants(u, fmt, kind)
        if v not in want:
            problems.append(f"{where}: variante inattesa")
            return
        i = want[v]
        record = u["records"][i] if i < len(u["records"]) else ""
        record_id = u["record_ids"][i] if i < len(u["record_ids"]) else ""
        if v and not record.endswith(f"-{v}"):
            problems.append(f"{where}: l'unità ha il record {record} al posto della variante {v}")
        if fmt == FORMAT_V1:
            fields = {"record": record, **fields}
        elif fields.get("record") != record or fields.get("record_id", record_id) != record_id:
            problems.append(f"{where}: associata al record {fields.get('record')} ({fields.get('record_id', '-')}) "
                            f"invece di {record} ({record_id})")
        index[(uid, v)] = fields

    for row in solutions:
        uid = cell(row.get("unita", ""))
        if fmt != FORMAT_V1:
            add("soluzioni", uid, cell(row.get("variante", "")), {f: cell(row.get(f, "")) for f in SAME_SOLUTION[fmt]})
            continue
        u = units.get(uid) or {}
        if u.get("coppia") and u.get("tabella") == "campione":
            parts = split_aggregated(row)
            if parts is None:
                problems.append(f"soluzioni {uid}: risposta aggregata della coppia non separabile")
                continue
            split.append(uid)
            for v, fields in parts.items():
                add("soluzioni", uid, v, fields)
        else:
            add("soluzioni", uid, "a" if u.get("coppia") else "",
                {f: cell(row.get(f, "")) for f in SAME_SOLUTION[FORMAT_V1][1:]})
    if fmt == FORMAT_V1:
        for uid, u in units.items():
            if u["con_qa"]:
                add("opzioni", uid, "a" if u["coppia"] else "", {"opzioni": cell(u["opzioni"] or "")})
    else:
        for row in options or []:
            add("opzioni", cell(row.get("unita", "")), cell(row.get("variante", "")),
                {f: cell(row.get(f, "")) for f in SAME_OPTION[fmt]})
    for kind, index in indexes.items():
        for uid, u in units.items():
            for v in expected_variants(u, fmt, kind):
                if (uid, v) not in index:
                    problems.append(f"{kind} {uid}/{v or '-'}: variante mancante")
    return indexes["soluzioni"], indexes["opzioni"], problems, split


def compare_index(old: dict, new: dict, fields: tuple, allowed_new: set) -> tuple[list, list, list, list]:
    """Confronto per (unita, variante): (campi diversi, chiavi solo nel precedente, chiavi solo nel nuovo non previste,
    chiavi solo nel nuovo previste dalla trasformazione dei formati)."""
    changed = []
    for key in sorted(set(old) & set(new)):
        bad = [f for f in fields if old[key].get(f) != new[key].get(f)]
        if bad:
            changed.append({"unita": key[0], "variante": key[1], "campi": bad,
                            "vecchio": {f: old[key].get(f) for f in bad}, "nuovo": {f: new[key].get(f) for f in bad}})

    def name(key: tuple) -> str:
        return f"{key[0]}/{key[1] or '-'}"

    added = sorted(set(new) - set(old))
    return (changed, [name(k) for k in sorted(set(old) - set(new))], [name(k) for k in added if k not in allowed_new],
            [name(k) for k in added if k in allowed_new])


def format_problems(prev: Path, fmt, old_c: list[dict], old_m: list[dict],
                    tables: dict[str, list[dict]]) -> tuple[list[str], list[str]]:
    """(problemi, tabelle vuote del vecchio scrittore). Il formato del foglio precedente si legge dallo schema del suo
    report.json e deve coincidere con le colonne dei file; le tabelle nuove devono avere le colonne del formato
    attuale. Un file con la sola intestazione "unita" e nessuna riga è la tabella vuota che main scriveva senza colonne
    prima del 2026-11-05: si accetta, e si dichiara, solo se le unità del foglio non attendono righe in quella tabella;
    altrimenti è un file incompleto."""
    if fmt not in SOLUTION_COLUMNS:
        return [f"schema del campione precedente non riconosciuto: {fmt!r} (noti: {sorted(SOLUTION_COLUMNS)})"], []
    problems, legacy = [], []
    units, _ = sheet_units(old_c, old_m)
    tables_of_fmt = [("soluzioni.tsv", "soluzioni", SOLUTION_COLUMNS[fmt])]
    if fmt == SCHEMA:
        tables_of_fmt.append(("opzioni.tsv", "opzioni", OPTION_COLUMNS))
    for name, kind, want in tables_of_fmt:
        cols = tsv_columns(prev / name)
        if tuple(cols) == want:
            continue
        if legacy_empty(prev / name):
            expected = sum(len(expected_variants(u, fmt, kind)) for u in units.values())
            if expected:
                problems.append(f"{name} del precedente vuoto e senza colonne, ma le sue unità attendono {expected} righe")
            else:
                legacy.append(name)
            continue
        problems.append(f"{name} del precedente assente o con le colonne {cols}: {fmt} vuole {list(want)}")
    if fmt == FORMAT_V1 and any("opzioni" not in row for row in old_c + old_m):
        problems.append(f"{fmt}: campione.tsv o mirati.tsv senza la colonna opzioni")
    for name, columns in (("soluzioni", SOLUTION_COLUMNS[SCHEMA]), ("opzioni", OPTION_COLUMNS)):
        if any(set(columns) - set(row) for row in tables[name]):
            problems.append(f"{name} nuove senza le colonne di {SCHEMA}")
    return problems, legacy


def compare_previous(prev: Path, tables: dict[str, list[dict]], design: dict, seed: int, manifest_sha: str,
                     out_dir: Path) -> dict:
    """Selezione, ranghi, soluzioni, opzioni e immagini v4 del campione precedente contro quelli nuovi, con il formato
    del precedente dichiarato dallo schema del suo report.json.

    Formato attuale contro attuale: soluzioni e opzioni per (unita, variante), con record e record_id associati,
    posizione, tipo, risposta, esito del checker, stato e stato della scena; opzioni per variante da opzioni.tsv. Ogni
    foglio si controlla anche da solo: righe ripetute, varianti mancanti o inattese, varianti associate a un record
    diverso da quello dell'unità.

    Dal formato 1 (campione-1) solo le trasformazioni previste: le risposte aggregate delle coppie casuali ("a: x | b:
    y") si separano e si confrontano per variante; le varianti b assenti nel formato 1 (soluzioni delle coppie mirate,
    opzioni di tutte le coppie) sono aggiunte senza confronto, elencate in `trasformazioni`, e non si certificano
    invariate: `confronto_completo` è falso.

    Tabelle vuote: main scrive soluzioni.tsv e opzioni.tsv con le colonne canoniche anche senza righe (write_sheet). Un
    precedente con la sola intestazione "unita" e nessuna riga, come le scriveva main prima del 2026-11-05, si accetta
    solo se le sue unità non attendono righe in quella tabella, ed è elencato in formati.tabelle_vuote_senza_colonne;
    altrimenti è un file incompleto.

    `selezione_invariata` è falso se cambia una sola unità, un rango, un record, una posizione, una QA, una risposta,
    un'opzione confrontata o un'immagine v4, o se un foglio ha un problema di struttura o un formato non riconosciuto."""
    old_report = json.loads((prev / "report.json").read_text(encoding="utf-8"))
    diffs: dict[str, list] = defaultdict(list)

    def same(name: str, old_rows: list[dict], new_rows: list[dict], fields: tuple) -> int:
        a = [tuple(row.get(f, "") for f in fields) for row in old_rows]
        b = [tuple(cell(row.get(f, "")) for f in fields) for row in new_rows]
        if a != b:
            diffs[name] += [{"vecchio": x, "nuovo": y} for x, y in zip(a, b) if x != y][:5] or [
                {"righe_vecchie": len(a), "righe_nuove": len(b)}]
        return len(a)

    out = {"precedente": str(prev), "report_precedente": file_entry(prev / "report.json")}
    old_design = old_report["campione"]["strati"]
    if old_report.get("seme") != seed or old_report.get("audit") != AUDIT:
        diffs["estrazione"].append({"seme": old_report.get("seme"), "audit": old_report.get("audit")})
    if old_report["manifest"]["manifest_sha256"] != manifest_sha:
        diffs["manifest"].append(old_report["manifest"]["manifest_sha256"])
    if json.loads(json.dumps(design)) != old_design:
        diffs["strati"].append({"vecchi": old_design, "nuovi": design})
    old_c, old_m = read_tsv(prev / "campione.tsv"), read_tsv(prev / "mirati.tsv")
    old_d = read_tsv(prev / "candidati_decontaminazione.tsv")
    out["unita_campione"] = same("campione", old_c, tables["campione"], SAME_SAMPLE)
    out["unita_mirate"] = same("mirati", old_m, tables["mirati"], SAME_TARGETED)
    out["candidati"] = same("candidati_decontaminazione", old_d, tables["candidati"], SAME_DECONTAM)
    # soluzioni e opzioni per (unita, variante), secondo il formato dichiarato del precedente
    fmt = old_report.get("schema")
    bad_format, legacy = format_problems(prev, fmt, old_c, old_m, tables)
    out["formati"] = {"precedente": fmt, "nuovo": SCHEMA, "tabelle_vuote_senza_colonne": legacy}
    if bad_format:
        diffs["formato"] += bad_format
        out["confronto_completo"] = False
    else:
        old_units, p_old = sheet_units(old_c, old_m)
        new_units, p_new = sheet_units(tables["campione"], tables["mirati"])
        old_sol, old_opt, p, split = sheet_index(old_units, read_tsv(prev / "soluzioni.tsv"),
                                                 None if fmt == FORMAT_V1 else read_tsv(prev / "opzioni.tsv"), fmt)
        p_old += p
        new_sol, new_opt, p, _ = sheet_index(new_units, tables["soluzioni"], tables["opzioni"], SCHEMA)
        p_new += p
        if p_old:
            diffs["struttura_precedente"] += p_old
        if p_new:
            diffs["struttura_nuova"] += p_new
        # aggiunte previste solo dal formato 1: le varianti b che quel formato non aveva
        pairs = {uid for uid, u in old_units.items() if u["coppia"]} if fmt == FORMAT_V1 else set()
        allowed = {"soluzioni": {(uid, "b") for uid in pairs if old_units[uid]["tabella"] == "mirati"},
                   "opzioni": {(uid, "b") for uid in pairs}}
        added = {}
        for kind, old_idx, new_idx, fields in (("soluzioni", old_sol, new_sol, SAME_SOLUTION[fmt]),
                                               ("opzioni", old_opt, new_opt, SAME_OPTION[fmt])):
            changed, only_old, unexpected, added[kind] = compare_index(old_idx, new_idx, fields, allowed[kind])
            if changed:
                diffs[kind] += changed
            if only_old:
                diffs[f"{kind}_mancanti_nel_nuovo"] += only_old
            if unexpected:
                diffs[f"{kind}_aggiunte_non_previste"] += unexpected
            out[f"{kind}_confrontate"] = len(set(old_idx) & set(new_idx))
        out["trasformazioni"] = {
            "risposte_aggregate_separate": split, "varianti_aggiunte_senza_confronto": added,
            "nota": "trasformazioni previste solo dal formato 1: risposte aggregate delle coppie casuali separate per "
                    "variante; varianti b assenti nel formato 1 (soluzioni delle coppie mirate, opzioni di tutte le "
                    "coppie) aggiunte senza confronto, quindi non certificate invariate"}
        out["seconde_varianti_aggiunte"] = len(added["soluzioni"])
        out["confronto_completo"] = not any(added.values())
    # anteprima del foglio precedente: prefisso dell'elenco completo
    full = {d["unita"]: d["elementi"].split("; ") if d["elementi"] else [] for d in tables["candidati"]}
    longer, extra = 0, 0
    for row in old_d:
        old_list = row["elementi"].split("; ") if row["elementi"] else []
        new_list = full.get(row["unita"], [])
        if new_list[:len(old_list)] != old_list:
            diffs["elementi_candidati"].append({"unita": row["unita"], "vecchi": old_list[:3], "nuovi": new_list[:3]})
        longer += len(new_list) > len(old_list)
        extra += max(0, len(new_list) - len(old_list))
    out["candidati_con_corrispondenze_mancanti_nel_foglio_precedente"] = longer
    out["corrispondenze_mancanti_nel_foglio_precedente"] = extra
    # immagini v4: stessi file, stessi byte
    def v4_images(rows: list[dict]) -> set[str]:
        return {p for row in rows for p in (row.get("immagini") or "").split("|")
                if p.startswith("immagini/synth-") and "-pannello-" not in p}

    old_png = v4_images(old_c + old_m + old_d)
    new_png = v4_images(tables["campione"] + tables["mirati"] + tables["candidati"])
    if old_png != new_png:
        diffs["immagini_v4"].append({"solo_vecchie": sorted(old_png - new_png)[:5],
                                     "solo_nuove": sorted(new_png - old_png)[:5]})
    bad = [p for p in sorted(old_png & new_png) if sha256_file(prev / p) != sha256_file(out_dir / p)]
    if bad:
        diffs["immagini_v4"].append({"byte_diversi": bad[:5]})
    out["immagini_v4_confrontate"] = len(old_png & new_png)
    out["differenze"] = dict(diffs)
    out["selezione_invariata"] = not diffs
    return out


# ---------------------------------------------------------------- job

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--index", type=Path, help="cartella dell'indice del pannello, per le immagini dei candidati")
    ap.add_argument("--review", type=Path, help="uscita di decontam_review.py per la scansione (review.jsonl)")
    ap.add_argument("--previous", type=Path, help="campione precedente da confrontare (stessa estrazione)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-threshold", type=int, default=8)
    ap.add_argument("--pairs-per-family", type=int, default=5)
    ap.add_argument("--contamination", type=int, default=10)
    ap.add_argument("--per-rule", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("synth_gold_sample.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: il campione si scrive una volta sola")
    t0 = time.time()

    # ---- manifest, shard ed esiti del checker
    m = mf.load(args.manifest)
    raw = (args.manifest / m.meta["records"]["file"]).read_bytes()
    records = [json.loads(x) for x in gzip.decompress(raw).decode("utf-8").splitlines()]
    by_key = {r["record"]: r for r in records}
    for s in m.meta["shards"]:
        p = mf.shard_path(s["percorso"])
        if p.stat().st_size != s["bytes"] or sha256_file(p) != s["sha256"]:
            sys.exit(f"{p}: sha256 o dimensione diversi da quelli del manifest")
    cand_entry = m.meta["ingressi"]["candidato_records"]
    if sha256_file(cand_entry["path"]) != cand_entry["sha256"]:
        sys.exit(f"{cand_entry['path']}: sha256 diverso da quello registrato nel manifest")
    checker = {}
    with open(cand_entry["path"], encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            checker[row["key"]] = row["questions"]

    # ---- campione casuale
    pops = populations(records, args.seed)
    sample, design = [], {}
    for stratum, n in SIZES.items():
        units = pops.get(stratum, [])
        take = units[:n]
        design[stratum] = {"N": len(units), "n": len(take), "p_inclusione": round(len(take) / len(units), 6)
                           if units else None, "unita": "coppia" if stratum == "coppie" else "QA",
                           "limite_superiore_95_con_zero_errori": upper_zero(len(units), len(take))}
        for i, u in enumerate(take, 1):
            sample.append({**u, "unita": f"C-{stratum}-{i:03d}", "strato": stratum, "N": len(units), "n": len(take)})
    in_sample = {(u["records"][0]["record"], u["posizione"]) for u in sample}
    sampled_pairs = {u.get("coppia") for u in sample if u.get("coppia")}

    # ---- casi mirati che non chiedono il layout
    targeted = []

    def add_target(category: str, rs: list[dict], pos, tipo, why: str, value=None) -> None:
        targeted.append({"categoria": category, "records": rs, "posizione": pos, "tipo": tipo, "motivo": why,
                         "misura": value})

    for r in records:  # quarantena per motivi propri della scena; i candidati della decontaminazione stanno sotto
        if r["stato"] == "quarantena" and set(r["motivo"] or {}) - {CHECK}:
            add_target("quarantena", [r], None, "scena", json.dumps(r["motivo"], ensure_ascii=False))
    pairs_by_family = defaultdict(list)
    for u in pops.get("coppie", []):
        if u["coppia"] not in sampled_pairs:
            pairs_by_family[u["records"][0]["famiglia"]].append(u)
    for fam in FAMILIES:
        for u in pairs_by_family[fam][:args.pairs_per_family]:
            add_target("coppie_per_famiglia", u["records"], 0, u["tipo"], f"coppia {u['coppia']}, rango {u['rango'][:12]}")
    by_group = defaultdict(list)
    for r in records:
        by_group[r["gruppo"]].append(r)
    excluded = sorted((r for r in records if r["stato"] == "escluso" and "contaminazione" in (r["motivo"] or {})),
                      key=lambda r: rank({"audit": AUDIT, "seme": args.seed, "escluso": r["record_id"]}))
    for r in excluded[:args.contamination]:
        evals = [x for x in by_group[r["gruppo"]] if x["insieme"] != "train"]
        same = [x for x in evals if x["stato_scena_sha256"] == r["stato_scena_sha256"]] or evals
        if same:
            add_target("esclusioni_contaminazione", [r, min(same, key=lambda x: x["record"])], None, "scena",
                       "stessa istanza della valutazione?" + (" stesso stato della scena" if same[0]["stato_scena_sha256"]
                                                              == r["stato_scena_sha256"] else " stato nel gruppo"))
    rule_units = defaultdict(list)
    for stratum in FAMILIES:
        for u in pops.get(stratum, []):
            if u["tipo"] in NEW_RULES and (u["records"][0]["record"], u["posizione"]) not in in_sample:
                rule_units[u["tipo"]].append(u)
    for t in NEW_RULES:
        for u in rule_units[t][:args.per_rule]:
            add_target("regole_nuove_v4", u["records"], u["posizione"], t, f"regola v4 di {t}, rango {u['rango'][:12]}")
    # candidati della decontaminazione esterna: tutti, in un foglio a parte (non sono casi delle golden answer)
    decontam = []
    for r in records:
        e = last_evidence(r)
        if e is not None and e.get("esito") in ("candidato", "corrispondenza_confermata"):
            decontam.append({"rango": rank({"audit": AUDIT, "seme": args.seed, "decontaminazione": r["record_id"]}),
                             "records": [r], "evidenza": e})
    decontam.sort(key=lambda x: x["rango"])
    for n, u in enumerate(decontam, 1):
        u["unita"] = f"D-{n:03d}"
    review = {}
    if args.review:  # somiglianza dei pixel di decontam_review.py, solo per ordinare la verifica
        with open(args.review / "review.jsonl", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                review[row["key"]] = row.get("image") or {}

    # ---- scansione completa, seguendo il riferimento delle evidenze
    dmeta = m.meta.get("decontaminazione") or {}
    if any(not (u["evidenza"].get("rif") or {}).get("file") for u in decontam):
        sys.exit("evidenza della decontaminazione senza riferimento alla scansione")
    refs = {(u["evidenza"]["rif"]["file"], u["evidenza"]["rif"]["sha256"]) for u in decontam}
    scan_rows, scan_info, matches_lines, summary = {}, {}, [], {}
    if decontam:
        if len(refs) != 1:
            sys.exit(f"le evidenze citano {len(refs)} scansioni diverse: {sorted(refs)[:3]}")
        (ref_file, ref_sha), = refs
        scan_path = Path(ref_file) if Path(ref_file).is_absolute() else ROOT / ref_file
        declared = dmeta.get("scansione") or {}
        for name in ("records", "summary", "matches"):
            entry = declared.get(name) or {}
            if not entry.get("path") or sha256_file(entry["path"]) != entry.get("sha256"):
                sys.exit(f"scansione: {name} assente o con sha256 diverso da quello del manifest ({entry.get('path')})")
        if Path(declared["records"]["path"]).resolve() != scan_path.resolve() \
                or declared["records"]["sha256"] != ref_sha or sha256_file(scan_path) != ref_sha:
            sys.exit(f"{scan_path}: il riferimento delle evidenze non è il records.jsonl.gz del manifest o ha un altro "
                     "sha256")
        summary = json.loads(Path(declared["summary"]["path"]).read_text(encoding="utf-8"))
        if summary.get("files", {}).get("records.jsonl.gz", {}).get("sha256") != ref_sha:
            sys.exit("summary.json della scansione non registra lo sha256 di records.jsonl.gz delle evidenze")
        with gzip.open(scan_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if row["key"] in scan_rows:
                        sys.exit(f"{row['key']}: riga ripetuta nella scansione")
                    scan_rows[row["key"]] = {k: row.get(k) for k in ("key", "shard", "posizione", "byte_sha256",
                                                                      "corrispondenze", "dettagli",
                                                                      "dettagli_completi")}
        with open(declared["matches"]["path"], encoding="utf-8") as fh:
            matches_lines = [json.loads(line) for line in fh if line.strip()]
        if len(scan_rows) != summary.get("samples") or set(scan_rows) != set(by_key):
            sys.exit(f"scansione: {len(scan_rows)} righe, summary {summary.get('samples')}, manifest {len(by_key)}")
        scan_info = {"records": file_entry(scan_path, ref_sha), "summary": declared["summary"],
                     "matches": declared["matches"], "job": summary.get("job_id"), "code": summary.get("code"),
                     "max_lines": summary.get("max_lines"), "radius": summary.get("radius")}

    # ---- lettura degli shard: margini sulle scene ammesse di train e dev, testi delle unità, immagini
    wanted = {r["record"] for u in sample + targeted + decontam for r in u["records"]}
    k = args.per_threshold
    near = {name: Best(k) for name in ("soglia_testo_basso", "soglia_riquadri_vicini", "soglia_intersezione_bordo",
                                       "soglia_grafici_vicini", "soglia_vertice_estremo", "coordinate_dalle_tacche")}
    metas, images, raws, seen = {}, {}, {}, 0
    for s in m.meta["shards"]:
        for index, (key, files) in enumerate(iter_shard(mf.shard_path(s["percorso"]))):
            r = by_key.get(key)
            if r is None or r["shard"] != s["percorso"] or r["indice"] != index:
                sys.exit(f"{key}: non è il record del manifest in {s['percorso']} posizione {index}")
            if hashlib.sha256(files["json"]).hexdigest() != r["json_sha256"] \
                    or hashlib.sha256(files["png"]).hexdigest() != r["immagini"][0]["sha256"]:
                sys.exit(f"{key}: JSON o PNG diversi da quelli del manifest")
            seen += 1
            meta = json.loads(files["json"])
            if key in wanted:
                metas[key], images[key], raws[key] = meta, files["png"], files["json"]
            if r["stato"] != "ammesso" or r["split"] not in ("train", "dev") or r["insieme"] != "train":
                continue
            layout = meta["layout"]
            # a parità di misura decide il rango della scena o della QA
            ties = {pos: rank({"audit": AUDIT, "seme": args.seed, "soglia": r["record_id"], "posizione": pos})
                    for pos in [None, *range(len(meta["qa"]))]}
            low, close = text_margins(layout)
            if low is not None:
                near["soglia_testo_basso"].add(low, ties[None], (key, None, "scena", f"testo più basso {low:.1f} px "
                                                                 f"(soglia {sc.MIN_TEXT_PX})", low))
            if close is not None:
                near["soglia_riquadri_vicini"].add(-close, ties[None], (key, None, "scena", f"riquadri vicini: "
                                                                        f"{close:.1f} px (violazione oltre "
                                                                        f"{sc.OVERLAP_PX})", close))
            qas = meta["qa"]
            if meta["family"] == "function":
                edge, gap = intersection_margins(meta)
                for pos, q in enumerate(qas):
                    if q["type"] != "intersections":
                        continue
                    if edge is not None:
                        near["soglia_intersezione_bordo"].add(edge, ties[pos], (key, pos, q["type"], f"intersezione a "
                                                                                f"{edge:.3f} dal bordo (soglia "
                                                                                f"{sc.EDGE})", edge))
                    if gap is not None:
                        near["soglia_grafici_vicini"].add(gap, ties[pos], (key, pos, q["type"], f"grafici a {gap:.3f} "
                                                                           f"senza incrociarsi (soglia {sc.TANGENT})",
                                                                           gap))
            if meta["family"] == "polygon":
                margins = extreme_margins(layout)
                for pos, q in enumerate(qas):
                    if q["type"] in margins:
                        v = margins[q["type"]]
                        near["soglia_vertice_estremo"].add(v, ties[pos], (key, pos, q["type"], f"stacco {v:.3f} del "
                                                                          f"lato (soglia {sc.EXTREME})", v))
            if meta["family"] == "coord":
                for pos, q in enumerate(qas):
                    if q["type"] == "coords" and coords_from_ticks(layout, q["answer"]):
                        near["coordinate_dalle_tacche"].add(0.0, ties[pos], (key, pos, q["type"], "coordinata senza "
                                                                             "etichetta: si contano le tacche", None))
                        break  # una QA per scena
    if seen != len(records):
        sys.exit(f"letti {seen} record, il manifest ne ha {len(records)}")
    for name, best in near.items():
        for key, pos, tipo, why, value in best.items():
            targeted.append({"categoria": name, "records": [by_key[key]], "posizione": pos, "tipo": tipo,
                             "motivo": why, "misura": value})
            wanted.add(key)
    missing = sorted(wanted - set(metas))
    if missing:  # margini scelti dopo la lettura: una seconda lettura solo per loro
        need = set(missing)
        for s in m.meta["shards"]:
            for key, files in iter_shard(mf.shard_path(s["percorso"])):
                if key in need:
                    if hashlib.sha256(files["png"]).hexdigest() != by_key[key]["immagini"][0]["sha256"] \
                            or hashlib.sha256(files["json"]).hexdigest() != by_key[key]["json_sha256"]:
                        sys.exit(f"{key}: JSON o PNG diversi da quelli del manifest")
                    metas[key], images[key], raws[key] = json.loads(files["json"]), files["png"], files["json"]

    # ---- pannello: voci e immagini di tutte le corrispondenze dei candidati
    entries, elements, panel_bytes, panel_errors, panel_info = {}, {}, {}, [], {}
    needed = {(d.get("elemento"), d.get("pos")) for u in decontam
              for d in (scan_rows.get(u["records"][0]["record"]) or {}).get("dettagli") or []}
    if decontam:
        el_entry = (dmeta.get("pannello") or {}).get("elements") or {}
        if not el_entry.get("path") or sha256_file(el_entry["path"]) != el_entry.get("sha256"):
            sys.exit(f"elementi del pannello assenti o con sha256 diverso da quello del manifest ({el_entry.get('path')})")
        with gzip.open(el_entry["path"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    e = json.loads(line)
                    elements[e["chiave"]] = {k: e.get(k) for k in ("ruolo", "gruppo", "metrica_primaria")}
        panel_info["elements"] = el_entry
    if args.index and decontam:
        import panel_index

        idx_entry = dmeta.get("indice") or {}
        index_path = args.index / "index.json"
        if sha256_file(index_path) != idx_entry.get("sha256"):
            sys.exit(f"{index_path}: sha256 diverso dall'indice registrato nel manifest")
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        for it in loaded["items"]:
            if "item" in it and it.get("pos") is not None:
                where = (f"{it['benchmark']}/{it['item']}", it["pos"])
                if where in needed:
                    if where in entries:
                        sys.exit(f"{where}: due voci dell'indice per la stessa immagine")
                    entries[where] = it
        # tutte le voci degli elementi citati: resolve_file confronta il numero di immagini dell'elemento con le voci
        cited = {el for el, _ in needed}
        subset = [it for it in loaded["items"] if f"{it['benchmark']}/{it.get('item')}" in cited
                  and it.get("pos") is not None]
        out_bytes, panel_errors = panel_index.resolve_entries(subset, loaded["convenzione_immagini"], args.workers)
        for it, data in zip(subset, out_bytes):
            where = (f"{it['benchmark']}/{it['item']}", it["pos"])
            if data and where in needed:
                panel_bytes[where] = data
        panel_info.update(index=file_entry(index_path, idx_entry.get("sha256")), voci_necessarie=len(needed),
                          voci_trovate=len(entries), immagini_ritrovate=len(panel_bytes))

    # ---- scrittura
    args.out.mkdir(parents=True)
    (args.out / "immagini").mkdir()
    (args.out / "immagini" / "pannello").mkdir()
    written = {}

    def image_file(key: str) -> str:
        if key not in written:
            path = args.out / "immagini" / f"{key}.png"
            path.write_bytes(images[key])
            if sha256_file(path) != by_key[key]["immagini"][0]["sha256"]:
                sys.exit(f"{path}: copia diversa dal PNG del manifest")
            written[key] = f"immagini/{key}.png"
        return written[key]

    panel_files = {}
    if panel_bytes:
        from PIL import Image

        for where, data in sorted(panel_bytes.items()):
            digest = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as im:
                ext = (im.format or "bin").lower().replace("jpeg", "jpg")
            path = args.out / "immagini" / "pannello" / f"{digest}.{ext}"
            if not path.exists():
                path.write_bytes(data)  # i byte della sorgente, come li legge l'indice
            panel_files[where] = {"file": f"immagini/pannello/{path.name}", "sha256": digest}

    def qa_of(r: dict, pos) -> dict | None:
        meta = metas[r["record"]]
        items = list(meta["qa"]) + ([meta["pair"]] if "pair" in meta else [])
        if pos is None:
            return None
        q = items[pos]
        if mf.qa_digest(mf.Digests(), q) != r["qa"][pos]["sha256"]:
            sys.exit(f"{r['record']}: QA {pos} con digest diverso da quello del manifest")
        return q

    def status_of(r: dict, pos) -> str:
        qs = checker.get(r["record"]) or []
        return qs[pos][2] if pos is not None and pos < len(qs) else ""

    prompts, prompt_stats = [], Counter()

    def prompt_cols(uid: str, rs: list[dict], pos) -> dict:
        ps = [prompt_for(r, metas[r["record"]], {"json": raws[r["record"]], "png": images[r["record"]]}, pos)
              for r in rs if r["insieme"] == "coppie" or r is rs[0]]
        p = ps[0]
        if any(x["prompt"] != p["prompt"] for x in ps):
            sys.exit(f"{uid}: le due varianti della coppia hanno prompt diversi")
        digest = mf.sha256_text(p["prompt"]) if p["prompt"] else ""
        prompt_stats[p["origine_prompt"]] += 1
        prompts.append({"unita": uid, "record": [r["record"] for r in rs], "posizione_qa": pos, **p,
                        "prompt_sha256": digest or None})
        return {"prompt": tsv_text(p["prompt"]), "prompt_sha256": digest, "origine_prompt": p["origine_prompt"],
                "numero_nel_prompt": "" if p["numero_nel_prompt"] is None else p["numero_nel_prompt"],
                "domande_nel_prompt": len(p["domande_nel_prompt"]),
                "consegnato_dal_loader": "sì" if p["consegnato_dal_loader"] else "no"}

    rows, mrows, solutions, options, form = [], [], [], [], []
    for u in sample:
        rs = u["records"]
        qs = [qa_of(r, u["posizione"]) for r in rs]
        pc = prompt_cols(u["unita"], rs, u["posizione"])
        if pc["origine_prompt"] == "loader" and pc["numero_nel_prompt"] != u["ordine_nel_loader"] + 1:
            sys.exit(f"{u['unita']}: numero nel prompt {pc['numero_nel_prompt']}, ordine nel loader {u['ordine_nel_loader']}")
        rows.append({"unita": u["unita"], "strato": u["strato"], "rango": u["rango"], "N": u["N"], "n": u["n"],
                     "p_inclusione": f"{u['n']}/{u['N']}", "record": "|".join(r["record"] for r in rs),
                     "record_id": "|".join(r["record_id"] for r in rs), "insieme": rs[0]["insieme"],
                     "split": rs[0]["split"], "famiglia": rs[0]["famiglia"],
                     "shard": "|".join(r["shard"] for r in rs), "indice": "|".join(str(r["indice"]) for r in rs),
                     "posizione_qa": u["posizione"], "ordine_nel_loader": u["ordine_nel_loader"], "tipo": u["tipo"],
                     "domanda": qs[0]["q"], **pc,
                     "immagini": "|".join(image_file(r["record"]) for r in rs),
                     "immagini_sha256": "|".join(r["immagini"][0]["sha256"] for r in rs),
                     "qa_sha256": u["qa_sha256"]})
        sols, opts, frm = unit_rows(u["unita"], rs, metas, qs, u["posizione"], status_of)
        solutions += sols
        options += opts
        form += frm
    counters = Counter()
    for u in targeted:
        counters[u["categoria"]] += 1
        uid = f"M-{u['categoria']}-{counters[u['categoria']]:03d}"
        rs = u["records"]
        qs = [qa_of(r, u["posizione"]) for r in rs] if u["posizione"] is not None else [None] * len(rs)
        q = qs[0]
        pc = prompt_cols(uid, rs, u["posizione"])
        mrows.append({"unita": uid, "categoria": u["categoria"], "motivo": u["motivo"],
                      "misura": "" if u["misura"] is None else round(u["misura"], 4),
                      "record": "|".join(r["record"] for r in rs), "record_id": "|".join(r["record_id"] for r in rs),
                      "insieme": "|".join(r["insieme"] for r in rs), "split": "|".join(r["split"] for r in rs),
                      "stato": "|".join(r["stato"] for r in rs), "famiglia": rs[0]["famiglia"],
                      "posizione_qa": "" if u["posizione"] is None else u["posizione"], "tipo": u["tipo"],
                      "domanda": q["q"] if q else "scena intera: leggibilità e univocità dell'immagine", **pc,
                      "immagini": "|".join(image_file(r["record"]) for r in rs)})
        sols, opts, frm = unit_rows(uid, rs, metas, qs, u["posizione"], status_of)
        solutions += sols
        options += opts
        form += frm
    # il loader di training non costruisce il prompt delle coppie (le loro QA sono solo nel campo pair)
    pair_loader = None
    first_pair = next((r for u in sample for r in u["records"] if r["insieme"] == "coppie"), None)
    if first_pair is not None:
        try:
            loader_prompt(first_pair["record"], {"json": raws[first_pair["record"]], "png": images[first_pair["record"]]})
            pair_loader = f"{first_pair['record']}: il loader costruisce un prompt"
        except ValueError as exc:
            pair_loader = f"{first_pair['record']}: ValueError del loader ({exc})"

    # ---- candidati della decontaminazione: tutte le corrispondenze
    drows, dmatches, fatal = decontam_tables(decontam, scan_rows, entries, elements, panel_files)
    if fatal:
        sys.exit(f"{len(fatal)} incoerenze fra manifest e scansione, foglio non scritto: {fatal[:5]}")
    recon_checks, recon = reconcile(scan_rows, summary, matches_lines, {u["records"][0]["record"] for u in decontam},
                                    dmatches) if decontam else ([], {})
    for row in drows:
        im_rev = review.get(row["record"]) or {}
        row.update({"somiglianza_pixel": im_rev.get("sim", ""), "rapporto_aspetto": im_rev.get("aspect_ratio", ""),
                    "elemento_piu_simile": f"{im_rev.get('benchmark') or ''}/{im_rev.get('bench_id') or ''}"
                    if im_rev else "",
                    "immagini": "|".join([image_file(row["record"]), *filter(None, row.pop("immagini_pannello")
                                                                             .split("|"))]),
                    "domanda": "stessa immagine o stesso contenuto di un elemento del pannello?"})

    def write_tsv(name: str, table: list[dict]) -> dict:
        return write_sheet(args.out, name, table)

    # ---- inventario
    inv = defaultdict(Counter)
    for r in records:
        statuses = checker.get(r["record"]) or []
        for pos, q in enumerate(r["qa"]):
            fam_type = (r["famiglia"], q["tipo"], "coppia" if r["insieme"] == "coppie" else "scena")
            inv[fam_type]["qa"] += 1
            inv[fam_type]["qa_scene_ammesse"] += r["stato"] == "ammesso"
            inv[fam_type][f"checker_{statuses[pos][2] if pos < len(statuses) else 'assente'}"] += 1
    inventory = []
    for (fam, t, kind), c in sorted(inv.items()):
        corr, risp = PAIR_CHECK if kind == "coppia" else CHECKS.get(fam, {}).get(t, ("non censito", "non censito"))
        inventory.append({"famiglia": fam, "tipo": t, "unita": kind, "qa": c["qa"],
                          "qa_scene_ammesse": c["qa_scene_ammesse"],
                          "checker": ", ".join(f"{k[8:]} {v}" for k, v in sorted(c.items()) if k.startswith("checker_")),
                          "correttezza_matematica": "R parziale (generatore e checker dello stesso progetto): " + corr,
                          "coerenza_annotazioni_immagine": COMMON["coerenza"],
                          "rispondibilita_da_immagine_e_domanda": risp + "; C: revisione visiva",
                          "spiegazione": COMMON["spiegazione"], "gold": "nessuno: record candidato"})
    prompt_path = args.out / "prompts.jsonl"
    prompt_path.write_text("".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in prompts),
                           encoding="utf-8")
    files = {"campione.tsv": write_tsv("campione.tsv", rows), "mirati.tsv": write_tsv("mirati.tsv", mrows),
             "prompts.jsonl": {**file_entry(prompt_path), "righe": len(prompts)},
             "opzioni.tsv": write_tsv("opzioni.tsv", options),
             "soluzioni.tsv": write_tsv("soluzioni.tsv", solutions),
             "esiti.tsv": write_tsv("esiti.tsv", form),
             "inventario.tsv": write_tsv("inventario.tsv", inventory),
             "candidati_decontaminazione.tsv": write_tsv("candidati_decontaminazione.tsv", drows),
             "corrispondenze_decontaminazione.tsv": write_tsv("corrispondenze_decontaminazione.tsv", dmatches),
             "esiti_decontaminazione.tsv": write_tsv("esiti_decontaminazione.tsv",
                                                     [{"unita": x["unita"], "record": x["record"]} for x in drows])}
    comparison = None
    if args.previous:
        comparison = compare_previous(args.previous, {"campione": rows, "mirati": mrows, "candidati": drows,
                                                      "soluzioni": solutions, "opzioni": options},
                                      design, args.seed, m.manifest_sha256, args.out)
        write_new_json(args.out / "confronto.json", comparison)
        files["confronto.json"] = file_entry(args.out / "confronto.json")
    review_entry = file_entry(args.review / "review.jsonl") if args.review else None
    incomplete = [x["unita"] for x in drows if x["materiale"] != "completo"]
    report = {
        "schema": SCHEMA, "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
        "manifest": {"path": str(args.manifest), "versione": m.meta.get("versione"), "manifest_sha256": m.manifest_sha256,
                     "records_sha256": m.meta["records"]["sha256"]},
        "checker": file_entry(cand_entry["path"]), "audit": AUDIT, "seme": args.seed,
        "proprieta": {"correttezza_matematica": "il target è vero per la scena descritta dalla specifica",
                      "coerenza_annotazioni_immagine": "il PNG mostra ciò che dicono specifica e layout",
                      "rispondibilita_da_immagine_e_domanda": "l'informazione serve visibile, leggibile e univoca",
                      "spiegazione": COMMON["spiegazione"]},
        "protocollo": {**PROTOCOL, "origini_del_prompt": PROMPT_ORIGINS, "prompt_per_origine": dict(prompt_stats),
                       "loader_di_training_sulle_coppie": pair_loader,
                       "opzioni_nel_testo_della_domanda": dict(Counter(
                           f"{o['tipo']}: {o['opzioni_nel_testo_della_domanda']}" for o in options))},
        "campione": {"strati": design, "regola": "le n unità col digest più piccolo di {audit, seme, record_id, "
                                                 "posizione, qa} (coppie: {audit, seme, coppia}); un n più piccolo, "
                                                 "fissato prima di guardare, è il prefisso in ordine di rango",
                     "unita_totali": len(rows),
                     "nota_limite": "informativo: vale con 0 errori osservati, incerti e non giudicabili contati come "
                                    "errori, e resta condizionato alla sensibilità del revisore (S4 §4)"},
        "mirati": {"categorie": dict(counters), "unita_totali": len(mrows),
                   "nota": "casi diagnostici scelti per regola, non stimano un tasso"},
        "soluzioni": {"righe": len(solutions), "per_variante": dict(Counter(s["variante"] or "-" for s in solutions)),
                      "modulo": dict(Counter(f["variante"] or "-" for f in form))},
        "candidati_decontaminazione": {
            "record": len(drows), "esiti": dict(Counter(x["esito"] for x in drows)),
            "per_insieme_split_stato": dict(Counter(f"{x['insieme']}/{x['split']}/{x['stato']}" for x in drows)),
            "scansione": scan_info, "pannello": panel_info, "riconciliazione": recon_checks, "conteggi": recon,
            "materiale_incompleto": incomplete, "revisione_pixel": review_entry,
            "nota": "tutte le corrispondenze della scansione completa, non solo l'anteprima dell'evidenza del manifest; "
                    "la somiglianza dei pixel (decontam_review.py) serve solo a ordinare la verifica visiva, non "
                    "risolve un candidato"},
        "immagini_copiate": len(written), "immagini_pannello": len({f["sha256"] for f in panel_files.values()}),
        "errori_pannello": panel_errors[:20], "confronto": comparison,
        "limiti": LIMITS, "inventario": inventory, "files": files, "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "report.json", report)
    print(json.dumps({"strati": design, "mirati": dict(counters), "prompt": dict(prompt_stats),
                      "soluzioni": report["soluzioni"], "candidati_decontaminazione": len(drows),
                      "corrispondenze": recon, "riconciliazione_fallita": [c["controllo"] for c in recon_checks
                                                                          if not c["esito"]],
                      "materiale_incompleto": len(incomplete), "immagini": len(written),
                      "immagini_pannello": report["immagini_pannello"], "errori_pannello": len(panel_errors),
                      "selezione_invariata": None if comparison is None else comparison["selezione_invariata"],
                      "confronto_completo": None if comparison is None else comparison["confronto_completo"],
                      "seconds": report["seconds"]}, ensure_ascii=False))
    problems = []
    if panel_errors:
        problems.append(f"{len(panel_errors)} immagini del pannello non ritrovate esattamente, per esempio "
                        f"{panel_errors[:3]}")
    if any(not c["esito"] for c in recon_checks):
        problems.append("riconciliazione fallita: " + "; ".join(c["controllo"] for c in recon_checks if not c["esito"]))
    if comparison is not None and not comparison["selezione_invariata"]:
        problems.append(f"selezione diversa dal campione precedente: {sorted(comparison['differenze'])}")
    if problems:
        sys.exit(" | ".join(problems))


if __name__ == "__main__":
    main()
