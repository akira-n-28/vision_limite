#!/usr/bin/env python3
"""Preparazione della verifica delle golden answer dei sintetici v4: inventario dei controlli disponibili e campione
riproducibile per la revisione visiva (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §§1, 2.8, 2.10; S4). Solo in un job
SLURM CPU. Manifest, shard e uscite dei controlli si leggono soltanto: nessun dato o spiegazione rigenerati, nessuna
revisione avviata, nessun agente.

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
delle regole nuove di v4 (intersections, shape_type e label_type con le opzioni); i candidati della decontaminazione
esterna, con le immagini del pannello se c'è --index.

**Fogli:** ogni unità ha le immagini copiate byte per byte dagli shard (immagini/, sha256 controllato), la domanda e le
opzioni. Le risposte attese stanno a parte in soluzioni.tsv, per rispondere prima senza vederle (S4 §5); esiti.tsv è il
modulo vuoto, con una colonna per proprietà.

    srun $RUN_PYTHON -u -B scripts/synth_gold_sample.py --manifest DIR --out DIR [--index DIR] [--seed 0]
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
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.f1c import pick_questions  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, sha256_file, write_new_json  # noqa: E402

AUDIT = "synth-v4-1"  # etichetta dell'estrazione (S2 §2.8)
FAMILIES = ("coord", "function", "polygon", "shapes")
SIZES = {"coord": 75, "function": 75, "polygon": 75, "shapes": 75, "test": 50, "coppie": 30}  # proposta S2 §2.8
EXTREME_TYPES = {"highest": (1, 1), "lowest": (1, -1), "leftmost": (0, -1), "rightmost": (0, 1)}
NEW_RULES = ("intersections", "shape_type", "label_type")
COORDS = re.compile(r"^\((-?\d+), (-?\d+)\)$")
REVIEW_COLUMNS = ("unita", "risposta_revisore", "corretta", "coerente_con_annotazioni", "rispondibile_da_immagine",
                  "spiegazione", "gravita", "tipo_difetto", "tempo_s", "note")

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--index", type=Path, help="cartella dell'indice del pannello, per le immagini dei candidati")
    ap.add_argument("--review", type=Path, help="uscita di decontam_review.py per la scansione (review.jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-threshold", type=int, default=8)
    ap.add_argument("--pairs-per-family", type=int, default=5)
    ap.add_argument("--contamination", type=int, default=10)
    ap.add_argument("--per-rule", type=int, default=10)
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
        if r["stato"] == "quarantena" and set(r["motivo"] or {}) - {"decontaminazione_pannello"}:
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
        for e in r.get("evidenze") or []:
            if e.get("controllo") == "decontaminazione_pannello" and e.get("esito") in ("candidato",
                                                                                         "corrispondenza_confermata"):
                decontam.append({"rango": rank({"audit": AUDIT, "seme": args.seed, "decontaminazione": r["record_id"]}),
                                 "records": [r], "evidenza": e})
    decontam.sort(key=lambda x: x["rango"])
    review = {}
    if args.review:  # somiglianza dei pixel di decontam_review.py, solo per ordinare la verifica
        with open(args.review / "review.jsonl", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                review[row["key"]] = row.get("image") or {}

    # ---- lettura degli shard: margini sulle scene ammesse di train e dev, testi delle unità, immagini
    wanted = {r["record"] for u in sample + targeted + decontam for r in u["records"]}
    k = args.per_threshold
    near = {name: Best(k) for name in ("soglia_testo_basso", "soglia_riquadri_vicini", "soglia_intersezione_bordo",
                                       "soglia_grafici_vicini", "soglia_vertice_estremo", "coordinate_dalle_tacche")}
    metas, images, seen = {}, {}, 0
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
                metas[key], images[key] = meta, files["png"]
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
                    if hashlib.sha256(files["png"]).hexdigest() != by_key[key]["immagini"][0]["sha256"]:
                        sys.exit(f"{key}: PNG diverso da quello del manifest")
                    metas[key], images[key] = json.loads(files["json"]), files["png"]

    # ---- pannello: immagini dei candidati della decontaminazione
    panel_images, panel_errors = {}, []
    if args.index and decontam:
        import panel_index

        loaded = json.loads((args.index / "index.json").read_text(encoding="utf-8"))
        wanted_el = {(x["elemento"], x["posizione"]): None for u in decontam for x in u["evidenza"]["elementi"]}
        # tutte le voci degli elementi citati: resolve_file confronta il numero di immagini dell'elemento con le voci
        elements = {el for el, _ in wanted_el}
        subset = [it for it in loaded["items"] if f"{it['benchmark']}/{it['item']}" in elements
                  and it.get("pos") is not None]
        out_bytes, panel_errors = panel_index.resolve_entries(subset, loaded["convenzione_immagini"], 4)
        for it, data in zip(subset, out_bytes):
            if data and (f"{it['benchmark']}/{it['item']}", it["pos"]) in wanted_el:
                panel_images[(f"{it['benchmark']}/{it['item']}", it["pos"])] = data

    # ---- scrittura
    args.out.mkdir(parents=True)
    (args.out / "immagini").mkdir()
    written = {}

    def image_file(key: str) -> str:
        if key not in written:
            path = args.out / "immagini" / f"{key}.png"
            path.write_bytes(images[key])
            if sha256_file(path) != by_key[key]["immagini"][0]["sha256"]:
                sys.exit(f"{path}: copia diversa dal PNG del manifest")
            written[key] = f"immagini/{key}.png"
        return written[key]

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

    rows, mrows, solutions = [], [], []
    for u in sample:
        rs = u["records"]
        qs = [qa_of(r, u["posizione"]) for r in rs]
        rows.append({"unita": u["unita"], "strato": u["strato"], "rango": u["rango"], "N": u["N"], "n": u["n"],
                     "p_inclusione": f"{u['n']}/{u['N']}", "record": "|".join(r["record"] for r in rs),
                     "record_id": "|".join(r["record_id"] for r in rs), "insieme": rs[0]["insieme"],
                     "split": rs[0]["split"], "famiglia": rs[0]["famiglia"],
                     "shard": "|".join(r["shard"] for r in rs), "indice": "|".join(str(r["indice"]) for r in rs),
                     "posizione_qa": u["posizione"], "ordine_nel_loader": u["ordine_nel_loader"], "tipo": u["tipo"],
                     "domanda": qs[0]["q"], "opzioni": " / ".join(qs[0]["options"]),
                     "immagini": "|".join(image_file(r["record"]) for r in rs),
                     "immagini_sha256": "|".join(r["immagini"][0]["sha256"] for r in rs),
                     "qa_sha256": u["qa_sha256"]})
        solutions.append({"unita": u["unita"], "risposta": " | ".join(f"{r['record'][-1] if len(rs) > 1 else ''}"
                                                                      f"{': ' if len(rs) > 1 else ''}{q['answer']}"
                                                                      for r, q in zip(rs, qs)),
                          "esito_checker": "|".join(status_of(r, u["posizione"]) for r in rs),
                          "stato_scena_sha256": "|".join(r["stato_scena_sha256"] for r in rs)})
    counters = Counter()
    for u in targeted:
        counters[u["categoria"]] += 1
        uid = f"M-{u['categoria']}-{counters[u['categoria']]:03d}"
        rs = u["records"]
        q = qa_of(rs[0], u["posizione"])
        mrows.append({"unita": uid, "categoria": u["categoria"], "motivo": u["motivo"],
                      "misura": "" if u["misura"] is None else round(u["misura"], 4),
                      "record": "|".join(r["record"] for r in rs), "record_id": "|".join(r["record_id"] for r in rs),
                      "insieme": "|".join(r["insieme"] for r in rs), "split": "|".join(r["split"] for r in rs),
                      "stato": "|".join(r["stato"] for r in rs), "famiglia": rs[0]["famiglia"],
                      "posizione_qa": "" if u["posizione"] is None else u["posizione"], "tipo": u["tipo"],
                      "domanda": q["q"] if q else "scena intera: leggibilità e univocità dell'immagine",
                      "opzioni": " / ".join(q["options"]) if q else "",
                      "immagini": "|".join(image_file(r["record"]) for r in rs)})
        if q is not None:
            solutions.append({"unita": uid, "risposta": q["answer"], "esito_checker": status_of(rs[0], u["posizione"]),
                              "stato_scena_sha256": rs[0]["stato_scena_sha256"]})
    drows = []
    for n, u in enumerate(decontam, 1):
        uid, (r, e) = f"D-{n:03d}", (u["records"][0], u["evidenza"])
        extra = []
        for j, x in enumerate(e["elementi"]):
            data = panel_images.get((x["elemento"], x["posizione"]))
            if data:
                from PIL import Image

                with Image.open(io.BytesIO(data)) as im:
                    ext = (im.format or "bin").lower().replace("jpeg", "jpg")
                path = args.out / "immagini" / f"{uid}-pannello-{j}.{ext}"
                path.write_bytes(data)  # i byte della sorgente, come li legge l'indice
                extra.append(f"immagini/{path.name}")
        im_rev = review.get(r["record"]) or {}
        drows.append({"unita": uid, "rango": u["rango"], "record": r["record"], "record_id": r["record_id"],
                      "insieme": r["insieme"], "split": r["split"], "stato": r["stato"], "famiglia": r["famiglia"],
                      "esito": e["esito"], "esito_gruppo": e["esito_gruppo"],
                      "corrispondenze": json.dumps(e["corrispondenze"], sort_keys=True),
                      "elementi": "; ".join(f"{x['tipo']} {x['elemento']} pos {x['posizione']} d={x['distanza']} "
                                            f"{x['ruolo']}" + ("" if x["metrica_primaria"] is None else
                                                               f" metrica_primaria={x['metrica_primaria']}")
                                            for x in e["elementi"]),
                      "somiglianza_pixel": im_rev.get("sim", ""), "rapporto_aspetto": im_rev.get("aspect_ratio", ""),
                      "elemento_piu_simile": f"{im_rev.get('benchmark') or ''}/{im_rev.get('bench_id') or ''}"
                      if im_rev else "",
                      "immagini": "|".join([image_file(r["record"]), *extra]),
                      "domanda": "stessa immagine o stesso contenuto di un elemento del pannello?"})

    def write_tsv(name: str, table: list[dict], columns=None) -> dict:
        path = args.out / name
        columns = columns or (list(table[0]) if table else ["unita"])
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for row in table:
                w.writerow({c: str(row.get(c, "")).replace("\t", " ").replace("\n", " ") for c in columns})
        return {**file_entry(path), "righe": len(table)}

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
    files = {"campione.tsv": write_tsv("campione.tsv", rows), "mirati.tsv": write_tsv("mirati.tsv", mrows),
             "soluzioni.tsv": write_tsv("soluzioni.tsv", solutions),
             "esiti.tsv": write_tsv("esiti.tsv", [{"unita": x["unita"], "spiegazione": "non_applicabile"}
                                                  for x in rows + mrows], REVIEW_COLUMNS),
             "inventario.tsv": write_tsv("inventario.tsv", inventory),
             "candidati_decontaminazione.tsv": write_tsv("candidati_decontaminazione.tsv", drows),
             "esiti_decontaminazione.tsv": write_tsv("esiti_decontaminazione.tsv", [{"unita": x["unita"]} for x in drows],
                                                     ("unita", "stesso_contenuto", "elemento_guardato", "note"))}
    review_entry = file_entry(args.review / "review.jsonl") if args.review else None
    report = {
        "schema": "synth-gold-sample/1", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
        "manifest": {"path": str(args.manifest), "versione": m.meta.get("versione"), "manifest_sha256": m.manifest_sha256,
                     "records_sha256": m.meta["records"]["sha256"]},
        "checker": file_entry(cand_entry["path"]), "audit": AUDIT, "seme": args.seed,
        "proprieta": {"correttezza_matematica": "il target è vero per la scena descritta dalla specifica",
                      "coerenza_annotazioni_immagine": "il PNG mostra ciò che dicono specifica e layout",
                      "rispondibilita_da_immagine_e_domanda": "l'informazione serve visibile, leggibile e univoca",
                      "spiegazione": COMMON["spiegazione"]},
        "campione": {"strati": design, "regola": "le n unità col digest più piccolo di {audit, seme, record_id, "
                                                 "posizione, qa} (coppie: {audit, seme, coppia}); un n più piccolo, "
                                                 "fissato prima di guardare, è il prefisso in ordine di rango",
                     "unita_totali": len(rows),
                     "nota_limite": "informativo: vale con 0 errori osservati, incerti e non giudicabili contati come "
                                    "errori, e resta condizionato alla sensibilità del revisore (S4 §4)"},
        "mirati": {"categorie": dict(counters), "unita_totali": len(mrows),
                   "nota": "casi diagnostici scelti per regola, non stimano un tasso"},
        "candidati_decontaminazione": {
            "record": len(drows), "esiti": dict(Counter(x["esito"] for x in drows)),
            "per_insieme_split_stato": dict(Counter(f"{x['insieme']}/{x['split']}/{x['stato']}" for x in drows)),
            "revisione_pixel": review_entry,
            "nota": "tutti i record con un candidato della decontaminazione esterna, con le immagini del pannello "
                    "citate nell'evidenza; la somiglianza dei pixel (decontam_review.py) serve solo a ordinare la "
                    "verifica visiva, non risolve un candidato"},
        "immagini_copiate": len(written), "immagini_pannello": len(panel_images), "errori_pannello": panel_errors[:20],
        "limiti": LIMITS, "inventario": inventory, "files": files, "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "report.json", report)
    print(json.dumps({"strati": design, "mirati": dict(counters), "candidati_decontaminazione": len(drows),
                      "immagini": len(written), "immagini_pannello": len(panel_images),
                      "errori_pannello": len(panel_errors), "seconds": report["seconds"]}, ensure_ascii=False))
    if panel_errors:
        sys.exit(f"{len(panel_errors)} immagini del pannello non ritrovate esattamente, per esempio {panel_errors[:3]}")


if __name__ == "__main__":
    main()
