#!/usr/bin/env python3
"""Pacchetto offline per la revisione visiva mirata di MAVIS-Geometry dopo C9 (revisioni/mavis-c9-campione-1; S2 §3).
Solo in un job SLURM CPU (slurm/mavis_review_package.sbatch). Manifest, scansione della decontaminazione, shard e zip
dello snapshot in sola lettura; nessuna chiamata a modelli e nessun giudizio compilato.

`costruisci` sceglie dal manifest (mavis-geometry-candidato-5, sha256 controllato) un campione diagnostico di 80
schede con le quote dell'utente del 2026-11-10 (PLAN): 1 controllo noto, 19 letture di valori, 30 classificazioni C9
recuperabili, 15 omesse, 15 altri compiti. Regole:
- record di train ammessi (il controllo noto è scelto per nome: eccezione documentata se non lo è); shape_type resta in
  quarantena e il manifest non cambia;
- una scheda per gruppo (quindi per record);
- in ogni strato le celle si percorrono a giro, profondità d3 per prima (per le letture alternando affollamento alto e
  basso: misure stampate del record sopra o non sopra la mediana della sua profondità); in ogni cella i gruppi in
  ordine di sha256({campione, seme 0, strato, gruppo}), nel gruppo l'unità con lo sha256({..., unita}) minore. Una cella
  esaurita si salta e si registra; una quota non raggiunta è un deficit registrato, mai compensato con un altro strato;
- gli strati si elaborano dal meno disponibile al più disponibile (a parità nell'ordine di PLAN), dopo il controllo;
- l'ordine delle schede è sha256({campione, seme 0, ordine: unita}): le categorie si mescolano.
Domande, opzioni e target si ricalcolano dai JSON originali degli shard (posizione e json_sha256 del manifest) col
codice di C9 e devono avere i digest del manifest; le immagini sono i caption.jpg originali di RuleBaseGeo.zip, coi
byte_sha256 e le dimensioni della scansione registrata nel manifest. Scrive nella cartella del pacchetto, che contiene
già l'interfaccia: dati/pacchetto.js (primo passaggio, senza target, annotazioni, esiti, categorie e motivi),
dati/soluzioni.js (secondo passaggio), immagini/, moduli/selezione.tsv e moduli/piano.json (inventario e piano, con i
deficit), pacchetto.json e SHA256SUMS.

`firme` riscrive SHA256SUMS (dopo una correzione dell'interfaccia; dati e immagini non cambiano).

`verifica` ricalcola selezione e schede dal manifest e dagli shard e controlla il pacchetto: corrispondenza
schede-selezione-manifest, condizioni di ogni strato scritte a parte, quote e deficit, record e gruppi distinti, nessun
collegamento, immagini e hash, separazione del primo passaggio, SHA256SUMS, riferimenti dell'interfaccia, nessuna rete.

`esegui-js` esegue dati, logica.js, app.js e test.js con Duktape (scripts/review_package.py), senza DOM e sul DOM
simulato; non è una prova in un browser.

    srun $RUN_PYTHON -u -B scripts/mavis_review_package.py costruisci --manifest DIR --manifest-sha256 HEX --out DIR
    srun $RUN_PYTHON -u -B scripts/mavis_review_package.py firme --pacchetto DIR
    srun $RUN_PYTHON -u -B scripts/mavis_review_package.py verifica --manifest DIR --manifest-sha256 HEX --pacchetto DIR \\
        --report FILE
    srun $RUN_PYTHON -u -B scripts/mavis_review_package.py esegui-js --pacchetto DIR --dom FILE --report FILE
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import mavis_c9 as mc  # noqa: E402  (shard_jsons: JSON degli shard con posizione e json_sha256 del manifest)
import review_package as rp  # noqa: E402  (Duktape, controllo lessicale, file window.X = ...)
from limite_vl import manifest as mf  # noqa: E402
from limite_vl import mavis_c9 as c9  # noqa: E402
from limite_vl.f1_mavis import describe, mavis_questions, mavis_truth  # noqa: E402
from limite_vl.f1_schema import latex_value  # noqa: E402
from limite_vl.provenance import code_identity, sha256_file, sha256_text  # noqa: E402

SCHEMA = "mavis-c9-revisione-pacchetto/1"
PACKAGE_ID = "mavis-c9-campione-1"
PACKAGE_VERSION = "1"
SEED = 0
CONTROL = ("d3-44919", "side:EH")  # l'unica rappresentazione corretta da candidato-5 (intero -> radice)
CELLS_VALUES = (("d3", "alto"), ("d2", "basso"), ("d1", "alto"), ("d0", "basso"),
                ("d3", "basso"), ("d2", "alto"), ("d1", "basso"), ("d0", "alto"))
CELLS_DEPTH = (("d3",), ("d2",), ("d1",), ("d0",))
PLAN = (  # (categoria, sottocategoria, quota): quote dell'utente del 2026-11-10
    ("controllo_noto", "radice_di_quadrato_perfetto", 1),
    ("lettura_valore", "decimale_arrotondato", 4),
    ("lettura_valore", "decimale_esatto", 4),
    ("lettura_valore", "radice", 4),
    ("lettura_valore", "intero", 4),
    ("lettura_valore", "angolo", 3),
    ("classificazione_recuperabile", "acute triangle", 6),
    ("classificazione_recuperabile", "right triangle", 6),
    ("classificazione_recuperabile", "obtuse triangle", 6),
    ("classificazione_recuperabile", "rectangle", 6),
    ("classificazione_recuperabile", "rhombus", 6),
    ("classificazione_omessa", "piu_opzioni_vere", 4),
    ("classificazione_omessa", "misure_insufficienti", 4),
    ("classificazione_omessa", "uguaglianza_apparente_arrotondamento", 4),
    ("classificazione_omessa", "segno_angolo_retto", 3),
    ("altro_compito", "describe", 3),
    ("altro_compito", "direction", 3),
    ("altro_compito", "count_points", 3),
    ("altro_compito", "removed_side", 3),
    ("altro_compito", "shared_side", 3),
)
TOTAL = 80
OTHER_TASKS = ("direction", "count_points", "removed_side", "shared_side")
OMITTED = {"piu_opzioni_vere": "piu_opzioni_vere", "richiede_segno_angolo_retto": "segno_angolo_retto"}
COMMON = ("record di train ammesso; una scheda per gruppo; gruppi in ordine di sha256({campione, seme 0, strato, "
          "gruppo}), nel gruppo l'unità con lo sha256 minore")
RULES = {
    "controllo_noto": "d3-44919 side:EH, scelto per nome: l'unica misura con la rappresentazione corretta da "
                      "candidato-5 (intero -> radice; esatto √841 = 29, scritto \\sqrt{841}); non indipendente; "
                      "eccezione documentata se il record non è di train ammesso",
    "lettura_valore": "misura con la rappresentazione della sottocategoria in `valori` (lati; per «angolo» ogni misura "
                      "di angolo) e la sua domanda di lettura del pool C9 (side_len o angle_val, campo elemento) col "
                      "compito ammesso; " + COMMON + "; celle a giro d3, d2, d1, d0 alternando affollamento alto e "
                      "basso (misure stampate del record sopra o non sopra la mediana della sua profondità fra i "
                      "record di train ammessi)",
    "classificazione_recuperabile": "figura con esito C9 recuperabile e risposta uguale alla sottocategoria "
                                    "(shape_type_c9), con la domanda C9 e le sue opzioni; il compito shape_type resta "
                                    "in quarantena; " + COMMON + "; celle a giro d3, d2, d1, d0",
    "classificazione_omessa": "figura con esito C9 omessa e motivo della sottocategoria: piu_opzioni_vere; "
                              "misure_insufficienti = non_deciso_dalle_misure_stampate senza uguaglianze apparenti; "
                              "uguaglianza_apparente_arrotondamento = non_deciso_dalle_misure_stampate con almeno due "
                              "lati della figura scritti uguali, uno dei quali decimale arrotondato; segno_angolo_retto "
                              "= richiede_segno_angolo_retto. Nessuna golden answer: richiesta diagnostica con "
                              "l'insieme fisso della famiglia; " + COMMON + "; celle a giro d3, d2, d1, d0",
    "altro_compito": "domanda del tipo della sottocategoria nel pool C9 (identica alla storica) col compito ammesso; "
                     "describe: richiesta e descrizione canonica del loader (f1_mavis.describe, sha256 della "
                     "descrizione del manifest); " + COMMON + "; celle a giro d3, d2, d1, d0",
}
CONSEGNE = {
    "valore": "Riporta il valore così come è scritto nella figura, carattere per carattere, senza trasformarlo. Se il "
              "valore non si legge o non c'è, segna «non determinabile».",
    "opzioni": "Rispondi con una delle opzioni della domanda. Se dall'immagine non si decide, segna «non "
               "determinabile» e spiega nelle note.",
    "diagnostica": "Richiesta diagnostica, non è una domanda del training. Indica tutte le opzioni vere per la figura "
                   "nominata: possono essere una, più di una o nessuna. Se l'immagine non basta, segna «non "
                   "determinabile» e scrivi nelle note che cosa manca.",
    "describe": "Descrivi la figura come la vedi: punti, figure, lati e angoli con le misure scritte. Bastano poche "
                "righe: nel secondo passaggio la confronterai con la descrizione candidata.",
    "domanda": "Rispondi alla domanda guardando l'immagine. Se dall'immagine non si può rispondere, segna «non "
               "determinabile».",
}
NOUN_IT = {"triangle": "il triangolo", "quadrilateral": "il quadrilatero"}
# chiavi che non devono comparire nei dati del primo passaggio (dati/pacchetto.js)
HIDDEN_KEYS = {"target", "risposta", "answer", "categoria", "sottocategoria", "strato", "cella", "regola", "motivo",
               "motivo_selezione", "esito", "c9", "valore", "valori", "valori_del_record", "annotazioni",
               "didascalia", "caption", "diagram_generation", "fatti", "confutazioni", "vere", "non_confutate",
               "ignote", "rappresentazione", "esatto", "visualizzato_previsto", "risposta_lettura", "storica",
               "verita_annotazioni", "split", "stato", "stato_compito", "compito", "qa_sha256", "tipo_dichiarato",
               "elemento", "figura", "rango", "affollamento"}
INTERFACE = ("index.html", "app.js", "logica.js", "stile.css", "verifica.html", "test.js", "LEGGIMI.md",
             "risultati/LEGGIMI.md")
NOT_SIGNED = {"SHA256SUMS", "verifica_pacchetto.json", "verifica_js.json"}
TSV_FIELDS = ("scheda", "posizione", "record", "record_id", "gruppo", "split", "stato_record", "profondita",
              "misure_stampate", "affollamento", "categoria", "sottocategoria", "cella", "posizione_nella_cella",
              "richiesta", "compito", "stato_compito", "elemento", "figura", "domanda", "opzioni", "qa_sha256",
              "target", "rango_ordine", "rango_gruppo", "rango_unita", "motivo_selezione", "eccezione", "immagine",
              "immagine_sha256", "larghezza", "altezza", "membro_zip", "json_sha256")


class Failed(Exception):
    pass


def rank(**fields) -> str:
    return mf.sha256_text(mf.canonical({"campione": PACKAGE_ID, "seme": SEED, **fields}))


def stratum_id(cat: str, sub: str) -> str:
    return f"{cat}/{sub}"


def task_state(r: dict, name: str) -> tuple[str | None, int | None]:
    for t in r.get("compiti") or []:
        if t["compito"] == name:
            return t["stato"], t.get("domande")
    return None, None


def task_admitted(r: dict, name: str) -> bool:
    st, n = task_state(r, name)
    return st == "ammesso" and (name == "describe" or (n or 0) > 0)


def record_admitted(r: dict) -> bool:
    return r["split"] == "train" and r["stato"] == "ammesso"


def apparent_rounding(sides: dict[str, str]) -> bool:
    """Due lati della figura con la stessa scrittura prevista, almeno uno decimale arrotondato: un'uguaglianza che il
    disegno sembra mostrare ma le misure stampate non dimostrano."""
    items = [(latex_value(x), c9.representation(x)) for x in sides.values()]
    return any(a[0] == b[0] and "decimale_arrotondato" in (a[1], b[1])
               for i, a in enumerate(items) for b in items[i + 1:])


def omitted_class(entry: dict) -> str | None:
    if entry.get("esito") != "omessa":
        return None
    motive = entry.get("motivo")
    if motive in OMITTED:
        return OMITTED[motive]
    if motive == "non_deciso_dalle_misure_stampate":
        return ("uguaglianza_apparente_arrotondamento" if apparent_rounding(entry["fatti"]["lati"])
                else "misure_insufficienti")
    return None


def units_of(r: dict) -> list[tuple[str, str, dict]]:
    """(strato, unità, specifica) delle unità candidate di un record di train ammesso. Il controllo noto è a parte."""
    key, out = r["record"], []
    if not record_admitted(r):
        return out
    qa = r.get("qa") or []
    asked = {q["elemento"]: q["tipo"] for q in qa if q.get("elemento")}
    for v in r.get("valori") or []:
        elem = v["elemento"]
        if (key, elem) == CONTROL or elem not in asked or not task_admitted(r, asked[elem]):
            continue
        sub = "angolo" if elem.startswith("angle:") else v["rappresentazione"]
        if sub in ("decimale_arrotondato", "decimale_esatto", "radice", "intero", "angolo"):
            out.append((stratum_id("lettura_valore", sub), f"{key}|valore|{elem}", {"tipo": "valore", "elemento": elem}))
    for i, f in enumerate(r.get("shape_type_c9") or []):
        spec = {"tipo": "forma", "indice": i, "figura": f["figura"]}
        if f["esito"] == "recuperabile":
            out.append((stratum_id("classificazione_recuperabile", f["risposta"]),
                        f"{key}|forma|{i}|{f['figura']}", spec))
        elif (sub := omitted_class(f)) is not None:
            out.append((stratum_id("classificazione_omessa", sub), f"{key}|forma|{i}|{f['figura']}", spec))
    if task_admitted(r, "describe") and r.get("descrizione_sha256"):
        out.append((stratum_id("altro_compito", "describe"), f"{key}|describe", {"tipo": "describe"}))
    seen = Counter()
    for q in qa:
        t = q["tipo"]
        if t in OTHER_TASKS:
            k = seen[t]
            seen[t] += 1
            if task_admitted(r, t):
                out.append((stratum_id("altro_compito", t), f"{key}|{t}|{k}", {"tipo": "compito", "compito": t, "k": k}))
    return out


def cell_of(stratum: str, depth: str, crowd: str) -> tuple:
    return (depth, crowd) if stratum.startswith("lettura_valore/") else (depth,)


def cells_for(stratum: str) -> tuple:
    return CELLS_VALUES if stratum.startswith("lettura_valore/") else CELLS_DEPTH


# ---------------------------------------------------------------- lettura del manifest

def stream_records(folder: Path, manifest_sha: str):
    """(meta, sha di manifest.json, generatore delle righe di records.jsonl.gz). Lo sha256 di records.jsonl.gz si
    controlla durante la lettura: alla fine del generatore, Failed se diverso da quello di manifest.json."""
    path = folder / "manifest.json"
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != manifest_sha:
        raise Failed(f"{path}: sha256 {sha}, atteso {manifest_sha}")
    meta = json.loads(raw)
    if meta.get("sorgente") != "mavis-geometry" or meta.get("schema") != mf.SCHEMA:
        raise Failed(f"{path}: non è un manifest di mavis-geometry ({meta.get('sorgente')}, {meta.get('schema')})")

    def lines():
        h, n = hashlib.sha256(), 0
        with open(folder / meta["records"]["file"], "rb") as fh:
            raw_gz = HashingReader(fh, h)
            with gzip.GzipFile(fileobj=raw_gz) as gz:
                for line in io.TextIOWrapper(gz, encoding="utf-8"):
                    n += 1
                    yield json.loads(line)
            while raw_gz.read(1 << 20):
                pass
        if h.hexdigest() != meta["records"]["sha256"] or n != meta["records"]["righe"]:
            raise Failed(f"records.jsonl.gz: sha256 o righe diversi da manifest.json ({n} righe)")
    return meta, sha, lines()


class HashingReader(io.RawIOBase):
    def __init__(self, fh, h):
        self.fh, self.h = fh, h

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self.fh.read(len(b))
        self.h.update(data)
        b[:len(data)] = data
        return len(data)


def scan_inputs(folder: Path, manifest_sha: str) -> dict:
    """Una lettura del manifest: per record le informazioni della selezione, le unità candidate, le posizioni per gli
    shard e le misure stampate; il controllo noto. Nessuna decisione."""
    meta, msha, it = stream_records(folder, manifest_sha)
    recs, units, positions = {}, [], defaultdict(dict)
    crowd_by_depth, control, counts = defaultdict(list), None, Counter()
    for r in it:
        key = r["record"]
        positions[r["shard"]][key] = (r["indice"], r["json_sha256"])
        counts[f"{r['split']}/{r['stato']}"] += 1
        n_values = len(r.get("valori") or [])
        if record_admitted(r):
            crowd_by_depth[r["famiglia"]].append(n_values)
        if key == CONTROL[0]:
            control = r
        mine = units_of(r)
        if mine or key == CONTROL[0]:
            recs[key] = {k: r[k] for k in ("record", "record_id", "gruppo", "split", "stato", "famiglia", "shard",
                                           "indice", "json_sha256", "descrizione_sha256", "motivo", "verifiche",
                                           "immagini") if k in r}
            recs[key].update(misure_stampate=n_values, qa=r.get("qa") or [], compiti=r.get("compiti") or [],
                             valori=r.get("valori") or [], shape_type_c9=r.get("shape_type_c9") or [])
        units += [(s, u, key, spec) for s, u, spec in mine]
    if control is None:
        raise Failed(f"controllo noto {CONTROL[0]} assente dal manifest")
    medians = {d: statistics.median(v) for d, v in sorted(crowd_by_depth.items())}
    return {"meta": meta, "manifest_sha256": msha, "records": recs, "units": units, "positions": positions,
            "medians": medians, "counts": dict(counts), "control": control}


# ---------------------------------------------------------------- selezione

def crowd_of(rec: dict, medians: dict) -> str:
    return "alto" if rec["misure_stampate"] > medians[rec["famiglia"]] else "basso"


def candidates_by_stratum(units, recs, medians) -> dict:
    """{strato: {cella: {gruppo: (rango dell'unità, unità, record, specifica)}}} con la migliore unità per gruppo e
    cella."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for s, u, key, spec in units:
        rec = recs[key]
        cell = cell_of(s, rec["famiglia"], crowd_of(rec, medians))
        ur = rank(strato=s, unita=u)
        g = rec["gruppo"]
        cur = out[s][cell].get(g)
        if cur is None or ur < cur[0]:
            out[s][cell][g] = (ur, u, key, spec)
    return out


def select(plan, cands: dict, reserved: set[str]) -> tuple[list[dict], list[dict], list[str]]:
    """Scelte, rapporto per strato e ordine di elaborazione. `cands` come candidates_by_stratum; `reserved`: gruppi già
    usati (il controllo). Celle a giro (cells_for), gruppi per rango nella cella, un gruppo una volta sola."""
    used = set(reserved)
    avail = {stratum_id(c, s): len({g for cell in cands.get(stratum_id(c, s), {}).values() for g in cell})
             for c, s, _ in plan if c != "controllo_noto"}
    order = sorted(avail, key=lambda s: (avail[s], [stratum_id(c, x) for c, x, _ in plan].index(s)))
    quota = {stratum_id(c, s): q for c, s, q in plan}
    picks, report = [], []
    for s in order:
        cells = cells_for(s)
        by_cell = {cell: sorted((rank(strato=s, gruppo=g), g, *v) for g, v in cands.get(s, {}).get(cell, {}).items())
                   for cell in cells}
        ptr = {cell: 0 for cell in cells}
        exhausted, log, taken, i = [], [], Counter(), 0
        conflicts = 0
        while sum(taken.values()) < quota[s] and len(exhausted) < len(cells):
            cell = cells[i % len(cells)]
            i += 1
            if cell in exhausted:
                continue
            lst, p = by_cell[cell], ptr[cell]
            while p < len(lst) and lst[p][1] in used:
                p += 1
                conflicts += 1
            if p >= len(lst):
                exhausted.append(cell)
                log.append(f"cella {'/'.join(cell)} esaurita dopo {taken[cell]} scelte")
                continue
            grank, g, urank, unit, key, spec = lst[p]
            ptr[cell] = p + 1
            used.add(g)
            taken[cell] += 1
            picks.append({"strato": s, "cella": cell, "gruppo": g, "rango_gruppo": grank, "rango_unita": urank,
                          "unita": unit, "record": key, "spec": spec, "posizione_nella_cella": p + 1})
        got = sum(taken.values())
        report.append({"strato": s, "quota": quota[s], "scelte": got, "deficit": quota[s] - got,
                       "gruppi_disponibili": avail[s],
                       "gruppi_per_cella": {"/".join(c): len(by_cell[c]) for c in cells},
                       "scelte_per_cella": {"/".join(c): taken[c] for c in cells if taken[c]},
                       "gruppi_saltati_perche_gia_usati": conflicts, "celle": log})
    return picks, report, order


# ---------------------------------------------------------------- schede dai JSON originali

def read_jsons(shards: dict[str, dict], wanted: set[str]) -> tuple[dict[str, bytes], dict]:
    """JSON dei record scelti, letti in sequenza con scripts/mavis_c9.shard_jsons: ogni JSON degli shard che li
    contengono ha la posizione e il json_sha256 del suo record del manifest, immagini saltate."""
    out, stats = {}, Counter()
    for rel, positions in sorted(shards.items()):
        keep = set(positions) & wanted
        if not keep:
            continue
        for key, data in mc.shard_jsons(mf.shard_path(rel), positions, stats):
            if key in keep:
                out[key] = data
    if set(out) != wanted:
        raise Failed(f"JSON non trovati: {sorted(wanted - set(out))[:5]}")
    return out, dict(stats)


def facts_doc(f: c9.ShapeFacts) -> dict:
    return mc.facts_doc(f)


def outcome_entry(o: dict, f: c9.ShapeFacts) -> dict:
    """Voce di shape_type_c9 come la scrive la costruzione (scripts/mavis_c9.c9_shard)."""
    entry = {k: o[k] for k in ("figura", "tipo_dichiarato", "esito", "motivo", "opzioni", "risposta", "confutazioni",
                               "non_confutate", "vere", "ignote", "dettaglio") if k in o}
    entry["fatti"] = facts_doc(f)
    return entry


def card_of(pick: dict, rec: dict, data: bytes, medians: dict) -> tuple[dict, dict]:
    """(dati del primo passaggio senza identificativo né immagine, soluzione) di una scelta, ricalcolati dal JSON
    originale col codice di C9 e confrontati coi digest del manifest."""
    key, spec = rec["record"], pick["spec"]
    meta = json.loads(data)
    if meta.get("id") != key:
        raise Failed(f"{key}: id del JSON {meta.get('id')!r}")
    te = meta["text_en"]
    gen = te["diagram_generation"]
    truth, why = mavis_truth(te["caption"], gen)
    if truth is None:
        raise Failed(f"{key}: verità non disponibile ({why})")
    pool, outcomes, values = c9.questions(key, truth, gen)
    got = [{"tipo": q["type"], "sha256": mf.qa_item_sha256(q), **({"elemento": q["elemento"]} if "elemento" in q
                                                                    else {})} for q in pool]
    if got != rec["qa"]:
        raise Failed(f"{key}: domande ricalcolate diverse da quelle del manifest")
    if [{"elemento": e, **v} for e, v in values.items()] != rec["valori"]:
        raise Failed(f"{key}: valori ricalcolati diversi da quelli del manifest")
    facts = c9.shape_facts(truth, gen)
    entries = [outcome_entry(o, f) for o, f in zip(outcomes, facts)]
    if entries != rec["shape_type_c9"]:
        raise Failed(f"{key}: esiti di shape_type ricalcolati diversi da quelli del manifest")
    cat, sub = pick["strato"].split("/", 1)
    first = {"richiesta": "qa", "controllo_noto": cat == "controllo_noto", "opzioni": []}
    sol = {"strato": pick["strato"], "categoria": cat, "sottocategoria": sub, "cella": "/".join(pick["cella"]),
           "regola": RULES[cat], "qa_candidata": True, "compito": None, "stato_compito": None, "qa_sha256": None,
           "elemento": None, "figura": None, "target": None, "valore": None, "c9": None, "verita_annotazioni": None,
           "confutazioni_stampate": None, "confutazioni_con_segno_retto": None, "fatti": None, "storica": None,
           "annotazioni": {"didascalia": te["caption"], "diagram_generation": gen},
           "valori_del_record": rec["valori"]}
    if spec["tipo"] == "valore":
        elem = spec["elemento"]
        (q,) = [q for q in pool if q.get("elemento") == elem]
        first.update(domanda=q["q"], consegna=CONSEGNE["valore"])
        st, _ = task_state(rec, q["type"])
        sol.update(compito=q["type"], stato_compito=st, qa_sha256=mf.qa_item_sha256(q), elemento=elem,
                   target=q["answer"], valore={"elemento": elem, **values[elem]})
        if values[elem]["risposta_lettura"] != q["answer"]:
            raise Failed(f"{key} {elem}: target diverso dalla risposta di lettura")
    elif spec["tipo"] == "forma":
        i = spec["indice"]
        o, f = outcomes[i], facts[i]
        if o["figura"] != spec["figura"]:
            raise Failed(f"{key}: figura {i} {o['figura']!r} invece di {spec['figura']!r}")
        hist = [q for q in mavis_questions(key, truth) if q["type"] == "shape_type"][i]
        if hist["q"] != f"What type of shape is {o['figura']}?":
            raise Failed(f"{key}: domanda storica non allineata alla figura {o['figura']}")
        truth_classes, contradiction = c9.exact_classes(f)
        st, _ = task_state(rec, "shape_type")
        sol.update(compito="shape_type", stato_compito=st, figura=o["figura"], c9=entries[i],
                   verita_annotazioni=truth_classes, fatti={**facts_doc(f), "tipo_dichiarato": f.kind},
                   storica={"domanda": hist["q"], "risposta": hist["answer"],
                            "nota": "domanda aperta storica, in quarantena: più risposte vere possibili"},
                   confutazioni_stampate=c9.printed_refutations(f))
        if contradiction:
            sol["contraddizione_annotazioni"] = contradiction
        if f.kind == "right triangle":
            sol["confutazioni_con_segno_retto"] = c9.printed_refutations(f, right_mark=True)
        if o["esito"] == "recuperabile":
            q = o["domanda"]
            first.update(domanda=q["q"], opzioni=list(o["opzioni"]), consegna=CONSEGNE["opzioni"])
            sol.update(target=o["risposta"], qa_sha256=mf.qa_item_sha256(q))
        else:
            options = c9.key_shuffle(key, o["figura"], c9.OPTIONS[f.family])
            first.update(richiesta="diagnostica", opzioni=options, consegna=CONSEGNE["diagnostica"],
                         domanda=f"Quali di queste opzioni descrivono {NOUN_IT[f.family]} {o['figura']}? Opzioni: "
                                 f"{', '.join(options)}.")
            sol.update(qa_candidata=False, target=None,
                       nota_target="nessuna golden answer: la figura è omessa da C9; le verità secondo le "
                                   "annotazioni sono materiale di confronto, non gold")
    elif spec["tipo"] == "describe":
        prompt, desc = describe(key, truth)
        if sha256_text(desc) != rec["descrizione_sha256"]:
            raise Failed(f"{key}: descrizione diversa dal digest del manifest")
        st, _ = task_state(rec, "describe")
        first.update(domanda=prompt, consegna=CONSEGNE["describe"])
        sol.update(compito="describe", stato_compito=st, target=desc,
                   nota_target="descrizione canonica del loader (f1_schema.render_geometry), sha256 "
                               f"{rec['descrizione_sha256']}")
    else:
        t, k = spec["compito"], spec["k"]
        q = [q for q in pool if q["type"] == t][k]
        st, _ = task_state(rec, t)
        first.update(domanda=q["q"], consegna=CONSEGNE["domanda"])
        sol.update(compito=t, stato_compito=st, qa_sha256=mf.qa_item_sha256(q), target=q["answer"])
    sol["record"] = {"record": key, "record_id": rec["record_id"], "gruppo": rec["gruppo"], "split": rec["split"],
                     "stato": rec["stato"], "motivo": rec.get("motivo"), "profondita": rec["famiglia"],
                     "misure_stampate": rec["misure_stampate"], "affollamento": crowd_of(rec, medians),
                     "mediana_della_profondita": medians[rec["famiglia"]], "json_sha256": rec["json_sha256"],
                     "verifiche": rec.get("verifiche"), "shard": rec["shard"], "indice": rec["indice"]}
    sol["selezione"] = {"unita": pick["unita"], "rango_gruppo": pick["rango_gruppo"], "rango_unita": pick["rango_unita"],
                        "posizione_nella_cella": pick["posizione_nella_cella"]}
    if cat == "controllo_noto":
        sol["nota_controllo"] = ("controllo noto, non indipendente: il caso è descritto nei rapporti di C9 (verifica "
                                 "corretta di candidato-4, correzione di candidato-5)")
    sol["motivo_selezione"] = (RULES[cat] if cat == "controllo_noto" else
                               f"{pick['strato']}, cella {'/'.join(pick['cella'])}: {pick['posizione_nella_cella']}° "
                               f"gruppo per rango nella cella")
    return first, sol


# ---------------------------------------------------------------- immagini originali

def scan_rows(meta: dict, keys: set[str]) -> tuple[dict, dict]:
    entry = meta["decontaminazione"]["scansione"]["records"]
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise Failed(f"{path}: sha256 diverso da quello registrato nel manifest")
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["record"] in keys:
                if row["record"] in out:
                    raise Failed(f"scansione: {row['record']} ripetuto")
                out[row["record"]] = {"json_sha256": row["json_sha256"], **row["originale"]}
    if set(out) != keys:
        raise Failed(f"scansione: record mancanti {sorted(keys - set(out))[:5]}")
    return out, {"percorso": um_rel(path), "sha256": entry["sha256"], "byte": entry["bytes"]}


def um_rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def zip_info(meta: dict) -> dict:
    snap = meta["identita"]["snapshot"]["zip"]
    path = Path(snap["percorso"])
    blob = path.resolve().name
    if not path.is_file() or blob != snap["sha256_atteso"] or path.stat().st_size != snap["byte"]:
        raise Failed(f"zip dello snapshot assente o diverso: {path} -> {blob}")
    return {"percorso": str(path), "blob": blob, "sha256_atteso": snap["sha256_atteso"], "byte": snap["byte"],
            "nota": "blob della cache Hugging Face indirizzato dal contenuto (nome = sha256 atteso); sha256 dell'intero "
                    "zip non ricalcolato qui (calcolato dalla scansione, job 30652); ogni immagine ha i byte_sha256 "
                    "della scansione"}


def original_images(zpath: str, scan: dict, jsons: dict[str, bytes]) -> dict[str, dict]:
    """{record: {byte, sha256, larghezza, altezza, membro}} dei caption.jpg originali, con i controlli: membro nella
    cartella del JSON, sha256 dei byte e dimensioni uguali alla scansione e alle dimensioni originali del JSON."""
    from PIL import Image

    out = {}
    with zipfile.ZipFile(zpath) as zf:
        for key in sorted(scan):
            s, meta = scan[key], json.loads(jsons[key])
            member = s.get("membro")
            if s.get("lettura") != "letta" or not member or not member.endswith("caption.jpg"):
                raise Failed(f"{key}: originale non letto dalla scansione ({s.get('lettura')}, {member})")
            if PurePosixPath(member).parent.as_posix() != meta.get("zip_dir"):
                raise Failed(f"{key}: membro {member} fuori dalla cartella {meta.get('zip_dir')} del JSON")
            if s["json_sha256"] != hashlib.sha256(jsons[key]).hexdigest():
                raise Failed(f"{key}: JSON della scansione diverso da quello del manifest")
            data = zf.read(member)  # zipfile controlla il CRC-32
            sha = hashlib.sha256(data).hexdigest()
            if sha != s["byte_sha256"]:
                raise Failed(f"{key}: sha256 dell'originale {sha} diverso dalla scansione {s['byte_sha256']}")
            with Image.open(io.BytesIO(data)) as im:
                size, fmt = list(im.size), im.format
            if size != list(s["dimensioni"]) or size != [meta.get("orig_width"), meta.get("orig_height")]:
                raise Failed(f"{key}: dimensioni {size}, scansione {s['dimensioni']}, JSON "
                             f"{[meta.get('orig_width'), meta.get('orig_height')]}")
            if fmt != "JPEG":
                raise Failed(f"{key}: formato {fmt}")
            out[key] = {"byte": data, "sha256": sha, "larghezza": size[0], "altezza": size[1], "membro": member}
    return out


# ---------------------------------------------------------------- costruzione

def derive(args) -> dict:
    """Selezione e schede dal manifest e dagli shard (stessa funzione in costruzione e verifica)."""
    t0 = time.time()
    inp = scan_inputs(args.manifest, args.manifest_sha256)
    recs, medians = inp["records"], inp["medians"]
    control = inp["control"]
    ctrl_values = {v["elemento"]: v for v in control.get("valori") or []}
    if CONTROL[1] not in ctrl_values or ctrl_values[CONTROL[1]]["rappresentazione"] != "radice":
        raise Failed(f"controllo noto {CONTROL}: misura assente o non registrata come radice")
    cands = candidates_by_stratum(inp["units"], recs, medians)
    ctrl = {"strato": stratum_id(*PLAN[0][:2]), "cella": (control["famiglia"],), "gruppo": control["gruppo"],
            "rango_gruppo": rank(strato=stratum_id(*PLAN[0][:2]), gruppo=control["gruppo"]),
            "rango_unita": rank(strato=stratum_id(*PLAN[0][:2]), unita=f"{CONTROL[0]}|valore|{CONTROL[1]}"),
            "unita": f"{CONTROL[0]}|valore|{CONTROL[1]}", "record": CONTROL[0],
            "spec": {"tipo": "valore", "elemento": CONTROL[1]}, "posizione_nella_cella": 1}
    picks, report, order = select(PLAN, cands, {control["gruppo"]})
    picks = [ctrl] + picks
    exception = None if record_admitted(control) else (
        f"record {control['split']}/{control['stato']}: scelto per nome come controllo noto; stato e split invariati")
    report.insert(0, {"strato": ctrl["strato"], "quota": 1, "scelte": 1, "deficit": 0, "gruppi_disponibili": 1,
                      "celle": [], "eccezione": exception})
    t1 = time.time()
    wanted = {p["record"] for p in picks}
    if len(wanted) != len(picks) or len({p["gruppo"] for p in picks}) != len(picks):
        raise Failed("record o gruppi ripetuti nella selezione")
    jsons, read_stats = read_jsons(inp["positions"], wanted)
    cards = []
    for p in picks:
        first, sol = card_of(p, recs[p["record"]], jsons[p["record"]], medians)
        cards.append({"pick": p, "first": first, "sol": sol,
                      "ordine": rank(ordine=p["unita"]), "eccezione": exception if p is ctrl else None})
    cards.sort(key=lambda c: c["ordine"])
    meta = inp["meta"]
    scan, scan_file = scan_rows(meta, wanted)
    zinfo = zip_info(meta)
    images = original_images(zinfo["percorso"], scan, jsons)
    for n, c in enumerate(cards, 1):
        key = c["pick"]["record"]
        im = images[key]
        c["id"] = f"M{n:02d}"
        c["posizione"] = n
        c["immagine"] = {"percorso": f"immagini/{key}.jpg", "sha256": im["sha256"], "larghezza": im["larghezza"],
                         "altezza": im["altezza"], "membro": im["membro"]}
    shards = [{"percorso": s["percorso"], "sha256": s["sha256"]} for s in meta["shards"]]
    return {"cards": cards, "images": images, "report": report, "order": order, "medians": medians,
            "counts": inp["counts"], "manifest": {"percorso": um_rel(args.manifest), "nome": meta["nome"],
                                                  "versione": meta["versione"],
                                                  "manifest_sha256": inp["manifest_sha256"],
                                                  "records_sha256": meta["records"]["sha256"]},
            "scansione": scan_file, "zip": zinfo, "shard": shards, "eccezione_controllo": exception,
            "lettura_shard": read_stats,
            "secondi": {"manifest_e_selezione": round(t1 - t0, 1), "schede_e_immagini": round(time.time() - t1, 1)}}


def package_objects(d: dict) -> tuple[dict, dict]:
    """(PACCHETTO, SOLUZIONI) con l'impronta: digest canonico del pacchetto senza impronta e delle soluzioni."""
    schede, sol = [], {}
    for c in d["cards"]:
        f, im = c["first"], c["immagine"]
        schede.append({"id": c["id"], "posizione": c["posizione"], "record": c["pick"]["record"],
                       "gruppo": c["pick"]["gruppo"], "immagine": im["percorso"], "immagine_sha256": im["sha256"],
                       "larghezza": im["larghezza"], "altezza": im["altezza"], "richiesta": f["richiesta"],
                       "controllo_noto": f["controllo_noto"], "domanda": f["domanda"], "opzioni": f["opzioni"],
                       "consegna": f["consegna"]})
        sol[c["id"]] = {**c["sol"], "immagine": {"sha256": im["sha256"], "membro_zip": im["membro"]}}
    pkg = {"schema": SCHEMA, "id": PACKAGE_ID, "versione": PACKAGE_VERSION, "seme": SEED,
           "sorgente": {"manifest": d["manifest"]["percorso"], "versione": d["manifest"]["versione"],
                        "manifest_sha256": d["manifest"]["manifest_sha256"],
                        "records_sha256": d["manifest"]["records_sha256"]},
           "schede": schede}
    digests = {"soluzioni": rp.canonical_digest(sol)}
    pkg["impronta"] = rp.canonical_digest({"pacchetto": pkg, **digests})
    pkg["impronte_dati"] = digests
    return pkg, sol


def tsv_rows(d: dict) -> list[dict]:
    rows = []
    for c in d["cards"]:
        p, f, s, im = c["pick"], c["first"], c["sol"], c["immagine"]
        rec = s["record"]
        rows.append({"scheda": c["id"], "posizione": c["posizione"], "record": p["record"], "record_id": rec["record_id"],
                     "gruppo": p["gruppo"], "split": rec["split"], "stato_record": rec["stato"],
                     "profondita": rec["profondita"], "misure_stampate": rec["misure_stampate"],
                     "affollamento": rec["affollamento"], "categoria": s["categoria"],
                     "sottocategoria": s["sottocategoria"], "cella": s["cella"],
                     "posizione_nella_cella": p["posizione_nella_cella"], "richiesta": f["richiesta"],
                     "compito": s["compito"], "stato_compito": s["stato_compito"], "elemento": s["elemento"] or "",
                     "figura": s["figura"] or "", "domanda": f["domanda"], "opzioni": " / ".join(f["opzioni"]),
                     "qa_sha256": s["qa_sha256"] or "", "target": "" if s["target"] is None else s["target"],
                     "rango_ordine": c["ordine"], "rango_gruppo": p["rango_gruppo"], "rango_unita": p["rango_unita"],
                     "motivo_selezione": s["motivo_selezione"], "eccezione": c["eccezione"] or "",
                     "immagine": im["percorso"], "immagine_sha256": im["sha256"], "larghezza": im["larghezza"],
                     "altezza": im["altezza"], "membro_zip": im["membro"], "json_sha256": rec["json_sha256"]})
    return rows


def tsv_text(rows: list[dict]) -> str:
    def cell(x) -> str:
        return str(x).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
    return "".join("\t".join(cell(r[k]) for k in TSV_FIELDS) + "\n" for r in [dict(zip(TSV_FIELDS, TSV_FIELDS))] + rows)


def plan_doc(d: dict) -> dict:
    deficits = [r for r in d["report"] if r["deficit"]]
    return {"schema": SCHEMA + "-piano", "campione": PACKAGE_ID, "versione": PACKAGE_VERSION, "seme": SEED,
            "natura": "campione diagnostico, non rappresentativo della frequenza naturale degli errori: nessun "
                      "intervallo di confidenza, nessuna stima di tasso, nessuna certificazione (S4); gli esiti non "
                      "ammettono record, non promuovono a gold e non attivano il training",
            "manifest": d["manifest"], "quote": [{"categoria": c, "sottocategoria": s, "quota": q} for c, s, q in PLAN],
            "totale_previsto": TOTAL, "totale_scelto": len(d["cards"]),
            "deficit": [{"strato": r["strato"], "mancano": r["deficit"], "gruppi_disponibili": r["gruppi_disponibili"],
                         "motivo": "gruppi disponibili esauriti in tutte le celle dello strato (record di train "
                                   "ammessi, gruppi non già usati): non compensato con altri strati"}
                        for r in deficits],
            "ordine_di_elaborazione": [PLAN[0][0] + "/" + PLAN[0][1]] + d["order"],
            "regola_ordine": "controllo per primo, poi gli strati dal meno disponibile (gruppi) al più disponibile, a "
                             "parità nell'ordine delle quote",
            "strati": d["report"], "regole": RULES,
            "celle": {"lettura_valore": ["/".join(c) for c in CELLS_VALUES],
                      "altri_strati": ["/".join(c) for c in CELLS_DEPTH]},
            "affollamento": {"definizione": "misure stampate del record (voci di `valori`) sopra la mediana della sua "
                                            "profondità fra i record di train ammessi = alto; altrimenti basso. È un "
                                            "indicatore della densità di etichette, non una misura della leggibilità",
                             "mediane": d["medians"]},
            "ordine_delle_schede": "sha256({campione, seme 0, ordine: unita}) crescente; identificativi M01-M80 nell'ordine",
            "ranghi": "sha256 del JSON canonico {campione, seme, strato, gruppo} e {campione, seme, strato, unita}",
            "controllo_noto": {"record": CONTROL[0], "elemento": CONTROL[1], "eccezione": d["eccezione_controllo"]},
            "record_del_manifest_per_split_e_stato": d["counts"],
            "ingressi": {"scansione": d["scansione"], "zip": d["zip"], "shard": d["shard"]}}


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        fh.write(data)


def signed_files(pkg_dir: Path) -> list[str]:
    out = []
    for p in sorted(pkg_dir.rglob("*")):
        rel = p.relative_to(pkg_dir).as_posix()
        if p.is_file() and rel not in NOT_SIGNED and not (rel.startswith("risultati/") and rel != "risultati/LEGGIMI.md"):
            out.append(rel)
    return out


def write_sums(pkg_dir: Path) -> int:
    names = signed_files(pkg_dir)
    text = "".join(f"{sha256_file(pkg_dir / n)}  {n}\n" for n in names)
    (pkg_dir / "SHA256SUMS").write_text(text, encoding="utf-8")
    return len(names)


def cmd_build(args) -> None:
    out = args.out
    missing = [n for n in INTERFACE if not (out / n).is_file()]
    if missing:
        raise Failed(f"interfaccia incompleta in {out}: {missing}")
    for n in ("dati", "immagini", "moduli", "pacchetto.json"):
        if (out / n).exists():
            raise Failed(f"{out / n} esiste già: il pacchetto si costruisce una volta sola")
    d = derive(args)
    pkg, sol = package_objects(d)
    plan = plan_doc(d)
    files = {"dati/pacchetto.js": rp.js_text("PACCHETTO", pkg).encode("utf-8"),
             "dati/soluzioni.js": rp.js_text("SOLUZIONI", sol).encode("utf-8"),
             "moduli/selezione.tsv": tsv_text(tsv_rows(d)).encode("utf-8"),
             "moduli/piano.json": (json.dumps(plan, indent=1, ensure_ascii=False) + "\n").encode("utf-8")}
    by_cat = Counter(c["sol"]["categoria"] for c in d["cards"])
    meta = {"schema": SCHEMA, "id": PACKAGE_ID, "versione": PACKAGE_VERSION, "impronta": pkg["impronta"],
            "impronte_dati": pkg["impronte_dati"], "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
            "ingressi": {"manifest": d["manifest"], "scansione": d["scansione"], "zip": d["zip"], "shard": d["shard"],
                         "lettura_shard": d["lettura_shard"]},
            "conteggi": {"schede": len(d["cards"]), "immagini": len(d["images"]), "per_categoria": dict(by_cat),
                         "deficit": sum(r["deficit"] for r in d["report"])},
            "file": {n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in files.items()},
            "secondi": d["secondi"],
            "nota": "campione diagnostico per la revisione visiva; nessun giudizio compilato, nessun esito automatico "
                    "promosso a corretto o gold"}
    files["pacchetto.json"] = (json.dumps(meta, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    files.update({f"immagini/{key}.jpg": im["byte"] for key, im in sorted(d["images"].items())})
    for name, data in files.items():  # tutto calcolato prima: nessun pacchetto a metà se un passo fallisce
        write_new(out / name, data)
    n = write_sums(out)
    print(json.dumps({"schede": len(d["cards"]), "per_categoria": dict(by_cat), "deficit": plan["deficit"],
                      "impronta": pkg["impronta"], "file_firmati": n, "secondi": d["secondi"]}, ensure_ascii=False))


# ---------------------------------------------------------------- verifica

def condition_ok(cat: str, sub: str, rec: dict, s: dict) -> bool:
    """Condizione dello strato sul record del manifest, scritta a parte dalla selezione (units_of)."""
    admitted = rec["split"] == "train" and rec["stato"] == "ammesso"
    if cat == "controllo_noto":
        v = {x["elemento"]: x for x in rec["valori"]}.get(CONTROL[1])
        return rec["record"] == CONTROL[0] and s["elemento"] == CONTROL[1] and v is not None \
            and v["rappresentazione"] == "radice" and v["esatto"] == "√841"
    if not admitted:
        return False
    states = {t["compito"]: t["stato"] for t in rec["compiti"]}
    if cat == "lettura_valore":
        v = {x["elemento"]: x for x in rec["valori"]}.get(s["elemento"])
        kind_ok = s["elemento"].startswith("angle:") if sub == "angolo" else (
            s["elemento"].startswith("side:") and v is not None and v["rappresentazione"] == sub)
        return v is not None and kind_ok and states.get(s["compito"]) == "ammesso" \
            and any(q.get("elemento") == s["elemento"] and q["tipo"] == s["compito"] for q in rec["qa"])
    if cat in ("classificazione_recuperabile", "classificazione_omessa"):
        e = [x for x in rec["shape_type_c9"] if x["figura"] == s["figura"]]
        if len(e) != 1 or states.get("shape_type", "quarantena") == "ammesso":
            return False
        e = e[0]
        if cat == "classificazione_recuperabile":
            return e["esito"] == "recuperabile" and e["risposta"] == sub
        if e["esito"] != "omessa":
            return False
        sides = e["fatti"]["lati"]
        shown = Counter(latex_value(x) for x in sides.values())
        rounded = any(shown[latex_value(x)] > 1 and c9.representation(x) == "decimale_arrotondato"
                      for x in sides.values())
        return {"piu_opzioni_vere": e["motivo"] == "piu_opzioni_vere",
                "segno_angolo_retto": e["motivo"] == "richiede_segno_angolo_retto",
                "misure_insufficienti": e["motivo"] == "non_deciso_dalle_misure_stampate" and not rounded,
                "uguaglianza_apparente_arrotondamento": e["motivo"] == "non_deciso_dalle_misure_stampate" and rounded
                }[sub]
    if cat == "altro_compito":
        return states.get(sub) == "ammesso" and s["compito"] == sub
    return False


def verify(args) -> int:
    t0 = time.time()
    pkg_dir = args.pacchetto
    checks: list[dict] = []

    def check(name: str, ok, detail=None) -> None:
        checks.append({"controllo": name, "esito": bool(ok), "dettaglio": detail})
        print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail is not None else ""), flush=True)

    pkg = rp.read_js(pkg_dir / "dati/pacchetto.js", "PACCHETTO")
    sol = rp.read_js(pkg_dir / "dati/soluzioni.js", "SOLUZIONI")
    meta = json.loads((pkg_dir / "pacchetto.json").read_text(encoding="utf-8"))
    plan = json.loads((pkg_dir / "moduli/piano.json").read_text(encoding="utf-8"))
    with open(pkg_dir / "moduli/selezione.tsv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    d = derive(args)
    pkg2, sol2 = package_objects(d)
    check("manifest con lo sha256 atteso, records.jsonl.gz con lo sha256 di manifest.json",
          pkg["sorgente"]["manifest_sha256"] == args.manifest_sha256 == d["manifest"]["manifest_sha256"]
          and pkg["sorgente"]["records_sha256"] == d["manifest"]["records_sha256"], args.manifest_sha256[:12])
    stated = {k: v for k, v in pkg.items() if k not in ("impronta", "impronte_dati")}
    digests = {"soluzioni": rp.canonical_digest(sol)}
    check("impronta ricalcolata da pacchetto.js e soluzioni.js, uguale a pacchetto.json",
          pkg["impronta"] == rp.canonical_digest({"pacchetto": stated, **digests}) == meta["impronta"]
          and pkg["impronte_dati"] == digests)
    check("schede-selezione-manifest: selezione, ordine, domande, opzioni e soluzioni ricalcolati dal manifest e dagli "
          "shard uguali al pacchetto", pkg2 == pkg and sol2 == sol, {"schede": len(pkg["schede"])})
    check("inventario moduli/selezione.tsv uguale alla selezione ricalcolata",
          tsv_text(tsv_rows(d)) == (pkg_dir / "moduli/selezione.tsv").read_text(encoding="utf-8"), len(rows))
    plan2 = plan_doc(d)
    check("piano moduli/piano.json uguale a quello ricalcolato (quote, disponibilità, celle, deficit)", plan2 == plan,
          [x["strato"] for x in plan["deficit"]])
    ids = [c["id"] for c in pkg["schede"]]
    n_cards = TOTAL - sum(r["deficit"] for r in plan["strati"])
    check(f"{n_cards} schede (80 meno i deficit) M01-M{n_cards:02d} nell'ordine, ciascuna con la sua soluzione",
          sum(q for _, _, q in PLAN) == TOTAL and ids == [f"M{i:02d}" for i in range(1, n_cards + 1)]
          and set(sol) == set(ids) and [c["posizione"] for c in pkg["schede"]] == list(range(1, n_cards + 1)),
          len(ids))
    order = [r["rango_ordine"] for r in rows]
    check("ordine delle schede per sha256({campione, seme 0, ordine: unità}), categorie mescolate",
          order == sorted(order) and all(r["rango_ordine"] == rank(ordine=sol[r["scheda"]]["selezione"]["unita"])
                                         for r in rows))
    quota = {stratum_id(c, s): q for c, s, q in PLAN}
    got = Counter(s["strato"] for s in sol.values())
    rep = {r["strato"]: r for r in plan["strati"]}
    bad_q = {s: (got[s], q, rep[s]["deficit"]) for s, q in quota.items()
             if got[s] != rep[s]["scelte"] or got[s] + rep[s]["deficit"] != q
             or (rep[s]["deficit"] and len([c for c in rep[s]["celle"] if "esaurita" in c]) != len(cells_for(s)))}
    check("quote: ogni strato ha la sua quota, o un deficit registrato con tutte le celle esaurite", not bad_q,
          bad_q or {s: got[s] for s in quota})
    recs_of = [c["record"] for c in pkg["schede"]]
    groups = [c["gruppo"] for c in pkg["schede"]]
    check("record e gruppi distinti: una scheda per gruppo, nessun collegamento fra schede",
          len(set(recs_of)) == len(recs_of) and len(set(groups)) == len(groups), len(set(groups)))
    recs = {}
    _, _, it = stream_records(args.manifest, args.manifest_sha256)
    want = set(recs_of)
    for r in it:
        if r["record"] in want:
            recs[r["record"]] = r
    bad = [cid for cid, s in sol.items()
           if not condition_ok(s["categoria"], s["sottocategoria"], recs[s["record"]["record"]], s)
           or recs[s["record"]["record"]]["gruppo"] != s["record"]["gruppo"]]
    check("condizioni degli strati riverificate sui record del manifest (codice scritto a parte dalla selezione)",
          not bad, bad[:5])
    depth = Counter(s["record"]["profondita"] for s in sol.values())
    check("profondità coperte, d3 compresa", set(depth) == {"d0", "d1", "d2", "d3"}, dict(depth))
    not_train = [cid for cid, s in sol.items() if (s["record"]["split"], s["record"]["stato"]) != ("train", "ammesso")]
    ctrl = [c["id"] for c in pkg["schede"] if c["controllo_noto"]]
    check("solo record di train ammessi, salvo il controllo noto (eccezione documentata se c'è)",
          set(not_train) <= set(ctrl) and (not not_train or plan["controllo_noto"]["eccezione"]),
          {"fuori": not_train, "eccezione": plan["controllo_noto"]["eccezione"]})
    check("un solo controllo noto, d3-44919 side:EH, segnato nel primo passaggio",
          len(ctrl) == 1 and sol[ctrl[0]]["record"]["record"] == CONTROL[0] and sol[ctrl[0]]["elemento"] == CONTROL[1]
          and sol[ctrl[0]]["categoria"] == "controllo_noto", ctrl)
    st_bad = [cid for cid, s in sol.items() if s["compito"] == "shape_type" and s["stato_compito"] == "ammesso"]
    check("shape_type resta in quarantena in ogni scheda di classificazione", not st_bad, st_bad)
    diag = [c["id"] for c in pkg["schede"] if c["richiesta"] == "diagnostica"]
    check("classificazioni omesse: richiesta diagnostica dichiarata, senza target; le altre schede sono QA candidate "
          "col target", set(diag) == {cid for cid, s in sol.items() if s["categoria"] == "classificazione_omessa"}
          and all(sol[x]["target"] is None and not sol[x]["qa_candidata"] for x in diag)
          and all(s["target"] is not None and s["qa_candidata"] for cid, s in sol.items() if cid not in diag), len(diag))
    opt_bad = [c["id"] for c in pkg["schede"] if (c["opzioni"] != []) != (sol[c["id"]]["compito"] == "shape_type")
               or (c["opzioni"] and sol[c["id"]]["qa_candidata"] and sol[c["id"]]["target"] not in c["opzioni"])]
    check("opzioni solo dove appartengono alla domanda (classificazioni), col target fra le opzioni delle QA", not opt_bad,
          opt_bad)

    def hidden(obj, path=""):
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in HIDDEN_KEYS:
                    found.append(f"{path}.{k}")
                found += hidden(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                found += hidden(v, f"{path}[{i}]")
        return found

    leaks = hidden(pkg["schede"])
    check("primo passaggio: nessun target, annotazione, esito, categoria, motivo o stato in pacchetto.js", not leaks,
          leaks[:5])
    secrets = secrets_of(pkg, sol)
    visible_leaks = [c["id"] for c in pkg["schede"] if any(rp.bounded(strings_of(c), t)
                                                           for t in secrets_card(c, sol[c["id"]]))]
    check("primo passaggio: i testi del secondo passaggio (target, didascalia, descrizione, categorie, motivi) non "
          "compaiono nella scheda, salvo dove sono parte della domanda o delle opzioni", not visible_leaks,
          {"stringhe": len(secrets), "schede": visible_leaks[:5]})
    titles = [c["consegna"] for c in pkg["schede"]]
    cat_words = ("arrotond", "esatt", "radice", "intero", "recuperabil", "omess", "controllo", "piu_opzioni", "quadrato")
    check("consegne uguali per tipo di richiesta e senza parole delle categorie",
          len(set(titles)) <= len(CONSEGNE) and not any(w in t.lower() for t in titles for w in cat_words),
          len(set(titles)))
    refs = {c["immagine"]: c["immagine_sha256"] for c in pkg["schede"]}
    img_bad = [p for p, sha in refs.items() if not (pkg_dir / p).is_file() or sha256_file(pkg_dir / p) != sha
               or sha != d["images"][Path(p).stem]["sha256"]]
    check("immagini: caption.jpg originali, presenti, con lo sha256 della scansione e le dimensioni registrate",
          not img_bad and all((c["larghezza"], c["altezza"]) == (d["images"][c["record"]]["larghezza"],
                                                                 d["images"][c["record"]]["altezza"])
                              for c in pkg["schede"]), {"immagini": len(refs), "diverse": img_bad[:5]})
    on_disk = {p.relative_to(pkg_dir).as_posix() for p in (pkg_dir / "immagini").rglob("*") if p.is_file()}
    check("immagini: nessun file in più", on_disk == set(refs), len(on_disk))
    lines = (pkg_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed = [ln[66:] for ln in lines]
    bad_sum = [ln for ln in lines if ln[64:66] != "  " or sha256_file(pkg_dir / ln[66:]) != ln[:64]]
    check("SHA256SUMS: ogni file del pacchetto (interfaccia, dati, immagini, moduli) elencato col suo sha256",
          lines and not bad_sum and listed == signed_files(pkg_dir), {"file": len(lines), "diversi": bad_sum[:3]})
    html = (pkg_dir / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r'<script src="([^"]+)"', html)
    check("index.html: dati, logica e interfaccia nell'ordine, file presenti",
          scripts == ["dati/pacchetto.js", "dati/soluzioni.js", "logica.js", "app.js"]
          and all((pkg_dir / s).is_file() for s in scripts) and (pkg_dir / "stile.css").is_file(), scripts)
    vscripts = re.findall(r'<script src="([^"]+)"', (pkg_dir / "verifica.html").read_text(encoding="utf-8"))
    check("verifica.html: dati, logica, interfaccia e autotest", vscripts == scripts + ["test.js"], vscripts)
    net = []
    for name in ("index.html", "app.js", "logica.js", "stile.css", "verifica.html", "test.js"):
        text = (pkg_dir / name).read_text(encoding="utf-8")
        for pat in ("fetch(", "XMLHttpRequest", "http://", "https://", "import(", "<link rel=\"preconnect",
                    "navigator.sendBeacon", "WebSocket"):
            if pat in text:
                net.append(f"{name}: {pat}")
    check("interfaccia: nessuna richiesta di rete", not net, net)
    lexical = {n: rp.js_lexical((pkg_dir / n).read_text(encoding="utf-8")) for n in ("app.js", "logica.js", "test.js")}
    check("JavaScript: commenti, stringhe, espressioni regolari e parentesi bilanciate (lessicale, non esecuzione)",
          not any(lexical.values()), {k: v for k, v in lexical.items() if v} or len(lexical))
    es6 = [n for n in ("app.js", "logica.js", "test.js")
           if re.search(r"=>|\blet\s|\bconst\s|`|\bclass\s+[A-Z]", (pkg_dir / n).read_text(encoding="utf-8"))]
    check("JavaScript ES5 (niente frecce, let, const, template, classi): gira anche con Duktape", not es6, es6)
    failed = [x["controllo"] for x in checks if not x["esito"]]
    out = {"schema": SCHEMA + "-verifica", "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
           "pacchetto": um_rel(pkg_dir), "impronta": pkg["impronta"], "controlli": checks, "falliti": failed,
           "nota": "verifica statica di dati, selezione e riferimenti; l'interfaccia non è provata in un browser",
           "secondi": round(time.time() - t0, 1)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(checks) - len(failed)} controlli superati su {len(checks)}")
    return 1 if failed else 0


def visible_text(card: dict) -> str:
    return "\n".join([card["id"], card["record"], card["domanda"], card["consegna"], *card["opzioni"]])


def strings_of(obj) -> str:
    """Tutti i testi di un oggetto (valori, non chiavi), uno per riga."""
    if isinstance(obj, dict):
        return "\n".join(strings_of(v) for v in obj.values())
    if isinstance(obj, list):
        return "\n".join(strings_of(v) for v in obj)
    return "" if obj is None or isinstance(obj, bool) else str(obj)


def secrets_card(card: dict, s: dict) -> list[str]:
    """Testi del secondo passaggio di una scheda (almeno 4 caratteri) che non sono nel suo testo visibile: target,
    didascalia, categorie, motivi, valori; come segreti() in test.js."""
    visible = visible_text(card)
    cand = [s["target"], s["sottocategoria"], s["categoria"], s["strato"], s["motivo_selezione"],
            s["annotazioni"]["didascalia"]]
    if s["valore"]:
        cand += [s["valore"]["esatto"], s["valore"]["visualizzato_previsto"], s["valore"]["rappresentazione"]]
    if s["c9"]:
        cand += [s["c9"].get("motivo"), s["c9"].get("risposta")]
    out = []
    for t in cand:
        t = "" if t is None else str(t).strip()
        if len(t) >= 4 and not rp.bounded(visible, t):
            out.append(t)
    return out


def secrets_of(pkg: dict, sol: dict) -> list[str]:
    return sorted({t for c in pkg["schede"] for t in secrets_card(c, sol[c["id"]])})


# ---------------------------------------------------------------- JavaScript con Duktape

JS_FILES = ("dati/pacchetto.js", "dati/soluzioni.js", "logica.js", "app.js", "test.js")
LINKS_JS = """JSON.stringify((function () {
  var C = window.Logica.contesto(window.PACCHETTO, window.SOLUZIONI), o = {};
  Object.keys(C.collegati).forEach(function (k) { o[k] = C.collegati[k].map(function (c) { return c.id; }).sort(); });
  return o;
})())"""


def run_once(lib: str, pkg_dir: Path, dom: Path | None) -> dict:
    duk = rp.Duktape(lib)
    res = {"dom": um_rel(dom) if dom else None, "file": {}, "errore": None, "righe": None, "collegati": None}
    try:
        res["versione_duktape"] = duk.eval("String(typeof Duktape === 'object' ? Duktape.version : '?')")
        if dom:
            duk.eval(dom.read_text(encoding="utf-8"), str(dom))
            res["file"][um_rel(dom)] = {"sha256": sha256_file(dom)}
        else:
            duk.eval("var window = this;", "prelude")
        for name in JS_FILES:
            t1 = time.time()
            duk.eval((pkg_dir / name).read_text(encoding="utf-8"), name)
            res["file"][name] = {"sha256": sha256_file(pkg_dir / name), "secondi": round(time.time() - t1, 2)}
        if dom:
            res["timer_rimasti"] = int(float(duk.eval("String(window.__DOM.svuotaTimer())", "timer")))
        res["righe"] = json.loads(duk.eval("JSON.stringify(window.RISULTATI_TEST || null)", "risultati"))
        res["collegati"] = json.loads(duk.eval(LINKS_JS, "collegamenti"))
    except (rp.JSError, rp.Failed) as exc:
        res["errore"] = str(exc)
    finally:
        duk.close()
    return res


def run_js(args) -> int:
    t0 = time.time()
    pkg_dir = args.pacchetto
    pkg = rp.read_js(pkg_dir / "dati/pacchetto.js", "PACCHETTO")
    sol = rp.read_js(pkg_dir / "dati/soluzioni.js", "SOLUZIONI")
    expected_links = {c["id"]: [] for c in pkg["schede"]}
    secrets = secrets_of(pkg, sol)
    out = {"schema": SCHEMA + "-esecuzione-js", "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
           "pacchetto": um_rel(pkg_dir), "impronta": pkg["impronta"],
           "motore": {"libreria": args.libreria, "sha256": sha256_file(Path(args.libreria))},
           "nota": "logica e interfaccia eseguite con Duktape, senza DOM e su un DOM simulato; non è una prova in un "
                   "browser (disegno, eventi veri, immagini e localStorage si provano con verifica.html)",
           "segreti_cercati": len(secrets), "esecuzioni": []}
    failed = False
    for dom in [None] + ([args.dom] if args.dom else []):
        res = run_once(args.libreria, pkg_dir, dom)
        print(f"== esecuzione {'DOM simulato' if dom else 'senza DOM'}", flush=True)
        rows = res["righe"]
        if res["errore"] or rows is None:
            print("ERRORE " + str(res["errore"]), flush=True)
            failed = True
        else:
            leaks = [i for i, r in enumerate(rows) if any(rp.bounded(r["nome"] + "\n" + r["dettaglio"], s)
                                                          for s in secrets)]
            for i in leaks:
                rows[i] = {"nome": f"riga {i + 1}", "ok": False, "dettaglio": "[nascosto: conteneva una soluzione]"}
            res["righe_con_soluzioni"] = leaks
            res["collegamenti_uguali_a_python"] = res["collegati"] == expected_links
            res["superati"] = sum(r["ok"] is True for r in rows)
            res["falliti"] = [r["nome"] for r in rows if r["ok"] is False]
            res["saltati"] = [r["nome"] for r in rows if r["ok"] is None]
            for r in rows:
                mark = {True: "ok   ", False: "FAIL ", None: "salt "}[r["ok"]]
                print(mark + r["nome"] + (f"  ({r['dettaglio']})" if r["dettaglio"] else ""), flush=True)
            print(f"{res['superati']} superati, {len(res['falliti'])} falliti, {len(res['saltati'])} saltati su "
                  f"{len(rows)}; nessun collegamento fra schede: {res['collegamenti_uguali_a_python']}; righe con "
                  f"soluzioni: {len(leaks)}", flush=True)
            failed |= bool(res["falliti"] or leaks or not res["collegamenti_uguali_a_python"])
        res["collegati"] = None if res["collegati"] == expected_links else res["collegati"]
        out["esecuzioni"].append(res)
    out["secondi"] = round(time.time() - t0, 1)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("costruisci", "verifica"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--manifest-sha256", required=True)
        if name == "costruisci":
            p.add_argument("--out", required=True, type=Path)
        else:
            p.add_argument("--pacchetto", required=True, type=Path)
            p.add_argument("--report", required=True, type=Path)
    f = sub.add_parser("firme")
    f.add_argument("--pacchetto", required=True, type=Path)
    j = sub.add_parser("esegui-js")
    j.add_argument("--pacchetto", required=True, type=Path)
    j.add_argument("--report", required=True, type=Path)
    j.add_argument("--dom", type=Path)
    j.add_argument("--libreria", default=rp.DUKTAPE)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("mavis_review_package.py va eseguito in un job SLURM")
    try:
        if args.cmd == "costruisci":
            cmd_build(args)
        elif args.cmd == "firme":
            print(json.dumps({"file_firmati": write_sums(args.pacchetto)}))
        elif args.cmd == "verifica":
            sys.exit(verify(args))
        else:
            sys.exit(run_js(args))
    except (Failed, mc.Failed, rp.Failed, mf.ManifestError) as exc:
        sys.exit(f"pacchetto non scritto o non valido: {exc}")


if __name__ == "__main__":
    main()
