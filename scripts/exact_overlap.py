#!/usr/bin/env python3
"""Primo sottoinsieme di C6 (S1 §4; docs/F0_IMPLEMENTATION_PLAN.md §3): identità esatte delle immagini fra i sintetici
v4, UniMER-1M, MAVIS-Geometry e PixMo-Cap, ricavate dai manifest candidati e dalle scansioni di decontaminazione già
fatte, senza rileggere immagini, shard, zip o parquet e senza ricalcolare hash di immagini. Solo in un job SLURM CPU:
ingressi in sola lettura, uscite in una cartella nuova. È una diagnostica: non cambia gruppi, split o stati, non scrive
liste per i loader e non sostituisce i group_id dei manifest; non è l'indice globale di S1 §4.

Ingressi, per sorgente (--manifest NOME CARTELLA SHA256, NOME fra synth, unimer, mavis, pixmo):
- manifest.json con lo sha256 atteso; records.jsonl.gz con lo sha256 e le righe di manifest.json; record_id ricalcolati
  da (sorgente, revisione, record), sorgente e revisione del manifest, shard registrati con i loro conteggi, voci
  (insieme, chiave) uniche, ogni gruppo in un solo split; per UniMER ogni riga ritagliata con la chiave, il record_id e il
  gruppo della sua formula;
- la scansione che il manifest cita, con gli sha256 registrati nel manifest: `decontaminazione.scansione` per sintetici,
  UniMER e MAVIS, `ingressi.file.scansione` per PixMo. Ogni riga si ricollega a una sola voce del manifest per (insieme,
  shard, posizione, chiave) e coi digest del manifest: byte dell'immagine dello shard per tutti, JSON per UniMER e MAVIS,
  pixel di shard e originale, file, riga e didascalia per PixMo. Ogni voce deve avere una sola riga; per PixMo le righe
  senza shard sono gli scarti dello sharding elencati nel manifest;
- facoltativo (--c9 CARTELLA SHA256): MAVIS candidato-5, riferimento della revisione C9. Si legge solo manifest.json, per
  controllare che derivi da candidato-3 (catena delle versioni) con le stesse righe: non è una sorgente in più.

Rappresentazioni (nessun hash ricalcolato):
- `originale`: il file dello snapshot (UniMER images/<i>.png e MAVIS <cartella>/<n>_caption.jpg degli zip, immagine del
  parquet di PixMo); per i sintetici non c'è un file distinto, perché il PNG dello shard è l'immagine generata;
- `shard`: il file immagine dello shard che il loader legge e passa al processor (PNG generato, PNG denso di UniMER, PNG
  a 2304 patch di MAVIS, JPEG a 1024 patch di PixMo); per le righe ritagliate di UniMER è l'immagine intera da cui il
  loader ritaglia;
- `ritaglio k`: la riga k di UniMER come la ritaglia il loader (data.unimer_line_sample), cioè l'immagine data al
  processor per transcribe_line e transcribe_line_select; solo pixel.
Tipi di identità: `byte` (sha256 dei byte del file; originale e shard) e `pixel` (sha256 di dimensioni e valori RGB dopo
processing.to_rgb, convenzione rgb-su-bianco/1, senza orientamento EXIF), usato solo se la scansione dichiara quella
convenzione. Il pixel_sha256 del manifest dei sintetici è RGBA diretto (synth_candidate.decode_png): un'altra convenzione,
mai confrontata. Ogni digest di un tipo si confronta con tutti i digest dello stesso tipo, di ogni sorgente e
rappresentazione. Un digest mancante (immagine illeggibile, convenzione diversa, problema registrato dalla scansione,
voce senza riga) rende il controllo non eseguibile per quella voce e rappresentazione e si conta nella copertura: non
vuol dire "nessuna sovrapposizione".

Componenti diagnostiche: i gruppi dei manifest (group_id) uniti in modo transitivo dalle classi di digest uguali (stesso
tipo e valore) che toccano almeno due gruppi. Per ogni componente con almeno due gruppi: voci con riferimento al manifest
(manifest, insieme, shard, posizione, chiave), record_id, gruppo, split, stato e compiti, classi che la legano (tipo,
digest, voci e rappresentazioni), categoria:
- `train_dev`: record caricabili (ammessi, nello split train o dev, con almeno un compito ammesso) sia in train sia in dev;
- `riservati`: record dello split test (valutazione) legati a record caricabili;
- `solo_train`, `solo_dev`: almeno due gruppi caricabili, tutti nello stesso split;
- `un_gruppo_caricabile`: un solo gruppo con record caricabili, legato a record non caricabili;
- `non_caricabili`: nessun record caricabile.
La stessa immagine è un legame visivo, non l'identità della domanda o della risposta: nessuno stato si propaga, e dei
record non caricabili si riportano i motivi, con la loro portata (immagine, testo o risposta, gruppo, da esaminare).
Una formula e la sua riga ritagliata (stessa chiave e stesso record_id) sono due voci e un solo record: i conteggi per
record usano il record_id.

Con --limit N si leggono solo le prime N righe di ogni records.jsonl.gz (prova limitata); gli sha256 dei file restano
completi. Il rapporto ha allora `parziale` e la stima del lavoro completo.

    srun $RUN_PYTHON -u -B scripts/exact_overlap.py --manifest synth DIR SHA256 --manifest unimer DIR SHA256 \\
        --manifest mavis DIR SHA256 --manifest pixmo DIR SHA256 [--c9 DIR SHA256] --out DIR [--summary FILE] [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import unimer_manifest as um  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.decontam import IMAGE_CONVENTION  # noqa: E402
from limite_vl.provenance import code_identity, write_new_json  # noqa: E402

SCHEMA = "exact-overlap/1"
SOURCES = {"synth": "synth", "unimer": "unimer", "mavis": "mavis-geometry", "pixmo": "pixmo-cap"}  # nome -> sorgente
ORDER = tuple(SOURCES)
SCAN_FORMAT = {"synth": "synth", "unimer": "panel", "mavis": "panel", "pixmo": "pixmo"}
KIND = {b"B": "byte", b"P": "pixel"}
SHIFT = 12  # codice di un'occorrenza: voce << SHIFT | rappresentazione (0 originale, 1 shard, 2 + k ritaglio k)
LOADABLE_SPLITS = ("train", "dev")
CATEGORIES = ("train_dev", "riservati", "solo_train", "solo_dev", "un_gruppo_caricabile", "non_caricabili")
SUMMARY_PER_CATEGORY = 30  # componenti per categoria nel riepilogo per git (tutte in componenti.jsonl.gz)
SUMMARY_MEMBERS = 10  # voci e classi per componente nel riepilogo per git
REPORT_MEMBERS = 500  # voci e classi per componente in report.json
REPRESENTATIONS = {
    "originale": "file dello snapshot: UniMER images/<i>.png e MAVIS <cartella>/<n>_caption.jpg degli zip, immagine del "
                 "parquet di PixMo; i sintetici non ne hanno uno distinto (il PNG dello shard è l'immagine generata)",
    "shard": "file immagine dello shard letto dal loader: PNG generato (sintetici), PNG denso (UniMER), PNG a 2304 patch "
             "(MAVIS), JPEG a 1024 patch (PixMo); per le righe ritagliate di UniMER l'immagine intera da cui si ritaglia",
    "ritaglio": "riga k di UniMER ritagliata come in data.unimer_line_sample (to_rgb), l'immagine data al processor per "
                "transcribe_line e transcribe_line_select; solo pixel",
}
TYPES = {"byte": "sha256 dei byte del file immagine (originale e shard; i ritagli non sono file)",
         "pixel": f"sha256 di dimensioni e valori RGB dopo processing.to_rgb, convenzione {IMAGE_CONVENTION} (trasparenze "
                  "su bianco, nessun orientamento EXIF), dalle scansioni che la dichiarano"}
# Portata dei motivi di un record non caricabile: se riguarda l'immagine può valere per chi ha la stessa immagine nella
# stessa rappresentazione; se riguarda testo, risposta, didascalia, formula o annotazioni no; `gruppo` se viene da un
# altro record del gruppo. Un motivo di decontaminazione prende la portata dalle corrispondenze del record stesso
# (evidenza di decontaminazione: pixel e pHash -> immagine, testo e formule -> testo o risposta; nessuna -> gruppo).
# Gli altri motivi si leggono con le parole di SCOPE_RULES (minuscolo), in quest'ordine; il resto è "da_esaminare".
IMAGE_MATCHES = ("pixel", "phash", "image")  # tipi di corrispondenza sull'immagine (synth_decontam usa "image" per pHash)
SCOPE_RULES = (
    ("gruppo", ("dal gruppo", "dalla formula", "ereditato:", "dell'altra variante", "contaminazione: ")),
    ("immagine", ("stessa immagine di unimer-test", "dimensioni diverse", "segnaposto", "orientamento", "exif",
                  "illeggibile", "difetto", "visivo_integrita", "testi_sovrapposti", "testo_su_marcatore",
                  "testo_piccolo")),
    ("testo_o_risposta", ("unimer_check", "formula di unimer-test", "rietichettat", "verità", "andata e ritorno",
                          "segnalato da", "errore_matematico", "formato_minore", "incerto")),
)
PROPOSALS = {
    "train_dev": "conflitto effettivo: un'immagine di un record caricabile nel train è anche in un record caricabile di dev. "
                 "Proposta, non applicata: nella versione successiva dei manifest coinvolti i gruppi della componente "
                 "diventano un solo gruppo globale con un solo split; correzione minima: i gruppi di dev della componente "
                 "passano al train (il dev resta il campione per hash degli altri gruppi), oppure tutta la componente va "
                 "in dev. Decisione dell'utente",
    "riservati": "un record caricabile ha la stessa immagine di un record riservato alla valutazione: proposta, non "
                 "applicata, di escludere dal training i gruppi caricabili della componente, come i gruppi dei sintetici "
                 "legati a test e coppie (train/test/escluso)",
    "solo_train": "nessun conflitto di split: la stessa immagine compare più volte nel training, anche con compiti "
                  "diversi; proposta: unire i gruppi nell'indice globale (lo split è già lo stesso) e contare le "
                  "ripetizioni; nessuna esclusione, perché compiti diversi sulla stessa immagine non si contraddicono",
    "solo_dev": "nessuna fuga train/dev: la stessa immagine compare più volte nel dev; proposta: unirli in un solo gruppo "
                "di dev e contare la ripetizione nelle metriche",
    "un_gruppo_caricabile": "un solo gruppo caricabile, legato a record in quarantena o esclusi: nessun conflitto fra dati "
                            "caricabili. Il motivo del record non caricabile si estende solo se riguarda l'immagine e la "
                            "stessa rappresentazione (portata `immagine`), dopo verifica; motivi di testo, risposta, "
                            "didascalia o formula non si estendono",
    "non_caricabili": "nessun effetto sui dati caricabili; da riconsiderare solo se uno dei record viene riammesso",
}


class Failed(Exception):
    """Ingresso diverso da quello atteso: il confronto non si fa."""


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def __call__(self, name: str, cond, detail=None) -> bool:
        self.items.append({"controllo": name, "esito": bool(cond), "dettaglio": detail})
        print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail is not None else ""),
              flush=True)
        return bool(cond)

    def failed(self) -> list[str]:
        return [c["controllo"] for c in self.items if not c["esito"]]


class Problems:
    """Problemi contati per tipo, con al più 5 esempi."""

    def __init__(self):
        self.n: Counter = Counter()
        self.examples: dict[str, list] = defaultdict(list)

    def __call__(self, kind: str, what) -> None:
        self.n[kind] += 1
        if len(self.examples[kind]) < 5:
            self.examples[kind].append(str(what))

    def detail(self) -> dict:
        return {k: {"n": v, "esempi": self.examples[k]} for k, v in sorted(self.n.items())}


class GzRows:
    """Righe JSON di un .jsonl.gz in streaming, con sha256 e byte del file compresso calcolati nella stessa lettura; con
    limit si interpretano solo le prime righe e il resto del file si legge solo per lo sha256."""

    def __init__(self, path: Path, limit: int | None = None):
        self.path, self.limit = Path(path), limit
        self.rows = 0
        self.text_bytes = 0
        self.complete = False
        self.sha256: str | None = None
        self.bytes: int | None = None
        self.seconds_rows = 0.0  # lettura delle righe, compreso il lavoro di chi le consuma
        self.seconds_tail = 0.0  # resto del file letto solo per lo sha256 (prova limitata)

    def __iter__(self):
        t0 = time.time()
        raw = um.HashingReader(self.path)
        with gzip.GzipFile(fileobj=io.BufferedReader(raw, buffer_size=1 << 20)) as gz:
            stopped = False
            for line in gz:
                if self.limit is not None and self.rows >= self.limit:
                    stopped = True
                    break
                self.text_bytes += len(line)
                if line.strip():
                    self.rows += 1
                    yield json.loads(line)
            self.complete = not stopped
        t1 = time.time()
        self.sha256, self.bytes = raw.finish()
        self.seconds_rows, self.seconds_tail = round(t1 - t0, 2), round(time.time() - t1, 2)

    def entry(self, declared) -> dict:
        return {"path": um.rel(self.path), "bytes": self.bytes, "sha256": self.sha256, "righe_lette": self.rows,
                "righe_dichiarate": declared, "completo": self.complete, "byte_decompressi": self.text_bytes,
                "secondi_righe": self.seconds_rows, "secondi_solo_sha256": self.seconds_tail}


class Store:
    """Voci dei manifest (nodi), gruppi e occorrenze dei digest."""

    def __init__(self):
        self.src: list[int] = []
        self.ref: list[tuple] = []  # (insieme, shard, posizione, chiave)
        self.rid: list[str] = []
        self.grp: list[int] = []
        self.split: list[str] = []
        self.state: list[str] = []
        self.tasks: list[tuple] = []  # ((compito, stato, utilizzabile), ...)
        self.motive: list = []  # None per gli ammessi; altrimenti (motivi, tipi di corrispondenza della decontaminazione)
        self.link: list[bytes | None] = []  # digest dei campi da ritrovare nella riga della scansione
        self.groups: list[tuple[int, str]] = []
        self.group_of: dict[str, int] = {}
        self.group_split: list[str] = []
        self.first: dict[bytes, int] = {}
        self.dups: dict[bytes, list[int]] = {}
        self._shared: dict = {}

    def share(self, x):
        return self._shared.setdefault(x, x)

    def add_digest(self, kind: bytes, digest: str, node: int, rep: int) -> None:
        key = kind + bytes.fromhex(digest)
        code = node << SHIFT | rep
        old = self.first.setdefault(key, code)
        if old != code:
            lst = self.dups.get(key)
            if lst is None:
                self.dups[key] = [old, code]
            else:
                lst.append(code)


def rep_name(rep: int) -> str:
    return ("originale", "shard")[rep] if rep < 2 else f"ritaglio {rep - 2}"


def rep_kind(rep: int) -> str:
    return ("originale", "shard")[rep] if rep < 2 else "ritaglio"


def link_digest(values: list) -> bytes:
    return hashlib.sha256(mf.canonical(values).encode()).digest()


def read_meta(folder: Path, expected: str) -> tuple[dict, dict]:
    path = Path(folder) / "manifest.json"
    if not path.is_file():
        raise Failed(f"manifest mancante: {path}")
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != mf.require_digest(expected, f"{path}: sha256 atteso"):
        raise Failed(f"{path}: sha256 {got}, atteso {expected}")
    return json.loads(raw), {"path": um.rel(path), "bytes": len(raw), "sha256": got}


def lineage(meta: dict) -> set[str]:
    """sha256 dei manifest da cui la versione deriva (precedente, split ereditato, catena delle versioni)."""
    out = set()
    for x in [meta.get("precedente") or {}, (meta.get("split") or {}).get("ereditato_da") or {},
              *(meta.get("catena_delle_versioni") or [])]:
        if x.get("manifest_sha256"):
            out.add(x["manifest_sha256"])
    return out


def tasks_of(name: str, r: dict) -> tuple:
    """Compiti della voce con stato e utilizzabilità, come manifest._record_tasks: MAVIS per compito (describe sempre,
    le domande se ce n'è almeno una); UniMER e PixMo coi nomi e lo stato del record; i sintetici con le domande della
    scena, tutte nello stato del record."""
    if name == "mavis":
        return tuple((t["compito"], t["stato"], t["compito"] == "describe" or t.get("domande", 0) > 0)
                     for t in r.get("compiti") or [])
    if name == "synth":
        return (("domande_della_scena", r["stato"], bool(r.get("qa"))),)
    return tuple((t, r["stato"], True) for t in r.get("compiti") or [])


def decontam_kinds(r: dict) -> tuple:
    """Tipi di corrispondenza dell'evidenza di decontaminazione del record (pixel, phash, testo, formula...)."""
    kinds = set()
    for e in r.get("evidenze") or []:
        if "decontaminazione" in str(e.get("controllo") or e.get("lista") or ""):
            kinds.update(k for k, v in (e.get("corrispondenze") or {}).items() if v)
    return tuple(sorted(kinds))


def motive_of(name: str, r: dict):
    """Motivi di un record non ammesso, come elenco di testi: per i sintetici le categorie col primo dettaglio, per
    PixMo anche i codici dei blocchi diretti ed ereditati."""
    if r["stato"] == "ammesso":
        return None
    m = r.get("motivo") or []
    texts = [f"{c}: {(xs or [''])[0]}" for c, xs in m.items()] if isinstance(m, dict) else [str(x) for x in m]
    if name == "pixmo":
        b = r.get("blocchi") or {}
        texts = list(b.get("diretti") or []) + [f"ereditato:{e.get('codice')}" for e in b.get("ereditati") or []] + texts
    return tuple(texts), decontam_kinds(r)


def loadable(state: str, split: str, tasks: tuple) -> bool:
    return state == "ammesso" and split in LOADABLE_SPLITS and any(st == "ammesso" and ok for _, st, ok in tasks)


def scope(texts, kinds) -> list[str]:
    """Portata dei motivi di un record non caricabile: decontaminazione dalle corrispondenze del record (`kinds`), gli
    altri motivi da SCOPE_RULES."""
    out = set()
    for t in texts:
        low = t.lower()
        if "decontaminazione" in low:
            own = set(kinds)
            if own & set(IMAGE_MATCHES):
                out.add("immagine")
            if own - set(IMAGE_MATCHES):
                out.add("testo_o_risposta")
            if not own:
                out.add("gruppo")
            continue
        out.add(next((s for s, words in SCOPE_RULES if any(w in low for w in words)), "da_esaminare"))
    return sorted(out)


# ---------------------------------------------------------------- lettura dei manifest

def read_manifest(name: str, folder: Path, expected: str, limit, store: Store, checks: Checks) -> dict:
    t0 = time.time()
    s = ORDER.index(name)
    meta, meta_entry = read_meta(folder, expected)
    if meta.get("schema") != mf.SCHEMA or meta.get("sorgente") != SOURCES[name]:
        raise Failed(f"{folder}: schema {meta.get('schema')!r} o sorgente {meta.get('sorgente')!r} inattesi per {name}")
    rev = meta.get("revisione")
    shards = {x["percorso"]: x for x in meta.get("shards") or []}
    rec = meta.get("records") or {}
    rows = GzRows(Path(folder) / str(rec.get("file")), limit)
    bad = Problems()
    per_shard: Counter = Counter()
    seen: set = set()
    entries: dict[tuple, int] = {}
    formula: dict[str, int] = {}
    counts: Counter = Counter()
    for r in rows:
        key, ins = r.get("record"), r.get("insieme")
        rid = r.get("record_id")
        if rid != mf.sha256_text(mf.canonical({"sorgente": r.get("sorgente"), "revisione": r.get("revisione"),
                                               "record": key})):
            bad("record_id diverso da quello ricalcolato", key)
        if r.get("sorgente") != SOURCES[name] or r.get("revisione") != rev:
            bad("sorgente o revisione diverse da quelle del manifest", key)
        if r.get("shard") not in shards:
            bad("shard non registrato", key)
        if (ins, key) in seen:
            bad("voce (insieme, chiave) ripetuta", f"{ins}/{key}")
        seen.add((ins, key))
        where = (r.get("shard"), r.get("indice"))
        if where in entries:
            bad("posizione ripetuta nello shard", where)
        gid, split, state = r.get("gruppo"), r.get("split"), r.get("stato")
        if not isinstance(gid, str) or not mf.HEX64.match(gid) or split not in mf.SPLITS or state not in mf.STATES:
            bad("gruppo, split o stato non validi", key)
            continue
        g = store.group_of.get(gid)
        if g is None:
            g = store.group_of[gid] = len(store.groups)
            store.groups.append((s, gid))
            store.group_split.append(split)
        elif store.groups[g][0] != s or store.group_split[g] != split:
            bad("gruppo in due split o in due sorgenti", key)
        images = r.get("immagini") or [{}]
        if SCAN_FORMAT[name] == "synth":
            link = [images[0].get("sha256")]
        elif SCAN_FORMAT[name] == "panel":
            link = [r.get("json_sha256"), images[0].get("sha256"), images[0].get("sha256")]
        else:
            o = r.get("originale") or {}
            link = [images[0].get("sha256"), images[0].get("pixel_sha256"), o.get("pixel_sha256"), o.get("file"),
                    o.get("riga"), r.get("didascalia_sha256")]
        n = len(store.src)
        store.src.append(s)
        store.ref.append((store.share(ins), store.share(r.get("shard")), r.get("indice"), key))
        store.grp.append(g)
        store.split.append(store.share(split))
        store.state.append(store.share(state))
        store.tasks.append(store.share(tasks_of(name, r)))
        store.motive.append(motive_of(name, r))
        store.link.append(link_digest(link))
        if name == "unimer" and ins == "train":
            formula[key] = n
            store.rid.append(rid)
        elif name == "unimer" and ins == "righe":
            f = formula.get(key)
            if f is None:
                if rows.limit is None:
                    bad("riga ritagliata senza la sua formula", key)
                store.rid.append(rid)
            else:
                if store.rid[f] != rid or store.grp[f] != g:
                    bad("riga ritagliata con record_id o gruppo diversi da quelli della formula", key)
                store.rid.append(store.rid[f] if store.rid[f] == rid else rid)  # stessa stringa: un solo record
        else:
            store.rid.append(rid)
        entries[where] = n
        per_shard[r.get("shard")] += 1
        counts[f"{ins}/{split}/{state}"] += 1
    complete = rows.complete
    checks(f"{name}: records.jsonl.gz con lo sha256 di manifest.json" + ("" if not complete else " e le sue righe"),
           rows.sha256 == rec.get("sha256") and (not complete or rows.rows == rec.get("righe")),
           {"sha256": rows.sha256, "righe_lette": rows.rows, "righe_dichiarate": rec.get("righe")})
    checks(f"{name}: record_id ricalcolati, sorgente e revisione, shard registrati, voci (insieme, chiave) e posizioni "
           "uniche, ogni gruppo in un solo split" + (", righe ritagliate con chiave, record_id e gruppo della formula"
                                                     if name == "unimer" else ""), not bad.n, bad.detail() or None)
    if complete:
        declared = {p: x.get("record") for p, x in shards.items()}
        checks(f"{name}: voci per shard uguali a quelle di manifest.json", dict(per_shard) == declared,
               None if dict(per_shard) == declared else {"lette": dict(per_shard), "dichiarate": declared})
    else:
        print(f"{name}: prova limitata, conteggi per shard non controllati", flush=True)
    discards = [x.get("chiave") for x in meta.get("scarti_dello_sharding") or []]
    return {"nome": name, "sorgente": SOURCES[name], "manifest_nome": f"{meta.get('nome')}-{meta.get('versione')}",
            "revisione": rev, "manifest": meta_entry, "records": rows.entry(rec.get("righe")), "meta": meta, "entries": entries, "shard_of_name": {x.get("nome"): x["percorso"] for x in shards.values()},
            "scarti": discards, "voci": len(entries), "per_insieme_split_stato": dict(sorted(counts.items())),
            "secondi_manifest": round(time.time() - t0, 1)}


def check_c9(folder: Path, expected: str, mavis: dict, checks: Checks) -> dict:
    """MAVIS candidato-5: solo manifest.json, derivazione da candidato-3 nella catena delle versioni e stesse righe."""
    meta, entry = read_meta(folder, expected)
    chain = meta.get("catena_delle_versioni") or []
    parent = mavis["manifest"]["sha256"]
    ok = meta.get("sorgente") == SOURCES["mavis"] and any(c.get("manifest_sha256") == parent for c in chain) \
        and (meta.get("records") or {}).get("righe") == mavis["records"]["righe_dichiarate"]
    checks("C9: mavis-geometry-candidato-5 deriva da candidato-3 (catena delle versioni) con le stesse righe: stessa "
           "sorgente, non confrontata come un'altra", ok, {"manifest_sha256": entry["sha256"], "genitore": parent})
    return {"manifest": entry, "nome": f"{meta.get('nome')}-{meta.get('versione')}",
            "catena": [{k: c.get(k) for k in ("versione", "manifest_sha256")} for c in chain],
            "uso": "riferimento separato della revisione C9: record, gruppi e split ereditati da candidato-3, quindi le "
                   "componenti trovate su candidato-3 valgono per gli stessi record; non è una sorgente in più"}


# ---------------------------------------------------------------- lettura delle scansioni

def scan_entries(name: str, meta: dict) -> tuple[dict, dict]:
    if name == "pixmo":
        f = ((meta.get("ingressi") or {}).get("file") or {}).get("scansione") or {}
        return f.get("records.jsonl.gz") or {}, f.get("summary.json") or {}
    s = (meta.get("decontaminazione") or {}).get("scansione") or {}
    return s.get("records") or {}, s.get("summary") or {}


def row_identity(fmt: str, row: dict, shard_of_name: dict) -> tuple:
    """(shard, posizione, chiave, insieme o None) di una riga della scansione, comparabili con la voce del manifest. Per
    i sintetici lo shard è un percorso assoluto, ricondotto alla radice del progetto una volta per shard
    (shard_of_name fa da cache)."""
    if fmt == "synth":
        path = row.get("shard")
        if path not in shard_of_name:
            shard_of_name[path] = um.rel(Path(path))
        return shard_of_name[path], row.get("posizione"), row.get("key"), None
    if fmt == "panel":
        return row.get("shard"), row.get("indice"), row.get("record"), row.get("insieme")
    shard = row.get("shard")
    return (shard_of_name.get(shard, f"?{shard}") if shard else None), row.get("posizione"), row.get("chiave"), None


def row_link(fmt: str, row: dict) -> list:
    """I campi della riga da ritrovare nella voce del manifest, nello stesso ordine di read_manifest."""
    if fmt == "synth":
        return [row.get("byte_sha256")]
    if fmt == "panel":
        return [row.get("json_sha256"), row.get("png_sha256"), (row.get("png") or {}).get("byte_sha256")]
    o, s = row.get("originale") or {}, row.get("immagine_shard") or {}
    return [s.get("byte_sha256"), s.get("pixel_sha256"), o.get("pixel_sha256"), row.get("file"), row.get("riga"),
            row.get("didascalia_sha256")]


def row_images(fmt: str, row: dict) -> list[tuple]:
    """(rappresentazione, byte o None, pixel o None, motivo se manca il byte, motivo se manca il pixel)."""
    def one(rep: int, im: dict | None, problem: bool) -> tuple:
        im = im or {}
        if problem:
            return rep, None, None, "problema_della_scansione", "problema_della_scansione"
        read = im.get("lettura")
        b = im.get("byte_sha256")
        p = im.get("pixel_sha256") if read == "letta" else None
        return rep, b, p, None if b else (read or "assente"), None if p else (read if read != "letta" else "assente")

    if fmt == "synth":
        read = row.get("immagine")
        p = row.get("pixel_sha256") if read == "letta" else None
        return [(1, row.get("byte_sha256"), p, None if row.get("byte_sha256") else read or "assente",
                 None if p else (read if read != "letta" else "assente"))]
    if fmt == "panel":
        problem = bool(row.get("problemi"))
        out = [one(0, row.get("originale"), problem), one(1, row.get("png"), False)]
        for c in row.get("ritagli") or []:
            out.append((2 + int(c["riga"]), None, c.get("pixel_sha256"), "non_applicabile",
                        None if c.get("pixel_sha256") else "assente"))
        return out
    return [one(0, row.get("originale"), False), one(1, row.get("immagine_shard"), False)]


def read_scan(name: str, info: dict, store: Store, limit, checks: Checks) -> dict:
    t0 = time.time()
    meta, fmt = info["meta"], SCAN_FORMAT[name]
    rec_entry, sum_entry = scan_entries(name, meta)
    if not rec_entry.get("path") or not sum_entry.get("path"):
        raise Failed(f"{name}: il manifest non cita una scansione")
    sum_path, rec_path = Path(sum_entry["path"]), Path(rec_entry["path"])
    raw = sum_path.read_bytes()
    summary = json.loads(raw)
    sum_sha = hashlib.sha256(raw).hexdigest()
    if sum_sha != sum_entry.get("sha256"):
        raise Failed(f"{sum_path}: sha256 {sum_sha}, il manifest registra {sum_entry.get('sha256')}")
    convention = summary.get("convenzione_immagini")
    pixels_ok = convention == IMAGE_CONVENTION  # altrimenti i pixel di questa scansione non si confrontano
    print(f"{name}: scansione nella convenzione {convention!r}: pixel {'confrontati' if pixels_ok else 'NON confrontati'}",
          flush=True)
    if fmt == "synth":
        declared_rows = summary.get("samples")
    elif fmt == "panel":
        declared_rows = (summary.get("manifest") or {}).get("righe")
        scanned = (summary.get("manifest") or {}).get("manifest_sha256")
        known = lineage(meta) | {info["manifest"]["sha256"]}
        checks(f"{name}: scansione completa di un manifest della catena di {info['manifest_nome']}",
               summary.get("parziale") is False and scanned in known, {"manifest_scansionato": scanned})
    else:
        declared_rows = summary.get("righe")
        checks(f"{name}: scansione completa (non parziale)", summary.get("parziale") is False,
               {"parziale": summary.get("parziale")})
    entries = info["entries"]
    seen = bytearray(len(store.src))
    bad, coverage, outside = Problems(), defaultdict(Counter), Counter()
    rows = GzRows(rec_path, limit)
    discards = set(info["scarti"])
    names = dict(info["shard_of_name"]) if fmt == "pixmo" else {}
    for row in rows:
        shard, pos, key, ins = row_identity(fmt, row, names)
        if fmt == "pixmo" and shard is None:
            if key in discards:
                outside["scarto dello sharding (nessuna voce del manifest)"] += 1
            else:
                bad("riga senza shard che non è uno scarto dello sharding", key)
            continue
        n = entries.get((shard, pos))
        if n is None:
            if rows.limit is not None:
                outside["fuori dalla prova limitata"] += 1
            else:
                bad("riga senza voce del manifest", (shard, pos, key))
            continue
        e_ins, _, _, e_key = store.ref[n]
        if e_key != key or (ins is not None and ins != e_ins):
            bad("chiave o insieme diversi da quelli della voce", (shard, pos, key))
            continue
        if seen[n]:
            bad("voce con due righe", (shard, pos, key))
            continue
        seen[n] = 1
        if link_digest(row_link(fmt, row)) != store.link[n]:
            bad("digest della riga diversi da quelli della voce", (shard, pos, key))
            continue
        for rep, b, p, why_b, why_p in row_images(fmt, row):
            where = f"{e_ins}/{rep_kind(rep)}"
            if why_b != "non_applicabile":
                if b and mf.HEX64.match(b):
                    store.add_digest(b"B", b, n, rep)
                    coverage[f"{where}/byte"]["disponibile"] += 1
                else:
                    coverage[f"{where}/byte"][why_b or "digest_non_valido"] += 1
            if not pixels_ok:
                coverage[f"{where}/pixel"]["convenzione_incompatibile"] += 1
            elif p and mf.HEX64.match(p):
                store.add_digest(b"P", p, n, rep)
                coverage[f"{where}/pixel"]["disponibile"] += 1
            else:
                coverage[f"{where}/pixel"][why_p or "digest_non_valido"] += 1
            if fmt == "panel" and e_ins == "righe" and rep == 1 and not row.get("ritagli"):
                coverage[f"{e_ins}/ritaglio/pixel"]["ritagli_non_costruiti"] += 1
    mine = [n for n in entries.values()]
    missing = [n for n in mine if not seen[n]]
    complete = rows.complete
    checks(f"{name}: scansione con gli sha256 registrati nel manifest (records.jsonl.gz e summary.json)",
           rows.sha256 == rec_entry.get("sha256"), {"records": rows.sha256, "summary": sum_sha})
    checks(f"{name}: ogni riga della scansione ricollegata a una sola voce coi digest del manifest", not bad.n,
           bad.detail() or None)
    if complete and limit is None:
        checks(f"{name}: ogni voce del manifest con una riga della scansione (righe dichiarate dalla scansione "
               f"{declared_rows})", not missing and rows.rows == declared_rows,
               {"voci_senza_riga": len(missing), "righe": rows.rows, "esempi": [store.ref[n] for n in missing[:5]]})
    for n in missing:
        coverage[f"{store.ref[n][0]}/voce/scansione"]["senza_riga_della_scansione"] += 1
    for n in mine:  # i campi del collegamento e la mappa delle posizioni non servono più
        store.link[n] = None
    info["entries"] = {}
    return {"records": rows.entry(declared_rows), "summary": {"path": um.rel(sum_path), "bytes": len(raw), "sha256": sum_sha},
            "schema": summary.get("schema") or f"decontam.py scan ({summary.get('kind')})",
            "job": summary.get("job_id"), "convenzione": convention, "pixel_confrontati": pixels_ok,
            "copertura": {k: dict(sorted(v.items())) for k, v in sorted(coverage.items())},
            "righe_fuori_confronto": dict(outside), "voci_senza_riga": len(missing),
            "secondi_scansione": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- componenti

def signature(store: Store, key: bytes, codes: list[int]) -> str:
    parts = sorted({f"{SOURCES[ORDER[store.src[c >> SHIFT]]]}/{rep_kind(c & ((1 << SHIFT) - 1))}" for c in codes})
    return f"{KIND[key[:1]]}: " + " = ".join(parts)


def member(store: Store, info: dict, n: int) -> dict:
    name = ORDER[store.src[n]]
    ins, shard, pos, key = store.ref[n]
    tasks = store.tasks[n]
    out = {"sorgente": SOURCES[name], "manifest": info[name]["manifest_nome"], "revisione": info[name]["revisione"],
           "insieme": ins, "shard": shard, "indice": pos, "chiave": key, "record_id": store.rid[n],
           "gruppo": store.groups[store.grp[n]][1], "split": store.split[n], "stato": store.state[n],
           "compiti": [{"compito": t, "stato": st, "utilizzabile": ok} for t, st, ok in tasks],
           "caricabile": loadable(store.state[n], store.split[n], tasks)}
    if store.motive[n] is not None:
        texts, kinds = store.motive[n]
        out.update(motivo=list(texts), corrispondenze_decontaminazione=list(kinds), portata=scope(texts, kinds))
    return out


def classify(members: list[dict]) -> str:
    train = {m["gruppo"] for m in members if m["caricabile"] and m["split"] == "train"}
    dev = {m["gruppo"] for m in members if m["caricabile"] and m["split"] == "dev"}
    if train and dev:
        return "train_dev"
    if any(m["split"] == "test" for m in members) and (train or dev):
        return "riservati"
    if len(train) >= 2:
        return "solo_train"
    if len(dev) >= 2:
        return "solo_dev"
    if train or dev:
        return "un_gruppo_caricabile"
    return "non_caricabili"


def describe_component(number: int, members: list[dict], classes: list[dict]) -> dict:
    by_source: dict[str, list] = defaultdict(list)
    for m in members:
        by_source[m["sorgente"]].append(m)
    groups: dict[str, dict] = {}
    for m in members:
        g = groups.setdefault(m["gruppo"], {"sorgente": m["sorgente"], "gruppo": m["gruppo"], "split": m["split"],
                                            "voci": 0, "record": set(), "stati": Counter(), "caricabili": 0})
        g["voci"] += 1
        g["record"].add(m["record_id"])
        g["stati"][f"{m['insieme']}/{m['stato']}"] += 1
        g["caricabili"] += m["caricabile"]
    direct = {(c["sorgente"], c["insieme"], c["shard"], c["indice"]) for cl in classes for c in cl["voci"]}
    not_loadable = [m for m in members if not m["caricabile"]]
    motives = Counter((m["sorgente"], m["split"], m["stato"], tuple(m.get("motivo") or ()),
                       tuple(m.get("portata") or ())) for m in not_loadable)
    category = classify(members)
    return {
        "numero": number, "nota": "numero d'ordine nel rapporto, non un identificatore",
        "categoria": category, "fra_sorgenti": len(by_source) > 1, "sorgenti": sorted(by_source),
        "con_riservati": any(m["split"] == "test" for m in members),
        "gruppi": [{**{k: v for k, v in g.items() if k not in ("record", "stati")}, "record": len(g["record"]),
                    "stati": dict(sorted(g["stati"].items()))} for g in sorted(groups.values(),
                                                                                key=lambda x: (x["sorgente"], x["gruppo"]))],
        "voci": len(members), "voci_per_sorgente": {s: len(ms) for s, ms in sorted(by_source.items())},
        "record_per_sorgente": {s: len({m["record_id"] for m in ms}) for s, ms in sorted(by_source.items())},
        "voci_con_legame_diretto": len(direct),
        "split_e_stati": dict(sorted(Counter(f"{m['sorgente']}/{m['insieme']}/{m['split']}/{m['stato']}"
                                             for m in members).items())),
        "compiti": dict(sorted(Counter(f"{m['sorgente']}/{t['compito']}/{m['split']}/{t['stato']}"
                                       for m in members for t in m["compiti"]).items())),
        "caricabili": {"train": sum(m["caricabile"] and m["split"] == "train" for m in members),
                       "dev": sum(m["caricabile"] and m["split"] == "dev" for m in members)},
        "non_caricabili": [{"sorgente": s, "split": sp, "stato": st, "motivo": list(mo), "portata": list(sc), "voci": c}
                           for (s, sp, st, mo, sc), c in sorted(motives.items())],
        "rappresentazioni": dict(sorted(Counter(cl["firma"] for cl in classes).items())),
        "classi": classes, "membri": members,
        "proposta": PROPOSALS[category]}


def build_components(store: Store, info: dict) -> dict:
    t0 = time.time()
    new, links = [], []
    within, same_node = Counter(), Counter()
    for key, codes in store.dups.items():
        nodes = {c >> SHIFT for c in codes}
        if len(nodes) < 2:
            same_node[signature(store, key, codes)] += 1
            continue
        groups = sorted({store.grp[n] for n in nodes})
        if len(groups) < 2:
            within[signature(store, key, codes)] += 1
            continue
        new.append((key, codes))
        links.extend((groups[0], g) for g in groups[1:])
    rep = um.components(len(store.groups), links)
    size = Counter(rep)
    interesting = {r for r, c in size.items() if c > 1}
    del size
    nodes_of: dict[int, list[int]] = defaultdict(list)
    if interesting:
        for n, g in enumerate(store.grp):
            if rep[g] in interesting:
                nodes_of[rep[g]].append(n)
    classes_of: dict[int, list] = defaultdict(list)
    for key, codes in new:
        classes_of[rep[store.grp[codes[0] >> SHIFT]]].append((key, codes))
    comps = []
    order = sorted(interesting, key=lambda r: (-len(nodes_of[r]), store.groups[r][1]))
    for number, r in enumerate(order, 1):
        members = [member(store, info, n) for n in nodes_of[r]]
        classes = []
        for key, codes in sorted(classes_of[r]):
            voci = []
            for c in sorted(set(codes)):
                n, rp = c >> SHIFT, c & ((1 << SHIFT) - 1)
                ins, shard, pos, k = store.ref[n]
                voci.append({"sorgente": SOURCES[ORDER[store.src[n]]], "insieme": ins, "shard": shard, "indice": pos,
                             "chiave": k, "rappresentazione": rep_name(rp)})
            classes.append({"tipo": KIND[key[:1]], "digest": key[1:].hex(), "firma": signature(store, key, codes),
                            "gruppi": len({store.grp[c >> SHIFT] for c in codes}), "voci": voci})
        comps.append(describe_component(number, members, classes))
    return {"componenti": comps, "classi_nuove": len(new), "legami_nei_gruppi": dict(sorted(within.items())),
            "stessa_voce": dict(sorted(same_node.items())), "classi_con_ripetizioni": len(store.dups),
            "digest_distinti": len(store.first), "gruppi": len(store.groups), "secondi": round(time.time() - t0, 1)}


def pair_stats(comps: list[dict], names: list[str]) -> dict:
    """Per coppia di sorgenti: componenti, classi (digest uguali) e record con un legame diretto o nella componente."""
    sources = [SOURCES[n] for n in names]
    out = {}
    for a, b in combinations(sources, 2):
        sel = [c for c in comps if a in c["sorgenti"] and b in c["sorgenti"]]
        classes = [cl for c in sel for cl in c["classi"] if {a, b} <= {v["sorgente"] for v in cl["voci"]}]
        direct = {s: set() for s in (a, b)}
        for c in sel:
            rid = {(m["sorgente"], m["insieme"], m["shard"], m["indice"]): m["record_id"] for m in c["membri"]}
            for cl in c["classi"]:
                if {a, b} <= {v["sorgente"] for v in cl["voci"]}:
                    for v in cl["voci"]:
                        if v["sorgente"] in direct:
                            direct[v["sorgente"]].add(rid[(v["sorgente"], v["insieme"], v["shard"], v["indice"])])
        out[f"{a} + {b}"] = {
            "componenti": len(sel), "classi_dirette": len(classes),
            "record_con_legame_diretto": {s: len(x) for s, x in direct.items()},
            "record_nelle_componenti": {s: len({m["record_id"] for c in sel for m in c["membri"] if m["sorgente"] == s})
                                        for s in (a, b)},
            "categorie": dict(Counter(c["categoria"] for c in sel))}
    for s in sources:
        sel = [c for c in comps if c["sorgenti"] == [s]]
        out[f"{s} (dentro la sorgente)"] = {"componenti": len(sel), "categorie": dict(Counter(c["categoria"] for c in sel)),
                                            "record_nelle_componenti": len({m["record_id"] for c in sel
                                                                            for m in c["membri"]})}
    return out


def compact(c: dict, members: int) -> dict:
    out = {k: v for k, v in c.items() if k not in ("membri", "classi")}
    out["membri"] = c["membri"][:members]
    out["membri_omessi"] = max(0, len(c["membri"]) - members)
    out["classi"] = [{**cl, "voci": cl["voci"][:members], "voci_omesse": max(0, len(cl["voci"]) - members)}
                     for cl in c["classi"][:members]]
    out["classi_omesse"] = max(0, len(c["classi"]) - members)
    return out


LIMITS = [
    "solo identità esatte: nessun pHash, nessuna somiglianza, nessun confronto di testi, formule, domande o risposte; la "
    "stessa immagine è un legame visivo, non dice se domanda o risposta coincidono",
    "immagine data al modello dopo il ridimensionamento del processor (tensore delle patch): non registrata. Due "
    "immagini con pixel uguali danno lo stesso ingresso solo con lo stesso numero massimo di patch; pixel diversi "
    "possono dare ingressi quasi uguali (per esempio la stessa immagine a due risoluzioni): non si vedono. Recupero: "
    "decodifica e ridimensionamento di tutte le immagini, non eseguito",
    "ritagli di UniMER: solo pixel, perché non sono file; lo stesso contenuto con margini diversi non si riconosce",
    "MAVIS: text.jpg e vision_dominant.jpg non sono negli shard e non si confrontano; PixMo: solo i 25 parquet locali "
    "su 75 (33,34% della scheda), le altre righe non sono confrontate",
    "pixel nella convenzione rgb-su-bianco/1 senza orientamento EXIF: due immagini uguali dopo la rotazione EXIF ma con "
    "byte diversi non si riconoscono",
    "tutte le scansioni sono state fatte con lo stesso ambiente (venvs/current -> build-29382) e quindi lo stesso "
    "decodificatore di Pillow; un decodificatore diverso potrebbe dare pixel diversi per gli stessi byte JPEG",
    "LLaVA-Pretrain, MMFineReason e le altre sorgenti locali sono fuori da questo lotto",
]


def estimate(info: dict, scans: dict, store: Store, rss: float, seconds: float) -> dict:
    """Stima del lavoro completo da una prova limitata: tempo per riga di ogni file (righe interpretate per il rapporto
    fra righe totali e lette, più la lettura per lo sha256, che nella prova copre già il resto del file) e memoria per
    voce."""
    files, total = [], 0.0
    for name, i in info.items():
        for what, x in (("manifest", i["records"]), ("scansione", scans[name]["records"])):
            ratio = (x["righe_dichiarate"] or x["righe_lette"]) / max(1, x["righe_lette"])
            est = x["secondi_righe"] * ratio + x["secondi_solo_sha256"]
            total += est
            files.append({"sorgente": name, "file": what, "righe_lette": x["righe_lette"],
                          "righe_totali": x["righe_dichiarate"], "secondi_righe": x["secondi_righe"],
                          "secondi_solo_sha256": x["secondi_solo_sha256"], "secondi_stimati": round(est, 1)})
    voci_tot = sum(i["records"]["righe_dichiarate"] or 0 for i in info.values())
    return {"file": files, "secondi_letture_stimati": round(total, 1), "voci_lette": len(store.src),
            "voci_totali": voci_tot, "rss_mib_prova": rss,
            "rss_mib_stimato": round(rss * max(1.0, voci_tot / max(1, len(store.src))), 1),
            "nota": "stima lineare per riga e per voce; la memoria stimata è un tetto, perché l'interprete e gli "
                    "indici dei file non crescono con le voci; le componenti non sono comprese (dipendono dai legami "
                    "trovati)", "secondi_prova_totali": round(seconds, 1)}


# ---------------------------------------------------------------- principale

def run(args) -> int:
    t0 = time.time()
    names = [m[0] for m in args.manifest]
    if not names or len(set(names)) != len(names) or any(n not in SOURCES for n in names):
        raise Failed(f"sorgenti {names}: ognuna al più una volta, fra {list(SOURCES)}")
    names = [n for n in ORDER if n in names]
    folders = {m[0]: (Path(m[1]), m[2]) for m in args.manifest}
    out = Path(args.out)
    if out.exists():
        raise Failed(f"{out} esiste già: le uscite si scrivono in una cartella nuova")
    if args.summary and Path(args.summary).exists():
        raise Failed(f"{args.summary} esiste già")
    checks, store = Checks(), Store()
    info, scans = {}, {}
    for name in names:
        info[name] = read_manifest(name, *folders[name], args.limit, store, checks)
        print(f"{name}: {info[name]['voci']} voci in {info[name]['secondi_manifest']} s, RSS {um.rss_mib()} MiB",
              flush=True)
    c9 = None
    if args.c9:
        if "mavis" not in info:
            raise Failed("--c9 richiede la sorgente mavis")
        c9 = check_c9(Path(args.c9[0]), args.c9[1], info["mavis"], checks)
    for name in names:
        scans[name] = read_scan(name, info[name], store, args.limit, checks)
        print(f"{name}: scansione in {scans[name]['secondi_scansione']} s, {len(store.first)} digest distinti, "
              f"RSS {um.rss_mib()} MiB", flush=True)
    built = build_components(store, info)
    comps = built.pop("componenti")
    rss = um.rss_mib()
    partial = args.limit is not None
    where = {True: "fra_sorgenti", False: "dentro_una_sorgente"}
    categories = {w: dict(Counter(c["categoria"] for c in comps if where[c["fra_sorgenti"]] == w))
                  for w in where.values()}
    td = [c for c in comps if c["categoria"] == "train_dev"]
    conflicts = {"componenti": len(td), "fra_sorgenti": sum(c["fra_sorgenti"] for c in td),
                 "record_caricabili_coinvolti": {"train": sum(c["caricabili"]["train"] for c in td),
                                                 "dev": sum(c["caricabili"]["dev"] for c in td)},
                 "per_sorgenti": dict(Counter(" + ".join(c["sorgenti"]) for c in td))}
    blocked = [x for c in comps for x in c["non_caricabili"]]
    with_blocked = {"componenti": sum(1 for c in comps if c["non_caricabili"]),
                    "voci_non_caricabili": sum(x["voci"] for x in blocked),
                    "per_sorgente_split_stato": dict(Counter(f"{x['sorgente']}/{x['split']}/{x['stato']}"
                                                             for x in blocked for _ in range(x["voci"]))),
                    "per_portata": dict(Counter(p for x in blocked for p in x["portata"] for _ in range(x["voci"])))}

    def per_category(members: int, k: int | None) -> dict:
        res = {}
        for cat in CATEGORIES:
            for flag, w in where.items():
                sel = [c for c in comps if c["categoria"] == cat and c["fra_sorgenti"] == flag]
                if sel:
                    res[f"{cat}/{w}"] = {"componenti": len(sel), "elenco": [compact(c, members) for c in sel[:k]],
                                         "omesse": max(0, len(sel) - k) if k else 0}
        return res

    report = {
        "schema": SCHEMA, "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
        "parziale": partial, "limite": args.limit,
        "perimetro": "identità esatte (byte e pixel nella stessa convenzione) fra sintetici v4, UniMER, MAVIS e PixMo dai "
                     "manifest e dalle scansioni esistenti; diagnostica, nessuna modifica a gruppi, split o stati; non è "
                     "l'indice globale di C6 né la decontaminazione globale",
        "rappresentazioni": REPRESENTATIONS, "tipi": TYPES,
        "ingressi": {n: {**{k: v for k, v in info[n].items() if k not in ("meta", "entries", "shard_of_name", "scarti")},
                         "scansione": {k: v for k, v in scans[n].items() if k != "copertura"},
                         "scarti_dello_sharding": len(info[n]["scarti"])} for n in names},
        "c9": c9,
        "controlli": checks.items, "falliti": checks.failed(),
        "copertura": {n: scans[n]["copertura"] for n in names},
        "copertura_nota": "per voce e rappresentazione (i ritagli per ritaglio): `disponibile` = digest confrontato con "
                          "tutti quelli dello stesso tipo; gli altri esiti sono controlli non eseguibili. Il pixel_sha256 "
                          "del manifest dei sintetici (RGBA diretto) non si usa: convenzione diversa",
        "classi": built,
        "componenti_con_piu_gruppi": len(comps),
        "categorie": categories,
        "coppie_di_sorgenti": pair_stats(comps, names),
        "conflitti_train_dev": conflicts,
        "legami_con_record_non_caricabili": with_blocked,
        "proposte": PROPOSALS, "limiti": LIMITS,
        "costi": {"secondi": round(time.time() - t0, 1), "rss_max_mib": rss,
                  "byte_letti_compressi": sum(i["manifest"]["bytes"] + (i["records"]["bytes"] or 0) for i in info.values())
                  + sum(s["summary"]["bytes"] + (s["records"]["bytes"] or 0) for s in scans.values())
                  + (c9["manifest"]["bytes"] if c9 else 0),
                  "byte_decompressi_interpretati": sum(i["records"]["byte_decompressi"] for i in info.values())
                  + sum(s["records"]["byte_decompressi"] for s in scans.values()),
                  "immagini_lette": 0},
    }
    if partial:
        report["stima_completa"] = estimate(info, scans, store, rss, time.time() - t0)
    out.mkdir(parents=True)
    lines = (json.dumps(c, ensure_ascii=False, sort_keys=True) for c in comps)
    sha, n = um.write_records_gz(out / "componenti.jsonl.gz", lines)
    report["files"] = {"componenti.jsonl.gz": {"path": um.rel(out / "componenti.jsonl.gz"), "sha256": sha, "righe": n}}
    write_new_json(out / "report.json", {**report, "componenti": per_category(REPORT_MEMBERS, None)})
    if args.summary:
        write_new_json(args.summary, {**report, "componenti": per_category(SUMMARY_MEMBERS, SUMMARY_PER_CATEGORY),
                                      "nota_componenti": f"al più {SUMMARY_PER_CATEGORY} componenti per categoria e "
                                                         f"{SUMMARY_MEMBERS} voci e classi per componente; tutte in "
                                                         "componenti.jsonl.gz (files)"})
    print(json.dumps({"componenti_con_piu_gruppi": len(comps), "categorie": categories,
                      "falliti": checks.failed(), "secondi": report["costi"]["secondi"], "rss_max_mib": rss},
                     ensure_ascii=False), flush=True)
    return 1 if checks.failed() else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs=3, action="append", required=True, metavar=("NOME", "CARTELLA", "SHA256"))
    ap.add_argument("--c9", nargs=2, metavar=("CARTELLA", "SHA256"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    try:
        code = run(args)
    except (Failed, mf.ManifestError) as exc:
        sys.exit(f"fermato: {exc}")
    sys.exit(code)


if __name__ == "__main__":
    main()
