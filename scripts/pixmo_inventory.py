#!/usr/bin/env python3
"""PixMo-Cap: inventario tecnico del sottoinsieme locale, collegamento originale -> shard, riproduzione dei segnali
storici di outputs/pixmo_check/run4 e decontaminazione contro pannello-v3, in un solo passaggio sui dati (pipeline
§§5.2-5.3, 5.7-5.8; S1 §§1, 4-5; docs/PANNELLO_VALUTAZIONE.md §6.2). Solo in un job SLURM CPU: snapshot, shard, indice,
pannello ed evidenze storiche si leggono soltanto; si scrive solo in cartelle nuove. Nessun gruppo, split, stato o
manifest delle ammissioni: le uscite sono evidenze, e le proposte di trattamento non sono lette da nessun loader.

Ingressi: lo snapshot anthracite-org/pixmo-cap-images (parquet `data/train-*-of-00075.parquet` con image, image_url,
caption e transcripts; README) e la cartella dello sharding (data/shards/pixmo-cap/3e42775e: plan.json, shards.jsonl,
manifest.json e i tar del job 29560, scripts/shard_parquet.py --kind pixmo); l'indice pannello-v3 (convenzione
rgb-su-bianco/1) e i suoi gruppi.

Identità e collegamento. La chiave `pixmo-<file>-<riga>` è quella di shard_parquet.py: <file> è la posizione del parquet
nell'elenco del piano, <riga> la riga nel parquet, contata anche per le righe scartate. Ogni riga del parquet ha una riga
dell'inventario, nell'ordine dei file e delle righe: file, riga e gruppo di righe; record dello shard (nome, posizione,
membri) o il motivo per cui manca (gli scarti di shard_parquet: immagine o didascalia mancante, immagine illeggibile);
record_id di S1 §1 (sorgente pixmo-cap, revisione di 40 caratteri, record = chiave). Digest:
- originale: sha256 dei byte, sha256 dei pixel e pHash nella convenzione dell'indice, dimensioni, formato, modo,
  orientamento EXIF diverso da 1 e fotogrammi se presenti;
- shard: sha256 del JSON e del JPEG, sha256 dei pixel, pHash e dimensioni del JPEG decodificato;
- trasformazione: quella di shard_parquet.convert (processing.to_rgb, target_size a 1024 patch, BILINEAR, JPEG 95)
  rifatta sull'originale e confrontata coi byte dello shard. Il JPEG dello shard è una copia ridimensionata e
  ricodificata: non si chiede che abbia i byte o i pixel dell'originale;
- testi: didascalia originale e del loader (data.pixmo_caption_sample legge `caption` del JSON, cioè l'originale senza
  spazi ai bordi), trascrizioni (numero e sha256), URL d'origine, richiesta del loader (indice in
  data.DETAILED_CAPTION_PROMPTS scelto da pick_prompt).
I difetti strutturali dimostrati stanno in `difetti`, separati dai segnali delle didascalie.

Segnali delle didascalie: transcripts.compare su didascalia originale e trascrizioni, come
scripts/pixmo_transcript_check.py (run4): numeri, colori e testi citati che nessuna trascrizione contiene. Sono segnali di
rischio, non difetti: l'assenza nelle trascrizioni non prova che il dettaglio sia falso nell'immagine.

Decontaminazione, con le regole documentate e gli helper di panel_decontam.py (strutture dell'indice, image_check,
text_hits, entry_ref): identità dei pixel normalizzati dell'originale e del JPEG dello shard contro ogni voce
dell'indice, in ogni ruolo; pHash entro 4 bit delle due immagini; 10-gram di normalize_words della didascalia del loader
contro i testi dell'indice (dal lato dell'indice senza gli n-gram in più di 25 testi diversi del pannello, regola di
TextIndex). Nessuna regola copre le didascalie: ogni corrispondenza testuale resta un candidato, qualunque sia la sua
frequenza in PixMo. Le sei richieste fisse del loader si confrontano una volta, con i 10-gram e col testo esatto
(decontam.exact_text). Esiti di una riga (calcolati dalla verifica): corrispondenza_confermata (pixel identici, legame
sull'immagine e non sulla domanda), candidato (pHash o testo, da verificare), non_verificabile (un controllo non
eseguibile), nessuna_corrispondenza (nei controlli eseguiti; non certifica la didascalia). Tutte le corrispondenze si
conservano, senza tagli.

Comandi
  scan      un passaggio: inventario -> --inventory (records.jsonl.gz, summary.json), scansione -> --scan
            (records.jsonl.gz, summary.json). --files N: solo i primi N parquet del piano (sonda).
  verifica  rilegge le uscite e controlla ingressi, sharding, copertura, chiavi, collegamento, trasformazione, scarti,
            duplicati, riproduzione di run4, liste storiche p6/f1eval e corrispondenze col pannello -> --report
            (report.json, duplicati.jsonl.gz, proposte_di_trattamento.jsonl.gz). --partial per la sonda.

    srun $RUN_PYTHON -u -B scripts/pixmo_inventory.py scan --snapshot DIR --shards DIR --revision HEX --index DIR \\
        --index-sha256 HEX --inventory DIR --scan DIR [--files N] [--workers N]
    srun $RUN_PYTHON -u -B scripts/pixmo_inventory.py verifica --snapshot DIR --shards DIR --revision HEX --index DIR \\
        --index-sha256 HEX --panel DIR --inventory DIR --scan DIR --run4 DIR --historical DIR [DIR ...] \\
        --verify-report TSV --report DIR [--partial]
"""
from __future__ import annotations

import argparse
import bisect
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
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import panel_decontam as pd  # noqa: E402
import synth_decontam as sd  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl import decontam as dc  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.data import DETAILED_CAPTION_PROMPTS, pick_prompt  # noqa: E402
from limite_vl.processing import DEFAULT_MAX_PATCHES, exact_patches, target_size  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, sha256_file, write_new_json  # noqa: E402
from limite_vl.transcripts import compare, number_kind  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # come shard_parquet.py: foto grandi, file verificati

SOURCE = "pixmo-cap"  # sorgente del record_id (S1 §1), come in data_lists e nell'inventario degli artefatti
REPO = "anthracite-org/pixmo-cap-images"
UPSTREAM = "allenai/pixmo-cap"  # didascalie e trascrizioni di Ai2; immagini riscaricate da anthracite-org dagli URL
SCHEMA_INVENTORY = "pixmo-inventory/1"
SCHEMA_SCAN = "pixmo-panel-scan/1"
SCHEMA_REPORT = "pixmo-inventory-verifica/1"
COLUMNS = ["image", "image_url", "caption", "transcripts"]
CONV = dc.IMAGE_CONVENTION
BATCH = 256  # righe per blocco del pool (circa 135 MB di immagini originali)
JPEG_QUALITY = 95  # shard_parquet.convert
REQUIRED = ("pixel_originale", "pixel_shard", "phash_originale", "phash_shard", "testo_didascalia")
OUTCOMES = ("corrispondenza_confermata", "candidato", "non_verificabile", "nessuna_corrispondenza")
KEY = re.compile(r"^pixmo-(\d{5})-(\d{6})$")
EXIF_ORIENTATION = 0x0112
NOT_VERIFIED = [
    "fedeltà della didascalia all'immagine: oggetti, posizioni, relazioni e attributi aggiunti dal riscrittore, ed errori "
    "percettivi degli annotatori (serve un campione con le immagini, D5)",
    "corrispondenza fra l'immagine riscaricata da anthracite-org dall'URL e quella vista dagli annotatori di Ai2 "
    "(collegamenti scaduti, segnaposto, immagini cambiate); i duplicati per byte con didascalie diverse sono solo un "
    "segnale",
    "rappresentatività del sottoinsieme locale: è il prefisso dei file 0-24 su 75, non un campione casuale",
    "orientamento: né lo sharding né l'indice applicano l'orientamento EXIF; per le immagini con un orientamento diverso "
    "da 1 non è verificato quale versione abbiano visto gli annotatori",
    "leggibilità dei dettagli alla risoluzione dello shard (1024 patch): non controllata",
]
NOT_RUN = [
    "trascrizioni contro il pannello: non sono negli shard e il modello non le vede",
    "URL e provenienza dichiarata: il pannello non registra URL, e PixMo non dichiara legami coi benchmark",
    "ritagli, ridisegni, rotazioni o riformulazioni delle immagini del pannello: solo pixel identici e pHash entro 4 bit",
    "didascalie più corte di 10 parole: nessun 10-gram (nel sottoinsieme sono contate in testi.confrontabili)",
]


class Failed(Exception):
    """Ingresso mancante o incoerente: niente uscite."""


def sha(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def key_order(key: str | None) -> tuple[int, int] | None:
    m = KEY.match(key or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


class GzLines:
    """Righe JSON in un gzip riproducibile (senza nome né data), scritto in streaming; il file non deve esistere.
    close() chiude il gzip (coda del formato) e poi il file, prima che se ne calcoli lo sha256."""

    def __init__(self, path: Path):
        if path.exists():
            raise Failed(f"{path} esiste già")
        self.raw = open(path, "wb")
        self.text = io.TextIOWrapper(gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, compresslevel=6,
                                                   mtime=0), encoding="utf-8")

    def write(self, line: str) -> None:
        self.text.write(line)

    def close(self) -> None:
        self.text.close()
        self.raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def cause(text: str) -> str:
    """Tipo di un difetto senza chiavi e numeri, per contarli."""
    return re.sub(r"\d+", "N", text)


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- ingressi: sharding e snapshot

def read_sharding(shards: Path, revision: str) -> dict:
    """Piano, registro e manifest dello sharding (shard_parquet.py), coerenti fra loro e con la revisione attesa."""
    for name in ("plan.json", "shards.jsonl", "manifest.json"):
        if not (shards / name).is_file():
            raise Failed(f"manca {shards / name}: non è una cartella di shard_parquet.py")
    plan = json.loads((shards / "plan.json").read_text())
    log = [json.loads(x) for x in (shards / "shards.jsonl").read_text().splitlines() if x.strip()]
    meta = json.loads((shards / "manifest.json").read_text())
    if plan.get("kind") != "pixmo" or plan.get("source") != f"{REPO}@{revision}":
        raise Failed(f"piano di {plan.get('kind')!r} da {plan.get('source')!r}, atteso pixmo da {REPO}@{revision}")
    doc = {k: plan[k] for k in ("kind", "source", "prefix", "files_per_shard", "files")}
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()  # shard_parquet.main
    per = plan["files_per_shard"]
    names = [f"{plan['prefix']}-{i:05d}.tar" for i in range((len(plan["files"]) + per - 1) // per)]
    by_name = {}
    for row in log:  # una riga per shard scritto; una ripresa rilegge la riga registrata
        by_name.setdefault(row["name"], row)
    units = []
    for s, name in enumerate(names):
        row = by_name.get(name)
        if row is None:
            raise Failed(f"{name} del piano non registrato in shards.jsonl")
        blobs = plan["files"][s * per:(s + 1) * per]
        if len(row["files"]) != len(blobs):
            raise Failed(f"{name}: {len(row['files'])} parquet registrati, {len(blobs)} nel piano")
        files = [{"indice_file": s * per + j, "snapshot": Path(p), "blob_atteso": b}
                 for j, (p, b) in enumerate(zip(row["files"], blobs))]
        units.append({"nome": name, "tar": shards / name, "registro": row, "parquet": files})
    return {"piano": plan, "digest_ricalcolato": digest, "registro": log, "manifest": meta, "unita": units,
            "sorgente": plan["source"], "revisione": revision}


def snapshot_file(snapshot: Path, logged: Path) -> Path:
    """Il parquet registrato dallo sharding, ritrovato nello snapshot indicato (stesso percorso relativo a `data/`)."""
    path = snapshot / "data" / logged.name
    if not path.is_file():
        raise Failed(f"parquet mancante: {path} (registrato come {logged})")
    return path


# ---------------------------------------------------------------- esame di una riga (processi del pool)

def original_check(data: bytes | None) -> tuple[dict, Image.Image | None]:
    """Originale nella convenzione dell'indice (come decontam.hash_image_bytes, che non restituisce l'immagine): esito
    della lettura, digest, dimensioni, formato e modo; l'immagine normalizzata serve anche a rifare la trasformazione."""
    if not data:  # come shard_parquet.rows_of: byte assenti o vuoti sono un'immagine mancante
        return {"lettura": "assente"}, None
    out = {"lettura": "letta", "byte_sha256": sha(data), "byte": len(data)}
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()  # come shard_parquet.convert: un'immagine troncata è illeggibile
            out.update(formato=im.format, modo=im.mode, dimensioni=[im.width, im.height])
            try:  # metadati: un EXIF guasto non rende illeggibile un'immagine che lo sharding ha letto
                orientation = im.getexif().get(EXIF_ORIENTATION)
                frames = getattr(im, "n_frames", 1)
            except Exception as exc:  # noqa: BLE001
                orientation, frames = None, 1
                out["errore_metadati"] = f"{type(exc).__name__}: {exc}"[:200]
            if orientation not in (None, 1):
                out["exif_orientamento"] = orientation
            if frames != 1:
                out["fotogrammi"] = frames
            norm = dc.normalize_image(im, CONV)
        out["pixel_sha256"] = dc.pixel_sha256(norm)
        out["phash"] = f"{dc.phash(norm):016x}"
        return out, norm
    except Exception as exc:  # noqa: BLE001 - illeggibile: registrata col motivo, come nello sharding
        return {"lettura": "illeggibile", "byte_sha256": out["byte_sha256"], "byte": len(data),
                "errore": f"{type(exc).__name__}: {exc}"[:200], "inizio_hex": data[:16].hex()}, None


def transform(norm: Image.Image) -> tuple[dict, bytes]:
    """shard_parquet.convert per --kind pixmo sull'immagine già normalizzata (to_rgb): dimensioni a 1024 patch,
    BILINEAR, JPEG qualità 95."""
    h, w = target_size(norm.height, norm.width, DEFAULT_MAX_PATCHES)
    image = norm.resize((w, h), Image.BILINEAR) if (w, h) != norm.size else norm
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    data = buf.getvalue()
    patches = exact_patches(h, w)
    return {"dimensioni_attese": [w, h], "patch": patches, "idempotente": target_size(h, w, patches) == (h, w),
            "jpeg_rifatto_sha256": sha(data)}, data


def expected_sharding(image: bytes | None, caption, original: dict, rep_error: str | None) -> str:
    """Esito che shard_parquet.rows_of e convert danno alla riga: nello shard o scartata col motivo registrato."""
    missing = [n for n, v in (("image", image), ("caption", caption)) if not v]
    if missing:
        return "scartata:missing_" + "_".join(missing)
    if original["lettura"] != "letta" or rep_error:
        return "scartata:illeggibile"
    return "nello_shard"


def caption_info(caption) -> dict:
    if caption is None:
        kind = "assente"
    elif not isinstance(caption, str):
        kind = "non_testo"
    elif caption == "":
        kind = "vuota"
    elif not caption.strip():
        kind = "solo_spazi"
    else:
        kind = "testo"
    out = {"originale": kind}
    if isinstance(caption, str):
        loader = caption.strip()
        out.update(originale_sha256=mf.sha256_text(caption), spazi_ai_bordi=caption != loader,
                   loader_sha256=mf.sha256_text(loader), caratteri=len(loader), parole=len(loader.split()))
    return out


def signals_of(caption, transcripts: list) -> dict:
    """I segnali di run4 (pixmo_transcript_check.py, schema /4): stessi ingressi, stessi tipi."""
    if not any(t and t.strip() for t in transcripts):
        return {"tipi": [], "senza_trascrizioni": True}
    r = compare(caption or "", transcripts)
    kinds = sorted({f"numero_{number_kind(v)}" for v in r["numbers"]} | ({"colore"} if r["colors"] else set())
                   | ({"testo_citato"} if r["quotes"] else set()))
    return {"tipi": kinds, "numbers": r["numbers"], "colors": r["colors"], "quotes": r["quotes"],
            "controllati": r["checked"]}


def shard_check(key: str, shard: dict, caption, url, original: dict, rep: dict | None, source: str,
                defects: list[str]) -> tuple[dict, dict | None, str | None]:
    """Record dello shard contro la riga: membri, JSON, didascalia del loader, URL, dimensioni e trasformazione.
    Restituisce il record, l'esito della lettura del JPEG e la didascalia del loader."""
    files, names = shard["files"], shard["membri"]
    out = {"nome": shard["nome"], "posizione": shard["posizione"], "membri": names,
           "json_sha256": sha(files.get("json")), "jpg_sha256": sha(files.get("jpg"))}
    if sorted(names) != sorted([f"{key}.jpg", f"{key}.json"]):
        defects.append(f"membri dello shard {names} invece di {key}.jpg e {key}.json")
    try:
        meta = json.loads(files["json"])
    except (KeyError, TypeError, ValueError):
        meta = None
        defects.append("JSON dello shard mancante o non valido")
    img = pd.image_check(files.get("jpg"), CONV, True)
    if img["lettura"] != "letta":
        defects.append(f"JPEG dello shard {img['lettura']}")
    loader_caption = None
    if isinstance(meta, dict):
        loader_caption = meta.get("caption")
        expected = {"id": key, "caption": caption.strip() if isinstance(caption, str) else None, "image_url": url,
                    "format": "JPEG", "stored_resized": True, "source": source}
        if original.get("dimensioni"):
            expected.update(orig_width=original["dimensioni"][0], orig_height=original["dimensioni"][1])
        if rep:
            expected.update(width=rep["dimensioni_attese"][0], height=rep["dimensioni_attese"][1],
                            max_patches=rep["patch"])
        diff = sorted(k for k, v in expected.items() if meta.get(k) != v)
        if diff:
            defects.append("JSON dello shard diverso dalla riga in " + ", ".join(diff))
            out["differenze"] = {k: [meta.get(k), expected[k]] for k in diff if k != "caption"}
        extra = sorted(set(meta) - set(expected) - {"orig_width", "orig_height", "width", "height", "max_patches"})
        if extra:
            defects.append("campi in più nel JSON dello shard: " + ", ".join(extra))
        if isinstance(loader_caption, str) and not loader_caption.strip():
            defects.append("didascalia del loader vuota")
        if img["lettura"] == "letta" and img["dimensioni"] != [meta.get("width"), meta.get("height")]:
            defects.append(f"JPEG {img['dimensioni']} diverso dalle dimensioni del JSON")
    return out, img, loader_caption


def examine(task: tuple) -> tuple[dict, dict, dict]:
    """Una riga: (riga dell'inventario, riga della scansione, tempi in secondi). Un errore imprevisto non ferma la
    scansione: la riga lo registra come difetto, con tutti i controlli non eseguiti, e la verifica lo conta."""
    try:
        return examine_row(task)
    except Exception as exc:  # noqa: BLE001
        file_index, file_name, row, rg, key, _, _, _, url, shard, _ = task
        where = None if shard is None else {k: shard[k] for k in ("nome", "posizione", "membri")}
        inv = {"chiave": key, "parquet": {"file": file_name, "indice_file": file_index, "riga": row,
                                          "gruppo_di_righe": rg[0], "riga_nel_gruppo": rg[1]},
               "originale": {"lettura": "errore"}, "didascalia": {"originale": "errore"}, "url": url,
               "trascrizioni": {"numero": 0, "non_vuote": 0, "sha256": None}, "richiesta": -1,
               "esito_atteso_dello_sharding": "errore", "esito_sharding": "errore_dell_esame", "shard": where,
               "difetti": [f"errore dell'esame: {type(exc).__name__}: {exc}"[:300]], "segnali": {"tipi": []}}
        scan = {"chiave": key, "file": file_name, "riga": row, "shard": where and where["nome"],
                "posizione": where and where["posizione"], "controlli": {c: False for c in REQUIRED},
                "testi": {}, "dettagli": [], "corrispondenze": {}, "ngram_in_comune": []}
        return inv, scan, {}


def examine_row(task: tuple) -> tuple[dict, dict, dict]:
    file_index, file_name, row, rg, key, image, caption, transcripts, url, shard, source = task
    tm = {}
    t0 = time.perf_counter()
    defects: list[str] = []
    original, norm = original_check(image)
    t1 = time.perf_counter()
    tm["originale"] = t1 - t0
    rep, rep_error = None, None
    if norm is not None:
        try:
            rep, _ = transform(norm)
        except Exception as exc:  # noqa: BLE001 - convert l'avrebbe scartata come illeggibile
            rep_error = f"{type(exc).__name__}: {exc}"[:200]
    del norm
    t2 = time.perf_counter()
    tm["trasformazione"] = t2 - t1
    expected = expected_sharding(image, caption, original, rep_error)
    transcripts = transcripts if isinstance(transcripts, list) else []
    inv = {"chiave": key, "parquet": {"file": file_name, "indice_file": file_index, "riga": row,
                                      "gruppo_di_righe": rg[0], "riga_nel_gruppo": rg[1]},
           "metadati_sha256": mf.sha256_text(mf.canonical({"caption": caption if isinstance(caption, str) else None,
                                                           "image_url": url if isinstance(url, str) else None,
                                                           "transcripts": list(transcripts)})),
           "originale": original, "didascalia": caption_info(caption), "url": url,
           "trascrizioni": {"numero": len(transcripts), "non_vuote": sum(bool(t and t.strip()) for t in transcripts),
                            "sha256": mf.sha256_text(mf.canonical(list(transcripts)))},
           "richiesta": DETAILED_CAPTION_PROMPTS.index(pick_prompt(key, DETAILED_CAPTION_PROMPTS)),
           "esito_atteso_dello_sharding": expected}
    if rep_error:
        inv["errore_trasformazione"] = rep_error
    img, loader_caption = None, None
    if shard is not None:
        inv["shard"], img, loader_caption = shard_check(key, shard, caption, url, original, rep, source, defects)
        inv["immagine_shard"] = img
        inv["esito_sharding"] = "nello_shard" if expected == "nello_shard" else "nello_shard_inatteso"
        if expected != "nello_shard":
            defects.append(f"record nello shard per una riga che lo sharding scarta ({expected})")
    else:
        inv["shard"] = None
        inv["esito_sharding"] = expected if expected != "nello_shard" else "mancante_nello_shard"
        if expected == "nello_shard":
            defects.append("riga senza record nello shard")
    if rep is not None:
        inv["trasformazione"] = rep
        if img is not None and img.get("phash") and original.get("phash"):
            rep["byte_uguali_allo_shard"] = rep["jpeg_rifatto_sha256"] == inv["shard"]["jpg_sha256"]
            rep["phash_distanza_originale_shard"] = dc.hamming(int(original["phash"], 16), int(img["phash"], 16))
    if original["lettura"] != "letta":
        defects.append(f"immagine originale {original['lettura']}")
    cap_kind = inv["didascalia"]["originale"]
    if cap_kind != "testo":
        defects.append(f"didascalia originale {cap_kind}")
    inv["difetti"] = defects  # senza trascrizioni i segnali non si calcolano: controllo non eseguibile, non un difetto
    t3 = time.perf_counter()
    tm["shard"] = t3 - t2
    inv["segnali"] = signals_of(caption, transcripts)
    t4 = time.perf_counter()
    tm["segnali"] = t4 - t3
    # ---- decontaminazione: tutte le corrispondenze, nessun taglio
    text = loader_caption if shard is not None else (caption.strip() if isinstance(caption, str) else None)
    found = []
    for where, im in (("originale", original), ("shard", img)):
        if im and im.get("pixel_sha256"):
            found += [{"tipo": "pixel", "immagine": where, **pd.entry_ref(j)}
                      for j in pd.G["by_pixels"].get(im["pixel_sha256"], ())]
    for where, im in (("originale", original), ("shard", img)):
        if im and im.get("phash"):
            for jj, d in pd.G["hidx"].query(int(im["phash"], 16)):
                found.append({"tipo": "phash", "immagine": where, "distanza": d, **pd.entry_ref(pd.G["hash_entry"][jj])})
    comparable = isinstance(text, str) and len(dc.normalize_words(text)) >= pd.NGRAM
    if isinstance(text, str) and text.strip():
        found += [{"tipo": "testo", "campo": "didascalia", "ngram": gs[0], "ngram_tutti": gs, **pd.entry_ref(j)}
                  for j, gs in pd.text_hits(text)]
    scan = {"chiave": key, "file": file_name, "riga": row,
            "shard": inv["shard"]["nome"] if inv["shard"] else None,
            "posizione": inv["shard"]["posizione"] if inv["shard"] else None,
            "originale": {k: original.get(k) for k in ("lettura", "byte_sha256", "pixel_sha256", "phash")},
            "immagine_shard": None if img is None else {k: img.get(k) for k in ("lettura", "byte_sha256",
                                                                                "pixel_sha256", "phash")},
            "didascalia_sha256": mf.sha256_text(text) if isinstance(text, str) else None,
            "controlli": {"pixel_originale": original["lettura"] == "letta",
                          "pixel_shard": bool(img and img["lettura"] == "letta"),
                          "phash_originale": "phash" in original, "phash_shard": bool(img and "phash" in img),
                          "testo_didascalia": isinstance(text, str) and bool(text.strip())},
            "testi": {"didascalia": {"testi": int(isinstance(text, str) and bool(text.strip())),
                                     "confrontabili": int(comparable)}},
            "dettagli": found, "corrispondenze": dict(Counter(d["tipo"] for d in found)),
            "ngram_in_comune": sorted({g for d in found if d["tipo"] == "testo" for g in d["ngram_tutti"]})}
    tm["decontaminazione"] = time.perf_counter() - t4
    return inv, scan, tm


# ---------------------------------------------------------------- esiti (puri, collaudati in tests/)

def outcome(row: dict) -> str:
    """Esito della decontaminazione di una riga della scansione."""
    found = row.get("corrispondenze") or {}
    if found.get("pixel"):
        return "corrispondenza_confermata"
    if found.get("phash") or found.get("testo"):
        return "candidato"
    if not all((row.get("controlli") or {}).get(c) for c in REQUIRED):
        return "non_verificabile"
    return "nessuna_corrispondenza"


def proposal(inv: dict, row: dict, elements: dict) -> dict | None:
    """Proposta di trattamento di una riga (non applicata, non letta dai loader): solo per difetti e corrispondenze."""
    esito = outcome(row)
    roles = sorted({pd.role_of(elements.get(d["elemento"])) or "?" for d in row["dettagli"]})
    in_shard = inv.get("shard") is not None
    if not in_shard:
        if inv["esito_sharding"] == "mancante_nello_shard":
            return {"proposta": "rigenerare lo shard o escludere la riga", "motivo": "riga senza record nello shard"}
        return None  # scarto dello sharding: fuori dal training per costruzione
    if esito == "corrispondenza_confermata":
        return {"proposta": "escludere", "motivo": "pixel identici a un'immagine del pannello (legame sull'immagine, "
                                                     "non sulla domanda)", "ruoli": roles}
    if inv["difetti"]:
        return {"proposta": "escludere o correggere", "motivo": "; ".join(inv["difetti"])}
    if esito == "candidato":
        kinds = sorted({d["tipo"] for d in row["dettagli"]})
        how = " e ".join({"phash": "verifica a vista (pHash)", "testo": "lettura del testo in comune"}[k] for k in kinds)
        return {"proposta": "quarantena fino alla verifica", "motivo": f"candidato: {how}", "ruoli": roles}
    if esito == "non_verificabile":
        return {"proposta": "quarantena", "motivo": "controllo incompleto: " + ", ".join(
            c for c in REQUIRED if not row["controlli"].get(c))}
    return None


# ---------------------------------------------------------------- scansione

class Hasher(threading.Thread):
    """sha256 e byte di un file letto per intero in sequenza, mentre pyarrow legge le colonne (hashlib rilascia il GIL)."""

    def __init__(self, path: Path):
        super().__init__(daemon=True)
        self.path, self.result, self.error, self.seconds = path, None, None, None

    def run(self) -> None:
        t0 = time.time()
        try:
            h, n = hashlib.sha256(), 0
            with open(self.path, "rb", buffering=0) as fh:
                while block := fh.read(16 << 20):
                    h.update(block)
                    n += len(block)
            self.result = (h.hexdigest(), n)
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        self.seconds = round(time.time() - t0, 1)


def tar_samples(tar: tarfile.TarFile):
    """(chiave, {estensione: byte}, membri) nell'ordine del tar, come data.iter_shard, con i nomi dei membri: un membro
    ripetuto non sovrascrive l'altro senza traccia."""
    key, files, names = None, {}, []
    for member in tar:
        if not member.isfile():
            continue
        stem, _, ext = member.name.rpartition("/")[2].partition(".")
        if stem != key and names:
            yield key, files, names
            files, names = {}, []
        key = stem
        names.append(member.name)
        if ext in files:
            ext = f"{ext}#{len(names)}"  # ripetuto: resta visibile fra le estensioni
        files[ext] = tar.extractfile(member).read()
    if names:
        yield key, files, names


def prompt_check(items: list[dict]) -> list[dict]:
    """Le richieste fisse del loader contro l'indice: 10-gram (text_hits) e testo esatto (forma conservativa)."""
    exact = defaultdict(list)
    for j, it in enumerate(items):
        if it.get("text"):
            exact[dc.exact_text(it["text"])].append(j)
    out = []
    for i, p in enumerate(DETAILED_CAPTION_PROMPTS):
        hits = pd.text_hits(p)
        out.append({"indice": i, "richiesta": p, "parole": len(dc.normalize_words(p)),
                    "confrontabile_10gram": len(dc.normalize_words(p)) >= pd.NGRAM,
                    "ngram_in_comune": [{**pd.entry_ref(j), "ngram_tutti": gs} for j, gs in hits],
                    "testo_esatto": [pd.entry_ref(j) for j in exact.get(dc.exact_text(p), ())]})
    return out


def cmd_scan(args) -> dict:
    t0 = time.time()
    for d in (args.inventory, args.scan):
        if d.exists() and any(d.iterdir()):
            raise Failed(f"{d} non è vuota: le uscite si scrivono in cartelle nuove")
    sh = read_sharding(args.shards, args.revision)
    units = sh["unita"][:args.files] if args.files else sh["unita"]
    for u in units:
        if not u["tar"].is_file():
            raise Failed(f"shard mancante: {u['tar']}")
        for f in u["parquet"]:
            f["percorso"] = snapshot_file(args.snapshot, f["snapshot"])
            f["blob"] = f["percorso"].resolve().name
            if f["blob"] != f["blob_atteso"]:
                raise Failed(f"{f['percorso']}: blob {f['blob']}, il piano registra {f['blob_atteso']}")
    readme = args.snapshot / "README.md"
    loaded, isha = pd.read_index(args.index)
    if isha != args.index_sha256:
        raise Failed(f"indice con sha256 {isha}, atteso {args.index_sha256}")
    try:
        convention = dc.index_convention(loaded)
    except ValueError as exc:
        raise Failed(f"indice: {exc}") from None
    if convention != CONV or loaded.get("schema") != pd.PANEL_SCHEMA:
        raise Failed(f"indice {convention!r} {loaded.get('schema')!r}: servono {CONV} e {pd.PANEL_SCHEMA}")
    items = loaded["items"]
    pd.G.clear()
    # il ramo di panel_decontam.structures con pHash e 10-gram (lo stesso della scansione di MAVIS): voci, pixel,
    # HashIndex entro 4 bit, TextIndex con n = 10 e max_df = 25 dal lato dell'indice
    pd.G.update(pd.structures(items, "mavis", None), convention=convention)
    index_info = {**file_entry(args.index / "index.json", isha), "schema": loaded.get("schema"),
                  "convenzione_immagini": convention, "voci": len(items),
                  "voci_con_pixel_sha256": sum(map(len, pd.G["by_pixels"].values())),
                  "voci_con_phash": len(pd.G["hash_entry"]), "ngram_indicizzati": len(pd.G["tidx"].grams),
                  "ngram_di_rito_esclusi": pd.G["tidx"].dropped}
    prompts = prompt_check(items)
    del loaded, items
    gc.collect()
    target_size(64, 64)  # moduli di transformers e di imagehash importati prima del fork: i processi li ereditano
    dc.phash(Image.new("RGB", (8, 8)))
    t_index = round(time.time() - t0, 1)
    pool = multiprocessing.get_context("fork").Pool(args.workers)
    args.inventory.mkdir(parents=True, exist_ok=True)
    args.scan.mkdir(parents=True, exist_ok=True)
    inv_out, scan_out = GzLines(args.inventory / "records.jsonl.gz"), GzLines(args.scan / "records.jsonl.gz")
    digests = mf.Digests()
    st = {k: Counter() for k in ("esito_sharding", "difetti", "lettura_originale", "lettura_shard", "formato", "modo",
                                 "exif_orientamento", "fotogrammi", "didascalia_originale", "spazi_ai_bordi",
                                 "trascrizioni", "segnali", "trasformazione", "phash_distanza_originale_shard",
                                 "esiti_propri", "corrispondenze_per_benchmark_tipo",
                                 "righe_con_corrispondenze_per_benchmark_tipo", "controlli_non_eseguiti",
                                 "richiesta", "parole_didascalia", "lato_lungo_originale")}
    times = Counter()
    ngram_df: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    seen_keys: set[str] = set()
    extra_records: list[dict] = []
    units_out = []

    def bucket(n: int, edges: tuple[int, ...]) -> str:
        for e in edges:
            if n < e:
                return f"<{e}"
        return f">={edges[-1]}"

    def handle(results) -> None:
        for inv, scan, tm in results:
            key = inv["chiave"]
            if key in seen_keys:
                inv["difetti"].append("chiave ripetuta")
            seen_keys.add(key)
            inv["record_id"] = mf.record_id(digests, SOURCE, args.revision, key)
            inv_out.write(dumps(inv) + "\n")
            scan_out.write(dumps(scan) + "\n")
            times.update(tm)
            st["esito_sharding"][inv["esito_sharding"]] += 1
            for d in inv["difetti"]:
                st["difetti"][cause(d)] += 1
                if len(examples["difetti"]) < 20:
                    examples["difetti"].append(f"{key}: {d}")
            o = inv["originale"]
            st["lettura_originale"][o["lettura"]] += 1
            st["formato"][o.get("formato") or "-"] += 1
            st["modo"][o.get("modo") or "-"] += 1
            if "exif_orientamento" in o:
                st["exif_orientamento"][str(o["exif_orientamento"])] += 1
            if "fotogrammi" in o:
                st["fotogrammi"][str(o["fotogrammi"])] += 1
            if o.get("dimensioni"):
                st["lato_lungo_originale"][bucket(max(o["dimensioni"]), (64, 256, 512, 1024, 2048, 4096, 8192))] += 1
            if inv.get("immagine_shard"):
                st["lettura_shard"][inv["immagine_shard"]["lettura"]] += 1
            c = inv["didascalia"]
            st["didascalia_originale"][c["originale"]] += 1
            st["spazi_ai_bordi"][str(c.get("spazi_ai_bordi"))] += 1
            if c.get("parole") is not None:
                st["parole_didascalia"][bucket(c["parole"], (10, 50, 100, 200, 300, 500))] += 1
            st["trascrizioni"][str(min(inv["trascrizioni"]["non_vuote"], 5))] += 1
            for k in inv["segnali"]["tipi"]:
                st["segnali"][k] += 1
            st["segnali"]["righe_con_almeno_un_segnale"] += bool(inv["segnali"]["tipi"])
            st["richiesta"][str(inv["richiesta"])] += 1
            rep = inv.get("trasformazione")
            if rep and "byte_uguali_allo_shard" in rep:
                st["trasformazione"]["byte_uguali" if rep["byte_uguali_allo_shard"] else "byte_diversi"] += 1
                st["phash_distanza_originale_shard"][str(min(rep["phash_distanza_originale_shard"], 12))] += 1
                if not rep["byte_uguali_allo_shard"] and len(examples["trasformazione"]) < 20:
                    examples["trasformazione"].append(key)
            if rep and not rep["idempotente"]:
                st["trasformazione"]["non_idempotente"] += 1
            st["esiti_propri"][("nello_shard/" if inv["shard"] else "fuori_shard/") + outcome(scan)] += 1
            for ck, v in scan["controlli"].items():
                if not v:
                    st["controlli_non_eseguiti"][ck] += 1
            seen = set()
            for d in scan["dettagli"]:
                k = f"{d['benchmark']}/{d['tipo']}"
                st["corrispondenze_per_benchmark_tipo"][k] += 1
                seen.add(k)
            for k in seen:
                st["righe_con_corrispondenze_per_benchmark_tipo"][k] += 1
                if len(examples[k]) < 10:
                    examples[k].append(key)
            ngram_df.update(scan["ngram_in_comune"])

    pending = None
    for u in units:
        tu = time.time()
        raw = um.HashingReader(u["tar"])
        tar = tarfile.open(fileobj=io.BufferedReader(raw, buffer_size=1 << 20), mode="r|")
        samples = tar_samples(tar)
        nxt = next(samples, None)
        position = 0
        files_out = []
        for f in u["parquet"]:
            hasher = Hasher(f["percorso"])
            hasher.start()
            pf = pq.ParquetFile(f["percorso"])
            md = pf.metadata
            starts = [0]
            col_bytes = 0
            for g in range(md.num_row_groups):
                rgm = md.row_group(g)
                starts.append(starts[-1] + rgm.num_rows)
                col_bytes += sum(rgm.column(c).total_compressed_size for c in range(rgm.num_columns)
                                 if rgm.column(c).path_in_schema.split(".")[0] in COLUMNS)
            fi, name = f["indice_file"], f"data/{f['percorso'].name}"
            row, batch = 0, []
            for b in pf.iter_batches(batch_size=BATCH, columns=COLUMNS):
                for rec in b.to_pylist():
                    key = f"pixmo-{fi:05d}-{row:06d}"
                    # record dello shard di questa riga: quelli prima della chiave, o non riconoscibili, sono in più
                    while nxt is not None and (key_order(nxt[0]) is None or key_order(nxt[0]) < (fi, row)):
                        extra_records.append({"shard": u["nome"], "posizione": position, "chiave": nxt[0],
                                              "membri": nxt[2]})
                        position += 1
                        nxt = next(samples, None)
                    shard = None
                    if nxt is not None and nxt[0] == key:
                        shard = {"nome": u["nome"], "posizione": position, "files": nxt[1], "membri": nxt[2]}
                        position += 1
                        nxt = next(samples, None)
                    g = bisect.bisect_right(starts, row) - 1
                    batch.append((fi, name, row, (g, row - starts[g]), key, (rec.get("image") or {}).get("bytes"),
                                  rec.get("caption"), rec.get("transcripts"), rec.get("image_url"), shard,
                                  sh["sorgente"]))
                    row += 1
                if len(batch) >= BATCH:
                    cur = pool.map_async(examine, batch, chunksize=8)
                    if pending is not None:
                        handle(pending.get())
                    pending, batch = cur, []
            if batch:
                cur = pool.map_async(examine, batch, chunksize=8)
                if pending is not None:
                    handle(pending.get())
                pending = cur
            tw = time.time()
            hasher.join()
            files_out.append({"file": name, "indice_file": fi, "percorso": str(f["percorso"]), "blob": f["blob"],
                              "righe_lette": row, "righe_metadati": md.num_rows, "gruppi_di_righe": md.num_row_groups,
                              "righe_per_gruppo": sorted(Counter(starts[i + 1] - starts[i]
                                                                 for i in range(md.num_row_groups)).items()),
                              "byte_colonne_lette": col_bytes, "sha256_calcolato": (hasher.result or (None,))[0],
                              "byte_letti_per_lo_sha256": (hasher.result or (None, None))[1],
                              "sha256_uguale_al_blob": bool(hasher.result) and hasher.result[0] == f["blob"],
                              "errore_sha256": hasher.error, "secondi_sha256": hasher.seconds,
                              "attesa_sha256_secondi": round(time.time() - tw, 1),
                              "colonne": pf.schema_arrow.names})
        while nxt is not None:  # record dopo l'ultima riga dei parquet dello shard
            extra_records.append({"shard": u["nome"], "posizione": position, "chiave": nxt[0], "membri": nxt[2]})
            position += 1
            nxt = next(samples, None)
        tar.close()
        tar_sha, tar_bytes = raw.finish()
        reg = u["registro"]
        units_out.append({"shard": u["nome"], "percorso": str(u["tar"]), "parquet": files_out,
                          "campioni_letti": position, "campioni_registrati": reg["samples"],
                          "sha256_letto": tar_sha, "byte_letti": tar_bytes, "sha256_registrato": reg["sha256"],
                          "byte_registrati": reg["bytes"],
                          "uguale_al_registro": tar_sha == reg["sha256"] and tar_bytes == reg["bytes"]
                          and position == reg["samples"], "secondi": round(time.time() - tu, 1)})
        print(f"{u['nome']}: {sum(x['righe_lette'] for x in files_out)} righe, {position} campioni "
              f"({time.time() - t0:.0f} s)", flush=True)
    if pending is not None:
        handle(pending.get())
    pool.close()
    pool.join()
    inv_out.close()
    scan_out.close()
    rows = sum(x["righe_lette"] for u in units_out for x in u["parquet"])
    common = {"job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "args": {k: str(v) for k, v in vars(args).items() if k != "func"},
              "sorgente": SOURCE, "repository": REPO, "revisione": args.revision, "origine_a_monte": UPSTREAM,
              "parziale": bool(args.files), "file_letti": len([x for u in units_out for x in u["parquet"]]),
              "file_del_piano": len(sh["piano"]["files"]), "righe": rows}
    inventory_summary = {
        "schema": SCHEMA_INVENTORY, **common,
        "readme": file_entry(readme) if readme.is_file() else None,
        "sharding": {"cartella": str(args.shards), "piano": file_entry(args.shards / "plan.json"),
                     "registro": file_entry(args.shards / "shards.jsonl"),
                     "manifest": file_entry(args.shards / "manifest.json"),
                     "digest_del_piano": sh["piano"].get("digest"), "digest_ricalcolato": sh["digest_ricalcolato"],
                     "job": sh["manifest"].get("job_id"), "conteggi": sh["manifest"].get("counts")},
        "unita": units_out, "record_dello_shard_senza_riga": extra_records,
        "statistiche": {k: dict(sorted(v.items())) for k, v in st.items() if not k.startswith(("esiti", "corrisp",
                                                                                                 "righe_con",
                                                                                                 "controlli"))},
        "esempi": {k: v for k, v in examples.items() if k in ("difetti", "trasformazione")},
        "proprieta_non_verificate": NOT_VERIFIED,
        "files": {"records.jsonl.gz": file_entry(args.inventory / "records.jsonl.gz")}}
    scan_summary = {
        "schema": SCHEMA_SCAN, **common, "indice": index_info, "convenzione_immagini": convention,
        "soglie": {"phash_bit": pd.RADIUS, "ngram": pd.NGRAM, "max_df_dell_indice": pd.MAX_DF},
        "regole": {"confermata": "pixel identici dopo la normalizzazione rgb-su-bianco/1 dell'originale o del JPEG dello "
                                 "shard (sha256 dei pixel uguale a quello di una voce dell'indice)",
                   "candidato": "pHash entro 4 bit dell'originale o del JPEG dello shard, oppure un 10-gram di "
                                "normalize_words in comune fra la didascalia del loader e un testo dell'indice (dal lato "
                                "dell'indice senza gli n-gram in più di 25 testi diversi del pannello). Nessuna regola "
                                "esplicita copre le didascalie: la frequenza nella sorgente non decide nulla",
                   "non_verificabile": "un controllo di " + ", ".join(REQUIRED) + " non eseguibile",
                   "nessuna_corrispondenza": "nessuna corrispondenza nei controlli eseguiti; non certifica la didascalia"},
        "richieste_del_loader": prompts, "controlli_non_eseguiti_per_costruzione": NOT_RUN,
        "statistiche": {k: dict(sorted(v.items())) for k, v in st.items() if k.startswith(("esiti", "corrisp",
                                                                                            "righe_con", "controlli"))},
        "ngram_in_comune_distinti": len(ngram_df),
        "ngram_piu_frequenti": [{"ngram": g, "righe": n} for g, n in ngram_df.most_common(30)],
        "nota_frequenze": "la frequenza dei 10-gram serve solo a ordinare i casi da esaminare, non decide nulla",
        "esempi_per_benchmark_tipo": {k: v for k, v in sorted(examples.items()) if "/" in k},
        "files": {"records.jsonl.gz": file_entry(args.scan / "records.jsonl.gz")}}
    cost = {"secondi": round(time.time() - t0, 1), "secondi_indice": t_index, "rss_max_mib": um.rss_mib(),
            "workers": args.workers, "secondi_cpu_per_fase": {k: round(v, 1) for k, v in sorted(times.items())},
            "byte_parquet_letti_per_lo_sha256": sum(x["byte_letti_per_lo_sha256"] or 0 for u in units_out
                                                    for x in u["parquet"]),
            "byte_colonne_parquet": sum(x["byte_colonne_lette"] for u in units_out for x in u["parquet"]),
            "byte_shard_letti": sum(u["byte_letti"] for u in units_out)}
    inventory_summary["costo"] = scan_summary["costo"] = cost
    write_new_json(args.inventory / "summary.json", inventory_summary)
    write_new_json(args.scan / "summary.json", scan_summary)
    print(json.dumps({"righe": rows, "esito_sharding": dict(st["esito_sharding"]), "difetti": dict(st["difetti"]),
                      "esiti": dict(st["esiti_propri"]), "trasformazione": dict(st["trasformazione"]), **cost},
                     ensure_ascii=False)[:4000])
    return inventory_summary


# ---------------------------------------------------------------- verifica

def readme_info(path: Path) -> dict:
    """Esempi, file e byte dichiarati nella scheda del dataset (intestazione YAML, letta con espressioni regolari)."""
    text = path.read_text(encoding="utf-8")
    num = re.search(r"num_examples:\s*(\d+)", text)
    size = re.search(r"download_size:\s*(\d+)", text)
    data = text.encode("utf-8")
    return {"esempi_dichiarati": int(num.group(1)) if num else None,
            "byte_scaricabili_dichiarati": int(size.group(1)) if size else None,
            "git_sha1": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}


def host(url) -> str:
    try:
        return urlsplit(url).hostname or "-"
    except (TypeError, ValueError):
        return "?"


def duplicates(inv: list[dict]) -> tuple[dict, list[dict]]:
    """Gruppi di righe con lo stesso valore (byte o pixel dell'originale, URL, didascalia del loader, byte del JPEG
    dello shard): conteggi e gruppi completi. Non sono gruppi di ammissione."""
    fields = {"byte_originale": lambda r: r["originale"].get("byte_sha256"),
              "pixel_originale": lambda r: r["originale"].get("pixel_sha256"),
              "url": lambda r: r.get("url"),
              "didascalia_del_loader": lambda r: r["didascalia"].get("loader_sha256") if r.get("shard") else None,
              "jpeg_dello_shard": lambda r: (r.get("shard") or {}).get("jpg_sha256")}
    stats, groups = {}, []
    for name, fn in fields.items():
        by = defaultdict(list)
        for r in inv:
            v = fn(r)
            if v:
                by[v].append(r)
        multi = [rs for rs in by.values() if len(rs) > 1]
        sizes = Counter(len(rs) for rs in multi)
        stats[name] = {"gruppi": len(multi), "righe": sum(map(len, multi)),
                       "dimensioni": dict(sorted((str(k), v) for k, v in sizes.items())),
                       "massimo": max(sizes) if sizes else 0}
        for rs in sorted(multi, key=lambda x: (-len(x), x[0]["chiave"])):
            urls = {r.get("url") for r in rs}
            groups.append({"campo": name, "righe": len(rs), "chiavi": [r["chiave"] for r in rs],
                           "url_distinti": len(urls),
                           "host": dict(Counter(host(u) for u in urls if u).most_common(5)),
                           "didascalie_distinte": len({r["didascalia"].get("loader_sha256") for r in rs}),
                           "pixel_distinti": len({r["originale"].get("pixel_sha256") for r in rs}),
                           "dimensioni_originale": rs[0]["originale"].get("dimensioni"),
                           "byte_originale": rs[0]["originale"].get("byte")})
    return stats, groups


def in_read_files(key: str, read_idx: set[int]) -> bool:
    o = key_order(key)
    return o is not None and o[0] in read_idx


def run4_reconcile(run4: Path, inv: list[dict], partial: bool, revision: str, check) -> dict:
    """run4 (pixmo_transcript_check.py, job 30044) collegato alle righe correnti: revisione e file letti, copertura,
    chiavi risolte in esattamente una riga (S1 §5), segnali uguali riga per riga a quelli ricalcolati dal codice
    corrente sugli stessi ingressi. Con --partial solo le chiavi dei parquet letti."""
    report = json.loads((run4 / "report.json").read_text(encoding="utf-8"))
    flagged = um.read_jsonl(run4 / "flagged.jsonl")
    by_key = {r["chiave"]: r for r in inv}
    read_files = {r["parquet"]["file"] for r in inv}
    read_idx = {r["parquet"]["indice_file"] for r in inv}
    files4 = [Path(f["path"]) for f in report.get("files") or []]
    check("run4: schema pixmo-transcript-check/4, parquet della stessa revisione",
          report.get("schema") == "pixmo-transcript-check/4" and bool(files4)
          and all(f"/snapshots/{revision}/data/" in str(p) for p in files4), report.get("schema"))
    out = {"report": file_entry(run4 / "report.json"), "flagged": file_entry(run4 / "flagged.jsonl"),
           "schema": report.get("schema"), "job": report.get("job_id"), "codice": report.get("code"),
           "righe": report.get("rows"), "righe_senza_trascrizioni": report.get("rows_without_transcript"),
           "righe_con_segnali": report.get("rows_with_candidates"), "file": len(files4)}
    keys4 = [r["id"] for r in flagged]
    dup = [k for k, n in Counter(keys4).items() if n > 1]
    in_scope = [k for k in keys4 if not partial or in_read_files(k, read_idx)]
    unresolved = [k for k in in_scope if k not in by_key]
    check("run4: chiavi di flagged.jsonl uniche e risolte in esattamente una riga dell'inventario",
          not dup and not unresolved, {"ripetute": dup[:5], "non_risolte": len(unresolved), "esempi": unresolved[:5]})
    mine = {k: r["segnali"] for k, r in by_key.items() if r["segnali"]["tipi"]}
    theirs = {r["id"]: r for r in flagged if r["id"] in by_key}
    diff = []
    for k in sorted(set(mine) | set(theirs)):
        a, b = mine.get(k), theirs.get(k)
        if a is None or b is None or a["tipi"] != b["risks"] or a["numbers"] != b["numbers"] \
                or a["colors"] != b["colors"] or a["quotes"] != b["quotes"]:
            diff.append(k)
    check("run4: segnali ricalcolati dal codice corrente uguali riga per riga a flagged.jsonl (tipi, numeri, colori, "
          "testi citati)", not diff, {"diversi": len(diff), "esempi": diff[:5]})
    if not partial:
        same_files = sorted(str(p.name) for p in files4) == sorted(Path(f).name for f in read_files)
        check("run4: stessi 25 parquet della revisione, stesse righe, nessuna senza trascrizioni",
              same_files and report.get("rows") == len(inv) and report.get("rows_without_transcript") == sum(
                  1 for r in inv if r["segnali"].get("senza_trascrizioni")),
              {"file": len(files4), "righe_run4": report.get("rows"), "righe_inventario": len(inv)})
        mine_rows = Counter(k for r in inv for k in r["segnali"]["tipi"])
        check("run4: righe per tipo di segnale uguali a report.json",
              dict(mine_rows) == report.get("risk_rows_by_type"), dict(mine_rows))
        files_ok = all(f.get("bytes") == (Path(f["path"]).stat().st_size if Path(f["path"]).exists() else None)
                       for f in report.get("files") or [])
        check("run4: byte dei parquet letti allora uguali a quelli di oggi", files_ok)
        out["righe_per_tipo_ricalcolate"] = dict(mine_rows)
    out["righe_confrontate"] = len(in_scope) if partial else len(inv)
    out["nota"] = ("run4 è del codice 00d80e3 con modifiche non salvate: la provenienza si stabilisce ricalcolando gli "
                   "stessi segnali con il codice corrente, che le contiene (25b86c8), sugli stessi ingressi")
    return out


def historical(folders: list[Path], by_key: dict, outcomes: dict, partial: bool, check) -> list[dict]:
    """Liste storiche di decontam.py scan sugli shard (p6 e f1eval, convenzione grigio-diretto/0, solo pHash): le chiavi
    si risolvono in esattamente una riga (S1 §5) e si confrontano con gli esiti nuovi. Diagnostico: non si importano.
    Con --partial solo le chiavi dei parquet letti."""
    read_idx = {r["parquet"]["indice_file"] for r in by_key.values()}
    out = []
    for folder in folders:
        raw, final = um.read_keys(folder / "exclude.txt"), um.read_keys(folder / "exclude_final.txt")
        if partial:
            raw = [k for k in raw if in_read_files(k, read_idx)]
            final = [k for k in final if in_read_files(k, read_idx)]
        problems = um.resolve(raw, by_key, str(folder)) + um.resolve(final, by_key, f"{folder} (finale)")
        summary = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").is_file() else {}
        check(f"storiche {folder.parent.name}/{folder.name}: chiavi risolte in esattamente una riga", not problems,
              problems[:5])
        found = [k for k in raw if k in outcomes]
        out.append({"cartella": str(folder), "exclude": file_entry(folder / "exclude.txt"),
                    "exclude_final": file_entry(folder / "exclude_final.txt"), "candidati_storici": len(raw),
                    "lista_finale": len(final), "indice_storico": summary.get("index"),
                    "esiti_nuovi_dei_candidati_storici": dict(Counter(outcomes[k] for k in found)),
                    "storici_non_candidati_nuovi_esempi": [k for k in found if outcomes[k] == "nessuna_corrispondenza"][:10],
                    "nota": "convenzione grigio-diretto/0 sui JPEG degli shard, solo pHash, contro indici diversi da "
                            "pannello-v3: confronto diagnostico, liste non importate"})
    return out


def cmd_verifica(args) -> int:
    t0 = time.time()
    if args.report.exists() and any(args.report.iterdir()):
        raise Failed(f"{args.report} non è vuota")
    checks: list[dict] = []

    def check(name: str, ok, detail=None) -> bool:
        checks.append({"controllo": name, "esito": bool(ok), "dettaglio": detail})
        print(("ok   " if ok else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail is not None else ""), flush=True)
        return bool(ok)

    # ---- indice e pannello
    loaded, isha = pd.read_index(args.index)
    try:
        convention = dc.index_convention(loaded)
    except ValueError as exc:
        convention = str(exc)
    check("indice: convenzione rgb-su-bianco/1 e schema panel-index/2",
          convention == CONV and loaded.get("schema") == pd.PANEL_SCHEMA, [convention, loaded.get("schema")])
    check("indice: sha256 uguale a quello atteso", isha == args.index_sha256, isha)
    items = loaded["items"]
    verifica = json.loads((args.index / "verifica.json").read_text(encoding="utf-8"))
    with_image = sum(1 for it in items if it.get("pos") is not None)
    check("indice: verifica.json, ogni voce con immagine ritrova la propria immagine",
          verifica.get("verificato") is True and verifica.get("errori") == 0
          and verifica.get("voci_con_immagine") == with_image and verifica.get("convenzione_immagini") == convention,
          {k: verifica.get(k) for k in ("verificato", "errori", "voci_con_immagine")})
    panel_summary = json.loads((args.panel / "summary.json").read_text(encoding="utf-8"))
    check("pannello costruito da questo indice", (panel_summary.get("indice") or {}).get("sha256") == isha)
    elements = {e["chiave"]: e for e in sd.read_jsonl_gz(args.panel / "elements.jsonl.gz")}
    index_elements = {pd.element_of(it) for it in items}
    check("perimetro: ogni elemento del pannello ha voci nell'indice e ogni voce un elemento",
          index_elements == set(elements), {"solo_indice": len(index_elements - set(elements)),
                                           "solo_pannello": len(set(elements) - index_elements)})
    perimeter = Counter(pd.role_of(e) for e in elements.values())
    check("perimetro: tutti i ruoli, compresi test non ammesso, diagnostica e solo_decontaminazione",
          all(perimeter[r] for r in ("sviluppo", "test_ammesso", "test_non_ammesso", "diagnostica",
                                     "solo_decontaminazione")), dict(perimeter))
    del loaded, items
    gc.collect()
    # ---- uscite della scansione
    inv_sum = json.loads((args.inventory / "summary.json").read_text(encoding="utf-8"))
    scan_sum = json.loads((args.scan / "summary.json").read_text(encoding="utf-8"))
    for what, folder, summ in (("inventario", args.inventory, inv_sum), ("scansione", args.scan, scan_sum)):
        got = sha256_file(folder / "records.jsonl.gz")
        check(f"{what}: records.jsonl.gz con lo sha256 di summary.json",
              got == summ["files"]["records.jsonl.gz"]["sha256"], got)
    check("scansione: stesso indice (sha256) e stessa convenzione",
          (scan_sum.get("indice") or {}).get("sha256") == isha and scan_sum.get("convenzione_immagini") == CONV)
    check("scansione e inventario dello stesso job, stessa revisione" + (", sonda dichiarata" if args.partial
                                                                         else ", completi"),
          inv_sum["job_id"] == scan_sum["job_id"] and inv_sum["revisione"] == scan_sum["revisione"] == args.revision
          and inv_sum["parziale"] == scan_sum["parziale"] == bool(args.partial))
    inv = read_gz(args.inventory / "records.jsonl.gz")
    scan = read_gz(args.scan / "records.jsonl.gz")
    check("una riga della scansione per riga dell'inventario, nello stesso ordine",
          [r["chiave"] for r in inv] == [r["chiave"] for r in scan], {"inventario": len(inv), "scansione": len(scan)})
    # ---- snapshot e sharding
    sh = read_sharding(args.shards, args.revision)
    units = {u["nome"]: u for u in sh["unita"]}
    check("sharding: digest del piano ricalcolato uguale a plan.json",
          sh["digest_ricalcolato"] == sh["piano"].get("digest") == sh["manifest"].get("plan_digest"))
    read_units = inv_sum["unita"]
    expected_units = list(units)[:len(read_units)] if args.partial else list(units)
    check("sharding: unità lette nell'ordine del piano" + ("" if args.partial else ", tutte"),
          [u["shard"] for u in read_units] == expected_units, [u["shard"] for u in read_units])
    check("shard: letti per intero, sha256, byte e campioni uguali a shards.jsonl",
          all(u["uguale_al_registro"] for u in read_units),
          [u["shard"] for u in read_units if not u["uguale_al_registro"]])
    pfiles = [x for u in read_units for x in u["parquet"]]
    check("snapshot: parquet coi blob del piano, sha256 ricalcolato uguale al nome del blob",
          all(x["sha256_uguale_al_blob"] for x in pfiles), [x["file"] for x in pfiles if not x["sha256_uguale_al_blob"]])
    check("snapshot: righe lette uguali alle righe dei metadati di ogni parquet",
          all(x["righe_lette"] == x["righe_metadati"] for x in pfiles))
    tsv = [line.split("\t") for line in args.verify_report.read_text().splitlines()[1:] if line.strip()]
    verified = {row[1].rsplit("/", 1)[-1]: row for row in tsv if REPO in row[4]}
    check("snapshot: ogni parquet letto verificato dal job 29554 (stato ok, byte e sha256 attesi)",
          all(verified.get(x["blob"], [None])[0] == "ok" and verified[x["blob"]][3] == f"sha256:{x['blob']}"
              and int(verified[x["blob"]][2]) == x["byte_letti_per_lo_sha256"] for x in pfiles),
          len(verified))
    readme = readme_info(args.snapshot / "README.md")
    check("snapshot: README con il git sha1 del blob registrato",
          (args.snapshot / "README.md").resolve().name == readme["git_sha1"], readme["git_sha1"])
    # ---- copertura, chiavi e collegamento
    by_key = {}
    problems = Counter()
    examples = defaultdict(list)
    file_of = {f["indice_file"]: f"data/{f['snapshot'].name}" for u in sh["unita"] for f in u["parquet"]}
    for r in inv:
        k = r["chiave"]
        o = key_order(k)
        p = r["parquet"]
        why = None
        if k in by_key:
            why = "chiave ripetuta"
        elif o is None or o != (p["indice_file"], p["riga"]):
            why = "chiave diversa da file e riga"
        elif p["file"] != file_of.get(p["indice_file"]):
            why = "file diverso da quello registrato dallo sharding per l'indice del file"
        if why:
            problems[why] += 1
            examples[why].append(k)
        by_key.setdefault(k, r)
    rows_expected = sum(x["righe_metadati"] for x in pfiles)
    check("chiavi: uniche, uguali a pixmo-<file>-<riga> del parquet, una per riga dei parquet letti",
          not problems and len(by_key) == rows_expected,
          {"problemi": dict(problems), "esempi": {k: v[:3] for k, v in examples.items()}, "righe": len(by_key),
           "attese": rows_expected})
    rids = Counter(r["record_id"] for r in inv)
    digests = mf.Digests()
    bad_rid = [r["chiave"] for r in inv if mf.record_id(digests, SOURCE, args.revision, r["chiave"]) != r["record_id"]]
    check("record_id: unici e ricalcolati da sorgente, revisione e chiave (S1 §1)",
          max(rids.values()) == 1 and not bad_rid, {"diversi": len(bad_rid)})
    extra = inv_sum["record_dello_shard_senza_riga"]
    check("collegamento: nessun record dello shard senza riga del parquet", not extra, extra[:5])
    order_bad = []
    last = {}
    for r in inv:
        s = r.get("shard")
        if s:
            if s["posizione"] <= last.get(s["nome"], -1):
                order_bad.append(r["chiave"])
            last[s["nome"]] = s["posizione"]
    check("collegamento: ogni record dello shard ritrova la sua riga, nello stesso ordine", not order_bad,
          order_bad[:5])
    sharding = Counter(r["esito_sharding"] for r in inv)
    defects = Counter()
    for r in inv:
        for d in r["difetti"]:
            defects[cause(d)] += 1
    check("esame: nessun errore imprevisto in una riga",
          not any(d.startswith("errore dell'esame") for d in defects),
          {d: n for d, n in defects.items() if d.startswith("errore dell'esame")})
    link_defects = {d: n for d, n in defects.items() if d.startswith(("JSON", "membri", "JPEG", "campi", "riga senza",
                                                                       "record nello shard", "chiave"))}
    check("collegamento: JSON, membri e JPEG di ogni record coerenti con la riga (chiave, didascalia, URL, dimensioni, "
          "formato, sorgente)", not link_defects, link_defects)
    # scarti: quelli di shards.jsonl (illeggibili per chiave, mancanti per motivo) e quelli ricalcolati
    logged_unreadable = {x.split(":", 1)[0] for u in read_units for x in units[u["shard"]]["registro"]["unreadable"]}
    logged_excluded = Counter()
    for u in read_units:
        logged_excluded.update(units[u["shard"]]["registro"]["excluded"])
    mine_unreadable = {r["chiave"] for r in inv if r["esito_sharding"] == "scartata:illeggibile"}
    mine_excluded = Counter(r["esito_sharding"].split(":", 1)[1] for r in inv
                            if r["esito_sharding"].startswith("scartata:missing"))
    check("scarti: illeggibili e mancanti ricalcolati uguali a quelli registrati dallo sharding",
          logged_unreadable == mine_unreadable and dict(logged_excluded) == dict(mine_excluded),
          {"illeggibili_registrati": sorted(logged_unreadable), "illeggibili_ricalcolati": sorted(mine_unreadable),
           "mancanti_registrati": dict(logged_excluded), "mancanti_ricalcolati": dict(mine_excluded)})
    if not args.partial:
        counts = sh["manifest"].get("counts") or {}
        check("sharding: campioni scritti e scarti uguali al manifest.json dello sharding",
              sharding["nello_shard"] == counts.get("written") and len(mine_unreadable) == counts.get("unreadable")
              and dict(mine_excluded) == counts.get("excluded"), {"manifest": counts, "inventario": dict(sharding)})
    rep = Counter()
    for r in inv:
        t = r.get("trasformazione") or {}
        if "byte_uguali_allo_shard" in t:
            rep["uguali" if t["byte_uguali_allo_shard"] else "diversi"] += 1
    check("trasformazione: shard_parquet.convert rifatta sull'originale dà i byte del JPEG dello shard",
          rep["diversi"] == 0 and rep["uguali"] == sharding["nello_shard"], dict(rep))
    # ---- duplicati, run4, decontaminazione
    dup_stats, dup_groups = duplicates(inv)
    run4 = run4_reconcile(args.run4, inv, args.partial, args.revision, check)
    outcomes = {s["chiave"]: outcome(s) for s in scan}
    hist = historical(args.historical, by_key, outcomes, args.partial, check)
    unknown = Counter(d["benchmark"] for s in scan for d in s["dettagli"] if d["elemento"] not in elements)
    check("corrispondenze: ogni elemento citato esiste nel pannello", not unknown, dict(unknown))
    in_shard = {r["chiave"] for r in inv if r.get("shard")}
    missing_checks = Counter(c for s in scan if s["chiave"] in in_shard for c in REQUIRED if not s["controlli"].get(c))
    by_outcome = Counter((("nello_shard/" if s["chiave"] in in_shard else "fuori_shard/") + outcomes[s["chiave"]])
                         for s in scan)
    per_bench, rows_per_bench, rows_per_kind = Counter(), Counter(), Counter()
    elements_hit = defaultdict(set)
    for s in scan:
        seen = set()
        for d in s["dettagli"]:
            role = pd.role_of(elements.get(d["elemento"]))
            per_bench[f"{d['benchmark']}/{role}/{d['tipo']}/{d.get('immagine') or d.get('campo')}"] += 1
            seen.add(f"{d['benchmark']}/{d['tipo']}")
            elements_hit[d["tipo"]].add(d["elemento"])
        for k in seen:
            rows_per_bench[k] += 1
        for k in {d["tipo"] for d in s["dettagli"]}:
            rows_per_kind[k] += 1
    proposals = []
    for r, s in zip(inv, scan):
        p = proposal(r, s, elements)
        if p:
            proposals.append({"chiave": r["chiave"], "record_id": r["record_id"], "esito": outcomes[r["chiave"]],
                              **p, "corrispondenze": s["corrispondenze"]})
    args.report.mkdir(parents=True, exist_ok=True)
    with GzLines(args.report / "duplicati.jsonl.gz") as fh:
        for g in dup_groups:
            fh.write(dumps(g) + "\n")
    with GzLines(args.report / "proposte_di_trattamento.jsonl.gz") as fh:
        for p in proposals:
            fh.write(dumps(p) + "\n")
    signals = Counter(k for r in inv for k in r["segnali"]["tipi"])
    totals = {int(m.group(1)) for x in pfiles if (m := re.search(r"-of-(\d{5})\.parquet$", x["file"]))}
    declared = readme["esempi_dichiarati"]
    local = {"file_letti": len(pfiles), "file_del_piano": len(sh["piano"]["files"]),
             "file_del_dataset": totals.pop() if len(totals) == 1 else sorted(totals),
             "righe_lette": len(inv), "esempi_dichiarati_dal_readme": declared,
             "quota_delle_righe": round(len(inv) / declared, 4) if declared else None,
             "byte_parquet_letti": sum(x["byte_letti_per_lo_sha256"] or 0 for x in pfiles),
             "byte_dichiarati_dal_readme": readme["byte_scaricabili_dichiarati"],
             "nota": "il sottoinsieme locale è il prefisso dei file 0-24 (filtro di downloads/sources.tsv), non un "
                     "campione casuale"}
    report = {
        "schema": SCHEMA_REPORT, "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
        "parziale": bool(args.partial), "sorgente": SOURCE, "repository": REPO, "revisione": args.revision,
        "controlli": checks, "falliti": [c["controllo"] for c in checks if not c["esito"]],
        "perimetro_pannello": dict(perimeter), "copertura_del_sottoinsieme_locale": local,
        "inventario": {"righe": len(inv), "esito_sharding": dict(sharding), "difetti_strutturali": dict(defects),
                       "difetti_esempi": {d: [r["chiave"] for r in inv if any(
                           cause(x) == d for x in r["difetti"])][:10] for d in defects},
                       "trasformazione": dict(rep), "statistiche": inv_sum["statistiche"],
                       "duplicati": dup_stats,
                       "duplicati_piu_grandi": [g for g in dup_groups if g["campo"] == "byte_originale"][:15]},
        "segnali_delle_didascalie": {"righe_per_tipo": dict(signals.most_common()),
                                     "righe_con_almeno_un_segnale": sum(1 for r in inv if r["segnali"]["tipi"]),
                                     "nota": "segnali di rischio, non difetti: un dettaglio assente dalle trascrizioni "
                                             "non è dimostrato falso nell'immagine"},
        "run4": run4, "candidati_storici": hist,
        "decontaminazione": {"esiti": dict(sorted(by_outcome.items())),
                             "controlli_mancanti_nelle_righe_dello_shard": dict(missing_checks),
                             "righe_per_tipo": dict(rows_per_kind),
                             "elementi_del_pannello_per_tipo": {k: len(v) for k, v in elements_hit.items()},
                             "corrispondenze_per_benchmark_ruolo_tipo_immagine": dict(sorted(per_bench.items())),
                             "righe_per_benchmark_tipo": dict(sorted(rows_per_bench.items())),
                             "richieste_del_loader": scan_sum["richieste_del_loader"],
                             "ngram_piu_frequenti": scan_sum["ngram_piu_frequenti"],
                             "regole": scan_sum["regole"],
                             "controlli_non_eseguiti_per_costruzione": NOT_RUN},
        "proposte_di_trattamento": dict(Counter(p["proposta"] for p in proposals)),
        "proprieta_non_verificate": NOT_VERIFIED,
        "scansione": {"costo": scan_sum["costo"], "unita": [{k: u[k] for k in ("shard", "secondi", "campioni_letti")}
                                                             for u in read_units]},
        "files": {n: file_entry(args.report / n) for n in ("duplicati.jsonl.gz", "proposte_di_trattamento.jsonl.gz")},
        "nota": "evidenze e proposte: nessuna lista è collegata ai loader, nessun record è ammesso o escluso",
        "secondi": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()}
    write_new_json(args.report / "report.json", report)
    print(f"{len(checks) - len(report['falliti'])} controlli superati su {len(checks)}")
    print(json.dumps({k: report[k] for k in ("proposte_di_trattamento",)} | {"esiti": by_outcome}, ensure_ascii=False))
    return 1 if report["falliti"] else 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "verifica"):
        p = sub.add_parser(name)
        p.add_argument("--snapshot", type=Path, required=True)
        p.add_argument("--shards", type=Path, required=True)
        p.add_argument("--revision", required=True)
        p.add_argument("--index", type=Path, required=True)
        p.add_argument("--index-sha256", required=True)
        p.add_argument("--inventory", type=Path, required=True)
        p.add_argument("--scan", type=Path, required=True)
    s = sub.choices["scan"]
    s.add_argument("--files", type=int, default=0, help="solo i primi N parquet del piano (sonda)")
    s.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    v = sub.choices["verifica"]
    v.add_argument("--panel", type=Path, required=True)
    v.add_argument("--run4", type=Path, required=True)
    v.add_argument("--historical", type=Path, nargs="+", required=True)
    v.add_argument("--verify-report", type=Path, required=True, help="report di verify_hashes.sbatch dello snapshot")
    v.add_argument("--report", type=Path, required=True)
    v.add_argument("--partial", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("pixmo_inventory.py va eseguito in un job SLURM")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        sys.exit(f"revisione {args.revision!r}: servono 40 caratteri esadecimali")
    try:
        if args.cmd == "scan":
            cmd_scan(args)
            return 0
        return cmd_verifica(args)
    except Failed as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
