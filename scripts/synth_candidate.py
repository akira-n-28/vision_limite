#!/usr/bin/env python3
"""Candidato dei sintetici v4 (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §2.6): controlli sugli artefatti salvati da
scripts/synth_render.py, duplicati e split, stato di ogni scena. Solo in un job SLURM CPU; legge gli shard in sola
lettura e non li modifica.

Per ogni record degli shard (train/, test/, pairs/ della cartella --shards):
1. integrità: sha256 e dimensione di ogni shard uguali al manifest del render, nessun altro file nelle cartelle; nel tar
   un PNG e un JSON per chiave, nessun altro membro, le chiavi attese nell'ordine del render; campi del JSON, risposte
   fra le opzioni; PNG che si decodifica, RGB, alle dimensioni del record, lati multipli di 32 e max_patches uguale alle
   patch dell'immagine, che il processor lascia invariata (processing.target_size, una volta per dimensione); layout
   salvato con png_sha256 uguale allo sha256 del PNG;
2. riproducibilità: specifica e domande rigenerate dalla chiave (seme: sha256 della chiave, synth.rng_for), PNG e layout
   ridisegnati con byte identici; per le coppie l'intera coppia rigenerata come in synth_render.scene;
3. verifica indipendente: synth_check.check_sample sul layout salvato, come scripts/synth_semantics_check.py. I campioni
   non verificati devono essere esattamente quelli del controllo (--check) sugli stessi shard;
4. coppie controfattuali: stessa domanda e stesse opzioni in a e b, risposte diverse, immagini diverse, un solo
   dettaglio cambiato secondo la famiglia (pair_contract);
5. formato: domande ripetute nella scena, spazi in testa, in coda o doppi.
Fra i record (pipeline §§5.5, 5.8; S1 §§3-4; la regola di scripts/split_duplicates.py):
6. legami confermati, solo identità esatte: stessi pixel (sha256 dei valori RGBA decodificati) o stesso stato della
   scena (specifica senza chiave, versione e stile, cioè la stessa istanza con un altro stile). I record legati formano
   un gruppo; la coppia a/b è un gruppo. Un record di training in un gruppo con un record di test o delle coppie
   (insiemi di valutazione) è escluso dal training; i duplicati interni a uno split si contano soltanto.

Stato della scena, per le coppie del gruppo a/b, con le categorie tenute distinte:
- `escluso`: errore_matematico (domanda errata, misura o angolo scritti incoerenti col disegno), visivo_integrita
  (violazione bloccante del renderer, etichette dei vertici, artefatto non integro o non riproducibile, contratto della
  coppia), contaminazione (legame con la valutazione);
- `quarantena`: incerto (testi sovrapposti, testo su un marcatore, testo piccolo, domanda non ammissibile o non
  verificabile, violazione non classificata), formato_minore;
- `candidato`: nessuno dei precedenti. Vuol dire approvato dai controlli dichiarati: non è gold certificato (S2 §2.4,
  S4) e non dimostra che ogni risposta sia corretta.

Uscite nella cartella nuova --out: report.json (conteggi, copertura per famiglia e compito prima e dopo i filtri,
duplicati e split, ingressi con sha256, identità del codice), records.jsonl (una riga per record, con stato, motivi e
hash), excluded.tsv e quarantine.tsv (chiave, split, famiglia, categorie, motivi). Esce con errore, dopo aver scritto
il report, se un controllo dell'insieme fallisce (shard, chiavi, confronto col controllo, conteggi).

    srun $RUN_PYTHON -u -B scripts/synth_candidate.py --shards DIR --check DIR --out DIR [--reference MANIFEST]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import synth, synth_check  # noqa: E402
from limite_vl.provenance import (HashingTextWriter, code_identity, file_entry, refuse_existing,  # noqa: E402
                                  sha256_file, write_new_json)
from synth_render import scene as render_scene  # noqa: E402  (stessa cartella scripts/: le coppie come nel render)

SCHEMA = "synth-candidate/1"
VERSION = 4
SPLITS = {"train": "training", "test": "valutazione", "pairs": "valutazione"}
STYLE = ("lw", "font", "font_family", "width", "height")  # synth._style: lo stile, non lo stato della scena
MATH = ("lunghezza_incoerente", "angolo_incoerente")
VISUAL = (*synth_check.BLOCKING, "etichette_dei_vertici")
UNCERTAIN = ("testi_sovrapposti", "testo_su_marcatore", "testo_piccolo")
EXCLUDING = ("errore_matematico", "visivo_integrita", "contaminazione")
QUARANTINE = ("incerto", "formato_minore")
CATEGORIES = EXCLUDING + QUARANTINE
STATES = ("candidato", "quarantena", "escluso")
FIELDS = ("id", "family", "width", "height", "max_patches", "qa", "spec", "layout")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def roundtrip(obj):
    """Come lo scrive e lo rilegge il JSON dello shard (tuple in liste)."""
    return json.loads(json.dumps(obj, ensure_ascii=False))


def write_tsv(path: Path, header: list[str], lines: list[str]) -> dict:
    """TSV in un file nuovo (scritto in .tmp e rinominato), con percorso, dimensione e sha256."""
    refuse_existing(path)
    text = "\t".join(header) + "\n" + "".join(line + "\n" for line in lines)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return {**file_entry(path, hashlib.sha256(text.encode("utf-8")).hexdigest()), "lines": len(lines)}


def scene_state(spec: dict) -> dict:
    """Stato della scena: la specifica senza chiave, versione del generatore e stile."""
    return {k: v for k, v in spec.items() if k not in ("key", "generator_version", *STYLE)}


def expected_keys(split: str, args: dict) -> list[tuple[str, str]]:
    """(chiave, famiglia) nell'ordine di synth_render.py, dagli argomenti registrati nel suo manifest."""
    fam = synth.FAMILIES
    if split == "train":
        return [(f"synth-{i:07d}", fam[i % len(fam)]) for i in range(int(args["train"]))]
    if split == "test":
        return [(f"synth-test-{i:05d}", fam[i % len(fam)]) for i in range(int(args["test"]))]
    return [(f"synth-pair-{i:05d}-{v}", fam[i % len(fam)]) for i in range(int(args["pairs"])) for v in "ab"]


# ---------------------------------------------------------------- un record

def read_tar(path: str, problems: list[str]):
    """(chiave, {estensione: byte}) nell'ordine del tar, come data.iter_shard; i problemi di struttura (membri non
    regolari, file ripetuti, chiavi non consecutive) vanno in `problems`."""
    key, files, seen = None, {}, set()
    with tarfile.open(path, "r|") as tar:
        for member in tar:
            if not member.isfile():
                problems.append(f"membro non regolare: {member.name}")
                continue
            stem, _, ext = member.name.rpartition("/")[2].partition(".")
            if stem != key:
                if files:
                    yield key, files
                if stem in seen:
                    problems.append(f"chiave non consecutiva: {stem}")
                seen.add(stem)
                key, files = stem, {}
            if ext in files:
                problems.append(f"file ripetuto: {member.name}")
            files[ext] = tar.extractfile(member).read()
    if files:
        yield key, files


def decode_png(data: bytes) -> tuple[tuple[int, int] | None, str | None, str | None]:
    """(dimensioni, modo, sha256 di dimensioni e valori RGBA) del PNG; (None, None, None) se non si decodifica."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data), formats=("PNG",))
        image.load()
        pixels = hashlib.sha256(f"{image.size[0]}x{image.size[1]}:".encode() + image.convert("RGBA").tobytes())
        return image.size, image.mode, pixels.hexdigest()
    except Exception:  # noqa: BLE001 - un PNG illeggibile è un problema d'integrità del record
        return None, None, None


def record_problems(key: str, family: str, files: dict[str, bytes]) -> tuple[dict | None, list[str], list[str], dict]:
    """(meta, problemi d'integrità, problemi di formato, hash) di un record letto dallo shard, senza ridisegno."""
    integrity, fmt = [], []
    hashes = {"png_sha256": None, "pixels_sha256": None, "json_sha256": None}
    if set(files) != {"png", "json"}:
        integrity.append(f"file del record {sorted(files)} invece di png e json")
    if "json" not in files:
        return None, integrity, fmt, hashes
    hashes["json_sha256"] = hashlib.sha256(files["json"]).hexdigest()
    try:
        meta = json.loads(files["json"])
    except ValueError:
        return None, integrity + ["JSON non valido"], fmt, hashes
    missing = [f for f in FIELDS if f not in meta]
    if missing:
        return None, integrity + [f"campi mancanti: {missing}"], fmt, hashes
    spec = meta["spec"]
    pair = key.startswith("synth-pair-")
    if meta["id"] != key:
        integrity.append(f"id {meta['id']} diverso dalla chiave")
    if not meta["family"] == spec.get("family") == family:
        integrity.append(f"famiglia {meta['family']}/{spec.get('family')} invece di {family}")
    if spec.get("generator_version") != VERSION:
        integrity.append(f"versione del generatore {spec.get('generator_version')}")
    w, h = meta["width"], meta["height"]
    if (w, h) != (spec.get("width"), spec.get("height")):
        integrity.append("dimensioni diverse da quelle della specifica")
    if w % 32 or h % 32:
        integrity.append(f"{w}x{h}: lati non multipli di 32")
    if meta["max_patches"] != (w // 16) * (h // 16):
        integrity.append(f"max_patches {meta['max_patches']} invece di {(w // 16) * (h // 16)}")
    qs = list(meta["qa"])
    if pair:
        if meta["qa"] or "pair" not in meta:
            integrity.append("record di coppia con domande proprie o senza pair")
        else:
            qs.append(meta["pair"])
    elif not meta["qa"] or "pair" in meta:
        integrity.append("scena senza domande o con pair")
    for q in qs:
        opts = q.get("options") or []
        if not all(isinstance(q.get(f), str) and q.get(f) for f in ("type", "q", "answer")):
            integrity.append(f"domanda incompleta: {q}")
            continue
        if q["answer"] not in opts or len(set(opts)) != len(opts) or not all(isinstance(o, str) and o for o in opts):
            integrity.append(f"{q['type']}: risposta {q['answer']!r} e opzioni {opts}")
        for field in ("q", "answer"):
            if q[field] != q[field].strip() or "  " in q[field]:
                fmt.append(f"spazi in {field}: {q[field]!r}")
    fmt += [f"domanda ripetuta {n} volte: {q}" for q, n in Counter(q.get("q") for q in meta["qa"]).items() if n > 1]
    if "png" in files:
        hashes["png_sha256"] = hashlib.sha256(files["png"]).hexdigest()
        size, mode, hashes["pixels_sha256"] = decode_png(files["png"])
        if size is None:
            integrity.append("PNG che non si decodifica")
        else:
            if size != (w, h):
                integrity.append(f"PNG {size[0]}x{size[1]} invece di {w}x{h}")
            if mode != "RGB":
                integrity.append(f"PNG in modo {mode}")
    layout = meta["layout"]
    if not isinstance(layout, dict) or layout.get("png_sha256") != hashes["png_sha256"]:
        integrity.append("layout senza png_sha256 uguale allo sha256 del PNG")
    elif layout.get("size") != [w, h]:
        integrity.append(f"layout di dimensioni {layout.get('size')}")
    return meta, integrity, fmt, hashes


def reproduce(key: str, family: str, meta: dict, png: bytes | None) -> list[str]:
    """Problemi di riproducibilità di una scena (non coppia): specifica e domande dalla chiave, PNG e layout ridisegnati."""
    out = []
    spec = synth.make_spec(key, family, VERSION)
    if roundtrip(spec) != meta["spec"]:
        out.append("specifica diversa da quella rigenerata dalla chiave")
    if roundtrip(synth.questions(spec)) != meta["qa"]:
        out.append("domande diverse da quelle rigenerate dalla chiave")
    image, layout = synth.render(meta["spec"], layout=True)
    data = synth.png_bytes(image)
    if data != png:
        out.append("PNG ridisegnato con byte diversi")
    layout["png_sha256"] = hashlib.sha256(data).hexdigest()
    if roundtrip(layout) != meta["layout"]:
        out.append("layout ridisegnato diverso da quello salvato")
    return out


def reproduce_pair(pair_id: str, family: str, saved: dict[str, tuple[dict, bytes | None]]) -> dict[str, list[str]]:
    """Problemi di riproducibilità di una coppia, per variante: la coppia rigenerata come in synth_render.scene."""
    out = {k: [] for k in saved}
    regen = {k: (meta, png) for k, png, meta in render_scene((pair_id, family, VERSION))}
    for k in saved:
        if k not in regen:
            out[k].append("variante assente nella coppia rigenerata")
            continue
        meta, png = saved[k]
        if roundtrip(regen[k][0]) != meta:
            out[k].append("record diverso da quello della coppia rigenerata")
        if regen[k][1] != png:
            out[k].append("PNG diverso da quello della coppia rigenerata")
    return out


def pair_contract(ma: dict, mb: dict, pa: bytes | None, pb: bytes | None) -> list[str]:
    """Problemi del contratto di una coppia controfattuale (synth.counterfactual), con controlli scritti qui: stessa
    domanda e stesse opzioni in a e b, risposte diverse fra le opzioni, immagini diverse, un solo dettaglio cambiato:
    coord un punto riflesso su un asse; function il coefficiente a di un grafico cambiato di segno (le etichette dei
    nomi possono spostarsi); polygon la figura riscalata in modo uniforme, con una sola misura scritta; shapes una
    figura in più o in meno."""
    A, B = ma.get("pair"), mb.get("pair")
    if not A or not B:
        return ["record senza pair"]
    out = []
    if A.get("id") != B.get("id") or (A.get("variant"), B.get("variant")) != ("a", "b"):
        out.append("identificativo o varianti della coppia")
    if (A.get("q"), A.get("type"), A.get("options")) != (B.get("q"), B.get("type"), B.get("options")):
        out.append("domanda, tipo o opzioni diversi fra a e b")
    if A.get("answer") == B.get("answer"):
        out.append("stessa risposta in a e b")
    if pa is None or pa == pb:
        out.append("immagini uguali o mancanti")
    sa, sb = ma["spec"], mb["spec"]
    fam = sa.get("family")
    if sb.get("family") != fam:
        return out + ["famiglie diverse fra a e b"]

    def rest_same(*fields: str) -> bool:
        return {k: v for k, v in sa.items() if k not in fields} == {k: v for k, v in sb.items() if k not in fields}

    ok = False
    if fam == "coord":
        pts_a, pts_b = sa["points"], sb["points"]
        diff = [(p, q) for p, q in zip(pts_a, pts_b) if p != q]
        if len(pts_a) == len(pts_b) and len(diff) == 1 and sa["line"] is None and rest_same("points"):
            p, q = diff[0]
            moved = [ax for ax in ("x", "y") if p[ax] != q[ax]]
            ok = p["label"] == q["label"] and len(moved) == 1 and q[moved[0]] == -p[moved[0]]
    elif fam == "function":
        ca, cb = sa["curves"], sb["curves"]

        def bare(c: dict) -> dict:
            return {k: v for k, v in c.items() if k != "label_at"}

        diff = [(c, d) for c, d in zip(ca, cb) if bare(c) != bare(d)]
        if len(ca) == len(cb) and len(diff) == 1 and rest_same("curves"):
            c, d = diff[0]
            ok = {k: v for k, v in bare(c).items() if k != "a"} == {k: v for k, v in bare(d).items() if k != "a"} \
                and d["a"] == -c["a"] and all(x.get("label_at") for x in cb)
    elif fam == "polygon":
        pts_a, pts_b = sa["points"], sb["points"]
        if len(pts_a) == len(pts_b) and len(sa["shown"]) == 1 and rest_same("points", "lengths"):
            ref = max(range(len(pts_a)), key=lambda j: abs(pts_a[j][0]) + abs(pts_a[j][1]))
            axis = 0 if abs(pts_a[ref][0]) >= abs(pts_a[ref][1]) else 1
            k = pts_b[ref][axis] / pts_a[ref][axis] if pts_a[ref][axis] else 0.0
            scale = max(abs(v) for p in pts_b for v in p)
            ok = k > 0 and k != 1 and all(abs(pb_[t] - k * pa_[t]) <= 1e-9 * scale
                                          for pa_, pb_ in zip(pts_a, pts_b) for t in (0, 1))
    elif fam == "shapes":
        if rest_same("shapes"):
            a, b = Counter(canonical(s) for s in sa["shapes"]), Counter(canonical(s) for s in sb["shapes"])
            ok = sum(((a - b) + (b - a)).values()) == 1
    if not ok:
        out.append(f"{fam}: il cambio fra a e b non è il solo dettaglio previsto")
    return out


def categorize(result: dict, integrity: list[str], fmt: list[str]) -> dict[str, list[str]]:
    """Motivi per categoria, dagli esiti di synth_check.check_sample e dai controlli sugli artefatti (S2 §2.6). Con una
    violazione bloccante le domande ok diventano non ammissibili: lì il motivo è la violazione, non la domanda."""
    cats: dict[str, list[str]] = defaultdict(list)
    for v in result["violations"]:
        kind = v["kind"]
        where = " ".join(v.get("gids") or [])
        if kind in MATH:
            cats["errore_matematico"].append(f"{kind} {where}: {v.get('detail')}")
        elif kind in VISUAL:
            cats["visivo_integrita"].append(f"{kind} {where}".strip())
        elif kind in UNCERTAIN:
            cats["incerto"].append(f"{kind} {where}".strip())
        else:
            cats["incerto"].append(f"violazione non classificata {kind} {where}".strip())
    for r in result["questions"]:
        if r["status"] == "errata":
            cats["errore_matematico"].append(f"{r['type']} errata: risposta {r['answer']}, si vede {r['expected']}")
        elif r["status"] in ("non_ammissibile", "non_verificabile"):
            if not result["blocking"]:
                cats["incerto"].append(f"{r['type']} {r['status']}: {r['why']}")
        elif r["status"] != "ok":
            cats["incerto"].append(f"{r['type']}: esito {r['status']}")
    if integrity:
        cats["visivo_integrita"].extend(integrity)
    if fmt:
        cats["formato_minore"].extend(fmt)
    return dict(cats)


def state_of(categories: dict) -> str:
    if any(c in categories for c in EXCLUDING):
        return "escluso"
    if any(c in categories for c in QUARANTINE):
        return "quarantena"
    return "candidato"


def _row(key, split, family, shard, meta, integrity, fmt, hashes, result) -> dict:
    spec = meta["spec"] if meta else {}
    checked = result is not None
    return {"key": key, "split": split, "shard": shard, "family": family,
            "pair": meta.get("pair", {}).get("id") if meta else None,
            **hashes,
            "state_sha256": digest(scene_state(spec)) if meta else None,
            "size": [meta["width"], meta["height"]] if meta else None,
            "max_patches": meta["max_patches"] if meta else None,
            "questions": [[r["type"], r["answer"], r["status"]] for r in result["questions"]] if checked else [],
            "violations": sorted({v["kind"] for v in result["violations"]}) if checked else [],
            "checker_verified": bool(checked and result["verified"]),
            "_integrity": integrity, "_format": fmt, "_result": result}


def shard_task(task: tuple[str, str, list[tuple[str, str]]]) -> dict:
    """Tutti i record di uno shard: integrità, riproducibilità, verifica indipendente, coppie, formato."""
    split, path, expected = task
    shard_problems: list[str] = []
    rows, seen = [], []
    fam_of = dict(expected)
    pending: dict[str, tuple] = {}
    for key, files in read_tar(path, shard_problems):
        seen.append(key)
        family = fam_of.get(key) or synth.FAMILIES[0]
        meta, integrity, fmt, hashes = record_problems(key, family, files)
        png = files.get("png")
        if key not in fam_of:
            integrity.append("chiave non attesa in questo shard")
        if meta is None:
            rows.append(_row(key, split, family, Path(path).name, None, integrity, fmt, hashes, None))
            continue
        # il layout salvato vale solo se è legato al PNG dal digest; altrimenti la semantica si legge dal ridisegno
        stale = any(p.startswith("layout ") for p in integrity)
        layout = synth.render(meta["spec"], layout=True)[1] if stale else meta["layout"]
        result = synth_check.check_sample(meta, layout)
        if split == "pairs":
            pending[key] = (meta, png, integrity, fmt, hashes, result)
            pair_id = key.rsplit("-", 1)[0]
            ka, kb = f"{pair_id}-a", f"{pair_id}-b"
            if ka in pending and kb in pending:
                (ma, pa, *_), (mb, pb, *_) = pending[ka], pending[kb]
                contract = pair_contract(ma, mb, pa, pb)
                repro = reproduce_pair(pair_id, family, {ka: (ma, pa), kb: (mb, pb)})
                for k in (ka, kb):
                    m, _, integ, f, hs, res = pending.pop(k)
                    integ = integ + repro[k] + [f"coppia: {c}" for c in contract]
                    rows.append(_row(k, split, family, Path(path).name, m, integ, f, hs, res))
            continue
        integrity += reproduce(key, family, meta, png)
        rows.append(_row(key, split, family, Path(path).name, meta, integrity, fmt, hashes, result))
    for k, (m, _, integ, f, hs, res) in pending.items():  # variante senza la sua compagna
        rows.append(_row(k, split, fam_of.get(k, synth.FAMILIES[0]), Path(path).name, m,
                         integ + ["coppia: variante senza la compagna nello shard"], f, hs, res))
    if seen != [k for k, _ in expected]:
        shard_problems.append(f"{len(seen)} chiavi lette, {len(expected)} attese, o in un altro ordine")
    for r in rows:
        r["categories"] = categorize(r.pop("_result") or {"violations": [], "questions": [], "blocking": []},
                                     r.pop("_integrity"), r.pop("_format"))
    return {"split": split, "path": path, "rows": rows, "problems": shard_problems}


# ---------------------------------------------------------------- fra i record

def link_groups(rows: list[dict]) -> dict[str, list[str]]:
    """Gruppi per legami confermati (identità esatte): stessi pixel, stesso stato della scena, coppia a/b. Restituisce
    rappresentante -> membri; aggiunge a ogni riga `group` e `group_size`."""
    parent: dict[str, str] = {r["key"]: r["key"] for r in rows}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by: dict[tuple, str] = {}
    for r in rows:
        for link in (("pixels", r.get("pixels_sha256")), ("stato", r.get("state_sha256")), ("coppia", r.get("pair"))):
            if link[1] is None:
                continue
            if link in by:
                union(by[link], r["key"])
            else:
                by[link] = r["key"]
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        groups[find(r["key"])].append(r["key"])
    for r in rows:
        g = find(r["key"])
        r["group"], r["group_size"] = g, len(groups[g])
    return dict(groups)


def apply_split_rules(rows: list[dict], groups: dict[str, list[str]]) -> dict:
    """Regole già previste (pipeline §§5.5, 5.8.5; S1 §3; scripts/split_duplicates.py): un record di training in un
    gruppo con record di valutazione è escluso dal training (categoria contaminazione, con le chiavi legate); i
    duplicati interni a uno split si contano soltanto. Restituisce i conteggi del report."""
    by_key = {r["key"]: r for r in rows}
    stats: dict = {"per_split": {}, "groups_across": Counter(), "train_excluded": 0, "train_excluded_by_family": Counter()}
    for split in SPLITS:
        members = [r for r in rows if r["split"] == split]
        pix = Counter(r["pixels_sha256"] for r in members if r["pixels_sha256"])
        st = Counter(r["state_sha256"] for r in members if r["state_sha256"])
        stats["per_split"][split] = {
            "records": len(members), "distinct_pixels": len(pix), "distinct_states": len(st),
            "records_with_pixel_duplicate": sum(n for n in pix.values() if n > 1),
            "records_with_state_duplicate": sum(n for n in st.values() if n > 1),
            "distinct_states_by_family": {f: len({r["state_sha256"] for r in members if r["family"] == f})
                                          for f in synth.FAMILIES}}
    for members in groups.values():
        splits = sorted({by_key[k]["split"] for k in members})
        if len(splits) > 1:
            stats["groups_across"]["+".join(splits)] += 1
        evaluation = sorted(k for k in members if SPLITS[by_key[k]["split"]] == "valutazione")
        if not evaluation:
            continue
        for k in members:
            r = by_key[k]
            if SPLITS[r["split"]] != "training":
                continue
            kinds = [name for name, field in (("pixel", "pixels_sha256"), ("stato", "state_sha256"))
                     if any(by_key[e][field] == r[field] for e in evaluation)]
            r["categories"].setdefault("contaminazione", []).append(
                f"gruppo con {len(evaluation)} record di valutazione ({', '.join(evaluation[:3])}"
                f"{', …' if len(evaluation) > 3 else ''}); legame: {'/'.join(kinds) or 'tramite altri record'}")
            stats["train_excluded"] += 1
            stats["train_excluded_by_family"][r["family"]] += 1
    stats["groups_across"] = dict(stats["groups_across"])
    stats["train_excluded_by_family"] = dict(stats["train_excluded_by_family"])
    return stats


def assign_states(rows: list[dict]) -> None:
    """Stato di ogni record; nelle coppie lo stato peggiore fra a e b, con le categorie di entrambe."""
    for r in rows:
        r["state"] = state_of(r["categories"])
    order = {s: i for i, s in enumerate(STATES)}
    pairs: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["pair"]:
            pairs[r["pair"]].append(r)
    for members in pairs.values():
        worst = max((r["state"] for r in members), key=order.get)
        for r in members:
            if r["state"] != worst:
                r["categories"].setdefault("coppia", []).append(f"stato {worst} dell'altra variante")
            r["state"] = worst


def coverage(rows: list[dict]) -> dict:
    """Scene e domande per split, famiglia e compito, prima e dopo i filtri, con gli esiti del checker."""
    scenes: dict = {}
    questions: dict = {}
    for r in rows:
        s = scenes.setdefault(r["split"], {}).setdefault(r["family"], Counter())
        s["generate"] += 1
        s[r["state"]] += 1
        for cat in r["categories"]:
            s[f"categoria/{cat}"] += 1
        for qtype, _, status in r["questions"]:
            q = questions.setdefault(r["split"], {}).setdefault(r["family"], {}).setdefault(qtype, Counter())
            q["prima"] += 1
            q[f"checker/{status}"] += 1
            q[f"stato/{r['state']}"] += 1
            q["dopo"] += r["state"] == "candidato"

    def plain(tree: dict) -> dict:  # i Counter sono dict: si riconoscono prima
        return {k: dict(sorted(v.items())) if isinstance(v, Counter) else plain(v) for k, v in sorted(tree.items())}

    return {"scenes": plain(scenes), "questions": plain(questions)}


def answers_after(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        if r["state"] == "candidato":
            for qtype, answer, _ in r["questions"]:
                out.setdefault(r["split"], {}).setdefault(qtype, Counter())[answer] += 1
    return {s: {t: dict(sorted(c.items())) for t, c in sorted(v.items())} for s, v in sorted(out.items())}


def compare_check(rows: list[dict], check_dir: Path, shard_sha: dict[str, str]) -> tuple[dict, list[str]]:
    """Confronto col controllo di scripts/synth_semantics_check.py sugli stessi shard: stessi file (sha256), stessi
    campioni, e i non verificati sono esattamente quelli che qui non superano il checker sul layout del PNG."""
    report = json.loads((check_dir / "report.json").read_text())
    problems = []
    inputs = {Path(i["path"]).name: i["sha256"] for i in report["inputs"] if "path" in i}
    if inputs != shard_sha:
        problems.append(f"il controllo ha letto altri shard: {sorted(inputs)} contro {sorted(shard_sha)}")
    total = report["samples"].get("shard", 0)
    if total != len(rows):
        problems.append(f"il controllo conta {total} campioni, qui {len(rows)}")
    theirs = set()
    with open(check_dir / "problems.jsonl", encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            if p.get("what") == "sample" and p.get("origin") == "shard":
                theirs.add(p["key"])
    ours = {r["key"] for r in rows if not (r["checker_verified"] and not any("png_sha256" in x for x in
                                                                              r["categories"].get("visivo_integrita", [])))}
    if theirs != ours:
        problems.append(f"campioni non verificati diversi: {len(theirs - ours)} solo nel controllo, "
                        f"{len(ours - theirs)} solo qui (es. {sorted(theirs ^ ours)[:5]})")
    if report.get("incoherent"):
        problems.append(f"il controllo ha conteggi incoerenti: {report['incoherent']}")
    summary = {"report_sha256": sha256_file(check_dir / "report.json"),
               "problems_sha256": sha256_file(check_dir / "problems.jsonl"),
               "job_id": report.get("job_id"), "code": report.get("code"), "samples": report.get("samples"),
               "samples_by_origin": report.get("samples_by_origin"), "non_verificati": len(theirs),
               "coincide": not problems}
    return summary, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, type=Path, help="cartella del render (manifest.json, train/, test/, pairs/)")
    ap.add_argument("--check", required=True, type=Path, help="uscita di synth_semantics_check.py sugli stessi shard")
    ap.add_argument("--reference", type=Path, help="manifest di un render precedente (v3) per la copertura")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("synth_candidate.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: scegliere un'altra cartella")
    manifest_path = args.shards / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("generator_version") != VERSION:
        sys.exit(f"{manifest_path}: generator_version {manifest.get('generator_version')}, attesa {VERSION}")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # shard del manifest: sha256, dimensione, nessun altro file
    fatal, tasks, shard_sha = [], [], {}
    for split in SPLITS:
        listed = manifest["sets"][split]
        present = sorted(p.name for p in (args.shards / split).iterdir() if p.name != "manifest.json")
        if present != sorted(r["name"] for r in listed):
            fatal.append(f"{split}: file presenti {present}, nel manifest {[r['name'] for r in listed]}")
        keys = expected_keys(split, manifest["args"])  # gli shard le contengono di seguito, nell'ordine del manifest
        start = 0
        for row in listed:
            expected = keys[start:start + row["samples"]]
            start += row["samples"]
            path = args.shards / split / row["name"]
            if not path.is_file():
                fatal.append(f"{path} mancante")
                continue
            sha = sha256_file(path)
            shard_sha[row["name"]] = sha
            if sha != row["sha256"] or path.stat().st_size != row["bytes"]:
                fatal.append(f"{row['name']}: sha256 o dimensione diversi dal manifest del render")
            tasks.append((split, str(path), expected))
        if start != len(keys):
            fatal.append(f"{split}: {start} record nel manifest, {len(keys)} attesi dai suoi argomenti")

    rows, shard_rows = [], []
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(shard_task, tasks):
            rows.extend(res["rows"])
            shard_rows.append({"path": res["path"], "records": len(res["rows"]), "problems": res["problems"]})
            fatal.extend(f"{Path(res['path']).name}: {p}" for p in res["problems"])
            print(f"{Path(res['path']).name}: {len(res['rows'])} record, problemi di struttura {len(res['problems'])} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    rows.sort(key=lambda r: (list(SPLITS).index(r["split"]), r["key"]))

    # dimensioni viste dal modello: il processor lascia l'immagine invariata
    from limite_vl import processing

    sizes = {}
    for w, h in sorted({tuple(r["size"]) for r in rows if r["size"]}):
        mp = (w // 16) * (h // 16)
        th, tw = processing.target_size(h, w, mp)  # (altezza, larghezza)
        sizes[f"{w}x{h}"] = {"max_patches": mp, "processor": f"{tw}x{th}", "invariata": (tw, th) == (w, h),
                             "entro_DIAGRAM_MAX_PATCHES": mp <= processing.DIAGRAM_MAX_PATCHES}
    changed = sorted(k for k, v in sizes.items() if not (v["invariata"] and v["entro_DIAGRAM_MAX_PATCHES"]))
    for r in rows:
        if r["size"] and f"{r['size'][0]}x{r['size'][1]}" in changed:
            r["categories"].setdefault("visivo_integrita", []).append("il processor cambia la dimensione dell'immagine")

    groups = link_groups(rows)
    split_stats = apply_split_rules(rows, groups)
    assign_states(rows)
    check_summary, check_problems = compare_check(rows, args.check, shard_sha)
    fatal += check_problems

    records = HashingTextWriter(args.out / "records.jsonl")
    lines: dict[str, list[str]] = {"escluso": [], "quarantena": []}
    for r in rows:
        records.write_json_line(r)
        if r["state"] in lines:
            reasons = " | ".join(f"{c}: {x}" for c, xs in r["categories"].items() for x in xs[:3])
            lines[r["state"]].append("\t".join([r["key"], r["split"], r["family"], ",".join(sorted(r["categories"])),
                                                reasons.replace("\t", " ").replace("\n", " ")]))
    files = {"records": records.close()}
    for name, state in (("excluded", "escluso"), ("quarantine", "quarantena")):
        files[name] = write_tsv(args.out / f"{name}.tsv", ["chiave", "split", "famiglia", "categorie", "motivi"],
                                lines[state])

    states = Counter((r["split"], r["state"]) for r in rows)
    cats = Counter((r["split"], c) for r in rows for c in r["categories"])
    pairs_by_state = Counter(r["state"] for r in rows if r["split"] == "pairs" and r["key"].endswith("-a"))
    incoherent = []
    for split in SPLITS:
        n = sum(1 for r in rows if r["split"] == split)
        if sum(states[(split, s)] for s in STATES) != n:
            incoherent.append(f"{split}: stati che non sommano a {n}")
    if sum(pairs_by_state.values()) * 2 != sum(1 for r in rows if r["split"] == "pairs"):
        incoherent.append("coppie: varianti a e b non appaiate")
    fatal += incoherent

    reference = None
    if args.reference:
        ref = json.loads(args.reference.read_text())
        counts = Counter()
        for r in rows:
            if r["split"] == "train":
                for qtype, _, _ in r["questions"]:
                    counts[(r["family"], qtype, "v4_prima")] += 1
                    counts[(r["family"], qtype, "v4_dopo")] += r["state"] == "candidato"
        for k, v in ref["counts"]["train"].items():
            if k.startswith("qa/"):
                _, fam, qtype = k.split("/")
                counts[(fam, qtype, "riferimento")] += v
        reference = {"manifest": str(args.reference), "sha256": sha256_file(args.reference), "train": {}}
        for (fam, qtype, what), v in sorted(counts.items()):
            reference["train"].setdefault(fam, {}).setdefault(qtype, {})[what] = v

    report = {
        "schema": SCHEMA, "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "render": {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
                   "job_id": manifest.get("job_id"), "args": manifest.get("args"), "code": manifest.get("code"),
                   "generator_version": manifest.get("generator_version"), "shards": shard_sha,
                   "seed": "synth.rng_for(chiave): sha256 della chiave, primi 8 byte little-endian; domande da "
                           "chiave + '/qa', punto segnato dei polygon da chiave + '/m', coppie da chiave + '/cf' "
                           "e tentativi chiave/0..19; famiglie a rotazione i % 4"},
        "splits": {s: {"ruolo": role, "records": sum(1 for r in rows if r["split"] == s)} for s, role in SPLITS.items()},
        "thresholds": {k: getattr(synth_check, k) for k in ("EDGE", "CLOSE_ROOTS", "TANGENT", "SIDE", "ON_LINE",
                                                             "OFF_LINE", "ANGLE_TOL", "LENGTH_TOL", "EXTREME",
                                                             "MIN_TEXT_PX", "OVERLAP_PX", "CLIP_PX", "TICK_PX",
                                                             "SCALE_TOL")},
        "categories": {"escluso": list(EXCLUDING), "quarantena": list(QUARANTINE),
                       "violazioni": {"errore_matematico": list(MATH), "visivo_integrita": list(VISUAL),
                                      "incerto": list(UNCERTAIN)}},
        "states": {s: {st: states[(s, st)] for st in STATES} for s in SPLITS},
        "categories_by_split": {s: {c: cats[(s, c)] for c in (*CATEGORIES, "coppia") if cats[(s, c)]} for s in SPLITS},
        "pairs": {"coppie": sum(pairs_by_state.values()), "per_stato": dict(pairs_by_state)},
        "integrity": {"shards": shard_rows, "image_sizes": sizes, "sizes_changed_by_processor": changed},
        "duplicates_and_splits": split_stats, "groups": len(groups),
        "check": check_summary, "coverage": coverage(rows), "answers_after": answers_after(rows),
        "reference": reference, "files": files, "fatal": fatal, "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("states", "categories_by_split", "pairs", "groups", "seconds")},
                     ensure_ascii=False))
    print(json.dumps(split_stats, ensure_ascii=False))
    if fatal:
        sys.exit("controlli dell'insieme falliti (report scritto): " + "; ".join(fatal[:10]))


if __name__ == "__main__":
    main()
