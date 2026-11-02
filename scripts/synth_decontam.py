#!/usr/bin/env python3
"""Decontaminazione esterna dei sintetici v4 contro il pannello di valutazione e nuova versione del manifest delle
ammissioni (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §§2.8-2.9, docs/PANNELLO_VALUTAZIONE.md §6, S1 §§2-5). Solo in
un job SLURM CPU. Manifest, shard, indice, pannello e scansione si leggono soltanto; si scrive solo nella cartella
nuova --out.

Ingressi:
- --manifest: il manifest della versione precedente (data/manifests/synth-v4-candidato-1), letto con manifest.load;
- --index: la cartella dell'indice del pannello (index.json e verifica.json di scripts/panel_index.py);
- --panel: la cartella dei gruppi del pannello (elements.jsonl.gz e summary.json di scripts/panel_groups.py);
- --scan: l'uscita di `scripts/decontam.py scan --kind synth` contro quell'indice (summary.json, records.jsonl.gz,
  matches.jsonl).

`verifica` controlla la compatibilità e scrive --out/report.json; il job esce con errore se un controllo fallisce.
- Indice: convenzione dichiarata `rgb-su-bianco/1`, schema, sha256 (anche contro --index-sha256 e contro quello
  registrato dal pannello), verifica.json con tutte le voci ritrovate nelle sorgenti.
- Perimetro: ogni elemento del pannello, in ogni ruolo (sviluppo, test ammesso o no alla metrica primaria,
  diagnostica, solo_decontaminazione), ha voci nell'indice e ogni voce ha il suo elemento. La scansione confronta
  tutte le voci, senza filtri per ruolo.
- Scansione: stesso indice (sha256) e stessa convenzione, tipo `synth`, identità dei pixel attiva, solo shard del
  manifest con gli sha256 del manifest.
- Identificativi: ogni riga della scansione corrisponde a un solo record del manifest, nello stesso shard e nella
  stessa posizione, con lo sha256 dei byte del PNG registrato nel manifest; ogni corrispondenza ritrova il suo elemento
  del pannello. Con --partial basta una scansione limitata (alcuni shard); senza, ogni record del manifest va
  scansionato esattamente una volta.

`manifest` ripete le verifiche senza --partial e scrive la versione nuova del manifest (S1 §7, scritta una volta):
- esito per record:
  - `corrispondenza_confermata`: pixel identici, dopo la normalizzazione, a un'immagine del pannello;
  - `candidato`: pHash entro il raggio o un 10-gram delle domande in comune. Sono somiglianze da verificare, non prove;
  - `non_verificabile`: immagine assente o illeggibile, o identità dei pixel non eseguibile;
  - `nessuna_corrispondenza`: nessuna corrispondenza nei controlli eseguiti. Non certifica la risposta né l'assenza di
    ogni contaminazione;
- stato per gruppo, perché la scena con le sue varianti è l'unità di separazione (S1 §§3, 5): un membro con una
  corrispondenza confermata esclude tutto il gruppo (`escluso`); un candidato o un controllo incompleto lo mette in
  `quarantena`. Lo stato non migliora mai: un record già escluso o in quarantena per altri motivi resta tale (gravità
  escluso > quarantena > ammesso). Split e gruppi restano quelli della versione precedente;
- ogni record riceve l'evidenza `decontaminazione_pannello`: esito proprio e del gruppo, controlli eseguiti, conteggi
  per tipo e fino a MAX_EVIDENCE elementi del pannello con benchmark, ruolo e gruppo, più il riferimento a
  records.jsonl.gz della scansione con lo sha256. Il manifest registra versioni e sha256 di tutti gli ingressi.

    srun $RUN_PYTHON -u -B scripts/synth_decontam.py verifica --manifest DIR --index DIR --panel DIR --scan DIR \\
        --out DIR [--partial] [--index-sha256 HEX]
    srun $RUN_PYTHON -u -B scripts/synth_decontam.py manifest --manifest DIR --index DIR --panel DIR --scan DIR \\
        --out DIR --version 2 [--index-sha256 HEX]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.decontam import IMAGE_CONVENTION, index_convention  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, refuse_existing, sha256_file, write_new_json  # noqa: E402
from synth_manifest import gzip_lines  # noqa: E402

CHECK = "decontaminazione_pannello"
SCHEMA = "synth-decontam/1"
MAX_EVIDENCE = 10  # elementi del pannello riportati nell'evidenza di un record (i conteggi sono completi)
SEVERITY = {"ammesso": 0, "quarantena": 1, "escluso": 2}
OUTCOMES = ("corrispondenza_confermata", "candidato", "non_verificabile", "nessuna_corrispondenza")
GROUP_STATE = {"corrispondenza_confermata": "escluso", "candidato": "quarantena", "non_verificabile": "quarantena"}
REASON = {"corrispondenza_confermata": "pixel identici a un'immagine del pannello",
          "candidato": "candidato da verificare (pHash o n-gram delle domande)",
          "non_verificabile": "controllo delle immagini incompleto"}
RULES = {
    "confermata": "pixel identici dopo la normalizzazione rgb-su-bianco/1 (sha256 dei pixel uguale a quello di una "
                  "voce dell'indice)",
    "candidato": "pHash entro il raggio della scansione, oppure un n-gram di normalize_words in comune fra una domanda "
                 "della scena e un testo dell'indice (senza gli n-gram di rito, in più di max_df testi). Una "
                 "somiglianza non dimostra contaminazione",
    "non_verificabile": "immagine assente o illeggibile, o sha256 dei pixel non calcolabile",
    "nessuna_corrispondenza": "nessuna corrispondenza nei controlli eseguiti; non certifica la risposta né l'assenza "
                              "di ogni contaminazione",
    "gruppo": "lo stato vale per il gruppo del manifest (stessa scena, stato della scena o coppia): confermata -> "
              "escluso, candidato o non verificabile -> quarantena; la gravità non diminuisce mai e split e gruppi "
              "non cambiano",
    "testo": "le domande più corte di n parole non si confrontano sul testo: per loro decide l'immagine",
}
NOT_COVERED_DECONTAM = (
    "indice globale fra sorgenti (S1 §4) e import delle esclusioni storiche (S1 §5): i gruppi sono quelli dei legami "
    "esatti dentro i sintetici v4 (pixel, stato della scena, coppia). Il confronto con i benchmark riservati è quello "
    "della decontaminazione esterna contro pannello-v3 (evidenza decontaminazione_pannello), non un indice globale")


class Failed(Exception):
    """Un controllo di compatibilità non superato: niente manifest."""


def rel(path: Path) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if ROOT in p.parents else str(p)


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- decisioni (pure, collaudate in tests/)

def record_outcome(row: dict) -> str:
    """Esito della decontaminazione di un record dalla sua riga di records.jsonl.gz della scansione."""
    found = row.get("corrispondenze") or {}
    if found.get("pixel"):
        return "corrispondenza_confermata"
    if found.get("image") or found.get("text"):
        return "candidato"
    checks = row.get("controlli") or {}
    if row.get("immagine") != "letta" or not checks.get("pixel") or not checks.get("phash"):
        return "non_verificabile"
    return "nessuna_corrispondenza"


def decide(records: list[dict], scan: dict[str, dict]) -> dict[str, dict]:
    """Per ogni record: esito proprio, esito e stato del gruppo, stato nuovo e membri del gruppo che lo causano.
    records: righe del manifest precedente; scan: chiave -> riga della scansione."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[r["gruppo"]].append(r)
    out = {}
    for rs in by_group.values():
        own = {r["record"]: record_outcome(scan[r["record"]]) for r in rs}
        worst = min(own.values(), key=OUTCOMES.index)
        group_state = GROUP_STATE.get(worst)
        causes = sorted(k for k, o in own.items() if o == worst) if group_state else []
        for r in rs:
            state = r["stato"]
            if group_state and SEVERITY[group_state] > SEVERITY[state]:
                state = group_state
            out[r["record"]] = {"esito": own[r["record"]], "esito_gruppo": worst, "stato_gruppo": group_state,
                                "stato": state, "cause": causes}
    return out


# ---------------------------------------------------------------- ingressi e compatibilità

def load_inputs(args, checks: list[dict]) -> dict:
    def check(name: str, ok: bool, detail=None) -> None:
        checks.append({"controllo": name, "esito": bool(ok), "dettaglio": detail})
        print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail is not None else ""), flush=True)

    # ---- indice del pannello
    index_path = args.index / "index.json"
    index_sha = sha256_file(index_path)
    loaded = json.loads(index_path.read_text(encoding="utf-8"))
    try:
        convention = index_convention(loaded)
    except ValueError as exc:
        convention = None
        check("indice: convenzione dichiarata", False, str(exc))
    check("indice: convenzione dichiarata rgb-su-bianco/1, la stessa della scansione e della revisione",
          convention == IMAGE_CONVENTION, convention)
    check("indice: schema dell'indice per immagine del pannello", loaded.get("schema") == "panel-index/2",
          loaded.get("schema"))
    if args.index_sha256:
        check("indice: sha256 uguale a quello atteso", index_sha == args.index_sha256, index_sha)
    verifica = json.loads((args.index / "verifica.json").read_text(encoding="utf-8"))
    items = loaded["items"]
    with_image = sum(1 for it in items if it.get("pos") is not None)
    check("indice: verifica.json, ogni voce ritrova la propria immagine nella sorgente",
          verifica.get("verificato") is True and verifica.get("errori") == 0
          and verifica.get("voci_con_immagine") == with_image
          and verifica.get("convenzione_immagini") == convention,
          {k: verifica.get(k) for k in ("verificato", "errori", "voci_con_immagine")})
    pixels = sum(1 for it in items if it.get("pixel_sha256"))
    check("indice: sha256 dei pixel per ogni voce con immagine leggibile",
          pixels == sum(1 for it in items if it.get("phash") and it.get("pos") is not None), pixels)

    # ---- pannello e perimetro
    panel_summary = json.loads((args.panel / "summary.json").read_text(encoding="utf-8"))
    check("pannello costruito da questo indice (sha256 registrato in summary.json)",
          (panel_summary.get("indice") or {}).get("sha256") == index_sha)
    elements = {r["chiave"]: r for r in read_jsonl_gz(args.panel / "elements.jsonl.gz")}
    index_elements = {f"{it['benchmark']}/{it['item']}" for it in items}
    check("perimetro: ogni elemento del pannello ha voci nell'indice e ogni voce un elemento",
          index_elements == set(elements), {"solo_indice": len(index_elements - set(elements)),
                                           "solo_pannello": len(set(elements) - index_elements)})
    perimeter = Counter()
    for e in elements.values():
        role = e["ruolo"]
        if role == "test":
            role = "test_ammesso" if e.get("metrica_primaria") is True else "test_non_ammesso"
        perimeter[role] += 1
    check("perimetro: tutti i ruoli presenti, compresi test non ammesso, diagnostica e solo_decontaminazione",
          all(perimeter[r] for r in ("sviluppo", "test_ammesso", "test_non_ammesso", "diagnostica",
                                     "solo_decontaminazione")), dict(perimeter))
    check("pannello: numero di elementi uguale a summary.json", len(elements) == panel_summary.get("elementi"),
          len(elements))

    # ---- manifest precedente e shard
    m = mf.load(args.manifest)
    raw = (args.manifest / m.meta["records"]["file"]).read_bytes()  # già verificato da mf.load
    records = [json.loads(line) for line in gzip.decompress(raw).decode("utf-8").splitlines()]
    check("manifest precedente letto e verificato (sha256, record_id, un solo split per gruppo)",
          len(records) == m.meta["records"]["righe"], len(records))
    keys = Counter(r["record"] for r in records)
    check("manifest: chiavi uniche fra tutti gli shard", max(keys.values()) == 1)
    shard_ok = []
    for s in m.meta["shards"]:
        p = mf.shard_path(s["percorso"])
        shard_ok.append(p.is_file() and p.stat().st_size == s["bytes"] and sha256_file(p) == s["sha256"])
    check("shard del manifest su disco con sha256 e dimensione registrati", all(shard_ok), shard_ok)

    # ---- scansione
    scan_summary = json.loads((args.scan / "summary.json").read_text(encoding="utf-8"))
    check("scansione: tipo synth (domande della scena e della coppia)", scan_summary.get("kind") == "synth",
          scan_summary.get("kind"))
    check("scansione: stessa convenzione dell'indice", scan_summary.get("convenzione_immagini") == convention)
    check("scansione: stesso indice (sha256)", (scan_summary.get("indice") or {}).get("sha256") == index_sha)
    check("scansione: identità dei pixel attiva", scan_summary.get("identita_pixel") is True)
    files = scan_summary.get("files") or {}
    ok_files = all(sha256_file(args.scan / name) == (files.get(name) or {}).get("sha256")
                   for name in ("records.jsonl.gz", "matches.jsonl"))
    check("scansione: records.jsonl.gz e matches.jsonl con gli sha256 di summary.json", ok_files)
    known = {str(mf.shard_path(s["percorso"])): s for s in m.meta["shards"]}
    scanned = [str((ROOT / p).resolve()) if not Path(p).is_absolute() else str(Path(p).resolve())
               for p in scan_summary.get("shards") or []]
    check("scansione: solo shard registrati nel manifest", scanned and all(p in known for p in scanned), scanned)
    rows = read_jsonl_gz(args.scan / "records.jsonl.gz")
    check("scansione: una riga per campione letto", len(rows) == scan_summary.get("samples"), len(rows))
    return {"index": loaded, "index_path": index_path, "index_sha": index_sha, "verifica": verifica,
            "convention": convention, "panel_summary": panel_summary, "elements": elements, "perimeter": perimeter,
            "manifest": m, "records": records, "scan_summary": scan_summary, "rows": rows, "known": known,
            "scanned": scanned, "check": check}


def match_scan(inp: dict, partial: bool, check) -> dict[str, dict]:
    """Ogni riga della scansione al suo record del manifest (chiave, shard, posizione, sha256 del PNG) e ogni
    corrispondenza al suo elemento del pannello. Senza partial ogni record va scansionato una volta."""
    by_key = {r["record"]: r for r in inp["records"]}
    scan, problems = {}, Counter()
    examples = []
    resolved: dict[str, str] = {}

    def where(path: str) -> str:  # percorso assoluto risolto, una volta per shard (GlusterFS)
        if path not in resolved:
            resolved[path] = str(mf.shard_path(path))
        return resolved[path]

    for row in inp["rows"]:
        key = row["key"]
        r = by_key.get(key)
        why = None
        if key in scan:
            why = "riga ripetuta"
        elif r is None:
            why = "chiave assente dal manifest"
        else:
            if where(row["shard"]) != where(r["shard"]):
                why = "shard diverso"
            elif row["posizione"] != r["indice"]:
                why = "posizione diversa"
            elif row.get("byte_sha256") != r["immagini"][0]["sha256"]:
                why = "byte del PNG diversi da quelli del manifest"
        if why:
            problems[why] += 1
            if len(examples) < 5:
                examples.append(f"{key}: {why}")
        scan.setdefault(key, row)
    check("identificativi: ogni riga della scansione ritrova un solo record (chiave, shard, posizione, sha256 del PNG)",
          not problems, {"problemi": dict(problems), "esempi": examples})
    scanned = set(inp["scanned"])
    expected = {r["record"] for r in inp["records"] if where(r["shard"]) in scanned} if partial else set(by_key)
    missing = expected - set(scan)
    check("copertura: ogni record degli shard scansionati ha la sua riga" if partial
          else "copertura: ogni record del manifest scansionato una volta", not missing, len(missing))
    unknown, details_cut = Counter(), 0
    for row in scan.values():
        details_cut += not row.get("dettagli_completi", True)
        for d in row.get("dettagli") or []:
            if d.get("elemento") not in inp["elements"]:
                unknown[d.get("benchmark")] += 1
    check("corrispondenze: ogni elemento citato esiste nel pannello", not unknown, dict(unknown))
    check("corrispondenze: dettagli completi per ogni record (nessun elenco tagliato da --max-lines)",
          details_cut == 0, details_cut)
    return scan


def evidence_of(row: dict, decision: dict, elements: dict, ref: dict, job) -> dict:
    found = []
    for d in (row.get("dettagli") or [])[:MAX_EVIDENCE]:
        e = elements.get(d.get("elemento")) or {}
        found.append({"tipo": d["match"], "elemento": d.get("elemento"), "posizione": d.get("pos"),
                      "distanza": d.get("distance"), "ruolo": e.get("ruolo"), "gruppo_pannello": e.get("gruppo"),
                      "metrica_primaria": e.get("metrica_primaria")})
    return {"controllo": CHECK, "versione": SCHEMA, "job": job, "esito": decision["esito"],
            "esito_gruppo": decision["esito_gruppo"], "controlli": row.get("controlli"),
            "corrispondenze": row.get("corrispondenze") or {}, "elementi": found, "rif": ref}


def summarize(inp: dict, scan: dict, decisions: dict) -> dict:
    """Conteggi per insieme, split, esito e stato, e corrispondenze per benchmark e ruolo del pannello."""
    out = {"esiti_record": Counter(), "esiti_gruppo": Counter(), "cambi_di_stato": Counter(),
           "per_insieme_split_esito": Counter(), "corrispondenze_per_benchmark_ruolo": Counter(),
           "record_con_corrispondenze_per_benchmark": Counter(), "testi": Counter()}
    groups = {}
    for r in inp["records"]:
        d = decisions.get(r["record"])
        if d is None:
            continue
        out["esiti_record"][d["esito"]] += 1
        groups[r["gruppo"]] = d["esito_gruppo"]
        out["per_insieme_split_esito"][f"{r['insieme']}/{r['split']}/{d['esito']}"] += 1
        if d["stato"] != r["stato"]:
            out["cambi_di_stato"][f"{r['insieme']}/{r['split']}: {r['stato']} -> {d['stato']}"] += 1
        row = scan[r["record"]]
        tc = (row.get("controlli") or {}).get("testo") or {}
        out["testi"]["domande"] += tc.get("testi", 0)
        out["testi"]["confrontabili"] += tc.get("confrontabili", 0)
        out["testi"]["record_senza_domande_confrontabili"] += tc.get("confrontabili", 0) == 0
        benches = set()
        for det in row.get("dettagli") or []:
            e = inp["elements"].get(det.get("elemento")) or {}
            role = e.get("ruolo")
            if role == "test":
                role = "test_ammesso" if e.get("metrica_primaria") is True else "test_non_ammesso"
            out["corrispondenze_per_benchmark_ruolo"][f"{det['benchmark']}/{role}/{det['match']}"] += 1
            benches.add(det["benchmark"])
        for b in benches:
            out["record_con_corrispondenze_per_benchmark"][b] += 1
    out["esiti_gruppo"] = Counter(groups.values())
    return {k: dict(sorted(v.items())) for k, v in out.items()}


# ---------------------------------------------------------------- comandi

def cmd_verifica(args) -> int:
    checks: list[dict] = []
    t0 = time.time()
    inp = load_inputs(args, checks)
    scan = match_scan(inp, args.partial, inp["check"])
    by_key = {r["record"]: r for r in inp["records"]}
    scanned = [by_key[k] for k in scan if k in by_key]
    decisions = decide(scanned, scan) if all(k in scan for k in (r["record"] for r in scanned)) else {}
    report = {"schema": SCHEMA + "-verifica", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
              "args": {k: str(v) for k, v in vars(args).items()}, "parziale": args.partial, "controlli": checks,
              "falliti": [c["controllo"] for c in checks if not c["esito"]],
              "perimetro_pannello": dict(inp["perimeter"]), "scansione": {k: inp["scan_summary"].get(k) for k in (
                  "samples", "lettura_immagini", "testi", "excluded", "excluded_by", "matches_per_benchmark",
                  "seconds", "shards")},
              "esiti_sul_sottoinsieme": summarize(inp, scan, decisions) if decisions else None,
              "nota": "con --partial gli esiti per gruppo valgono solo fra i record scansionati",
              "seconds": round(time.time() - t0, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    write_new_json(args.out / "report.json", report)
    failed = report["falliti"]
    print(f"{len(checks) - len(failed)} controlli superati su {len(checks)}")
    if decisions:
        print(json.dumps(report["esiti_sul_sottoinsieme"], ensure_ascii=False)[:4000])
    return 1 if failed else 0


def cmd_manifest(args) -> int:
    checks: list[dict] = []
    t0 = time.time()
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: una versione del manifest si scrive una volta sola")
    inp = load_inputs(args, checks)
    scan = match_scan(inp, False, inp["check"])
    failed = [c["controllo"] for c in checks if not c["esito"]]
    if failed:
        sys.exit("controlli di compatibilità falliti, manifest non scritto: " + "; ".join(failed))
    m, records = inp["manifest"], inp["records"]
    decisions = decide(records, scan)
    scan_ref = {"file": rel(args.scan / "records.jsonl.gz"),
                "sha256": inp["scan_summary"]["files"]["records.jsonl.gz"]["sha256"]}
    job = inp["scan_summary"].get("job_id")
    lines, new = [], []
    for r in records:
        d = decisions[r["record"]]
        rec = dict(r)
        rec["evidenze"] = list(r["evidenze"]) + [evidence_of(scan[r["record"]], d, inp["elements"], scan_ref, job)]
        if d["stato_gruppo"]:
            motivo = dict(r["motivo"] or {})
            motivo[CHECK] = [f"{REASON[d['esito_gruppo']]}: {', '.join(d['cause'][:5])}"
                             + (f" e altri {len(d['cause']) - 5}" if len(d["cause"]) > 5 else "")]
            rec["motivo"] = motivo
        rec["stato"] = d["stato"]
        new.append(rec)
        lines.append(json.dumps(rec, sort_keys=True, ensure_ascii=False))
    # ---- coerenza prima di scrivere
    worse = [x["record"] for x, r in zip(new, records) if SEVERITY[x["stato"]] < SEVERITY[r["stato"]]]
    moved = [x["record"] for x, r in zip(new, records)
             if (x["split"], x["gruppo"], x["record_id"]) != (r["split"], r["gruppo"], r["record_id"])]
    unexplained = [x["record"] for x, r in zip(new, records) if x["stato"] != r["stato"] and CHECK not in x["motivo"]]
    if worse or moved or unexplained:
        sys.exit(f"incoerenze: stato migliorato {worse[:3]}, split o gruppo cambiati {moved[:3]}, cambi senza motivo "
                 f"{unexplained[:3]}")

    args.out.mkdir(parents=True, exist_ok=True)
    data = gzip_lines(lines)
    data_path = args.out / "records.jsonl.gz"
    refuse_existing(data_path)
    data_path.write_bytes(data)
    keys = {"train": [x["record"] for x in new if x["split"] == "train" and x["stato"] == "ammesso"],
            "dev": [x["record"] for x in new if x["split"] == "dev" and x["stato"] == "ammesso"],
            "test": [x["record"] for x in new if x["insieme"] == "test" and x["stato"] == "ammesso"],
            "coppie": [x["record"] for x in new if x["insieme"] == "coppie" and x["stato"] == "ammesso"],
            "non_ammessi": [x["record"] for x in new if x["stato"] != "ammesso"]}
    (args.out / "keys").mkdir()
    derived = {}
    for name, ks in keys.items():
        path = args.out / "keys" / f"{name}.txt"
        text = "".join(k + "\n" for k in sorted(ks))
        path.write_text(text, encoding="utf-8")
        derived[f"keys/{name}.txt"] = {**file_entry(path, mf.sha256_text(text)), "righe": len(ks),
                                       "contenuto": "chiavi ammesse" if name != "non_ammessi" else "quarantena ed esclusi"}
    counts = {"per_insieme_split_stato": Counter(), "per_split_famiglia_stato": Counter(),
              "qa_per_split_famiglia_tipo": Counter(), "gruppi_per_split": Counter()}
    split_of = {}
    for x in new:
        counts["per_insieme_split_stato"][f"{x['insieme']}/{x['split']}/{x['stato']}"] += 1
        counts["per_split_famiglia_stato"][f"{x['split']}/{x['famiglia']}/{x['stato']}"] += 1
        split_of[x["gruppo"]] = x["split"]
        if x["stato"] == "ammesso":
            for q in x["qa"]:
                counts["qa_per_split_famiglia_tipo"][f"{x['split']}/{x['famiglia']}/{q['tipo']}"] += 1
    for s in split_of.values():
        counts["gruppi_per_split"][s] += 1
    dev_now = Counter(x["famiglia"] for x in new if x["split"] == "dev" and x["stato"] == "ammesso")
    meta0 = m.meta
    stats = summarize(inp, scan, decisions)
    meta = {**meta0,
            "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
            "precedente": {"percorso": rel(args.manifest), "versione": meta0["versione"],
                           "manifest_sha256": m.manifest_sha256, "records_sha256": meta0["records"]["sha256"]},
            "split": {**meta0["split"],
                      "ereditato_da": {"percorso": rel(args.manifest), "versione": meta0["versione"],
                                       "manifest_sha256": m.manifest_sha256},
                      "regola_versione": "split e gruppi della versione precedente, invariati: la decontaminazione "
                                         "cambia solo stati, motivi ed evidenze",
                      "dev_ammessi_per_famiglia": dict(sorted(dev_now.items()))},
            "stati": {"ammesso": "scena candidato del controllo (S2 §2.6), senza corrispondenze col pannello nei "
                                 "controlli eseguiti: ammessa nel manifest, non certificata",
                      "quarantena": "testi sovrapposti (S2 §2.6), oppure gruppo con un candidato di decontaminazione "
                                    "o un controllo incompleto: da rivedere",
                      "escluso": "contaminazione con la valutazione dei sintetici (S2 §2.6), oppure gruppo con pixel "
                                 "identici a un'immagine del pannello"},
            "decontaminazione": {
                "controllo": CHECK, "versione": SCHEMA, "regole": RULES,
                "indice": {**file_entry(inp["index_path"], inp["index_sha"]), "schema": inp["index"].get("schema"),
                           "convenzione_immagini": inp["convention"], "voci": len(inp["index"]["items"]),
                           "verifica": file_entry(args.index / "verifica.json")},
                "pannello": {"elements": file_entry(args.panel / "elements.jsonl.gz"),
                             "summary": file_entry(args.panel / "summary.json"),
                             "job": inp["panel_summary"].get("job_id"), "codice": inp["panel_summary"].get("code"),
                             "perimetro": dict(sorted(inp["perimeter"].items()))},
                "scansione": {"summary": file_entry(args.scan / "summary.json"),
                              "records": file_entry(args.scan / "records.jsonl.gz"),
                              "matches": file_entry(args.scan / "matches.jsonl"),
                              "job": job, "codice": inp["scan_summary"].get("code"),
                              **{k: inp["scan_summary"].get(k) for k in ("radius", "ngram", "max_df", "max_lines",
                                                                         "samples", "lettura_immagini", "testi")}},
                "controlli_di_compatibilita": checks, **stats},
            "conteggi": {k: dict(sorted(v.items())) for k, v in counts.items()},
            "records": {"file": "records.jsonl.gz", "sha256": hashlib.sha256(data).hexdigest(), "righe": len(lines)},
            "derivati": derived,
            "fuori_dal_sottoinsieme": [NOT_COVERED_DECONTAM if x.startswith("indice globale fra sorgenti") else x
                                       for x in meta0["fuori_dal_sottoinsieme"]],
            "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "manifest.json", meta)
    loaded = mf.load(args.out)  # rilettura con le verifiche dei loader
    print(json.dumps({"record": len(lines), "stati": meta["conteggi"]["per_insieme_split_stato"],
                      "esiti": stats["esiti_record"], "cambi_di_stato": stats["cambi_di_stato"],
                      "riletto": len(loaded.by_shard), "seconds": meta["seconds"]}, ensure_ascii=False))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("verifica", "manifest"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--index", required=True, type=Path, help="cartella con index.json e verifica.json")
        p.add_argument("--panel", required=True, type=Path, help="cartella con elements.jsonl.gz e summary.json")
        p.add_argument("--scan", required=True, type=Path, help="uscita di decontam.py scan --kind synth")
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--index-sha256", default=None)
        if name == "verifica":
            p.add_argument("--partial", action="store_true", help="scansione limitata ad alcuni shard")
        else:
            p.add_argument("--version", required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("synth_decontam.py va eseguito in un job SLURM")
    sys.exit(cmd_verifica(args) if args.cmd == "verifica" else cmd_manifest(args))


if __name__ == "__main__":
    main()
