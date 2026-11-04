#!/usr/bin/env python3
"""Pacchetto portatile per la revisione umana offline dei fogli di campione-2 (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md
§§2.8, 2.11-2.12). Solo in un job SLURM CPU (slurm/review_package.sbatch). Nessun dato rigenerato: campione, ranghi,
identificativi, prompt, risposte attese e immagini vengono dai file di campione-2, verificati con gli sha256 del suo
report.json; specifica e layout dei record selezionati vengono dagli shard, con lo sha256 del JSON registrato nel
manifest; gli esiti del checker dal file del candidato registrato nel manifest.

`build` aggiunge alla cartella del pacchetto, che contiene già l'interfaccia scritta a mano (index.html, app.js,
logica.js, stile.css, verifica.html, test.js, LEGGIMI.md, risultati/LEGGIMI.md):
- dati/pacchetto.js (window.PACCHETTO): le unità come si vedono al primo passaggio, senza target, opzioni ausiliarie,
  esiti del checker, motivi o misure;
- dati/soluzioni.js (window.SOLUZIONI): per unità le risposte attese, le opzioni (dal JSON del record), l'esito del
  checker, lo stato della scena, motivo e misura dei casi mirati;
- dati/evidenze.js (window.EVIDENZE): per record il JSON dello shard (specifica, QA, layout) e la riga del checker;
- immagini/: i PNG v4 e le immagini del pannello, copiati byte per byte e controllati;
- moduli/: i file originali di campione-2, byte per byte;
- pacchetto.json (versione, impronta, sha256 degli ingressi, conteggi) e SHA256SUMS (per `shasum -a 256 -c`).
I file .js si caricano con <script>, quindi l'interfaccia si apre da file:// senza server.

`verifica` rilegge il pacchetto e lo confronta con campione-2: conteggi (380 unità casuali, 175 mirate, 264 candidati,
755 corrispondenze), identificativi, prompt esatti, risposte e opzioni, immagini, moduli, SHA256SUMS, separazione del
primo passaggio, collegamenti fra unità per record e per scena, riferimenti dell'interfaccia e assenza di richieste di
rete. Scrive --report ed esce con errore se un controllo fallisce. Non prova l'interfaccia in un browser.

`esegui-js` esegue il JavaScript del pacchetto (dati, logica.js, app.js, test.js) con Duktape, la libreria di sistema
di lagrange0 caricata con ctypes, senza DOM e su un DOM simulato (scripts/review_package_dom.js); confronta i
collegamenti di logica.js con quelli calcolati qui e cerca soluzioni nell'output dell'autotest. Non è una prova in un
browser.

    srun $RUN_PYTHON -u -B scripts/review_package.py build --campione DIR --manifest DIR --out DIR
    srun $RUN_PYTHON -u -B scripts/review_package.py verifica --campione DIR --pacchetto DIR --manifest DIR --report FILE
    srun $RUN_PYTHON -u -B scripts/review_package.py esegui-js --pacchetto DIR --dom FILE --report FILE
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, sha256_file, sha256_text  # noqa: E402

SCHEMA = "synth-v4-revisione-pacchetto/1"
PACKAGE_ID = "synth-v4-campione-2"
PACKAGE_VERSION = "1"
CAMPIONE_FILES = ("campione.tsv", "mirati.tsv", "prompts.jsonl", "opzioni.tsv", "soluzioni.tsv", "esiti.tsv",
                  "inventario.tsv", "candidati_decontaminazione.tsv", "corrispondenze_decontaminazione.tsv",
                  "esiti_decontaminazione.tsv", "confronto.json")
INTERFACE = ("index.html", "app.js", "logica.js", "stile.css", "verifica.html", "test.js", "LEGGIMI.md",
             "risultati/LEGGIMI.md")
DATA_FILES = ("dati/pacchetto.js", "dati/soluzioni.js", "dati/evidenze.js")
EXPECTED = {"campione": 380, "mirati": 175, "decontaminazione": 264, "corrispondenze": 755, "righe_esiti": 655}
# chiavi che non devono comparire nei dati del primo passaggio (dati/pacchetto.js)
HIDDEN_KEYS = {"risposta", "answer", "opzioni", "options", "esito_checker", "motivo", "misura", "stato_scena_sha256",
               "questions"}
CONSEGNE = {
    "loader": "Rispondi alla domanda numero {n} del prompt; le altre domande sono il contesto che il modello vede "
              "insieme a lei.",
    "loader_uno": "Rispondi alla domanda del prompt.",
    "domanda_singola_qa_non_scelta": "Rispondi alla domanda. Il loader non sceglie questa QA per la scena: il modello "
                                     "non la vede mai in training, e qui è mostrata da sola.",
    "domanda_singola_test": "Rispondi alla domanda (test: ogni QA da sola, protocollo del foglio).",
    "domanda_singola_coppia": "Rispondi alla domanda separatamente per la variante a e per la variante b, guardando "
                              "ciascuna immagine.",
    "loader_scena": "Caso sulla scena intera: giudica se l'immagine è leggibile e univoca. Il prompt mostra le domande "
                    "che il loader costruisce per questa scena; nel campo risposta puoi rispondere o descrivere ciò "
                    "che vedi (facoltativo).",
    "nessuno": "Confronta le due immagini: sono la stessa istanza (la stessa scena)? Scrivi sì o no nel campo risposta "
               "e giudica la leggibilità.",
}


class Failed(Exception):
    pass


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def js_text(var: str, obj) -> str:
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return f"window.{var} = " + text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + ";\n"


def read_js(path: Path, var: str):
    text = path.read_text(encoding="utf-8")
    head = f"window.{var} = "
    if not text.startswith(head) or not text.endswith(";\n"):
        raise Failed(f"{path}: non è un file window.{var} = ...;")
    return json.loads(text[len(head):-2])


def canonical_digest(obj) -> str:
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def number(text: str):
    if text in ("", "None"):
        return None
    try:
        return int(text)
    except ValueError:
        return float(text)


def boolish(text: str):
    return {"True": True, "False": False, "None": None, "": None, "sì": True, "no": False}.get(text, text)


# ---------------------------------------------------------------- lettura di campione-2

def load_campione(src: Path) -> dict:
    report = json.loads((src / "report.json").read_text(encoding="utf-8"))
    bad = [name for name, entry in report["files"].items()
           if sha256_file(src / name) != entry["sha256"] or (src / name).stat().st_size != entry["bytes"]]
    if bad:
        raise Failed(f"campione-2: file diversi da quelli registrati in report.json: {bad}")
    missing = [name for name in CAMPIONE_FILES if name not in report["files"]]
    if missing:
        raise Failed(f"campione-2: file non registrati in report.json: {missing}")
    tables = {name: read_tsv(src / name) for name in CAMPIONE_FILES if name.endswith(".tsv")}
    prompts = [json.loads(line) for line in (src / "prompts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return {"report": report, "report_entry": file_entry(src / "report.json"), "tables": tables,
            "prompts": {p["unita"]: p for p in prompts}, "n_prompts": len(prompts)}


# ---------------------------------------------------------------- costruzione

def build(args) -> None:
    out, src = args.out, args.campione
    missing = [name for name in INTERFACE if not (out / name).is_file()]
    if missing:
        sys.exit(f"{out}: mancano i file dell'interfaccia {missing}")
    present = [p for p in ("dati", "immagini", "moduli", "pacchetto.json", "SHA256SUMS") if (out / p).exists()]
    if present:
        sys.exit(f"{out}: già presenti {present}; il pacchetto si scrive una volta sola")
    t0 = time.time()
    c = load_campione(src)
    tab, prompts, report = c["tables"], c["prompts"], c["report"]
    m = mf.load(args.manifest)
    if m.manifest_sha256 != report["manifest"]["manifest_sha256"]:
        raise Failed("il manifest non è quello usato da campione-2")
    by_key = {k: r for shard in m.by_shard.values() for k, r in shard.items()}

    # ---- record selezionati: JSON dagli shard e righe del checker
    unit_rows = [("campione", row) for row in tab["campione.tsv"]] + [("mirati", row) for row in tab["mirati.tsv"]]
    wanted = {k for _, row in unit_rows for k in row["record"].split("|")}
    metas, raw_text = {}, {}
    for s in m.meta["shards"]:
        for index, (key, files) in enumerate(iter_shard(mf.shard_path(s["percorso"]))):
            if key not in wanted:
                continue
            r = by_key[key]
            if r["shard"] != s["percorso"] or r["indice"] != index:
                raise Failed(f"{key}: posizione diversa da quella del manifest")
            digest = hashlib.sha256(files["json"]).hexdigest()
            if digest != r["json_sha256"] or hashlib.sha256(files["png"]).hexdigest() != r["immagini"][0]["sha256"]:
                raise Failed(f"{key}: JSON o PNG diversi da quelli del manifest")
            metas[key], raw_text[key] = json.loads(files["json"]), files["json"].decode("utf-8")
    if set(metas) != wanted:
        raise Failed(f"record non trovati negli shard: {sorted(wanted - set(metas))[:5]}")
    cand = m.meta["ingressi"]["candidato_records"]
    if sha256_file(cand["path"]) != cand["sha256"]:
        raise Failed(f"{cand['path']}: sha256 diverso da quello del manifest")
    checker = {}
    with open(cand["path"], encoding="utf-8") as fh:
        for line in fh:
            if line.startswith('{"key": "') and line[9:line.index('"', 9)] in wanted:
                row = json.loads(line)
                checker[row["key"]] = row

    # ---- unità del primo passaggio e soluzioni
    sol_by_unit, opt_by_unit = defaultdict(list), defaultdict(list)
    for row in tab["soluzioni.tsv"]:
        sol_by_unit[row["unita"]].append(row)
    for row in tab["opzioni.tsv"]:
        opt_by_unit[row["unita"]].append(row)
    rows_by_unit = defaultdict(list)
    for row in tab["esiti.tsv"]:
        rows_by_unit[row["unita"]].append(row)
    sections = {"campione": [], "mirati": []}
    solutions, images = {}, {}
    for section, row in unit_rows:
        uid = row["unita"]
        keys = row["record"].split("|")
        recs = [by_key[k] for k in keys]
        if row["record_id"].split("|") != [r["record_id"] for r in recs]:
            raise Failed(f"{uid}: record_id diversi dal manifest")
        pair = len(recs) == 2 and all(r["insieme"] == "coppie" for r in recs)
        labels = []
        for r in recs:
            v = (metas[r["record"]].get("pair") or {}).get("variant") if pair else ""
            if pair and (v not in ("a", "b") or not r["record"].endswith(f"-{v}")):
                raise Failed(f"{uid}: variante {v!r} incoerente con {r['record']}")
            labels.append(v)
        if pair and labels != ["a", "b"]:
            raise Failed(f"{uid}: varianti {labels}")
        p = prompts[uid]
        if p["prompt"] and sha256_text(p["prompt"]) != p["prompt_sha256"]:
            raise Failed(f"{uid}: prompt con sha256 diverso")
        if row["prompt"] != p["prompt"].replace("\n", "\\n") or row["origine_prompt"] != p["origine_prompt"]:
            raise Failed(f"{uid}: prompt del TSV diverso da prompts.jsonl")
        pos = None if row["posizione_qa"] == "" else int(row["posizione_qa"])
        origin = p["origine_prompt"]
        if origin == "loader":
            consegna = (CONSEGNE["loader"].format(n=p["numero_nel_prompt"]) if len(p["domande_nel_prompt"]) > 1
                        else CONSEGNE["loader_uno"])
        else:
            consegna = CONSEGNE[origin]
        shown = []
        for r, v in zip(recs, labels):
            if pair:
                label = f"variante {v}"
            elif len(recs) > 1:
                label = "escluso (train)" if r["insieme"] == "train" else f"valutazione ({r['insieme']})"
            else:
                label = ""
            shown.append({"etichetta": label, "variante": v, "record": r["record"], "record_id": r["record_id"],
                          "insieme": r["insieme"], "split": r["split"], "stato": r["stato"], "famiglia": r["famiglia"],
                          "shard": r["shard"], "indice": r["indice"], "immagine": f"immagini/{r['record']}.png",
                          "immagine_sha256": r["immagini"][0]["sha256"]})
            images[f"immagini/{r['record']}.png"] = r["immagini"][0]["sha256"]
        variants = ["a", "b"] if pair else [""]
        record_of = ({v: r["record"] for r, v in zip(recs, labels)} if pair else {"": "|".join(keys)})
        expected_rows = ([("a", record_of["a"]), ("b", record_of["b"]), ("coppia", "|".join(keys))] if pair
                         else [("", "|".join(keys))])
        if [(x["variante"], x["record"]) for x in rows_by_unit[uid]] != expected_rows:
            raise Failed(f"{uid}: righe di esiti.tsv diverse dalle varianti attese")
        unit = {"id": uid, "sezione": section, "gruppo": row["strato"] if section == "campione" else row["categoria"],
                "tipo": row["tipo"], "famiglia": row["famiglia"], "livello": "qa" if pos is not None else "scena",
                "coppia": pair, "varianti": variants, "record_per_variante": record_of, "immagini": shown,
                "posizione_qa": pos, "domanda": row["domanda"], "prompt": p["prompt"],
                "prompt_sha256": p["prompt_sha256"], "origine_prompt": origin,
                "numero_nel_prompt": p["numero_nel_prompt"], "domande_nel_prompt": p["domande_nel_prompt"],
                "consegnato_dal_loader": p["consegnato_dal_loader"], "consegna": consegna}
        if section == "campione":
            unit.update(rango=row["rango"], N=int(row["N"]), n=int(row["n"]), p_inclusione=row["p_inclusione"],
                        ordine_nel_loader=int(row["ordine_nel_loader"]), qa_sha256=row["qa_sha256"])
        sections[section].append(unit)
        # soluzioni: risposte dal foglio, opzioni dal JSON del record (lista esatta), controllate contro opzioni.tsv
        sols = []
        opts = {(o["variante"], o["record"]): o for o in opt_by_unit[uid]}
        for s in sol_by_unit[uid]:
            meta = metas[s["record"]]
            items = list(meta["qa"]) + ([meta["pair"]] if "pair" in meta else [])
            q = items[int(s["posizione_qa"])]
            o = opts.get((s["variante"], s["record"]))
            if q["answer"] != s["risposta"] or o is None or " / ".join(q["options"]) != o["opzioni"]:
                raise Failed(f"{uid}: risposta o opzioni del JSON diverse da soluzioni.tsv e opzioni.tsv")
            sols.append({"variante": s["variante"], "record": s["record"], "record_id": s["record_id"],
                         "posizione_qa": int(s["posizione_qa"]), "tipo": s["tipo"], "risposta": s["risposta"],
                         "esito_checker": s["esito_checker"], "stato": s["stato"],
                         "stato_scena_sha256": s["stato_scena_sha256"], "opzioni": list(q["options"]),
                         "opzioni_nel_testo_della_domanda": o["opzioni_nel_testo_della_domanda"] == "sì"})
        solutions[uid] = {"varianti": sols, "motivo": row.get("motivo"), "misura": row.get("misura")}
    if len(c["prompts"]) != len(sections["campione"]) + len(sections["mirati"]):
        raise Failed("prompts.jsonl ha unità che non sono nei fogli")

    # ---- decontaminazione: tutte le corrispondenze
    matches = defaultdict(list)
    for row in tab["corrispondenze_decontaminazione.tsv"]:
        matches[row["unita"]].append(row)
    decontam = []
    for row in tab["candidati_decontaminazione.tsv"]:
        uid, r = row["unita"], by_key[row["record"]]
        ms = matches[uid]
        if [int(x["n"]) for x in ms] != list(range(1, len(ms) + 1)) or len(ms) != int(row["corrispondenze_esportate"]):
            raise Failed(f"{uid}: corrispondenze non complete o fuori ordine")
        listed = []
        for x in ms:
            path = x["immagine_pannello"]
            if path:
                images[path] = x["immagine_pannello_sha256"]
            listed.append({"id": x["corrispondenza"], "n": int(x["n"]), "di": int(x["di"]), "tipo": x["tipo"],
                           "distanza": number(x["distanza"]), "benchmark": x["benchmark"], "elemento": x["elemento"],
                           "posizione_immagine": number(x["posizione_immagine"]), "voce_indice": x["voce_indice"],
                           "ruolo": x["ruolo"], "gruppo_pannello": x["gruppo_pannello"],
                           "metrica_primaria": boolish(x["metrica_primaria"]),
                           "pixel_sha256_pannello": x["pixel_sha256_pannello"],
                           "nell_anteprima_del_manifest": x["nell_anteprima_del_manifest"] == "sì",
                           "immagine": path, "immagine_sha256": x["immagine_pannello_sha256"],
                           "problemi": x["problemi"]})
        images[f"immagini/{r['record']}.png"] = r["immagini"][0]["sha256"]
        decontam.append({"id": uid, "rango": row["rango"], "record": row["record"], "record_id": row["record_id"],
                         "insieme": row["insieme"], "split": row["split"], "stato": row["stato"],
                         "famiglia": row["famiglia"], "esito": row["esito"], "esito_gruppo": row["esito_gruppo"],
                         "corrispondenze": json.loads(row["corrispondenze"]),
                         "corrispondenze_nella_scansione": number(row["corrispondenze_nella_scansione"]),
                         "materiale": row["materiale"], "ruoli": row["ruoli"],
                         "somiglianza_pixel": number(row["somiglianza_pixel"]),
                         "rapporto_aspetto": number(row["rapporto_aspetto"]),
                         "elemento_piu_simile": row["elemento_piu_simile"],
                         "immagine": f"immagini/{r['record']}.png", "immagine_sha256": r["immagini"][0]["sha256"],
                         "elenco": listed})

    # ---- immagini: copie byte per byte, controllate
    src_of = {p: src / p for p in images}
    bad = [p for p, sha in images.items() if not src_of[p].is_file() or sha256_file(src_of[p]) != sha]
    if bad:
        raise Failed(f"immagini di campione-2 assenti o diverse: {bad[:5]}")
    # il testo esatto del JSON dello shard (l'interfaccia lo legge con JSON.parse), per verificarne lo sha256
    evidence = {k: {"json": raw_text[k], "json_sha256": by_key[k]["json_sha256"], "checker": checker.get(k)}
                for k in sorted(metas)}
    righe = [{"unita": x["unita"], "variante": x["variante"], "record": x["record"]} for x in tab["esiti.tsv"]]
    righe_d = [{"unita": x["unita"], "record": x["record"]} for x in tab["esiti_decontaminazione.tsv"]]
    pkg = {"schema": SCHEMA, "id": PACKAGE_ID, "versione": PACKAGE_VERSION,
           "sorgente": {"campione": "outputs/synth_gold_review/campione-2", "report_sha256": c["report_entry"]["sha256"],
                        "job_campione": report.get("job_id"), "audit": report["audit"], "seme": report["seme"],
                        "manifest": report["manifest"], "codice_campione": report["code"]},
           "campione": sections["campione"], "mirati": sections["mirati"], "decontaminazione": decontam,
           "righe_esiti": righe, "righe_esiti_decontaminazione": righe_d,
           "strati": report["campione"]["strati"], "categorie_mirati": report["mirati"]["categorie"]}
    digests = {"soluzioni": canonical_digest(solutions), "evidenze": canonical_digest(evidence)}
    pkg["impronta"] = canonical_digest({"pacchetto": pkg, **digests})
    pkg["impronte_dati"] = digests

    # ---- scrittura
    (out / "dati").mkdir()
    (out / "moduli").mkdir()
    (out / "immagini" / "pannello").mkdir(parents=True)
    for name, var, obj in (("dati/pacchetto.js", "PACCHETTO", pkg), ("dati/soluzioni.js", "SOLUZIONI", solutions),
                           ("dati/evidenze.js", "EVIDENZE", evidence)):
        (out / name).write_text(js_text(var, obj), encoding="utf-8")
    for p in sorted(images):
        shutil.copyfile(src_of[p], out / p)
        if sha256_file(out / p) != images[p]:
            raise Failed(f"{p}: copia diversa dall'originale")
    for name in CAMPIONE_FILES + ("report.json",):
        shutil.copyfile(src / name, out / "moduli" / name)
    generated = sorted([*DATA_FILES, *images, *[f"moduli/{n}" for n in CAMPIONE_FILES + ("report.json",)]])
    sums = {p: sha256_file(out / p) for p in generated}
    meta = {"schema": SCHEMA, "id": PACKAGE_ID, "versione": PACKAGE_VERSION, "impronta": pkg["impronta"],
            "impronte_dati": digests, "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
            "ingressi": {"campione": {"cartella": str(src), "report": c["report_entry"],
                                      "file": report["files"]},
                         "manifest": {"cartella": str(args.manifest), "manifest_sha256": m.manifest_sha256,
                                      "records_sha256": m.meta["records"]["sha256"]},
                         "checker": file_entry(cand["path"], cand["sha256"]),
                         "shard": [{"percorso": s["percorso"], "sha256": s["sha256"]} for s in m.meta["shards"]]},
            "conteggi": {"campione": len(sections["campione"]), "mirati": len(sections["mirati"]),
                         "decontaminazione": len(decontam), "corrispondenze": sum(len(d["elenco"]) for d in decontam),
                         "righe_esiti": len(righe), "righe_esiti_decontaminazione": len(righe_d),
                         "record_con_evidenze": len(evidence), "immagini_v4": sum(not p.startswith("immagini/pannello/")
                                                                                  for p in images),
                         "immagini_pannello": sum(p.startswith("immagini/pannello/") for p in images)},
            "file": {p: {"sha256": sums[p], "bytes": (out / p).stat().st_size} for p in generated},
            "interfaccia": list(INTERFACE), "seconds": round(time.time() - t0, 1)}
    (out / "pacchetto.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sums["pacchetto.json"] = sha256_file(out / "pacchetto.json")
    (out / "SHA256SUMS").write_text("".join(f"{sums[p]}  {p}\n" for p in sorted(sums)), encoding="utf-8")
    sizes = {p: (out / p).stat().st_size for p in DATA_FILES}
    print(json.dumps({"conteggi": meta["conteggi"], "impronta": pkg["impronta"], "byte_dati": sizes,
                      "byte_immagini": sum((out / p).stat().st_size for p in images),
                      "seconds": meta["seconds"]}, ensure_ascii=False))


# ---------------------------------------------------------------- verifica

REGEX_BEFORE = set("(,=:[!&|?{};+-*%<>~^") | {"", "return", "typeof"}


def js_lexical(text: str) -> str | None:
    """Controllo lessicale minimo di un file JavaScript: commenti e stringhe chiusi, espressioni regolari riconosciute
    dal token precedente, parentesi tonde, quadre e graffe bilanciate. Non è un parser e non esegue il codice: un errore
    di sintassi d'altro tipo (una virgola mancante) non si vede. Restituisce il primo problema o None."""
    pairs, stack, i, n, prev = {")": "(", "]": "[", "}": "{"}, [], 0, len(text), ""

    def where(k: int) -> str:
        return f"riga {text.count(chr(10), 0, k) + 1}"

    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                return f"commento non chiuso ({where(i)})"
            i = j + 2
        elif ch in "\"'`":
            j = i + 1
            while j < n and text[j] != ch:
                if text[j] == "\\":
                    j += 1
                elif text[j] == "\n" and ch != "`":
                    return f"stringa non chiusa ({where(i)})"
                j += 1
            if j >= n:
                return f"stringa non chiusa ({where(i)})"
            i, prev = j + 1, "stringa"
        elif ch == "/" and prev in REGEX_BEFORE:
            j, in_class = i + 1, False
            while j < n:
                c = text[j]
                if c == "\\":
                    j += 2
                    continue
                if c == "\n":
                    return f"espressione regolare non chiusa ({where(i)})"
                if c == "[":
                    in_class = True
                elif c == "]":
                    in_class = False
                elif c == "/" and not in_class:
                    break
                j += 1
            i = j + 1
            while i < n and text[i].isalpha():
                i += 1
            prev = "regex"
        elif ch in "([{":
            stack.append((ch, i))
            i, prev = i + 1, ch
        elif ch in ")]}":
            if not stack or stack[-1][0] != pairs[ch]:
                return f"{ch} non atteso ({where(i)})"
            stack.pop()
            i, prev = i + 1, ch
        elif ch.isalnum() or ch in "_$":
            j = i
            while j < n and (text[j].isalnum() or text[j] in "_$"):
                j += 1
            i, prev = j, text[i:j]
        else:
            i, prev = i + 1, ch
    return f"{stack[-1][0]} non chiuso ({where(stack[-1][1])})" if stack else None


STYLE = ("lw", "font", "font_family", "width", "height")  # come synth_candidate.STYLE: stile, non stato della scena


def scene_key(json_text: str) -> str:
    """Chiave dello stato della scena di un record, calcolata come in logica.js (chiaveScena): la specifica senza
    chiave, versione del generatore e stile (synth_candidate.scene_state), in forma canonica."""
    spec = json.loads(json_text)["spec"]
    return json.dumps({k: v for k, v in spec.items() if k not in ("key", "generator_version", *STYLE)},
                      sort_keys=True, ensure_ascii=False)


def links_map(units: list[dict], ev: dict):
    """(unità per record, record per scena, chiave della scena per record, unità collegate per unità): mostrare la
    soluzione di un'unità espone i suoi record, e con loro le scene uguali; le unità collegate sono quelle che mostrano
    uno di questi record o un record con la stessa scena (logica.js, contesto)."""
    rec_units: dict[str, list[str]] = defaultdict(list)
    for u in units:
        for x in u["immagini"]:
            if u["id"] not in rec_units[x["record"]]:
                rec_units[x["record"]].append(u["id"])
    key_of = {r: scene_key(ev[r]["json"]) for r in rec_units}
    by_scene: dict[str, list[str]] = defaultdict(list)
    for r, k in key_of.items():
        by_scene[k].append(r)
    linked: dict[str, set] = {u["id"]: set() for u in units}
    for u in units:
        for x in u["immagini"]:
            for r in by_scene[key_of[x["record"]]]:
                linked[u["id"]].update(w for w in rec_units[r] if w != u["id"])
    return rec_units, by_scene, key_of, linked


def unit_links(units: list[dict], ev: dict, by_key: dict) -> dict:
    """Statistiche dei collegamenti fra unità (links_map), con il controllo che le scene uguali calcolate dal JSON dei
    record coincidano con lo stato della scena registrato nel manifest."""
    rec_units, by_scene, _, linked = links_map(units, ev)
    by_state: dict[str, list[str]] = defaultdict(list)
    for r in rec_units:
        by_state[by_key[r]["stato_scena_sha256"]].append(r)
    same_partition = sorted(map(sorted, by_scene.values())) == sorted(map(sorted, by_state.values()))
    seen, sizes = set(), []
    for uid in linked:  # componenti connesse della relazione "collegate"
        if uid in seen:
            continue
        stack, comp = [uid], 0
        seen.add(uid)
        while stack:
            cur = stack.pop()
            comp += 1
            for w in linked[cur]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        sizes.append(comp)
    shared = {r: us for r, us in rec_units.items() if len(us) > 1}
    scenes = [rs for rs in by_scene.values() if len(rs) > 1]
    return {"partizione_uguale_al_manifest": same_partition,
            "record": len(rec_units), "record_con_piu_unita": len(shared),
            "unita_su_record_condivisi": len({w for us in shared.values() for w in us}),
            "scene_con_piu_record": len(scenes), "record_in_scene_condivise": sum(map(len, scenes)),
            "unita_con_collegamenti": sum(1 for v in linked.values() if v),
            "max_collegate_per_unita": max(map(len, linked.values())),
            "componenti": dict(sorted(Counter(sizes).items())),
            "esempi_record": {r: us for r, us in sorted(shared.items())[:5]},
            "esempi_scene": [{r: rec_units[r] for r in rs} for rs in scenes[:5]]}


def verify(args) -> int:
    pkg_dir, src = args.pacchetto, args.campione
    checks: list[dict] = []

    def check(name: str, ok, detail=None) -> None:
        checks.append({"controllo": name, "esito": bool(ok), "dettaglio": detail})
        print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail is not None else ""), flush=True)

    t0 = time.time()
    c = load_campione(src)
    tab, prompts, report = c["tables"], c["prompts"], c["report"]
    pkg = read_js(pkg_dir / "dati/pacchetto.js", "PACCHETTO")
    sol = read_js(pkg_dir / "dati/soluzioni.js", "SOLUZIONI")
    ev = read_js(pkg_dir / "dati/evidenze.js", "EVIDENZE")
    meta = json.loads((pkg_dir / "pacchetto.json").read_text(encoding="utf-8"))
    check("campione-2: file con gli sha256 del suo report.json", True, c["report_entry"]["sha256"][:12])
    check("pacchetto: sorgente = campione-2 (sha256 di report.json, manifest)",
          pkg["sorgente"]["report_sha256"] == c["report_entry"]["sha256"]
          and pkg["sorgente"]["manifest"] == report["manifest"])
    stated = {k: v for k, v in pkg.items() if k not in ("impronta", "impronte_dati")}
    digests = {"soluzioni": canonical_digest(sol), "evidenze": canonical_digest(ev)}
    check("impronta del pacchetto ricalcolata da pacchetto.js, soluzioni.js ed evidenze.js",
          pkg["impronta"] == canonical_digest({"pacchetto": stated, **digests}) == meta["impronta"]
          and pkg["impronte_dati"] == digests)
    counts = {"campione": len(pkg["campione"]), "mirati": len(pkg["mirati"]),
              "decontaminazione": len(pkg["decontaminazione"]),
              "corrispondenze": sum(len(d["elenco"]) for d in pkg["decontaminazione"]),
              "righe_esiti": len(pkg["righe_esiti"])}
    check("conteggi: 380 unità casuali, 175 mirate, 264 candidati, 755 corrispondenze, 655 righe del modulo",
          counts == EXPECTED, counts)
    strata = Counter(u["gruppo"] for u in pkg["campione"])
    check("campione: n per strato uguale al disegno di campione-2",
          strata == Counter({s: d["n"] for s, d in report["campione"]["strati"].items()}), dict(strata))
    check("mirati: casi per categoria uguali a campione-2",
          Counter(u["gruppo"] for u in pkg["mirati"]) == Counter(report["mirati"]["categorie"]))
    fields = ("unita", "strato", "rango", "record", "record_id", "posizione_qa", "ordine_nel_loader", "tipo", "domanda",
              "qa_sha256", "N", "n")
    same = [tuple(r[f] for f in fields) for r in tab["campione.tsv"]] == [
        (u["id"], u["gruppo"], u["rango"], "|".join(x["record"] for x in u["immagini"]),
         "|".join(x["record_id"] for x in u["immagini"]), "" if u["posizione_qa"] is None else str(u["posizione_qa"]),
         str(u["ordine_nel_loader"]), u["tipo"], u["domanda"], u["qa_sha256"], str(u["N"]), str(u["n"]))
        for u in pkg["campione"]]
    check("campione: stesse unità nello stesso ordine, con rango, record, posizione, tipo, domanda e digest", same)
    tfields = ("unita", "categoria", "record", "record_id", "posizione_qa", "tipo", "domanda")
    same = [tuple(r[f] for f in tfields) for r in tab["mirati.tsv"]] == [
        (u["id"], u["gruppo"], "|".join(x["record"] for x in u["immagini"]),
         "|".join(x["record_id"] for x in u["immagini"]), "" if u["posizione_qa"] is None else str(u["posizione_qa"]),
         u["tipo"], u["domanda"]) for u in pkg["mirati"]]
    check("mirati: stessi casi nello stesso ordine", same)
    units = pkg["campione"] + pkg["mirati"]
    check("identificativi unici", len({u["id"] for u in units} | {d["id"] for d in pkg["decontaminazione"]})
          == len(units) + len(pkg["decontaminazione"]))
    bad_prompt = [u["id"] for u in units if u["prompt"] != prompts[u["id"]]["prompt"]
                  or (u["prompt"] and sha256_text(u["prompt"]) != u["prompt_sha256"])
                  or u["numero_nel_prompt"] != prompts[u["id"]]["numero_nel_prompt"]]
    check("prompt esatti (con gli a capo) e numero della domanda uguali a prompts.jsonl", not bad_prompt,
          bad_prompt[:5])
    multi = sum(1 for u in units if "\n" in u["prompt"])
    check("prompt con più domande: a capo veri, non la sequenza \\n", multi > 0 and not any(
        "\\n" in u["prompt"] for u in units), multi)

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

    leaks = hidden(units)
    check("primo passaggio: nessun target, opzione ausiliaria, esito del checker, motivo o misura in pacchetto.js",
          not leaks, leaks[:5])
    sol_rows = defaultdict(list)
    for r in tab["soluzioni.tsv"]:
        sol_rows[r["unita"]].append((r["variante"], r["record"], r["risposta"], r["esito_checker"],
                                     r["stato_scena_sha256"]))
    got = {uid: [(s["variante"], s["record"], s["risposta"], s["esito_checker"], s["stato_scena_sha256"])
                 for s in v["varianti"]] for uid, v in sol.items() if v["varianti"]}
    check("soluzioni: risposte, esiti del checker e stati uguali a soluzioni.tsv (righe per variante)",
          got == dict(sol_rows), {"unita": len(got), "righe": sum(map(len, got.values()))})
    opt_rows = {(r["unita"], r["variante"], r["record"]): r["opzioni"] for r in tab["opzioni.tsv"]}
    got_opt = {(uid, s["variante"], s["record"]): " / ".join(s["opzioni"]) for uid, v in sol.items()
               for s in v["varianti"]}
    check("opzioni: uguali a opzioni.tsv", got_opt == opt_rows, len(got_opt))
    mot = {r["unita"]: (r["motivo"], r["misura"]) for r in tab["mirati.tsv"]}
    check("motivi e misure dei casi mirati solo in soluzioni.js, uguali a mirati.tsv",
          all((sol[u][f"motivo"], sol[u]["misura"]) == mot[u] for u in mot))
    pairs = [u for u in units if u["coppia"]]
    rows_by_unit = defaultdict(list)
    for r in pkg["righe_esiti"]:
        rows_by_unit[r["unita"]].append(r["variante"])
    check("coppie: 50 unità con varianti a e b, soluzioni per entrambe e righe a, b e coppia nel modulo",
          len(pairs) == 50 and all(u["varianti"] == ["a", "b"] and [s["variante"] for s in sol[u["id"]]["varianti"]]
                                   == ["a", "b"] and rows_by_unit[u["id"]] == ["a", "b", "coppia"] for u in pairs),
          len(pairs))
    check("modulo: righe di esiti.tsv e di esiti_decontaminazione.tsv nell'ordine originale",
          [(r["unita"], r["variante"], r["record"]) for r in tab["esiti.tsv"]]
          == [(r["unita"], r["variante"], r["record"]) for r in pkg["righe_esiti"]]
          and [(r["unita"], r["record"]) for r in tab["esiti_decontaminazione.tsv"]]
          == [(r["unita"], r["record"]) for r in pkg["righe_esiti_decontaminazione"]])
    mrows = [(r["corrispondenza"], r["unita"], r["record"], r["elemento"], r["posizione_immagine"], r["voce_indice"],
              r["immagine_pannello"]) for r in tab["corrispondenze_decontaminazione.tsv"]]
    got_m = [(x["id"], d["id"], d["record"], x["elemento"], "" if x["posizione_immagine"] is None
              else str(x["posizione_immagine"]), x["voce_indice"], x["immagine"])
             for d in pkg["decontaminazione"] for x in d["elenco"]]
    check("decontaminazione: tutte le corrispondenze con record, elemento, posizione, voce e immagine", mrows == got_m,
          len(got_m))
    check("decontaminazione: candidati nello stesso ordine e con gli stessi esiti",
          [(r["unita"], r["record"], r["esito"], r["materiale"]) for r in tab["candidati_decontaminazione.tsv"]]
          == [(d["id"], d["record"], d["esito"], d["materiale"]) for d in pkg["decontaminazione"]])
    recs = {x["record"] for u in units for x in u["immagini"]}
    m = mf.load(args.manifest)
    by_key = {k: r for shard in m.by_shard.values() for k, r in shard.items()}
    check("evidenze: testo JSON dei record selezionati con lo sha256 del manifest, e riga del checker",
          set(ev) == recs and all(hashlib.sha256(e["json"].encode("utf-8")).hexdigest() == e["json_sha256"]
                                  == by_key[k]["json_sha256"] for k, e in ev.items())
          and all(e["checker"] is not None and e["checker"]["key"] == k for k, e in ev.items()), len(ev))
    links = unit_links(units, ev, by_key)
    check("collegamenti fra unità per record e per stato della scena (specifica senza chiave, versione e stile, come "
          "in logica.js): stessa partizione dello stato registrato nel manifest", links["partizione_uguale_al_manifest"],
          links)
    refs = {x["immagine"]: x["immagine_sha256"] for u in units for x in u["immagini"]}
    refs.update({d["immagine"]: d["immagine_sha256"] for d in pkg["decontaminazione"]})
    refs.update({x["immagine"]: x["immagine_sha256"] for d in pkg["decontaminazione"] for x in d["elenco"]
                 if x["immagine"]})
    bad = [p for p, sha in refs.items() if not (pkg_dir / p).is_file() or sha256_file(pkg_dir / p) != sha
           or sha256_file(src / p) != sha]
    check("immagini: ogni immagine citata presente, con byte identici a campione-2", not bad,
          {"immagini": len(refs), "diverse": bad[:5]})
    on_disk = {str(p.relative_to(pkg_dir)) for p in (pkg_dir / "immagini").rglob("*") if p.is_file()}
    check("immagini: nessun file in più", on_disk == set(refs), len(on_disk))
    names = CAMPIONE_FILES + ("report.json",)
    check("moduli: file originali di campione-2, byte per byte", all(
        sha256_file(pkg_dir / "moduli" / n) == sha256_file(src / n) for n in names))
    lines = (pkg_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    bad = [ln for ln in lines if sha256_file(pkg_dir / ln[66:]) != ln[:64] or ln[64:66] != "  "]
    check("SHA256SUMS: ogni file generato ha lo sha256 elencato", lines and not bad, {"file": len(lines),
                                                                                       "diversi": bad[:3]})
    html = (pkg_dir / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r'<script src="([^"]+)"', html)
    check("index.html: carica dati, logica e interfaccia nell'ordine, file presenti",
          scripts == ["dati/pacchetto.js", "dati/soluzioni.js", "dati/evidenze.js", "logica.js", "app.js"]
          and all((pkg_dir / s).is_file() for s in scripts) and (pkg_dir / "stile.css").is_file(), scripts)
    vhtml = (pkg_dir / "verifica.html").read_text(encoding="utf-8")
    vscripts = re.findall(r'<script src="([^"]+)"', vhtml)
    check("verifica.html: file presenti", vscripts and all((pkg_dir / s).is_file() for s in vscripts), vscripts)
    net = []
    for name in ("index.html", "app.js", "logica.js", "stile.css", "verifica.html", "test.js"):
        text = (pkg_dir / name).read_text(encoding="utf-8")
        for pat in ("fetch(", "XMLHttpRequest", "http://", "https://", "import(", "<link rel=\"preconnect"):
            if pat in text:
                net.append(f"{name}: {pat}")
    check("interfaccia: nessuna richiesta di rete né caricamento con fetch", not net, net)
    lexical = {name: js_lexical((pkg_dir / name).read_text(encoding="utf-8")) for name in ("app.js", "logica.js",
                                                                                         "test.js")}
    check("JavaScript: commenti, stringhe, espressioni regolari e parentesi bilanciate (controllo lessicale, non "
          "esecuzione)", not any(lexical.values()), {k: v for k, v in lexical.items() if v} or len(lexical))
    failed = [x["controllo"] for x in checks if not x["esito"]]
    out = {"schema": SCHEMA + "-verifica", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
           "pacchetto": str(pkg_dir), "impronta": pkg["impronta"], "controlli": checks, "falliti": failed,
           "nota": "verifica statica dei dati e dei riferimenti; l'interfaccia non è provata in un browser",
           "seconds": round(time.time() - t0, 1)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(checks) - len(failed)} controlli superati su {len(checks)}")
    return 1 if failed else 0


# ---------------------------------------------------------------- esecuzione del JavaScript con Duktape

DUKTAPE = "/lib/x86_64-linux-gnu/libduktape.so.207"  # libreria di sistema di lagrange0 (Duktape 2.7, ES5.1)
JS_FILES = ("dati/pacchetto.js", "dati/soluzioni.js", "dati/evidenze.js", "logica.js", "app.js", "test.js")


class JSError(Exception):
    pass


class Duktape:
    """Motore JavaScript Duktape caricato con ctypes. Ogni sorgente si valuta con l'eval globale del motore (eval
    indiretto: ambito globale, come uno <script>), chiamato in modo protetto: un errore JavaScript diventa JSError con
    il messaggio e la riga, senza fermare l'interprete."""

    def __init__(self, path: str):
        import ctypes
        self.c = ctypes
        lib = ctypes.CDLL(path)
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        lib.duk_create_heap.restype, lib.duk_create_heap.argtypes = vp, [vp] * 5
        lib.duk_destroy_heap.restype, lib.duk_destroy_heap.argtypes = None, [vp]
        lib.duk_get_global_string.restype, lib.duk_get_global_string.argtypes = ctypes.c_uint, [vp, ctypes.c_char_p]
        lib.duk_push_lstring.restype, lib.duk_push_lstring.argtypes = vp, [vp, ctypes.c_char_p, sz]
        lib.duk_pcall.restype, lib.duk_pcall.argtypes = ctypes.c_int, [vp, ctypes.c_int]
        lib.duk_safe_to_lstring.restype = vp
        lib.duk_safe_to_lstring.argtypes = [vp, ctypes.c_int, ctypes.POINTER(sz)]
        lib.duk_pop.restype, lib.duk_pop.argtypes = None, [vp]
        lib.duk_get_top.restype, lib.duk_get_top.argtypes = ctypes.c_int, [vp]
        self.lib, self.ctx = lib, lib.duk_create_heap(None, None, None, None, None)
        if not self.ctx:
            raise Failed(f"{path}: heap di Duktape non creato")

    def eval(self, source: str, name: str = "codice") -> str:
        lib, ctx, c = self.lib, self.ctx, self.c
        top = lib.duk_get_top(ctx)
        lib.duk_get_global_string(ctx, b"eval")
        data = source.encode("utf-8")
        lib.duk_push_lstring(ctx, data, len(data))
        rc = lib.duk_pcall(ctx, 1)
        n = c.c_size_t()
        ptr = lib.duk_safe_to_lstring(ctx, -1, c.byref(n))
        text = c.string_at(ptr, n.value).decode("utf-8", "replace") if ptr else ""
        lib.duk_pop(ctx)
        if lib.duk_get_top(ctx) != top:
            raise Failed(f"{name}: pila di Duktape non bilanciata")
        if rc != 0:
            raise JSError(f"{name}: {text}")
        return text

    def close(self) -> None:
        self.lib.duk_destroy_heap(self.ctx)


LINKS_JS = """JSON.stringify((function () {
  var C = window.Logica.contesto(window.PACCHETTO, window.EVIDENZE), o = {};
  Object.keys(C.collegati).forEach(function (k) { o[k] = C.collegati[k].map(function (c) { return c.id; }).sort(); });
  return o;
})())"""


WORD = re.compile(r"[A-Za-z0-9_]")


def bounded(text: str, s: str) -> bool:
    """s compare in text come parola intera, o con bordi non alfanumerici (come confinato() in test.js)."""
    i = text.find(s)
    while i >= 0:
        before = text[i - 1] if i else " "
        after = text[i + len(s)] if i + len(s) < len(text) else " "
        if ((not WORD.match(s[0]) or not WORD.match(before))
                and (not WORD.match(s[-1]) or not WORD.match(after))):
            return True
        i = text.find(s, i + 1)
    return False


def secrets_of(units: list[dict], sol: dict) -> list[str]:
    """Come segreti() in test.js: risposte attese, opzioni, motivi e misure di un'unità (almeno 4 caratteri) che non
    compaiono nel suo testo visibile al primo passaggio (id, prompt, domanda, consegna, record)."""
    out = set()
    for u in units:
        s = sol.get(u["id"])
        if not s:
            continue
        visible = "\n".join([u["id"], u["prompt"] or "", u["domanda"] or "", u["consegna"] or "",
                             *[x["record"] for x in u["immagini"]]])
        cand = [s.get("motivo"), s.get("misura")]
        for x in s["varianti"]:
            cand += [x["risposta"], *x["opzioni"]]
        for t in cand:
            t = ("" if t is None else str(t)).strip()
            if len(t) >= 4 and not bounded(visible, t):
                out.add(t)
    return sorted(out)


def run_once(lib: str, pkg_dir: Path, dom: Path | None) -> dict:
    duk = Duktape(lib)
    res = {"dom": str(dom) if dom else None, "file": {}, "errore": None, "righe": None, "collegati": None}
    try:
        res["versione_duktape"] = duk.eval("String(typeof Duktape === 'object' ? Duktape.version : '?')")
        if dom:
            duk.eval(dom.read_text(encoding="utf-8"), str(dom))
            res["file"][str(dom)] = {"sha256": sha256_file(dom)}
        else:
            duk.eval("var window = this;", "prelude")
        for name in JS_FILES:
            path = pkg_dir / name
            t1 = time.time()
            duk.eval(path.read_text(encoding="utf-8"), name)
            res["file"][name] = {"sha256": sha256_file(path), "secondi": round(time.time() - t1, 2)}
        if dom:  # timer in sospeso (salvataggi ritardati dell'interfaccia)
            res["timer_rimasti"] = int(float(duk.eval("String(window.__DOM.svuotaTimer())", "timer")))
        res["righe"] = json.loads(duk.eval("JSON.stringify(window.RISULTATI_TEST || null)", "risultati"))
        res["collegati"] = json.loads(duk.eval(LINKS_JS, "collegamenti"))
    except (JSError, Failed) as exc:
        res["errore"] = str(exc)
    finally:
        duk.close()
    return res


def run_js(args) -> int:
    """Esegue con Duktape (libreria di sistema, via ctypes) i dati, logica.js, app.js e test.js del pacchetto, come
    verifica.html ma fuori dal browser: una volta senza DOM e, con --dom, una volta sul DOM simulato
    (scripts/review_package_dom.js). Confronta i collegamenti fra unità calcolati da logica.js con quelli calcolati qui
    e cerca nelle righe dell'autotest le soluzioni del secondo passaggio. Scrive --report ed esce con errore se una riga
    fallisce, se il codice non si esegue o se l'output contiene una soluzione. Non è una prova in un browser: niente
    disegno della pagina, eventi veri, immagini o localStorage."""
    t0 = time.time()
    pkg_dir = args.pacchetto
    pkg = read_js(pkg_dir / "dati/pacchetto.js", "PACCHETTO")
    sol = read_js(pkg_dir / "dati/soluzioni.js", "SOLUZIONI")
    ev = read_js(pkg_dir / "dati/evidenze.js", "EVIDENZE")
    units = pkg["campione"] + pkg["mirati"]
    _, _, _, linked = links_map(units, ev)
    expected_links = {k: sorted(v) for k, v in linked.items()}
    secrets = secrets_of(units, sol)
    out = {"schema": SCHEMA + "-esecuzione-js", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
           "pacchetto": str(pkg_dir), "impronta": pkg["impronta"],
           "motore": {"libreria": args.libreria, "sha256": sha256_file(Path(args.libreria))},
           "nota": "logica e interfaccia eseguite con Duktape, senza DOM e su un DOM simulato; non è una prova in un "
                   "browser (disegno, eventi veri, immagini e localStorage si provano con verifica.html)",
           "segreti_cercati": len(secrets), "esecuzioni": []}
    failed = False
    for dom in [None] + ([args.dom] if args.dom else []):
        res = run_once(args.libreria, pkg_dir, dom)
        label = "DOM simulato" if dom else "senza DOM"
        print(f"== esecuzione {label}", flush=True)
        rows = res["righe"]
        if res["errore"] or rows is None:
            print("ERRORE " + str(res["errore"]), flush=True)
            failed = True
        else:
            leaks = [i for i, r in enumerate(rows)
                     if any(bounded(r["nome"] + "\n" + r["dettaglio"], s) for s in secrets)]
            for i in leaks:  # nel rapporto (tracciato in git e scaricato con il pacchetto) non resta nulla
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
                  f"{len(rows)}; collegamenti uguali a quelli calcolati in Python: {res['collegamenti_uguali_a_python']}; "
                  f"righe con soluzioni: {len(leaks)}", flush=True)
            failed |= bool(res["falliti"] or leaks or not res["collegamenti_uguali_a_python"])
        res["collegati"] = None if res["collegati"] == expected_links else res["collegati"]
        out["esecuzioni"].append(res)
    out["seconds"] = round(time.time() - t0, 1)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--campione", required=True, type=Path)
    b.add_argument("--manifest", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    v = sub.add_parser("verifica")
    v.add_argument("--campione", required=True, type=Path)
    v.add_argument("--pacchetto", required=True, type=Path)
    v.add_argument("--manifest", required=True, type=Path)
    v.add_argument("--report", required=True, type=Path)
    j = sub.add_parser("esegui-js")
    j.add_argument("--pacchetto", required=True, type=Path)
    j.add_argument("--report", required=True, type=Path)
    j.add_argument("--dom", type=Path, help="DOM simulato da caricare prima dei dati (scripts/review_package_dom.js)")
    j.add_argument("--libreria", default=DUKTAPE)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("review_package.py va eseguito in un job SLURM")
    try:
        if args.cmd == "build":
            build(args)
        elif args.cmd == "esegui-js":
            sys.exit(run_js(args))
        else:
            sys.exit(verify(args))
    except Failed as exc:
        sys.exit(f"pacchetto non scritto o non valido: {exc}")


if __name__ == "__main__":
    main()
