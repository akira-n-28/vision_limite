#!/usr/bin/env python3
"""Integrazione verificata degli esiti della revisione umana di synth-v4-campione-2 (docs/specs/
S2_AMMISSIONE_PRIME_FAMIGLIE.md §2.13). Adattatore minimo sugli strumenti esistenti: riusa scripts/review_package.py
(lettura del pacchetto, verifica statica, collegamenti per record e per scena, Duktape) e la logica dell'interfaccia
(logica.js: controllaImport, tsvEsiti, tsvDecontaminazione). Solo in un job SLURM CPU (slurm/review_results.sbatch).

Gli esiti caricati e il pacchetto si leggono e non si modificano; nessun manifest, gruppo, split, target o ruolo del
pannello cambia; nessuna stima (C8) e nessuna certificazione. Fa:
1. identità: sha256 dei tre file di risultati uguali a quelli attesi (riga di comando); pacchetto, versione e impronta
   del file uguali al pacchetto; verifica statica del pacchetto ripetuta (review_package verifica, rapporto nella
   cartella d'uscita); logica.js con lo sha256 eseguito nell'ultima verifica del JavaScript (verifica_js.json);
2. collegamento: manifest synth-v4-candidato-2 letto e verificato (manifest.load), con gli sha256 del pacchetto; unità,
   varianti, record e corrispondenze del file uguali al pacchetto, nessuna mancante, ripetuta o in più;
3. JSON e TSV: logica.js (Duktape) importa il file senza errori né cambi e rigenera i due TSV, confrontati byte per byte
   con quelli caricati; in Python ogni riga dei TSV si ricostruisce dai campi del JSON;
4. cronologia: registro delle esposizioni ricostruito dalle soluzioni mostrate; per ogni unità la prima esposizione
   pertinente (stesso record o stessa scena) confrontata col salvataggio del primo giudizio, e con le annotazioni;
5. completezza dei giudizi richiesti, con non_applicabile distinto da non_compilato;
6. conteggi descrittivi per sezione e prime risposte confrontate coi target, separate dai giudizi finali;
7. materiale per il riesame dei candidati incerti: JSON del record dallo shard (sha256 del manifest), esami e note
   delle corrispondenze, testo dell'elemento del pannello (indice) e riga del benchmark;
8. proposta per record e gruppo di aggiornamento delle ammissioni, non applicata; il riesame tecnico facoltativo
   (--riesame) resta separato dal giudizio umano.

    srun $RUN_PYTHON -u -B scripts/review_results.py --pacchetto DIR --avanzamento FILE SHA256 --esiti FILE SHA256 \\
        --esiti-decontaminazione FILE SHA256 --campione DIR --manifest DIR --indice DIR --pannello DIR \\
        [--riesame FILE] --out DIR --docs PREFISSO
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import review_package as rp  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, sha256_file, write_new_json  # noqa: E402

SCHEMA = "synth-v4-revisione-integrazione/1"
PROGRESS = "synth-v4-revisione-avanzamento/2"
NON, NA = "non_compilato", "non_applicabile"
FIELDS = ("corretta", "coerente_con_annotazioni", "rispondibile_da_immagine", "gravita")
TSV_ESITI = ("unita", "variante", "record", "risposta_revisore", "corretta", "coerente_con_annotazioni",
             "rispondibile_da_immagine", "spiegazione", "contratto_coppia", "gravita", "tipo_difetto", "tempo_s", "note")
TSV_DECONTAM = ("unita", "record", "stesso_contenuto", "corrispondenze_guardate", "corrispondenza_decisiva", "note")
LINKS = {"stesso_record": "stesso record", "stessa_scena": "stessa scena", "stessa_unita": "questa unità"}
# motivi del manifest dei sintetici (synth_candidate: QUARANTINE ed EXCLUDING; synth_decontam.CHECK). La quarantena
# visiva è il motivo `incerto`, i cui dettagli cominciano col tipo di violazione: i casi mirati di quarantena esaminano
# solo `testi_sovrapposti`
DECONTAM = "decontaminazione_pannello"
VISUAL = "incerto"
VISUAL_KINDS = ("testi_sovrapposti",)
EXCLUDING = ("errore_matematico", "visivo_integrita", "contaminazione")
TECHNICAL = ("contenuto_diverso", "stesso_contenuto", "irrisolto")
QUARANTINE_CATEGORY = "quarantena"  # casi mirati: scene in quarantena per testi sovrapposti
# regola per riesaminare una quarantena visiva dai giudizi: fissata qui, prima di applicarla, ed elencata nel rapporto
VISUAL_RULE = {"leggibilita": "si", "coerente_con_annotazioni": "si", "rispondibile_da_immagine": "si",
               "gravita": "nessun_difetto", "primo_giudizio": "indipendente"}


class Failed(Exception):
    """Ingresso incoerente: niente rapporto."""


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def __call__(self, name: str, ok, detail=None) -> bool:
        self.items.append({"controllo": name, "esito": bool(ok), "dettaglio": detail})
        print(("ok   " if ok else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail is not None else ""),
              flush=True)
        return bool(ok)

    def failed(self) -> list[str]:
        return [c["controllo"] for c in self.items if not c["esito"]]


# ---------------------------------------------------------------- funzioni pure (collaudate in tests/)

def when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def js_round(x: float) -> int:
    return math.floor(x + 0.5)  # Math.round di JavaScript


def unescape(cell: str) -> str:
    """Come leggiCella di logica.js: \\n, \\t e \\\\ nelle celle dei TSV."""
    return re.sub(r"\\(\\|n|t)", lambda m: {"n": "\n", "t": "\t", "\\": "\\"}[m.group(1)], cell)


def read_tsv(text: str, columns: tuple) -> list[dict]:
    """TSV esportato dall'interfaccia: intestazione uguale alle colonne del modulo, ogni riga con lo stesso numero di
    campi, celle senza virgolette e con gli a capo come \\n."""
    if not text.endswith("\n"):
        raise Failed("TSV senza a capo finale")
    lines = text[:-1].split("\n")
    if tuple(lines[0].split("\t")) != columns:
        raise Failed(f"intestazione {lines[0][:80]!r} diversa dalle colonne del modulo")
    rows = []
    for n, line in enumerate(lines[1:], 2):
        cells = line.split("\t")
        if len(cells) != len(columns):
            raise Failed(f"riga {n}: {len(cells)} campi invece di {len(columns)}")
        rows.append(dict(zip(columns, map(unescape, cells))))
    return rows


def describe_exposure(e: dict) -> str:  # descriviEsposizione di logica.js
    return f"{', '.join(e['record'])} esposto il {e['quando']} tramite {e['tramite']} ({LINKS.get(e['collegamento'], e['collegamento'])})"


def first_exposures(units: list[dict], state: dict, scene_of: dict) -> dict:
    """Prima esposizione pertinente per unità, ricalcolata dal registro: un'esposizione riguarda un'unità se il record
    esposto è uno dei suoi o ha la stessa scena di uno dei suoi. {unità: {quando, tramite, record, collegamento}}."""
    by_scene: dict[str, set] = defaultdict(set)
    for r, k in scene_of.items():
        by_scene[k].add(r)
    out = {}
    events = sorted(state["esposizioni"], key=lambda e: (when(e["quando"]), e["record"], e["tramite"]))
    for u in units:
        mine = [x["record"] for x in u["immagini"]]
        near = set().union(*(by_scene[scene_of[r]] if r in scene_of else {r} for r in mine))
        for e in events:
            if e["record"] in near:
                link = ("stessa_unita" if e["tramite"] == u["id"] else
                        "stesso_record" if e["record"] in mine else "stessa_scena")
                out[u["id"]] = {"quando": e["quando"], "tramite": e["tramite"], "record": [e["record"]],
                                "collegamento": link}
                break
    return out


def independence(units: list[dict], state: dict, scene_of: dict) -> dict:
    """Per unità: indipendente se il primo giudizio è salvato non dopo la prima esposizione pertinente; confronto con
    le annotazioni del file (esposta_senza_primo_giudizio e provenienza del blocco)."""
    first = first_exposures(units, state, scene_of)
    out = {}
    for u in units:
        v = state["unita"].get(u["id"])
        e = first.get(u["id"])
        saved = v["prima"]["salvata_il"] if v and v.get("prima") else None
        ind = bool(saved and (e is None or when(saved) <= when(e["quando"])))
        noted = v.get("esposta_senza_primo_giudizio") if v else None
        block = (v or {}).get("prima_fissata") or {}
        want_block = None
        if e and saved:
            own = e["tramite"] == u["id"]
            want_block = ({"quando": e["quando"], "motivo": "soluzione_mostrata" if own else "record_esposto",
                           "tramite": e["tramite"]} if ind else
                          {"quando": saved, "motivo": "record_gia_esposto", "tramite": e["tramite"]})
        got_block = {k: block.get(k) for k in ("quando", "motivo", "tramite")} if block else None
        out[u["id"]] = {"prima_esposizione": e, "primo_giudizio_salvato": saved, "indipendente": ind,
                        "annotazione_coerente": (noted is None) == ind and (ind or noted == e),
                        "blocco_coerente": got_block == want_block}
    return out


def expected_rows(pkg: dict, state: dict) -> list[list[str]]:
    """Righe di esiti.tsv ricostruite dal JSON (stesse regole di tsvEsiti in logica.js, scritte qui di nuovo)."""
    units = {u["id"]: u for u in pkg["campione"] + pkg["mirati"]}
    rows = []
    for r in pkg["righe_esiti"]:
        u, v = units[r["unita"]], state["unita"].get(r["unita"])
        tempo = str(js_round(v["tempo_ms"] / 1000)) if v else ""
        shown = bool(v and v["soluzione_mostrata_il"])
        if r["variante"] == "coppia":
            c = v["dopo"]["coppia"] if shown else None
            rows.append([r["unita"], "coppia", r["record"], NA, NA, NA, NA, NA, c["contratto_coppia"] if c else NON,
                         c["gravita"] if c else NON, c["tipo_difetto"] if c else "", tempo, c["note"] if c else ""])
            continue
        p = v["prima"]["varianti"][r["variante"]] if v and v["prima"] else None
        d = v["dopo"]["varianti"][r["variante"]] if shown else None
        answer = NON if not p else "non_riesco_a_rispondere" if p["non_riesco"] else p["risposta"]
        if p and u["livello"] == "scena" and not p["non_riesco"] and not p["risposta"].strip():
            answer = "nessuna_risposta_caso_sulla_scena"
        note = d["note"] if d else ""
        if p and v["esposta_senza_primo_giudizio"]:
            note = (f"[primo giudizio non indipendente: {describe_exposure(v['esposta_senza_primo_giudizio'])}] "
                    f"risposta: {answer}" + (f"\n{note}" if note else ""))
            answer = "non_indipendente"
        rows.append([r["unita"], r["variante"], r["record"], answer, *(d[f] if d else NON for f in FIELDS[:3]), NA, NA,
                     d["gravita"] if d else NON, d["tipo_difetto"] if d else "", tempo, note])
    return rows


def expected_decontam_rows(pkg: dict, state: dict) -> list[list[str]]:
    """Righe di esiti_decontaminazione.tsv ricostruite dal JSON (regole di tsvDecontaminazione)."""
    rows = []
    didx = {d["id"]: d for d in pkg["decontaminazione"]}
    for r in pkg["righe_esiti_decontaminazione"]:
        d = didx[r["unita"]]
        v = state["decontaminazione"].get(d["id"])
        exams = [(x["id"], v["corrispondenze"][x["id"]]["esame"] if v else "non_esaminata") for x in d["elenco"]]
        seen = sum(e != "non_esaminata" for _, e in exams)
        outcome = v["esito"] if v else NON
        bad = decontam_errors(d, v) if v else []
        if bad and outcome != NON:
            outcome = "incompleto_" + outcome
        note = v["note"] if v else ""
        for x in d["elenco"] if v else []:
            if v["corrispondenze"][x["id"]]["note"]:
                note += ("\n" if note else "") + f"{x['id']}: {v['corrispondenze'][x['id']]['note']}"
        rows.append([r["unita"], r["record"], outcome,
                     f"{seen}/{len(exams)}: " + "; ".join(f"{i}={e}" for i, e in exams),
                     v["decisiva"] if v and v["esito"] == "si" else "", note])
    return rows


def decontam_errors(d: dict, v: dict) -> list[str]:
    """Regole dell'esito (controlloDecontam): "no" solo con tutte le corrispondenze contenuto_diverso e materiale
    completo; "si" con la corrispondenza decisiva esaminata come stesso contenuto."""
    out = []
    if v["esito"] == "no":
        if any(v["corrispondenze"][x["id"]]["esame"] != "contenuto_diverso" for x in d["elenco"]):
            out.append("no con corrispondenze non esaminate come contenuto diverso")
        if d["materiale"] != "completo" or any(not x["immagine"] or x["problemi"] for x in d["elenco"]):
            out.append("no con materiale incompleto")
    if v["esito"] == "si" and (v["corrispondenze"].get(v["decisiva"], {}).get("esame") != "stesso_contenuto"):
        out.append("si senza corrispondenza decisiva esaminata come stesso contenuto")
    if v["esito"] != "si" and v["decisiva"]:
        out.append("corrispondenza decisiva senza esito si")
    return out


def norm_answer(text: str):
    """Forma normalizzata di una risposta breve, per riconoscere equivalenze di scrittura (maiuscole, spazi, punto
    finale, gradi, zeri decimali, coordinate): non decide equivalenze semantiche."""
    s = re.sub(r"\s+", " ", text.strip().lower().replace("\u2212", "-").replace("°", "")).rstrip(".").strip()
    t = s.replace(" ", "")
    try:
        return ("numero", float(t))
    except ValueError:
        pass
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)?", t)
    if m:
        return ("coordinate", float(m[1]), float(m[2]))
    return ("testo", s)


def compare_answer(first: dict, target: str) -> str:
    if first["non_riesco"]:
        return "non_riesco"
    if first["risposta"].strip() == target.strip():
        return "uguale"
    return "equivalente_nella_scrittura" if norm_answer(first["risposta"]) == norm_answer(target) else "diversa"


def visual_positive(unit_state: dict, independent: bool) -> tuple[bool, dict]:
    """Regola VISUAL_RULE su un caso mirato di quarantena (unità sulla scena intera, variante unica)."""
    p, d = unit_state["prima"]["varianti"][""], unit_state["dopo"]["varianti"][""]
    got = {"leggibilita": p["leggibilita"], "coerente_con_annotazioni": d["coerente_con_annotazioni"],
           "rispondibile_da_immagine": d["rispondibile_da_immagine"], "gravita": d["gravita"],
           "primo_giudizio": "indipendente" if independent else "non_indipendente"}
    return got == VISUAL_RULE, got


def decontam_outcome(rec: dict) -> str | None:
    """Esito proprio della decontaminazione del record (evidenza decontaminazione_pannello del manifest)."""
    ev = [e for e in rec.get("evidenze") or [] if e.get("controllo") == DECONTAM]
    return ev[-1]["esito"] if ev else None


def severity(motives: list[str]) -> str:
    if any(m in EXCLUDING for m in motives):
        return "escluso"
    return "quarantena" if motives else "ammesso"


def propose(records: list[dict], human: dict, visual: dict, technical: dict) -> tuple[list[dict], list[dict]]:
    """Proposta per record e per gruppo, non applicata.
    records: record del manifest; human: record -> esito umano della decontaminazione (si, no, incerto, ...);
    visual: record -> (positivo, dettaglio, unità) dei casi mirati di quarantena; technical: record -> conclusione del
    riesame tecnico (TECHNICAL), separata dal giudizio umano.
    Per ogni motivo di un record non ammesso: risolto_dalla_revisione, risolvibile_col_riesame_tecnico (da approvare),
    riesaminabile_dai_giudizi (quarantena visiva), valido (blocco indipendente o non riesaminato), irrisolto,
    confermato. Lo stato proposto toglie solo i motivi risolti dalla revisione o riesaminabili dai giudizi; split,
    gruppi e target non cambiano."""
    own = {r["record"]: decontam_outcome(r) for r in records}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["gruppo"]].append(r)
    group_rows, group_status = [], {}
    for g, rs in groups.items():
        causes = sorted(r["record"] for r in rs if own[r["record"]] == "candidato")
        if not causes:
            continue
        outs = {c: human.get(c, "non_esaminato") for c in causes}
        if all(o == "no" for o in outs.values()):
            status = "risolto_dalla_revisione"
        elif any(o == "si" for o in outs.values()):
            status = "confermato"
        elif all(o == "no" or technical.get(c) == "contenuto_diverso" for c, o in outs.items()):
            status = "risolvibile_col_riesame_tecnico"
        else:
            status = "irrisolto"
        group_status[g] = status
        group_rows.append({"gruppo": g, "membri": sorted(r["record"] for r in rs), "cause": outs,
                           "riesame_tecnico": {c: technical[c] for c in causes if c in technical},
                           "stato_decontaminazione": status, "split": rs[0]["split"],
                           "stati": dict(Counter(r["stato"] for r in rs))})
    rows = []
    for r in records:
        motives = sorted(r.get("motivo") or {})
        if r["stato"] == "ammesso" and not motives:
            continue
        per, removable, remaining = {}, [], []
        for m in motives:
            if m == DECONTAM:
                status = group_status.get(r["gruppo"], "valido")
            elif m == VISUAL:
                covered = all(str(x).split(" ", 1)[0] in VISUAL_KINDS for x in r["motivo"][m])
                status = "riesaminabile_dai_giudizi" if covered and visual.get(r["record"], (False,))[0] else "valido"
            else:
                status = "valido"
            per[m] = status
            (removable if status in ("risolto_dalla_revisione", "riesaminabile_dai_giudizi") else remaining).append(m)
        proposed = severity(remaining)
        after_technical = severity([m for m in remaining if per[m] != "risolvibile_col_riesame_tecnico"])
        if any(per[m] == "confermato" for m in remaining):  # regola di v2: corrispondenza confermata esclude il gruppo
            proposed = after_technical = "escluso"
        if r["stato"] == "ammesso":
            category = "gia_ammesso"  # motivi solo informativi
        elif proposed == "escluso" and r["stato"] != "escluso":
            category = "esclusione_proposta"
        elif proposed == "ammesso":
            category = "riammissione_proposta"
        elif after_technical == "ammesso":
            category = "riammissione_dopo_riesame_tecnico"
        elif any(per[m] in ("valido", "confermato") for m in remaining):
            category = "resta_bloccato"  # almeno un blocco indipendente da quelli riesaminati
        else:
            category = "irrisolto"
        v = visual.get(r["record"])
        rows.append({"record": r["record"], "record_id": r["record_id"], "insieme": r["insieme"], "split": r["split"],
                     "gruppo": r["gruppo"], "famiglia": r["famiglia"], "coppia": r.get("coppia"),
                     "stato_attuale": r["stato"], "motivi": per,
                     "esito_proprio_decontaminazione": own[r["record"]], "esito_umano": human.get(r["record"]),
                     "riesame_tecnico": technical.get(r["record"]),
                     "caso_quarantena": v[2] if v else None, "motivi_rimovibili": removable,
                     "blocchi_rimanenti": remaining, "stato_proposto": proposed,
                     "stato_dopo_riesame_tecnico": after_technical, "categoria": category})
    return rows, group_rows


# ---------------------------------------------------------------- letture dei dati reali

def read_results(args, checks: Checks) -> dict:
    out = {}
    for name, (path, sha) in (("avanzamento", args.avanzamento), ("esiti", args.esiti),
                              ("esiti_decontaminazione", args.esiti_decontaminazione)):
        path = Path(path)
        entry = file_entry(path)
        if entry["sha256"] != sha:
            raise Failed(f"{path}: sha256 {entry['sha256']}, atteso {sha}")
        # byte esatti: nessuna conversione degli a capo, così il confronto coi TSV rigenerati è byte per byte
        out[name] = {**entry, "path": str(path.resolve().relative_to(ROOT)), "text": path.read_bytes().decode("utf-8")}
    checks("file di risultati con gli sha256 attesi", True,
           {k: v["sha256"][:12] for k, v in out.items()})
    return out


def run_logic(pkg_dir: Path, text: str) -> dict:
    """logica.js con Duktape: controllaImport sul file di avanzamento, poi i TSV e il riepilogo dallo stato importato,
    e il confronto dello stato importato col file (nessun cambio)."""
    duk = rp.Duktape(rp.DUKTAPE)
    try:
        duk.eval("var window = this;", "prelude")
        for name in ("dati/pacchetto.js", "dati/soluzioni.js", "dati/evidenze.js", "logica.js"):
            duk.eval((pkg_dir / name).read_text(encoding="utf-8"), name)
        literal = json.dumps(text).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        res = duk.eval(f"""JSON.stringify((function () {{
  var L = window.Logica, ctx = L.contesto(window.PACCHETTO, window.EVIDENZE), testo = {literal};
  var r = L.controllaImport(ctx, testo), o = {{errori: r.errori, migrazione: r.migrazione,
    riconciliazione: r.riconciliazione || null}};
  if (r.stato) {{
    var d = JSON.parse(testo), keys = ["unita", "decontaminazione", "esposizioni", "archivio", "migrazioni"];
    o.invariato = keys.every(function (k) {{ return L.canonico(r.stato[k]) === L.canonico(d[k]); }});
    o.tsv = L.tsvEsiti(ctx, r.stato);
    o.tsv_decontaminazione = L.tsvDecontaminazione(ctx, r.stato);
    o.riepilogo = L.riepilogo(ctx, r.stato);
  }}
  return o;
}})())""", "importazione")
        return json.loads(res)
    finally:
        duk.close()


TABLES: dict[Path, list[dict]] = {}


def bench_row(entry: dict) -> dict | None:
    """Campi scalari (non immagini) della riga del benchmark per una voce dell'indice dei benchmark in parquet; None per
    gli altri formati (resta il testo dell'indice)."""
    import panel_index as pi
    import pyarrow.parquet as pq

    spec = pi.BENCHMARKS.get(entry["benchmark"])
    if spec is None:
        return None
    path = pi.snap(spec["repo"], spec["rev"]) / entry["file"]
    if path not in TABLES:
        keep = [f.name for f in pq.read_schema(path)
                if not str(f.type).startswith(("binary", "large_binary", "struct", "list", "large_list"))]
        TABLES[path] = pq.read_table(path, columns=keep).to_pylist()
    ident = entry["item"].split(":", 1)[1]
    for i, row in enumerate(TABLES[path]):
        if (str(row.get(spec["id"])) if spec["id"] else str(i)) == ident:
            return {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) for k, v in row.items()}
    return None


def incerti_material(pkg: dict, state: dict, by_key: dict, groups: dict, index_dir: Path, panel_dir: Path,
                     checks: Checks) -> list[dict]:
    """Materiale del riesame per i candidati con esito umano incerto: record (JSON dallo shard, sha256 del manifest),
    gruppo, corrispondenze con esame e note, elementi del pannello (voce dell'indice, riga del benchmark, gruppo e
    ruoli del pannello)."""
    incerti = [d for d in pkg["decontaminazione"] if state["decontaminazione"][d["id"]]["esito"] == "incerto"]
    keys = {d["record"] for d in incerti}
    found = {}
    for shard in sorted({by_key[k]["shard"] for k in keys}):
        want = {k for k in keys if by_key[k]["shard"] == shard}
        for key, files in rp.iter_shard(ROOT / shard):
            if key in want:
                found[key] = files["json"].decode("utf-8")
                want.discard(key)
                if not want:
                    break
    ok = all(k in found and hashlib.sha256(found[k].encode("utf-8")).hexdigest() == by_key[k]["json_sha256"]
             for k in keys)
    checks("riesame: JSON dei record incerti letto dagli shard, con lo sha256 del manifest", ok, len(found))
    index = json.loads((index_dir / "index.json").read_text(encoding="utf-8"))["items"]
    wanted = {(x["benchmark"], x["voce_indice"]) for d in incerti for x in d["elenco"]}
    entries = {(e["benchmark"], e["id"]): e for e in index if (e["benchmark"], e["id"]) in wanted}
    del index
    with gzip.open(panel_dir / "elements.jsonl.gz", "rt", encoding="utf-8") as fh:
        elements = [json.loads(line) for line in fh if line.strip()]
    panel_groups = {x["gruppo_pannello"] for d in incerti for x in d["elenco"]}
    by_group = defaultdict(list)
    for e in elements:
        if e.get("gruppo") in panel_groups:
            by_group[e["gruppo"]].append({"elemento": e["chiave"], "ruolo": e.get("ruolo"),
                                          "metrica_primaria": e.get("metrica_primaria")})
    checks("riesame: ogni corrispondenza incerta ritrova la sua voce dell'indice", len(entries) == len(wanted),
           {"voci": len(entries), "attese": len(wanted)})
    out = []
    for d in incerti:
        v = state["decontaminazione"][d["id"]]
        rec = by_key[d["record"]]
        data = json.loads(found[d["record"]])
        members = [{"record": m["record"], "insieme": m["insieme"], "split": m["split"], "stato": m["stato"],
                    "esito_proprio": decontam_outcome(m)} for m in groups[rec["gruppo"]]]
        matches = []
        for x in d["elenco"]:
            e = entries.get((x["benchmark"], x["voce_indice"]))
            matches.append({**{k: x[k] for k in ("id", "benchmark", "elemento", "voce_indice", "posizione_immagine",
                                                 "distanza", "ruolo", "metrica_primaria", "gruppo_pannello", "immagine")},
                            "esame": v["corrispondenze"][x["id"]]["esame"], "nota": v["corrispondenze"][x["id"]]["note"],
                            "testo_indice": e["text"] if e else None, "indizi_indice": e["hints"] if e else None,
                            "riga_benchmark": bench_row(e) if e else None,
                            "gruppo_del_pannello": by_group.get(x["gruppo_pannello"], [])})
        out.append({"unita": d["id"], "record": d["record"], "record_id": rec["record_id"], "insieme": rec["insieme"],
                    "split": rec["split"], "stato": rec["stato"], "gruppo": rec["gruppo"], "motivo": rec["motivo"],
                    "famiglia": rec["famiglia"], "immagine": d["immagine"], "somiglianza_pixel": d["somiglianza_pixel"],
                    "esito_umano": v["esito"], "nota_umana": v["note"], "spec": data.get("spec"),
                    "qa": data.get("qa"), "domanda_della_coppia": data.get("pair"), "membri_del_gruppo": members,
                    "corrispondenze": matches})
    return out


def read_technical(path: Path | None, incerti: dict) -> dict:
    """Riesame tecnico (TSV: unita, record, conclusione, motivazione), separato dal giudizio umano: solo candidati
    incerti, una riga per unità, conclusioni fra TECHNICAL."""
    if path is None:
        return {}
    rows = read_tsv(path.read_text(encoding="utf-8"), ("unita", "record", "conclusione", "motivazione"))
    seen, out = set(), {}
    for r in rows:
        if r["unita"] not in incerti or incerti[r["unita"]] != r["record"]:
            raise Failed(f"riesame: {r['unita']} {r['record']} non è un candidato incerto del pacchetto")
        if r["unita"] in seen or r["conclusione"] not in TECHNICAL or not r["motivazione"].strip():
            raise Failed(f"riesame: riga di {r['unita']} ripetuta, conclusione {r['conclusione']!r} o motivazione vuota")
        seen.add(r["unita"])
        out[r["record"]] = r["conclusione"]
    return out


# ---------------------------------------------------------------- esecuzione

def run(args) -> int:
    t0 = time.time()
    out_dir, docs = args.out, Path(args.docs)
    for p in (out_dir, Path(f"{docs}.json"), Path(f"{docs}-proposta.tsv")):
        if p.exists():
            raise Failed(f"{p} esiste già: le uscite si scrivono una volta")
    checks = Checks()
    res = read_results(args, checks)
    pkg_dir = args.pacchetto
    pkg = rp.read_js(pkg_dir / "dati/pacchetto.js", "PACCHETTO")
    sol = rp.read_js(pkg_dir / "dati/soluzioni.js", "SOLUZIONI")
    ev = rp.read_js(pkg_dir / "dati/evidenze.js", "EVIDENZE")
    meta = json.loads((pkg_dir / "pacchetto.json").read_text(encoding="utf-8"))
    state = json.loads(res["avanzamento"]["text"])
    out_dir.mkdir(parents=True)

    # ---- 1. identità
    checks("file di avanzamento: formato 2, pacchetto, versione e impronta del pacchetto",
           state.get("schema") == PROGRESS and state.get("pacchetto") == {"id": pkg["id"], "versione": pkg["versione"],
                                                                          "impronta": meta["impronta"]}
           and pkg["impronta"] == meta["impronta"], state.get("pacchetto"))
    rc = rp.verify(argparse.Namespace(campione=args.campione, pacchetto=pkg_dir, manifest=args.manifest,
                                      report=out_dir / "verifica_pacchetto.json"))
    checks("pacchetto: verifica statica ripetuta superata (review_package verifica)", rc == 0,
           str((out_dir / "verifica_pacchetto.json").relative_to(ROOT)))
    js = json.loads((pkg_dir / "verifica_js.json").read_text(encoding="utf-8"))
    ran = {e["file"].get("logica.js", {}).get("sha256") for e in js["esecuzioni"]}
    checks("logica.js uguale a quella eseguita dall'ultima verifica del JavaScript (verifica_js.json)",
           ran == {sha256_file(pkg_dir / "logica.js")}, {"job": js.get("job_id")})

    # ---- 2. collegamento al manifest e completezza degli identificativi
    m = mf.load(args.manifest)
    by_key = {k: r for shard in m.by_shard.values() for k, r in shard.items()}
    src = pkg["sorgente"]["manifest"]
    checks("manifest synth-v4-candidato-2 letto e verificato, con gli sha256 registrati nel pacchetto",
           m.manifest_sha256 == src["manifest_sha256"] and m.meta["records"]["sha256"] == src["records_sha256"],
           m.manifest_sha256[:12])
    units = pkg["campione"] + pkg["mirati"]
    uidx = {u["id"]: u for u in units}
    checks("unità del file = unità del pacchetto (380 + 175), nessuna mancante o in più",
           set(state["unita"]) == set(uidx) and len(uidx) == 555, len(state["unita"]))
    bad = [u["id"] for u in units if set(state["unita"][u["id"]]["varianti"]) != set(u["varianti"])
           or any(state["unita"][u["id"]]["varianti"][k]["record"] != u["record_per_variante"][k] for k in u["varianti"])]
    checks("varianti e record di ogni unità uguali al pacchetto", not bad, bad[:5])
    bad = [x["record"] for u in units for x in u["immagini"]
           if x["record"] not in by_key or by_key[x["record"]]["record_id"] != x["record_id"]]
    checks("record delle unità nel manifest, con lo stesso record_id", not bad, bad[:5])
    didx = {d["id"]: d for d in pkg["decontaminazione"]}
    n_matches = sum(len(d["elenco"]) for d in pkg["decontaminazione"])
    bad = [i for i, d in didx.items() if i not in state["decontaminazione"]
           or state["decontaminazione"][i]["record"] != d["record"]
           or set(state["decontaminazione"][i]["corrispondenze"]) != {x["id"] for x in d["elenco"]}]
    checks("decontaminazione: 264 candidati e 755 corrispondenze del pacchetto, nessuno mancante o in più",
           not bad and set(state["decontaminazione"]) == set(didx) and len(didx) == 264 and n_matches == 755, bad[:5])
    bad = [d["record"] for d in pkg["decontaminazione"]
           if d["record"] not in by_key or by_key[d["record"]]["record_id"] != d["record_id"]
           or decontam_outcome(by_key[d["record"]]) != "candidato"]
    cand = sorted(k for k, r in by_key.items() if decontam_outcome(r) == "candidato")
    checks("candidati del pacchetto = record con esito candidato nel manifest, con lo stesso record_id",
           not bad and cand == sorted(didx[i]["record"] for i in didx), {"manifest": len(cand), "diversi": bad[:5]})
    open_units = [i for i, v in state["unita"].items() if not v.get("prima") or not v.get("soluzione_mostrata_il")]
    checks("ogni unità ha il primo giudizio salvato e la soluzione mostrata", not open_units, open_units[:5])
    if checks.failed():  # identificativi o revisione incompleti: il resto non si può ricalcolare
        raise Failed("ingressi incoerenti col pacchetto o revisione incompleta: " + "; ".join(checks.failed()))

    # ---- 3. JSON e TSV
    logic = run_logic(pkg_dir, res["avanzamento"]["text"])
    checks("logica.js (Duktape): importazione senza errori, migrazioni o riclassificazioni, stato invariato",
           not logic["errori"] and logic["migrazione"] is None and logic["riconciliazione"] is None
           and logic.get("invariato"), logic["errori"][:3] or logic["riconciliazione"])
    checks("logica.js: esiti.tsv rigenerato dal JSON uguale byte per byte a quello caricato",
           logic.get("tsv") == res["esiti"]["text"])
    checks("logica.js: esiti_decontaminazione.tsv rigenerato uguale byte per byte",
           logic.get("tsv_decontaminazione") == res["esiti_decontaminazione"]["text"])
    tsv = read_tsv(res["esiti"]["text"], TSV_ESITI)
    tsvd = read_tsv(res["esiti_decontaminazione"]["text"], TSV_DECONTAM)
    scene_of = {r: rp.scene_key(e["json"]) for r, e in ev.items()}
    indep = independence(units, state, scene_of)
    got = [[r[c] for c in TSV_ESITI] for r in tsv]
    want = expected_rows(pkg, state)
    diff = [g[0] + "/" + g[1] for g, w in zip(got, want) if g != w]
    checks("esiti.tsv: 655 righe nell'ordine del modulo, ogni campo uguale a quello ricostruito in Python dal JSON",
           len(got) == len(want) == 655 and not diff, diff[:5])
    gotd = [[r[c] for c in TSV_DECONTAM] for r in tsvd]
    wantd = expected_decontam_rows(pkg, state)
    diff = [g[0] for g, w in zip(gotd, wantd) if g != w]
    checks("esiti_decontaminazione.tsv: 264 righe, ogni campo uguale a quello ricostruito in Python",
           len(gotd) == len(wantd) == 264 and not diff, diff[:5])
    checks("TSV: nessuna riga ripetuta", len({(r["unita"], r["variante"]) for r in tsv}) == len(tsv)
           and len({r["unita"] for r in tsvd}) == len(tsvd))

    # ---- 4. cronologia ed esposizioni
    implicit = {(x, state["unita"][u["id"]]["soluzione_mostrata_il"], u["id"]) for u in units
                if state["unita"][u["id"]]["soluzione_mostrata_il"] for x in dict.fromkeys(i["record"] for i in u["immagini"])}
    registry = {(e["record"], e["quando"], e["tramite"]) for e in state["esposizioni"]}
    checks("registro delle esposizioni = record esposti dalle soluzioni mostrate, senza voci in più o mancanti",
           registry == implicit and len(registry) == len(state["esposizioni"])
           and all(e["origine"] == "soluzione_mostrata" for e in state["esposizioni"]),
           {"registro": len(registry), "dalle_soluzioni": len(implicit)})
    order = [u["id"] for u in units if not (when(state["unita"][u["id"]]["prima"]["salvata_il"])
                                            <= when(state["unita"][u["id"]]["soluzione_mostrata_il"]))]
    checks("primo giudizio salvato prima della propria soluzione, per ogni unità", not order, order[:5])
    late = [u["id"] for u in units for e in state["unita"][u["id"]]["eventi"]
            if e["azione"] == "giudizio_finale" and when(e["quando"]) < when(state["unita"][u["id"]]["soluzione_mostrata_il"])]
    checks("giudizi finali registrati dopo la soluzione", not late, late[:5])
    linked_units = {k: sorted(v) for k, v in rp.links_map(units, ev)[3].items() if v}
    nonind = sorted(k for k, v in indep.items() if not v["indipendente"])
    independence_facts = {"indipendenti": len(units) - len(nonind), "non_indipendenti": nonind,
                          "unita_collegate": linked_units,
                          "unita_collegate_con_prima_esposizione_altrui": sorted(
                              k for k, v in indep.items() if v["prima_esposizione"]
                              and v["prima_esposizione"]["tramite"] != k)}
    print("indipendenza ricalcolata: " + json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v)
                                                    for k, v in independence_facts.items()}), flush=True)
    bad = [k for k, v in indep.items() if not v["annotazione_coerente"] or not v["blocco_coerente"]]
    checks("annotazioni del file (esposizione senza primo giudizio, provenienza del blocco) uguali al ricalcolo",
           not bad, bad[:5])
    fixed = [u["id"] for u in units if state["unita"][u["id"]]["bozza"] != state["unita"][u["id"]]["prima"]["varianti"]]
    checks("bozza uguale al primo giudizio fissato (nessuna modifica dopo la fissazione)", not fixed, fixed[:5])

    # ---- 5. completezza dei giudizi
    missing, values = [], defaultdict(Counter)
    for u in units:
        v = state["unita"][u["id"]]
        for k in u["varianti"]:
            p, d = v["prima"]["varianti"][k], v["dopo"]["varianti"][k]
            if u["livello"] == "qa" and not p["non_riesco"] and not p["risposta"].strip():
                missing.append(f"{u['id']}/{k}: risposta")
            if p["leggibilita"] == NON:
                missing.append(f"{u['id']}/{k}: leggibilita")
            values[f"leggibilita/{u['livello']}"][p["leggibilita"]] += 1
            for f in FIELDS:
                values[f"{f}/{u['livello']}"][d[f]] += 1
                if d[f] == NON:
                    missing.append(f"{u['id']}/{k}: {f}")
            if d["gravita"] in ("minore", "grave") and not d["tipo_difetto"].strip():
                missing.append(f"{u['id']}/{k}: tipo_difetto")
        if u["coppia"]:
            c = v["dopo"]["coppia"]
            values["contratto_coppia"][c["contratto_coppia"]] += 1
            values["gravita_coppia"][c["gravita"]] += 1
            missing += [f"{u['id']}/coppia: {f}" for f in ("contratto_coppia", "gravita") if c[f] == NON]
    dbad = []
    for d in pkg["decontaminazione"]:
        v = state["decontaminazione"][d["id"]]
        values["esito_decontaminazione"][v["esito"]] += 1
        for x in d["elenco"]:
            values["esame_corrispondenza"][v["corrispondenze"][x["id"]]["esame"]] += 1
        if v["esito"] == NON:
            missing.append(f"{d['id']}: esito")
        dbad += [f"{d['id']}: {e}" for e in decontam_errors(d, v)]
    checks("giudizi richiesti tutti compilati (non_compilato = mancante; non_applicabile ammesso)", not missing,
           missing[:5])
    checks("esiti della decontaminazione coerenti con le regole (no, si, decisiva, materiale)", not dbad, dbad[:5])
    na_qa = sorted(f"{u['id']}/{k}" for u in units if u["livello"] == "qa" for k in u["varianti"]
                   if state["unita"][u["id"]]["dopo"]["varianti"][k]["corretta"] == NA)
    checks("correttezza non_applicabile solo nei casi sulla scena intera", not na_qa, na_qa[:5])

    # ---- 6. conteggi descrittivi e prime risposte
    sections = {"campione": pkg["campione"], "mirati": pkg["mirati"]}
    by_section = {}
    per_unit = []
    for name, us in sections.items():
        table = defaultdict(lambda: defaultdict(Counter))
        for u in us:
            v = state["unita"][u["id"]]
            s = sol[u["id"]]
            targets = {x["variante"]: x["risposta"] for x in s["varianti"]}
            for k in u["varianti"]:
                p, d = v["prima"]["varianti"][k], v["dopo"]["varianti"][k]
                cmp_ = compare_answer(p, targets[k]) if u["livello"] == "qa" else "caso_sulla_scena"
                targets.setdefault(k, None)
                g = table[u["gruppo"]]
                g["varianti"]["n"] += 1
                g["prima_risposta"][cmp_] += 1
                g["leggibilita"][p["leggibilita"]] += 1
                for f in FIELDS:
                    g[f][d[f]] += 1
                per_unit.append({"sezione": name, "gruppo": u["gruppo"], "unita": u["id"], "variante": k,
                                 "record": u["record_per_variante"][k], "livello": u["livello"], "tipo": u["tipo"],
                                 "prima_risposta": "non_riesco_a_rispondere" if p["non_riesco"] else p["risposta"],
                                 "target": targets[k], "confronto": cmp_, "leggibilita": p["leggibilita"],
                                 **{f: d[f] for f in FIELDS}, "tipo_difetto": d["tipo_difetto"], "note": d["note"],
                                 "risposta_dopo_soluzione": d["risposta_dopo_soluzione"],
                                 "indipendente": indep[u["id"]]["indipendente"]})
            if u["coppia"]:
                c = v["dopo"]["coppia"]
                g = table[u["gruppo"]]
                g["contratto_coppia"][c["contratto_coppia"]] += 1
                g["gravita_coppia"][c["gravita"]] += 1
        by_section[name] = {grp: {k: dict(c) for k, c in t.items()} for grp, t in table.items()}
    positive = {"corretta": {"si", NA}, "coerente_con_annotazioni": {"si", NA}, "rispondibile_da_immagine": {"si", NA},
                "gravita": {"nessun_difetto"}}
    not_pos = [f"{x['unita']}/{x['variante']}: {f}={x[f]}" for x in per_unit for f in FIELDS if x[f] not in positive[f]]
    not_pos += [f"{u['id']}/coppia: {c}" for u in units if u["coppia"]
                for c in [state["unita"][u["id"]]["dopo"]["coppia"]] if (c["contratto_coppia"], c["gravita"])
                != ("si", "nessun_difetto")]
    diverse = [x for x in per_unit if x["confronto"] == "diversa"]
    notes = [x for x in per_unit if x["note"].strip() or x["risposta_dopo_soluzione"].strip()]
    dec = Counter(state["decontaminazione"][i]["esito"] for i in didx)
    exams = values["esame_corrispondenza"]
    stated = {"campione": 380, "mirati": 175, "righe_tsv": 655, "candidati": 264, "corrispondenze": 755,
              "esito_no": 249, "esito_incerto": 15, "esito_si": 0, "corrispondenze_diverse": 727,
              "corrispondenze_incerte": 28}
    observed = {"campione": len(pkg["campione"]), "mirati": len(pkg["mirati"]), "righe_tsv": len(tsv),
                "candidati": len(didx), "corrispondenze": n_matches, "esito_no": dec["no"],
                "esito_incerto": dec["incerto"], "esito_si": dec["si"],
                "corrispondenze_diverse": exams["contenuto_diverso"], "corrispondenze_incerte": exams["incerto"]}
    checks("conteggi della revisione statica confermati dai dati", observed == stated,
           {k: (observed[k], stated[k]) for k in stated if observed[k] != stated[k]} or observed)
    checks("giudizi finali applicabili sulla qualità tutti positivi (sì o non applicabile, nessun difetto; contratti "
           "sì)", not not_pos, not_pos[:5])

    # ---- 7. riesame dei candidati incerti: materiale
    groups = defaultdict(list)
    for r in by_key.values():
        groups[r["gruppo"]].append(r)
    incerti = incerti_material(pkg, state, by_key, groups, args.indice, args.pannello, checks)
    technical = read_technical(args.riesame, {x["unita"]: x["record"] for x in incerti})

    # ---- 8. proposta per record e gruppo
    human = {didx[i]["record"]: state["decontaminazione"][i]["esito"] for i in didx}
    visual = {}
    for u in pkg["mirati"]:
        if u["gruppo"] == QUARANTINE_CATEGORY:
            r = u["record_per_variante"][""]
            ok, detail = visual_positive(state["unita"][u["id"]], indep[u["id"]]["indipendente"])
            visual[r] = (ok, detail, u["id"])
    quarantined = {k for k, r in by_key.items() if VISUAL in (r.get("motivo") or {})}
    kinds = Counter(str(x).split(" ", 1)[0] for k in quarantined for x in by_key[k]["motivo"][VISUAL])
    checks("casi mirati di quarantena = record col motivo incerto nel manifest (67), dettagli per tipo",
           set(visual) == quarantined and len(visual) == 67, {"casi": len(visual), "manifest": len(quarantined),
                                                               "dettagli": dict(kinds)})
    rows, group_rows = propose(list(by_key.values()), human, visual, technical)
    worse = [x["record"] for x in rows if mf_sev(x["stato_proposto"]) > mf_sev(x["stato_attuale"])
             and x["categoria"] != "esclusione_proposta"]
    checks("proposta: stati più gravi solo per contaminazioni confermate, altrimenti solo motivi tolti; split e gruppi "
           "invariati", not worse, worse[:5])
    summary = {
        "categorie": dict(Counter(x["categoria"] for x in rows)),
        "per_insieme_split": {k: dict(c) for k, c in sorted(_by(rows, lambda x: f"{x['insieme']}/{x['split']}",
                                                                 lambda x: f"{x['stato_attuale']} -> {x['stato_proposto']}").items())},
        "motivi": {k: dict(c) for k, c in sorted(_motive_counts(rows).items())},
        "gruppi_con_candidati": dict(Counter(g["stato_decontaminazione"] for g in group_rows)),
        "regola_quarantena_visiva": VISUAL_RULE,
        "quarantene_visive": dict(Counter("positiva" if v[0] else "non_positiva" for v in visual.values())),
    }

    # ---- uscite
    report = {"schema": SCHEMA, "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "ingressi": {"risultati": {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in res.items()},
                           "pacchetto": {"cartella": str(pkg_dir.resolve().relative_to(ROOT)), "id": pkg["id"],
                                         "versione": pkg["versione"], "impronta": pkg["impronta"]},
                           "manifest": {"cartella": str(Path(args.manifest).resolve().relative_to(ROOT)),
                                        "manifest_sha256": m.manifest_sha256,
                                        "records_sha256": m.meta["records"]["sha256"]},
                           "riesame_tecnico": file_entry(args.riesame) if args.riesame else None},
              "controlli": checks.items, "falliti": checks.failed(),
              "indipendenza": independence_facts, "archivio": len(state["archivio"]),
              "cronologia": {"creato_il": state["creato_il"], "esportato_il": state["esportato_il"],
                             "primo_giudizio": min(x["primo_giudizio_salvato"] for x in indep.values()),
                             "ultimo_giudizio": max(x["primo_giudizio_salvato"] for x in indep.values()),
                             "esposizioni": len(state["esposizioni"]), "migrazioni": state["migrazioni"]},
              "riepilogo_esportato": state["riepilogo"], "riepilogo_ricalcolato_da_logica": logic.get("riepilogo"),
              "conteggi": {"osservati": observed, "dichiarati": stated, "valori": {k: dict(c) for k, c in values.items()},
                           "per_sezione": by_section},
              "prime_risposte_diverse": diverse, "note_del_revisore": notes,
              "riesame_incerti": incerti, "riesame_tecnico": technical,
              "proposta": {"riepilogo": summary, "gruppi": group_rows},
              "c8": {"stato": "non applicato: la corrispondenza fra esiti del pacchetto ed esiti di C8 è una proposta "
                              "(S4 §9) non approvata prima della revisione",
                     "prerequisiti_mancanti": ["corrispondenza fra valori del pacchetto ed esiti di C8 approvata",
                                               "affermazioni per proprietà e strato fissate (S4 §9)",
                                               "D5: numerosità e budget degli audit",
                                               "sensibilità del revisore non misurata: conclusioni condizionate"],
                     "certificazione": "nessuna"},
              "costi": {"secondi": round(time.time() - t0, 1)}}
    write_new_json(out_dir / "report.json", report)
    write_tsv(out_dir / "esiti_per_unita.tsv", per_unit)
    flat = [{**{k: v for k, v in x.items() if k not in ("motivi", "motivi_rimovibili", "blocchi_rimanenti")},
             "motivi": "; ".join(f"{k}={v}" for k, v in x["motivi"].items()),
             "motivi_rimovibili": ",".join(x["motivi_rimovibili"]), "blocchi_rimanenti": ",".join(x["blocchi_rimanenti"])}
            for x in rows]
    write_tsv(out_dir / "proposta_ammissioni.tsv", flat)
    touched = {g for g in (x["gruppo"] for x in group_rows)} | {by_key[r]["gruppo"] for r in visual}
    Path(f"{docs}.json").parent.mkdir(parents=True, exist_ok=True)
    write_tsv(Path(f"{docs}-proposta.tsv"), [x for x in flat if x["gruppo"] in touched])
    light = {**report, "file_completi": {n: str((out_dir / n).relative_to(ROOT)) for n in
                                         ("report.json", "esiti_per_unita.tsv", "proposta_ammissioni.tsv")}}
    write_new_json(Path(f"{docs}.json"), light)
    print(json.dumps({"controlli": len(checks.items), "falliti": checks.failed(), "categorie": summary["categorie"],
                      "secondi": report["costi"]["secondi"]}, ensure_ascii=False), flush=True)
    return 1 if checks.failed() else 0


def mf_sev(state: str) -> int:
    return {"ammesso": 0, "quarantena": 1, "escluso": 2}[state]


def _by(rows, key, val):
    out = defaultdict(Counter)
    for x in rows:
        out[key(x)][val(x)] += 1
    return out


def _motive_counts(rows):
    out = defaultdict(Counter)
    for x in rows:
        for m, s in x["motivi"].items():
            out[m][s] += 1
    return out


def write_tsv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise Failed(f"{path} esiste già")
    cols = list(rows[0]) if rows else []
    cell = lambda v: "" if v is None else str(v).replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")  # noqa: E731
    path.write_text("\t".join(cols) + "\n" + "".join("\t".join(cell(r[c]) for c in cols) + "\n" for r in rows),
                    encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pacchetto", type=Path, required=True)
    for name in ("--avanzamento", "--esiti", "--esiti-decontaminazione"):
        ap.add_argument(name, nargs=2, required=True, metavar=("FILE", "SHA256"))
    ap.add_argument("--campione", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--indice", type=Path, required=True)
    ap.add_argument("--pannello", type=Path, required=True)
    ap.add_argument("--riesame", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", required=True, help="prefisso del riepilogo versionato (.json e -proposta.tsv)")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("review_results.py va eseguito in un job SLURM")
    try:
        sys.exit(run(args))
    except (Failed, rp.Failed, mf.ManifestError) as exc:
        sys.exit(f"fermato: {exc}")


if __name__ == "__main__":
    main()
