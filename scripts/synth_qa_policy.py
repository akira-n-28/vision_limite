#!/usr/bin/env python3
"""Scorciatoia testuale di function/exists nei sintetici v4 e politica delle domande (S2 §2.14). Solo in un job SLURM
CPU; manifest, shard, revisione e campione in sola lettura; scrive solo nelle cartelle e nei file nuovi indicati.

Il problema: in v3 e v4 (synth._function_qa) i grafici presenti si chiamano sempre f e, se c'è, g; la domanda exists
positiva sceglie fra i presenti e quella negativa fra h, p, q. Le risposte sono vere, ma il nome chiesto le dà senza
guardare l'immagine (f, g -> yes; h, p, q -> no). Il checker approva risposte vere e non vede la distribuzione delle
domande; la revisione visiva non prova l'assenza di scorciatoie.

Sottocomandi:
- impatto: sul manifest synth-v4-candidato-2 e sugli shard v4 (JSON letti con gli sha256 del manifest), senza modelli:
  QA function/exists del pool per insieme, split, stato e risposta; tabella etichetta x risposta; quelle che il loader
  mette nei prompt (f1c.pick_questions sui record ammessi, come data.synth_qa_sample); esito della regola testuale;
  record toccati e QA che restano; scelta del loader col filtro (prompt cambiati); misura storica di f1c_eval (tutte le
  QA delle 300 scene di test); per le altre domande di presenza (coord, polygon, shapes exists; shapes has_type)
  vocabolari di sì e no, regola per etichetta e contesto del prompt; effetto del filtro sui prompt revisionati di
  campione-2.
- diagnostica: generatore v5 su chiavi diagnostiche, fuori da ogni manifest: tabelle etichetta x risposta di v4 e v5
  prima e dopo la scelta del loader (data.pick_synth_questions), confronto con v4 alla stessa chiave, e un piccolo
  insieme disegnato (stessa domanda, immagini diverse, risposte opposte, per ogni etichetta) controllato da
  limite_vl.synth_check sul layout.
- deriva: synth-v4-candidato-3 da candidato-2 (immutato): stati e motivi dalla proposta della revisione (riammissioni,
  riesame tecnico se le sue evidenze sono confermate, irrisolto e contaminazioni invariati), evidenza per record, e i
  compiti per famiglia e tipo con function/exists escluso (politica_domande). Record, gruppi, split, immagini, JSON e QA
  invariati.

Funzioni pure usate anche da scripts/manifest_check.py: policy_checks, review_checks, expected_review_change.

    srun $RUN_PYTHON -u -B scripts/synth_qa_policy.py impatto --manifest DIR --campione DIR --out DIR
    srun $RUN_PYTHON -u -B scripts/synth_qa_policy.py diagnostica --out DIR [--chiavi 4000 --per-cella 6]
    srun $RUN_PYTHON -u -B scripts/synth_qa_policy.py deriva --parent DIR SHA256 --proposta TSV --riesame TSV SHA256 \
        --integrazione REPORT [--accetta-riesame] --out DIR
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.data import iter_shard, pick_synth_questions, synth_task  # noqa: E402
from limite_vl.f1c import pick_questions, qa_messages  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, sha256_file, write_new_json  # noqa: E402

SCHEMA_IMPACT = "synth-scorciatoia-exists/1"
SCHEMA_DIAG = "synth-v5-diagnostica/1"
POLICY_VERSION = "synth-domande/1"
BLOCKED = {"function/exists": "scorciatoia testuale: in v3 e v4 i grafici presenti si chiamano f o g e gli assenti h, p "
                              "o q, quindi il nome chiesto dà la risposta (f, g -> yes; h, p, q -> no) senza guardare "
                              "l'immagine. Le risposte sono vere: si escludono entrambe le polarità (S2 §2.14)"}
REVIEW = "revisione_synth_v4_campione_2"
REVIEW_VERSION = "synth-revisione/1"
REVIEW_FIELD = "revisione_umana"  # blocco della derivazione dalla revisione nel meta ("revisione" è la revisione della sorgente)
TECH_STATUS = "risolvibile_col_riesame_tecnico"
READMIT = "riammissione_proposta"
READMIT_TECH = "riammissione_dopo_riesame_tecnico"
PROPOSAL_COLUMNS = ("record", "record_id", "insieme", "split", "gruppo", "famiglia", "coppia", "stato_attuale",
                    "esito_proprio_decontaminazione", "esito_umano", "riesame_tecnico", "caso_quarantena",
                    "stato_proposto", "stato_dopo_riesame_tecnico", "categoria", "motivi", "motivi_rimovibili",
                    "blocchi_rimanenti")
SEVERITY = {"ammesso": 0, "quarantena": 1, "escluso": 2}
PRESENCE = (("function", "exists"), ("coord", "exists"), ("polygon", "exists"), ("shapes", "exists"),
            ("shapes", "has_type"))
ASKED = {"exists": re.compile(r"labeled (\w+)\?$"), "has_type": re.compile(r"^Is there an? (\w+) in the figure\?$")}
COUNT_TYPE = re.compile(r"^How many (\w+?)s are in the figure\?$")
V4_PRESENT = ("f", "g")  # v3, v4: i nomi dei grafici presenti; gli assenti sono h, p, q


# ---------------------------------------------------------------- utilità pure (collaudate in tests/test_synth_policy.py)

def asked_label(q: dict) -> str | None:
    """Etichetta (exists) o tipo di figura (has_type) chiesti da una domanda di presenza."""
    m = ASKED.get(q["type"]) and ASKED[q["type"]].search(q["q"])
    return m.group(1) if m else None


def text_rule(label: str) -> str:
    """La regola testuale di v3 e v4 per function/exists: f o g -> yes, altrimenti no."""
    return "yes" if label in V4_PRESENT else "no"


def mentioned(label: str, family: str, texts: list[str]) -> bool:
    """L'etichetta compare in uno dei testi (domande che presuppongono la presenza di ciò che nominano): i nomi dei grafici
    come parola di una lettera minuscola, i punti e le figure come lettera di una parola maiuscola (side AB, ABC)."""
    if family == "function":
        return any(label in re.findall(r"\b[a-z]\b", t) for t in texts)
    return any(label in tok for t in texts for tok in re.findall(r"\b[A-Z]+\b", t))


def pool_of(meta: dict) -> list[dict]:
    """Il pool del record come lo registra il manifest: le QA e, per una coppia, la sua domanda."""
    return list(meta["qa"]) + ([meta["pair"]] if meta.get("pair") else [])


def label_table(rows) -> dict:
    """{etichetta: {yes, no}} da coppie (etichetta, risposta), con vocabolari e regola per etichetta: accuratezza della
    risposta più frequente per etichetta (1 = l'etichetta determina la risposta)."""
    t: dict[str, Counter] = defaultdict(Counter)
    for label, answer in rows:
        t[label][answer] += 1
    total = sum(sum(c.values()) for c in t.values())
    best = sum(max(c.values()) for c in t.values())
    yes = sorted(k for k, c in t.items() if c["yes"])
    no = sorted(k for k, c in t.items() if c["no"])
    return {"tabella": {k: dict(sorted(t[k].items())) for k in sorted(t)}, "totale": total,
            "regola_per_etichetta": {"corrette": best, "accuratezza": round(best / total, 4) if total else None},
            "vocabolario_si": yes, "vocabolario_no": no, "comuni": sorted(set(yes) & set(no)),
            "solo_si": sorted(set(yes) - set(no)), "solo_no": sorted(set(no) - set(yes))}


def compiti_of(record: dict, blocked: dict = BLOCKED) -> list[dict]:
    """Compiti di un record per la politica: uno per famiglia e tipo delle sue QA (registrate in r["qa"]), con il numero
    di domande; escluso se il compito è fra i bloccati, altrimenti ammesso (lo stato del record resta a parte)."""
    n = Counter(synth_task(record["famiglia"], q["tipo"]) for q in record["qa"])
    return [{"compito": t, "stato": "escluso" if t in blocked else "ammesso", "domande": n[t],
             **({"motivo": "scorciatoia_testuale"} if t in blocked else {})} for t in sorted(n)]


def admitted_pool(record: dict, pool: list[dict]) -> list[dict]:
    """Le domande del pool dei compiti ammessi dal record (indipendente da data.synth_qa_sample)."""
    states = {c["compito"]: c["stato"] for c in record["compiti"]}
    return [q for q in pool if states.get(synth_task(record["famiglia"], q["type"])) == "ammesso"]


def parse_motivi(cell: str) -> dict[str, str]:
    return dict(x.split("=", 1) for x in cell.split("; ") if x) if cell else {}


def unescape(cell: str) -> str:
    out, i = [], 0
    while i < len(cell):
        if cell[i] == "\\" and i + 1 < len(cell):
            out.append({"n": "\n", "t": "\t", "\\": "\\"}.get(cell[i + 1], cell[i + 1]))
            i += 2
        else:
            out.append(cell[i])
            i += 1
    return "".join(out)


def read_tsv(text: str, columns: tuple[str, ...] | None = None) -> list[dict]:
    lines = text.split("\n")
    head = lines[0].split("\t")
    if columns is not None and tuple(head) != columns:
        raise ValueError(f"intestazione {head[:4]}... diversa da quella attesa")
    rows = []
    for line in lines[1:]:
        if line:
            cells = line.split("\t")
            if len(cells) != len(head):
                raise ValueError(f"riga con {len(cells)} campi invece di {len(head)}")
            rows.append({k: unescape(v) for k, v in zip(head, cells)})
    return rows


def expected_review_change(parent: dict, row: dict | None, accepted: bool) -> tuple[str, dict, dict]:
    """(stato, motivo, motivi risolti) di un record di candidato-3 dalla riga della proposta (S2 §2.13). Si tolgono i
    motivi rimovibili della riga e, se il riesame tecnico è accettato, quelli risolvibili col riesame; lo stato diventa
    ammesso solo per le riammissioni (quelle dopo il riesame solo se accettato) e solo senza altri motivi. Senza riga il
    record non cambia."""
    motivo = dict(parent.get("motivo") or {})
    if row is None:
        return parent["stato"], motivo, {}
    removable = {x for x in row["motivi_rimovibili"].split(",") if x}
    if accepted:
        removable |= {k for k, s in parse_motivi(row["motivi"]).items() if s == TECH_STATUS}
    solved = {k: v for k, v in motivo.items() if k in removable}
    left = {k: v for k, v in motivo.items() if k not in removable}
    readmit = row["categoria"] == READMIT or (row["categoria"] == READMIT_TECH and accepted)
    state = "ammesso" if readmit and not left else parent["stato"]
    return state, left, solved


# ---------------------------------------------------------------- verifiche pure per manifest_check.py

def policy_checks(meta: dict, records: list[dict], check) -> None:
    """Politica delle domande di un manifest dei sintetici: versione, compiti bloccati e compiti di ogni record."""
    pol = meta.get(mf.POLICY_FIELD) or {}
    check("politica delle domande: versione e compiti bloccati dichiarati (function/exists, escluso)",
          pol.get("versione") == POLICY_VERSION and sorted(pol.get("compiti_bloccati", {})) == sorted(BLOCKED),
          {"versione": pol.get("versione"), "bloccati": sorted(pol.get("compiti_bloccati", {}))})
    bad = [r["record"] for r in records if r.get("compiti") != compiti_of(r)]
    check("politica delle domande: ogni record ha un compito per famiglia e tipo delle sue QA, col numero di domande; "
          "escluso solo function/exists", not bad, {"casi": len(bad), "esempi": bad[:3]})
    empty = [r["record"] for r in records if r["stato"] == "ammesso"
             and not any(c["stato"] == "ammesso" and c["domande"] for c in r.get("compiti") or [])]
    check("politica delle domande: record ammessi senza domande di compiti ammessi (il reader li salta col motivo "
          "nessun_compito_ammesso)", True, len(empty))


def read_proposal(path: Path) -> dict[str, dict]:
    rows = read_tsv(path.read_text(encoding="utf-8"), PROPOSAL_COLUMNS)
    out = {r["record"]: r for r in rows}
    if len(out) != len(rows):
        raise mf.ManifestError(f"{path}: record ripetuti nella proposta")
    return out


def review_checks(meta: dict, records: list[dict], parent_records: list[dict], check) -> None:
    """Derivazione dalla revisione (meta[REVIEW_FIELD]) rispetto al genitore immediato: proposta e riesame con gli sha256
    registrati; ogni record della proposta col genitore e l'identità della riga; stato e motivi del record come dice la
    regola (e come dicono le colonne della proposta); evidenze del genitore conservate e una sola evidenza nuova, della
    revisione, per i record della proposta; gli altri record senza cambi; le riammissioni sono esattamente quelle
    previste; QA e immagini invariate."""
    rev = meta[REVIEW_FIELD]
    paths = {k: ROOT / rev[k]["percorso"] for k in ("proposta", "riesame_tecnico")}
    same = {k: p.is_file() and sha256_file(p) == rev[k]["sha256"] for k, p in paths.items()}
    check("revisione: proposta e riesame tecnico registrati presenti con gli sha256 del manifest", all(same.values()),
          same)
    if not all(same.values()):
        return
    rows = read_proposal(paths["proposta"])
    accepted = bool(rev["riesame_tecnico"].get("accettato"))
    old = {r["record"]: r for r in parent_records}
    issues: dict[str, list] = defaultdict(list)
    improved, expected_up = [], []
    for r in records:
        p = old.get(r["record"])
        if p is None:
            issues["record senza genitore"].append(r["record"])
            continue
        row = rows.get(r["record"])
        ev, pev = r.get("evidenze") or [], p.get("evidenze") or []
        if ev[:len(pev)] != pev:
            issues["evidenze del genitore non conservate"].append(r["record"])
        new = ev[len(pev):]
        if any(r[f] != p[f] for f in ("qa", "immagini", "json_sha256", "stato_scena_sha256", "coppia")):
            issues["QA, immagini o JSON cambiati"].append(r["record"])
        if SEVERITY[r["stato"]] < SEVERITY[p["stato"]]:
            improved.append(r["record"])
        if row is None:
            if new or r["stato"] != p["stato"] or (r.get("motivo") or {}) != (p.get("motivo") or {}):
                issues["cambi senza riga della proposta"].append(r["record"])
            continue
        if any(row[f] != str(p[f] or "") for f in ("record_id", "insieme", "split", "gruppo", "famiglia")) \
                or row["stato_attuale"] != p["stato"]:
            issues["riga della proposta diversa dal genitore"].append(r["record"])
        state, left, solved = expected_review_change(p, row, accepted)
        if state != p["stato"] and SEVERITY[state] < SEVERITY[p["stato"]]:
            expected_up.append(r["record"])
        column = row["stato_dopo_riesame_tecnico"] if accepted else row["stato_proposto"]
        if r["stato"] != state or r["stato"] != column:
            issues["stato diverso dalla regola o dalla colonna della proposta"].append(r["record"])
        if (r.get("motivo") or {}) != left:
            issues["motivi diversi da quelli del genitore meno i risolti"].append(r["record"])
        if (r["stato"] == "ammesso") == bool(left):
            issues["stato ammesso con motivi, o non ammesso senza motivi"].append(r["record"])
        if len(new) != 1 or new[0].get("controllo") != REVIEW or new[0].get("categoria") != row["categoria"] \
                or new[0].get("motivi_risolti") != solved or new[0].get("stato") != r["stato"] \
                or new[0].get("stato_genitore") != p["stato"]:
            issues["evidenza della revisione mancante o diversa"].append(r["record"])
    missing = sorted(set(rows) - set(old))
    check("revisione: ogni riga della proposta è un record del genitore", not missing, missing[:3])
    for name in ("record senza genitore", "evidenze del genitore non conservate", "QA, immagini o JSON cambiati",
                 "cambi senza riga della proposta", "riga della proposta diversa dal genitore",
                 "stato diverso dalla regola o dalla colonna della proposta",
                 "motivi diversi da quelli del genitore meno i risolti",
                 "stato ammesso con motivi, o non ammesso senza motivi", "evidenza della revisione mancante o diversa"):
        check(f"revisione: {name}", not issues[name], {"casi": len(issues[name]), "esempi": issues[name][:3]})
    check("revisione: i record che passano a uno stato meno grave sono esattamente le riammissioni previste",
          sorted(improved) == sorted(expected_up), {"migliorati": len(improved), "previsti": len(expected_up)})


# ---------------------------------------------------------------- lettura degli shard

def _read_shard(task: tuple[str, dict]) -> tuple[list, int]:
    """(chiave, sottoinsieme del JSON) dei record attesi di uno shard, col JSON verificato contro il manifest."""
    rel, expected = task
    out, bad = [], 0
    for key, files in iter_shard(ROOT / rel):
        if key not in expected:
            continue
        raw = files["json"]
        if hashlib.sha256(raw).hexdigest() != expected[key]:
            bad += 1
            continue
        meta = json.loads(raw)
        names = [c["name"] for c in meta["spec"].get("curves", [])] if meta["family"] == "function" else []
        out.append((key, {"family": meta["family"], "qa": meta["qa"], "pair": meta.get("pair"), "curve": names}))
    return out, bad


def read_jsons(m: mf.Manifest, workers: int) -> tuple[dict[str, dict], int]:
    tasks = [(rel, {k: r["json_sha256"] for k, r in recs.items()}) for rel, recs in sorted(m.by_shard.items())]
    with Pool(max(1, min(workers, len(tasks)))) as pool:
        parts = pool.map(_read_shard, tasks)
    metas = {k: v for part, _ in parts for k, v in part}
    return metas, sum(b for _, b in parts)


def stratum(r: dict) -> str:
    """train, dev, test o coppie per un record ammesso; insieme/split/stato per gli altri."""
    if r["stato"] != "ammesso":
        return f"{r['insieme']}/{r['split']}/{r['stato']}"
    return r["insieme"] if r["insieme"] != "train" else r["split"]


def qa_key(q: dict) -> str:
    return mf.sha256_text(mf.canonical({k: q[k] for k in ("type", "q", "options", "answer")}))


# ---------------------------------------------------------------- impatto

def presence_stats(selected: list[tuple[str, list[dict], int]]) -> dict:
    """Contesto dei prompt per le domande di presenza scelte: (famiglia, scelte, indice della domanda di presenza)."""
    out: dict[str, Counter] = defaultdict(Counter)
    for fam, picked, i in selected:
        q = picked[i]
        label, answer = asked_label(q), q["answer"]
        name = f"{fam}/{q['type']}"
        c = out[name]
        c["scelte"] += 1
        others = [x for j, x in enumerate(picked) if j != i]
        if not others:
            c["da_sole"] += 1
            continue
        c["con_altre_domande"] += 1
        presupposing = [x["q"] for x in others if (fam, x["type"]) not in PRESENCE]
        if q["type"] == "exists":
            hit = mentioned(label, fam, presupposing)
            c[f"etichetta_nominata_da_altre/{answer}"] += hit
            c["regola_contesto_corrette"] += (answer == "yes") == hit
            if any(x["type"] == "exists" for x in others):
                c["con_l_altra_exists_nello_stesso_prompt"] += 1
        else:
            same = [j for j, x in enumerate(picked) if x["type"] == "count_type" and COUNT_TYPE.search(x["q"])
                    and COUNT_TYPE.search(x["q"]).group(1) == label]
            c[f"count_type_dello_stesso_tipo/{answer}"] += bool(same)
            c[f"count_type_dello_stesso_tipo_prima/{answer}"] += any(j < i for j in same)
            if any(x["type"] == "has_type" for x in others):
                c["con_l_altra_has_type_nello_stesso_prompt"] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def cmd_impatto(args) -> int:
    t0 = time.time()
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)
    m = mf.load(args.manifest)
    if m.manifest_sha256 != args.manifest_sha256:
        sys.exit(f"manifest.json ha sha256 {m.manifest_sha256}, atteso {args.manifest_sha256}")
    records = {k: r for shard in m.by_shard.values() for k, r in shard.items()}
    metas, bad = read_jsons(m, args.workers)
    checks = {"json_letti_con_lo_sha256_del_manifest": len(metas) == len(records) and bad == 0,
              "json_letti": len(metas), "json_diversi": bad}
    pool_fe = defaultdict(Counter)  # strato -> risposta, per function/exists
    labels_all, labels_adm = [], []  # (etichetta, risposta) function/exists
    rule_pool = Counter()
    sel_fe = defaultdict(Counter)
    labels_sel = []
    rule_sel = Counter()
    presence_pool: dict[str, list] = defaultdict(list)
    presence_sel: dict[str, list] = defaultdict(list)
    context: list = []
    func = defaultdict(Counter)  # strato -> conteggi sui record function ammessi
    rest_hist = Counter()
    changed = defaultdict(Counter)
    types_before, types_after = Counter(), Counter()
    f1c_test = Counter()
    for key, r in records.items():
        meta = metas.get(key)
        if meta is None:
            continue
        fam, st = r["famiglia"], stratum(r)
        pool = pool_of(meta)
        if [qa_key(q) for q in pool] != [q["sha256"] for q in r["qa"]]:
            checks.setdefault("pool_diverso_dai_digest", []).append(key)
        for q in pool:
            if (fam, q["type"]) in PRESENCE:
                lab = asked_label(q)
                presence_pool[f"{fam}/{q['type']}"].append((lab, q["answer"]))
                if (fam, q["type"]) == ("function", "exists"):
                    pool_fe[st][q["answer"]] += 1
                    labels_all.append((lab, q["answer"]))
                    rule_pool["corrette"] += text_rule(lab) == q["answer"]
                    rule_pool["totale"] += 1
                    if r["stato"] == "ammesso":
                        labels_adm.append((lab, q["answer"]))
        if r["insieme"] == "test":  # f1c_eval (storico) valuta ogni QA delle 300 scene di test, senza manifest
            for q in meta["qa"]:
                f1c_test["qa"] += 1
                if (fam, q["type"]) == ("function", "exists"):
                    f1c_test[f"function_exists/{q['answer']}"] += 1
                    f1c_test["regola_testuale_corrette"] += text_rule(asked_label(q)) == q["answer"]
        if r["stato"] != "ammesso":
            continue
        old = pick_questions(key, pool)
        for i, q in enumerate(old):
            if (fam, q["type"]) in PRESENCE:
                presence_sel[f"{fam}/{q['type']}"].append((asked_label(q), q["answer"]))
                context.append((fam, old, i))
            if (fam, q["type"]) == ("function", "exists"):
                sel_fe[st][q["answer"]] += 1
                labels_sel.append((asked_label(q), q["answer"]))
                rule_sel["corrette"] += text_rule(asked_label(q)) == q["answer"]
                rule_sel["totale"] += 1
        kept = [q for q in pool if synth_task(fam, q["type"]) not in BLOCKED]
        new = pick_synth_questions(key, kept) if kept else []
        changed[st][f"{fam}/{'cambiato' if [qa_key(q) for q in new] != [qa_key(q) for q in old] else 'uguale'}"] += 1
        if fam == "function":
            c = func[st]
            c["record"] += 1
            c["qa_exists_tolte"] += len(pool) - len(kept)
            c["qa_che_restano"] += len(kept)
            c["prompt_con_exists_prima"] += any(q["type"] == "exists" for q in old)
            c["record_senza_domande_dopo"] += not kept
            rest_hist[len(kept)] += 1
            types_before.update(q["type"] for q in old)
            types_after.update(q["type"] for q in new)
    checks["pool_uguale_ai_digest_del_manifest"] = "pool_diverso_dai_digest" not in checks
    campione = review_effect(args.campione, records, metas)
    checks["prompt_storici_di_campione_2_riprodotti"] = (
        campione["unita_col_prompt_del_loader"] > 0
        and campione["prompt_storici_riprodotti"] == campione["unita_col_prompt_del_loader"])
    report = {
        "schema": SCHEMA_IMPACT, "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
        "manifest": {"cartella": str(args.manifest), "manifest_sha256": m.manifest_sha256,
                     "records_sha256": m.meta["records"]["sha256"]},
        "controlli": checks,
        "function_exists": {
            "pool_per_strato_e_risposta": {k: dict(v) for k, v in sorted(pool_fe.items())},
            "etichetta_per_risposta_tutti_i_record": label_table(labels_all),
            "etichetta_per_risposta_record_ammessi": label_table(labels_adm),
            "regola_testuale_sul_pool": dict(rule_pool),
            "scelte_dal_loader_per_strato_e_risposta": {k: dict(v) for k, v in sorted(sel_fe.items())},
            "etichetta_per_risposta_scelte": label_table(labels_sel),
            "regola_testuale_sulle_scelte": dict(rule_sel),
            "record_function_ammessi": {k: dict(v) for k, v in sorted(func.items())},
            "qa_che_restano_per_record": dict(sorted(rest_hist.items())),
            "tipi_scelti_nei_function_prima": dict(sorted(types_before.items())),
            "tipi_scelti_nei_function_dopo": dict(sorted(types_after.items())),
            "f1c_eval_storico_sul_test": dict(f1c_test)},
        "scelta_del_loader_col_filtro": {k: dict(v) for k, v in sorted(changed.items())},
        "altre_domande_di_presenza": {
            "pool": {k: label_table(v) for k, v in sorted(presence_pool.items())},
            "scelte": {k: label_table(v) for k, v in sorted(presence_sel.items())},
            "contesto_dei_prompt": presence_stats(context)},
        "campione_2": campione,
        "secondi": round(time.time() - t0, 1)}
    write_new_json(args.out / "report.json", report)
    if args.docs:
        light = {k: v for k, v in report.items()}
        light["campione_2"] = {k: v for k, v in campione.items() if k != "unita"}
        write_new_json(args.docs, light)
    ok = checks["json_letti_con_lo_sha256_del_manifest"] and checks["pool_uguale_ai_digest_del_manifest"] \
        and checks["prompt_storici_di_campione_2_riprodotti"]
    print(json.dumps({"function_exists": report["function_exists"]["regola_testuale_sul_pool"],
                      "scelte": report["function_exists"]["regola_testuale_sulle_scelte"], "ok": ok,
                      "secondi": report["secondi"]}, ensure_ascii=False), flush=True)
    return 0 if ok else 1


def review_effect(folder: Path, records: dict, metas: dict) -> dict:
    """Prompt dei record di campione-2 col filtro, confrontati con quelli revisionati (campione.tsv, mirati.tsv)."""
    units, table = [], Counter()
    for name in ("campione", "mirati"):
        for u in read_tsv((folder / f"{name}.tsv").read_text(encoding="utf-8")):
            key = u["record"]
            meta, r = metas.get(key), records.get(key)
            if meta is None or r is None:
                table[f"{name}/record_assente"] += 1
                continue
            fam = r["famiglia"]
            pool = pool_of(meta)
            old_prompt = qa_messages(pick_questions(key, pool))[0]
            kept = [q for q in pool if synth_task(fam, q["type"]) not in BLOCKED]
            new_prompt = qa_messages(pick_synth_questions(key, kept))[0] if kept else None
            stored, origin = u.get("prompt", ""), u.get("origine_prompt", "")
            from_loader = origin in ("loader", "loader_scena")  # le altre unità mostrano una domanda singola
            reproduced = from_loader and stored == old_prompt
            blocked_unit = synth_task(fam, u.get("tipo", "")) in BLOCKED
            outcome = ("domanda_bloccata" if blocked_unit else "domanda_singola" if not from_loader
                       else "prompt_uguale" if new_prompt == old_prompt else "prompt_cambiato")
            table[f"{name}/{fam}/{origin}/{outcome}"] += 1
            units.append({"sezione": name, "unita": u["unita"], "record": key, "famiglia": fam, "tipo": u.get("tipo"),
                          "origine_prompt": origin, "prompt_storico_riprodotto": reproduced, "esito": outcome})
    loader = [x for x in units if x["origine_prompt"] in ("loader", "loader_scena")]
    return {"unita": units, "per_sezione_famiglia_prompt_esito": dict(sorted(table.items())),
            "unita_col_prompt_del_loader": len(loader),
            "prompt_storici_riprodotti": sum(x["prompt_storico_riprodotto"] for x in loader),
            "unita_function": sum(x["famiglia"] == "function" for x in units),
            "unita_con_domanda_bloccata": sum(x["esito"] == "domanda_bloccata" for x in units),
            "unita_col_prompt_cambiato": sum(x["esito"] == "prompt_cambiato" for x in units),
            "nota": "la revisione umana ha giudicato i prompt storici: col filtro i record function hanno un'altra "
                    "scelta di domande, che quel giudizio non copre"}


# ---------------------------------------------------------------- diagnostica v5

def _strip(spec: dict) -> dict:
    s = json.loads(json.dumps(spec))
    s.pop("generator_version", None)
    for c in s.get("curves", []):
        c.pop("name", None)
    return s


def _exists_q(qs: list[dict]) -> list[dict]:
    return [q for q in qs if q["type"] == "exists"]


def cmd_diagnostica(args) -> int:
    from limite_vl import synth, synth_check

    t0 = time.time()
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota")
    args.out.mkdir(parents=True, exist_ok=True)
    checks: dict[str, object] = {}
    keys = [f"synth-v5-diag-{i:06d}" for i in range(args.chiavi)]
    v4_pool, v5_pool, v4_sel, v5_sel = [], [], [], []
    by_curves, alone, with_others, geometry_diff, other_diff = Counter(), 0, 0, [], []
    examples_with_exists = 0
    for key in keys:
        s4, s5 = synth.make_spec(key, "function", 4), synth.make_spec(key, "function", 5)
        q4, q5 = synth.questions(s4), synth.questions(s5)
        if _strip(s4) != _strip(s5):
            geometry_diff.append(key)
        same = Counter((q["type"], tuple(q["options"]), q["answer"]) for q in q4 if q["type"] != "exists") == \
            Counter((q["type"], tuple(q["options"]), q["answer"]) for q in q5 if q["type"] != "exists")
        names5 = {c["name"] for c in s5["curves"]}
        # lettere minuscole isolate delle domande: nomi dei grafici, più l'articolo e gli assi (x-axis, y-axis)
        named = all(set(re.findall(r"\b[a-z]\b", q["q"])) - {"a", "x", "y"} <= names5
                    for q in q5 if q["type"] != "exists")
        if not (same and named):
            other_diff.append(key)
        v4_pool += [(asked_label(q), q["answer"]) for q in _exists_q(q4)]
        e5 = _exists_q(q5)
        if len(e5) != 1 or not e5[0].get(synth.SOLO_FIELD):
            checks.setdefault("v5_senza_una_sola_exists_da_sola", []).append(key)
        v5_pool += [(asked_label(q), q["answer"]) for q in e5]
        by_curves[f"{len(s5['curves'])}/{e5[0]['answer'] if e5 else 'nessuna'}"] += 1
        v4_sel += [(asked_label(q), q["answer"]) for q in _exists_q(pick_questions(key, q4))]
        p5 = pick_synth_questions(key, q5)
        if _exists_q(p5):
            examples_with_exists += 1
            alone += len(p5) == 1
            with_others += len(p5) > 1
        v5_sel += [(asked_label(q), q["answer"]) for q in _exists_q(p5)]
    checks["v5_una_sola_exists_da_sola_per_scena"] = "v5_senza_una_sola_exists_da_sola" not in checks
    checks["v5_stessa_geometria_di_v4_a_parte_i_nomi"] = not geometry_diff
    checks["v5_stesse_altre_domande_di_v4_e_nomi_dei_grafici_disegnati"] = not other_diff
    checks["v5_exists_scelta_sempre_da_sola"] = with_others == 0
    fam_same = {}
    for fam in ("coord", "polygon", "shapes"):
        ks = [f"synth-v5-diag-{fam}-{i:04d}" for i in range(args.altre)]
        fam_same[fam] = all(_strip(synth.make_spec(k, fam, 4)) == _strip(synth.make_spec(k, fam, 5))
                            and synth.questions(synth.make_spec(k, fam, 4)) == synth.questions(synth.make_spec(k, fam, 5))
                            for k in ks)
    checks["v5_altre_famiglie_uguali_a_v4"] = fam_same
    # insieme disegnato: per ogni etichetta e risposta, per_cella scene con la domanda exists su quell'etichetta
    cells: dict[tuple[str, str], list[str]] = {(n, a): [] for n in synth.FUNCTION_NAMES for a in ("yes", "no")}
    i = 0
    while any(len(v) < args.per_cella for v in cells.values()) and i < 200000:
        key = f"synth-v5-diag-disegno-{i:06d}"
        i += 1
        e = _exists_q(synth.questions(synth.make_spec(key, "function", 5)))[0]
        cell = (asked_label(e), e["answer"])
        if len(cells[cell]) < args.per_cella:
            cells[cell].append(key)
    rows, statuses, problems = [], Counter(), []
    tar_path = args.out / "diagnostica.tar"
    with tarfile.open(tar_path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for (name, answer), ks in sorted(cells.items()):
            for key in ks:
                spec = synth.make_spec(key, "function", 5)
                qs = synth.questions(spec)
                image, layout = synth.render(spec, layout=True)
                png = synth.png_bytes(image)
                out = synth_check.check_sample({"spec": spec, "qa": qs}, layout)
                for q in out["questions"]:
                    statuses[f"{q['type']}/{q['status']}"] += 1
                # la risposta di exists dal contenuto disegnato (etichette del layout), a parte le violazioni del renderer
                e = next(q for q in synth_check.check_questions(spec, qs, layout) if q["type"] == "exists")
                if e["status"] != "ok" or e["expected"] != answer:
                    problems.append(key)
                meta = {"id": key, "gruppo": key, "family": "function", "spec": spec, "qa": qs,
                        "png_sha256": hashlib.sha256(png).hexdigest(), "verifica": out}
                for ext, data in (("png", png), ("json", json.dumps(meta, ensure_ascii=False).encode())):
                    info = tarfile.TarInfo(f"{key}.{ext}")
                    info.size, info.mtime = len(data), 0
                    tar.addfile(info, io.BytesIO(data))
                rows.append({"chiave": key, "etichetta": name, "risposta": answer,
                             "domanda": next(q["q"] for q in qs if q["type"] == "exists"),
                             "grafici": [c["name"] for c in spec["curves"]], "verificata": out["verified"],
                             "violazioni": sorted({v["kind"] for v in out["violations"]})})
    texts = defaultdict(set)
    for r in rows:
        texts[r["etichetta"]].add(r["domanda"])
    checks["insieme_disegnato_celle_complete"] = all(len(v) == args.per_cella for v in cells.values())
    checks["insieme_disegnato_stessa_domanda_per_etichetta_con_risposte_opposte"] = all(
        len(v) == 1 for v in texts.values()) and len(texts) == len(synth.FUNCTION_NAMES)
    checks["insieme_disegnato_exists_verificata_dal_contenuto_disegnato"] = not problems
    report = {"schema": SCHEMA_DIAG, "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "generatore": {"versioni": list(synth.VERSIONS), "default": synth.GENERATOR_VERSION,
                             "nomi_dei_grafici_v5": list(synth.FUNCTION_NAMES)},
              "fuori_dai_manifest": "chiavi diagnostiche synth-v5-diag-*: nessun manifest, nessuno split; ogni scena "
                                    "è il suo gruppo",
              "controlli": checks,
              "scene_non_disegnate": {"chiavi": args.chiavi,
                                      "v4_pool": label_table(v4_pool), "v4_scelte_dal_loader": label_table(v4_sel),
                                      "v5_pool": label_table(v5_pool), "v5_scelte_dal_loader": label_table(v5_sel),
                                      "v5_grafici_per_risposta": dict(sorted(by_curves.items())),
                                      "v5_esempi_con_exists": examples_with_exists, "v5_exists_da_sola": alone,
                                      "v5_exists_con_altre_domande": with_others,
                                      "differenze_di_geometria": geometry_diff[:5],
                                      "differenze_nelle_altre_domande": other_diff[:5]},
              "insieme_disegnato": {"per_cella": args.per_cella, "scene": len(rows),
                                    "tar": file_entry(tar_path), "esiti_delle_domande": dict(sorted(statuses.items())),
                                    "exists_non_verificate": problems, "domande_per_etichetta": {
                                        k: sorted(v) for k, v in sorted(texts.items())},
                                    "scene": rows},
              "secondi": round(time.time() - t0, 1)}
    write_new_json(args.out / "report.json", report)
    if args.docs:
        write_new_json(args.docs, {**report, "insieme_disegnato": {k: v for k, v in report["insieme_disegnato"].items()
                                                                   if k != "scene"}})
    ok = all(v if isinstance(v, bool) else all(v.values()) for k, v in checks.items() if not isinstance(v, list))
    print(json.dumps({"controlli": checks, "secondi": report["secondi"]}, ensure_ascii=False), flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------- deriva (synth-v4-candidato-3)

def cmd_deriva(args) -> int:
    from synth_manifest import gzip_lines

    t0 = time.time()
    if args.out.exists():
        sys.exit(f"{args.out} esiste già: una versione del manifest si scrive una volta sola")
    parent_dir, parent_sha = Path(args.parent[0]), args.parent[1]
    m = mf.load(parent_dir)
    if m.manifest_sha256 != parent_sha:
        sys.exit(f"{parent_dir}: manifest.json con sha256 {m.manifest_sha256}, atteso {parent_sha}")
    if mf.POLICY_FIELD in m.meta or m.meta.get(REVIEW_FIELD):
        sys.exit("il genitore ha già una politica delle domande o una revisione: derivazione non prevista")
    integ = json.loads(args.integrazione.read_text(encoding="utf-8"))
    tech_path, tech_sha = Path(args.riesame[0]), args.riesame[1]
    if sha256_file(tech_path) != tech_sha:
        sys.exit(f"{tech_path}: sha256 diverso da quello atteso")
    prop_entry = file_entry(args.proposta)
    confronto = integ.get("confronto_col_precedente") or {}
    if integ.get("falliti") or integ["ingressi"]["manifest"]["manifest_sha256"] != m.manifest_sha256 \
            or confronto.get("proposta_docs", {}).get("sha256") != prop_entry["sha256"] \
            or not confronto.get("proposta_docs", {}).get("uguale") \
            or (integ["ingressi"].get("riesame_tecnico") or {}).get("sha256") != tech_sha:
        sys.exit("l'integrazione indicata non conferma questa proposta, questo riesame o questo manifest")
    provenance_ok = [c for c in integ["controlli"] if c["controllo"].startswith("riesame: ingresso")]
    material_ok = [c for c in integ["controlli"] if c["controllo"].startswith("riesame: materiale uguale")]
    confirmed = bool(provenance_ok) and all(c["esito"] for c in provenance_ok) and bool(material_ok) \
        and all(c["esito"] for c in material_ok)
    if args.accetta_riesame and not confirmed:
        sys.exit("riesame tecnico non accettabile: ingressi o materiale non confermati dall'integrazione")
    accepted = bool(args.accetta_riesame and confirmed)
    rows = read_proposal(args.proposta)
    records = [r for shard in m.by_shard.values() for r in shard.values()]
    by_key = {r["record"]: r for r in records}
    missing = sorted(set(rows) - set(by_key))
    if missing:
        sys.exit(f"righe della proposta senza record: {missing[:3]}")
    job = os.environ.get("SLURM_JOB_ID")
    new, changes = [], Counter()
    for r in records:
        row = rows.get(r["record"])
        if row is not None and (any(row[f] != str(r[f] or "") for f in ("record_id", "insieme", "split", "gruppo",
                                                                         "famiglia")) or row["stato_attuale"] != r["stato"]):
            sys.exit(f"{r['record']}: riga della proposta diversa dal record del genitore")
        state, left, solved = expected_review_change(r, row, accepted)
        rec = dict(r)
        rec["compiti"] = compiti_of(r)
        if row is not None:
            column = row["stato_dopo_riesame_tecnico"] if accepted else row["stato_proposto"]
            if state != column:
                sys.exit(f"{r['record']}: stato {state} dalla regola, la proposta dice {column}")
            rec["stato"], rec["motivo"] = state, left
            rec["evidenze"] = list(r["evidenze"] or []) + [{
                "controllo": REVIEW, "versione": REVIEW_VERSION, "categoria": row["categoria"],
                "esito_umano": row["esito_umano"] or None, "riesame_tecnico": row["riesame_tecnico"] or None,
                "caso_quarantena": row["caso_quarantena"] or None, "motivi_risolti": solved,
                "stato_genitore": r["stato"], "stato": state, "job": job,
                "riga_proposta_sha256": mf.sha256_text(mf.canonical(row))}]
            changes[f"{r['insieme']}/{r['split']}: {r['stato']} -> {state}"] += 1
        new.append(rec)
    lines = [json.dumps(x, sort_keys=True, ensure_ascii=False) for x in new]
    args.out.mkdir(parents=True)
    data = gzip_lines(lines)
    (args.out / "records.jsonl.gz").write_bytes(data)
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
    counts = {k: Counter() for k in ("per_insieme_split_stato", "per_split_famiglia_stato", "qa_per_split_famiglia_tipo",
                                     "qa_ammesse_per_split_famiglia_tipo", "compiti_per_split_stato_del_compito",
                                     "gruppi_per_split")}
    split_of = {}
    for x in new:
        counts["per_insieme_split_stato"][f"{x['insieme']}/{x['split']}/{x['stato']}"] += 1
        counts["per_split_famiglia_stato"][f"{x['split']}/{x['famiglia']}/{x['stato']}"] += 1
        split_of[x["gruppo"]] = x["split"]
        if x["stato"] == "ammesso":
            states = {c["compito"]: c["stato"] for c in x["compiti"]}
            for q in x["qa"]:
                counts["qa_per_split_famiglia_tipo"][f"{x['split']}/{x['famiglia']}/{q['tipo']}"] += 1
                if states[synth_task(x["famiglia"], q["tipo"])] == "ammesso":
                    counts["qa_ammesse_per_split_famiglia_tipo"][f"{x['split']}/{x['famiglia']}/{q['tipo']}"] += 1
            for c in x["compiti"]:
                counts["compiti_per_split_stato_del_compito"][f"{x['split']}/{c['stato']}/{c['compito']}"] += c["domande"]
    for s in split_of.values():
        counts["gruppi_per_split"][s] += 1
    dev_now = Counter(x["famiglia"] for x in new if x["split"] == "dev" and x["stato"] == "ammesso")
    meta0 = m.meta
    rel = lambda p: str(Path(p).resolve().relative_to(ROOT))  # noqa: E731
    meta = {k: v for k, v in meta0.items() if k != "decontaminazione"}  # la decontaminazione è della versione 2
    meta.update({
        "versione": args.version, "job_id": job, "codice": code_identity(),
        "precedente": {"percorso": rel(parent_dir), "versione": meta0["versione"], "manifest_sha256": m.manifest_sha256,
                       "records_sha256": meta0["records"]["sha256"]},
        "split": {**meta0["split"],
                  "ereditato_da": {"percorso": rel(parent_dir), "versione": meta0["versione"],
                                   "manifest_sha256": m.manifest_sha256},
                  "regola_versione": "split e gruppi della versione precedente, invariati: la revisione cambia solo "
                                     "stati, motivi ed evidenze dei record della proposta; la politica delle domande "
                                     "aggiunge i compiti",
                  "dev_ammessi_per_famiglia": dict(sorted(dev_now.items()))},
        "stati": {"ammesso": "record ammesso nel manifest candidato, non certificato: le sue domande si costruiscono solo "
                             "dai compiti ammessi (politica_domande)",
                  "quarantena": "testi sovrapposti o gruppo con un candidato di decontaminazione non risolto: da "
                                "rivedere",
                  "escluso": "contaminazione con la valutazione dei sintetici (S2 §2.6), oppure gruppo con pixel "
                             "identici a un'immagine del pannello"},
        mf.POLICY_FIELD: {
            "versione": POLICY_VERSION,
            "unita": "compito = famiglia/tipo di domanda (data.synth_task); ogni record registra i compiti delle sue QA "
                     "con stato e numero di domande. Ammissione del record e ammissione delle domande sono separate: un "
                     "record riammesso non riabilita un compito escluso",
            "compiti_bloccati": {t: {"stato": "escluso", "motivo": "scorciatoia_testuale", "dettaglio": d}
                                 for t, d in BLOCKED.items()},
            "applicazione": "manifest.TASKED_BY_POLICY: il tipo synth_qa legge train, test e coppie come sorgente per "
                            "compito; data.synth_qa_sample lega il pool al digest qa_sha256 e toglie le domande dei "
                            "compiti non ammessi prima di pick_synth_questions; un record senza domande ammesse si salta "
                            "(nessun_compito_ammesso), senza ripiegare su quelle bloccate",
            "effetto_sulla_scelta": "il pool più corto cambia numero e scelta delle altre domande dei record function: "
                                    "la revisione umana dei prompt storici non copre i prompt nuovi",
            "evidenze": {"impatto": {**file_entry(args.impatto), "percorso": rel(args.impatto)} if args.impatto else None}},
        REVIEW_FIELD: {
            "controllo": REVIEW, "versione": REVIEW_VERSION, "pacchetto": "revisioni/synth-v4-campione-2",
            "risultati": integ["ingressi"]["risultati"],
            "integrazione": {"rapporto": {**file_entry(args.integrazione), "percorso": rel(args.integrazione)},
                             "job": integ.get("job_id"), "codice": integ.get("codice")},
            "proposta": {"percorso": rel(args.proposta), "sha256": prop_entry["sha256"], "bytes": prop_entry["bytes"],
                         "righe": len(rows)},
            "riesame_tecnico": {"percorso": rel(tech_path), "sha256": tech_sha, "accettato": accepted,
                                "ingressi_confermati": confirmed,
                                "regola": "i motivi risolvibili col riesame tecnico si tolgono solo se accettato e se "
                                          "l'integrazione conferma ingressi e materiale del riesame"},
            "regole": {"motivi": "si tolgono i motivi rimovibili della riga della proposta e, col riesame accettato, "
                                 "quelli risolvibili col riesame; gli altri restano",
                       "stato": "ammesso solo per riammissione_proposta e, col riesame accettato, "
                                "riammissione_dopo_riesame_tecnico, senza altri motivi; gli altri record restano nello "
                                "stato del genitore",
                       "evidenza": "una per record della proposta (motivi risolti con i dettagli del genitore, categoria, "
                                   "esito umano, riesame, sha256 della riga)",
                       "invarianti": "record, gruppi, split, insiemi, immagini, JSON e QA del genitore"},
            "cambi_di_stato": dict(sorted(changes.items()))},
        "conteggi": {k: dict(sorted(v.items())) for k, v in counts.items()},
        "records": {"file": "records.jsonl.gz", "sha256": hashlib.sha256(data).hexdigest(), "righe": len(lines)},
        "derivati": derived,
        "fuori_dal_sottoinsieme": [
            x for x in meta0["fuori_dal_sottoinsieme"] if not x.startswith("ammissione della singola QA")] + [
            "ammissione per domanda: per compito (famiglia/tipo), non per singola QA; le QA di un compito escluso non si "
            "costruiscono"],
        "seconds": round(time.time() - t0, 1)})
    write_new_json(args.out / "manifest.json", meta)
    loaded = mf.load(args.out)
    for split in ("train", "dev", "test"):
        shards = [str(mf.shard_path(s["percorso"])) for s in meta["shards"]
                  if (s["insieme"] == "train") == (split != "test")]
        loaded.admission(shards, split, kind="synth_qa")
    print(json.dumps({"record": len(lines), "cambi_di_stato": meta[REVIEW_FIELD]["cambi_di_stato"],
                      "riesame_accettato": accepted, "seconds": meta["seconds"]}, ensure_ascii=False), flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("impatto")
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--manifest-sha256", required=True)
    a.add_argument("--campione", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--docs", type=Path)
    a.add_argument("--workers", type=int, default=4)
    d = sub.add_parser("diagnostica")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--docs", type=Path)
    d.add_argument("--chiavi", type=int, default=4000)
    d.add_argument("--altre", type=int, default=100)
    d.add_argument("--per-cella", type=int, default=6)
    v = sub.add_parser("deriva")
    v.add_argument("--parent", nargs=2, required=True, metavar=("DIR", "SHA256"))
    v.add_argument("--proposta", type=Path, required=True)
    v.add_argument("--riesame", nargs=2, required=True, metavar=("TSV", "SHA256"))
    v.add_argument("--integrazione", type=Path, required=True)
    v.add_argument("--impatto", type=Path)
    v.add_argument("--accetta-riesame", action="store_true")
    v.add_argument("--version", default="3")
    v.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("synth_qa_policy.py va eseguito in un job SLURM")
    sys.exit({"impatto": cmd_impatto, "diagnostica": cmd_diagnostica, "deriva": cmd_deriva}[args.cmd](args))


if __name__ == "__main__":
    main()
