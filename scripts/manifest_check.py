#!/usr/bin/env python3
"""Verifiche CPU del manifest delle ammissioni dei sintetici v4 e dei loader che lo applicano (S1 §§6, 8-9; S2 §2.7).
Solo in un job SLURM CPU; manifest e shard in sola lettura, scrive solo nella cartella nuova --out.

1. Integrità: manifest.load (sha256 di records.jsonl.gz, record_id ricalcolati, un solo split per gruppo, conteggi per
   shard); shard su disco con sha256 e dimensione del manifest; file derivati (keys/) con i loro sha256 e uguali alle
   chiavi del manifest.
2. Split: ricalcolato due volte con la regola, la quota e il seme del manifest (manifest.assign_splits) e uguale a
   quello scritto; con un altro seme è diverso. I legami esatti ricalcolati dagli hash del manifest (pixel, stato della
   scena, coppia) danno gli stessi gruppi; nessun gruppo è in due split; dev per famiglia rispetto alla quota.
   Una versione derivata (`split.ereditato_da`, es. dopo la decontaminazione esterna) ha due verifiche separate
   (verify_splits):
   - rispetto al genitore immediato: stessi record, gruppi, insiemi, split e posizioni; nessuno stato meno grave; con
     la decontaminazione, le evidenze e i motivi del genitore conservati, una sola evidenza nuova per record, ogni
     cambio di stato spiegato dal suo motivo e uguale al più grave fra lo stato del genitore e quello dato dall'esito
     del gruppo (nessuna riammissione);
   - rispetto all'assegnazione originale: la catena `ereditato_da` si segue fino alla versione che assegnò gli split
     (la prima senza `ereditato_da`), ogni anello letto con manifest.load e con lo sha256 di manifest.json registrato
     dal figlio (anche in `precedente`). La regola si ricalcola sui record di quella versione: sui record del genitore
     un gruppo di dev messo tutto in quarantena tornerebbe in train. Regola, identità e split restano gli stessi lungo
     tutta la catena, e nessun anello migliora uno stato.
3. Loader: sorgenti di stream.make_sources con il manifest per train e dev (shard train) e per test e coppie (split
   test). Ogni sorgente si legge per un'epoca con --workers worker di SourceReader: le chiavi consegnate devono essere
   esattamente quelle ammesse dello split, ciascuna una volta, nessuna esclusa o in quarantena, e i gruppi letti non
   devono comparire in altri split. Le domande costruite (f1c.pick_questions, come data.synth_qa_sample) hanno gli
   stessi digest e gli stessi conteggi per famiglia e compito attesi dal manifest. validation_samples restituisce lo
   split dev. Un breve MixedRowStream col processor di Violetto consegna solo chiavi di train e conta gli esempi.
   Con la politica delle domande (politica_domande, synth-v4-candidato-3; S2 §2.14) la sorgente è per compito: la
   scelta attesa è sul pool dei soli compiti ammessi, e ogni campione si costruisce col loader reale
   (data.synth_qa_sample sulla decisione allegata dal reader): prompt e target uguali alla scelta attesa, nessuna domanda
   di un compito escluso nel prompt, nessun yes/no nei target dei function, domande tolte e compiti bloccati uguali a
   quelli del manifest. Una versione dalla revisione (`revisione_umana`) riammette solo i record della proposta, con la sua
   evidenza (synth_qa_policy.review_checks); i compiti dei record seguono la politica (synth_qa_policy.policy_checks).
4. Arresti espliciti: manifest mancante, records.jsonl.gz alterato, sha256 di uno shard diverso, gruppo spostato in un
   altro split, record_id alterato, shard v3 col manifest v4, shard v4 senza manifest, split senza record ammessi,
   split sconosciuto, manifest senza split. Ognuno deve dare ManifestError; le copie alterate stanno in una cartella
   temporanea del job.

Uscite: --out/report.json con esiti e conteggi; il job esce con errore se un controllo fallisce.

    srun $RUN_PYTHON -u -B scripts/manifest_check.py --manifest DIR --shards DIR --out DIR [--workers 2]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.f1c import pick_questions  # noqa: E402
from limite_vl.provenance import code_identity, sha256_file, write_new_json  # noqa: E402

VIOLETTO = ROOT / "hf/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
SOURCES: dict = {}  # nome -> Source, ereditate dai processi del pool
RESULTS: list[dict] = []
SEVERITY = {"ammesso": 0, "quarantena": 1, "escluso": 2}
IDENTITY = ("record_id", "gruppo", "split", "insieme", "shard", "indice", "json_sha256")
RULE_FIELDS = ("valori", "quota_dev", "seme", "dev_per_famiglia")  # la regola degli split, ereditata invariata
MAX_CHAIN = 64


def check(name: str, ok: bool, detail=None) -> bool:
    RESULTS.append({"controllo": name, "esito": bool(ok), "dettaglio": detail})
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail is not None else ""), flush=True)
    return ok


# ---------------------------------------------------------------- versioni derivate e split (pure, collaudate in tests/)

def all_records(m: mf.Manifest) -> list[dict]:
    return [r for shard in m.by_shard.values() for r in shard.values()]


def chain_of(m: mf.Manifest) -> list[mf.Manifest]:
    """[m, genitore, ..., origine]: segue split.ereditato_da fino alla versione che assegnò gli split, la prima senza
    ereditato_da. Ogni anello è letto con manifest.load (sha256 e righe dei record, record_id, un solo split per
    gruppo); lo sha256 del suo manifest.json deve essere quello registrato dal figlio in ereditato_da e, se c'è, in
    precedente, con lo sha256 dei record. Un anello mancante o alterato, un ciclo o una catena più lunga di MAX_CHAIN
    sono un ManifestError."""
    chain, seen = [m], {m.manifest_sha256}
    while True:
        child = chain[-1]
        ref = (child.meta.get("split") or {}).get("ereditato_da")
        if not ref:
            return chain
        if len(chain) > MAX_CHAIN:
            raise mf.ManifestError(f"catena delle versioni più lunga di {MAX_CHAIN}")
        parent = mf.load(mf.ROOT / ref["percorso"])
        where = f"{child.folder} -> {ref['percorso']}"
        if parent.manifest_sha256 != ref.get("manifest_sha256"):
            raise mf.ManifestError(f"{where}: manifest.json con sha256 {parent.manifest_sha256[:12]}, il figlio registra "
                                   f"{str(ref.get('manifest_sha256'))[:12]}")
        prev = child.meta.get("precedente")
        if prev is not None and (prev.get("percorso") != ref["percorso"]
                                 or prev.get("manifest_sha256") != parent.manifest_sha256
                                 or prev.get("records_sha256") != parent.meta["records"]["sha256"]):
            raise mf.ManifestError(f"{where}: precedente ed ereditato_da non indicano la stessa versione")
        if parent.manifest_sha256 in seen:
            raise mf.ManifestError(f"{where}: ciclo nella catena delle versioni")
        seen.add(parent.manifest_sha256)
        chain.append(parent)


def _same_identity(records: list[dict], other: list[dict]) -> bool:
    old = {r["record"]: r for r in other}
    return len(old) == len(records) == len(other) and all(
        r["record"] in old and all(r[f] == old[r["record"]][f] for f in IDENTITY) for r in records)


REVIEW_FIELD = "revisione_umana"  # synth_qa_policy.REVIEW_FIELD; "revisione" del meta è la revisione della sorgente


def justified(meta: dict, r: dict) -> bool:
    """Un record che passa a uno stato meno grave in una versione derivata dalla revisione (meta[REVIEW_FIELD]): la sua
    ultima evidenza è quella della revisione dichiarata, con lo stato nuovo (S2 §2.14)."""
    rev = meta.get(REVIEW_FIELD)
    last = (r.get("evidenze") or [{}])[-1]
    return isinstance(rev, dict) and last.get("controllo") == rev.get("controllo") and last.get("stato") == r["stato"]


def parent_checks(meta: dict, records: list[dict], parent_records: list[dict], check=check) -> None:
    """Verifiche rispetto al genitore immediato di una versione derivata. Una versione dalla revisione (meta[REVIEW_FIELD],
    synth-v4-candidato-3) può riammettere solo i record della proposta, con la sua evidenza (synth_qa_policy.review_checks);
    le altre non migliorano nessuno stato."""
    old = {r["record"]: r for r in parent_records}
    check("genitore immediato: stessi record, gruppi, insiemi, split e posizioni", _same_identity(records, parent_records))
    better = [r["record"] for r in records if r["record"] in old
              and SEVERITY[r["stato"]] < SEVERITY[old[r["record"]]["stato"]]]
    if isinstance(meta.get(REVIEW_FIELD), dict):
        import synth_qa_policy as sqp

        sqp.review_checks(meta, records, parent_records, check)
    else:
        check("genitore immediato: nessun record torna ammesso o passa a uno stato meno grave", not better,
              {"casi": len(better), "esempi": better[:3]})
    if meta.get(mf.POLICY_FIELD):
        import synth_qa_policy as sqp

        sqp.policy_checks(meta, records, check)
    if not meta.get("decontaminazione"):
        return
    import synth_decontam as sd

    name = meta["decontaminazione"]["controllo"]
    kept, one_new, kept_reasons, unexplained, wrong_state = [], [], [], [], []
    group_outcome: dict[str, set] = defaultdict(set)
    for r in records:
        p = old.get(r["record"])
        if p is None:
            continue
        ev, pev = r["evidenze"] or [], p["evidenze"] or []
        if ev[:len(pev)] != pev:
            kept.append(r["record"])
        new = ev[len(pev):]
        if len(new) != 1 or new[0].get("controllo") != name:
            one_new.append(r["record"])
            continue
        motivo, pmotivo = r["motivo"] or {}, p["motivo"] or {}
        if any((motivo.get(k) or [])[:len(v or [])] != (v or []) for k, v in pmotivo.items()):
            kept_reasons.append(r["record"])
        if r["stato"] != p["stato"] and len(motivo.get(name) or []) <= len(pmotivo.get(name) or []):
            unexplained.append(r["record"])
        outcome = new[0].get("esito_gruppo")
        group_outcome[r["gruppo"]].add(outcome)
        target = sd.GROUP_STATE.get(outcome)
        expected = p["stato"] if target is None or SEVERITY[target] <= SEVERITY[p["stato"]] else target
        if outcome not in sd.OUTCOMES or r["stato"] != expected:
            wrong_state.append(r["record"])
    mixed = [g for g, o in group_outcome.items() if len(o) > 1]
    check(f"{name}: evidenze del genitore conservate", not kept, len(kept))
    check(f"{name}: una e una sola evidenza nuova per record", not one_new, len(one_new))
    check(f"{name}: motivi del genitore conservati", not kept_reasons, len(kept_reasons))
    check(f"{name}: ogni cambio di stato ha il suo motivo nuovo", not unexplained, len(unexplained))
    check(f"{name}: stato = il più grave fra quello del genitore e quello dell'esito del gruppo (nessuna riammissione)",
          not wrong_state, {"casi": len(wrong_state), "esempi": wrong_state[:3]})
    check(f"{name}: un solo esito per gruppo", not mixed, len(mixed))


def verify_splits(m: mf.Manifest, records: list[dict], check=check) -> dict:
    """Split di un manifest (sezione 2 della docstring). Restituisce catena, origine e dev per famiglia."""
    meta, rule = m.meta, m.meta["split"]
    inherited = rule.get("ereditato_da")
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[r["gruppo"]].append(r)
    chain, chain_error = [m], None
    if inherited:
        try:
            chain = chain_of(m)
        except mf.ManifestError as exc:
            chain_error = str(exc)
        check("catena delle versioni: ogni anello letto e verificato, con lo sha256 di manifest.json registrato dal "
              "figlio, fino alla versione che assegnò gli split", chain_error is None,
              chain_error or [f"{c.meta.get('versione')}: {c.manifest_sha256[:12]}" for c in chain])
        if chain_error is None:
            parent_checks(meta, records, all_records(chain[1]), check)
    origin = chain[-1]
    lineage = {"catena": [{"percorso": str(c.folder), "versione": c.meta.get("versione"),
                           "manifest_sha256": c.manifest_sha256} for c in chain],
               "origine": {"percorso": str(origin.folder), "versione": origin.meta.get("versione"),
                           "manifest_sha256": origin.manifest_sha256}}
    if chain_error is not None:
        check("split ricalcolato sulla versione che li assegnò", False, "catena non verificabile")
        return {**lineage, "dev": {}}
    base = all_records(origin)
    if len(chain) > 1:
        rules = [{k: c.meta["split"].get(k) for k in RULE_FIELDS} for c in chain]
        check("regola degli split (valori, quota, seme, dev per famiglia) uguale lungo la catena",
              all(x == rules[-1] for x in rules))
        check("assegnazione originale: stessi record, gruppi, insiemi, split e posizioni in ogni versione della catena",
              all(_same_identity(records, all_records(c)) for c in chain[1:]))
        worse, allowed = 0, 0
        for child, parent in zip(chain, chain[1:]):
            old = {r["record"]: r["stato"] for r in all_records(parent)}
            for r in all_records(child):
                if r["record"] in old and SEVERITY[r["stato"]] < SEVERITY[old[r["record"]]]:
                    if justified(child.meta, r):
                        allowed += 1
                    else:
                        worse += 1
        if any(isinstance(c.meta.get(REVIEW_FIELD), dict) for c in chain):
            check("catena: nessuna versione migliora lo stato di un record della precedente, salvo le riammissioni con "
                  "l'evidenza della revisione dichiarata dalla versione", worse == 0,
                  {"non_giustificati": worse, "riammissioni_della_revisione": allowed})
        else:
            check("catena: nessuna versione migliora lo stato di un record della precedente", worse == 0, worse)
    base_group: dict[str, list[dict]] = defaultdict(list)
    for r in base:
        base_group[r["gruppo"]].append(r)
    info = {g: {"famiglia": rs[0]["famiglia"], "valutazione": any(r["insieme"] != "train" for r in rs),
                "ammessi": sum(r["stato"] == "ammesso" for r in rs)} for g, rs in base_group.items()}
    again = [mf.assign_splits(info, rule["quota_dev"], rule["seme"]) for _ in range(2)]
    written = {g: rs[0]["split"] for g, rs in base_group.items()}
    current = {g: rs[0]["split"] for g, rs in by_group.items()}
    check("split ricalcolato due volte con quota e seme del manifest" + (
        f" sulla versione {origin.meta.get('versione')} che li assegnò" if len(chain) > 1 else "")
          + ": identico a quello scritto", again[0] == again[1] == written == current,
          sum(again[0].get(g) != current.get(g) for g in current))
    other = mf.assign_splits(info, rule["quota_dev"], rule["seme"] + 1)
    moved = sum(other[g] != written[g] for g in written)
    check("con un altro seme lo split di dev cambia (la scelta dipende dal seme)", moved > 0, moved)
    dev_info = {}
    for fam, d in rule["dev_per_famiglia"].items():
        biggest = max((i["ammessi"] for g, i in info.items() if i["famiglia"] == fam and written[g] == "dev"),
                      default=0)
        dev_info[fam] = {**d, "gruppo_dev_piu_grande": biggest,
                         "entro_la_quota": d["obiettivo"] <= d["dev"] < d["obiettivo"] + max(biggest, 1)}
    check("dev per famiglia (alla versione che assegnò gli split): almeno l'obiettivo, superato al più dall'ultimo "
          "gruppo", all(v["entro_la_quota"] for v in dev_info.values()), dev_info)
    return {**lineage, "dev": dev_info}


def specs(manifest: Path, shards: Path, sha: str | None = None) -> dict[str, dict]:
    train = str(shards / "train" / "*.tar")
    ident = {"manifest": str(manifest), **({"manifest_sha256": sha} if sha else {})}
    return {"train": {"name": "synth-train", "kind": "synth_qa", "shards": train, "weight": 1.0, **ident,
                      "split": "train"},
            "dev": {"name": "synth-dev", "kind": "synth_qa", "shards": train, "weight": 1.0, **ident, "split": "dev"},
            "test": {"name": "synth-test", "kind": "synth_qa", "shards": str(shards / "test" / "*.tar"), "weight": 1.0,
                     **ident, "split": "test"},
            "coppie": {"name": "synth-coppie", "kind": "synth_qa", "shards": str(shards / "pairs" / "*.tar"),
                       "weight": 1.0, **ident, "split": "test"}}


REC: dict[str, dict] = {}  # chiave -> record del manifest, ereditato dai processi del pool (manifest con politica)


def expected_items(r: dict, items: list) -> list:
    """Le voci (QA del JSON o {tipo, sha256} del manifest) che il loader può scegliere: tutte per un manifest senza
    politica, solo quelle dei compiti ammessi per un manifest con politica_domande (S2 §2.14)."""
    if "compiti" not in r:
        return list(items)
    states = {c["compito"]: c["stato"] for c in r["compiti"]}
    return [q for q in items if states.get(f"{r['famiglia']}/{q.get('type', q.get('tipo'))}") == "ammesso"]


def read_epoch(task: tuple[str, int, int]) -> dict:
    """Un'epoca di un worker di SourceReader: chiavi consegnate, contatori del reader, QA costruite. Con un manifest con
    politica delle domande il campione si costruisce col loader reale (data.synth_qa_sample, decisione allegata dal
    reader) e si confronta con la scelta attesa, calcolata qui dal pool del JSON filtrato coi compiti del manifest:
    prompt e target uguali, nessuna domanda di un compito escluso nel prompt."""
    import io

    from PIL import Image

    import limite_vl.data as data_mod
    from limite_vl.data import synth_qa_sample
    from limite_vl.f1c import qa_messages
    from limite_vl.stream import SourceReader

    # nel worker l'immagine si apre senza decodificare i pixel: qui si verificano le domande; prompt e target sono
    # quelli del loader, e il MixedRowStream finale costruisce esempi veri con le immagini
    data_mod.open_image = lambda raw: Image.open(io.BytesIO(raw))
    name, worker, n_workers = task
    source = SOURCES[name]
    reader = SourceReader(source, seed=0, worker=worker, n_workers=n_workers, val_count=0)
    keys, picked, built = [], {}, Counter()
    digests = mf.Digests()
    while True:
        key, files = reader.next()
        if reader.epoch > 0:  # primo campione dell'epoca successiva: l'epoca è finita
            break
        keys.append(key)
        meta = json.loads(files["json"])
        items = list(meta["qa"]) if meta["qa"] else [meta["pair"]]
        if source.admission.tasked:
            r = REC[key]
            usable = expected_items(r, items)
            exp = pick_questions(key, usable) if usable else []
            sample = synth_qa_sample(key, files)
            prompt, answer = sample.messages[0]["content"][1]["text"], sample.messages[1]["content"]
            built["campioni"] += 1
            built["prompt_e_target_uguali_alla_scelta_attesa"] += (prompt, answer) == qa_messages(exp)
            leaked = [q for q in items if q not in usable and q["q"] in prompt]
            built["domande_escluse_nel_prompt"] += len(leaked)
            if meta["family"] == "function":  # nei function solo exists ha le risposte yes e no
                built["risposte_yes_no_nei_target_function"] += sum(
                    line.split(". ", 1)[-1] in ("yes", "no") for line in answer.split("\n"))
            built["domande_tolte_prima_della_scelta"] += len(items) - len(usable)
            built.update(f"compito_costruito/{t}" for t in sample.tasks)
            built.update({f"compito_bloccato/{t}": n for t, n in (sample.blocked or {}).items()})
            picked[key] = [[q["type"], mf.qa_digest(digests, q)] for q in exp]
        else:
            picked[key] = [[q["type"], mf.qa_digest(digests, q)] for q in pick_questions(key, items)]
        problem = source.admission.check(key, files)
        if problem:
            picked[key].append(["problema", problem])
    return {"name": name, "worker": worker, "keys": keys, "picked": picked, "counts": dict(reader.counts),
            "costruiti": dict(built)}


def expect_error(name: str, fn) -> None:
    try:
        fn()
    except mf.ManifestError as exc:
        check(f"arresto: {name}", True, str(exc)[:160])
        return
    except Exception as exc:  # noqa: BLE001 - un errore di altro tipo non è l'arresto esplicito atteso
        check(f"arresto: {name}", False, f"{type(exc).__name__}: {exc}"[:200])
        return
    check(f"arresto: {name}", False, "nessun errore")


def rewrite_records(folder: Path, edit) -> None:
    """Riscrive records.jsonl.gz di una copia del manifest con edit(lista di record) e aggiorna sha256 e righe in
    manifest.json, così l'unica incoerenza è quella introdotta da edit."""
    meta = json.loads((folder / "manifest.json").read_text())
    lines = gzip.decompress((folder / "records.jsonl.gz").read_bytes()).decode().splitlines()
    rows = [json.loads(x) for x in lines]
    edit(rows)
    data = gzip.compress("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode(),
                         mtime=0)
    (folder / "records.jsonl.gz").write_bytes(data)
    meta["records"].update(sha256=hashlib.sha256(data).hexdigest(), righe=len(rows))
    (folder / "manifest.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--v3-shards", type=Path, default=ROOT / "data/shards/synth/v3")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=2, help="worker di SourceReader per ogni sorgente")
    ap.add_argument("--stream-batches", type=int, default=30)
    ap.add_argument("--config", type=Path, help="config di collaudo con le sorgenti (config/runs/f0-c7-reader-synth.json)")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("manifest_check.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: scegliere un'altra cartella")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 1. integrità
    m = mf.load(args.manifest)
    meta = m.meta
    records = [r for shard in m.by_shard.values() for r in shard.values()]
    policy = bool(meta.get(mf.POLICY_FIELD))
    check("manifest letto e verificato (sha256 dei record, record_id, un solo split per gruppo, conteggi per shard)",
          len(records) == meta["records"]["righe"], len(records))
    disk = [(s["nome"], sha256_file(mf.shard_path(s["percorso"])) == s["sha256"]) for s in meta["shards"]]
    check("shard su disco con gli sha256 del manifest", all(ok for _, ok in disk), [n for n, ok in disk if not ok])
    expected_keys = {
        "train": sorted(r["record"] for r in records if r["split"] == "train" and r["stato"] == "ammesso"),
        "dev": sorted(r["record"] for r in records if r["split"] == "dev" and r["stato"] == "ammesso"),
        "test": sorted(r["record"] for r in records if r["insieme"] == "test" and r["stato"] == "ammesso"),
        "coppie": sorted(r["record"] for r in records if r["insieme"] == "coppie" and r["stato"] == "ammesso"),
        "non_ammessi": sorted(r["record"] for r in records if r["stato"] != "ammesso")}
    rec_by_key = {r["record"]: r for r in records}
    # con la politica il reader salta anche i record ammessi senza domande di compiti ammessi (nessun_compito_ammesso)
    without_tasks = sorted(k for k in expected_keys["train"] + expected_keys["dev"] + expected_keys["test"]
                           + expected_keys["coppie"] if policy and not expected_items(rec_by_key[k], rec_by_key[k]["qa"]))
    readable = {name: [k for k in ks if k not in set(without_tasks)] for name, ks in expected_keys.items()}
    derived_ok = []
    for name, ks in expected_keys.items():
        entry = meta["derivati"][f"keys/{name}.txt"]
        path = args.manifest / "keys" / f"{name}.txt"
        derived_ok.append(path.read_text().splitlines() == ks and sha256_file(path) == entry["sha256"]
                          and entry["righe"] == len(ks))
    check("file derivati (keys/) uguali alle chiavi del manifest e ai loro sha256", all(derived_ok), derived_ok)

    # ---- 2. split e gruppi
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[r["gruppo"]].append(r)
    inherited = meta["split"].get("ereditato_da")
    lineage = verify_splits(m, records)
    dev_info = lineage["dev"]
    check("ogni gruppo ha una sola famiglia", all(len({r["famiglia"] for r in rs}) == 1 for rs in by_group.values()))
    parent: dict[str, str] = {r["record"]: r["record"] for r in records}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    first: dict[tuple, str] = {}
    for r in records:
        for link in (("pixel", r["immagini"][0]["pixel_sha256"]), ("stato", r["stato_scena_sha256"]),
                     ("coppia", r["coppia"])):
            if link[1] is None:
                continue
            if link in first:
                a, b = find(first[link]), find(r["record"])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[link] = r["record"]
    ours = defaultdict(set)
    for r in records:
        ours[find(r["record"])].add(r["record"])
    check("legami esatti ricalcolati dagli hash del manifest: stessi gruppi",
          sorted(map(sorted, ours.values())) == sorted(sorted(r["record"] for r in rs) for rs in by_group.values()),
          len(ours))
    crossing = [g for g, rs in by_group.items() if len({r["split"] for r in rs}) > 1]
    check("nessun gruppo in due split (train, dev, test)", not crossing, len(crossing))
    eval_train = [r["record"] for r in records if r["split"] in ("train", "dev")
                  and any(x["insieme"] != "train" for x in by_group[r["gruppo"]])]
    check("nessun gruppo legato a test o coppie in train o dev", not eval_train, len(eval_train))

    # ---- 3. loader
    from limite_vl.stream import MixedRowStream, make_sources, validation_samples  # importa torch: solo qui

    sp = specs(args.manifest, args.shards, m.manifest_sha256)
    if args.config:  # le sorgenti della config di collaudo: stesse di specs(), col manifest identificato
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        by_name = {s["name"]: s for s in cfg["sources"]}
        same = sorted(by_name) == sorted(s["name"] for s in sp.values()) and all(
            Path(by_name[s["name"]]["manifest"]).resolve() == Path(s["manifest"]).resolve()
            and by_name[s["name"]].get("manifest_sha256") == m.manifest_sha256
            and by_name[s["name"]]["split"] == s["split"] and by_name[s["name"]]["kind"] == s["kind"]
            and sorted(glob.glob(str(ROOT / by_name[s["name"]]["shards"]))) == sorted(glob.glob(s["shards"]))
            for s in sp.values())
        check(f"config di collaudo {args.config.name}: le quattro sorgenti del manifest, identificato dal suo sha256",
              same, sorted(by_name))
        if same:
            sp = {k: {**by_name[s["name"]], "shards": str(ROOT / by_name[s["name"]]["shards"]),
                      "manifest": str(ROOT / by_name[s["name"]]["manifest"])} for k, s in sp.items()}
    sources = make_sources(list(sp.values()))
    for key_name, source in zip(sp, sources):
        SOURCES[key_name] = source
    if policy:
        REC.update(rec_by_key)
    check("tipo synth_qa letto per compito se e solo se il manifest dichiara la politica delle domande",
          all(s.admission.tasked == policy for s in sources), {"politica": policy})
    with Pool(min(len(SOURCES) * args.workers, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))) as pool:
        epochs = pool.map(read_epoch, [(n, w, args.workers) for n in SOURCES for w in range(args.workers)])
    loaded: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, Counter] = defaultdict(Counter)
    picked: dict[str, dict] = defaultdict(dict)
    built: dict[str, Counter] = defaultdict(Counter)
    for e in epochs:
        loaded[e["name"]].extend(e["keys"])
        counts[e["name"]].update(e["counts"])
        picked[e["name"]].update(e["picked"])
        built[e["name"]].update(e["costruiti"])
    group_split = {g: rs[0]["split"] for g, rs in by_group.items()}
    load_report = {}
    if policy:
        check("politica delle domande: record ammessi senza domande di compiti ammessi, saltati dal reader",
              all(counts[n].get("saltati_manifest/nessun_compito_ammesso", 0) == sum(
                  k in set(expected_keys[n]) for k in without_tasks) for n in SOURCES), len(without_tasks))
    for name in SOURCES:
        ks = loaded[name]
        ok_once = len(ks) == len(set(ks))
        ok_set = sorted(ks) == readable[name]
        not_admitted = [k for k in ks if rec_by_key[k]["stato"] != "ammesso"]
        wrong_split = [k for k in ks if group_split[rec_by_key[k]["gruppo"]] != sp[name]["split"]]
        problems = [k for k, p in picked[name].items() if p and p[-1][0] == "problema"]
        check(f"{name}: un'epoca con {args.workers} worker consegna esattamente le chiavi ammesse dello split, una volta",
              ok_once and ok_set and not not_admitted and not wrong_split and not problems,
              {"consegnati": len(ks), "attesi": len(readable[name]), "doppi": len(ks) - len(set(ks)),
               "non_ammessi": len(not_admitted), "altro_split": len(wrong_split), "problemi": len(problems)})
        fam_loaded, task_loaded, task_expected, digest_bad = Counter(), Counter(), Counter(), 0
        removed = 0
        for k in ks:
            r = rec_by_key[k]
            fam_loaded[r["famiglia"]] += 1
            usable = expected_items(r, r["qa"])  # con la politica, solo i compiti ammessi, prima della scelta
            removed += len(r["qa"]) - len(usable)
            exp = pick_questions(k, usable)  # la stessa scelta del loader: dipende solo da chiave e numero di QA
            got = [p for p in picked[name][k] if p[0] != "problema"]
            digest_bad += [[q["tipo"], q["sha256"]] for q in exp] != got
            for q in exp:
                task_expected[f"{r['famiglia']}/{q['tipo']}"] += 1
            for t, _ in got:
                task_loaded[f"{r['famiglia']}/{t}"] += 1
        check(f"{name}: domande costruite con gli stessi digest delle QA ammesse e gli stessi conteggi per compito",
              not digest_bad and task_loaded == task_expected, digest_bad)
        if policy:
            b = built[name]
            blocked_exp = {f"compito_bloccato/{t}": n for t, n in sorted(Counter(
                f"{rec_by_key[k]['famiglia']}/{q['tipo']}" for k in ks for q in rec_by_key[k]["qa"]
                if q not in expected_items(rec_by_key[k], rec_by_key[k]["qa"])).items())}
            check(f"{name}: col loader reale (data.synth_qa_sample) prompt e target uguali alla scelta attesa sul pool "
                  "filtrato, per ogni campione", b["campioni"] == len(ks)
                  and b["prompt_e_target_uguali_alla_scelta_attesa"] == len(ks),
                  {"campioni": b["campioni"], "uguali": b["prompt_e_target_uguali_alla_scelta_attesa"]})
            check(f"{name}: nessuna domanda di un compito escluso nei prompt, nessuna risposta yes o no nei target dei "
                  "function", not b["domande_escluse_nel_prompt"] and not b["risposte_yes_no_nei_target_function"],
                  {"escluse_nei_prompt": b["domande_escluse_nel_prompt"],
                   "yes_no_nei_function": b["risposte_yes_no_nei_target_function"]})
            check(f"{name}: domande tolte prima della scelta e compiti bloccati del loader uguali a quelli attesi dal "
                  "manifest", b["domande_tolte_prima_della_scelta"] == removed
                  and {t: n for t, n in b.items() if t.startswith("compito_bloccato/")} == blocked_exp,
                  {"tolte": b["domande_tolte_prima_della_scelta"], "attese": removed, "bloccati": blocked_exp})
        load_report[name] = {"split": sp[name]["split"], "consegnati": len(ks), "contatori_reader": dict(counts[name]),
                             "costruiti_col_loader": dict(sorted(built[name].items())),
                             "per_famiglia": dict(sorted(fam_loaded.items())),
                             "domande_costruite_per_famiglia_e_compito": dict(sorted(task_loaded.items())),
                             "atteso_dal_manifest": SOURCES[name].admission.describe()}
    seen = {name: {rec_by_key[k]["gruppo"] for k in loaded[name]} for name in SOURCES}
    overlap = {f"{a}+{b}": len(seen[a] & seen[b]) for a in ("train", "dev") for b in SOURCES if b != a and
               (a, b) != ("dev", "train")}
    check("gruppi letti: nessuno in comune fra train, dev e valutazione", not any(overlap.values()), overlap)
    val = [k for k, _ in validation_samples(SOURCES["train"], 10 ** 9)]
    check("validation_samples della sorgente di train: tutto lo split dev del manifest, nell'ordine degli shard",
          sorted(val) == expected_keys["dev"] and len(val) == len(set(val)), len(val))
    from limite_vl.processing import LimiteVLProcessor

    processor = LimiteVLProcessor.from_pretrained(str(VIOLETTO))
    stream = MixedRowStream([SOURCES["train"]], processor, max_len=4096, micro_rows=2, seed=0, shuffle_buffer=64,
                            pack_buffer=32)
    keys, last = [], None
    for n, batch in enumerate(stream.iter_worker(0, 1)):
        keys += [k.split("/", 1)[1] for k in batch["meta"]["keys"]]
        last = batch["meta"]
        if n + 1 >= args.stream_batches:
            break
    train_set = set(expected_keys["train"])
    check("MixedRowStream col processor vero: solo chiavi di train ammesse, esempi contati, nessun errore del manifest",
          keys and all(k in train_set for k in keys) and last["stats"].get("loaded/synth-train", 0) > 0
          and not last["stats"].get("manifest_errors/synth-train") and not last["stats"].get("errors/synth-train"),
          {"esempi": len(keys), "stats": last["stats"], "reader": last["readers"]})

    # ---- 4. arresti espliciti
    with tempfile.TemporaryDirectory(prefix="manifest_check_") as tmp:
        tmp = Path(tmp)

        def copy(name: str) -> Path:
            dst = tmp / name
            shutil.copytree(args.manifest, dst)
            return dst

        base = {k: v for k, v in sp["train"].items() if k != "manifest_sha256"}

        def at(folder: Path, **kw) -> list[dict]:
            """Sorgente sulla cartella, identificata dallo sha256 del suo manifest.json: l'unico errore è quello voluto."""
            mj = folder / "manifest.json"
            return [{**base, "manifest": str(folder), **({"manifest_sha256": sha256_file(mj)} if mj.is_file() else {}),
                     **kw}]

        same = copy("uguale")
        check("controllo dei casi: la copia non alterata si carica", bool(make_sources(at(same))))
        expect_error("manifest mancante", lambda: make_sources(at(tmp / "inesistente")))
        broken = copy("records-alterati")
        data = bytearray((broken / "records.jsonl.gz").read_bytes())
        data[len(data) // 2] ^= 0xFF
        (broken / "records.jsonl.gz").write_bytes(bytes(data))
        expect_error("records.jsonl.gz alterato", lambda: make_sources(at(broken)))
        other_sha = copy("sha-shard")
        meta2 = json.loads((other_sha / "manifest.json").read_text())
        meta2["shards"][0]["sha256"] = "0" * 64
        (other_sha / "manifest.json").write_text(json.dumps(meta2, indent=1, ensure_ascii=False))
        expect_error("sha256 di uno shard diverso", lambda: make_sources(at(other_sha)))
        moved_dir = copy("gruppo-spostato")
        multi = next(g for g, rs in by_group.items() if len(rs) > 1 and rs[0]["split"] == "train")

        def move(rows):
            target = next(r for r in rows if r["gruppo"] == multi)
            target["split"] = "dev"

        rewrite_records(moved_dir, move)
        expect_error("gruppo in due split", lambda: make_sources(at(moved_dir)))
        renamed = copy("record-id")

        def rename(rows):
            rows[0]["record"] = rows[0]["record"] + "x"

        rewrite_records(renamed, rename)
        expect_error("record_id non ricalcolabile", lambda: make_sources(at(renamed)))
        expect_error("shard v3 col manifest v4", lambda: make_sources(
            at(args.manifest, shards=str(args.v3_shards / "train" / "*.tar"))))
        expect_error("shard v4 senza manifest", lambda: make_sources(
            [{k: v for k, v in base.items() if k not in ("manifest", "split")}]))
        expect_error("split senza record ammessi (test sugli shard di train)", lambda: make_sources(
            at(args.manifest, split="test")))
        expect_error("split sconosciuto", lambda: make_sources(at(args.manifest, split="val")))
        expect_error("manifest senza split", lambda: make_sources(
            [{k: v for k, v in at(args.manifest)[0].items() if k != "split"}]))
        if policy:  # la politica delle domande rende la sorgente per compito: identificazione e niente liste
            expect_error("politica delle domande: manifest non identificato (manca manifest_sha256)",
                         lambda: make_sources([{**base, "manifest": str(args.manifest)}]))
            listed = tmp / "lista.txt"
            listed.write_text(expected_keys["train"][0] + "\n")
            expect_error("politica delle domande: lista exclude nella config",
                         lambda: make_sources(at(args.manifest, exclude=[str(listed)])))
            unknown = copy("politica-sconosciuta")
            meta3 = json.loads((unknown / "manifest.json").read_text())
            meta3[mf.POLICY_FIELD]["versione"] = "synth-domande/0"
            (unknown / "manifest.json").write_text(json.dumps(meta3, indent=1, ensure_ascii=False))
            expect_error("politica delle domande: versione sconosciuta", lambda: make_sources(at(unknown)))

    failed = [r["controllo"] for r in RESULTS if not r["esito"]]
    coverage = {}
    for key, n in meta["conteggi"]["qa_per_split_famiglia_tipo"].items():
        split, fam, qtype = key.split("/")
        coverage.setdefault(f"{fam}/{qtype}", {})[split] = n
    report = {"schema": "manifest-check/1", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
              "args": {k: str(v) for k, v in vars(args).items()},
              "manifest": {"path": str(args.manifest), "manifest_sha256": m.manifest_sha256,
                           "records_sha256": meta["records"]["sha256"], "split_ereditato_da": inherited,
                           "catena": lineage["catena"], "split_assegnati_da": lineage["origine"]},
              "controlli": RESULTS, "falliti": failed, "dev": dev_info, "loader": load_report,
              "gruppi_letti_in_comune": overlap, "qa_ammesse_per_compito_e_split": coverage,
              "conteggi_manifest": meta["conteggi"], "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "report.json", report)
    print(f"{len(RESULTS) - len(failed)} controlli superati su {len(RESULTS)} ({report['seconds']} s)")
    if failed:
        sys.exit("controlli falliti: " + "; ".join(failed))


if __name__ == "__main__":
    main()
