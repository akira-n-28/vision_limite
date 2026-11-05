#!/usr/bin/env python3
"""Manifest candidato di MAVIS-Geometry (docs/F0_CAMPIONE_CANDIDATO.md §6 punto 3; S1 §§1-3, 5, 7, 9; S2 §3), col
sottoinsieme di S1 dei sintetici v4 e di UniMER: riusa da scripts/unimer_manifest.py la lettura verificata degli shard,
i componenti dei legami esatti, il rango del campione candidato, la scrittura riproducibile e le regole degli stati. Solo
in un job SLURM CPU: shard ed evidenze in sola lettura, scrive cartelle nuove. Organizza candidati ed evidenze: non
certifica la qualità, non assegna gold e non autorizza il training. C9 (opzioni di shape_type, valore visualizzato ed
esatto nel record) non è implementato e nessuna risposta è inventata: domande e descrizioni sono quelle del codice
esistente (f1_mavis, f1_schema).

Ingressi, verificati prima di scrivere:
- shard `CaraJ/MAVIS-Geometry@9c80fff3`, variante caption (--shards, job 29419): 11 tar con sha256 e dimensione del
  manifest dello sharding, letti e hashati in un solo passaggio; lo snapshot di Hugging Face con il blob dichiarato;
- evidenze (--evidence, cartella outputs): f1_mavis/run2 (job 30052: verità da diagram_generation confermata dalla
  didascalia, andata e ritorno della descrizione, domande), mavis_caption_check/run1 (job 30025), mavis_geometry/run1
  (job 30042), f1_mavis/fractions_run1 (job 30049: razionali disegnati come decimali), split_duplicates/run1/mavis
  (job 30043, solo controllo incrociato dei gruppi).

Record, una riga per campione (caption.jpg di una cartella RuleBaseGeo/depthK/<i>): compiti `describe` e i tipi di
domanda di f1_mavis.mavis_questions, ciascuno con il suo stato e il numero di domande; QA con tipo e digest, costruite
con la chiave del tar come fa il loader (data.mavis_f1_sample); digest della descrizione canonica, della verità, di
diagram_generation e della didascalia; misure razionali il cui valore disegnato (decimale a due cifre) è diverso
dall'esatto, che il record non contiene ancora (C9).

Gruppi (S1 §3): componenti dei legami esatti, cioè gli stessi byte del PNG oppure lo stesso diagram_generation (la
stessa figura del motore).

Stati:
- record `quarantena` se la verità manca (didascalia diversa da diagram_generation, vertice non univoco), se l'andata e
  ritorno della descrizione non è perfetta, se il PNG ha dimensioni diverse dal JSON, se una lista storica lo segnala, o
  se nel suo gruppo, cioè nella stessa immagine o figura, la stessa domanda ha risposte diverse o le descrizioni
  differiscono: almeno una verità non è fedele alla figura; altrimenti `ammesso`, candidato non certificato;
- compito: `shape_type` sempre in quarantena (classi che non formano una catena; servono opzioni con una sola vera,
  C9, S2 §3), senza togliere gli altri compiti del record, che ne seguono lo stato.
Due proprietà separate per record: coerenza con le annotazioni (verificata dal codice: didascalia, diagram_generation,
andata e ritorno) e fedeltà all'immagine (da verificare con la revisione visiva e la stima di S4).

Riconciliazione prima di scrivere: il ricalcolo deve coincidere con run2 (motivi, campioni pronti per profondità, domande
per tipo, risposte yes/no, domande per esempio e le 60 righe di sample.jsonl, ricalcolati con la chiave `mavis-<id>`
che run2 usava, diversa da quella del loader); caption check e geometry check senza segnalazioni sui 199.050; le chiavi
escluse allo sharding non sono negli shard; i duplicati della vecchia validazione stanno in gruppi di almeno due record.

Split (manifest.assign_splits): dev per circa --dev-fraction dei record ammessi di ogni profondità (d0-d3; un gruppo con
più profondità va in quella minima), gruppi indivisibili in ordine di sha256({seme, gruppo}) con --seed; rango_f0 per
gruppo e rango_f0_sviluppo per i gruppi di dev (F0_CAMPIONE_CANDIDATO §1).

Uscite come scripts/unimer_manifest.py: --out (records.jsonl.gz, manifest.json, keys/), --report/verifica.json,
--summary (rapporto sintetico per git); --probe legge un solo tar e scrive --report/probe.json.

    srun $RUN_PYTHON -u -B scripts/mavis_manifest.py --shards DIR --evidence DIR --out DIR --report DIR --summary FILE \\
        [--name mavis-geometry-candidato] [--version 1] [--dev-fraction 0.05] [--seed 0] [--workers 4] [--probe]
"""
from __future__ import annotations

import argparse
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
import unimer_manifest as um  # noqa: E402
from f1_mavis_check import perfect  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.f1_mavis import describe, mavis_questions, mavis_truth, questions_example  # noqa: E402
from limite_vl.f1_schema import geometry_scores  # noqa: E402
from limite_vl.mavis_gen import measures  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, write_new_json  # noqa: E402

SOURCE = "mavis-geometry"
REVISION = re.compile(r"^CaraJ/MAVIS-Geometry@([0-9a-f]{40})$")
QUARANTINED_TASKS = {"shape_type": "classi che non formano una catena (quadrato, rettangolo, rombo, aquilone, "
                                   "parallelogramma; triangolo rettangolo e isoscele): servono opzioni con una sola vera "
                                   "secondo le misure note (C9, S2 §3)"}
TYPE_ORDER = ("side_len", "angle_val", "removed_side", "direction", "shape_type", "shared_side", "count_points",
              "exists_point", "exists_shape")
RATIONAL = re.compile(r"(\d+)/(\d+)")
NOT_COVERED = [
    "variant_id e uid (S1 §1): dipendono dal contratto di risposta (S3, D3). Il loader sceglie dalla chiave il compito "
    "(descrizione o domande), il prompt della descrizione e da 1 a 5 domande; il manifest elenca tutte le domande "
    "possibili del record con il loro digest",
    "hash dei token di input e di target, tokenizer, template e contratto (S1 §2, §7): dipendono dal contratto",
    "indice globale fra sorgenti (S1 §4): i gruppi sono i legami esatti dentro MAVIS-Geometry; le altre due immagini "
    "della cartella (text.jpg, vision_dominant.jpg) non sono negli shard",
    "decontaminazione esterna contro pannello-v3 (pixel, pHash e testo, comprese le geometrie di MathVerse, degli "
    "stessi autori): non fatta, è il lotto successivo",
    "reader col manifest (C7): gli shard di MAVIS non sono in manifest.MANIFEST_REQUIRED e il loader attuale "
    "(data.mavis_f1_sample) può ancora scegliere shape_type; un reader che applichi il manifest deve escluderla",
    "C9: opzioni di shape_type costruite dalle misure note, valore visualizzato ed esatto nel record; qui solo il "
    "conteggio delle misure razionali disegnate come decimali",
    "campo gold (S2 §3): nessuno; target_controllato dopo il campione di S4",
]
MISSING_CHECKS = [
    "fedeltà all'immagine: il renderer di MAVIS non è nostro, nessuna invariante; revisione visiva campionaria e stima "
    "di S4 da fare (finora 40 immagini guardate, outputs/f1_mavis/contact_run1)",
    "decontaminazione esterna di MAVIS contro pannello-v3",
    "osservabilità alla risoluzione scelta (oggi 2304 patch, oltre le 1024 di addestramento di NaFlex)",
    "shape_type in quarantena finché C9 non costruisce le opzioni",
    "valori razionali: la risposta è il decimale disegnato; l'esatto non è nel record (C9)",
]


def rational_measures(generation: str) -> int:
    """Misure razionali senza radice (p/q non intero) di diagram_generation: il motore le disegna come decimali a due
    cifre (job 30049), quindi il valore visualizzato è diverso dall'esatto."""
    gm = measures(generation)
    n = 0
    for v in list(gm["side"].values()) + list(gm["angle"].values()):
        m = RATIONAL.fullmatch(v)
        n += bool(m) and int(m.group(1)) % int(m.group(2)) != 0
    return n


def qa_sha(q: dict) -> str:
    return mf.sha256_text(mf.canonical({k: q[k] for k in ("type", "q", "answer")}))


def read_shard(task: tuple[str, int, tuple]) -> dict:
    """Un tar letto e hashato in un passaggio: righe compatte, più gli esempi di run2 per le chiavi di sample.jsonl."""
    path, limit, sample_keys = task
    sample_keys = set(sample_keys)
    t0 = time.time()
    raw = um.HashingReader(Path(path))
    rows, problems, examples = [], [], {}
    # contatori di run2 (f1_mavis_check.py), con le sue regole: motivi, campioni e pronti per profondità, domande
    agg = {"reasons": Counter(), "by_depth": defaultdict(Counter), "qtypes": Counter(), "answers": Counter(),
           "n_questions": Counter()}
    with tarfile.open(fileobj=io.BufferedReader(raw, buffer_size=1 << 20), mode="r|") as tar:
        for index, (key, files) in enumerate(um.iter_samples(tar)):
            if limit and index >= limit:
                break
            if set(files) != {"json", "png"}:
                problems.append(f"{key}: file {sorted(files)}")
                continue
            meta = json.loads(files["json"])
            if meta.get("id") != key or meta.get("variant") != "caption":
                problems.append(f"{key}: id {meta.get('id')!r} o variante {meta.get('variant')!r}")
                continue
            te, png = meta["text_en"], files["png"]
            caption, generation = te["caption"], te["diagram_generation"]
            truth, why = mavis_truth(caption, generation)
            dims_ok = um.png_size(png) == (meta["width"], meta["height"])
            agg["by_depth"][str(meta["depth"])]["campioni"] += 1
            agg["reasons"][why] += 1
            qa, pairs, desc_sha, truth_sha, rt = [], (), None, None, None
            if truth is not None:
                pool = mavis_questions(key, truth)  # chiave del tar, come il loader
                qa = [(q["type"], qa_sha(q)) for q in pool]
                pairs = tuple(sorted({(q["q"], q["answer"]) for q in pool}))
                _, description = describe(key, truth)
                desc_sha = hashlib.sha256(description.encode()).digest()
                truth_sha = hashlib.sha256(mf.canonical(truth).encode()).digest()
                # riproduzione di run2 (f1_mavis_check.py usava la chiave "mavis-<id>", non quella del loader)
                k2 = f"mavis-{key}"
                prompt2, description2 = describe(k2, truth)
                rt = perfect(geometry_scores(description2, truth))
                if not rt:
                    agg["reasons"]["andata_e_ritorno_imperfetto"] += 1
                else:
                    agg["by_depth"][str(meta["depth"])]["pronti"] += 1
                    pool2 = mavis_questions(k2, truth)
                    agg["qtypes"].update(q["type"] for q in pool2)
                    agg["answers"].update(q["answer"] for q in pool2 if q["type"].startswith("exists"))
                    if any(not q["answer"] for q in pool2):
                        agg["reasons"]["risposta_vuota"] += 1
                    q_prompt2, q_answer2, picked2 = questions_example(k2, truth)
                    agg["n_questions"][str(len(picked2))] += 1
                    if key in sample_keys:
                        examples[key] = {"describe_prompt": prompt2, "description": description2,
                                         "questions_prompt": q_prompt2, "questions_answer": q_answer2}
            rows.append((key, index, hashlib.sha256(files["json"]).digest(), hashlib.sha256(png).digest(),
                         meta["depth"], hashlib.sha256(caption.encode()).digest(),
                         hashlib.sha256(generation.encode()).digest(), truth_sha, why, dims_ok, desc_sha, qa, pairs,
                         rational_measures(generation), rt, meta["width"], meta["height"]))
    sha, size = raw.finish() if not limit else (None, None)
    return {"path": path, "sha256": sha, "bytes": size, "rows": rows, "problems": problems, "examples": examples,
            "agg": agg, "seconds": round(time.time() - t0, 1)}


def shard_plan(shards: Path) -> dict:
    main = json.loads((shards / "manifest.json").read_text())
    m = REVISION.match(main.get("source", ""))
    if not m:
        raise um.Problem(f"sorgente degli shard non riconosciuta: {main.get('source')!r}")
    return {"revision": m.group(1), "main": main, "sets": {"train": [(shards / s["name"], s) for s in main["shards"]]},
            "files": {"manifest": file_entry(shards / "manifest.json"), "plan": file_entry(shards / "plan.json")}}


def snapshot_identity(main: dict) -> dict:
    path = Path(main["zip"])
    blob = path.resolve()
    return {"zip": {"percorso": str(path), "esiste": path.is_file(), "byte": path.stat().st_size if path.is_file() else None,
                    "blob": blob.name, "sha256_atteso": main["zip_sha256_expected"],
                    "blob_coincide": blob.name == main["zip_sha256_expected"]}}


def evidence_files(evidence: Path) -> dict:
    """File delle evidenze storiche, per nome della lista."""
    return {"f1_mavis_run2": {n: evidence / "f1_mavis" / "run2" / n for n in ("report.json", "sample.jsonl",
                                                                                "failures.jsonl")},
            "mavis_caption_check_run1": {n: evidence / "mavis_caption_check" / "run1" / n for n in ("report.json",
                                                                                                 "issues.jsonl")},
            "mavis_geometry_run1": {n: evidence / "mavis_geometry" / "run1" / n for n in ("report.json",
                                                                                      "violations.jsonl")},
            "fractions_run1": {"report.json": evidence / "f1_mavis" / "fractions_run1" / "report.json"},
            "split_duplicates_run1": {"report.json": evidence / "split_duplicates" / "run1" / "report.json",
                                      "val_with_duplicates.jsonl": evidence / "split_duplicates" / "run1" / "mavis" /
                                      "val_with_duplicates.jsonl"}}


def build(args, plan: dict) -> dict:
    """Righe del manifest e metadati, dopo tutte le riconciliazioni; um.Problem se un ingresso è incoerente."""
    problems: list[str] = []
    ev = evidence_files(Path(args.evidence))
    sample2 = um.read_jsonl(ev["f1_mavis_run2"]["sample.jsonl"])
    sample_keys = tuple(sorted(r["id"] for r in sample2))
    tasks = [(str(p), 0, sample_keys) for p, _ in plan["sets"]["train"]]
    results: dict[str, dict] = {}
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(read_shard, tasks):
            results[res["path"]] = res
            print(f"{Path(res['path']).name}: {len(res['rows'])} campioni in {res['seconds']} s", flush=True)
    shards_meta, rows, row_shard, examples = [], [], [], {}
    agg = {"reasons": Counter(), "by_depth": defaultdict(Counter), "qtypes": Counter(), "answers": Counter(),
           "n_questions": Counter()}
    for p, s in plan["sets"]["train"]:
        res = results[str(p)]
        problems += [f"{p.name}: {x}" for x in res["problems"]]
        if res["sha256"] != s["sha256"] or res["bytes"] != s["bytes"] or len(res["rows"]) != s["samples"]:
            problems.append(f"{p.name}: sha256, byte o campioni diversi dal manifest dello sharding")
        shards_meta.append({"insieme": "train", "nome": p.name, "percorso": um.rel(p), "sha256": res["sha256"],
                            "bytes": res["bytes"], "record": len(res["rows"])})
        rows += res["rows"]
        row_shard += [um.rel(p)] * len(res["rows"])
        examples.update(res["examples"])
        for k in ("reasons", "qtypes", "answers", "n_questions"):
            agg[k].update(res["agg"][k])
        for d, c in res["agg"]["by_depth"].items():
            agg["by_depth"][d].update(c)
    if problems:
        raise um.Problem("; ".join(problems[:10]))
    index = {}
    for i, row in enumerate(rows):
        if row[0] in index:
            raise um.Problem(f"{row[0]}: in due shard ({row_shard[index[row[0]]]}, {row_shard[i]})")
        index[row[0]] = i

    # ---- esclusi allo sharding: non sono negli shard, i loro primi esemplari sì
    excluded = plan["main"].get("excluded") or {}
    incomplete = excluded.get("incomplete") or []
    dup_caps = excluded.get("duplicate_captions") or []
    counts_sharding = plan["main"].get("counts") or {}
    present = [k for k in incomplete + [d[0] for d in dup_caps] if k in index]
    firsts = [d[1] for d in dup_caps if d[1] not in index]
    if present or firsts or len(incomplete) != counts_sharding.get("incomplete") \
            or len(dup_caps) != counts_sharding.get("duplicate_captions"):
        problems.append(f"sharding: esclusi presenti negli shard {present[:3]}, primi esemplari assenti {firsts[:3]}, o "
                        "conteggi diversi dal manifest dello sharding")

    # ---- run2 ricalcolato (chiave mavis-<id>), caption check e geometry check
    report2 = json.loads(ev["f1_mavis_run2"]["report.json"].read_text())
    shard_sha = {s["sha256"] for s in shards_meta}
    reports = {"f1_mavis_run2": report2}
    for name in ("mavis_caption_check_run1", "mavis_geometry_run1"):
        reports[name] = json.loads(ev[name]["report.json"].read_text())
    for name, rep in reports.items():
        if {s["sha256"] for s in rep.get("shards", [])} != shard_sha or rep.get("samples") != len(rows):
            problems.append(f"{name}: shard o campioni diversi da quelli letti")
    for name in ("fractions_run1", "split_duplicates_run1"):  # senza shard nel report: servono il job e i file
        reports[name] = json.loads(ev[name]["report.json"].read_text())
    ours2 = {"reasons": dict(agg["reasons"]), "by_depth": {k: dict(v) for k, v in agg["by_depth"].items()},
             "question_types_in_pool": dict(agg["qtypes"]), "presence_answers": dict(agg["answers"]),
             "questions_per_example": dict(agg["n_questions"])}
    theirs2 = {k: report2.get(k) for k in ours2}
    if ours2 != theirs2:
        problems.append(f"run2 non riprodotto: {[k for k in ours2 if ours2[k] != theirs2[k]]}")
    bad_examples = [r["id"] for r in sample2 if examples.get(r["id"]) != {k: r[k] for k in (
        "describe_prompt", "description", "questions_prompt", "questions_answer")}]
    if bad_examples or len(examples) != len(sample2):
        problems.append(f"run2: esempi di sample.jsonl non riprodotti: {bad_examples[:3]}")
    failures2 = um.read_jsonl(ev["f1_mavis_run2"]["failures.jsonl"])
    issues = um.read_jsonl(ev["mavis_caption_check_run1"]["issues.jsonl"])
    violations = um.read_jsonl(ev["mavis_geometry_run1"]["violations.jsonl"])
    flagged = {"f1_mavis_run2": [r.get("id") for r in failures2],
               "mavis_caption_check_run1": [r.get("key", r.get("id")) for r in issues],
               "mavis_geometry_run1": [r.get("key", r.get("id")) for r in violations]}
    for name, keys in flagged.items():
        problems += um.resolve(keys, index, name)
    if reports["mavis_caption_check_run1"].get("samples_with_issues") != len(issues) \
            or reports["mavis_geometry_run1"].get("samples_with_violations") != len(violations):
        problems.append("caption check o geometry check: segnalazioni del report diverse dai file")
    caption_issues = sum(row[8] == "didascalia_diversa" for row in rows)
    if caption_issues != len(issues):
        problems.append(f"caption check non riprodotto: {caption_issues} didascalie diverse, {len(issues)} nel file")
    dup_rows = um.read_jsonl(ev["split_duplicates_run1"]["val_with_duplicates.jsonl"])
    problems += um.resolve([r["val_key"] for r in dup_rows], index, "split_duplicates val_with_duplicates")
    if problems:
        raise um.Problem("; ".join(problems[:10]))

    # ---- gruppi: stessi byte del PNG o stesso diagram_generation
    links = um.same_key_links([row[3] for row in rows]) + um.same_key_links([row[6] for row in rows])
    rep = um.components(len(rows), links)
    members = defaultdict(list)
    for i in range(len(rows)):
        members[rep[i]].append(i)
    dup_bad = [r["val_key"] for r in dup_rows if r.get("immagine") and len(members[rep[index[r["val_key"]]]]) < 2]
    if dup_bad:
        raise um.Problem(f"split_duplicates: {len(dup_bad)} chiavi con immagine ripetuta in un gruppo da solo")
    digests = mf.Digests()
    rid = [mf.record_id(digests, SOURCE, plan["revision"], row[0]) for row in rows]
    index_name = {"nome": "legami esatti di MAVIS-Geometry: stessi byte del PNG (caption.jpg a 2304 patch) o stesso "
                          "diagram_generation", "sharding_manifest_sha256": plan["files"]["manifest"]["sha256"]}
    gid_of_rep = {r: mf.group_id(digests, index_name, sorted({rid[i] for i in ms})) for r, ms in members.items()}

    # ---- stati: verità, andata e ritorno, dimensioni, liste storiche, coerenza dentro il gruppo
    state, motive = ["ammesso"] * len(rows), [[] for _ in rows]
    for i, row in enumerate(rows):
        if row[7] is None:
            state[i], motive[i] = "quarantena", [f"verità non disponibile: {row[8]}"]
        elif not row[14]:
            state[i], motive[i] = "quarantena", ["andata e ritorno della descrizione imperfetto"]
        if not row[9]:
            state[i] = "quarantena"
            motive[i].append("PNG con dimensioni diverse dal JSON")
    for name, keys in flagged.items():
        for k in keys:
            state[index[k]] = "quarantena"
            motive[index[k]].append(f"segnalato da {name}")
    conflict_groups = 0
    for r, ms in members.items():
        if len(ms) < 2:
            continue
        answers_by_q = defaultdict(set)
        for i in ms:
            for q, a in rows[i][12]:
                answers_by_q[q].add(a)
        descs = {rows[i][10] for i in ms}
        if any(len(a) > 1 for a in answers_by_q.values()) or len(descs) > 1:
            conflict_groups += 1
            for i in ms:
                state[i] = um.worst(state[i], "quarantena")
                motive[i].append("stessa immagine o figura con verità diverse nel gruppo: almeno una non è fedele")

    # ---- split per profondità e ranghi
    info, mixed = {}, 0
    for r, ms in members.items():
        depths = {rows[i][4] for i in ms}
        mixed += len(depths) > 1
        info[gid_of_rep[r]] = {"famiglia": f"d{min(depths)}", "valutazione": False,
                               "ammessi": sum(state[i] == "ammesso" for i in ms)}
    split_of = mf.assign_splits(info, args.dev_fraction, args.seed)
    rank = {g: um.f0_rank(args.seed, g) for g in info}
    rank_dev = {g: um.f0_rank(args.seed, g, "sviluppo") for g in info if split_of[g] == "dev"}
    reproduced = {"run2": {"uguale": True, **ours2, "esempi_riprodotti": len(examples),
                           "nota": "ricalcolato con la chiave 'mavis-<id>' di f1_mavis_check.py; le domande del manifest "
                                   "usano la chiave del tar, come il loader"},
                  "caption_check": {"didascalie_diverse": caption_issues, "segnalazioni_nel_file": len(issues)},
                  "geometry_check": {"violazioni_nel_file": len(violations), "ricalcolato": False},
                  "sharding": {"incompleti": len(incomplete), "didascalie_duplicate": len(dup_caps)},
                  "gruppi": {"in_conflitto": conflict_groups, "con_piu_record": sum(len(m) > 1 for m in members.values()),
                             "record_in_gruppi_con_piu_record": sum(len(m) for m in members.values() if len(m) > 1),
                             "con_piu_profondita": mixed},
                  "duplicati_vecchia_validazione_controllati": sum(1 for r in dup_rows if r.get("immagine"))}
    return {"rows": rows, "row_shard": row_shard, "rep": rep, "gid_of_rep": gid_of_rep, "rid": rid, "state": state,
            "motive": motive, "split_of": split_of, "rank": rank, "rank_dev": rank_dev, "info": info,
            "index_name": index_name, "shards_meta": shards_meta, "reproduced": reproduced, "flagged": flagged,
            "reports": reports, "revision": plan["revision"], "ev": ev}


def tasks_of(row, record_state: str) -> list[dict]:
    """Compiti del record: describe e i tipi di domanda, con lo stato (shape_type sempre in quarantena)."""
    if row[7] is None:
        return [{"compito": "describe", "stato": record_state}]
    by_type = Counter(t for t, _ in row[11])
    out = [{"compito": "describe", "stato": record_state}]
    for t in sorted(by_type, key=lambda t: (TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER), t)):
        st = um.worst(record_state, "quarantena") if t in QUARANTINED_TASKS else record_state
        out.append({"compito": t, "stato": st, "domande": by_type[t]})
    return out


def records(b: dict):
    rows = b["rows"]
    flags = {name: set(keys) for name, keys in b["flagged"].items()}
    for i, row in enumerate(rows):
        g = b["gid_of_rep"][b["rep"][i]]
        st = b["state"][i]
        evid = [{"lista": "f1_mavis_run2", "esito": "fallito" if row[0] in flags["f1_mavis_run2"]
                 else row[8] if row[7] is None else "ok" if row[14] else "andata_e_ritorno_imperfetto"},
                {"lista": "mavis_caption_check_run1", "esito": "segnalato" if row[0] in flags["mavis_caption_check_run1"]
                 else "nessuna_segnalazione"},
                {"lista": "mavis_geometry_run1", "esito": "violazione" if row[0] in flags["mavis_geometry_run1"]
                 else "nessuna_violazione"}]
        coherent = row[7] is not None and row[14]
        yield {"record_id": b["rid"][i], "sorgente": SOURCE, "revisione": b["revision"], "record": row[0],
               "insieme": "train", "shard": b["row_shard"][i], "indice": row[1], "famiglia": f"d{row[4]}",
               "compiti": tasks_of(row, st), "gruppo": g, "split": b["split_of"][g], "stato": st,
               "motivo": b["motive"][i], "gold": None,
               "immagini": [{"sha256": row[3].hex(), "ordine": 0, "griglia": [row[16] // 16, row[15] // 16]}],
               "json_sha256": row[2].hex(), "didascalia_sha256": row[5].hex(),
               "diagram_generation_sha256": row[6].hex(), "verita_sha256": row[7].hex() if row[7] else None,
               "descrizione_sha256": row[10].hex() if row[10] else None,
               "qa": [{"tipo": t, "sha256": s} for t, s in row[11]],
               "verifiche": {"coerenza_annotazioni": "verificata" if coherent else "non_verificata",
                             "fedelta_immagine": "da_verificare"},
               "misure_razionali_come_decimali": row[13], "evidenze": evid, "trasformazione": None, "genitori": [],
               "rango_f0": b["rank"][g], "rango_f0_sviluppo": b["rank_dev"].get(g)}


def summarize(recs: list[dict]) -> dict:
    c = {"per_split_famiglia_stato": Counter(), "per_compito_split_stato": Counter(), "domande_per_tipo_stato": Counter(),
         "per_motivo": Counter(), "gruppi_per_split": Counter(), "misure_razionali": Counter()}
    groups = {}
    for r in recs:
        c["per_split_famiglia_stato"][f"{r['split']}/{r['famiglia']}/{r['stato']}"] += 1
        for t in r["compiti"]:
            c["per_compito_split_stato"][f"{t['compito']}/{r['split']}/{t['stato']}"] += 1
            if "domande" in t:
                c["domande_per_tipo_stato"][f"{t['compito']}/{t['stato']}"] += t["domande"]
        for m in r["motivo"]:
            c["per_motivo"][m] += 1
        c["misure_razionali"]["record_con_misure_razionali"] += r["misure_razionali_come_decimali"] > 0
        groups[r["gruppo"]] = r["split"]
    for s in groups.values():
        c["gruppi_per_split"][s] += 1
    return {k: dict(sorted(v.items())) for k, v in c.items()}


def verify(folder: Path, ev: dict) -> dict:
    """Controlli rileggendo il manifest scritto (manifest.load) e le liste storiche, senza gli shard."""
    checks = []

    def ok(name: str, cond: bool, detail=None) -> None:
        checks.append({"controllo": name, "esito": bool(cond), "dettaglio": detail})

    m = mf.load(folder)
    ok("manifest letto e verificato (sha256 dei record, record_id ricalcolati, un solo split per gruppo, conteggi per "
       "shard)", True)
    recs = [r for rows in m.by_shard.values() for r in rows.values()]
    meta = m.meta
    ok("identità uniche: record_id e chiave", len({r["record_id"] for r in recs}) == len(recs)
       == len({r["record"] for r in recs}))
    links = um.same_key_links([r["immagini"][0]["sha256"] for r in recs])
    links += um.same_key_links([r["diagram_generation_sha256"] for r in recs])
    rep = um.components(len(recs), links)
    ours, theirs = defaultdict(set), defaultdict(set)
    for n, r in enumerate(recs):
        ours[rep[n]].add(n)
        theirs[r["gruppo"]].add(n)
    ok("gruppi ricalcolati dai byte del PNG e da diagram_generation: stessa partizione",
       sorted(map(sorted, ours.values())) == sorted(map(sorted, theirs.values())), len(theirs))
    digests = mf.Digests()
    bad_gid = sum(mf.group_id(digests, meta["indice"], sorted(recs[n]["record_id"] for n in ms))
                  != recs[next(iter(ms))]["gruppo"] for ms in theirs.values())
    ok("group_id ricalcolati dai membri", bad_gid == 0, bad_gid)
    split = defaultdict(set)
    for r in recs:
        split[r["gruppo"]].add(r["split"])
    ok("nessun gruppo in due split (in particolare fra train e dev)", all(len(s) == 1 for s in split.values()))
    info = {}
    for g, ms in theirs.items():
        depths = {recs[n]["famiglia"] for n in ms}
        info[g] = {"famiglia": min(depths, key=lambda d: int(d[1:])), "valutazione": False,
                   "ammessi": sum(recs[n]["stato"] == "ammesso" for n in ms)}
    rule = meta["split"]
    again = mf.assign_splits(info, rule["quota_dev"], rule["seme"])
    ok("split ricalcolato con quota e seme del manifest: uguale a quello scritto",
       all(again[g] == next(iter(split[g])) for g in info))
    other = mf.assign_splits(info, rule["quota_dev"], rule["seme"] + 1)
    moved = sum(other[g] != again[g] for g in info)
    ok("con un altro seme il dev cambia (la scelta dipende dal seme)", moved > 0, moved)
    per_family = {}
    for fam in sorted({i["famiglia"] for i in info.values()}):
        admitted = sum(i["ammessi"] for i in info.values() if i["famiglia"] == fam)
        dev = sum(i["ammessi"] for g, i in info.items() if i["famiglia"] == fam and again[g] == "dev")
        top = max((i["ammessi"] for i in info.values() if i["famiglia"] == fam), default=0)
        per_family[fam] = {"ammessi": admitted, "dev": dev, "ok": round(rule["quota_dev"] * admitted) <= dev
                           <= round(rule["quota_dev"] * admitted) + top}
    ok("dev per profondità: almeno l'obiettivo, superato al più dall'ultimo gruppo", all(v["ok"] for v in
                                                                                          per_family.values()),
       per_family)
    bad_tasks = [r["record"] for r in recs for t in r["compiti"]
                 if (t["compito"] in QUARANTINED_TASKS and t["stato"] == "ammesso")
                 or (t["compito"] not in QUARANTINED_TASKS and t["stato"] != r["stato"])]
    ok("compiti: shape_type mai ammessa, gli altri con lo stato del record", not bad_tasks, bad_tasks[:5])
    qa_types = Counter(q["tipo"] for r in recs for q in r["qa"])
    task_q = Counter()
    for r in recs:
        for t in r["compiti"]:
            if "domande" in t:
                task_q[t["compito"]] += t["domande"]
    ok("domande dei compiti uguali alle QA del record", qa_types == task_q)
    not_ok = [r["record"] for r in recs if r["stato"] == "ammesso"
              and (r["verifiche"]["coerenza_annotazioni"] != "verificata" or r["verita_sha256"] is None)]
    ok("ogni record ammesso ha la coerenza con le annotazioni verificata; la fedeltà all'immagine resta da verificare "
       "per tutti", not not_ok and all(r["verifiche"]["fedelta_immagine"] == "da_verificare" for r in recs), not_ok[:5])
    flagged = {"f1_mavis_run2": [r.get("id") for r in um.read_jsonl(ev["f1_mavis_run2"]["failures.jsonl"])],
               "mavis_caption_check_run1": [r.get("key", r.get("id")) for r in um.read_jsonl(
                   ev["mavis_caption_check_run1"]["issues.jsonl"])],
               "mavis_geometry_run1": [r.get("key", r.get("id")) for r in um.read_jsonl(
                   ev["mavis_geometry_run1"]["violations.jsonl"])]}
    by_key = {r["record"]: r for r in recs}
    unresolved = {k: sum(1 for x in v if x not in by_key) for k, v in flagged.items()}
    ok("liste storiche: ogni chiave segnalata risolta in un solo record, e non ammessa",
       not any(unresolved.values()) and all(by_key[x]["stato"] != "ammesso" for v in flagged.values() for x in v),
       unresolved)
    conflicted = {g: ms for g, ms in theirs.items()
                  if len(ms) > 1 and len({recs[n]["descrizione_sha256"] for n in ms}) > 1}
    spread = [recs[n]["record"] for ms in conflicted.values() for n in ms if recs[n]["stato"] == "ammesso"]
    ok("gruppi con descrizioni diverse: nessun record ammesso", not spread, spread[:5])
    ok("conteggi per split, profondità, stato, compito e motivo uguali a manifest.json", summarize(recs)
       == meta["conteggi"])
    bad_rank = [r["record"] for r in recs if r["rango_f0"] != um.f0_rank(rule["seme"], r["gruppo"])
                or r["rango_f0_sviluppo"] != (um.f0_rank(rule["seme"], r["gruppo"], "sviluppo")
                                              if r["split"] == "dev" else None)]
    ok("ranghi del campione candidato ricalcolati", not bad_rank, bad_rank[:5])
    derived_ok = []
    for name, entry in meta["derivati"].items():
        text = (folder / name).read_text(encoding="utf-8")
        derived_ok.append(mf.sha256_text(text) == entry["sha256"] and text.count("\n") == entry["righe"])
    ok("file derivati (keys/) con gli sha256 registrati", all(derived_ok))
    return {"manifest": str(folder), "manifest_sha256": m.manifest_sha256, "controlli": checks,
            "superati": sum(c["esito"] for c in checks), "totale": len(checks),
            "esito": all(c["esito"] for c in checks)}


def probe(args, plan: dict) -> None:
    p, s = plan["sets"]["train"][0]
    t0 = time.time()
    res = read_shard((str(p), 0, ()))
    rate = len(res["rows"]) / max(res["seconds"], 0.1)
    total = sum(x["samples"] for _, x in plan["sets"]["train"])
    report = {"schema": "mavis-manifest-probe/1", "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "tar": p.name, "campioni": len(res["rows"]), "secondi": res["seconds"],
              "campioni_al_secondo": round(rate, 1), "sha256_coincide": res["sha256"] == s["sha256"],
              "problemi": res["problems"][:5], "campioni_totali": total, "rss_max_mib": um.rss_mib(),
              "workers": args.workers, "stima_lettura_completa_secondi": round(total / rate / max(args.workers, 1), 1),
              "secondi_della_prova": round(time.time() - t0, 1),
              "senza_verita": sum(r[7] is None for r in res["rows"]),
              "domande_per_record": round(sum(len(r[11]) for r in res["rows"]) / max(len(res["rows"]), 1), 2)}
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "probe.json", report)
    print(json.dumps(report, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--evidence", required=True, type=Path, help="cartella outputs con le uscite storiche")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--name", default="mavis-geometry-candidato")
    ap.add_argument("--version", default="1")
    ap.add_argument("--dev-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("mavis_manifest.py va eseguito in un job SLURM")
    try:
        run(args)
    except um.Problem as exc:
        sys.exit(str(exc))


def run(args) -> dict | None:
    if args.report.exists() and any(args.report.iterdir()):
        raise um.Problem(f"{args.report} non è vuota")
    t0 = time.time()
    plan = shard_plan(args.shards)
    if args.probe:
        probe(args, plan)
        return None
    if args.out is None or args.summary is None:
        raise um.Problem("servono --out e --summary")
    if args.out.exists() and any(args.out.iterdir()):
        raise um.Problem(f"{args.out} non è vuota: una versione del manifest si scrive una volta sola")
    snapshot = snapshot_identity(plan["main"])
    if not all(s["esiste"] and s["blob_coincide"] for s in snapshot.values()):
        raise um.Problem(f"snapshot di Hugging Face diverso da quello dello sharding: {snapshot}")
    try:
        b = build(args, plan)
    except um.Problem as exc:
        raise um.Problem(f"controlli prima della scrittura falliti: {exc}") from None
    t_build = time.time() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    recs = []

    def lines():
        for rec in records(b):
            recs.append({k: rec[k] for k in ("record", "split", "stato", "famiglia", "gruppo", "compiti", "motivo",
                                             "misure_razionali_come_decimali")})
            yield json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    data_sha, n_lines = um.write_records_gz(args.out / "records.jsonl.gz", lines())
    keys = {"train.txt": [r["record"] for r in recs if r["split"] == "train" and r["stato"] == "ammesso"],
            "dev.txt": [r["record"] for r in recs if r["split"] == "dev" and r["stato"] == "ammesso"],
            "non_ammessi.txt": [r["record"] for r in recs if r["stato"] != "ammesso"]}
    (args.out / "keys").mkdir()
    derived = {}
    for name, ks in keys.items():
        text = "".join(k + "\n" for k in sorted(ks))
        (args.out / "keys" / name).write_text(text, encoding="utf-8")
        derived[f"keys/{name}"] = {**file_entry(args.out / "keys" / name, mf.sha256_text(text)), "righe": len(ks)}
    counts = summarize(recs)
    dev_target = {}
    for fam in sorted({i["famiglia"] for i in b["info"].values()}):
        admitted = sum(i["ammessi"] for i in b["info"].values() if i["famiglia"] == fam)
        got = sum(i["ammessi"] for g, i in b["info"].items() if i["famiglia"] == fam and b["split_of"][g] == "dev")
        dev_target[fam] = {"ammessi": admitted, "obiettivo": round(args.dev_fraction * admitted), "dev": got,
                           "quota": round(got / admitted, 5) if admitted else None}
    meta = {
        "schema": mf.SCHEMA, "nome": args.name, "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"),
        "codice": code_identity(), "sorgente": SOURCE, "revisione": b["revision"],
        "identita": {"repository": "CaraJ/MAVIS-Geometry", "revisione": b["revision"], "snapshot": snapshot,
                     "catalogo": "downloads/catalog_status.tsv (hash dello snapshot verificati nei job 29413 e 29554)"},
        "indice": b["index_name"],
        "ingressi": {"sharding": plan["files"], "sharding_job": plan["main"].get("job_id"),
                     "variante": plan["main"].get("variant"), "max_patches": plan["main"].get("max_patches"),
                     "esclusi_allo_sharding": b["reproduced"]["sharding"]},
        "shards": b["shards_meta"],
        "liste_importate": {
            name: {"file": {n: file_entry(p) for n, p in files.items()},
                   "job": (b["reports"].get(name) or {}).get("job_id")}
            for name, files in b["ev"].items()},
        "riconciliazione": b["reproduced"],
        "tokenizer": None, "template": None, "contratti": None,
        "split": {"valori": list(mf.SPLITS), "unita": "gruppo", "quota_dev": args.dev_fraction, "seme": args.seed,
                  "strati": "profondità d0-d3 (un gruppo con più profondità va in quella minima)",
                  "regola": "per profondità e in ordine di sha256({seme, gruppo}), dev finché i record ammessi di dev "
                            "sono meno di round(quota * ammessi della profondità); il resto train "
                            "(manifest.assign_splits)",
                  "dev_per_profondita": dev_target},
        "rango_f0": {"campione": um.SAMPLE, "seme": args.seed,
                     "regola": "sha256 del JSON canonico {campione, seme, gruppo}; per lo sviluppo anche ruolo "
                               "'sviluppo' (F0_CAMPIONE_CANDIDATO §1)"},
        "stati": {"ammesso": "candidato del manifest, non certificato: coerente con le annotazioni, fedeltà "
                             "all'immagine da verificare",
                  "quarantena": "verità mancante, andata e ritorno imperfetto, dimensioni diverse, segnalato da una "
                                "lista storica, o verità diverse nella stessa immagine o figura",
                  "escluso": "nessuna regola di esclusione per questa sorgente"},
        "compiti_in_quarantena": QUARANTINED_TASKS,
        "conteggi": counts,
        "records": {"file": "records.jsonl.gz", "sha256": data_sha, "righe": n_lines},
        "derivati": derived,
        "fuori_dal_sottoinsieme": NOT_COVERED,
        "verifiche_mancanti": MISSING_CHECKS,
        "secondi_costruzione": round(t_build, 1), "rss_max_mib": um.rss_mib()}
    write_new_json(args.out / "manifest.json", meta)
    ev = b["ev"]
    del recs, b
    verification = verify(args.out, ev)
    verification.update({"job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
                         "secondi_totali": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()})
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "verifica.json", verification)
    summary = {"schema": "limite-vl-manifest-riepilogo/1", "manifest": um.rel(args.out), "nome": args.name,
               "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"),
               "manifest_sha256": verification["manifest_sha256"], "records_sha256": data_sha, "righe": n_lines,
               "sorgente": SOURCE, "revisione": meta["revisione"], "shards": meta["shards"],
               "liste_importate": {k: {n: e["sha256"] for n, e in v["file"].items()}
                                   for k, v in meta["liste_importate"].items()},
               "split": meta["split"]["dev_per_profondita"], "conteggi": counts,
               "riconciliazione": {k: v for k, v in meta["riconciliazione"].items() if k != "run2"}
               | {"run2_uguale": True, "run2_esempi_riprodotti": meta["riconciliazione"]["run2"]["esempi_riprodotti"]},
               "controlli": [{k: c[k] for k in ("controllo", "esito")} for c in verification["controlli"]],
               "esito_verifica": verification["esito"], "compiti_in_quarantena": QUARANTINED_TASKS,
               "fuori_dal_sottoinsieme": NOT_COVERED, "verifiche_mancanti": MISSING_CHECKS,
               "nota": "organizza candidati ed evidenze: non certifica la qualità e non autorizza il training"}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.summary, summary)
    print(json.dumps({"righe": n_lines, "conteggi": counts["per_split_famiglia_stato"], "split": dev_target,
                      "verifica": f"{verification['superati']}/{verification['totale']}",
                      "secondi": verification["secondi_totali"], "rss_max_mib": verification["rss_max_mib"]},
                     ensure_ascii=False))
    if not verification["esito"]:
        raise um.Problem("verifica dopo la scrittura fallita: vedi " + str(args.report / "verifica.json"))
    return summary


if __name__ == "__main__":
    main()
