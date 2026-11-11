#!/usr/bin/env python3
"""Manifest candidato di PixMo-Cap (S1 §§1-3, 7, 9; docs/F0_CAMPIONE_CANDIDATO.md §7) dall'inventario, dalla verifica e
dalla scansione di scripts/pixmo_inventory.py (job 30813), senza rileggere parquet né immagini. Solo in un job SLURM CPU:
ingressi in sola lettura, uscite in cartelle nuove. Organizza candidati ed evidenze: non certifica le didascalie, non
assegna gold e non autorizza il training.

Ingressi, verificati prima di derivare il manifest:
- inventario (--inventory: records.jsonl.gz e summary.json), verifica (--verifica: report.json senza controlli falliti,
  duplicati.jsonl.gz, proposte_di_trattamento.jsonl.gz) e scansione (--scan: records.jsonl.gz e summary.json) dello
  stesso job, completi, della revisione --revision, coi digest di summary.json e con quelli attesi
  (--inventory-sha256, --scan-sha256); scansione contro l'indice --index-sha256;
- sharding (--shards: plan.json, shards.jsonl, manifest.json, job 29560): sorgente e revisione, tar presenti con la
  dimensione registrata; i loro sha256 vengono dal registro, già ritrovati nella lettura del job 30813 (nessun nuovo
  hash dei 20 GB), e il reader li verifica a ogni caricamento;
- run4 (--run4) e liste storiche (--historical) solo per riferimento: la verifica del job 30813 le ha riconciliate.

Record: uno per record degli shard (insieme `train`, compito pixmo_caption), con shard, posizione, sha256 del JSON e del
JPEG, pixel del JPEG, didascalia del loader, URL (sha256), richiesta del loader, originale (file, riga, pixel). La riga
illeggibile scartata dallo sharding non ha record: è contata in `scarti_dello_sharding`.

Gruppi (S1 §3): unioni transitive dei legami esatti fra record:
- pixel identici dell'immagine originale;
- pixel identici del JPEG dello shard;
- byte identici del JPEG dello shard;
- stesso URL originale, non vuoto, per uguaglianza esatta.
Gli input finali identici stanno quindi nello stesso gruppo anche con originali diversi.

Blocchi diretti di un record (S2, politica di panel_decontam.decide: stato per gruppo, gravità che non diminuisce):
- `decontaminazione_confermata` (pixel identici al pannello): escluso;
- `decontaminazione_candidato` (pHash entro 4 bit o 10-gram): quarantena; `decontaminazione_non_verificabile`: quarantena;
- `possibile_segnaposto`: membro di un gruppo di immagini originali identiche (pixel) da almeno --min-segnaposto righe,
  dall'elenco completo dei duplicati della verifica: quarantena cautelativa, non un errore dimostrato;
- `orientamento_exif`: orientamento EXIF da 2 a 8, che lo shard non applica: quarantena (nessuna rotazione);
- `difetto_strutturale`: difetto dimostrato dall'inventario: escluso.
I segnali di run4 restano evidenze di rischio: nessun effetto sullo stato. Lo stato del gruppo è il più grave fra quelli
dei suoi record, e ogni record prende il più grave fra il proprio e quello del gruppo: `blocchi.diretti` sono i motivi del
record, `blocchi.ereditati` quelli degli altri record del gruppo che il record non ha. Un record senza blocchi è
`ammesso`: candidato per i controlli operativi effettuati (collegamento, integrità, decontaminazione nei controlli
dichiarati); la fedeltà delle didascalie ai pixel resta da valutare e il training richiede una decisione successiva.

Split (manifest.assign_splits): dev per circa --dev-fraction dei record ammessi, gruppi indivisibili in ordine di
sha256({seme, gruppo}) con --seed; il resto train. Ranghi del campione candidato di F0 come per UniMER e MAVIS.

Comandi:
  costruzione  ingressi verificati, manifest scritto in --out (records.jsonl.gz, manifest.json, keys/; una volta sola),
               verifica rileggendolo -> --report/verifica.json, riepilogo per git --summary.
  verifica     rilegge un manifest già scritto con gli stessi ingressi -> --report/verifica.json.

    srun $RUN_PYTHON -u -B scripts/pixmo_manifest.py costruzione --inventory DIR --verifica DIR --scan DIR \\
        --shards DIR --revision HEX --inventory-sha256 HEX --scan-sha256 HEX --index-sha256 HEX --run4 DIR \\
        --historical DIR DIR --out DIR --report DIR --summary FILE --gruppi-segnaposto N --righe-segnaposto N \\
        --exif-attesi N --candidati-attesi N [--name pixmo-cap-candidato] [--version 1] [--dev-fraction 0.05] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import pixmo_inventory as pi  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, sha256_file, write_new_json  # noqa: E402

SOURCE = pi.SOURCE  # "pixmo-cap"
TASK = "pixmo_caption"
INSIEME = "train"
SEVERITY = um.SEVERITY
CODES = {  # codice del blocco -> (stato minimo, motivo)
    "decontaminazione_confermata": ("escluso", "decontaminazione_pannello: pixel identici a un'immagine di pannello-v3 "
                                               "(legame sull'immagine, non sulla domanda)"),
    "difetto_strutturale": ("escluso", "difetto strutturale dimostrato dall'inventario"),
    "decontaminazione_candidato": ("quarantena", "decontaminazione_pannello: candidato da verificare (pHash entro 4 bit o "
                                                 "10-gram in comune)"),
    "decontaminazione_non_verificabile": ("quarantena", "decontaminazione_pannello: controllo non eseguibile"),
    "possibile_segnaposto": ("quarantena", "quarantena cautelativa: possibile segnaposto, gruppo di immagini originali "
                                           "identiche con almeno tre righe (docs/F0_CAMPIONE_CANDIDATO.md §7.2); non è "
                                           "un errore dimostrato"),
    "orientamento_exif": ("quarantena", "orientamento EXIF che lo shard non applica: da decidere, nessuna rotazione "
                                        "applicata"),
}
OUTCOME_CODE = {"corrispondenza_confermata": "decontaminazione_confermata", "candidato": "decontaminazione_candidato",
                "non_verificabile": "decontaminazione_non_verificabile"}
EXIF_BLOCKED = frozenset(range(2, 9))  # 1 = nessuna trasformazione; 0 e i valori fuori scala non sono validi
RELATIONS = ("pixel_originale", "pixel_shard", "jpeg_shard", "url")
INDEX_NAME = ("legami esatti di PixMo-Cap: pixel dell'immagine originale, pixel e byte del JPEG dello shard, stesso URL "
              "originale non vuoto (uguaglianza esatta); unioni transitive")
STATES_DOC = {
    "ammesso": "candidato per i controlli operativi effettuati (collegamento originale -> shard, integrità, "
               "decontaminazione contro pannello-v3 nei controlli dichiarati), per il compito pixmo_caption; la fedeltà "
               "delle didascalie ai pixel resta da valutare e il training richiede una decisione successiva",
    "quarantena": "candidati della decontaminazione, possibili segnaposto (cautelativa, non un errore dimostrato), "
                  "orientamento EXIF non applicato, e i record del loro gruppo",
    "escluso": "identità d'immagine confermata col pannello o difetto strutturale dimostrato, e i record del loro gruppo "
               "(quanti, nei conteggi)",
}
NOT_COVERED = [
    "variant_id e uid (S1 §1): dipendono dal contratto di risposta (S3, D3); l'ammissione è per record e compito, la "
    "richiesta è scelta dal loader con la chiave (data.pick_prompt), registrata in `richiesta`",
    "hash dei token di input e di target, tokenizer, template e contratti (S1 §2, §7)",
    "indice globale fra sorgenti (S1 §4): i gruppi sono i legami esatti dentro PixMo-Cap",
    "campo gold: nessuno; una didascalia di PixMo è al più un target controllato (pipeline §5.7)",
]
MISSING_CHECKS = [
    "fedeltà delle didascalie all'immagine (oggetti, posizioni, relazioni, attributi, testi): da valutare con un campione "
    "e immagini (D5); i segnali di run4 sono solo evidenze di rischio",
    "verifica a vista dei 248 candidati della decontaminazione e dei gruppi di possibili segnaposto (nessuno è risolto)",
    "corrispondenza fra l'immagine riscaricata dagli URL e quella vista dagli annotatori",
    "orientamento EXIF: quale versione abbiano visto gli annotatori; nessuno shard rigenerato",
    "leggibilità dei dettagli a 1024 patch; i 50 parquet non locali",
]


class Problem(Exception):
    """Ingresso o derivazione incoerente: il manifest non si scrive (o la verifica fallisce)."""


def sha(text: str | None) -> str | None:
    return None if not text else mf.sha256_text(text)


# ---------------------------------------------------------------- ingressi

def load_json(path: Path) -> dict:
    if not path.is_file():
        raise Problem(f"ingresso mancante: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def inputs(args) -> dict:
    """Ingressi verificati: digest, revisione, completezza e compatibilità degli artefatti del job dell'inventario."""
    inv_sum = load_json(args.inventory / "summary.json")
    scan_sum = load_json(args.scan / "summary.json")
    report = load_json(args.verifica / "report.json")
    problems = []
    if inv_sum.get("schema") != pi.SCHEMA_INVENTORY or scan_sum.get("schema") != pi.SCHEMA_SCAN \
            or report.get("schema") != pi.SCHEMA_REPORT:
        problems.append("schema di inventario, scansione o verifica non riconosciuto")
    for what, s in (("inventario", inv_sum), ("scansione", scan_sum), ("verifica", report)):
        if s.get("revisione") != args.revision or s.get("parziale") is not False:
            problems.append(f"{what}: revisione {s.get('revisione')!r} o parziale {s.get('parziale')!r}")
    jobs = {inv_sum.get("job_id"), scan_sum.get("job_id"), report.get("job_id")}
    if len(jobs) != 1:
        problems.append(f"inventario, scansione e verifica di job diversi: {sorted(map(str, jobs))}")
    if report.get("falliti"):
        problems.append(f"verifica con controlli falliti: {report['falliti'][:3]}")
    files = {"inventario": args.inventory / "records.jsonl.gz", "scansione": args.scan / "records.jsonl.gz"}
    digests = {k: sha256_file(p) for k, p in files.items()}
    expected = {"inventario": args.inventory_sha256, "scansione": args.scan_sha256}
    for k in files:
        recorded = (inv_sum if k == "inventario" else scan_sum)["files"]["records.jsonl.gz"]["sha256"]
        if digests[k] != recorded or digests[k] != expected[k]:
            problems.append(f"{k}: records.jsonl.gz con sha256 {digests[k]}, registrato {recorded}, atteso {expected[k]}")
        by_report = [c["dettaglio"] for c in report["controlli"]
                     if c["controllo"].startswith(f"{k}: records.jsonl.gz con lo sha256")]
        if by_report != [digests[k]]:
            problems.append(f"{k}: sha256 diverso da quello ritrovato dalla verifica ({by_report})")
    verifica_files = {}
    for name in ("duplicati.jsonl.gz", "proposte_di_trattamento.jsonl.gz"):
        got = sha256_file(args.verifica / name)
        if got != report["files"][name]["sha256"]:
            problems.append(f"{name}: sha256 {got} diverso da quello del rapporto")
        verifica_files[name] = file_entry(args.verifica / name, got)
    if (scan_sum.get("indice") or {}).get("sha256") != args.index_sha256 \
            or scan_sum.get("convenzione_immagini") != pi.CONV:
        problems.append("scansione contro un indice o una convenzione diversi da quelli attesi")
    try:
        sh = pi.read_sharding(args.shards, args.revision)
    except pi.Failed as exc:
        raise Problem(f"sharding: {exc}") from None
    if sh["digest_ricalcolato"] != sh["piano"].get("digest"):
        problems.append("sharding: digest del piano diverso da plan.json")
    read_units = {u["shard"]: u for u in inv_sum["unita"]}
    shards = []
    for u in sh["unita"]:
        reg, got = u["registro"], read_units.get(u["nome"])
        if got is None or not got["uguale_al_registro"] or got["sha256_letto"] != reg["sha256"] \
                or got["byte_letti"] != reg["bytes"] or got["campioni_letti"] != reg["samples"]:
            problems.append(f"{u['nome']}: non ritrovato dall'inventario con sha256, byte e campioni del registro")
        if not u["tar"].is_file() or u["tar"].stat().st_size != reg["bytes"]:
            problems.append(f"{u['nome']}: tar mancante o con un'altra dimensione")
        shards.append({"insieme": INSIEME, "nome": u["nome"], "percorso": um.rel(u["tar"]), "sha256": reg["sha256"],
                       "bytes": reg["bytes"], "record": reg["samples"]})
    if len(read_units) != len(sh["unita"]):
        problems.append("l'inventario non ha letto tutti gli shard del piano")
    if problems:
        raise Problem("; ".join(problems[:8]))
    run4 = {n: file_entry(args.run4 / n) for n in ("report.json", "flagged.jsonl")}
    historical = {f"{Path(d).parent.name}/{Path(d).name}": {n: file_entry(Path(d) / n)
                                                          for n in ("exclude.txt", "exclude_final.txt")}
                  for d in args.historical}
    return {"inv_sum": inv_sum, "scan_sum": scan_sum, "report": report, "digests": digests, "sharding": sh,
            "shards": shards, "verifica_files": verifica_files, "run4": run4, "historical": historical,
            "files": {"inventario": {"summary.json": file_entry(args.inventory / "summary.json"),
                                     "records.jsonl.gz": file_entry(files["inventario"], digests["inventario"])},
                      "scansione": {"summary.json": file_entry(args.scan / "summary.json"),
                                    "records.jsonl.gz": file_entry(files["scansione"], digests["scansione"])},
                      "verifica": {"report.json": file_entry(args.verifica / "report.json"), **verifica_files},
                      "sharding": {n: file_entry(args.shards / n) for n in ("plan.json", "shards.jsonl",
                                                                           "manifest.json")}}}


def rows_of(args, inp: dict) -> tuple[list[dict], list[dict]]:
    """Record compatti degli shard (nell'ordine dei file e delle righe, che è quello degli shard) e scarti dello
    sharding, dall'inventario e dalla scansione, con le coerenze richieste prima di derivare."""
    shard_rel = {s["nome"]: s["percorso"] for s in inp["shards"]}
    scan = {r["chiave"]: r for r in pi.read_gz(args.scan / "records.jsonl.gz")}
    rows, discarded, problems = [], [], []
    digests = mf.Digests()
    n = 0
    for r in pi.read_gz(args.inventory / "records.jsonl.gz"):
        n += 1
        k = r["chiave"]
        if r["record_id"] != mf.record_id(digests, SOURCE, args.revision, k):
            problems.append(f"{k}: record_id diverso da quello ricalcolato")
        s = scan.get(k)
        if s is None:
            problems.append(f"{k}: senza riga nella scansione")
            continue
        if r.get("shard") is None:
            discarded.append({"chiave": k, "record_id": r["record_id"], "esito_sharding": r["esito_sharding"],
                              "file": r["parquet"]["file"], "riga": r["parquet"]["riga"], "difetti": r["difetti"],
                              "originale": {x: r["originale"].get(x) for x in ("lettura", "byte_sha256", "byte",
                                                                                "errore")},
                              "esito_decontaminazione": pi.outcome(s)})
            continue
        sh_, img, t = r["shard"], r["immagine_shard"], r.get("trasformazione") or {}
        if s.get("shard") != sh_["nome"] or s.get("posizione") != sh_["posizione"] \
                or s.get("didascalia_sha256") != r["didascalia"].get("loader_sha256") \
                or (s.get("immagine_shard") or {}).get("pixel_sha256") != img.get("pixel_sha256"):
            problems.append(f"{k}: scansione e inventario non coincidono")
        if t.get("byte_uguali_allo_shard") is not True:
            problems.append(f"{k}: trasformazione dello sharding non riprodotta")
        rows.append({"key": k, "rid": r["record_id"], "shard": shard_rel[sh_["nome"]], "pos": sh_["posizione"],
                     "json": sh_["json_sha256"], "jpg": sh_["jpg_sha256"], "px_shard": img.get("pixel_sha256"),
                     "dims": img.get("dimensioni"), "px_orig": r["originale"].get("pixel_sha256"),
                     "file": r["parquet"]["file"], "riga": r["parquet"]["riga"], "url": r.get("url") or None,
                     "caption": r["didascalia"].get("loader_sha256"), "richiesta": r["richiesta"],
                     "exif": r["originale"].get("exif_orientamento"), "defects": list(r["difetti"]),
                     "signals": list(r["segnali"]["tipi"]), "outcome": pi.outcome(s),
                     "matches": dict(s.get("corrispondenze") or {})})
    if n != inp["inv_sum"]["righe"] or len(scan) != n:
        problems.append(f"righe: inventario {n}, scansione {len(scan)}, dichiarate {inp['inv_sum']['righe']}")
    per_shard = Counter(x["shard"] for x in rows)
    for s in inp["shards"]:
        pos = sorted(x["pos"] for x in rows if x["shard"] == s["percorso"])
        if per_shard[s["percorso"]] != s["record"] or pos != list(range(s["record"])):
            problems.append(f"{s['nome']}: posizioni dei record diverse da 0..{s['record'] - 1}")
    if problems:
        raise Problem("; ".join(problems[:8]))
    return rows, discarded


def placeholder_groups(args, rows: list[dict]) -> list[dict]:
    """Gruppi cautelativi: dall'elenco completo dei duplicati della verifica, i gruppi per pixel dell'immagine originale
    con almeno --min-segnaposto righe; ogni gruppo deve essere esattamente l'insieme dei record con quei pixel."""
    by_px = defaultdict(list)
    for x in rows:
        by_px[x["px_orig"]].append(x["key"])
    px_of = {x["key"]: x["px_orig"] for x in rows}
    out, problems = [], []
    for g in pi.read_gz(args.verifica / "duplicati.jsonl.gz"):
        if g["campo"] != "pixel_originale" or g["righe"] < args.min_segnaposto:
            continue
        values = {px_of.get(k) for k in g["chiavi"]}
        if len(values) != 1 or None in values or sorted(by_px[next(iter(values))]) != sorted(g["chiavi"]) \
                or len(g["chiavi"]) != g["righe"]:
            problems.append(f"gruppo di {g['righe']} righe ({g['chiavi'][:2]}): non coincide coi pixel dell'inventario")
        out.append(g)
    if problems:
        raise Problem("; ".join(problems[:5]))
    return out


# ---------------------------------------------------------------- derivazione (pure, collaudate in tests/)

def link_lists(rows: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """Legami per relazione (uguaglianza esatta; valori vuoti esclusi)."""
    return {"pixel_originale": um.same_key_links([x["px_orig"] for x in rows]),
            "pixel_shard": um.same_key_links([x["px_shard"] for x in rows]),
            "jpeg_shard": um.same_key_links([x["jpg"] for x in rows]),
            "url": um.same_key_links([x["url"] or None for x in rows])}


def group_sizes(rep: list[int]) -> dict:
    sizes = Counter(Counter(rep).values())
    multi = {s: n for s, n in sizes.items() if s > 1}
    return {"gruppi": sum(sizes.values()), "gruppi_da_almeno_due": sum(multi.values()),
            "record_in_gruppi_da_almeno_due": sum(s * n for s, n in multi.items()), "massimo": max(sizes, default=0)}


def union_stats(rows: list[dict], links: dict) -> dict:
    """Conteggi prima e dopo le unioni: ogni relazione da sola, poi le relazioni aggiunte una alla volta."""
    n = len(rows)
    out = {"record": n, "per_relazione": {}, "cumulative": []}
    acc: list = []
    for name in RELATIONS:
        values = Counter(v for v in ([x["px_orig"] for x in rows] if name == "pixel_originale" else
                                     [x["px_shard"] for x in rows] if name == "pixel_shard" else
                                     [x["jpg"] for x in rows] if name == "jpeg_shard" else
                                     [x["url"] for x in rows]) if v)
        repeated = {v: c for v, c in values.items() if c > 1}
        out["per_relazione"][name] = {"legami": len(links[name]), "valori_ripetuti": len(repeated),
                                      "record_coinvolti": sum(repeated.values()),
                                      **group_sizes(um.components(n, links[name]))}
        acc += links[name]
        out["cumulative"].append({"fino_a": name, **group_sizes(um.components(n, acc))})
    return out


def decide(rows: list[dict], rep: list[int], placeholder_keys: set[str]) -> list[dict]:
    """Blocchi diretti, stato del gruppo e stato di ogni record, con i blocchi ereditati (quelli degli altri record del
    gruppo che il record non ha). Nessun effetto dei segnali di run4."""
    direct = []
    for x in rows:
        codes = []
        if x["outcome"] in OUTCOME_CODE:
            codes.append(OUTCOME_CODE[x["outcome"]])
        if x["key"] in placeholder_keys:
            codes.append("possibile_segnaposto")
        if x["exif"] in EXIF_BLOCKED:
            codes.append("orientamento_exif")
        if x["defects"]:
            codes.append("difetto_strutturale")
        direct.append(codes)
    own = [um.worst("ammesso", *(CODES[c][0] for c in codes)) for codes in direct]
    members = defaultdict(list)
    for i, g in enumerate(rep):
        members[g].append(i)
    group_state = {g: um.worst("ammesso", *(own[i] for i in ms)) for g, ms in members.items()}
    out = []
    for i, x in enumerate(rows):
        ms = members[rep[i]]
        inherited = []
        if group_state[rep[i]] != "ammesso":
            by_code = defaultdict(list)
            for j in ms:
                if j != i:
                    for c in direct[j]:
                        if c not in direct[i]:
                            by_code[c].append(rows[j]["key"])
            inherited = [{"codice": c, "da": sorted(ks)[:10], "n": len(ks)} for c, ks in sorted(by_code.items())]
        state = um.worst(own[i], group_state[rep[i]])
        motive = [CODES[c][1] for c in direct[i]]
        motive += [f"dal gruppo: {CODES[e['codice']][1]} (record {', '.join(e['da'][:3])}"
                   f"{', …' if e['n'] > 3 else ''})" for e in inherited]
        out.append({"diretti": direct[i], "ereditati": inherited, "stato": state, "motivo": motive,
                    "stato_del_gruppo": group_state[rep[i]]})
    return out


def summarize(recs: list[dict]) -> dict:
    """Conteggi per split e stato, compito, motivo diretto ed ereditato, sovrapposizioni, gruppi, segnali di run4."""
    c = {k: Counter() for k in ("per_split_stato", "per_compito_split_stato", "per_motivo_diretto",
                                "per_motivo_ereditato", "sovrapposizioni_dirette", "gruppi_per_split",
                                "gruppi_per_stato", "segnali_run4_per_stato", "non_ammessi_per_origine")}
    groups: dict[str, tuple] = {}
    for r in recs:
        c["per_split_stato"][f"{r['split']}/{r['stato']}"] += 1
        for t in r["compiti"]:
            c["per_compito_split_stato"][f"{t}/{r['split']}/{r['stato']}"] += 1
        d, e = r["blocchi"]["diretti"], [x["codice"] for x in r["blocchi"]["ereditati"]]
        for x in d:
            c["per_motivo_diretto"][x] += 1
        for x in e:
            c["per_motivo_ereditato"][x] += 1
        if d:
            c["sovrapposizioni_dirette"]["+".join(sorted(d))] += 1
        if r["stato"] != "ammesso":
            c["non_ammessi_per_origine"][f"{r['stato']}/{'diretto' if d else 'solo_ereditato'}"] += 1
        signals = [e2 for e2 in r["evidenze"] if e2["lista"] == "pixmo_check_run4"]
        if signals:
            c["segnali_run4_per_stato"][f"{r['split']}/{r['stato']}"] += 1
        g = groups.setdefault(r["gruppo"], (r["split"], r["stato"]))
        if SEVERITY[r["stato"]] > SEVERITY[g[1]]:
            groups[r["gruppo"]] = (g[0], r["stato"])
    for split, state in groups.values():
        c["gruppi_per_split"][split] += 1
        c["gruppi_per_stato"][f"{split}/{state}"] += 1
    return {k: dict(sorted(v.items())) for k, v in c.items()}


# ---------------------------------------------------------------- costruzione

def build(args, inp: dict) -> dict:
    rows, discarded = rows_of(args, inp)
    placeholders = placeholder_groups(args, rows)
    ph_keys = {k for g in placeholders for k in g["chiavi"]}
    counts = {"gruppi_segnaposto": len(placeholders), "righe_segnaposto": len(ph_keys),
              "orientamento_exif": sum(1 for x in rows if x["exif"] in EXIF_BLOCKED),
              "candidati": sum(1 for x in rows if x["outcome"] == "candidato")}
    wanted = {"gruppi_segnaposto": args.gruppi_segnaposto, "righe_segnaposto": args.righe_segnaposto,
              "orientamento_exif": args.exif_attesi, "candidati": args.candidati_attesi}
    if counts != wanted:
        raise Problem(f"conteggi diversi da quelli autorizzati: trovati {counts}, attesi {wanted}")
    proposals = {p["chiave"] for p in pi.read_gz(args.verifica / "proposte_di_trattamento.jsonl.gz")
                 if p["esito"] == "candidato"}
    if proposals != {x["key"] for x in rows if x["outcome"] == "candidato"}:
        raise Problem("candidati della scansione diversi da quelli delle proposte di trattamento")
    links = link_lists(rows)
    stats = union_stats(rows, links)
    rep = um.components(len(rows), [p for name in RELATIONS for p in links[name]])
    decisions = decide(rows, rep, ph_keys)
    digests = mf.Digests()
    index = {"nome": INDEX_NAME, "inventario_sha256": inp["digests"]["inventario"],
             "scansione_sha256": inp["digests"]["scansione"]}
    members = defaultdict(list)
    for i, g in enumerate(rep):
        members[g].append(i)
    gid_of = {g: mf.group_id(digests, index, sorted({rows[i]["rid"] for i in ms})) for g, ms in members.items()}
    info = {gid_of[g]: {"famiglia": SOURCE, "valutazione": False,
                        "ammessi": sum(decisions[i]["stato"] == "ammesso" for i in ms)} for g, ms in members.items()}
    split_of = mf.assign_splits(info, args.dev_fraction, args.seed)
    return {"rows": rows, "discarded": discarded, "placeholders": placeholders, "counts": counts, "stats": stats,
            "rep": rep, "decisions": decisions, "gid_of": gid_of, "info": info, "split_of": split_of, "index": index}


def records(b: dict, args):
    """Righe del manifest nell'ordine degli shard e delle posizioni."""
    for i, x in enumerate(b["rows"]):
        g = b["gid_of"][b["rep"][i]]
        d = b["decisions"][i]
        ev = [{"lista": "decontaminazione_pannello_pixmo", "esito": x["outcome"], "corrispondenze": x["matches"]}]
        if x["signals"]:
            ev.append({"lista": "pixmo_check_run4", "esito": "segnali", "tipi": x["signals"],
                       "effetto": "nessuno: evidenza di rischio, non un difetto"})
        if "possibile_segnaposto" in d["diretti"]:
            ev.append({"lista": "duplicati_inventario", "esito": "gruppo_cautelativo",
                       "regola": f"pixel_originale, almeno {args.min_segnaposto} righe"})
        w, h = x["dims"] or (0, 0)  # JPEG illeggibile: record con un difetto strutturale, escluso
        yield {"record_id": x["rid"], "sorgente": SOURCE, "revisione": args.revision, "record": x["key"],
               "insieme": INSIEME, "shard": x["shard"], "indice": x["pos"], "compiti": [TASK], "gruppo": g,
               "split": b["split_of"][g], "stato": d["stato"], "motivo": d["motivo"], "gold": None,
               "blocchi": {"diretti": d["diretti"], "ereditati": d["ereditati"]},
               "immagini": [{"sha256": x["jpg"], "ordine": 0, "griglia": [h // 16, w // 16],
                             "pixel_sha256": x["px_shard"]}],
               "json_sha256": x["json"], "didascalia_sha256": x["caption"], "url_sha256": sha(x["url"]),
               "richiesta": x["richiesta"],
               "originale": {"file": x["file"], "riga": x["riga"], "pixel_sha256": x["px_orig"],
                             "orientamento_exif": x["exif"]},
               "evidenze": ev, "trasformazione": None, "genitori": [],
               "rango_f0": um.f0_rank(args.seed, g),
               "rango_f0_sviluppo": um.f0_rank(args.seed, g, "sviluppo") if b["split_of"][g] == "dev" else None}


# ---------------------------------------------------------------- verifica dopo la scrittura

def verify(args, folder: Path, inp: dict) -> dict:
    """Controlli rileggendo il manifest scritto (manifest.load) e gli ingressi, senza le strutture della costruzione:
    record e digest uguali all'inventario, scarti, partizione e group_id ricalcolati dai campi dei record, ogni legame
    dentro un gruppo e uno split, split ricalcolato, blocchi e stati ricalcolati dalle evidenze originali, segnali di
    run4 senza effetto, conteggi, ranghi e file derivati."""
    checks = []

    def ok(name: str, cond, detail=None) -> None:
        checks.append({"controllo": name, "esito": bool(cond), "dettaglio": detail})
        print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail is not None else ""),
              flush=True)

    m = mf.load(folder)
    ok("manifest letto e verificato da manifest.load (sha256 dei record, record_id ricalcolati, un solo split per "
       "gruppo, conteggi per shard)", True, m.manifest_sha256)
    meta = m.meta
    recs = sorted((r for rows in m.by_shard.values() for r in rows.values()), key=lambda r: (r["shard"], r["indice"]))
    ok("identità: sorgente, revisione e repository; solo l'insieme train e il compito pixmo_caption",
       meta["sorgente"] == SOURCE and meta["revisione"] == args.revision
       and meta["identita"]["repository"] == pi.REPO
       and all(r["insieme"] == INSIEME and r["compiti"] == [TASK] for r in recs))
    rows, discarded = rows_of(args, inp)
    by_key = {x["key"]: x for x in rows}
    differ = [r["record"] for r in recs if (x := by_key.get(r["record"])) is None
              or (r["shard"], r["indice"], r["json_sha256"], r["immagini"][0]["sha256"], r["immagini"][0]["pixel_sha256"],
                  r["didascalia_sha256"], r["url_sha256"], r["originale"]["pixel_sha256"], r["richiesta"])
              != (x["shard"], x["pos"], x["json"], x["jpg"], x["px_shard"], x["caption"], sha(x["url"]), x["px_orig"],
                  x["richiesta"])]
    ok("record: uno per record degli shard dell'inventario, con shard, posizione, sha256 di JSON e JPEG, pixel, "
       "didascalia, URL, originale e richiesta uguali", len(recs) == len(rows) and not differ,
       {"record": len(recs), "inventario": len(rows), "diversi": len(differ), "esempi": differ[:5]})
    gone = {d["chiave"] for d in discarded}
    ok("scarti dello sharding: fuori dai record e contati in manifest.json",
       not gone & {r["record"] for r in recs} and sorted(d["chiave"] for d in meta["scarti_dello_sharding"]) == sorted(gone),
       sorted(gone))
    # partizione ricalcolata dai campi del manifest
    fields = {"pixel_originale": [r["originale"]["pixel_sha256"] for r in recs],
              "pixel_shard": [r["immagini"][0]["pixel_sha256"] for r in recs],
              "jpeg_shard": [r["immagini"][0]["sha256"] for r in recs], "url": [r["url_sha256"] for r in recs]}
    rep = um.components(len(recs), [p for name in RELATIONS for p in um.same_key_links(fields[name])])
    ours, theirs = defaultdict(set), defaultdict(set)
    for i, r in enumerate(recs):
        ours[rep[i]].add(i)
        theirs[r["gruppo"]].add(i)
    ok("gruppi ricalcolati dai pixel dell'originale e dello shard, dai byte del JPEG e dall'URL: stessa partizione",
       sorted(map(sorted, ours.values())) == sorted(map(sorted, theirs.values())), len(theirs))
    digests = mf.Digests()
    bad_gid = sum(mf.group_id(digests, meta["indice"], sorted(recs[i]["record_id"] for i in ms)) != recs[next(iter(ms))]["gruppo"]
                  for ms in theirs.values())
    ok("group_id ricalcolati dai membri e dall'indice", bad_gid == 0, bad_gid)
    crossing = {}
    for name, values in fields.items():
        seen: dict[str, tuple] = {}
        bad = 0
        for r, v in zip(recs, values):
            if v:
                if seen.setdefault(v, (r["gruppo"], r["split"])) != (r["gruppo"], r["split"]):
                    bad += 1
        crossing[name] = bad
    ok("ogni legame dentro un solo gruppo e un solo split: gli input finali identici (pixel e byte del JPEG) non "
       "attraversano train e dev", not any(crossing.values()), crossing)
    # blocchi e stati ricalcolati dalle evidenze originali
    placeholders = placeholder_groups(args, rows)
    ph_keys = {k for g in placeholders for k in g["chiavi"]}
    order = {x["key"]: i for i, x in enumerate(rows)}
    rep_rows = [0] * len(rows)
    for i, r in enumerate(recs):
        rep_rows[order[r["record"]]] = rep[i]
    expected = decide(rows, rep_rows, ph_keys)
    bad_state = [r["record"] for r in recs
                 if (e := expected[order[r["record"]]])["stato"] != r["stato"] or e["diretti"] != r["blocchi"]["diretti"]
                 or e["ereditati"] != r["blocchi"]["ereditati"] or e["motivo"] != r["motivo"]]
    ok("blocchi diretti ed ereditati, motivi e stato di ogni record ricalcolati dalla scansione, dall'elenco completo dei "
       "duplicati e dall'inventario (stato del gruppo il più grave, nessuna riammissione)", not bad_state,
       {"diversi": len(bad_state), "esempi": bad_state[:5]})
    by_group_state = defaultdict(set)
    for r in recs:
        by_group_state[r["gruppo"]].add(r["stato"])
    ok("ogni gruppo con un solo stato (propagazione completa)", all(len(s) == 1 for s in by_group_state.values()))
    signal_admitted = sum(1 for r in recs if r["stato"] == "ammesso"
                          and any(e["lista"] == "pixmo_check_run4" for e in r["evidenze"]))
    run4_keys = {x["key"] for x in rows if x["signals"]}
    ok("segnali di run4: evidenze di tutti e soli i record con segnali, nessun blocco ne deriva (ammessi con segnali)",
       {r["record"] for r in recs if any(e["lista"] == "pixmo_check_run4" for e in r["evidenze"])} == run4_keys
       and signal_admitted > 0 and not any(c not in CODES for r in recs for c in r["blocchi"]["diretti"]),
       {"record_con_segnali": len(run4_keys), "ammessi_con_segnali": signal_admitted})
    # split
    info = {}
    for r in recs:
        i = info.setdefault(r["gruppo"], {"famiglia": SOURCE, "valutazione": False, "ammessi": 0})
        i["ammessi"] += r["stato"] == "ammesso"
    split = {r["gruppo"]: r["split"] for r in recs}
    rule = meta["split"]
    again = mf.assign_splits(info, rule["quota_dev"], rule["seme"])
    ok("split ricalcolato con quota e seme del manifest: uguale a quello scritto",
       all(again[g] == split[g] for g in info))
    admitted = sum(i["ammessi"] for i in info.values())
    dev = sum(i["ammessi"] for g, i in info.items() if split[g] == "dev")
    ok("dev: almeno l'obiettivo di record ammessi, superato al più dall'ultimo gruppo; quarantene ed esclusi in train",
       round(rule["quota_dev"] * admitted) <= dev <= round(rule["quota_dev"] * admitted)
       + max((i["ammessi"] for i in info.values()), default=0)
       and all(split[g] == "train" for g, i in info.items() if i["ammessi"] == 0),
       {"ammessi": admitted, "dev": dev, "quota": round(dev / admitted, 5) if admitted else None})
    ok("conteggi uguali a manifest.json", summarize(recs) == meta["conteggi"])
    bad_rank = [r["record"] for r in recs if r["rango_f0"] != um.f0_rank(rule["seme"], r["gruppo"])
                or r["rango_f0_sviluppo"] != (um.f0_rank(rule["seme"], r["gruppo"], "sviluppo") if r["split"] == "dev"
                                              else None)]
    ok("ranghi del campione candidato ricalcolati", not bad_rank, bad_rank[:5])
    derived_ok = []
    for name, entry in meta["derivati"].items():
        text = (folder / name).read_text(encoding="utf-8")
        derived_ok.append(mf.sha256_text(text) == entry["sha256"] and text.count("\n") == entry["righe"])
    ok("file derivati (keys/) con gli sha256 registrati", all(derived_ok))
    return {"manifest": str(folder), "manifest_sha256": m.manifest_sha256, "controlli": checks,
            "superati": sum(c["esito"] for c in checks), "totale": len(checks),
            "esito": all(c["esito"] for c in checks)}


def cmd_costruzione(args) -> int:
    t0 = time.time()
    for p in (args.out, args.report):
        if p.exists() and any(p.iterdir()):
            raise Problem(f"{p} non è vuota: le uscite si scrivono in cartelle nuove")
    if args.summary.exists():
        raise Problem(f"{args.summary} esiste già")
    inp = inputs(args)
    b = build(args, inp)
    t_build = time.time() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    recs = []

    def lines():
        for rec in records(b, args):
            recs.append({k: rec[k] for k in ("record", "split", "stato", "gruppo", "compiti", "blocchi", "evidenze")})
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
    admitted = sum(i["ammessi"] for i in b["info"].values())
    dev = sum(i["ammessi"] for g, i in b["info"].items() if b["split_of"][g] == "dev")
    groups_dev = sum(1 for g in b["info"] if b["split_of"][g] == "dev")
    sh = inp["sharding"]
    meta = {
        "schema": mf.SCHEMA, "nome": args.name, "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"),
        "codice": code_identity(), "sorgente": SOURCE, "revisione": args.revision,
        "identita": {"repository": pi.REPO, "revisione": args.revision, "origine_a_monte": pi.UPSTREAM,
                     "snapshot": {"readme": inp["inv_sum"].get("readme"),
                                  "parquet": [{k: x[k] for k in ("file", "blob", "sha256_calcolato",
                                                                 "byte_letti_per_lo_sha256")}
                                              for u in inp["inv_sum"]["unita"] for x in u["parquet"]]},
                     "catalogo": "downloads/catalog_status.tsv (hash dello snapshot verificati nel job 29554 e "
                                 "ricalcolati nel job dell'inventario)"},
        "indice": b["index"],
        "ingressi": {"inventario_job": inp["inv_sum"]["job_id"], "codice_dell_inventario": inp["inv_sum"]["codice"],
                     "file": inp["files"], "sharding_job": sh["manifest"].get("job_id"),
                     "pannello": {"indice_sha256": args.index_sha256,
                                  "convenzione_immagini": inp["scan_sum"].get("convenzione_immagini")}},
        "shards": inp["shards"],
        "liste_importate": {
            "decontaminazione_pannello_pixmo": {
                "controllo": "scripts/pixmo_inventory.py (scan e verifica)", "job": inp["inv_sum"]["job_id"],
                "file": {"scansione": inp["files"]["scansione"], "proposte": inp["verifica_files"]
                         ["proposte_di_trattamento.jsonl.gz"]},
                "stato_proposto": {"corrispondenza_confermata": "escluso, per gruppo",
                                   "candidato": "quarantena, per gruppo (da verificare)",
                                   "non_verificabile": "quarantena, per gruppo",
                                   "nessuna_corrispondenza": "nessun effetto"}},
            "duplicati_inventario": {
                "controllo": "scripts/pixmo_inventory.py verifica (elenco completo dei duplicati)",
                "file": {"duplicati.jsonl.gz": inp["verifica_files"]["duplicati.jsonl.gz"]},
                "regola": f"gruppi per pixel dell'immagine originale con almeno {args.min_segnaposto} righe: quarantena "
                          "cautelativa per gruppo (possibili segnaposto, non errori dimostrati); gli altri duplicati "
                          "si raggruppano senza scarti",
                "gruppi": [{"righe": g["righe"], "url_distinti": g["url_distinti"],
                            "didascalie_distinte": g["didascalie_distinte"], "host": g["host"],
                            "dimensioni_originale": g["dimensioni_originale"], "byte_originale": g["byte_originale"],
                            "chiavi": g["chiavi"]} for g in b["placeholders"]]},
            "orientamento_exif": {"controllo": "inventario (originale.exif_orientamento)",
                                  "regola": "valori da 2 a 8: quarantena per gruppo, nessuna rotazione; 0 e i valori "
                                            "fuori scala non sono validi e restano senza effetto"},
            "pixmo_check_run4": {"controllo": "scripts/pixmo_transcript_check.py (job 30044), riprodotto riga per riga "
                                              "nel job dell'inventario", "file": inp["run4"],
                                 "uso": "evidenza di rischio nei record con segnali; nessun effetto sullo stato"},
            "liste_storiche": {"file": inp["historical"],
                               "uso": "solo confronto diagnostico (job dell'inventario): tutti i candidati storici "
                                      "sono candidati anche nella scansione nuova; nessun effetto proprio"}},
        "tokenizer": None, "template": None, "contratti": None,
        "split": {"valori": list(mf.SPLITS), "unita": "gruppo", "quota_dev": args.dev_fraction, "seme": args.seed,
                  "strati": "un solo strato (PixMo-Cap)",
                  "regola": "in ordine di sha256({seme, gruppo}), dev finché i record ammessi di dev sono meno di "
                            "round(quota * record ammessi); il resto train (manifest.assign_splits); i gruppi senza "
                            "record ammessi restano in train, dove nessun record si legge",
                  "record_ammessi": admitted, "record_ammessi_dev": dev, "gruppi_dev": groups_dev,
                  "quota_ottenuta": round(dev / admitted, 5) if admitted else None},
        "rango_f0": {"campione": um.SAMPLE, "seme": args.seed,
                     "regola": "sha256 del JSON canonico {campione, seme, gruppo}; per lo sviluppo anche ruolo "
                               "'sviluppo' (F0_CAMPIONE_CANDIDATO §1)"},
        "stati": STATES_DOC, "blocchi": {c: {"stato": s, "motivo": t} for c, (s, t) in CODES.items()},
        "compiti": {INSIEME: [TASK]},
        "unioni": b["stats"], "conteggi_autorizzati": b["counts"], "conteggi": counts,
        "scarti_dello_sharding": b["discarded"],
        "records": {"file": "records.jsonl.gz", "sha256": data_sha, "righe": n_lines},
        "derivati": derived, "fuori_dal_sottoinsieme": NOT_COVERED, "verifiche_mancanti": MISSING_CHECKS,
        "secondi_costruzione": round(t_build, 1), "rss_max_mib": um.rss_mib()}
    write_new_json(args.out / "manifest.json", meta)
    del recs, b
    verification = verify(args, args.out, inp)
    verification.update({"job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
                         "secondi_totali": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()})
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "verifica.json", verification)
    summary = {"schema": "limite-vl-manifest-riepilogo/1", "manifest": um.rel(args.out), "nome": args.name,
               "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"),
               "manifest_sha256": verification["manifest_sha256"], "records_sha256": data_sha, "righe": n_lines,
               "sorgente": SOURCE, "revisione": args.revision,
               "ingressi": {k: {n: e["sha256"] for n, e in v.items()} for k, v in inp["files"].items()},
               "shards": meta["shards"], "split": meta["split"], "unioni": meta["unioni"],
               "conteggi_autorizzati": meta["conteggi_autorizzati"], "conteggi": counts,
               "scarti_dello_sharding": meta["scarti_dello_sharding"],
               "controlli": [{k: c[k] for k in ("controllo", "esito")} for c in verification["controlli"]],
               "esito_verifica": verification["esito"], "stati": STATES_DOC, "fuori_dal_sottoinsieme": NOT_COVERED,
               "verifiche_mancanti": MISSING_CHECKS,
               "nota": "organizza candidati ed evidenze: non certifica le didascalie e non autorizza il training",
               "secondi_totali": verification["secondi_totali"], "rss_max_mib": verification["rss_max_mib"]}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.summary, summary)
    print(json.dumps({"righe": n_lines, "per_split_stato": counts["per_split_stato"], "split": meta["split"],
                      "verifica": f"{verification['superati']}/{verification['totale']}",
                      "secondi": verification["secondi_totali"], "rss_max_mib": verification["rss_max_mib"]},
                     ensure_ascii=False))
    return 0 if verification["esito"] else 1


def cmd_verifica(args) -> int:
    if args.report.exists() and any(args.report.iterdir()):
        raise Problem(f"{args.report} non è vuota")
    t0 = time.time()
    verification = verify(args, args.out, inputs(args))
    verification.update({"job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
                         "secondi_totali": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()})
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "verifica.json", verification)
    print(f"{verification['superati']} controlli superati su {verification['totale']}")
    return 0 if verification["esito"] else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("costruzione", "verifica"):
        p = sub.add_parser(name)
        for opt in ("--inventory", "--verifica", "--scan", "--shards", "--run4", "--out", "--report"):
            p.add_argument(opt, type=Path, required=True)
        p.add_argument("--historical", type=Path, nargs="+", required=True)
        for opt in ("--revision", "--inventory-sha256", "--scan-sha256", "--index-sha256"):
            p.add_argument(opt, required=True)
        for opt in ("--gruppi-segnaposto", "--righe-segnaposto", "--exif-attesi", "--candidati-attesi"):
            p.add_argument(opt, type=int, required=True, help="conteggio autorizzato: la costruzione si ferma se diverso")
        p.add_argument("--min-segnaposto", type=int, default=3)
        p.add_argument("--name", default="pixmo-cap-candidato")
        p.add_argument("--version", default="1")
        p.add_argument("--dev-fraction", type=float, default=0.05)
        p.add_argument("--seed", type=int, default=0)
    sub.choices["costruzione"].add_argument("--summary", type=Path, required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("pixmo_manifest.py va eseguito in un job SLURM")
    try:
        return cmd_costruzione(args) if args.cmd == "costruzione" else cmd_verifica(args)
    except Problem as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
