#!/usr/bin/env python3
"""Smoke test di C7 sui manifest e sugli shard reali delle sorgenti di manifest.TASKED (UniMER-1M, MAVIS-Geometry,
PixMo-Cap): i reader del progetto (stream.make_sources, SourceReader, validation_samples, MixedRowStream,
data.build_example) con i manifest candidati della config (--config). Solo in un job SLURM CPU: manifest e shard in sola
lettura, scrive solo --out e --summary. Nessun modello: il processor di Violetto serve solo a tokenizzare. Per PixMo
anche: nessun record non ammesso consegnato, ammessi coi segnali di run4 consegnati (evidenze, non blocchi), didascalia,
richiesta e JPEG di ogni esempio invariati e coi digest del manifest.

1. Sorgenti della config con i manifest identificati da sha256: identità, shard verificati, attesi del manifest per
   split, stato e compito, rietichettature; tempo e memoria.
2. Sottoinsieme deterministico: il primo shard in ordine di nome di ogni sorgente (per le righe ritagliate l'unico),
   letto per intero da un SourceReader per lo split train e per il dev. Le chiavi consegnate si confrontano
   esattamente coi record del manifest che dovrebbero entrare, ricavati dai record grezzi (records.jsonl.gz rilettto
   con lo sha256 di manifest.json, senza Admission), e così i motivi dei saltati.
3. Esempi col processor: i primi --examples consegnati di train di ogni sorgente e i casi mirati del sottoinsieme
   (formule rietichettate ammesse; righe di formule ammesse o rietichettate; record di MAVIS con shape_type registrata),
   più i primi --dev di validation_samples. Compiti emessi solo fra gli ammessi, shape_type mai, rietichettature
   applicate (sha256 dell'etichetta nuova del manifest), target delle righe fra le righe del record.
4. MixedRowStream con tutte le sorgenti per --stream-batches micro-batch: solo chiavi ammesse di train, contatori.

I conteggi del manifest valgono per la sorgente intera; quelli osservati per il sottoinsieme o per letture parziali (il
flusso ripete i record di una sorgente piccola). Lo smoke test non è una verifica completa del training.

    srun $RUN_PYTHON -u -B scripts/reader_smoke.py --config config/runs/f0-c7-reader-unimer-mavis.json \\
        --out DIR --summary FILE [--examples 300] [--dev 200] [--stream-batches 20]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import io
import json
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, sha256_file, write_new_json  # noqa: E402

VIOLETTO = ROOT / "hf/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
CHECKS: list[dict] = []
TARGETED = 40  # casi mirati per categoria e sorgente


def check(name: str, ok, detail=None) -> bool:
    CHECKS.append({"controllo": name, "esito": bool(ok), "dettaglio": detail})
    print(("ok   " if ok else "FAIL ") + name + (f"  ({str(detail)[:400]})" if detail is not None else ""), flush=True)
    return bool(ok)


def rss_mib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def raw_records(folder: Path, sha: str) -> dict[str, dict[str, tuple]]:
    """Record del manifest in forma compatta, dai byte di records.jsonl.gz con lo sha256 registrato in manifest.json
    (che deve essere quello della config): percorso dello shard -> chiave -> (insieme, split, stato, compiti, nome della
    trasformazione, sha256 dell'etichetta nuova, indice, dati di PixMo). compiti: [(nome, stato, domande)]. Dati di PixMo
    (None per le altre sorgenti): segnali di run4, sha256 della didascalia e del JPEG, blocchi diretti ed ereditati.
    Indipendente da Admission."""
    if sha256_file(folder / "manifest.json") != sha:
        raise SystemExit(f"{folder}: manifest.json diverso da quello della config")
    meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    raw = (folder / meta["records"]["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["records"]["sha256"]:
        raise SystemExit(f"{folder}: records.jsonl.gz diverso da quello registrato")
    out: dict[str, dict[str, tuple]] = defaultdict(dict)
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
        for line in fh:
            r = json.loads(line)
            tasks = tuple((t, r["stato"], 1) if isinstance(t, str) else (t["compito"], t["stato"], t.get("domande", 0))
                          for t in r.get("compiti") or [])
            t = r.get("trasformazione") or {}
            pixmo = None
            if r.get("didascalia_sha256"):
                b = r.get("blocchi") or {}
                pixmo = {"segnali": any(e.get("lista") == "pixmo_check_run4" for e in r.get("evidenze") or []),
                         "didascalia_sha256": r["didascalia_sha256"], "immagine_sha256": r["immagini"][0]["sha256"],
                         "diretti": tuple(b.get("diretti") or ()),
                         "ereditati": tuple(e["codice"] for e in b.get("ereditati") or ())}
            out[r["shard"]][r["record"]] = (r["insieme"], r["split"], r["stato"], tasks, t.get("nome"),
                                            t.get("etichetta_sha256"), r["indice"], pixmo)
    return out


def entering(rec: tuple, kind: str, split: str) -> str | None:
    """Motivo per cui il record non entra nello split per questo tipo di loader, None se entra (dalla specifica di C7:
    split, stato del record, compito del tipo o almeno un compito ammesso)."""
    source = next(s for s, kinds in mf.TASKED.items() if kind in kinds)
    insieme, task = mf.TASKED[source][kind]
    if rec[0] != insieme:
        return f"insieme_{rec[0]}"
    if rec[1] != split:
        return f"split_{rec[1]}"
    if rec[2] != "ammesso":
        return rec[2]
    if task is not None:
        return None if task in [t[0] for t in rec[3]] else f"compito_non_ammesso/{task}"
    usable = [t for t in rec[3] if t[1] == "ammesso" and (t[0] == "describe" or t[2] > 0)]
    return None if usable else "nessun_compito_ammesso"


def read_pass(source, keep: set[str], first_n: int) -> dict:
    """Un'epoca di un SourceReader con un worker: chiavi consegnate, contatori dell'epoca (fotografati quando finisce:
    next() li fa avanzare fino al primo campione dell'epoca dopo), file dei primi first_n e di `keep`."""
    from limite_vl.stream import SourceReader

    class EpochReader(SourceReader):
        def _advance_shard(self) -> None:
            super()._advance_shard()
            if self.epoch == 1 and not hasattr(self, "epoch0"):
                self.epoch0 = dict(self.counts)

    t0 = time.time()
    reader = EpochReader(source, seed=0, worker=0, n_workers=1, val_count=0)
    keys, files_kept = [], {}
    while True:
        key, files = reader.next()
        if reader.epoch > 0:
            break
        keys.append(key)
        if key in keep or len(files_kept) < first_n:
            files_kept[key] = files
    return {"keys": keys, "files": files_kept, "counts": reader.epoch0, "secondi": round(time.time() - t0, 1)}


def build_report(processor, source, items, recs: dict, relabel: dict) -> dict:
    """Esempi costruiti dai file consegnati: compiti per tipo, bloccati, token, errori, controlli per record."""
    from limite_vl.data import DETAILED_CAPTION_PROMPTS, SAMPLE_LOADERS, admission_of, build_example, pick_prompt

    per_task, blocked, tokens = Counter(), Counter(), []
    errors, bad_task, relabel_ok, relabel_bad, line_bad = [], [], 0, [], []
    pixmo_ok, pixmo_bad = 0, []
    for key, files in items:
        try:
            ex = build_example(processor, source.kind, key, files)
        except Exception as exc:  # noqa: BLE001 - contato e riportato
            errors.append(f"{key}: {type(exc).__name__}: {exc}"[:200])
            continue
        tokens.append(len(ex))
        per_task.update(ex.tasks)
        blocked.update(ex.blocked)
        adm = admission_of(files) or {}
        if not ex.tasks or not set(ex.tasks) <= set(adm.get("compiti_ammessi") or []) or "shape_type" in ex.tasks:
            bad_task.append(key)
        sample = SAMPLE_LOADERS[source.kind](key, files)
        target = sample.messages[-1]["content"]
        rec = recs[key]
        if source.kind == "unimer_formula" and rec[4] == "rietichettatura":
            if target == relabel.get(key, (None, None))[1] and hashlib.sha256(target.encode()).hexdigest() == rec[5]:
                relabel_ok += 1
            else:
                relabel_bad.append(key)
        if source.kind in ("unimer_line", "unimer_line_select") and target not in json.loads(files["json"])["rows"]:
            line_bad.append(key)
        if source.kind == "pixmo_caption":  # didascalia, richiesta e immagine invariate, coi digest del manifest
            px = rec[7] or {}
            same = (target == json.loads(files["json"])["caption"]
                    and hashlib.sha256(target.encode()).hexdigest() == px.get("didascalia_sha256")
                    and sample.messages[0]["content"][1]["text"] == pick_prompt(key, DETAILED_CAPTION_PROMPTS)
                    and hashlib.sha256(files["jpg"]).hexdigest() == px.get("immagine_sha256")
                    and ex.tasks == ("pixmo_caption",))
            if same:
                pixmo_ok += 1
            else:
                pixmo_bad.append(key)
    out = {"costruiti": len(tokens), "errori": errors[:10], "n_errori": len(errors),
           "compiti_emessi": dict(sorted(per_task.items())), "domande_bloccate": dict(sorted(blocked.items())),
           "token_medi": round(sum(tokens) / len(tokens), 1) if tokens else None, "token_max": max(tokens, default=None),
           "fuori_dai_compiti_ammessi": bad_task[:10], "n_fuori": len(bad_task),
           "rietichettati_controllati": relabel_ok, "rietichettati_sbagliati": relabel_bad[:10],
           "righe_con_target_fuori_dal_record": line_bad[:10]}
    if source.kind == "pixmo_caption":
        out.update(pixmo_invariati=pixmo_ok, pixmo_diversi=pixmo_bad[:10])
    return out


def pixmo_unchanged(kind: str, rep: dict) -> bool:
    """PixMo: ogni esempio costruito con didascalia, richiesta e JPEG invariati (vero per le altre sorgenti)."""
    return kind != "pixmo_caption" or (rep["pixmo_invariati"] == rep["costruiti"] and not rep["pixmo_diversi"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--examples", type=int, default=300)
    ap.add_argument("--dev", type=int, default=200)
    ap.add_argument("--stream-batches", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("reader_smoke.py va eseguito in un job SLURM")
    for p in (args.out,):
        if p.exists() and any(p.iterdir()):
            sys.exit(f"{p} non è vuota")
    if args.summary.exists():
        sys.exit(f"{args.summary} esiste già")
    t0 = time.time()
    phases: dict[str, dict] = {}
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    specs = cfg["sources"]

    from limite_vl.stream import MixedRowStream, make_sources, validation_samples  # importa torch: solo qui

    # ---- 1. sorgenti della config, più le sotto-sorgenti del sottoinsieme (stessi manifest: verifiche condivise)
    sub_specs = []
    for spec in specs:
        first = sorted(glob.glob(spec["shards"]))[0]
        for split in ("train", "dev"):
            sub_specs.append({**spec, "name": f"{spec['name']}@{Path(first).stem}/{split}", "shards": first,
                              "split": split})
    t = time.time()
    built = make_sources(specs + sub_specs)
    sources, subs = built[:len(specs)], built[len(specs):]
    phases["make_sources"] = {"secondi": round(time.time() - t, 1), "rss_max_mib": rss_mib()}
    described = {}
    for spec, s in zip(specs, sources):
        d = s.admission.describe()
        described[s.name] = {**d, "rietichettature": len(s.relabel or {}), "n_shard": len(s.shards)}
        check(f"{s.name}: manifest identificato dallo sha256 della config, della sorgente, revisione, insieme e tipo",
              d["manifest_sha256"] == spec["manifest_sha256"] and d["tipo"] == spec["kind"]
              and d["sorgente"] == next(src for src, kinds in mf.TASKED.items() if spec["kind"] in kinds),
              {k: d[k] for k in ("manifest_sha256", "sorgente", "revisione", "insieme", "tipo", "compito_del_tipo")})
    # ---- record grezzi dei manifest
    t = time.time()
    raw = {}
    for spec in specs:
        if spec["manifest"] not in raw:
            raw[spec["manifest"]] = raw_records(ROOT / spec["manifest"], spec["manifest_sha256"])
    phases["record_grezzi"] = {"secondi": round(time.time() - t, 1), "rss_max_mib": rss_mib()}

    def recs_of(source) -> dict[str, tuple]:
        spec = next(x for x in specs + sub_specs if x["name"] == source.name)
        by_shard = raw[spec["manifest"]]
        out = {}
        for p in source.shards:
            rel = str(Path(p).resolve().relative_to(ROOT))
            out.update(by_shard[rel])
        return out

    # ---- 2. sottoinsieme: un'epoca per sotto-sorgente, confronto esatto coi record grezzi
    processor = None
    subset = {}
    kept_files: dict[str, dict] = {}
    t = time.time()
    for s in subs:
        recs = recs_of(s)
        split = s.admission.split
        ordered = sorted(recs, key=lambda k: recs[k][6])
        want = [k for k in ordered if entering(recs[k], s.kind, split) is None]
        reasons = Counter(entering(recs[k], s.kind, split) for k in ordered)
        reasons.pop(None, None)
        keep: set[str] = set()
        if split == "train":
            if s.kind == "unimer_formula":
                keep = {k for k in want if recs[k][4] == "rietichettatura"}
            elif s.kind == "mavis_f1":
                keep = {k for k in want if any(x[0] == "shape_type" for x in recs[k][3])}
            elif s.kind == "pixmo_caption":  # ammessi coi segnali di run4: evidenze, non blocchi
                keep = {k for k in want if (recs[k][7] or {}).get("segnali")}
            keep = set(sorted(keep, key=lambda k: recs[k][6])[:TARGETED])
        res = read_pass(s, keep, args.examples if split == "train" else 0)
        got = res["keys"]
        observed = {k.split("/", 1)[1]: v for k, v in res["counts"].items() if k.startswith("saltati_manifest/")}
        expected_reasons = {k: v for k, v in reasons.items() if not k.startswith("insieme_")}
        ok = check(f"{s.name}: un'epoca consegna esattamente i record che il manifest ammette in questo shard, una volta "
                   "e nell'ordine dello shard, e salta gli altri coi motivi attesi",
                   got == want and len(set(got)) == len(got)
                   and observed == dict(expected_reasons) and not res["counts"].get("errori_manifest"),
                   {"attesi": len(want), "consegnati": len(got), "motivi_attesi": dict(expected_reasons),
                    "motivi_osservati": observed})
        subset[s.name] = {"shard": Path(s.shards[0]).name, "split": split, "tipo": s.kind,
                          "record_nello_shard": len(recs), "attesi_dal_manifest": len(want), "consegnati": len(got),
                          "uguali": ok, "motivi_attesi": dict(sorted(expected_reasons.items())),
                          "contatori_del_reader": res["counts"], "secondi": res["secondi"],
                          "casi_mirati_tenuti": len(keep & set(res["files"]))}
        if s.kind == "pixmo_caption":  # casi non ammessi dello shard per origine del blocco: nessuno è consegnato
            cases = Counter()
            for k, r in recs.items():
                px = r[7] or {}
                if r[2] != "ammesso":
                    cases[f"{r[1]}/{r[2]}/" + ("+".join(px.get("diretti") or ()) or
                                               "ereditato:" + "+".join(sorted(set(px.get("ereditati") or ()))))] += 1
            subset[s.name]["non_ammessi_per_origine"] = dict(sorted(cases.items()))
            subset[s.name]["ammessi_con_segnali_run4"] = sum(1 for k in want if (recs[k][7] or {}).get("segnali"))
            check(f"{s.name}: nessun record in quarantena o escluso consegnato; ammessi coi segnali di run4 consegnati",
                  not [k for k in got if recs[k][2] != "ammesso"]
                  and (split != "train" or subset[s.name]["ammessi_con_segnali_run4"] > 0),
                  {"non_ammessi_nello_shard": sum(cases.values()),
                   "ammessi_con_segnali": subset[s.name]["ammessi_con_segnali_run4"]})
        if split == "train":
            kept_files[s.name] = res["files"]
    phases["sottoinsieme"] = {"secondi": round(time.time() - t, 1), "rss_max_mib": rss_mib()}
    # righe e formule con la stessa chiave: decisioni proprie (solo se la config ha entrambe le sorgenti di UniMER)
    line_src = next((s for s in sources if s.kind == "unimer_line"), None)
    formula_src = next((s for s in sources if s.kind == "unimer_formula"), None)
    if line_src is not None and formula_src is not None:
        line_recs, formula_recs = recs_of(line_src), recs_of(formula_src)
        pairs = Counter()
        for k, r in line_recs.items():
            f = formula_recs.get(k)
            pairs[f"riga {r[2]} / formula {f[2] if f else 'assente'}{' rietichettata' if f and f[4] else ''}"] += 1
        check("UniMER chiavi condivise: ogni riga ritagliata ha la sua formula (stessa chiave), una riga di formula "
              "rietichettata non è ammessa, la decisione della riga è la sua",
              all("assente" not in k for k in pairs) and not any(k.startswith("riga ammesso") and "rietichettata" in k
                                                                 for k in pairs), dict(pairs))
    # ---- 3. esempi col processor
    from limite_vl.processing import LimiteVLProcessor

    t = time.time()
    processor = LimiteVLProcessor.from_pretrained(str(VIOLETTO))
    examples = {}
    for s in subs:
        if s.admission.split != "train":
            continue
        rep = build_report(processor, s, list(kept_files[s.name].items()), recs_of(s), s.relabel or {})
        examples[s.name] = rep
        check(f"{s.name}: esempi costruiti senza errori, compiti emessi solo fra gli ammessi (shape_type mai), "
              "rietichettature applicate e target delle righe dal record",
              rep["costruiti"] > 0 and not rep["n_errori"] and not rep["n_fuori"] and not rep["rietichettati_sbagliati"]
              and not rep["righe_con_target_fuori_dal_record"] and "shape_type" not in rep["compiti_emessi"]
              and (s.kind != "unimer_formula" or rep["rietichettati_controllati"] > 0)
              and (s.kind != "mavis_f1" or rep["domande_bloccate"].get("shape_type", 0) > 0)
              and pixmo_unchanged(s.kind, rep),
              {k: rep[k] for k in ("costruiti", "n_errori", "compiti_emessi", "domande_bloccate",
                                   "rietichettati_controllati", "pixmo_invariati") if k in rep})
    dev_report = {}
    for s in sources:
        recs = recs_of(s)
        val = validation_samples(s, args.dev)
        wrong = [k for k, _ in val if entering(recs[k], s.kind, "dev") is not None]
        rep = build_report(processor, s, val, recs, s.relabel or {})
        dev_report[s.name] = {"letti": len(val), "fuori_dal_dev_ammesso": wrong[:10], **rep}
        check(f"{s.name}: validazione dal dev del manifest (record ammessi col compito), esempi costruiti",
              len(val) > 0 and not wrong and rep["costruiti"] == len(val) and not rep["n_fuori"]
              and pixmo_unchanged(s.kind, rep),
              {"letti": len(val), "costruiti": rep["costruiti"], "compiti": rep["compiti_emessi"]})
    phases["esempi"] = {"secondi": round(time.time() - t, 1), "rss_max_mib": rss_mib()}
    # ---- 4. flusso misto
    t = time.time()
    admitted = {}
    for s in sources:
        recs = recs_of(s)
        admitted[s.name] = {k for k, r in recs.items() if entering(r, s.kind, "train") is None}
    stream = MixedRowStream(sources, processor, max_len=args.max_len, micro_rows=4, seed=0, shuffle_buffer=128,
                            pack_buffer=64)
    keys, meta = [], {}
    for n, batch in enumerate(stream.iter_worker(0, 1)):
        keys += [k.split("/", 1) for k in batch["meta"]["keys"]]
        meta = batch["meta"]
        if n + 1 >= args.stream_batches:
            break
    stats = meta.get("stats", {})
    outside = [f"{s}/{k}" for s, k in keys if k not in admitted[s]]
    per_source = Counter(s for s, _ in keys)
    check("flusso misto: solo chiavi ammesse di train di ogni sorgente, esempi da tutte, nessun errore, nessun "
          "shape_type, contatori per sorgente e compito",
          keys and not outside and set(per_source) == {s.name for s in sources}
          and not any(k.startswith(("errors/", "manifest_errors/", "senza_compiti/")) for k in stats)
          and not any(k.endswith("/shape_type") and k.startswith("compiti/") for k in stats),
          {"esempi": len(keys), "per_sorgente": dict(per_source), "fuori": outside[:5]})
    phases["flusso"] = {"secondi": round(time.time() - t, 1), "rss_max_mib": rss_mib()}
    failed = [c["controllo"] for c in CHECKS if not c["esito"]]
    report = {"schema": "reader-smoke/1", "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "config": {**file_entry(args.config), "notes": cfg.get("notes")},
              "sorgenti": described, "sottoinsieme": subset, "esempi_train": examples, "esempi_dev": dev_report,
              "flusso": {"micro_batch": args.stream_batches, "esempi": len(keys), "per_sorgente": dict(per_source),
                         "fuori_ammissione": outside[:20], "stats": stats, "readers": meta.get("readers")},
              "fasi": phases, "controlli": CHECKS, "superati": len(CHECKS) - len(failed), "totale": len(CHECKS),
              "falliti": failed, "secondi": round(time.time() - t0, 1), "rss_max_mib": rss_mib(),
              "nota": "attesi del manifest per la sorgente intera; osservati su un sottoinsieme deterministico (primo "
                      "shard di ogni tipo) e su letture parziali (esempi, flusso): smoke test, non verifica completa "
                      "del training. Ammesso = decisione operativa del candidato, non gold"}
    args.out.mkdir(parents=True, exist_ok=True)
    write_new_json(args.out / "report.json", report)
    summary = {k: report[k] for k in ("schema", "job_id", "codice", "config", "fasi", "superati", "totale", "falliti",
                                      "secondi", "rss_max_mib", "nota")}
    summary["sorgenti"] = {n: {k: d.get(k) for k in ("manifest", "manifest_sha256", "sorgente", "revisione", "nome",
                                                     "versione", "insieme", "tipo", "compito_del_tipo", "split",
                                                     "n_shard", "attesi", "attesi_per_compito", "ammessi_nello_split",
                                                     "record_in_questi_shard", "rietichettature")}
                           for n, d in described.items()}
    summary["sottoinsieme"] = {n: {k: v for k, v in x.items() if k != "contatori_del_reader"} for n, x in subset.items()}
    summary["esempi_train"] = {n: {k: v for k, v in x.items() if k != "errori"} for n, x in examples.items()}
    summary["esempi_dev"] = {n: {k: v for k, v in x.items() if k != "errori"} for n, x in dev_report.items()}
    summary["flusso"] = {k: v for k, v in report["flusso"].items() if k != "readers"}
    summary["controlli"] = [{k: c[k] for k in ("controllo", "esito")} for c in CHECKS]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.summary, summary)
    print(f"{report['superati']} controlli superati su {report['totale']}; {report['secondi']} s, "
          f"RSS {report['rss_max_mib']} MiB")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
