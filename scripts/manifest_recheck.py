#!/usr/bin/env python3
"""Riverifica in sola lettura dei manifest candidati già scritti di UniMER-1M (job 30625) e MAVIS-Geometry (job 30630),
coi verificatori correnti (S1 §9; S2 §§3-4). Non ricostruisce nulla e non modifica manifest, shard o evidenze. Scrive
una cartella nuova di rapporti, legata allo sha256 del manifest.json controllato, e un rapporto sintetico per git. Solo
in un job SLURM CPU. Le anomalie si riportano con identificativi, conteggi e causa, senza correzioni automatiche:
un manifest da correggere diventa una versione nuova, su proposta.

Ingressi verificati prima dei controlli: un'incoerenza ferma il job prima di scrivere.
- manifest.json con lo sha256 atteso (--sha256), letto e verificato da manifest.load;
- i file delle evidenze che la verifica rilegge, con gli sha256 registrati nel manifest (liste_importate).

unimer:
- unimer_manifest.verify: righe ritagliate con lo stesso PNG della formula, stati richiesti dalle evidenze, split col
  seme dichiarato;
- tutti i record dell'insieme `righe` contro lo shard delle righe ritagliate registrato nel manifest, letto e hashato
  in un passaggio. Per ogni campione: JSON e PNG coi byte del manifest, etichetta, numero di righe, griglia; righe e
  fasce strutturalmente valide (unimer_manifest.band_problems), che non provano la corrispondenza visiva;
- con i digest del manifest, la formula genitore: stesso PNG, etichetta e griglia, genitore registrato uguale, nessun
  errore di dimensioni in run4.
Gli shard di train non si rileggono: i digest dei genitori sono quelli che il job 30625 ha calcolato leggendo i tar
verificati contro lo sharding. Qui se ne controlla solo la dimensione su disco.

mavis: mavis_manifest.verify, con lo split ricalcolato col seme dichiarato e senza il controllo sul cambio di seme.

    srun $RUN_PYTHON -u -B scripts/manifest_recheck.py unimer --manifest DIR --sha256 HEX --check DIR --overlap DIR \\
        --report DIR --summary FILE
    srun $RUN_PYTHON -u -B scripts/manifest_recheck.py mavis --manifest DIR --sha256 HEX --evidence DIR \\
        --report DIR --summary FILE
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
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import mavis_manifest as mm  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.provenance import code_identity, sha256_file, write_new_json  # noqa: E402

SCHEMA = "limite-vl-riverifica/1"
EXAMPLES = 20  # identificativi per causa nel rapporto sintetico
INLINE = 200  # anomalie riportate per intero in verifica.json; tutte in anomalie.jsonl
NOTE = ("riverifica in sola lettura di un manifest già scritto, coi verificatori correnti: non ricostruisce e non "
        "corregge nulla; non certifica la qualità e non autorizza il training")


class Problem(Exception):
    """Ingresso incoerente: la riverifica non si fa."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cause(text: str) -> str:
    """Causa di un problema senza numeri, per contarle."""
    return re.sub(r"-?\d+(\.\d+)?", "N", text)


# ---------------------------------------------------------------- ingressi

def load_expected(folder: Path, expected: str) -> mf.Manifest:
    """Il manifest, solo se manifest.json ha lo sha256 atteso; poi manifest.load (records, record_id, split, shard)."""
    path = folder / "manifest.json"
    if not path.is_file():
        raise Problem(f"manifest mancante: {path}")
    found = sha256_file(path)
    if found != expected:
        raise Problem(f"{path}: sha256 {found}, atteso {expected}: non è il manifest da riverificare")
    try:
        m = mf.load(folder)
    except mf.ManifestError as exc:
        raise Problem(f"manifest non valido: {exc}") from None
    if m.manifest_sha256 != expected:
        raise Problem(f"{path}: cambiato durante la lettura")
    return m


def evidence_identity(recorded: dict[str, dict[str, dict]], actual: dict[str, dict[str, Path]]) -> list[dict]:
    """Per ogni file registrato nel manifest, quello che la verifica rilegge, con lo sha256 su disco; Problem se uno
    manca o è diverso."""
    rows, bad = [], []
    for lista, files in recorded.items():
        for name, entry in files.items():
            path = actual.get(lista, {}).get(name)
            sha = sha256_file(path) if path is not None and path.is_file() else None
            rows.append({"lista": lista, "file": name, "percorso": um.rel(path) if path else None,
                         "percorso_registrato": entry.get("path"), "sha256_registrato": entry.get("sha256"),
                         "sha256_su_disco": sha, "uguale": sha == entry.get("sha256")})
            if sha != entry.get("sha256"):
                bad.append(f"{lista}/{name}")
    if bad:
        raise Problem(f"evidenze su disco diverse da quelle registrate nel manifest (o mancanti): {bad[:5]}")
    return rows


# ---------------------------------------------------------------- unimer: righe ritagliate

def line_problems(key: str, index: int, files: dict, r: dict | None, p: dict | None) -> tuple[dict, dict]:
    """Problemi di un campione dello shard delle righe, per gruppo di controlli (shard, fasce, genitore), e i suoi
    numeri (fasce, altezza)."""
    out = {"shard": [], "fasce": [], "genitore": []}
    if r is None:
        out["shard"].append("campione dello shard senza record righe nel manifest")
        return out, {}
    if set(files) != {"json", "png"}:
        out["shard"].append(f"file del campione {sorted(files)}")
        return out, {}
    png, raw = files["png"], files["json"]
    try:
        meta = json.loads(raw)
    except ValueError:
        out["shard"].append("JSON non leggibile")
        return out, {}
    if not isinstance(meta, dict):
        out["shard"].append("JSON senza un oggetto")
        return out, {}
    png_sha, label = sha256_hex(png), meta.get("label")
    if meta.get("id") != key:
        out["shard"].append(f"id del JSON {meta.get('id')!r}")
    if r["indice"] != index:
        out["shard"].append(f"posizione {index} nel tar, {r['indice']} nel manifest")
    if sha256_hex(raw) != r["json_sha256"]:
        out["shard"].append("JSON con byte diversi da quelli del manifest")
    if png_sha != r["immagini"][0]["sha256"]:
        out["shard"].append("PNG con byte diversi da quelli del manifest")
    if not isinstance(label, str):
        out["shard"].append("etichetta assente")
        label = ""
    label_sha = sha256_hex(label.encode())
    if label_sha != r["etichetta_sha256"]:
        out["shard"].append("etichetta diversa da quella del manifest")
    n_rows = len(meta["rows"]) if isinstance(meta.get("rows"), list) else 0
    if n_rows != r["righe"]:
        out["shard"].append(f"{n_rows} righe nello shard, {r['righe']} nel manifest")
    out["fasce"] += um.band_problems(meta, label, png)
    size = um.png_size(png)
    if size is not None and r["immagini"][0].get("griglia") != [size[1] // 16, size[0] // 16]:
        out["fasce"].append("griglia del manifest diversa dalle dimensioni del PNG")
    if p is None:
        out["genitore"].append("formula genitore assente dal manifest")
    else:
        g = (r["genitori"] or [{}])[0]
        if p["immagini"][0]["sha256"] != png_sha:
            out["genitore"].append("PNG diverso da quello della formula genitore")
        if p["etichetta_sha256"] != label_sha:
            out["genitore"].append("etichetta diversa da quella della formula genitore")
        if p["immagini"][0].get("griglia") != r["immagini"][0].get("griglia"):
            out["genitore"].append("griglia diversa da quella della formula genitore")
        if (g.get("record_id"), g.get("json_sha256"), g.get("png_sha256"), g.get("shard"), g.get("indice")) \
                != (p["record_id"], p["json_sha256"], p["immagini"][0]["sha256"], p["shard"], p["indice"]):
            out["genitore"].append("genitore registrato diverso dalla formula del manifest")
        if any("dimensioni_diverse_dal_json" in (e.get("categorie") or []) for e in p["evidenze"]
               if e.get("lista") == "unimer_check_run4"):
            out["genitore"].append("formula genitore con dimensioni del JSON diverse dal PNG (run4)")
    bands = meta.get("lines")
    numbers = {"fasce": bands, "altezza": size[1] if size else None, "righe": n_rows}
    return out, numbers


def band_stats(bands, height) -> dict | None:
    """Fasce valide: numero, altezza minima, spazio minimo fra fasce vicine, fasce adiacenti."""
    if not isinstance(bands, list) or not bands or not all(
            isinstance(b, list) and len(b) == 2 and all(type(y) is int for y in b) for b in bands):
        return None
    gaps = [b[0] - a[1] for a, b in zip(bands, bands[1:])]
    return {"n": len(bands), "altezza_minima": min(b[1] - b[0] for b in bands), "spazio_minimo": min(gaps, default=None),
            "adiacenti": sum(x == 0 for x in gaps), "fino_al_bordo": height is not None and bands[-1][1] == height}


def recheck_lines(m: mf.Manifest) -> dict:
    """Tutti i record `righe` contro lo shard delle righe ritagliate registrato nel manifest (un passaggio, hashato) e,
    coi digest del manifest, contro la formula genitore."""
    base = {}
    for s in m.meta["shards"]:
        if s["insieme"] == "train":
            base.update(m.by_shard.get(s["percorso"], {}))
    shards, anomalies = [], []
    per_group, causes, examples = Counter(), Counter(), {}
    n_bands, stats = Counter(), {"altezza_minima": None, "spazio_minimo": None, "record_con_fasce_adiacenti": 0,
                                 "record_con_l_ultima_fascia_fino_al_bordo": 0}
    checked = 0
    for s in (s for s in m.meta["shards"] if s["insieme"] == "righe"):
        recs = m.by_shard.get(s["percorso"], {})
        path = mf.shard_path(s["percorso"])
        seen, n_samples, t0 = set(), 0, time.time()
        raw = um.HashingReader(path)
        with tarfile.open(fileobj=io.BufferedReader(raw, buffer_size=1 << 20), mode="r|") as tar:
            for index, (key, files) in enumerate(um.iter_samples(tar)):
                n_samples += 1
                r = recs.get(key)
                if r is not None:
                    seen.add(key)
                    checked += 1
                found, numbers = line_problems(key, index, files, r, base.get(key))
                st = band_stats(numbers.get("fasce"), numbers.get("altezza"))
                if st is not None:
                    n_bands[st["n"]] += 1
                    for k in ("altezza_minima", "spazio_minimo"):
                        if st[k] is not None and (stats[k] is None or st[k] < stats[k]):
                            stats[k] = st[k]
                    stats["record_con_fasce_adiacenti"] += st["adiacenti"] > 0
                    stats["record_con_l_ultima_fascia_fino_al_bordo"] += st["fino_al_bordo"]
                if any(found.values()):
                    anomalies.append({"record": key, "shard": s["percorso"], "indice": index,
                                      "stato": r["stato"] if r else None, "split": r["split"] if r else None,
                                      "problemi": {k: v for k, v in found.items() if v}, **numbers})
                    for group, items in found.items():
                        per_group[group] += bool(items)
                        for x in items:
                            c = cause(x)
                            causes[c] += 1
                            examples.setdefault(c, [])
                            if len(examples[c]) < EXAMPLES:
                                examples[c].append(key)
        sha, size = raw.finish()
        missing = sorted(set(recs) - seen)
        for key in missing:
            why = "record righe del manifest assente dallo shard"
            anomalies.append({"record": key, "shard": s["percorso"], "indice": recs[key]["indice"],
                              "stato": recs[key]["stato"], "split": recs[key]["split"], "problemi": {"shard": [why]}})
            per_group["shard"] += 1
            causes[why] += 1
            if len(examples.setdefault(why, [])) < EXAMPLES:
                examples[why].append(key)
        shards.append({"percorso": s["percorso"], "sha256_registrato": s["sha256"], "sha256_letto": sha,
                       "byte_registrati": s["bytes"], "byte_letti": size, "record_registrati": s["record"],
                       "campioni_letti": n_samples, "record_trovati": len(seen),
                       "record_assenti": len(missing), "secondi": round(time.time() - t0, 1),
                       "uguale": sha == s["sha256"] and size == s["bytes"]})
    train_sizes = []
    for s in m.meta["shards"]:
        if s["insieme"] == "train":
            path = mf.shard_path(s["percorso"])
            on_disk = path.stat().st_size if path.is_file() else None
            train_sizes.append({"percorso": s["percorso"], "byte_registrati": s["bytes"], "byte_su_disco": on_disk,
                                "uguale": on_disk == s["bytes"]})
    righe_total = sum(len(m.by_shard.get(s["percorso"], {})) for s in m.meta["shards"] if s["insieme"] == "righe")
    checks = [
        ("shard delle righe ritagliate uguale a quello registrato nel manifest (sha256, byte, campioni)",
         all(x["uguale"] and x["campioni_letti"] == x["record_registrati"] for x in shards),
         [{k: x[k] for k in ("percorso", "uguale", "campioni_letti", "record_registrati")} for x in shards]),
        ("ogni record righe del manifest nello shard, ogni campione dello shard nel manifest; JSON, PNG, etichetta, "
         "posizione e numero di righe coi valori del manifest", per_group["shard"] == 0,
         {"record_righe": righe_total, "controllati": checked, "con_problemi": per_group["shard"]}),
        ("righe e fasce strutturalmente valide (unimer_manifest.band_problems: coppie di interi dentro l'immagine, in "
         "ordine, senza sovrapposizioni, una per riga; dimensioni del JSON uguali al PNG) e griglia del manifest "
         "coerente", per_group["fasce"] == 0, {"con_problemi": per_group["fasce"]}),
        ("formula genitore (digest del manifest): stesso PNG, stessa etichetta e griglia, genitore registrato uguale, "
         "nessun errore di dimensioni in run4", per_group["genitore"] == 0, {"con_problemi": per_group["genitore"]}),
        ("shard di train non riletti: dimensione su disco uguale a quella registrata", all(x["uguale"] for x in train_sizes),
         {"shard": len(train_sizes), "diversi": sum(not x["uguale"] for x in train_sizes)}),
    ]
    return {"shard": shards, "shard_di_train_non_riletti": train_sizes, "record_righe": righe_total,
            "controllati": checked, "con_anomalie": len(anomalies),
            "anomalie_per_gruppo_di_controlli": dict(per_group), "anomalie_per_causa": dict(causes.most_common()),
            "esempi_per_causa": examples, "fasce_per_record": dict(sorted(n_bands.items())), "fasce": stats,
            "controlli": [{"controllo": n, "esito": bool(c), "dettaglio": d} for n, c, d in checks],
            "anomalie": anomalies}


# ---------------------------------------------------------------- rapporti

def write_reports(args, m: mf.Manifest, inputs: list[dict], parts: dict[str, dict], t0: float) -> dict:
    """verifica.json (e anomalie.jsonl se ce ne sono) in --report, rapporto sintetico in --summary."""
    checks = [{"parte": name, **c} for name, part in parts.items() for c in part["controlli"]]
    anomalies = [a for part in parts.values() for a in part.get("anomalie", [])]
    passed = all(c["esito"] for c in checks)
    common = {"schema": SCHEMA, "sorgente": args.source, "manifest": um.rel(args.manifest),
              "nome": m.meta.get("nome"), "versione": m.meta.get("versione"), "manifest_sha256": m.manifest_sha256,
              "manifest_sha256_atteso": args.sha256, "records_sha256": m.meta["records"]["sha256"],
              "job_costruzione": m.meta.get("job_id"), "codice_costruzione": m.meta.get("codice"),
              "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "verificatori": {name: {"percorso": um.rel(p), "sha256": sha256_file(p)}
                               for name, p in (("unimer_manifest", ROOT / "scripts" / "unimer_manifest.py"),
                                               ("mavis_manifest", ROOT / "scripts" / "mavis_manifest.py"),
                                               ("manifest_recheck", Path(__file__).resolve()))},
              "evidenze": inputs}
    args.report.mkdir(parents=True, exist_ok=True)
    if anomalies:
        with open(args.report / "anomalie.jsonl.tmp", "w", encoding="utf-8") as fh:
            for a in anomalies:
                fh.write(json.dumps(a, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(args.report / "anomalie.jsonl.tmp", args.report / "anomalie.jsonl")
    detail = {name: {k: v for k, v in part.items() if k not in ("anomalie", "controlli")}
              for name, part in parts.items()}
    report = {**common, "parti": detail, "controlli": checks, "superati": sum(c["esito"] for c in checks),
              "totale": len(checks), "esito": passed, "anomalie": len(anomalies),
              "anomalie_in_questo_file": anomalies[:INLINE],
              "anomalie_file": "anomalie.jsonl" if anomalies else None,
              "anomalie_sha256": sha256_file(args.report / "anomalie.jsonl") if anomalies else None,
              "secondi": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib(), "nota": NOTE}
    write_new_json(args.report / "verifica.json", report)
    summary = {**{k: v for k, v in common.items() if k not in ("codice_costruzione", "evidenze")},
               "evidenze_uguali_a_quelle_registrate": all(x["uguale"] for x in inputs),
               "evidenze": {f"{x['lista']}/{x['file']}": x["sha256_su_disco"] for x in inputs},
               "rapporto": um.rel(args.report / "verifica.json"),
               "rapporto_sha256": sha256_file(args.report / "verifica.json"),
               "controlli": [{k: c[k] for k in ("parte", "controllo", "esito")} for c in checks],
               "superati": report["superati"], "totale": report["totale"], "esito": passed,
               "anomalie": len(anomalies)}
    lines = parts.get("righe_ritagliate")
    if lines:
        summary["righe_ritagliate"] = {k: lines[k] for k in ("record_righe", "controllati", "con_anomalie",
                                                             "anomalie_per_gruppo_di_controlli", "anomalie_per_causa",
                                                             "esempi_per_causa", "fasce_per_record", "fasce")}
        summary["righe_ritagliate"]["shard"] = [{k: x[k] for k in ("percorso", "sha256_letto", "uguale")}
                                                for x in lines["shard"]]
    summary["nota"] = NOTE
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.summary, summary)
    return report


def run(args) -> dict:
    """Riverifica: Problem se gli ingressi non sono quelli registrati, prima di scrivere; altrimenti i rapporti, anche
    con controlli falliti (esito false)."""
    t0 = time.time()
    if args.report.exists() and any(args.report.iterdir()):
        raise Problem(f"{args.report} non è vuota: i rapporti si scrivono una volta sola")
    if args.summary.exists():
        raise Problem(f"{args.summary} esiste già")
    m = load_expected(args.manifest, args.sha256)
    if m.meta.get("sorgente") != {"unimer": um.SOURCE, "mavis": mm.SOURCE}[args.source]:
        raise Problem(f"il manifest è della sorgente {m.meta.get('sorgente')!r}, non di {args.source}")
    lists = m.meta.get("liste_importate") or {}
    if args.source == "unimer":
        recorded = {name: lists.get(name, {}).get("file", {}) for name in ("unimer_check_run4", "unimer_overlap_run2")}
        actual = {"unimer_check_run4": {n: args.check / n for n in recorded["unimer_check_run4"]},
                  "unimer_overlap_run2": {n: args.overlap / n for n in recorded["unimer_overlap_run2"]}}
        inputs = evidence_identity(recorded, actual)
        print(f"ingressi verificati in {time.time() - t0:.1f} s", flush=True)
        parts = {"verifica": um.verify(args.manifest, args.check, args.overlap, m=m)}
        print(f"verifica: {parts['verifica']['superati']}/{parts['verifica']['totale']} in {time.time() - t0:.1f} s",
              flush=True)
        parts["righe_ritagliate"] = recheck_lines(m)
    else:
        recorded = {name: entry.get("file", {}) for name, entry in lists.items()}
        ev = mm.evidence_files(args.evidence)
        inputs = evidence_identity(recorded, ev)
        print(f"ingressi verificati in {time.time() - t0:.1f} s", flush=True)
        parts = {"verifica": mm.verify(args.manifest, ev, m=m)}
    return write_reports(args, m, inputs, parts, t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=("unimer", "mavis"))
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--sha256", required=True, help="sha256 atteso di manifest.json")
    ap.add_argument("--check", type=Path, help="unimer: outputs/unimer_check/run4")
    ap.add_argument("--overlap", type=Path, help="unimer: outputs/unimer_overlap/run2")
    ap.add_argument("--evidence", type=Path, help="mavis: cartella outputs con le uscite storiche")
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("manifest_recheck.py va eseguito in un job SLURM")
    if not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        sys.exit("--sha256: servono 64 cifre esadecimali")
    needed = ("check", "overlap") if args.source == "unimer" else ("evidence",)
    if any(getattr(args, k) is None for k in needed):
        sys.exit(f"{args.source}: servono --{' --'.join(needed)}")
    try:
        report = run(args)
    except Problem as exc:
        sys.exit(f"riverifica non eseguita: {exc}")
    print(json.dumps({"esito": report["esito"], "controlli": f"{report['superati']}/{report['totale']}",
                      "anomalie": report["anomalie"], "secondi": report["secondi"],
                      "rss_max_mib": report["rss_max_mib"]}, ensure_ascii=False))
    if not report["esito"]:
        print("controlli non superati: vedi " + str(args.report / "verifica.json"), file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
