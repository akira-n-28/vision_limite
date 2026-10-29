#!/usr/bin/env python3
"""Verifica semantica e del renderer dei sintetici (limite_vl.synth_check) in un job SLURM CPU. Non scrive shard e
non modifica quelli esistenti.

Modi, anche insieme:
- --shards GLOB: record di shard esistenti, in sola lettura; ogni shard letto è registrato con il suo sha256.
- --generate N --generator-version V: N scene generate e disegnate in memoria (chiavi synth-check-NNNNNNN, famiglie a
  rotazione), senza salvare immagini.
- --cases CHIAVE:FAMIGLIA:VERSIONE ...: casi singoli disegnati e salvati in OUT/cases/ (PNG, layout ed esiti), per
  esempio la scena tagliata synth-0000043 in v3 e la stessa chiave in v4.

Esiti di un record letto da uno shard, tenuti distinti (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §2.4):
- semantica della scena: `verificata` se synth_check.check_sample sul layout usato non trova violazioni e tutte le
  domande sono ok, altrimenti `non_verificata`;
- immagine, cioè la corrispondenza fra il rendering descritto da quel layout e il PNG presente nello shard:
  - `digest_corrispondente`: il layout salvato porta `png_sha256` uguale allo sha256 dei byte del PNG dello shard;
  - `byte_identici`: la scena ridisegnata dalla specifica dà gli stessi byte del PNG dello shard;
  - `pixel_identici`: byte diversi, ma il PNG dello shard decodificato ha le stesse dimensioni e gli stessi valori RGBA
    del ridisegno;
  - `diversa_da_verificare`: pixel diversi, o PNG che non si decodifica; non prova che l'immagine o la risposta siano
    sbagliate: va guardato;
  - `png_mancante`: il record non ha il PNG;
- provenienza del layout usato per la semantica:
  - `salvato_con_digest`: quello del record, legato al PNG dal digest;
  - `ridisegnato_layout_assente`: record senza layout (v3), layout del ridisegno;
  - `ridisegnato_senza_digest`: record con layout ma senza `png_sha256` (precedenti al digest): il layout salvato
    non prova il legame col PNG e non si usa;
  - `ridisegnato_digest_diverso`: digest diverso dal PNG dello shard, il layout salvato non è di quell'immagine e non
    si usa;
  - `salvato_png_mancante`: PNG assente, semantica sul layout del record;
- esito del campione: `verificato` solo con semantica verificata e immagine `digest_corrispondente`, `byte_identici` o
  `pixel_identici`; altrimenti `non_verificato`.
Una scena generata in memoria (--generate) ha immagine `in_memoria`, layout `generato` ed esito `scena_verificata` o
`scena_non_verificata`: verifica il generatore e il renderer, non è un campione di dati.

Uscite nella cartella nuova --out:
- report.json: per origine (shard, generato) campioni per esito, immagine, semantica, layout, semantica per immagine
  e famiglia; `incoherent`, le incoerenze fra questi conteggi (vuoto se tornano; altrimenti il job esce con errore dopo
  aver scritto il report); esiti delle domande e violazioni del renderer per origine, famiglia e tipo; ingressi;
  identità del codice;
- problems.jsonl: una riga per ogni campione o scena non verificati con i motivi (`what` = "sample", sempre scritta),
  più esempi di domande non ok e di violazioni (al più PROBLEM_CAP per shard, o blocco generato, per famiglia, tipo ed
  esito); i conteggi del report sono sempre completi.

    srun $RUN_PYTHON -u -B scripts/synth_semantics_check.py --out DIR [--shards GLOB ...] \\
        [--generate N --generator-version 4] [--cases KEY:FAMILY:VERSION ...] [--workers W]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import synth, synth_check  # noqa: E402
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.provenance import HashingTextWriter, code_identity, sha256_file, write_new_json  # noqa: E402

PROBLEM_CAP = 200
IMAGE_OK = ("digest_corrispondente", "byte_identici", "pixel_identici")  # corrispondenza col PNG dello shard dimostrata
ORIGINS = ("shard", "generato")
VERIFIED = ("verificato", "scena_verificata")


def _keep(problems: list[dict], kept: Counter, new: list[dict]) -> None:
    """Aggiunge i problemi: le righe per campione sempre, gli esempi di domande e violazioni fino a PROBLEM_CAP per
    famiglia, tipo ed esito."""
    for p in new:
        if p["what"] == "sample":
            problems.append(p)
            continue
        k = (p["family"], p["what"], p.get("type") or p.get("kind"), p.get("status"))
        if kept[k] < PROBLEM_CAP:
            kept[k] += 1
            problems.append(p)


def png_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def same_pixels(image, png: bytes) -> bool:
    """Confronto deterministico dei pixel decodificati, distinto da quello dei byte: stesse dimensioni e stessi valori
    RGBA decodificati da Pillow (un'immagine RGB ha alfa 255). I metadati del PNG (gamma, profili, testi) non contano.
    Un file che non si decodifica come PNG non prova nulla."""
    from PIL import Image

    try:
        other = Image.open(io.BytesIO(png), formats=("PNG",))
        other.load()
        return other.size == image.size and other.convert("RGBA").tobytes() == image.convert("RGBA").tobytes()
    except Exception:  # noqa: BLE001 - PNG illeggibile o non convertibile: nessuna prova di corrispondenza
        return False


def shard_evidence(meta: dict, png: bytes | None) -> tuple[dict, str, str]:
    """(layout per la semantica, stato dell'immagine, provenienza del layout) di un record letto da uno shard.

    Il layout salvato si usa solo se porta `png_sha256` uguale allo sha256 del PNG dello shard. Negli altri casi la
    scena si ridisegna dalla specifica e il rendering si confronta col PNG: prima i byte, poi i pixel decodificati."""
    stored = meta.get("layout")
    if png is None:
        if stored is not None:
            return stored, "png_mancante", "salvato_png_mancante"
        return synth.render(meta["spec"], layout=True)[1], "png_mancante", "ridisegnato_layout_assente"
    if stored is not None and stored.get("png_sha256") == png_digest(png):
        return stored, "digest_corrispondente", "salvato_con_digest"
    if stored is None:
        source = "ridisegnato_layout_assente"
    elif "png_sha256" not in stored:
        source = "ridisegnato_senza_digest"
    else:
        source = "ridisegnato_digest_diverso"
    image, layout = synth.render(meta["spec"], layout=True)
    if synth.png_bytes(image) == png:
        return layout, "byte_identici", source
    if same_pixels(image, png):
        return layout, "pixel_identici", source
    return layout, "diversa_da_verificare", source


def _check(key: str, meta: dict, origin: str, png: bytes | None = None,
           layout: dict | None = None) -> tuple[Counter, list[dict], dict]:
    """Esiti di un campione: contatori, problemi e stato {origin, image, layout, semantic, outcome, reasons}.

    - origin "shard": record letto da uno shard; `png` sono i byte del suo PNG (None se manca); il layout viene da
      shard_evidence.
    - origin "generato": scena appena disegnata in memoria; `layout` è il suo layout."""
    spec = meta["spec"]
    fam = spec["family"]
    if origin == "shard":
        if layout is not None:
            raise ValueError(f"{key}: il layout di un record di shard viene dal record (shard_evidence)")
        layout, image, source = shard_evidence(meta, png)
    elif origin == "generato":
        if layout is None or png is not None:
            raise ValueError(f"{key}: una scena generata richiede il suo layout e nessun PNG")
        image, source = "in_memoria", "generato"
    else:
        raise ValueError(f"origine {origin!r}: attesa una di {ORIGINS}")
    result = synth_check.check_sample(meta, layout)
    semantic = "verificata" if result["verified"] else "non_verificata"
    if origin == "shard":
        outcome = "verificato" if result["verified"] and image in IMAGE_OK else "non_verificato"
    else:
        outcome = "scena_verificata" if result["verified"] else "scena_non_verificata"
    reasons = []  # vuoti se e solo se l'esito è verificato
    if origin == "shard" and image not in IMAGE_OK:
        reasons.append(f"immagine: {image}")
    if result["blocking"]:
        reasons.append("violazioni bloccanti: " + ", ".join(result["blocking"]))
    bad = Counter(r["status"] for r in result["questions"] if r["status"] != "ok")
    if bad:
        reasons.append("domande non ok: " + ", ".join(f"{k} {v}" for k, v in sorted(bad.items())))
    other = sorted({v["kind"] for v in result["violations"]} - set(result["blocking"]))
    if other:
        reasons.append("altre violazioni: " + ", ".join(other))
    state = {"origin": origin, "image": image, "layout": source, "semantic": semantic, "outcome": outcome,
             "reasons": reasons}
    counts, problems = Counter(), []
    for r in result["questions"]:
        counts[f"questions/{origin}/{fam}/{r['type']}/{r['status']}"] += 1
        if r["status"] != "ok":
            problems.append({"key": key, "family": fam, "what": "question", "origin": origin, **r})
    for v in result["violations"]:
        counts[f"violations/{origin}/{fam}/{v['kind']}"] += 1
        problems.append({"key": key, "family": fam, "what": "violation", "origin": origin, **v})
    for name, value in (("sample", outcome), ("image", image), ("semantic", semantic), ("layout", source)):
        counts[f"{name}/{origin}/{value}"] += 1
    counts[f"joint/{origin}/{semantic}/{image}"] += 1
    counts[f"family/{origin}/{fam}/{outcome}"] += 1
    if outcome not in VERIFIED:
        problems.append({"key": key, "family": fam, "what": "sample", **state})
    return counts, problems, state


def summarize(counts: Counter) -> dict:
    """Conteggi per origine di campioni, immagine, semantica, layout e semantica per immagine, più `incoherent`, le
    incoerenze trovate (vuoto se i conteggi tornano): ogni campione è contato una volta in ciascun gruppo, anche per
    famiglia; i verificati di uno shard sono esattamente quelli con semantica verificata e immagine corrispondente, le
    scene verificate quelle con semantica verificata."""
    def tally(prefix: str) -> dict:
        out: dict = {}
        for k, v in sorted(counts.items()):
            if k.startswith(prefix + "/"):
                origin, value = k[len(prefix) + 1:].split("/", 1)
                out.setdefault(origin, {})[value] = v
        return out

    groups = {name: tally(name) for name in ("sample", "image", "semantic", "layout", "joint", "family")}
    incoherent = []
    for origin in ORIGINS:
        sizes = {name: sum(g.get(origin, {}).values()) for name, g in groups.items()}
        if len(set(sizes.values())) > 1:
            incoherent.append(f"{origin}: totali diversi fra i gruppi {sizes}")
    proven = sum(v for k, v in groups["joint"].get("shard", {}).items()
                 if k.split("/")[0] == "verificata" and k.split("/")[1] in IMAGE_OK)
    if groups["sample"].get("shard", {}).get("verificato", 0) != proven:
        incoherent.append(f"shard: verificati diversi dai {proven} con semantica verificata e immagine corrispondente")
    scenes = groups["semantic"].get("generato", {}).get("verificata", 0)
    if groups["sample"].get("generato", {}).get("scena_verificata", 0) != scenes:
        incoherent.append(f"generato: scene verificate diverse dalle {scenes} con semantica verificata")
    out = {name: groups[name] for name in ("sample", "image", "semantic", "layout", "joint")}
    out["total"] = {origin: sum(groups["sample"].get(origin, {}).values()) for origin in ORIGINS}
    out["incoherent"] = incoherent
    return out


def shard_task(path: str) -> dict:
    counts, problems, kept = Counter(), [], Counter()
    for key, files in iter_shard(path):
        c, p, _ = _check(key, json.loads(files["json"]), "shard", png=files.get("png"))
        counts.update(c)
        _keep(problems, kept, p)
    return {"input": {"path": path, "sha256": sha256_file(path)}, "counts": counts, "problems": problems}


def generate_task(task: tuple[int, int, int]) -> dict:
    start, stop, version = task
    counts, problems, kept = Counter(), [], Counter()
    for i in range(start, stop):
        key, family = f"synth-check-{i:07d}", synth.FAMILIES[i % len(synth.FAMILIES)]
        spec = synth.make_spec(key, family, version)
        _, layout = synth.render(spec, layout=True)
        meta = {"id": key, "family": family, "qa": synth.questions(spec), "spec": spec}
        c, p, _ = _check(key, meta, "generato", layout=layout)
        counts.update(c)
        _keep(problems, kept, p)
    return {"input": {"generate": [start, stop], "generator_version": version}, "counts": counts, "problems": problems}


def save_cases(cases: list[str], out: Path) -> list[dict]:
    """Disegna i casi CHIAVE:FAMIGLIA:VERSIONE e salva PNG, layout (con lo sha256 del PNG) ed esiti in out/, come file
    nuovi. Sono scene disegnate in memoria: il loro esito è di scena."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        key, family, version = case.rsplit(":", 2)
        spec = synth.make_spec(key, family, int(version))
        image, layout = synth.render(spec, layout=True)
        data = synth.png_bytes(image)
        layout["png_sha256"] = png_digest(data)
        meta = {"id": key, "family": family, "qa": synth.questions(spec), "spec": spec}
        result = synth_check.check_sample(meta, layout)
        stem = f"{key}-v{version}"
        png_path = out / f"{stem}.png"
        if png_path.exists():
            sys.exit(f"{png_path} esiste già")
        png_path.write_bytes(data)
        write_new_json(out / f"{stem}.json", {"case": case, "spec": spec, "layout": layout, "result": result})
        rows.append({"case": case, "png": png_path.name, "png_sha256": layout["png_sha256"],
                     "outcome": "scena_verificata" if result["verified"] else "scena_non_verificata",
                     "blocking": result["blocking"],
                     "questions": dict(Counter(r["status"] for r in result["questions"]))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shards", nargs="*", default=[], help="glob di shard tar da controllare (sola lettura)")
    ap.add_argument("--generate", type=int, default=0, help="scene da generare e controllare in memoria")
    ap.add_argument("--generator-version", type=int, choices=synth.VERSIONS, help="versione per --generate")
    ap.add_argument("--cases", nargs="*", default=[], help="casi CHIAVE:FAMIGLIA:VERSIONE da disegnare e salvare")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("synth_semantics_check.py va eseguito in un job SLURM")
    if args.generate and args.generator_version is None:
        sys.exit("--generate richiede --generator-version")
    paths = sorted({p for g in args.shards for p in glob.glob(g)})
    if args.shards and not paths:
        sys.exit(f"nessuno shard per {args.shards}")
    if not paths and not args.generate and not args.cases:
        sys.exit("niente da controllare: --shards, --generate o --cases")
    for case in args.cases:
        parts = case.rsplit(":", 2)
        if len(parts) != 3 or parts[1] not in synth.FAMILIES or not parts[2].isdigit() or int(parts[2]) not in synth.VERSIONS:
            sys.exit(f"caso non valido: {case} (atteso CHIAVE:FAMIGLIA:VERSIONE)")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: scegliere un'altra cartella")
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    case_rows = save_cases(args.cases, args.out / "cases") if args.cases else []
    counts, inputs = Counter(), []
    writer = HashingTextWriter(args.out / "problems.jsonl")
    with Pool(args.workers) as pool:
        jobs = [pool.imap_unordered(shard_task, paths)]
        if args.generate:
            chunks = [(i, min(i + args.chunk, args.generate), args.generator_version)
                      for i in range(0, args.generate, args.chunk)]
            jobs.append(pool.imap_unordered(generate_task, chunks))
        for it in jobs:
            for res in it:
                counts.update(res["counts"])
                inputs.append(res["input"])
                for p in res["problems"]:
                    writer.write_json_line(p)
                n = sum(v for k, v in res["counts"].items() if k.startswith("sample/"))
                print(f"{res['input']}: {n} campioni, {len(res['problems'])} righe di problemi "
                      f"({time.time() - t0:.0f} s)", flush=True)
    problems_file = writer.close()
    summary = summarize(counts)

    def group(prefix: str) -> dict:
        out: dict = {}
        for k, v in sorted(counts.items()):
            if k.startswith(prefix):
                parts = k[len(prefix):].split("/")
                node = out
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = v
        return out

    questions: dict = {}
    for k, v in counts.items():
        if k.startswith("questions/"):
            _, origin, _family, _type, status = k.split("/")
            questions.setdefault(origin, Counter())[status] += v
    report = {"schema": "synth-check/3", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
              "args": {k: str(v) for k, v in vars(args).items()},
              "thresholds": {k: getattr(synth_check, k) for k in ("EDGE", "CLOSE_ROOTS", "TANGENT", "SIDE", "ON_LINE",
                                                                   "OFF_LINE", "ANGLE_TOL", "LENGTH_TOL", "EXTREME",
                                                                   "MIN_TEXT_PX", "OVERLAP_PX", "CLIP_PX")},
              "blocking_kinds": list(synth_check.BLOCKING), "image_ok": list(IMAGE_OK),
              "inputs": sorted(inputs, key=json.dumps), "samples": summary["total"],
              "samples_by_origin": summary["sample"], "image_by_origin": summary["image"],
              "semantic_by_origin": summary["semantic"], "layout_by_origin": summary["layout"],
              "semantic_image_by_origin": summary["joint"], "by_family": group("family/"),
              "incoherent": summary["incoherent"],
              "questions": {o: dict(c) for o, c in questions.items()}, "by_type": group("questions/"),
              "violations": group("violations/"), "cases": case_rows,
              "problems": problems_file, "problem_cap": PROBLEM_CAP, "seconds": round(time.time() - t0, 1)}
    write_new_json(args.out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("samples", "samples_by_origin", "image_by_origin", "semantic_by_origin",
                                             "seconds")}, ensure_ascii=False))
    for row in case_rows:
        print("caso", json.dumps(row, ensure_ascii=False))
    print(json.dumps(report["violations"], ensure_ascii=False))
    if summary["incoherent"]:
        sys.exit("conteggi incoerenti (report scritto): " + "; ".join(summary["incoherent"]))


if __name__ == "__main__":
    main()
