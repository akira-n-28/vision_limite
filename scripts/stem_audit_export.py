#!/usr/bin/env python3
"""Audit automatico STEM di MMFineReason: controlli deterministici, campione, calibrazione e lotti per gli agenti.
Solo in un job SLURM CPU.

Popolazione: i problemi sft, meno quelli che non entrano mai in training (--exclude-lists: decontaminazione p6,
domande in giapponese o coreano). Campione stratificato per sorgente originale (colonna `source`): --n diviso in
parti uguali con limite.stem_audit.allocate (screening), oppure --per-source problemi delle sole --sources
(certificazione), con un seed fisso e senza le chiavi di --exclude-keys (campioni indipendenti).

Controlli deterministici (prima degli agenti):
- su tutta la popolazione, dal testo: campi vuoti, segnaposto <image> diversi da uno, domande a scelta multipla con
  la risposta lettera fuori dalle opzioni, domande e coppie domanda-risposta duplicate, descrizioni di più immagini
  (population_checks.json, per sorgente);
- sul campione: integrità dell'immagine (decodifica completa, formato, modo, immagine quasi vuota, proporzioni),
  risoluzione di training (data.stem_image: lato, patch, fattore di ingrandimento), duplicati dello stesso problema
  (stessa domanda, immagine uguale per pHash) con risposta diversa, risposta che math_verify non sa leggere,
  candidati alla contaminazione p6 non confermati (exclude.txt grezzo), metadati (deterministic.json).

Calibrazione (verità nota, mescolata alla cieca nei lotti; verità in _gold/calibration.json): diagrammi MAVIS con
la didascalia vera e due affermazioni alterate, problemi MMFineReason con due affermazioni alterate, diagrammi
sintetici con domande dalla risposta nota; un terzo con la risposta di riferimento sbagliata apposta; figure
rimpicciolite fino a renderle illeggibili; domande scambiate fra problemi di sorgenti diverse.

Gli agenti vedono solo id opachi (p001, f001): niente sorgente, niente segno della calibrazione. Scrive in OUT:
- batches/SA-NN.json, SB-NN.json: risolvere dalla sola immagine e domanda, leggibilità, coerenza (senza descrizione);
- batches/DA-NN.json, DB-NN.json: affermazioni della descrizione da dividere in fatti e verificare sull'immagine;
- batches/FA-NN.json, FB-NN.json (con --f2-selected): soluzioni F2 di Violetto da verificare contro l'immagine;
- img/: immagini come le vede il modello in training; _gold/: items.json, gold.json, idmap.json, calibration.json,
  items_f2.json (mai nei lotti); deterministic.json, population_checks.json, source_sizes.json, manifest.json.

    srun $RUN_PYTHON -u -B scripts/stem_audit_export.py --out DIR --parquet-glob GLOB --n 300 --seed 20261019 \
        --exclude-lists L1 L2 --tag-lists L3 L4 --contam-candidates EXCLUDE_TXT --mavis-shard TAR --synth-test TAR \
        [--f2-selected outputs/f2_select/f2-sft120k-v4/selected.jsonl --f2-n 100]
"""
from __future__ import annotations

import argparse
import bisect
import glob
import hashlib
import io
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.data import iter_shard, open_image, stem_image  # noqa: E402
from limite_vl.decontam import hamming, phash  # noqa: E402
from limite_vl.f1c import mavis_questions  # noqa: E402
from limite_vl.f2 import answers_match  # noqa: E402
from limite_vl.mmfine import mmfine_key, rank, split_of  # noqa: E402
from limite_vl.stem_audit import (allocate, answer_letter, claim_units, mc_options, mutate_text,  # noqa: E402
                                  norm_question, wrong_answer)

Image.MAX_IMAGE_PIXELS = None
CAL = {"mavis": 30, "mmf_injected": 30, "synth": 20, "illegible": 10, "incoherent": 10}


def h8(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def read_keys(paths: list[str]) -> dict[str, set[str]]:
    return {Path(p).name: {line.strip() for line in open(p) if line.strip()} for p in paths}


# ---------------------------------------------------------------- popolazione e controlli sul testo

def scan(paths: list[str]) -> tuple[list[dict], dict]:
    """Una riga leggera per ogni problema con domanda, risposta e descrizione, e i controlli sul testo per sorgente."""
    import pyarrow.parquet as pq

    rows, stats = [], defaultdict(Counter)
    for path in paths:
        split = split_of(Path(path))
        t = pq.read_table(path, columns=["id", "source", "pass_rate", "question", "answer", "caption"]).to_pydict()
        for i in range(len(t["id"])):
            src = t["source"][i]
            q, a, c = t["question"][i] or "", (t["answer"][i] or "").strip(), t["caption"][i] or ""
            st = stats[src]
            st["righe"] += 1
            if not q.replace("<image>", "").strip() or not a or not c.strip():
                st["campi_vuoti"] += 1
                continue
            opts, letter = mc_options(q.replace("<image>", " ")), answer_letter(a)
            st["segnaposto_diversi_da_uno"] += q.count("<image>") != 1
            st["scelta_multipla"] += bool(opts)
            st["lettera_fuori_dalle_opzioni"] += bool(opts and letter and letter not in opts)
            st["opzioni_ma_risposta_non_lettera"] += bool(opts and not letter)
            st["descrizione_di_piu_immagini"] += bool(re.search(r"Image\s*\[?2\b", c))
            rows.append({"key": mmfine_key(split, src, t["id"][i]), "source": src, "file": path, "row": i,
                         "pass_rate": t["pass_rate"][i], "qhash": h8(norm_question(q)),
                         "qahash": h8(norm_question(q) + "\x00" + a.lower())})
        del t
    qcount = Counter(r["qhash"] for r in rows)
    qacount = Counter(r["qahash"] for r in rows)
    for r in rows:
        st = stats[r["source"]]
        st["domanda_duplicata"] += qcount[r["qhash"]] > 1
        st["domanda_e_risposta_duplicate"] += qacount[r["qahash"]] > 1
    for r in rows:
        r["qdups"] = qcount[r["qhash"]] - 1
    return rows, {s: dict(c) for s, c in stats.items()}


def load_full(chosen: list[dict]) -> dict[str, dict]:
    """Testo completo (domanda grezza, risposta, descrizione) dei problemi scelti."""
    import pyarrow.parquet as pq

    by_file, out = defaultdict(list), {}
    for p in chosen:
        by_file[p["file"]].append(p)
    for path, items in sorted(by_file.items()):
        t = pq.read_table(path, columns=["question", "answer", "caption"]).take([p["row"] for p in items]).to_pylist()
        for p, r in zip(items, t):
            out[p["key"]] = {"question_raw": r["question"] or "", "question": (r["question"] or "").replace("<image>", "").strip(),
                             "answer": (r["answer"] or "").strip(), "caption": (r["caption"] or "").strip()}
    return out


def image_bytes(chosen: list[dict]) -> dict[str, bytes | None]:
    """Byte originali delle immagini, leggendo dai parquet solo i gruppi di righe che servono."""
    import pyarrow.parquet as pq

    by_file, out = defaultdict(list), {}
    for p in chosen:
        by_file[p["file"]].append(p)
    for path, items in sorted(by_file.items()):
        pf = pq.ParquetFile(path)
        starts, s = [], 0
        for g in range(pf.metadata.num_row_groups):
            starts.append(s)
            s += pf.metadata.row_group(g).num_rows
        by_group = defaultdict(list)
        for p in items:
            by_group[bisect.bisect_right(starts, p["row"]) - 1].append(p)
        for g, ps in sorted(by_group.items()):
            col = pf.read_row_group(g, columns=["image"]).column("image").to_pylist()
            for p in ps:
                out[p["key"]] = (col[p["row"] - starts[g]] or {}).get("bytes")
    return out


# ---------------------------------------------------------------- controlli sull'immagine

def image_checks(raw: bytes | None) -> tuple[Image.Image | None, dict]:
    """Decodifica completa e statistiche dell'immagine originale e di quella di training."""
    if not raw:
        return None, {"ok": False, "error": "immagine mancante"}
    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt, mode, frames = im.format, im.mode, getattr(im, "n_frames", 1)
            im.load()
            orig = im.copy()
        image, patches = stem_image({"png": raw})
    except Exception as exc:  # noqa: BLE001 - immagine che non si decodifica: illeggibile per costruzione
        return None, {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    gray = orig.convert("L")
    hist = gray.histogram()
    w, h = orig.size
    tw, th = image.size
    checks = {"ok": True, "format": fmt, "mode": mode, "frames": frames, "bytes": len(raw), "orig_size": [w, h],
              "train_size": [tw, th], "patches": patches, "image_tokens": patches // 4, "scale": round(tw / w, 3),
              "gray_std": round(ImageStat.Stat(gray).stddev[0], 2), "dominant_frac": round(max(hist) / (w * h), 4),
              "aspect": round(max(w, h) / max(1, min(w, h)), 2), "phash": f"{phash(image):016x}"}
    flags = []
    if min(w, h) < 32:
        flags.append("lato originale sotto 32 px")
    if checks["gray_std"] < 3 or checks["dominant_frac"] > 0.995:
        flags.append("immagine quasi vuota")
    if checks["aspect"] > 8:
        flags.append("proporzioni estreme")
    if frames > 1:
        flags.append("più fotogrammi")
    if checks["scale"] > 3:
        flags.append("ingrandita più di 3x per il training")
    checks["flags"] = flags
    return image, checks


def text_checks(t: dict, gold: str) -> dict:
    q = t["question"]
    opts, letter = mc_options(q), answer_letter(gold)
    flags = []
    if t["question_raw"].count("<image>") != 1:
        flags.append(f"segnaposto <image>: {t['question_raw'].count('<image>')}")
    if opts and letter and letter not in opts:
        flags.append("risposta lettera fuori dalle opzioni")
    if letter and not opts:
        flags.append("risposta lettera senza opzioni riconosciute nella domanda")
    if opts and not letter:
        flags.append("scelta multipla con risposta non lettera")
    if re.search(r"Image\s*\[?2\b", t["caption"]):
        flags.append("descrizione di più immagini")
    if len(gold) > 200:
        flags.append("risposta lunga più di 200 caratteri")
    parseable = None
    try:
        from math_verify import parse

        parseable = bool(parse("\\boxed{" + gold + "}", parsing_timeout=5)) if not letter else True
    except Exception:  # noqa: BLE001
        parseable = False
    if not parseable:
        flags.append("risposta non leggibile da math_verify")
    return {"options": opts, "answer_letter": letter, "gold_parseable": parseable, "flags": flags,
            "gold_wrong_by_code": bool(opts and letter and letter not in opts)}


def duplicate_checks(items: list[dict], rows_by_q: dict, texts: dict, max_dups: int = 3, max_group: int = 20) -> dict:
    """Per ogni problema del campione, fino a max_dups altri problemi con la stessa domanda (solo gruppi piccoli: le
    domande modello come "find x" ricorrono migliaia di volte con figure diverse): stessa immagine per pHash e
    risposta diversa = duplicato in conflitto. Più le coppie di immagini quasi uguali dentro il campione."""
    want = {}
    for it in items:
        group = [r for r in rows_by_q.get(it["qhash"], []) if r["key"] != it["key"]]
        if 0 < len(group) <= max_group:
            want[it["key"]] = sorted(group, key=lambda r: rank(r["key"], 7))[:max_dups]
    flat = [r for g in want.values() for r in g]
    raws, dtexts = image_bytes(flat), load_full(flat)
    out = {}
    for it in items:
        res = []
        for r in want.get(it["key"], []):
            raw = raws.get(r["key"])
            try:
                hp = phash(stem_image({"png": raw})[0]) if raw else None
            except Exception:  # noqa: BLE001
                hp = None
            same = hp is not None and it.get("phash") and hamming(hp, int(it["phash"], 16)) <= 4
            other = dtexts.get(r["key"], {}).get("answer", "")
            agree = answers_match(other, texts[it["key"]]["answer"]) if same and other else None
            res.append({"key": r["key"], "same_image": bool(same), "answer": other, "answer_agrees": agree})
        out[it["key"]] = {"same_question": len(rows_by_q.get(it["qhash"], [])) - 1, "checked": res,
                          "conflict": any(d["same_image"] and d["answer_agrees"] is False for d in res)}
    hashes = [(it["key"], int(it["phash"], 16)) for it in items if it.get("phash")]
    for i, (a, ha) in enumerate(hashes):
        near = [b for b, hb in hashes[i + 1:] if hamming(ha, hb) <= 4]
        for b in near:
            out[a].setdefault("near_in_sample", []).append(b)
            out[b].setdefault("near_in_sample", []).append(a)
    return out


# ---------------------------------------------------------------- calibrazione

def mc_text(question: str, options: list[str]) -> tuple[str, list[str]]:
    letters = "ABCDEFGH"[:len(options)]
    return question.strip() + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, options)), list(letters)


def inject_units(units: list[dict], rng: random.Random, n: int, visible_only: bool) -> dict[str, dict]:
    """Altera n affermazioni (con numeri o parole di direzione): {cid: {"original", "changed", "what"}}."""
    cands = [u for u in units if (u["visible"] or not visible_only) and mutate_text(u["text"], random.Random(0))]
    out = {}
    for u in rng.sample(cands, min(n, len(cands))):
        new, what = mutate_text(u["text"], rng)
        out[u["cid"]] = {"original": u["text"], "changed": new, "what": what}
        u["text"] = new
    return out


def calibration(args, population: list[dict], skip: set[str], rng: random.Random) -> list[dict]:
    """Elementi di calibrazione: dict come quelli del campione più "truth" (verità nota o errori introdotti)."""
    out = []
    # MAVIS: didascalia vera su caption.jpg, due affermazioni alterate, domanda dalla risposta nota
    pool = []
    for n, (key, files) in enumerate(iter_shard(args.mavis_shard)):
        if n >= 1500:
            break
        pool.append((key, files))
    pool.sort(key=lambda kf: rank(kf[0], args.seed + 20))
    for key, files in pool[:CAL["mavis"]]:
        meta = json.loads(files["json"])
        caption = meta["text_en"]["caption"].strip()
        te, choices, ai = meta["text_en"], meta.get("choices"), meta.get("answer_index")
        if te.get("question") and choices and isinstance(ai, int) and 0 <= ai < len(choices):
            question, letters = mc_text(te["question"], [str(c) for c in choices])
            gold = letters[ai]
        else:
            qs = [q for q in mavis_questions(caption, meta["depth"], key) if q["type"] in ("side_len", "angle_val", "count")]
            if not qs:
                continue
            q = qs[0]
            question, letters = mc_text(q["q"], q["options"])
            gold = letters[q["options"].index(q["answer"])]
        units = claim_units(caption)
        injected = inject_units(units, rng, 2, visible_only=False)
        out.append({"key": f"cal-mavis-{key}", "source": "_cal_mavis", "question": question, "question_raw": "<image>" + question,
                    "answer": gold, "caption": caption, "units": units, "raw": files["png"],
                    "truth": {"kind": "mavis", "gold_true": gold, "units_true": [u["cid"] for u in units if u["cid"] not in injected],
                              "units_injected": injected, "legible": True, "coherent": True}})
    # diagrammi sintetici: domanda con opzioni dalla risposta nota, niente descrizione
    pool = sorted(iter_shard(args.synth_test), key=lambda kf: rank(kf[0], args.seed + 21))[:CAL["synth"]]
    for key, files in pool:
        meta = json.loads(files["json"])
        q = random.Random(key).choice([q for q in meta["qa"] if q["type"] != "exists"] or meta["qa"])
        question, letters = mc_text(q["q"], q["options"])
        gold = letters[q["options"].index(q["answer"])]
        out.append({"key": f"cal-synth-{key}", "source": "_cal_synth", "question": question, "question_raw": "<image>" + question,
                    "answer": gold, "caption": "", "units": [], "raw": files["png"],
                    "truth": {"kind": "synth", "gold_true": gold, "units_true": [], "units_injected": {}, "legible": True,
                              "coherent": True}})
    # problemi reali: affermazioni alterate, figure rimpicciolite, domande scambiate
    real = sorted((p for p in population if p["key"] not in skip), key=lambda p: rank(p["key"], args.seed + 22))
    need = CAL["mmf_injected"] + CAL["illegible"] + 2 * CAL["incoherent"]
    cand = real[: need * 2]
    texts = load_full(cand)
    raws = image_bytes(cand)
    cand = [p for p in cand if raws.get(p["key"]) and texts[p["key"]]["question"]]
    take = iter(cand)
    for _ in range(CAL["mmf_injected"]):
        p = next(take)
        t = texts[p["key"]]
        units = claim_units(t["caption"])
        injected = inject_units(units, rng, 2, visible_only=True)
        out.append({"key": f"cal-inj-{p['key']}", "source": "_cal_injected", "question": t["question"],
                    "question_raw": t["question_raw"], "answer": t["answer"], "caption": t["caption"], "units": units,
                    "raw": raws[p["key"]], "origin": p["key"],
                    "truth": {"kind": "mmf_injected", "gold_true": None, "units_true": [], "units_injected": injected,
                              "legible": None, "coherent": None}})
    for _ in range(CAL["illegible"]):
        p = next(take)
        t = texts[p["key"]]
        out.append({"key": f"cal-tiny-{p['key']}", "source": "_cal_illegible", "question": t["question"],
                    "question_raw": t["question_raw"], "answer": t["answer"], "caption": t["caption"],
                    "units": claim_units(t["caption"]), "raw": raws[p["key"]], "shrink": 40, "origin": p["key"],
                    "truth": {"kind": "illegible", "gold_true": None, "units_true": [], "units_injected": {}, "legible": False,
                              "coherent": None}})
    pairs = [next(take) for _ in range(2 * CAL["incoherent"])]
    images, questions = pairs[::2], pairs[1::2]
    for a, b in zip(images, questions):
        ta, tb = texts[a["key"]], texts[b["key"]]
        out.append({"key": f"cal-swap-{a['key']}", "source": "_cal_incoherent", "question": tb["question"],
                    "question_raw": tb["question_raw"], "answer": tb["answer"], "caption": ta["caption"],
                    "units": claim_units(ta["caption"]), "raw": raws[a["key"]], "origin": a["key"], "question_from": b["key"],
                    "truth": {"kind": "incoherent", "gold_true": None, "units_true": [], "units_injected": {}, "legible": None,
                              "coherent": False}})
    # risposta di riferimento sbagliata apposta in un terzo degli elementi con risposta modificabile
    for it in out:
        if it["truth"]["kind"] in ("mavis", "synth", "mmf_injected") and rng.random() < 1 / 3:
            bad = wrong_answer(it["answer"], mc_options(it["question"]), rng)
            if bad and not answers_match(bad, it["answer"]):
                it["truth"]["gold_injected"] = {"original": it["answer"], "changed": bad}
                it["truth"]["gold_true"] = None
                it["answer"] = bad
    return out


# ---------------------------------------------------------------- lotti

def write_batches(out: Path, prefix: str, items: list[dict], seed: int, size: int, build) -> list[str]:
    order = sorted(items, key=lambda it: rank(it["pid"], seed))
    names = []
    for b in range(0, len(order), size):
        name = f"{prefix}-{b // size + 1:02d}"
        (out / "batches" / f"{name}.json").write_text(json.dumps([build(it) for it in order[b:b + size]], indent=1,
                                                                 ensure_ascii=False))
        names.append(name)
    return names


def solve_task(it: dict) -> dict:
    return {"id": it["pid"], "image": it["image"], "question": it["question"], "check": ["solve", "legibility", "coherence"]}


def describe_task(it: dict) -> dict:
    return {"id": it["pid"], "image": it["image"], "units": [{k: u[k] for k in ("cid", "section", "text")} for u in it["units"]]}


def f2_items(selected: Path, n: int, seed: int, by_key: dict[str, dict], out: Path) -> list[dict]:
    """Una soluzione scelta (la prima) per problema, problemi stratificati per sorgente."""
    sols = {}
    with open(selected) as fh:
        for line in fh:
            r = json.loads(line)
            sols.setdefault(r["key"], r)
    by_src = defaultdict(list)
    for key, r in sols.items():
        if key in by_key:
            by_src[r["source_dataset"]].append(key)
    alloc = allocate({s: len(v) for s, v in by_src.items()}, n)
    chosen = [k for s in sorted(alloc) for k in sorted(by_src[s], key=lambda k: rank(k, seed))[:alloc[s]]]
    texts, raws = load_full([by_key[k] for k in chosen]), image_bytes([by_key[k] for k in chosen])
    items = []
    for i, key in enumerate(sorted(chosen, key=lambda k: rank(k, seed + 1))):
        image, checks = image_checks(raws.get(key))
        if image is None:
            continue
        fid = f"f{len(items) + 1:03d}"
        path = out / "img" / f"{fid}.png"
        image.convert("RGB").save(path)
        r = sols[key]
        items.append({"fid": fid, "key": key, "source": r["source_dataset"], "question": texts[key]["question"],
                      "gold": r["answer"], "solution": r["solution"], "image": str(path)})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--parquet-glob", required=True, help="parquet sft dello snapshot (percorsi dello snapshot)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=300, help="screening: problemi in tutto, in parti uguali fra le sorgenti")
    ap.add_argument("--sources", nargs="*", default=None, help="certificazione: sorgenti da campionare")
    ap.add_argument("--per-source", type=int, default=0, help="certificazione: problemi per sorgente")
    ap.add_argument("--exclude-lists", nargs="*", default=[], help="chiavi fuori dalla popolazione (mai in training)")
    ap.add_argument("--tag-lists", nargs="*", default=[], help="chiavi solo da segnare (es. descrizioni CJK)")
    ap.add_argument("--contam-candidates", type=Path, help="exclude.txt grezzo di decontam.py scan (candidati)")
    ap.add_argument("--exclude-keys", nargs="*", default=[], type=Path, help="_gold/items.json di campioni precedenti")
    ap.add_argument("--mavis-shard", type=Path, required=True)
    ap.add_argument("--synth-test", type=Path, required=True)
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--solve-batch", type=int, default=12)
    ap.add_argument("--describe-batch", type=int, default=6)
    ap.add_argument("--f2-selected", type=Path)
    ap.add_argument("--f2-n", type=int, default=100)
    ap.add_argument("--f2-batch", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} esiste e non è vuota: un campione non si riscrive")
    for d in ("img", "batches", "answers", "_gold"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + 30)

    paths = sorted(glob.glob(args.parquet_glob))
    rows, pop_checks = scan(paths)
    by_key = {r["key"]: r for r in rows}
    excluded, tags = read_keys(args.exclude_lists), read_keys(args.tag_lists)
    out_keys = set().union(*excluded.values()) if excluded else set()
    population = [r for r in rows if r["key"] not in out_keys]
    sizes = Counter(r["source"] for r in population)
    rows_by_q = defaultdict(list)
    for r in population:
        rows_by_q[r["qhash"]].append(r)
    print(f"{len(paths)} parquet, {len(rows)} problemi con testo, popolazione {len(population)} in {len(sizes)} sorgenti "
          f"({time.time() - t0:.0f} s)", flush=True)

    skip = set()
    for f in args.exclude_keys:
        skip |= {it["key"] for it in json.loads(f.read_text())}
    if args.sources:
        missing = [s for s in args.sources if s not in sizes]
        if missing:
            raise SystemExit(f"sorgenti sconosciute: {missing}")
        alloc = {s: min(args.per_source, sizes[s]) for s in args.sources}
    else:
        alloc = allocate(dict(sizes), args.n)
    pools = defaultdict(list)
    for r in population:
        if r["key"] not in skip:
            pools[r["source"]].append(r)
    chosen = [r for s, n in sorted(alloc.items()) for r in sorted(pools[s], key=lambda r: rank(r["key"], args.seed))[:n]]
    texts = load_full(chosen)
    raws = image_bytes(chosen)
    items = []
    for r in chosen:
        t = texts[r["key"]]
        items.append({"key": r["key"], "source": r["source"], "question": t["question"], "question_raw": t["question_raw"],
                      "answer": t["answer"], "caption": t["caption"], "units": claim_units(t["caption"]),
                      "raw": raws.get(r["key"]), "qhash": r["qhash"], "qwen4b_pass_rate": r["pass_rate"],
                      "tags": sorted(name for name, keys in tags.items() if r["key"] in keys)})
    cal = [] if args.no_calibration else calibration(args, population, skip | {it["key"] for it in items}, rng)
    print(f"campione {len(items)}, calibrazione {len(cal)} ({time.time() - t0:.0f} s)", flush=True)

    contam = {line.strip() for line in open(args.contam_candidates)} if args.contam_candidates else set()
    everything = items + cal
    order = sorted(everything, key=lambda it: rank(it["key"], args.seed + 9))
    det, gold, idmap, truth = {}, {}, {}, {}
    for i, it in enumerate(order):
        pid = f"p{i + 1:03d}"
        it["pid"] = pid
        image, ic = image_checks(it.pop("raw"))
        if image is not None and it.get("shrink"):
            s = it["shrink"] / max(image.size)
            image = image.resize((max(8, round(image.width * s)), max(8, round(image.height * s))), Image.BILINEAR)
            ic["train_size"], ic["note"] = list(image.size), "rimpicciolita apposta (calibrazione)"
        if image is not None:
            path = args.out / "img" / f"{pid}.png"
            image.convert("RGB").save(path)
            it["image"] = str(path)
        else:
            it["image"] = None
        it["phash"] = ic.get("phash")
        tc = text_checks(it, it["answer"])
        det[pid] = {"image": ic, "text": tc, "contamination_candidate": it["key"] in contam,
                    "calibration": it["source"].startswith("_cal")}
        gold[pid] = it["answer"]
        idmap[pid] = {"key": it["key"], "source": it["source"]}
        if "truth" in it:
            truth[pid] = it.pop("truth")
    dups = duplicate_checks([it for it in items if it.get("phash")], rows_by_q, texts)
    for it in items:
        det[it["pid"]]["duplicates"] = dups.get(it["key"], {"same_question": it.get("qhash") and len(rows_by_q[it["qhash"]]) - 1,
                                                             "checked": [], "conflict": False})
    print(f"controlli deterministici fatti ({time.time() - t0:.0f} s)", flush=True)

    master = [{k: it[k] for k in ("pid", "key", "source", "question", "caption", "units", "image") if k in it}
              | {"tags": it.get("tags", []), "calibration": it["source"].startswith("_cal")} for it in order]
    (args.out / "_gold" / "items.json").write_text(json.dumps(master, indent=1, ensure_ascii=False))
    for name, obj in (("gold.json", gold), ("idmap.json", idmap), ("calibration.json", truth)):
        (args.out / "_gold" / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    (args.out / "deterministic.json").write_text(json.dumps(det, indent=1, ensure_ascii=False))
    (args.out / "population_checks.json").write_text(json.dumps(pop_checks, indent=1, ensure_ascii=False))
    (args.out / "source_sizes.json").write_text(json.dumps(dict(sizes.most_common()), indent=1, ensure_ascii=False))

    ok = [it for it in order if it["image"]]
    batches = {"SA": write_batches(args.out, "SA", ok, args.seed + 1, args.solve_batch, solve_task),
               "SB": write_batches(args.out, "SB", ok, args.seed + 2, args.solve_batch, solve_task)}
    described = [it for it in ok if it["units"]]
    batches["DA"] = write_batches(args.out, "DA", described, args.seed + 3, args.describe_batch, describe_task)
    batches["DB"] = write_batches(args.out, "DB", described, args.seed + 4, args.describe_batch, describe_task)
    units = [len(it["units"]) for it in described]
    print(f"lotti: risolvere {len(batches['SA'])} x 2, descrizioni {len(batches['DA'])} x 2; affermazioni per descrizione "
          f"media {sum(units) / max(1, len(units)):.1f}, massimo {max(units, default=0)}", flush=True)

    f2 = []
    if args.f2_selected:
        f2 = f2_items(args.f2_selected, args.f2_n, args.seed + 5, by_key, args.out)
        (args.out / "_gold" / "items_f2.json").write_text(json.dumps(f2, indent=1, ensure_ascii=False))
        task = lambda it: {"id": it["fid"], "image": it["image"], "question": it["question"], "solution": it["solution"]}  # noqa: E731
        f2p = [{**it, "pid": it["fid"]} for it in f2]
        batches["FA"] = write_batches(args.out, "FA", f2p, args.seed + 6, args.f2_batch, task)
        batches["FB"] = write_batches(args.out, "FB", f2p, args.seed + 7, args.f2_batch, task)
        print(f"soluzioni F2: {len(f2)} in {len(batches['FA'])} lotti x 2", flush=True)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.count("\n")
    flags = Counter(f for d in det.values() if not d["calibration"] for f in d["image"].get("flags", []) + d["text"]["flags"])
    manifest = {"args": {k: str(v) for k, v in vars(args).items()}, "commit": commit, "dirty_files": dirty,
                "parquet": len(paths), "problems_with_text": len(rows), "population": len(population),
                "excluded": {name: sum(r["key"] in keys for r in rows) for name, keys in excluded.items()},
                "allocation": alloc, "sampled": len(items), "calibration": dict(Counter(t["kind"] for t in truth.values())),
                "gold_injected": sum("gold_injected" in t for t in truth.values()),
                "units_injected": sum(len(t["units_injected"]) for t in truth.values()),
                "unreadable": sum(1 for d in det.values() if not d["image"]["ok"]), "deterministic_flags": dict(flags),
                "duplicate_conflicts": sum(1 for d in det.values() if d.get("duplicates", {}).get("conflict")),
                "contamination_candidates": sum(d["contamination_candidate"] for d in det.values()),
                "batches": batches, "f2_items": len(f2), "seconds": round(time.time() - t0, 1)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("population", "excluded", "sampled", "calibration", "gold_injected",
                                               "units_injected", "unreadable", "deterministic_flags",
                                               "duplicate_conflicts", "contamination_candidates", "f2_items")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
