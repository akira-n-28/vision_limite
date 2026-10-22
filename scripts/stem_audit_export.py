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
- batches/FA-NN.json, FB-NN.json (con --f2-selected): soluzioni F2 di Violetto da verificare contro l'immagine; con
  --f2-v2 (selezione v2) l'unità è la soluzione, gli item portano sid e hash, la calibrazione ha misure, posizioni e
  controlli negativi (f2_v2_items) e il certificato si calcola poi con scripts/f2_certify.py;
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
from limite_vl.f2_records import record_version  # noqa: E402
from limite_vl.mmfine import mmfine_key, rank, split_of  # noqa: E402
from limite_vl.provenance import code_identity, sha256_file, sha256_text  # noqa: E402
from limite_vl.stem_audit import (allocate, answer_letter, claim_units, mc_options, mutate_text,  # noqa: E402
                                  norm_question, wrong_answer)

Image.MAX_IMAGE_PIXELS = None
# Elementi di calibrazione per tipo e frazione con la risposta di riferimento sbagliata apposta. "cert" (certificazione
# per RL, senza descrizioni) ha più risposte sbagliate introdotte, da cui dipende il limite inferiore della sensibilità.
CAL_PROFILES = {"screening": ({"mavis": 30, "mmf_injected": 30, "synth": 20, "illegible": 10, "incoherent": 10}, 1 / 3),
                "cert": ({"mavis": 30, "mmf_injected": 60, "synth": 30, "illegible": 15, "incoherent": 15}, 1 / 2)}


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
    CAL, gold_frac = CAL_PROFILES[args.cal_profile]
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
    # risposta di riferimento sbagliata apposta in una frazione (gold_frac) degli elementi con risposta modificabile
    for it in out:
        if it["truth"]["kind"] in ("mavis", "synth", "mmf_injected") and rng.random() < gold_frac:
            bad = wrong_answer(it["answer"], mc_options(it["question"]), rng)
            if bad and not answers_match(bad, it["answer"]):
                it["truth"]["gold_injected"] = {"original": it["answer"], "changed": bad}
                it["truth"]["gold_true"] = None
                it["answer"] = bad
    return out


# ---------------------------------------------------------------- MAVIS-Geometry (dati F2 esatti per costruzione)

MAVIS_CAL = {"units": 15, "gold": 20, "illegible": 10, "incoherent": 10}


def mavis_items(args, skip_ids: set[str], rng: random.Random, recheck_ids: set[str] = frozenset()
                ) -> tuple[list[dict], list[dict], Counter, dict]:
    """Campione di --mavis-sample problemi MAVIS-Geometry (in parti uguali fra le profondità 0-3) come li vedrà F2:
    caption.jpg, domanda con la struttura della figura e le opzioni, ponte con la didascalia riscritta
    (limite_vl.mavis); e una calibrazione dagli stessi shard (affermazioni alterate, risposte sbagliate apposta, figure
    rimpicciolite, domande scambiate fra profondità diverse). I problemi di recheck_ids (esclusi o in quarantena in un
    audit precedente) entrano tutti, come sorgente "MAVIS-Geometry ricontrollo"."""
    import heapq

    from limite_vl.mavis import f2_problem

    per_depth = args.mavis_sample // 4
    reserve = 18  # per profondità, per la calibrazione
    heaps, sizes, failed, rechecks = defaultdict(list), Counter(), Counter(), []
    for shard in sorted(glob.glob(args.mavis_shards)):
        for key, files in iter_shard(shard):
            meta = json.loads(files["json"])
            src = f"MAVIS-Geometry d{meta['depth']}"
            p = f2_problem(meta)
            if p is None:
                failed[src] += 1
                continue
            sizes[src] += 1
            if meta["id"] in recheck_ids:
                rechecks.append((p, files["png"]))
                continue
            if meta["id"] in skip_ids:
                continue
            entry = (-rank(p["key"], args.seed), p["key"], p, files["png"])
            h = heaps[src]
            if len(h) < per_depth + reserve:
                heapq.heappush(h, entry)
            elif entry[0] > h[0][0]:
                heapq.heapreplace(h, entry)
    def item(p: dict, png: bytes, src: str) -> dict:
        return {"key": p["key"], "source": src, "question": p["question"], "question_raw": "<image>" + p["question"],
                "answer": p["answer_letter"], "caption": p["caption"], "units": claim_units(p["caption"]), "raw": png,
                "qhash": None, "qwen4b_pass_rate": None, "tags": [f"riscritte {p['rewritten']}"] if p["rewritten"] else []}

    missing = recheck_ids - {p["key"][len("mavis-"):] for p, _ in rechecks}
    if missing:
        raise SystemExit(f"problemi da ricontrollare non trovati o non più validi: {sorted(missing)}")
    items = [item(p, png, "MAVIS-Geometry ricontrollo") for p, png in sorted(rechecks, key=lambda x: x[0]["key"])]
    pool = []
    for src in sorted(heaps):
        ranked = [e[2:] for e in sorted(heaps[src], key=lambda e: -e[0])]
        for i, (p, png) in enumerate(ranked):
            (items if i < per_depth else pool).append(item(p, png, src))
    pool.sort(key=lambda it: rank(it["key"], args.seed + 22))
    take = iter(pool)
    cal = []
    for _ in range(MAVIS_CAL["units"]):
        it = dict(next(take))
        units = claim_units(it["caption"])
        injected = inject_units(units, rng, 2, visible_only=True)
        cal.append(it | {"key": f"cal-mavis-units-{it['key']}", "source": "_cal_mavis", "units": units, "tags": [],
                         "truth": {"kind": "mavis", "gold_true": it["answer"], "units_true": [u["cid"] for u in units if u["cid"] not in injected],
                                   "units_injected": injected, "legible": True, "coherent": True}})
    for _ in range(MAVIS_CAL["gold"]):
        it = dict(next(take))
        bad = wrong_answer(it["answer"], mc_options(it["question"]), rng)
        cal.append(it | {"key": f"cal-mavis-gold-{it['key']}", "source": "_cal_mavis", "answer": bad, "tags": [],
                         "truth": {"kind": "mavis", "gold_true": None, "gold_injected": {"original": it["answer"], "changed": bad},
                                   "units_true": [u["cid"] for u in it["units"]], "units_injected": {}, "legible": True, "coherent": True}})
    for _ in range(MAVIS_CAL["illegible"]):
        it = dict(next(take))
        cal.append(it | {"key": f"cal-tiny-{it['key']}", "source": "_cal_illegible", "shrink": 40, "tags": [],
                         "truth": {"kind": "illegible", "gold_true": None, "units_true": [], "units_injected": {}, "legible": False,
                                   "coherent": None}})
    rest = list(take)
    pairs = [(a, b) for a, b in zip(rest[::2], rest[1::2]) if a["source"] != b["source"]][:MAVIS_CAL["incoherent"]]
    for a, b in pairs:
        cal.append(a | {"key": f"cal-swap-{a['key']}", "source": "_cal_incoherent", "question": b["question"],
                        "question_raw": b["question_raw"], "answer": b["answer"], "question_from": b["key"], "tags": [],
                        "truth": {"kind": "incoherent", "gold_true": None, "units_true": [], "units_injected": {}, "legible": None,
                                  "coherent": False}})
    checks = {src: {"popolazione": sizes[src], "didascalia_non_riscrivibile": failed[src]} for src in sorted(set(sizes) | set(failed))}
    return items, cal, sizes, checks


_SOL_NUMBER = re.compile(r"(?<![\w.^/])(\d+(?:\.\d+)?)(?!\.?\d|[\w^/])")  # anche "30." a fine frase


def inject_solution(solution: str, caption: str, rng: random.Random) -> tuple[str, dict] | None:
    """Errore introdotto in una soluzione: una misura della figura (un numero delle frasi con cifre della didascalia,
    tutte disegnate in caption.jpg) cambiata in un punto del ragionamento. None se la soluzione non ne cita."""
    measures = {m.group(1) for s in re.split(r"(?<=\.)\s+", caption) if re.search(r"\d", s)
                for m in _SOL_NUMBER.finditer(s)} - {"0", "1", "90", "180", "360"}
    end = solution.rfind("\\boxed")
    spots = [m for m in _SOL_NUMBER.finditer(solution[:end if end > 0 else len(solution)]) if m.group(1) in measures]
    if not spots:
        return None
    m = rng.choice(spots)
    v = m.group(1)
    if "." in v:
        new = f"{float(v) + rng.choice([0.5, 1.5, 2.5]):.{len(v.split('.')[1])}f}"
    else:
        iv = int(v)
        new = str(iv + rng.choice([2, 3, 4, 5]) if iv < 20 or rng.random() < 0.5 else iv * 2)
    text = solution[:m.start(1)] + new + solution[m.end(1):]
    return text, {"original": v, "changed": new, "context": text[max(0, m.start(1) - 80):m.start(1) + 80],
                  "start": m.start(1), "end": m.start(1) + len(new)}


# Parole di posizione e orientamento (sottoinsieme di stem_audit.SWAPS più alto/basso): l'errore dominante delle soluzioni
# del ponte MAVIS è la disposizione (mavis-f2sol, 132 citazioni su 138), che le misure cambiate non calibrano.
LAYOUT_SWAPS = [("left", "right"), ("above", "below"), ("horizontal", "vertical"), ("upward", "downward"),
                ("inside", "outside"), ("top", "bottom"), ("upper", "lower"), ("clockwise", "counterclockwise")]


def inject_layout(solution: str, rng: random.Random) -> tuple[str, dict] | None:
    """Errore di disposizione introdotto nel ragionamento: una parola di posizione o orientamento sostituita con
    l'opposta, prima di </think>. Il testo originale non è verificato: se era già falso e lo scambio lo rende vero,
    l'errore risulta non trovato e la sensibilità esce più bassa del vero (direzione prudente). None se non ce ne sono."""
    end = solution.find("</think>")
    head = solution[:end if end > 0 else len(solution)]
    spots = [(m, y) for a, b in LAYOUT_SWAPS for x, y in ((a, b), (b, a)) for m in re.finditer(rf"\b{x}\b", head, re.IGNORECASE)]
    if not spots:
        return None
    m, repl = rng.choice(sorted(spots, key=lambda s: s[0].start()))
    word = m.group(0)
    repl = repl.upper() if word.isupper() else repl.capitalize() if word[0].isupper() else repl
    text = solution[:m.start()] + repl + solution[m.end():]
    return text, {"original": word, "changed": repl, "context": text[max(0, m.start() - 80):m.start() + 80],
                  "start": m.start(), "end": m.start() + len(repl)}


def read_v2_selection(path: Path) -> tuple[list[dict], dict]:
    """Record f2-sol/2 di una selezione (f2_select.py o f2_backfill.py) e voce del file con lo sha256."""
    h, recs = hashlib.sha256(), []
    with open(path, "rb") as fh:
        for n, line in enumerate(fh, 1):
            h.update(line)
            rec = json.loads(line)
            if record_version(rec) != 2:
                raise SystemExit(f"{path}:{n}: --f2-v2 vuole una selezione v2 (selected.v2.jsonl), non selected.jsonl")
            recs.append(rec)
    return recs, {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def f2_v2_items(args, out: Path, rng: random.Random) -> tuple[list[dict], dict, dict]:
    """Audit v2 delle soluzioni F2 (unità = soluzione, revisione tecnica sezione 6): campione di --f2-n soluzioni fra
    quelle della selezione che superano i controlli che scartano, stratificato per source_dataset (profondità per
    MAVIS); ogni item porta sid, text_sha256 e question_sha256. Calibrazione da altre soluzioni della stessa
    selezione: --f2-cal-measure misure cambiate, --f2-cal-layout parole di posizione scambiate, --f2-negatives
    controlli negativi (MAVIS: il CoT del motore, coerente con la figura, dei problemi con didascalia senza frasi
    riscritte). Domanda: quella salvata nella generazione; se manca, ricostruita (MAVIS con f2_problem, MMFineReason dal
    parquet), e una domanda salvata diversa da quella ricostruita è contata in question_mismatch."""
    from limite_vl.mavis import f2_problem

    recs, sel_entry = read_v2_selection(args.f2_selected)
    eligible = [r for r in recs if not r["failed"]]
    if len({r["sid"] for r in eligible}) != len(eligible):
        raise SystemExit("sid ripetuti nella selezione")
    by_src = defaultdict(list)
    for r in eligible:
        by_src[r["source_dataset"]].append(r)
    alloc = allocate({s: len(v) for s, v in by_src.items()}, args.f2_n)
    chosen = [r for s in sorted(alloc) for r in sorted(by_src[s], key=lambda r: rank(r["sid"], args.seed))[:alloc[s]]]
    chosen_sids = {r["sid"] for r in chosen}
    pool = sorted((r for r in eligible if r["sid"] not in chosen_sids), key=lambda r: rank(r["sid"], args.seed + 3))

    keys = {r["key"] for r in eligible}
    problems: dict[str, dict] = {}  # chiave -> domanda ricostruita, didascalia, immagine, cot
    mavis = all(k.startswith("mavis-") for k in keys)
    if mavis:
        if not args.mavis_shards:
            raise SystemExit("soluzioni MAVIS: serve --mavis-shards")
        for shard in sorted(glob.glob(args.mavis_shards)):
            for _, files in iter_shard(shard):
                meta = json.loads(files["json"])
                key = f"mavis-{meta['id']}"
                if key in keys and (p := f2_problem(meta)):
                    problems[key] = {"question": p["question"], "caption": p["caption"], "png": files["png"],
                                     "cot": p["cot"] if p["rewritten"] == 0 else "", "letter": p["answer_letter"]}
    else:
        if not args.parquet_glob:
            raise SystemExit("soluzioni MMFineReason: serve --parquet-glob")
        rows, _ = scan(sorted(glob.glob(args.parquet_glob)))
        need = [r for r in rows if r["key"] in keys]
        texts, raws = load_full(need), image_bytes(need)
        problems = {k: {"question": texts[k]["question"], "caption": texts[k]["caption"], "png": raws.get(k), "cot": "",
                        "letter": None} for k in texts}

    cal: list[tuple[str, dict, str, dict]] = []  # (tipo, record base, testo, verità)
    want = {"measure": args.f2_cal_measure, "layout": args.f2_cal_layout, "negative": args.f2_negatives}
    used: set[str] = set()
    for kind in ("measure", "layout", "negative"):
        n_kind = 0
        for r in pool:
            if n_kind >= want[kind]:
                break
            p = problems.get(r["key"])
            if r["sid"] in used or p is None:
                continue
            if kind == "measure":
                got = inject_solution(r["solution"], p["caption"], rng)
            elif kind == "layout":
                got = inject_layout(r["solution"], rng)
            else:
                got = (f"<think>{p['cot'].strip()}</think>\n\nThe answer is \\boxed{{{p['letter']}}}.", {}) if p["cot"] else None
            if got:
                cal.append((kind, r, got[0], got[1]))
                used.add(r["sid"])
                n_kind += 1

    entries = [("real", r, r["solution"], {}) for r in chosen] + cal
    items, truth, mismatch = [], {}, 0
    for kind, r, text, info in sorted(entries, key=lambda e: rank(f"{e[0]}:{e[1]['sid']}", args.seed + 1)):
        p = problems.get(r["key"])
        image, _ = image_checks(p["png"] if p else None)
        if p is None or image is None:
            continue
        if r.get("question") is not None and r["question"].strip() != p["question"].strip():
            mismatch += 1
        question = r["question"] if r.get("question") is not None else p["question"]
        fid = f"f{len(items) + 1:03d}"
        path = out / "img" / f"{fid}.png"
        image.convert("RGB").save(path)
        it = {"fid": fid, "key": r["key"], "source": r["source_dataset"] if kind == "real" else "_cal_f2", "question": question,
              "question_sha256": r.get("question_sha256"), "gold": r["answer"], "solution": text, "image": str(path),
              "calibration": kind != "real", "text_sha256": sha256_text(text)}
        if kind == "real":
            it["sid"] = r["sid"]
        else:
            it.update({"sid": None, "source_sid": r["sid"], "source_text_sha256": r["text_sha256"]})
            truth[fid] = {"kind": kind, "key": r["key"], "source_sid": r["sid"], **info}
        items.append(it)
    design = {"unit": "soluzione", "strata": "source_dataset", "seed": args.seed, "sample": sum(not it["calibration"] for it in items),
              "allocation": alloc, "eligible": len(eligible), "selection_records": len(recs),
              "eligible_by_source": {s: len(v) for s, v in sorted(by_src.items())},
              "calibration": dict(Counter(t["kind"] for t in truth.values())), "calibration_requested": want,
              "recheck_frac": args.f2_recheck_frac, "question_mismatch": mismatch,
              "questions_saved": sum(it["question_sha256"] is not None for it in items)}
    return items, truth, {"selection": sel_entry, "design": design}


def mavis_f2_items(selected: Path, n: int, n_cal: int, seed: int, shards: str, out: Path,
                   rng: random.Random) -> tuple[list[dict], dict]:
    """Audit delle soluzioni F2 di Violetto sui problemi MAVIS (f2_select.py): una soluzione per problema, n problemi
    in parti uguali fra le profondità, più n_cal soluzioni di altri problemi con una misura della figura cambiata
    (verità nota: visual_ok falso). La domanda è quella di F2 (limite_vl.mavis.f2_problem), l'immagine caption.jpg."""
    from limite_vl.mavis import f2_problem

    sols = {}
    with open(selected) as fh:
        for line in fh:
            r = json.loads(line)
            sols.setdefault(r["key"], r)
    by_src = defaultdict(list)
    for key, r in sols.items():
        by_src[r["source_dataset"]].append(key)
    alloc = allocate({s: len(v) for s, v in by_src.items()}, n)
    chosen = {k for s in sorted(alloc) for k in sorted(by_src[s], key=lambda k: rank(k, seed))[:alloc[s]]}
    rest = sorted((k for k in sols if k not in chosen), key=lambda k: rank(k, seed + 3))
    wanted, metas, pngs = set(sols), {}, {}
    for shard in sorted(glob.glob(shards)):
        for _, files in iter_shard(shard):
            meta = json.loads(files["json"])
            key = f"mavis-{meta['id']}"
            if key in wanted:
                metas[key], pngs[key] = meta, files["png"]
    cal, truth = [], {}
    for key in rest:
        if len(cal) >= n_cal:
            break
        p = f2_problem(metas[key]) if key in metas else None
        got = inject_solution(sols[key]["solution"], p["caption"], rng) if p else None
        if got:
            cal.append((key, got[0]))
            truth[key] = got[1]
    entries = [(k, sols[k]["solution"], False) for k in chosen] + [(k, s, True) for k, s in cal]
    items, cal_truth = [], {}
    for key, solution, is_cal in sorted(entries, key=lambda e: rank(e[0], seed + 1)):
        p = f2_problem(metas[key]) if key in metas else None
        image, _ = image_checks(pngs.get(key))
        if p is None or image is None:
            continue
        fid = f"f{len(items) + 1:03d}"
        path = out / "img" / f"{fid}.png"
        image.convert("RGB").save(path)
        items.append({"fid": fid, "key": key, "source": "_cal_f2" if is_cal else sols[key]["source_dataset"],
                      "question": p["question"], "gold": sols[key]["answer"], "solution": solution, "image": str(path),
                      "calibration": is_cal})
        if is_cal:
            cal_truth[fid] = {"kind": "f2_injected", "key": key, **truth[key]}
    return items, cal_truth


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
    ap.add_argument("--parquet-glob", help="parquet sft dello snapshot (percorsi dello snapshot)")
    ap.add_argument("--mavis-sample", type=int, default=0,
                    help="audit di MAVIS-Geometry come dati F2: problemi in tutto (in parti uguali fra le profondità)")
    ap.add_argument("--mavis-shards", help="glob degli shard caption di MAVIS-Geometry (con --mavis-sample)")
    ap.add_argument("--mavis-recheck", nargs="*", default=[], type=Path,
                    help="liste di chiavi mavis-<id> da ricontrollare tutte (es. exclude.txt e quarantine.txt di un audit precedente)")
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
    ap.add_argument("--cal-profile", choices=sorted(CAL_PROFILES), default="screening")
    ap.add_argument("--no-describe", action="store_true",
                    help="niente lotti DA/DB (certificazione per RL: la descrizione non entra in training)")
    ap.add_argument("--solve-batch", type=int, default=12)
    ap.add_argument("--describe-batch", type=int, default=6)
    ap.add_argument("--f2-selected", type=Path)
    ap.add_argument("--f2-n", type=int, default=100)
    ap.add_argument("--f2-batch", type=int, default=6)
    ap.add_argument("--f2-mavis", action="store_true",
                    help="solo l'audit delle soluzioni F2 di Violetto sui problemi MAVIS (--f2-selected, --mavis-shards), "
                         "senza campione di problemi")
    ap.add_argument("--f2-cal", type=int, default=0, help="con --f2-mavis: soluzioni con una misura della figura cambiata")
    ap.add_argument("--f2-v2", action="store_true",
                    help="audit v2 delle soluzioni F2 (--f2-selected = selected.v2.jsonl): unità = soluzione, sid e hash negli "
                         "item, calibrazione per tipo, controlli negativi, ricontrollo dei consensi; istruzioni *_v2.md")
    ap.add_argument("--f2-cal-measure", type=int, default=60, help="con --f2-v2: soluzioni con una misura cambiata")
    ap.add_argument("--f2-cal-layout", type=int, default=60, help="con --f2-v2: soluzioni con una parola di posizione scambiata")
    ap.add_argument("--f2-negatives", type=int, default=30, help="con --f2-v2: controlli negativi (solo MAVIS)")
    ap.add_argument("--f2-recheck-frac", type=float, default=0.15,
                    help="con --f2-v2: frazione delle soluzioni approvate da FA e FB all'unanimità ricontrollata da FC")
    args = ap.parse_args()
    if args.f2_v2 and not args.f2_selected:
        raise SystemExit("--f2-v2 vuole --f2-selected (selected.v2.jsonl)")
    t0 = time.time()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} esiste e non è vuota: un campione non si riscrive")
    for d in ("img", "batches", "answers", "_gold"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + 30)

    skip = set()
    for f in args.exclude_keys:
        skip |= {it["key"] for it in json.loads(f.read_text())}
    if args.f2_mavis or args.f2_v2:  # solo le soluzioni (lotti FA/FB), nessun campione di problemi
        if args.f2_mavis and not (args.mavis_shards and args.f2_selected):
            raise SystemExit("--f2-mavis vuole --mavis-shards e --f2-selected")
        items, cal, sizes, pop_checks = [], [], Counter(), {}
        paths, rows, population, by_key, excluded, rows_by_q, texts, alloc = [], [], [], {}, {}, defaultdict(list), {}, {}
    elif args.mavis_sample:
        if not args.mavis_shards:
            raise SystemExit("--mavis-sample vuole --mavis-shards")
        skip_ids = ({k.split("cal-mavis-", 1)[1] for k in skip if k.startswith("cal-mavis-")}
                    | {m[1] for k in skip if (m := re.search(r"mavis-(d\d-\d+)$", k))})  # anche cal-*-mavis-<id>, campioni MAVIS
        recheck_ids = {k[len("mavis-"):] for f in args.mavis_recheck for k in f.read_text().split() if k.startswith("mavis-")}
        items, cal, sizes, pop_checks = mavis_items(args, skip_ids, rng, recheck_ids)
        cal = [] if args.no_calibration else cal
        paths, rows, population, by_key, excluded, rows_by_q, texts = [], [], [], {}, {}, defaultdict(list), {}
        alloc = {src: sum(it["source"] == src for it in items) for src in sorted(set(sizes) | {it["source"] for it in items})}
        print(f"MAVIS: popolazione {sum(sizes.values())}, campione {len(items)}, calibrazione {len(cal)} "
              f"({time.time() - t0:.0f} s)", flush=True)
    else:
        if not args.parquet_glob:
            raise SystemExit("serve --parquet-glob (oppure --mavis-sample)")
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
    described = [] if args.no_describe else [it for it in ok if it["units"]]
    if described:
        batches["DA"] = write_batches(args.out, "DA", described, args.seed + 3, args.describe_batch, describe_task)
        batches["DB"] = write_batches(args.out, "DB", described, args.seed + 4, args.describe_batch, describe_task)
    else:
        batches["DA"] = batches["DB"] = []
    units = [len(it["units"]) for it in described]
    print(f"lotti: risolvere {len(batches['SA'])} x 2, descrizioni {len(batches['DA'])} x 2; affermazioni per descrizione "
          f"media {sum(units) / max(1, len(units)):.1f}, massimo {max(units, default=0)}", flush=True)

    f2, f2_truth, f2_v2 = [], {}, None
    if args.f2_v2:
        f2, f2_truth, f2_v2 = f2_v2_items(args, args.out, rng)
        (args.out / "_gold" / "calibration_f2.json").write_text(json.dumps(f2_truth, indent=1, ensure_ascii=False))
        files = {"FA": "f2_solution_v2.md", "FB": "f2_solution_v2.md", "FC": "f2_solution_v2.md", "FR1": "f2_arbiter_v2.md",
                 "FR2": "f2_arbiter_v2.md", "rules": "_rules.md"}
        f2_v2["instructions"] = {r: {"path": f"config/audit/{f}", "sha256": sha256_file(ROOT / "config" / "audit" / f)}
                                 for r, f in files.items()}
        f2_v2["code"] = code_identity()
    elif args.f2_mavis:
        f2, f2_truth = mavis_f2_items(args.f2_selected, args.f2_n, args.f2_cal, args.seed + 5, args.mavis_shards, args.out, rng)
        (args.out / "_gold" / "calibration_f2.json").write_text(json.dumps(f2_truth, indent=1, ensure_ascii=False))
    elif args.f2_selected and not args.mavis_sample:  # le soluzioni F2 di Violetto sono di MMFineReason
        f2 = f2_items(args.f2_selected, args.f2_n, args.seed + 5, by_key, args.out)
    if f2:
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
                "parquet": len(paths), "problems_with_text": len(rows), "population": len(population) or sum(sizes.values()),
                "excluded": {name: sum(r["key"] in keys for r in rows) for name, keys in excluded.items()},
                "allocation": alloc, "sampled": len(items), "calibration": dict(Counter(t["kind"] for t in truth.values())),
                "gold_injected": sum("gold_injected" in t for t in truth.values()),
                "units_injected": sum(len(t["units_injected"]) for t in truth.values()),
                "unreadable": sum(1 for d in det.values() if not d["image"]["ok"]), "deterministic_flags": dict(flags),
                "duplicate_conflicts": sum(1 for d in det.values() if d.get("duplicates", {}).get("conflict")),
                "contamination_candidates": sum(d["contamination_candidate"] for d in det.values()),
                "batches": batches, "f2_items": len(f2), "f2_calibration": len(f2_truth), "seconds": round(time.time() - t0, 1),
                "mavis_question": "domanda con la struttura della figura e la richiesta di vision_dominant (structure_question)"
                if args.mavis_sample else None}
    if f2_v2 is not None:
        manifest |= {"audit_version": 2, "f2_v2": f2_v2}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("population", "excluded", "sampled", "calibration", "gold_injected",
                                               "units_injected", "unreadable", "deterministic_flags",
                                               "duplicate_conflicts", "contamination_candidates", "f2_items")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
