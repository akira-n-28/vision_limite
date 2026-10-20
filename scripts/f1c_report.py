#!/usr/bin/env python3
"""Confronto di F1b, controllo e F1c sulle valutazioni di scripts/f1c_eval.py. Solo libreria standard (job CPU).

Per ogni criterio del piano di F1c riporta il valore di ogni modello, le differenze rispetto a F1b e al controllo
con intervallo al 95% (bootstrap appaiato sui campioni, 2000 ricampionamenti) e, per gli esiti sì/no, il p-value
del test di McNemar esatto; poi esempi di errori corretti e introdotti. I modelli si passano come NOME=CARTELLA;
servono i nomi f1b, control e f1c.

    python scripts/f1c_report.py --out OUT.json f1b=DIR control=DIR f1c=DIR
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from f1_error_report import formula_analysis, is_loop, mean, token_tables  # noqa: E402

from limite_vl.facts import compare_facts, mavis_facts  # noqa: E402

RELATIONS = {"quadrant", "side_x", "side_y", "pair_lr", "pair_ud", "highest", "lowest", "leftmost", "rightmost",
             "m_side", "line_on", "vertex_ud", "vertex_lr", "intercept", "shared_side", "relation"}
NUMBERS = {"coords", "side_len", "angle_val", "count", "count_type", "intersections", "value"}
RECOGNITION = {"shape_type", "label_type", "has_type", "exists", "opens", "slope", "right_vertex", "label"}


def group_of(qtype: str) -> str:
    return "relazioni" if qtype in RELATIONS else "valori" if qtype in NUMBERS else "riconoscimento"


def bootstrap(pairs: list[tuple[float, float]], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Differenza media b - a con intervallo al 95% (bootstrap appaiato)."""
    if not pairs:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    diff = [b - a for a, b in pairs]
    point = sum(diff) / len(diff)
    stats = sorted(sum(rng.choice(diff) for _ in diff) / len(diff) for _ in range(n))
    return point, stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def mcnemar(pairs: list[tuple[bool, bool]]) -> float:
    """p-value bilaterale esatto sulle coppie discordanti."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(per_model: dict[str, dict], metric: str, binary: bool = True) -> dict:
    """per_model: {modello: {id: valore}}. Differenze di f1c rispetto a f1b e control sugli id comuni."""
    out = {m: mean(v.values()) for m, v in per_model.items()}
    for base in ("f1b", "control"):
        if base not in per_model or "f1c" not in per_model:
            continue
        ids = sorted(set(per_model[base]) & set(per_model["f1c"]))
        pairs = [(float(per_model[base][i]), float(per_model["f1c"][i])) for i in ids]
        d, lo, hi = bootstrap(pairs)
        out[f"f1c-{base}"] = {"diff": d, "ci95": [lo, hi], "n": len(ids)}
        if binary:
            out[f"f1c-{base}"]["mcnemar_p"] = mcnemar([(bool(a), bool(b)) for a, b in pairs])
    return out


# ---------------------------------------------------------------- MAVIS

def mavis_rows(rows: list[dict]) -> dict:
    """Per didascalia con immagine vera: figura inventata, del tutto giusta, conteggi per il richiamo."""
    out = {}
    for r in rows:
        if r.get("source") != "mavis" or r["mode"] != "real":
            continue
        c = compare_facts(mavis_facts(r["reference"]), mavis_facts(r["output"]))
        bad = sum(c[k]["wrong"] + c[k]["extra"] for k in ("shape", "angle", "side"))
        miss = sum(c[k]["missing"] for k in ("shape", "angle", "side"))
        fluent = r["eos"] and not is_loop(r["output"], r["eos"])
        out[r["key"]] = {"invented": c["shape"]["gen"] > c["shape"]["ref"], "correct": fluent and not bad and not miss,
                         "facts": c, "output": r["output"], "reference": r["reference"]}
    return out


def mavis_dependence(rows: list[dict]) -> dict:
    prec = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("source") != "mavis":
            continue
        c = compare_facts(mavis_facts(r["reference"]), mavis_facts(r["output"]))
        for k in ("shape", "angle", "side"):
            prec[r["mode"]][0] += c[k]["ok"]
            prec[r["mode"]][1] += c[k]["gen"]
    return {m: v[0] / max(1, v[1]) for m, v in prec.items()}


def recall(rows: dict, keys: list[str], kind: str) -> float:
    ok = sum(rows[k]["facts"][kind]["ok"] for k in keys)
    return ok / max(1, sum(rows[k]["facts"][kind]["ref"] for k in keys))


# ---------------------------------------------------------------- report

def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="NOME=CARTELLA di f1c_eval (f1b, control, f1c)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    dirs = {m.partition("=")[0]: Path(m.partition("=")[2]) for m in args.models}
    rep: dict = {"models": {k: str(v) for k, v in dirs.items()}, "criteria": {}}
    gen = {m: load(d / "gen.jsonl") for m, d in dirs.items()}
    choice = {m: load(d / "choice.jsonl") for m, d in dirs.items()}

    # MAVIS, didascalie: stessi 80 dell'analisi e tutti i 256
    mav = {m: mavis_rows(rows) for m, rows in gen.items()}
    keys256 = sorted(set.intersection(*(set(v) for v in mav.values()))) if mav else []
    first80 = [r["key"] for r in gen["f1b"] if r.get("source") == "mavis" and r["mode"] == "real"][:80]
    for label, keys in (("80", first80), ("256", keys256)):
        sec = {}
        for metric in ("invented", "correct"):
            sec[metric] = compare({m: {k: v[k][metric] for k in keys if k in v} for m, v in mav.items()}, metric)
        sec["recall"] = {m: {kind: recall(v, keys, kind) for kind in ("shape", "angle", "side")} for m, v in mav.items()}
        rep[f"mavis_{label}"] = sec
    rep["mavis_dependence"] = {m: mavis_dependence(rows) for m, rows in gen.items()}
    m80 = rep["mavis_80"]
    rep["criteria"]["mavis_inventate<=43/80"] = {m: round(80 * m80["invented"][m]) for m in mav}
    rep["criteria"]["mavis_giuste>=15/80"] = {m: round(80 * m80["correct"][m]) for m in mav}
    rep["criteria"]["mavis_richiamo_calo<=3pp"] = {
        kind: {m: round(100 * (m80["recall"][m][kind] - m80["recall"]["f1b"][kind]), 1) for m in mav} for kind in ("shape", "angle", "side")}

    # UniMER: per insieme di input, stessi 200 (indice < 100) e indipendenti (100-199)
    rep["unimer"] = {}
    for m, d in dirs.items():
        for path in sorted(d.glob("formulas-*.jsonl")):
            rows = load(path)
            test = path.stem.split("-", 1)[1]
            for part, sel in (("stessi", [r for r in rows if r["index"] < 100]), ("indipendenti", [r for r in rows if r["index"] >= 100])):
                a = formula_analysis(sel)
                rep["unimer"].setdefault(f"{test}/{part}", {})[m] = {
                    k: a[k] for k in ("n", "acc_visible", "exact_visible", "loops", "no_eos", "by", "acc_by_length_and_density",
                                      "isolated_per_100_ref_tokens", "error_ops_by_run_length")}
                rep["unimer"][f"{test}/{part}"][m]["per_sample"] = {p["key"]: p for p in a["per_sample"]}
    for part in ("stessi", "indipendenti"):  # F1c con i suoi input (dense), F1b e controllo con i loro (x4) e con dense
        cmp = {}
        for metric in ("acc_visible", "exact_visible"):
            per = {"f1b": {k: v[metric] for k, v in rep["unimer"].get(f"x4/{part}", {}).get("f1b", {}).get("per_sample", {}).items()},
                   "control": {k: v[metric] for k, v in rep["unimer"].get(f"x4/{part}", {}).get("control", {}).get("per_sample", {}).items()},
                   "f1c": {k: v[metric] for k, v in rep["unimer"].get(f"dense/{part}", {}).get("f1c", {}).get("per_sample", {}).items()}}
            cmp[metric] = compare(per, metric, binary=metric == "exact_visible")
        rep[f"unimer_{part}"] = cmp
    u = rep["unimer"]
    if "dense/stessi" in u and "f1c" in u["dense/stessi"]:
        rep["criteria"]["unimer_acc>=0.70"] = {"f1b x4": u["x4/stessi"]["f1b"]["acc_visible"], "control x4": u["x4/stessi"].get("control", {}).get("acc_visible"),
                                               "f1c dense": u["dense/stessi"]["f1c"]["acc_visible"]}
        rep["criteria"]["unimer_esatte>=30/200"] = {"f1b x4": u["x4/stessi"]["f1b"]["exact_visible"],
                                                    "control x4": u["x4/stessi"].get("control", {}).get("exact_visible"),
                                                    "f1c dense": u["dense/stessi"]["f1c"]["exact_visible"]}
        rep["criteria"]["unimer_cicli<=10/200"] = {"f1b x4": u["x4/stessi"]["f1b"]["loops"], "control x4": u["x4/stessi"].get("control", {}).get("loops"),
                                                   "f1c dense": u["dense/stessi"]["f1c"]["loops"]}

    # scelta forzata: per insieme, gruppo di domande e modo
    acc: dict = defaultdict(dict)
    for m, rows in choice.items():
        for r in rows:
            gid = f"{r['set']}/{group_of(r['type'])}/{r['mode']}"
            acc[gid].setdefault(m, {})[f"{r['key']}|{r['q']}"] = r["correct"]
            acc[f"{r['set']}/tutte/{r['mode']}"].setdefault(m, {})[f"{r['key']}|{r['q']}"] = r["correct"]
    rep["choice"] = {g: compare(v, g) for g, v in sorted(acc.items())}
    # coppie controfattuali: riuscita se sono giuste entrambe le varianti
    pairs = {}
    for m, rows in choice.items():
        by = defaultdict(dict)
        for r in rows:
            if r["set"] == "pairs":
                by[r["mode"]].setdefault(r["pair"], []).append(r["correct"])
        pairs[m] = {mode: {p: all(v) and len(v) == 2 for p, v in d.items()} for mode, d in by.items()}
    rep["pairs"] = {mode: compare({m: v.get(mode, {}) for m, v in pairs.items()}, mode) for mode in ("real", "white", "mismatch")}
    rel = rep["choice"].get("synth/relazioni/real", {})
    rep["criteria"]["stem_relazioni_+10pp(sintetici)"] = rel
    if "stem_real/tutte/real" in rep["choice"]:
        rep["criteria"]["stem_reali"] = rep["choice"]["stem_real/tutte/real"]
    rep["criteria"]["dipendenza_coppie"] = rep["pairs"].get("real")

    # teacher forcing: guadagno nell'ultimo decimo rispetto al primo
    rep["decay"] = {}
    for m, d in dirs.items():
        tok = d / "tokens.json"
        if tok.exists():
            for src, t in token_tables(json.loads(tok.read_text())).items():
                g = [x["gain"] for x in t["deciles"]]
                rep["decay"].setdefault(src, {})[m] = {"first": g[0], "last": g[-1], "ratio": g[-1] / g[0] if g[0] else None}

    # esempi: MAVIS corrette o peggiorate da F1c rispetto a F1b
    ex = {"mavis_corrette": [], "mavis_introdotte": []}
    if "f1c" in mav:
        for k in first80:
            a, b = mav["f1b"].get(k), mav["f1c"].get(k)
            if a and b and a["invented"] != b["invented"]:
                bucket = "mavis_corrette" if a["invented"] else "mavis_introdotte"
                if len(ex[bucket]) < 5:
                    ex[bucket].append({"key": k, "riferimento": a["reference"], "f1b": a["output"], "f1c": b["output"]})
    rep["examples"] = ex
    for sec in rep["unimer"].values():
        for v in sec.values():
            v.pop("per_sample", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"criteria": rep["criteria"], "mavis_80": rep["mavis_80"], "unimer_stessi": rep.get("unimer_stessi"),
                      "pairs": rep["pairs"], "decay": rep["decay"],
                      "choice": {k: v for k, v in rep["choice"].items() if k.endswith("/real") or "tutte" in k}},
                     indent=1, ensure_ascii=False, default=str))
    print(f"\nreport: {args.out}")


if __name__ == "__main__":
    main()
