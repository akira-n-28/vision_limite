#!/usr/bin/env python3
"""Analisi degli errori degli adapter di F1: dove e come sbagliano. Solo libreria standard (job CPU).

Sottocomandi:
- formulas FILE.jsonl...: predizioni di scripts/eval_f1_formulas.py. I confronti si fanno sui token visibili
  (latex.latex_tokens senza spazi, graffe, comandi di font, \\left/\\right, ambienti array e sinonimi come
  \\ge/\\geq), così una notazione diversa per la stessa formula non conta come errore. Dalle predizioni senza EOS
  si toglie la coda periodica (ciclo), contata a parte. Predizione e riferimento si allineano con la distanza di
  edit: sostituzioni (glifo simile come a/\\alpha o l/1, maiuscola/minuscola, stessa classe, classe diversa),
  omissioni (token del riferimento che mancano) e aggiunte, per classe di token, per posizione nella formula e
  per densità (token visibili per token immagine).
- captions DIR...: uscite di scripts/f1_error_gen.py, una cartella per adapter.
  - teacher forcing: loss e top-1 con immagine vera e bianca per tipo di token (numeri, maiuscole cioè etichette,
    forme, colori, LaTeX, parole funzionali, formato, contenuto) e per decimo della risposta, più le confusioni;
  - generazione: cicli e EOS; MAVIS: fatti estratti dal testo (figure con il loro tipo, angoli e lati con il
    valore) confrontati con il riferimento; STEM: lettere dei punti, numeri e concetti geometrici; PixMo e LLaVA:
    parole di contenuto, colori e numeri. Ogni misura anche con l'immagine bianca, che dà il livello di quello che
    il decoder scrive senza vedere nulla.
Ogni sottocomando stampa le tabelle e scrive il JSON completo in --out.

    python scripts/f1_error_report.py formulas --out OUT.json NOME=FILE.jsonl ...
    python scripts/f1_error_report.py captions --out OUT.json NOME=DIR ...
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.facts import SHAPE_WORDS, compare_facts, mavis_facts  # noqa: E402,F401
from limite_vl.latex import INVISIBLE, latex_tokens, tclass, visible_tokens  # noqa: E402,F401

# ---------------------------------------------------------------- formule

# Coppie che si confondono a vista (in qualunque ordine).
SIMILAR = {frozenset(p) for p in [
    ("a", r"\alpha"), ("v", r"\nu"), ("u", r"\mu"), ("w", r"\omega"), ("p", r"\rho"), ("B", r"\beta"),
    ("x", r"\chi"), ("x", r"\times"), ("X", r"\chi"), ("e", r"\epsilon"), ("e", r"\varepsilon"),
    (r"\epsilon", r"\varepsilon"), (r"\epsilon", r"\in"), (r"\varepsilon", r"\in"), (r"\phi", r"\varphi"),
    (r"\phi", r"\Phi"), (r"\theta", r"\vartheta"), (r"\rho", r"\varrho"), (r"\sigma", r"\varsigma"),
    (r"\pi", r"\varpi"), ("n", r"\eta"), ("y", r"\gamma"), ("k", r"\kappa"), ("K", r"\kappa"), ("t", r"\tau"),
    ("i", r"\iota"), ("o", r"\sigma"), ("o", "0"), ("O", "0"), ("O", r"\Theta"), ("l", "1"), ("l", "|"),
    ("1", "|"), ("I", "l"), ("I", "1"), ("I", "|"), ("d", r"\partial"), (r"\partial", r"\delta"),
    ("E", r"\Sigma"), (".", r"\cdot"), (r"\cdot", r"\bullet"), (r"\ldots", r"\cdots"), ("S", "5"), ("Z", "2"),
    ("z", "2"), ("q", "9"), ("g", "9"), ("b", "6"), ("B", "8"), ("A", r"\Lambda"), ("V", r"\nu"),
    ("u", r"\upsilon"), ("v", r"\upsilon"), (r"\psi", r"\varphi"), ("<", r"\langle"), (">", r"\rangle"),
    ("h", r"\hbar"), ("T", r"\top"), ("U", r"\cup"), ("c", "e"), ("n", "h"), ("m", "n"), ("r", "v"), ("l", "t"),
    ("i", "j")]}


def align(pred: list[str], ref: list[str]) -> list[tuple]:
    """Allineamento di edit minimo: (op, token predetto, token del riferimento, posizione nel riferimento)
    con op in eq, sub, del (token del riferimento mancante), ins (token in più nella predizione)."""
    n, m = len(pred), len(ref)
    dp = [list(range(m + 1))]
    for i in range(1, n + 1):
        prev, cur, x = dp[-1], [i] + [0] * m, pred[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != ref[j - 1]))
        dp.append(cur)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (pred[i - 1] != ref[j - 1]):
            ops.append(("eq" if pred[i - 1] == ref[j - 1] else "sub", pred[i - 1], ref[j - 1], j - 1))
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("del", None, ref[j - 1], j - 1))
            j -= 1
        else:
            ops.append(("ins", pred[i - 1], None, j))
            i -= 1
    return ops[::-1]


def accuracy(ops: list[tuple], ref_len: int) -> float:
    return max(0.0, 1 - sum(op[0] != "eq" for op in ops) / max(1, ref_len))


def loop_start(tokens: list, min_cover: int = 20, min_reps: int = 3, max_period: int = 80) -> int | None:
    """Inizio della coda periodica più lunga (almeno min_reps ripetizioni e min_cover token), oppure None."""
    best, n = None, len(tokens)
    for k in range(1, min(max_period, n // min_reps) + 1):
        s = n - k
        while s > 0 and tokens[s - 1] == tokens[s - 1 + k]:
            s -= 1
        if n - s >= max(min_cover, min_reps * k) and (best is None or s < best):
            best = s
    return best


def sub_kind(pred: str, ref: str) -> str:
    if frozenset((pred, ref)) in SIMILAR:
        return "glifo simile"
    cp, cr = tclass(pred), tclass(ref)
    if cp == cr == "lettera" and pred.lower() == ref.lower():
        return "maiuscola/minuscola"
    if cp == cr == "greca" and pred.lower() == ref.lower():
        return "maiuscola/minuscola"
    if cp == cr:
        return f"stessa classe: {cp}"
    return "classe diversa"


def bucket(value: float, edges: tuple) -> str:
    for lo, hi in zip((0,) + edges, edges + (float("inf"),)):
        if lo <= value < hi:
            return f"{lo}-{hi if hi != float('inf') else ''}"
    return "?"


def mean(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def formula_analysis(rows: list[dict]) -> dict:
    per = []
    for r in rows:
        ref = visible_tokens(r["reference"])
        pred_all = visible_tokens(r["prediction"])
        start = None if r["eos"] else loop_start(pred_all)
        pred = pred_all if start is None else pred_all[:start]
        ops = align(pred, ref)
        raw_ref = latex_tokens(r["reference"])
        per.append({"key": r["key"], "subset": r["subset"], "eos": r["eos"], "ref": ref, "pred": pred, "ops": ops,
                    "loop_start": start, "loop_tokens": 0 if start is None else len(pred_all) - start,
                    "acc_raw": r["token_acc"], "acc_visible": accuracy(ops, len(ref)),
                    "acc_visible_untrimmed": accuracy(align(pred_all, ref), len(ref)) if start is not None else None,
                    "exact_visible": pred == ref, "notation_only": pred == ref and latex_tokens(r["prediction"]) != raw_ref,
                    "image_tokens": r.get("image_tokens"),  # assente nelle uscite più vecchie (job 29631)
                    "density": len(ref) / r["image_tokens"] if r.get("image_tokens") else None})
    n_ref = sum(len(p["ref"]) for p in per)
    ops_all = [op for p in per for op in p["ops"]]
    count = Counter(op[0] for op in ops_all)
    subs = [(op[1], op[2]) for op in ops_all if op[0] == "sub"]
    kinds = Counter(sub_kind(a, b) for a, b in subs)
    dels = [op[2] for op in ops_all if op[0] == "del"]
    ins = [op[1] for op in ops_all if op[0] == "ins"]
    deciles = [[0, 0] for _ in range(10)]
    ref_class_err = defaultdict(lambda: [0, 0])
    for p in per:
        err_at = {op[3] for op in p["ops"] if op[0] in ("sub", "del")}
        for j, tok in enumerate(p["ref"]):
            d = min(9, 10 * j // len(p["ref"]))
            deciles[d][0] += 1
            deciles[d][1] += j in err_at
            ref_class_err[tclass(tok)][0] += 1
            ref_class_err[tclass(tok)][1] += j in err_at
    loops = [p for p in per if p["loop_start"] is not None]
    ratio = [p["loop_start"] / max(1, len(p["ref"])) for p in loops]
    # errori isolati (una sola operazione fra due token giusti: simbolo letto male, dettaglio saltato) contro tratti
    # di errori consecutivi (la predizione ha perso il filo e l'allineamento accoppia token non correlati)
    run_ops, isolated = Counter(), defaultdict(list)
    for p in per:
        run = []
        for op in p["ops"] + [("eq", None, None, None)]:
            if op[0] != "eq":
                run.append(op)
                continue
            if run:
                run_ops[bucket(len(run), (2, 4, 10))] += len(run)
                if len(run) == 1:
                    isolated[run[0][0]].append(run[0])
                run = []
    iso_subs = [(op[1], op[2]) for op in isolated["sub"]]
    cross = defaultdict(list)
    for p in per:
        if p["density"] is not None:
            cross[(bucket(len(p["ref"]), (40, 80, 160)), bucket(p["density"], (0.25, 0.5)))].append(p["acc_visible"])
    by = {}
    for name, field, edges in (("densita", "density", (0.25, 0.5, 1.0, 2.0)),
                               ("lunghezza", None, (20, 40, 80, 160))):
        groups = defaultdict(list)
        for p in per:
            value = p[field] if field else len(p["ref"])
            if value is not None:
                groups[bucket(value, edges)].append(p)
        by[name] = {k: {"n": len(v), "acc_visible": mean(x["acc_visible"] for x in v),
                        "loops": sum(x["loop_start"] is not None for x in v)}
                    for k, v in sorted(groups.items(), key=lambda kv: float(kv[0].split("-")[0]))}
    return {
        "n": len(per), "ref_tokens": n_ref,
        "acc_raw": mean(p["acc_raw"] for p in per), "acc_visible": mean(p["acc_visible"] for p in per),
        "exact_visible": sum(p["exact_visible"] for p in per), "notation_only": sum(p["notation_only"] for p in per),
        "no_eos": sum(not p["eos"] for p in per), "loops": len(loops),
        "loop_start_over_ref_len": {"<0.5": sum(x < 0.5 for x in ratio), "0.5-1": sum(0.5 <= x < 1 for x in ratio),
                                    ">=1": sum(x >= 1 for x in ratio)},
        "acc_visible_loops_trimmed": mean(p["acc_visible"] for p in loops),
        "acc_visible_loops_untrimmed": mean(p["acc_visible_untrimmed"] for p in loops),
        "per_100_ref_tokens": {k: 100 * count[k] / n_ref for k in ("sub", "del", "ins")},
        "error_ops_by_run_length": {k: v / max(1, sum(run_ops.values())) for k, v in sorted(
            run_ops.items(), key=lambda kv: float(kv[0].split("-")[0]))},
        "isolated_per_100_ref_tokens": {k: 100 * len(isolated[k]) / n_ref for k in ("sub", "del", "ins")},
        "isolated_sub_kinds": dict(Counter(sub_kind(a, b) for a, b in iso_subs).most_common()),
        "top_isolated_subs": [[f"{b} -> {a}", n] for (a, b), n in Counter(iso_subs).most_common(30)],
        "isolated_del_by_class": dict(Counter(tclass(op[2]) for op in isolated["del"]).most_common()),
        "top_isolated_dels": Counter(op[2] for op in isolated["del"]).most_common(15),
        "top_isolated_ins": Counter(op[1] for op in isolated["ins"]).most_common(15),
        "acc_by_length_and_density": {f"{a} | {b}": f"n {len(v)} acc {mean(v):.2f}" for (a, b), v in sorted(
            cross.items(), key=lambda kv: (float(kv[0][0].split("-")[0]), float(kv[0][1].split("-")[0])))},
        "sub_kinds": dict(kinds.most_common()),
        "top_subs": [[f"{b} -> {a}", n] for (a, b), n in Counter(subs).most_common(40)],
        "del_by_class": dict(Counter(tclass(t) for t in dels).most_common()),
        "top_dels": Counter(dels).most_common(25),
        "ins_by_class": dict(Counter(tclass(t) for t in ins).most_common()),
        "top_ins": Counter(ins).most_common(25),
        "error_rate_by_ref_class": {k: {"tokens": v[0], "error_rate": v[1] / v[0]}
                                    for k, v in sorted(ref_class_err.items(), key=lambda kv: -kv[1][0])},
        "error_rate_by_decile": [d[1] / max(1, d[0]) for d in deciles],
        "by": by,
        "per_sample": [{k: p[k] for k in ("key", "subset", "eos", "loop_start", "loop_tokens", "acc_raw",
                                          "acc_visible", "exact_visible", "image_tokens")} | {"ref_len": len(p["ref"])}
                       for p in per],
    }


def print_formulas(name: str, a: dict) -> None:
    print(f"\n=== formule {name}: {a['n']} campioni, {a['ref_tokens']} token visibili nei riferimenti")
    print(f"accuratezza sui token: come scritti {a['acc_raw']:.3f}, visibili {a['acc_visible']:.3f}; esatte a vista "
          f"{a['exact_visible']} (di cui solo notazione diversa {a['notation_only']})")
    print(f"senza EOS {a['no_eos']}, cicli {a['loops']} (inizio rispetto alla lunghezza del riferimento: "
          f"{a['loop_start_over_ref_len']}); accuratezza dei cicli prima della coda {a['acc_visible_loops_trimmed']}, "
          f"con la coda {a['acc_visible_loops_untrimmed']}")
    print("operazioni per 100 token del riferimento (cicli esclusi):",
          {k: round(v, 1) for k, v in a["per_100_ref_tokens"].items()})
    print("operazioni di errore per lunghezza del tratto di errori consecutivi:",
          {k: round(v, 2) for k, v in a["error_ops_by_run_length"].items()})
    print("errori isolati per 100 token del riferimento:", {k: round(v, 1) for k, v in a["isolated_per_100_ref_tokens"].items()})
    print("sostituzioni isolate per tipo:", a["isolated_sub_kinds"])
    print("sostituzioni isolate più frequenti:", a["top_isolated_subs"][:25])
    print("omissioni isolate per classe:", a["isolated_del_by_class"], "più frequenti:", a["top_isolated_dels"][:12])
    print("aggiunte isolate più frequenti:", a["top_isolated_ins"][:12])
    print("accuratezza per lunghezza | densità:", a["acc_by_length_and_density"])
    print("sostituzioni per tipo:", a["sub_kinds"])
    print("sostituzioni più frequenti (riferimento -> predizione):", a["top_subs"][:25])
    print("omissioni per classe:", a["del_by_class"])
    print("omissioni più frequenti:", a["top_dels"][:15])
    print("aggiunte per classe:", a["ins_by_class"])
    print("aggiunte più frequenti:", a["top_ins"][:15])
    print("errori per classe del token del riferimento:",
          {k: f"{v['error_rate']:.2f} ({v['tokens']})" for k, v in a["error_rate_by_ref_class"].items()})
    print("errori per decimo della formula:", [round(x, 2) for x in a["error_rate_by_decile"]])
    for k, v in a["by"].items():
        print(f"per {k}:", {b: f"n {x['n']} acc {x['acc_visible']:.2f} cicli {x['loops']}" for b, x in v.items()})


def compare_formulas(a: dict, b: dict) -> dict:
    pa = {p["key"]: p for p in a["per_sample"]}
    pairs = [(pa[p["key"]], p) for p in b["per_sample"] if p["key"] in pa]
    diff = [q["acc_visible"] - p["acc_visible"] for p, q in pairs]
    return {"pairs": len(pairs), "mean_diff": mean(diff), "better": sum(d > 0.05 for d in diff),
            "worse": sum(d < -0.05 for d in diff), "loops": [sum(p["loop_start"] is not None for p, _ in pairs),
                                                              sum(q["loop_start"] is not None for _, q in pairs)]}


# ---------------------------------------------------------------- didascalie

STOPWORDS = set("""a an the of in on at to for from by with without and or but nor so as is are was were be been being
am it its it's this that these those there here which who whom whose what when where while than then also very
into onto over under above below between among through during before after up down out off about against along
across behind beyond near next each every all both either neither any some no not only own same such too more most
other another can could may might must shall should will would do does did has have had having he she they them
their his her we our you your i me my one's s t just""".split())
GENERIC = set("""image picture photo photograph shows show showing depicts depicting depicted visible background
foreground detailed detail details scene appears appear appearing features featuring feature left right top bottom
side sides center centre middle overall front view seen area part slightly various several displayed display
displaying shown include includes including located positioned position set setting likely possibly""".split())
GEOMETRY = set(SHAPE_WORDS) | set("""polygon pentagon hexagon quadrilateral line lines segment segments angle angles arc
chord diameter radius tangent parallel perpendicular midpoint bisector diagonal vertex vertices intersection cube
cylinder cone sphere prism pyramid ellipse parabola hyperbola graph axis axes coordinate grid bar chart table curve
function arrow point points""".split())
COLORS = set("""red orange yellow green blue purple pink brown black white gray grey silver gold golden beige tan teal
turquoise navy maroon violet cyan magenta cream ivory olive""".split())
NUMBER_WORDS = {w: str(i) for i, w in enumerate("""zero one two three four five six seven eight nine ten eleven twelve
thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty""".split())}
TOKEN_CATEGORIES = ("numero", "maiuscole", "forma", "colore", "latex", "formato", "punteggiatura", "funzionale",
                    "contenuto", "spazi", "eos")


def token_category(tok: str) -> str:
    t = tok.strip()
    if t.startswith("<|") and t.endswith("|>"):
        return "eos"
    if not t:
        return "spazi"
    if any(c.isdigit() for c in t):
        return "numero"
    if t.startswith("\\") or t in ("^", "_", "{", "}", "$"):
        return "latex"
    low = t.lower()
    if low in SHAPE_WORDS or low in ("triangles", "squares", "rectangles", "circles", "sectors"):
        return "forma"
    if low in COLORS:
        return "colore"
    if re.fullmatch(r"[A-Z]{1,4}", t):
        return "maiuscole"
    if re.fullmatch(r"[*#`|>\-]+", t):
        return "formato"
    if re.fullmatch(r"[^\w\s]+", t):
        return "punteggiatura"
    if low in STOPWORDS:
        return "funzionale"
    return "contenuto"


def token_tables(tokens: dict) -> dict:
    out = {}
    for source, data in tokens.items():
        cats = defaultdict(lambda: [0, 0.0, 0.0, 0, 0])
        for tok, v in data["by_token"].items():
            acc = cats[token_category(tok)]
            for i in range(5):
                acc[i] += v[i]
        n = sum(v[0] for v in cats.values())
        loss = sum(v[1] for v in cats.values())
        table = {}
        for cat in TOKEN_CATEGORIES:
            if cat in cats:
                c, lr, lw, okr, okw = cats[cat]
                table[cat] = {"token_share": c / n, "loss_share": lr / loss, "loss_real": lr / c, "loss_white": lw / c,
                              "gain": (lw - lr) / c, "top1_real": okr / c, "top1_white": okw / c, "tokens": c}
        deciles = [{"loss_real": d[1] / max(1, d[0]), "gain": (d[2] - d[1]) / max(1, d[0]),
                    "top1_real": d[3] / max(1, d[0]), "top1_white": d[4] / max(1, d[0])} for d in data["by_decile"]]
        conf = defaultdict(list)
        for target, pred, count in data["confusions"]:
            cat = token_category(target)
            if cat in ("numero", "maiuscole", "forma", "colore", "latex", "contenuto") and len(conf[cat]) < 15:
                conf[cat].append([f"{target.strip()} -> {pred.strip()}", count])
        out[source] = {"categories": table, "deciles": deciles, "confusions": dict(conf)}
    return out


WORD = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z\d]")


def words(text: str) -> list[str]:
    return WORD.findall(text)


def is_loop(text: str, eos: bool) -> bool:
    return not eos and loop_start(words(text), min_cover=12) is not None


def distinct4(text: str) -> float | None:
    w = words(text)
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / len(grams) if grams else None


def content_words(text: str) -> set[str]:
    return {w for w in (x.lower() for x in re.findall(r"[A-Za-z]+", text))
            if len(w) >= 3 and w not in STOPWORDS and w not in GENERIC}


def numbers(text: str) -> set[str]:
    nums = set(re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", text))
    return nums | {NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in NUMBER_WORDS}


BACKTICK = re.compile(r"`([^`]*)`")
UPPER_RUN = re.compile(r"(?<![A-Za-z])([A-Z]{1,4})(?![A-Za-z])")
POINT_WORD = re.compile(r"\b(?:[Pp]oints?|[Vv]ertex|[Vv]ertices|[Ll]ine|[Ss]egment|[Ss]ide|[Aa]ngle|[Tt]riangle|"
                        r"[Cc]ircle|[Cc]enter|[Qq]uadrilateral|[Ss]quare|[Rr]ectangle)\s+([A-Z]{1,4})\b")
HEADER = re.compile(r"^\s*#+\s*Image\s*\d+\s*:\s*(.+)$", re.M)


def stem_items(text: str) -> dict:
    header = HEADER.search(text)
    body = HEADER.sub("", text)
    labels = [m for span in BACKTICK.findall(body) for m in UPPER_RUN.findall(span)] + POINT_WORD.findall(body)
    return {"points": {c for lab in labels for c in lab}, "numbers": numbers(body),
            "geometry": {w for w in (x.lower() for x in re.findall(r"[A-Za-z]+", body)) if w in GEOMETRY},
            "type": header.group(1).strip().lower() if header else None}


def pr(ref: set, gen: set) -> tuple[int, int, int]:
    return len(ref & gen), len(gen), len(ref)


def ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def generation_analysis(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["source"], r["mode"])].append(r)
    out = {}
    for (source, mode), items in sorted(by.items()):
        a = {"n": len(items), "eos": mean(r["eos"] for r in items), "loops": mean(is_loop(r["output"], r["eos"]) for r in items),
             "new_tokens": mean(r["new_tokens"] for r in items),
             "distinct4": mean(d for d in (distinct4(r["output"]) for r in items) if d is not None),
             "distinct4_reference": mean(d for d in (distinct4(r["reference"]) for r in items) if d is not None)}
        if source == "mavis":
            tot = defaultdict(Counter)
            labels, hits, depth = Counter(), defaultdict(list), defaultdict(list)
            for r in items:
                c = compare_facts(mavis_facts(r["reference"]), mavis_facts(r["output"]))
                for kind in ("shape", "angle", "side"):
                    tot[kind].update(c[kind])
                for i, h in enumerate(c["shape_hits"]):
                    hits[min(i, 3)].append(h)
                bad = sum(c[k]["wrong"] + c[k]["extra"] for k in ("shape", "angle", "side"))
                miss = sum(c[k]["missing"] for k in ("shape", "angle", "side"))
                if is_loop(r["output"], r["eos"]) or not r["eos"]:
                    label = "ciclo o troncata"
                elif bad:
                    label = "plausibile ma sbagliata"
                elif miss:
                    label = "giusta ma incompleta"
                else:
                    label = "giusta"
                labels[label] += 1
                d = re.match(r"d(\d)-", r["key"])
                depth[d.group(1) if d else "?"].append(label == "giusta")
                r["_mavis"] = {"label": label, "facts": c}
            a["facts"] = {k: dict(v) | {"precision": ratio(v["ok"], v["gen"]), "recall": ratio(v["ok"], v["ref"])}
                          for k, v in tot.items()}
            a["shape_count"] = {"more": sum(r["_mavis"]["facts"]["shape"]["gen"] > r["_mavis"]["facts"]["shape"]["ref"] for r in items),
                                "fewer": sum(r["_mavis"]["facts"]["shape"]["gen"] < r["_mavis"]["facts"]["shape"]["ref"] for r in items)}
            a["recall_by_shape_index"] = {("4+" if i == 3 else str(i + 1)): mean(v) for i, v in sorted(hits.items())}
            a["labels"] = dict(labels)
            a["correct_by_depth"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(depth.items())}
        elif source == "stem":
            tot = defaultdict(lambda: [0, 0, 0])
            same_type = []
            for r in items:
                ref, gen = stem_items(r["reference"]), stem_items(r["output"])
                for k in ("points", "numbers", "geometry"):
                    for i, x in enumerate(pr(ref[k], gen[k])):
                        tot[k][i] += x
                same_type.append(ref["type"] == gen["type"])
            a["items"] = {k: {"precision": ratio(v[0], v[1]), "recall": ratio(v[0], v[2]), "gen": v[1], "ref": v[2]}
                          for k, v in tot.items()}
            a["same_type"] = mean(same_type)
        else:
            tot = defaultdict(lambda: [0, 0, 0])
            for r in items:
                for k, f in (("content", content_words), ("colors", lambda t: content_words(t) & COLORS),
                             ("numbers", numbers)):
                    for i, x in enumerate(pr(f(r["reference"]), f(r["output"]))):
                        tot[k][i] += x
            a["items"] = {k: {"precision": ratio(v[0], v[1]), "recall": ratio(v[0], v[2]), "gen": v[1], "ref": v[2]}
                          for k, v in tot.items()}
        out[f"{source}/{mode}"] = a
    # quanto la generazione con l'immagine vera somiglia a quella con l'immagine bianca dello stesso campione
    sim = defaultdict(list)
    index = {(r["source"], r["key"], r["mode"]): r for r in rows}
    for (source, key, mode), r in index.items():
        w = index.get((source, key, "white"))
        if mode == "real" and w is not None:
            a, b = content_words(r["output"]), content_words(w["output"])
            sim[source].append(len(a & b) / max(1, len(a | b)))
    for source, v in sim.items():
        out.setdefault("real_vs_white_jaccard", {})[source] = mean(v)
    return out


def print_captions(name: str, tables: dict, gen: dict) -> None:
    print(f"\n=== didascalie {name}")
    for source, t in tables.items():
        print(f"-- {source}: per tipo di token (quota token, quota loss, loss vera/bianca, guadagno, top-1 vera/bianca)")
        for cat, v in t["categories"].items():
            print(f"   {cat:13s} {100 * v['token_share']:5.1f}% {100 * v['loss_share']:5.1f}%  {v['loss_real']:.2f}/"
                  f"{v['loss_white']:.2f}  {v['gain']:+.2f}  {v['top1_real']:.2f}/{v['top1_white']:.2f}")
        print("   per decimo: loss", [round(d["loss_real"], 2) for d in t["deciles"]],
              "guadagno", [round(d["gain"], 2) for d in t["deciles"]])
        for cat, items in t["confusions"].items():
            print(f"   confusioni {cat}: {items[:10]}")
    for key, a in gen.items():
        if key == "real_vs_white_jaccard":
            print("somiglianza fra generazione con immagine vera e bianca (Jaccard delle parole di contenuto):",
                  {k: round(v, 2) for k, v in a.items()})
            continue
        line = (f"-- {key}: n {a['n']}, EOS {a['eos']:.2f}, cicli {a['loops']:.2f}, token {a['new_tokens']:.0f}, "
                f"distinct-4 {a['distinct4']} (riferimenti {a['distinct4_reference']})")
        print(line)
        for k in ("facts", "items"):
            if k in a:
                for kind, v in a[k].items():
                    print(f"   {kind}: " + ", ".join(f"{x} {round(y, 3) if isinstance(y, float) else y}"
                                                   for x, y in v.items()))
        for k in ("shape_count", "recall_by_shape_index", "labels", "correct_by_depth", "same_type"):
            if k in a:
                print(f"   {k}: {a[k]}")


def parse_named(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        name, sep, path = item.partition("=")
        out.append((name, Path(path)) if sep else (Path(item).stem, Path(item)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["formulas", "captions"])
    ap.add_argument("inputs", nargs="+", help="NOME=FILE.jsonl (formulas) o NOME=DIR (captions)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    report = {}
    if args.command == "formulas":
        for name, path in parse_named(args.inputs):
            rows = [json.loads(line) for line in path.open()]
            report[name] = formula_analysis(rows)
            print_formulas(name, report[name])
        names = list(report)
        report["compare"] = {f"{a} -> {b}": compare_formulas(report[a], report[b])
                             for i, a in enumerate(names) for b in names[i + 1:]}
        print("\nconfronti per campione (accuratezza visibile; migliori/peggiori di oltre 5 punti):", report["compare"])
    else:
        for name, path in parse_named(args.inputs):
            tables = token_tables(json.loads((path / "tokens.json").read_text()))
            rows = [json.loads(line) for line in (path / "gen.jsonl").open()]
            gen = generation_analysis(rows)
            report[name] = {"tokens": tables, "generation": gen,
                            "mavis_samples": [{"key": r["key"], "mode": r["mode"], **r["_mavis"]}
                                              for r in rows if "_mavis" in r]}
            print_captions(name, tables, gen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\nreport: {args.out}")


if __name__ == "__main__":
    main()
