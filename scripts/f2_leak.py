#!/usr/bin/env python3
"""Fughe dal ponte testuale nelle soluzioni F2 di Violetto: quanto le soluzioni scelte copiano la descrizione.

Da eseguire in un job CPU (slurm/f2_leak.sbatch). Violetto ha scritto le soluzioni leggendo la descrizione di Qwen3-VL
al posto dell'immagine; in training il modello vede l'immagine, quindi una frase copiata da parti della descrizione
che nell'immagine non ci sono diventa un'affermazione sulla figura inventata. Per ogni soluzione di selected.jsonl
(scripts/f2_select.py) si prendono domanda e descrizione dai parquet di MMFineReason e si misurano:
- copia_nascosta: un n-gramma di parole (--n, default 8) della soluzione presente nelle sezioni non visibili della
  descrizione ("Implicit and Inferred Properties", "Identified Ambiguities") e assente dalla domanda;
- copia_visibile: lo stesso con le sezioni sul contenuto visibile (riassunto, componenti, relazioni);
- risposta_nascosta: la risposta di riferimento compare fra le parole delle sezioni non visibili e non nella domanda
  (la descrizione dà la risposta); "stretta" solo per risposte non banali (almeno 3 caratteri, o con / . √ π);
- tracce: frasi del ponte rimaste ("the figure says", "shown but not given", nomi delle sezioni, "description");
- copertura: frazione delle parole della soluzione dentro n-grammi copiati dalla descrizione e non dalla domanda.
Scrive OUT/stats.json (tassi totali e per sorgente), OUT/flags.jsonl (soluzione, motivi, copertura) e OUT/examples.md.

    srun $RUN_PYTHON -u -B scripts/f2_leak.py --selected outputs/f2_select/<run>/selected.jsonl --parquet-glob GLOB --out DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.mmfine import mmfine_key, split_of  # noqa: E402

HIDDEN = {"implicit and inferred properties", "identified ambiguities", "inferred properties", "question context",
          "problem context", "answer inference", "correct inference"}
_SECTION = re.compile(r"^\s*\*\*\s*([^*:]{2,60}?)\s*:?\s*\*\*\s*:?\s*")
_TITLE = re.compile(r"^\s*\**\s*#{1,6}\s*Image\b", re.IGNORECASE)
WORD = re.compile(r"[a-z0-9]+|[√π]")
TRACES = {
    "the_figure_says": re.compile(r"\bthe (?:figure|image|diagram|picture) (?:says|states|tells us|reads|notes|mentions|lists)\b", re.I),
    "not_given": re.compile(r"\b(?:shown|described|referenced|mentioned) but not (?:given|provided|shown)\b|\bnot (?:given|provided) (?:to us|here)\b", re.I),
    "section_names": re.compile(r"\b(?:scene summary|explicit component|interactions and relationships|implicit and inferred|identified ambiguities)\b", re.I),
    "description_word": re.compile(r"\b(?:description|caption)\b", re.I),
}


def sections(caption: str) -> tuple[str, str]:
    """(testo visibile, testo non visibile) della descrizione, divisa per titoli in grassetto."""
    visible, hidden, current = [], [], "scene summary"
    for line in caption.splitlines():
        if _TITLE.match(line):
            continue
        m = _SECTION.match(line)
        if m and line.strip().startswith("**"):
            current = m.group(1).strip().lower()
            line = line[m.end():]
        (hidden if current in HIDDEN else visible).append(line)
    return "\n".join(visible), "\n".join(hidden)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(ws: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def norm_answer(ans: str) -> str:
    ans = re.sub(r"\\boxed\{(.*)\}", r"\1", ans.strip()).strip("$ .")
    return " ".join(words(ans))


def analyze(solution: str, question: str, caption: str, answer: str, n: int) -> dict:
    vis, hid = sections(caption)
    sw, qn = words(solution), ngrams(words(question), n)
    hn, vn = ngrams(words(hid), n) - qn, ngrams(words(vis), n) - qn
    copied_h = [g for g in ngrams(sw, n) if g in hn]
    copied_v = [g for g in ngrams(sw, n) if g in vn and g not in hn]
    covered, covered_h = [False] * len(sw), [False] * len(sw)
    for i in range(len(sw) - n + 1):
        g = tuple(sw[i:i + n])
        if g in hn or g in vn:
            for j in range(i, i + n):
                covered[j] = True
                covered_h[j] = covered_h[j] or g in hn
    a = norm_answer(answer)
    hid_text, q_text = " " + " ".join(words(hid)) + " ", " " + " ".join(words(question)) + " "
    in_hidden = bool(a) and f" {a} " in hid_text and f" {a} " not in q_text
    strict = in_hidden and (len(a.replace(" ", "")) >= 3 or bool(re.search(r"[/.√π]", answer)))
    return {"copia_nascosta": bool(copied_h), "copia_visibile": bool(copied_v), "risposta_nascosta": in_hidden,
            "risposta_nascosta_stretta": strict, "copertura": sum(covered) / max(1, len(sw)),
            "nascosti": len(copied_h), "copertura_nascosta": sum(covered_h) / max(1, len(sw)),
            "tracce": [k for k, p in TRACES.items() if p.search(solution)],
            "esempio_nascosto": " ".join(copied_h[0]) if copied_h else None,
            "esempio_visibile": " ".join(copied_v[0]) if copied_v else None, "hidden_chars": len(hid)}


def load_texts(keys: set[str], paths: list[str]) -> dict[str, dict]:
    """Domanda e descrizione delle chiavi date, leggendo dai parquet solo le righe che servono."""
    import pyarrow.parquet as pq

    out = {}
    for path in paths:
        split = split_of(Path(path))
        meta = pq.read_table(path, columns=["id", "source"]).to_pydict()
        rows = [i for i, (id_, src) in enumerate(zip(meta["id"], meta["source"])) if mmfine_key(split, src, id_) in keys]
        if not rows:
            continue
        table = pq.read_table(path, columns=["id", "source", "question", "caption"]).take(rows).to_pylist()
        for r in table:
            out[mmfine_key(split, r["source"], r["id"])] = {"question": (r["question"] or "").replace("<image>", "").strip(),
                                                           "caption": (r["caption"] or "").strip()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", required=True, type=Path)
    ap.add_argument("--parquet-glob", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=8, help="lunghezza degli n-grammi di parole")
    args = ap.parse_args()
    t0 = time.time()
    sols = [json.loads(line) for line in args.selected.open()]
    paths = sorted(glob.glob(args.parquet_glob))
    texts = load_texts({s["key"] for s in sols}, paths)
    print(f"{len(sols)} soluzioni di {len({s['key'] for s in sols})} problemi; testi trovati {len(texts)} ({time.time() - t0:.0f} s)",
          flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    tot, by_src, examples = Counter(), defaultdict(Counter), defaultdict(list)
    cover, hidden_hits, hidden_cov = [], [], []
    with open(args.out / "flags.jsonl", "w") as fl:
        for i, s in enumerate(sols):
            t = texts.get(s["key"])
            if t is None:
                tot["senza_testo"] += 1
                continue
            r = analyze(s["solution"], t["question"], t["caption"], s["answer"], args.n)
            cover.append(r["copertura"])
            hidden_hits.append(r["nascosti"])
            hidden_cov.append(r["copertura_nascosta"])
            reasons = [k for k in ("copia_nascosta", "copia_visibile", "risposta_nascosta", "risposta_nascosta_stretta") if r[k]]
            reasons += [f"traccia:{k}" for k in r["tracce"]]
            tot["soluzioni"] += 1
            by_src[s.get("source_dataset")]["soluzioni"] += 1
            for k in reasons:
                tot[k] += 1
                by_src[s.get("source_dataset")][k] += 1
                if len(examples[k]) < 8:
                    examples[k].append({"key": s["key"], "esempio": r["esempio_nascosto"] if k == "copia_nascosta"
                                        else r["esempio_visibile"] if k == "copia_visibile" else None,
                                        "risposta": s["answer"], "inizio": s["solution"][:300]})
            if r["copia_nascosta"] or r["risposta_nascosta_stretta"] or r["tracce"]:
                tot["almeno_una_fuga"] += 1
            fl.write(json.dumps({"index": i, "key": s["key"], "motivi": reasons, "copertura": round(r["copertura"], 4),
                                 "nascosti": r["nascosti"], "copertura_nascosta": round(r["copertura_nascosta"], 4)}) + "\n")
    n = max(1, tot["soluzioni"])
    cover.sort()
    stats = {"selected": str(args.selected), "n_gram": args.n, "counts": dict(tot),
             "rates": {k: round(v / n, 4) for k, v in tot.items() if k != "soluzioni"},
             "copertura": {"media": round(sum(cover) / max(1, len(cover)), 4),
                           "mediana": cover[len(cover) // 2] if cover else None,
                           "p90": cover[int(0.9 * len(cover))] if cover else None},
             "soglie_nascoste": {f">= {k} n-grammi": round(sum(h >= k for h in hidden_hits) / n, 4) for k in (1, 3, 5, 10, 20)}
             | {f"copertura >= {c:.0%}": round(sum(x >= c for x in hidden_cov) / n, 4) for c in (0.01, 0.02, 0.05, 0.10)},
             "per_sorgente": {src: {"soluzioni": c["soluzioni"], **{k: round(c[k] / max(1, c["soluzioni"]), 4) for k in c if k != "soluzioni"}}
                              for src, c in sorted(by_src.items(), key=lambda kv: -kv[1]["soluzioni"])},
             "seconds": round(time.time() - t0, 1)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    md = ["# Fughe dal ponte: esempi\n"]
    for k, items in examples.items():
        md.append(f"\n## {k}\n")
        for e in items:
            md.append(f"- `{e['key']}` risposta {e['risposta']!r} | n-gramma: {e['esempio']} | {e['inizio'][:200]!r}")
    (args.out / "examples.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: stats[k] for k in ("counts", "rates", "copertura", "soglie_nascoste")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
