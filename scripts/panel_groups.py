#!/usr/bin/env python3
"""Gruppi e ruoli del pannello di valutazione (docs/PANNELLO_VALUTAZIONE.md §3), dall'indice di scripts/panel_index.py.
Criteri, seme e assegnazione sono fissati qui, prima di qualsiasi valutazione del modello. Solo in un job SLURM CPU.

Legami **confermati** (fondono i gruppi):
- pixel identici: stesso `pixel_sha256` (dimensioni e valori RGB normalizzati) fra due elementi qualsiasi;
- provenienza o stesso problema, dichiarati nei dati: MathVerse `problem_index`, We-Math `ID`, DynaMath `id` fra le
  varianti, MathVision `id` fra testmini e test, DocVQA `docId`, ChartQA `imgname` nello stesso split, MUIRBench
  `counterpart_idx`, MMStar `meta_info` verso MathVista (stesso nome di file d'immagine), UniMER-Test con la stessa
  formula, benchmark di solo testo con lo stesso testo normalizzato (almeno 8 parole).
**Candidati** (non fondono nulla): pHash entro --radius bit fra immagini di gruppi diversi, e testi uguali (almeno 8
parole) fra elementi visivi di gruppi diversi. **Irrisolti**: candidati che collegano un elemento di sviluppo a uno di
test; l'elemento di test resta nel test ma è contato a parte finché il legame non si conferma o si esclude.

Ruoli fissi (RULES) per benchmark e split; `casuale` vuol dire estratto per gruppo con seme e quota: ogni gruppo va in
sviluppo se il suo rango (manifest.split_rank) è sotto la quota, altrimenti nel test. Un gruppo con un membro di
sviluppo (o già consultato) va tutto in sviluppo; altrimenti, con un membro di test, tutto nel test. I membri
`solo_decontaminazione` (senza risposte, fuori lingua o non verificabili) e `diagnostica` restano tali, ma il loro
gruppo segue la regola. Tutti i ruoli sono esclusi dal training.

Uscite nella cartella nuova --out: elements.jsonl.gz (un elemento per riga: benchmark, file, split, item, gruppo,
ruolo fisso, ruolo finale, irrisolto), links.jsonl.gz (legami confermati, per tipo), candidates.jsonl.gz,
unresolved.jsonl.gz, summary.json (conteggi, regole, seme, sha256 dell'indice, identità del codice).

    srun $RUN_PYTHON -u -B scripts/panel_groups.py --index FILE --unimer-shards DIR --out DIR [--seed 0] [--radius 4]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.decontam import HashIndex, normalize_words  # noqa: E402
from limite_vl.manifest import Digests, split_rank  # noqa: E402
from limite_vl.provenance import code_identity, sha256_file, write_new_json  # noqa: E402

CONSULTED_UNIMER = 200  # i primi 200 di spe e di cpe: criterio di F1 e valutazioni F1c (job 29448-29761)
FRACTIONS = {"ocrbench": 0.5, "docvqa": 0.5, "muirbench": 0.5, "mmstar": 0.5, "unimer_test": 0.05}
TEXT_ONLY_DEV = {"gsm8k", "ifeval", "aime24", "aime25", "aime26", "hmmt25", "hmmt26"}
TEXT_ONLY_TEST = {"math500", "beyondaime", "apex"}
MIN_WORDS = 8
MAX_NEIGHBORS = 200


def split_of(benchmark: str, file: str) -> str:
    stem = PurePosixPath(file).name
    for name in ("testmini", "test", "validation", "val", "train"):
        if stem.startswith(name):
            return name
    return PurePosixPath(file).parent.name if benchmark == "olympiadbench" else PurePosixPath(stem).stem


def fixed_role(e: dict, consulted: set[str]) -> str:
    """Ruolo fisso dell'elemento: sviluppo, test, casuale, diagnostica o solo_decontaminazione."""
    b, s, h = e["benchmark"], e["split"], e["hints"]
    if b == "mathvista":
        return "sviluppo" if s == "testmini" and h.get("language") == "english" else "solo_decontaminazione"
    if b in ("mathverse", "wemath", "geoperception"):
        return "sviluppo"
    if b == "mathvision":
        return "sviluppo" if s == "testmini" else "test"
    if b in ("dynamath", "visonlyqa_real", "visonlyqa_synth", "mvmath"):
        return "test"
    if b == "olympiadbench":
        return "test" if h.get("subset") in ("OE_MM_maths_en_COMP", "OE_MM_physics_en_COMP", "OE_TO_maths_en_COMP") \
            else "solo_decontaminazione"
    if b == "chartqa":
        return "sviluppo" if s == "val" else "test"
    if b == "unimer_test":
        if h.get("subset") == "hwe":
            return "diagnostica"
        return "sviluppo" if e["item"].split(":", 1)[-1] in consulted else "casuale"
    if b in FRACTIONS:
        return "casuale"
    if b in TEXT_ONLY_DEV:
        return "sviluppo"
    if b in TEXT_ONLY_TEST:
        return "test"
    raise ValueError(f"benchmark senza regola: {b}")


class Union:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, k):
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


def write_gz(path: Path, rows) -> dict:
    data = gzip.compress("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode(),
                         mtime=0)
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "sha256": sha256_file(path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--unimer-shards", required=True, type=Path, help="shard di test di UniMER (chiavi consultate)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=4)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("panel_groups.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: un'assegnazione si scrive una volta sola")

    # ---- chiavi di UniMER-Test già consultate: le prime 200 di spe e di cpe nell'ordine degli shard
    consulted: set[str] = set()
    for sub in ("spe", "cpe"):
        for n, (key, _) in enumerate(iter_shard(args.unimer_shards / f"unimer-test-{sub}.tar")):
            if n >= CONSULTED_UNIMER:
                break
            consulted.add(key)

    # ---- elementi
    index = json.loads(args.index.read_text(encoding="utf-8"))["items"]
    elements: dict[str, dict] = {}
    for it in index:
        key = f"{it['benchmark']}/{it['item']}"
        e = elements.setdefault(key, {"key": key, "benchmark": it["benchmark"], "file": it["file"],
                                      "split": split_of(it["benchmark"], it["file"]), "item": it["item"],
                                      "hints": it["hints"], "text": it["text"], "images": []})
        if it["pixel_sha256"]:
            e["images"].append((it["pixel_sha256"], int(it["phash"], 16)))
    for e in elements.values():
        e["fixed"] = fixed_role(e, consulted)
    keys = sorted(elements)
    uf = Union(keys)
    link_counts, link_rows = Counter(), []

    def link(a: str, b: str, kind: str) -> None:
        if a != b and uf.union(a, b):
            link_counts[kind] += 1
            link_rows.append({"tipo": kind, "a": a, "b": b})

    def link_by(kind: str, select, value) -> None:
        first: dict = {}
        for k in keys:
            e = elements[k]
            if not select(e):
                continue
            v = value(e)
            if v in (None, ""):
                continue
            if v in first:
                link(first[v], k, kind)
            else:
                first[v] = k

    # ---- legami confermati
    by_pixels: dict[str, str] = {}
    for k in keys:
        for pixels, _ in elements[k]["images"]:
            if pixels in by_pixels:
                link(by_pixels[pixels], k, "pixel_identici")
            else:
                by_pixels[pixels] = k
    link_by("mathverse_problem_index", lambda e: e["benchmark"] == "mathverse", lambda e: e["hints"].get("problem_index"))
    link_by("wemath_ID", lambda e: e["benchmark"] == "wemath", lambda e: e["hints"].get("ID"))
    link_by("dynamath_varianti", lambda e: e["benchmark"] == "dynamath", lambda e: e["hints"].get("seed"))
    link_by("mathvision_testmini_test", lambda e: e["benchmark"] == "mathvision", lambda e: e["hints"].get("id"))
    link_by("docvqa_documento", lambda e: e["benchmark"] == "docvqa", lambda e: e["hints"].get("docId"))
    link_by("chartqa_grafico", lambda e: e["benchmark"] == "chartqa",
            lambda e: f"{e['split']}/{e['hints'].get('imgname')}" if e["hints"].get("imgname") else None)
    link_by("unimer_stessa_formula", lambda e: e["benchmark"] == "unimer_test",
            lambda e: (e["hints"].get("formula") or "").strip() or None)
    link_by("testo_identico_solo_testo", lambda e: not e["images"] and (e["benchmark"] in TEXT_ONLY_DEV | TEXT_ONLY_TEST
                                                                          or e["benchmark"] == "olympiadbench"),
            lambda e: " ".join(w) if len(w := normalize_words(e["text"])) >= MIN_WORDS else None)
    muir = {e["hints"].get("id"): k for k, e in elements.items() if e["benchmark"] == "muirbench"}
    for k, e in elements.items():
        if e["benchmark"] == "muirbench" and e["hints"].get("counterpart_idx") in muir:
            link(k, muir[e["hints"]["counterpart_idx"]], "muirbench_controparte")
    vista = {PurePosixPath(str(e["hints"].get("image"))).name: k for k, e in elements.items()
             if e["benchmark"] == "mathvista" and e["hints"].get("image")}
    for k, e in elements.items():
        if e["benchmark"] == "mmstar" and "mathvista" in str(e["hints"].get("source") or "").lower():
            target = vista.get(PurePosixPath(str(e["hints"].get("image_path"))).name)
            if target:
                link(k, target, "mmstar_provenienza_mathvista")

    # ---- gruppi e ruoli
    members: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        members[uf.find(k)].append(k)
    digests = Digests()
    group_of, group_role = {}, {}
    for root, ms in members.items():
        gid = digests({"indice": "pannello-v1", "membri": sorted(ms)})
        fixed = {elements[m]["fixed"] for m in ms}
        if "sviluppo" in fixed:
            role, why = "sviluppo", "membro di sviluppo o già consultato"
        elif "test" in fixed:
            role, why = "test", "membro di test"
        elif "casuale" in fixed:
            frac = min(FRACTIONS[elements[m]["benchmark"]] for m in ms if elements[m]["fixed"] == "casuale")
            u = int(split_rank(args.seed, gid)[:15], 16) / 16 ** 15
            role, why = ("sviluppo" if u < frac else "test"), f"estratto: rango {u:.4f}, quota {frac}"
        elif "diagnostica" in fixed:
            role, why = "diagnostica", "solo membri di diagnostica"
        else:
            role, why = "solo_decontaminazione", "nessun membro valutato"
        for m in ms:
            group_of[m] = gid
        group_role[gid] = (role, why, len(ms))

    def final(k: str) -> str:
        e = elements[k]
        if e["fixed"] in ("solo_decontaminazione", "diagnostica"):
            return e["fixed"]
        return group_role[group_of[k]][0]

    # ---- candidati e irrisolti. Le formule di UniMER-Test, quasi bianche, danno vicini pHash senza senso (job 30039):
    # per loro contano pixel identici e formula identica. Un'immagine con più di MAX_NEIGHBORS vicini è un cluster
    # denso: si registrano i primi e il cluster si conta a parte.
    flat = [(k, pixels, h) for k in keys if elements[k]["benchmark"] != "unimer_test"
            for pixels, h in elements[k]["images"]]
    hidx = HashIndex([h for _, _, h in flat], radius=args.radius)
    cand_rows, cand_counts, unresolved = [], Counter(), {}
    seen_pairs = set()
    for i, (k, _, h) in enumerate(flat):
        near = hidx.query(h)
        if len(near) > MAX_NEIGHBORS:
            cand_counts[f"cluster_denso/{elements[k]['benchmark']}"] += 1
            near = sorted(near, key=lambda x: x[1])[:MAX_NEIGHBORS]
        for j, d in near:
            k2 = flat[j][0]
            if j <= i or group_of[k] == group_of[k2] or (k, k2) in seen_pairs:
                continue
            seen_pairs.add((k, k2))
            pair = " + ".join(sorted((elements[k]["benchmark"], elements[k2]["benchmark"])))
            cand_counts[f"phash/{pair}"] += 1
            r1, r2 = final(k), final(k2)
            cand_rows.append({"tipo": "phash", "a": k, "b": k2, "distanza": d, "ruoli": [r1, r2]})
            if {r1, r2} == {"sviluppo", "test"}:
                unresolved[k if r1 == "test" else k2] = k2 if r1 == "test" else k
    by_text: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        e = elements[k]
        if e["images"] and len(w := normalize_words(e["text"])) >= MIN_WORDS:
            by_text[" ".join(w)].append(k)
    for ks in by_text.values():
        groups = sorted({group_of[k] for k in ks})
        if len(groups) < 2:
            continue
        reps = [next(k for k in ks if group_of[k] == g) for g in groups]
        for a, b in zip(reps, reps[1:]):
            pair = " + ".join(sorted((elements[a]["benchmark"], elements[b]["benchmark"])))
            cand_counts[f"testo/{pair}"] += 1
            cand_rows.append({"tipo": "testo", "a": a, "b": b, "distanza": None, "ruoli": [final(a), final(b)]})

    # ---- uscite
    args.out.mkdir(parents=True, exist_ok=True)
    rows, per_bench, per_bench_split = [], Counter(), Counter()
    for k in keys:
        e = elements[k]
        role = final(k)
        gid = group_of[k]
        rows.append({"chiave": k, "benchmark": e["benchmark"], "file": e["file"], "split": e["split"], "item": e["item"],
                     "gruppo": gid, "dimensione_gruppo": group_role[gid][2], "ruolo_fisso": e["fixed"], "ruolo": role,
                     "motivo_gruppo": group_role[gid][1], "irrisolto_con": unresolved.get(k)})
        per_bench[f"{e['benchmark']}/{role}"] += 1
        per_bench_split[f"{e['benchmark']}/{e['split']}/{role}"] += 1
    files = {"elements": write_gz(args.out / "elements.jsonl.gz", rows),
             "links": write_gz(args.out / "links.jsonl.gz", link_rows),
             "candidates": write_gz(args.out / "candidates.jsonl.gz", cand_rows),
             "unresolved": write_gz(args.out / "unresolved.jsonl.gz",
                                    [{"test": t, "sviluppo": d} for t, d in sorted(unresolved.items())])}
    cross = Counter()
    for ms in members.values():
        benches = sorted({elements[m]["benchmark"] for m in ms})
        if len(benches) > 1:
            cross[" + ".join(benches)] += 1
    summary = {
        "schema": "panel-groups/1", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
        "indice": {"path": str(args.index), "sha256": sha256_file(args.index), "voci": len(index)},
        "seme": args.seed, "radius": args.radius, "quote_casuali": FRACTIONS, "parole_minime_testo": MIN_WORDS,
        "unimer_consultati": sorted(consulted), "elementi": len(keys), "gruppi": len(members),
        "gruppi_fra_benchmark": dict(cross.most_common()),
        "legami_confermati": dict(link_counts), "candidati": dict(cand_counts.most_common()),
        "irrisolti_test": len(unresolved),
        "irrisolti_per_benchmark": dict(Counter(elements[k]["benchmark"] for k in unresolved).most_common()),
        "ruoli_per_benchmark": dict(sorted(per_bench.items())),
        "ruoli_per_benchmark_split": dict(sorted(per_bench_split.items())), "files": files}
    write_new_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("elementi", "gruppi", "legami_confermati", "irrisolti_test",
                                              "irrisolti_per_benchmark", "ruoli_per_benchmark")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
