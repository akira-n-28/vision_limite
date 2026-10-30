#!/usr/bin/env python3
"""Gruppi e ruoli del pannello di valutazione (docs/PANNELLO_VALUTAZIONE.md §3), dall'indice di scripts/panel_index.py.
Criteri, seme e assegnazione sono fissati qui, prima di qualsiasi valutazione del modello. Solo in un job SLURM CPU.

Legami **confermati** (fondono i gruppi):
- pixel identici: stesso `pixel_sha256` (dimensioni e valori RGB normalizzati) fra due elementi qualsiasi;
- provenienza o stesso problema, dichiarati nei dati: MathVerse `problem_index`, We-Math `ID`, DynaMath `id` fra le
  varianti, MathVision `id` fra testmini e test, DocVQA `docId` e documento UCSF (`ucsf_document_id`: le pagine dello
  stesso documento stanno insieme), ChartQA `imgname` nello stesso split, MUIRBench `counterpart_idx`, MMStar
  `meta_info` verso MathVista (stesso nome di file d'immagine), UniMER-Test con la stessa formula, benchmark di solo
  testo con lo stesso testo normalizzato (almeno 8 parole).
I legami si contano come unioni nuove, nell'ordine delle regole; quelli che confermano un gruppo già formato sono in
`legami_ridondanti`.

**Candidati** (non fondono nulla):
- pHash entro --radius bit fra immagini di gruppi diversi. Un pHash vicino, o uguale, indica un candidato duplicato e
  non dimostra che le immagini siano identiche;
- lo stesso `imgname` di ChartQA in split diversi;
- testi uguali (almeno 8 parole) fra elementi visivi di gruppi diversi. Un testo condiviso da più di MAX_TEXT_GROUPS
  gruppi è una domanda di rito e non conta.
Un candidato è **forte** se pHash e testo collegano la stessa coppia di gruppi.

**Irrisolti**: candidati che collegano un elemento di sviluppo a uno di test. Contano i pHash, i nomi di file e i testi
fra benchmark diversi con almeno TEXT_UNRESOLVED_WORDS parole; dentro lo stesso benchmark un testo uguale su
un'immagine diversa è un altro elemento. L'elemento di test resta nel test, ma è contato a parte finché il legame non
si conferma o si esclude (verifica visiva).

Ruoli fissi (fixed_role) per benchmark e split; `casuale` vuol dire estratto per gruppo con seme e quota: ogni gruppo
va in sviluppo se il suo rango (manifest.split_rank) è sotto la quota, altrimenti nel test. Un gruppo con un membro di
sviluppo (o già consultato) va tutto in sviluppo; altrimenti, con un membro di test, tutto nel test. I membri
`solo_decontaminazione` (senza risposte, fuori lingua o non verificabili) e `diagnostica` restano tali, ma il loro
gruppo segue la regola. Tutti i ruoli sono esclusi dal training.

Uscite nella cartella nuova --out: elements.jsonl.gz (un elemento per riga: benchmark, file, split, item, gruppo,
ruolo fisso, ruolo finale, irrisolto), links.jsonl.gz (legami confermati, per tipo), candidates.jsonl.gz,
unresolved.jsonl.gz, summary.json (conteggi, copertura per ruolo, regole, seme, sha256 dell'indice, identità del
codice).

    srun $RUN_PYTHON -u -B scripts/panel_groups.py --index FILE --unimer-shards DIR --out DIR [--seed 0] [--radius 4]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
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
TEXT_UNRESOLVED_WORDS = 15  # come --propose-words di decontam_review.py
MAX_TEXT_GROUPS = 25  # come --max-df di decontam_review.py: un testo in più gruppi è una domanda di rito
MAX_NEIGHBORS = 200
# Etichetta dello schema degli ID di gruppo, non del run: a parità di membri un gruppo ha lo stesso ID, e quindi la
# stessa estrazione, in ogni run con lo stesso seme.
GROUP_SCHEME = "pannello-v1"
NONE_OPTION = re.compile(r"\bnone of\b", re.I)  # opzione corretta delle varianti non rispondibili di MUIRBench
# Campi contati per ruolo (le liste contano ogni valore); `immagini` e `senza_risposta` sono derivati.
COVERAGE = {
    "docvqa": ("question_types",),
    "ocrbench": ("question_type", "dataset"),
    "chartqa": ("type",),
    "muirbench": ("task", "image_relation", "image_type", "senza_risposta", "immagini"),
    "mmstar": ("category", "l2_category", "source"),
    "mathvista": ("context", "task", "category", "source"),
    "mathverse": ("version",),
    "geoperception": ("predicate",),
    "mvmath": ("subject", "answer_type", "image_relavance", "immagini"),
    "olympiadbench": ("subset", "immagini"),
}
NAMED_SPLITS = {"testmini", "test", "val", "validation"}


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


def unanswerable(e: dict) -> bool:
    """Variante non rispondibile di MUIRBench: l'opzione indicata dalla risposta è del tipo "None of ..."."""
    options, answer = e["hints"].get("options") or [], str(e["hints"].get("answer") or "").strip().upper()
    i = ord(answer) - ord("A") if len(answer) == 1 and "A" <= answer <= "Z" else -1
    return bool(0 <= i < len(options) and NONE_OPTION.search(str(options[i])))


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


def plain(d):
    """defaultdict e Counter annidati -> dict ordinati, per il JSON."""
    return {k: plain(v) for k, v in sorted(d.items())} if isinstance(d, dict) else d


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
        e["derived"] = {"immagini": len(e["images"])}
        if e["benchmark"] == "muirbench":
            e["derived"]["senza_risposta"] = unanswerable(e)
    keys = sorted(elements)
    uf = Union(keys)
    link_counts, redundant, link_rows = Counter(), Counter(), []

    def link(a: str, b: str, kind: str) -> None:
        if a == b:
            return
        if uf.union(a, b):
            link_counts[kind] += 1
            link_rows.append({"tipo": kind, "a": a, "b": b})
        else:
            redundant[kind] += 1

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
    link_by("docvqa_documento_ucsf", lambda e: e["benchmark"] == "docvqa", lambda e: e["hints"].get("ucsf"))
    link_by("chartqa_grafico", lambda e: e["benchmark"] == "chartqa",
            lambda e: f"{e['split']}/{e['hints'].get('imgname')}" if e["hints"].get("imgname") else None)
    link_by("unimer_stessa_formula", lambda e: e["benchmark"] == "unimer_test",
            lambda e: (e["hints"].get("formula") or "").strip() or None)
    link_by("testo_identico_solo_testo", lambda e: not e["images"] and (e["benchmark"] in TEXT_ONLY_DEV | TEXT_ONLY_TEST
                                                                          or e["benchmark"] == "olympiadbench"),
            lambda e: " ".join(w) if len(w := normalize_words(e["text"])) >= MIN_WORDS else None)
    muir = {e["hints"].get("id"): k for k, e in elements.items() if e["benchmark"] == "muirbench"}
    muir_refs, muir_pairs, none_texts = Counter(), Counter(), Counter()
    seen_muir = set()
    for k in keys:
        e = elements[k]
        if e["benchmark"] != "muirbench":
            continue
        ref = muir.get(e["hints"].get("counterpart_idx"))
        if ref is None or ref == k:
            muir_refs["assente" if ref is None else "se_stessa"] += 1
            continue
        muir_refs["trovata"] += 1
        link(k, ref, "muirbench_controparte")
        pair = tuple(sorted((k, ref)))
        if pair not in seen_muir:
            seen_muir.add(pair)
            flags = sorted(str(elements[m]["derived"]["senza_risposta"]) for m in pair)
            muir_pairs["+".join(flags)] += 1
    for k in keys:
        e = elements[k]
        if e["benchmark"] == "muirbench":
            opts, ans = e["hints"].get("options") or [], str(e["hints"].get("answer") or "").strip().upper()
            i = ord(ans) - ord("A") if len(ans) == 1 and "A" <= ans <= "Z" else -1
            if 0 <= i < len(opts) and "none" in str(opts[i]).lower():
                none_texts[str(opts[i])] += 1
    vista = {PurePosixPath(str(e["hints"].get("image"))).name: k for k, e in elements.items()
             if e["benchmark"] == "mathvista" and e["hints"].get("image")}
    for k in keys:
        e = elements[k]
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
        gid = digests({"indice": GROUP_SCHEME, "membri": sorted(ms)})
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

    def set_of(k: str) -> str:
        return f"{elements[k]['benchmark']}/{elements[k]['split']}"

    # ---- candidati e irrisolti
    cand_rows, cand_counts = [], Counter()
    pair_kinds: dict[tuple, set] = defaultdict(set)
    unresolved: dict[str, list] = defaultdict(list)

    def candidate(kind: str, a: str, b: str, distance=None, unresolving: bool = True) -> None:
        pair = " + ".join(sorted((elements[a]["benchmark"], elements[b]["benchmark"])))
        cand_counts[f"{kind}/{pair}"] += 1
        r1, r2 = final(a), final(b)
        cand_rows.append({"tipo": kind, "a": a, "b": b, "distanza": distance, "ruoli": [r1, r2]})
        pair_kinds[tuple(sorted((group_of[a], group_of[b])))].add(kind)
        if unresolving and {r1, r2} == {"sviluppo", "test"}:
            t, d = (a, b) if r1 == "test" else (b, a)
            unresolved[t].append({"sviluppo": d, "tipo": kind, "distanza": distance})

    # pHash. Le formule di UniMER-Test, quasi bianche, danno vicini pHash senza senso (job 30039): per loro contano
    # pixel identici e formula identica. Un'immagine con più di MAX_NEIGHBORS vicini è un cluster denso: si registrano
    # i più vicini e il cluster si conta a parte.
    flat = [(k, pixels, h) for k in keys if elements[k]["benchmark"] != "unimer_test"
            for pixels, h in elements[k]["images"]]
    hidx = HashIndex([h for _, _, h in flat], radius=args.radius)
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
            candidate("phash", k, k2, d)
    # stesso nome di file di ChartQA in split diversi
    by_name: dict[str, dict] = defaultdict(dict)
    for k in keys:
        e = elements[k]
        if e["benchmark"] == "chartqa" and e["hints"].get("imgname"):
            by_name[e["hints"]["imgname"]].setdefault(e["split"], k)
    for reps in by_name.values():
        if len(reps) > 1:
            a, b = sorted(reps.values())[:2]
            if group_of[a] != group_of[b]:
                candidate("nome_file", a, b)
    # testi uguali fra elementi visivi
    by_text: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        e = elements[k]
        if e["images"] and len(w := normalize_words(e["text"])) >= MIN_WORDS:
            by_text[" ".join(w)].append(k)
    for text, ks in sorted(by_text.items()):
        reps: dict[str, str] = {}
        for k in ks:
            reps.setdefault(group_of[k], k)
        if len(reps) < 2:
            continue
        if len(reps) > MAX_TEXT_GROUPS:
            cand_counts["testo_di_rito/" + "+".join(sorted({elements[k]["benchmark"] for k in ks}))] += 1
            continue
        groups = sorted(reps)
        long_text = len(text.split()) >= TEXT_UNRESOLVED_WORDS
        for n, g1 in enumerate(groups):
            for g2 in groups[n + 1:]:
                a, b = reps[g1], reps[g2]
                candidate("testo", a, b, None, long_text and elements[a]["benchmark"] != elements[b]["benchmark"])
    strong = {p for p, kinds in pair_kinds.items() if {"phash", "testo"} <= kinds}

    def is_strong(t: str) -> bool:
        return any(tuple(sorted((group_of[t], group_of[c["sviluppo"]]))) in strong for c in unresolved[t])

    # ---- uscite
    args.out.mkdir(parents=True, exist_ok=True)
    rows, per_bench, per_bench_split = [], Counter(), Counter()
    groups_per_role: dict[str, set] = defaultdict(set)
    coverage = defaultdict(lambda: defaultdict(Counter))
    docs = defaultdict(lambda: defaultdict(set))
    for k in keys:
        e = elements[k]
        role = final(k)
        gid = group_of[k]
        rows.append({"chiave": k, "benchmark": e["benchmark"], "file": e["file"], "split": e["split"], "item": e["item"],
                     "gruppo": gid, "dimensione_gruppo": group_role[gid][2], "ruolo_fisso": e["fixed"], "ruolo": role,
                     "motivo_gruppo": group_role[gid][1],
                     "irrisolto_con": unresolved[k][0]["sviluppo"] if k in unresolved else None,
                     "irrisolto_forte": is_strong(k) if k in unresolved else None})
        per_bench[f"{e['benchmark']}/{role}"] += 1
        per_bench_split[f"{e['benchmark']}/{e['split']}/{role}"] += 1
        groups_per_role[f"{e['benchmark']}/{role}"].add(gid)
        for f in COVERAGE.get(e["benchmark"], ()):
            v = e["derived"][f] if f in e["derived"] else e["hints"].get(f)
            for x in (v if isinstance(v, list) else [v]):
                coverage[f"{e['benchmark']}/{f}"][str(x)][role] += 1
        if e["benchmark"] == "docvqa":
            docs[role]["documenti_ucsf"].add(e["hints"].get("ucsf"))
            docs[role]["pagine_docId"].add(e["hints"].get("docId"))
    ucsf_roles = Counter()
    for role, d in docs.items():
        for u in d["documenti_ucsf"]:
            ucsf_roles[u] += 1
    unresolved_rows = [{"test": t, "candidati": cs[:10], "n_candidati": len(cs), "forte": is_strong(t)}
                       for t, cs in sorted(unresolved.items())]
    files = {"elements": write_gz(args.out / "elements.jsonl.gz", rows),
             "links": write_gz(args.out / "links.jsonl.gz", link_rows),
             "candidates": write_gz(args.out / "candidates.jsonl.gz", cand_rows),
             "unresolved": write_gz(args.out / "unresolved.jsonl.gz", unresolved_rows)}
    cross, cross_sets, largest = Counter(), Counter(), []
    for ms in members.values():
        benches = sorted({elements[m]["benchmark"] for m in ms})
        if len(benches) > 1:
            cross[" + ".join(benches)] += 1
        sets = {set_of(m) for m in ms}
        if len(sets) > 1:
            for m in ms:
                s = set_of(m)
                for o in sets:
                    if o != s and (elements[m]["benchmark"] != o.split("/")[0]
                                   or {elements[m]["split"], o.split("/", 1)[1]} <= NAMED_SPLITS):
                        cross_sets[f"{s} -> {o}"] += 1
        largest.append((len(ms), " + ".join(f"{b}:{n}" for b, n in sorted(Counter(elements[m]["benchmark"]
                                                                                     for m in ms).items()))))
    largest.sort(key=lambda x: (-x[0], x[1]))
    test_counts = Counter(elements[k]["benchmark"] for k in keys if final(k) == "test")
    unresolved_by = Counter(elements[k]["benchmark"] for k in unresolved)
    summary = {
        "schema": "panel-groups/2", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
        "indice": {"path": str(args.index), "sha256": sha256_file(args.index), "voci": len(index)},
        "seme": args.seed, "radius": args.radius, "schema_id_gruppo": GROUP_SCHEME, "quote_casuali": FRACTIONS,
        "parole_minime_testo": MIN_WORDS, "parole_testo_irrisolto": TEXT_UNRESOLVED_WORDS,
        "gruppi_massimi_testo": MAX_TEXT_GROUPS, "unimer_consultati": sorted(consulted), "elementi": len(keys),
        "gruppi": len(members), "gruppi_fra_benchmark": dict(cross.most_common()),
        "elementi_legati_fra_insiemi": dict(cross_sets.most_common()),
        "gruppi_piu_grandi": [{"membri": n, "composizione": c} for n, c in largest[:15]],
        "legami_confermati": dict(link_counts), "legami_ridondanti": dict(redundant),
        "muirbench_controparti": {"riferimenti": dict(muir_refs), "coppie_per_senza_risposta": dict(muir_pairs),
                                  "testi_della_risposta_con_none": dict(none_texts.most_common(10))},
        "docvqa_documenti": {r: {n: len(v) for n, v in d.items()} for r, d in sorted(docs.items())},
        "docvqa_documenti_in_piu_ruoli": sum(1 for n in ucsf_roles.values() if n > 1),
        "candidati": dict(cand_counts.most_common()), "candidati_forti_coppie_di_gruppi": len(strong),
        "irrisolti_test": len(unresolved), "irrisolti_per_benchmark": dict(unresolved_by.most_common()),
        "irrisolti_forti_per_benchmark": dict(Counter(elements[k]["benchmark"] for k in unresolved
                                                      if is_strong(k)).most_common()),
        "irrisolti_per_tipo": dict(Counter(f"{elements[t]['benchmark']}/{'+'.join(sorted({c['tipo'] for c in cs}))}"
                                           for t, cs in unresolved.items()).most_common()),
        "irrisolti_per_controparte": dict(Counter(f"{set_of(t)} <- {o}" for t, cs in unresolved.items()
                                                  for o in sorted({set_of(c['sviluppo']) for c in cs})).most_common()),
        "test_senza_irrisolti": {b: {"test": n, "irrisolti": unresolved_by[b], "senza_irrisolti": n - unresolved_by[b]}
                                 for b, n in sorted(test_counts.items())},
        "ruoli_per_benchmark": dict(sorted(per_bench.items())),
        "gruppi_per_benchmark_ruolo": {k: len(v) for k, v in sorted(groups_per_role.items())},
        "ruoli_per_benchmark_split": dict(sorted(per_bench_split.items())),
        "copertura": plain(coverage), "files": files}
    write_new_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("elementi", "gruppi", "legami_confermati", "legami_ridondanti",
                                              "irrisolti_test", "irrisolti_per_benchmark", "test_senza_irrisolti",
                                              "ruoli_per_benchmark")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
