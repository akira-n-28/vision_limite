#!/usr/bin/env python3
"""Gruppi e ruoli del pannello di valutazione (docs/PANNELLO_VALUTAZIONE.md §3), dall'indice di scripts/panel_index.py.
Criteri, seme e assegnazione sono fissati qui, prima di qualsiasi valutazione del modello. Solo in un job SLURM CPU.

Legami **confermati** (fondono i gruppi):
- pixel identici: stesso `pixel_sha256` (convenzione `rgb-su-bianco/1` dell'indice) fra due elementi qualsiasi;
- provenienza o stesso problema, dichiarati nei dati: MathVerse `problem_index`, We-Math `ID`, DynaMath `id` fra le
  varianti, MathVision `id` fra testmini e test, DocVQA `docId` e documento UCSF (`ucsf_document_id`: le pagine dello
  stesso documento stanno insieme), ChartQA `imgname` nello stesso split, MUIRBench `counterpart_idx`, MMStar
  `meta_info` verso MathVista (stesso nome di file d'immagine), UniMER-Test con la stessa formula;
- benchmark di solo testo con lo stesso testo nella forma conservativa (`decontam.exact_text`: NFC e spazi compressi;
  operatori, segni, numeri e condizioni intatti), di almeno 8 parole. La forma di `normalize_words`, che toglie
  operatori e segni, dà solo candidati (`testo_permissivo`).
I legami si contano come unioni nuove, nell'ordine delle regole; quelli che confermano un gruppo già formato sono in
`legami_ridondanti`.

**Candidati** (non fondono nulla e non spostano nulla nello sviluppo):
- pHash entro --radius bit fra immagini di gruppi diversi. Un pHash vicino, o uguale, indica un candidato duplicato e
  non dimostra che le immagini siano identiche. Per ogni immagine i vicini dello stesso gruppo si tolgono prima del
  limite operativo --max-compare; se restano più vicini del limite si confrontano i più vicini e il gruppo
  dell'elemento ha copertura incompleta. Le coppie si deduplicano senza assumere che le liste siano simmetriche;
- lo stesso `imgname` di ChartQA in split diversi;
- testi uguali per normalize_words fra elementi visivi di gruppi diversi (un testo condiviso da più di
  MAX_TEXT_GROUPS gruppi è una domanda di rito e non conta), e fra elementi di solo testo con forma conservativa
  diversa (`testo_permissivo`).
Un candidato è **forte** se pHash e testo collegano la stessa coppia di gruppi. --max-saved limita solo i dettagli
salvati per la revisione (candidates.jsonl.gz), non il confronto.

**Irrisolti**: un candidato che collega un elemento di sviluppo a uno di test è un'evidenza diretta sull'elemento di
test. Contano i pHash, i nomi di file, i testi fra benchmark diversi con almeno TEXT_UNRESOLVED_WORDS parole e i
`testo_permissivo`; dentro lo stesso benchmark un testo uguale su un'immagine diversa è un altro elemento. Le
evidenze restano sul singolo elemento, ma l'inammissibilità alla metrica primaria vale per tutti i membri di test del
gruppo confermato (documenti, varianti e coppie di MUIRBench restano indivisibili). Anche un gruppo con copertura
incompleta è inammissibile. Nessun elemento cambia ruolo per un candidato.

Ruoli fissi (fixed_role) per benchmark e split; uno split non previsto da EXPECTED_SPLITS ferma il run. `casuale` vuol
dire estratto per gruppo con seme e quota: ogni gruppo va in sviluppo se il suo rango (manifest.split_rank) è sotto la
quota, altrimenti nel test. Un gruppo con un membro di sviluppo (o già consultato) va tutto in sviluppo; altrimenti,
con un membro di test, tutto nel test. I membri `solo_decontaminazione` e `diagnostica` restano tali, ma il loro
gruppo segue la regola. Tutti i ruoli sono esclusi dal training.

Uscite nella cartella nuova --out: elements.jsonl.gz (un elemento per riga: gruppo, ruolo, evidenze dirette, gruppo
irrisolto, copertura, ammissione alla metrica primaria), links.jsonl.gz, candidates.jsonl.gz, unresolved.jsonl.gz
(evidenze dirette), groups_excluded.jsonl.gz (gruppi con membri di test inammissibili e motivi), summary.json
(conteggi, copertura per ruolo, regole, seme, sha256 dell'indice, identità del codice; con --previous il riesame dei
legami testuali e il confronto col pannello precedente).

    srun $RUN_PYTHON -u -B scripts/panel_groups.py --index FILE --unimer-shards DIR --out DIR [--seed 0] [--radius 4] \\
        [--previous DIR]
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
from limite_vl.decontam import (  # noqa: E402
    CONVENTIONS, IMAGE_CONVENTION, HashIndex, exact_text, index_convention, normalize_words)
from limite_vl.manifest import Digests, split_rank  # noqa: E402
from limite_vl.provenance import code_identity, sha256_file, write_new_json  # noqa: E402

CONSULTED_UNIMER = 200  # i primi 200 di spe e di cpe: criterio di F1 e valutazioni F1c (job 29448-29761)
FRACTIONS = {"ocrbench": 0.5, "docvqa": 0.5, "muirbench": 0.5, "mmstar": 0.5, "unimer_test": 0.05}
TEXT_ONLY_DEV = {"gsm8k", "ifeval", "aime24", "aime25", "aime26", "hmmt25", "hmmt26"}
TEXT_ONLY_TEST = {"math500", "beyondaime", "apex"}
MIN_WORDS = 8
TEXT_UNRESOLVED_WORDS = 15  # come --propose-words di decontam_review.py
MAX_TEXT_GROUPS = 25  # come --max-df di decontam_review.py: un testo in più gruppi è una domanda di rito
MAX_COMPARE = 50_000  # limite operativo dei vicini di altri gruppi confrontati per immagine; oltre, copertura incompleta
MAX_SAVED = 50  # coppie candidate salvate al più per elemento, solo per la revisione
MAX_EVIDENCE = 10  # evidenze salvate per elemento irrisolto (il conteggio è completo)
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
OLYMPIAD_TEST = {"OE_MM_maths_en_COMP", "OE_MM_physics_en_COMP", "OE_TO_maths_en_COMP"}
# Split attesi per benchmark, quelli autorizzati nel catalogo: uno split diverso (per esempio il train di ChartQA, se
# entrasse nell'indice) ferma il run invece di diventare test.
EXPECTED_SPLITS = {
    "mathvista": {"testmini", "test"},
    "mathverse": {"testmini"},
    "mathvision": {"testmini", "test"},
    "wemath": {"testmini"},
    "dynamath": {f"sample{n}-00000-of-{n:05d}" for n in range(1, 11)},
    "geoperception": {"train"},
    "visonlyqa_real": {f"{c}-00000-of-00001" for c in (
        "charts__extraction", "charts__intersection", "chemistry__shape_multi", "chemistry__shape_single",
        "geometry__angle", "geometry__area", "geometry__diameter_radius", "geometry__length", "geometry__quadrilateral",
        "geometry__triangle")},
    "visonlyqa_synth": {f"{c}-00000-of-00001" for c in (
        "3d__angle", "3d__size", "syntheticgeometry__angle", "syntheticgeometry__area", "syntheticgeometry__length",
        "syntheticgeometry__quadrilateral", "syntheticgeometry__triangle")},
    "olympiadbench": {
        "OE_MM_maths_en_COMP", "OE_MM_maths_zh_CEE", "OE_MM_maths_zh_COMP", "OE_MM_physics_en_COMP",
        "OE_MM_physics_zh_CEE", "OE_TO_maths_en_COMP", "OE_TO_maths_zh_CEE", "OE_TO_maths_zh_COMP",
        "OE_TO_physics_en_COMP", "OE_TO_physics_zh_CEE", "TP_MM_maths_en_COMP", "TP_MM_maths_zh_CEE",
        "TP_MM_maths_zh_COMP", "TP_MM_physics_en_COMP", "TP_TO_maths_en_COMP", "TP_TO_maths_zh_CEE",
        "TP_TO_maths_zh_COMP", "TP_TO_physics_en_COMP"},
    "unimer_test": {"spe", "cpe", "sce", "hwe"},
    "ocrbench": {"test"},
    "docvqa": {"validation"},
    "chartqa": {"val", "test"},
    "muirbench": {"test"},
    "mmstar": {"mmstar"},
    "mvmath": {"MV-MATH"},
    "gsm8k": {"test"},
    "ifeval": {"ifeval_input_data"},
    "aime24": {"train"},
    "aime25": {"aime2025-I", "aime2025-II"},
    "aime26": {"train"},
    "hmmt25": {"train"},
    "hmmt26": {"train"},
    "apex": {"train"},
    "beyondaime": {"test"},
    "math500": {"test"},
}


def split_of(benchmark: str, file: str) -> str:
    stem = PurePosixPath(file).name
    for name in ("testmini", "test", "validation", "val", "train"):
        if stem.startswith(name):
            return name
    return PurePosixPath(file).parent.name if benchmark == "olympiadbench" else PurePosixPath(stem).stem


def fixed_role(e: dict, consulted: set[str], fractions: dict | None = None) -> str:
    """Ruolo fisso dell'elemento: sviluppo, test, casuale, diagnostica o solo_decontaminazione."""
    fractions = FRACTIONS if fractions is None else fractions
    b, s, h = e["benchmark"], e["split"], e["hints"]
    if b not in EXPECTED_SPLITS:
        raise ValueError(f"benchmark senza regola: {b}")
    if s not in EXPECTED_SPLITS[b]:
        raise ValueError(f"{b}: split inatteso {s!r} ({e['file']}), non previsto dal pannello riservato")
    if b == "mathvista":
        return "sviluppo" if s == "testmini" and h.get("language") == "english" else "solo_decontaminazione"
    if b in ("mathverse", "wemath", "geoperception"):
        return "sviluppo"
    if b == "mathvision":
        return {"testmini": "sviluppo", "test": "test"}[s]
    if b in ("dynamath", "visonlyqa_real", "visonlyqa_synth", "mvmath"):
        return "test"
    if b == "olympiadbench":
        return "test" if s in OLYMPIAD_TEST else "solo_decontaminazione"
    if b == "chartqa":
        return {"val": "sviluppo", "test": "test"}[s]
    if b == "unimer_test":
        if s == "hwe":
            return "diagnostica"
        return "sviluppo" if e["item"].split(":", 1)[-1] in consulted else "casuale"
    if b in fractions:
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


def phash_neighbors(images: list[tuple[str, str, int]], radius: int, max_compare: int | None):
    """Coppie candidate per pHash fra immagini di gruppi diversi.

    images: (chiave dell'elemento, gruppo, pHash), una voce per immagine. Per ogni immagine si cercano i vicini entro
    radius bit e si tolgono quelli dello stesso gruppo. Se ne restano più di max_compare si tiene il più vicino di ogni
    altro gruppo (il limite conta i gruppi, non i loro membri); se i gruppi sono ancora troppi si confrontano i più
    vicini (a parità di distanza in ordine di chiave) e l'elemento è segnato come troncato. Le coppie sono non ordinate
    e si raccolgono da entrambe le direzioni, senza assumere che le liste troncate siano simmetriche.
    Restituisce ({(a, b): distanza minima}, elementi troncati, {elemento: vicini di altri gruppi, al massimo per
    immagine})."""
    hidx = HashIndex([h for _, _, h in images], radius=radius)
    cache: dict[int, list[tuple[int, int]]] = {}
    pairs: dict[tuple[str, str], int] = {}
    truncated: set[str] = set()
    most: dict[str, int] = {}
    for key, group, h in images:
        near = cache.get(h)
        if near is None:
            near = cache[h] = hidx.query(h)
        cross = [(j, d) for j, d in near if images[j][1] != group]
        most[key] = max(most.get(key, 0), len(cross))
        if max_compare is not None and len(cross) > max_compare:
            closest: dict[str, tuple[int, int]] = {}
            for j, d in sorted(cross, key=lambda x: (x[1], images[x[0]][0])):
                closest.setdefault(images[j][1], (j, d))
            cross = sorted(closest.values(), key=lambda x: (x[1], images[x[0]][0]))
            if len(cross) > max_compare:
                truncated.add(key)
                cross = cross[:max_compare]
        for j, d in cross:
            other = images[j][0]
            pair = (key, other) if key < other else (other, key)
            if pairs.get(pair, radius + 1) > d:
                pairs[pair] = d
    return pairs, truncated, most


def plain(d):
    """defaultdict e Counter annidati -> dict ordinati, per il JSON."""
    return {k: plain(v) for k, v in sorted(d.items())} if isinstance(d, dict) else d


def build(index_items: list[dict], consulted: set[str], *, seed: int = 0, radius: int = 4,
          max_compare: int | None = MAX_COMPARE, max_saved: int = MAX_SAVED, fractions: dict | None = None,
          previous: dict | None = None) -> dict:
    """Gruppi, ruoli, candidati e ammissione alla metrica primaria dalle voci dell'indice. previous (facoltativo):
    {"links", "elements", "summary"} di un pannello precedente, per il riesame dei legami testuali e il confronto."""
    fractions = FRACTIONS if fractions is None else fractions
    # ---- elementi
    elements: dict[str, dict] = {}
    for it in index_items:
        key = f"{it['benchmark']}/{it['item']}"
        e = elements.get(key)
        if e is None:
            e = elements[key] = {"key": key, "benchmark": it["benchmark"], "file": it["file"],
                                 "split": split_of(it["benchmark"], it["file"]), "item": it["item"],
                                 "hints": it["hints"], "text": it["text"], "images": [], "positions": set()}
        elif e["file"] != it["file"]:
            raise ValueError(f"{key}: voci in file diversi ({e['file']}, {it['file']})")
        if it.get("pos") in e["positions"]:
            raise ValueError(f"{key}: posizione {it.get('pos')} ripetuta nell'indice")
        e["positions"].add(it.get("pos"))
        if it["pixel_sha256"]:
            e["images"].append((it["pixel_sha256"], int(it["phash"], 16)))
    for e in elements.values():
        e["fixed"] = fixed_role(e, consulted, fractions)
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
    # solo testo: confermati soltanto i testi uguali nella forma conservativa; le classi di normalize_words con forme
    # diverse danno candidati `testo_permissivo` fra i rappresentanti
    permissive: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        e = elements[k]
        if e["images"] or not (e["benchmark"] in TEXT_ONLY_DEV | TEXT_ONLY_TEST or e["benchmark"] == "olympiadbench"):
            continue
        words = normalize_words(e["text"])
        if len(words) >= MIN_WORDS:
            permissive[" ".join(words)].append(k)
    permissive_classes = []
    for ks in permissive.values():
        if len(ks) < 2:
            continue
        exact: dict[str, list[str]] = defaultdict(list)
        for k in ks:
            exact[exact_text(elements[k]["text"])].append(k)
        for cks in exact.values():
            for k in cks[1:]:
                link(cks[0], k, "testo_identico_conservativo")
        if len(exact) > 1:
            permissive_classes.append([cks[0] for cks in exact.values()])
    muir = {e["hints"].get("id"): k for k, e in elements.items() if e["benchmark"] == "muirbench"}
    muir_refs, muir_flags, none_texts = Counter(), Counter(), Counter()
    muir_pairs: set[tuple[str, str]] = set()
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
        if pair not in muir_pairs:
            muir_pairs.add(pair)
            muir_flags["+".join(sorted(str(elements[m]["derived"]["senza_risposta"]) for m in pair))] += 1
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
            frac = min(fractions[elements[m]["benchmark"]] for m in ms if elements[m]["fixed"] == "casuale")
            u = int(split_rank(seed, gid)[:15], 16) / 16 ** 15
            role, why = ("sviluppo" if u < frac else "test"), f"estratto: rango {u:.4f}, quota {frac}"
        elif "diagnostica" in fixed:
            role, why = "diagnostica", "solo membri di diagnostica"
        else:
            role, why = "solo_decontaminazione", "nessun membro valutato"
        for m in ms:
            group_of[m] = gid
        group_role[gid] = (role, why, len(ms))
    group_members = {group_of[ms[0]]: ms for ms in members.values()}

    def final(k: str) -> str:
        e = elements[k]
        if e["fixed"] in ("solo_decontaminazione", "diagnostica"):
            return e["fixed"]
        return group_role[group_of[k]][0]

    def set_of(k: str) -> str:
        return f"{elements[k]['benchmark']}/{elements[k]['split']}"

    # ---- candidati ed evidenze dirette
    cand_rows, cand_counts = [], Counter()
    pair_kinds: dict[tuple, set] = defaultdict(set)
    evidence: dict[str, list] = defaultdict(list)

    def candidate(kind: str, a: str, b: str, distance=None, unresolving: bool = True) -> None:
        if group_of[a] == group_of[b]:
            return
        pair = " + ".join(sorted((elements[a]["benchmark"], elements[b]["benchmark"])))
        cand_counts[f"{kind}/{pair}"] += 1
        r1, r2 = final(a), final(b)
        cand_rows.append({"tipo": kind, "a": a, "b": b, "distanza": distance, "ruoli": [r1, r2]})
        pair_kinds[tuple(sorted((group_of[a], group_of[b])))].add(kind)
        if unresolving and {r1, r2} == {"sviluppo", "test"}:
            t, d = (a, b) if r1 == "test" else (b, a)
            evidence[t].append({"sviluppo": d, "tipo": kind, "distanza": distance})

    # pHash. Le formule di UniMER-Test, quasi bianche, danno vicini pHash senza senso (job 30039): per loro contano
    # pixel identici e formula identica.
    images = [(k, group_of[k], h) for k in keys if elements[k]["benchmark"] != "unimer_test"
              for _, h in elements[k]["images"]]
    pairs, truncated, most = phash_neighbors(images, radius, max_compare)
    for (a, b), d in sorted(pairs.items()):
        candidate("phash", a, b, d)
    # stesso nome di file di ChartQA in split diversi
    by_name: dict[str, dict] = defaultdict(dict)
    for k in keys:
        e = elements[k]
        if e["benchmark"] == "chartqa" and e["hints"].get("imgname"):
            by_name[e["hints"]["imgname"]].setdefault(e["split"], k)
    for reps in by_name.values():
        if len(reps) > 1:
            a, b = sorted(reps.values())[:2]
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
    # solo testo: stessa forma permissiva, forma conservativa diversa
    for rep_keys in permissive_classes:
        for n, a in enumerate(rep_keys):
            for b in rep_keys[n + 1:]:
                candidate("testo_permissivo", a, b, None, True)
    strong = {p for p, kinds in pair_kinds.items() if {"phash", "testo"} <= kinds}

    def is_strong(t: str) -> bool:
        return any(tuple(sorted((group_of[t], group_of[c["sviluppo"]]))) in strong for c in evidence[t])

    # ---- inammissibilità per gruppo
    unresolved_groups = {group_of[t] for t in evidence}
    incomplete_groups = {group_of[k] for k in truncated if final(k) == "test"}

    def admitted(k: str) -> bool:
        return final(k) == "test" and group_of[k] not in unresolved_groups and group_of[k] not in incomplete_groups

    status = defaultdict(set)
    for k in keys:
        if final(k) == "test":
            status[group_of[k]].add(admitted(k))
    if any(len(s) > 1 for s in status.values()):
        raise RuntimeError("un gruppo ha membri di test ammessi e non ammessi")
    muir_check = Counter()
    for a, b in sorted(muir_pairs):
        ra, rb = final(a), final(b)
        if ra != rb:
            muir_check["ruoli_diversi"] += 1
        elif ra != "test":
            muir_check[ra] += 1
        else:
            sa, sb = admitted(a), admitted(b)
            muir_check["test_ammesse_entrambe" if sa and sb else "test_escluse_entrambe" if not (sa or sb)
                       else "test_divise"] += 1
    if muir_check["ruoli_diversi"] or muir_check["test_divise"]:
        raise RuntimeError(f"coppie di MUIRBench divise: {dict(muir_check)}")

    # ---- righe
    rows, per_bench, per_bench_split = [], Counter(), Counter()
    groups_per_role: dict[str, set] = defaultdict(set)
    coverage = defaultdict(lambda: defaultdict(Counter))
    docs = defaultdict(lambda: defaultdict(set))
    usable = defaultdict(Counter)
    for k in keys:
        e = elements[k]
        role = final(k)
        gid = group_of[k]
        ev = sorted(evidence.get(k, []), key=lambda c: (c["distanza"] if c["distanza"] is not None else radius + 1,
                                                       c["tipo"], c["sviluppo"]))
        ok = admitted(k)
        rows.append({"chiave": k, "benchmark": e["benchmark"], "file": e["file"], "split": e["split"], "item": e["item"],
                     "gruppo": gid, "dimensione_gruppo": group_role[gid][2], "ruolo_fisso": e["fixed"], "ruolo": role,
                     "motivo_gruppo": group_role[gid][1], "evidenze_dirette": len(ev),
                     "irrisolto_con": ev[0]["sviluppo"] if ev else None, "irrisolto_forte": is_strong(k) if ev else None,
                     "gruppo_irrisolto": gid in unresolved_groups, "copertura_incompleta": gid in incomplete_groups,
                     "metrica_primaria": ok if role == "test" else None})
        per_bench[f"{e['benchmark']}/{role}"] += 1
        per_bench_split[f"{e['benchmark']}/{e['split']}/{role}"] += 1
        groups_per_role[f"{e['benchmark']}/{role}"].add(gid)
        for f in COVERAGE.get(e["benchmark"], ()):
            v = e["derived"][f] if f in e["derived"] else e["hints"].get(f)
            for x in (v if isinstance(v, list) else [v]):
                coverage[f"{e['benchmark']}/{f}"][str(x)][role] += 1
                if ok:
                    coverage[f"{e['benchmark']}/{f}"][str(x)]["test_ammesso"] += 1
        if e["benchmark"] == "docvqa":
            for r in (role, "test_ammesso") if ok else (role,):
                docs[r]["documenti_ucsf"].add(e["hints"].get("ucsf"))
                docs[r]["pagine_docId"].add(e["hints"].get("docId"))
        if role == "test":
            u = usable[e["benchmark"]]
            u["test"] += 1
            u["evidenze_dirette"] += bool(ev)
            u["solo_per_gruppo"] += (gid in unresolved_groups) and not ev
            u["copertura_incompleta"] += gid in incomplete_groups
            u["ammessi"] += ok
    unresolved_rows = [{"test": t, "gruppo": group_of[t], "n_candidati": len(cs), "forte": is_strong(t),
                        "candidati": sorted(cs, key=lambda c: (c["distanza"] if c["distanza"] is not None else radius + 1,
                                                               c["tipo"], c["sviluppo"]))[:MAX_EVIDENCE]}
                       for t, cs in sorted(evidence.items())]
    excluded_rows = []
    for gid in sorted(unresolved_groups | incomplete_groups):
        tests = [m for m in group_members[gid] if final(m) == "test"]
        if not tests:
            continue
        excluded_rows.append({"gruppo": gid, "motivi": [m for m, cond in (("irrisolto", gid in unresolved_groups),
                                                                          ("copertura_incompleta",
                                                                           gid in incomplete_groups)) if cond],
                              "benchmark": sorted({elements[m]["benchmark"] for m in group_members[gid]}),
                              "membri": len(group_members[gid]), "membri_test": len(tests),
                              "con_evidenza_diretta": sum(m in evidence for m in tests),
                              "troncati": sum(m in truncated for m in group_members[gid])})
    # dettagli salvati: per elemento le max_saved coppie più vicine (il confronto resta completo)
    per_elem: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(cand_rows):
        per_elem[r["a"]].append(idx)
        per_elem[r["b"]].append(idx)
    keep: set[int] = set()
    for idxs in per_elem.values():
        idxs.sort(key=lambda i: (cand_rows[i]["distanza"] if cand_rows[i]["distanza"] is not None else radius + 1,
                                 cand_rows[i]["tipo"], cand_rows[i]["a"], cand_rows[i]["b"]))
        keep.update(idxs[:max_saved])
    saved_rows = [cand_rows[i] for i in sorted(keep)]

    # ---- riepilogo
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
    ucsf_roles = Counter(u for r, d in docs.items() if r != "test_ammesso" for u in d["documenti_ucsf"])
    neigh = defaultdict(int)
    over200 = Counter()
    for k, n in most.items():
        b = elements[k]["benchmark"]
        neigh[b] = max(neigh[b], n)
        over200[b] += n > 200
    unresolved_by = Counter(elements[k]["benchmark"] for k in evidence)
    summary = {
        "seme": seed, "radius": radius, "max_compare": max_compare, "max_saved": max_saved, "max_evidence": MAX_EVIDENCE,
        "schema_id_gruppo": GROUP_SCHEME, "quote_casuali": fractions, "parole_minime_testo": MIN_WORDS,
        "parole_testo_irrisolto": TEXT_UNRESOLVED_WORDS, "gruppi_massimi_testo": MAX_TEXT_GROUPS,
        "unimer_consultati": sorted(consulted), "elementi": len(keys), "gruppi": len(members),
        "gruppi_fra_benchmark": dict(cross.most_common()),
        "elementi_legati_fra_insiemi": dict(cross_sets.most_common()),
        "gruppi_piu_grandi": [{"membri": n, "composizione": c} for n, c in largest[:15]],
        "legami_confermati": dict(link_counts), "legami_ridondanti": dict(redundant),
        "testo_solo_testo": {"classi_permissive_con_forme_diverse": len(permissive_classes),
                             "candidati_testo_permissivo": sum(v for k, v in cand_counts.items()
                                                               if k.startswith("testo_permissivo/"))},
        "muirbench_controparti": {"riferimenti": dict(muir_refs), "coppie_per_senza_risposta": dict(muir_flags),
                                  "testi_della_risposta_con_none": dict(none_texts.most_common(10)),
                                  "verifica_coppie": dict(muir_check)},
        "docvqa_documenti": {r: {n: len(v) for n, v in d.items()} for r, d in sorted(docs.items())},
        "docvqa_documenti_in_piu_ruoli": sum(1 for n in ucsf_roles.values() if n > 1),
        "phash": {"immagini": len(images), "coppie_candidate": len(pairs), "elementi_troncati": len(truncated),
                  "vicini_massimi_per_benchmark": dict(sorted(neigh.items())),
                  "elementi_oltre_200_vicini": {b: n for b, n in sorted(over200.items()) if n}},
        "candidati": dict(cand_counts.most_common()), "candidati_totali": len(cand_rows),
        "candidati_salvati": len(saved_rows), "candidati_forti_coppie_di_gruppi": len(strong),
        "irrisolti_test": len(evidence), "irrisolti_per_benchmark": dict(unresolved_by.most_common()),
        "irrisolti_forti_per_benchmark": dict(Counter(elements[k]["benchmark"] for k in evidence
                                                      if is_strong(k)).most_common()),
        "irrisolti_per_tipo": dict(Counter(f"{elements[t]['benchmark']}/{'+'.join(sorted({c['tipo'] for c in cs}))}"
                                           for t, cs in evidence.items()).most_common()),
        "irrisolti_per_controparte": dict(Counter(f"{set_of(t)} <- {o}" for t, cs in evidence.items()
                                                  for o in sorted({set_of(c['sviluppo']) for c in cs})).most_common()),
        "gruppi_inammissibili": {"irrisolti": len(unresolved_groups), "copertura_incompleta": len(incomplete_groups),
                                 "con_membri_di_test": len(excluded_rows)},
        "test_utilizzabile": {b: dict(c) for b, c in sorted(usable.items())},
        "ruoli_per_benchmark": dict(sorted(per_bench.items())),
        "gruppi_per_benchmark_ruolo": {k: len(v) for k, v in sorted(groups_per_role.items())},
        "ruoli_per_benchmark_split": dict(sorted(per_bench_split.items())),
        "copertura": plain(coverage)}
    if previous:
        summary["riesame_testo_precedente"] = review_text_links(previous.get("links") or [], elements, group_of)
        summary["confronto_precedente"] = compare_panels(previous, rows, summary)
    return {"rows": rows, "links": link_rows, "candidates": saved_rows, "unresolved": unresolved_rows,
            "excluded_groups": excluded_rows, "summary": summary, "group_of": group_of, "role": final,
            "admitted": admitted, "evidence": evidence, "truncated": truncated, "pairs": pairs}


def review_text_links(links: list[dict], elements: dict, group_of: dict) -> dict:
    """Riesame delle unioni della vecchia regola testo_identico_solo_testo (normalize_words): restano confermate se i
    due testi coincidono nella forma conservativa, altrimenti sono candidati."""
    out = []
    for r in links:
        if r.get("tipo") != "testo_identico_solo_testo":
            continue
        a, b = elements.get(r["a"]), elements.get(r["b"])
        if a is None or b is None:
            out.append({"a": r["a"], "b": r["b"], "esito": "elemento assente"})
            continue
        ta, tb = exact_text(a["text"]), exact_text(b["text"])
        row = {"a": r["a"], "b": r["b"], "esito": "confermato" if ta == tb else "candidato",
               "stesso_gruppo": group_of[r["a"]] == group_of[r["b"]]}
        if ta != tb:
            i = next((n for n, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            row["differenza"] = {"posizione": i, "a": ta[max(0, i - 30):i + 50], "b": tb[max(0, i - 30):i + 50]}
        out.append(row)
    return {"unioni": len(out), "confermate": sum(r["esito"] == "confermato" for r in out),
            "candidate": sum(r["esito"] == "candidato" for r in out), "dettagli": out}


def compare_panels(previous: dict, rows: list[dict], summary: dict) -> dict:
    """Differenze per benchmark rispetto a un pannello precedente (v2: test utilizzabile = test senza irrisolti)."""
    prev = {r["chiave"]: r for r in previous.get("elements") or []}
    cur = {r["chiave"]: r for r in rows}
    per = defaultdict(Counter)
    for k in sorted(cur.keys() & prev.keys()):
        p, c = prev[k], cur[k]
        b = c["benchmark"]
        if p["ruolo"] != c["ruolo"]:
            per[b][f"ruolo {p['ruolo']} -> {c['ruolo']}"] += 1
        per[b]["gruppo cambiato"] += p["gruppo"] != c["gruppo"]
        p_ok = p["ruolo"] == "test" and p.get("irrisolto_con") is None
        c_ok = bool(c["metrica_primaria"])
        per[b]["test utilizzabile prima"] += p_ok
        per[b]["test utilizzabile ora"] += c_ok
        per[b]["evidenze dirette prima"] += p.get("irrisolto_con") is not None
        per[b]["evidenze dirette ora"] += bool(c["evidenze_dirette"])
        per[b]["usciti dal test utilizzabile"] += p_ok and not c_ok
        per[b]["entrati nel test utilizzabile"] += c_ok and not p_ok
    ps = previous.get("summary") or {}
    return {"elementi_solo_prima": len(prev.keys() - cur.keys()), "elementi_solo_ora": len(cur.keys() - prev.keys()),
            "gruppi": {"prima": ps.get("gruppi"), "ora": summary["gruppi"]},
            "candidati": {"prima": sum((ps.get("candidati") or {}).values()), "ora": sum(summary["candidati"].values())},
            "irrisolti_diretti": {"prima": ps.get("irrisolti_test"), "ora": summary["irrisolti_test"]},
            "per_benchmark": {b: {k: v for k, v in c.items() if v} for b, c in sorted(per.items())}}


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


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
    ap.add_argument("--max-compare", type=int, default=MAX_COMPARE)
    ap.add_argument("--max-saved", type=int, default=MAX_SAVED)
    ap.add_argument("--previous", type=Path, help="cartella di un pannello precedente (riesame e confronto)")
    ap.add_argument("--convention", choices=CONVENTIONS, default=None,
                    help="convenzione di un indice che non la dichiara (pannello-v1 e v2: rgb-su-bianco/1)")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("panel_groups.py va eseguito in un job SLURM")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} non è vuota: un'assegnazione si scrive una volta sola")
    loaded = json.loads(args.index.read_text(encoding="utf-8"))
    try:
        convention = index_convention(loaded, args.convention)
    except ValueError as e:
        sys.exit(f"{args.index}: {e}")
    if convention != IMAGE_CONVENTION:
        sys.exit(f"il pannello usa lo sha256 dei pixel, che esiste solo nella convenzione {IMAGE_CONVENTION}")
    # chiavi di UniMER-Test già consultate: le prime 200 di spe e di cpe nell'ordine degli shard
    consulted: set[str] = set()
    for sub in ("spe", "cpe"):
        for n, (key, _) in enumerate(iter_shard(args.unimer_shards / f"unimer-test-{sub}.tar")):
            if n >= CONSULTED_UNIMER:
                break
            consulted.add(key)
    previous = None
    if args.previous:
        previous = {"links": read_gz(args.previous / "links.jsonl.gz"),
                    "elements": read_gz(args.previous / "elements.jsonl.gz"),
                    "summary": json.loads((args.previous / "summary.json").read_text(encoding="utf-8")),
                    "path": str(args.previous)}
    result = build(loaded["items"], consulted, seed=args.seed, radius=args.radius, max_compare=args.max_compare,
                   max_saved=args.max_saved, previous=previous)
    args.out.mkdir(parents=True, exist_ok=True)
    files = {"elements": write_gz(args.out / "elements.jsonl.gz", result["rows"]),
             "links": write_gz(args.out / "links.jsonl.gz", result["links"]),
             "candidates": write_gz(args.out / "candidates.jsonl.gz", result["candidates"]),
             "unresolved": write_gz(args.out / "unresolved.jsonl.gz", result["unresolved"]),
             "groups_excluded": write_gz(args.out / "groups_excluded.jsonl.gz", result["excluded_groups"])}
    summary = {"schema": "panel-groups/3", "job_id": os.environ.get("SLURM_JOB_ID"), "code": code_identity(),
               "indice": {"path": str(args.index), "sha256": sha256_file(args.index), "voci": len(loaded["items"]),
                          "convenzione_immagini": convention},
               "precedente": str(args.previous) if args.previous else None, **result["summary"], "files": files}
    write_new_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("elementi", "gruppi", "legami_confermati", "irrisolti_test",
                                              "gruppi_inammissibili", "test_utilizzabile", "phash")},
                     ensure_ascii=False))
    if previous:
        print(json.dumps({k: summary[k] for k in ("riesame_testo_precedente", "confronto_precedente")},
                         ensure_ascii=False)[:20000])


if __name__ == "__main__":
    main()
