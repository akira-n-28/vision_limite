#!/usr/bin/env python3
"""C9 di MAVIS-Geometry sui record locali (src/limite_vl/mavis_c9.py; S2 §3, S1 §9). Solo in un job SLURM CPU: shard,
manifest e pannello in sola lettura, uscite in cartelle nuove.

Legge solo i JSON degli 11 tar (intestazioni in sequenza, dati delle immagini saltati), ciascuno legato al record di
mavis-geometry-candidato-3 da posizione e json_sha256. Le immagini non cambiano e non si rileggono.

Sottocomandi:
- `sonda`: ricognizione dei fatti nelle annotazioni (tipi dichiarati, misure per figura, forme dei valori, vertici
  degli isosceli), nessuna decisione e nessun manifest -> --out/report.json.
- `costruzione`: per ogni record pool storico e descrizione ricalcolati e uguali ai digest del genitore (compatibilità
  della costruzione predefinita), poi pool C9 (mavis_c9.questions), valori esatti/visualizzati/richiesti ed esiti di
  shape_type per figura, ogni domanda recuperabile riverificata dal controllo indipendente. Le domande nuove si
  confrontano coi testi di pannello-v3 (10-gram, regole di panel_decontam): un candidato manda il gruppo in quarantena.
  Scrive la scansione (--out/report.json), la versione nuova derivata dal genitore (--new: stessi record, gruppi, split,
  evidenze e stati; costruzione_qa, qa, compiti di shape_type, valori, shape_type_c9 ed evidenza mavis-c9 nuovi), la
  verifica (--report/verifica.json) e il riepilogo per git (--summary). Niente manifest se un controllo fallisce.
- `verifica`: la verifica della versione C9, rileggendo genitore e figlio.
- `lettore`: collegamento col reader (C7) sul primo shard, col manifest C9 e col genitore.

    srun $RUN_PYTHON -u -B scripts/mavis_c9.py sonda --manifest DIR --manifest-sha256 HEX --out DIR [--workers 4]
    srun $RUN_PYTHON -u -B scripts/mavis_c9.py costruzione --manifest DIR --manifest-sha256 HEX --new DIR --out DIR \\
        --report DIR --index DIR --index-sha256 HEX --summary FILE --version N [--workers 4]
    srun $RUN_PYTHON -u -B scripts/mavis_c9.py lettore --manifest DIR --manifest-sha256 HEX --new DIR --new-sha256 HEX \\
        --shards GLOB --out DIR
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import mavis_manifest as mm  # noqa: E402
import unimer_manifest as um  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl import mavis_c9 as c9  # noqa: E402
from limite_vl.f1_mavis import describe, mavis_questions, mavis_truth  # noqa: E402
from limite_vl.mavis_c9 import angle_degrees, measure_values, provably_equal, shape_facts, shown_of  # noqa: E402
from limite_vl.mavis_gen import FIGURE  # noqa: E402
from limite_vl.provenance import code_identity, file_entry, sha256_file, write_new_json  # noqa: E402

EXAMPLES = 4


class Failed(Exception):
    pass


class CountingFile(io.FileIO):
    """File senza buffer che conta i byte letti (come panel_decontam.CountingFile)."""

    letti = 0

    def read(self, size=-1):
        data = super().read(size)
        self.letti += len(data or b"")
        return data

    def readinto(self, b):
        n = super().readinto(b)
        self.letti += n or 0
        return n


def shard_jsons(path: Path, expected: dict[str, tuple[int, str]], stats: Counter):
    """(chiave, byte del JSON) di uno shard, nell'ordine del tar, saltando i dati delle immagini. Ogni JSON deve avere
    la posizione e lo sha256 del record del manifest; ogni record del manifest deve essere trovato. `stats` conta
    intestazioni, JSON e byte letti."""
    seen = set()
    with CountingFile(path, "rb") as raw, tarfile.open(fileobj=raw, mode="r:") as tar:
        pos, last = -1, None
        for member in tar:
            stats["intestazioni"] += 1
            if not member.isfile():
                continue
            stem, _, ext = member.name.rpartition("/")[2].partition(".")
            if stem != last:
                pos, last = pos + 1, stem
            if ext != "json":
                continue
            data = tar.extractfile(member).read()
            stats["json"] += 1
            want = expected.get(stem)
            if want is None or stem in seen:
                raise Failed(f"{path.name}: {stem} assente dal manifest o ripetuto")
            if want != (pos, hashlib.sha256(data).hexdigest()):
                raise Failed(f"{path.name}: {stem} con posizione o json_sha256 diversi dal manifest")
            seen.add(stem)
            yield stem, data
        stats["byte_letti"] += raw.letti
    if seen != set(expected):
        raise Failed(f"{path.name}: {len(set(expected) - seen)} record del manifest senza JSON")


def _side_pattern(name: str, sides: dict) -> str:
    n = len(name)
    idx = sorted(i for i in range(n) if name[i] + name[(i + 1) % n] in sides)
    if len(idx) == 2:
        return "adiacenti" if (idx[1] - idx[0]) % n in (1, n - 1) else "opposti"
    return str(len(idx))


def probe_shard(task) -> dict:
    """Conteggi della sonda su uno shard (nel processo del pool)."""
    path, expected = task
    t0 = time.time()
    c, ex, stats = Counter(), defaultdict(list), Counter()

    def example(k, key, text=""):
        if len(ex[k]) < EXAMPLES:
            ex[k].append(key if not text else f"{key}: {text[:300]}")

    for key, data in shard_jsons(Path(path), expected, stats):
        te = json.loads(data)["text_en"]
        gen = te["diagram_generation"]
        truth, why = mavis_truth(te["caption"], gen)
        c["record"] += 1
        if truth is None:
            c[f"senza_verita/{why}"] += 1
            continue
        c["shape_type_storiche"] += sum(q["type"] == "shape_type" for q in mavis_questions(key, truth))
        vals = measure_values(gen)
        reps = Counter(v["rappresentazione"] for v in vals.values())
        for k, v in vals.items():
            c[f"valori/{k.split(':')[0]}/{v['rappresentazione']}"] += 1
            if v["rappresentazione"] in ("decimale_arrotondato", "non_analizzabile"):
                example(f"valori/{v['rappresentazione']}", key, f"{k}={v['esatto']} -> {v['visualizzato_previsto']}")
        c["record_con_decimale_arrotondato"] += reps["decimale_arrotondato"] > 0
        c["record_con_decimale_esatto"] += reps["decimale_esatto"] > 0
        c["record_con_razionale_non_intero"] += (reps["decimale_arrotondato"] + reps["decimale_esatto"]) > 0
        for f in shape_facts(truth, gen):
            k = f.kind
            c[f"figure/{k}"] += 1
            if f.family == "other":
                c[f"figure_altra_famiglia/{k}/{len(f.name)}"] += 1
                example(f"altra_famiglia/{k}", key, f.sentence)
            m = FIGURE.search(f.sentence)
            c[f"forma/{k}/{m['form'] if m else '?'}"] += 1
            c[f"lati/{k}/{_side_pattern(f.name, f.sides) if f.family != 'sector' else len(f.sides)}"] += 1
            c[f"angoli_dati/{k}/{len(f.angles)}"] += 1
            if f.inscribed_circle:
                c[f"cerchio_inscritto/{k}"] += 1
            for vertex, v in f.angles.items():
                c[f"angolo/{k}/pos{f.name.index(vertex)}/{v}"] += 1
            shown = {xy: shown_of(v) for xy, v in f.sides.items()}
            if any(s is None for s in shown.values()):
                c[f"lati_non_analizzabili/{k}"] += 1
                continue
            vals_sides = list(shown.values())
            if len(vals_sides) >= 2:
                eq = any(provably_equal(a, b) for i, a in enumerate(vals_sides) for b in vals_sides[i + 1:])
                same_text = len({s.text for s in vals_sides}) < len(vals_sides)
                c[f"lati_uguali_stampati/{k}/{'dimostrato' if eq else 'scrittura_uguale' if same_text else 'no'}"] += 1
                if same_text and not eq:
                    example(f"uguaglianza_apparente/{k}", key, str({xy: s.text for xy, s in shown.items()}))
            if k == "isosceles triangle":
                c[f"isoscele/regola/{f.apex_rule}"] += 1
                if f.apex is None:
                    example(f"isoscele/{f.apex_rule}", key, f.sentence)
                else:
                    c[f"isoscele/vertice_pos{f.name.index(f.apex)}"] += 1
                    for vertex, v in f.angles.items():
                        c[f"isoscele/angolo_{'al_vertice' if vertex == f.apex else 'alla_base'}/{v}"] += 1
                    legs = [xy for xy in f.sides if f.apex in xy]
                    c[f"isoscele/lati_dati_al_vertice/{len(legs)}"] += 1
            if k == "right triangle":
                degs = [angle_degrees(s) for s in map(shown_of, f.angles.values()) if s is not None]
                c[f"rettangolo/novanta_dato/{any(d == 90 for d in degs)}"] += 1
                example(f"rettangolo/{len(f.angles)}angoli", key, f.sentence)
            if k in ("triangle", "equilateral triangle", "rhombus", "kite", "trapezoid"):
                example(f"tipo/{k}", key, f.sentence)
            if k == "parallelogram" and len(f.sides) >= 2 and len(f.angles) >= 1:
                ang = sorted(f.angles.values())
                c[f"parallelogramma/angolo/{ang[0]}"] += 1
    return {"path": path, "c": c, "ex": ex, "stats": stats, "secondi": round(time.time() - t0, 1)}


def load_parent(args) -> mf.Manifest:
    m = mf.load(args.manifest)
    if m.manifest_sha256 != args.manifest_sha256:
        raise Failed(f"manifest con sha256 {m.manifest_sha256}, atteso {args.manifest_sha256}")
    return m


def tasks_of(m: mf.Manifest) -> list[tuple[str, dict]]:
    return [(str(mf.shard_path(rel)), {k: (r["indice"], r["json_sha256"]) for k, r in rows.items()})
            for rel, rows in sorted(m.by_shard.items())]


def cmd_sonda(args) -> int:
    t0 = time.time()
    if args.out.exists() and any(args.out.iterdir()):
        raise Failed(f"{args.out} non è vuota")
    pool = Pool(args.workers)  # prima del manifest: i processi non ereditano i record
    m = load_parent(args)
    total, ex, stats = Counter(), defaultdict(list), Counter()
    with pool:
        for res in pool.imap_unordered(probe_shard, tasks_of(m)):
            total.update(res["c"])
            stats.update(res["stats"])
            for k, v in res["ex"].items():
                ex[k] = (ex[k] + v)[:EXAMPLES]
            print(f"{Path(res['path']).name}: {res['c']['record']} record in {res['secondi']} s", flush=True)
    report = {"schema": "mavis-c9-sonda/1", "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "manifest": {"percorso": str(args.manifest), "manifest_sha256": m.manifest_sha256},
              "lettura": dict(stats), "conteggi": dict(sorted(total.items())), "esempi": dict(sorted(ex.items())),
              "secondi": round(time.time() - t0, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    write_new_json(args.out / "report.json", report)
    print(json.dumps({"record": total["record"], "shape_type_storiche": total["shape_type_storiche"],
                      "secondi": report["secondi"]}))
    return 0


# ---------------------------------------------------------------- costruzione del candidato C9

CHECK = "mavis-c9"
HIT_REASON = ("mavis-c9: domanda shape_type nuova con un 10-gram in comune con un testo di pannello-v3, non coperto da "
              "una regola esplicita (gruppo in quarantena)")
SHAPE_REASON = ("domande C9 con un insieme fisso di opzioni per famiglia (mavis-c9/1: una sola vera secondo le "
                "annotazioni, le altre confutate dalle misure stampate); escluse dal percorso operativo in questo lotto "
                "per decisione dell'utente del 2026-11-10: da approvare, non gold")


def facts_doc(f: c9.ShapeFacts) -> dict:
    return {"famiglia": f.family, "lati": dict(f.sides), "angoli": dict(f.angles), "vertice": f.apex}


def facts_of(entry: dict) -> c9.ShapeFacts:
    d = entry["fatti"]
    return c9.ShapeFacts(entry["figura"], entry["tipo_dichiarato"], d["famiglia"], "", sides=dict(d["lati"]),
                         angles=dict(d["angoli"]), apex=d.get("vertice"))


def reading_sha(elem: str, answer: str) -> tuple[str, str]:
    """(tipo, digest) della domanda di lettura storica sulla misura `elem` con la risposta `answer`."""
    kind, name = elem.split(":")
    q = f"What is the length of {name}?" if kind == "side" else f"What is the measure of angle {name}?"
    t = "side_len" if kind == "side" else "angle_val"
    return t, mf.qa_item_sha256({"type": t, "q": q, "answer": answer})


def c9_shard(task) -> dict:
    """Costruzione C9 di uno shard (nel processo del pool). Per ogni record: pool storico e descrizione ricalcolati e
    uguali ai digest del genitore (la costruzione predefinita non cambia), pool C9, valori, esiti di shape_type, ogni
    domanda recuperabile riverificata da mavis_c9.check_outcome."""
    path, expected = task
    t0 = time.time()
    stats, c, sets, problems, rows = Counter(), Counter(), Counter(), [], {}
    for key, data in shard_jsons(Path(path), {k: v[:2] for k, v in expected.items()}, stats):
        _, _, parent_qa, parent_desc, depth = expected[key]
        te = json.loads(data)["text_en"]
        gen = te["diagram_generation"]
        truth, why = mavis_truth(te["caption"], gen)
        if truth is None:  # record senza verità (in quarantena nel genitore, senza domande né descrizione)
            if parent_qa != mf.qa_list_sha256([]) or parent_desc is not None:
                problems.append(f"{key}: verità non disponibile ({why}) ma il genitore ha domande o descrizione")
            rows[key] = {"qa": [], "valori": [], "forme": [], "testi": []}
            c["record"] += 1
            c[f"record/senza_verita/{why}"] += 1
            continue
        hist = mavis_questions(key, truth)
        if mf.qa_list_sha256((q["type"], mf.qa_item_sha256(q)) for q in hist) != parent_qa:
            problems.append(f"{key}: domande della costruzione storica diverse dai digest del genitore")
            continue
        if mf.sha256_text(describe(key, truth)[1]) != parent_desc:
            problems.append(f"{key}: descrizione diversa da quella del genitore")
            continue
        try:
            pool, outcomes, values = c9.questions(key, truth, gen)
        except ValueError as exc:
            problems.append(f"{key}: {exc}")
            continue
        facts = c9.shape_facts(truth, gen)
        same = [(q["type"], mf.qa_item_sha256(q)) for q in hist if q["type"] != "shape_type"]
        if [(q["type"], mf.qa_item_sha256(q)) for q in pool if q["type"] != "shape_type"] != same:
            problems.append(f"{key}: domande diverse da shape_type cambiate dalla costruzione C9")
        shapes, texts = [], []
        for f, o in zip(facts, outcomes):
            bad = c9.check_outcome(key, f, o)
            if bad:
                problems.append(f"{key} {f.name}: controllo indipendente: {bad[:3]}")
            entry = {k: o[k] for k in ("figura", "tipo_dichiarato", "esito", "motivo", "opzioni", "risposta",
                                       "confutazioni", "non_confutate", "vere", "ignote", "dettaglio") if k in o}
            entry["fatti"] = facts_doc(f)
            shapes.append(entry)
            label = o["motivo"] if o["esito"] == "omessa" else o["risposta"]
            c[f"forme/{f.kind}/{o['esito']}/{label}"] += 1
            c[f"profondita/d{depth}/{f.kind}/{o['esito']}/{label}"] += 1
            c[f"esiti/{o['esito']}"] += 1
            if o["esito"] == "recuperabile":
                texts.append(o["domanda"]["q"])
                sets[f"{f.family}|{','.join(sorted(o['opzioni']))}|{o['risposta']}"] += 1
                c[f"posizione/{f.family}/{o['opzioni'].index(o['risposta'])}"] += 1
                c[f"risposte/{f.family}/{o['risposta']}"] += 1
        for elem, v in values.items():
            c[f"valori/{elem.split(':')[0]}/{v['rappresentazione']}"] += 1
        for q in pool:
            if "elemento" in q:
                rep = values[q["elemento"]]["rappresentazione"]
                c[f"lettura/{q['type']}/{rep}"] += 1
                c["lettura/esatto_diverso_dal_target"] += rep == "decimale_arrotondato"
        reps = {v["rappresentazione"] for v in values.values()}
        c["record/con_decimale_arrotondato"] += "decimale_arrotondato" in reps
        c["record/con_decimale_esatto"] += "decimale_esatto" in reps
        c["record/descrizione_con_decimale_arrotondato"] += "decimale_arrotondato" in reps
        c["record"] += 1
        rows[key] = {"qa": [(q["type"], mf.qa_item_sha256(q), q.get("elemento")) for q in pool],
                     "valori": [{"elemento": e, **v} for e, v in values.items()], "forme": shapes, "testi": texts}
    return {"path": path, "rows": rows, "c": c, "sets": sets, "stats": stats, "problems": problems,
            "secondi": round(time.time() - t0, 1)}


def text_check(index_dir: Path, expected_sha: str | None, texts: list[str]) -> tuple[dict, dict]:
    """10-gram delle domande nuove contro tutti i testi dell'indice del pannello, con le regole di panel_decontam
    (normalize_words, n = 10, n-gram di rito in più di 25 testi del pannello esclusi): {testo: {elemento: [10-gram]}}
    e le informazioni sull'indice. Nessuna regola esplicita copre questo campo: ogni corrispondenza è un candidato."""
    import panel_decontam as pd
    from limite_vl.decontam import TextIndex, normalize_words

    loaded, sha = pd.read_index(index_dir)
    if expected_sha and sha != expected_sha:
        raise Failed(f"indice del pannello con sha256 {sha}, atteso {expected_sha}")
    if loaded.get("schema") != pd.PANEL_SCHEMA:
        raise Failed(f"indice con schema {loaded.get('schema')!r}, atteso {pd.PANEL_SCHEMA}")
    items = loaded["items"]
    tidx = TextIndex.build([it.get("text") or "" for it in items], n=pd.NGRAM, max_df=pd.MAX_DF)
    out = {}
    for t in texts:
        words, hits = normalize_words(t), defaultdict(set)
        for i in range(len(words) - tidx.n + 1):
            g = tuple(words[i:i + tidx.n])
            for j in tidx.grams.get(g, ()):
                hits[pd.element_of(items[j])].add(" ".join(g))
        if hits:
            out[t] = {e: sorted(gs) for e, gs in sorted(hits.items())}
    info = {"indice": {**file_entry(index_dir / "index.json", sha), "schema": loaded.get("schema"), "voci": len(items)},
            "ngram": pd.NGRAM, "max_df": pd.MAX_DF, "ngram_di_rito_esclusi": tidx.dropped,
            "campo": "domande shape_type della costruzione mavis-c9/1", "regole_esplicite": "nessuna per questo campo",
            "testi_distinti": len(texts), "testi_con_corrispondenze": len(out)}
    return out, info


def parent_lines(folder: Path, meta: dict) -> list[dict]:
    raw = (folder / meta["records"]["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["records"]["sha256"]:
        raise Failed(f"{folder}: records.jsonl.gz diverso da quello registrato")
    return [json.loads(x) for x in gzip.decompress(raw).decode("utf-8").splitlines()]


def new_tasks(compiti: list[dict], state: str, n_shape: int) -> list[dict]:
    """Compiti del record C9: quelli del genitore, con lo stato del record (shape_type almeno in quarantena) e le
    domande shape_type contate sul pool C9 (il compito sparisce se tutte le figure sono omesse)."""
    out = []
    for t in compiti:
        if t["compito"] == "shape_type":
            if n_shape:
                out.append({"compito": "shape_type", "domande": n_shape, "stato": um.worst(state, "quarantena")})
        else:
            out.append({**t, "stato": state})
    return out


def cmd_costruzione(args) -> int:
    import panel_decontam as pd

    t0 = time.time()
    for p in (args.new, args.report, args.out):
        if p.exists() and any(p.iterdir()):
            raise Failed(f"{p} non è vuota: si scrive una volta sola")
    if args.summary.exists():
        raise Failed(f"{args.summary} esiste già")
    pool = Pool(args.workers)  # prima del manifest: i processi non ereditano i record
    try:
        m = load_parent(args)
        if m.meta.get("sorgente") != "mavis-geometry":
            raise Failed(f"genitore della sorgente {m.meta.get('sorgente')!r}")
        if any(r.get("costruzione_qa") for rows in m.by_shard.values() for r in rows.values()):
            raise Failed("il genitore dichiara già una costruzione delle domande: C9 si deriva da un manifest storico")
    except BaseException:
        pool.terminate()
        raise
    meta0 = m.meta
    tasks = [(str(mf.shard_path(rel)),
              {k: (r["indice"], r["json_sha256"], mf.qa_list_sha256((q["tipo"], q["sha256"]) for q in r["qa"]),
                   r["descrizione_sha256"], r["famiglia"][1:]) for k, r in rows.items()})
             for rel, rows in sorted(m.by_shard.items())]
    parent_sha, folder0 = m.manifest_sha256, m.folder
    del m
    gc.collect()
    rows, c, sets, stats, problems, per_shard = {}, Counter(), Counter(), Counter(), [], {}
    with pool:
        for res in pool.imap_unordered(c9_shard, tasks):
            rows.update(res["rows"])
            c.update(res["c"])
            sets.update(res["sets"])
            stats.update(res["stats"])
            problems += res["problems"]
            per_shard[Path(res["path"]).name] = {"record": len(res["rows"]), "secondi": res["secondi"]}
            print(f"{Path(res['path']).name}: {len(res['rows'])} record in {res['secondi']} s", flush=True)
    scan_seconds = round(time.time() - t0, 1)
    if problems:
        raise Failed(f"{len(problems)} problemi, manifest non scritto: {problems[:5]}")
    texts = sorted({t for r in rows.values() for t in r["testi"]})
    hits, text_info = text_check(args.index, args.index_sha256, texts)
    lines = parent_lines(folder0, meta0)
    hit_groups = {r["gruppo"] for r in lines if any(t in hits for t in rows[r["record"]]["testi"])}
    new, out_lines, changes = [], [], Counter()
    for r in lines:
        res = rows[r["record"]]
        rec = dict(r)
        state = um.worst(r["stato"], "quarantena") if r["gruppo"] in hit_groups else r["stato"]
        if state != r["stato"]:
            changes[f"{r['split']}/{r['stato']}->{state}"] += 1
            rec["motivo"] = list(r["motivo"] or []) + [HIT_REASON]
        own = sorted({e for t in res["testi"] for e in hits.get(t, {})})
        rec["stato"] = state
        rec["costruzione_qa"] = c9.CONSTRUCTION
        rec["qa"] = [{"tipo": t, "sha256": s, **({"elemento": e} if e else {})} for t, s, e in res["qa"]]
        n_shape = sum(t == "shape_type" for t, _, _ in res["qa"])
        rec["compiti"] = new_tasks(r["compiti"], state, n_shape)
        rec["valori"] = res["valori"]
        rec["shape_type_c9"] = res["forme"]
        rec["evidenze"] = list(r["evidenze"] or []) + [{
            "controllo": CHECK, "versione": c9.CONSTRUCTION, "job": os.environ.get("SLURM_JOB_ID"),
            "shape_type": dict(Counter(f["esito"] for f in res["forme"])), "valori": len(res["valori"]),
            "testo_nuovo": "candidato" if own else "nessuna_corrispondenza", "elementi": own,
            "esito_gruppo": "candidato" if r["gruppo"] in hit_groups else "nessuna_corrispondenza",
            "rif": {"report": um.rel(args.out / "report.json")}}]
        new.append(rec)
        out_lines.append(json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    del rows, lines
    gc.collect()
    args.new.mkdir(parents=True, exist_ok=True)
    data_sha, n_lines = um.write_records_gz(args.new / "records.jsonl.gz", iter(out_lines))
    del out_lines
    (args.new / "keys").mkdir()
    derived = {}
    for name, ks in pd.derived_keys("mavis", new).items():
        text = "".join(k + "\n" for k in sorted(ks))
        (args.new / "keys" / name).write_text(text, encoding="utf-8")
        derived[f"keys/{name}"] = {**file_entry(args.new / "keys" / name, mf.sha256_text(text)), "righe": len(ks)}
    split = dict(meta0["split"])
    split["ereditato_da"] = {"percorso": um.rel(args.manifest), "versione": meta0["versione"],
                             "manifest_sha256": parent_sha}
    split["regola_versione"] = ("split e gruppi della versione precedente, invariati: C9 cambia la costruzione delle "
                                "domande shape_type, aggiunge valori ed esiti per figura; gli stati cambiano solo per "
                                "un candidato del controllo testuale delle domande nuove")
    admitted = Counter(f"{x['split']}/{x['famiglia']}" for x in new if x["stato"] == "ammesso")
    split["ammessi_dopo_c9"] = dict(sorted(admitted.items()))
    chain = [dict(x) for x in meta0.get("catena_delle_versioni") or []]
    if chain and chain[-1].get("versione") == meta0["versione"]:
        chain[-1]["manifest_sha256"] = parent_sha
    chain.append({"versione": args.version, "percorso": um.rel(args.new), "manifest_sha256": None,
                  "job": os.environ.get("SLURM_JOB_ID"),
                  "decisione": "C9: valori esatti e visualizzati nel record, domande shape_type con opzioni "
                               "(mavis-c9/1), controllo testuale mirato delle domande nuove"})
    stati = dict(meta0["stati"])
    stati["quarantena"] += "; oppure gruppo con una domanda C9 nuova in comune con un testo del pannello: da verificare"
    report = {"schema": "mavis-c9-costruzione/1", "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "costruzione": c9.CONSTRUCTION,
              "genitore": {"percorso": um.rel(args.manifest), "manifest_sha256": parent_sha},
              "lettura": dict(stats), "per_shard": per_shard, "conteggi": dict(sorted(c.items())),
              "insiemi_di_opzioni": dict(sorted(sets.items())), "controllo_testuale": text_info,
              "corrispondenze_testuali": hits, "gruppi_in_quarantena_per_il_testo": len(hit_groups),
              "cambi_di_stato": dict(changes), "secondi_scansione": scan_seconds, "rss_max_mib": um.rss_mib()}
    args.out.mkdir(parents=True, exist_ok=True)
    write_new_json(args.out / "report.json", report)
    meta = {**meta0, "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
            "precedente": {"percorso": um.rel(args.manifest), "versione": meta0["versione"],
                           "manifest_sha256": parent_sha, "records_sha256": meta0["records"]["sha256"]},
            "split": split, "stati": stati,
            "costruzione_qa": {
                "nome": c9.CONSTRUCTION, "modulo": "src/limite_vl/mavis_c9.py",
                "regola": "pool storico di f1_mavis.mavis_questions con le domande shape_type sostituite (insieme fisso "
                          "di opzioni per famiglia, ordine per chiave) o tolte con il motivo; le altre domande "
                          "identiche, con la misura che chiedono (elemento); il loader la usa solo se il record la "
                          "dichiara (costruzione_qa)",
                "opzioni": {k: list(v) for k, v in c9.OPTIONS.items()}, "classi_escluse": c9.EXCLUDED,
                "valori": {"esatto": c9.VALUE_SOURCE, "visualizzato_previsto": c9.DISPLAY_RULE,
                           "risposta_lettura": "risposta richiesta dai compiti di lettura (side_len, angle_val, "
                                               "describe): il visualizzato previsto, come i target storici"},
                "report": file_entry(args.out / "report.json"), "controllo_testuale": text_info},
            "compiti_in_quarantena": {"shape_type": SHAPE_REASON},
            "conteggi": mm.summarize(new),
            "records": {"file": "records.jsonl.gz", "sha256": data_sha, "righe": n_lines},
            "derivati": derived, "catena_delle_versioni": chain,
            "fuori_dal_sottoinsieme": [x for x in meta0["fuori_dal_sottoinsieme"] if not x.startswith("C9:")]
            + ["C9 (mavis-c9/1): shape_type costruita e controllata ma esclusa dal percorso operativo; nessun gold"],
            "verifiche_mancanti": [x for x in meta0["verifiche_mancanti"]
                                   if not x.startswith(("shape_type in quarantena", "valori razionali"))]
            + ["shape_type C9: approvazione dell'utente e verifica a vista delle figure e delle misure stampate su cui "
               "poggiano le confutazioni; la sonda e la scansione provano proprietà delle annotazioni, non dei pixel",
               "valori: visualizzato previsto dalla regola del renderer, non letto dai pixel"],
            "secondi_costruzione": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()}
    write_new_json(args.new / "manifest.json", meta)
    del new
    gc.collect()
    result = verify_c9(args.new, args.manifest, args.manifest_sha256, args.out / "report.json")
    result.update({"job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
                   "secondi_totali": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib()})
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "verifica.json", result)
    summary = {"schema": "limite-vl-manifest-riepilogo/1", "manifest": um.rel(args.new), "nome": meta["nome"],
               "versione": args.version, "job_id": os.environ.get("SLURM_JOB_ID"),
               "manifest_sha256": sha256_file(args.new / "manifest.json"), "records_sha256": data_sha,
               "righe": n_lines, "precedente": meta["precedente"], "costruzione_qa": c9.CONSTRUCTION,
               "conteggi_c9": {k: v for k, v in sorted(c.items()) if not k.startswith("profondita/")},
               "insiemi_di_opzioni": dict(sorted(sets.items())), "controllo_testuale": text_info,
               "cambi_di_stato": dict(changes), "conteggi": meta["conteggi"], "split_ammessi": split["ammessi_dopo_c9"],
               "controlli": [{k: x[k] for k in ("controllo", "esito")} for x in result["controlli"]],
               "esito_verifica": result["esito"],
               "nota": "C9 documenta le domande shape_type recuperabili e i valori: non certifica le risposte, non le "
                       "promuove a gold e non attiva il candidato in nessuna configurazione"}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.summary, summary)
    print(json.dumps({"righe": n_lines, "esiti": {k: v for k, v in c.items() if k.startswith("esiti/")},
                      "verifica": f"{result['superati']}/{result['totale']}", "secondi": result["secondi_totali"],
                      "rss_max_mib": result["rss_max_mib"]}, ensure_ascii=False))
    if not result["esito"]:
        raise Failed("verifica della versione nuova fallita: vedi " + str(args.report / "verifica.json"))
    return 0


INHERITED = ("record_id", "sorgente", "revisione", "insieme", "shard", "indice", "famiglia", "gruppo", "split",
             "immagini", "json_sha256", "didascalia_sha256", "diagram_generation_sha256", "verita_sha256",
             "descrizione_sha256", "misure_razionali_come_decimali", "rango_f0", "rango_f0_sviluppo", "gold",
             "trasformazione", "genitori")


def verify_c9(child_dir: Path, parent_dir: Path, parent_sha: str, report_path: Path) -> dict:
    """Controlli rileggendo il candidato C9 e il genitore (manifest.load), senza gli shard: identità, gruppi, split ed
    evidenze ereditati; stati cambiati solo per i gruppi del controllo testuale; domande diverse da shape_type uguali al
    genitore; ogni domanda di lettura ricalcolata dal valore visualizzato registrato; ogni domanda shape_type
    ricalcolata dalle opzioni e riverificata da mavis_c9.check_question coi fatti registrati; conteggi e file derivati."""
    checks = []

    def ok(name, cond, detail=None):
        checks.append({"controllo": name, "esito": bool(cond), "dettaglio": detail})

    parent, child = mf.load(parent_dir), mf.load(child_dir)
    ok("genitore con lo sha256 atteso", parent.manifest_sha256 == parent_sha, parent.manifest_sha256)
    pm, cm = parent.meta, child.meta
    ok("precedente e split.ereditato_da indicano il genitore",
       cm["precedente"]["manifest_sha256"] == parent_sha == cm["split"]["ereditato_da"]["manifest_sha256"]
       and cm["precedente"]["records_sha256"] == pm["records"]["sha256"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hits = report["corrispondenze_testuali"]
    P = {(s, k): r for s, rows in parent.by_shard.items() for k, r in rows.items()}
    C = {(s, k): r for s, rows in child.by_shard.items() for k, r in rows.items()}
    ok("stessi record negli stessi shard", set(P) == set(C), (len(P), len(C)))
    bad = Counter()
    first: dict[str, str] = {}

    def fail(what, key):
        bad[what] += 1
        first.setdefault(what, key)

    hit_groups = set()
    for key, r in C.items():
        for e in r["shape_type_c9"]:
            if e["esito"] == "recuperabile" and c9.question_text(e["fatti"]["famiglia"], e["figura"],
                                                                 e["opzioni"]) in hits:
                hit_groups.add(r["gruppo"])
    for key, p in P.items():
        r = C.get(key)
        if r is None:
            continue
        k = f"{key[1]}"
        if any(r.get(f) != p.get(f) for f in INHERITED):
            fail("campi ereditati diversi dal genitore", k)
        expected_state = um.worst(p["stato"], "quarantena") if p["gruppo"] in hit_groups else p["stato"]
        if r["stato"] != expected_state:
            fail("stato diverso da quello atteso (genitore o gruppo del controllo testuale)", k)
        if r["motivo"][:len(p["motivo"])] != p["motivo"] or (len(r["motivo"]) - len(p["motivo"])) != (
                expected_state != p["stato"]):
            fail("motivi del genitore non conservati o motivo nuovo senza cambio di stato", k)
        if r["evidenze"][:-1] != p["evidenze"] or r["evidenze"][-1].get("controllo") != CHECK:
            fail("evidenze del genitore non conservate o evidenza C9 mancante", k)
        if r.get("costruzione_qa") != c9.CONSTRUCTION:
            fail("costruzione delle domande non dichiarata", k)
        pq = [(q["tipo"], q["sha256"]) for q in p["qa"] if q["tipo"] != "shape_type"]
        rq = [(q["tipo"], q["sha256"]) for q in r["qa"] if q["tipo"] != "shape_type"]
        if pq != rq:
            fail("domande diverse da shape_type cambiate", k)
        values = {v["elemento"]: v for v in r["valori"]}
        for q in r["qa"]:
            if q["tipo"] in ("side_len", "angle_val"):
                v = values.get(q.get("elemento"))
                if v is None or reading_sha(q["elemento"], v["risposta_lettura"]) != (q["tipo"], q["sha256"]):
                    fail("domanda di lettura non ricalcolata dal visualizzato registrato", k)
                elif v["risposta_lettura"].rstrip("°") != v["visualizzato_previsto"]:
                    fail("risposta di lettura diversa dal visualizzato previsto", k)
        ptasks = [t for t in p["compiti"] if t["compito"] != "shape_type"]
        rtasks = [t for t in r["compiti"] if t["compito"] != "shape_type"]
        if [{**t, "stato": r["stato"]} for t in ptasks] != rtasks:
            fail("compiti diversi da shape_type cambiati", k)
        shape_q = [q for q in r["qa"] if q["tipo"] == "shape_type"]
        st = [t for t in r["compiti"] if t["compito"] == "shape_type"]
        if (st and (st[0]["stato"] == "ammesso" or st[0]["domande"] != len(shape_q))) or (not st and shape_q):
            fail("compito shape_type ammesso o con un numero di domande diverso dal pool", k)
        n_hist = next((t["domande"] for t in p["compiti"] if t["compito"] == "shape_type"), 0)
        if len(r["shape_type_c9"]) != n_hist:
            fail("esiti per figura diversi dalle domande shape_type storiche", k)
        recovered = [e for e in r["shape_type_c9"] if e["esito"] == "recuperabile"]
        shas = [mf.qa_item_sha256({"type": "shape_type", "q": c9.question_text(e["fatti"]["famiglia"], e["figura"],
                                                                               e["opzioni"]), "answer": e["risposta"]})
                for e in recovered]
        if shas != [q["sha256"] for q in shape_q]:
            fail("domande shape_type del pool diverse da quelle ricalcolate dagli esiti", k)
        for e in recovered:
            if c9.check_question(r["record"], facts_of(e), e["opzioni"], e["risposta"],
                                 c9.question_text(e["fatti"]["famiglia"], e["figura"], e["opzioni"])):
                fail("domanda shape_type rifiutata dal controllo indipendente coi fatti registrati", k)
    ok("nessuna anomalia nei record (campi ereditati, stati, motivi, evidenze, domande, valori, compiti, esiti)",
       not bad, {"conteggi": dict(bad), "primi": first})
    recs = list(C.values())
    ok("conteggi uguali a manifest.json", mm.summarize(recs) == cm["conteggi"])
    import panel_decontam as pd

    keys_ok = all(mf.sha256_text("".join(x + "\n" for x in sorted(ks))) == cm["derivati"][f"keys/{n}"]["sha256"]
                  for n, ks in pd.derived_keys("mavis", recs).items())
    ok("file derivati (keys/) ricalcolati dai record", keys_ok)
    ok("gruppi del controllo testuale ricalcolati dagli esiti", len(hit_groups) == report[
        "gruppi_in_quarantena_per_il_testo"], len(hit_groups))
    passed = sum(x["esito"] for x in checks)
    return {"schema": "mavis-c9-verifica/1", "manifest": um.rel(child_dir), "genitore": um.rel(parent_dir),
            "controlli": checks, "superati": passed, "totale": len(checks), "esito": passed == len(checks)}


def cmd_verifica(args) -> int:
    result = verify_c9(args.new, args.manifest, args.manifest_sha256, args.out / "report.json")
    print(json.dumps({"verifica": f"{result['superati']}/{result['totale']}"}))
    args.report.mkdir(parents=True, exist_ok=True)
    write_new_json(args.report / "verifica.json", {**result, "job_id": os.environ.get("SLURM_JOB_ID"),
                                                   "codice": code_identity()})
    return 0 if result["esito"] else 1


# ---------------------------------------------------------------- collegamento col reader

def cmd_lettore(args) -> int:
    """Il primo shard letto per intero, in train e in dev, col manifest C9 e col genitore (C7): chiavi consegnate uguali
    ai record ammessi dei record grezzi; ogni campione costruito dal loader col manifest C9 (digest del pool C9
    verificato), compiti solo fra gli ammessi, shape_type mai, domande shape_type C9 contate fra le bloccate, e gli stessi
    messaggi del genitore; senza la costruzione dichiarata il pool C9 viene rifiutato."""
    import glob

    import reader_smoke as rs
    from limite_vl import data as dt
    from limite_vl.stream import make_sources

    t0 = time.time()
    if args.out.exists() and any(args.out.iterdir()):
        raise Failed(f"{args.out} non è vuota")
    first = sorted(glob.glob(args.shards))[0]
    specs = [{"name": f"{name}/{split}", "kind": "mavis_f1", "shards": first, "weight": 1.0, "manifest": str(folder),
              "manifest_sha256": sha, "split": split}
             for name, folder, sha in (("c9", args.new, args.new_sha256), ("c7", args.manifest, args.manifest_sha256))
             for split in ("train", "dev")]
    sources = dict(zip((s["name"] for s in specs), make_sources(specs)))
    raw = {"c9": rs.raw_records(Path(args.new), args.new_sha256), "c7": rs.raw_records(Path(args.manifest),
                                                                                      args.manifest_sha256)}
    rel = str(Path(first).resolve().relative_to(ROOT))
    checks, out = [], {}

    def ok(name, cond, detail=None):
        checks.append({"controllo": name, "esito": bool(cond), "dettaglio": detail})
        print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail is not None else ""),
              flush=True)

    for split in ("train", "dev"):
        got = {}
        for name in ("c9", "c7"):
            s = sources[f"{name}/{split}"]
            recs = raw[name][rel]
            want = [k for k in sorted(recs, key=lambda k: recs[k][6]) if rs.entering(recs[k], "mavis_f1", split) is None]
            res = rs.read_pass(s, set(), 10 ** 9)
            ok(f"{name} {split}: chiavi consegnate uguali ai record ammessi dei record grezzi, nell'ordine dello shard",
               res["keys"] == want and not res["counts"].get("errori_manifest"), (len(res["keys"]), len(want)))
            got[name] = res
        bad, blocked9, blocked7, tasks, refused = Counter(), Counter(), Counter(), Counter(), 0
        for key in got["c9"]["keys"]:
            f9, f7 = got["c9"]["files"][key], got["c7"]["files"].get(key)
            s9 = dt.SAMPLE_LOADERS["mavis_f1"](key, f9)
            adm = dt.admission_of(f9)
            n_shape = next((t[2] for t in raw["c9"][rel][key][3] if t[0] == "shape_type"), 0)
            tasks.update(s9.tasks)
            blocked9.update(s9.blocked or {})
            if adm.get("costruzione_qa") != c9.CONSTRUCTION:
                bad["costruzione non dichiarata nella decisione allegata"] += 1
            if "shape_type" in s9.tasks or not set(s9.tasks) <= set(adm["compiti_ammessi"]):
                bad["compiti fuori dagli ammessi o shape_type"] += 1
            expect = 0 if s9.tasks == ("describe",) else n_shape  # la descrizione non tocca il pool delle domande
            if (s9.blocked or {}).get("shape_type", 0) != expect:
                bad["domande shape_type C9 bloccate diverse dal pool"] += 1
            if f7 is None:
                bad["record consegnato col C9 ma non col genitore"] += 1
                continue
            s7 = dt.SAMPLE_LOADERS["mavis_f1"](key, f7)
            blocked7.update(s7.blocked or {})
            if s9.messages != s7.messages or s9.tasks != s7.tasks:
                bad["messaggi o compiti diversi dal genitore"] += 1
            stripped = {**f9, mf.ADMISSION_FILE: mf.canonical({k: v for k, v in adm.items()
                                                                if k != "costruzione_qa"}).encode()}
            try:
                dt.SAMPLE_LOADERS["mavis_f1"](key, stripped)
            except mf.ManifestError:
                refused += 1
        with_shape = sum(1 for k in got["c9"]["keys"]
                         if any(t[0] == "shape_type" for t in raw["c7"][rel][k][3]))
        ok(f"{split}: campioni costruiti col manifest C9, compiti solo fra gli ammessi, shape_type mai, stessi messaggi "
           "del genitore", not bad, dict(bad))
        ok(f"{split}: senza la costruzione dichiarata il pool C9 è rifiutato dal digest (record con shape_type storiche)",
           refused == with_shape, {"rifiutati": refused, "record_con_shape_type_storiche": with_shape})
        out[split] = {"consegnati": len(got["c9"]["keys"]), "compiti_emessi": dict(sorted(tasks.items())),
                      "bloccate_c9": dict(blocked9), "bloccate_genitore": dict(blocked7),
                      "contatori_reader_c9": got["c9"]["counts"], "secondi": round(time.time() - t0, 1)}
    passed = sum(x["esito"] for x in checks)
    result = {"schema": "mavis-c9-lettore/1", "job_id": os.environ.get("SLURM_JOB_ID"), "codice": code_identity(),
              "shard": Path(first).name, "c9": {"manifest": str(args.new), "manifest_sha256": args.new_sha256},
              "genitore": {"manifest": str(args.manifest), "manifest_sha256": args.manifest_sha256},
              "controlli": checks, "superati": passed, "totale": len(checks), "esito": passed == len(checks),
              "per_split": out, "secondi": round(time.time() - t0, 1), "rss_max_mib": um.rss_mib(),
              "nota": "lettura di un solo shard: collega il candidato C9 al reader, non verifica il flusso intero"}
    args.out.mkdir(parents=True, exist_ok=True)
    write_new_json(args.out / "lettore.json", result)
    print(json.dumps({"lettore": f"{passed}/{len(checks)}", "secondi": result["secondi"]}))
    return 0 if result["esito"] else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sonda")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    for name in ("costruzione", "verifica"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True, help="genitore (mavis-geometry-candidato-3)")
        p.add_argument("--manifest-sha256", required=True)
        p.add_argument("--new", type=Path, required=True, help="cartella nuova del candidato C9")
        p.add_argument("--out", type=Path, required=True, help="cartella della scansione (report.json)")
        p.add_argument("--report", type=Path, required=True, help="cartella della verifica")
        if name == "costruzione":
            p.add_argument("--index", type=Path, required=True, help="indice del pannello (pannello-v3)")
            p.add_argument("--index-sha256", required=True)
            p.add_argument("--summary", type=Path, required=True)
            p.add_argument("--version", required=True)
            p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("lettore")
    p.add_argument("--manifest", type=Path, required=True, help="genitore (C7)")
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--new", type=Path, required=True, help="candidato C9")
    p.add_argument("--new-sha256", required=True)
    p.add_argument("--shards", required=True, help="glob degli shard; si legge il primo in ordine di nome")
    p.add_argument("--out", type=Path, required=True)
    return ap


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("va eseguito in un job SLURM")
    args = parser().parse_args()
    try:
        sys.exit({"sonda": cmd_sonda, "costruzione": cmd_costruzione, "verifica": cmd_verifica,
                  "lettore": cmd_lettore}[args.cmd](args))
    except Failed as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
