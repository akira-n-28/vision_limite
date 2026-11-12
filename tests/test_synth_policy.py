#!/usr/bin/env python3
"""Test CPU della politica delle domande dei sintetici e del generatore v5 (S2 §2.14). Solo in un job
(slurm/synth_qa_policy.sbatch, slurm/tests_cpu.sbatch).

Controlla:
- vecchie versioni: specifica e domande dei primi record degli shard v4 (train, test, coppie) e v3 riprodotte da
  synth.make_spec e synth.questions; il default del generatore resta 4;
- v5: nomi dei grafici da FUNCTION_NAMES dopo la geometria (uguale a v4), una sola exists marcata da_sola con lo stesso
  vocabolario per sì e no, ogni etichetta con sì e con no, intersezioni coi nomi disegnati, altre famiglie uguali a v4;
  pick_synth_questions uguale a pick_questions senza domande da sole, e una domanda da sola mai insieme ad altre;
  synth_check sul layout di scene v5 disegnate (exists e intersezioni coi nomi nuovi; domanda coi nomi sbagliati non
  verificabile);
- manifest e loader: un manifest dei sintetici senza politica si legge per record come prima (stessa scelta di domande,
  exists compresa); con politica_domande è per compito: manifest_sha256 obbligatorio, liste della config rifiutate,
  decisione allegata dal reader, function/exists tolta prima della scelta (prompt e target senza la domanda, scelta
  uguale a pick_questions sul pool filtrato, compiti e bloccati contati), altre famiglie con exists invariata, coppia
  con la sua domanda, record senza compiti ammessi saltato col motivo, NoAdmittedTask dal loader senza ripiego, digest
  delle domande e famiglia legati al manifest, versione della politica sconosciuta rifiutata; liste storiche non
  richieste col manifest con politica;
- synth_qa_policy: regola della revisione (expected_review_change), review_checks e policy_checks su derivazioni
  corrette e alterate, strumenti della misura (etichette, menzioni, tabelle, contesto)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import synth_qa_policy as sqp  # noqa: E402
from limite_vl import data as dt  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl import synth  # noqa: E402
from limite_vl.data_lists import missing_required  # noqa: E402
from limite_vl.f1c import pick_questions, qa_messages  # noqa: E402

V3, V4 = ROOT / "data/shards/synth/v3", ROOT / "data/shards/synth/v4"
DONE, FAILS = [], []


def check(name, cond, detail=""):
    DONE.append(name)
    print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:300]})" if detail != "" and not cond else ""),
          flush=True)
    if not cond:
        FAILS.append(name)


def raises(fn, exc=mf.ManifestError) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception as e:  # noqa: BLE001
        print("    eccezione diversa:", type(e).__name__, e)
        return False
    return False


# ---------------------------------------------------------------- generatore

def test_old_versions() -> None:
    check("generatore: la versione di default resta 4 e v5 è fra le versioni esplicite",
          synth.GENERATOR_VERSION == 4 and synth.VERSIONS == (3, 4, 5))
    for folder, version, tar, n in ((V4, 4, "train/synth-train-00000.tar", 400), (V4, 4, "test/synth-test.tar", 60),
                                    (V3, 3, "train/synth-train-00000.tar", 200)):
        same, total = 0, 0
        for key, files in dt.iter_shard(folder / tar):
            meta = json.loads(files["json"])
            spec = synth.make_spec(key, meta["family"], version)
            same += json.loads(json.dumps(spec)) == meta["spec"] and synth.questions(spec) == meta["qa"]
            total += 1
            if total >= n:
                break
        check(f"v{version}: specifica e domande dei primi {n} record di {tar} riprodotte dal generatore", same == total == n,
              (same, total))
    pairs = 0
    for key, files in dt.iter_shard(V4 / "pairs/synth-pairs.tar"):
        meta = json.loads(files["json"])
        pairs += meta["qa"] == [] and meta["pair"]["type"] in ("side_x", "side_y", "opens", "slope", "side_len", "count")
        if pairs >= 40:
            break
    check("v4: le domande delle coppie non sono mai exists (side_x, side_y, opens, slope, side_len, count)", pairs == 40)


def test_v5_generator() -> None:
    keys = [f"test-v5-{i:05d}" for i in range(3000)]
    cells, geometry, single, names_ok, inter = Counter(), 0, 0, 0, 0
    for key in keys:
        s4, s5 = synth.make_spec(key, "function", 4), synth.make_spec(key, "function", 5)
        q5 = synth.questions(s5)
        if sqp._strip(s4) == sqp._strip(s5):
            geometry += 1
        ex = [q for q in q5 if q["type"] == "exists"]
        if len(ex) == 1 and ex[0].get(synth.SOLO_FIELD) is True:
            single += 1
            cells[(sqp.asked_label(ex[0]), ex[0]["answer"])] += 1
        names = [c["name"] for c in s5["curves"]]
        names_ok += len(set(names)) == len(names) and set(names) <= set(synth.FUNCTION_NAMES)
        for q in q5:
            if q["type"] == "intersections":
                inter += 1
                a, b = sorted(names)
                if q["q"] != f"How many times do the graphs of {a} and {b} intersect in the figure?":
                    names_ok -= 1
    check("v5: geometria e stile uguali a v4 alla stessa chiave, a parte i nomi", geometry == len(keys), geometry)
    check("v5: una sola exists per scena, marcata da_sola", single == len(keys), single)
    check("v5: nomi dei grafici distinti, da FUNCTION_NAMES; intersezioni coi nomi dei due grafici",
          names_ok == len(keys) and inter > 0, (names_ok, inter))
    both = [n for n in synth.FUNCTION_NAMES if cells[(n, "yes")] and cells[(n, "no")]]
    check("v5: ogni etichetta compare con yes e con no (stesso vocabolario)", both == list(synth.FUNCTION_NAMES),
          dict(cells))
    yes = sum(v for (n, a), v in cells.items() if a == "yes")
    check("v5: sì e no bilanciati entro il 5% su 3000 scene", abs(yes / len(keys) - 0.5) < 0.05, yes)
    fam_same = all(synth.make_spec(k, f, 4) | {"generator_version": 5} == synth.make_spec(k, f, 5)
                   and synth.questions(synth.make_spec(k, f, 4)) == synth.questions(synth.make_spec(k, f, 5))
                   for f in ("coord", "polygon", "shapes") for k in keys[:150])
    check("v5: coord, polygon e shapes uguali a v4 (solo generator_version cambia)", fam_same)
    v4_rule = [(sqp.asked_label(q), q["answer"]) for k in keys[:500]
               for q in synth.questions(synth.make_spec(k, "function", 4)) if q["type"] == "exists"]
    check("v4: la regola testuale f, g -> yes; h, p, q -> no dà tutte le risposte (la scorciatoia c'è)",
          all(sqp.text_rule(lab) == a for lab, a in v4_rule) and len(v4_rule) == 1000)


def test_pick() -> None:
    import random

    rng = random.Random(0)
    same = True
    for i in range(400):
        items = [{"type": "t", "q": f"q{j}", "options": ["a"], "answer": "a"} for j in range(rng.randint(1, 9))]
        same &= dt.pick_synth_questions(f"k{i}", items) == pick_questions(f"k{i}", items)
    check("pick_synth_questions senza domande da sole = f1c.pick_questions", same)
    alone_ok, seen_alone, seen_multi = True, 0, 0
    for i in range(3000):
        items = [{"type": "t", "q": f"q{j}", "options": ["a"], "answer": "a"} for j in range(4)]
        items.append({"type": "exists", "q": "Is there a graph labeled h?", "options": ["yes", "no"], "answer": "no",
                      synth.SOLO_FIELD: True})
        picked = dt.pick_synth_questions(f"s{i}", items)
        if any(q.get(synth.SOLO_FIELD) for q in picked):
            seen_alone += 1
            alone_ok &= len(picked) == 1
        else:
            seen_multi += len(picked) > 1
    check("pick_synth_questions: la domanda da sola entra solo da sola (e qualche volta entra)",
          alone_ok and seen_alone > 0 and seen_multi > 0, (seen_alone, seen_multi))
    only = [{"type": "exists", "q": "Is there a graph labeled f?", "options": ["yes", "no"], "answer": "yes",
             synth.SOLO_FIELD: True}]
    check("pick_synth_questions: con le sole domande da sole si sceglie una domanda",
          all(len(dt.pick_synth_questions(f"o{i}", only)) == 1 for i in range(50)))


def test_v5_rendered() -> None:
    from limite_vl import synth_check

    found, ok, inter_ok, wrong_named = 0, 0, 0, 0
    for i in range(400):
        key = f"test-v5-render-{i:04d}"
        spec = synth.make_spec(key, "function", 5)
        if len(spec["curves"]) != 2 or set(c["name"] for c in spec["curves"]) == {"f", "g"}:
            continue
        qs = synth.questions(spec)
        if not any(q["type"] == "intersections" for q in qs):
            continue
        image, layout = synth.render(spec, layout=True)
        out = synth_check.check_questions(spec, qs, layout)
        found += 1
        ok += all(r["status"] == "ok" for r in out if r["type"] == "exists")
        inter_ok += all(r["status"] in ("ok", "non_ammissibile") for r in out if r["type"] == "intersections")
        bad = [{**q, "q": "How many times do the graphs of f and g intersect in the figure?"} for q in qs
               if q["type"] == "intersections"]
        res = synth_check.check_questions(spec, bad, layout)
        wrong_named += all(r["status"] == "non_verificabile" for r in res)
        if found >= 6:
            break
    check("v5 disegnate: exists verificata dal layout e intersezioni coi nomi nuovi accettate dal checker",
          found >= 6 and ok == found and inter_ok == found, (found, ok, inter_ok))
    check("checker: una domanda sulle intersezioni che non nomina i grafici disegnati non è verificabile",
          wrong_named == found, wrong_named)


# ---------------------------------------------------------------- manifest e loader

def png(color: int = 255, size: int = 448) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), (color, color, color)).save(buf, format="PNG")
    return buf.getvalue()


def q(t, text, options, answer):
    return {"type": t, "q": text, "options": options, "answer": answer}


YN = ["yes", "no"]
RECORDS = {  # chiave -> (insieme, split, stato, famiglia, QA, domanda della coppia)
    "f1": ("train", "train", "ammesso", "function", [
        q("opens", "Does the graph of f open upward or downward?", ["upward", "downward"], "upward"),
        q("intercept", "Does the graph of f cross the y-axis above or below the origin?", ["above", "below"], "above"),
        q("exists", "Is there a graph labeled f?", YN, "yes"), q("exists", "Is there a graph labeled h?", YN, "no")], None),
    "f2": ("train", "train", "ammesso", "function", [
        q("slope", "Is the graph of f increasing or decreasing?", ["increasing", "decreasing"], "increasing"),
        q("intercept", "Does the graph of f cross the y-axis above or below the origin?", ["above", "below"], "below"),
        q("slope", "Is the graph of g increasing or decreasing?", ["increasing", "decreasing"], "decreasing"),
        q("intersections", "How many times do the graphs of f and g intersect in the figure?", ["0", "1", "2"], "1"),
        q("exists", "Is there a graph labeled g?", YN, "yes"), q("exists", "Is there a graph labeled p?", YN, "no")], None),
    "s1": ("train", "train", "ammesso", "shapes", [
        q("count", "How many shapes are in the figure?", ["1", "2", "3"], "2"),
        q("exists", "Is there a shape labeled A?", YN, "yes"), q("exists", "Is there a shape labeled K?", YN, "no"),
        q("has_type", "Is there a circle in the figure?", YN, "yes")], None),
    "e1": ("train", "train", "ammesso", "function", [
        q("exists", "Is there a graph labeled f?", YN, "yes"), q("exists", "Is there a graph labeled q?", YN, "no")], None),
    "d1": ("train", "dev", "ammesso", "function", [
        q("opens", "Does the graph of f open upward or downward?", ["upward", "downward"], "downward"),
        q("exists", "Is there a graph labeled f?", YN, "yes"), q("exists", "Is there a graph labeled h?", YN, "no")], None),
    "x1": ("train", "test", "escluso", "function", [
        q("opens", "Does the graph of f open upward or downward?", ["upward", "downward"], "upward"),
        q("exists", "Is there a graph labeled f?", YN, "yes")], None),
    "t1": ("test", "test", "ammesso", "coord", [
        q("quadrant", "In which quadrant is point A?", ["first", "second", "third", "fourth"], "first"),
        q("exists", "Is there a point labeled A?", YN, "yes"), q("exists", "Is there a point labeled B?", YN, "no")], None),
    "p1-a": ("coppie", "test", "ammesso", "function", [],
             {"id": "p1", "variant": "a", "q": "Does the graph of f open upward or downward?", "type": "opens",
              "options": ["upward", "downward"], "answer": "upward"}),
}


def write_tar(path: Path, items: list[tuple[str, bytes, bytes]]) -> None:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for key, js, image in items:
            for name, data in ((f"{key}.png", image), (f"{key}.json", js)):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))


def build(tmp: Path, policy: bool, records=RECORDS, version: str = sqp.POLICY_VERSION) -> dict:
    image = png()
    by_set = {"train": [], "test": [], "coppie": []}
    for k, v in records.items():
        by_set[v[0]].append(k)
    digests = mf.Digests()
    shards, lines, blobs, info = [], [], {}, {}
    for insieme, ks in by_set.items():
        items = []
        for k in ks:
            ins, split, state, fam, qa, pair = records[k]
            meta = {"id": k, "family": fam, "width": 448, "height": 448, "max_patches": 784, "qa": qa, "spec": {}}
            if pair:
                meta["pair"] = pair
            blobs[k] = json.dumps(meta).encode()
            items.append((k, blobs[k], image))
        path = tmp / f"{insieme}.tar"
        write_tar(path, items)
        shards.append({"insieme": insieme, "nome": path.name, "percorso": str(path),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                       "record": len(ks)})
    for s, (insieme, ks) in zip(shards, by_set.items()):
        for index, k in enumerate(ks):
            ins, split, state, fam, qa, pair = records[k]
            pool = list(qa) + ([pair] if pair else [])
            rec = {"record_id": mf.record_id(digests, "synth", "v4@" + "0" * 40, k), "sorgente": "synth",
                   "revisione": "v4@" + "0" * 40, "record": k, "insieme": ins, "shard": s["percorso"], "indice": index,
                   "famiglia": fam, "coppia": pair["id"] if pair else None, "gruppo": mf.sha256_text("g" + k),
                   "split": split, "stato": state, "motivo": {} if state == "ammesso" else {"contaminazione": ["x"]},
                   "gold": None, "immagini": [{"sha256": hashlib.sha256(image).hexdigest(), "ordine": 0}],
                   "json_sha256": hashlib.sha256(blobs[k]).hexdigest(),
                   "qa": [{"tipo": x["type"], "sha256": mf.qa_digest(digests, x)} for x in pool]}
            if policy:
                rec["compiti"] = sqp.compiti_of(rec)
            lines.append(json.dumps(rec, sort_keys=True))
            info[k] = rec
    folder = tmp / ("manifest-politica" if policy else "manifest")
    folder.mkdir()
    data = gzip.compress("".join(x + "\n" for x in lines).encode(), mtime=0)
    (folder / "records.jsonl.gz").write_bytes(data)
    meta = {"schema": mf.SCHEMA, "sorgente": "synth", "revisione": "v4@" + "0" * 40, "shards": shards, "derivati": {},
            "records": {"file": "records.jsonl.gz", "sha256": hashlib.sha256(data).hexdigest(), "righe": len(lines)}}
    if policy:
        meta[mf.POLICY_FIELD] = {"versione": version, "compiti_bloccati": {t: {"stato": "escluso"} for t in sqp.BLOCKED}}
    (folder / "manifest.json").write_text(json.dumps(meta))
    return {"folder": folder, "records": info, "tmp": tmp, "sha": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()}


def read_all(source) -> dict[str, dict]:
    from limite_vl.stream import SourceReader

    out = {}
    reader = SourceReader(source, seed=0, worker=0, n_workers=1, val_count=0)
    while True:
        key, files = reader.next()
        if reader.epoch > 0:
            break
        out[key] = files
    return out, dict(reader.counts)


def test_manifest_and_loader(tmp: Path) -> None:
    from limite_vl.stream import make_sources

    old = build(tmp / "senza", False) if (tmp / "senza").mkdir() is None else None
    new = build(tmp / "con", True) if (tmp / "con").mkdir() is None else None
    spec = lambda f, split, shards, **kw: {"name": "s", "kind": "synth_qa", "shards": shards, "weight": 1.0,  # noqa: E731
                                           "manifest": str(f["folder"]), "split": split, **kw}
    # senza politica: per record, come prima
    src = make_sources([spec(old, "train", str(old["tmp"] / "train.tar"))])[0]
    files, counts = read_all(src)
    check("senza politica: sorgente per record, nessuna decisione allegata", not src.admission.tasked
          and all(mf.ADMISSION_FILE not in f for f in files.values()) and sorted(files) == ["e1", "f1", "f2", "s1"],
          sorted(files))
    same = all(dt.synth_qa_sample(k, f).messages[1]["content"]
               == qa_messages(pick_questions(k, json.loads(f["json"])["qa"]))[1] for k, f in files.items())
    check("senza politica: la scelta delle domande è quella di prima (exists compresa)", same)
    fake = "data/shards/synth/v4/train/*.tar"  # missing_required guarda solo la cartella e il manifest
    check("senza politica: la lista storica dei sintetici resta obbligatoria",
          bool(missing_required([spec(old, "train", fake)])))
    # con politica: per compito
    check("con politica: manifest non identificato rifiutato",
          raises(lambda: make_sources([spec(new, "train", str(new["tmp"] / "train.tar"))])))
    lst = tmp / "lista.txt"
    lst.write_text("f1\n")
    check("con politica: liste exclude della config rifiutate", raises(lambda: make_sources(
        [spec(new, "train", str(new["tmp"] / "train.tar"), manifest_sha256=new["sha"], exclude=[str(lst)])])))
    check("con politica: nessuna lista storica richiesta",
          not missing_required([spec(new, "train", fake, manifest_sha256=new["sha"])]))
    src = make_sources([spec(new, "train", str(new["tmp"] / "train.tar"), manifest_sha256=new["sha"])])[0]
    files, counts = read_all(src)
    check("con politica: sorgente per compito, record senza compiti ammessi saltato col motivo",
          src.admission.tasked and sorted(files) == ["f1", "f2", "s1"]
          and counts.get("saltati_manifest/nessun_compito_ammesso") == 1, (sorted(files), counts))
    leak, same, tasks_ok = [], True, True
    for k, f in files.items():
        sample = dt.synth_qa_sample(k, f)
        prompt, target = sample.messages[0]["content"][1]["text"], sample.messages[1]["content"]
        meta = json.loads(f["json"])
        fam = meta["family"]
        kept = [x for x in meta["qa"] if f"{fam}/{x['type']}" != "function/exists"]
        same &= (prompt, target) == qa_messages(pick_questions(k, kept))
        if fam == "function":
            leak += [k for x in meta["qa"] if x["type"] == "exists" and x["q"] in prompt]
            tasks_ok &= sample.blocked == {"function/exists": 2} and all(t.startswith("function/") and t !=
                                                                          "function/exists" for t in sample.tasks)
        else:
            tasks_ok &= sample.blocked == {}
    check("con politica: function/exists tolta prima della scelta; prompt e target = pick_questions sul pool filtrato",
          same and not leak, leak)
    check("con politica: compiti svolti e domande bloccate contati (function/exists: 2 per record function)", tasks_ok)
    s1 = json.loads(files["s1"]["json"])
    check("con politica: le exists delle altre famiglie restano nel pool",
          all(x in s1["qa"] for x in s1["qa"] if x["type"] == "exists")
          and sorted(json.loads(files["s1"][mf.ADMISSION_FILE])["compiti_ammessi"]) ==
          ["shapes/count", "shapes/exists", "shapes/has_type"])
    # digest e famiglia legati al manifest, nessun ripiego
    f1 = files["f1"]
    adm = json.loads(f1[mf.ADMISSION_FILE])
    altered = {**f1, mf.ADMISSION_FILE: mf.canonical({**adm, "qa_sha256": "0" * 64}).encode()}
    check("loader: domande diverse dal digest del manifest rifiutate", raises(lambda: dt.synth_qa_sample("f1", altered)))
    other = {**f1, mf.ADMISSION_FILE: mf.canonical({**adm, "famiglia": "shapes"}).encode()}
    check("loader: famiglia diversa da quella del manifest rifiutata", raises(lambda: dt.synth_qa_sample("f1", other)))
    none = {**f1, mf.ADMISSION_FILE: mf.canonical({**adm, "compiti_ammessi": ["function/exists"][:0]}).encode()}
    check("loader: senza compiti ammessi NoAdmittedTask, nessun ripiego sulle domande bloccate",
          raises(lambda: dt.synth_qa_sample("f1", none), dt.NoAdmittedTask))
    # dev, test e coppie
    dev = make_sources([spec(new, "dev", str(new["tmp"] / "train.tar"), manifest_sha256=new["sha"])])[0]
    dfiles, _ = read_all(dev)
    check("con politica: dev = solo il record di dev, senza exists nel prompt",
          sorted(dfiles) == ["d1"] and "labeled" not in dt.synth_qa_sample("d1", dfiles["d1"]).messages[0]["content"][1]["text"])
    test = make_sources([spec(new, "test", str(new["tmp"] / "test.tar"), manifest_sha256=new["sha"])])[0]
    tfiles, _ = read_all(test)
    t1 = dt.synth_qa_sample("t1", tfiles["t1"])
    check("con politica: test coord con le sue domande (exists dei punti ammessa)", sorted(tfiles) == ["t1"]
          and t1.blocked == {} and all(t.startswith("coord/") for t in t1.tasks))
    pairs = make_sources([spec(new, "test", str(new["tmp"] / "coppie.tar"), manifest_sha256=new["sha"])])[0]
    pfiles, _ = read_all(pairs)
    p = dt.synth_qa_sample("p1-a", pfiles["p1-a"])
    check("con politica: la coppia si costruisce con la sua domanda (opens), non è tolta perché function",
          p.tasks == ("function/opens",) and p.messages[1]["content"] == "upward")
    check("senza politica: la coppia senza QA non ha domande per il loader (come prima)",
          raises(lambda: dt.synth_qa_sample("p1-a", {k: v for k, v in pfiles["p1-a"].items() if k != mf.ADMISSION_FILE}),
                 ValueError))
    (tmp / "ignota").mkdir()
    bad = build(tmp / "ignota", True, version="synth-domande/0")
    check("politica di versione sconosciuta rifiutata", raises(lambda: make_sources(
        [spec(bad, "train", str(bad["tmp"] / "train.tar"), manifest_sha256=bad["sha"])])))
    m = mf.load(new["folder"])
    check("tasked_source: synth_qa sul manifest con politica è synth; un altro tipo è rifiutato",
          m.tasked_source("synth_qa") == "synth" and raises(lambda: m.tasked_source("mavis_f1")))
    check("tasked_source: synth_qa sul manifest senza politica resta per record",
          mf.load(old["folder"]).tasked_source("synth_qa") is None)


# ---------------------------------------------------------------- synth_qa_policy

def row(record, categoria, stato, motivi, rimovibili="", blocchi="", proposto=None, dopo=None, **kw):
    base = {c: "" for c in sqp.PROPOSAL_COLUMNS}
    base.update(record=record, categoria=categoria, stato_attuale=stato, motivi=motivi, motivi_rimovibili=rimovibili,
                blocchi_rimanenti=blocchi, stato_proposto=proposto or stato, stato_dopo_riesame_tecnico=dopo or proposto
                or stato, **kw)
    return base


def test_review_rule(tmp: Path) -> None:
    dec = {"decontaminazione_pannello": ["candidato"]}
    p = {"stato": "quarantena", "motivo": dec}
    check("regola: riammissione proposta toglie il motivo risolto e ammette",
          sqp.expected_review_change(p, row("a", sqp.READMIT, "quarantena", "decontaminazione_pannello=risolto_dalla_revisione",
                                            "decontaminazione_pannello", proposto="ammesso"), False)
          == ("ammesso", {}, dec))
    tech = row("b", sqp.READMIT_TECH, "quarantena", "decontaminazione_pannello=risolvibile_col_riesame_tecnico",
               blocchi="decontaminazione_pannello", proposto="quarantena", dopo="ammesso")
    check("regola: dopo il riesame tecnico ammette solo se il riesame è accettato",
          sqp.expected_review_change(p, tech, True) == ("ammesso", {}, dec)
          and sqp.expected_review_change(p, tech, False) == ("quarantena", dec, {}))
    both = {"stato": "escluso", "motivo": {"contaminazione": ["c"], **dec}}
    blocked = row("c", "resta_bloccato", "escluso", "contaminazione=valido; decontaminazione_pannello=risolto_dalla_revisione",
                  "decontaminazione_pannello", "contaminazione")
    check("regola: resta bloccato toglie il motivo risolto e resta escluso per la contaminazione",
          sqp.expected_review_change(both, blocked, True) == ("escluso", {"contaminazione": ["c"]}, dec))
    unres = row("d", "irrisolto", "quarantena", "decontaminazione_pannello=irrisolto", blocchi="decontaminazione_pannello")
    check("regola: irrisolto e record senza riga invariati",
          sqp.expected_review_change(p, unres, True) == ("quarantena", dec, {})
          and sqp.expected_review_change(p, None, True) == ("quarantena", dec, {}))
    # review_checks su una derivazione piccola
    parent = [{"record": "a", "record_id": "ra", "insieme": "train", "split": "train", "gruppo": "ga", "famiglia": "shapes",
               "stato": "quarantena", "motivo": dec, "evidenze": [{"controllo": "x"}], "qa": [{"tipo": "count", "sha256": "1"}],
               "immagini": [], "json_sha256": "j", "stato_scena_sha256": "s", "coppia": None, "shard": "x.tar", "indice": 0},
              {"record": "z", "record_id": "rz", "insieme": "train", "split": "train", "gruppo": "gz", "famiglia": "shapes",
               "stato": "quarantena", "motivo": dec, "evidenze": [], "qa": [], "immagini": [], "json_sha256": "j2",
               "stato_scena_sha256": "s2", "coppia": None, "shard": "x.tar", "indice": 1}]
    prop = tmp / "proposta.tsv"
    r = row("a", sqp.READMIT, "quarantena", "decontaminazione_pannello=risolto_dalla_revisione",
            "decontaminazione_pannello", proposto="ammesso", record_id="ra", insieme="train", split="train", gruppo="ga",
            famiglia="shapes")
    prop.write_text("\t".join(sqp.PROPOSAL_COLUMNS) + "\n" + "\t".join(r[c] for c in sqp.PROPOSAL_COLUMNS) + "\n")
    tech_file = tmp / "riesame.tsv"
    tech_file.write_text("unita\trecord\tconclusione\tmotivazione\n")
    meta = {"revisione": "v4@" + "0" * 40,  # la revisione della sorgente, come nei manifest veri
            sqp.REVIEW_FIELD: {"controllo": sqp.REVIEW,
                               "proposta": {"percorso": str(prop), "sha256": hashlib.sha256(prop.read_bytes()).hexdigest()},
                               "riesame_tecnico": {"percorso": str(tech_file),
                                                   "sha256": hashlib.sha256(tech_file.read_bytes()).hexdigest(),
                                                   "accettato": True}}}
    good = [dict(parent[0], stato="ammesso", motivo={}, evidenze=parent[0]["evidenze"] + [
        {"controllo": sqp.REVIEW, "categoria": sqp.READMIT, "motivi_risolti": dec, "stato": "ammesso",
         "stato_genitore": "quarantena"}]), dict(parent[1])]

    def run(records):
        results = []
        sqp.review_checks(meta, records, parent, lambda name, ok, detail=None: results.append((name, bool(ok))))
        return [n for n, ok in results if not ok]

    check("review_checks: derivazione corretta senza errori", run(good) == [], run(good))
    stray = [good[0], dict(parent[1], stato="ammesso", motivo={})]
    check("review_checks: un record riammesso senza riga della proposta è un errore", bool(run(stray)))
    no_ev = [dict(good[0], evidenze=parent[0]["evidenze"]), good[1]]
    check("review_checks: una riammissione senza l'evidenza della revisione è un errore", bool(run(no_ev)))
    qa = [dict(good[0], qa=[]), good[1]]
    check("review_checks: QA cambiate sono un errore", bool(run(qa)))
    # manifest_check: la revisione della sorgente (stringa "revisione") non è una derivazione dalla revisione umana
    import manifest_check as mc

    plain = {"revisione": "v4@" + "0" * 40}
    results = []
    mc.parent_checks(plain, [dict(x) for x in parent], parent, lambda n, ok, d=None: results.append(bool(ok)))
    check("manifest_check: un manifest con la sola revisione della sorgente si verifica come prima (nessun "
          "miglioramento ammesso, nessuna revisione umana)", results and all(results)
          and not mc.justified(plain, good[0]) and mc.justified(meta, good[0]))
    results = []
    mc.parent_checks(meta, good, parent, lambda n, ok, d=None: results.append(bool(ok)))
    check("manifest_check: la derivazione dalla revisione umana passa da review_checks", results and all(results))
    meta[sqp.REVIEW_FIELD]["proposta"]["sha256"] = "0" * 64
    check("review_checks: proposta con sha256 diverso da quello registrato", bool(run(good)))


def test_policy_and_tools() -> None:
    rec = {"record": "f", "famiglia": "function", "stato": "ammesso",
           "qa": [{"tipo": "opens", "sha256": "1"}, {"tipo": "exists", "sha256": "2"}, {"tipo": "exists", "sha256": "3"}]}
    rec["compiti"] = sqp.compiti_of(rec)
    check("compiti_of: un compito per famiglia e tipo, function/exists escluso col numero di domande",
          rec["compiti"] == [{"compito": "function/exists", "stato": "escluso", "domande": 2, "motivo": "scorciatoia_testuale"},
                             {"compito": "function/opens", "stato": "ammesso", "domande": 1}])
    res = []
    meta = {mf.POLICY_FIELD: {"versione": sqp.POLICY_VERSION, "compiti_bloccati": {"function/exists": {}}}}
    sqp.policy_checks(meta, [rec], lambda n, ok, d=None: res.append(ok))
    altered = dict(rec, compiti=[dict(c, stato="ammesso") for c in rec["compiti"]])
    res2 = []
    sqp.policy_checks(meta, [altered], lambda n, ok, d=None: res2.append(ok))
    check("policy_checks: compiti corretti passano, function/exists dichiarata ammessa no", all(res) and not all(res2))
    check("asked_label e text_rule", sqp.asked_label({"type": "exists", "q": "Is there a graph labeled h?"}) == "h"
          and sqp.asked_label({"type": "has_type", "q": "Is there a circle in the figure?"}) == "circle"
          and sqp.text_rule("g") == "yes" and sqp.text_rule("p") == "no")
    check("mentioned: punti nelle parole maiuscole, grafici come lettere minuscole",
          sqp.mentioned("A", "polygon", ["What is the length of side AB?"])
          and not sqp.mentioned("K", "coord", ["In which quadrant is point A?"])
          and sqp.mentioned("g", "function", ["Is the graph of g increasing or decreasing?"])
          and not sqp.mentioned("h", "function", ["Does the graph of f open upward or downward?"]))
    t = sqp.label_table([("f", "yes"), ("f", "yes"), ("h", "no"), ("A", "yes"), ("A", "no")])
    check("label_table: vocabolari e regola per etichetta", t["solo_si"] == ["f"] and t["solo_no"] == ["h"]
          and t["comuni"] == ["A"] and t["regola_per_etichetta"]["corrette"] == 4)
    picked = [{"type": "quadrant", "q": "In which quadrant is point A?"},
              {"type": "exists", "q": "Is there a point labeled A?", "answer": "yes"},
              {"type": "exists", "q": "Is there a point labeled K?", "answer": "no"}]
    st = sqp.presence_stats([("coord", picked, 1), ("coord", picked, 2)])["coord/exists"]
    check("presence_stats: etichetta nominata da un'altra domanda, regola del contesto, coppia sì/no nel prompt",
          st["etichetta_nominata_da_altre/yes"] == 1 and st["etichetta_nominata_da_altre/no"] == 0
          and st["regola_contesto_corrette"] == 2 and st["con_l_altra_exists_nello_stesso_prompt"] == 2)


def main() -> None:
    test_old_versions()
    test_v5_generator()
    test_pick()
    test_v5_rendered()
    tmp = Path(tempfile.mkdtemp(prefix="synth_policy_"))
    try:
        test_manifest_and_loader(tmp)
        (tmp / "regola").mkdir()
        test_review_rule(tmp / "regola")
        test_policy_and_tools()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\ntest_synth_policy: {len(DONE) - len(FAILS)} controlli superati su {len(DONE)}"
          + (f", FALLITI: {len(FAILS)}" if FAILS else ""))
    if FAILS:
        for f in FAILS:
            print(" -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
