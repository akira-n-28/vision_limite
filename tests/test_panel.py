#!/usr/bin/env python3
"""Test CPU del pannello (scripts/panel_index.py, scripts/panel_groups.py) e della convenzione delle immagini
(limite_vl.decontam, scripts/decontam.py, scripts/decontam_review.py), su dati piccoli costruiti qui. Solo in un job
(slurm/tests_cpu.sbatch).

Coprono: irrisolto su un membro di una coppia o di un documento; problemi diversi per operatore, disuguaglianza o
segno; vicinati pHash asimmetrici dopo il limite e vicini dello stesso gruppo; PNG trasparenti fra indice e scansione;
recupero delle singole immagini dal nuovo indice; file train di ChartQA nella cache senza inclusione nel pannello."""
from __future__ import annotations

import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import decontam_review  # noqa: E402
import panel_groups as pg  # noqa: E402
import panel_index as pi  # noqa: E402
from limite_vl import decontam as dc  # noqa: E402

FAILS = []
ENV = dict(os.environ, SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID", "test"))
FRACTIONS_TEST = {"ocrbench": 0.0, "docvqa": 0.0, "muirbench": 0.0, "mmstar": 0.0, "unimer_test": 0.0}  # tutto test


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def far_hashes(n: int, seed: int = 0, min_dist: int = 16) -> list[int]:
    rng, out = random.Random(seed), []
    while len(out) < n:
        h = rng.getrandbits(64)
        if all(dc.hamming(h, o) >= min_dist for o in out):
            out.append(h)
    return out


def voice(benchmark, file, ident, *, h=None, pixels=None, pos=None, text="", hints=None):
    item = f"{Path(file).stem}:{ident}"
    return {"benchmark": benchmark, "file": file, "item": item, "id": item if pos in (None, 0) else f"{item}#{pos}",
            "pos": pos, "phash": None if h is None else f"{h:016x}", "pixel_sha256": pixels, "size": None,
            "text": text, "hints": hints or {}}


def key(v: dict) -> str:
    return f"{v['benchmark']}/{v['item']}"


# ---------------------------------------------------------------- 1. irrisolti per gruppo
def test_group_unresolved():
    hs = far_hashes(12)
    dvf = "DocVQA/validation-00000-of-00006.parquet"
    mf = "data/test-00000-of-00005.parquet"
    none = ["first", "second", "None of the choices provided"]
    dev_chart = voice("chartqa", "data/val-00000-of-00001.parquet", "0", h=hs[0], pixels="dev-chart", pos=0,
                      hints={"imgname": "a.png", "type": "human"})
    dev_verse = voice("mathverse", "testmini.parquet", "1", h=hs[3], pixels="dev-verse", pos=0,
                      hints={"problem_index": "p1", "version": "Vision Only"})
    q1 = voice("docvqa", dvf, "q1", h=hs[1], pixels="q1", pos=0, hints={"docId": 1, "ucsf": "docA"})
    q2 = voice("docvqa", dvf, "q2", h=hs[0] ^ 1, pixels="q2", pos=0, hints={"docId": 2, "ucsf": "docA"})
    q3 = voice("docvqa", dvf, "q3", h=hs[2], pixels="q3", pos=0, hints={"docId": 3, "ucsf": "docB"})
    m1 = [voice("muirbench", mf, "1", h=hs[4], pixels="m1a", pos=0,
                hints={"id": "1", "counterpart_idx": "2", "options": none, "answer": "A"}),
          voice("muirbench", mf, "1", h=hs[3] ^ 3, pixels="m1b", pos=1,
                hints={"id": "1", "counterpart_idx": "2", "options": none, "answer": "A"})]
    m2 = [voice("muirbench", mf, "2", h=hs[5], pixels="m2a", pos=0,
                hints={"id": "2", "counterpart_idx": "1", "options": none, "answer": "C"}),
          voice("muirbench", mf, "2", h=hs[6], pixels="m2b", pos=1,
                hints={"id": "2", "counterpart_idx": "1", "options": none, "answer": "C"})]
    m3 = [voice("muirbench", mf, "3", h=hs[7], pixels="m3a", pos=0,
                hints={"id": "3", "counterpart_idx": "4", "options": none, "answer": "B"})]
    m4 = [voice("muirbench", mf, "4", h=hs[8], pixels="m4a", pos=0,
                hints={"id": "4", "counterpart_idx": "3", "options": none, "answer": "C"})]
    items = [dev_chart, dev_verse, q1, q2, q3, *m1, *m2, *m3, *m4]
    r = pg.build(items, set(), fractions=FRACTIONS_TEST)
    g, role, adm, ev = r["group_of"], r["role"], r["admitted"], r["evidence"]
    k1, k2, k3, km1, km2, km3, km4 = (key(v) for v in (q1, q2, q3, m1[0], m2[0], m3[0], m4[0]))
    check("documento: le due pagine stanno nello stesso gruppo", g[k1] == g[k2])
    check("documento: il candidato non fonde il gruppo con lo sviluppo", g[k2] != g[key(dev_chart)])
    check("documento: nessuno spostamento nello sviluppo", role(k1) == role(k2) == "test")
    check("documento: evidenza diretta solo sulla pagina coinvolta", k2 in ev and k1 not in ev)
    check("documento: inammissibili entrambe le pagine", not adm(k1) and not adm(k2))
    check("documento senza candidati ammesso", adm(k3))
    check("MUIRBench: coppia nello stesso gruppo", g[km1] == g[km2])
    check("MUIRBench: evidenza diretta solo sull'elemento coinvolto", km1 in ev and km2 not in ev)
    check("MUIRBench: coppia esclusa per intero", not adm(km1) and not adm(km2))
    check("MUIRBench: coppia senza candidati ammessa per intero", adm(km3) and adm(km4))
    verifica = r["summary"]["muirbench_controparti"]["verifica_coppie"]
    check("MUIRBench: verifica delle coppie ammesse e escluse, nessuna divisa",
          verifica == {"test_escluse_entrambe": 1, "test_ammesse_entrambe": 1}, str(verifica))
    rows = {row["chiave"]: row for row in r["rows"]}
    check("manifest: pagina senza evidenza ma nel gruppo irrisolto",
          rows[k1]["evidenze_dirette"] == 0 and rows[k1]["gruppo_irrisolto"] and rows[k1]["metrica_primaria"] is False)
    use = r["summary"]["test_utilizzabile"]["docvqa"]
    check("riepilogo: test utilizzabile di DocVQA", use == {"test": 3, "evidenze_dirette": 1, "solo_per_gruppo": 1,
                                                            "copertura_incompleta": 0, "ammessi": 1}, str(use))


# ---------------------------------------------------------------- 2. testo che preserva la matematica
def test_text_math():
    pairs = [("If x + 3 = 7, what is the value of x squared minus one?",
              "If x - 3 = 7, what is the value of x squared minus one?"),
             ("Find the number of integers n with n < 10 and n greater than zero.",
              "Find the number of integers n with n > 10 and n greater than zero."),
             ("The temperature is -5 degrees at noon; find the temperature after it rises by eight degrees.",
              "The temperature is 5 degrees at noon; find the temperature after it rises by eight degrees."),
             ("Compute x² + 1 for the given value of x in the problem statement.",
              "Compute x2 + 1 for the given value of x in the problem statement.")]
    for a, b in pairs:
        check(f"normalize_words li confonde, exact_text no: {a[:40]!r}",
              dc.normalize_words(a) == dc.normalize_words(b) and dc.exact_text(a) != dc.exact_text(b))
    check("exact_text: solo spazi compressi", dc.exact_text("Let  x +\n y = 2 . ") == "Let x + y = 2 .")
    a24 = voice("aime24", "data/train-00000-of-00001.parquet", "1", text=pairs[0][0])
    m500 = voice("math500", "test.jsonl", "test/a.json", text=pairs[0][1])
    a25 = voice("aime25", "aime2025-I.jsonl", "2", text="Let N be the number of ordered pairs (a, b) with a + b = 20.")
    apex = voice("apex", "data/train-00000-of-00001.parquet", "3",
                 text="Let N  be the number of ordered pairs (a, b) with a + b = 20.\n")
    m500b = voice("math500", "test.jsonl", "test/b.json", text=pairs[1][1])
    hm = voice("hmmt25", "data/train-00000-of-00001.parquet", "4", text=pairs[1][0])
    previous = {"links": [{"tipo": "testo_identico_solo_testo", "a": key(a24), "b": key(m500)},
                          {"tipo": "testo_identico_solo_testo", "a": key(a25), "b": key(apex)}]}
    r = pg.build([a24, m500, a25, apex, m500b, hm], set(), previous=previous)
    g, role, adm, ev = r["group_of"], r["role"], r["admitted"], r["evidence"]
    check("operatore diverso: nessuna fusione", g[key(a24)] != g[key(m500)])
    check("operatore diverso: resta nel test, inammissibile per il candidato permissivo",
          role(key(m500)) == "test" and not adm(key(m500))
          and any(c["tipo"] == "testo_permissivo" for c in ev[key(m500)]))
    check("disuguaglianza diversa: nessuna fusione, candidato", g[key(hm)] != g[key(m500b)] and key(m500b) in ev)
    check("stesso testo (solo spazi diversi): legame confermato e gruppo nello sviluppo",
          g[key(a25)] == g[key(apex)] and role(key(apex)) == "sviluppo")
    review = r["summary"]["riesame_testo_precedente"]
    check("riesame della vecchia regola: una confermata, una candidata",
          (review["confermate"], review["candidate"]) == (1, 1) and review["unioni"] == 2, str(review)[:200])


# ---------------------------------------------------------------- 3. vicinati pHash
def old_rule_pairs(images, radius, limit):
    """La regola di pannello-v2: taglio prima del filtro di gruppo e coppia scartata se j <= i."""
    hidx = dc.HashIndex([h for _, _, h in images], radius=radius)
    out = set()
    for i, (k, g, h) in enumerate(images):
        near = hidx.query(h)
        if len(near) > limit:
            near = sorted(near, key=lambda x: x[1])[:limit]
        for j, _ in near:
            if j <= i or images[j][1] == g:
                continue
            out.add(tuple(sorted((k, images[j][0]))))
    return out


def test_neighbors():
    # B ha cinque vicini a distanza 1 (C0-C4) e A a distanza 4; A, che viene dopo B, ha solo B. Con limite 3 la lista
    # di B non contiene A: la coppia esiste solo dal lato di A.
    images = [("B", "gB", 0)] + [(f"C{i}", f"gC{i}", 1 << (8 + i)) for i in range(5)] + [("A", "gA", 0b1111)]
    pairs, truncated, most = pg.phash_neighbors(images, radius=4, max_compare=3)
    check("vicinati asimmetrici: la coppia si trova dal lato non troncato", pairs.get(("A", "B")) == 4, str(pairs))
    check("vicinati asimmetrici: B troncato, A no", "B" in truncated and "A" not in truncated)
    check("vicinati asimmetrici: la regola di v2 perdeva la coppia", ("A", "B") not in old_rule_pairs(images, 4, 3))
    full, trunc_full, _ = pg.phash_neighbors(images, radius=4, max_compare=None)
    check("senza limite: tutte le coppie e nessun troncamento", ("A", "B") in full and not trunc_full
          and len(full) == 5 + 1 + 10, str(sorted(full)))
    # cinque copie dello stesso gruppo non consumano il limite: Y resta un candidato di X; dal lato di Y i sei membri
    # di gX contano come un gruppo solo
    images = [("X", "gX", 0)] + [(f"X{i}", "gX", 0) for i in range(5)] + [("Y", "gY", 1)]
    pairs, truncated, _ = pg.phash_neighbors(images, radius=4, max_compare=3)
    check("vicini dello stesso gruppo tolti prima del limite", ("X", "Y") in pairs and not truncated, str(truncated))
    check("vicini dello stesso gruppo: tutte le coppie dai lati completi",
          all((f"X{i}", "Y") in pairs for i in range(5)), str(sorted(pairs)))
    check("vicini dello stesso gruppo: la regola di v2 perdeva il candidato", ("X", "Y") not in old_rule_pairs(images, 4, 3))
    # più gruppi del limite: si tengono i più vicini e la copertura è incompleta
    images = [("Z", "gZ", 0)] + [(f"W{i}", f"gW{i}", 1 << i) for i in range(4)]
    pairs, truncated, _ = pg.phash_neighbors(images, radius=4, max_compare=3)
    check("più gruppi del limite: elemento troncato", "Z" in truncated, str(truncated))
    # copertura incompleta: un gruppo con un membro troncato non entra nella metrica primaria
    df = "data/sample1-00000-of-00001.parquet"
    items = [voice("dynamath", df, str(i), h=h, pixels=f"d{i}", pos=0, hints={"seed": str(i)})
             for i, h in enumerate((0, 1, 2, 4))]
    r = pg.build(items, set(), max_compare=2)
    check("limite superato: copertura incompleta e test non ammesso",
          all(not r["admitted"](key(v)) for v in items) and r["summary"]["phash"]["elementi_troncati"] == 4
          and all(row["copertura_incompleta"] for row in r["rows"]))
    r = pg.build(items, set(), max_compare=None)
    check("confronto completo: gli stessi elementi senza irrisolti sono ammessi",
          all(r["admitted"](key(v)) for v in items) and r["summary"]["phash"]["elementi_troncati"] == 0)


# ---------------------------------------------------------------- 4. immagini trasparenti
def transparent_png(seed: int = 0) -> bytes:
    im = Image.new("RGBA", (96, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((20 + seed, 10, 70, 55), outline=(0, 0, 0, 255), width=3)
    d.line((5, 60, 90, 5 + seed), fill=(200, 30, 30, 255), width=2)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def opaque_jpeg(color, size=(80, 60)) -> bytes:
    im = Image.new("RGB", size, color)
    ImageDraw.Draw(im).rectangle((10, 10, 40, 30), fill=(255 - color[0], 40, 90))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def write_parquet(path: Path, columns: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def snapshot(name: str) -> Path:
    spec = pi.BENCHMARKS.get(name) or {"mvmath": pi.MVMATH, "unimer_test": pi.UNIMER}[name]
    folder = pi.snap(spec["repo"], spec["rev"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def test_transparent(tmp: Path):
    data = transparent_png()
    new = dc.hash_image_bytes(data)
    old = dc.hash_image_bytes(data, dc.LEGACY_CONVENTION)
    check("PNG trasparente: le due convenzioni danno pHash lontani", dc.hamming(new[0], old[0]) > 4,
          f"distanza {dc.hamming(new[0], old[0])}")
    check("copia identica: stesso pHash, stessi pixel", dc.hash_image_bytes(bytes(data)) == new)
    check("convenzione storica senza sha256 dei pixel", old[1] is None)
    folder = snapshot("ocrbench")
    write_parquet(folder / "data/test-00000-of-00001.parquet",
                  {"dataset": ["finto"], "question": ["What is drawn?"], "question_type": ["Doc-oriented VQA"],
                   "answer": [["circle"]], "image": [{"bytes": data, "path": "t.png"}]})
    res = pi.file_task(("ocrbench", "data/test-00000-of-00001.parquet"))
    check("indice: voce del PNG trasparente con la convenzione corrente",
          res["items"][0]["phash"] == f"{new[0]:016x}" and res["items"][0]["pixel_sha256"] == new[1])
    (tmp / "index.json").write_text(json.dumps({"schema": pi.SCHEMA, "convenzione_immagini": dc.IMAGE_CONVENTION,
                                                "items": res["items"], "report": {}}))
    (tmp / "index-senza.json").write_text(json.dumps({"items": res["items"], "report": {}}))
    shard = tmp / "train-00000.tar"
    with tarfile.open(shard, "w") as tar:
        for name, payload in (("copia.png", data), ("copia.json", b'{"question": "x"}'),
                              ("altra.jpg", opaque_jpeg((10, 200, 30))), ("altra.json", b'{"question": "y"}')):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    scan = [sys.executable, "-B", str(ROOT / "scripts/decontam.py"), "scan", "--kind", "mmfinereason",
            "--shards", str(shard), "--workers", "1"]
    subprocess.run(scan + ["--index", str(tmp / "index.json"), "--out", str(tmp / "scan")], check=True, env=ENV)
    matches = [json.loads(line) for line in (tmp / "scan/matches.jsonl").read_text().splitlines()]
    summary = json.loads((tmp / "scan/summary.json").read_text())
    check("scan: copia del PNG trasparente ritrovata a distanza 0",
          any(m["key"] == "copia" and m["match"] == "image" and m["distance"] == 0 for m in matches)
          and "altra" not in {m["key"] for m in matches}, str(matches)[:200])
    check("scan: convenzione registrata", summary.get("convenzione_immagini") == dc.IMAGE_CONVENTION)
    refused = subprocess.run(scan + ["--index", str(tmp / "index-senza.json"), "--out", str(tmp / "scan2")], env=ENV,
                             capture_output=True, text=True)
    check("scan: indice senza convenzione rifiutato", refused.returncode != 0 and "convenzione" in refused.stderr,
          refused.stderr[-200:])
    subprocess.run(scan + ["--index", str(tmp / "index-senza.json"), "--out", str(tmp / "scan3"),
                           "--convention", dc.IMAGE_CONVENTION], check=True, env=ENV)
    check("scan: convenzione esplicita accettata", "copia" in (tmp / "scan3/exclude.txt").read_text().split())
    try:
        dc.index_convention({"convenzione_immagini": dc.IMAGE_CONVENTION}, dc.LEGACY_CONVENTION)
        mixed = False
    except ValueError:
        mixed = True
    check("index_convention: convenzioni diverse rifiutate", mixed)


# ---------------------------------------------------------------- 5. immagini dal nuovo indice
def test_resolve(tmp: Path):
    expected = {}
    pngs = [transparent_png(s) for s in range(1, 6)]
    jpgs = [opaque_jpeg((20 * i, 100, 200 - 20 * i)) for i in range(1, 6)]
    folder = snapshot("muirbench")
    rows = {"idx": ["1", "2"], "task": ["Counting", "Counting"], "image_relation": ["Temporal", "Temporal"],
            "image_type": ["Photography", "Photography"], "question": ["How many?", "How many?"],
            "options": [["1", "2", "None of the choices provided"]] * 2, "answer": ["A", "C"],
            "image_list": [[{"bytes": pngs[0], "path": "a"}, {"bytes": pngs[1], "path": "b"},
                            {"bytes": jpgs[0], "path": "c"}],
                           [{"bytes": pngs[2], "path": "d"}, {"bytes": jpgs[1], "path": "e"}]],
            "counterpart_idx": ["2", "1"]}
    write_parquet(folder / "data/test-00000-of-00001.parquet", rows)
    for item, imgs in (("test-00000-of-00001:1", rows["image_list"][0]), ("test-00000-of-00001:2", rows["image_list"][1])):
        for pos, v in enumerate(imgs):
            expected[("muirbench", item, pos)] = v["bytes"]
    folder = snapshot("mvmath")
    problems = [{"problem_id": 1, "question": "Compare the figures.", "input_image": ["images/1_1.jpg", "images/1_2.jpg"],
                 "answer_type": "choice", "subject": "Algebra", "image_relavance": 1, "is_multi_img": True,
                 "difficulty": "Low"},
                {"problem_id": 2, "question": "Which one?", "input_image": ["images/2_1.jpg", "images/2_2.jpg",
                                                                            "images/2_3.jpg"],
                 "answer_type": "choice", "subject": "Logic", "image_relavance": 0, "is_multi_img": True,
                 "difficulty": "High"}]
    (folder / pi.MVMATH["file"]).write_text(json.dumps(problems))
    with zipfile.ZipFile(folder / "images.zip", "w") as zf:
        for n, p in enumerate(["images/1_1.jpg", "images/1_2.jpg", "images/2_1.jpg", "images/2_2.jpg", "images/2_3.jpg"]):
            zf.writestr(p, jpgs[n])
    for n, (pid, pos) in enumerate([(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]):
        expected[("mvmath", f"MV-MATH:{pid}", pos)] = jpgs[n]
    folder = snapshot("unimer_test")
    with zipfile.ZipFile(folder / pi.UNIMER["zip"], "w") as zf:
        zf.writestr("UniMER-Test/spe.txt", "a+b\nc-d\n")
        zf.writestr("UniMER-Test/spe/0.png", pngs[3])
        zf.writestr("UniMER-Test/spe/1.png", pngs[4])
    expected[("unimer_test", "spe-0000000", 0)] = pngs[3]
    expected[("unimer_test", "spe-0000001", 0)] = pngs[4]
    entries = []
    for task in (("muirbench", "data/test-00000-of-00001.parquet"), ("mvmath", pi.MVMATH["file"]), ("unimer_test", "spe")):
        entries += pi.file_task(task)["items"]
    check("indice: una voce per immagine con la posizione", len(entries) == len(expected)
          and all((e["benchmark"], e["item"], e["pos"]) in expected for e in entries))
    out, errors = pi.resolve_entries(entries, dc.IMAGE_CONVENTION)
    check("risoluzione: ogni voce ritrova esattamente la propria immagine", not errors and all(
        out[i] == expected[(e["benchmark"], e["item"], e["pos"])] for i, e in enumerate(entries)), str(errors)[:200])
    (tmp / "idx").mkdir()
    (tmp / "idx/index.json").write_text(json.dumps({"schema": pi.SCHEMA, "convenzione_immagini": dc.IMAGE_CONVENTION,
                                                    "items": entries, "report": {}}))
    report = pi.verify_index(tmp / "idx", workers=1)
    check("verifica dell'indice: tutte le voci ok", report["verificato"] and report["esiti"] == {"ok": len(entries)},
          str(report["esiti"]))
    swapped = [dict(e) for e in entries]
    a, b = (i for i, e in enumerate(swapped) if e["benchmark"] == "muirbench" and e["item"].endswith(":1")
            and e["pos"] in (0, 1))
    swapped[a]["pos"], swapped[b]["pos"] = swapped[b]["pos"], swapped[a]["pos"]
    out, errors = pi.resolve_entries(swapped, dc.IMAGE_CONVENTION)
    check("posizioni scambiate: segnalate, nessuna immagine sostituita",
          len(errors) == 2 and out[a] is None and out[b] is None, str(errors)[:200])
    broken = [dict(e) for e in entries]
    broken[0] = {**broken[0], "pos": 7}
    broken.append({**entries[-1], "item": "spe-0009999", "id": "spe-0009999"})
    out, errors = pi.resolve_entries(broken, dc.IMAGE_CONVENTION)
    check("posizione oltre le immagini e elemento assente: segnalati",
          any("posizione 7 mancante" in x for x in errors) and any("assente" in x for x in errors), str(errors)[:300])
    got = decontam_review.bench_images([], entries, dc.IMAGE_CONVENTION)
    check("revisione: immagini del pannello ritrovate per voce",
          all(got[i] == expected[(e["benchmark"], e["item"], e["pos"])] for i, e in enumerate(entries)))
    try:
        decontam_review.bench_images([], swapped, dc.IMAGE_CONVENTION)
        stopped = False
    except SystemExit:
        stopped = True
    check("revisione: un mancato abbinamento ferma la revisione", stopped)


# ---------------------------------------------------------------- 6. split di ChartQA
def test_chartqa_splits():
    folder = snapshot("chartqa")
    for split in ("train", "val", "test"):
        write_parquet(folder / f"data/{split}-00000-of-00001.parquet",
                      {"imgname": ["x.png"], "query": ["q"], "label": ["1"], "type": ["human"],
                       "image": [opaque_jpeg((1, 2, 3))]})
    files, missing = pi.benchmark_files("chartqa")
    check("ChartQA: con il train nella cache l'indice prende solo val e test",
          files == ["data/val-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"] and not missing, str(files))
    check("ChartQA: un glob generico avrebbe preso anche il train",
          "data/train-00000-of-00001.parquet" in {str(p.relative_to(folder)) for p in folder.glob("data/*.parquet")})
    item = voice("chartqa", "data/train-00000-of-00001.parquet", "0", h=1, pixels="t", pos=0, hints={"imgname": "x.png"})
    try:
        pg.build([item], set())
        refused = False
    except ValueError as e:
        refused = "split inatteso" in str(e)
    check("ruoli: uno split inatteso ferma il run invece di diventare test", refused)
    (folder / "data/val-00000-of-00001.parquet").unlink()
    files, missing = pi.benchmark_files("chartqa")
    check("ChartQA: un file autorizzato mancante è segnalato", missing == ["data/val-00000-of-00001.parquet"])


if __name__ == "__main__":
    test_group_unresolved()
    test_text_math()
    test_neighbors()
    tmp = ROOT / "outputs" / "tmp" / f"test-panel-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    tmp.mkdir(parents=True, exist_ok=True)
    pi.HF = tmp / "hub"  # snapshot finti: i lettori di panel_index.py li cercano qui
    try:
        test_transparent(tmp)
        test_resolve(tmp)
        test_chartqa_splits()
    finally:
        shutil.rmtree(tmp)
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
