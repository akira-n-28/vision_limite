"""PixMo-Cap: manifest candidato (scripts/pixmo_manifest.py) e reader col manifest (manifest.TASKED, stream.make_sources,
data.pixmo_caption_sample). Fixture costruita qui con gli helper di tests/test_pixmo_inventory.py: parquet in stile
Hugging Face, shard veri di scripts/shard_parquet.py, run4 vero, inventario e scansione veri di scripts/pixmo_inventory.py
contro un pannello finto con tutti i ruoli. Nessun dato reale tranne l'esistenza degli shard locali, per l'arresto senza
manifest, che avviene prima di leggerli. Solo in un job SLURM (slurm/tests_cpu.sbatch).

Controlla:
- unioni transitive (pixel dell'originale, pixel e byte del JPEG dello shard, URL esatto), comprese immagini finali
  identiche con originali diversi; URL vuoti senza legami;
- blocchi diretti e propagazione al gruppo (quarantena ereditata distinta dal motivo diretto, sovrapposizioni,
  esclusione per identità confermata), segnali di run4 senza effetto;
- manifest della fixture: conteggi autorizzati, scarto dello sharding fuori dai record, nessun gruppo in due split,
  verifica dopo la scrittura, arresti per conteggi o ingressi diversi;
- reader: manifest obbligatorio per gli shard locali, identificazione, tipo e liste della config; un'epoca consegna
  esattamente gli ammessi dello split, mai quarantena o esclusi; validazione dal dev; immagine, didascalia e richiesta
  invariate; JPEG o didascalia alterati fermati."""
from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("SLURM_JOB_ID", "test")
import pixmo_inventory as pi  # noqa: E402
import pixmo_manifest as pm  # noqa: E402
import test_pixmo_inventory as tpi  # noqa: E402
from limite_vl import data as dt  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl import stream as st  # noqa: E402
from limite_vl.data_lists import missing_required  # noqa: E402
from limite_vl.processing import to_rgb  # noqa: E402
from limite_vl.provenance import sha256_file  # noqa: E402
from PIL import Image  # noqa: E402

REV = tpi.REV
FAILS: list[str] = []
DONE: list[str] = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({str(detail)[:400]})" if detail != "" and not cond else ""))
    DONE.append(name)
    if not cond:
        FAILS.append(name)


def refused(fn) -> str:
    try:
        fn()
    except (mf.ManifestError, pm.Problem) as exc:
        return str(exc) or "rifiutato"
    except Exception:  # noqa: BLE001 - un errore di altro tipo non è l'arresto esplicito atteso
        return ""
    return ""


# ---------------------------------------------------------------- funzioni pure

def row(key, px_orig, px_shard, jpg, url, outcome="nessuna_corrispondenza", exif=None, defects=(), signals=()):
    return {"key": key, "px_orig": px_orig, "px_shard": px_shard, "jpg": jpg, "url": url, "outcome": outcome,
            "exif": exif, "defects": list(defects), "signals": list(signals)}


def test_pure() -> None:
    rows = [row("a", "O1", "S1", "J1", "u1"), row("b", "O1", "S2", "J2", "u2"),   # a-b: pixel dell'originale
            row("c", "O2", "S3", "J2", "u3"),                                     # b-c: byte del JPEG
            row("d", "O3", "S4", "J4", "u3"),                                     # c-d: URL
            row("e", "O5", "S5", "J5", "u5"), row("f", "O6", "S5", "J6", "u6"),   # e-f: pixel dello shard, originali diversi
            row("g", "O7", "S7", "J7", ""), row("h", "O8", "S8", "J8", None),     # URL vuoti: nessun legame
            row("i", "O9", "S9", "J9", "u9")]
    links = pm.link_lists(rows)
    rep = mf_components(rows, links)
    groups = sorted(sorted(rows[i]["key"] for i in range(len(rows)) if rep[i] == r) for r in set(rep))
    check("unioni transitive: originale, JPEG e URL in catena; JPEG uguali con originali diversi; URL vuoti senza legami",
          groups == [["a", "b", "c", "d"], ["e", "f"], ["g"], ["h"], ["i"]], groups)
    check("legami per relazione contati separatamente", {k: len(v) for k, v in links.items()}
          == {"pixel_originale": 1, "pixel_shard": 1, "jpeg_shard": 1, "url": 1}, {k: len(v) for k, v in links.items()})
    stats = pm.union_stats(rows, links)
    check("conteggi prima e dopo le unioni: 9 record, 8 gruppi col solo originale, 5 dopo tutte le relazioni",
          stats["per_relazione"]["pixel_originale"]["gruppi"] == 8 and stats["cumulative"][-1]["gruppi"] == 5
          and stats["cumulative"][-1]["massimo"] == 4, stats["cumulative"])
    # blocchi: a candidato, c EXIF 6, e segnaposto, i difetto; b, d ereditano; f eredita il segnaposto; g segnali
    rows2 = [dict(r) for r in rows]
    rows2[0]["outcome"] = "candidato"
    rows2[2]["exif"] = 6
    rows2[6]["signals"] = ["colore"]
    rows2[7]["exif"] = 0  # valore non valido: nessun effetto
    rows2[8]["defects"] = ["didascalia del loader vuota"]
    d = pm.decide(rows2, rep, {"e"})
    by = {r["key"]: x for r, x in zip(rows2, d)}
    check("blocchi diretti: candidato, EXIF 6 e segnaposto in quarantena, difetto escluso, EXIF 0 senza effetto",
          by["a"]["diretti"] == ["decontaminazione_candidato"] and by["c"]["diretti"] == ["orientamento_exif"]
          and by["e"]["diretti"] == ["possibile_segnaposto"] and by["i"]["diretti"] == ["difetto_strutturale"]
          and by["i"]["stato"] == "escluso" and by["h"]["diretti"] == [] and by["h"]["stato"] == "ammesso")
    check("propagazione: tutto il gruppo in quarantena; ereditati i blocchi degli altri record, distinti dai diretti",
          all(by[k]["stato"] == "quarantena" for k in "abcdef")
          and by["b"]["diretti"] == [] and {e["codice"] for e in by["b"]["ereditati"]}
          == {"decontaminazione_candidato", "orientamento_exif"}
          and [e["codice"] for e in by["a"]["ereditati"]] == ["orientamento_exif"]
          and by["f"]["ereditati"] == [{"codice": "possibile_segnaposto", "da": ["e"], "n": 1}]
          and any(m.startswith("dal gruppo: ") for m in by["d"]["motivo"]), by["b"])
    check("segnali di run4: nessun effetto sullo stato (record g ammesso coi segnali)",
          by["g"]["stato"] == "ammesso" and by["g"]["diretti"] == [] and not by["g"]["ereditati"])
    rows3 = [dict(r) for r in rows]
    rows3[0]["outcome"] = "corrispondenza_confermata"
    rows3[1]["outcome"] = "candidato"
    d3 = {r["key"]: x for r, x in zip(rows3, pm.decide(rows3, rep, set()))}
    check("identità confermata: il gruppo intero escluso, la quarantena di un altro record non lo attenua",
          all(d3[k]["stato"] == "escluso" for k in "abcd") and d3["b"]["diretti"] == ["decontaminazione_candidato"])
    recs = [{"split": "train", "stato": x["stato"], "gruppo": f"g{rep[i]}", "compiti": ["pixmo_caption"],
             "blocchi": {"diretti": x["diretti"], "ereditati": x["ereditati"]},
             "evidenze": [{"lista": "pixmo_check_run4"}] if rows2[i]["signals"] else []}
            for i, x in enumerate(d)]
    s = pm.summarize(recs)
    check("riepilogo: sovrapposizioni dirette, quarantena solo ereditata, segnali per stato",
          s["non_ammessi_per_origine"].get("quarantena/solo_ereditato") == 3
          and s["segnali_run4_per_stato"] == {"train/ammesso": 1} and s["per_motivo_diretto"]["possibile_segnaposto"] == 1,
          s)


def mf_components(rows, links):
    import unimer_manifest as um

    return um.components(len(rows), [p for name in pm.RELATIONS for p in links[name]])


# ---------------------------------------------------------------- fixture

def uniform(size, color, poke=None) -> bytes:
    """Immagine uniforme; con `poke` un solo pixel cambiato di poco: originali diversi che il ridimensionamento a 1024
    patch rende identici (la sua media pesa ogni pixel meno di 1/4 di livello)."""
    im = Image.new("RGB", size, color)
    if poke is not None:
        im.putpixel(poke, tuple(c + 1 for c in color))
    return tpi.encode(im, "PNG")


def build_fixture(tmp: Path) -> dict:
    base = tmp / "hub" / "datasets--anthracite-org--pixmo-cap-images"
    exif = Image.Exif()
    exif[pi.EXIF_ORIENTATION] = 6
    placeholder = tpi.encode(tpi.picture(50, (161, 81)), quality=90, exif=exif.tobytes())
    candidate = tpi.encode(tpi.picture(3, (800, 600)), quality=92)
    confirmed = tpi.encode(tpi.picture(2, (500, 300)), quality=90)
    rotated = tpi.encode(tpi.picture(7, (400, 300)), quality=90, exif=exif.tobytes())
    tr = ["a photo"]
    rows0 = [tpi.row(tpi.encode(tpi.picture(1, (640, 480)), quality=90), "A red car near a tree.",
                     ["a red car near a tree"], "http://a.example/0"),
             *[tpi.row(placeholder, f"A scene number {i} with a dog.", ["a dog"], f"http://ph.example/{i}")
               for i in range(1, 4)],
             tpi.row(candidate, "Two green chairs stand on a terrace.", ["two green chairs"], "http://c.example/4"),
             tpi.row(rotated, "A door seen from the left.", ["a door"], "http://d.example/5"),
             tpi.row(tpi.encode(tpi.picture(8, (512, 512)), quality=90), "A tall lighthouse.", ["a lighthouse"],
                     "http://same.example/x"),
             tpi.row(b"<html>not an image</html>", "A broken file.", ["broken"], "http://e.example/7"),
             tpi.row(tpi.encode(tpi.picture(9, (300, 300)), quality=90), "A blue bowl.", ["a bowl"],
                     "http://c.example/4"),
             tpi.row(tpi.encode(tpi.picture(10, (300, 200)), quality=90), "Seven cats sit on a wooden porch.",
                     ["some cats sit on a wooden porch"], "http://g.example/9"),
             tpi.row(confirmed, "A blue boat on a lake.", ["a blue boat"], "http://b.example/10"),
             tpi.row(uniform((2048, 1536), (90, 120, 150)), "A plain blue wall.", ["a wall"], "http://u.example/11"),
             tpi.row(uniform((2048, 1536), (90, 120, 150), poke=(1000, 700)), "Another plain blue wall.", ["a wall"],
                     "http://u.example/12")]
    rows0 += [tpi.row(tpi.encode(tpi.picture(100 + i, (320, 240)), quality=90), f"A photo of object {i}.", tr,
                      f"http://n.example/{i}") for i in range(17)]
    rows1 = [tpi.row(tpi.encode(tpi.picture(300, (320, 240)), quality=90), "A tower at night.", ["a tower"],
                     "http://same.example/x")]
    rows1 += [tpi.row(tpi.encode(tpi.picture(400 + i, (320, 240)), quality=90), f"Another photo {i}.", tr,
                      f"http://m.example/{i}") for i in range(14)]
    p0 = tpi.blob(base, "data/train-00000-of-00075.parquet", tpi.parquet_bytes(rows0))
    p1 = tpi.blob(base, "data/train-00001-of-00075.parquet", tpi.parquet_bytes(rows1))
    tpi.blob(base, "README.md", b"---\ndataset_info:\n  splits:\n  - name: train\n    num_examples: 200\n"
                                b"  download_size: 123456\n---\n# PixMo-Cap (fixture)\n", git=True)
    snapshot = base / "snapshots" / REV
    shards = tmp / "shards"
    env = {**os.environ, "SLURM_JOB_ID": "test"}
    for cmd in ([sys.executable, "-B", str(ROOT / "scripts/shard_parquet.py"), "--kind", "pixmo", "--files", str(p0),
                 str(p1), "--out", str(shards), "--source", f"{pi.REPO}@{REV}", "--prefix", "pixmo-cap",
                 "--files-per-shard", "1", "--workers", "1"],
                [sys.executable, "-B", str(ROOT / "scripts/pixmo_transcript_check.py"), "--shards-jsonl",
                 str(shards / "shards.jsonl"), "--out", str(tmp / "run4")]):
        done = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"{cmd[2]}: {done.stderr[-2000:]}")
    reenc = tpi.encode(to_rgb(Image.open(io.BytesIO(candidate))), quality=40)
    near = to_rgb(Image.open(io.BytesIO(placeholder)))  # stessi pixel tranne uno: pHash uguale, identità no
    near.putpixel((0, 0), tuple((v + 128) % 256 for v in near.getpixel((0, 0))))
    ph_reenc = tpi.encode(near, "PNG")
    entries = [{"benchmark": "mmstar", "item": "test:1", "image": tpi.png_of_pixels(confirmed), "text": "What is it?",
                "ruolo": "test", "metrica": True},
               {"benchmark": "muirbench", "item": "test:2", "image": reenc, "text": "Which one?", "ruolo": "test",
                "metrica": False},
               {"benchmark": "mathvista", "item": "testmini:3", "image": ph_reenc, "text": "How many?",
                "ruolo": "sviluppo"},
               {"benchmark": "docvqa", "item": "val:4", "image": tpi.encode(tpi.picture(901, (60, 40)), "PNG"),
                "text": "What is the date?", "ruolo": "diagnostica"},
               {"benchmark": "ocrbench", "item": "test:5", "text": "Read the text in the image please.",
                "ruolo": "solo_decontaminazione"}]
    index_dir, panel_dir, isha = tpi.panel_fixture(tmp, entries)
    hist = []
    for name in ("p6", "f1eval"):
        d = tmp / "storiche" / name / "pixmo"
        d.mkdir(parents=True)
        (d / "exclude.txt").write_text("")
        (d / "exclude_final.txt").write_text("")
        (d / "summary.json").write_text(json.dumps({"kind": "pixmo"}))
        hist.append(d)
    tsv = tmp / "report-verify.tsv"
    lines = ["status\tdest\tsize\texpected\tlabel"]
    for p in (p0, p1):
        b = p.resolve()
        lines.append(f"ok\t{b}\t{b.stat().st_size}\tsha256:{b.name}\t{pi.REPO}@{REV[:8]}:data/{p.name}")
    tsv.write_text("\n".join(lines) + "\n")
    fx = {"snapshot": snapshot, "shards": shards, "run4": tmp / "run4", "index": index_dir, "panel": panel_dir,
          "isha": isha, "historical": hist, "tsv": tsv}
    out = tmp / "inventario"
    if tpi.pi.main(tpi.scan_args(fx, out)) != 0:
        raise RuntimeError("pixmo_inventory scan fallita sulla fixture")
    code = tpi.pi.main(tpi.verify_args(fx, out, out / "verifica"))
    if code != 0:
        raise RuntimeError(f"pixmo_inventory verifica fallita sulla fixture: {tpi.failed(out / 'verifica')}")
    fx.update(inventory=out / "inventario", scan=out / "scansione", verifica=out / "verifica")
    return fx


def manifest_args(fx: dict, tmp: Path, name: str, **over) -> list[str]:
    a = {"inventory": fx["inventory"], "verifica": fx["verifica"], "scan": fx["scan"], "shards": fx["shards"],
         "run4": fx["run4"], "out": tmp / name / "manifest", "report": tmp / name / "verifica",
         "summary": tmp / name / "riepilogo.json", "revision": REV,
         "inventory-sha256": sha256_file(fx["inventory"] / "records.jsonl.gz"),
         "scan-sha256": sha256_file(fx["scan"] / "records.jsonl.gz"), "index-sha256": fx["isha"],
         "gruppi-segnaposto": 1, "righe-segnaposto": 3, "exif-attesi": 4, "candidati-attesi": 4,
         "dev-fraction": 0.2, **over}
    argv = ["costruzione"]
    for k, v in a.items():
        argv += [f"--{k}", str(v)]
    return argv + ["--historical", *[str(h) for h in fx["historical"]]]


def key(f: int, r: int) -> str:
    return f"pixmo-{f:05d}-{r:06d}"


def test_manifest(tmp: Path, fx: dict) -> dict:
    code = pm.main(manifest_args(fx, tmp, "m1"))
    folder = tmp / "m1" / "manifest"
    check("costruzione e verifica dopo la scrittura superate", code == 0,
          json.loads((tmp / "m1" / "verifica" / "verifica.json").read_text())["controlli"]
          if (tmp / "m1" / "verifica" / "verifica.json").is_file() else "nessun rapporto")
    m = mf.load(folder)
    recs = {r["record"]: r for rows in m.by_shard.values() for r in rows.values()}
    check("record: uno per record degli shard; la riga illeggibile fuori dai record e contata negli scarti",
          len(recs) == 30 + 15 - 1 and key(0, 7) not in recs
          and [d["chiave"] for d in m.meta["scarti_dello_sharding"]] == [key(0, 7)], len(recs))
    st_of = {k: r["stato"] for k, r in recs.items()}
    d_of = {k: r["blocchi"]["diretti"] for k, r in recs.items()}
    check("segnaposto, candidati ed EXIF in quarantena coi motivi diretti, sovrapposti nel gruppo segnaposto",
          all(st_of[key(0, i)] == "quarantena" and sorted(d_of[key(0, i)]) == sorted(
              ["decontaminazione_candidato", "possibile_segnaposto", "orientamento_exif"]) for i in (1, 2, 3))
          and d_of[key(0, 4)] == ["decontaminazione_candidato"] and d_of[key(0, 5)] == ["orientamento_exif"],
          {k: d_of[k] for k in (key(0, 1), key(0, 4), key(0, 5))})
    check("quarantena ereditata: stesso URL del candidato, nessun motivo diretto",
          st_of[key(0, 8)] == "quarantena" and d_of[key(0, 8)] == []
          and [e["codice"] for e in recs[key(0, 8)]["blocchi"]["ereditati"]] == ["decontaminazione_candidato"]
          and recs[key(0, 8)]["gruppo"] == recs[key(0, 4)]["gruppo"])
    check("identità confermata col pannello: escluso", st_of[key(0, 10)] == "escluso"
          and d_of[key(0, 10)] == ["decontaminazione_confermata"])
    check("segnale di run4: evidenza, record ammesso", st_of[key(0, 9)] == "ammesso"
          and any(e["lista"] == "pixmo_check_run4" for e in recs[key(0, 9)]["evidenze"]))
    check("duplicati fuori dai gruppi cautelativi raggruppati senza scarti: stesso URL in due shard, JPEG identici con "
          "originali diversi",
          recs[key(0, 6)]["gruppo"] == recs[key(1, 0)]["gruppo"] and st_of[key(0, 6)] == st_of[key(1, 0)] == "ammesso"
          and recs[key(0, 11)]["immagini"][0]["sha256"] == recs[key(0, 12)]["immagini"][0]["sha256"]
          and recs[key(0, 11)]["originale"]["pixel_sha256"] != recs[key(0, 12)]["originale"]["pixel_sha256"]
          and recs[key(0, 11)]["gruppo"] == recs[key(0, 12)]["gruppo"] and st_of[key(0, 11)] == "ammesso",
          {k: recs[k]["gruppo"][:8] for k in (key(0, 6), key(1, 0), key(0, 11), key(0, 12))})
    split_of = {}
    for r in recs.values():
        split_of.setdefault(r["gruppo"], set()).add(r["split"])
    check("nessun gruppo in due split; dev non vuoto; quarantene ed esclusi in train",
          all(len(s) == 1 for s in split_of.values()) and any(r["split"] == "dev" for r in recs.values())
          and all(r["split"] == "train" for r in recs.values() if r["stato"] != "ammesso"))
    check("compito pixmo_caption e digest della didascalia in ogni record",
          all(r["compiti"] == ["pixmo_caption"] and mf.HEX64.match(r["didascalia_sha256"]) for r in recs.values()))
    # arresti prima della scrittura
    check("conteggi diversi da quelli autorizzati: costruzione fermata, nessun manifest",
          pm.main(manifest_args(fx, tmp, "m2", **{"gruppi-segnaposto": 2})) == 2
          and not (tmp / "m2" / "manifest" / "manifest.json").exists())
    check("inventario con un altro sha256 atteso: costruzione fermata",
          pm.main(manifest_args(fx, tmp, "m3", **{"inventory-sha256": "0" * 64})) == 2)
    check("uscite esistenti: nessuna sovrascrittura", pm.main(manifest_args(fx, tmp, "m1")) == 2)
    # verifica di un manifest alterato: uno stato riammesso
    bad = tmp / "m-alterato"
    shutil.copytree(folder, bad)
    lines = [json.loads(x) for x in gzip.decompress((bad / "records.jsonl.gz").read_bytes()).decode().splitlines()]
    for r in lines:
        if r["record"] == key(0, 8):
            r["stato"], r["motivo"], r["blocchi"]["ereditati"] = "ammesso", [], []
    (bad / "records.jsonl.gz").unlink()
    import unimer_manifest as um

    sha, _ = um.write_records_gz(bad / "records.jsonl.gz",
                                 (json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for r in lines))
    meta = json.loads((bad / "manifest.json").read_text())
    meta["records"]["sha256"] = sha
    (bad / "manifest.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    argv = manifest_args(fx, tmp, "m4")
    argv[0] = "verifica"
    i = argv.index("--out")
    argv[i + 1] = str(bad)
    j = argv.index("--summary")
    del argv[j:j + 2]
    check("verifica di un manifest con un record riammesso: fallita", pm.main(argv) == 1)
    return {"folder": folder, "sha": sha256_file(folder / "manifest.json"), "recs": recs}


# ---------------------------------------------------------------- reader

def epoch(source) -> list[tuple[str, dict]]:
    reader = st.SourceReader(source, seed=3, worker=0, n_workers=1, val_count=0)
    out = []
    while True:
        k, files = reader.next()
        if reader.epoch > 0:
            return out
        out.append((k, files))


def test_reader(tmp: Path, fx: dict, mm: dict) -> None:
    glob_ = str(fx["shards"] / "pixmo-cap-*.tar")

    def spec(name, split, **over):
        return {"name": name, "kind": "pixmo_caption", "shards": glob_, "weight": 1.0, "manifest": str(mm["folder"]),
                "manifest_sha256": mm["sha"], "split": split, **over}

    recs = mm["recs"]
    want = {s: sorted(k for k, r in recs.items() if r["split"] == s and r["stato"] == "ammesso") for s in ("train", "dev")}
    tr, dv = st.make_sources([spec("p", "train"), spec("pd", "dev")])
    got_tr, got_dv = epoch(tr), epoch(dv)
    check("reader: un'epoca consegna esattamente gli ammessi dello split, una volta (train e dev)",
          sorted(k for k, _ in got_tr) == want["train"] and sorted(k for k, _ in got_dv) == want["dev"]
          and len({k for k, _ in got_tr}) == len(got_tr), (len(got_tr), len(got_dv)))
    check("reader: quarantena ed esclusi mai consegnati", all(recs[k]["stato"] == "ammesso" for k, _ in got_tr + got_dv))
    delivered = dict(got_tr + got_dv)
    raw = {k: f for p in tr.shards for k, f in dt.iter_shard(p)}
    same = []
    for k, files in delivered.items():
        s = dt.SAMPLE_LOADERS["pixmo_caption"](k, files)
        meta = json.loads(raw[k]["json"])
        same.append(s.messages[-1]["content"] == meta["caption"]
                    and s.messages[0]["content"][1]["text"] == dt.pick_prompt(k, dt.DETAILED_CAPTION_PROMPTS)
                    and files["jpg"] == raw[k]["jpg"] and s.tasks == ("pixmo_caption",)
                    and mf.sha256_text(meta["caption"]) == recs[k]["didascalia_sha256"])
    check("reader: immagine, didascalia e richiesta degli ammessi invariate, didascalia col digest del manifest",
          same and all(same), sum(same))
    val = st.validation_samples(tr, 1000)
    order = [k for k, r in sorted(recs.items(), key=lambda x: (x[1]["shard"], x[1]["indice"]))
             if r["split"] == "dev" and r["stato"] == "ammesso"]
    check("validazione dal dev del manifest, nell'ordine degli shard", [k for k, _ in val] == order, len(val))
    # decisione allegata alterata: didascalia diversa fermata dal loader
    k0 = want["train"][0]
    files = dict(delivered[k0])
    meta = json.loads(files["json"])
    meta["caption"] = meta["caption"] + " Extra."
    files["json"] = json.dumps(meta).encode()
    check("loader: didascalia diversa dal digest del manifest fermata",
          bool(refused(lambda: dt.SAMPLE_LOADERS["pixmo_caption"](k0, files))))
    altered = dict(raw[k0])
    altered["jpg"] = altered["jpg"][:-2] + b"\x00\x00"
    check("reader: JPEG con altri byte rifiutato dal controllo del manifest",
          tr.admission.check(k0, altered) == "JPEG diverso da quello del manifest" and tr.admission.check(k0, raw[k0])
          is None)
    q = next(k for k, r in recs.items() if r["stato"] == "quarantena")
    check("reader: motivo del rifiuto di un record in quarantena", tr.admission.refusal(q) == "quarantena")
    # arresti
    check("manifest non identificato: rifiutato", bool(refused(lambda: st.make_sources([spec("x", "train",
                                                                                             manifest_sha256="0" * 64)]))))
    check("liste della config col manifest di PixMo: rifiutate",
          bool(refused(lambda: st.make_sources([spec("x", "train", exclude=[str(fx["historical"][0] / "exclude.txt")])]))))
    check("tipo di loader di un'altra sorgente col manifest di PixMo: rifiutato",
          bool(refused(lambda: st.make_sources([{**spec("x", "train"), "kind": "llava"}]))))
    real = "data/shards/pixmo-cap/3e42775e/pixmo-cap-*.tar"
    if list(ROOT.glob(real)):
        check("shard locali di PixMo senza manifest: rifiutati prima di leggerli",
              "si leggono solo con un manifest" in refused(lambda: st.make_sources(
                  [{"name": "r", "kind": "pixmo_caption", "shards": str(ROOT / real), "weight": 1.0}])))
    check("liste storiche non richieste con il manifest, richieste senza",
          not missing_required([{"name": "p", "shards": real, "manifest": "x"}])
          and len(missing_required([{"name": "p", "shards": real}])) == 2)


def main() -> None:
    test_pure()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fx = build_fixture(tmp)
        mm = test_manifest(tmp, fx)
        test_reader(tmp, fx, mm)
    print(f"{len(DONE) - len(FAILS)} controlli su {len(DONE)}")
    if FAILS:
        print("falliti: " + "; ".join(FAILS))
        sys.exit(1)


if __name__ == "__main__":
    main()
