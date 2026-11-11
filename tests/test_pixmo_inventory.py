"""PixMo-Cap: inventario, collegamento originale -> shard, segnali di run4 e decontaminazione contro un pannello di prova
(scripts/pixmo_inventory.py). Le fixture si costruiscono qui: parquet con lo schema di anthracite-org/pixmo-cap-images in
uno snapshot in stile Hugging Face (blob col nome dello sha256), shard veri di scripts/shard_parquet.py, run4 vero di
scripts/pixmo_transcript_check.py, indice e pannello finti con tutti i ruoli. Nessun dato reale. Solo in un job SLURM.

Rischi coperti: collegamento originale/shard e righe escluse dallo sharding; immagine identica (pixel dell'originale e
del JPEG dello shard) e copia ricodificata (resta candidata); somiglianza pHash che resta candidata; testo in comune,
anche ripetuto in 27 righe; immagine originale mancante o illeggibile, parquet e shard mancanti; dettaglio assente dalle
trascrizioni che resta un segnale; manomissioni dello shard, di run4 e delle liste storiche."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("SLURM_JOB_ID", "test")
import pixmo_inventory as pi  # noqa: E402
from limite_vl import decontam as dc  # noqa: E402
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.data import DETAILED_CAPTION_PROMPTS, pick_prompt  # noqa: E402
from limite_vl.processing import to_rgb  # noqa: E402
from limite_vl.provenance import sha256_file  # noqa: E402

REV = "0123456789abcdef0123456789abcdef01234567"
FAILS, DONE = [], []
SPECIFIC = "What is the total length of the red fence around the vegetable garden in meters?"
SPECIFIC2 = "How many yellow umbrellas are open on the crowded beach next to the old lighthouse?"
BOILER = "Hint: Please answer the question and provide the correct option letter at the end."
SCHEMA = pa.schema([("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                    ("image_url", pa.string()), ("caption", pa.string()), ("transcripts", pa.list_(pa.string()))])


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    DONE.append(name)
    if not cond:
        FAILS.append(name)


def picture(seed: int, size: tuple[int, int]) -> Image.Image:
    """Rettangoli colorati su fondo colorato, diversi per seme: pHash informativo."""
    rng = random.Random(seed)
    im = Image.new("RGB", size, tuple(rng.randrange(256) for _ in range(3)))
    d = ImageDraw.Draw(im)
    w, h = size
    for _ in range(12):
        x0, y0 = rng.randrange(w), rng.randrange(h)
        d.rectangle((x0, y0, x0 + rng.randrange(4, w // 2 + 5), y0 + rng.randrange(4, h // 2 + 5)),
                    fill=tuple(rng.randrange(256) for _ in range(3)))
    return im


def encode(im: Image.Image, fmt: str = "JPEG", **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format=fmt, **kw)
    return buf.getvalue()


def png_of_pixels(data: bytes) -> bytes:
    """Gli stessi pixel normalizzati (rgb-su-bianco/1) in un PNG: identità dei pixel, byte diversi."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return encode(to_rgb(im), "PNG")


def row(image: bytes | None, caption, transcripts, url) -> dict:
    return {"image": {"bytes": image, "path": None}, "image_url": url, "caption": caption, "transcripts": transcripts}


def blob(base: Path, rel: str, data: bytes, git: bool = False) -> Path:
    """File dello snapshot come in hf/hub: blob col nome dello sha256 (o del git sha1) e link simbolico relativo."""
    name = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() if git else hashlib.sha256(data).hexdigest()
    (base / "blobs").mkdir(parents=True, exist_ok=True)
    (base / "blobs" / name).write_bytes(data)
    link = base / "snapshots" / REV / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(base / "blobs" / name, link.parent))
    return link


def parquet_bytes(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), buf, row_group_size=4)
    return buf.getvalue()


def rows_of_fixture() -> tuple[list[dict], list[dict], dict]:
    """Righe dei due parquet e immagini da usare nel pannello."""
    r0 = encode(picture(1, (640, 480)), quality=90)
    r1 = encode(picture(2, (500, 300)), quality=90)
    r2 = encode(picture(3, (800, 600)), quality=92)
    r11 = encode(picture(5, (900, 700)), quality=90)
    rgba = picture(6, (300, 200)).convert("RGBA")
    px = rgba.load()
    for x in range(100):
        for y in range(80):
            px[x, y] = (10, 20, 30, 0)  # trasparenza: sul bianco nella convenzione dell'indice
    r12 = encode(rgba, "PNG")
    exif = Image.Exif()
    exif[pi.EXIF_ORIENTATION] = 6
    r13 = encode(picture(7, (400, 300)), quality=90, exif=exif.tobytes())
    tr = ["there's a red car parked next to three tall trees on a quiet street at noon"]
    rows0 = [
        row(r0, "A red car is parked next to three tall trees on a quiet street at noon.", tr, "http://a.example/0.jpg"),
        row(r1, "A blue boat floats on a calm lake under a cloudy sky.", ["a blue boat on a lake"], "http://b.example/1"),
        row(r2, "Two green chairs stand on a terrace.", ["two green chairs on a terrace"], "http://c.example/2"),
        row(encode(picture(4, (640, 400)), quality=90),
            f"A sign in the garden reads: {SPECIFIC} Below it there is a wooden bench.",
            ["a sign in a garden with a question and a bench"], "http://d.example/3"),
        row(None, "An image that was never downloaded.", ["nothing"], "http://e.example/4"),
        row(b"<html>not an image</html>", "A broken file.", ["broken"], "http://f.example/5"),
        row(encode(picture(8, (512, 512)), quality=90), "Seven cats sit on a wooden porch while two dogs sleep nearby.",
            ["some cats sit on a wooden porch and two dogs sleep nearby"], "http://g.example/6"),
        row(encode(picture(9, (320, 240)), quality=90), None, ["a caption that is missing"], "http://h.example/7"),
        row(encode(picture(10, (320, 240)), quality=90), "   ", ["only spaces"], "http://i.example/8"),
        row(encode(picture(11, (320, 240)), quality=90), "  A bowl of fruit on a table.  ", ["a bowl of fruit"],
            "http://j.example/9"),
        row(r0, "A different caption for the same downloaded image of a street.", ["a street"], "http://k.example/10"),
        row(r11, "A yellow kite in the wind above a field.", ["a yellow kite"], "http://l.example/11"),
        row(r12, "A transparent sticker on a white page.", ["a sticker"], "http://m.example/12"),
        row(r13, "A rotated photo of a door.", ["a door"], "http://n.example/13"),
        row(encode(picture(12, (400, 300)), quality=90), f"Question 7 of the quiz. {BOILER}", ["a quiz"],
            "http://o.example/14"),
    ]
    rows1 = [row(encode(picture(100 + i, (96, 64)), quality=90), f"A poster asks: {SPECIFIC2}", ["a poster"],
                 f"http://p.example/{i}") for i in range(27)]
    rows1 += [row(encode(picture(200, (300, 300)), quality=90), "A plain wall.", ["a wall"], "http://q.example/27"),
              row(encode(picture(201, (300, 300)), quality=90), "A plain floor.", ["a floor"], "http://q.example/28")]
    return rows0, rows1, {"r1": r1, "r2": r2, "r11": r11, "r12": r12}


def key(f: int, r: int) -> str:
    return f"pixmo-{f:05d}-{r:06d}"


def shard_members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}


def rewrite_tar(src: Path, dst: Path, edit) -> None:
    """Copia del tar con i membri trasformati da edit(nome, byte) -> byte o None (membro tolto)."""
    with tarfile.open(src) as tin, tarfile.open(dst, "w", format=tarfile.USTAR_FORMAT) as tout:
        for m in tin:
            data = edit(m.name, tin.extractfile(m).read())
            if data is None:
                continue
            info = tarfile.TarInfo(m.name)
            info.size = len(data)
            tout.addfile(info, io.BytesIO(data))


def panel_fixture(tmp: Path, entries: list[dict]) -> tuple[Path, Path, str]:
    items, elements = [], {}
    for e in entries:
        img = e.get("image")
        h, pixels, size = dc.hash_image_bytes(img) if img else (None, None, None)
        items.append({"benchmark": e["benchmark"], "file": "data/x.parquet", "item": e["item"], "id": e["item"],
                      "pos": 0 if img else None, "phash": None if h is None else f"{h:016x}", "pixel_sha256": pixels,
                      "size": size, "text": e.get("text", ""), "hints": {}})
        k = f"{e['benchmark']}/{e['item']}"
        elements[k] = {"chiave": k, "benchmark": e["benchmark"], "ruolo": e["ruolo"],
                       "gruppo": hashlib.sha256(k.encode()).hexdigest(), "metrica_primaria": e.get("metrica")}
    index_dir, panel_dir = tmp / "indice", tmp / "pannello"
    index_dir.mkdir()
    panel_dir.mkdir()
    (index_dir / "index.json").write_text(json.dumps({"schema": "panel-index/2", "convenzione_immagini": dc.IMAGE_CONVENTION,
                                                      "items": items}))
    (index_dir / "verifica.json").write_text(json.dumps({
        "verificato": True, "errori": 0, "voci_con_immagine": sum(it["pos"] is not None for it in items),
        "convenzione_immagini": dc.IMAGE_CONVENTION}))
    (panel_dir / "elements.jsonl.gz").write_bytes(gzip.compress(
        "".join(json.dumps(x) + "\n" for x in elements.values()).encode(), mtime=0))
    sha = sha256_file(index_dir / "index.json")
    (panel_dir / "summary.json").write_text(json.dumps({"indice": {"sha256": sha}, "elementi": len(elements)}))
    return index_dir, panel_dir, sha


def build(tmp: Path) -> dict:
    base = tmp / "hub" / "datasets--anthracite-org--pixmo-cap-images"
    rows0, rows1, imgs = rows_of_fixture()
    p0 = blob(base, "data/train-00000-of-00075.parquet", parquet_bytes(rows0))
    p1 = blob(base, "data/train-00001-of-00075.parquet", parquet_bytes(rows1))
    blob(base, "README.md", b"---\ndataset_info:\n  splits:\n  - name: train\n    num_examples: 200\n"
                            b"  download_size: 123456\n---\n# PixMo-Cap (fixture)\n", git=True)
    snapshot = base / "snapshots" / REV
    shards = tmp / "shards"
    env = {**os.environ, "SLURM_JOB_ID": "test"}
    done = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/shard_parquet.py"), "--kind", "pixmo", "--files",
                           str(p0), str(p1), "--out", str(shards), "--source", f"{pi.REPO}@{REV}", "--prefix",
                           "pixmo-cap", "--files-per-shard", "1", "--workers", "1"], env=env, capture_output=True,
                          text=True)
    if done.returncode:
        raise RuntimeError(f"shard_parquet.py: {done.stderr[-2000:]}")
    run4 = tmp / "run4"
    done = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/pixmo_transcript_check.py"), "--shards-jsonl",
                           str(shards / "shards.jsonl"), "--out", str(run4)], env=env, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"pixmo_transcript_check.py: {done.stderr[-2000:]}")
    members = shard_members(shards / "pixmo-cap-00000.tar")
    shard_r11 = members[f"{key(0, 11)}.jpg"]
    reenc = encode(to_rgb(Image.open(io.BytesIO(imgs["r2"]))), quality=40)
    entries = [
        {"benchmark": "mmstar", "item": "test:1", "image": png_of_pixels(imgs["r1"]), "text": "What is in the lake?",
         "ruolo": "test", "metrica": True},
        {"benchmark": "muirbench", "item": "test:2", "image": reenc, "text": "Which chair is closer?", "ruolo": "test",
         "metrica": False},
        {"benchmark": "mathvista", "item": "testmini:3", "image": encode(picture(900, (50, 40)), "PNG"),
         "text": SPECIFIC, "ruolo": "sviluppo"},
        {"benchmark": "docvqa", "item": "val:4", "image": png_of_pixels(shard_r11), "text": "What is the date?",
         "ruolo": "diagnostica"},
        {"benchmark": "ocrbench", "item": "test:5", "text": SPECIFIC2, "ruolo": "solo_decontaminazione"},
    ]
    entries += [{"benchmark": "geoperception", "item": f"test:{100 + i}", "text": f"Item {i} asks about figure {i}. {BOILER}",
                 "ruolo": "sviluppo"} for i in range(26)]
    index_dir, panel_dir, isha = panel_fixture(tmp, entries)
    hist = []
    for name, keys in (("p6", [key(0, 2)]), ("f1eval", [key(0, 0)])):
        d = tmp / "storiche" / name / "pixmo"
        d.mkdir(parents=True)
        (d / "exclude.txt").write_text("".join(k + "\n" for k in keys))
        (d / "exclude_final.txt").write_text("")
        (d / "summary.json").write_text(json.dumps({"kind": "pixmo", "index": f"data/decontam/{name}/index.json"}))
        hist.append(d)
    tsv = tmp / "report-verify.tsv"
    lines = ["status\tdest\tsize\texpected\tlabel"]
    for p in (p0, p1):
        b = p.resolve()
        lines.append(f"ok\t{b}\t{b.stat().st_size}\tsha256:{b.name}\t{pi.REPO}@{REV[:8]}:data/{p.name}")
    tsv.write_text("\n".join(lines) + "\n")
    return {"snapshot": snapshot, "shards": shards, "run4": run4, "index": index_dir, "panel": panel_dir, "isha": isha,
            "historical": hist, "tsv": tsv, "imgs": imgs, "rows0": rows0, "rows1": rows1, "reenc": reenc,
            "shard_r11": shard_r11}


def scan_args(fx: dict, out: Path, **over) -> list[str]:
    a = {"snapshot": fx["snapshot"], "shards": fx["shards"], "index": fx["index"], "index-sha256": fx["isha"],
         "inventory": out / "inventario", "scan": out / "scansione", **over}
    argv = ["scan", "--revision", REV, "--workers", "2"]
    for k, v in a.items():
        argv += [f"--{k}", str(v)]
    return argv


def verify_args(fx: dict, out: Path, report: Path, **over) -> list[str]:
    a = {"snapshot": fx["snapshot"], "shards": fx["shards"], "index": fx["index"], "index-sha256": fx["isha"],
         "inventory": out / "inventario", "scan": out / "scansione", "panel": fx["panel"], "run4": fx["run4"],
         "verify-report": fx["tsv"], "report": report, **over}
    argv = ["verifica", "--revision", REV]
    for k, v in a.items():
        if k != "historical":
            argv += [f"--{k}", str(v)]
    argv += ["--historical", *[str(h) for h in over.get("historical", fx["historical"])]]
    return argv


def failed(report: Path) -> list[str]:
    path = report / "report.json"
    return json.loads(path.read_text())["falliti"] if path.is_file() else ["(verifica fermata prima del report)"]


def by_key(path: Path) -> dict[str, dict]:
    return {r["chiave"]: r for r in pi.read_gz(path)}


def test_pure() -> None:
    check("chiavi: key_order legge file e riga, rifiuta le altre", pi.key_order(key(3, 17)) == (3, 17)
          and pi.key_order("pixmo-3-17") is None and pi.key_order(None) is None)
    check("sharding: immagine mancante, didascalia mancante, entrambe, illeggibile, nello shard",
          [pi.expected_sharding(None, "c", {"lettura": "assente"}, None),
           pi.expected_sharding(b"x", "", {"lettura": "letta"}, None),
           pi.expected_sharding(b"", None, {"lettura": "assente"}, None),
           pi.expected_sharding(b"x", "c", {"lettura": "illeggibile"}, None),
           pi.expected_sharding(b"x", "c", {"lettura": "letta"}, "OSError"),
           pi.expected_sharding(b"x", "c", {"lettura": "letta"}, None)]
          == ["scartata:missing_image", "scartata:missing_caption", "scartata:missing_image_caption",
              "scartata:illeggibile", "scartata:illeggibile", "nello_shard"])
    check("originale: byte vuoti come immagine mancante", pi.original_check(b"")[0] == {"lettura": "assente"})
    full = {c: True for c in pi.REQUIRED}
    rows = [({"pixel": 1, "phash": 2}, full), ({"phash": 1}, full), ({"testo": 3}, full), ({}, {**full, "pixel_shard": False}),
            ({}, full), ({"phash": 1}, {**full, "phash_shard": False})]
    check("esiti: pixel -> confermata, pHash o testo -> candidato, controllo mancante -> non verificabile",
          [pi.outcome({"corrispondenze": c, "controlli": k}) for c, k in rows]
          == ["corrispondenza_confermata", "candidato", "candidato", "non_verificabile", "nessuna_corrispondenza",
              "candidato"])
    data = encode(picture(3, (120, 90)), "PNG")
    o, norm = pi.original_check(data)
    h, px, size = dc.hash_image_bytes(data)
    check("originale: pixel e pHash uguali a decontam.hash_image_bytes (stessa convenzione)",
          o["pixel_sha256"] == px and o["phash"] == f"{h:016x}" and o["dimensioni"] == size and norm.mode == "RGB")


def test_scan(tmp: Path, fx: dict) -> None:
    out = tmp / "pulito"
    check("scan: uscita 0", pi.main(scan_args(fx, out)) == 0)
    report = out / "verifica"
    code = pi.main(verify_args(fx, out, report))
    check("verifica: tutti i controlli superati sulla fixture pulita", code == 0, failed(report) if code else "")
    rep = json.loads((report / "report.json").read_text())
    inv = by_key(out / "inventario" / "records.jsonl.gz")
    scan = by_key(out / "scansione" / "records.jsonl.gz")
    rows0, rows1 = fx["rows0"], fx["rows1"]
    check("inventario: una riga per riga dei parquet, nell'ordine dei file e delle righe",
          list(inv) == [key(0, i) for i in range(len(rows0))] + [key(1, i) for i in range(len(rows1))])
    k = key(0, 0)
    r = inv[k]
    check("collegamento: file, riga, gruppo di righe, shard e posizione della prima riga",
          r["parquet"] == {"file": "data/train-00000-of-00075.parquet", "indice_file": 0, "riga": 0,
                           "gruppo_di_righe": 0, "riga_nel_gruppo": 0}
          and r["shard"]["nome"] == "pixmo-cap-00000.tar" and r["shard"]["posizione"] == 0
          and sorted(r["shard"]["membri"]) == [f"{k}.jpg", f"{k}.json"])
    r9 = inv[key(0, 9)]
    check("collegamento: gruppo di righe e posizione dopo gli scarti (riga 9: gruppo 2, settimo record dopo tre scarti)",
          r9["parquet"]["gruppo_di_righe"] == 2 and r9["parquet"]["riga_nel_gruppo"] == 1
          and r9["shard"]["posizione"] == 6, r9["shard"])
    members = shard_members(fx["shards"] / "pixmo-cap-00000.tar")
    meta = json.loads(members[f"{k}.json"])
    check("collegamento: sha256 di JSON e JPEG uguali ai membri dello shard",
          r["shard"]["json_sha256"] == hashlib.sha256(members[f"{k}.json"]).hexdigest()
          and r["shard"]["jpg_sha256"] == hashlib.sha256(members[f"{k}.jpg"]).hexdigest())
    check("originale: sha256 dei byte e dimensioni della riga del parquet",
          r["originale"]["byte_sha256"] == hashlib.sha256(rows0[0]["image"]["bytes"]).hexdigest()
          and r["originale"]["dimensioni"] == [640, 480] and r["originale"]["formato"] == "JPEG")
    check("trasformazione: dimensioni a 1024 patch del JSON, byte del JPEG rifatti uguali allo shard",
          r["trasformazione"]["dimensioni_attese"] == [meta["width"], meta["height"]]
          and r["trasformazione"]["byte_uguali_allo_shard"] is True
          and all(x["trasformazione"]["byte_uguali_allo_shard"] for x in inv.values() if x.get("shard")))
    check("trasformazione: copia ridimensionata, pixel diversi dall'originale ma pHash vicino",
          r["immagine_shard"]["pixel_sha256"] != r["originale"]["pixel_sha256"]
          and r["trasformazione"]["phash_distanza_originale_shard"] <= 4)
    check("didascalia: quella del loader è il campo caption del JSON dello shard",
          r["didascalia"]["loader_sha256"] == mf.sha256_text(meta["caption"]) and r["didascalia"]["parole"] == 16)
    check("record_id di S1 §1 e richiesta del loader",
          r["record_id"] == mf.record_id(mf.Digests(), "pixmo-cap", REV, k)
          and DETAILED_CAPTION_PROMPTS[r["richiesta"]] == pick_prompt(k, DETAILED_CAPTION_PROMPTS))
    # scarti dello sharding e difetti
    r4, r5, r7, r8 = inv[key(0, 4)], inv[key(0, 5)], inv[key(0, 7)], inv[key(0, 8)]
    check("scarti: immagine mancante e didascalia mancante, senza record nello shard",
          r4["esito_sharding"] == "scartata:missing_image" and r4["shard"] is None
          and r4["originale"]["lettura"] == "assente" and r7["esito_sharding"] == "scartata:missing_caption"
          and r7["shard"] is None and "didascalia originale assente" in r7["difetti"])
    check("scarti: immagine illeggibile come nello sharding, controlli non eseguibili",
          r5["esito_sharding"] == "scartata:illeggibile" and r5["originale"]["lettura"] == "illeggibile"
          and "immagine originale illeggibile" in r5["difetti"] and r5["originale"]["inizio_hex"]
          and pi.outcome(scan[key(0, 5)]) == "non_verificabile")
    check("difetto: didascalia di soli spazi vuota nel loader",
          r8["shard"] is not None and "didascalia del loader vuota" in r8["difetti"]
          and "didascalia originale solo_spazi" in r8["difetti"])
    r9d = r9["didascalia"]
    check("didascalia: spazi ai bordi tolti dal loader, non un difetto",
          r9d["spazi_ai_bordi"] is True and r9d["loader_sha256"] == mf.sha256_text("A bowl of fruit on a table.")
          and not r9["difetti"])
    check("scarti: conteggi della verifica uguali a quelli dello sharding",
          rep["inventario"]["esito_sharding"] == {"nello_shard": 41, "scartata:illeggibile": 1,
                                                  "scartata:missing_caption": 1, "scartata:missing_image": 1},
          rep["inventario"]["esito_sharding"])
    # convenzione, EXIF, duplicati
    r12 = inv[key(0, 12)]
    h, px, _ = dc.hash_image_bytes(fx["imgs"]["r12"])
    check("convenzione: PNG trasparente normalizzato su bianco come nell'indice",
          r12["originale"]["pixel_sha256"] == px and r12["originale"]["phash"] == f"{h:016x}"
          and r12["originale"]["modo"] == "RGBA")
    check("originale: orientamento EXIF registrato", inv[key(0, 13)]["originale"].get("exif_orientamento") == 6)
    dup = [g for g in pi.read_gz(report / "duplicati.jsonl.gz") if g["campo"] == "byte_originale"]
    check("duplicati: stessa immagine in due righe con URL e didascalie diverse",
          len(dup) == 1 and dup[0]["chiavi"] == [key(0, 0), key(0, 10)] and dup[0]["url_distinti"] == 2
          and dup[0]["didascalie_distinte"] == 2, dup)
    # decontaminazione
    s1, s2, s3, s11 = scan[key(0, 1)], scan[key(0, 2)], scan[key(0, 3)], scan[key(0, 11)]
    check("pixel identici sull'originale (PNG con gli stessi pixel): corrispondenza confermata",
          pi.outcome(s1) == "corrispondenza_confermata"
          and any(d["tipo"] == "pixel" and d["immagine"] == "originale" and d["elemento"] == "mmstar/test:1"
                  for d in s1["dettagli"]))
    check("pixel identici sul JPEG dello shard: corrispondenza confermata, non sull'originale",
          pi.outcome(s11) == "corrispondenza_confermata"
          and any(d["tipo"] == "pixel" and d["immagine"] == "shard" and d["elemento"] == "docvqa/val:4"
                  for d in s11["dettagli"])
          and not any(d["tipo"] == "pixel" and d["immagine"] == "originale" for d in s11["dettagli"]))
    o2 = int(s2["originale"]["phash"], 16)
    re_h, re_px, _ = dc.hash_image_bytes(fx["reenc"])
    check("fixture: la copia ricodificata ha pixel diversi e pHash entro 4 bit",
          re_px != s2["originale"]["pixel_sha256"] and dc.hamming(o2, re_h) <= 4)
    check("copia ricodificata: somiglianza pHash, resta candidata (nessun pixel identico)",
          pi.outcome(s2) == "candidato" and not s2["corrispondenze"].get("pixel")
          and any(d["tipo"] == "phash" and d["elemento"] == "muirbench/test:2" for d in s2["dettagli"]))
    grams = [" ".join(dc.normalize_words(SPECIFIC)[i:i + 10]) for i in range(len(dc.normalize_words(SPECIFIC)) - 9)]
    texts = [d for d in s3["dettagli"] if d["tipo"] == "testo"]
    check("testo in comune: candidato, con tutti i 10-gram conservati",
          pi.outcome(s3) == "candidato" and len(texts) == 1 and texts[0]["elemento"] == "mathvista/testmini:3"
          and texts[0]["ngram_tutti"] == grams and s3["ngram_in_comune"] == sorted(grams), texts)
    rep27 = [scan[key(1, i)] for i in range(27)]
    check("testo ripetuto in 27 righe: ognuna resta candidata (la frequenza nella sorgente non decide)",
          all(pi.outcome(s) == "candidato" and any(d["elemento"] == "ocrbench/test:5" for d in s["dettagli"])
              for s in rep27))
    check("testo di rito del pannello (in più di 25 testi dell'indice): nessuna corrispondenza testuale",
          not scan[key(0, 14)]["corrispondenze"].get("testo"))
    r6, s6 = inv[key(0, 6)], scan[key(0, 6)]
    check("dettaglio assente dalle trascrizioni: resta un segnale, non un difetto né un'esclusione",
          r6["segnali"]["tipi"] == ["numero_conteggio_2_12"] and r6["segnali"]["numbers"] == ["7"]
          and not r6["difetti"] and not s6["corrispondenze"].get("testo"))
    props = {p["chiave"]: p for p in pi.read_gz(report / "proposte_di_trattamento.jsonl.gz")}
    check("proposte: escludere le identità confermate, quarantena per i candidati, niente per segnali e scarti",
          props[key(0, 1)]["proposta"] == "escludere" and props[key(0, 1)]["ruoli"] == ["test_ammesso"]
          and props[key(0, 11)]["proposta"] == "escludere" and props[key(0, 11)]["ruoli"] == ["diagnostica"]
          and props[key(0, 2)]["proposta"].startswith("quarantena")
          and props[key(0, 3)]["proposta"].startswith("quarantena")
          and props[key(0, 8)]["proposta"] == "escludere o correggere"
          and key(0, 6) not in props and key(0, 4) not in props and key(0, 5) not in props, sorted(props))
    hist = {Path(h["cartella"]).parent.name: h for h in rep["candidati_storici"]}
    check("storiche: chiavi risolte e confrontate con gli esiti nuovi (diagnostico)",
          hist["p6"]["esiti_nuovi_dei_candidati_storici"] == {"candidato": 1}
          and hist["f1eval"]["candidati_storici"] == 1)
    check("run4: riprodotto riga per riga dal codice corrente", rep["run4"]["righe_confrontate"] == len(inv)
          and not [c for c in rep["falliti"] if c.startswith("run4")])
    check("costi registrati: secondi, memoria, byte letti", all(
        k in rep["scansione"]["costo"] for k in ("secondi", "rss_max_mib", "byte_parquet_letti_per_lo_sha256",
                                                 "byte_shard_letti", "secondi_cpu_per_fase")))
    check("copertura: righe lette su esempi dichiarati dal README", rep["copertura_del_sottoinsieme_locale"][
        "esempi_dichiarati_dal_readme"] == 200 and rep["copertura_del_sottoinsieme_locale"]["file_del_dataset"] == 75)
    # sonda: un solo parquet
    probe = tmp / "sonda"
    check("sonda: --files 1", pi.main(scan_args(fx, probe) + ["--files", "1"]) == 0)
    code = pi.main(verify_args(fx, probe, probe / "verifica") + ["--partial"])
    check("sonda: verifica parziale superata", code == 0, failed(probe / "verifica") if code else "")
    # run4 e liste storiche alterate, sulle stesse uscite
    run4 = tmp / "run4-alterato"
    shutil.copytree(fx["run4"], run4)
    lines = (run4 / "flagged.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["risks"] = sorted(first["risks"] + ["numero_anno"])  # un tipo che la riga non ha: il valore cambia davvero
    (run4 / "flagged.jsonl").write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    code = pi.main(verify_args(fx, out, tmp / "v-run4", run4=run4))
    check("run4 alterato: la riproduzione riga per riga lo trova", code == 1
          and any(c.startswith("run4: segnali ricalcolati") for c in failed(tmp / "v-run4")))
    bad = tmp / "storiche-alterate" / "p6" / "pixmo"
    shutil.copytree(fx["historical"][0], bad)
    (bad / "exclude.txt").write_text(f"{key(0, 2)}\n{key(0, 999)}\n")
    code = pi.main(verify_args(fx, out, tmp / "v-storiche", historical=[bad, fx["historical"][1]]))
    check("lista storica con una chiave senza riga: errore dell'import", code == 1
          and any(c.startswith("storiche") for c in failed(tmp / "v-storiche")))


def test_tampering(tmp: Path, fx: dict) -> None:
    def shards_with(name: str, edit) -> Path:
        dst = tmp / name
        shutil.copytree(fx["shards"], dst, ignore=shutil.ignore_patterns("*.tar"))
        for t in sorted(fx["shards"].glob("*.tar")):
            if t.name == "pixmo-cap-00000.tar" and edit is not None:
                rewrite_tar(t, dst / t.name, edit)
            else:
                shutil.copy(t, dst / t.name)
        return dst

    gone = shards_with("shard-senza-riga", lambda n, d: None if n.startswith(key(0, 9)) else d)
    out = tmp / "senza-riga"
    check("shard senza un record: scan completa", pi.main(scan_args(fx, out, shards=gone)) == 0)
    inv = by_key(out / "inventario" / "records.jsonl.gz")
    code = pi.main(verify_args(fx, out, out / "verifica", shards=gone))
    f = failed(out / "verifica")
    check("riga senza record nello shard: difetto, verifica fallita (sha256 e collegamento)",
          inv[key(0, 9)]["esito_sharding"] == "mancante_nello_shard"
          and "riga senza record nello shard" in inv[key(0, 9)]["difetti"] and code == 1
          and any(c.startswith("shard: letti per intero") for c in f)
          and any(c.startswith("collegamento: JSON, membri") for c in f), f)
    props = {p["chiave"]: p for p in pi.read_gz(out / "verifica" / "proposte_di_trattamento.jsonl.gz")}
    check("riga senza record: proposta di rigenerare lo shard", props[key(0, 9)]["proposta"].startswith("rigenerare"))

    def caption_changed(n, d):
        if n == f"{key(0, 0)}.json":
            meta = json.loads(d)
            meta["caption"] = "A completely different caption."
            return json.dumps(meta).encode()
        return d

    altered = shards_with("shard-json-alterato", caption_changed)
    out = tmp / "json-alterato"
    pi.main(scan_args(fx, out, shards=altered))
    inv = by_key(out / "inventario" / "records.jsonl.gz")
    code = pi.main(verify_args(fx, out, out / "verifica", shards=altered))
    check("JSON dello shard con un'altra didascalia: difetto di collegamento",
          any(d.startswith("JSON dello shard diverso dalla riga in caption") for d in inv[key(0, 0)]["difetti"])
          and code == 1 and any(c.startswith("collegamento: JSON, membri") for c in failed(out / "verifica")))
    # ingressi mancanti: arresto chiaro, nessuna uscita
    snap2 = tmp / "snapshot-parziale" / "snapshots" / REV
    (snap2 / "data").mkdir(parents=True)
    for rel in ("data/train-00000-of-00075.parquet", "README.md"):
        (snap2 / rel).symlink_to((fx["snapshot"] / rel).resolve())
    out = tmp / "senza-parquet"
    check("parquet mancante: arresto prima di leggere", pi.main(scan_args(fx, out, snapshot=snap2)) == 2
          and not (out / "inventario" / "records.jsonl.gz").exists())
    no_tar = tmp / "shard-mancante"
    shutil.copytree(fx["shards"], no_tar, ignore=shutil.ignore_patterns("pixmo-cap-00001.tar"))
    out = tmp / "senza-shard"
    check("shard mancante: arresto prima di leggere", pi.main(scan_args(fx, out, shards=no_tar)) == 2
          and not (out / "inventario" / "records.jsonl.gz").exists())
    out = tmp / "indice-diverso"
    check("indice con un altro sha256: arresto", pi.main(scan_args(fx, out, **{"index-sha256": "0" * 64})) == 2)
    out = tmp / "pulito"
    check("uscite esistenti: nessuna sovrascrittura", pi.main(scan_args(fx, out)) == 2)


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_pure()
        fx = build(tmp)
        test_scan(tmp, fx)
        test_tampering(tmp, fx)
    print(f"{len(DONE) - len(FAILS)} controlli su {len(DONE)}")
    if FAILS:
        print("falliti: " + "; ".join(FAILS))
        sys.exit(1)


if __name__ == "__main__":
    main()
