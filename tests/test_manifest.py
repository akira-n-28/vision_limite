#!/usr/bin/env python3
"""Test del manifest delle ammissioni (limite_vl.manifest, docs/specs/S1_RECORD_MANIFEST.md §§8-9) e dei reader che lo
applicano (limite_vl.stream), su shard e manifest piccoli costruiti qui. Importa torch (stream) e carica il tokenizer di
Violetto: solo in un job (slurm/tests_cpu.sbatch)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl import manifest as mf  # noqa: E402
from limite_vl.data_lists import provenance  # noqa: E402
from limite_vl.stream import MixedRowStream, SourceReader, make_sources, validation_samples  # noqa: E402

VIOLETTO = ROOT / "hf/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, exc=mf.ManifestError) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception as other:  # noqa: BLE001 - un errore di altro tipo non è quello atteso
        print(f"     errore inatteso: {type(other).__name__}: {other}")
        return False
    return False


def test_identity():
    a = mf.canonical({"b": 1, "a": [1, "è", None, True, {}, []], "c": ""})
    b = mf.canonical({"c": "", "a": [1, "è", None, True, {}, []], "b": 1})
    check("forma canonica: stesso contenuto con chiavi in ordine diverso, stessa stringa", a == b)
    check("forma canonica: andata e ritorno, Unicode non sostituito, nessuno spazio",
          json.loads(a) == {"b": 1, "a": [1, "è", None, True, {}, []], "c": ""} and "è" in a and " " not in a, a)
    check("contenuti diversi, digest diversi", mf.sha256_text(a) != mf.sha256_text(mf.canonical({"b": 2})))
    check("numeri non interi rifiutati nella forma canonica", raises(lambda: mf.canonical({"x": 0.5}), TypeError))
    # collisione forzata: digest troncato a 8 caratteri, due record distinti con lo stesso prefisso (S1 §8)
    first, i = {}, 0
    while True:
        p = mf.sha256_text(mf.canonical({"record": str(i)}))[:8]
        if p in first:
            break
        first[p] = i
        i += 1
    d = mf.Digests(fn=lambda text: mf.sha256_text(text)[:8])
    d({"record": str(first[p])})
    check("collisione forzata con il digest troncato: errore, nessuna fusione",
          raises(lambda: d({"record": str(i)})) and len(d.seen) == 1, (first[p], i))
    check("stesso contenuto due volte: nessuna collisione", d({"record": str(first[p])}) == p)
    full = mf.Digests()
    rid = mf.record_id(full, "synth", "v4@" + "0" * 40, "synth-0000001")
    check("prefissi: un identificatore breve non vale come chiave", raises(lambda: mf.require_digest(rid[:8], "id"))
          and raises(lambda: mf.group_id(full, "indice", [rid[:12]])) and mf.require_digest(rid, "id") == rid)


def test_assign_splits():
    groups = {}
    sizes = Counter()
    for fam in ("coord", "function"):
        for n in range(400):
            g = mf.sha256_text(f"{fam}/{n}")
            size = 1 + (n * 7) % 5
            groups[g] = {"famiglia": fam, "valutazione": n % 50 == 0, "ammessi": 0 if n % 97 == 1 else size}
            sizes[fam] += 0 if groups[g]["valutazione"] else groups[g]["ammessi"]
    s1, s2 = mf.assign_splits(groups, 0.05, 0), mf.assign_splits(groups, 0.05, 0)
    other = mf.assign_splits(groups, 0.05, 1)
    check("split: deterministico con lo stesso seme, diverso con un altro", s1 == s2 and s1 != other)
    check("split: i gruppi con record di valutazione vanno in test",
          all(s1[g] == "test" for g, i in groups.items() if i["valutazione"])
          and all(s1[g] != "test" for g, i in groups.items() if not i["valutazione"]))
    check("split: un gruppo senza scene ammesse resta in train",
          all(s1[g] == "train" for g, i in groups.items() if not i["ammessi"] and not i["valutazione"]))
    ok = []
    for fam in ("coord", "function"):
        dev = sum(i["ammessi"] for g, i in groups.items() if i["famiglia"] == fam and s1[g] == "dev")
        target = round(0.05 * sizes[fam])
        ok.append(target <= dev < target + 5)
    check("split: dev per famiglia almeno all'obiettivo, superato al più da un gruppo", all(ok), ok)
    check("split: quota fuori da [0, 1) rifiutata", raises(lambda: mf.assign_splits(groups, 1.0, 0), ValueError))


def png(color: int = 255, size: int = 448) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), (color, color, color)).save(buf, format="PNG")
    return buf.getvalue()


def write_tar(path: Path, items: list[tuple[str, bytes, bytes]]) -> None:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for key, js, image in items:
            for name, data in ((f"{key}.png", image), (f"{key}.json", js)):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))


def scene(key: str, n: int) -> bytes:
    qa = [{"type": "count", "q": f"How many shapes are in figure {n}?", "options": ["1", "2"], "answer": "1"},
          {"type": "exists", "q": f"Is there a shape labeled A{n}?", "options": ["yes", "no"], "answer": "yes"}]
    return json.dumps({"id": key, "family": "shapes", "width": 448, "height": 448, "max_patches": 784, "qa": qa,
                       "spec": {"family": "shapes", "n": n}}).encode()


def build_tiny(tmp: Path) -> dict:
    """Due shard di train da 6 record e uno di test da 2, con il loro manifest: gruppi (t0-t1 e t6-t7 per stato
    comune, t8 con un record di test), stati (t2 in quarantena, t3 escluso per contaminazione con s0) e split."""
    image = png()
    keys = {"train-a": [f"t{i}" for i in range(6)], "train-b": [f"t{i}" for i in range(6, 12)], "test": ["s0", "s1"]}
    group_of = {"t0": "g01", "t1": "g01", "t6": "g67", "t7": "g67", "t3": "gs0", "s0": "gs0", "t8": "gs1", "s1": "gs1"}
    state = {"t2": "quarantena", "t3": "escluso", "t8": "escluso"}
    dev = {"g4", "g67", "g10"}
    folder = tmp / "manifest"
    shards, lines, info = [], [], {}
    digests = mf.Digests()
    blobs = {}
    for name, ks in keys.items():
        items = [(k, scene(k, n), image) for n, k in enumerate(ks)]
        path = tmp / f"{name}.tar"
        write_tar(path, items)
        blobs.update({k: js for k, js, _ in items})
        shards.append({"insieme": "test" if name == "test" else "train", "nome": path.name, "percorso": str(path),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                       "record": len(ks)})
    gid = {}
    for name, ks in keys.items():
        for k in ks:
            g = group_of.get(k, "g" + k[1:])
            gid.setdefault(g, mf.sha256_text(g))
    for s, (name, ks) in zip(shards, keys.items()):
        for index, k in enumerate(ks):
            g = group_of.get(k, "g" + k[1:])
            split = "test" if g.startswith("gs") else "dev" if g in dev else "train"
            rec = {"record_id": mf.record_id(digests, "synth", "v4@" + "0" * 40, k), "sorgente": "synth",
                   "revisione": "v4@" + "0" * 40, "record": k, "insieme": s["insieme"], "shard": s["percorso"],
                   "indice": index, "famiglia": "shapes", "coppia": None, "gruppo": gid[g], "split": split,
                   "stato": state.get(k, "ammesso"), "motivo": {}, "gold": None,
                   "immagini": [{"sha256": hashlib.sha256(image).hexdigest(), "ordine": 0}],
                   "json_sha256": hashlib.sha256(blobs[k]).hexdigest(), "qa": []}
            lines.append(json.dumps(rec, sort_keys=True))
            info[k] = rec
    folder.mkdir()
    data = gzip.compress("".join(x + "\n" for x in lines).encode(), mtime=0)
    (folder / "records.jsonl.gz").write_bytes(data)
    meta = {"schema": mf.SCHEMA, "shards": shards, "derivati": {},
            "records": {"file": "records.jsonl.gz", "sha256": hashlib.sha256(data).hexdigest(), "righe": len(lines)}}
    (folder / "manifest.json").write_text(json.dumps(meta))
    return {"folder": folder, "shards": shards, "records": info, "tmp": tmp}


def epoch(source, n_workers: int) -> list[str]:
    out = []
    for w in range(n_workers):
        reader = SourceReader(source, seed=3, worker=w, n_workers=n_workers, val_count=0)
        while True:
            key, _ = reader.next()
            if reader.epoch > 0:
                break
            out.append(key)
    return out


def test_reader():
    with tempfile.TemporaryDirectory(prefix="test_manifest_") as d:
        t = build_tiny(Path(d))
        folder, tmp = t["folder"], t["tmp"]
        train_glob = str(tmp / "train-*.tar")
        base = {"name": "s", "kind": "synth_qa", "shards": train_glob, "weight": 1.0, "manifest": str(folder)}
        recs = t["records"]
        want = {split: sorted(k for k, r in recs.items() if r["split"] == split and r["stato"] == "ammesso")
                for split in ("train", "dev", "test")}
        (train,) = make_sources([{**base, "split": "train"}])
        (dev,) = make_sources([{**base, "split": "dev"}])
        (test,) = make_sources([{**base, "shards": str(tmp / "test.tar"), "split": "test"}])
        got = {"train": epoch(train, 2), "dev": epoch(dev, 2), "test": epoch(test, 2)}
        check("reader: un'epoca consegna esattamente le chiavi ammesse dello split, una volta, con più worker",
              all(sorted(got[s]) == want[s] and len(got[s]) == len(set(got[s])) for s in got), (got, want))
        check("reader: quarantena, esclusi e dev non arrivano al training",
              not {"t2", "t3", "t8", "t4", "t6", "t7", "t10"} & set(got["train"]))
        r = SourceReader(train, seed=3, worker=0, n_workers=1, val_count=0)
        for _ in range(len(want["train"])):
            r.next()
        check("reader: contatori di letti, saltati per il manifest e consegnati",
              r.counts["consegnati"] == len(want["train"]) and r.counts["saltati_manifest"] == 12 - len(want["train"])
              and r.counts["letti"] == 12, dict(r.counts))
        val = [k for k, _ in validation_samples(train, 100)]
        check("validazione dal manifest: lo split dev, nell'ordine degli shard", val == ["t4", "t6", "t7", "t10"], val)
        check("validazione dal manifest: al più count record",
              [k for k, _ in validation_samples(train, 2)] == ["t4", "t6"])
        (legacy,) = make_sources([{k: v for k, v in base.items() if k != "manifest"}])
        legacy_all = sorted(epoch(legacy, 1))
        r = SourceReader(legacy, seed=3, worker=0, n_workers=1, val_count=2)
        legacy_val = []
        while True:
            key, _ = r.next()
            if r.epoch > 0:
                break
            legacy_val.append(key)
        check("sorgente senza manifest (fuori da MANIFEST_REQUIRED): comportamento di prima",
              legacy.admission is None and legacy_all == sorted(f"t{i}" for i in range(12))
              and sorted(legacy_val) == sorted(f"t{i}" for i in range(12) if i not in (6, 7))
              and [k for k, _ in validation_samples(legacy, 2)] == ["t6", "t7"], (legacy_all, sorted(legacy_val)))
        files = {"json": b"{}", "png": png()}
        check("record diverso da quello ammesso: motivo dal controllo degli sha256",
              train.admission.check("t0", files) == "JSON diverso da quello del manifest"
              and train.admission.check("zz", files) == "chiave assente dal manifest")
        check("provenienza del run: il manifest con sha256 e split",
              provenance([{**base, "split": "train"}])["s"]["manifest"]["split"] == "train")

        # arresti espliciti
        check("arresto: manifest mancante", raises(lambda: make_sources([{**base, "manifest": str(tmp / "no"),
                                                                           "split": "train"}])))
        check("arresto: manifest senza split, o split senza manifest",
              raises(lambda: make_sources([base])) and raises(lambda: make_sources(
                  [{k: v for k, v in base.items() if k != "manifest"} | {"split": "train"}])))
        check("arresto: split sconosciuto", raises(lambda: make_sources([{**base, "split": "val"}])))
        check("arresto: split senza record ammessi in questi shard", raises(lambda: make_sources([{**base, "split": "test"}])))
        write_tar(tmp / "other.tar", [("u0", scene("u0", 0), png())])
        check("arresto: shard non registrato nel manifest",
              raises(lambda: make_sources([{**base, "shards": str(tmp / "other.tar"), "split": "train"}])))
        check("arresto: shard v4 senza manifest (MANIFEST_REQUIRED)", raises(lambda: make_sources(
            [{"name": "v4", "kind": "synth_qa", "shards": str(ROOT / "data/shards/synth/v4/train/*.tar"), "weight": 1}])))

        def altered(edit_meta=None, edit_rows=None, edit_bytes=None) -> Path:
            dst = tmp / f"m{len(list(tmp.glob('m*')))}"
            dst.mkdir()
            meta = json.loads((folder / "manifest.json").read_text())
            data = (folder / "records.jsonl.gz").read_bytes()
            if edit_rows:
                rows = [json.loads(x) for x in gzip.decompress(data).decode().splitlines()]
                edit_rows(rows)
                data = gzip.compress("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows).encode(), mtime=0)
                meta["records"]["sha256"] = hashlib.sha256(data).hexdigest()
            if edit_bytes:
                data = edit_bytes(data)
            if edit_meta:
                edit_meta(meta)
            (dst / "records.jsonl.gz").write_bytes(data)
            (dst / "manifest.json").write_text(json.dumps(meta))
            return dst

        def load_with(path: Path):
            return lambda: make_sources([{**base, "manifest": str(path), "split": "train"}])

        flip = altered(edit_bytes=lambda b: b[:-5] + bytes([b[-5] ^ 1]) + b[-4:])
        check("arresto: records.jsonl.gz alterato", raises(load_with(flip)))

        def split_group(rows):
            next(r for r in rows if r["record"] == "t1")["split"] = "dev"

        check("arresto: un gruppo in due split", raises(load_with(altered(edit_rows=split_group))))

        def rename(rows):
            rows[0]["record"] = "t0x"

        check("arresto: record_id che non si ricalcola", raises(load_with(altered(edit_rows=rename))))

        def short_group(rows):
            rows[0]["gruppo"] = rows[0]["gruppo"][:16]

        check("arresto: gruppo con un identificatore breve", raises(load_with(altered(edit_rows=short_group))))

        def count(meta):
            meta["shards"][0]["record"] += 1

        check("arresto: conteggio di uno shard diverso dai record", raises(load_with(altered(edit_meta=count))))

        def schema(meta):
            meta["schema"] = "altro"

        check("arresto: schema sconosciuto", raises(load_with(altered(edit_meta=schema))))
        shard = Path(t["shards"][0]["percorso"])
        raw = bytearray(shard.read_bytes())
        raw[700] ^= 1  # dentro il PNG del primo record: stessa dimensione, sha256 diverso
        shard.write_bytes(bytes(raw))
        check("arresto: shard con sha256 diverso da quello del manifest",
              raises(lambda: make_sources([{**base, "split": "train"}])))

    # MixedRowStream: un record con sha256 diverso dal manifest è un errore contato, non un passaggio silenzioso
    from limite_vl.processing import LimiteVLProcessor

    processor = LimiteVLProcessor.from_pretrained(str(VIOLETTO))
    with tempfile.TemporaryDirectory(prefix="test_manifest_stream_") as d:
        t = build_tiny(Path(d))
        (train,) = make_sources([{"name": "s", "kind": "synth_qa", "shards": str(t["tmp"] / "train-*.tar"), "weight": 1.0,
                                  "manifest": str(t["folder"]), "split": "train"}])
        bad = "t5"
        e = train.admission.entries[bad]
        train.admission.entries[bad] = (e[0], e[1], "0" * 64, e[3])
        stream = MixedRowStream([train], processor, max_len=2048, micro_rows=1, seed=0, shuffle_buffer=2, pack_buffer=2)
        keys, stats = [], {}
        for n, batch in enumerate(stream.iter_worker(0, 1)):
            keys += [k.split("/", 1)[1] for k in batch["meta"]["keys"]]
            stats = batch["meta"]["stats"]
            if n >= 6:
                break
        admitted = {k for k, r in t["records"].items() if r["split"] == "train" and r["stato"] == "ammesso"}
        check("flusso: solo chiavi ammesse di train, il record alterato saltato e contato",
              set(keys) <= admitted - {bad} and stats.get("manifest_errors/s", 0) >= 1 and stats.get("loaded/s", 0) > 0,
              (sorted(set(keys)), stats))


if __name__ == "__main__":
    test_identity()
    test_assign_splits()
    test_reader()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
