#!/usr/bin/env python3
"""Test CPU della decontaminazione (src/limite_vl/decontam.py, scripts/decontam.py). Da eseguire in
slurm/tests_cpu.sbatch: la prova completa lancia index e scan su un finto benchmark."""
from __future__ import annotations

import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.data import iter_shard  # noqa: E402
from limite_vl.decontam import HashIndex, TextIndex, hamming, normalize_words, phash  # noqa: E402

MMFINE = ROOT / "data/shards/mmfinereason/7c93f724-rl/mmfinereason-rl-00000.tar"
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_hash_index():
    rng = random.Random(0)
    hashes = [rng.getrandbits(64) for _ in range(3000)]
    for i in range(0, 300):  # vicini piantati a 1-4 bit
        h = hashes[i]
        for b in rng.sample(range(64), rng.randint(1, 4)):
            h ^= 1 << b
        hashes.append(h)
    index = HashIndex(hashes, radius=4)
    ok = True
    for q in rng.sample(hashes, 400) + [rng.getrandbits(64) for _ in range(100)]:
        brute = sorted(((i, hamming(q, h)) for i, h in enumerate(hashes) if hamming(q, h) <= 4), key=lambda x: (x[1], x[0]))
        got = sorted(index.query(q), key=lambda x: (x[1], x[0]))
        ok &= got == brute
    check("HashIndex: stessi vicini della ricerca esaustiva (radius 4)", ok)


def reencode(image: Image.Image, scale: float, quality: int) -> bytes:
    w, h = image.size
    small = image.convert("RGB").resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def sample_images(n: int):
    out = []
    for key, files in iter_shard(MMFINE):
        ext = "png" if "png" in files else "jpg"
        out.append((key, files[ext], json.loads(files["json"]), files))
        if len(out) >= n:
            break
    return out


def test_phash(samples):
    same, diff = [], []
    hashes = []
    for _, data, _, _ in samples[:40]:
        with Image.open(io.BytesIO(data)) as im:
            h0 = phash(im)
            with Image.open(io.BytesIO(reencode(im, 0.7, 70))) as im2:
                same.append(hamming(h0, phash(im2)))
        hashes.append(h0)
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            diff.append(hamming(hashes[i], hashes[j]))
    check("pHash: stessa immagine ridimensionata e ricompressa entro 4 bit", max(same) <= 4, f"max {max(same)}")
    close = sum(d <= 4 for d in diff)
    check("pHash: immagini diverse quasi mai entro 4 bit", close <= len(diff) * 0.01,
          f"{close} coppie su {len(diff)}, minimo {min(diff)}")


def test_text_index():
    bench = ["In the figure, AB is the diameter of circle O and the chord CD is perpendicular to AB; find the length of CE.",
             "What is the value of x?"]
    idx = TextIndex.build(bench, n=10)
    check("TextIndex: domanda che contiene un 10-gram del benchmark",
          idx.query("Look carefully: in the figure, AB is the diameter of circle O and the chord CD crosses it.") == [0])
    check("TextIndex: domanda diversa non trovata", idx.query("Compute the area of the shaded triangle in the plot.") == [])
    check("TextIndex: domanda corta non confrontata sul testo", idx.query("What is the value of x?") == [])
    check("normalize_words: ideogrammi uno per uno, larghezza piena ridotta",
          normalize_words("如图，若∠１＝55°，则 \\frac{x}{2}") == ["如", "图", "若", "1", "55", "则", "frac", "x", "2"],
          str(normalize_words("如图，若∠１＝55°，则 \\frac{x}{2}")))
    zh = TextIndex.build(["如图，直线a∥b，直线l与直线a相交于点O，与直线b相交于点P，OM⊥l于点O．若∠1＝55°，则∠2＝（）"], n=10)
    check("TextIndex: domanda cinese diversa con le stesse scelte non trovata",
          zh.query("如图，把一块直角三角尺的直角顶点放在直尺的一边上，若∠1＝55°，则∠2的度数为（） A. 35° B. 45° C. 55° D. 25°") == []
          and zh.query("已知：如图，直线a∥b，直线l与直线a相交于点O，与直线b相交于点P") == [0])
    hint = " Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
    many = [f"Question {i}: how many {w} are left in the picture?" + hint for i, w in enumerate(["red cubes"] * 30)]
    many.append("A ladder of length 13 leans against a wall and its foot is 5 meters away from the base." + hint)
    many += [many[-1]] * 5  # varianti identiche: un solo testo
    idx = TextIndex.build(many, n=10, max_df=25)
    check("TextIndex: formula di rito presente in più di max_df testi ignorata",
          idx.query("Which bar is tallest?" + hint) == [] and idx.dropped > 0, f"dropped {idx.dropped}")
    check("TextIndex: domanda specifica trovata anche con la formula di rito, varianti identiche contate una volta",
          idx.query("A ladder of length 13 leans against a wall and its foot is 5 meters away." + hint)
          == list(range(30, 36)))


def test_end_to_end(samples, tmp: Path):
    planted = samples[:20]
    rows = {"pid": [], "question": [], "decoded_image": []}
    for i, (key, data, meta, _) in enumerate(planted):
        with Image.open(io.BytesIO(data)) as im:
            rows["pid"].append(str(i))
            rows["question"].append("Unrelated benchmark wording number %d." % i)
            rows["decoded_image"].append({"bytes": reencode(im, 0.8, 80), "path": f"{i}.jpg"})
    bench = tmp / "bench"
    bench.mkdir()
    pq.write_table(pa.table(rows), bench / "testmini.parquet")
    shard = tmp / "train-00000.tar"
    with tarfile.open(shard, "w") as tar:
        for key, _, _, files in samples:
            for ext, data in files.items():
                info = tarfile.TarInfo(f"{key}.{ext}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    env = dict(os.environ, SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID", "test"))
    py = sys.executable
    subprocess.run([py, "-B", str(ROOT / "scripts/decontam.py"), "index", "--benchmark", f"finto={bench}",
                    "--out", str(tmp / "index.json")], check=True, env=env)
    subprocess.run([py, "-B", str(ROOT / "scripts/decontam.py"), "scan", "--index", str(tmp / "index.json"),
                    "--kind", "mmfinereason", "--shards", str(shard), "--out", str(tmp / "scan"), "--workers", "2"],
                   check=True, env=env)
    excluded = set((tmp / "scan" / "exclude.txt").read_text().split())
    expected = {k for k, _, _, _ in planted}
    extra = excluded - expected
    check("scan: ritrova tutte le immagini piantate", expected <= excluded, f"mancano {len(expected - excluded)}")
    check("scan: pochi falsi positivi sugli altri campioni", len(extra) <= 2, f"{len(extra)} in più su {len(samples) - 20}")


if __name__ == "__main__":
    test_hash_index()
    test_text_index()
    if not MMFINE.is_file():
        check("shard MMFineReason presente", False, str(MMFINE))
    else:
        samples = sample_images(300)
        test_phash(samples)
        tmp = ROOT / "outputs" / "tmp" / f"test-decontam-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            test_end_to_end(samples, tmp)
        finally:
            shutil.rmtree(tmp)
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
