#!/usr/bin/env python3
"""Test CPU dei dati di F1: esempi MAVIS da uno shard vero, flusso misto (partizione fra worker,
validazione esclusa, determinismo, ripresa) su shard sintetici. Da eseguire in slurm/tests_cpu.sbatch."""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.data import (FORMULA_PROMPTS, MAVIS_CAPTION_PROMPTS, STEM_CAPTION_PROMPTS, build_example,  # noqa: E402
                            generation_prompt, iter_shard, mavis_caption_prompt)
from limite_vl.latex import despace, latex_tokens, same_latex  # noqa: E402
from limite_vl.processing import (DIAGRAM_MAX_PATCHES, EOS_ID, IGNORE, IMAGE_PAD_ID, PATCH_SIZE,  # noqa: E402
                                  LimiteVLProcessor, capped_size, exact_patches, image_tokens, target_size)
from limite_vl.stream import MixedRowStream, Source, SourceReader  # noqa: E402

VIOLETTO = ROOT / "hf/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
MAVIS = ROOT / "data/shards/mavis-geometry/9c80fff3-caption/mavis-geometry-caption-00000.tar"
UNIMER = ROOT / "data/shards/unimer/2343ddd9/unimer-train-00000.tar"
MMFINE = ROOT / "data/shards/mmfinereason/7c93f724-rl/mmfinereason-rl-00000.tar"
THINK, END_THINK = 151665, 151666
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_mavis_example(proc):
    if not MAVIS.is_file():
        check("shard MAVIS presente", False, str(MAVIS))
        return
    key, files = next(iter_shard(MAVIS))
    meta = json.loads(files["json"])
    ex = build_example(proc, "mavis_caption", key, files)
    with Image.open(io.BytesIO(files["png"])) as im:
        size = (im.height, im.width)
    rows, cols = ex.grids[0]
    check("MAVIS: PNG già alla dimensione di 2304 patch", size == target_size(*size, DIAGRAM_MAX_PATCHES)
          and (rows * PATCH_SIZE, cols * PATCH_SIZE) == size, f"{size}, griglia {ex.grids[0]}")
    check("MAVIS: token immagine = griglia/4", int((ex.input_ids == IMAGE_PAD_ID).sum()) == image_tokens(ex.grids[0])
          == (rows // 2) * (cols // 2), str(image_tokens(ex.grids[0])))
    labelled = ex.input_ids[ex.labels != IGNORE].tolist()
    caption = meta["text_en"]["caption"].strip()
    check("MAVIS: label = didascalia + EOS", labelled[-1] == EOS_ID
          and proc.tokenizer.decode(labelled[:-1]) == caption, proc.tokenizer.decode(labelled[:-1])[:80])
    first = int(np.argmax(ex.labels != IGNORE))
    check("MAVIS: prefill <think></think> senza label subito prima della didascalia",
          ex.input_ids[first - 2:first].tolist() == [THINK, END_THINK]
          and (ex.labels[first - 2:first] == IGNORE).all())
    prompt = mavis_caption_prompt(key)
    check("MAVIS: richiesta dalla lista, deterministica", prompt in MAVIS_CAPTION_PROMPTS
          and prompt == mavis_caption_prompt(key) and prompt in proc.tokenizer.decode(ex.input_ids.tolist()))
    prompts = {mavis_caption_prompt(f"d0-{i}") for i in range(200)}
    check("MAVIS: tutte le richieste usate", len(prompts) == len(MAVIS_CAPTION_PROMPTS), str(len(prompts)))
    gen, reference = generation_prompt(proc, "mavis_caption", key, files)
    check("MAVIS: prompt di generazione = esempio fino al prefill incluso, senza label",
          gen.input_ids.tolist() == ex.input_ids[:first].tolist() and (gen.labels == IGNORE).all()
          and reference == caption)
    check("MAVIS: chiave e sorgente nell'esempio", ex.key == key and ex.source == "mavis_caption")
    white = build_example(proc, "mavis_caption", key, files, image_mode="white")
    check("immagine bianca: stessi token e label, patch tutte 255",
          (white.input_ids == ex.input_ids).all() and (white.labels == ex.labels).all()
          and white.grids == ex.grids and all((p == 255).all() for p in white.patches))
    none = build_example(proc, "mavis_caption", key, files, image_mode="none")
    check("senza immagine: niente token immagine né patch, stessa risposta con label",
          not (none.input_ids == IMAGE_PAD_ID).any() and not none.patches
          and none.input_ids[none.labels != IGNORE].tolist() == labelled
          and prompt in proc.tokenizer.decode(none.input_ids.tolist()))


def test_latex():
    cases = {"\\frac { 1 } { 2 }": "\\frac{1}{2}", "\\alpha b": "\\alpha b", "0 . 5": "0.5",
             "a \\ b": "a\\ b", "\\left\\{ x \\right\\}": "\\left\\{x\\right\\}", "x \\\\ y": "x\\\\y",
             "\\mathrm { m g }": "\\mathrm{mg}", "\\alpha 2": "\\alpha2", "\\ \\mathrm { i f }": "\\ \\mathrm{if}"}
    got = {k: despace(k) for k in cases}
    check("despace: casi noti", got == cases, str({k: v for k, v in got.items() if v != cases[k]}))
    check("despace: stessi token LaTeX", all(latex_tokens(k) == latex_tokens(v) for k, v in cases.items()))
    check("same_latex: uguali a meno degli spazi, diversi se cambia un token",
          same_latex("x ^ { 2 }", "x^{2}") and not same_latex("x^{2}", "x^{3}") and not same_latex("\\alpha b", "\\alphab"))


def test_capped_size():
    sizes = [(32, 32), (32, 64), (64, 640), (32, 2528), (1056, 2000), (224, 768), (37, 91), (443, 1296)]
    ok, detail = True, []
    for h, w in sizes:
        hh, ww = capped_size(h, w, 2.0, 1024)
        patches = exact_patches(hh, ww)
        good = (hh % 32 == 0 and ww % 32 == 0 and patches <= 1024 and hh <= 2 * h + 31 and ww <= 2 * w + 31
                and target_size(hh, ww, patches) == (hh, ww))
        ok &= good
        detail.append(f"{h}x{w}->{hh}x{ww}")
    check("capped_size: multipli di 32, al più 2x e 1024 patch, idempotente per il processor", ok, ", ".join(detail))


def test_unimer_example(proc):
    if not UNIMER.is_file():
        check("shard UniMER presente", False, str(UNIMER))
        return
    key, files = next(iter_shard(UNIMER))
    meta = json.loads(files["json"])
    ex = build_example(proc, "unimer_formula", key, files)
    with Image.open(io.BytesIO(files["png"])) as im:
        size = (im.height, im.width)
    rows, cols = ex.grids[0]
    check("UniMER: PNG già alla dimensione finale", (rows * PATCH_SIZE, cols * PATCH_SIZE) == size
          and meta["max_patches"] == rows * cols, f"{size}, griglia {ex.grids[0]}")
    labelled = ex.input_ids[ex.labels != IGNORE].tolist()
    check("UniMER: label = formula senza spazi + EOS", labelled[-1] == EOS_ID
          and proc.tokenizer.decode(labelled[:-1]) == meta["label"] and same_latex(meta["label"], meta["label_spaced"]),
          meta["label"][:80])
    text = proc.tokenizer.decode(ex.input_ids.tolist())
    check("UniMER: richiesta di trascrizione dalla lista", any(p in text for p in FORMULA_PROMPTS))
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_f1_formulas import rescale
    same = rescale(files, 2.0, 1024)
    check("eval_f1_formulas.rescale: con 2x e 1024 patch l'immagine resta quella dello shard",
          json.loads(same["json"])["max_patches"] == meta["max_patches"])
    big = rescale(files, 4.0, 2304)
    big_meta = json.loads(big["json"])
    ex_big = build_example(proc, "unimer_formula", key, big)
    rows, cols = ex_big.grids[0]
    with Image.open(io.BytesIO(big["png"])) as im:
        big_size = (im.height, im.width)
    check("eval_f1_formulas.rescale: con 4x più patch, PNG alla dimensione finale, il processor non la tocca",
          meta["max_patches"] < big_meta["max_patches"] <= 2304 and big_meta["max_patches"] == rows * cols
          and (rows * PATCH_SIZE, cols * PATCH_SIZE) == big_size, f"{meta['max_patches']} -> {big_meta['max_patches']}")


def test_stem_caption_example(proc):
    if not MMFINE.is_file():
        check("shard MMFineReason presente", False, str(MMFINE))
        return
    from limite_vl.data import STEM_MAX_SCALE
    sizes_ok, bigger, n = True, 0, 0
    for key, files in iter_shard(MMFINE):
        meta = json.loads(files["json"])
        ex = build_example(proc, "stem_caption", key, files)
        rows, cols = ex.grids[0]
        with Image.open(io.BytesIO(files["png"] if "png" in files else files["jpg"])) as im:
            want = capped_size(im.height, im.width, STEM_MAX_SCALE, 2304)
        sizes_ok &= (rows * PATCH_SIZE, cols * PATCH_SIZE) == want and rows * cols == exact_patches(*want)
        bigger += rows * cols > meta["max_patches"]  # il JSON dello shard ha la regola a 2x
        if n == 0:
            labelled = ex.input_ids[ex.labels != IGNORE].tolist()
            check("MMFineReason: label = descrizione + EOS", labelled[-1] == EOS_ID
                  and proc.tokenizer.decode(labelled[:-1]) == meta["caption"], meta["caption"][:60])
            text = proc.tokenizer.decode(ex.input_ids.tolist())
            check("MMFineReason: richiesta di descrizione dalla lista", any(q in text for q in STEM_CAPTION_PROMPTS))
        n += 1
        if n >= 50:
            break
    check(f"MMFineReason: immagine originale portata a capped_size({STEM_MAX_SCALE}x, 2304 patch) (50 campioni)", sizes_ok)
    check("MMFineReason: con il tetto a 4x più patch che con quello a 2x del JSON in almeno metà dei campioni",
          bigger >= n // 2, f"{bigger}/{n}")


def write_shard(path: Path, prefix: str, n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    keys = []
    with tarfile.open(path, "w") as tar:
        for i in range(n):
            key = f"{prefix}-{i:03d}"
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(buf, format="JPEG")
            meta = json.dumps({"prompt": "<image>\nDescribe the image.", "caption": f"Caption number {i} of {prefix}."})
            for ext, data in (("jpg", buf.getvalue()), ("json", meta.encode())):
                info = tarfile.TarInfo(f"{key}.{ext}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            keys.append(key)
    return keys


def epoch_keys(source: Source, worker: int, n_workers: int, val_count: int) -> list[str]:
    reader = SourceReader(source, 0, worker, n_workers, val_count)
    keys = []
    while reader.epoch == 0:
        key, _ = reader.next()
        keys.append(key)
        if len(keys) > 10_000:
            break
    # next() può aver letto il primo campione dell'epoca 1 prima di tornare: lo tolgo
    if reader.epoch == 1:
        keys.pop()
    return keys


def test_stream(proc, tmp: Path):
    a_keys = sum((write_shard(tmp / f"a-{s}.tar", f"a{s}", 40, s) for s in range(3)), [])
    sum((write_shard(tmp / f"b-{s}.tar", f"b{s}", 30, 10 + s) for s in range(2)), [])
    a = Source("a", "llava", sorted(str(p) for p in tmp.glob("a-*.tar")), 3.0)
    b = Source("b", "llava", sorted(str(p) for p in tmp.glob("b-*.tar")), 1.0)
    val = 4

    # partizione: due worker, un'epoca, chiavi disgiunte che coprono tutto tranne la validazione
    w0, w1 = epoch_keys(a, 0, 2, val), epoch_keys(a, 1, 2, val)
    expected = set(a_keys) - {f"a2-{i:03d}" for i in range(val)}
    check("flusso: worker disgiunti", not set(w0) & set(w1), str(set(w0) & set(w1)))
    check("flusso: un'epoca copre tutto tranne la validazione", set(w0) | set(w1) == expected
          and len(w0) + len(w1) == len(expected), f"{len(w0)}+{len(w1)} contro {len(expected)}")
    check("flusso: validazione = primi campioni dell'ultimo shard", a.val_shard.endswith("a-2.tar"))

    def batches(stream, n, worker=0, n_workers=1):
        out, it = [], stream.iter_worker(worker, n_workers)
        for _ in range(n):
            out.append(next(it))
        return out

    def make(resume=None):
        return MixedRowStream([a, b], proc, max_len=1024, micro_rows=1, seed=0, shuffle_buffer=8, pack_buffer=6,
                              val_count=val, resume=resume)

    first, second = batches(make(), 6), batches(make(), 6)
    check("flusso: deterministico", all((x["input_ids"] == y["input_ids"]).all() and x["meta"]["keys"] == y["meta"]["keys"]
                                       for x, y in zip(first, second)))
    seen = [k for x in first for k in x["meta"]["keys"]]
    check("flusso: nessuna chiave di validazione", not any(k.split("/")[1] in {f"a2-{i:03d}" for i in range(val)}
                                                            or k.split("/")[1] in {f"b1-{i:03d}" for i in range(val)}
                                                            for k in seen))
    check("flusso: micro-batch con immagini e righe da max_len", first[0]["input_ids"].shape == (1, 1024)
          and first[0]["spatial_shapes"].shape[0] == len(first[0]["meta"]["keys"]))
    counts = {"a": sum(x["meta"]["sources"].get("a", 0) for x in first),
              "b": sum(x["meta"]["sources"].get("b", 0) for x in first)}
    check("flusso: entrambe le sorgenti, a più frequente (peso 3:1)", counts["a"] > counts["b"] > 0, str(counts))

    # ripresa dallo stato del terzo micro-batch: niente ripetizioni, stessa continuazione dei lettori
    state = first[2]["meta"]["state"]
    resumed = batches(make({"0": state}), 3)
    before = {k for x in first[:3] for k in x["meta"]["keys"]}
    after = [k for x in resumed for k in x["meta"]["keys"]]
    check("flusso: ripresa senza esempi già consegnati", not before & set(after), str(before & set(after)))
    again = batches(make({"0": state}), 3)
    check("flusso: ripresa deterministica", [x["meta"]["keys"] for x in again] == [x["meta"]["keys"] for x in resumed])
    later = resumed[-1]["meta"]["state"]["sources"]["a"]
    check("flusso: posizione dei lettori avanza", tuple(later[:3]) > tuple(state["sources"]["a"][:3]),
          f"{state['sources']['a']} -> {later}")
    try:
        next(make({"0": state}).iter_worker(0, 2))
        check("flusso: rifiuta un numero di worker diverso", False)
    except ValueError:
        check("flusso: rifiuta un numero di worker diverso", True)


if __name__ == "__main__":
    processor = LimiteVLProcessor.from_pretrained(str(VIOLETTO))
    test_latex()
    test_capped_size()
    test_mavis_example(processor)
    test_unimer_example(processor)
    test_stem_caption_example(processor)
    tmp = ROOT / "outputs" / "tmp" / f"test-f1-data-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        test_stream(processor, tmp)
    finally:
        shutil.rmtree(tmp)
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
