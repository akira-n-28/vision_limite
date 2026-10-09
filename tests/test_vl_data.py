"""Test CPU di processor, collator e lettura degli shard di Limite-VL (tokenizer di Violetto, nessun peso).
Su Lagrange: in un job CPU con l'ambiente del progetto.

    srun $RUN_PYTHON -B tests/test_vl_data.py
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.collator import PackedCollator, iter_packed_rows, pack_first_fit  # noqa: E402
from limite_vl.data import iter_shard, llava_pretrain_example  # noqa: E402
from limite_vl.processing import (DEFAULT_MAX_PATCHES, DIAGRAM_MAX_PATCHES, EOS_ID, IGNORE,  # noqa: E402
                                  IMAGE_PAD_ID, MAX_IMAGE_TOKENS, Example, LimiteVLProcessor,
                                  image_to_patches, image_tokens, target_size)

VIOLETTO = ROOT / "hf/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
SHARD = ROOT / "data/shards/llava-pretrain/70f9d1e5/llava-pretrain-00000.tar"
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def noise(width, height, seed=0, mode="RGB"):
    rng = np.random.default_rng(seed)
    channels = len(mode)
    return Image.fromarray(rng.integers(0, 256, (height, width, channels), dtype=np.uint8), mode)


def test_target_size():
    rng = random.Random(0)
    sizes = [(1, 1), (1, 4000), (4000, 1), (336, 336), (1080, 1920), (31, 33)]
    sizes += [(rng.randint(1, 5000), rng.randint(1, 5000)) for _ in range(300)]
    bad = []
    for max_patches in (DEFAULT_MAX_PATCHES, DIAGRAM_MAX_PATCHES):
        for h, w in sizes:
            th, tw = target_size(h, w, max_patches)
            patches = (th // 16) * (tw // 16)
            if th % 32 or tw % 32 or patches > max_patches or patches // 4 > MAX_IMAGE_TOKENS or th < 32 or tw < 32:
                bad.append((h, w, max_patches, th, tw))
    check("target_size: lati multipli di 32 e budget di patch rispettato", not bad, str(bad[:3]))
    check("target_size: 336x336 -> 512x512 (256 token)", target_size(336, 336) == (512, 512))
    try:
        target_size(64, 64, 4096)
        check("target_size: rifiuta più di 768 token", False)
    except ValueError:
        check("target_size: rifiuta più di 768 token", True)


def test_patches_match_siglip2():
    from transformers.models.siglip2.image_processing_pil_siglip2 import Siglip2ImageProcessorPil

    image = noise(320, 224, seed=1)  # già a lati multipli di 32: nessun ridimensionamento
    patches, grid = image_to_patches(image, max_patches=1024)
    reference = Siglip2ImageProcessorPil(patch_size=16, max_num_patches=grid[0] * grid[1])(
        images=[image], return_tensors="np")
    ref = reference["pixel_values"][0][: grid[0] * grid[1]]
    mine = (patches.astype(np.float32) / 255.0 - 0.5) / 0.5
    check("patch: griglia come SigLIP2", tuple(reference["spatial_shapes"][0]) == grid, f"{grid}")
    check("patch: stessi valori e ordine di SigLIP2", ref.shape == mine.shape and np.abs(ref - mine).max() < 1e-6,
          f"max diff {np.abs(ref - mine).max() if ref.shape == mine.shape else ref.shape}")


def test_alpha_on_white():
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    patches, _ = image_to_patches(image)
    check("trasparenza su fondo bianco", int(patches.min()) == 255)


def test_encode_single_turn(proc):
    tok = proc.tokenizer
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Find x."}]},
                {"role": "assistant", "reasoning_content": "The angle is 30.", "content": "\\boxed{30}"}]
    ex = proc.encode(messages, [noise(400, 300)])
    n_img = image_tokens(ex.grids[0])
    ids, labels = ex.input_ids.tolist(), ex.labels
    check("encode: token <|image_pad|> = token dell'immagine", ids.count(IMAGE_PAD_ID) == n_img, f"{n_img}")
    check("encode: patch coerenti con la griglia", ex.patches[0].shape == (ex.grids[0][0] * ex.grids[0][1], 768))
    text = tok.decode(ids, skip_special_tokens=False)
    check("encode: system prompt e immagine prima della domanda",
          text.startswith("<|endoftext|><|im_start|>system\n") and "<|vision_end|>Find x.<|im_end|>" in text)
    target = tok.decode(ex.input_ids[labels != IGNORE].tolist(), skip_special_tokens=False)
    expected = "<think>The angle is 30.</think>\\boxed{30}<|endoftext|>"
    check("encode: label solo sulla risposta, formato di Violetto e EOS", target == expected, repr(target))
    check("encode: niente <|im_end|> dopo l'ultima risposta", not text.endswith("<|im_end|>\n<|endoftext|>"))
    first = int(np.argmax(labels != IGNORE))
    check("encode: label contigue fino alla fine", bool((labels[first:] != IGNORE).all()) and ids[-1] == EOS_ID)


def test_encode_multi_turn(proc):
    tok = proc.tokenizer
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is shown?"}]},
                {"role": "assistant", "content": "A triangle."},
                {"role": "user", "content": "Its area?"},
                {"role": "assistant", "content": "6."}]
    ex = proc.encode(messages, [noise(64, 64)])
    labelled = ex.input_ids.copy()
    labelled[ex.labels == IGNORE] = -1
    spans, current = [], []
    for t in labelled.tolist():
        if t == -1:
            if current:
                spans.append(current)
            current = []
        else:
            current.append(t)
    if current:
        spans.append(current)
    decoded = [tok.decode(s, skip_special_tokens=False) for s in spans]
    check("multi-turno: due turni dell'assistente con label",
          decoded == ["A triangle.", "6.<|endoftext|>"], repr(decoded))
    text = tok.decode(ex.input_ids.tolist(), skip_special_tokens=False)
    check("multi-turno: <|im_end|> nella cronologia, EOS alla fine",
          "A triangle.<|im_end|>\n<|im_start|>user\nIts area?" in text and text.endswith("6.<|endoftext|>"))


def test_rejects(proc):
    cases = {
        "immagine dopo il testo": ([{"role": "user", "content": [{"type": "text", "text": "Q"}, {"type": "image"}]}],
                                   [noise(64, 64)]),
        "token speciale nel testo": ([{"role": "user", "content": "<|image_pad|> hi"}], []),
        "numero di immagini": ([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q"}]}], []),
    }
    for name, (messages, images) in cases.items():
        try:
            proc.encode(messages, images)
            check(f"rifiuta: {name}", False)
        except ValueError:
            check(f"rifiuta: {name}", True)


def test_generation_prompt(proc):
    ex = proc.encode([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q"}]}],
                     [noise(64, 64)], add_generation_prompt=True)
    text = proc.tokenizer.decode(ex.input_ids.tolist(), skip_special_tokens=False)
    check("prompt di generazione: finisce con l'apertura dell'assistente, senza label",
          text.endswith("<|im_start|>assistant\n") and bool((ex.labels == IGNORE).all()))


def fake(n, n_images=0, seed=0):
    rng = np.random.default_rng(seed)
    grids = [(2 * int(rng.integers(1, 4)), 2 * int(rng.integers(1, 4))) for _ in range(n_images)]
    patches = [rng.integers(0, 256, (g[0] * g[1], 768), dtype=np.uint8) for g in grids]
    ids = rng.integers(0, 1000, n).astype(np.int32)
    return Example(ids, ids.copy(), patches, grids)


def test_pack_first_fit():
    rng = random.Random(0)
    lengths = [rng.randint(1, 5000) for _ in range(500)]
    bins = pack_first_fit(lengths, 4096)
    flat = [i for b in bins for i in b]
    check("first-fit: ogni indice al più una volta, gruppi entro max_len",
          len(flat) == len(set(flat)) and all(sum(lengths[i] for i in b) <= 4096 for b in bins))
    check("first-fit: fuori solo i troppo lunghi", set(flat) == {i for i, n in enumerate(lengths) if n <= 4096})
    fill = sum(lengths[i] for i in flat) / (len(bins) * 4096)
    check("first-fit: riempimento > 95%", fill > 0.95, f"{fill:.3f}")


def test_iter_packed_rows():
    rng = random.Random(1)
    examples = [fake(rng.randint(10, 900), seed=i) for i in range(400)] + [fake(5000, seed=999)]
    stats = Counter()
    rows = list(iter_packed_rows(examples, 2048, buffer_size=64, stats=stats))
    seen = [id(e) for r in rows for e in r]
    check("righe: ogni esempio una volta, il troppo lungo escluso",
          len(seen) == len(set(seen)) == 400 and stats["too_long"] == 1, str(dict(stats)))
    check("righe: entro max_len", all(sum(len(e) for e in r) <= 2048 for r in rows))
    check("righe: riempimento medio > 90%", stats["tokens"] / (stats["rows"] * 2048) > 0.9,
          f"{stats['tokens'] / (stats['rows'] * 2048):.3f}")


def test_collator():
    a, b, c = fake(100, 1, seed=1), fake(50, 2, seed=2), fake(30, 1, seed=3)
    batch = PackedCollator(256)([[a, b], [c]])
    pos = batch["position_ids"]
    check("collate: forme", tuple(batch["input_ids"].shape) == (2, 256) and batch["pixel_values"].shape[0] == 4)
    check("collate: posizioni ripartono a ogni esempio e nella coda",
          pos[0, :100].tolist() == list(range(100)) and pos[0, 100:150].tolist() == list(range(50))
          and pos[0, 150] == 0 and pos[1, 30] == 0)
    check("collate: coda con EOS e label ignorate",
          bool((batch["input_ids"][0, 150:] == EOS_ID).all()) and bool((batch["labels"][0, 150:] == IGNORE).all()))
    order = [a.grids[0], b.grids[0], b.grids[1], c.grids[0]]
    check("collate: immagini in ordine di apparizione", [tuple(s) for s in batch["spatial_shapes"].tolist()] == order)
    n = [g[0] * g[1] for g in order]
    mask = batch["pixel_attention_mask"]
    check("collate: maschera delle patch", mask.sum(1).tolist() == n and
          bool((batch["pixel_values"][1, : n[1]].numpy() == b.patches[0]).all()))


def test_llava_shard(proc):
    if not SHARD.is_file():
        check("shard LLaVA-Pretrain presente", False, str(SHARD))
        return
    examples, errors = [], []
    for key, files in iter_shard(SHARD):
        try:
            examples.append(llava_pretrain_example(proc, files))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
        if len(examples) + len(errors) >= 300:
            break
    check("LLaVA: 300 campioni tokenizzati senza errori", not errors and len(examples) == 300, str(errors[:3]))
    lengths = [len(e) for e in examples]
    tokens = Counter(image_tokens(e.grids[0]) for e in examples)
    text = [int((e.labels != IGNORE).sum()) for e in examples]
    print(f"     lunghezza media {np.mean(lengths):.0f} (min {min(lengths)}, max {max(lengths)}); "
          f"token per immagine {dict(tokens.most_common(5))}; token con label media {np.mean(text):.1f}")
    stats = Counter()
    rows = list(iter_packed_rows(examples, 4096, buffer_size=300, stats=stats))
    fill = stats["tokens"] / (stats["rows"] * 4096)
    print(f"     packing a 4096: {stats['rows']} righe, riempimento {fill:.3f}, "
          f"{np.mean([len(r) for r in rows]):.1f} esempi per riga")
    check("LLaVA: un'immagine per esempio, griglie pari", all(len(e.grids) == 1 and e.grids[0][0] % 2 == 0
                                                             and e.grids[0][1] % 2 == 0 for e in examples))
    batch = PackedCollator(4096)(rows[:2])
    check("LLaVA: collate di due righe", int((batch["input_ids"] == IMAGE_PAD_ID).sum())
          == sum(image_tokens(tuple(g)) for g in batch["spatial_shapes"].tolist()))


if __name__ == "__main__":
    test_target_size()
    test_patches_match_siglip2()
    test_alpha_on_white()
    processor = LimiteVLProcessor.from_pretrained(str(VIOLETTO))
    test_encode_single_turn(processor)
    test_encode_multi_turn(processor)
    test_rejects(processor)
    test_generation_prompt(processor)
    test_pack_first_fit()
    test_iter_packed_rows()
    test_collator()
    test_llava_shard(processor)
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
