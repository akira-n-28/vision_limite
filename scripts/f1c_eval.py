#!/usr/bin/env python3
"""Valutazione di un adapter per F1c (F1b, controllo, F1c sugli stessi campioni). Job con GPU.

1. Scelta forzata: per ogni domanda verificabile si confronta la probabilità di ogni opzione come risposta breve
   (somma dei log-prob dei token dell'opzione e di <|endoftext|>, con la stessa richiesta "Answer the question with a
   short answer." e lo stesso prefill di F1). Non serve che il modello abbia visto domande in training, quindi F1b e
   il controllo si confrontano con F1c alla pari. Insiemi: mavis (domande dai 256 diagrammi di validazione, mai in
   training), synth (scene di test), pairs (coppie controfattuali: riuscita se sono giuste entrambe le varianti),
   stem_real (fatti verificati a mano su immagini STEM di validazione, file JSON facoltativo). Modi: real, white
   (immagine bianca della stessa dimensione) e mismatch (immagine di un altro campione dello stesso insieme), questi
   due su un sottoinsieme fisso (--dependence-frac) per mavis e synth e su tutto per pairs e stem_real.
2. Generazione a batch (limite_vl.generation): didascalie MAVIS dei 256 diagrammi di validazione (vere) e dei primi
   --mavis-dependence con immagine bianca e di un altro diagramma; formule di UniMER-Test, i primi 200 di spe e cpe
   (0-99 = l'analisi degli errori, 100-199 = test indipendente) per ogni cartella di --unimer-test.
3. Teacher forcing per decimo della risposta (scripts/f1_error_gen.teacher_forcing) sulle sorgenti --tf-sources.
Uscite in OUT_DIR: choice.jsonl, gen.jsonl, formulas-<nome>.jsonl, tokens.json, summary.json. Con --check-batched N
confronta solo la generazione a batch con quella a un esempio alla volta su N didascalie e N formule.

    srun $RUN_PYTHON -u -B scripts/f1c_eval.py --config CONFIG --adapter DIR --name NOME --out-dir DIR \\
        --synth-dir DIR --unimer-test x4=DIR dense=DIR [--stem-real FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_f1_formulas import edit_distance  # noqa: E402  (stessa cartella scripts/)
from f1_error_gen import teacher_forcing  # noqa: E402
from train_f1 import load_config, seconds_left_in_job, to_cuda  # noqa: E402

from limite_vl.collator import PackedCollator  # noqa: E402
from limite_vl.data import SAMPLE_LOADERS, generation_prompt, iter_shard, open_image  # noqa: E402
from limite_vl.f1c import QA_ONE, key_rng, mavis_questions  # noqa: E402
from limite_vl.generation import generate_all  # noqa: E402
from limite_vl.latex import latex_tokens  # noqa: E402
from limite_vl.modeling import LimiteVLForConditionalGeneration, VLConfig  # noqa: E402
from limite_vl.processing import F1_PREFILL, IGNORE, IMAGE_PAD_ID, LimiteVLProcessor, image_to_patches  # noqa: E402
from limite_vl.stream import make_sources, validation_samples  # noqa: E402
from limite_vl.train_utils import allow_flex_recompiles  # noqa: E402

MAX_LEN = 4096


# ---------------------------------------------------------------- scelta forzata

def choice_items(cfg: dict, args) -> list[dict]:
    """Domande con opzioni: {set, key, image, max_patches, q, options, answer, type, pair}."""
    items = []
    sets = set(args.choice_sets.split(","))
    if "mavis" in sets:
        src = next(s for s in make_sources(cfg["sources"]) if s.name == "mavis")
        for key, files in validation_samples(src, cfg["val_count"]):
            meta = json.loads(files["json"])
            image = open_image(files["png"])
            for q in mavis_questions(meta["text_en"]["caption"].strip(), meta["depth"], key):
                items.append({"set": "mavis", "key": key, "image": image, "max_patches": meta["max_patches"], **q})
    for name, shard in (("synth", "test/synth-test.tar"), ("pairs", "pairs/synth-pairs.tar")):
        if name not in sets:
            continue
        for key, files in iter_shard(args.synth_dir / shard):
            meta = json.loads(files["json"])
            image = open_image(files["png"])
            qs = [meta["pair"]] if name == "pairs" else meta["qa"]
            for q in qs:
                items.append({"set": name, "key": key, "image": image, "max_patches": meta["max_patches"],
                              "family": meta["family"], "pair": meta.get("pair", {}).get("id"),
                              **{k: q[k] for k in ("q", "options", "answer", "type")}})
    if "stem_real" in sets and args.stem_real:
        for row in json.loads(args.stem_real.read_text()):
            image = open_image(Path(row["image"]).read_bytes())
            items.append({"set": "stem_real", "key": row["key"], "image": image, "max_patches": row["max_patches"],
                          **{k: row[k] for k in ("q", "options", "answer", "type")}})
    return items


def with_modes(items: list[dict], frac: float) -> list[dict]:
    """Ogni domanda con l'immagine vera; un sottoinsieme fisso anche con l'immagine bianca e di un altro campione."""
    out = []
    by_set: dict[str, list[dict]] = {}
    for it in items:
        by_set.setdefault(it["set"], []).append(it)
    for name, group in by_set.items():
        keys = sorted({it["key"] for it in group})
        images, whites = {}, {}
        for it in group:
            images.setdefault(it["key"], (it["image"], it["max_patches"]))
            whites.setdefault(it["key"], Image.new("RGB", it["image"].size, "white"))  # una per immagine, non per domanda
        for it in group:
            out.append({**it, "mode": "real"})
            full = name in ("pairs", "stem_real")
            if not full and key_rng(it["key"] + it["q"] + "/dep").random() >= frac:
                continue
            out.append({**it, "mode": "white", "image": whites[it["key"]]})
            other = keys[(keys.index(it["key"]) + len(keys) // 2) % len(keys)]
            img, mp = images[other]
            out.append({**it, "mode": "mismatch", "image": img, "max_patches": mp, "other": other})
    return out


@torch.no_grad()
def score_options(model, processor, items: list[dict], log_every: int = 2000) -> None:
    """Aggiunge a ogni item "scores" (log-prob di ogni opzione) e "pred". Gli esempi si costruiscono uno alla volta:
    tutti insieme (circa 50 mila con le patch delle immagini) superavano i 24 GB del job (29759, 29760). Le patch di
    un'immagine si calcolano una volta e servono per tutte le opzioni e le domande che la usano."""
    patches: OrderedDict = OrderedDict()

    def image_input(it: dict):
        k = (id(it["image"]), it["max_patches"])
        if k in patches:
            patches.move_to_end(k)
        else:
            patches[k] = image_to_patches(it["image"], it["max_patches"] or processor.max_patches)
            if len(patches) > 64:
                patches.popitem(last=False)
        return patches[k]

    def examples():
        for i, it in enumerate(items):
            user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QA_ONE + it["q"]}]}
            image = image_input(it)
            for j, opt in enumerate(it["options"]):
                yield processor.encode([user, {"role": "assistant", "content": opt, "prefill": F1_PREFILL}], [image],
                                       max_patches=it["max_patches"]), (i, j)

    for it in items:
        it["scores"] = [None] * len(it["options"])
    total = sum(len(it["options"]) for it in items)
    collate = PackedCollator(MAX_LEN)
    row, row_owners, t0 = [], [], time.time()

    def flush():
        batch = to_cuda({k: v for k, v in collate([row]).items() if k != "num_examples"})
        labels = batch.pop("labels")
        hidden = model(**batch).hidden_states[:, :-1]
        y = labels[:, 1:]
        mask = y != IGNORE
        h, target = hidden[mask], y[mask]
        loss = torch.cat([F.cross_entropy(model.logits(h[s:s + 1024]).float(), target[s:s + 1024], reduction="none")
                          for s in range(0, h.shape[0], 1024)])
        starts = mask & (torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1) == 0)
        answer = (torch.cumsum(starts.flatten().long(), 0).view_as(mask) - 1)[mask]
        sums = torch.zeros(len(row_owners), device=loss.device).index_add_(0, answer, loss)
        for (i, j), s in zip(row_owners, sums.tolist()):
            items[i]["scores"][j] = -s

    used = 0
    for n, (ex, owner) in enumerate(examples()):
        if row and used + len(ex) > MAX_LEN:
            flush()
            row, row_owners, used = [], [], 0
        row.append(ex)
        row_owners.append(owner)
        used += len(ex)
        if n and n % log_every == 0:
            print(f"   scelta forzata: {n}/{total} opzioni ({time.time() - t0:.0f} s)", flush=True)
    if row:
        flush()
    for it in items:
        it["pred"] = it["options"][max(range(len(it["options"])), key=lambda j: it["scores"][j])]
        it["correct"] = it["pred"] == it["answer"]


# ---------------------------------------------------------------- generazione

def mavis_prompts(cfg: dict, n_dependence: int, processor) -> list[dict]:
    src = next(s for s in make_sources(cfg["sources"]) if s.name == "mavis")
    raw = validation_samples(src, cfg["val_count"])
    out = []
    for i, (key, files) in enumerate(raw):
        sample = SAMPLE_LOADERS["mavis_caption"](key, files)
        *history, answer = sample.messages
        modes = [("real", sample.image)]
        if i < n_dependence:
            other = SAMPLE_LOADERS["mavis_caption"](*raw[(i + len(raw) // 2) % len(raw)]).image
            modes += [("white", Image.new("RGB", sample.image.size, "white")), ("mismatch", other)]
        for mode, image in modes:
            ex = processor.encode(history, [image], max_patches=sample.max_patches, add_generation_prompt=True,
                                  prefill=F1_PREFILL)
            out.append({"source": "mavis", "key": key, "mode": mode, "example": ex, "reference": answer["content"]})
    return out


def formula_prompts(test_dir: Path, processor, per_subset: int) -> list[dict]:
    out = []
    for subset in ("spe", "cpe"):
        for n, (key, files) in enumerate(iter_shard(test_dir / f"unimer-test-{subset}.tar")):
            if n >= per_subset:
                break
            ex, reference = generation_prompt(processor, "unimer_formula", key, files)
            out.append({"subset": subset, "key": key, "index": n, "example": ex, "reference": reference,
                        "image_tokens": int((ex.input_ids == IMAGE_PAD_ID).sum())})
    return out


def run_generation(model, processor, prompts: list[dict], max_new: int, batch_size: int) -> None:
    results = generate_all(model, processor.tokenizer, [p["example"] for p in prompts], max_new, batch_size)
    for p, r in zip(prompts, results):
        p.update(output=r["text"], eos=r["eos"], new_tokens=r["new_tokens"])
        p.pop("example")


@torch.no_grad()
def single(model, processor, ex, max_new: int) -> str:
    batch = to_cuda(PackedCollator(len(ex))([[ex]]))
    out = model.generate(batch["input_ids"], pixel_values=batch.get("pixel_values"),
                         pixel_attention_mask=batch.get("pixel_attention_mask"), spatial_shapes=batch.get("spatial_shapes"),
                         do_sample=False, max_new_tokens=max_new, use_cache=True)
    new = out[0, len(ex):].tolist()
    if processor.tokenizer.eos_token_id in new:
        new = new[: new.index(processor.tokenizer.eos_token_id)]
    return processor.tokenizer.decode(new, skip_special_tokens=False)


def check_batched(model, processor, cfg: dict, args) -> dict:
    """Generazione a batch contro un esempio alla volta, su didascalie MAVIS e formule."""
    report = {}
    for name, prompts, max_new in (
            ("mavis", [p for p in mavis_prompts(cfg, 0, processor)][: args.check_batched], 256),
            ("unimer", formula_prompts(args.unimer_test[0][1], processor, args.check_batched // 2 + 1)[: args.check_batched], 512)):
        t = time.time()
        ref = [single(model, processor, p["example"], max_new) for p in prompts]
        t_single = time.time() - t
        t = time.time()
        got = generate_all(model, processor.tokenizer, [p["example"] for p in prompts], max_new, args.batch_size)
        t_batch = time.time() - t
        same = [r == g["text"] for r, g in zip(ref, got)]
        prefix = []
        for r, g in zip(ref, got):
            k = next((i for i, (a, b) in enumerate(zip(r, g["text"])) if a != b), min(len(r), len(g["text"])))
            prefix.append(k / max(1, len(r)))
        report[name] = {"n": len(prompts), "identical": sum(same), "mean_common_prefix": sum(prefix) / len(prefix),
                        "seconds_single": round(t_single, 1), "seconds_batched": round(t_batch, 1),
                        "examples": [{"single": r[:300], "batched": g["text"][:300]} for r, g, s in zip(ref, got, same)
                                     if not s][:3]}
        print(f"controllo a batch {name}: identiche {sum(same)}/{len(prompts)}, prefisso comune medio "
              f"{report[name]['mean_common_prefix']:.3f}, {t_single:.0f} s contro {t_batch:.0f} s", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="config con le sorgenti di validazione (mavis, stem, ...)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--synth-dir", required=True, type=Path)
    ap.add_argument("--unimer-test", nargs="+", default=[], help="NOME=cartella dei test (es. x4=..., dense=...)")
    ap.add_argument("--stem-real", type=Path)
    ap.add_argument("--choice-sets", default="mavis,synth,pairs,stem_real")
    ap.add_argument("--dependence-frac", type=float, default=0.35)
    ap.add_argument("--mavis-dependence", type=int, default=100)
    ap.add_argument("--per-subset", type=int, default=200)
    ap.add_argument("--tf-sources", default="stem,mavis,unimer")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--check-batched", type=int, default=0)
    args = ap.parse_args()
    args.unimer_test = [(s.partition("=")[0], Path(s.partition("=")[2])) for s in args.unimer_test]
    cfg = load_config(args.config)
    t0 = time.time()
    processor = LimiteVLProcessor.from_pretrained(cfg["decoder"])
    model = LimiteVLForConditionalGeneration.from_pretrained(
        VLConfig(decoder_path=cfg["decoder"], vision_path=cfg["vision"],
                 vision_precision=cfg.get("vision_precision", "tf32")), dtype=torch.bfloat16, adapter_dir=args.adapter)
    gen_attn = model.language_model.config._attn_implementation
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"name": args.name, "adapter": args.adapter, "config": str(args.config), "seconds": {}}

    if args.check_batched:
        model.eval()
        summary["check_batched"] = check_batched(model, processor, cfg, args)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        return

    # 1. scelta forzata (come la validazione: modalità train, flex_attention)
    t = time.time()
    model.language_model.config._attn_implementation = cfg["attn"]
    allow_flex_recompiles()
    model.train()
    items = with_modes(choice_items(cfg, args), args.dependence_frac)
    print(f"scelta forzata: {len(items)} domande", flush=True)
    score_options(model, processor, items)
    with open(args.out_dir / "choice.jsonl", "w") as fh:
        for it in items:
            fh.write(json.dumps({k: v for k, v in it.items() if k != "image"}, ensure_ascii=False) + "\n")
    summary["seconds"]["choice"] = round(time.time() - t, 1)
    acc = {}
    for it in items:
        acc.setdefault(f"{it['set']}/{it['mode']}", []).append(it["correct"])
    print("scelta forzata:", {k: f"{sum(v) / len(v):.3f} ({len(v)})" for k, v in sorted(acc.items())}, flush=True)

    # 2. teacher forcing per decimo della risposta
    t = time.time()
    tokens = {}
    cache: dict[int, str] = {}

    def decode(tid: int) -> str:
        if tid not in cache:
            cache[tid] = processor.tokenizer.decode([tid], skip_special_tokens=False)
        return cache[tid]

    srcs = {s.name: s for s in make_sources(cfg["sources"])}
    for name in filter(None, args.tf_sources.split(",")):
        if name in srcs:
            tokens[name] = teacher_forcing(model, processor, validation_samples(srcs[name], cfg["val_count"]),
                                           srcs[name].kind, cfg["max_len"], decode)
    (args.out_dir / "tokens.json").write_text(json.dumps(tokens))
    summary["seconds"]["teacher_forcing"] = round(time.time() - t, 1)

    # 3. generazione a batch
    model.language_model.config._attn_implementation = gen_attn
    model.eval()
    t = time.time()
    prompts = mavis_prompts(cfg, args.mavis_dependence, processor)
    run_generation(model, processor, prompts, 256, args.batch_size)
    with open(args.out_dir / "gen.jsonl", "w") as fh:
        for p in prompts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    summary["seconds"]["mavis_generation"] = round(time.time() - t, 1)
    print(f"didascalie MAVIS: {len(prompts)} ({time.time() - t:.0f} s)", flush=True)
    for test_name, test_dir in args.unimer_test:
        left = seconds_left_in_job()
        if left is not None and left < 600:
            print(f"tempo quasi finito: formule {test_name} saltate", flush=True)
            summary.setdefault("skipped", []).append(test_name)
            continue
        t = time.time()
        prompts = formula_prompts(test_dir, processor, args.per_subset)
        run_generation(model, processor, prompts, 512, args.batch_size)
        with open(args.out_dir / f"formulas-{test_name}.jsonl", "w") as fh:
            for p in prompts:
                pred = p.pop("output").strip()
                ref_t, pred_t = latex_tokens(p["reference"]), latex_tokens(pred)
                p.update(prediction=pred, exact=ref_t == pred_t, ref_len=len(ref_t),
                         token_acc=max(0.0, 1 - edit_distance(pred_t, ref_t) / max(1, len(ref_t))))
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        summary["seconds"][f"formulas_{test_name}"] = round(time.time() - t, 1)
        print(f"formule {test_name}: {len(prompts)} ({time.time() - t:.0f} s)", flush=True)
    summary["seconds"]["total"] = round(time.time() - t0, 1)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"fatto in {time.time() - t0:.0f} s -> {args.out_dir}")


if __name__ == "__main__":
    main()
