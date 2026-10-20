#!/usr/bin/env python3
"""Materiale per l'analisi degli errori di F1 (scripts/f1_error_report.py) con un adapter. Job con GPU.

Usa gli stessi campioni di validazione del training (i primi val_count dell'ultimo shard di ogni sorgente):
1. teacher forcing con immagine vera e bianca. Le risposte sono identiche, cambia solo l'immagine: per ogni token
   con label si confrontano loss e top-1 nei due modi. Si salvano aggregati per stringa del token (conteggio,
   somme delle loss, top-1 giusti), per decimo di posizione nella risposta e le confusioni token atteso -> top-1
   con l'immagine vera, così il report può raggruppare i token per tipo senza rifare il passaggio su GPU;
2. generazione sui primi campioni di ogni sorgente, con lo stesso prompt e lo stesso prefill del training:
   greedy con l'immagine vera (real), greedy con l'immagine bianca (white: quello che il decoder scrive senza
   vedere nulla) e campionata con l'immagine vera (sampled, temperatura 0.6, top_p 0.95: come la valutazione di
   Violetto). Ogni riga si scrive appena pronta; se il tempo del job sta per finire la generazione si ferma.
Uscite in OUT_DIR: tokens.json, gen.jsonl, summary.json.

    srun $RUN_PYTHON -u -B scripts/f1_error_gen.py --config CONFIG --adapter DIR --name NOME --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_f1 import load_config, seconds_left_in_job, to_cuda  # noqa: E402  (stessa cartella scripts/)

from limite_vl.collator import PackedCollator, iter_packed_rows  # noqa: E402
from limite_vl.data import build_example, generation_prompt  # noqa: E402
from limite_vl.modeling import LimiteVLForConditionalGeneration, VLConfig  # noqa: E402
from limite_vl.processing import IGNORE, IMAGE_PAD_ID, LimiteVLProcessor  # noqa: E402
from limite_vl.stream import make_sources, validation_samples  # noqa: E402
from limite_vl.train_utils import allow_flex_recompiles  # noqa: E402

GEN_MODES = ("real", "white", "sampled")
DEFAULT_COUNTS = {"real": "mavis=80,stem=20,pixmo=30,llava=40", "white": "mavis=80,stem=20,pixmo=15,llava=40",
                  "sampled": "stem=10,pixmo=15"}
DEFAULT_MAX_NEW = "mavis=256,stem=1024,pixmo=512,llava=96"
GEN_ORDER = ("mavis", "stem", "pixmo", "llava")


def parse_spec(spec: str) -> dict[str, int]:
    out = {}
    for item in filter(None, spec.split(",")):
        name, _, value = item.partition("=")
        out[name.strip()] = int(value)
    return out


@torch.no_grad()
def token_outputs(model, batch: dict) -> tuple[torch.Tensor, ...]:
    """Per i token con label: atteso, loss, top-1 e indice della risposta (run contigue di label nella riga)."""
    labels = batch.pop("labels")
    hidden = model(**batch).hidden_states[:, :-1]
    y = labels[:, 1:]
    mask = y != IGNORE
    h, target = hidden[mask], y[mask]
    losses, preds = [], []
    for s in range(0, h.shape[0], 1024):
        logits = model.logits(h[s:s + 1024]).float()
        losses.append(F.cross_entropy(logits, target[s:s + 1024], reduction="none"))
        preds.append(logits.argmax(-1))
    starts = mask & (torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1) == 0)
    answer = (torch.cumsum(starts.flatten().long(), 0).view_as(mask) - 1)[mask]
    return target, torch.cat(losses), torch.cat(preds), answer


def teacher_forcing(model, processor, raw: list, kind: str, max_len: int, decode) -> dict:
    """Aggregati di loss e top-1 con immagine vera e bianca sugli stessi token."""
    rows = {mode: list(iter_packed_rows([build_example(processor, kind, k, f, image_mode=mode) for k, f in raw],
                                        max_len, buffer_size=len(raw)))
            for mode in ("real", "white")}
    if [[len(e) for e in r] for r in rows["real"]] != [[len(e) for e in r] for r in rows["white"]]:
        sys.exit(f"{kind}: righe diverse fra immagine vera e bianca")
    collate = PackedCollator(max_len)
    by_token: dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0, 0, 0])
    by_decile = [[0, 0.0, 0.0, 0, 0] for _ in range(10)]
    confusions: Counter = Counter()
    for i in range(len(rows["real"])):
        out = {}
        for mode in ("real", "white"):
            batch = to_cuda({k: v for k, v in collate(rows[mode][i:i + 1]).items() if k != "num_examples"})
            out[mode] = [t.cpu() for t in token_outputs(model, batch)]
        target, loss_r, pred_r, answer = out["real"]
        target_w, loss_w, pred_w, _ = out["white"]
        if not torch.equal(target, target_w):
            sys.exit(f"{kind}: token con label diversi fra immagine vera e bianca")
        lengths = torch.bincount(answer)
        offsets = torch.cumsum(lengths, 0) - lengths
        position = torch.arange(len(answer)) - offsets[answer]
        decile = torch.clamp((10 * position) // lengths[answer], max=9)
        for t, lr, lw, pr, pw, d in zip(target.tolist(), loss_r.tolist(), loss_w.tolist(), pred_r.tolist(),
                                        pred_w.tolist(), decile.tolist()):
            ok_r, ok_w = int(pr == t), int(pw == t)
            for acc in (by_token[decode(t)], by_decile[d]):
                acc[0] += 1
                acc[1] += lr
                acc[2] += lw
                acc[3] += ok_r
                acc[4] += ok_w
            if not ok_r:
                confusions[(decode(t), decode(pr))] += 1
    return {"by_token": dict(by_token), "by_decile": by_decile,
            "confusions": [[a, b, n] for (a, b), n in confusions.most_common(600)]}


@torch.no_grad()
def generate(model, processor, example, mode: str, max_new_tokens: int, seed: int) -> tuple[str, bool, int]:
    batch = to_cuda(PackedCollator(len(example))([[example]]))
    kwargs = {"do_sample": False}
    if mode == "sampled":
        torch.manual_seed(seed)
        kwargs = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 0}
    out = model.generate(batch["input_ids"], pixel_values=batch.get("pixel_values"),
                         pixel_attention_mask=batch.get("pixel_attention_mask"),
                         spatial_shapes=batch.get("spatial_shapes"), max_new_tokens=max_new_tokens,
                         use_cache=True, **kwargs)
    new = out[0, len(example):].tolist()
    eos = processor.tokenizer.eos_token_id in new
    if eos:
        new = new[:new.index(processor.tokenizer.eos_token_id)]
    return processor.tokenizer.decode(new, skip_special_tokens=False), eos, len(new)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="config di training (sorgenti, decoder, vision)")
    ap.add_argument("--adapter", required=True, help="cartella dell'adapter")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tf-sources", default="pixmo,stem,mavis,unimer,llava", help="sorgenti per il teacher forcing")
    for mode in GEN_MODES:
        ap.add_argument(f"--{mode}", default=DEFAULT_COUNTS[mode], help=f"campioni per sorgente in generazione {mode}")
    ap.add_argument("--max-new-tokens", default=DEFAULT_MAX_NEW)
    ap.add_argument("--reserve-s", type=float, default=180, help="ferma la generazione quando al job resta meno di così")
    args = ap.parse_args()
    cfg = load_config(args.config)
    counts = {mode: parse_spec(getattr(args, mode)) for mode in GEN_MODES}
    max_new = parse_spec(args.max_new_tokens)
    t0 = time.time()
    processor = LimiteVLProcessor.from_pretrained(cfg["decoder"])
    tokenizer = processor.tokenizer
    cache: dict[int, str] = {}

    def decode(tid: int) -> str:
        if tid not in cache:
            cache[tid] = tokenizer.decode([tid], skip_special_tokens=False)
        return cache[tid]

    sources = {s.name: s for s in make_sources(cfg["sources"])}
    model = LimiteVLForConditionalGeneration.from_pretrained(
        VLConfig(decoder_path=cfg["decoder"], vision_path=cfg["vision"],
                 vision_precision=cfg.get("vision_precision", "tf32")), dtype=torch.bfloat16, adapter_dir=args.adapter)
    gen_attn = model.language_model.config._attn_implementation
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"name": args.name, "adapter": args.adapter, "config": str(args.config), "val_count": cfg["val_count"],
               "counts": counts, "max_new_tokens": max_new, "teacher_forcing_s": {}, "generation": {}}

    # 1. teacher forcing (come la validazione del training: modalità train, flex_attention, nessun dropout)
    model.language_model.config._attn_implementation = cfg["attn"]
    allow_flex_recompiles()
    model.train()
    tokens = {}
    for name in filter(None, args.tf_sources.split(",")):
        s, t = sources[name], time.time()
        raw = validation_samples(s, cfg["val_count"])
        tokens[name] = teacher_forcing(model, processor, raw, s.kind, cfg["max_len"], decode)
        n = sum(v[0] for v in tokens[name]["by_token"].values())
        lr = sum(v[1] for v in tokens[name]["by_token"].values()) / n
        lw = sum(v[2] for v in tokens[name]["by_token"].values()) / n
        summary["teacher_forcing_s"][name] = round(time.time() - t, 1)
        print(f"teacher forcing {name}: {len(raw)} esempi, {n} token, loss vera {lr:.3f} bianca {lw:.3f} "
              f"({time.time() - t:.0f} s)", flush=True)
    (args.out_dir / "tokens.json").write_text(json.dumps(tokens))

    # 2. generazione (modalità eval e attenzione di default, come eval_f1_formulas e i campioni del training)
    model.language_model.config._attn_implementation = gen_attn
    model.eval()
    stopped = False
    with open(args.out_dir / "gen.jsonl", "w") as out:
        for name in GEN_ORDER:
            if name not in sources or stopped:
                continue
            s, t = sources[name], time.time()
            n_max = max(counts[m].get(name, 0) for m in GEN_MODES)
            done = Counter()
            for index, (key, files) in enumerate(validation_samples(s, n_max)):
                left = seconds_left_in_job()
                if left is not None and left < args.reserve_s:
                    stopped = True
                    print(f"tempo quasi finito ({left:.0f} s): generazione fermata a {name} {index}", flush=True)
                    break
                for mode in GEN_MODES:
                    if index >= counts[mode].get(name, 0):
                        continue
                    example, reference = generation_prompt(processor, s.kind, key, files,
                                                           image_mode="white" if mode == "white" else "real")
                    tg = time.time()
                    text, eos, n_new = generate(model, processor, example, mode, max_new[name], seed=index)
                    out.write(json.dumps({"source": name, "key": key, "mode": mode, "output": text, "eos": eos,
                                          "new_tokens": n_new, "seconds": round(time.time() - tg, 2),
                                          "image_tokens": int((example.input_ids == IMAGE_PAD_ID).sum()),
                                          "reference": reference}, ensure_ascii=False) + "\n")
                    out.flush()
                    done[mode] += 1
            summary["generation"][name] = {"rows": dict(done), "seconds": round(time.time() - t, 1)}
            print(f"generazione {name}: {dict(done)} ({time.time() - t:.0f} s)", flush=True)
    summary["stopped_early"] = stopped
    summary["seconds"] = round(time.time() - t0, 1)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"fatto in {time.time() - t0:.0f} s -> {args.out_dir}")


if __name__ == "__main__":
    main()
