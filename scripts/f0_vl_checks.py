#!/usr/bin/env python3
"""Verifiche F0 di LimiteVLForConditionalGeneration (SigLIP2 so400m + proiettore + Violetto), in un job con 1 GPU.

Ogni controllo è indipendente: un errore viene registrato e si passa al successivo. Risultati in JSON
in --out. Dati: i primi campioni del primo shard di LLaVA-Pretrain. Nessun download.

Controlli
  1. caricamento: parametri per componente, chiavi dell'encoder
  2. parità solo testo: stessi hidden state di Violetto senza hook, in train ed eval; loss a blocchi
     contro la loss ufficiale
  3. hook: <|image_pad|> riceve le feature, value embeddings azzerati sulle posizioni immagine
  4. forward con immagini: loss finita su una riga impacchettata da 4096
  5. packing: loss per token di due esempi impacchettati uguale a quella degli esempi da soli
     (maschera a blocchi dalle position_ids), in fp32 e bf16; controllo negativo con posizioni continue
  5b. flex_attention contro sdpa sulla stessa riga impacchettata, in fp32 e bf16
  6. gradient checkpointing sul modello VL: gradienti del proiettore con e senza, in fp32 e bf16,
     accanto al rumore fra due backward identici
  7. mini-training F1 (proiettore e righe dei token visivi, AdamW 1e-3): la loss scende, il decoder
     non riceve gradienti
  8. throughput: configurazione F1 (righe da 4096) e F2 (riga da 32768, decoder addestrabile),
     attenzione sdpa e flex_attention
  9. generazione: inizio e fine di tre risposte testuali greedy (formato di <think> e dell'EOS nei
     dati) e generazione con immagine
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.collator import PackedCollator, iter_packed_rows  # noqa: E402
from limite_vl.data import iter_shard, llava_pretrain_example  # noqa: E402
from limite_vl.modeling import LimiteVLForConditionalGeneration, VLConfig  # noqa: E402
from limite_vl.processing import IGNORE, IMAGE_PAD_ID, LimiteVLProcessor, image_to_patches  # noqa: E402
from limite_vl.train_utils import LayerCheckpointing, chunked_ce  # noqa: E402

RESULTS: dict[str, object] = {}


def record(name: str):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            print(f"\n=== {name} ===", flush=True)
            try:
                RESULTS[name] = {"ok": True, **(fn(*a, **k) or {})}
            except Exception as exc:  # noqa: BLE001 - vogliamo continuare
                traceback.print_exc()
                RESULTS[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            RESULTS[name]["seconds"] = round(time.time() - t0, 2)  # type: ignore[index]
            print(json.dumps(RESULTS[name], indent=1, default=str), flush=True)
        return run
    return deco


def to_cuda(batch: dict) -> dict:
    return {k: v.cuda(non_blocking=True) for k, v in batch.items() if k != "num_examples"}


def token_losses(model, batch: dict) -> torch.Tensor:
    """Loss per token (posizioni con label) in fp32, nell'ordine della riga."""
    out = model(**{k: v for k, v in batch.items() if k != "labels"})
    h, y = out.hidden_states[:, :-1], batch["labels"][:, 1:]
    keep = y != IGNORE
    logits = model.logits(h[keep]).float()
    return F.cross_entropy(logits, y[keep], reduction="none")


class Unhooked:
    """Toglie temporaneamente gli hook sugli embedding: il decoder torna Violetto originale."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for h in self.model._hooks:
            h.remove()

    def __exit__(self, *exc):
        m = self.model
        m._hooks = [m.language_model.get_input_embeddings().register_forward_hook(m._embed_hook)]
        if m.config.image_value_embeds == "zero":
            m._hooks.append(m.language_model.model.value_embeds.register_forward_hook(m._value_embed_hook))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True, help="snapshot di Violetto")
    ap.add_argument("--vision", required=True, help="snapshot di SigLIP2 NaFlex")
    ap.add_argument("--shard", required=True, help="shard tar di LLaVA-Pretrain")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f0_vl_checks.py va eseguito dentro un job SLURM")
    if not torch.cuda.is_available():
        sys.exit("nessuna GPU CUDA disponibile")
    torch.manual_seed(args.seed)
    RESULTS["env"] = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(0), "decoder": args.decoder, "vision": args.vision,
                      "shard": args.shard}

    t0 = time.time()
    processor = LimiteVLProcessor.from_pretrained(args.decoder)
    examples = []
    for _, files in iter_shard(args.shard):
        examples.append(llava_pretrain_example(processor, files))
        if len(examples) >= args.samples:
            break
    stats = Counter()
    rows_4k = list(iter_packed_rows(examples[:700], 4096, buffer_size=700, stats=stats))
    rows_32k = list(iter_packed_rows(examples[700:], 32768, buffer_size=len(examples) - 700))
    collate_4k, collate_32k = PackedCollator(4096), PackedCollator(32768)
    RESULTS["data"] = {"examples": len(examples), "seconds": round(time.time() - t0, 1), "rows_4096": len(rows_4k),
                       "fill_4096": round(stats["tokens"] / (stats["rows"] * 4096), 3),
                       "examples_per_row_4096": round(len(examples[:700]) / max(1, len(rows_4k)), 1),
                       "row_32768_tokens": sum(len(e) for e in rows_32k[0]) if rows_32k else 0}
    print(RESULTS["data"], flush=True)

    t0 = time.time()
    config = VLConfig(decoder_path=args.decoder, vision_path=args.vision)
    model = LimiteVLForConditionalGeneration.from_pretrained(config, dtype=torch.bfloat16)
    lm = model.language_model
    load_seconds = time.time() - t0

    @record("1_load")
    def _():
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        return {"seconds_load": round(load_seconds, 1), "vision_params": count(model.vision),
                "projector_params": count(model.projector), "decoder_params": count(lm),
                "vision_token_rows": list(model.vision_token_embeds.shape),
                "vision_dtype": str(model.vision.embeddings.patch_embedding.weight.dtype),
                "projector_dtype": str(model.projector.fc1.weight.dtype),
                "gpu_mem_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}
    _()

    @record("2_text_parity")
    def _():
        ids = torch.randint(0, 150000, (1, 2048), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
        out = {}
        with torch.no_grad():
            for mode in ("eval", "train"):
                model.train(mode == "train")
                vl = model(input_ids=ids).hidden_states
                with Unhooked(model):
                    ref = lm.model(input_ids=ids, use_cache=False).last_hidden_state
                out[f"{mode}_identical"] = bool(torch.equal(vl, ref))
            model.eval()
            official = lm(input_ids=ids, labels=ids, use_cache=False).loss.float().item()
            mine = model(input_ids=ids, labels=ids).loss.item()
        out.update(loss_official=official, loss_vl_chunked=mine, abs_diff=abs(official - mine))
        out["pass"] = out["eval_identical"] and out["train_identical"] and abs(official - mine) < 2e-2
        return out
    _()

    @record("3_hooks")
    def _():
        model.eval()
        batch = to_cuda(collate_4k(rows_4k[:1]))
        ids = batch["input_ids"]
        mask = ids == IMAGE_PAD_ID
        n = int(mask.sum())
        fake = torch.randn(n, lm.config.hidden_size, device="cuda")
        embed = lm.get_input_embeddings()
        with torch.no_grad():
            with Unhooked(model):
                plain = embed(ids)
            model._image_features = fake
            hooked = embed(ids)
            ve = lm.model.value_embeds(ids)
        leftover = model._image_features is not None
        model._image_features = None
        return {"image_tokens": n,
                "features_in_place": bool(torch.equal(hooked[mask], fake.to(hooked.dtype))),
                "text_rows_unchanged": bool(torch.equal(hooked[~mask], plain[~mask])),
                "value_embeds_zero_on_images": bool((ve[mask] == 0).all()),
                "value_embeds_text_nonzero": bool((ve[~mask].abs().sum(-1) > 0).float().mean() > 0.99),
                "features_consumed": not leftover}
    _()

    @record("4_image_forward")
    def _():
        model.eval()
        batch = to_cuda(collate_4k(rows_4k[:1]))
        with torch.no_grad():
            features = model.encode_images(batch["pixel_values"], batch["pixel_attention_mask"], batch["spatial_shapes"])
            loss = model(**batch).loss.item()
        emb = lm.get_input_embeddings().weight
        return {"images": int(batch["spatial_shapes"].shape[0]), "image_tokens": int(features.shape[0]),
                "feature_rms": features.float().pow(2).mean().sqrt().item(),
                "text_embedding_rms": emb.float().pow(2).mean().sqrt().item(),
                "loss_untrained_projector": loss, "finite": bool(torch.isfinite(torch.tensor(loss)))}
    _()

    def backbone_dtype(dtype: torch.dtype) -> None:
        """Encoder e decoder in `dtype` (proiettore sempre fp32); ricostruisce le costanti di inferenza di Limite."""
        mode = model.training
        model.vision.to(dtype)
        lm.to(dtype)
        model.train(mode)

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a - b).norm() / a.norm()).item()

    # Confronti in fp32 e in modalità train: in eval il MUDD usa costanti bf16 (modeling_limite.py), e in bf16
    # le differenze fra kernel di attenzione coprirebbero un eventuale errore. Limite non ha dropout.
    @record("5_packing")
    def _():
        a, b = rows_4k[0][0], rows_4k[0][1]
        n_a = int((torch.from_numpy(a.labels[1:]) != IGNORE).sum())
        out = {}
        try:
            for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
                backbone_dtype(dtype)
                model.train()
                with torch.no_grad():
                    alone = torch.cat([token_losses(model, to_cuda(PackedCollator(len(e))([[e]]))) for e in (a, b)])
                    packed_batch = to_cuda(PackedCollator(len(a) + len(b))([[a, b]]))
                    packed = token_losses(model, packed_batch)
                    control_batch = dict(packed_batch)
                    control_batch["position_ids"] = torch.arange(len(a) + len(b), device="cuda").unsqueeze(0)
                    control = token_losses(model, control_batch)
                diff = (packed - alone).abs()
                out[name] = {"mean_loss_alone": alone.mean().item(), "mean_loss_packed": packed.mean().item(),
                             "max_abs_diff": diff.max().item(), "mean_abs_diff": diff.mean().item(),
                             "control_mean_abs_diff_b": (control[n_a:] - alone[n_a:]).abs().mean().item()}
        finally:
            backbone_dtype(torch.bfloat16)
            model.eval()
        out["labelled_tokens"] = n_a + int((torch.from_numpy(b.labels[1:]) != IGNORE).sum())
        fp32 = out["fp32"]
        out["pass"] = fp32["max_abs_diff"] < 1e-3 and fp32["control_mean_abs_diff_b"] > 100 * fp32["mean_abs_diff"]
        return out
    _()

    @record("5b_flex_parity")
    def _():
        batch = to_cuda(collate_4k(rows_4k[:1]))
        out = {}
        try:
            for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
                backbone_dtype(dtype)
                model.train()
                losses = {}
                for attn in ("sdpa", "flex_attention"):
                    lm.config._attn_implementation = attn
                    with torch.no_grad():
                        losses[attn] = token_losses(model, batch)
                lm.config._attn_implementation = "sdpa"
                out[name] = {"loss_sdpa": losses["sdpa"].mean().item(), "loss_flex": losses["flex_attention"].mean().item(),
                             "max_abs_diff": (losses["sdpa"] - losses["flex_attention"]).abs().max().item()}
        finally:
            lm.config._attn_implementation = "sdpa"
            backbone_dtype(torch.bfloat16)
            model.eval()
        out["pass"] = out["fp32"]["max_abs_diff"] < 1e-3
        return out
    _()

    @record("6_gradient_checkpointing_vl")
    def _():
        batch = to_cuda(collate_4k(rows_4k[1:2]))
        model.set_trainable("f1")

        def grads():
            model.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            return loss.item(), torch.cat([p.grad.flatten().float() for p in model.projector.parameters()])

        out = {}
        try:
            for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
                backbone_dtype(dtype)
                model.train()
                loss_a, g_a = grads()
                _, g_again = grads()  # rumore fra due backward identici (kernel non deterministici)
                with LayerCheckpointing(lm):
                    loss_b, g_b = grads()
                out[name] = {"loss_plain": loss_a, "loss_ckpt": loss_b, "rel_diff_repeat": rel(g_a, g_again),
                             "rel_diff_ckpt": rel(g_a, g_b)}
        finally:
            model.zero_grad(set_to_none=True)
            backbone_dtype(torch.bfloat16)
        out["pass"] = out["fp32"]["rel_diff_ckpt"] < 1e-4
        return out
    _()

    @record("7_f1_smoke_training")
    def _():
        model.train()
        model.set_trainable("f1")
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
        losses, tokens, t_start = [], 0, None
        steps = min(40, len(rows_4k) - 2)
        with LayerCheckpointing(lm):
            for step in range(steps):
                batch = to_cuda(collate_4k(rows_4k[2 + step:3 + step]))
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(round(loss.item(), 4))
                if step == 1:
                    torch.cuda.synchronize()
                    t_start = time.time()
                elif step > 1:
                    tokens += batch["input_ids"].numel()
        torch.cuda.synchronize()
        decoder_grads = sum(p.grad is not None for p in lm.parameters())
        trainable = sum(p.numel() for p in params)
        del opt
        first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
        return {"steps": steps, "trainable_params": trainable, "losses": losses, "first10": first, "last10": last,
                "decoder_params_with_grad": decoder_grads,
                "tokens_per_s": round(tokens / (time.time() - t_start)) if t_start else None,
                "pass": last < first and decoder_grads == 0}
    _()

    def bench(rows_per_step: list[list], collate, phase: str, use_ckpt: bool, steps: int = 4):
        model.train()
        model.set_trainable(phase)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-6, fused=True)
        torch.cuda.reset_peak_memory_stats()
        times, vision_times = [], []
        ctx = LayerCheckpointing(lm) if use_ckpt else None
        if ctx:
            ctx.__enter__()
        try:
            for i in range(steps + 1):
                batch = to_cuda(collate(rows_per_step[i % len(rows_per_step)]))
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    model.encode_images(batch["pixel_values"], batch["pixel_attention_mask"], batch["spatial_shapes"])
                torch.cuda.synchronize()
                t1 = time.time()
                loss = model(**batch).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if i:  # il primo passo scalda i kernel
                    vision_times.append(t1 - t0)
                    times.append(time.time() - t1)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
            del opt
            model.zero_grad(set_to_none=True)
            lm.requires_grad_(False)
        dt = sum(times) / len(times)
        n_tok = batch["input_ids"].numel()
        return {"phase": phase, "tokens_per_step": n_tok, "checkpointing": use_ckpt,
                "attn": lm.config._attn_implementation, "s_per_step": round(dt, 3),
                "tokens_per_s": round(n_tok / dt), "vision_s_per_step": round(sum(vision_times) / len(vision_times), 3),
                "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}

    @record("8_throughput")
    def _():
        runs = []
        f1_rows_1 = [[r] for r in rows_4k[:5]]
        f1_rows_8 = [rows_4k[i:i + 8] for i in range(0, 40, 8)]
        plan = [("sdpa", f1_rows_1, collate_4k, "f1", True), ("sdpa", f1_rows_1, collate_4k, "f1", False),
                ("sdpa", f1_rows_8, collate_4k, "f1", True)]
        if rows_32k:
            plan.append(("sdpa", [rows_32k[:1]], collate_32k, "f2", True))
        plan += [("flex_attention", f1_rows_1, collate_4k, "f1", True)]
        if rows_32k:
            plan.append(("flex_attention", [rows_32k[:1]], collate_32k, "f2", True))
        for attn, rows, collate, phase, ckpt in plan:
            lm.config._attn_implementation = attn
            try:
                runs.append(bench(rows, collate, phase, ckpt))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                runs.append({"phase": phase, "attn": attn, "checkpointing": ckpt,
                             "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            finally:
                lm.config._attn_implementation = "sdpa"
                torch.cuda.empty_cache()
            print(runs[-1], flush=True)
        return {"runs": runs}
    _()

    @record("9_generation")
    def _():
        model.eval()
        tok = processor.tokenizer
        result = {"text": []}
        for question in ("What is 7 times 8? Answer briefly.", "Find the remainder when 2^10 is divided by 7.",
                         "How many positive divisors does 36 have?"):
            prompt = processor.encode([{"role": "user", "content": question}], add_generation_prompt=True)
            ids = torch.from_numpy(prompt.input_ids).long().unsqueeze(0).cuda()
            out = model.generate(ids, do_sample=False, max_new_tokens=1536, use_cache=True)
            new = out[0, ids.shape[1]:].tolist()
            result["text"].append({"new_tokens": len(new), "tail_ids": new[-4:],
                                   "head": tok.decode(new[:12], skip_special_tokens=False),
                                   "tail": tok.decode(new[-12:], skip_special_tokens=False)})
        _, files = next(iter_shard(args.shard))
        from io import BytesIO

        from PIL import Image
        with Image.open(BytesIO(files["jpg"])) as image:
            image.load()
            patches, grid = image_to_patches(image)
        ex = processor.encode([{"role": "user", "content": [{"type": "image"},
                                                            {"type": "text", "text": "Describe the image."}]}],
                              [(patches, grid)], add_generation_prompt=True)
        batch = to_cuda(PackedCollator(len(ex))([[ex]]))
        out = model.generate(batch["input_ids"], pixel_values=batch["pixel_values"],
                             pixel_attention_mask=batch["pixel_attention_mask"],
                             spatial_shapes=batch["spatial_shapes"], do_sample=False, max_new_tokens=32, use_cache=True)
        result["image_prompt_tokens"] = len(ex)
        result["image_untrained_output"] = tok.decode(out[0, len(ex):].tolist(), skip_special_tokens=False)
        return result
    _()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\nrisultati: {args.out}")
    failed = [k for k, v in RESULTS.items() if isinstance(v, dict) and (v.get("ok") is False or v.get("pass") is False)]
    print("controlli falliti:", failed or "nessuno")


if __name__ == "__main__":
    main()
