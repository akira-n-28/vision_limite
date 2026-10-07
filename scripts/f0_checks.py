#!/usr/bin/env python3
"""Verifiche F0 su Limite 1B - Violetto, da eseguire in un job SLURM con 1 GPU.

Ogni controllo è indipendente: un errore viene registrato e si passa al successivo.
Risultati in JSON in --out. Nessun download: il modello si carica da un percorso locale.

Controlli
  1. caricamento offline, dtype, numero di parametri
  2. token visivi: id e norma delle righe di embedding rispetto alla media
  3. comportamento di inputs_embeds (atteso: ValueError)
  4. gradient checkpointing: supporto nativo e wrapper per layer con torch.utils.checkpoint,
     confronto dei gradienti con il calcolo senza checkpointing
  5. loss a blocchi con lo stesso soft-capping: confronto con la loss ufficiale
  6. throughput fwd+bwd+AdamW e memoria di picco per alcune lunghezze di sequenza
  7. nomi e forme dei parametri, e loro assegnazione ai gruppi della Tabella 1 (NorMuon/AdamW)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VISION_TOKENS = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|vision_pad|>"]
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


def load(path: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
    ).cuda()
    return tok, model


def decoder(model):
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        raise RuntimeError("non trovo model.model.layers: controllare la struttura in modeling_limite.py")
    return inner


def chunked_ce(model, hidden: torch.Tensor, labels: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """CE con soft-capping calcolata a blocchi di token; hidden = output dopo la norm finale."""
    softcap = getattr(model, "_softcapped_logits", None)
    if softcap is None:
        raise RuntimeError("manca model._softcapped_logits")
    h = hidden[:, :-1]
    y = labels[:, 1:]
    total = hidden.new_zeros((), dtype=torch.float32)
    count = (y != -100).sum()
    for s in range(0, h.shape[1], chunk):
        def piece(hh, yy):
            logits = softcap(hh).float()
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yy.reshape(-1),
                                   ignore_index=-100, reduction="sum")
        total = total + checkpoint(piece, h[:, s:s + chunk], y[:, s:s + chunk], use_reentrant=False)
    return total / count.clamp(min=1)


class LayerCheckpointing:
    """Avvolge ogni decoder layer in torch.utils.checkpoint (non rientrante)."""

    def __init__(self, model):
        self.layers = decoder(model).layers
        self.orig = [layer.forward for layer in self.layers]

    def __enter__(self):
        for layer, fwd in zip(self.layers, self.orig):
            layer.forward = (lambda f: (lambda *a, **k: checkpoint(f, *a, use_reentrant=False, **k)))(fwd)
        return self

    def __exit__(self, *exc):
        for layer, fwd in zip(self.layers, self.orig):
            layer.forward = fwd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="cartella locale dello snapshot (snapshots/<commit>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-lens", default="8192,32768")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("f0_checks.py va eseguito dentro un job SLURM")
    if not torch.cuda.is_available():
        sys.exit("nessuna GPU CUDA disponibile")
    torch.manual_seed(args.seed)
    RESULTS["env"] = {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "model_path": args.model,
    }

    tok, model = load(args.model, torch.bfloat16)
    vocab = model.get_input_embeddings().weight.shape[0]

    @record("1_load")
    def _():
        n = sum(p.numel() for p in model.parameters())
        return {"params": n, "dtype": str(next(model.parameters()).dtype), "class": type(model).__name__,
                "vocab_rows": vocab, "supports_gradient_checkpointing":
                    bool(getattr(model, "supports_gradient_checkpointing", False))}
    _()

    @record("2_vision_tokens")
    def _():
        emb = model.get_input_embeddings().weight.float()
        norms = emb.norm(dim=-1)
        out = {"mean_row_norm": norms.mean().item(), "median_row_norm": norms.median().item()}
        for t in VISION_TOKENS:
            i = tok.convert_tokens_to_ids(t)
            out[t] = {"id": i, "norm": norms[i].item() if isinstance(i, int) and 0 <= i < vocab else None}
        return out
    _()

    @record("3_inputs_embeds")
    def _():
        ids = torch.randint(0, 150000, (1, 16), device="cuda")
        emb = model.get_input_embeddings()(ids)
        try:
            model(inputs_embeds=emb)
            return {"inputs_embeds_accepted": True}
        except ValueError as exc:
            return {"inputs_embeds_accepted": False, "message": str(exc)}
    _()

    @record("4_gradient_checkpointing")
    def _():
        native = None
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.gradient_checkpointing_disable()
            native = "enable() accettato (verificare che abbia effetto)"
        except Exception as exc:  # noqa: BLE001
            native = f"{type(exc).__name__}: {exc}"
        m32 = model.float()
        m32.train()
        ids = torch.randint(0, 150000, (1, 1536), device="cuda", generator=torch.Generator("cuda").manual_seed(1))

        def grads():
            m32.zero_grad(set_to_none=True)
            out = m32(input_ids=ids, labels=ids)
            out.loss.backward()
            return out.loss.item(), torch.cat([p.grad.flatten() for p in m32.parameters() if p.grad is not None])

        loss_a, g_a = grads()
        with LayerCheckpointing(m32):
            loss_b, g_b = grads()
        rel = ((g_a - g_b).norm() / g_a.norm()).item()
        m32.zero_grad(set_to_none=True)
        model.to(torch.bfloat16)
        return {"native": native, "loss_plain": loss_a, "loss_ckpt": loss_b,
                "grad_rel_diff": rel, "pass": rel < 1e-4}
    _()

    @record("5_chunked_loss")
    def _():
        model.eval()
        ids = torch.randint(0, 150000, (1, 4096), device="cuda", generator=torch.Generator("cuda").manual_seed(2))
        with torch.no_grad():
            ref = model(input_ids=ids, labels=ids).loss.float().item()
            hidden = decoder(model)(input_ids=ids).last_hidden_state
            mine = chunked_ce(model, hidden, ids).item()
        return {"loss_official": ref, "loss_chunked": mine, "abs_diff": abs(ref - mine),
                "pass": abs(ref - mine) < 2e-2}
    _()

    def bench(seq: int, use_ckpt: bool, steps: int = 4):
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, 150000, (1, seq), device="cuda")
        times = []
        ctx = LayerCheckpointing(model) if use_ckpt else None
        if ctx:
            ctx.__enter__()
        try:
            for i in range(steps + 1):
                torch.cuda.synchronize()
                t0 = time.time()
                hidden = decoder(model)(input_ids=ids).last_hidden_state
                loss = chunked_ce(model, hidden, ids)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if i:  # il primo passo scalda i kernel
                    times.append(time.time() - t0)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
            del opt
        dt = sum(times) / len(times)
        return {"seq": seq, "checkpointing": use_ckpt, "s_per_step": round(dt, 3),
                "tokens_per_s": round(seq / dt), "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}

    @record("6_throughput_bf16_params")
    def _():
        # Nota: pesi bf16 + AdamW, solo per misurare tempi e memoria delle attivazioni.
        # Il training vero userà pesi master fp32 (circa 16 GB in più di stato fisso).
        rows = []
        for seq in [int(s) for s in args.seq_lens.split(",")]:
            for use_ckpt in (True, False):
                try:
                    rows.append(bench(seq, use_ckpt))
                except torch.cuda.OutOfMemoryError:
                    rows.append({"seq": seq, "checkpointing": use_ckpt, "oom": True})
                torch.cuda.empty_cache()
                print(rows[-1], flush=True)
        return {"runs": rows}
    _()

    @record("7_param_groups")
    def _():
        from limite_vl.optim.groups import assign_groups, describe
        names = [(n, list(p.shape)) for n, p in model.named_parameters()]
        out = {"param_names": names}
        try:
            groups = assign_groups(model)
            text = describe(groups)
            print(text)
            out["assignment"] = {g: [n for n, _ in items] for g, items in groups.items()}
            out["assigned_ok"] = True
        except ValueError as exc:
            out["assigned_ok"] = False
            out["unassigned"] = str(exc)
        return out
    _()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\nrisultati: {args.out}")
    failed = [k for k, v in RESULTS.items() if isinstance(v, dict) and v.get("ok") is False]
    print("controlli falliti:", failed or "nessuno")


if __name__ == "__main__":
    main()
