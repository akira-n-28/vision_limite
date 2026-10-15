#!/usr/bin/env python3
"""Test CPU delle parti di scripts/train_f2.py che non richiedono la GPU: pesi master fp32 (train_utils.MasterWeights),
scheduler, costruzione degli ottimizzatori nei due modi (normuon, adamw) e un passo completo su un modello giocattolo.
Da eseguire in slurm/tests_cpu.sbatch."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from limite_vl.train_utils import MasterWeights  # noqa: E402
from train_f2 import build_optimizer, lr_factor  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


# Parametri che Limite tiene in fp32 anche con il modello in bf16 (_fp32_parameter: gate, lambda residue, MUDD).
FP32_NAMES = {"language_model.model.layers.0.attn_gate.weight", "language_model.model.layers.0.mlp_lambda"}


def toy_named():
    """Nomi come quelli del decoder dentro il modello VL (prefisso language_model.), pesi bf16 tranne FP32_NAMES."""
    torch.manual_seed(0)
    shapes = {"language_model.model.embed_tokens.weight": (50, 16),
              "language_model.model.layers.0.self_attn.q_proj.weight": (16, 16),
              "language_model.model.layers.0.mlp.down_proj.weight": (16, 32),
              "language_model.model.layers.0.attn_gate.weight": (2, 4),
              "language_model.model.layers.0.mlp_lambda": ()}
    return [(n, torch.nn.Parameter((torch.randn(s) * 0.02).to(torch.float32 if n in FP32_NAMES else torch.bfloat16)))
            for n, s in shapes.items()]


def test_lr_factor():
    f = [lr_factor(s, 100, 0.02) for s in range(100)]
    check("lr_factor: warmup lineare su 2 passi", f[0] == 0.5 and f[1] == 1.0, str(f[:3]))
    check("lr_factor: decadimento lineare verso zero", abs(f[51] - 0.5) < 0.02 and 0 < f[99] < 0.02, f"{f[51]} {f[99]}")


def test_masters():
    named = toy_named()
    masters = MasterWeights(named)
    check("MasterWeights: copie fp32 con gli stessi nomi", all(m.dtype == torch.float32 for m in masters.parameters())
          and [n for n, _ in masters.named_parameters()] == [n for n, _ in named])
    check("MasterWeights: i parametri fp32 sono master di se stessi, i bf16 hanno una copia",
          all((m is p) == (n in FP32_NAMES) for (n, p), m in zip(named, masters.parameters())))
    for _, p in named:
        p.grad = torch.ones_like(p)
    masters.grads_to_masters(4.0)
    check("grads_to_masters: gradienti fp32 divisi, quelli bf16 liberati",
          all(m.grad.dtype == torch.float32 and torch.all(m.grad == 0.25) for m in masters.parameters())
          and all((p.grad is None) == (n not in FP32_NAMES) for n, p in named))
    with torch.no_grad():
        for m in masters.parameters():
            m.add_(1e-5)  # passo che in bf16 si perderebbe
    masters.masters_to_model()
    check("masters_to_model: pesi bf16 = master arrotondati",
          all(torch.equal(p, m.to(p.dtype)) for (_, p), m in zip(named, masters.parameters())))
    state = {k: v.clone() for k, v in masters.state_dict().items()}
    other = MasterWeights(toy_named())
    other.load_state_dict(state)
    check("state_dict/load_state_dict: stessi master e stessi pesi bf16",
          all(torch.equal(a, b) for a, b in zip(other.parameters(), masters.parameters())))


def test_optimizers():
    adapter = [torch.nn.Parameter(torch.randn(8, 4) * 0.02), torch.nn.Parameter(torch.zeros(4))]
    cfg = {"optimizer": "normuon", "phase_scale": 0.15, "projector_lr": 1e-4, "adamw_lr": 1e-5}
    named = toy_named()
    masters = MasterWeights(named)
    opts, text = build_optimizer(cfg, masters, adapter)
    groups = {g["name"]: g for o in opts for g in o.param_groups}
    print(text)
    check("normuon: matrici nascoste a 0.0027, adapter a 1e-4, peak_lr in ogni gruppo",
          abs(groups["hidden_matrices"]["peak_lr"] - 0.0027) < 1e-12 and groups["adapter"]["peak_lr"] == 1e-4
          and all("peak_lr" in g for g in groups.values()), str({k: g["lr"] for k, g in groups.items()}))
    check("normuon: ogni master in un solo gruppo", sum(len(g["params"]) for n, g in groups.items() if n != "adapter")
          == len(masters.parameters()))
    before = [p.detach().clone() for _, p in named]
    for _, p in named:
        p.grad = torch.randn_like(p)
    for p in adapter:
        p.grad = torch.randn_like(p)
    masters.grads_to_masters(1.0)
    for o in opts:
        o.step()
        o.zero_grad(set_to_none=True)
    masters.masters_to_model()
    changed = [not torch.equal(b, p) for b, (_, p) in zip(before, named)]
    check("normuon: un passo cambia i pesi bf16 del modello", all(changed), str(changed))
    cfg["optimizer"] = "adamw"
    opts, _ = build_optimizer(cfg, MasterWeights(toy_named()), adapter)
    groups = {g["name"]: g for o in opts for g in o.param_groups}
    check("adamw: decoder a 1e-5 in un solo gruppo, adapter a 1e-4",
          set(groups) == {"decoder", "adapter"} and groups["decoder"]["peak_lr"] == 1e-5, str(set(groups)))


if __name__ == "__main__":
    test_lr_factor()
    test_masters()
    test_optimizers()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
