"""Gruppi di parametri e ottimizzatori di Limite secondo la Tabella 1 del report.

Le regole sui NOMI dei parametri sono un'ipotesi: vanno verificate in F0 con l'elenco reale
(scripts/f0_checks.py, controllo "1_load" → `param_names`). Ogni parametro deve cadere in
esattamente un gruppo, altrimenti `assign_groups` fallisce con l'elenco dei casi dubbi.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from .normuon import NorMuon


@dataclass(frozen=True)
class GroupSpec:
    name: str
    optimizer: str            # "normuon" | "adamw"
    peak_lr: float            # picco del pretraining (Tabella 1)
    betas: tuple[float, float]
    wd_mult: float
    patterns: tuple[str, ...]  # regex sul nome completo del parametro
    ndim: tuple[int, ...] = (1, 2)


# Tabella 1 (picco del pretraining). L'ordine conta: vince la prima regola che corrisponde.
TABLE_1: tuple[GroupSpec, ...] = (
    GroupSpec("embed_tied", "adamw", 0.0019, (0.5, 0.95), 150, (r"embed_tokens\.weight$", r"lm_head\.weight$"), (2,)),
    GroupSpec("value_embeds", "adamw", 0.14, (0.75, 0.95), 5, (r"value_embeds\.weight$",), (2,)),
    # dense2 e dense2_mlp di Violetto sono 3D (48, 3, 32): job 29378, controllo 7_param_groups.
    GroupSpec("mudd", "adamw", 0.0019, (0.9, 0.95), 0, (r"\bmudd\b",), (0, 1, 2, 3)),
    GroupSpec("gates_xsa", "adamw", 0.0019, (0.9, 0.99), 0, (r"gate_(attn|ve|beta)|attn_gate|ve_gate|beta_gate|xsa|gamma",), (0, 1, 2)),
    GroupSpec("post_lambdas", "adamw", 0.0080, (0.9, 0.95), 0, (r"post.*lambda|lambda.*post|lambda_prime",), (0, 1)),
    GroupSpec("residual_lambdas", "adamw", 0.0016, (0.9, 0.95), 0, (r"lambda",), (0, 1)),
    GroupSpec("hidden_matrices", "normuon", 0.018, (0.95, 0.90), 1,
              (r"layers\.\d+\..*(q_proj|k_proj|v_proj|o_proj|qkv|wqkv|W_?[qkvo]|gate_proj|up_proj|down_proj|w_?gate|w_?up|w_?down)",), (2,)),
    GroupSpec("scalars", "adamw", 0.04, (0.9, 0.99), 0, (r".*",), (0, 1)),
)


def assign_groups(model: torch.nn.Module, specs=TABLE_1) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    out: dict[str, list] = {s.name: [] for s in specs}
    unmatched = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for s in specs:
            if p.ndim in s.ndim and any(re.search(pat, name) for pat in s.patterns):
                out[s.name].append((name, p))
                break
        else:
            unmatched.append((name, tuple(p.shape)))
    if unmatched:
        raise ValueError("parametri senza gruppo (correggere le regole in TABLE_1):\n" +
                         "\n".join(f"  {n} {s}" for n, s in unmatched))
    return out


def describe(groups) -> str:
    lines = []
    for g, items in groups.items():
        n = sum(p.numel() for _, p in items)
        lines.append(f"{g:18s} {len(items):4d} tensori {n / 1e6:9.2f} M")
        for name, p in items[:6]:
            lines.append(f"    {name} {tuple(p.shape)}")
        if len(items) > 6:
            lines.append(f"    … altri {len(items) - 6}")
    return "\n".join(lines)


def build_optimizers(model: torch.nn.Module, phase_scale: float, specs=TABLE_1, extra_adamw=()):
    """Restituisce (normuon, adamw) con learning rate = picco × phase_scale (0.15 per l'SFT del report).

    `extra_adamw`: gruppi già pronti per moduli nuovi (es. il proiettore), come dict per AdamW.
    """
    groups = assign_groups(model, specs)
    by_name = {s.name: s for s in specs}
    muon_groups, adam_groups = [], []
    for gname, items in groups.items():
        if not items:
            continue
        s = by_name[gname]
        params = [p for _, p in items]
        lr = s.peak_lr * phase_scale
        if s.optimizer == "normuon":
            muon_groups.append(dict(params=params, lr=lr, momentum=s.betas[0], beta2=s.betas[1],
                                    weight_decay=1.2, wd_mult=s.wd_mult, name=gname))
        else:
            adam_groups.append(dict(params=params, lr=lr, betas=s.betas, weight_decay=0.0,
                                    wd_mult=s.wd_mult, name=gname))
    adam_groups.extend(extra_adamw)
    muon = NorMuon(muon_groups) if muon_groups else None
    adam = torch.optim.AdamW(adam_groups, eps=1e-10) if adam_groups else None
    return muon, adam
