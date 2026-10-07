"""NorMuon per Limite-VL, a GPU singola.

Implementa l'Algoritmo 1 del report tecnico di Limite 1B - Violetto (Appendice A):
  1. momento di Nesterov in forma EMA:  B ← μB + (1−μ)G ;  G̃ ← (1−μ)G + μB
  2. ortogonalizzazione Polar Express a 5 iterazioni (bf16, accumulo fp32)
  3. riduzione di varianza NorMuon (statistica per neurone lungo la dimensione più lunga)
  4. cautious weight decay stretto, applicato come λ·η²:  W ← W − λη²(C⊙W) − ηΔ,  C = 1[Δ⊙W > 0]

Coefficienti di Polar Express: modded-nanogpt (MIT, © 2024 Keller Jordan), commit
4ea6b937337a4889b8cfe3f38a93d120048d8f71, `train_gpt_medium.py`, calcolati per
num_iters=5, safety_factor=2e-2, cushion=2 — gli stessi parametri dichiarati nel report.

Differenze note rispetto a modded-nanogpt (qui si segue il report):
  - maschera cautious stretta (> 0) invece di >= 0
  - nessun moltiplicatore di LR legato alla forma (upstream: sqrt(max(1, righe/colonne)));
    attivabile con `shape_lr_mult=True` per le ablation
  - nessuna logica distribuita: una GPU, parametri interi

I parametri in input devono essere le copie master in fp32. Matrici fuse (es. QKV in un
unico peso) si ortogonalizzano per blocchi indicando `split=[righe_q, righe_k, righe_v]`.
"""
from __future__ import annotations

from typing import Iterable

import torch

POLAR_EXPRESS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def polar_express(g: torch.Tensor) -> torch.Tensor:
    """Approssimazione del fattore polare di una matrice 2D (Algoritmo 1, passi 3-12)."""
    if g.ndim != 2:
        raise ValueError("polar_express accetta solo matrici 2D")
    x = g.to(torch.bfloat16)
    x = x / (1.02 * x.norm() + 1e-6)  # ‖X‖₂ ≤ ‖X‖_F, quindi norma spettrale < 1
    m, n = x.shape
    for a, b, c in POLAR_EXPRESS_COEFFS:
        if m > n:
            A = x.mT @ x
            x = a * x + x @ (b * A + c * (A @ A))
        else:
            A = x @ x.mT
            x = a * x + (b * A + c * (A @ A)) @ x
    return x.float()


def normuon_direction(u: torch.Tensor, nu: torch.Tensor, beta2: float, r: int) -> torch.Tensor:
    """Algoritmo 1, passi 13-19: aggiorna `nu` in place e restituisce Δ con ‖Δ‖_F = ‖U‖_F."""
    n_r = u.shape[r]
    s = (u * u).mean(dim=r)                              # s ∈ R^m oppure R^n
    rho = (n_r * s.sum()).sqrt()                         # = ‖U‖_F
    nu.mul_(beta2).add_(s, alpha=1 - beta2)
    sigma = nu.clamp_min(1e-10).rsqrt()
    rho_p = (n_r * (s * sigma.square()).sum()).sqrt()    # = ‖U⊙σ‖_F
    sig_b = sigma.unsqueeze(1) if r == 1 else sigma.unsqueeze(0)
    return u * sig_b * (rho / rho_p.clamp_min(1e-10))


class NorMuon(torch.optim.Optimizer):
    """NorMuon dell'Appendice A del report di Limite.

    Opzioni per gruppo:
      lr            η (picco già scalato per la fase, es. 0.15 × 0.018 in F2)
      momentum      μ (0.95 nel report, con warm-up da 0.85 solo nel pretraining)
      beta2         β₂ della statistica di secondo ordine (0.9)
      weight_decay  λ (1.2), applicato come λ·η² e moltiplicato per `wd_mult`
      wd_mult       moltiplicatore per gruppo (1 per le matrici nascoste)
      reduce        "auto" (dimensione più lunga), "cols" o "rows"; "cols" per W_v come nel report
      split         None oppure lista di numeri di righe per ortogonalizzare a blocchi una matrice fusa
      shape_lr_mult False (report); True = moltiplicatore di modded-nanogpt
    """

    def __init__(self, params: Iterable, lr: float = 0.0027, momentum: float = 0.95, beta2: float = 0.9,
                 weight_decay: float = 1.2, wd_mult: float = 1.0, reduce: str = "auto",
                 split: list[int] | None = None, shape_lr_mult: bool = False):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay, wd_mult=wd_mult,
                        reduce=reduce, split=split, shape_lr_mult=shape_lr_mult)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"NorMuon gestisce solo matrici 2D, trovato shape {tuple(p.shape)}")
                if p.dtype != torch.float32:
                    raise ValueError("NorMuon richiede pesi master in fp32")
                if group["split"] is not None and sum(group["split"]) != p.shape[0]:
                    raise ValueError(f"split {group['split']} non somma alle righe {p.shape[0]}")

    @staticmethod
    def _reduce_dim(shape: torch.Size, reduce: str) -> int:
        m, n = shape
        if reduce == "cols":
            return 1  # media sulle colonne → una statistica per riga
        if reduce == "rows":
            return 0
        return 1 if m >= n else 0  # "auto": r = colonne se m ≥ n, altrimenti righe

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, beta2 = group["lr"], group["momentum"], group["beta2"]
            lam = group["weight_decay"] * group["wd_mult"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                r = self._reduce_dim(p.shape, group["reduce"])
                if not state:
                    state["B"] = torch.zeros_like(p, dtype=torch.float32)
                    state["nu"] = torch.zeros(p.shape[0] if r == 1 else p.shape[1], device=p.device,
                                              dtype=torch.float32)
                    state["step"] = 0
                state["step"] += 1

                # 1. Nesterov (EMA)
                B = state["B"]
                B.lerp_(g, 1 - mu)
                g_tilde = g.lerp(B, mu)

                # 2. Polar Express, per blocchi se la matrice è fusa
                if group["split"]:
                    u = torch.cat([polar_express(blk) for blk in torch.split(g_tilde, group["split"], dim=0)], dim=0)
                else:
                    u = polar_express(g_tilde)

                # 3. riduzione di varianza NorMuon
                delta = normuon_direction(u, state["nu"], beta2, r)

                # 4. cautious weight decay (stretto) e aggiornamento
                eta = lr * (max(1.0, p.shape[0] / p.shape[1]) ** 0.5 if group["shape_lr_mult"] else 1.0)
                if lam:
                    mask = (delta * p) > 0
                    p.sub_(p * mask, alpha=lam * eta * eta)
                p.sub_(delta, alpha=eta)
        return loss


def sync_adamw_weight_decay(optimizer: torch.optim.Optimizer, base_lambda: float = 0.005) -> None:
    """Weight decay di AdamW come nel report: λ·η²·mult (torch applica lr·wd, quindi wd = λ·η·mult).

    Da chiamare prima di ogni `optimizer.step()` quando il learning rate cambia con lo scheduler.
    Ogni gruppo deve avere la chiave `wd_mult` (0 = nessun decay).
    """
    for group in optimizer.param_groups:
        group["weight_decay"] = base_lambda * group["lr"] * group.get("wd_mult", 0.0)
