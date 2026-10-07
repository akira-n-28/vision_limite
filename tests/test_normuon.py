"""Test di NorMuon (CPU, nessun download). Su Lagrange: in un job CPU con l'ambiente del progetto.

    srun $RUN_PYTHON -B tests/test_normuon.py

Il confronto con modded-nanogpt usa una copia delle sue funzioni di riferimento
(MIT, © 2024 Keller Jordan, commit 4ea6b937337a4889b8cfe3f38a93d120048d8f71,
train_gpt_medium.py), senza torch.compile.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_vl.optim import NorMuon, normuon_direction, polar_express  # noqa: E402
from limite_vl.optim.groups import assign_groups, build_optimizers, describe  # noqa: E402


# ---- riferimento upstream (modded-nanogpt), copiato senza modifiche sostanziali ----
def ref_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
    v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v_chunk.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v_chunk.mul_(final_scale.type_as(v_chunk))


def ref_cautious_update(p, v, wd, lr):
    mask = (v * p) >= 0
    p.copy_(p - (p * mask * wd * lr) - (v * lr))
# ------------------------------------------------------------------------------------

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_polar_express():
    torch.manual_seed(0)
    for shape in [(256, 128), (128, 256), (1280, 3328), (3328, 1280)]:
        g = torch.randn(shape)
        x = polar_express(g)
        u, s, vh = torch.linalg.svd(g, full_matrices=False)
        exact = u @ vh
        rel = ((x - exact).norm() / exact.norm()).item()
        sv = torch.linalg.svdvals(x)
        check(f"polar_express {shape}: vicino al fattore polare", rel < 0.25,
              f"err rel {rel:.3f}, valori singolari in [{sv.min():.2f}, {sv.max():.2f}]")


def test_matches_upstream():
    torch.manual_seed(1)
    for shape in [(384, 1280), (3328, 1280)]:
        p0 = torch.randn(shape) * 0.02
        g = torch.randn(shape)
        lr, wd, mu, beta2 = 0.0027, 1.2, 0.95, 0.9
        # nostro
        p_ours = p0.clone().requires_grad_(False)
        opt = NorMuon([p_ours], lr=lr, momentum=mu, beta2=beta2, weight_decay=wd)
        p_ours.grad = g.clone()
        opt.step()
        # upstream: momento, polar express, riduzione, update (stessa sequenza)
        B = torch.zeros_like(p0)
        B.lerp_(g, 1 - mu)
        gt = g.lerp(B, mu)
        v = polar_express(gt)
        red_dim = -1 if shape[0] >= shape[1] else -2
        buf = torch.zeros(shape[0], 1) if red_dim == -1 else torch.zeros(1, shape[1])
        v = ref_variance_reduction(v, buf, beta2, red_dim)
        p_ref = p0.clone()
        ref_cautious_update(p_ref, v, wd * lr, lr)  # upstream: eff_wd = wd·lr, poi × lr nel kernel
        # la direzione Δ deve coincidere (stesso input U, entrambi fp32)
        u = polar_expr_cached = polar_express(gt)
        r = 1 if shape[0] >= shape[1] else 0
        nu = torch.zeros(shape[0] if r == 1 else shape[1])
        d_ours = normuon_direction(u.clone(), nu, beta2, r)
        buf2 = torch.zeros(shape[0], 1) if red_dim == -1 else torch.zeros(1, shape[1])
        d_ref = ref_variance_reduction(polar_expr_cached.clone(), buf2, beta2, red_dim)
        dd = ((d_ours - d_ref).norm() / d_ref.norm()).item()
        check(f"stessa direzione Δ di modded-nanogpt {shape}", dd < 1e-5, f"diff rel {dd:.2e}")
        # l'update sui pesi coincide a meno dell'arrotondamento fp32 dei pesi (pochi ulp)
        upd = (p_ref - p0).norm()
        diff = ((p_ours - p_ref).norm() / upd).item()
        ulp = (torch.finfo(torch.float32).eps * p0.abs()).norm() / upd
        check(f"stesso update di modded-nanogpt {shape} (entro l'arrotondamento fp32)", diff < 10 * ulp.item(),
              f"diff rel {diff:.2e}, scala di 1 ulp {ulp.item():.2e}")


def test_delta_norm_and_cautious():
    torch.manual_seed(2)
    p = torch.randn(64, 32) * 0.1
    p0 = p.clone()
    g = torch.randn(64, 32)
    opt = NorMuon([p], lr=0.01, weight_decay=0.0)
    p.grad = g
    opt.step()
    delta = (p0 - p) / 0.01
    u = polar_express(g.lerp(g * 0.05, 0.95))
    check("‖Δ‖_F = ‖U‖_F", abs(delta.norm().item() - u.norm().item()) / u.norm().item() < 1e-4,
          f"{delta.norm():.4f} vs {u.norm():.4f}")
    # cautious: con lr minuscolo e λ enorme, il decay agisce solo dove Δ⊙W > 0
    p = torch.randn(64, 32)
    p_before = p.clone()
    opt = NorMuon([p], lr=1e-8, weight_decay=1e12)
    p.grad = torch.randn(64, 32)
    opt.step()
    moved = (p - p_before).abs() > 1e-4
    st = opt.state[p]
    check("cautious weight decay solo dove Δ⊙W > 0", moved.any().item() and not moved.all().item(),
          f"decadute {moved.float().mean():.2f} delle entrate; step {st['step']}")


def test_converges():
    torch.manual_seed(3)
    W_true = torch.randn(32, 64) / 8
    lin = torch.nn.Linear(64, 32, bias=False)
    opt = NorMuon(lin.parameters(), lr=0.02, weight_decay=0.0)
    x = torch.randn(4096, 64)
    y = x @ W_true.T
    first = None
    for _ in range(300):
        loss = ((lin(x) - y) ** 2).mean()
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    check("converge su un problema lineare", loss.item() < first * 0.05, f"loss {first:.4f} → {loss.item():.5f}")


def test_groups_on_toy_model():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 16, bias=False)
            self.gate_proj = torch.nn.Linear(16, 32, bias=False)
            self.attn_gate = torch.nn.Linear(4, 2, bias=False)
            self.mlp_lambda = torch.nn.Parameter(torch.ones(()))

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(100, 16)
            self.value_embeds = torch.nn.Embedding(100, 8)
            self.layers = torch.nn.ModuleList([Layer(), Layer()])
            self.norm_scale = torch.nn.Parameter(torch.ones(16))

    m = Toy()
    groups = assign_groups(m)
    print(describe(groups))
    check("gruppi: matrici nascoste in NorMuon", len(groups["hidden_matrices"]) == 4)
    check("gruppi: attn_gate in AdamW", len(groups["gates_xsa"]) == 2)
    check("gruppi: lambda scalari nel gruppo residual_lambdas", len(groups["residual_lambdas"]) == 2)
    muon, adam = build_optimizers(m, phase_scale=0.15)
    lrs = {g["name"]: g["lr"] for g in muon.param_groups + adam.param_groups}
    check("LR al 15% del picco", abs(lrs["hidden_matrices"] - 0.0027) < 1e-12, str(lrs))


if __name__ == "__main__":
    test_polar_express()
    test_matches_upstream()
    test_delta_norm_and_cautious()
    test_converges()
    test_groups_on_toy_model()
    print("\nfalliti:", FAILS or "nessuno")
    sys.exit(1 if FAILS else 0)
