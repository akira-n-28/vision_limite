"""Loss a blocchi e gradient checkpointing per layer, validati da scripts/f0_checks.py (job 29378); pesi master
fp32 per il decoder addestrabile di F2."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# flex_attention: transformers compila create_block_mask e dynamo ricompila per ogni tipo di maschera
# (righe impacchettate, layer globali e a finestra, prefill e decodifica della generazione). Con il limite
# di default (8) la validazione a tre sorgenti lo supera e le maschere tornano in eager: 2 mila token/s
# invece di 16 mila (job 29441). Le varianti sono poche e fisse, quindi basta un limite più alto.
FLEX_RECOMPILE_LIMIT = 64


def allow_flex_recompiles(limit: int = FLEX_RECOMPILE_LIMIT) -> None:
    import torch._dynamo

    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, limit)


def decoder(model):
    """LimiteModel (con .layers) di un LimiteForCausalLM."""
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        raise RuntimeError("non trovo model.model.layers: controllare la struttura in modeling_limite.py")
    return inner


def chunked_ce(model, hidden: torch.Tensor, labels: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """CE con soft-capping calcolata a blocchi di token; hidden = output dopo la norm finale.

    `model` è il LimiteForCausalLM (serve `_softcapped_logits`). Media sui token con label != -100.
    """
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


class MasterWeights:
    """Pesi master fp32 per i parametri bf16 del decoder (F2): il modello calcola in bf16 come in F0 (job 29402),
    l'ottimizzatore aggiorna le copie fp32 (NorMuon le richiede; in bf16 un passo da 1e-5 su un peso da 0.02 si
    perderebbe nell'arrotondamento) e i pesi bf16 si ricopiano dopo ogni passo.

    I parametri che Limite tiene già in fp32 anche con il modello in bf16 (`_fp32_parameter`: scale, lambda
    residue, gate, pesi del MUDD; job 29664) sono master di se stessi: niente copia né ricopiatura.

    `named_parameters()` restituisce le copie fp32 con i nomi del modello, per optim.groups.assign_groups.
    """

    def __init__(self, named_params):
        self.pairs = []
        for name, p in named_params:
            if p.dtype == torch.bfloat16:
                self.pairs.append((name, p, torch.nn.Parameter(p.detach().float().clone())))
            elif p.dtype == torch.float32:
                self.pairs.append((name, p, p))
            else:
                raise ValueError(f"{name}: atteso bf16 o fp32, trovato {p.dtype}")

    def named_parameters(self):
        return [(name, m) for name, _, m in self.pairs]

    def parameters(self):
        return [m for _, _, m in self.pairs]

    @torch.no_grad()
    def grads_to_masters(self, scale: float | torch.Tensor = 1.0) -> None:
        """Gradienti bf16 del modello -> fp32 sulle copie master, divisi per `scale`; libera quelli del modello."""
        for _, p, m in self.pairs:
            if p.grad is None:
                m.grad = None
            elif m is p:
                p.grad.div_(scale)
            else:
                m.grad = p.grad.float().div_(scale)
                p.grad = None

    @torch.no_grad()
    def masters_to_model(self) -> None:
        for _, p, m in self.pairs:
            if m is not p:
                p.copy_(m)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: m.detach() for name, _, m in self.pairs}

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        names = {name for name, _, _ in self.pairs}
        if set(state) != names:
            raise ValueError(f"pesi master: mancano {sorted(names - set(state))[:5]}, in più {sorted(set(state) - names)[:5]}")
        for name, p, m in self.pairs:
            m.copy_(state[name])
            if m is not p:
                p.copy_(m)


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
