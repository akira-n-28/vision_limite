"""Loss a blocchi e gradient checkpointing per layer, validati da scripts/f0_checks.py (job 29378)."""
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
