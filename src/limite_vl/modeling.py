"""LimiteVLForConditionalGeneration: encoder SigLIP2 NaFlex, proiettore pixel shuffle 2x2 + MLP, Limite invariato.

Il decoder è il LimiteForCausalLM originale (copia verificata, limite_code.py), senza modifiche al
suo forward: LimiteModel rifiuta inputs_embeds perché i value embeddings sono un secondo lookup per
token, quindi le immagini entrano con due forward hook sugli embedding.

- embed_tokens: le posizioni <|image_pad|> ricevono l'output del proiettore, prima della rms_norm
  iniziale (che è anche il tap x(0) del MUDD); <|vision_start|> e <|vision_end|> ricevono righe
  addestrabili (vision_token_embeds, F1) inizializzate dalle righe originali.
- value_embeds: contributo azzerato sulle posizioni <|image_pad|> (image_value_embeds="zero"),
  altrimenti tutte riceverebbero il vettore di <|image_pad|>.

Così il ciclo dei layer, il MUDD, le cache e generate() restano quelli del modello originale, e un
input solo testo dà esattamente gli stessi logit di Violetto. Le feature di un forward si consumano al
primo lookup degli embedding (il prefill, in generate()).
"""
from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import limite_code
from .processing import IMAGE_PAD_ID, MERGE, VISION_END_ID, VISION_START_ID
from .train_utils import chunked_ce

# Normalizzazione di SigLIP2 (preprocessor_config.json): rescale 1/255, media e deviazione 0.5.
PIXEL_MEAN = 0.5
PIXEL_STD = 0.5
ADAPTER_WEIGHTS = "vl_adapter.safetensors"
ADAPTER_CONFIG = "vl_config.json"


@dataclass
class VLConfig:
    decoder_path: str  # snapshot di Violetto (o checkpoint del decoder addestrato)
    vision_path: str  # snapshot di SigLIP2 NaFlex (so400m o base)
    projector_hidden: int = 2560
    image_value_embeds: str = "zero"  # "zero" | "token" (ablation: vettore di <|image_pad|>)
    vision_token_rows: bool = True


@dataclass
class VLOutput:
    loss: Tensor | None
    hidden_states: Tensor  # dopo la norm finale
    past_key_values: object | None = None


def load_vision_tower(path: str, dtype: torch.dtype, attn_implementation: str = "sdpa"):
    """Siglip2VisionModel senza testa di pooling, pesi letti dalle sole chiavi vision_model.* del checkpoint."""
    from safetensors import safe_open
    from transformers import Siglip2Config, Siglip2VisionModel

    config = Siglip2Config.from_pretrained(path, local_files_only=True).vision_config
    config.vision_use_head = False
    config._attn_implementation = attn_implementation
    model = Siglip2VisionModel(config)
    prefix, state = "vision_model.", {}
    with safe_open(str(Path(path) / "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix) and not key.startswith(prefix + "head."):
                state[key[len(prefix):]] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"encoder visivo: chiavi mancanti {missing[:5]}, inattese {unexpected[:5]}")
    return model.to(dtype).eval()


class PixelShuffleProjector(nn.Module):
    """Pixel shuffle 2x2 (1152 -> 4608) per immagine, poi MLP 4608 -> 2560 -> 1280 con GELU."""

    def __init__(self, vision_dim: int, text_dim: int, hidden: int, merge: int = MERGE):
        super().__init__()
        self.merge = merge
        self.fc1 = nn.Linear(vision_dim * merge * merge, hidden)
        self.fc2 = nn.Linear(hidden, text_dim)

    def shuffle(self, features: Tensor, spatial_shapes: Tensor) -> Tensor:
        """(N, P, C) con padding -> (token totali, C·m·m), token riga per riga, immagini in ordine."""
        m, out = self.merge, []
        for i, (rows, cols) in enumerate(spatial_shapes.tolist()):
            if rows % m or cols % m:
                raise ValueError(f"immagine {i}: griglia {rows}x{cols} non divisibile per {m}")
            x = features[i, : rows * cols].view(rows // m, m, cols // m, m, -1)
            out.append(x.permute(0, 2, 1, 3, 4).reshape((rows // m) * (cols // m), -1))
        return torch.cat(out)

    def forward(self, features: Tensor, spatial_shapes: Tensor) -> Tensor:
        x = self.shuffle(features, spatial_shapes).to(self.fc1.weight.dtype)
        return self.fc2(F.gelu(self.fc1(x)))


class LimiteVLForConditionalGeneration(nn.Module):
    def __init__(self, config: VLConfig, language_model: nn.Module, vision: nn.Module):
        super().__init__()
        if config.image_value_embeds not in ("zero", "token"):
            raise ValueError(f"image_value_embeds={config.image_value_embeds!r}")
        self.config = config
        self.language_model = language_model
        self.vision = vision
        self.projector = PixelShuffleProjector(int(vision.config.hidden_size), int(language_model.config.hidden_size),
                                               config.projector_hidden)
        embed = language_model.get_input_embeddings()
        if config.vision_token_rows:
            rows = embed.weight.detach()[[VISION_START_ID, VISION_END_ID]].float().clone()
            self.vision_token_embeds = nn.Parameter(rows)
        else:
            self.register_parameter("vision_token_embeds", None)
        self._image_features: Tensor | None = None
        self._hooks = [embed.register_forward_hook(self._embed_hook)]
        if config.image_value_embeds == "zero":
            self._hooks.append(language_model.model.value_embeds.register_forward_hook(self._value_embed_hook))

    # ---- caricamento e salvataggio ----
    @classmethod
    def from_pretrained(cls, config: VLConfig, dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
                        attn_implementation: str = "sdpa", adapter_dir: str | None = None):
        modeling = limite_code.load()
        language_model = modeling.LimiteForCausalLM.from_pretrained(
            config.decoder_path, local_files_only=True, dtype=dtype, attn_implementation=attn_implementation)
        model = cls(config, language_model, load_vision_tower(config.vision_path, dtype))
        if adapter_dir:
            model.load_adapter(adapter_dir)
        return model.to(device)

    def adapter_state_dict(self) -> dict[str, Tensor]:
        state = {f"projector.{k}": v for k, v in self.projector.state_dict().items()}
        if self.vision_token_embeds is not None:
            state["vision_token_embeds"] = self.vision_token_embeds.detach()
        return state

    def save_adapter(self, directory: str) -> None:
        """Proiettore e righe dei token visivi (F1); il decoder addestrato si salva a parte."""
        from safetensors.torch import save_file

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous().cpu() for k, v in self.adapter_state_dict().items()}, str(out / ADAPTER_WEIGHTS))
        (out / ADAPTER_CONFIG).write_text(json.dumps(asdict(self.config), indent=1) + "\n")

    def load_adapter(self, directory: str) -> None:
        from safetensors.torch import load_file

        state = load_file(str(Path(directory) / ADAPTER_WEIGHTS))
        self.projector.load_state_dict({k[len("projector."):]: v for k, v in state.items() if k.startswith("projector.")})
        if self.vision_token_embeds is not None:
            with torch.no_grad():
                self.vision_token_embeds.copy_(state["vision_token_embeds"])

    def set_trainable(self, phase: str) -> None:
        """f1: proiettore e righe dei token visivi; f2: anche tutto il decoder. Encoder sempre congelato."""
        if phase not in ("f1", "f2"):
            raise ValueError(f"fase {phase!r}")
        self.vision.requires_grad_(False)
        self.language_model.requires_grad_(phase == "f2")
        self.projector.requires_grad_(True)
        if self.vision_token_embeds is not None:
            self.vision_token_embeds.requires_grad_(True)

    # ---- hook sugli embedding ----
    def _embed_hook(self, module: nn.Module, args: tuple, output: Tensor) -> Tensor:
        input_ids = args[0]
        if self.vision_token_embeds is not None:
            rows = self.vision_token_embeds.to(output.dtype)
            output = torch.where((input_ids == VISION_START_ID).unsqueeze(-1), rows[0], output)
            output = torch.where((input_ids == VISION_END_ID).unsqueeze(-1), rows[1], output)
        features, self._image_features = self._image_features, None
        if features is not None:
            output = output.masked_scatter((input_ids == IMAGE_PAD_ID).unsqueeze(-1), features.to(output.dtype))
        return output

    def _value_embed_hook(self, module: nn.Module, args: tuple, output: Tensor) -> Tensor:
        return output.masked_fill((args[0] == IMAGE_PAD_ID).unsqueeze(-1), 0)

    # ---- forward ----
    def encode_images(self, pixel_values: Tensor, pixel_attention_mask: Tensor, spatial_shapes: Tensor) -> Tensor:
        """Patch (uint8 o già normalizzate) -> embedding per i token <|image_pad|>, (token totali, 1280)."""
        dtype = self.vision.embeddings.patch_embedding.weight.dtype
        pixels = pixel_values.float()
        if pixel_values.dtype == torch.uint8:
            pixels = (pixels / 255.0 - PIXEL_MEAN) / PIXEL_STD
        frozen = not any(p.requires_grad for p in self.vision.parameters())
        with torch.no_grad() if frozen else nullcontext():
            features = self.vision(pixel_values=pixels.to(dtype), pixel_attention_mask=pixel_attention_mask,
                                   spatial_shapes=spatial_shapes).last_hidden_state
        return self.projector(features, spatial_shapes)

    def _stage_images(self, input_ids: Tensor, pixel_values, pixel_attention_mask, spatial_shapes) -> None:
        n_pad = int((input_ids == IMAGE_PAD_ID).sum())
        if pixel_values is None:
            if n_pad:
                raise ValueError(f"{n_pad} token <|image_pad|> senza immagini")
            return
        features = self.encode_images(pixel_values, pixel_attention_mask, spatial_shapes)
        if features.shape[0] != n_pad:
            raise ValueError(f"{n_pad} token <|image_pad|> per {features.shape[0]} feature delle immagini")
        self._image_features = features

    def forward(self, input_ids: Tensor, pixel_values: Tensor | None = None,
                pixel_attention_mask: Tensor | None = None, spatial_shapes: Tensor | None = None,
                labels: Tensor | None = None, position_ids: Tensor | None = None,
                attention_mask: Tensor | None = None, past_key_values=None, use_cache: bool = False,
                **kwargs) -> VLOutput:
        """Con labels: loss a blocchi (chunked_ce) mediata sui token con label != -100. Logit con logits()."""
        kwargs.pop("num_examples", None)
        if kwargs:
            raise TypeError(f"argomenti inattesi: {sorted(kwargs)}")
        self._stage_images(input_ids, pixel_values, pixel_attention_mask, spatial_shapes)
        try:
            out = self.language_model.model(input_ids=input_ids, attention_mask=attention_mask,
                                            position_ids=position_ids, past_key_values=past_key_values,
                                            use_cache=use_cache, return_dict=True)
        finally:
            self._image_features = None
        hidden = out.last_hidden_state
        loss = chunked_ce(self.language_model, hidden, labels) if labels is not None else None
        return VLOutput(loss=loss, hidden_states=hidden, past_key_values=out.past_key_values)

    def logits(self, hidden_states: Tensor) -> Tensor:
        return self.language_model._softcapped_logits(hidden_states)

    @torch.no_grad()
    def generate(self, input_ids: Tensor, pixel_values: Tensor | None = None,
                 pixel_attention_mask: Tensor | None = None, spatial_shapes: Tensor | None = None, **kwargs):
        """generate() di Limite; le feature delle immagini entrano al prefill."""
        self._stage_images(input_ids, pixel_values, pixel_attention_mask, spatial_shapes)
        try:
            return self.language_model.generate(input_ids=input_ids, **kwargs)
        finally:
            self._image_features = None
