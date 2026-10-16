#!/usr/bin/env python3
"""Decoder di un run F2 -> checkpoint Hugging Face nel formato dello snapshot di Violetto (per vLLM e la valutazione
testuale).

train_f2 salva i pesi master fp32 del decoder (decoder_master.safetensors) con i nomi del modello VL in memoria:
prefisso `language_model.` e proiezioni fuse `qkv_proj` e `gate_up_proj` (registration.py di Limite le fonde al
caricamento). Qui si torna al formato dello snapshot:
- nomi senza prefisso, `q_proj`/`k_proj`/`v_proj` e `gate_proj`/`up_proj` separati (stesse dimensioni di
  registration.py: Q = teste x head_dim, K e V = teste KV x head_dim; gate e up a metà);
- tipo di ogni tensore uguale a quello dello snapshot (bf16 per le matrici, fp32 per scale, lambda, gate e MUDD);
- `lm_head` legato all'embedding (tie_word_embeddings), quindi non salvato, come nello snapshot.
Gli altri file (config, generation_config, tokenizer, chat template, codice) si copiano senza modifiche.

Controlli: stesso insieme di nomi, forme e tipi dello snapshot. Con --roundtrip non serve un run: i pesi di Violetto
caricati come in train_f2 (bf16, proiezioni fuse) e passati per le copie fp32 devono tornare bit per bit uguali allo
snapshot.

Da eseguire in un job (slurm/export_decoder.sbatch): carica pesi.

    srun $RUN_PYTHON -u -B scripts/export_decoder.py --decoder SNAPSHOT (--weights decoder_master.safetensors | --roundtrip) --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PREFIX = "language_model."
WEIGHTS = "model.safetensors"


def split_fused(state: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    """Nomi del modello in memoria (senza prefisso) -> nomi dello snapshot, proiezioni fuse separate."""
    q = int(config["num_attention_heads"]) * int(config["head_dim"])
    kv = int(config["num_key_value_heads"]) * int(config["head_dim"])
    out = {}
    for name, t in state.items():
        if name.endswith("self_attn.qkv_proj.weight"):
            if t.shape[0] != q + 2 * kv:
                raise ValueError(f"{name}: {t.shape[0]} righe, attese {q + 2 * kv}")
            base = name.removesuffix("qkv_proj.weight")
            for part, chunk in zip(("q_proj", "k_proj", "v_proj"), t.split((q, kv, kv), dim=0)):
                out[f"{base}{part}.weight"] = chunk
        elif name.endswith("mlp.gate_up_proj.weight"):
            if t.shape[0] % 2:
                raise ValueError(f"{name}: numero di righe dispari {t.shape[0]}")
            base = name.removesuffix("gate_up_proj.weight")
            gate, up = t.chunk(2, dim=0)
            out[f"{base}gate_proj.weight"], out[f"{base}up_proj.weight"] = gate, up
        else:
            out[name] = t
    return out


def snapshot_layout(path: Path) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    with safe_open(str(path), framework="pt") as f:
        layout = {}
        for key in f.keys():
            s = f.get_slice(key)
            layout[key] = (s.get_dtype(), tuple(s.get_shape()))
    return layout


DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def master_state(weights: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(weights), framework="pt") as f:
        names = list(f.keys())
        if not all(n.startswith(PREFIX) for n in names):
            raise ValueError(f"{weights}: attesi nomi con il prefisso {PREFIX!r}")
        return {n.removeprefix(PREFIX): f.get_tensor(n) for n in names}


def roundtrip_state(decoder: Path) -> dict[str, torch.Tensor]:
    """Violetto caricato come in train_f2 (bf16, proiezioni fuse) e passato per le copie fp32 di MasterWeights."""
    from limite_vl import limite_code

    modeling = limite_code.load()
    model = modeling.LimiteForCausalLM.from_pretrained(str(decoder), local_files_only=True, dtype=torch.bfloat16)
    return {n: p.detach().float().clone() for n, p in model.named_parameters()}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoder", required=True, help="snapshot di Violetto (formato e file di contorno)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", help="decoder_master.safetensors di un run F2 (checkpoints/ o snapshots/)")
    src.add_argument("--roundtrip", action="store_true", help="esporta Violetto stesso e confronta con lo snapshot")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("export_decoder.py va eseguito in un job SLURM")

    decoder, out = Path(args.decoder), Path(args.out)
    if out.exists():
        sys.exit(f"{out} esiste già: scegliere un'altra cartella")
    config = json.loads((decoder / "config.json").read_text())
    if not config.get("tie_word_embeddings"):
        sys.exit("lo snapshot non lega lm_head all'embedding: l'esportazione non lo prevede")
    layout = snapshot_layout(decoder / WEIGHTS)

    state = roundtrip_state(decoder) if args.roundtrip else master_state(Path(args.weights))
    state.pop("lm_head.weight", None)  # legato a model.embed_tokens.weight
    state = split_fused(state, config)

    missing, extra = sorted(set(layout) - set(state)), sorted(set(state) - set(layout))
    if missing or extra:
        sys.exit(f"nomi diversi dallo snapshot: mancano {missing[:5]} ({len(missing)}), in più {extra[:5]} ({len(extra)})")
    tensors = {}
    for name, (dtype_name, shape) in layout.items():
        t = state[name]
        if tuple(t.shape) != shape:
            sys.exit(f"{name}: forma {tuple(t.shape)}, nello snapshot {shape}")
        tensors[name] = t.to(DTYPES[dtype_name]).contiguous()
    print(f"{len(tensors)} tensori, {sum(t.numel() for t in tensors.values()) / 1e9:.3f} miliardi di parametri", flush=True)

    if args.roundtrip:
        different = []
        with safe_open(str(decoder / WEIGHTS), framework="pt") as f:
            for name, t in tensors.items():
                if not torch.equal(t, f.get_tensor(name)):
                    different.append(name)
        print(f"roundtrip: {len(tensors) - len(different)} tensori uguali bit per bit allo snapshot, diversi {len(different)} "
              f"{different[:5]}", flush=True)
        if different:
            sys.exit(1)

    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for item in sorted(decoder.iterdir()):
        if item.name != WEIGHTS and item.is_file() and not item.name.startswith("."):
            shutil.copyfile(item, tmp / item.name)  # copia del file vero (lo snapshot ha link ai blob)
    save_file(tensors, str(tmp / WEIGHTS), metadata={"format": "pt"})
    provenance = {"decoder": str(decoder.resolve()), "weights": str(Path(args.weights).resolve()) if args.weights else "roundtrip",
                  "job_id": os.environ.get("SLURM_JOB_ID"), "sha256": {WEIGHTS: sha256_file(tmp / WEIGHTS)}}
    if args.weights:
        provenance["weights_sha256"] = sha256_file(Path(args.weights))
    (tmp / "export.json").write_text(json.dumps(provenance, indent=1) + "\n")
    tmp.rename(out)
    print(f"checkpoint in {out} (model.safetensors sha256 {provenance['sha256'][WEIGHTS][:12]})", flush=True)


if __name__ == "__main__":
    main()
