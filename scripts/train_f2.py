#!/usr/bin/env python3
"""F2: SFT multimodale, proiettore e tutto il decoder addestrabili, encoder congelato.

Da eseguire SOLO in un job SLURM con GPU (slurm/train_f2.sbatch).

- Partenza: Violetto e il proiettore di F1 (`init_adapter`, la cartella final del run di F1).
- Dati: limite_vl.stream.MixedRowStream con le sorgenti della config (soluzioni del ponte, f2_solution; più avanti
  replay testuale), righe impacchettate da max_len (32k), rows_per_step righe per passo (circa 256k token).
- Gate: ogni sorgente f2_solution deve avere la catena del certificato verificata (f2_records.check_f2_shards: manifest
  f2-build/2, certificato `certified`, samples.jsonl) oppure un `uncertified_reason` esplicito nella sorgente (ablation
  di ricerca su dati non certificati, scritta nel manifest del run); senza l'una e l'altro il run non parte. Le config
  storiche (f2-smoke, f2-abl-*) puntano a shard costruiti prima del gate: si rilanciano solo da una copia con il motivo.
- Precisione: decoder in bf16 per il calcolo, come in F0 (job 29402); pesi master fp32 (train_utils.MasterWeights)
  aggiornati dall'ottimizzatore e ricopiati in bf16 dopo ogni passo. Proiettore e righe dei token visivi in fp32.
- Ottimizzatore (`optimizer`):
  - "normuon": NorMuon sulle matrici nascoste e AdamW sugli altri gruppi della Tabella 1, picchi × phase_scale
    (0.15, come l'SFT di Paradigma: matrici nascoste 0.0027), weight decay del report;
  - "adamw": AdamW su tutto il decoder a adamw_lr (1e-5), l'ablation del piano.
  Proiettore e righe dei token visivi sempre in AdamW a projector_lr (1e-4). Warmup lineare (warmup_frac) e
  decadimento lineare a zero; clipping della norma del gradiente su tutti i parametri addestrabili.
- Loss = CE media sui token con label di tutto il passo; gradient checkpointing per layer; flex_attention.
- Validazione ogni val_every passi: loss per sorgente con l'immagine vera e bianca (val_image_gain), soluzioni
  generate per alcuni campioni.
- Checkpoint in OUT/checkpoints/step-NNNNNN (pesi master fp32 del decoder, adapter, ottimizzatori, stato dei dati)
  ogni ckpt_minutes, alla fine e prima della scadenza del job; ripresa automatica con la stessa OUT. Copie dei
  pesi in OUT/snapshots/step-NNNNNN a ogni snapshot_frac del training (per la media degli ultimi, come nel report)
  e alla fine.
- OUT/manifest.json, OUT/metrics.jsonl, OUT/samples.jsonl come in F1.

    srun $RUN_PYTHON -u -B scripts/train_f2.py --config config/runs/f2-....json --out OUT
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from limite_vl.f2_records import f2_data_certification  # noqa: E402
from limite_vl.modeling import LimiteVLForConditionalGeneration, VLConfig  # noqa: E402
from limite_vl.optim.groups import build_optimizers, describe, assign_groups  # noqa: E402
from limite_vl.optim.normuon import sync_adamw_weight_decay  # noqa: E402
from limite_vl.processing import IGNORE, LimiteVLProcessor  # noqa: E402
from limite_vl.stream import MixedRowStream, make_sources  # noqa: E402
from limite_vl.train_utils import LayerCheckpointing, MasterWeights, allow_flex_recompiles  # noqa: E402
from train_f1 import (cgroup_memory_gib, check_vision_groups, config_digest, generate_samples, git_info,  # noqa: E402
                      load_config, log, seconds_left_in_job, shard_hashes, to_cuda, validate, validation_sets)

DECODER_PARAMS = 1_035_254_176  # Violetto (tests/test_normuon.py, job 29378)
ADAPTER_PARAMS = 15_077_120 + 2 * 1280
MASTER_WEIGHTS = "decoder_master.safetensors"


def lr_factor(step: int, total: int, warmup_frac: float) -> float:
    warmup = max(1, round(warmup_frac * total))
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def build_optimizer(cfg: dict, masters: MasterWeights, adapter_params: list) -> tuple[list, str]:
    """Ottimizzatori della config (lista) e descrizione dei gruppi. Ogni gruppo ha `peak_lr` per lo scheduler."""
    extra = [dict(params=adapter_params, lr=cfg["projector_lr"], betas=(0.9, 0.95), weight_decay=0.0,
                  wd_mult=0.0, name="adapter")]
    if cfg["optimizer"] == "normuon":
        muon, adam = build_optimizers(masters, phase_scale=cfg["phase_scale"], extra_adamw=extra)
        opts = [o for o in (muon, adam) if o is not None]
        text = describe(assign_groups(masters))
    elif cfg["optimizer"] == "adamw":
        adam = torch.optim.AdamW([dict(params=masters.parameters(), lr=cfg["adamw_lr"], wd_mult=0.0, name="decoder"),
                                  *extra], betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0)
        opts = [adam]
        text = f"decoder {sum(p.numel() for p in masters.parameters()) / 1e6:.2f} M in AdamW a {cfg['adamw_lr']}"
    else:
        raise ValueError(f"optimizer {cfg['optimizer']!r}: atteso normuon o adamw")
    for opt in opts:
        for group in opt.param_groups:
            group["peak_lr"] = group["lr"]
    return opts, text


def save_weights(folder: Path, model, masters: MasterWeights) -> None:
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in masters.state_dict().items()}, str(folder / MASTER_WEIGHTS))
    model.save_adapter(str(folder))


def save_checkpoint(out: Path, step: int, model, masters, opts, trainer_state: dict, keep: int) -> Path:
    ckpts = out / "checkpoints"
    final, tmp = ckpts / f"step-{step:06d}", ckpts / f"step-{step:06d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    save_weights(tmp, model, masters)
    torch.save([o.state_dict() for o in opts], tmp / "optimizer.pt")
    (tmp / "trainer_state.json").write_text(json.dumps(trainer_state, indent=1))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    pointer = ckpts / "latest.tmp"
    pointer.write_text(final.name + "\n")
    os.replace(pointer, ckpts / "latest")
    old = sorted(p for p in ckpts.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp"))
    for p in old[:-keep]:
        shutil.rmtree(p)
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stop-after-steps", type=int, help="ferma (con checkpoint) dopo N passi di questo avvio")
    ap.add_argument("--margin-min", type=float, default=15, help="minuti prima della scadenza del job per fermarsi")
    args = ap.parse_args()

    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        sys.exit("train_f2.py va eseguito in un job SLURM")
    if not torch.cuda.is_available():
        sys.exit("nessuna GPU CUDA disponibile")
    t_start = time.time()
    cfg = load_config(args.config)
    digest = config_digest(cfg)
    out = args.out
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    left = seconds_left_in_job()
    deadline = t_start + left - args.margin_min * 60 if left else None

    # ---- ripresa ----
    trainer_state, ckpt_dir = None, None
    latest = out / "checkpoints" / "latest"
    if latest.is_file():
        ckpt_dir = out / "checkpoints" / latest.read_text().strip()
        trainer_state = json.loads((ckpt_dir / "trainer_state.json").read_text())
        if trainer_state["config_digest"] != digest:
            sys.exit(f"config diversa da quella del checkpoint {ckpt_dir.name}: usare un'altra cartella --out")
        log(f"riprendo da {ckpt_dir.name} (passo {trainer_state['step']})")
    elif not Path(cfg["init_adapter"]).is_dir():
        sys.exit(f"manca il proiettore di F1: {cfg['init_adapter']}")

    torch.manual_seed(cfg["seed"])
    processor = LimiteVLProcessor.from_pretrained(cfg["decoder"])
    sources = make_sources(cfg["sources"])
    try:
        certification = f2_data_certification(cfg["sources"], [s.shards for s in sources])
    except ValueError as e:
        sys.exit(str(e))
    for c in certification:
        log(f"sorgente {c['name']}: " + ("shard certificati" if c["certified"] else
                                        f"shard NON certificati ({c['why']}); uncertified_reason: {c['uncertified_reason']}"))
    tokens_per_step = cfg["max_len"] * cfg["rows_per_step"]
    total_steps = cfg.get("total_steps") or math.ceil(float(cfg["total_tokens"]) / tokens_per_step)
    if cfg["rows_per_step"] % cfg["micro_rows"]:
        sys.exit("rows_per_step deve essere multiplo di micro_rows")
    accum = cfg["rows_per_step"] // cfg["micro_rows"]
    snapshot_every = max(1, round(cfg.get("snapshot_frac", 0.1) * total_steps))

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "run": cfg.get("run"), "config_digest": digest, "seed": cfg["seed"], "total_steps": total_steps,
        "tokens_per_step": tokens_per_step, "data": [shard_hashes(s) for s in sources],
        "data_certification": certification, "init_adapter": cfg["init_adapter"], "jobs": []}
    manifest["config"] = cfg
    manifest["jobs"].append({"job_id": job, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             "resumed_from": ckpt_dir.name if ckpt_dir else None, **git_info(out, job),
                             "data_certification": certification,
                             "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                             "seconds_left_at_start": left})
    manifest_path.write_text(json.dumps(manifest, indent=1))

    # ---- modello, pesi master e ottimizzatori ----
    model = LimiteVLForConditionalGeneration.from_pretrained(
        VLConfig(decoder_path=cfg["decoder"], vision_path=cfg["vision"], vision_precision=cfg.get("vision_precision", "tf32")),
        dtype=torch.bfloat16, adapter_dir=str(ckpt_dir) if ckpt_dir else cfg["init_adapter"])
    lm = model.language_model
    lm.config._attn_implementation = cfg["attn"]
    allow_flex_recompiles()
    model.set_trainable("f2")
    model.train()
    decoder_named = [(f"language_model.{n}", p) for n, p in lm.named_parameters() if p.requires_grad]
    adapter_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("language_model.")]
    n_dec, n_ad = sum(p.numel() for _, p in decoder_named), sum(p.numel() for p in adapter_params)
    if n_dec != DECODER_PARAMS or n_ad != ADAPTER_PARAMS:
        sys.exit(f"parametri addestrabili: decoder {n_dec} (attesi {DECODER_PARAMS}), adapter {n_ad} (attesi {ADAPTER_PARAMS})")
    if any(p.dtype != torch.float32 for p in adapter_params):
        sys.exit("proiettore e righe dei token visivi devono essere in fp32")
    masters = MasterWeights(decoder_named)
    if ckpt_dir:
        from safetensors.torch import load_file
        masters.load_state_dict({k: v.cuda() for k, v in load_file(str(ckpt_dir / MASTER_WEIGHTS)).items()})
    opts, groups_text = build_optimizer(cfg, masters, adapter_params)
    if ckpt_dir:
        for opt, state in zip(opts, torch.load(ckpt_dir / "optimizer.pt", map_location="cuda", weights_only=True)):
            opt.load_state_dict(state)
    trainable = masters.parameters() + adapter_params
    step = trainer_state["step"] if trainer_state else 0
    tokens_seen = trainer_state["tokens_seen"] if trainer_state else 0
    data_state = trainer_state["data_state"] if trainer_state else None
    log(f"modello pronto: decoder {n_dec} + adapter {n_ad} parametri, ottimizzatore {cfg['optimizer']}, {total_steps} "
        f"passi da {tokens_per_step} token ({accum} micro-batch da {cfg['micro_rows']} righe), passo iniziale {step}, "
        f"copie dei pesi ogni {snapshot_every} passi, {(time.time() - t_start):.0f} s")
    log("gruppi:\n" + groups_text)

    val_batches, prompts = validation_sets(sources, processor, cfg)
    metrics = open(out / "metrics.jsonl", "a")
    samples_log = open(out / "samples.jsonl", "a")

    def run_validation() -> None:
        t = time.time()
        losses = validate(model, val_batches)
        gain = {k: losses["white"][k] - v for k, v in losses["real"].items() if k in losses.get("white", {})}
        rows = generate_samples(model, processor, prompts, cfg.get("sample_max_new_tokens", 512))
        for r in rows:
            samples_log.write(json.dumps({"step": step, **r}, ensure_ascii=False) + "\n")
        samples_log.flush()
        metrics.write(json.dumps({"step": step, "val_loss": losses["real"], "val_loss_white": losses.get("white", {}),
                                  "val_image_gain": gain, "val_seconds": round(time.time() - t, 1)}) + "\n")
        metrics.flush()
        log(f"validazione al passo {step}: " + ", ".join(
            f"{k} {v:.4f} (bianca {losses['white'][k]:.4f}, guadagno immagine {gain[k]:.4f})" if k in gain
            else f"{k} {v:.4f} (solo testo)" for k, v in losses["real"].items()) + f" ({time.time() - t:.0f} s)")
        for r in rows[:2 * len(prompts)]:
            log(f"  [{r['source']} {r['key']}] {r['output'][:300]!r}")

    if step == 0:
        run_validation()

    stream = MixedRowStream(sources, processor, cfg["max_len"], cfg["micro_rows"], seed=cfg["seed"],
                            shuffle_buffer=cfg["shuffle_buffer"], pack_buffer=cfg["pack_buffer"],
                            val_count=cfg["val_count"], resume=data_state)
    loader = DataLoader(stream, batch_size=None, num_workers=cfg["num_workers"], pin_memory=True,
                        prefetch_factor=cfg.get("prefetch_factor", 2) if cfg["num_workers"] else None)
    batches = iter(loader)
    worker_state = dict(data_state or {})
    ckpt_ctx = LayerCheckpointing(lm) if cfg["checkpointing"] else None
    if ckpt_ctx:
        ckpt_ctx.__enter__()

    last_ckpt = time.time()
    steps_this_run = 0
    window = Counter()
    t_window = time.time()
    torch.cuda.reset_peak_memory_stats()

    def checkpoint(reason: str) -> None:
        nonlocal last_ckpt
        state = {"step": step, "tokens_seen": tokens_seen, "config_digest": digest, "data_state": worker_state,
                 "num_workers": cfg["num_workers"], "job_id": job, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        path = save_checkpoint(out, step, model, masters, opts, state, cfg.get("keep_checkpoints", 2))
        last_ckpt = time.time()
        log(f"checkpoint {path.name} ({reason})")

    stop_reason = None
    vision_checked = False
    while step < total_steps:
        t0 = time.time()
        factor = lr_factor(step, total_steps, cfg["warmup_frac"])
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = group["peak_lr"] * factor
            if isinstance(opt, torch.optim.AdamW) and cfg["optimizer"] == "normuon":
                sync_adamw_weight_decay(opt)  # λ·η·mult del report, con l'lr di questo passo
        loss_sum = torch.zeros((), device="cuda")
        count = torch.zeros((), device="cuda", dtype=torch.long)
        wait = 0.0
        for _ in range(accum):
            t_wait = time.time()
            batch = next(batches)
            wait += time.time() - t_wait
            meta = batch.pop("meta")
            worker_state[str(meta["worker"])] = meta["state"]
            window.update({f"examples/{k}": v for k, v in meta["sources"].items()})
            batch.pop("num_examples")
            window["images"] += int(batch["spatial_shapes"].shape[0]) if "spatial_shapes" in batch else 0
            window["tokens"] += batch["input_ids"].numel()
            batch = to_cuda(batch)
            if not vision_checked and "pixel_values" in batch:
                check_vision_groups(model, batch)
                vision_checked = True
            n = (batch["labels"][:, 1:] != IGNORE).sum()
            loss = model(**batch).loss
            (loss * n).backward()
            loss_sum += loss.detach() * n
            count += n
        denom = count.clamp(min=1)
        masters.grads_to_masters(denom)
        for p in adapter_params:
            if p.grad is not None:
                p.grad.div_(denom)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        masters.masters_to_model()
        model.zero_grad(set_to_none=True)
        step += 1
        steps_this_run += 1
        tokens_seen += tokens_per_step
        window["steps"] += 1
        window["wait"] += wait
        window["loss_sum"] += float(loss_sum)
        window["labelled"] += int(count)
        window["grad_norm"] += float(grad_norm)
        window["step_seconds"] += time.time() - t0

        if step % cfg["log_every"] == 0 or step == total_steps:
            dt = time.time() - t_window
            row = {"step": step, "lr_factor": factor, "loss": window["loss_sum"] / max(1, window["labelled"]),
                   "grad_norm": window["grad_norm"] / window["steps"], "tokens_per_s": window["tokens"] / dt,
                   "labelled_per_step": window["labelled"] / window["steps"],
                   "images_per_step": window["images"] / window["steps"],
                   "data_wait_s_per_step": window["wait"] / window["steps"],
                   "s_per_step": window["step_seconds"] / window["steps"],
                   "examples": {k[9:]: v for k, v in window.items() if k.startswith("examples/")},
                   "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30, "ram_gib": cgroup_memory_gib(),
                   "tokens_seen": tokens_seen, "elapsed_s": time.time() - t_start}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            log(f"passo {step}/{total_steps} loss {row['loss']:.4f} lr x{factor:.3f} |g| {row['grad_norm']:.3f} "
                f"{row['tokens_per_s']:.0f} tok/s, token con label {row['labelled_per_step']:.0f}/passo, attesa dati "
                f"{row['data_wait_s_per_step']:.2f} s/passo, GPU {row['peak_gpu_gib']:.1f} GiB, "
                f"RAM {row['ram_gib'].get('total', 0):.1f} GiB (processi {row['ram_gib'].get('anon', 0):.1f}, "
                f"cache {row['ram_gib'].get('file', 0):.1f}), esempi {row['examples']}")
            window, t_window = Counter(), time.time()

        if step % snapshot_every == 0 and step < total_steps:
            save_weights(out / "snapshots" / f"step-{step:06d}", model, masters)
            log(f"copia dei pesi al passo {step}")
        if cfg["val_every"] and step % cfg["val_every"] == 0 and step < total_steps:
            run_validation()
        if args.stop_after_steps and steps_this_run >= args.stop_after_steps:
            stop_reason = f"--stop-after-steps {args.stop_after_steps}"
        elif deadline and time.time() > deadline:
            stop_reason = "scadenza del job"
        if stop_reason:
            break
        if time.time() - last_ckpt > cfg["ckpt_minutes"] * 60:
            checkpoint("periodico")

    if ckpt_ctx:
        ckpt_ctx.__exit__(None, None, None)
    if step >= total_steps:
        run_validation()
        checkpoint("fine")
        save_weights(out / "snapshots" / f"step-{step:06d}", model, masters)
        log(f"F2 completato: {step} passi, {tokens_seen / 1e9:.3f} miliardi di token")
    else:
        checkpoint(stop_reason)
        log(f"fermato al passo {step}/{total_steps}: {stop_reason}")
    manifest = json.loads(manifest_path.read_text())
    manifest["jobs"][-1].update({"end": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "end_step": step,
                                 "stop_reason": stop_reason or "fine"})
    manifest_path.write_text(json.dumps(manifest, indent=1))
    metrics.close()
    samples_log.close()
    del batches, loader


if __name__ == "__main__":
    main()
