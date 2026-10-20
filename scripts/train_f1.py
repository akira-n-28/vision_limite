#!/usr/bin/env python3
"""F1: allineamento del proiettore (e delle righe dei token visivi) con encoder e decoder congelati.

Da eseguire SOLO in un job SLURM con GPU (slurm/train_f1.sbatch, slurm/train_f1_smoke.sbatch).

- Dati: limite_vl.stream.MixedRowStream (sorgenti miste, righe impacchettate da max_len, ripresa).
- Passo: rows_per_step righe (batch circa 128k token con 32 righe da 4096), accumulate a micro-batch di
  micro_rows righe. Loss = CE media sui token con label di tutto il passo (non media delle medie).
- Ottimizzatore: AdamW sui parametri addestrabili di set_trainable("f1"), lr con warmup lineare e
  coseno fino a min_lr_ratio·lr, clipping della norma del gradiente.
- Validazione: loss per sorgente sui campioni esclusi dal training, con l'immagine vera e con un'immagine
  bianca della stessa dimensione (val_image_gain = bianca - vera: quanto il modello usa il contenuto
  dell'immagine, job 29423), ogni val_every passi e alla fine; didascalie generate (greedy, con il
  prefill F1) per alcuni campioni di validazione.
- Checkpoint in OUT/checkpoints/step-NNNNNN (adapter, ottimizzatore, stato dei dati e del passo) ogni
  ckpt_minutes, alla fine e prima della scadenza del job; OUT/checkpoints/latest punta all'ultimo.
  Un nuovo avvio con la stessa OUT riprende da lì (stessa config, stesso numero di worker).
- `init_adapter` (facoltativo): cartella di un adapter da cui partire, con ottimizzatore nuovo (supplemento di F1
  con la regola di risoluzione a 4x, dopo il run f1).
- OUT/manifest.json: config effettiva, commit e diff, shard con hash, seed, un elemento per job.
  OUT/metrics.jsonl e OUT/samples.jsonl: una riga per passo registrato e per campione generato.

    srun $RUN_PYTHON -u -B scripts/train_f1.py --config config/runs/f1.json --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from limite_vl.collator import PackedCollator, iter_packed_rows  # noqa: E402
from limite_vl.data import TEXT_ONLY_KINDS, build_example, generation_prompt  # noqa: E402
from limite_vl.modeling import LimiteVLForConditionalGeneration, VLConfig  # noqa: E402
from limite_vl.processing import IGNORE, LimiteVLProcessor  # noqa: E402
from limite_vl.stream import MixedRowStream, make_sources, validation_samples  # noqa: E402
from limite_vl.train_utils import LayerCheckpointing, allow_flex_recompiles  # noqa: E402

# Campi che non cambiano il risultato del training: si possono modificare fra un job e l'altro.
CONTROL_KEYS = {"run", "log_every", "val_every", "ckpt_minutes", "keep_checkpoints", "samples_per_source",
                "sample_max_new_tokens"}
F1_PARAMS = 15_077_120 + 2 * 1280  # proiettore + righe di <|vision_start|> e <|vision_end|>


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())

    def expand(v):
        if isinstance(v, str):
            out = os.path.expandvars(v)
            if "$" in out:
                raise ValueError(f"variabile d'ambiente non definita in {v!r}")
            return out
        if isinstance(v, list):
            return [expand(x) for x in v]
        if isinstance(v, dict):
            return {k: expand(x) for k, x in v.items()}
        return v
    return expand(cfg)


def config_digest(cfg: dict) -> str:
    core = {k: v for k, v in cfg.items() if k not in CONTROL_KEYS}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def lr_at(step: int, total: int, cfg: dict) -> float:
    warmup = max(1, round(cfg["warmup_frac"] * total))
    if step < warmup:
        return cfg["lr"] * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    floor = cfg.get("min_lr_ratio", 0.0)
    return cfg["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def seconds_left_in_job() -> float | None:
    """Tempo rimasto al job secondo squeue (%L: [D-]HH:MM:SS, MM:SS)."""
    job = os.environ.get("SLURM_JOB_ID")
    try:
        left = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not left or left in ("UNLIMITED", "NOT_SET", "INVALID"):
        return None
    days, _, rest = left.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days or 0) * 24 + parts[0]) * 3600 + parts[1] * 60 + parts[2]


def cgroup_memory_gib() -> dict:
    """Memoria del cgroup del job step (cgroup v2), cioè questo processo più i worker del DataLoader:
    totale, memoria dei processi (anon) e cache dei file (file, che il kernel libera prima di un OOM).
    Nel run 29425 il totale è salito in modo lineare da 8.0 a 11.7 GiB."""
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        folder = Path("/sys/fs/cgroup") / rel.lstrip("/")
        stat = dict(line.split() for line in (folder / "memory.stat").read_text().splitlines())
        return {"total": int((folder / "memory.current").read_text()) / 2**30,
                "anon": int(stat["anon"]) / 2**30, "file": int(stat["file"]) / 2**30}
    except (OSError, IndexError, ValueError, KeyError):
        return {}


def git_info(out: Path, job: str) -> dict:
    def run(*cmd):
        return subprocess.run(["git", "-C", str(ROOT), *cmd], capture_output=True, text=True).stdout
    diff = run("diff", "HEAD")
    info = {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(run("status", "--porcelain").strip())}
    if diff:
        path = out / f"diff-{job}.patch"
        path.write_text(diff)
        info["diff"] = path.name
    return info


def shard_hashes(source) -> dict:
    """Hash degli shard dal manifest della loro cartella (scritto dallo script di sharding)."""
    folder = Path(source.shards[0]).parent
    manifest = folder / "manifest.json"
    entry = {"name": source.name, "kind": source.kind, "weight": source.weight, "folder": str(folder),
             "val_shard": Path(source.val_shard).name}
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        by_name = {s["name"]: s for s in data.get("shards", [])}
        entry["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        entry["source"] = data.get("source")
        entry["shards"] = [{"name": Path(p).name, "sha256": by_name.get(Path(p).name, {}).get("sha256"),
                            "samples": by_name.get(Path(p).name, {}).get("samples")} for p in source.shards]
        missing = [s["name"] for s in entry["shards"] if s["sha256"] is None]
        if missing:
            raise RuntimeError(f"{source.name}: shard non registrati nel manifest: {missing[:5]}")
    else:
        raise FileNotFoundError(f"{source.name}: manca {manifest}")
    return entry


def to_cuda(batch: dict) -> dict:
    return {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


VAL_MODES = ("real", "white")


def validation_sets(sources, processor, cfg) -> tuple[dict, dict]:
    """Micro-batch di validazione per (sorgente, immagine vera o bianca) e campioni per la generazione.

    Le sorgenti di solo testo (data.TEXT_ONLY_KINDS) hanno solo il modo real. Una sorgente può chiedere meno
    esempi con `val_count` nella sua voce della config (es. il replay testuale, con soluzioni lunghe): sono i primi
    della parte esclusa dal training, che resta quella di cfg["val_count"]."""
    collate = PackedCollator(cfg["max_len"])
    batches, prompts = {}, {}
    specs = {spec["name"]: spec for spec in cfg["sources"]}
    for s in sources:
        count = min(cfg["val_count"], specs.get(s.name, {}).get("val_count", cfg["val_count"]))
        raw = validation_samples(s, count)
        for mode in (("real",) if s.kind in TEXT_ONLY_KINDS else VAL_MODES):
            examples = [build_example(processor, s.kind, k, f, image_mode=mode) for k, f in raw]
            rows = list(iter_packed_rows(examples, cfg["max_len"], buffer_size=len(examples)))
            batches[(s.name, mode)] = [collate(rows[i:i + cfg["micro_rows"]])
                                       for i in range(0, len(rows), cfg["micro_rows"])]
        prompts[s.name] = [(k, *generation_prompt(processor, s.kind, k, f))
                           for k, f in raw[: cfg.get("samples_per_source", 2)]]
        log(f"validazione {s.name}: {len(raw)} esempi, {len(rows)} righe per modo")
    return batches, prompts


@torch.no_grad()
def validate(model, batches: dict) -> dict:
    """{modo: {sorgente: loss media sui token con label}}."""
    out: dict = {}
    for (name, mode), items in batches.items():
        total, count = 0.0, 0
        for batch in items:
            batch = to_cuda({k: v for k, v in batch.items() if k != "num_examples"})
            n = int((batch["labels"][:, 1:] != IGNORE).sum())
            total += model(**batch).loss.item() * n
            count += n
        out.setdefault(mode, {})[name] = total / max(1, count)
    return out


@torch.no_grad()
def generate_samples(model, processor, prompts: dict, max_new_tokens: int) -> list[dict]:
    model.eval()
    rows = []
    try:
        for name, items in prompts.items():
            for key, example, reference in items:
                collate = PackedCollator(len(example))
                batch = to_cuda(collate([[example]]))
                out = model.generate(batch["input_ids"], pixel_values=batch.get("pixel_values"),
                                     pixel_attention_mask=batch.get("pixel_attention_mask"),
                                     spatial_shapes=batch.get("spatial_shapes"), do_sample=False,
                                     max_new_tokens=max_new_tokens, use_cache=True)
                text = processor.tokenizer.decode(out[0, len(example):].tolist(), skip_special_tokens=False)
                rows.append({"source": name, "key": key, "output": text, "reference": reference[:600]})
    finally:
        model.train()
    return rows


@torch.no_grad()
def check_vision_groups(model, batch: dict) -> None:
    """Sul primo micro-batch del job: encoder a gruppi di lunghezza simile contro un solo gruppo allineato
    alla più lunga, in fp32, dove devono coincidere (job 29445: 1e-5)."""
    args = (batch["pixel_values"], batch["pixel_attention_mask"], batch["spatial_shapes"])
    if model.config.vision_precision == "bf16":
        log("controllo dell'encoder a gruppi saltato: in bf16 l'output dipende dal batch (job 29445)")
        return
    precision, model.config.vision_precision = model.config.vision_precision, "fp32"  # confronto esatto
    try:
        single = model.encode_images(*args, max_group_patches=None).float()
        lengths = batch["pixel_attention_mask"].sum(dim=1)
        for cap in (32768, 4096):  # il secondo obbliga a più gruppi anche con poche immagini
            grouped = model.encode_images(*args, max_group_patches=cap).float()
            rel = ((grouped - single).norm() / single.norm().clamp(min=1e-12)).item()
            log(f"encoder a gruppi (al più {cap} patch) contro gruppo unico, in fp32: differenza relativa {rel:.2e} "
                f"su {len(lengths)} immagini (patch da {int(lengths.min())} a {int(lengths.max())})")
            if rel > 1e-3:
                raise RuntimeError(f"encoder a gruppi diverso dal gruppo unico (differenza relativa {rel:.2e})")
    finally:
        model.config.vision_precision = precision


def save_checkpoint(out: Path, step: int, model, opt, trainer_state: dict, keep: int) -> Path:
    ckpts = out / "checkpoints"
    final = ckpts / f"step-{step:06d}"
    tmp = ckpts / f"step-{step:06d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_adapter(str(tmp))
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
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
    ap.add_argument("--margin-min", type=float, default=10, help="minuti prima della scadenza del job per fermarsi")
    args = ap.parse_args()

    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        sys.exit("train_f1.py va eseguito in un job SLURM")
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
    elif cfg.get("init_adapter") and not Path(cfg["init_adapter"]).is_dir():
        sys.exit(f"init_adapter mancante: {cfg['init_adapter']}")

    torch.manual_seed(cfg["seed"])
    processor = LimiteVLProcessor.from_pretrained(cfg["decoder"])
    sources = make_sources(cfg["sources"])
    tokens_per_step = cfg["max_len"] * cfg["rows_per_step"]
    total_steps = cfg.get("total_steps") or math.ceil(float(cfg["total_tokens"]) / tokens_per_step)
    if cfg["rows_per_step"] % cfg["micro_rows"]:
        sys.exit("rows_per_step deve essere multiplo di micro_rows")
    accum = cfg["rows_per_step"] // cfg["micro_rows"]

    # ---- manifest ----
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "run": cfg.get("run"), "config_digest": digest, "seed": cfg["seed"], "total_steps": total_steps,
        "tokens_per_step": tokens_per_step, "data": [shard_hashes(s) for s in sources],
        "init_adapter": cfg.get("init_adapter"), "jobs": []}
    manifest["config"] = cfg
    manifest["jobs"].append({"job_id": job, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             "resumed_from": ckpt_dir.name if ckpt_dir else None, **git_info(out, job),
                             "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                             "seconds_left_at_start": left})
    manifest_path.write_text(json.dumps(manifest, indent=1))

    # ---- modello e ottimizzatore ----
    model = LimiteVLForConditionalGeneration.from_pretrained(
        VLConfig(decoder_path=cfg["decoder"], vision_path=cfg["vision"], vision_precision=cfg.get("vision_precision", "tf32")), dtype=torch.bfloat16,
        adapter_dir=str(ckpt_dir) if ckpt_dir else cfg.get("init_adapter"))
    lm = model.language_model
    lm.config._attn_implementation = cfg["attn"]
    allow_flex_recompiles()
    model.set_trainable("f1")
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    if n_params != F1_PARAMS:
        sys.exit(f"parametri addestrabili {n_params}, attesi {F1_PARAMS}")
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"],
                            weight_decay=cfg["weight_decay"], fused=True)
    if ckpt_dir:
        opt.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location="cuda", weights_only=True))
    step = trainer_state["step"] if trainer_state else 0
    tokens_seen = trainer_state["tokens_seen"] if trainer_state else 0
    data_state = trainer_state["data_state"] if trainer_state else None
    log(f"modello pronto: {n_params} parametri addestrabili, {total_steps} passi da {tokens_per_step} token "
        f"({accum} micro-batch da {cfg['micro_rows']} righe), passo iniziale {step}, "
        f"{(time.time() - t_start):.0f} s")

    val_batches, prompts = validation_sets(sources, processor, cfg)
    metrics = open(out / "metrics.jsonl", "a")
    samples_log = open(out / "samples.jsonl", "a")

    def run_validation() -> None:
        t = time.time()
        losses = validate(model, val_batches)
        gain = {k: losses["white"][k] - v for k, v in losses["real"].items()}
        rows = generate_samples(model, processor, prompts, cfg.get("sample_max_new_tokens", 96))
        for r in rows:
            samples_log.write(json.dumps({"step": step, **r}, ensure_ascii=False) + "\n")
        samples_log.flush()
        metrics.write(json.dumps({"step": step, "val_loss": losses["real"], "val_loss_white": losses["white"],
                                  "val_image_gain": gain, "val_seconds": round(time.time() - t, 1)}) + "\n")
        metrics.flush()
        log(f"validazione al passo {step}: " + ", ".join(
            f"{k} {v:.4f} (bianca {losses['white'][k]:.4f}, guadagno immagine {gain[k]:.4f})"
            for k, v in losses["real"].items()) + f" ({time.time() - t:.0f} s)")
        for r in rows[:2 * len(prompts)]:
            log(f"  [{r['source']} {r['key']}] {r['output'][:200]!r}")

    if step == 0:
        run_validation()

    stream = MixedRowStream(sources, processor, cfg["max_len"], cfg["micro_rows"], seed=cfg["seed"],
                            shuffle_buffer=cfg["shuffle_buffer"], pack_buffer=cfg["pack_buffer"],
                            val_count=cfg["val_count"], resume=data_state)
    loader = DataLoader(stream, batch_size=None, num_workers=cfg["num_workers"], pin_memory=True,
                        prefetch_factor=cfg.get("prefetch_factor", 4) if cfg["num_workers"] else None)
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
        path = save_checkpoint(out, step, model, opt, state, cfg.get("keep_checkpoints", 2))
        last_ckpt = time.time()
        log(f"checkpoint {path.name} ({reason})")

    stop_reason = None
    vision_checked = False
    while step < total_steps:
        t0 = time.time()
        lr = lr_at(step, total_steps, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
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
        for p in params:
            if p.grad is not None:
                p.grad.div_(denom)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
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
            row = {"step": step, "lr": lr, "loss": window["loss_sum"] / max(1, window["labelled"]),
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
            log(f"passo {step}/{total_steps} loss {row['loss']:.4f} lr {lr:.2e} |g| {row['grad_norm']:.3f} "
                f"{row['tokens_per_s']:.0f} tok/s, attesa dati {row['data_wait_s_per_step']:.2f} s/passo, "
                f"GPU {row['peak_gpu_gib']:.1f} GiB, RAM {row['ram_gib'].get('total', 0):.1f} GiB "
                f"(processi {row['ram_gib'].get('anon', 0):.1f}, cache {row['ram_gib'].get('file', 0):.1f}), "
                f"esempi {row['examples']}")
            window, t_window = Counter(), time.time()

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
        model.save_adapter(str(out / "final"))
        log(f"F1 completato: {step} passi, {tokens_seen / 1e9:.3f} miliardi di token")
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
