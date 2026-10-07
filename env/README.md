# Ambiente Python del progetto

Il venv viene creato in `/home/prignano/vision_limite/venvs/build-<jobid>` con `scripts/setup_env.sh`
(vedi la sezione "Ambiente" di `CLAUDE.md`). Qui si sceglie solo **cosa** installare.

Il `uv.lock` ufficiale di `limite-violetto` fissa torch 2.11.0+cu130 e vLLM 0.26.0 con extra
`cu13`: serve un driver NVIDIA 580 o superiore. Su `lagrange0` il driver osservato era
570.211.01 (CUDA 12.8). Il job `slurm/f0_gpu_probe.sbatch` dice quale caso vale.

## Caso A — driver ≥ 580

```bash
cp env/pyproject.cu13.template.toml env/pyproject.toml
cp third_party/limite-violetto/uv.lock env/uv.lock   # parte dalle versioni verificate da Paradigma
scripts/setup_env.sh lock
```

## Caso B — driver 570 (CUDA 12.8)

1. Controllare sulla pagina delle release GitHub di vLLM che esista una wheel CUDA 12
   (`+cu128` o `+cu129`) della versione scelta e quale torch richiede; il plugin Limite
   non rifiuta altre versioni di vLLM, ma la combinazione verificata da Paradigma è la 0.26.0.
2. `cp env/pyproject.cu12.template.toml env/pyproject.toml` e correggere versioni e URL.
3. `scripts/setup_env.sh lock`. Se una libreria CUDA 12.9 non parte sul driver 570 al primo
   test GPU, tornare a build cu128 oppure chiedere agli amministratori l'aggiornamento del
   driver. Non forzare librerie con `LD_LIBRARY_PATH`.

## Se `uv lock` viene fermato dal guardiano

La risoluzione completa usa circa 116-127 MiB di RSS, vicino al limite di 128 MiB della guida.
Se `guarded_run.sh` la ferma, generare il lock sul portatile:

```bash
cd env && uv lock --no-build && git add uv.lock && git commit -m "env: lock"
```

poi, sul cluster, dopo `git pull`: `SKIP_RESOLVE=1 scripts/setup_env.sh lock`.

## Dipendenze di training già nei template

pillow, numpy, pyarrow, safetensors, webdataset, imagehash (decontaminazione), matplotlib e
sympy (generatore procedurale), math-verify (valutazione), tqdm, hatchling (per installare
offline il plugin Limite).
