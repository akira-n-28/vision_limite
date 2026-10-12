#!/usr/bin/env bash
# F2, ponte testuale: Violetto risolve i problemi di MMFineReason dalla descrizione (scripts/f2_generate.py).
# Uso: sbatch [--time=... --mem=...] slurm/f2_generate.sbatch NOME_RUN (--shards GLOB | --parquet FILE...) [opzioni]
# Uscita in outputs/f2_gen/NOME_RUN: rilanciando con lo stesso nome riprende dalle parti già scritte.
#SBATCH --job-name=lvl-f2-gen
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 2 )) || { echo "uso: f2_generate.sbatch NOME_RUN (--shards GLOB | --parquet FILE...) [opzioni]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
RUN=$1
shift
[[ $RUN =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome del run non valido: $RUN" >&2; exit 2; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Come slurm/vllm_smoke.sbatch: plugin di Limite, ninja del venv per i JIT, cache nel progetto.
export VLLM_PLUGINS=limite
export PATH="$(dirname "$RUN_PYTHON"):$PATH"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
VIOLETTO="$HF_HOME/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
test -f "$VIOLETTO/model.safetensors" || { echo "snapshot di Violetto mancante: $VIOLETTO" >&2; exit 1; }
OUT="$OUTPUT_ROOT/f2_gen/$RUN"
if srun "$RUN_PYTHON" -u -B scripts/f2_generate.py --model "$VIOLETTO" --out "$OUT" "$@"; then
  ledger_note ok "F2 generazione $RUN -> $OUT"
else
  ledger_note failed "F2 generazione $RUN -> $OUT"
  exit 1
fi
