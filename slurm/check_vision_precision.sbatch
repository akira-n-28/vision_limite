#!/usr/bin/env bash
# Precisione e costo di SigLIP2 in bf16, autocast, TF32 e fp32 (scripts/check_vision_precision.py). Uso: sbatch slurm/check_vision_precision.sbatch CONFIG.json [opzioni]
#SBATCH --job-name=lvl-vis-prec
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 1 )) || { echo "indicare la config" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=$1
shift
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS=1
out="$OUTPUT_ROOT/check_vision_precision/$SLURM_JOB_ID.json"
if srun "$RUN_PYTHON" -u -B scripts/check_vision_precision.py --config "$CONFIG_PATH" --out "$out" "$@"; then
  ledger_note ok "precisione di SigLIP2 -> $out"
else
  ledger_note failed "precisione di SigLIP2"
  exit 1
fi
