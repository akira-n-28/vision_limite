#!/usr/bin/env bash
# L'output di SigLIP2 dipende dal padding? (scripts/check_vision_padding.py). Uso: sbatch slurm/check_vision_padding.sbatch CONFIG.json
#SBATCH --job-name=lvl-vis-pad
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
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS=1
out="$OUTPUT_ROOT/check_vision_padding/$SLURM_JOB_ID.json"
if srun "$RUN_PYTHON" -u -B scripts/check_vision_padding.py --config "$CONFIG_PATH" --out "$out"; then
  ledger_note ok "padding di SigLIP2 -> $out"
else
  ledger_note failed "padding di SigLIP2"
  exit 1
fi
