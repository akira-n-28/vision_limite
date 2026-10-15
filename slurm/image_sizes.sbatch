#!/usr/bin/env bash
# scripts/image_sizes.py in un job CPU. Uso: sbatch [--time=... --mem=...] slurm/image_sizes.sbatch [opzioni di image_sizes.py]
#SBATCH --job-name=lvl-img-sizes
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
if srun "$RUN_PYTHON" -u -B scripts/image_sizes.py "$@"; then
  ledger_note ok "image_sizes: $*"
else
  ledger_note failed "image_sizes: $*"
  exit 1
fi
