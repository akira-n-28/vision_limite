#!/usr/bin/env bash
# scripts/f2_build.py in un job CPU. Uso: sbatch [--time=... --mem=...] slurm/f2_build.sbatch [opzioni di f2_build.py]
#SBATCH --job-name=lvl-f2-build
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
if srun "$RUN_PYTHON" -u -B scripts/f2_build.py "$@"; then
  ledger_note ok "f2_build: $*"
else
  ledger_note failed "f2_build: $*"
  exit 1
fi
