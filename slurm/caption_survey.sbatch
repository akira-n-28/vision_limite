#!/usr/bin/env bash
# scripts/caption_survey.py in un job CPU. Uso: sbatch [--time=... --mem=...] slurm/caption_survey.sbatch [opzioni di caption_survey.py]
#SBATCH --job-name=lvl-cap-survey
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
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
if srun "$RUN_PYTHON" -u -B scripts/caption_survey.py "$@"; then
  ledger_note ok "caption_survey: $*"
else
  ledger_note failed "caption_survey: $*"
  exit 1
fi
