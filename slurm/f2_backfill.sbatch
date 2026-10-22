#!/usr/bin/env bash
# scripts/f2_backfill.py in un job CPU: selected.v2.jsonl accanto a una selezione storica, senza toccarla.
# Uso: sbatch [--time=... --mem=...] slurm/f2_backfill.sbatch --gen outputs/f2_gen/RUN --selected outputs/f2_select/SEL/selected.jsonl --out-dir outputs/f2_backfill/SEL [--audit DIR ...]
#SBATCH --job-name=lvl-f2-backfill
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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
if srun "$RUN_PYTHON" -u -B scripts/f2_backfill.py "$@"; then
  ledger_note ok "f2_backfill: $*"
else
  ledger_note failed "f2_backfill: $*"
  exit 1
fi
