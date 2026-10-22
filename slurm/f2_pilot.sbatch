#!/usr/bin/env bash
# scripts/f2_pilot.py in un job CPU: pilota dell audit F2 v2 (export del campione, turni, metriche).
# Uso: sbatch slurm/f2_pilot.sbatch export --selection SEL --mavis-shards GLOB --audit-map MAPPA --out DIR --seed S  | round --dir DIR | final --dir DIR
#SBATCH --job-name=lvl-f2-pilot
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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
if srun "$RUN_PYTHON" -u -B scripts/f2_pilot.py "$@"; then
  ledger_note ok "f2_pilot: $*"
else
  ledger_note failed "f2_pilot: $*"
  exit 1
fi
