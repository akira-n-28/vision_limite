#!/usr/bin/env bash
# valori frazionari di MAVIS per denominatore, con immagini da guardare (scripts/mavis_fraction_values.py). Job solo CPU.
# Uso: sbatch slurm/mavis_fraction_values.sbatch OUT
#SBATCH --job-name=lvl-mavis-frac
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) || { echo "uso: mavis_fraction_values.sbatch OUT" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SHARDS="$DATA_ROOT/shards/mavis-geometry/9c80fff3-caption"
ls "$SHARDS"/*.tar >/dev/null || { echo "shard mancanti in $SHARDS" >&2; exit 1; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
if srun "$RUN_PYTHON" -u -B scripts/mavis_fraction_values.py --shards "$SHARDS"/*.tar --out "$1"; then
  ledger_note ok "mavis_fraction_values -> $1"
else
  ledger_note failed "mavis_fraction_values -> $1"
  exit 1
fi
