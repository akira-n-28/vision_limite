#!/usr/bin/env bash
# Confronto delle tre varianti d'immagine di MAVIS-Geometry (scripts/mavis_variants.py). Nessuna GPU.
# Uso: sbatch slurm/mavis_variants.sbatch FILE.parquet [RIGA ...]
#SBATCH --job-name=lvl-mavis-var
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# > 0 )) || { echo "indicare il file parquet" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
parquet=$1; shift
rows=("${@:-0 1 2}")
out="$OUTPUT_ROOT/mavis_variants/$SLURM_JOB_ID"
# shellcheck disable=SC2068
if srun "$RUN_PYTHON" -B scripts/mavis_variants.py "$parquet" --out "$out" --rows ${rows[@]}; then
  ledger_note ok "varianti MAVIS-Geometry: $out"
else
  ledger_note failed "varianti MAVIS-Geometry"
  exit 1
fi
