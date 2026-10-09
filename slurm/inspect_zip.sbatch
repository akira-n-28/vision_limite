#!/usr/bin/env bash
# Contenuto di un archivio zip senza estrarlo (scripts/inspect_zip.py). Nessuna GPU.
# Uso: sbatch slurm/inspect_zip.sbatch ARCHIVIO.zip [opzioni di inspect_zip.py]
#SBATCH --job-name=lvl-zip
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
(( $# > 0 )) || { echo "indicare l'archivio" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
out="$OUTPUT_ROOT/inspect_zip/$SLURM_JOB_ID"
if srun "$RUN_PYTHON" -I scripts/inspect_zip.py "$@" --out "$out"; then
  ledger_note ok "contenuto zip: $1 -> $out"
else
  ledger_note failed "contenuto zip: $1"
  exit 1
fi
