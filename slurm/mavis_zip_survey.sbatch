#!/usr/bin/env bash
# Analisi di RuleBaseGeo.zip di MAVIS-Geometry senza estrarlo (scripts/mavis_zip_survey.py). Nessuna GPU.
# Uso: sbatch slurm/mavis_zip_survey.sbatch
#SBATCH --job-name=lvl-mavis-zip
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
snap="$HF_HOME/hub/datasets--CaraJ--MAVIS-Geometry/snapshots/9c80fff300f4f2cb3a76f71e03428eebdd571907"
out="$OUTPUT_ROOT/mavis_zip_survey/$SLURM_JOB_ID"
if srun "$RUN_PYTHON" -u -B scripts/mavis_zip_survey.py "$snap/RuleBaseGeo.zip" \
    --parquet "$snap/visualization_small.parquet" --out "$out"; then
  ledger_note ok "analisi RuleBaseGeo.zip: $out"
else
  ledger_note failed "analisi RuleBaseGeo.zip"
  exit 1
fi
