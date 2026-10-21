#!/usr/bin/env bash
# scripts/mavis_caption_survey.py su tutti gli shard caption di MAVIS-Geometry. Job solo CPU.
# Uso: sbatch slurm/mavis_caption_survey.sbatch OUT
#SBATCH --job-name=lvl-mavis-capsurvey
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) || { echo "uso: mavis_caption_survey.sbatch OUT" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SHARDS="$DATA_ROOT/shards/mavis-geometry/9c80fff3-caption"
ls "$SHARDS"/*.tar >/dev/null || { echo "shard mancanti in $SHARDS" >&2; exit 1; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
srun "$RUN_PYTHON" -u -B scripts/mavis_caption_survey.py --shards "$SHARDS"/*.tar --out "$1"
