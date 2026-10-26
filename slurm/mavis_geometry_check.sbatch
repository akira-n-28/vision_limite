#!/usr/bin/env bash
# Coerenza geometrica delle costruzioni MAVIS (scripts/mavis_geometry_check.py) su tutti gli shard caption. Prima i test
# (tests/test_mavis_geometry.py); se falliscono il controllo non parte. Job solo CPU.
# Uso: sbatch slurm/mavis_geometry_check.sbatch OUT
#SBATCH --job-name=lvl-mavis-geom
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) || { echo "uso: mavis_geometry_check.sbatch OUT" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SHARDS="$DATA_ROOT/shards/mavis-geometry/9c80fff3-caption"
ls "$SHARDS"/*.tar >/dev/null || { echo "shard mancanti in $SHARDS" >&2; exit 1; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
if ! srun "$RUN_PYTHON" -B tests/test_mavis_geometry.py; then
  ledger_note failed "mavis_geometry_check: tests/test_mavis_geometry.py fallito, controllo non eseguito"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/mavis_geometry_check.py --shards "$SHARDS"/*.tar --out "$1"; then
  ledger_note ok "mavis_geometry_check -> $1"
else
  ledger_note failed "mavis_geometry_check -> $1"
  exit 1
fi
