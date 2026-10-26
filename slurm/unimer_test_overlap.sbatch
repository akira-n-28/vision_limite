#!/usr/bin/env bash
# UniMER: sovrapposizioni fra training e UniMER-Test (scripts/unimer_test_overlap.py). Job solo CPU.
# Uso: sbatch slurm/unimer_test_overlap.sbatch OUT
#SBATCH --job-name=lvl-unimer-overlap
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) || { echo "uso: unimer_test_overlap.sbatch OUT" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SHARDS="$DATA_ROOT/shards/unimer/2343ddd9-dense"
ls "$SHARDS"/unimer-train-*.tar "$SHARDS"/test/unimer-test-*.tar >/dev/null || { echo "shard mancanti in $SHARDS" >&2; exit 1; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
if srun "$RUN_PYTHON" -u -B scripts/unimer_test_overlap.py --test-shards "$SHARDS"/test/unimer-test-*.tar \
    --train-shards "$SHARDS"/unimer-train-*.tar --out "$1"; then
  ledger_note ok "unimer_test_overlap -> $1"
else
  ledger_note failed "unimer_test_overlap -> $1"
  exit 1
fi
