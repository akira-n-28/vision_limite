#!/usr/bin/env bash
# Duplicati fra validazione e training per le sorgenti di F1-clean (scripts/split_duplicates.py). Job solo CPU.
# Uso: sbatch slurm/split_duplicates.sbatch OUT
#SBATCH --job-name=lvl-split-dups
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
(( $# == 1 )) || { echo "uso: split_duplicates.sbatch OUT" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
S="$DATA_ROOT/shards"
if srun "$RUN_PYTHON" -u -B scripts/split_duplicates.py \
    --source "mavis:mavis=$S/mavis-geometry/9c80fff3-caption/*.tar" \
    --source "unimer:unimer=$S/unimer/2343ddd9-dense/unimer-train-*.tar" \
    --source "pixmo:pixmo=$S/pixmo-cap/3e42775e/*.tar" \
    --source "synth:synth=$S/synth/v3/train/*.tar" \
    --out "$1"; then
  ledger_note ok "split_duplicates -> $1"
else
  ledger_note failed "split_duplicates -> $1"
  exit 1
fi
