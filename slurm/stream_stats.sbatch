#!/usr/bin/env bash
# Statistiche del flusso di training senza GPU (scripts/stream_stats.py): errori per sorgente, velocità.
# Uso: sbatch slurm/stream_stats.sbatch CONFIG.json [opzioni]
#SBATCH --job-name=lvl-stream
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# > 0 )) || { echo "indicare la config" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=$1
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
cfg=$1; shift
if srun "$RUN_PYTHON" -u -B scripts/stream_stats.py --config "$cfg" "$@"; then
  ledger_note ok "statistiche del flusso: $cfg"
else
  ledger_note failed "statistiche del flusso: $cfg"
  exit 1
fi
