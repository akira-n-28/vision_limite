#!/usr/bin/env bash
# Controllo CPU di una config di training (scripts/check_train_config.py). Job solo CPU.
# Uso: sbatch slurm/check_train_config.sbatch CONFIG.json
#SBATCH --job-name=lvl-cfg-check
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) || { echo "uso: check_train_config.sbatch CONFIG.json" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=$1
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
if srun "$RUN_PYTHON" -u -B scripts/check_train_config.py --config "$CONFIG_PATH"; then
  ledger_note ok "controllo della config $CONFIG_PATH"
else
  ledger_note failed "controllo della config $CONFIG_PATH"
  exit 1
fi
