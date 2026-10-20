#!/usr/bin/env bash
# Report di F1c (scripts/f1c_report.py, solo libreria standard). Job solo CPU.
# Uso: sbatch slurm/f1c_report.sbatch OUT.json f1b=DIR control=DIR f1c=DIR
#SBATCH --job-name=lvl-f1c-report
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
(( $# >= 2 )) || { echo "uso: f1c_report.sbatch OUT.json NOME=DIR..." >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
OUT=$1
shift
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
if srun "$RUN_PYTHON" -u -B scripts/f1c_report.py --out "$OUT" "$@"; then
  ledger_note ok "report F1c -> $OUT"
else
  ledger_note failed "report F1c"
  exit 1
fi
