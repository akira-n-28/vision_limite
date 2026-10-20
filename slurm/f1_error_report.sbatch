#!/usr/bin/env bash
# Report dell'analisi degli errori di F1 (scripts/f1_error_report.py, solo libreria standard). Job solo CPU.
# Uso: sbatch slurm/f1_error_report.sbatch formulas|captions OUT.json NOME=FILE_O_CARTELLA...
#SBATCH --job-name=lvl-f1-report
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 3 )) || { echo "uso: f1_error_report.sbatch formulas|captions OUT.json NOME=PERCORSO..." >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
COMMAND=$1
OUT=$2
shift 2
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
if srun "$RUN_PYTHON" -u -B scripts/f1_error_report.py "$COMMAND" --out "$OUT" "$@"; then
  ledger_note ok "report degli errori di F1 ($COMMAND) -> $OUT"
else
  ledger_note failed "report degli errori di F1 ($COMMAND)"
  exit 1
fi
