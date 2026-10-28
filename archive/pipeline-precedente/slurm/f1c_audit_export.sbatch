#!/usr/bin/env bash
# Materiale per il controllo delle risposte di riferimento e dei dati di F1c (scripts/f1c_audit_export.py). Job solo CPU.
#SBATCH --job-name=lvl-f1c-audit
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=$PROJECT_ROOT/config/runs/f1b.json
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
out="$OUTPUT_ROOT/f1c/audit"
if srun "$RUN_PYTHON" -u -B scripts/f1c_audit_export.py --config "$CONFIG_PATH" --out "$out" "$@"; then
  ledger_note ok "materiale per l audit di F1c -> $out"
else
  ledger_note failed "materiale per l audit di F1c"
  exit 1
fi
