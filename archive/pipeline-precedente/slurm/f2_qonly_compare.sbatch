#!/usr/bin/env bash
# Prova solo domanda contro il ponte testuale (scripts/f2_qonly_compare.py). Job solo CPU.
# Uso: sbatch slurm/f2_qonly_compare.sbatch RUN_SOLO_DOMANDA RUN_PONTE   (cartelle di outputs/f2_gen)
#SBATCH --job-name=lvl-f2-qonly
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 2 )) || { echo "uso: f2_qonly_compare.sbatch RUN_SOLO_DOMANDA RUN_PONTE" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
out="$OUTPUT_ROOT/f2_probe/$1"
if srun "$RUN_PYTHON" -u -B scripts/f2_qonly_compare.py --qonly "$OUTPUT_ROOT/f2_gen/$1" --bridge "$OUTPUT_ROOT/f2_gen/$2" --out "$out"; then
  ledger_note ok "prova solo domanda ($1 contro $2) -> $out"
else
  ledger_note failed "prova solo domanda ($1 contro $2)"
  exit 1
fi
