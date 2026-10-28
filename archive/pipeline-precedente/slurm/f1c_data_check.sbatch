#!/usr/bin/env bash
# Controllo dei dati di F1c (scripts/f1c_data_check.py). Job solo CPU.
# Uso: sbatch slurm/f1c_data_check.sbatch NOME [opzioni di f1c_data_check.py]   -> outputs/f1c/check/NOME
#SBATCH --job-name=lvl-f1c-check
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 1 )) || { echo "uso: f1c_data_check.sbatch NOME [opzioni]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
NAME=$1
shift
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
out="$OUTPUT_ROOT/f1c/check/$NAME"
if srun "$RUN_PYTHON" -u -B scripts/f1c_data_check.py --out "$out" "$@"; then
  ledger_note ok "controllo dei dati F1c ($NAME) -> $out"
else
  ledger_note failed "controllo dei dati F1c ($NAME)"
  exit 1
fi
