#!/usr/bin/env bash
# Audit automatico STEM (scripts/stem_audit_round.py): stato dei giudizi, verifica con codice, lotti del turno
# successivo; con --final i risultati. Job solo CPU.
# Uso: sbatch slurm/stem_audit_round.sbatch NOME [--final]
#SBATCH --job-name=lvl-stem-round
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 1 )) || { echo "uso: stem_audit_round.sbatch NOME [--final]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
NAME=$1
shift
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
dir="$OUTPUT_ROOT/stem_audit/$NAME"
test -f "$dir/manifest.json" || { echo "audit mancante: $dir" >&2; exit 1; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
if srun "$RUN_PYTHON" -u -B scripts/stem_audit_round.py --dir "$dir" "$@"; then
  ledger_note ok "audit STEM, turno $NAME $*"
else
  ledger_note failed "audit STEM, turno $NAME $*"
  exit 1
fi
