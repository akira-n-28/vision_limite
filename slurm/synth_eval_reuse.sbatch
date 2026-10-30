#!/usr/bin/env bash
# Stati condivisi fra test e coppie dei sintetici v3 (consultati nelle valutazioni F1c) e v4 con le stesse chiavi
# (scripts/synth_eval_reuse.py): solo specifiche, nessun disegno. Job solo
# CPU, in sola lettura.
# Uso: sbatch slurm/synth_eval_reuse.sbatch NOME   -> outputs/synth_eval_reuse/NOME
#SBATCH --job-name=lvl-synth-reuse
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
(( $# == 1 )) || { echo "uso: synth_eval_reuse.sbatch NOME" >&2; exit 2; }
NAME=$1
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
OUT="$OUTPUT_ROOT/synth_eval_reuse/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
if srun "$RUN_PYTHON" -u -B scripts/synth_eval_reuse.py --out "$OUT"; then
  ledger_note ok "synth_eval_reuse -> $OUT"
else
  ledger_note failed "synth_eval_reuse -> $OUT"
  exit 1
fi
