#!/usr/bin/env bash
# Sovrapposizioni fra i benchmark già indicizzati (scripts/panel_overlaps.py): pHash identici o vicini, testi
# identici e ID condivisi fra benchmark e split, per raggruppare prima di assegnare i ruoli. Job solo
# CPU, in sola lettura.
# Uso: sbatch slurm/panel_overlaps.sbatch NOME   -> outputs/panel_overlaps/NOME
#SBATCH --job-name=lvl-panel-ovl
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
(( $# == 1 )) || { echo "uso: panel_overlaps.sbatch NOME" >&2; exit 2; }
NAME=$1
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
OUT="$OUTPUT_ROOT/panel_overlaps/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
if srun "$RUN_PYTHON" -u -B scripts/panel_overlaps.py --out "$OUT"; then
  ledger_note ok "panel_overlaps -> $OUT"
else
  ledger_note failed "panel_overlaps -> $OUT"
  exit 1
fi
