#!/usr/bin/env bash
# Inventario dei benchmark locali per il pannello di valutazione (scripts/panel_inventory.py): schema e conteggi dei
# file degli snapshot, contro i manifest dei download e le verifiche degli hash; indici di decontaminazione. Job solo
# CPU, in sola lettura; non estrae archivi.
# Uso: sbatch slurm/panel_inventory.sbatch NOME   -> outputs/panel_inventory/NOME
#SBATCH --job-name=lvl-panel-inv
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
(( $# == 1 )) || { echo "uso: panel_inventory.sbatch NOME" >&2; exit 2; }
NAME=$1
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
OUT="$OUTPUT_ROOT/panel_inventory/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
if srun "$RUN_PYTHON" -u -B scripts/panel_inventory.py --out "$OUT"; then
  ledger_note ok "panel_inventory -> $OUT"
else
  ledger_note failed "panel_inventory -> $OUT"
  exit 1
fi
