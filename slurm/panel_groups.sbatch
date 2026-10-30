#!/usr/bin/env bash
# Indice del pannello (scripts/panel_index.py: una voce per immagine con pHash e sha256 dei pixel normalizzati, voci di
# solo testo) e poi gruppi e ruoli (scripts/panel_groups.py: legami confermati, candidati, irrisolti, sviluppo e test
# con seme). Job solo CPU; snapshot in sola lettura.
# Uso: sbatch slurm/panel_groups.sbatch NOME   -> data/decontam/NOME (indice) e data/panel/NOME (gruppi e ruoli)
#SBATCH --job-name=lvl-panel-groups
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) || { echo "uso: panel_groups.sbatch NOME" >&2; exit 2; }
NAME=$1
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
INDEX="$DATA_ROOT/decontam/$NAME"
PANEL="$DATA_ROOT/panel/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
if ! srun "$RUN_PYTHON" -u -B scripts/panel_index.py --out "$INDEX" --workers 4; then
  ledger_note failed "panel_index -> $INDEX"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/panel_groups.py --index "$INDEX/index.json" \
    --unimer-shards "$DATA_ROOT/shards/unimer/2343ddd9/test" --out "$PANEL" --seed 0; then
  ledger_note ok "panel_index -> $INDEX, panel_groups -> $PANEL"
else
  ledger_note failed "panel_groups -> $PANEL"
  exit 1
fi
