#!/usr/bin/env bash
# Indice del pannello (scripts/panel_index.py: una voce per immagine con pHash e sha256 dei pixel normalizzati, voci di
# solo testo) e poi gruppi e ruoli (scripts/panel_groups.py: legami confermati, candidati, irrisolti, sviluppo e test
# con seme, copertura per ruolo). Job solo CPU; snapshot in sola lettura.
# Uso: sbatch slurm/panel_groups.sbatch NOME [INDICE]
#   NOME   -> data/panel/NOME (gruppi e ruoli) e, senza INDICE, data/decontam/NOME (indice nuovo)
#   INDICE -> riusa data/decontam/INDICE/index.json già costruito, senza ricostruirlo
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
(( $# == 1 || $# == 2 )) || { echo "uso: panel_groups.sbatch NOME [INDICE]" >&2; exit 2; }
NAME=$1
INDEX_NAME=${2:-$1}
for n in "$NAME" "$INDEX_NAME"; do
  [[ $n =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $n" >&2; exit 2; }
done
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
INDEX="$DATA_ROOT/decontam/$INDEX_NAME"
PANEL="$DATA_ROOT/panel/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
if (( $# == 1 )); then
  if ! srun "$RUN_PYTHON" -u -B scripts/panel_index.py --out "$INDEX" --workers 4; then
    ledger_note failed "panel_index -> $INDEX"
    exit 1
  fi
else
  test -s "$INDEX/index.json" || { echo "indice mancante: $INDEX/index.json" >&2; ledger_note failed "indice mancante $INDEX"; exit 1; }
  echo "indice riusato: $INDEX/index.json"
fi
if srun "$RUN_PYTHON" -u -B scripts/panel_groups.py --index "$INDEX/index.json" \
    --unimer-shards "$DATA_ROOT/shards/unimer/2343ddd9/test" --out "$PANEL" --seed 0; then
  ledger_note ok "panel_index -> $INDEX, panel_groups -> $PANEL"
else
  ledger_note failed "panel_groups -> $PANEL"
  exit 1
fi
