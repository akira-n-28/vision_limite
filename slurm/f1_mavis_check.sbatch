#!/usr/bin/env bash
# Dati MAVIS di F1-clean su tutti gli shard caption (scripts/f1_mavis_check.py): verità, descrizione canonica, domande,
# andata e ritorno. Prima i test (tests/test_mavis_gen.py e tests/test_f1_mavis.py); se falliscono non parte.
# Job solo CPU. Uso: sbatch slurm/f1_mavis_check.sbatch OUT
#SBATCH --job-name=lvl-f1-mavis
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) || { echo "uso: f1_mavis_check.sbatch OUT" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SHARDS="$DATA_ROOT/shards/mavis-geometry/9c80fff3-caption"
ls "$SHARDS"/*.tar >/dev/null || { echo "shard mancanti in $SHARDS" >&2; exit 1; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
for t in tests/test_mavis_gen.py tests/test_f1_mavis.py; do
  if ! srun "$RUN_PYTHON" -B "$t"; then
    ledger_note failed "f1_mavis_check: $t fallito, costruzione non eseguita"
    exit 1
  fi
done
if srun "$RUN_PYTHON" -u -B scripts/f1_mavis_check.py --shards "$SHARDS"/*.tar --out "$1"; then
  ledger_note ok "f1_mavis_check -> $1"
else
  ledger_note failed "f1_mavis_check -> $1"
  exit 1
fi
