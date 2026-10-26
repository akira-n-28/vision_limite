#!/usr/bin/env bash
# Controlli di UniMER senza LLM su tutti gli shard di train densi (scripts/unimer_check.py). Prima i test della struttura
# del LaTeX (tests/test_latex_structure.py) e del confronto col disegno di matplotlib (tests/test_latex_render.py);
# se falliscono il controllo non parte. Job solo CPU.
# Uso: sbatch slurm/unimer_check.sbatch OUT
#SBATCH --job-name=lvl-unimer-check
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) || { echo "uso: unimer_check.sbatch OUT" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SHARDS="$DATA_ROOT/shards/unimer/2343ddd9-dense"
ls "$SHARDS"/unimer-train-*.tar >/dev/null || { echo "shard mancanti in $SHARDS" >&2; exit 1; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
for t in tests/test_latex_structure.py tests/test_latex_render.py; do
  if ! srun "$RUN_PYTHON" -B "$t"; then
    ledger_note failed "unimer_check: $t fallito, controllo non eseguito"
    exit 1
  fi
done
if srun "$RUN_PYTHON" -u -B scripts/unimer_check.py --shards "$SHARDS"/unimer-train-*.tar --out "$1"; then
  ledger_note ok "unimer_check -> $1"
else
  ledger_note failed "unimer_check -> $1"
  exit 1
fi
