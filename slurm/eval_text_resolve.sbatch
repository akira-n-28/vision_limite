#!/usr/bin/env bash
# Prova CPU di scripts/eval_text_local.py --resolve-only: il valutatore riconosce un decoder esportato?
# Uso: sbatch slurm/eval_text_resolve.sbatch CHECKPOINT_DIR NOME
#SBATCH --job-name=lvl-eval-resolve
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 2 )) || { echo "uso: eval_text_resolve.sbatch CHECKPOINT_DIR NOME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
EVAL_PY="$VENVS_DIR/evals-current/bin/python"
test -x "$EVAL_PY" || { echo "manca il client del valutatore: $EVAL_PY" >&2; exit 1; }
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
srun "$EVAL_PY" -I "$PROJECT_ROOT/scripts/eval_text_local.py" --checkpoint "$(readlink -f "$1")" --name "$2" \
  --links "$OUTPUT_ROOT/eval-models" --resolve-only
