#!/usr/bin/env bash
# Criterio di F1: trascrizione di formule di UniMER-Test (scripts/eval_f1_formulas.py) con un adapter.
# Uso: sbatch slurm/eval_f1_formulas.sbatch CONFIG.json ADAPTER_DIR|random NOME [opzioni]
#SBATCH --job-name=lvl-f1-formulas
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 3 )) || { echo "uso: eval_f1_formulas.sbatch CONFIG.json ADAPTER_DIR|random NOME [opzioni]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=$1
ADAPTER=$2
NAME=$3
shift 3
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
TEST="$DATA_ROOT/shards/unimer/2343ddd9/test"
out="$OUTPUT_ROOT/eval_f1_formulas/$NAME-$SLURM_JOB_ID.json"
if srun "$RUN_PYTHON" -u -B scripts/eval_f1_formulas.py --config "$CONFIG_PATH" --adapter "$ADAPTER" \
    --test-dir "$TEST" --out "$out" "$@"; then
  ledger_note ok "formule F1 ($NAME) -> $out"
else
  ledger_note failed "formule F1 ($NAME)"
  exit 1
fi
