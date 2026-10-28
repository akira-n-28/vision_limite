#!/usr/bin/env bash
# Diagnostica di F1 dopo la prova 29421 (loss di LLaVA che scende poco):
# 1. 60 passi solo LLaVA (config/runs/f1-diag-llava.json), stessi iperparametri della prova;
# 2. loss di validazione con immagine vera, bianca e assente (scripts/eval_f1_val.py) per il proiettore
#    casuale, quello della prova e quello solo LLaVA, sui campioni di validazione di f1-smoke.
#SBATCH --job-name=lvl-f1-diag
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=config/runs/f1-diag-llava.json
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
OUT="$OUTPUT_ROOT/f1-diag-$SLURM_JOB_ID"
SMOKE="$OUTPUT_ROOT/f1-smoke-29421/final"
test -f "$SMOKE/vl_adapter.safetensors" || { echo "manca l'adapter della prova: $SMOKE" >&2; exit 1; }
if srun "$RUN_PYTHON" -u -B scripts/train_f1.py --config "$CONFIG_PATH" --out "$OUT/llava-only" \
   && srun "$RUN_PYTHON" -u -B scripts/eval_f1_val.py --config config/runs/f1-smoke.json --adapter random \
        --adapter "smoke=$SMOKE" --adapter "llava_only=$OUT/llava-only/final" --out "$OUT/eval.json"; then
  ledger_note ok "diagnostica F1 -> $OUT"
else
  ledger_note failed "diagnostica F1 -> $OUT"
  exit 1
fi
