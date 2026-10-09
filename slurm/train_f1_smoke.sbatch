#!/usr/bin/env bash
# Prova breve di F1 (scripts/train_f1.py, config/runs/f1-smoke.json): 60 passi da 32 righe di 4096 token,
# interrotta dopo 30 passi con checkpoint e ripresa nello stesso job, per verificare loss, throughput,
# attesa dei dati, memoria e ripresa. Non è il run di F1.
#SBATCH --job-name=lvl-f1-smoke
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
export SEED=0 CONFIG_PATH=config/runs/f1-smoke.json
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS=1  # i worker del DataLoader sono processi separati
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
OUT="$OUTPUT_ROOT/f1-smoke-$SLURM_JOB_ID"
if srun "$RUN_PYTHON" -u -B scripts/train_f1.py --config "$CONFIG_PATH" --out "$OUT" --stop-after-steps 30 \
   && srun "$RUN_PYTHON" -u -B scripts/train_f1.py --config "$CONFIG_PATH" --out "$OUT"; then
  ledger_note ok "prova F1 -> $OUT"
else
  ledger_note failed "prova F1 -> $OUT"
  exit 1
fi
