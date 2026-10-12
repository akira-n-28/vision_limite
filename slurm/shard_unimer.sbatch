#!/usr/bin/env bash
# UniMER (p3) -> shard tar per F1 (scripts/shard_unimer.py). Job solo CPU.
# Legge UniMER-1M.zip e UniMER-Test.zip dallo snapshot HF verificato (job 29429); riprende dagli shard già
# completi se viene rilanciato. Variabile LIMIT: prova sui primi N campioni del train, in una cartella a parte.
#SBATCH --job-name=lvl-shard-unimer
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1  # un processo per CPU nel pool

COMMIT=2343ddd963290469da36ca83e3a56c66e068add9
SNAP="$HF_HOME/hub/datasets--wanderkid--UniMER_Dataset/snapshots/$COMMIT"
OUT="$DATA_ROOT/shards/unimer/${COMMIT:0:8}"
extra=()
if [[ -n ${LIMIT:-} ]]; then
  OUT="$OUT-prova$LIMIT"
  extra=(--limit "$LIMIT")
fi
for f in UniMER-1M.zip UniMER-Test.zip; do
  test -f "$SNAP/$f" || { echo "file mancante nello snapshot: $SNAP/$f" >&2; exit 1; }
done
df -hT "$DATA_ROOT" | tail -n 1
if srun "$RUN_PYTHON" -u -B scripts/shard_unimer.py --zip "$SNAP/UniMER-1M.zip" --test-zip "$SNAP/UniMER-Test.zip" \
    --out "$OUT" --source "wanderkid/UniMER_Dataset@$COMMIT" "${extra[@]}"; then
  ledger_note ok "shard unimer ${LIMIT:+prova $LIMIT }-> $OUT"
  if [[ -n ${LIMIT:-} ]]; then  # prova: 3 campioni da guardare
    srun "$RUN_PYTHON" -B scripts/peek_shard.py "$OUT/unimer-train-00000.tar" --out "$OUTPUT_ROOT/peek_shard/$SLURM_JOB_ID" --n 3
  fi
else
  ledger_note failed "shard unimer ${LIMIT:+prova $LIMIT }-> $OUT"
  exit 1
fi
