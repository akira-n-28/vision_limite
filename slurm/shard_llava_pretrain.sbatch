#!/usr/bin/env bash
# LLaVA-Pretrain (p3) -> shard tar da circa 1 GB (scripts/shard_llava_pretrain.py). Job solo CPU.
# Legge images.zip e le didascalie dallo snapshot HF verificato (job 29387); riprende dagli shard
# già completi se viene rilanciato.
#SBATCH --job-name=lvl-shard-llava
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

COMMIT=70f9d1e5e1a697fe35830875cfc7de1dd590d727
SNAP="$HF_HOME/hub/datasets--liuhaotian--LLaVA-Pretrain/snapshots/$COMMIT"
OUT="$DATA_ROOT/shards/llava-pretrain/${COMMIT:0:8}"
for f in images.zip blip_laion_cc_sbu_558k.json; do
  test -f "$SNAP/$f" || { echo "file mancante nello snapshot: $SNAP/$f" >&2; exit 1; }
done
df -hT "$DATA_ROOT" | tail -n 1
if srun "$RUN_PYTHON" -u -B scripts/shard_llava_pretrain.py --zip "$SNAP/images.zip" \
    --captions "$SNAP/blip_laion_cc_sbu_558k.json" --out "$OUT" \
    --source "liuhaotian/LLaVA-Pretrain@$COMMIT"; then
  ledger_note ok "shard llava-pretrain -> $OUT"
else
  ledger_note failed "shard llava-pretrain -> $OUT"
  exit 1
fi
