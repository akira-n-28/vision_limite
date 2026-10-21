#!/usr/bin/env bash
# Primi campioni di uno shard tar (scripts/peek_shard.py): JSON troncato e immagini in OUT. Job solo CPU.
# Uso: sbatch slurm/peek_shard.sbatch SHARD.tar OUT [N] [MAX_CHARS]
#SBATCH --job-name=lvl-peek-shard
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 2 )) || { echo "uso: peek_shard.sbatch SHARD.tar OUT [N] [MAX_CHARS]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SHARD=$1 OUT=$2 N=${3:-3} MAXC=${4:-1200}
test -f "$SHARD" || { echo "shard mancante: $SHARD" >&2; exit 1; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
mkdir -p "$OUT"
srun "$RUN_PYTHON" -u -B scripts/peek_shard.py "$SHARD" --out "$OUT" --n "$N" --max-chars "$MAXC"
