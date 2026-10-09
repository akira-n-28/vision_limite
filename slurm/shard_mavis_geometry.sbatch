#!/usr/bin/env bash
# MAVIS-Geometry (p3) -> shard tar da circa 1 GB (scripts/shard_mavis_geometry.py). Job solo CPU.
# Legge RuleBaseGeo.zip dallo snapshot HF verificato (job 29413); riprende dagli shard già completi
# se viene rilanciato. Variabili: VARIANT (caption, default), LIMIT (prova sui primi N campioni,
# scritta in una cartella separata).
#SBATCH --job-name=lvl-shard-mavis
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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

COMMIT=9c80fff300f4f2cb3a76f71e03428eebdd571907
SNAP="$HF_HOME/hub/datasets--CaraJ--MAVIS-Geometry/snapshots/$COMMIT"
VARIANT=${VARIANT:-caption}
OUT="$DATA_ROOT/shards/mavis-geometry/${COMMIT:0:8}-$VARIANT"
extra=()
if [[ -n ${LIMIT:-} ]]; then
  OUT="$OUT-prova$LIMIT"
  extra=(--limit "$LIMIT")
fi
test -f "$SNAP/RuleBaseGeo.zip" || { echo "file mancante nello snapshot: $SNAP/RuleBaseGeo.zip" >&2; exit 1; }
df -hT "$DATA_ROOT" | tail -n 1
if srun "$RUN_PYTHON" -u -B scripts/shard_mavis_geometry.py --zip "$SNAP/RuleBaseGeo.zip" --out "$OUT" \
    --source "CaraJ/MAVIS-Geometry@$COMMIT" --variant "$VARIANT" "${extra[@]}"; then
  ledger_note ok "shard mavis-geometry $VARIANT ${LIMIT:+prova $LIMIT }-> $OUT"
  if [[ -n ${LIMIT:-} ]]; then  # prova: 3 campioni del primo shard da guardare
    srun "$RUN_PYTHON" -B scripts/peek_shard.py "$OUT/mavis-geometry-$VARIANT-00000.tar" \
      --out "$OUTPUT_ROOT/peek_shard/$SLURM_JOB_ID" --n 3
  fi
else
  ledger_note failed "shard mavis-geometry $VARIANT ${LIMIT:+prova $LIMIT }-> $OUT"
  exit 1
fi
