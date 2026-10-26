#!/usr/bin/env bash
# PixMo-Cap: descrizioni contro trascrizioni degli annotatori (scripts/pixmo_transcript_check.py). Prima i test
# (tests/test_transcripts.py); se falliscono il controllo non parte. Legge solo due colonne di testo dei parquet.
# Job solo CPU. Uso: sbatch slurm/pixmo_transcript_check.sbatch OUT
#SBATCH --job-name=lvl-pixmo-check
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
(( $# == 1 )) || { echo "uso: pixmo_transcript_check.sbatch OUT" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SHARDS_JSONL="$DATA_ROOT/shards/pixmo-cap/3e42775e/shards.jsonl"
test -r "$SHARDS_JSONL" || { echo "manca $SHARDS_JSONL" >&2; exit 1; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
if ! srun "$RUN_PYTHON" -B tests/test_transcripts.py; then
  ledger_note failed "pixmo_transcript_check: tests/test_transcripts.py fallito, controllo non eseguito"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/pixmo_transcript_check.py --shards-jsonl "$SHARDS_JSONL" --out "$1"; then
  ledger_note ok "pixmo_transcript_check -> $1"
else
  ledger_note failed "pixmo_transcript_check -> $1"
  exit 1
fi
