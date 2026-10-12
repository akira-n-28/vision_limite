#!/usr/bin/env bash
# Analisi di UniMER-1M.zip e UniMER-Test.zip senza estrarli (scripts/unimer_survey.py). Nessuna GPU.
# Uso: sbatch slurm/unimer_survey.sbatch [opzioni]
#SBATCH --job-name=lvl-unimer
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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
export OMP_NUM_THREADS=1
SNAP="$HF_HOME/hub/datasets--wanderkid--UniMER_Dataset/snapshots/2343ddd963290469da36ca83e3a56c66e068add9"
VIOLETTO="$HF_HOME/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
out="$OUTPUT_ROOT/unimer_survey/$SLURM_JOB_ID"
if srun "$RUN_PYTHON" -u -B scripts/unimer_survey.py --train-zip "$SNAP/UniMER-1M.zip" --test-zip "$SNAP/UniMER-Test.zip" \
    --tokenizer "$VIOLETTO" --out "$out" "$@"; then
  ledger_note ok "analisi UniMER: $out"
else
  ledger_note failed "analisi UniMER"
  exit 1
fi
