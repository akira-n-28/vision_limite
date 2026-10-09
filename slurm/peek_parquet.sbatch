#!/usr/bin/env bash
# Ispezione di file Parquet (scripts/peek_parquet.py) con lunghezze in token di Violetto. Nessuna GPU.
# Uso: sbatch slurm/peek_parquet.sbatch FILE.parquet [FILE.parquet ...]
#SBATCH --job-name=lvl-peek
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
(( $# > 0 )) || { echo "indicare almeno un file parquet" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
VIOLETTO="$HF_HOME/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
if srun "$RUN_PYTHON" -I scripts/peek_parquet.py "$@" --tokenizer "$VIOLETTO"; then
  ledger_note ok "peek parquet: $*"
else
  ledger_note failed "peek parquet: $*"
  exit 1
fi
