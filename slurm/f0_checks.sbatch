#!/usr/bin/env bash
# Verifiche F0 su Violetto (scripts/f0_checks.py). Richiede l'ambiente in $RUN_PYTHON
# e lo snapshot di Violetto già collegato nella cache HF (link_hf_snapshot.py).
#SBATCH --job-name=lvl-f0-checks
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

VIOLETTO="$HF_HOME/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
test -f "$VIOLETTO/model.safetensors" || { echo "snapshot di Violetto mancante: $VIOLETTO" >&2; exit 1; }
out="$OUTPUT_ROOT/f0/checks-${SLURM_JOB_ID}.json"
if srun "$RUN_PYTHON" -u -B scripts/f0_checks.py --model "$VIOLETTO" --out "$out"; then
  ledger_note ok "f0 checks -> $out"
else
  ledger_note failed "f0 checks -> $out"
  exit 1
fi
