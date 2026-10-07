#!/usr/bin/env bash
# Test CPU del codice del progetto (ottimizzatore e gruppi). Usa solo l'ambiente del progetto.
#SBATCH --job-name=lvl-tests
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
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
if srun "$RUN_PYTHON" -B tests/test_normuon.py; then
  ledger_note ok "tests_cpu"
else
  ledger_note failed "tests_cpu"
  exit 1
fi
