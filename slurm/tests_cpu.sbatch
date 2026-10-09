#!/usr/bin/env bash
# Test CPU del codice del progetto (tests/test_*.py: ottimizzatore e gruppi, processor e collator).
# Usa solo l'ambiente del progetto e i file locali (tokenizer di Violetto, shard di LLaVA-Pretrain).
#SBATCH --job-name=lvl-tests
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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
failed=()
for t in tests/test_*.py; do
  echo "=== $t"
  srun "$RUN_PYTHON" -B "$t" || failed+=("$t")
done
if (( ${#failed[@]} == 0 )); then
  ledger_note ok "tests_cpu"
else
  ledger_note failed "tests_cpu: ${failed[*]}"
  exit 1
fi
