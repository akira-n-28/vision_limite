#!/usr/bin/env bash
# Test CPU del codice del progetto (tests/test_*.py: ottimizzatore e gruppi, processor e collator).
# Usa solo l'ambiente del progetto e i file locali (tokenizer di Violetto, shard di LLaVA-Pretrain).
# Uso: sbatch slurm/tests_cpu.sbatch [tests/test_X.py ...]
#   senza argomenti tutti i tests/test_*.py; con argomenti solo quelli indicati (test mirati alle dipendenze
#   modificate). Il log riporta l'id git (git hash-object) dei file modificati o nuovi rispetto a HEAD, da confrontare
#   con git ls-tree dopo il commit.
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
tests=("$@")
(( ${#tests[@]} )) || tests=(tests/test_*.py)
for t in "${tests[@]}"; do
  [[ $t == tests/test_*.py && -f $t ]] || { echo "test non valido: $t" >&2; exit 2; }
done
label=""
if (( $# )); then label=" (${tests[*]})"; fi
source slurm/prologue.sh
lagrange_prologue cpu
echo "=== file modificati o nuovi rispetto a HEAD: id git del contenuto provato"
git ls-files -m -o --exclude-standard | while IFS= read -r f; do
  if [[ -f $f ]]; then printf '%s  %s\n' "$(git hash-object -- "$f")" "$f"; fi
done
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# test_synth.py e test_latex_render.py disegnano con matplotlib: configurazione e cache dentro il progetto
export MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
failed=()
for t in "${tests[@]}"; do
  echo "=== $t"
  srun "$RUN_PYTHON" -B "$t" || failed+=("$t")
done
if (( ${#failed[@]} == 0 )); then
  ledger_note ok "tests_cpu$label"
else
  ledger_note failed "tests_cpu$label: ${failed[*]}"
  exit 1
fi
