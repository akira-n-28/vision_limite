#!/usr/bin/env bash
# Verifica CPU, offline, che il valutatore carichi i dataset dei tasksets dalla cartella di link
# (scripts/link_eval_datasets.py) con la propria funzione tasksets.load_rows. Nessuna GPU.
# Uso: sbatch slurm/eval_datasets_check.sbatch aime25,aime26,hmmt26
#SBATCH --job-name=lvl-eval-data
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
TASKSETS=${1:?indicare i tasksets, es. aime25,aime26,hmmt26}
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
EVALS_PY="$VENVS_DIR/evals-current/bin/python"
test -x "$EVALS_PY" || { echo "manca il client del valutatore (venvs/evals-current)" >&2; exit 1; }
source slurm/prologue.sh
lagrange_prologue cpu

DS_ROOT="$DATA_ROOT/eval-datasets"
"$EVALS_PY" -I scripts/link_eval_datasets.py --tasksets "$TASKSETS" --root "$DS_ROOT"
cd "$DS_ROOT"
if srun "$EVALS_PY" -I - "$TASKSETS" <<'PY'
import sys
from limite_evals import suites, tasksets

for name in sys.argv[1].split(","):
    spec = suites.taskset("math-extended", name)
    rows = tasksets.load_rows(spec)
    problem, gold = rows[0]
    print(f"{name}: {spec.dataset}@{spec.revision[:8]} split={spec.split} -> {len(rows)} problemi; "
          f"primo gold {gold!r}; primo problema {problem[:80]!r}", flush=True)
PY
then
  ledger_note ok "caricamento offline dei dataset $TASKSETS da $DS_ROOT"
else
  ledger_note failed "caricamento offline dei dataset $TASKSETS da $DS_ROOT"
  exit 1
fi
