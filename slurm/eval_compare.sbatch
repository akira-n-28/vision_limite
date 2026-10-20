#!/usr/bin/env bash
# Confronto problema per problema fra run di limite-eval (scripts/eval_compare.py). Job solo CPU.
# Uso: sbatch slurm/eval_compare.sbatch TASK RUN_DIR [RUN_DIR ...]   (il primo run è il riferimento)
#SBATCH --job-name=lvl-eval-compare
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 3 )) || { echo "uso: eval_compare.sbatch TASK RUN_DIR RUN_DIR [...]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
TASK=$1
shift
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
srun "$RUN_PYTHON" -u -B scripts/eval_compare.py --task "$TASK" "$@"
