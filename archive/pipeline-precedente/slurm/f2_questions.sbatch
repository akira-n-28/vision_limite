#!/usr/bin/env bash
# scripts/f2_questions.py in un job CPU: ricostruzione non distruttiva della domanda nei record di generazioni storiche, in una cartella nuova.
# Uso: sbatch slurm/f2_questions.sbatch --selected SEL/selected.v2.jsonl --gen outputs/f2_gen/RUN --gen-logs LOG... (--mavis-shards GLOB | --parquet P... --data-manifest downloads/manifests/p3.tsv) --out-dir outputs/f2_questions/NOME [--tokenizer MODELLO]
#SBATCH --job-name=lvl-f2-questions
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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
if srun "$RUN_PYTHON" -u -B scripts/f2_questions.py "$@"; then
  ledger_note ok "f2_questions: $*"
else
  ledger_note failed "f2_questions: $*"
  exit 1
fi
