#!/usr/bin/env bash
# scripts/f2_certify.py in un job CPU: certificato di un lotto di soluzioni F2 da un audit e dalla politica, in una cartella nuova.
# Uso: sbatch slurm/f2_certify.sbatch --audit outputs/stem_audit/NOME --selection SEL/selected.v2.jsonl --policy config/audit/f2_policy.json --out outputs/f2_cert/NOME [--audit-map MAPPA]
#SBATCH --job-name=lvl-f2-certify
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
if srun "$RUN_PYTHON" -u -B scripts/f2_certify.py "$@"; then
  ledger_note ok "f2_certify: $*"
else
  ledger_note failed "f2_certify: $*"
  exit 1
fi
