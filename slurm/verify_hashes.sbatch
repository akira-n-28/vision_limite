#!/usr/bin/env bash
# Verifica SHA-256 / git-sha1 dei file scaricati. Job solo CPU.
# Uso: sbatch --parsable slurm/verify_hashes.sbatch downloads/manifests/p1-modelli.tsv [altri.tsv]
#SBATCH --job-name=lvl-verify
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 1 )) || { echo "indicare almeno un manifest" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
mkdir -p "$OUTPUT_ROOT/verify"
report="$OUTPUT_ROOT/verify/report-${SLURM_JOB_ID}.tsv"
if srun /usr/bin/python3 -I -B scripts/verify_manifest.py "$@" --report "$report"; then
  ledger_note ok "verify $* -> $report"
else
  ledger_note failed "verify $* -> $report"
  exit 1
fi
