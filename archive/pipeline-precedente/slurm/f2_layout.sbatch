#!/usr/bin/env bash
# scripts/f2_layout.py: soluzioni F2 che descrivono la disposizione della figura (parole e coordinate). Job solo CPU.
# Uso: sbatch slurm/f2_layout.sbatch --selected SEL.jsonl --out OUT.json [--audit DIR]
#SBATCH --job-name=lvl-f2-layout
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 4 )) || { echo "uso: f2_layout.sbatch --selected SEL.jsonl --out OUT.json [--audit DIR]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
srun "$RUN_PYTHON" -u -B scripts/f2_layout.py "$@"
