#!/usr/bin/env bash
# Candidato dei sintetici v4 (scripts/synth_candidate.py): integrità e riproducibilità degli artefatti salvati, verifica
# indipendente confrontata col controllo, coppie, duplicati e split, stato di ogni scena. Job solo CPU; legge gli shard
# in sola lettura e non scrive shard.
# Uso: sbatch slurm/synth_candidate.sbatch NOME --shards DIR --check DIR [--reference MANIFEST]
#      -> outputs/synth_candidate/NOME
# Esempio (render v4 e controllo synth_semantics_check v4-run1):
#   sbatch slurm/synth_candidate.sbatch v4-run1 --shards /home/prignano/vision_limite/data/shards/synth/v4 \
#     --check /home/prignano/vision_limite/outputs/synth_check/v4-run1 \
#     --reference /home/prignano/vision_limite/data/shards/synth/v3/manifest.json
#SBATCH --job-name=lvl-synth-cand
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 1 )) || { echo "uso: synth_candidate.sbatch NOME [opzioni]" >&2; exit 2; }
NAME=$1
shift
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
OUT="$OUTPUT_ROOT/synth_candidate/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
if srun "$RUN_PYTHON" -u -B scripts/synth_candidate.py --out "$OUT" "$@"; then
  ledger_note ok "synth_candidate -> $OUT"
else
  ledger_note failed "synth_candidate -> $OUT"
  exit 1
fi
