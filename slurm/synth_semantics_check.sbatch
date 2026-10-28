#!/usr/bin/env bash
# Verifica semantica e del renderer dei sintetici (scripts/synth_semantics_check.py): domande contro il disegno e
# invarianti del renderer, su shard esistenti in sola lettura e su scene generate in memoria. Prima i test dei
# sintetici (tests/test_synth.py); se falliscono il controllo non parte. Job solo CPU; non scrive shard.
# Uso: sbatch slurm/synth_semantics_check.sbatch NOME [opzioni di synth_semantics_check.py]   -> outputs/synth_check/NOME
# Esempio (collaudo di C1-C2 su v3 e su 20.000 scene v4 in memoria):
#   sbatch slurm/synth_semantics_check.sbatch c1c2-run1 \
#     --shards '/home/prignano/vision_limite/data/shards/synth/v3/*/*.tar' --generate 20000 --generator-version 4
#SBATCH --job-name=lvl-synth-check
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
(( $# >= 1 )) || { echo "uso: synth_semantics_check.sbatch NOME [opzioni]" >&2; exit 2; }
NAME=$1
shift
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
OUT="$OUTPUT_ROOT/synth_check/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
if ! srun "$RUN_PYTHON" -B tests/test_synth.py; then
  ledger_note failed "synth_check $NAME: tests/test_synth.py fallito, controllo non eseguito"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/synth_semantics_check.py --out "$OUT" "$@"; then
  ledger_note ok "synth_check -> $OUT"
else
  ledger_note failed "synth_check -> $OUT"
  exit 1
fi
