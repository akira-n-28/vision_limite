#!/usr/bin/env bash
# Analisi degli scarti del controllo dei sintetici (scripts/synth_review.py): coppie controfattuali polygon di v3,
# riquadri di testo sovrapposti nelle scene v4, domande coords non ammissibili. Legge il controllo (problems.jsonl,
# report.json) e lo shard delle coppie v3 in sola lettura e ridisegna in memoria; non scrive shard. Job solo CPU.
# Uso: sbatch slurm/synth_review.sbatch NOME [opzioni di synth_review.py]   -> outputs/synth_review/NOME
#SBATCH --job-name=lvl-synth-review
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 1 )) || { echo "uso: synth_review.sbatch NOME [opzioni]" >&2; exit 2; }
NAME=$1
shift
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
OUT="$OUTPUT_ROOT/synth_review/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
if srun "$RUN_PYTHON" -u -B scripts/synth_review.py --out "$OUT" "$@"; then
  ledger_note ok "synth_review -> $OUT"
else
  ledger_note failed "synth_review -> $OUT"
  exit 1
fi
