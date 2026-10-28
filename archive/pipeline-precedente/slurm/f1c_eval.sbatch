#!/usr/bin/env bash
# Valutazione di un adapter per F1c (scripts/f1c_eval.py): scelta forzata, teacher forcing, generazione a batch.
# Uso: sbatch slurm/f1c_eval.sbatch ADAPTER_DIR NOME [opzioni di f1c_eval.py]
#   config di validazione config/runs/f1b.json (stessi campioni per tutti i modelli), sintetici data/shards/synth/v3,
#   formule UniMER-Test con gli input di f1b (x4) e densi (dense).
#SBATCH --job-name=lvl-f1c-eval
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 2 )) || { echo "uso: f1c_eval.sbatch ADAPTER_DIR NOME [opzioni]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=$PROJECT_ROOT/config/runs/f1b.json
ADAPTER=$1
NAME=$2
shift 2
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
test -d "$ADAPTER" || { echo "adapter mancante: $ADAPTER" >&2; exit 1; }
SYNTH="$DATA_ROOT/shards/synth/v3"
X4="$DATA_ROOT/shards/unimer/2343ddd9-x4/test"
DENSE="$DATA_ROOT/shards/unimer/2343ddd9-dense/test"
for d in "$SYNTH/test" "$SYNTH/pairs" "$X4" "$DENSE"; do test -d "$d" || { echo "cartella mancante: $d" >&2; exit 1; }; done
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
extra=()
STEM_REAL="$OUTPUT_ROOT/f1c/stem_real/facts.json"
[[ -f $STEM_REAL ]] && extra+=(--stem-real "$STEM_REAL")
out="$OUTPUT_ROOT/f1c/eval/$NAME-$SLURM_JOB_ID"
if srun "$RUN_PYTHON" -u -B scripts/f1c_eval.py --config "$CONFIG_PATH" --adapter "$ADAPTER" --name "$NAME" \
    --out-dir "$out" --synth-dir "$SYNTH" --unimer-test x4="$X4" dense="$DENSE" "${extra[@]}" "$@"; then
  ledger_note ok "valutazione F1c ($NAME) -> $out"
else
  ledger_note failed "valutazione F1c ($NAME)"
  exit 1
fi
