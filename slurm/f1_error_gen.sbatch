#!/usr/bin/env bash
# Materiale per l'analisi degli errori di F1 (scripts/f1_error_gen.py): teacher forcing con immagine vera e bianca
# per tipo di token e generazioni (vera, bianca, campionata) sui campioni di validazione, per un adapter.
# Uso: sbatch slurm/f1_error_gen.sbatch CONFIG.json ADAPTER_DIR NOME [opzioni]
#SBATCH --job-name=lvl-f1-errors
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 3 )) || { echo "uso: f1_error_gen.sbatch CONFIG.json ADAPTER_DIR NOME [opzioni]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=$1
ADAPTER=$2
NAME=$3
shift 3
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
test -d "$ADAPTER" || { echo "adapter mancante: $ADAPTER" >&2; exit 1; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
out="$OUTPUT_ROOT/f1_errors/$NAME-$SLURM_JOB_ID"
if srun "$RUN_PYTHON" -u -B scripts/f1_error_gen.py --config "$CONFIG_PATH" --adapter "$ADAPTER" --name "$NAME" \
    --out-dir "$out" "$@"; then
  ledger_note ok "materiale per l analisi degli errori di F1 ($NAME) -> $out"
else
  ledger_note failed "materiale per l analisi degli errori di F1 ($NAME)"
  exit 1
fi
