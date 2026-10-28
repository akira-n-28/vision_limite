#!/usr/bin/env bash
# Training F1 (scripts/train_f1.py). Uso: sbatch [--time=... --mem=...] slurm/train_f1.sbatch CONFIG.json NOME_RUN [opzioni]
# Uscita in outputs/NOME_RUN (non dipende dal job): rilanciando con lo stesso nome il training riprende
# dall'ultimo checkpoint (--dependency=afterany:<jobid> per concatenare i job). Si ferma con un
# checkpoint --margin-min minuti prima della scadenza del job (default 10).
#SBATCH --job-name=lvl-f1
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 2 )) || { echo "uso: train_f1.sbatch CONFIG.json NOME_RUN [opzioni]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=$1
RUN=$2
shift 2
[[ $RUN =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome del run non valido: $RUN" >&2; exit 2; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS=1  # i worker del DataLoader sono processi separati
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
OUT="$OUTPUT_ROOT/$RUN"
if srun "$RUN_PYTHON" -u -B scripts/train_f1.py --config "$CONFIG_PATH" --out "$OUT" "$@"; then
  ledger_note ok "F1 $RUN -> $OUT"
else
  ledger_note failed "F1 $RUN -> $OUT"
  exit 1
fi
