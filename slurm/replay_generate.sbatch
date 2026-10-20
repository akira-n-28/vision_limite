#!/usr/bin/env bash
# F2, replay testuale: Violetto risolve problemi di testo con il budget pieno (scripts/replay_generate.py).
# Uso: sbatch [--time=... --mem=...] slurm/replay_generate.sbatch NOME_RUN --select N [opzioni]
#   aggiungere --plan-only (e --gres=gpu:0) per scrivere solo il piano.
# Uscita in outputs/replay_gen/NOME_RUN: rilanciando con lo stesso nome riprende dalle parti già scritte.
#SBATCH --job-name=lvl-replay-gen
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 2 )) || { echo "uso: replay_generate.sbatch NOME_RUN --select N [opzioni]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
RUN=$1
shift
[[ $RUN =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome del run non valido: $RUN" >&2; exit 2; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
if [[ " $* " == *" --plan-only "* ]]; then lagrange_prologue cpu; else lagrange_prologue gpu; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export VLLM_PLUGINS=limite
export PATH="$(dirname "$RUN_PYTHON"):$PATH"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
VIOLETTO="$HF_HOME/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
test -f "$VIOLETTO/model.safetensors" || { echo "snapshot di Violetto mancante: $VIOLETTO" >&2; exit 1; }
DSR="$HF_HOME/hub/datasets--agentica-org--DeepScaleR-Preview-Dataset/snapshots/b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29"
DM="$HF_HOME/hub/datasets--zwhe99--DeepMath-103K/snapshots/5cf055d1fe3d7a2eb19719ac020211469736ae44"
test -f "$DSR/deepscaler.json" -a -d "$DM/data" || { echo "snapshot del replay mancanti" >&2; exit 1; }
DECONTAM="$DATA_ROOT/decontam/text/replay/exclude_final.txt"
test -f "$DECONTAM" || { echo "manca la decontaminazione del replay: $DECONTAM (slurm/replay_decontam.sbatch)" >&2; exit 1; }
OUT="$OUTPUT_ROOT/replay_gen/$RUN"
if srun "$RUN_PYTHON" -u -B scripts/replay_generate.py --model "$VIOLETTO" --deepscaler "$DSR/deepscaler.json" \
     --deepmath "$DM"/data/train-*.parquet --exclude "$DECONTAM" --out "$OUT" "$@"; then
  ledger_note ok "replay generazione $RUN -> $OUT"
else
  ledger_note failed "replay generazione $RUN -> $OUT"
  exit 1
fi
