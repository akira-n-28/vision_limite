#!/usr/bin/env bash
# Decoder di un run F2 -> checkpoint nel formato di Violetto (scripts/export_decoder.py). Job solo CPU.
# Uso: sbatch slurm/export_decoder.sbatch (WEIGHTS.safetensors | roundtrip) OUT_DIR
#   WEIGHTS: decoder_master.safetensors in outputs/<run>/snapshots/step-N/ o checkpoints/step-N/
#   roundtrip: esporta Violetto stesso e controlla che torni uguale bit per bit allo snapshot
#SBATCH --job-name=lvl-export
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 2 )) || { echo "uso: export_decoder.sbatch (WEIGHTS.safetensors | roundtrip) OUT_DIR" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SRC=$1
OUT=$2
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
VIOLETTO="$HF_HOME/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
if [[ $SRC == roundtrip ]]; then
  src_args=(--roundtrip)
else
  test -f "$SRC" || { echo "pesi mancanti: $SRC" >&2; exit 1; }
  src_args=(--weights "$SRC")
fi
if srun "$RUN_PYTHON" -u -B scripts/export_decoder.py --decoder "$VIOLETTO" "${src_args[@]}" --out "$OUT"; then
  ledger_note ok "export decoder $SRC -> $OUT"
else
  ledger_note failed "export decoder $SRC -> $OUT"
  exit 1
fi
