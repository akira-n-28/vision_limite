#!/usr/bin/env bash
# Verifiche F0 del modello VL (scripts/f0_vl_checks.py): parità con Violetto, hook, packing, gradient
# checkpointing, mini-training F1, throughput F1/F2 (sdpa e flex_attention), generazione.
# Richiede venvs/current, il codice di Limite in third_party/limite-modeling (vendor_limite_modeling.py),
# gli snapshot di Violetto e SigLIP2 so400m e il primo shard di LLaVA-Pretrain.
#SBATCH --job-name=lvl-f0-vl
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue gpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

VIOLETTO="$HF_HOME/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
SIGLIP2="$HF_HOME/hub/models--google--siglip2-so400m-patch16-naflex/snapshots/cc24074f717b612951c2dead130904ab9b65a81e"
SHARD="$DATA_ROOT/shards/llava-pretrain/70f9d1e5/llava-pretrain-00000.tar"
for f in "$VIOLETTO/model.safetensors" "$SIGLIP2/model.safetensors" "$SHARD"; do
  test -f "$f" || { echo "file mancante: $f" >&2; exit 1; }
done
out="$OUTPUT_ROOT/f0/vl-checks-${SLURM_JOB_ID}.json"
if srun "$RUN_PYTHON" -u -B scripts/f0_vl_checks.py --decoder "$VIOLETTO" --vision "$SIGLIP2" --shard "$SHARD" --out "$out"; then
  ledger_note ok "f0 VL checks -> $out"
else
  ledger_note failed "f0 VL checks -> $out"
  exit 1
fi
