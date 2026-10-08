#!/usr/bin/env bash
# Prova di generazione con vLLM e il plugin Limite (scripts/vllm_smoke.py): vLLM cu129 sul driver
# del nodo, registrazione di LimiteForCausalLM, quattro problemi facili con risposta nota.
# Richiede l'ambiente in $RUN_PYTHON e lo snapshot di Violetto collegato nella cache HF.
# Argomenti extra passati allo script, es.: sbatch slurm/vllm_smoke.sbatch --enforce-eager
#SBATCH --job-name=lvl-vllm-smoke
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

# Il plugin si attiva solo con VLLM_PLUGINS; la bin del venv serve per ninja (JIT di FlashInfer e
# Inductor). Cache di Triton, Inductor e vLLM nei percorsi del progetto (config/paths.env).
export VLLM_PLUGINS=limite
export PATH="$(dirname "$RUN_PYTHON"):$PATH"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"

echo "=== compilatori visibili (Triton e FlashInfer compilano a runtime) ==="
for c in gcc g++ cc ninja nvcc; do printf '%-6s %s\n' "$c" "$(command -v "$c" || echo assente)"; done

VIOLETTO="$HF_HOME/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
test -f "$VIOLETTO/model.safetensors" || { echo "snapshot di Violetto mancante: $VIOLETTO" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/vllm_smoke"
out="$OUTPUT_ROOT/vllm_smoke/smoke-${SLURM_JOB_ID}.json"
if srun "$RUN_PYTHON" -u -B scripts/vllm_smoke.py --model "$VIOLETTO" --out "$out" "$@"; then
  ledger_note ok "vllm smoke -> $out"
else
  ledger_note failed "vllm smoke -> $out"
  exit 1
fi
