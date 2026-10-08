#!/usr/bin/env bash
# F0: baseline testuale di Violetto con il valutatore ufficiale (limite-evals @ 7ddad34f) sul nostro
# stack: client in venvs/evals-current, engine vLLM 0.26.0 cu129 + plugin Limite in venvs/current.
# Tasksets del piano: AIME 2025, AIME 2026, HMMT Feb 2026 (avg@32, T=0.6, top-p 0.95, seed 0,
# budget = contesto nativo meno la riserva del prompt), alle revisioni fissate dal valutatore (p1).
# Ripresa dopo un timeout: RESUME_FROM=<cartella del tentativo> sbatch slurm/eval_text_baseline.sbatch
#SBATCH --job-name=lvl-eval-text
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
EVAL_BIN="$VENVS_DIR/evals-current/bin/limite-eval"
test -x "$EVAL_BIN" || { echo "manca il client del valutatore: $EVAL_BIN (ENV_NAME=evals)" >&2; exit 1; }
source slurm/prologue.sh
lagrange_prologue gpu

# Engine: il plugin Limite (VLLM_PLUGINS) e le cache del progetto; tutto offline (config/paths.env).
export VLLM_PLUGINS=limite
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
VIOLETTO="$HF_HOME/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
test -f "$VIOLETTO/model.safetensors" || { echo "snapshot di Violetto mancante: $VIOLETTO" >&2; exit 1; }
ENGINE_VLLM=$("$RUN_PYTHON" -I -c 'import importlib.metadata as m; print(m.version("vllm").split("+")[0])')
echo "engine: $(readlink -f "$VENVS_DIR/current") vLLM $ENGINE_VLLM; client: $(readlink -f "$VENVS_DIR/evals-current")"

mkdir -p "$OUTPUT_ROOT/eval"
args=("$VIOLETTO" --tasksets aime25,aime26,hmmt26
      --engine-venv "$(readlink -f "$VENVS_DIR/current")" --expected-vllm-version "$ENGINE_VLLM"
      --out-root "$OUTPUT_ROOT/eval" --run-id "violetto-f0-text-${SLURM_JOB_ID}")
[[ -n ${RESUME_FROM:-} ]] && args+=(--resume-from "$RESUME_FROM")
echo "comando: limite-eval ${args[*]}"
if srun "$EVAL_BIN" "${args[@]}"; then
  ledger_note ok "baseline testuale F0 -> $OUTPUT_ROOT/eval/violetto-f0-text-${SLURM_JOB_ID}"
else
  ledger_note failed "baseline testuale F0 -> $OUTPUT_ROOT/eval/violetto-f0-text-${SLURM_JOB_ID}"
  exit 1
fi
