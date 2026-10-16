#!/usr/bin/env bash
# Valutazione testuale di un decoder esportato dal progetto (scripts/export_decoder.py) con il valutatore
# ufficiale invariato (limite-evals @ 7ddad34f), tramite scripts/eval_text_local.py (opzione 1 dell'utente,
# 2026-10-16). Stesso engine, client, dataset e protocollo della baseline F0 (slurm/eval_text_baseline.sbatch):
# avg@32, T=0.6, top-p 0.95, seed 0, budget = contesto nativo meno la riserva del prompt.
# Uso: sbatch [--time=...] slurm/eval_text_local.sbatch CHECKPOINT_DIR NOME [TASKSETS] [argomenti di limite-eval]
#   TASKSETS di default aime26; es. argomenti: --limit 2 per una prova corta.
# Ripresa dopo un timeout: RESUME_FROM=<cartella del tentativo> sbatch slurm/eval_text_local.sbatch ...
#SBATCH --job-name=lvl-eval-local
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 2 )) || { echo "uso: eval_text_local.sbatch CHECKPOINT_DIR NOME [TASKSETS] [argomenti di limite-eval]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
CKPT=$1
NAME=$2
TASKSETS=${3:-aime26}
shift $(( $# >= 3 ? 3 : 2 ))
[[ $NAME =~ ^[a-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
EVAL_PY="$VENVS_DIR/evals-current/bin/python"
test -x "$EVAL_PY" || { echo "manca il client del valutatore: $EVAL_PY (ENV_NAME=evals)" >&2; exit 1; }
test -f "$CKPT/model.safetensors" -a -f "$CKPT/export.json" || { echo "checkpoint esportato mancante: $CKPT" >&2; exit 1; }
source slurm/prologue.sh
lagrange_prologue gpu

export VLLM_PLUGINS=limite
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
ENGINE_VLLM=$("$RUN_PYTHON" -I -c 'import importlib.metadata as m; print(m.version("vllm").split("+")[0])')
echo "engine: $(readlink -f "$VENVS_DIR/current") vLLM $ENGINE_VLLM; client: $(readlink -f "$VENVS_DIR/evals-current")"

mkdir -p "$OUTPUT_ROOT/eval"
DS_ROOT="$DATA_ROOT/eval-datasets"
"$EVAL_PY" -I scripts/link_eval_datasets.py --tasksets "$TASKSETS" --root "$DS_ROOT"
RUN_ID="$NAME-text-${SLURM_JOB_ID}"
args=(--tasksets "$TASKSETS"
      --engine-venv "$(readlink -f "$VENVS_DIR/current")" --expected-vllm-version "$ENGINE_VLLM"
      --out-root "$OUTPUT_ROOT/eval" --run-id "$RUN_ID" "$@")
[[ -n ${RESUME_FROM:-} ]] && args+=(--resume-from "$RESUME_FROM")
SCRIPT="$PROJECT_ROOT/scripts/eval_text_local.py"
CKPT=$(readlink -f "$CKPT")
cd "$DS_ROOT"  # il valutatore legge i dataset dai link in questa cartella (job 29391)
echo "comando: eval_text_local.py --checkpoint $CKPT --name $NAME -- ${args[*]}"
if srun "$EVAL_PY" -I "$SCRIPT" --checkpoint "$CKPT" --name "$NAME" --links "$OUTPUT_ROOT/eval-models" -- "${args[@]}"; then
  ledger_note ok "valutazione testuale $NAME ($TASKSETS) -> $OUTPUT_ROOT/eval/$RUN_ID"
else
  ledger_note failed "valutazione testuale $NAME ($TASKSETS) -> $OUTPUT_ROOT/eval/$RUN_ID"
  exit 1
fi
