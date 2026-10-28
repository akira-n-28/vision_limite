#!/usr/bin/env bash
# Piano di un lotto F2 senza generare (scripts/f2_generate.py --plan-only), in un job CPU: problemi scelti, inclusi ed
# esclusi, plan.json e keys.txt in outputs/f2_gen/NOME_RUN. La generazione vera (slurm/f2_generate.sbatch con gli
# stessi argomenti, senza --plan-only) controlla che il piano coincida.
# Uso: sbatch slurm/f2_plan.sbatch NOME_RUN --parquet FILE... [opzioni di f2_generate.py]
#SBATCH --job-name=lvl-f2-plan
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 2 )) || { echo "uso: f2_plan.sbatch NOME_RUN --parquet FILE... [opzioni]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
RUN=$1
shift
[[ $RUN =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome del run non valido: $RUN" >&2; exit 2; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export VLLM_PLUGINS=limite  # richiesto da f2_generate.py; con --plan-only vLLM non si importa
VIOLETTO="$HF_HOME/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
OUT="$OUTPUT_ROOT/f2_gen/$RUN"
if srun "$RUN_PYTHON" -u -B scripts/f2_generate.py --model "$VIOLETTO" --out "$OUT" --plan-only "$@"; then
  ledger_note ok "F2 piano $RUN -> $OUT/plan.json"
else
  ledger_note failed "F2 piano $RUN -> $OUT/plan.json"
  exit 1
fi
