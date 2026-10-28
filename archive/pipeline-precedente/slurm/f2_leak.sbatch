#!/usr/bin/env bash
# Fughe dal ponte testuale nelle soluzioni F2 di Violetto (scripts/f2_leak.py). Job solo CPU.
# Uso: sbatch slurm/f2_leak.sbatch RUN_SELEZIONE   (es. f2-sft120k-v4 -> outputs/f2_select/f2-sft120k-v4/selected.jsonl)
#SBATCH --job-name=lvl-f2-leak
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) || { echo "uso: f2_leak.sbatch RUN_SELEZIONE" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
RUN=$1
[[ $RUN =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $RUN" >&2; exit 2; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
SEL="$OUTPUT_ROOT/f2_select/$RUN/selected.jsonl"
test -f "$SEL" || { echo "manca $SEL" >&2; exit 1; }
SNAP="$HF_HOME/hub/datasets--OpenDataArena--MMFineReason-1.8M/snapshots/7c93f724791d01246f5425e1182b14ec8d013847/data"
source slurm/prologue.sh
lagrange_prologue cpu
out="$OUTPUT_ROOT/f2_leak/$RUN"
if srun "$RUN_PYTHON" -u -B scripts/f2_leak.py --selected "$SEL" --parquet-glob "$SNAP/sft-*.parquet" --out "$out"; then
  ledger_note ok "fughe dal ponte ($RUN) -> $out"
else
  ledger_note failed "fughe dal ponte ($RUN)"
  exit 1
fi
