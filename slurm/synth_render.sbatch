#!/usr/bin/env bash
# Diagrammi sintetici di F1c (scripts/synth_render.py): train, test e coppie controfattuali. Job solo CPU.
# Uso: sbatch slurm/synth_render.sbatch [VERSIONE] --generator-version 3|4 [opzioni di synth_render.py]
#      (cartella data/shards/synth/VERSIONE, default v1; --generator-version è obbligatorio: 3 riproduce v3, 4 è la
#      versione corretta di docs/specs/S2 §2 e salva anche il layout. Non rigenerare v4 prima del collaudo di C1-C2.)
#SBATCH --job-name=lvl-synth
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
VERSION=${1:-v1}
shift || true
[[ $VERSION =~ ^[A-Za-z0-9._-]+$ ]] || { echo "versione non valida: $VERSION" >&2; exit 2; }
OUT="$DATA_ROOT/shards/synth/$VERSION"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
df -hT "$DATA_ROOT" | tail -n 1
if srun "$RUN_PYTHON" -u -B scripts/synth_render.py --out "$OUT" "$@"; then
  ledger_note ok "diagrammi sintetici F1c -> $OUT"
else
  ledger_note failed "diagrammi sintetici F1c -> $OUT"
  exit 1
fi
