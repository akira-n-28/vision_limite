#!/usr/bin/env bash
# Manifest candidato di UniMER-1M (scripts/unimer_manifest.py; docs/F0_CAMPIONE_CANDIDATO.md §6 punto 3), dopo
# tests/test_unimer_manifest.py: se il test fallisce non si legge nulla. Job solo CPU; shard ed evidenze in sola lettura.
# Ingressi fissi: data/shards/unimer/2343ddd9-dense (job 29729), outputs/unimer_check/run4 (job 30047),
# outputs/unimer_overlap/run2 (job 30048), outputs/split_duplicates/run1/unimer (job 30043).
# Uso:
#   sbatch slurm/unimer_manifest.sbatch prova NOME
#     -> outputs/manifest_probe/NOME/probe.json: un tar per insieme, tempi, memoria, stima del lavoro e dello spazio
#   sbatch slurm/unimer_manifest.sbatch costruzione NOME VERSIONE
#     -> data/manifests/NOME-VERSIONE (scritto una volta sola), outputs/manifest_check/NOME-VERSIONE/verifica.json,
#        docs/manifests/NOME-VERSIONE.json (rapporto sintetico per git)
# Risorse di partenza; si ridimensionano dopo la prova (sbatch --cpus-per-task/--mem/--time).
#SBATCH --job-name=lvl-unimer-manifest
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
MODE=${1:-} NAME=${2:-} VERSION=${3:-}
[[ ($MODE == prova && $# -eq 2) || ($MODE == costruzione && $# -eq 3) ]] || {
  echo "uso: unimer_manifest.sbatch prova NOME | costruzione NOME VERSIONE" >&2; exit 2; }
[[ $NAME =~ ^[A-Za-z0-9._-]+$ && ($MODE == prova || $VERSION =~ ^[A-Za-z0-9._]+$) ]] || {
  echo "nome o versione non validi" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SHARDS="$DATA_ROOT/shards/unimer/2343ddd9-dense"
CHECK="$OUTPUT_ROOT/unimer_check/run4"
OVERLAP="$OUTPUT_ROOT/unimer_overlap/run2"
DUPS="$OUTPUT_ROOT/split_duplicates/run1/unimer"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
WORKERS=${SLURM_CPUS_PER_TASK:-1}
if ! srun "$RUN_PYTHON" -B tests/test_unimer_manifest.py; then
  ledger_note failed "unimer_manifest $MODE $NAME: tests/test_unimer_manifest.py fallito, nessuna lettura"
  exit 1
fi
if [[ $MODE == prova ]]; then
  OUT="$OUTPUT_ROOT/manifest_probe/$NAME"
  if srun "$RUN_PYTHON" -u -B scripts/unimer_manifest.py --shards "$SHARDS" --check "$CHECK" --overlap "$OVERLAP" \
      --duplicates "$DUPS" --report "$OUT" --workers "$WORKERS" --probe; then
    ledger_note ok "unimer_manifest prova $NAME -> $OUT/probe.json"
  else
    ledger_note failed "unimer_manifest prova $NAME"
    exit 1
  fi
  exit 0
fi
MANIFEST="$DATA_ROOT/manifests/$NAME-$VERSION"
REPORT="$OUTPUT_ROOT/manifest_check/$NAME-$VERSION"
SUMMARY="$PROJECT_ROOT/docs/manifests/$NAME-$VERSION.json"
if srun "$RUN_PYTHON" -u -B scripts/unimer_manifest.py --shards "$SHARDS" --check "$CHECK" --overlap "$OVERLAP" \
    --duplicates "$DUPS" --out "$MANIFEST" --report "$REPORT" --summary "$SUMMARY" --name "$NAME" \
    --version "$VERSION" --workers "$WORKERS"; then
  ledger_note ok "unimer_manifest -> $MANIFEST, verifica -> $REPORT/verifica.json, riepilogo -> $SUMMARY"
else
  ledger_note failed "unimer_manifest $NAME-$VERSION -> $REPORT"
  exit 1
fi
