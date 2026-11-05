#!/usr/bin/env bash
# Manifest candidato di MAVIS-Geometry (scripts/mavis_manifest.py; docs/F0_CAMPIONE_CANDIDATO.md §6 punto 3), dopo
# tests/test_mavis_manifest.py: se il test fallisce non si legge nulla. Job solo CPU; shard ed evidenze in sola lettura.
# Ingressi fissi: data/shards/mavis-geometry/9c80fff3-caption (job 29419) e le uscite storiche in outputs/
# (f1_mavis/run2 job 30052, mavis_caption_check/run1 job 30025, mavis_geometry/run1 job 30042, f1_mavis/fractions_run1
# job 30049, split_duplicates/run1 job 30043).
# Uso:
#   sbatch slurm/mavis_manifest.sbatch prova NOME
#     -> outputs/manifest_probe/NOME/probe.json: un tar, tempi, memoria, stima del lavoro completo
#   sbatch slurm/mavis_manifest.sbatch costruzione NOME VERSIONE
#     -> data/manifests/NOME-VERSIONE (scritto una volta sola), outputs/manifest_check/NOME-VERSIONE/verifica.json,
#        docs/manifests/NOME-VERSIONE.json (rapporto sintetico per git)
# Risorse di partenza; si ridimensionano dopo la prova (sbatch --cpus-per-task/--mem/--time).
#SBATCH --job-name=lvl-mavis-manifest
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
  echo "uso: mavis_manifest.sbatch prova NOME | costruzione NOME VERSIONE" >&2; exit 2; }
[[ $NAME =~ ^[A-Za-z0-9._-]+$ && ($MODE == prova || $VERSION =~ ^[A-Za-z0-9._]+$) ]] || {
  echo "nome o versione non validi" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SHARDS="$DATA_ROOT/shards/mavis-geometry/9c80fff3-caption"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
WORKERS=${SLURM_CPUS_PER_TASK:-1}
if ! srun "$RUN_PYTHON" -B tests/test_mavis_manifest.py; then
  ledger_note failed "mavis_manifest $MODE $NAME: tests/test_mavis_manifest.py fallito, nessuna lettura"
  exit 1
fi
if [[ $MODE == prova ]]; then
  OUT="$OUTPUT_ROOT/manifest_probe/$NAME"
  if srun "$RUN_PYTHON" -u -B scripts/mavis_manifest.py --shards "$SHARDS" --evidence "$OUTPUT_ROOT" --report "$OUT" \
      --workers "$WORKERS" --probe; then
    ledger_note ok "mavis_manifest prova $NAME -> $OUT/probe.json"
  else
    ledger_note failed "mavis_manifest prova $NAME"
    exit 1
  fi
  exit 0
fi
MANIFEST="$DATA_ROOT/manifests/$NAME-$VERSION"
REPORT="$OUTPUT_ROOT/manifest_check/$NAME-$VERSION"
SUMMARY="$PROJECT_ROOT/docs/manifests/$NAME-$VERSION.json"
if srun "$RUN_PYTHON" -u -B scripts/mavis_manifest.py --shards "$SHARDS" --evidence "$OUTPUT_ROOT" --out "$MANIFEST" \
    --report "$REPORT" --summary "$SUMMARY" --name "$NAME" --version "$VERSION" --workers "$WORKERS"; then
  ledger_note ok "mavis_manifest -> $MANIFEST, verifica -> $REPORT/verifica.json, riepilogo -> $SUMMARY"
else
  ledger_note failed "mavis_manifest $NAME-$VERSION -> $REPORT"
  exit 1
fi
