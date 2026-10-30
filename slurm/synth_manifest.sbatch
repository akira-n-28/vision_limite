#!/usr/bin/env bash
# Manifest delle ammissioni del candidato dei sintetici v4 (scripts/synth_manifest.py) e sue verifiche CPU
# (scripts/manifest_check.py), dopo tests/test_manifest.py: se il test fallisce non si scrive nulla. Job solo CPU; shard e
# uscite del candidato in sola lettura.
# Uso: sbatch slurm/synth_manifest.sbatch NOME VERSIONE CANDIDATO SHARD [opzioni di synth_manifest.py]
#      -> data/manifests/NOME-VERSIONE (scritto una volta sola) e outputs/manifest_check/NOME-VERSIONE
# Esempio (candidato del job 30449):
#   sbatch slurm/synth_manifest.sbatch synth-v4-candidato 1 \
#     /home/prignano/vision_limite/outputs/synth_candidate/v4-run1 /home/prignano/vision_limite/data/shards/synth/v4 \
#     --dev-fraction 0.05 --seed 0
#SBATCH --job-name=lvl-manifest
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 4 )) || { echo "uso: synth_manifest.sbatch NOME VERSIONE CANDIDATO SHARD [opzioni]" >&2; exit 2; }
NAME=$1 VERSION=$2 CANDIDATE=$3 SHARDS=$4
shift 4
[[ $NAME =~ ^[A-Za-z0-9._-]+$ && $VERSION =~ ^[A-Za-z0-9._]+$ ]] || { echo "nome o versione non validi" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
MANIFEST="$DATA_ROOT/manifests/$NAME-$VERSION"
CHECK="$OUTPUT_ROOT/manifest_check/$NAME-$VERSION"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
if ! srun "$RUN_PYTHON" -B tests/test_manifest.py; then
  ledger_note failed "manifest $NAME-$VERSION: tests/test_manifest.py fallito, manifest non scritto"
  exit 1
fi
if ! srun "$RUN_PYTHON" -u -B scripts/synth_manifest.py --candidate "$CANDIDATE" --shards "$SHARDS" --out "$MANIFEST" \
    --name "$NAME" --version "$VERSION" "$@"; then
  ledger_note failed "manifest -> $MANIFEST"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/manifest_check.py --manifest "$MANIFEST" --shards "$SHARDS" --out "$CHECK"; then
  ledger_note ok "manifest -> $MANIFEST, verifiche -> $CHECK"
else
  ledger_note failed "verifiche del manifest -> $CHECK"
  exit 1
fi
