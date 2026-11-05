#!/usr/bin/env bash
# Riverifica in sola lettura di un manifest candidato già scritto (scripts/manifest_recheck.py), dopo i test mirati: se
# un test fallisce non si legge nulla. Job solo CPU; manifest, shard ed evidenze in sola lettura, nessuna ricostruzione.
# Ingressi fissi: unimer -> outputs/unimer_check/run4, outputs/unimer_overlap/run2 e lo shard delle righe ritagliate
# registrato nel manifest; mavis -> la cartella outputs con le uscite storiche (come slurm/mavis_manifest.sbatch).
# Uso:
#   sbatch slurm/manifest_recheck.sbatch unimer|mavis NOME SHA256 RUN
#     NOME: cartella data/manifests/NOME; SHA256: sha256 atteso del suo manifest.json; RUN: etichetta della riverifica
#     -> outputs/manifest_recheck/RUN/NOME-<sha8>/verifica.json (cartella nuova; anomalie.jsonl se ce ne sono),
#        docs/manifests/RUN/NOME-<sha8>.json (rapporto sintetico per git)
# Esito: 0 controlli superati; 3 controlli non superati (rapporti scritti, anomalie riportate, nulla corretto); altro
# codice: riverifica non eseguita (ingressi diversi da quelli registrati, cartella già scritta).
#SBATCH --job-name=lvl-manifest-recheck
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
SOURCE=${1:-} NAME=${2:-} SHA=${3:-} RUN=${4:-}
[[ $# -eq 4 && ($SOURCE == unimer || $SOURCE == mavis) ]] || {
  echo "uso: manifest_recheck.sbatch unimer|mavis NOME SHA256 RUN" >&2; exit 2; }
[[ $NAME =~ ^[A-Za-z0-9._-]+$ && $SHA =~ ^[0-9a-f]{64}$ && $RUN =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "nome, sha256 o etichetta non validi" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
TESTS=(tests/test_manifest_recheck.py)
if [[ $SOURCE == unimer ]]; then TESTS+=(tests/test_unimer_manifest.py); else TESTS+=(tests/test_mavis_manifest.py); fi
for t in "${TESTS[@]}"; do
  if ! srun "$RUN_PYTHON" -B "$t"; then
    ledger_note failed "manifest_recheck $SOURCE $NAME: $t fallito, nessuna lettura"
    exit 1
  fi
done
MANIFEST="$DATA_ROOT/manifests/$NAME"
TAG="$NAME-${SHA:0:8}"
REPORT="$OUTPUT_ROOT/manifest_recheck/$RUN/$TAG"
SUMMARY="$PROJECT_ROOT/docs/manifests/$RUN/$TAG.json"
if [[ $SOURCE == unimer ]]; then
  INPUTS=(--check "$OUTPUT_ROOT/unimer_check/run4" --overlap "$OUTPUT_ROOT/unimer_overlap/run2")
else
  INPUTS=(--evidence "$OUTPUT_ROOT")
fi
rc=0
srun "$RUN_PYTHON" -u -B scripts/manifest_recheck.py "$SOURCE" --manifest "$MANIFEST" --sha256 "$SHA" \
  "${INPUTS[@]}" --report "$REPORT" --summary "$SUMMARY" || rc=$?
if (( rc == 0 )); then
  ledger_note ok "manifest_recheck $SOURCE $NAME: controlli superati -> $REPORT/verifica.json, riepilogo -> $SUMMARY"
elif (( rc == 3 )); then
  ledger_note failed "manifest_recheck $SOURCE $NAME: controlli non superati, anomalie riportate e nulla corretto -> $REPORT/verifica.json"
  exit 3
else
  ledger_note failed "manifest_recheck $SOURCE $NAME: riverifica non eseguita (codice $rc), vedi il log"
  exit "$rc"
fi
