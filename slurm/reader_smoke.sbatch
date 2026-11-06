#!/usr/bin/env bash
# Smoke test di C7 dei reader col manifest delle ammissioni di UniMER e MAVIS (scripts/reader_smoke.py), sui manifest
# candidati e sugli shard reali della config config/runs/f0-c7-reader-unimer-mavis.json. Job solo CPU: nessun modello
# (il processor di Violetto solo per tokenizzare), manifest e shard in sola lettura. Prima i test mirati: se falliscono
# non si leggono i dati reali.
# Uso: sbatch slurm/reader_smoke.sbatch NOME
#   -> outputs/reader_smoke/NOME/report.json e il riepilogo per git docs/reader_smoke/NOME.json
#SBATCH --job-name=lvl-reader-smoke
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) || { echo "uso: reader_smoke.sbatch NOME" >&2; exit 2; }
NAME=$1
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
CONFIG="$PROJECT_ROOT/config/runs/f0-c7-reader-unimer-mavis.json"
OUT="$OUTPUT_ROOT/reader_smoke/$NAME"
SUMMARY="$PROJECT_ROOT/docs/reader_smoke/$NAME.json"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
echo "=== file modificati o nuovi rispetto a HEAD: id git del contenuto eseguito"
git ls-files -m -o --exclude-standard | while IFS= read -r f; do
  if [[ -f $f ]]; then printf '%s  %s\n' "$(git hash-object -- "$f")" "$f"; fi
done
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
test -s "$CONFIG" || { echo "config mancante: $CONFIG" >&2; exit 1; }
[[ ! -e $OUT && ! -e $SUMMARY ]] || { echo "uscite già presenti: $OUT o $SUMMARY" >&2; exit 1; }
for t in tests/test_reader_manifest.py tests/test_manifest.py; do
  if ! srun "$RUN_PYTHON" -B "$t"; then
    ledger_note failed "reader_smoke $NAME: $t fallito, nessuna lettura dei dati reali"
    exit 1
  fi
done
if srun "$RUN_PYTHON" -u -B scripts/reader_smoke.py --config "$CONFIG" --out "$OUT" --summary "$SUMMARY"; then
  ledger_note ok "reader_smoke $NAME -> $OUT/report.json, riepilogo $SUMMARY"
else
  ledger_note failed "reader_smoke $NAME: controlli falliti o errore -> $OUT"
  exit 1
fi
