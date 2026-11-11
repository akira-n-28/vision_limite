#!/usr/bin/env bash
# PixMo-Cap: manifest candidato dall'inventario, dalla verifica e dalla scansione del job 30813 (scripts/pixmo_manifest.py,
# senza rileggere parquet né immagini) e collaudo del reader col manifest sui dati reali (scripts/reader_smoke.py con la
# config di collaudo config/runs/f0-c7-reader-pixmo.json). Job solo CPU: ingressi in sola lettura, uscite in cartelle
# nuove. Prima il test mirato: se fallisce non si legge nulla.
# Uso:
#   sbatch --cpus-per-task 4 --mem 24G slurm/pixmo_manifest.sbatch costruzione VERSIONE
#     -> data/manifests/pixmo-cap-candidato-VERSIONE (scritto una volta sola),
#        outputs/manifest_check/pixmo-cap-candidato-VERSIONE/verifica.json, riepilogo docs/manifests/pixmo-cap-candidato-VERSIONE.json
#   sbatch slurm/pixmo_manifest.sbatch lettore NOME
#     -> outputs/reader_smoke/NOME/report.json, riepilogo docs/reader_smoke/NOME.json
#SBATCH --job-name=lvl-pixmo-manifest
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 2 )) || { echo "uso: pixmo_manifest.sbatch costruzione VERSIONE | lettore NOME" >&2; exit 2; }
MODE=$1 ARG=$2
[[ ($MODE == costruzione || $MODE == lettore) && $ARG =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "uso: pixmo_manifest.sbatch costruzione VERSIONE | lettore NOME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
REVISION=3e42775eea79dd4874c41379230c7f871b10218e
INVENTORY="$OUTPUT_ROOT/pixmo_inventory/completo-1/inventario"
INVENTORY_SHA256=ade3e0c26bd19dd8ac9df60039ada6edf6743eb19b05c2ef1285f2b9e6458578  # job 30813
VERIFICA="$OUTPUT_ROOT/pixmo_inventory/completo-1/verifica"
SCAN="$DATA_ROOT/decontam/pannello-v3/pixmo-cap"
SCAN_SHA256=767658e01af10762f772e39de6a5a954c9bc314620675aa65a025732c0f9b27d  # job 30813
INDEX_SHA256=02fb3cbfc36cbfd17bba92e4511c18a5760d330bc0145855fff2e7aa5130d127  # pannello-v3, job 30469
SHARDS="$DATA_ROOT/shards/pixmo-cap/3e42775e"  # job 29560
RUN4="$OUTPUT_ROOT/pixmo_check/run4"  # job 30044
HIST=("$DATA_ROOT/decontam/p6/pixmo" "$DATA_ROOT/decontam/f1eval/pixmo")
# conteggi autorizzati dall'utente il 2026-11-11 (docs/F0_CAMPIONE_CANDIDATO.md §7): la costruzione si ferma se diversi
COUNTS=(--gruppi-segnaposto 18 --righe-segnaposto 526 --exif-attesi 59 --candidati-attesi 248)
CONFIG="$PROJECT_ROOT/config/runs/f0-c7-reader-pixmo.json"
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
if [[ $MODE == costruzione ]]; then
  NAME="pixmo-cap-candidato-$ARG"
  OUT="$DATA_ROOT/manifests/$NAME" REPORT="$OUTPUT_ROOT/manifest_check/$NAME" SUMMARY="$PROJECT_ROOT/docs/manifests/$NAME.json"
  for f in "$INVENTORY/records.jsonl.gz" "$INVENTORY/summary.json" "$VERIFICA/report.json" "$VERIFICA/duplicati.jsonl.gz" \
      "$VERIFICA/proposte_di_trattamento.jsonl.gz" "$SCAN/records.jsonl.gz" "$SCAN/summary.json" "$SHARDS/plan.json" \
      "$SHARDS/shards.jsonl" "$SHARDS/manifest.json" "$RUN4/report.json" "$RUN4/flagged.jsonl"; do
    test -s "$f" || { echo "ingresso mancante: $f" >&2; ledger_note failed "pixmo_manifest $MODE $ARG: manca $f"; exit 1; }
  done
  [[ ! -e $OUT && ! -e $REPORT && ! -e $SUMMARY ]] || { echo "uscite già presenti: $OUT, $REPORT o $SUMMARY" >&2; exit 1; }
else
  OUT="$OUTPUT_ROOT/reader_smoke/$ARG" SUMMARY="$PROJECT_ROOT/docs/reader_smoke/$ARG.json"
  test -s "$CONFIG" || { echo "config mancante: $CONFIG" >&2; exit 1; }
  [[ ! -e $OUT && ! -e $SUMMARY ]] || { echo "uscite già presenti: $OUT o $SUMMARY" >&2; exit 1; }
fi
if ! srun "$RUN_PYTHON" -B tests/test_pixmo_manifest.py; then
  ledger_note failed "pixmo_manifest $MODE $ARG: tests/test_pixmo_manifest.py fallito, nessuna lettura dei dati reali"
  exit 1
fi
if [[ $MODE == costruzione ]]; then
  if srun "$RUN_PYTHON" -u -B scripts/pixmo_manifest.py costruzione --inventory "$INVENTORY" --verifica "$VERIFICA" \
      --scan "$SCAN" --shards "$SHARDS" --revision "$REVISION" --inventory-sha256 "$INVENTORY_SHA256" \
      --scan-sha256 "$SCAN_SHA256" --index-sha256 "$INDEX_SHA256" --run4 "$RUN4" --historical "${HIST[@]}" \
      --out "$OUT" --report "$REPORT" --summary "$SUMMARY" --version "$ARG" "${COUNTS[@]}"; then
    ledger_note ok "pixmo_manifest costruzione $ARG: manifest -> $OUT, verifica -> $REPORT/verifica.json, riepilogo -> $SUMMARY"
  else
    ledger_note failed "pixmo_manifest costruzione $ARG: costruzione o verifica -> $OUT, $REPORT"
    exit 1
  fi
else
  if srun "$RUN_PYTHON" -u -B scripts/reader_smoke.py --config "$CONFIG" --out "$OUT" --summary "$SUMMARY"; then
    ledger_note ok "pixmo_manifest lettore $ARG -> $OUT/report.json, riepilogo $SUMMARY"
  else
    ledger_note failed "pixmo_manifest lettore $ARG: controlli falliti o errore -> $OUT"
    exit 1
  fi
fi
