#!/usr/bin/env bash
# Decontaminazione esterna di UniMER-1M e MAVIS-Geometry contro pannello-v3 (scripts/panel_decontam.py;
# docs/PANNELLO_VALUTAZIONE.md §6.2, S2 §§3-4). Job solo CPU; manifest, shard, snapshot, indice, pannello ed evidenze in
# sola lettura. Prima il test mirato: se fallisce non si legge nulla.
# Uso:
#   sbatch [--time 00:40:00] slurm/panel_decontam.sbatch prova unimer|mavis NOME LIMITE
#     scansione dei primi LIMITE campioni di ogni shard -> outputs/panel_decontam/NOME/scan e verifica di compatibilità
#     con --partial -> outputs/panel_decontam/NOME/verifica (tempo, memoria, volume delle corrispondenze)
#   sbatch slurm/panel_decontam.sbatch completo unimer|mavis NOME VERSIONE
#     scansione completa -> data/decontam/pannello-v3/<unimer|mavis-geometry>, verifica completa
#     -> outputs/panel_decontam/NOME/verifica, versione nuova data/manifests/<candidato>-VERSIONE con la sua verifica
#     -> outputs/manifest_check/<candidato>-VERSIONE/verifica.json e il riepilogo docs/manifests/<candidato>-VERSIONE.json
#   sbatch [--cpus-per-task 4 --time 01:30:00] slurm/panel_decontam.sbatch riesame mavis NOME VERSIONE
#     decisioni testuali di mavis-geometry-candidato-2 (panel-decontam/1, regola di frequenza) ricalcolate con le
#     regole esplicite sulla scansione esistente data/decontam/pannello-v3/mavis-geometry, senza rileggere immagini:
#     regole, decisioni e riepilogo -> data/decontam/pannello-v3/mavis-geometry-NOME, versione nuova
#     data/manifests/mavis-geometry-candidato-VERSIONE derivata da candidato-2, verifica della catena
#     -> outputs/manifest_check/mavis-geometry-candidato-VERSIONE, riepilogo docs/manifests/mavis-geometry-candidato-VERSIONE.json
#SBATCH --job-name=lvl-panel-decontam
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
usage() { echo "uso: panel_decontam.sbatch prova unimer|mavis NOME LIMITE | completo unimer|mavis NOME VERSIONE" \
  "| riesame mavis NOME VERSIONE" >&2; exit 2; }
(( $# == 4 )) || usage
MODE=$1 SOURCE=$2 NAME=$3 ARG=$4
[[ ($MODE == prova || $MODE == completo) && ($SOURCE == unimer || $SOURCE == mavis) ]] \
  || [[ $MODE == riesame && $SOURCE == mavis ]] || usage
[[ $NAME =~ ^[A-Za-z0-9._-]+$ && $ARG =~ ^[0-9]+$ ]] || { echo "nome, limite o versione non validi" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
INDEX="$DATA_ROOT/decontam/pannello-v3"
INDEX_SHA256=02fb3cbfc36cbfd17bba92e4511c18a5760d330bc0145855fff2e7aa5130d127  # job 30469
PANEL="$DATA_ROOT/panel/pannello-v3"
if [[ $SOURCE == unimer ]]; then
  CANDIDATE=unimer-candidato SCAN_NAME=unimer
  MANIFEST="$DATA_ROOT/manifests/unimer-candidato-1"
  MANIFEST_SHA256=bcc851982598a80cdcf60ab00eb2adb0ff0adf581180ec7b296ed3d498d4dfcb  # job 30625, riverifica-2 (30639)
  CHECKS=(--overlap "$OUTPUT_ROOT/unimer_overlap/run2")
  EVIDENCE=(--check "$OUTPUT_ROOT/unimer_check/run4")
else
  CANDIDATE=mavis-geometry-candidato SCAN_NAME=mavis-geometry
  MANIFEST="$DATA_ROOT/manifests/mavis-geometry-candidato-1"
  MANIFEST_SHA256=dc0f264c60c1549cc1c9412ccae15d0fa6040ef1e55ac99cda0046957d6dbc2e  # job 30630, riverifica-2 (30640)
  CHECKS=(--historical "$DATA_ROOT/decontam/p6/mavis" "$DATA_ROOT/decontam/f1eval/mavis")
  EVIDENCE=(--evidence "$OUTPUT_ROOT")
fi
OUT="$OUTPUT_ROOT/panel_decontam/$NAME"
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
for f in "$INDEX/index.json" "$INDEX/verifica.json" "$PANEL/elements.jsonl.gz" "$PANEL/summary.json" \
    "$MANIFEST/manifest.json"; do
  test -s "$f" || { echo "ingresso mancante: $f" >&2; ledger_note failed "panel_decontam $MODE $SOURCE $NAME: manca $f"; exit 1; }
done
[[ ! -e $OUT ]] || { echo "$OUT esiste già" >&2; exit 1; }
if ! srun "$RUN_PYTHON" -B tests/test_panel_decontam.py; then
  ledger_note failed "panel_decontam $MODE $SOURCE $NAME: tests/test_panel_decontam.py fallito, nessuna lettura"
  exit 1
fi
common=(--manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA256" --index "$INDEX" --index-sha256 "$INDEX_SHA256")
if [[ $MODE == riesame ]]; then
  # genitore: candidato-2 (job 30652, panel-decontam/1); origine: candidato-1 (job 30630); la scansione di candidato-2
  PARENT="$DATA_ROOT/manifests/mavis-geometry-candidato-2"
  PARENT_SHA256=8c549a23d60934d68f3b4db1ee3d2a24e26ec822b5c5a079bb5411a5f9c81f67  # job 30652
  SCAN="$INDEX/mavis-geometry"
  RIESAME="$INDEX/mavis-geometry-$NAME"
  NEW="$DATA_ROOT/manifests/$CANDIDATE-$ARG"
  CHECK="$OUTPUT_ROOT/manifest_check/$CANDIDATE-$ARG"
  SUMMARY="$PROJECT_ROOT/docs/manifests/$CANDIDATE-$ARG.json"
  for f in "$PARENT/manifest.json" "$SCAN/records.jsonl.gz" "$SCAN/summary.json"; do
    test -s "$f" || { echo "ingresso mancante: $f" >&2; ledger_note failed "panel_decontam riesame $NAME: manca $f"; exit 1; }
  done
  [[ ! -e $RIESAME && ! -e $NEW && ! -e $CHECK && ! -e $SUMMARY ]] || {
    echo "uscite già presenti: $RIESAME, $NEW, $CHECK o $SUMMARY" >&2; exit 1; }
  if srun "$RUN_PYTHON" -u -B scripts/panel_decontam.py riesame mavis --origine "$MANIFEST" \
      --origine-sha256 "$MANIFEST_SHA256" --manifest "$PARENT" --manifest-sha256 "$PARENT_SHA256" --index "$INDEX" \
      --index-sha256 "$INDEX_SHA256" --panel "$PANEL" --scan "$SCAN" "${EVIDENCE[@]}" --out "$RIESAME" --new "$NEW" \
      --version "$ARG" --report "$CHECK" --summary "$SUMMARY"; then
    ledger_note ok "panel_decontam riesame mavis $NAME: decisioni -> $RIESAME, manifest -> $NEW, verifica -> $CHECK/verifica.json, riepilogo -> $SUMMARY"
  else
    ledger_note failed "panel_decontam riesame mavis $NAME: riesame o sua verifica -> $RIESAME, $NEW, $CHECK"
    exit 1
  fi
  exit 0
fi
if [[ $MODE == prova ]]; then
  SCAN="$OUT/scan"
  if ! srun "$RUN_PYTHON" -u -B scripts/panel_decontam.py scan "$SOURCE" "${common[@]}" --out "$SCAN" --limit "$ARG"; then
    ledger_note failed "panel_decontam prova $SOURCE $NAME: scansione limitata -> $SCAN"
    exit 1
  fi
  if srun "$RUN_PYTHON" -u -B scripts/panel_decontam.py verifica "$SOURCE" "${common[@]}" --panel "$PANEL" \
      "${CHECKS[@]}" --scan "$SCAN" --report "$OUT/verifica" --partial; then
    ledger_note ok "panel_decontam prova $SOURCE $NAME (primi $ARG campioni per shard) -> $OUT"
  else
    ledger_note failed "panel_decontam prova $SOURCE $NAME: verifica di compatibilità -> $OUT/verifica"
    exit 1
  fi
  exit 0
fi
SCAN="$INDEX/$SCAN_NAME"
NEW="$DATA_ROOT/manifests/$CANDIDATE-$ARG"
CHECK="$OUTPUT_ROOT/manifest_check/$CANDIDATE-$ARG"
SUMMARY="$PROJECT_ROOT/docs/manifests/$CANDIDATE-$ARG.json"
[[ ! -e $SCAN && ! -e $NEW && ! -e $CHECK && ! -e $SUMMARY ]] || {
  echo "uscite già presenti: $SCAN, $NEW, $CHECK o $SUMMARY" >&2; exit 1; }
if ! srun "$RUN_PYTHON" -u -B scripts/panel_decontam.py scan "$SOURCE" "${common[@]}" --out "$SCAN"; then
  ledger_note failed "panel_decontam completo $SOURCE $NAME: scansione -> $SCAN"
  exit 1
fi
if ! srun "$RUN_PYTHON" -u -B scripts/panel_decontam.py verifica "$SOURCE" "${common[@]}" --panel "$PANEL" \
    "${CHECKS[@]}" --scan "$SCAN" --report "$OUT/verifica"; then
  ledger_note failed "panel_decontam completo $SOURCE $NAME: verifica di compatibilità -> $OUT/verifica"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/panel_decontam.py manifest "$SOURCE" "${common[@]}" --panel "$PANEL" \
    "${CHECKS[@]}" "${EVIDENCE[@]}" --scan "$SCAN" --new "$NEW" --version "$ARG" --report "$CHECK" \
    --summary "$SUMMARY"; then
  ledger_note ok "panel_decontam completo $SOURCE $NAME: scansione -> $SCAN, manifest -> $NEW, verifica -> $CHECK/verifica.json, riepilogo -> $SUMMARY"
else
  ledger_note failed "panel_decontam completo $SOURCE $NAME: manifest o sua verifica -> $NEW, $CHECK"
  exit 1
fi
