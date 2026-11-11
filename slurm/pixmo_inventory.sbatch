#!/usr/bin/env bash
# PixMo-Cap @ 3e42775e: inventario tecnico del sottoinsieme locale, collegamento originale -> shard, riproduzione di
# outputs/pixmo_check/run4 e decontaminazione contro pannello-v3 (scripts/pixmo_inventory.py;
# docs/PANNELLO_VALUTAZIONE.md §6.2). Job solo CPU: snapshot, shard, indice, pannello ed evidenze storiche in sola
# lettura, uscite in cartelle nuove; nessun manifest delle ammissioni, nessuna lista per i loader. Prima i test mirati:
# se falliscono non si legge nulla.
# Uso:
#   sbatch --time 00:40:00 slurm/pixmo_inventory.sbatch sonda NOME
#     primo parquet e primo shard: inventario, scansione e verifica parziale -> outputs/pixmo_inventory/NOME/
#     (tempo, memoria, byte letti e copertura per il preventivo della scansione completa)
#   sbatch slurm/pixmo_inventory.sbatch completo NOME
#     i 25 parquet del piano: inventario -> outputs/pixmo_inventory/NOME/inventario, scansione ->
#     data/decontam/pannello-v3/pixmo-cap, verifica -> outputs/pixmo_inventory/NOME/verifica
#SBATCH --job-name=lvl-pixmo-inv
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
usage() { echo "uso: pixmo_inventory.sbatch sonda|completo NOME" >&2; exit 2; }
(( $# == 2 )) || usage
MODE=$1 NAME=$2
[[ $MODE == sonda || $MODE == completo ]] || usage
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
REVISION=3e42775eea79dd4874c41379230c7f871b10218e
SNAPSHOT="$PROJECT_ROOT/hf/hub/datasets--anthracite-org--pixmo-cap-images/snapshots/$REVISION"
SHARDS="$DATA_ROOT/shards/pixmo-cap/3e42775e"  # job 29560
INDEX="$DATA_ROOT/decontam/pannello-v3"
INDEX_SHA256=02fb3cbfc36cbfd17bba92e4511c18a5760d330bc0145855fff2e7aa5130d127  # job 30469
PANEL="$DATA_ROOT/panel/pannello-v3"
RUN4="$OUTPUT_ROOT/pixmo_check/run4"  # job 30044
HIST=("$DATA_ROOT/decontam/p6/pixmo" "$DATA_ROOT/decontam/f1eval/pixmo")
VERIFY="$OUTPUT_ROOT/verify/report-29554.tsv"  # sha256 dei 25 parquet e del README, job 29554
OUT="$OUTPUT_ROOT/pixmo_inventory/$NAME"
if [[ $MODE == completo ]]; then
  SCAN="$INDEX/pixmo-cap" EXTRA_SCAN=() EXTRA_VERIFY=()
else
  SCAN="$OUT/scansione" EXTRA_SCAN=(--files 1) EXTRA_VERIFY=(--partial)
fi
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
    "$SHARDS/plan.json" "$SHARDS/shards.jsonl" "$SHARDS/manifest.json" "$SNAPSHOT/README.md" "$RUN4/report.json" \
    "$RUN4/flagged.jsonl" "$VERIFY" "${HIST[0]}/exclude.txt" "${HIST[1]}/exclude.txt"; do
  test -s "$f" || { echo "ingresso mancante: $f" >&2; ledger_note failed "pixmo_inventory $MODE $NAME: manca $f"; exit 1; }
done
for d in "${HIST[@]}"; do
  test -f "$d/exclude_final.txt" || { echo "ingresso mancante: $d/exclude_final.txt" >&2; exit 1; }
done
[[ ! -e $OUT && ! -e $SCAN ]] || { echo "uscite già presenti: $OUT o $SCAN" >&2; exit 1; }
for t in tests/test_pixmo_inventory.py tests/test_transcripts.py; do
  if ! srun "$RUN_PYTHON" -B "$t"; then
    ledger_note failed "pixmo_inventory $MODE $NAME: $t fallito, nessuna lettura dei dati"
    exit 1
  fi
done
common=(--snapshot "$SNAPSHOT" --shards "$SHARDS" --revision "$REVISION" --index "$INDEX"
        --index-sha256 "$INDEX_SHA256" --inventory "$OUT/inventario" --scan "$SCAN")
if ! srun "$RUN_PYTHON" -u -B scripts/pixmo_inventory.py scan "${common[@]}" ${EXTRA_SCAN[@]+"${EXTRA_SCAN[@]}"}; then
  ledger_note failed "pixmo_inventory $MODE $NAME: scansione -> $OUT/inventario e $SCAN"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/pixmo_inventory.py verifica "${common[@]}" --panel "$PANEL" --run4 "$RUN4" \
    --historical "${HIST[@]}" --verify-report "$VERIFY" --report "$OUT/verifica" \
    ${EXTRA_VERIFY[@]+"${EXTRA_VERIFY[@]}"}; then
  ledger_note ok "pixmo_inventory $MODE $NAME: inventario -> $OUT/inventario, scansione -> $SCAN, verifica -> $OUT/verifica/report.json"
else
  ledger_note failed "pixmo_inventory $MODE $NAME: verifica -> $OUT/verifica/report.json"
  exit 1
fi
