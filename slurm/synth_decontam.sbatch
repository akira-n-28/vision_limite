#!/usr/bin/env bash
# Decontaminazione esterna dei sintetici v4 contro pannello-v3 (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §2.9). Job
# solo CPU; indice, pannello, manifest e shard in sola lettura. Prima i test mirati: se falliscono non si scansiona.
# Uso:
#   sbatch slurm/synth_decontam.sbatch controllo NOME
#     test mirati (pannello, decontaminazione, sintetici), scansione limitata agli shard di test di v4 e verifica di
#     compatibilità con --partial -> outputs/synth_decontam/NOME
#   sbatch slurm/synth_decontam.sbatch completo NOME VERSIONE
#     scansione di tutti gli shard v4 -> data/decontam/pannello-v3/synth-v4, verifica completa
#     -> outputs/synth_decontam/NOME, manifest nuovo data/manifests/synth-v4-candidato-VERSIONE e sue verifiche
#     (scripts/manifest_check.py) -> outputs/manifest_check/synth-v4-candidato-VERSIONE
#SBATCH --job-name=lvl-synth-decontam
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
usage() { echo "uso: synth_decontam.sbatch controllo NOME | completo NOME VERSIONE" >&2; exit 2; }
(( $# >= 2 )) || usage
MODE=$1 NAME=$2 VERSION=${3:-}
[[ $MODE == controllo || ( $MODE == completo && -n $VERSION ) ]] || usage
[[ $NAME =~ ^[A-Za-z0-9._-]+$ && ( -z $VERSION || $VERSION =~ ^[0-9]+$ ) ]] || { echo "nome o versione non validi" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
PANEL_NAME=pannello-v3
INDEX="$DATA_ROOT/decontam/$PANEL_NAME"
INDEX_SHA256=02fb3cbfc36cbfd17bba92e4511c18a5760d330bc0145855fff2e7aa5130d127  # job 30469
PANEL="$DATA_ROOT/panel/$PANEL_NAME"
SHARDS="$DATA_ROOT/shards/synth/v4"
MANIFEST="$DATA_ROOT/manifests/synth-v4-candidato-1"
OUT="$OUTPUT_ROOT/synth_decontam/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
for f in "$INDEX/index.json" "$INDEX/verifica.json" "$PANEL/elements.jsonl.gz" "$PANEL/summary.json" \
    "$MANIFEST/manifest.json"; do
  test -s "$f" || { echo "ingresso mancante: $f" >&2; ledger_note failed "synth_decontam $MODE $NAME: manca $f"; exit 1; }
done
[[ ! -e $OUT ]] || { echo "$OUT esiste già" >&2; exit 1; }
tests=(tests/test_synth_decontam.py tests/test_manifest.py)
[[ $MODE == controllo ]] && tests=(tests/test_panel.py tests/test_decontam.py "${tests[@]}")
for t in "${tests[@]}"; do
  echo "=== $t"
  if ! srun "$RUN_PYTHON" -B "$t"; then
    ledger_note failed "synth_decontam $MODE $NAME: $t fallito, nessuna scansione"
    exit 1
  fi
done
common=(--manifest "$MANIFEST" --index "$INDEX" --panel "$PANEL" --index-sha256 "$INDEX_SHA256")
if [[ $MODE == controllo ]]; then
  if ! srun "$RUN_PYTHON" -u -B scripts/decontam.py scan --kind synth --index "$INDEX/index.json" \
      --shards "$SHARDS/test/*.tar" --out "$OUT/scan-test" --max-lines 200 --workers 4; then
    ledger_note failed "synth_decontam controllo $NAME: scansione limitata"
    exit 1
  fi
  if srun "$RUN_PYTHON" -u -B scripts/synth_decontam.py verifica "${common[@]}" --scan "$OUT/scan-test" \
      --out "$OUT/verifica" --partial; then
    ledger_note ok "synth_decontam controllo $NAME -> $OUT"
  else
    ledger_note failed "synth_decontam controllo $NAME: verifica di compatibilità -> $OUT/verifica"
    exit 1
  fi
  exit 0
fi
SCAN="$INDEX/synth-v4"
NEW="$DATA_ROOT/manifests/synth-v4-candidato-$VERSION"
CHECK="$OUTPUT_ROOT/manifest_check/synth-v4-candidato-$VERSION"
[[ ! -e $SCAN && ! -e $NEW && ! -e $CHECK ]] || { echo "uscite già presenti: $SCAN, $NEW o $CHECK" >&2; exit 1; }
if ! srun "$RUN_PYTHON" -u -B scripts/decontam.py scan --kind synth --index "$INDEX/index.json" \
    --shards "$SHARDS/*/*.tar" --out "$SCAN" --max-lines 200 --workers 4; then
  ledger_note failed "synth_decontam completo $NAME: scansione -> $SCAN"
  exit 1
fi
if ! srun "$RUN_PYTHON" -u -B scripts/synth_decontam.py verifica "${common[@]}" --scan "$SCAN" --out "$OUT/verifica"; then
  ledger_note failed "synth_decontam completo $NAME: verifica di compatibilità -> $OUT/verifica"
  exit 1
fi
if ! srun "$RUN_PYTHON" -u -B scripts/synth_decontam.py manifest "${common[@]}" --scan "$SCAN" --out "$NEW" \
    --version "$VERSION"; then
  ledger_note failed "synth_decontam completo $NAME: manifest -> $NEW"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/manifest_check.py --manifest "$NEW" --shards "$SHARDS" --out "$CHECK"; then
  ledger_note ok "synth_decontam completo $NAME: scansione -> $SCAN, manifest -> $NEW, verifiche -> $CHECK"
else
  ledger_note failed "synth_decontam completo $NAME: verifiche del manifest -> $CHECK"
  exit 1
fi
