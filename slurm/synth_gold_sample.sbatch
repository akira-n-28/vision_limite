#!/usr/bin/env bash
# Preparazione della verifica delle golden answer dei sintetici v4 e dei candidati della decontaminazione esterna
# (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §§2.9-2.11). Job solo CPU; manifest, shard, indice e scansione in sola
# lettura; nessun dato rigenerato, nessuna revisione avviata.
# Uso:
#   sbatch slurm/synth_gold_sample.sbatch NOME
#     tests/test_synth_gold.py, poi somiglianza dei pixel dei candidati (scripts/decontam_review.py) in
#     outputs/decontam_review/NOME/synth-v4 e campione, casi mirati, candidati e inventario (scripts/synth_gold_sample.py)
#     in outputs/synth_gold_review/NOME (così il job 30522, campione-1)
#   sbatch slurm/synth_gold_sample.sbatch NOME REVISIONE PRECEDENTE
#     fogli nuovi dagli artefatti esistenti: tests/test_synth_gold.py e tests/test_synth_decontam.py; verifiche del
#     manifest col validatore attuale (scripts/manifest_check.py, sola lettura) in
#     outputs/manifest_check/synth-v4-candidato-2-NOME; la somiglianza dei pixel già calcolata in
#     outputs/decontam_review/REVISIONE/synth-v4, senza ricalcolarla; confronto con outputs/synth_gold_review/PRECEDENTE
#     (stessa estrazione: il job esce con errore se la selezione cambia)
#SBATCH --job-name=lvl-synth-gold
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
(( $# == 1 || $# == 3 )) || { echo "uso: synth_gold_sample.sbatch NOME [REVISIONE PRECEDENTE]" >&2; exit 2; }
NAME=$1 REVIEW_NAME=${2:-} PREVIOUS_NAME=${3:-}
for n in "$NAME" ${REVIEW_NAME:+"$REVIEW_NAME"} ${PREVIOUS_NAME:+"$PREVIOUS_NAME"}; do
  [[ $n =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $n" >&2; exit 2; }
done
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
MANIFEST="$DATA_ROOT/manifests/synth-v4-candidato-2"
SHARDS="$DATA_ROOT/shards/synth/v4"
INDEX="$DATA_ROOT/decontam/pannello-v3"
SCAN="$INDEX/synth-v4"
OUT="$OUTPUT_ROOT/synth_gold_review/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
for f in "$MANIFEST/manifest.json" "$INDEX/index.json" "$SCAN/summary.json"; do
  test -s "$f" || { echo "ingresso mancante: $f" >&2; ledger_note failed "synth_gold_sample $NAME: manca $f"; exit 1; }
done
[[ ! -e $OUT ]] || { echo "uscita già presente: $OUT" >&2; exit 1; }
tests=(tests/test_synth_gold.py)
[[ -n $REVIEW_NAME ]] && tests+=(tests/test_synth_decontam.py)
for t in "${tests[@]}"; do
  echo "=== $t"
  if ! srun "$RUN_PYTHON" -B "$t"; then
    ledger_note failed "synth_gold_sample $NAME: $t fallito"
    exit 1
  fi
done
extra=()
if [[ -z $REVIEW_NAME ]]; then
  REVIEW="$OUTPUT_ROOT/decontam_review/$NAME"
  [[ ! -e $REVIEW ]] || { echo "uscita già presente: $REVIEW" >&2; exit 1; }
  if ! srun "$RUN_PYTHON" -u -B scripts/decontam_review.py --index "$INDEX/index.json" --scan "$SCAN" --out "$REVIEW" \
      --examples 25 --workers 8; then
    ledger_note failed "synth_gold_sample $NAME: decontam_review -> $REVIEW"
    exit 1
  fi
else
  REVIEW="$OUTPUT_ROOT/decontam_review/$REVIEW_NAME"
  PREVIOUS="$OUTPUT_ROOT/synth_gold_review/$PREVIOUS_NAME"
  CHECK="$OUTPUT_ROOT/manifest_check/synth-v4-candidato-2-$NAME"
  for f in "$REVIEW/synth-v4/review.jsonl" "$PREVIOUS/report.json"; do
    test -s "$f" || { echo "ingresso mancante: $f" >&2; ledger_note failed "synth_gold_sample $NAME: manca $f"; exit 1; }
  done
  [[ ! -e $CHECK ]] || { echo "uscita già presente: $CHECK" >&2; exit 1; }
  if ! srun "$RUN_PYTHON" -u -B scripts/manifest_check.py --manifest "$MANIFEST" --shards "$SHARDS" --out "$CHECK"; then
    ledger_note failed "synth_gold_sample $NAME: verifiche del manifest -> $CHECK"
    exit 1
  fi
  extra=(--previous "$PREVIOUS")
fi
if srun "$RUN_PYTHON" -u -B scripts/synth_gold_sample.py --manifest "$MANIFEST" --index "$INDEX" \
    --review "$REVIEW/synth-v4" --out "$OUT" --workers 8 "${extra[@]}"; then
  ledger_note ok "synth_gold_sample $NAME: revisione dei candidati -> $REVIEW, campione -> $OUT"
else
  ledger_note failed "synth_gold_sample $NAME: campione -> $OUT"
  exit 1
fi
