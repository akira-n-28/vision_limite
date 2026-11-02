#!/usr/bin/env bash
# Preparazione della verifica delle golden answer dei sintetici v4 e dei candidati della decontaminazione esterna
# (docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §§2.9-2.10). Job solo CPU; manifest, shard, indice e scansione in sola
# lettura; nessun dato rigenerato, nessuna revisione avviata. Prima tests/test_synth_gold.py.
# Uso: sbatch slurm/synth_gold_sample.sbatch NOME
#   -> somiglianza dei pixel dei candidati (scripts/decontam_review.py) in outputs/decontam_review/NOME/synth-v4 e
#      campione, casi mirati, candidati e inventario (scripts/synth_gold_sample.py) in outputs/synth_gold_review/NOME
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
(( $# == 1 )) || { echo "uso: synth_gold_sample.sbatch NOME" >&2; exit 2; }
NAME=$1
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
MANIFEST="$DATA_ROOT/manifests/synth-v4-candidato-2"
INDEX="$DATA_ROOT/decontam/pannello-v3"
SCAN="$INDEX/synth-v4"
REVIEW="$OUTPUT_ROOT/decontam_review/$NAME"
OUT="$OUTPUT_ROOT/synth_gold_review/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
for f in "$MANIFEST/manifest.json" "$INDEX/index.json" "$SCAN/summary.json"; do
  test -s "$f" || { echo "ingresso mancante: $f" >&2; ledger_note failed "synth_gold_sample $NAME: manca $f"; exit 1; }
done
[[ ! -e $REVIEW && ! -e $OUT ]] || { echo "uscite già presenti: $REVIEW o $OUT" >&2; exit 1; }
if ! srun "$RUN_PYTHON" -B tests/test_synth_gold.py; then
  ledger_note failed "synth_gold_sample $NAME: tests/test_synth_gold.py fallito"
  exit 1
fi
if ! srun "$RUN_PYTHON" -u -B scripts/decontam_review.py --index "$INDEX/index.json" --scan "$SCAN" --out "$REVIEW" \
    --examples 25 --workers 8; then
  ledger_note failed "synth_gold_sample $NAME: decontam_review -> $REVIEW"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/synth_gold_sample.py --manifest "$MANIFEST" --index "$INDEX" \
    --review "$REVIEW/synth-v4" --out "$OUT"; then
  ledger_note ok "synth_gold_sample $NAME: revisione dei candidati -> $REVIEW, campione -> $OUT"
else
  ledger_note failed "synth_gold_sample $NAME: campione -> $OUT"
  exit 1
fi
