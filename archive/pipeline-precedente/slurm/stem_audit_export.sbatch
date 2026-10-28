#!/usr/bin/env bash
# Audit automatico STEM di MMFineReason (scripts/stem_audit_export.py): controlli deterministici, campione
# stratificato, calibrazione e lotti per gli agenti. Job solo CPU.
# Uso: sbatch slurm/stem_audit_export.sbatch NOME [opzioni di stem_audit_export.py, es. --n 300 --seed 20261019 --f2-n 100]
#SBATCH --job-name=lvl-stem-audit
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 1 )) || { echo "uso: stem_audit_export.sbatch NOME [opzioni]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
NAME=$1
shift
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
SNAP="$HF_HOME/hub/datasets--OpenDataArena--MMFineReason-1.8M/snapshots/7c93f724791d01246f5425e1182b14ec8d013847"
MAVIS="$DATA_ROOT/shards/mavis-geometry/9c80fff3-caption/mavis-geometry-caption-00000.tar"
SYNTH="$DATA_ROOT/shards/synth/v3/test/synth-test.tar"
F2SEL="$OUTPUT_ROOT/f2_select/f2-sft120k-v4/selected.jsonl"
for f in "$MAVIS" "$SYNTH" "$F2SEL" "$DATA_ROOT/decontam/p6/mmfinereason/exclude_final.txt" "$DATA_ROOT/decontam/p6/mmfinereason/exclude.txt" \
         "$DATA_ROOT/lang/mmfinereason/question_ja_ko.txt" "$DATA_ROOT/lang/mmfinereason/caption_cjk.txt" "$DATA_ROOT/lang/mmfinereason/question_non_english.txt"; do
  test -f "$f" || { echo "file mancante: $f" >&2; exit 1; }
done
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
out="$OUTPUT_ROOT/stem_audit/$NAME"
if srun "$RUN_PYTHON" -u -B scripts/stem_audit_export.py --out "$out" --parquet-glob "$SNAP/data/sft-*.parquet" \
    --exclude-lists "$DATA_ROOT/decontam/p6/mmfinereason/exclude_final.txt" "$DATA_ROOT/lang/mmfinereason/question_ja_ko.txt" \
    --tag-lists "$DATA_ROOT/lang/mmfinereason/caption_cjk.txt" "$DATA_ROOT/lang/mmfinereason/question_non_english.txt" \
    --contam-candidates "$DATA_ROOT/decontam/p6/mmfinereason/exclude.txt" --mavis-shard "$MAVIS" --synth-test "$SYNTH" \
    --f2-selected "$F2SEL" "$@"; then
  ledger_note ok "audit STEM, esportazione $NAME -> $out"
else
  ledger_note failed "audit STEM, esportazione $NAME"
  exit 1
fi
