#!/usr/bin/env bash
# Scorciatoia di function/exists nei sintetici v4, politica delle domande e synth-v4-candidato-3
# (scripts/synth_qa_policy.py, scripts/manifest_check.py; docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §2.14). Job solo
# CPU, offline; manifest, shard, revisione e campione in sola lettura; candidato-2 non cambia.
# Uso:
#   sbatch slurm/synth_qa_policy.sbatch misura NOME
#     test mirati (test_synth_policy, test_review_results, test_manifest, test_reader_manifest, test_synth), poi
#     impatto su synth-v4-candidato-2 -> outputs/synth_qa_policy/NOME/impatto (riepilogo
#     docs/inventari/synth-v4-scorciatoia-exists-NOME.json) e diagnostica v5 -> outputs/synth_qa_policy/NOME/diagnostica
#     (riepilogo docs/inventari/synth-v5-diagnostica-NOME.json)
#   sbatch slurm/synth_qa_policy.sbatch deriva NOME INTEGRAZIONE [accetta-riesame]
#     data/manifests/synth-v4-candidato-3 da candidato-2 con la proposta di integrazione-2 (blob di 5038c66), il riesame
#     tecnico (cfb9651f...) e il rapporto outputs/review_results/INTEGRAZIONE/report.json; impatto da
#     outputs/synth_qa_policy/NOME/impatto
#   sbatch slurm/synth_qa_policy.sbatch verifica NOME CONFIG
#     manifest_check.py su synth-v4-candidato-3 con le sorgenti della config di collaudo
#     -> outputs/manifest_check/synth-v4-candidato-3-NOME (riepilogo docs/manifests/synth-v4-candidato-3-NOME.json)
#   sbatch slurm/synth_qa_policy.sbatch verifica-precedente NOME
#     manifest_check.py col codice nuovo su synth-v4-candidato-2 (senza politica: lettura per record come prima)
#     -> outputs/manifest_check/synth-v4-candidato-2-NOME
#SBATCH --job-name=lvl-synth-qa-policy
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
MODE=${1:-} NAME=${2:-}
[[ $MODE =~ ^(misura|deriva|verifica|verifica-precedente)$ && $NAME =~ ^[A-Za-z0-9_-]+$ ]] \
  || { echo "uso: synth_qa_policy.sbatch misura|deriva|verifica|verifica-precedente NOME [...]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
echo "=== file modificati o nuovi rispetto a HEAD: id git del contenuto eseguito"
git ls-files -m -o --exclude-standard | while IFS= read -r f; do
  if [[ -f $f ]]; then printf '%s  %s\n' "$(git hash-object -- "$f")" "$f"; fi
done
M2="$DATA_ROOT/manifests/synth-v4-candidato-2"
M2_SHA=9a8104cdea4b56ef333e7d1e9629d7cfd638ccd1a6be18336137c717ae2d824e
M3="$DATA_ROOT/manifests/synth-v4-candidato-3"
OUT="$OUTPUT_ROOT/synth_qa_policy/$NAME"
PROPOSTA=docs/revisioni/synth-v4-campione-2-integrazione-2-proposta.tsv
RIESAME=docs/revisioni/synth-v4-campione-2-riesame-incerti.tsv
RIESAME_SHA=cfb9651faaa883c4cbca149ba2b155da829c4c59080786c7d52c711bb7d83217
if [[ -e $M3/manifest.json ]]; then  # identità del candidato-3 nel log, per la config di collaudo
  echo "=== synth-v4-candidato-3: $(sha256sum "$M3/manifest.json")"
  echo "=== synth-v4-candidato-3: $(sha256sum "$M3/records.jsonl.gz")"
fi

case $MODE in
misura)
  [[ ! -e $OUT/impatto && ! -e $OUT/diagnostica ]] || { echo "uscite già presenti per $NAME" >&2; exit 1; }
  for t in tests/test_synth_policy.py tests/test_review_results.py tests/test_manifest.py tests/test_reader_manifest.py \
           tests/test_synth.py; do
    echo "=== $t"
    if ! srun "$RUN_PYTHON" -B "$t"; then
      ledger_note failed "synth_qa_policy misura $NAME: $t fallito, nessuna lettura dei dati reali"
      exit 1
    fi
  done
  mkdir -p "$OUT" docs/inventari
  srun "$RUN_PYTHON" -u -B scripts/synth_qa_policy.py impatto --manifest "$M2" --manifest-sha256 "$M2_SHA" \
    --campione "$OUTPUT_ROOT/synth_gold_review/campione-2" --out "$OUT/impatto" --workers 4 \
    --docs "docs/inventari/synth-v4-scorciatoia-exists-$NAME.json" \
    || { ledger_note failed "synth_qa_policy misura $NAME: impatto fallito -> $OUT/impatto"; exit 1; }
  srun "$RUN_PYTHON" -u -B scripts/synth_qa_policy.py diagnostica --out "$OUT/diagnostica" \
    --docs "docs/inventari/synth-v5-diagnostica-$NAME.json" \
    || { ledger_note failed "synth_qa_policy misura $NAME: diagnostica v5 fallita -> $OUT/diagnostica"; exit 1; }
  ledger_note ok "synth_qa_policy misura $NAME -> $OUT/impatto/report.json, $OUT/diagnostica/report.json"
  ;;
deriva)
  INTEG=${3:-} ACCEPT=${4:-}
  [[ $INTEG =~ ^[A-Za-z0-9_-]+$ && -s $OUTPUT_ROOT/review_results/$INTEG/report.json ]] \
    || { echo "rapporto di integrazione mancante: $INTEG" >&2; exit 2; }
  [[ -z $ACCEPT || $ACCEPT == accetta-riesame ]] || { echo "quarto argomento: accetta-riesame o niente" >&2; exit 2; }
  [[ ! -e $M3 ]] || { echo "$M3 esiste già" >&2; exit 1; }
  for f in "$PROPOSTA" "$RIESAME"; do  # gli ingressi versionati: uguali ai blob del commit che li ha registrati
    want=$(git rev-parse "5038c66:$f") got=$(git hash-object -- "$f")
    [[ $got == "$want" ]] || { ledger_note failed "synth_qa_policy deriva $NAME: $f diverso da 5038c66"; exit 1; }
  done
  EXTRA=()
  [[ -n $ACCEPT ]] && EXTRA=(--accetta-riesame)
  srun "$RUN_PYTHON" -B tests/test_synth_policy.py \
    || { ledger_note failed "synth_qa_policy deriva $NAME: tests/test_synth_policy.py fallito, nessuna derivazione"; exit 1; }
  srun "$RUN_PYTHON" -u -B scripts/synth_qa_policy.py deriva --parent "$M2" "$M2_SHA" --proposta "$PROPOSTA" \
    --riesame "$RIESAME" "$RIESAME_SHA" --integrazione "$OUTPUT_ROOT/review_results/$INTEG/report.json" \
    --impatto "$OUT/impatto/report.json" "${EXTRA[@]}" --out "$M3" \
    || { ledger_note failed "synth_qa_policy deriva $NAME: derivazione fallita"; exit 1; }
  ledger_note ok "synth_qa_policy deriva $NAME -> $M3 (manifest.json $(sha256sum "$M3/manifest.json" | cut -c1-12)...)"
  ;;
verifica)
  CONFIG=${3:-}
  [[ -s $CONFIG ]] || { echo "config mancante: $CONFIG" >&2; exit 2; }
  CHECK="$OUTPUT_ROOT/manifest_check/synth-v4-candidato-3-$NAME"
  [[ ! -e $CHECK ]] || { echo "$CHECK esiste già" >&2; exit 1; }
  if srun "$RUN_PYTHON" -u -B scripts/manifest_check.py --manifest "$M3" --shards "$DATA_ROOT/shards/synth/v4" \
      --out "$CHECK" --workers 2 --config "$CONFIG"; then
    mkdir -p docs/manifests
    cp "$CHECK/report.json" "docs/manifests/synth-v4-candidato-3-$NAME.json"
    ledger_note ok "synth_qa_policy verifica $NAME -> $CHECK/report.json, docs/manifests/synth-v4-candidato-3-$NAME.json"
  else
    ledger_note failed "synth_qa_policy verifica $NAME: controlli falliti -> $CHECK"
    exit 1
  fi
  ;;
verifica-precedente)
  CHECK="$OUTPUT_ROOT/manifest_check/synth-v4-candidato-2-$NAME"
  [[ ! -e $CHECK ]] || { echo "$CHECK esiste già" >&2; exit 1; }
  if srun "$RUN_PYTHON" -u -B scripts/manifest_check.py --manifest "$M2" --shards "$DATA_ROOT/shards/synth/v4" \
      --out "$CHECK" --workers 2; then
    ledger_note ok "synth_qa_policy verifica-precedente $NAME -> $CHECK/report.json"
  else
    ledger_note failed "synth_qa_policy verifica-precedente $NAME: controlli falliti -> $CHECK"
    exit 1
  fi
  ;;
esac
