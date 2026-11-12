#!/usr/bin/env bash
# Integrazione verificata degli esiti della revisione umana di synth-v4-campione-2 (scripts/review_results.py;
# docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §2.13). Job solo CPU: esiti caricati, pacchetto, campione-2, manifest
# synth-v4-candidato-2, indice e gruppi di pannello-v3 in sola lettura; nessun manifest, config o ruolo cambiato.
# Prima: i tre file di risultati uguali ai blob del commit che li ha caricati (777e30c) e pacchetto senza modifiche
# locali; poi il test mirato; se uno dei due fallisce non si legge nulla.
# Uso: sbatch slurm/review_results.sbatch NOME [RIESAME [PRECEDENTE]]
#   RIESAME: TSV del riesame tecnico dei candidati incerti (unita, record, conclusione, motivazione), facoltativo
#   PRECEDENTE: nome di un run precedente (outputs/review_results/PRECEDENTE): materiale del riesame e proposta si
#     confrontano con i suoi; con la proposta uguale il TSV per git resta quello del run precedente
#   Gli ingressi del riesame (indice, gruppi del pannello, file dei benchmark) si verificano sempre contro le identità
#   registrate (manifest e downloads/manifests) prima di usarli.
#   -> outputs/review_results/NOME/{report.json, verifica_pacchetto.json, esiti_per_unita.tsv, proposta_ammissioni.tsv}
#      e, per git, docs/revisioni/synth-v4-campione-2-NOME.json e docs/revisioni/synth-v4-campione-2-NOME-proposta.tsv
#SBATCH --job-name=lvl-review-results
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 1 && $# <= 3 )) && [[ $1 =~ ^[A-Za-z0-9_-]+$ ]] || { echo "uso: review_results.sbatch NOME [RIESAME [PRECEDENTE]]" >&2; exit 2; }
NAME=$1 TECH=${2:-} PREV=${3:-}
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
PKG="$PROJECT_ROOT/revisioni/synth-v4-campione-2"
RES="revisioni/synth-v4-campione-2/risultati"
UPLOAD=777e30c  # commit dell'utente con l'ultimo file caricato (201839b i TSV, 777e30c il JSON)
FILES=("$RES/synth-v4-campione-2_avanzamento_2026-11-11T20-12-26.json"
       "$RES/esiti_2026-11-11T20-12-29.tsv"
       "$RES/esiti_decontaminazione_2026-11-11T20-12-30.tsv")
SHAS=(4a88d60cb1111ea6a7b4027f58f70865306a4245e4da3398d7c14036d8035b7f
      57ec0275bf2931c8021cd50b25a6473c4728d06b800c2e68fe5ff229679f8cb5
      a694cfad5114f317e38598d3f5f21f71bf00e9d3cb9473620d2fe047e606438b)
OUT="$OUTPUT_ROOT/review_results/$NAME"
DOCS="$PROJECT_ROOT/docs/revisioni/synth-v4-campione-2-$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
echo "=== file modificati o nuovi rispetto a HEAD: id git del contenuto eseguito"
git ls-files -m -o --exclude-standard | while IFS= read -r f; do
  if [[ -f $f ]]; then printf '%s  %s\n' "$(git hash-object -- "$f")" "$f"; fi
done
echo "=== esiti caricati: blob uguali a quelli di $UPLOAD, pacchetto senza modifiche locali"
for f in "${FILES[@]}"; do
  want=$(git rev-parse "$UPLOAD:$f") got=$(git hash-object -- "$f")
  echo "$got  $f (atteso $want)"
  [[ $got == "$want" ]] || { ledger_note failed "review_results $NAME: $f diverso dal blob di $UPLOAD"; exit 1; }
done
if [[ -n $(git status --porcelain -- revisioni/synth-v4-campione-2) ]]; then
  git status --porcelain -- revisioni/synth-v4-campione-2
  ledger_note failed "review_results $NAME: il pacchetto revisioni/synth-v4-campione-2 ha modifiche locali"
  exit 1
fi
EXTRA=()
if [[ -n $TECH ]]; then
  test -s "$TECH" || { echo "riesame mancante: $TECH" >&2; exit 2; }
  EXTRA=(--riesame "$TECH")
fi
if [[ -n $PREV ]]; then
  [[ $PREV =~ ^[A-Za-z0-9_-]+$ && -s $OUTPUT_ROOT/review_results/$PREV/report.json ]] \
    || { echo "run precedente mancante: $PREV" >&2; exit 2; }
  EXTRA+=(--precedente "$PREV")
fi
[[ ! -e $OUT && ! -e $DOCS.json && ! -e $DOCS-proposta.tsv ]] || { echo "uscite già presenti per $NAME" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")" "$(dirname "$DOCS")"
export OMP_NUM_THREADS=1
if ! srun "$RUN_PYTHON" -B tests/test_review_results.py; then
  ledger_note failed "review_results $NAME: tests/test_review_results.py fallito, nessuna lettura dei dati reali"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/review_results.py --pacchetto "$PKG" \
    --avanzamento "${FILES[0]}" "${SHAS[0]}" --esiti "${FILES[1]}" "${SHAS[1]}" \
    --esiti-decontaminazione "${FILES[2]}" "${SHAS[2]}" \
    --campione "$OUTPUT_ROOT/synth_gold_review/campione-2" --manifest "$DATA_ROOT/manifests/synth-v4-candidato-2" \
    --indice "$DATA_ROOT/decontam/pannello-v3" --pannello "$DATA_ROOT/panel/pannello-v3" "${EXTRA[@]}" \
    --out "$OUT" --docs "$DOCS"; then
  ledger_note ok "review_results $NAME -> $OUT/report.json, riepilogo $DOCS.json$([[ -e $DOCS-proposta.tsv ]] && echo " e $DOCS-proposta.tsv")"
else
  ledger_note failed "review_results $NAME: controlli falliti o arresto -> $OUT"
  exit 1
fi
if [[ -n $(git status --porcelain -- revisioni/synth-v4-campione-2) ]]; then
  git status --porcelain -- revisioni/synth-v4-campione-2
  ledger_note failed "review_results $NAME: il pacchetto è cambiato durante il job"
  exit 1
fi
