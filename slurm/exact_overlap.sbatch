#!/usr/bin/env bash
# Primo sottoinsieme di C6: identità esatte delle immagini fra sintetici v4, UniMER, MAVIS e PixMo-Cap
# (scripts/exact_overlap.py) dai manifest candidati e dalle scansioni già fatte, senza leggere immagini, shard, zip o
# parquet. Job solo CPU: ingressi in sola lettura, uscite in cartelle nuove. Prima il test mirato: se fallisce non si
# legge nulla. Diagnostica: nessun manifest, gruppo, split o stato cambia, nessuna lista per i loader.
# Uso:
#   sbatch slurm/exact_overlap.sbatch prova NOME     prime 20.000 righe di ogni records.jsonl.gz (sha256 completi),
#                                                     misure e stima -> outputs/exact_overlap/NOME
#   sbatch slurm/exact_overlap.sbatch completo NOME  confronto completo -> outputs/exact_overlap/NOME, riepilogo per
#                                                     git docs/indice_globale/identita-esatte-NOME.json
#SBATCH --job-name=lvl-exact-overlap
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
usage() { echo "uso: exact_overlap.sbatch prova NOME | completo NOME" >&2; exit 2; }
(( $# == 2 )) || usage
MODE=$1 NAME=$2
[[ ($MODE == prova || $MODE == completo) && $NAME =~ ^[A-Za-z0-9._-]+$ ]] || usage
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
M="$DATA_ROOT/manifests"
# manifest di riferimento con lo sha256 di manifest.json (S2 §2.9, config di collaudo di C7, docs/manifests)
SOURCES=(--manifest synth "$M/synth-v4-candidato-2" 9a8104cdea4b56ef333e7d1e9629d7cfd638ccd1a6be18336137c717ae2d824e
         --manifest unimer "$M/unimer-candidato-2" 97af10332520c01717b80ae2e7b094f87cb5e435a6d03b0ead656175b837c126
         --manifest mavis "$M/mavis-geometry-candidato-3" f3cca47367799ccd8a7c8f4d6b63c5c37e0cd06e22d921daa5507c6aa442b9c6
         --manifest pixmo "$M/pixmo-cap-candidato-1" 794787abffd51a9ddc7f12227d7ac6a2eb3a922bc3c4f5fac496cb96a7b9d850)
# MAVIS candidato-5 (revisione C9): solo manifest.json, per la catena; non è una sorgente in più
C9=(--c9 "$M/mavis-geometry-candidato-5" 314a0a527e20ce422399765f7db19c3abfe297ec5e174e7f3b158586c604ef72)
OUT="$OUTPUT_ROOT/exact_overlap/$NAME"
SUMMARY="$PROJECT_ROOT/docs/indice_globale/identita-esatte-$NAME.json"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
echo "=== file modificati o nuovi rispetto a HEAD: id git del contenuto eseguito"
git ls-files -m -o --exclude-standard | while IFS= read -r f; do
  if [[ -f $f ]]; then printf '%s  %s\n' "$(git hash-object -- "$f")" "$f"; fi
done
export OMP_NUM_THREADS=1
[[ ! -e $OUT ]] || { echo "$OUT esiste già" >&2; exit 1; }
extra=(--limit 20000) done_msg="$OUT/report.json"
if [[ $MODE == completo ]]; then
  [[ ! -e $SUMMARY ]] || { echo "$SUMMARY esiste già" >&2; exit 1; }
  mkdir -p "$(dirname "$SUMMARY")"
  extra=(--summary "$SUMMARY") done_msg="$OUT/report.json, riepilogo $SUMMARY"
fi
mkdir -p "$(dirname "$OUT")"
if ! srun "$RUN_PYTHON" -B tests/test_exact_overlap.py; then
  ledger_note failed "exact_overlap $MODE $NAME: tests/test_exact_overlap.py fallito, nessuna lettura dei dati reali"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/exact_overlap.py "${SOURCES[@]}" "${C9[@]}" --out "$OUT" "${extra[@]}"; then
  ledger_note ok "exact_overlap $MODE $NAME -> $done_msg"
else
  ledger_note failed "exact_overlap $MODE $NAME: controlli falliti o arresto -> $OUT"
  exit 1
fi
