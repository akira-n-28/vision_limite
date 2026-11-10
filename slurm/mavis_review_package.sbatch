#!/usr/bin/env bash
# Pacchetto offline per la revisione visiva mirata di MAVIS-Geometry dopo C9 (scripts/mavis_review_package.py; S2 §3).
# Job solo CPU; manifest mavis-geometry-candidato-5, scansione della decontaminazione, shard e zip dello snapshot in sola
# lettura; nessuna chiamata a modelli. La cartella revisioni/mavis-c9-campione-1 deve contenere già l'interfaccia
# (index.html, app.js, logica.js, stile.css, verifica.html, test.js, LEGGIMI.md, risultati/LEGGIMI.md).
# Uso: sbatch slurm/mavis_review_package.sbatch [costruzione | verifica | firme]
#   costruzione (predefinito): tests/test_mavis_review_package.py, poi selezione e schede -> dati/, immagini/, moduli/,
#     pacchetto.json, SHA256SUMS; poi le verifiche;
#   verifica: solo le verifiche di un pacchetto già costruito;
#   firme: riscrive SHA256SUMS dopo una correzione dell'interfaccia (dati e immagini invariati), poi le verifiche.
#   Verifiche: mavis_review_package.py verifica (selezione e schede ricalcolate dal manifest e dagli shard, condizioni
#   degli strati, quote e deficit, immagini e hash, primo passaggio, SHA256SUMS, riferimenti) -> verifica_pacchetto.json;
#   esegui-js con Duktape senza DOM e sul DOM simulato -> verifica_js.json. Nessuna delle due è una prova in un browser.
#SBATCH --job-name=lvl-mavis-review
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
MODE=${1:-costruzione}
[[ $MODE == costruzione || $MODE == verifica || $MODE == firme ]] || {
  echo "uso: mavis_review_package.sbatch [costruzione | verifica | firme]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
MANIFEST="$DATA_ROOT/manifests/mavis-geometry-candidato-5"
MANIFEST_SHA=314a0a527e20ce422399765f7db19c3abfe297ec5e174e7f3b158586c604ef72
PKG="$PROJECT_ROOT/revisioni/mavis-c9-campione-1"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
echo "=== file modificati o nuovi rispetto a HEAD (codice e interfaccia): id git del contenuto"
git ls-files -m -o --exclude-standard -- scripts src slurm tests revisioni/mavis-c9-campione-1/*.js \
    revisioni/mavis-c9-campione-1/*.html revisioni/mavis-c9-campione-1/*.css revisioni/mavis-c9-campione-1/*.md |
  while IFS= read -r f; do
    if [[ -f $f ]]; then printf '%s  %s\n' "$(git hash-object -- "$f")" "$f"; fi
  done
for f in "$MANIFEST/manifest.json" "$PKG/index.html" "$PKG/app.js" "$PKG/logica.js" "$PKG/test.js"; do
  test -s "$f" || { echo "ingresso mancante: $f" >&2; ledger_note failed "mavis_review_package: manca $f"; exit 1; }
done
PY=("$RUN_PYTHON" -u -B scripts/mavis_review_package.py)
if [[ $MODE == costruzione ]]; then
  if ! srun "$RUN_PYTHON" -B tests/test_mavis_review_package.py; then
    ledger_note failed "mavis_review_package: tests/test_mavis_review_package.py"
    exit 1
  fi
  if ! srun "${PY[@]}" costruisci --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" --out "$PKG"; then
    ledger_note failed "mavis_review_package costruzione: pacchetto non scritto (vedi il log)"
    exit 1
  fi
elif [[ $MODE == firme ]]; then
  srun "${PY[@]}" firme --pacchetto "$PKG"
fi
status=0
if srun "${PY[@]}" verifica --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" --pacchetto "$PKG" \
    --report "$PKG/verifica_pacchetto.json"; then
  ledger_note ok "mavis_review_package ($MODE): $PKG, verifica statica superata"
else
  ledger_note failed "mavis_review_package ($MODE): verifica statica -> $PKG/verifica_pacchetto.json"
  status=1
fi
# Esecuzione del JavaScript (dati, logica.js, app.js, test.js) con Duktape, libreria di sistema di lagrange0, senza DOM
# e su un DOM simulato: è l'esecuzione della logica e dell'interfaccia fuori da un browser, non una prova in un browser.
if srun "${PY[@]}" esegui-js --pacchetto "$PKG" --dom scripts/review_package_dom.js --report "$PKG/verifica_js.json"; then
  ledger_note ok "mavis_review_package ($MODE): JavaScript eseguito con Duktape senza DOM e sul DOM simulato, autotest superato"
else
  ledger_note failed "mavis_review_package ($MODE): esecuzione del JavaScript con Duktape -> $PKG/verifica_js.json"
  status=1
fi
exit $status
