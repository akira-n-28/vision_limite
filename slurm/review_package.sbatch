#!/usr/bin/env bash
# Pacchetto portatile per la revisione umana offline dei fogli di un campione delle golden answer dei sintetici v4
# (scripts/review_package.py; docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §2.12). Job solo CPU; campione, manifest e
# shard in sola lettura; nessun dato rigenerato. La cartella revisioni/NOME deve contenere già l'interfaccia (index.html,
# app.js, logica.js, stile.css, verifica.html, test.js, LEGGIMI.md, risultati/LEGGIMI.md).
# Uso: sbatch slurm/review_package.sbatch CAMPIONE NOME [verifica]
#   CAMPIONE: cartella in outputs/synth_gold_review (es. campione-2); NOME: cartella in revisioni (es. synth-v4-campione-2)
#   senza "verifica": costruzione -> revisioni/NOME/{dati,immagini,moduli,pacchetto.json,SHA256SUMS}, poi le verifiche;
#   con "verifica": solo le verifiche di un pacchetto già costruito.
#   Verifica statica: scripts/review_package.py verifica -> revisioni/NOME/verifica_pacchetto.json. Non esegue il
#   JavaScript: su lagrange0 non c'è un motore JS (job 30558) e non è una prova in un browser; l'interfaccia si collauda
#   aprendo revisioni/NOME/verifica.html.
#SBATCH --job-name=lvl-review-package
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 2 || $# == 3 )) || { echo "uso: review_package.sbatch CAMPIONE NOME [verifica]" >&2; exit 2; }
SAMPLE=$1 NAME=$2 MODE=${3:-costruzione}
[[ $MODE == costruzione || $MODE == verifica ]] || { echo "terzo argomento: verifica" >&2; exit 2; }
for n in "$SAMPLE" "$NAME"; do
  [[ $n =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $n" >&2; exit 2; }
done
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SRC="$OUTPUT_ROOT/synth_gold_review/$SAMPLE"
MANIFEST="$DATA_ROOT/manifests/synth-v4-candidato-2"
PKG="$PROJECT_ROOT/revisioni/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
for f in "$SRC/report.json" "$MANIFEST/manifest.json" "$PKG/index.html"; do
  test -s "$f" || { echo "ingresso mancante: $f" >&2; ledger_note failed "review_package $NAME: manca $f"; exit 1; }
done
if [[ $MODE == costruzione ]]; then
  if ! srun "$RUN_PYTHON" -u -B scripts/review_package.py build --campione "$SRC" --manifest "$MANIFEST" --out "$PKG"; then
    ledger_note failed "review_package $NAME: costruzione da $SRC"
    exit 1
  fi
fi
if srun "$RUN_PYTHON" -u -B scripts/review_package.py verifica --campione "$SRC" --pacchetto "$PKG" --manifest "$MANIFEST" \
    --report "$PKG/verifica_pacchetto.json"; then
  ledger_note ok "review_package $NAME ($MODE): pacchetto da $SRC -> $PKG, verifica statica superata"
else
  ledger_note failed "review_package $NAME ($MODE): verifica -> $PKG/verifica_pacchetto.json"
  exit 1
fi
