#!/usr/bin/env bash
# Pacchetto portatile per la revisione umana offline dei fogli di un campione delle golden answer dei sintetici v4
# (scripts/review_package.py; docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §2.12). Job solo CPU; campione, manifest e
# shard in sola lettura; nessun dato rigenerato. La cartella revisioni/NOME deve contenere già l'interfaccia (index.html,
# app.js, logica.js, stile.css, verifica.html, test.js, LEGGIMI.md, risultati/LEGGIMI.md) e niente dati.
# Uso: sbatch slurm/review_package.sbatch CAMPIONE NOME
#   CAMPIONE: cartella in outputs/synth_gold_review (es. campione-2); NOME: cartella in revisioni (es. synth-v4-campione-2)
#   -> revisioni/NOME/{dati,immagini,moduli,pacchetto.json,SHA256SUMS,verifica_pacchetto.json}
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
(( $# == 2 )) || { echo "uso: review_package.sbatch CAMPIONE NOME" >&2; exit 2; }
SAMPLE=$1 NAME=$2
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
if ! srun "$RUN_PYTHON" -u -B scripts/review_package.py build --campione "$SRC" --manifest "$MANIFEST" --out "$PKG"; then
  ledger_note failed "review_package $NAME: costruzione da $SRC"
  exit 1
fi
if srun "$RUN_PYTHON" -u -B scripts/review_package.py verifica --campione "$SRC" --pacchetto "$PKG" --manifest "$MANIFEST" \
    --report "$PKG/verifica_pacchetto.json"; then
  ledger_note ok "review_package $NAME: pacchetto da $SRC -> $PKG"
else
  ledger_note failed "review_package $NAME: verifica -> $PKG/verifica_pacchetto.json"
  exit 1
fi
