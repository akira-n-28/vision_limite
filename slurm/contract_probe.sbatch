#!/usr/bin/env bash
# Sonda CPU del contratto di risposta (D3; scripts/contract_probe.py): tokenizer e template dello snapshot di Violetto,
# esempi scritti a mano, nessun peso e nessun dataset. Diagnostica: non modifica processor, loader, tokenizer, template
# dello snapshot o config; le alternative esistono solo nella sonda.
# Uso: sbatch slurm/contract_probe.sbatch NOME
#   -> outputs/contract_probe/NOME/report.json, riepilogo per git docs/contratto/NOME.json e docs/contratto/NOME.md
#SBATCH --job-name=lvl-contract-probe
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# == 1 )) && [[ $1 =~ ^[A-Za-z0-9._-]+$ ]] || { echo "uso: contract_probe.sbatch NOME" >&2; exit 2; }
NAME=$1
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
SNAPSHOT="$PROJECT_ROOT/hf/hub/models--paradigma-inc--limite-1b-violetto/snapshots/b1f3d572ccacb6919f4d64c321b70ba034ddaef2"
OUT="$OUTPUT_ROOT/contract_probe/$NAME"
DOCS="$PROJECT_ROOT/docs/contratto/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
test -d "$SNAPSHOT"
source slurm/prologue.sh
lagrange_prologue cpu
echo "=== file modificati o nuovi rispetto a HEAD: id git del contenuto eseguito"
git ls-files -m -o --exclude-standard | while IFS= read -r f; do
  if [[ -f $f ]]; then printf '%s  %s\n' "$(git hash-object -- "$f")" "$f"; fi
done
export OMP_NUM_THREADS=1
[[ ! -e $OUT && ! -e $DOCS.json && ! -e $DOCS.md ]] || { echo "uscite già presenti per $NAME" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")" "$(dirname "$DOCS")"
if srun "$RUN_PYTHON" -u -B scripts/contract_probe.py --snapshot "$SNAPSHOT" --out "$OUT" --docs "$DOCS"; then
  ledger_note ok "contract_probe $NAME -> $OUT/report.json, riepilogo $DOCS.json e $DOCS.md"
else
  ledger_note failed "contract_probe $NAME: controlli falliti o arresto -> $OUT"
  exit 1
fi
