#!/usr/bin/env bash
# Catalogo di recupero del backup (scripts/backup.py catalogo): repository, commit, file esatti, dimensioni, hash e
# presenza locale; riferimento degli hash dei dati per il confronto con le copie sul Mac; derivati; ambienti; link
# assoluti o esterni; file con la radice assoluta. Sola lettura del progetto; scrive in trasferimento/catalogo/.
# Uso: sbatch --parsable slurm/backup_catalogo.sbatch INVENTARIO GENERAZIONE
#SBATCH --job-name=lvl-bk-cat
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/trasferimento/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/trasferimento/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
inv=${1:?inventario}; gen=${2:?generazione}
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
out="$PROJECT_ROOT/trasferimento/catalogo/$gen-$SLURM_JOB_ID"
if srun /usr/bin/python3 -I -B scripts/backup.py catalogo --root "$PROJECT_ROOT" --inventario "$inv" \
    --stato "$PROJECT_ROOT/trasferimento/stato/$gen" --out "$out"; then
  ledger_note ok "backup catalogo $gen da $inv -> $out"
else
  ledger_note failed "backup catalogo $gen da $inv"
  exit 1
fi
