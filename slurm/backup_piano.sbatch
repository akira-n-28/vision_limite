#!/usr/bin/env bash
# Piano dei volumi del backup (scripts/backup.py piano) da un inventario di slurm/backup_inventario.sbatch.
# Scrive in trasferimento/piani/<uscita>; nessun file del progetto viene modificato.
# Uso: sbatch --parsable slurm/backup_piano.sbatch INVENTARIO USCITA GENERAZIONE [opzioni di backup.py piano...]
#   es.  ... trasferimento/inventari/inv-30941 g1-20261112 g1-20261112 --escludi-categorie codice
#SBATCH --job-name=lvl-bk-piano
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/trasferimento/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/trasferimento/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
inv=${1:?inventario}; name=${2:?uscita}; gen=${3:?generazione}; shift 3
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
mkdir -p "$PROJECT_ROOT/trasferimento/piani"
out="$PROJECT_ROOT/trasferimento/piani/$name"
if srun /usr/bin/python3 -I -B scripts/backup.py piano --inventario "$inv" --out "$out" --gen "$gen" "$@"; then
  ledger_note ok "backup piano $gen da $inv -> $out"
else
  ledger_note failed "backup piano $gen da $inv"
  exit 1
fi
