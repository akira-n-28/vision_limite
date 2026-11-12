#!/usr/bin/env bash
# Verifica finale di una generazione del backup (scripts/backup.py completezza): volumi pronti e verificati su Drive
# parte per parte, indici su Drive con gli stessi hash, membri dei piani contro gli inventari, indici contro membri.
# Sola lettura; scrive in trasferimento/completezza/<generazione>-<job>.
# Uso: sbatch --parsable slurm/backup_completezza.sbatch GENERAZIONE "PIANO..." "INVENTARIO..." "CATEGORIE..."
#   es.  ... g1-20261112 "trasferimento/piani/g1-20261112 trasferimento/piani/g1-20261112-codice" \
#            "trasferimento/inventari/inv-30943 trasferimento/inventari/inv-NNNNN" "tranne:codice codice"
#SBATCH --job-name=lvl-bk-compl
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
gen=${1:?generazione}; read -r -a plans <<< "${2:?piani}"; read -r -a invs <<< "${3:?inventari}"
read -r -a cats <<< "${4:?categorie}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
out="$PROJECT_ROOT/trasferimento/completezza/$gen-$SLURM_JOB_ID"
if srun /usr/bin/python3 -I -B scripts/backup.py completezza --stato "$PROJECT_ROOT/trasferimento/stato/$gen" \
    --piani "${plans[@]}" --inventari "${invs[@]}" --categorie-inventari "${cats[@]}" --out "$out"; then
  ledger_note ok "backup completezza $gen -> $out"
else
  ledger_note failed "backup completezza $gen -> $out"
  exit 1
fi
