#!/usr/bin/env bash
# Verifica finale di una generazione del backup (scripts/backup.py completezza), con esito e codice d'uscita:
# piani integri e inventari registrati senza errori, marcatori di questo piano, parti e indici verificati su Drive,
# copertura calcolata dalle voci davvero archiviate meno le esclusioni dichiarate. Sola lettura; scrive in
# trasferimento/completezza/<generazione>-<job>.
# Uso: sbatch --parsable slurm/backup_completezza.sbatch GENERAZIONE "PIANO..." [--solo-nucleo]
#   es.  ... g1-20261112 "trasferimento/piani/g1-20261112 trasferimento/piani/g1-20261112-codice" --solo-nucleo
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
gen=${1:?generazione}; read -r -a plans <<< "${2:?piani}"; shift 2   # altri argomenti (--solo-nucleo) a backup.py
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
out="$PROJECT_ROOT/trasferimento/completezza/$gen-$SLURM_JOB_ID"
if srun /usr/bin/python3 -I -B scripts/backup.py completezza --stato "$PROJECT_ROOT/trasferimento/stato/$gen" \
    --piani "${plans[@]}" --out "$out" "$@"; then
  ledger_note ok "backup completezza $gen: completo -> $out"
else
  ledger_note failed "backup completezza $gen: incompleto o non verificabile -> $out"
  exit 1
fi
