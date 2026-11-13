#!/usr/bin/env bash
# Confronti in sola lettura del backup (scripts/backup.py firme | rimossi):
#   firme   --piano-base PIANO.json --inventario INV --out FILE   categorie uguali, cambiate, nuove o scomparse
#   rimossi --base INDICI... --nuovo INDICI... --out FILE         percorsi da togliere dopo il ripristino di un supplemento
# Uso: sbatch --parsable slurm/backup_confronto.sbatch firme|rimossi [argomenti di backup.py...]
#SBATCH --job-name=lvl-bk-conf
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/prignano/vision_limite/trasferimento/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/trasferimento/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
mode=${1:?firme o rimossi}; shift
[[ $mode == firme || $mode == rimossi ]] || { echo "modo sconosciuto: $mode" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
if srun /usr/bin/python3 -I -B scripts/backup.py "$mode" "$@"; then
  ledger_note ok "backup $mode $*"
else
  ledger_note failed "backup $mode $*"
  exit 1
fi
