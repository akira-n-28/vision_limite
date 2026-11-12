#!/usr/bin/env bash
# Prova di ripristino di un volume con gli strumenti standard (scripts/backup.py ripristino): sha256 delle parti,
# cat | tar -xp in una cartella di prova, confronto con l'indice (tipi, permessi, date, link, hard link, sha256) e
# sha256sum -c; la copia di prova si rimuove alla fine. Le parti possono essere quelle in staging o una copia
# riscaricata da Drive.
# Uso: sbatch --parsable slurm/backup_ripristino.sbatch GENERAZIONE VOLUME CARTELLA_PARTI
#SBATCH --job-name=lvl-bk-ripr
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/trasferimento/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/trasferimento/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
gen=${1:?generazione}; vol=${2:?volume}; parts=${3:?cartella delle parti}
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
S="$PROJECT_ROOT/trasferimento/stato/$gen"
mkdir -p "$S/ripristino" "$PROJECT_ROOT/trasferimento/tmp"
report="$S/ripristino/$vol-$SLURM_JOB_ID.json"
if srun /usr/bin/python3 -I -B scripts/backup.py ripristino --pronto "$S/vol/$vol.pronto.json" --cartella "$parts" \
    --indice "$S/indici/$vol.jsonl.gz" --sha256sum "$S/indici/$vol.sha256" \
    --dest "$PROJECT_ROOT/trasferimento/tmp/ripristino-$SLURM_JOB_ID" --report "$report" --rimuovi; then
  ledger_note ok "backup ripristino di prova $gen/$vol da $parts -> $report"
else
  ledger_note failed "backup ripristino di prova $gen/$vol da $parts -> $report"
  exit 1
fi
