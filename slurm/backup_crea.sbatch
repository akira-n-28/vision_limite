#!/usr/bin/env bash
# Crea i volumi del backup (scripts/backup.py crea): un file letto una volta, sha256 per file e per parte, rilettura
# completa ed elenco di GNU tar prima di dichiarare un volume pronto. Staging limitato: con --attendi il job aspetta
# che scripts/backup_upload.sh carichi, verifichi e liberi lo staging. Riprende dai volumi non ancora pronti.
# Arresto gentile: touch /home/prignano/vision_limite/trasferimento/stato/STOP (prima del volume successivo).
# Uso: sbatch --parsable slurm/backup_crea.sbatch PIANO [opzioni di backup.py crea...]
#   es.  ... trasferimento/piani/g1-20261112 --attendi --limite-staging-gib 30
#SBATCH --job-name=lvl-bk-crea
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=/home/prignano/vision_limite/trasferimento/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/trasferimento/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
plan=${1:?cartella del piano}; shift
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
T="$PROJECT_ROOT/trasferimento"
mkdir -p "$T/staging" "$T/stato"
if srun /usr/bin/python3 -I -B scripts/backup.py crea --piano "$plan" --staging "$T/staging" --stato "$T/stato" "$@"; then
  ledger_note ok "backup crea $plan $*"
else
  ledger_note failed "backup crea $plan $*"
  exit 1
fi
