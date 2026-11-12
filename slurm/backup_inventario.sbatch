#!/usr/bin/env bash
# Inventario della cartella del progetto per il backup (scripts/backup.py inventario, docs/BACKUP_RIPRISTINO.md).
# Solo lettura del progetto: scrive in trasferimento/inventari/inv-<job>, escluso dal backup e da git.
# Uso: sbatch --parsable slurm/backup_inventario.sbatch
#SBATCH --job-name=lvl-bk-inv
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/prignano/vision_limite/trasferimento/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/trasferimento/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
out="$PROJECT_ROOT/trasferimento/inventari/inv-$SLURM_JOB_ID"
mkdir -p "$PROJECT_ROOT/trasferimento/inventari"
if srun /usr/bin/python3 -I -B scripts/backup.py inventario --root "$PROJECT_ROOT" --out "$out" --segreti; then
  ledger_note ok "backup inventario -> $out"
else
  ledger_note failed "backup inventario -> $out"
  exit 1
fi
