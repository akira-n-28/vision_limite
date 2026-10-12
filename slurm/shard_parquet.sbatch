#!/usr/bin/env bash
# Parquet con immagini incluse -> shard tar (scripts/shard_parquet.py). Job solo CPU.
# Uso: sbatch slurm/shard_parquet.sbatch KIND PREFISSO CARTELLA_USCITA SORGENTE FILE.parquet [FILE.parquet ...]
#SBATCH --job-name=lvl-shard-pq
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 5 )) || { echo "uso: shard_parquet.sbatch KIND PREFISSO CARTELLA_USCITA SORGENTE FILE..." >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
export OMP_NUM_THREADS=1
KIND=$1; PREFIX=$2; OUT=$3; SOURCE=$4
shift 4
case $OUT in "$DATA_ROOT"/shards/*) ;; *) echo "la cartella di uscita deve stare in $DATA_ROOT/shards" >&2; exit 2 ;; esac
df -hT "$DATA_ROOT" | tail -n 1
if srun "$RUN_PYTHON" -u -B scripts/shard_parquet.py --kind "$KIND" --prefix "$PREFIX" --out "$OUT" --source "$SOURCE" --files "$@"; then
  ledger_note ok "shard $KIND $PREFIX -> $OUT"
else
  ledger_note failed "shard $KIND $PREFIX -> $OUT"
  exit 1
fi
