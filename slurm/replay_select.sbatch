#!/usr/bin/env bash
# F2, replay testuale: correzione delle generazioni e shard di training (scripts/replay_select.py). Job solo CPU.
# Uso: sbatch slurm/replay_select.sbatch NOME GEN_RUN [GEN_RUN ...]
#   GEN_RUN: nomi in outputs/replay_gen/; uscita in outputs/replay_select/NOME e data/shards/replay/NOME
#SBATCH --job-name=lvl-replay-select
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
(( $# >= 2 )) || { echo "uso: replay_select.sbatch NOME GEN_RUN [GEN_RUN ...]" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
NAME=$1
shift
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido: $NAME" >&2; exit 2; }
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
DSR="$HF_HOME/hub/datasets--agentica-org--DeepScaleR-Preview-Dataset/snapshots/b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29"
DM="$HF_HOME/hub/datasets--zwhe99--DeepMath-103K/snapshots/5cf055d1fe3d7a2eb19719ac020211469736ae44"
gen=()
for g in "$@"; do test -d "$OUTPUT_ROOT/replay_gen/$g" || { echo "generazione mancante: $g" >&2; exit 1; }; gen+=("$OUTPUT_ROOT/replay_gen/$g"); done
SHARDS="$DATA_ROOT/shards/replay/$NAME"
test ! -e "$SHARDS" || { echo "$SHARDS esiste già: scegliere un altro nome" >&2; exit 1; }
if srun "$RUN_PYTHON" -u -B scripts/replay_select.py --gen "${gen[@]}" --deepscaler "$DSR/deepscaler.json" \
     --deepmath "$DM"/data/train-*.parquet --out "$OUTPUT_ROOT/replay_select/$NAME" --shards "$SHARDS"; then
  ledger_note ok "replay selezione $NAME ($*) -> $SHARDS"
else
  ledger_note failed "replay selezione $NAME ($*) -> $SHARDS"
  exit 1
fi
