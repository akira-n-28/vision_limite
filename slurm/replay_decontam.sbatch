#!/usr/bin/env bash
# Decontaminazione del replay testuale di F2 (scripts/replay_decontam.py) contro i benchmark testuali della suite
# math-extended (regola per il solo testo: copertura >= 0.2 o parole uguali). Job solo CPU.
# Uso: sbatch slurm/replay_decontam.sbatch [OUT_DIR]   (default data/decontam/text/replay)
#SBATCH --job-name=lvl-replay-decontam
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
OUT=${1:-$DATA_ROOT/decontam/text/replay}
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
H="$HF_HOME/hub"
snap() { echo "$H/datasets--$1/snapshots/$2"; }
DSR=$(snap agentica-org--DeepScaleR-Preview-Dataset b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29)
DM=$(snap zwhe99--DeepMath-103K 5cf055d1fe3d7a2eb19719ac020211469736ae44)
bench=(
  aime24="$(snap HuggingFaceH4--aime_2024 2fe88a2f1091d5048c0f36abc874fb997b3dd99a)"
  aime25="$(snap opencompass--AIME2025 a6ad95f611d72cf628a80b58bd0432ef6638f958)"
  aime26="$(snap MathArena--aime_2026 10b4e45b7a503075d4da8a0d57916a4f06ce6bd2)"
  hmmt25="$(snap MathArena--hmmt_feb_2025 6fdc4277120810ff75aa22d2d5489b91f7a262a1)"
  hmmt26="$(snap MathArena--hmmt_feb_2026 02fba4f74d8e68e73e66a02d540fd979c05c274c)"
  beyondaime="$(snap ByteDance-Seed--BeyondAIME c705198ae1043810b1e1693bd879250b51a7a523)"
  apex="$(snap MathArena--apex-shortlist f3efdf224ef665f129ddaae37699f6098c65781b)"
  math500="$(snap HuggingFaceH4--MATH-500 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be)"
  olympiadbench="$(snap Hothan--OlympiadBench 91184b52131e7fc9455fef848035173aea8cc01a)"
)
for d in "$DSR" "$DM" "${bench[@]#*=}"; do test -d "$d" || { echo "snapshot mancante: $d" >&2; exit 1; }; done
args=(--deepscaler "$DSR/deepscaler.json" --deepmath "$DM"/data/train-*.parquet --out "$OUT")
for b in "${bench[@]}"; do args+=(--benchmark "$b"); done
if srun "$RUN_PYTHON" -u -B scripts/replay_decontam.py "${args[@]}"; then
  ledger_note ok "decontaminazione del replay testuale -> $OUT"
else
  ledger_note failed "decontaminazione del replay testuale -> $OUT"
  exit 1
fi
