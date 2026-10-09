#!/usr/bin/env bash
# CPU visibili a un task srun con e senza --cpus-per-task esplicito (Slurm 23.11). Nessuna GPU.
# Uso: sbatch slurm/cpu_affinity_probe.sbatch
#SBATCH --job-name=lvl-cpu-probe
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
probe='import os; print(len(os.sched_getaffinity(0)), sorted(os.sched_getaffinity(0)))'
echo "script batch: $(python3 -c "$probe")"
echo "srun senza opzioni: $(srun "$RUN_PYTHON" -I -c "$probe")"
echo "srun --cpus-per-task=$SLURM_CPUS_PER_TASK: $(srun --cpus-per-task="$SLURM_CPUS_PER_TASK" "$RUN_PYTHON" -I -c "$probe")"
echo "SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK: $(SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK srun "$RUN_PYTHON" -I -c "$probe")"
ledger_note ok "sonda affinità CPU"
