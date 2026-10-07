#!/usr/bin/env bash
# Primo job: legge driver, GPU, glibc e filesystem del nodo assegnato. Non installa nulla
# e non usa interpreti o ambienti di altri progetti.
#SBATCH --job-name=lvl-f0-probe
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue gpu

echo "=== nodo ==="
ldd --version 2>&1 | sed -n 1p
command -v python3 && python3 --version
ls -d /usr/local/cuda* 2> /dev/null || echo "nessuna /usr/local/cuda* su questo nodo"
command -v nvcc && nvcc --version | tail -n 1 || true
df -hT /home /dev/shm /tmp 2> /dev/null || true
df -i /home | tail -n 1
command -v tmux || echo "tmux non nel PATH del nodo di calcolo"

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p)
major=${driver%%.*}
if (( major >= 580 )); then
  echo "DRIVER $driver: compatibile con lo stack CUDA 13.0 del repository (torch cu130, vLLM 0.26.0)"
else
  echo "DRIVER $driver: NON compatibile con CUDA 13.0 -> usare l'ambiente cu128/cu129 (vedi CLAUDE.md)"
fi

ledger_note ok "probe driver=$driver"
