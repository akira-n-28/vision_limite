#!/usr/bin/env bash
# Verifica SHA-256 ed estrae il binario di uv (l'estrazione va in allocazione, non sul frontend).
#SBATCH --job-name=lvl-tools
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu

TOOLS=$TOOLS_DIR
TARBALL="$TOOLS/uv-x86_64-unknown-linux-gnu.tar.gz"
test -f "$TARBALL" && test -f "$TARBALL.sha256"
want=$(awk '{ print $1 }' "$TARBALL.sha256")
got=$(sha256sum "$TARBALL" | awk '{ print $1 }')
[[ $want == "$got" ]] || { echo "SHA-256 del tarball di uv non corrisponde" >&2; ledger_note failed "uv sha256"; exit 1; }
dir="$TOOLS/extract-$SLURM_JOB_ID"
mkdir -p "$dir"
tar -xzf "$TARBALL" -C "$dir"
bin=$(find "$dir" -name uv -type f -print -quit)
install -m 755 "$bin" "$TOOLS/uv.new"
mv -f "$TOOLS/uv.new" "$TOOLS/uv"
"$TOOLS/uv" --version
ledger_note ok "uv $("$TOOLS/uv" --version | awk '{ print $2 }') in $TOOLS/uv"
