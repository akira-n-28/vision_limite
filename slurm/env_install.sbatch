#!/usr/bin/env bash
# Costruisce l'ambiente Python OFFLINE da wheel già scaricate. Job solo CPU.
#
# Input (variabili d'ambiente al momento di sbatch, oppure default):
#   LOCK_DIR   cartella con pyproject.toml + uv.lock da installare (default: env/ del repo)
#   UV_TARBALL tarball di uv scaricato con fetch.sh (default: $TOOLS_DIR/uv-x86_64-unknown-linux-gnu.tar.gz)
#   BASE_PYTHON interprete 3.12 del nodo (default: /usr/bin/python3)
# Output: un ambiente NUOVO in $VENVS_DIR/build-<jobid>. Non tocca ambienti esistenti.
# Per adottarlo, dopo aver letto il log:  ln -sfn build-<jobid> $VENVS_DIR/current
#SBATCH --job-name=lvl-env
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu

LOCK_DIR=${LOCK_DIR:-$PROJECT_ROOT/env}
TOOLS=$TOOLS_DIR
UV_TARBALL=${UV_TARBALL:-$TOOLS/uv-x86_64-unknown-linux-gnu.tar.gz}
BASE_PYTHON=${BASE_PYTHON:-$(command -v python3.12 || echo /usr/bin/python3)}
mkdir -p "$VENVS_DIR"
NEW_ENV="$VENVS_DIR/build-${SLURM_JOB_ID}"
export UV_OFFLINE=1 UV_NO_PROGRESS=1 UV_CACHE_DIR="$CACHE_ROOT/uv"

test -f "$LOCK_DIR/uv.lock" || { echo "manca $LOCK_DIR/uv.lock" >&2; exit 1; }
test -d "$WHEELHOUSE" || { echo "manca $WHEELHOUSE" >&2; exit 1; }
"$BASE_PYTHON" --version
[[ $("$BASE_PYTHON" -I -c 'import sys; print(sys.version_info[:2] == (3, 12))') == True ]] \
  || { echo "serve Python 3.12 come BASE_PYTHON" >&2; exit 1; }

UV="$TOOLS/uv"
if [[ ! -x $UV ]]; then
  test -f "$UV_TARBALL" || { echo "manca uv: scaricare $UV_TARBALL con fetch.sh" >&2; exit 1; }
  mkdir -p "$TOOLS/extract-$SLURM_JOB_ID"
  tar -xzf "$UV_TARBALL" -C "$TOOLS/extract-$SLURM_JOB_ID"
  install -m 755 "$(find "$TOOLS/extract-$SLURM_JOB_ID" -name uv -type f -print -quit)" "$UV"
fi
"$UV" --version

reqs="$OUTPUT_ROOT/env/requirements-${SLURM_JOB_ID}.txt"
mkdir -p "$OUTPUT_ROOT/env"
(cd "$LOCK_DIR" && "$UV" export --frozen --offline --no-emit-project --no-dev \
    --format requirements-txt -o "$reqs")
# Sorgenti URL (es. la wheel di vLLM nel caso B): il job è offline, quindi le punto alla copia
# già scaricata nel wheelhouse; l'hash del lock (--hash) resta e viene verificato da uv.
url_re='^([A-Za-z0-9._-]+) @ https://[^ ]+/([^/ ]+\.whl)(.*)$'
while IFS= read -r line; do
  if [[ $line =~ $url_re ]]; then
    fname=${BASH_REMATCH[2]//%2B/+}
    [[ -f $WHEELHOUSE/$fname ]] && line="${BASH_REMATCH[1]} @ file://$WHEELHOUSE/$fname${BASH_REMATCH[3]}"
  fi
  printf '%s\n' "$line"
done < "$reqs" > "$reqs.local"
mv -f "$reqs.local" "$reqs"
if grep -E '^[A-Za-z0-9._-]+ @ https://' "$reqs"; then
  echo "sorgenti URL senza copia locale nel wheelhouse (sopra): il job è offline, mi fermo" >&2
  exit 1
fi
grep -E ' @ file://' "$reqs" | cut -c1-200 || true
echo "requisiti: $(grep -c '==' "$reqs") pacchetti ($reqs)"

"$UV" venv --python "$BASE_PYTHON" "$NEW_ENV"
srun "$UV" pip install --offline --no-index --find-links "$WHEELHOUSE" \
    --python "$NEW_ENV/bin/python" -r "$reqs"

# Plugin Limite (serve hatchling nel wheelhouse per la build offline)
if [[ -d $PROJECT_ROOT/third_party/limite-violetto ]]; then
  "$UV" pip install --offline --no-index --find-links "$WHEELHOUSE" --no-deps \
      --python "$NEW_ENV/bin/python" "$PROJECT_ROOT/third_party/limite-violetto"
fi

"$UV" pip freeze --python "$NEW_ENV/bin/python" > "$OUTPUT_ROOT/env/freeze-${SLURM_JOB_ID}.txt"
"$NEW_ENV/bin/python" -I -c 'import torch, transformers; print("torch", torch.__version__, "cuda build", torch.version.cuda); print("transformers", transformers.__version__)'
# torchvision/torchaudio devono avere la stessa build CUDA di torch; transformers.modeling_utils
# importa torchvision (import pigro: `import transformers` da solo non lo verifica).
"$NEW_ENV/bin/python" -I -c 'import torch, torchvision, torchaudio, torchcodec, transformers.modeling_utils; print("torchvision", torchvision.__version__, "torchaudio", torchaudio.__version__, "torchcodec", torchcodec.__version__, "ops nms:", torch.ops.torchvision.nms is not None)'

# Nessuna libreria nativa compilata per un'altra CUDA (es. wheel PyPI per cu130 con torch cu129:
# job 29375 torchvision, job 29379 torchcodec). Solo intestazioni ELF, nessun caricamento.
if command -v readelf > /dev/null; then
  cuda_major=$("$NEW_ENV/bin/python" -I -c 'import torch; print(torch.version.cuda.split(".")[0])')
  bad=$( { find "$NEW_ENV/lib" -type f \( -name '*.so' -o -name '*.so.*' \) -print0 \
           | xargs -0 -n 50 readelf -d 2> /dev/null || true; } \
    | awk -v want="$cuda_major" '/^File: /{f=$2}
        /\(NEEDED\)/ && match($0, /lib(cudart|nvrtc|cublas|cublasLt|nvJitLink)\.so\.[0-9]+/) {
          s = substr($0, RSTART, RLENGTH); n = split(s, p, "."); if (p[n] != want) print f " -> " s }')
  if [[ -n $bad ]]; then
    echo "librerie per una CUDA diversa da $cuda_major (torch):" >&2
    echo "$bad" >&2
    exit 1
  fi
  echo "librerie native: nessuna dipendenza da CUDA diversa da $cuda_major"
else
  echo "readelf assente: controllo delle librerie CUDA saltato" >&2
fi
"$NEW_ENV/bin/python" -I -c 'import vllm; print("vllm", vllm.__version__)' || echo "vllm non importabile in CPU (da verificare in un job GPU)"
echo "ambiente pronto: $NEW_ENV"
ledger_note ok "env $NEW_ENV da $LOCK_DIR"
