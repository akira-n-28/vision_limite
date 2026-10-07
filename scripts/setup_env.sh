#!/usr/bin/env bash
# setup_env.sh — costruisce il venv del progetto in vision_limite/venvs senza carico sul frontend.
#
# Sul frontend girano solo trasferimenti controllati (fetch.sh, guarded_run.sh) e script leggeri;
# verifica degli hash, estrazione e installazione vanno in job SLURM CPU.
#
#   scripts/setup_env.sh status    stato di ogni fase e prossimo passo
#   scripts/setup_env.sh uv        scarica il binario di uv e invia il job che lo verifica/estrae
#   scripts/setup_env.sh lock      risolve le dipendenze di env/pyproject.toml (uv lock sotto guardia)
#                                  e genera il manifest delle wheel
#   scripts/setup_env.sh wheels    avvia in background il download delle wheel (una alla volta)
#   scripts/setup_env.sh install   invia il job che crea il venv offline dalle wheel
#
# Ogni comando fa una sola fase e torna subito: i job e i download proseguono da soli.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname -- "$HERE")
# shellcheck source=/dev/null
source "$ROOT/config/paths.env"

UV_VERSION=${UV_VERSION:-0.11.32}
TOOLS=$TOOLS_DIR
UV_TARBALL="$TOOLS/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
UV="$TOOLS/uv"
LOCK_DIR="$ROOT/env"
WHEEL_MANIFEST="$ROOT/downloads/manifests/wheels.tsv"
LOGS="$DATA_ROOT/logs"
BASE_PYTHON=${BASE_PYTHON:-$(command -v python3.12 || echo /usr/bin/python3)}

[[ $(hostname -s) == lagrangectl || -n ${SETUP_ENV_ANY_HOST:-} ]] \
  || { echo "da eseguire sul frontend lagrangectl (host attuale: $(hostname -s))" >&2; exit 2; }
mkdir -p "$TOOLS" "$LOGS" "$OUTPUT_ROOT/logs" "$WHEELHOUSE" "$CACHE_ROOT/uv" "$VENVS_DIR"

submit() {  # script [env assignments...]
  local script=$1; shift
  local id
  id=$(env "$@" sbatch --parsable "$ROOT/slurm/$script")
  id=${id%%;*}
  printf '%s\t%s\t%s\tsubmitted\t%s\n' "$(date --iso-8601=seconds)" "$id" "$script" "setup_env" >> "$ROOT/runs/ledger.tsv"
  echo "inviato $script: job $id (log in $OUTPUT_ROOT/logs/)"
}

wheels_missing() {
  [[ -f $WHEEL_MANIFEST ]] || { echo "?"; return; }
  awk -F'\t' '!/^#/ && NF >= 3 { print $2 "\t" $3 }' "$WHEEL_MANIFEST" | {
    n=0
    while IFS=$'\t' read -r dest size; do
      if [[ ! -f $dest ]] || { [[ $size != "-" ]] && [[ $(stat -c %s -- "$dest") != "$size" ]]; }; then n=$((n + 1)); fi
    done
    echo "$n"
  }
}

cmd_status() {
  local next=""
  if [[ -x $UV ]]; then echo "uv:        $("$UV" --version)"; else echo "uv:        assente (tarball $([[ -f $UV_TARBALL ]] && echo presente || echo assente))"; fi
  [[ -x $UV ]] || next=${next:-"scripts/setup_env.sh uv"}
  if [[ -f $LOCK_DIR/pyproject.toml ]]; then
    echo "pyproject: $LOCK_DIR/pyproject.toml"
  else
    echo "pyproject: assente -> scegliere il caso in env/README.md (dipende dal driver, vedi job di sonda)"
    next=${next:-"creare env/pyproject.toml dal template giusto (chiedere all'utente)"}
  fi
  if [[ -f $LOCK_DIR/uv.lock && $LOCK_DIR/uv.lock -nt $LOCK_DIR/pyproject.toml ]]; then
    echo "uv.lock:   aggiornato"
  else
    echo "uv.lock:   da generare"
    next=${next:-"scripts/setup_env.sh lock"}
  fi
  local miss; miss=$(wheels_missing)
  echo "wheel:     mancanti $miss (manifest: $WHEEL_MANIFEST)"
  local pidf="$DATA_ROOT/.fetch_manifest.pid"
  if [[ -f $pidf ]] && kill -0 "$(cat "$pidf")" 2> /dev/null; then
    echo "download:  in corso (PID $(cat "$pidf")); ultimo log: $(ls -t "$LOGS"/fetch-*.log 2> /dev/null | sed -n 1p)"
    next=${next:-"attendere la fine dei download (tail sul log)"}
  elif [[ $miss != 0 ]]; then
    next=${next:-"scripts/setup_env.sh wheels"}
  fi
  echo "venv:      $(readlink "$ENV_ROOT" 2> /dev/null || echo "nessun link venvs/current"); build presenti: $(find "$VENVS_DIR" -maxdepth 1 -name 'build-*' -printf '%f ' 2> /dev/null)"
  [[ -L $ENV_ROOT ]] || next=${next:-"scripts/setup_env.sh install, poi leggere il log e con l'ok dell'utente: ln -sfn build-<jobid> $ENV_ROOT"}
  echo "job:       $(squeue -u "$USER" -h -o '%i %j %T' 2> /dev/null | tr '\n' ';')"
  echo "PROSSIMO PASSO: ${next:-ambiente pronto; lanciare slurm/f0_checks.sbatch}"
}

cmd_uv() {
  [[ -x $UV ]] && { echo "uv già presente: $("$UV" --version)"; return; }
  "$HERE/fetch.sh" "$UV_URL" "$UV_TARBALL"
  "$HERE/fetch.sh" "$UV_URL.sha256" "$UV_TARBALL.sha256"
  submit extract_tools.sbatch
}

cmd_lock() {
  [[ -x $UV ]] || { echo "manca uv: scripts/setup_env.sh uv" >&2; exit 1; }
  [[ -f $LOCK_DIR/pyproject.toml ]] || { echo "manca env/pyproject.toml (vedi env/README.md)" >&2; exit 1; }
  # Un download alla volta, nessuna build di sorgenti, nessun Python scaricato da uv.
  export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
  export UV_PYTHON_DOWNLOADS=never UV_NO_PROGRESS=1 UV_CACHE_DIR="$CACHE_ROOT/uv"
  export RAYON_NUM_THREADS=1 TOKIO_WORKER_THREADS=1 MALLOC_ARENA_MAX=1
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  if [[ -n ${SKIP_RESOLVE:-} ]]; then
    [[ -f $LOCK_DIR/uv.lock ]] || { echo "SKIP_RESOLVE impostato ma manca env/uv.lock" >&2; exit 1; }
    echo "uso env/uv.lock esistente senza risolvere"
  elif ! "$HERE/guarded_run.sh" uv-lock -- "$UV" lock --directory "$LOCK_DIR" --no-build --python "$BASE_PYTHON"; then
    cat >&2 << 'EOF'
uv lock non è riuscito: vedere il messaggio sopra e download-log.tsv.
Se è stato fermato per memoria o RSS (codici 20/21), piano B:
Piano B: generare il lock su una macchina con Internet (`cd env && uv lock --no-build`),
fare commit di env/uv.lock e poi:  SKIP_RESOLVE=1 scripts/setup_env.sh lock
EOF
    exit 1
  fi
  /usr/bin/python3 -I "$HERE/uvlock_manifest.py" "$LOCK_DIR/uv.lock" --wheelhouse "$WHEELHOUSE" \
    --out "$WHEEL_MANIFEST" --max-glibc "${MAX_GLIBC:-2.35}"
  echo "versioni chiave nel lock:"
  awk '/^name = "(torch|vllm|transformers)"$/ { n = $3; getline; print "  " n " " $3 }' "$LOCK_DIR/uv.lock"
}

cmd_wheels() {
  [[ -f $WHEEL_MANIFEST ]] || { echo "manca il manifest: scripts/setup_env.sh lock" >&2; exit 1; }
  local log; log="$LOGS/fetch-wheels-$(date +%Y%m%d-%H%M%S).log"
  nohup setsid "$HERE/fetch_manifest.sh" "$WHEEL_MANIFEST" > "$log" 2>&1 < /dev/null &
  echo "download avviato in background, log: $log"
}

cmd_install() {
  local miss; miss=$(wheels_missing)
  [[ $miss == 0 ]] || { echo "mancano ancora $miss wheel" >&2; exit 1; }
  submit env_install.sbatch LOCK_DIR="$LOCK_DIR" BASE_PYTHON="$BASE_PYTHON"
}

case ${1:-status} in
  status) cmd_status ;;
  uv) cmd_uv ;;
  lock) cmd_lock ;;
  wheels) cmd_wheels ;;
  install) cmd_install ;;
  *) echo "uso: setup_env.sh status|uv|lock|wheels|install" >&2; exit 2 ;;
esac
