# slurm/prologue.sh — provenienza del job (LAGRANGE.md §7). Da usare con `source` dentro uno sbatch.
# Uso:  source "$PROJECT_ROOT/slurm/prologue.sh"; lagrange_prologue gpu|cpu
# Richiede: PROJECT_ROOT, OUTPUT_ROOT, SEED, CONFIG_PATH (anche "-" se il job non ha config).

lagrange_prologue() {
  local mode=${1:-gpu}
  : "${SLURM_JOB_ID:?Eseguire tramite sbatch/srun}"
  : "${PROJECT_ROOT:?Impostare PROJECT_ROOT}"
  : "${OUTPUT_ROOT:?Impostare OUTPUT_ROOT}"
  echo "=== provenienza job ${SLURM_JOB_ID} ==="
  date --iso-8601=seconds
  hostname
  pwd
  git -C "$PROJECT_ROOT" rev-parse HEAD 2> /dev/null || echo "git: non disponibile"
  { git -C "$PROJECT_ROOT" status --short 2> /dev/null || true; } | sed -n 1,50p
  printf 'SLURM_JOB_ID=%s\nSLURM_JOB_NODELIST=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
    "${SLURM_JOB_ID:-unset}" "${SLURM_JOB_NODELIST:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'CPUS_PER_TASK=%s\nCPUS_ON_NODE=%s\nMEM_PER_NODE_MiB=%s\nMEM_PER_CPU_MiB=%s\n' \
    "${SLURM_CPUS_PER_TASK:-unset}" "${SLURM_CPUS_ON_NODE:-unset}" \
    "${SLURM_MEM_PER_NODE:-unset}" "${SLURM_MEM_PER_CPU:-unset}"
  scontrol show job "$SLURM_JOB_ID" | grep -E 'JobName|TRES|TimeLimit|WorkDir|StdOut' || true
  printf 'PROJECT_ROOT=%s\nDATA_ROOT=%s\nOUTPUT_ROOT=%s\nCACHE_ROOT=%s\nCHECKPOINT_ROOT=%s\nHF_HOME=%s\n' \
    "$PROJECT_ROOT" "${DATA_ROOT:-unset}" "$OUTPUT_ROOT" "${CACHE_ROOT:-unset}" \
    "${CHECKPOINT_ROOT:-unset}" "${HF_HOME:-unset}"
  printf 'HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s\n' "${HF_HUB_OFFLINE:-unset}" "${TRANSFORMERS_OFFLINE:-unset}"
  printf 'SEED=%s\nCONFIG_PATH=%s\nRUN_PYTHON=%s\n' "${SEED:?Impostare SEED}" "${CONFIG_PATH:?Impostare CONFIG_PATH}" "${RUN_PYTHON:-unset}"
  if [[ $CONFIG_PATH != "-" ]]; then
    test -r "$CONFIG_PATH"
    sha256sum "$CONFIG_PATH"
  fi
  if [[ $mode == gpu ]]; then
    nvidia-smi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  fi
  echo "=== fine provenienza ==="
}

# Riga nel registro dei job del repository (piccolo file di testo versionato).
ledger_note() {  # stato nota
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${SLURM_JOB_ID:-}" \
    "${SLURM_JOB_NAME:-}" "$1" "$2" >> "$PROJECT_ROOT/runs/ledger.tsv"
}
