# Lagrange cluster operating guide

## User-authorized download clarification — 2026-10-06

The user confirms that `lagrangectl` is the only Internet-connected node and
has **3.8 GiB RAM shared by all users**. Controlled downloads on this frontend
are authorized, including large files streamed directly to persistent disk.
This clarification supersedes the generic frontend-download restriction below;
all other allocation, environment, storage and scheduler constraints remain.

Before any transfer read `MemTotal` and **MemAvailable** from `/proc/meminfo`,
check the destination filesystem/space/inodes and this task's active transfers,
and estimate complete/partial files, tokenized outputs and caches. One curl/wget
transfer at a time, no hidden workers, initially at most **10 MiB/s**. Do not start
below **1 GiB MemAvailable**. Check available memory and downloader RSS every
**2 seconds**; stop below **768 MiB MemAvailable** or above **128 MiB downloader
RSS**, preserve `.part`, and record the reason. Do not lower thresholds or
continuously restart under memory pressure. A nonprivileged per-downloader
cgroup memory limit may add protection; a virtual-address-space limit is not RSS.
Only signal processes created by this task. These protections cannot guarantee
against OOM caused by sudden allocations from other users.

Use explicit external `/home/prignano/` destinations, never frontend root or
`/dev/shm` caches. Validate HTTP completion/ranges before atomic promotion;
extended hashes, Parquet parsing, tokenization, model/checkpoint loading,
numerical imports, installation/extraction, compilation and training still
belong in verified SLURM allocations. Compute jobs remain offline: no proxy,
tunnel, DNS changes or compute-node downloads.

Reusable handoff for humans and coding agents working in any research repository
on Lagrange, the cluster of the Dipartimento di Matematica Guido Castelnuovo,
Sapienza Università di Roma. Read this file completely, then the new project's
own instructions. This guide describes cluster operation, not a project's
scientific or numerical contract.

## 1. Evidence and facts to revalidate

**Last lightweight verification: 2026-10-02, from frontend `lagrangectl`.**
Checks were read-only: host identity, memory, filesystem mounts, command paths,
compiler/interpreter versions, static package metadata, and SLURM configuration
and accounting policy. No compute job, package import requiring GPU libraries,
installation, download, training, or benchmark was performed for this guide.
Earlier compute-allocation reports dated 2026-09-25 supply the explicitly labeled
GPU and compute-toolkit observations. Revalidate inside an allocation before
depending on those observations; filesystem and policy checks must also be
repeated when starting a new project.

| Fact | Current / observed value | How to verify | Stability |
| --- | --- | --- | --- |
| SSH frontend | `lagrangectl.mat.uniroma1.it`; current local hostname `lagrangectl` | SSH to the public address; `hostname` | Expected stable cluster property; public address from prior documentation |
| Partition | `mat`, currently default, UP | `sinfo`; `scontrol show partition mat` | Cluster configuration; must revalidate |
| Compute node | `lagrange0` | `sinfo -N`; `scontrol show node lagrange0` | Cluster configuration; must revalidate |
| Node CPU inventory | 96 logical CPUs: 2 sockets × 24 cores × 2 threads (48 physical cores) | `scontrol show node lagrange0` | Verified scheduler inventory; allocation-dependent entitlement |
| Node memory inventory | `RealMemory=773714` MiB (about 755.6 GiB) | `scontrol show node lagrange0` | Verified scheduler inventory; job memory is separately limited |
| GPU inventory | SLURM currently configures `Gres=gpu:4`; four physical GPUs previously observed | `scontrol show node lagrange0`; administrator / allocated GPU inspection | Inventory is not permission to allocate four GPUs |
| GPU model / memory | Previously allocated NVIDIA H100 NVL, approximately 95,830 MiB per device | `nvidia-smi` **inside an allocation** | Previously observed; revalidate hardware assigned to job |
| NVIDIA driver | Previously `570.211.01` on compute | `nvidia-smi` inside allocation | Software-specific; must revalidate |
| CUDA toolkit | Frontend `/usr/bin/nvcc`: 12.0, V12.0.140 now; compute previously `/usr/local/cuda-12.6/bin/nvcc`: 12.6, V12.6.77 | `command -v nvcc`; `nvcc --version` on the relevant host | Host-specific; toolkit may be outside PATH |
| Frontend RAM | 3.8 GiB total now; available memory fluctuates | `free -h` | Host capacity; availability time-varying |
| Frontend root filesystem | ext4, approximately 59G total, 34G available now | `df -hT /` | Free space time-varying; not persistent shared experiment storage |
| Shared `/home` | GlusterFS, approximately 14T total, 12T available now; earlier verified on compute too | `df -hT /home`; `findmnt -T /home`; repeat in job | Persistent shared storage; space/quota/retention must revalidate |
| Frontend `/dev/shm` | 2.0G tmpfs now | `df -h /dev/shm` on relevant host | Host-specific; compute differs; RAM usage still counts toward job limits |
| System Python on frontend | `/usr/bin/python3`, 3.12.3; bare `python` absent | `command -v python3`; `python3 --version` | Host/environment-specific |
| Example existing environments | `/home/prignano/modded-nanogpt/.venv`, `/home/prignano/pg` both exist; Python 3.10.20 | Inspect explicit path; `bin/python -I -S -B --version` | Other projects' environments; availability/permission not guaranteed |
| Packages in those examples | Static files now report PyTorch `2.10.0+cu128`, CUDA build `12.8`, NumPy `2.2.6`; earlier compute runtime checks agreed | Static metadata first; runtime imports inside allocation | Environment-specific; version alone does not validate kernels |
| Inspected account / QoS | `prignano`: account `castelnuovo`, default QoS `castelnuovo_qos` | `sacctmgr -nP show assoc where user="$USER" format=User,Account,Partition,QOS,DefaultQOS` | User-specific; must revalidate |
| Inspected allocation quota | That QoS currently has `MaxTRESPU=cpu=12,gres/gpu=1,mem=150G` | `sacctmgr -nP show qos format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU` | Aggregate per-user QoS limits; policy can change; other limits may apply |
| Personal storage allowance | Not established; filesystem free space is not a personal quota | Available quota tools / administrator-confirmed policy | Must establish before large storage commitments |

The partition currently reports an unlimited maximum time and no default time;
always request an explicit sensible limit. Another QoS named
`castelnuovo_qos_2gpu` exists, but its presence does **not** establish entitlement
to use it. Never select a different account or QoS to bypass limits. Scheduler
configuration, account associations, availability and administrative policy all
constrain allocation.

`/scratch`, `/data`, `/datasets`, `/checkpoints`, `/usr/local/cuda` and
`/usr/local/cuda-12.6` were absent **on the frontend** during this check. This
does not establish their presence or absence on compute. Earlier compute
inspection found `/lagrangeGFS`, but its permitted use, sharing and retention
were not established: it is not a default artifact destination.

## 2. Connect and identify the machine

From your own terminal, using your authorized cluster account:

```bash
ssh <user>@lagrangectl.mat.uniroma1.it
```

Use existing approved SSH authentication. This document contains no credentials;
do not copy passwords, tokens or private keys into repositories or logs.
Immediately check `hostname`, `whoami`, `pwd` and whether `SLURM_JOB_ID` is set.
A shell on the compute node is not sufficient proof of an authorized allocation:
inspect the job and its assigned resources too.

| Machine / context | Appropriate work | Inappropriate work |
| --- | --- | --- |
| Login/frontend `lagrangectl` | Source editing, git, small logs, metadata, lightweight interpreter inspection, `sbatch`, `squeue`, `sacct`, user-authorized memory-monitored curl/wget transfers directly to disk | Parquet parsing, training, GPU benchmarks, preprocessing/tokenization, numerical-stack imports, large tensor/checkpoint loading, large test suites, heavy installation/compilation/extraction |
| SLURM allocation on `lagrange0` | CPU/GPU computation within assigned CPU, RAM, GPU and time limits | Assuming host totals are your resources; using unallocated GPUs or unapproved storage |

**Source editing and git operations are fine on the frontend; computation belongs
in allocations.** Even importing a large numerical stack may be inappropriate
on the constrained frontend. Inspect metadata there and validate imports in a
small allocation. Never run heavy work on the frontend first "just to test".

Seeing 96 CPUs, hundreds of GiB of host RAM or four GPUs does not grant those
resources to a job. SLURM currently uses `select/cons_tres` with
`CR_CORE_MEMORY`; whole-core allocation can round a CPU request to logical
threads. Inspect `AllocTRES`, `SLURM_CPUS_PER_TASK`, affinity and actual memory
limits instead of inferring entitlement from `free`, `nproc --all` or inventory.

## 3. Persistent files and SSH disconnection

Files saved under persistent shared `/home` survive an SSH disconnection.
Earlier allocations verified frontend-to-compute visibility and that compute
logs written there were readable from the frontend. Persistence is not a
backup guarantee: verify quota, retention and backup policy separately.

An accepted `sbatch` job is managed by SLURM and continues after the submitting
SSH connection closes. Ordinary commands started in a login shell may receive
a hangup or lose their controlling terminal and terminate. An interactive
`srun --pty` session may also end when its terminal disappears. Submit long
experiments through `sbatch`; a detached login terminal is not a compute policy.

After reconnecting, inspect the job rather than submitting a duplicate:

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS -P
```

With no overrides, `sbatch` normally writes stdout and stderr together to
`slurm-JOBID.out` in the submission working directory. Prefer explicit persistent
paths with `%j` (job ID), and preserve the job ID alongside the run manifest.
Create log directories **before submission**: SLURM opens the logs before the
script can create them. Logs may be buffered; absence of new output does not by
itself prove a hang.

## 4. Storage and local data

Separate these roles and configure them explicitly:

| Role | Recommended practice |
| --- | --- |
| Source repository | Small source/configuration files under `/home/<user>/<project>` |
| Datasets / tokenizers | Already-local, documented inputs in an approved persistent directory; avoid duplicate copies |
| Caches / environments | Explicit project-owned paths, outside source where large; protect stable environments |
| Checkpoints | Selected recovery/reproducibility checkpoints; estimate model and optimizer bytes × retained count |
| Run outputs | For example `/home/<user>/<project>-runs/<study-or-job>`; another verified convention is fine |
| Temporary files | Explicit `TMPDIR` appropriate to job size and verified filesystem; do not assume `/tmp` is shared or persistent |

Useful configuration names are `PROJECT_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT`,
`CACHE_ROOT` and `CHECKPOINT_ROOT`. Never repurpose `HOME`. Give package/runtime
caches explicit paths when needed; do not assume a library respects `CACHE_ROOT`
unless its own configuration is set. Large artifacts must live outside the
source tree: datasets, repeated checkpoints, profiler traces, package caches,
downloaded archives and tensor dumps do not belong in git or repository logs.

Before a large run, inspect the chosen destinations with `df -hT`, `df -i` and
`findmnt -T`; verify access, quota and expected output size plus temporary
headroom. The frontend root filesystem is small. `/dev/shm` is RAM-backed and
host-specific; large shared-memory loaders/caches can fail even with large GPU
memory. Compute-host memory totals and tmpfs capacity do not override the job's
cgroup memory limit. Do not assume node-local files survive a job or are visible
from the frontend.

Use local assets with explicit paths. Missing dataset, tokenizer or checkpoint
means **fail clearly**; never let training automatically fetch a replacement.
Do not assume Internet availability in compute jobs: prior compute probes had
DNS failures. A prepared experiment should operate offline. Record input path,
size, relevant hashes and the exact prefix/range/permutation used. Hashing or
scanning a large dataset is compute/I/O work and belongs in an allocation.
Avoid recursive `du`/`find`/`rg` scans of all `/home`; inspect named paths only.
Do not delete unfamiliar files or shared caches to recover space.

## 5. Python and CUDA environments

First inspect an existing environment without changing it. Two previously
validated examples are `/home/prignano/modded-nanogpt/.venv` and
`/home/prignano/pg`; these are examples from previous work, not prerequisites for
a new repository or permission to alter another project's environment.

Prefer an explicit interpreter path over reliance on shell activation:

```bash
/path/to/environment/bin/python -I -S -B --version
```

Record the executable, interpreter version, package versions, dependency lock and
any deliberate `PYTHONPATH` additions. A login-shell activation may not be
reproduced in batch. If dependencies differ materially, use an approved
project-specific environment. Do not casually mutate a stable reference/shared
environment, change shell startup files, or modify CUDA/library paths to make a
new project work. Do not assume `module`, conda or a container runtime exists;
earlier compute discovery did not find those tools on PATH.

Install/download only for an explicit need and with a deliberate resource,
storage and dependency plan. Under the explicit clarification above, frontend
transfers stream to disk with memory/RSS/bandwidth monitoring, including large
datasets; they do not parse data or load model weights there. Heavy installation,
builds and extraction require suitable compute resources and approved package
sources. No installation is part of using this guide by default.

CUDA has three distinct version contexts: NVIDIA driver compatibility reported
by `nvidia-smi`, the compiler toolkit reported by `nvcc`, and the runtime/build
used by PyTorch (`torch.version.cuda`). They can differ, including between login
and compute hosts. Do not force a system CUDA library via `LD_LIBRARY_PATH`
solely because a version string differs. Test the intended operations in the
allocated environment before trusting compatibility.

Check dtype support per operation. In the previously validated H100 / PyTorch
2.10.0+cu128 setup, CUDA bf16 `torch.linalg.svd` raised
`NotImplementedError`; this is observed software behavior, not a universal H100
limitation. bf16 matmul/autocast support does not imply bf16 support for every
linear-algebra primitive. Run numerical smoke checks, SVD benchmarks and other
large `torch.linalg` operations only inside allocations.

## 6. Submit work through SLURM

Check current policy before selecting resources:

```bash
sinfo -o '%P %a %l %D %N %G %c %m'
scontrol show partition mat
scontrol show node lagrange0
sacctmgr -nP show assoc where user="$USER" format=User,Account,Partition,QOS,DefaultQOS
sacctmgr -nP show qos format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU
```

If accounting policy is inaccessible or unclear, obtain the applicable limits
before a large request. Begin with a small smoke job, one GPU when needed, and
appropriate CPUs/RAM/time. Account QoS limits may apply across simultaneous jobs.
Do not request all node resources because the inventory lists them.

### Single-GPU batch template

**Replace every `<...>` placeholder before use.** This is a template, not a
ready-to-submit resource prescription. `--mem` is job host RAM, not GPU VRAM.
For a small smoke test, previously accepted requests included 1 GPU, 2 CPUs,
12G RAM and eight minutes; estimate your actual workload rather than reusing
those values for training.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=<short-name>
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=<appropriate-integer>
#SBATCH --mem=<appropriate-size-e.g.-12G>
#SBATCH --time=<appropriate-HH:MM:SS>
#SBATCH --output=/home/<user>/<project>-runs/logs/%x-%j.out
#SBATCH --error=/home/<user>/<project>-runs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
: "${PROJECT_ROOT:?Set explicit repository path}"
: "${RUN_PYTHON:?Set explicit interpreter path}"
: "${OUTPUT_ROOT:?Set persistent output directory}"
test -d "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
test -d "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Insert the provenance block below, validate required local inputs, then:
srun "$RUN_PYTHON" -u <entrypoint.py> <nonsecret-arguments>
```

Do not override `CUDA_VISIBLE_DEVICES`; SLURM owns device assignment. Do not
choose a physical GPU index manually. A one-GPU job should normally see one
assigned CUDA device even though the node contains more.

### CPU-only batch template

```bash
#!/usr/bin/env bash
#SBATCH --job-name=<short-cpu-name>
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<appropriate-integer>
#SBATCH --mem=<appropriate-size>
#SBATCH --time=<appropriate-HH:MM:SS>
#SBATCH --output=/home/<user>/<project>-runs/logs/%x-%j.out
#SBATCH --error=/home/<user>/<project>-runs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
: "${PROJECT_ROOT:?Set repository path}"
: "${RUN_PYTHON:?Set interpreter path}"
: "${OUTPUT_ROOT:?Set persistent output directory}"
test -x "$RUN_PYTHON"
test -d "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Insert provenance (omit GPU-only inspection) and validate local inputs.
srun "$RUN_PYTHON" -u <cpu_entrypoint.py> <nonsecret-arguments>
```

CPU-only jobs omit `--gres=gpu:1`. CPU tests, preprocessing, compilation and
large package builds are still allocation work. Match worker/thread counts to
the allocation rather than the host's CPU inventory.

### Submission, monitoring and interactive work

Set nonsecret job configuration explicitly in your submission shell, for example
`export PROJECT_ROOT=... RUN_PYTHON=... OUTPUT_ROOT=...`. Create only the intended
small directory structure before submission:

```bash
mkdir -p "$OUTPUT_ROOT/logs"
sbatch --output="$OUTPUT_ROOT/logs/%x-%j.out" \
       --error="$OUTPUT_ROOT/logs/%x-%j.err" script.sbatch
sbatch --parsable script.sbatch   # alternative submission, not a second copy
squeue -u "$USER"
squeue -j JOBID -o '%.18i %.9P %.8T %.10M %.6D %R'
scontrol show job JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS -P
scancel JOBID
```

Choose **one** submission command. Save the returned job ID (`--parsable` may
include a `;cluster` suffix). `#SBATCH` directives do not expand shell variables;
use literal edited paths or the shell-expanded CLI log options above. Submit
from a deliberate working directory or specify `--chdir` explicitly.

For a short interactive GPU check, the installed `srun` and current `mat` GPU
configuration support this request form; actual acceptance depends on policy
and free resources (no interactive allocation was launched for this guide):

```bash
srun --partition=mat --nodes=1 --ntasks=1 --gres=gpu:1 \
     --cpus-per-task=2 --mem=12G --time=00:10:00 --pty bash
```

For CPU-only inspection omit `--gres=gpu:1` and size CPU/RAM accordingly. Check
`hostname` and the allocation after the shell starts. Exit when finished; use
`sbatch` for long experiments that need to survive disconnection.

## 7. Reproducibility prologue inside the allocated job

Set the listed paths, `SEED` and `CONFIG_PATH` explicitly for your project. Keep
the config/CLI free of secrets. Use the block at the top of an important job,
after entering `PROJECT_ROOT`; stdout goes to the explicit persistent job log.
For CPU-only jobs omit GPU inspection and make PyTorch inspection optional if
the project does not use it.

```bash
date --iso-8601=seconds
hostname
pwd
git rev-parse HEAD
git status --short
printf 'SLURM_JOB_ID=%s\nSLURM_JOB_NODELIST=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "${SLURM_JOB_ID:-unset}" "${SLURM_JOB_NODELIST:-unset}" \
  "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'CPUS_PER_TASK=%s\nCPUS_ON_NODE=%s\nMEM_PER_NODE_MiB=%s\nMEM_PER_CPU_MiB=%s\n' \
  "${SLURM_CPUS_PER_TASK:-unset}" "${SLURM_CPUS_ON_NODE:-unset}" \
  "${SLURM_MEM_PER_NODE:-unset}" "${SLURM_MEM_PER_CPU:-unset}"
scontrol show job "$SLURM_JOB_ID"
printf 'PROJECT_ROOT=%s\nDATA_ROOT=%s\nOUTPUT_ROOT=%s\nCACHE_ROOT=%s\nCHECKPOINT_ROOT=%s\n' \
  "$PROJECT_ROOT" "${DATA_ROOT:-unset}" "$OUTPUT_ROOT" \
  "${CACHE_ROOT:-unset}" "${CHECKPOINT_ROOT:-unset}"
printf 'SEED=%s\nCONFIG_PATH=%s\nRUN_PYTHON=%s\n' \
  "${SEED:?Set seed}" "${CONFIG_PATH:?Set config path}" "$RUN_PYTHON"
test -r "$CONFIG_PATH"
sha256sum "$CONFIG_PATH"
"$RUN_PYTHON" --version
nvidia-smi
"$RUN_PYTHON" -B - <<'PY'
import sys
import torch
print("python_executable:", sys.executable)
print("python_version:", sys.version)
print("torch_version:", torch.__version__)
print("torch_cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("GPU job has no usable CUDA device")
for device in range(torch.cuda.device_count()):
    print("gpu_name:", torch.cuda.get_device_name(device))
    print("gpu_properties:", torch.cuda.get_device_properties(device))
PY
```

Record the exact nonsecret CLI arguments and effective model/training/solver
configuration as a structured run manifest. A git commit alone is insufficient
with a dirty tree: retain the relevant patch or a source snapshot/hashes. Also
record data/tokenizer/checkpoint identity, seed, precision, deterministic flags,
batch order and resume position. Keep job ID, allocation details and output path
in experiment reports. Measure GPU operations with correct synchronization.
Do not dump all environment variables or print credentials in command lines.

## 8. New-project bootstrap checklist for agents

1. Read `LAGRANGE.md` completely; identify dated observations that need checking.
2. Read the repository's `AGENTS.md`, README and relevant setup instructions.
3. Run `hostname`, `whoami`, `pwd`; establish frontend versus allocated compute.
4. Inspect `df -hT / /home /dev/shm` and `findmnt -T` for intended paths; establish quota before large writes.
5. Identify an explicit interpreter; inspect versions without mutating it.
6. Verify named local datasets/tokenizers/checkpoints and required prefix/range; fail on missing assets.
7. Inspect partition, node and your account/QoS limits; choose modest resources.
8. Create a project-owned persistent output/log root outside the repository.
9. Prepare lightweight smoke/unit tests; route numerical or memory-heavy tests to SLURM.
10. Prepare batch scripts with explicit interpreter, paths, resource/time limits and offline behavior.
11. Include provenance; submit once and retain job ID, script and configuration.
12. Inspect `squeue`, logs and `sacct`; validate the smoke result before scaling.
13. Never silently change dependencies, data, hardware assumptions or experimental configuration.

## 9. Agent operating and security rules

- Use SLURM for all heavy CPU/GPU work; never test it on the frontend first.
- Inspect before assuming paths, tools, Internet connectivity or device access.
- Do not assume all physical GPUs are allocatable, or that `/scratch` exists.
- Install/download only for explicit need; preserve stable reference environments
  unless explicitly instructed to change them.
- Avoid recursive scans of huge filesystems and deletion of unknown home/shared files.
- Keep large artifacts outside source; preserve compact failure metadata and logs.
- Preserve job IDs and check status before resubmitting after disconnection.
- Fail clearly on missing prerequisites or incompatible environments.
- Keep project changes separate from cluster/environment changes; do not change
  scheduler, mount, driver, networking or shell settings as a hidden workaround.
- Never store passwords, API secrets, tokens or SSH private keys in this guide,
  source, manifests or logs. Inspect only an allowlist of nonsecret environment
  variables; never print an entire environment or private-key contents.

## 10. Failure and debugging guide

| Symptom | Safe first checks / action |
| --- | --- |
| Job pending | `squeue -j JOBID -o '%.18i %.8T %R'`; `scontrol show job JOBID`. Distinguish resources, priority and QoS limits. `QOSMaxGRESPerUser` can mean your other job consumes the GPU allowance. Do not bypass policy. |
| Failed immediately | Read explicit `.out`/`.err`; `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed -P`. Check literal paths, pre-existing log directories, interpreter, executable/read permissions, config and working directory. |
| CUDA not visible | Inside allocation inspect `hostname`, job `AllocTRES`, `CUDA_VISIBLE_DEVICES`, `nvidia-smi`, chosen interpreter and `torch.cuda.is_available()`. Check GPU request; do not force device visibility. |
| OOM / killed process | Check accounting state, batch/step `MaxRSS` (may be unavailable), host-memory request and GPU-memory diagnostics. Host/cgroup OOM differs from CUDA VRAM OOM; `/dev/shm` can also fill. Adjust only deliberate workload/resources within policy. |
| Python environment mismatch | Compare explicit executable and package/runtime versions with manifest. Verify batch activation/path assumptions; do not repair by installing into another project's environment. |
| Missing dataset/tokenizer/checkpoint | Check the exact configured path and permissions on compute. Stop; do not fetch remotely or silently substitute data. |
| SSH disconnected / job still running | Reconnect and use `squeue -u "$USER"`, then inspect persistent logs. `sbatch` jobs survive; do not duplicate the run. |
| Job disappeared from `squeue` | Query `sacct` including `.batch`/step rows for completion/failure/timeout/cancellation. Accounting may lag; disappearance is not proof of success. |
| Disk full | `df -hT` and `df -i` on the actual destination; check quota. Inspect known run sizes only. Stop large writes; do not blindly delete caches or shared data. |
| CUDA/PyTorch mismatch | Inside allocation compare driver, toolkit, PyTorch runtime and failing operation/dtype. Preserve traceback; run a tiny allocated smoke test. Do not upgrade drivers or force library paths as an unreviewed fix. |

Use `tail -n 100 /explicit/path/job.err` or bounded `rg -n`/`grep -n` on named
logs. Cancel only jobs you intend to stop with `scancel JOBID`. Retain evidence
before cleanup or retries; numerical failures must not be hidden by restarts.

## 11. Daily command reference

Replace placeholders; GPU commands belong inside allocations.

| Purpose | Commands |
| --- | --- |
| Connection / identity | `ssh <user>@lagrangectl.mat.uniroma1.it`; `hostname`; `whoami`; `pwd` |
| Git | `git status --short`; `git rev-parse HEAD`; `git diff --stat`; `git diff --check` |
| Lightweight environment | `command -v python3 sbatch srun nvcc`; `/path/to/env/bin/python -I -S -B --version`; `nvcc --version` |
| Filesystem | `df -hT / /home /dev/shm`; `df -i /home`; `findmnt -T /explicit/path`; `ls -ld /explicit/path` |
| SLURM submit | `sbatch --parsable script.sbatch` (save ID; submit once) |
| SLURM inventory / status | `sinfo`; `squeue -u "$USER"`; `scontrol show job JOBID`; `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS -P` |
| Cancel | `scancel JOBID` |
| Job logs | `tail -n 100 /explicit/path/job.out`; `tail -f /explicit/path/job.err`; `rg -n -e ERROR -e Traceback -e OOM /explicit/path/job.err` |
| GPU, inside allocation | `nvidia-smi`; `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`; PyTorch version/device-properties block above |
