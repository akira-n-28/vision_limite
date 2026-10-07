# vision_limite (Limite-VL) su Lagrange — istruzioni per Claude Code

Progetto: aggiungere la visione a Limite 1B - Violetto (encoder SigLIP2 + proiettore +
decoder Limite) su una H100 NVL del cluster Lagrange. Il piano operativo, con fasi,
budget e criteri di avanzamento, è qui:
https://claude.ai/code/artifact/d3fe8c7e-5458-49c5-88e3-d113ce6d2483

## All'inizio di ogni sessione

1. Leggere **tutto** `LAGRANGE.md` (guida del cluster, incluso il chiarimento sui download
   del 2026-10-06), poi questo file.
2. `hostname; whoami; pwd; echo "${SLURM_JOB_ID:-nessun job}"`. Claude Code lavora sul
   frontend `lagrangectl`: se l'host è un altro, fermarsi e chiedere.
3. `squeue -u "$USER"` e `tail -n 20 runs/ledger.tsv`: sapere cosa è già in corso prima
   di lanciare qualsiasi cosa. Mai duplicare un job dopo una disconnessione.
4. `source config/paths.env`.

## Regole che non si negoziano

- Il frontend ha 3.8 GiB di RAM condivisi da tutti. Lì si fanno solo editing, git,
  `sbatch`/`squeue`/`sacct`/`scancel` sui propri job, lettura di log con `tail`/`grep`
  limitati, script di sola libreria standard che non caricano dati (manifest, link).
- **Mai sul frontend:** `import torch`, test, tokenizzazione, parsing di Parquet/JSON grandi,
  hash completi, estrazione di archivi, installazioni, compilazioni, caricamento di pesi.
  Tutto questo va in un job SLURM, anche "solo per provare".
- I job di calcolo sono offline. Niente proxy, tunnel, download nei job. Un file mancante
  fa fallire il job con un messaggio chiaro: non si sostituisce e non si scarica al volo.
- Download solo con `scripts/fetch.sh` / `scripts/fetch_manifest.sh` (uno alla volta,
  10 MiB/s, soglie di memoria della guida già codificate); l'unico altro comando di rete
  ammesso è `uv lock` tramite `scripts/setup_env.sh lock`, che lo esegue sotto
  `scripts/guarded_run.sh`. Non lanciare `curl`, `wget`, `pip download`, `uv sync`,
  `huggingface-cli` o `hf_transfer` direttamente. Non abbassare le soglie e non riavviare
  in loop dopo uno stop per memoria: riferire all'utente.
- Destinazioni solo dentro `/home/prignano/vision_limite`, nei percorsi di `config/paths.env`; mai
  `/dev/shm`, mai la root del frontend, mai file grandi dentro il repository.
- QoS `castelnuovo_qos`: in totale 1 GPU, 12 CPU, 150 GB di RAM fra tutti i job attivi.
  Non usare altre QoS o account, non toccare `CUDA_VISIBLE_DEVICES`, non usare
  `LD_LIBRARY_PATH` per forzare librerie CUDA.
- **Una sola cartella.** Tutto il progetto vive in `/home/prignano/vision_limite` (la cartella da
  cui gira Claude Code) e usa solo ciò che sta lì dentro: niente interpreti, venv, codice, cache o
  dati di altri progetti (es. `/home/prignano/modded-nanogpt`, `/home/prignano/pg`), nemmeno
  in sola lettura. Quello che serve si scarica con gli script del kit o si scrive nel repository.
- Non cancellare file che non ha creato questo progetto. Niente `du`/`find`/`rg` ricorsivi
  su tutta `/home`.
- Nessun segreto in comandi, log o repository. Il token Hugging Face sta in
  `$HF_TOKEN_FILE` (permessi 600) e viene letto dagli script; non stamparlo mai.

## Percorsi

Tutto sotto `/home/prignano/vision_limite`, definito in `config/paths.env`. La radice è il
repository git; le cartelle pesanti sono sottocartelle escluse da `.gitignore`:

| Cartella | Contenuto | In git |
| --- | --- | --- |
| `./` (CLAUDE.md, config, scripts, slurm, src, tests, env, downloads, runs/ledger.tsv) | codice e configurazione | sì |
| `data/` | dataset, shard, log e stato dei download, specifiche degli snapshot | no |
| `hf/` | cache Hugging Face con le revisioni fissate | no |
| `wheels/`, `tools/` | wheel scaricate, binario di uv | no |
| `venvs/` | `build-<jobid>` e il link `current` al build adottato | no |
| `cache/` | cache di uv, Triton, Inductor, vLLM | no |
| `outputs/` | log dei job, checkpoint, manifest dei run, report | no |
| `secrets/` | token Hugging Face (cartella 700, file 600) | no |
| `third_party/<repo>/` | repository esterni clonati, commit annotato nel README | no |

Mai `git add -A` senza controllare `git status`: nulla di quelle cartelle deve entrare in git.
Le cartelle dei log devono esistere **prima** di `sbatch` (`mkdir -p outputs/logs`).

## Download

1. Controllare lo spazio: `df -hT /home; df -i /home`. Nessuna quota personale (confermato
   dall'utente il 2026-10-07), ma /home è condiviso: avvisare l'utente prima che il progetto
   superi 1 TB in totale o se lo spazio libero su /home scende sotto 2 TB.
2. Generare i manifest con `downloads/build_manifests.sh <gruppo>` (righe di
   `downloads/sources.tsv`; quelle con VERIFICARE vanno prima corrette controllando la
   pagina del repository Hugging Face). Per le wheel: `scripts/uvlock_manifest.py`.
3. Avviare: `nohup setsid scripts/fetch_manifest.sh downloads/manifests/<gruppo>.tsv >
   /home/prignano/vision_limite/data/logs/fetch-<data>.log 2>&1 &` (creare prima la cartella).
   Un solo `fetch_manifest.sh` alla volta. Stop gentile: `touch /home/prignano/vision_limite/data/.fetch.stop`.
4. A download finito: `scripts/link_hf_snapshot.py` (crea gli snapshot nella cache HF) e
   `sbatch --parsable slurm/verify_hashes.sbatch downloads/manifests/<gruppo>.tsv`.
5. Gli archivi (es. `images.zip` di LLaVA-Pretrain) si estraggono e si convertono in shard
   da circa 1 GB in un job CPU. Mai milioni di file piccoli su GlusterFS.

Ordine di priorità: p1 (Violetto, SigLIP2, benchmark testuali alle revisioni del
valutatore) → wheel e `uv` → p3 (dati F1) → p4 (teacher, VLM di riferimento) → p5 (dati
F2 e RL) → p6 (benchmark visivi) → p7 (FineVision, solo sottoinsiemi scelti).

## Ambiente (venv in vision_limite/venvs, senza carico sul frontend)

Tutto passa da `scripts/setup_env.sh`, una fase per comando; `status` dice sempre il prossimo passo.

1. `slurm/f0_gpu_probe.sbatch` → driver ≥ 580: copiare `env/pyproject.cu13.template.toml` in
   `env/pyproject.toml` (e il `uv.lock` ufficiale in `env/uv.lock`); driver 570: partire da
   `env/pyproject.cu12.template.toml` verificando l'URL di vLLM. Chiedere conferma all'utente.
2. `scripts/setup_env.sh uv` — scarica il binario di uv con `fetch.sh` e invia
   `slurm/extract_tools.sbatch` (verifica SHA-256 ed estrazione in allocazione).
3. `scripts/setup_env.sh lock` — `uv lock --no-build` sotto `scripts/guarded_run.sh`: un download
   alla volta, nessuna build, stop se la RSS supera 128 MiB o MemAvailable scende sotto 768 MiB.
   Misurato su un grafo con vLLM e transformers (226 pacchetti): circa 45 MB scaricati, RSS
   massima 116-127 MiB, quindi vicino alla soglia. Se viene fermato (codici 20/21) **non
   riprovare in loop**: chiedere all'utente di generare `env/uv.lock` sul portatile
   (`cd env && uv lock --no-build`), fare pull e poi `SKIP_RESOLVE=1 scripts/setup_env.sh lock`.
4. `scripts/setup_env.sh wheels` — download delle wheel in background (circa 4 GB con torch, ~7 minuti
   a 10 MiB/s), sopravvive alla disconnessione. Controllare con `status`.
5. `scripts/setup_env.sh install` — invia `slurm/env_install.sbatch`: venv offline in
   `/home/prignano/vision_limite/venvs/build-<jobid>`, hash verificati, plugin Limite compilato offline.
6. Leggere il log del job; con l'ok dell'utente `ln -sfn build-<jobid> venvs/current`.
7. Primo test GPU dell'ambiente: `slurm/f0_checks.sbatch`.

Mai `pip install`, `uv sync`, `uv pip install` o creazione di venv sul frontend.

## SLURM

Risorse di partenza (stimare sempre il carico reale; il totale non deve superare la QoS):

| Tipo di job | GPU | CPU | RAM | Tempo |
| --- | --- | --- | --- | --- |
| Sonda / smoke | 1 | 2 | 12G | 00:15:00 |
| Verifiche F0 | 1 | 8 | 64G | 01:30:00 |
| Generazione con vLLM (teacher o Violetto) | 1 | 8 | 96G | ≤ 12:00:00, a shard con ripresa |
| Training F1/F2/F4 | 1 | 8 | 120G | ≤ 24:00:00, con ripresa dal checkpoint |
| CPU: hash, estrazione, shard, render | 0 | 2-4 | 4-24G | ≤ 12:00:00 |

Procedura per ogni job:

1. Script in `slurm/`, con percorsi letterali nelle direttive `#SBATCH` e il prologo
   (`source slurm/prologue.sh; lagrange_prologue gpu|cpu`).
2. Un solo invio: `jobid=$(sbatch --parsable slurm/<script>.sbatch [argomenti])`, poi
   annotare subito in `runs/ledger.tsv`: data, job ID, nome, `submitted`, scopo e config.
3. Controllo: `squeue -j $jobid -o '%.18i %.8T %.10M %R'`, `tail -n 50` sul log in
   `/home/prignano/vision_limite/outputs/logs/`, a fine job `sacct -j $jobid --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS -P`.
4. Pending per `QOSMaxGRESPerUser` = un altro nostro job usa la GPU: aspettare, non aggirare.
5. Job lunghi: salvare checkpoint e posizione nel dataset almeno ogni ora; il job
   successivo riparte dall'ultimo checkpoint (`--dependency=afterany:<jobid>` per concatenarli).
6. Ogni run scrive un manifest (config effettiva, commit git, eventuale diff, dati con hash,
   seed, job ID) in `/home/prignano/vision_limite/outputs/<run>/manifest.json`.

## Fasi e decisioni

Le fasi F0-F4 e i criteri di avanzamento sono nel piano. Claude Code **non passa alla fase
successiva da solo**: a fine fase riporta all'utente le metriche (AIME 2026 testuale,
benchmark visivi, test dell'immagine bianca) e aspetta l'ok. Stesso per: adottare un
ambiente, superare 1 TB di spazio totale, lanciare job più lunghi di 24 ore, cambiare dati o
iperparametri rispetto al piano.

## Fatti verificati sul codice di Limite (`modeling_limite.py` @ b1f3d572)

- `LimiteModel.forward` rifiuta `inputs_embeds` (ValueError): servono `input_ids` perché i
  value embeddings sono un secondo lookup per token. Il modello VL deve avere un forward
  proprio: `embed_tokens(input_ids)` → sostituzione delle posizioni `<|image_pad|>` con
  l'output del proiettore → `rms_norm` → ciclo dei layer copiato dall'originale.
- Lo stato iniziale `rms_norm(embed_tokens(input_ids))` è anche il tap x(0) del MUDD.
- `value_embeds(input_ids)`: azzerare il contributo sulle posizioni immagine.
- Nessun supporto dichiarato al gradient checkpointing: wrapper per layer con
  `torch.utils.checkpoint` (non rientrante), validato da `scripts/f0_checks.py`.
- La loss ufficiale materializza tutti i logit in fp32: usare la loss a blocchi con
  `model._softcapped_logits` (vedi `chunked_ce` in `scripts/f0_checks.py`).
- Token visivi già nel vocabolario: `<|vision_start|>` 151652, `<|image_pad|>` 151655,
  `<|vision_end|>` 151653.
- Copiare i file remoti del modello in `third_party/limite-modeling/` con il loro hash;
  il training non deve scaricare codice con `trust_remote_code` a runtime.

## Ottimizzatore

- F1: AdamW (proiettore e righe di embedding dei token visivi).
- F2: NorMuon sulle matrici nascoste e AdamW sugli altri gruppi, come nell'SFT di Paradigma,
  con i picchi della Tabella 1 scalati al 15% (matrici nascoste circa 0.0027). Proiettore in
  AdamW. Ablation breve contro AdamW a 1e-5 prima del run completo.
- F4 (RL): AdamW.
- Implementazione in `src/limite_vl/optim/` (Algoritmo 1 del report; coefficienti di Polar
  Express da modded-nanogpt @ 4ea6b937, MIT). Test: `slurm/tests_cpu.sbatch`. Le regole sui
  nomi dei parametri in `groups.py` sono ipotesi: verificarle con l'output di `f0_checks.py`
  (sezione `7_param_groups`) prima di F2.

## Struttura del repository

- `CLAUDE.md`, `LAGRANGE.md` — istruzioni
- `config/paths.env` — percorsi e variabili offline
- `downloads/` — `sources.tsv`, `build_manifests.sh`, `manifests/`
- `scripts/` — ambiente (`setup_env.sh`, `guarded_run.sh`, `uvlock_manifest.py`), download
  (`fetch.sh`, `fetch_manifest.sh`, `hf_manifest.py`, `link_hf_snapshot.py`), verifica
  (`verify_manifest.py`), F0 (`f0_checks.py`)
- `slurm/` — `prologue.sh` e gli script `.sbatch`
- `env/` — pyproject e lock dell'ambiente
- `third_party/` — `limite-violetto` (plugin vLLM e valutatore) e copia del codice del modello
- `src/limite_vl/` — codice del progetto (`optim/`: NorMuon e gruppi della Tabella 1)
- `tests/` — test CPU (`test_normuon.py`)
- `runs/ledger.tsv` — registro dei job (testo piccolo, versionato)
