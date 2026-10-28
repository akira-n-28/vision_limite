# vision_limite (Limite-VL) su Lagrange — istruzioni per Claude Code

Progetto: aggiungere la visione a Limite 1B - Violetto su una H100 NVL del cluster Lagrange, per leggere immagini,
documenti e formule, collegare più immagini e risolvere problemi di matematica e scienze.

**Riferimento corrente:** [`PIPELINE_ADDESTRAMENTO_LIMITE_VL.md`](PIPELINE_ADDESTRAMENTO_LIMITE_VL.md), versione 1.4
del 2026-10-28. Contiene obiettivi, fasi F0-F4, protocollo dei dati (gold, split, decontaminazione, audit) e decisioni
aperte. È un progetto: non è una configurazione eseguibile e non autorizza a lanciare job.

- Stato reale del codice e prossimi interventi: [`docs/F0_IMPLEMENTATION_PLAN.md`](docs/F0_IMPLEMENTATION_PLAN.md).
- Migrazione, artefatti e percorsi non toccati: [`docs/MIGRATION_REPORT.md`](docs/MIGRATION_REPORT.md),
  [`docs/ARTIFACT_INVENTORY.tsv`](docs/ARTIFACT_INVENTORY.tsv), [`docs/REPOSITORY_INDEX.tsv`](docs/REPOSITORY_INDEX.tsv).
- Rilievi delle revisioni indipendenti: [`docs/REVIEW_ACTIONS.md`](docs/REVIEW_ACTIONS.md). I rapporti in `docs/reviews/`
  restano immutati.

La strategia precedente (F1, F1b, F1c, ponte teacher di F2, gate f2-cert, replay generato da Violetto) è in
`archive/pipeline-precedente/`, con la copia integrale delle vecchie istruzioni. Non si riprendono i run storici, non
si rilanciano i loro launcher e non si usano i vecchi esiti come gold o come prova che una scelta sia validata.

Il progetto è **solo di ricerca** (decisione dell'utente, 2026-10-12): la licenza di un dataset non è un
motivo per escluderlo (es. MAVIS-Geometry senza licenza dichiarata, `images.zip` di LLaVA-Pretrain "per la
ricerca"). Si citano sempre le fonti, i dati non si ridistribuiscono e si rispettano le richieste di
rimozione. Se l'utente parla di pubblicare i pesi o di uso commerciale, riaprire la questione.

## Dove si lavora

| Luogo | Cosa si fa |
| --- | --- |
| Portatile dell'utente | Stesura di pipeline e revisioni, consegnate tramite GitHub; `uv lock` se il frontend lo ferma |
| Frontend `lagrangectl` (Claude Code) | Editing, git, `sbatch`/`squeue`/`sacct`/`scancel` sui propri job, download con gli script del kit, script di sola libreria standard che non caricano dati |
| Job SLURM su `lagrange0` | Tutto il calcolo: test, import di torch, tokenizzazione, hash, estrazione, shard, training, valutazione |

Il pacchetto del portatile contiene documenti, non codice da sovrapporre in blocco alla cartella del cluster.

## All'inizio di ogni sessione

1. Leggere **tutto** `LAGRANGE.md` (guida del cluster, incluso il chiarimento sui download
   del 2026-10-06), poi questo file, poi le sezioni della pipeline e di `docs/F0_IMPLEMENTATION_PLAN.md` che
   riguardano il lavoro richiesto.
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
  su tutta `/home`. Dataset, checkpoint e ambienti del progetto non si cancellano senza una proposta
  (motivo, dimensione, dipendenze, recuperabilità) approvata dall'utente.
- Nessun segreto in comandi, log o repository. Il token Hugging Face sta in
  `$HF_TOKEN_FILE` (permessi 600) e viene letto dagli script; non stamparlo mai.

## Percorsi

Tutto sotto `/home/prignano/vision_limite`, definito in `config/paths.env`. La radice è il
repository git; le cartelle pesanti sono sottocartelle escluse da `.gitignore`:

| Cartella | Contenuto | In git |
| --- | --- | --- |
| `./` (CLAUDE.md, pipeline, config, scripts, slurm, src, tests, env, downloads, docs, runs/ledger.tsv) | codice, configurazione e documenti correnti | sì |
| `archive/pipeline-precedente/` | ricette, launcher e istruzioni della strategia precedente (`MOVES.tsv`) | sì |
| `data/` | dataset, shard, liste di decontaminazione, log e stato dei download, specifiche degli snapshot | no |
| `hf/` | cache Hugging Face con le revisioni fissate | no |
| `wheels/`, `tools/` | wheel scaricate, binario di uv | no |
| `venvs/` | `build-<jobid>` e i link `current`, `evals-current` ai build adottati | no |
| `cache/` | cache di uv, Triton, Inductor, vLLM | no |
| `outputs/` | log dei job, checkpoint, manifest dei run, report | no |
| `secrets/` | token Hugging Face (cartella 700, file 600) | no |
| `third_party/<repo>/` | repository esterni clonati, commit annotato nel README | no |

L'inventario degli artefatti fuori da git (stato, dipendenze, dimensioni, azione proposta) è
`docs/ARTIFACT_INVENTORY.tsv`. Mai `git add -A` senza controllare `git status`: nulla delle cartelle escluse deve
entrare in git. Le cartelle dei log devono esistere **prima** di `sbatch` (`mkdir -p outputs/logs`).

## Download

Catalogo: `downloads/sources.tsv` (identità, revisioni, filtri; letto da `build_manifests.sh`) e
`downloads/catalog_status.tsv` (disponibilità locale, verifica degli hash, ruolo nella pipeline, ammissione).
Scaricato non vuol dire ammesso: ogni sorgente resta candidata fino ai controlli di F0 (pipeline §4). Nuovi
trasferimenti solo su richiesta esplicita dell'utente.

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

## Ambiente (venv in vision_limite/venvs, senza carico sul frontend)

Ambiente adottato: `venvs/current` → `build-29382` (Python 3.12, torch 2.11.0+cu129, transformers 5.6.2, vLLM
0.26.0+cu129, plugin Limite); client del valutatore in `venvs/evals-current` → `evals-build-29390`. Lock e pyproject
in `env/` e `env-evals/` sono artefatti riproducibili. Un encoder candidato diverso da SigLIP2 (ViT di Qwen) può
richiedere un'altra versione di transformers o di vLLM: va verificato prima del confronto, con un nuovo build e l'ok
dell'utente, senza toccare quello adottato.

Tutto passa da `scripts/setup_env.sh`, una fase per comando; `status` dice sempre il prossimo passo.

1. `slurm/f0_gpu_probe.sbatch` → driver ≥ 580: copiare `env/pyproject.cu13.template.toml` in
   `env/pyproject.toml` (e il `uv.lock` ufficiale in `env/uv.lock`); driver 570: partire da
   `env/pyproject.cu12.template.toml` verificando l'URL di vLLM. Chiedere conferma all'utente.
   Su Lagrange (driver 570.211.01) vale il caso B: `env/pyproject.toml` prende torch,
   torchvision e torchaudio dall'indice cu129. Le build PyPI di torchvision e torchaudio sono
   per il torch cu130 e con torch cu129 non caricano le estensioni (job 29375).
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
| Generazione con vLLM | 1 | 8 | 96G | ≤ 12:00:00, a shard con ripresa |
| Training (piloti e fasi) | 1 | 8 | 120G | ≤ 24:00:00, con ripresa dal checkpoint |
| CPU: test, hash, estrazione, shard, render | 0 | 2-4 | 4-24G | ≤ 12:00:00 |

Le CPU di Slurm su lagrange0 sono thread, non core: `--cpus-per-task=4` dà due coppie di thread
sorelli, cioè 2 core fisici (job 29418, CPU 0, 1, 48, 49). `srun` le vede tutte anche senza
`--cpus-per-task`. Per i pool di processi (shard, DataLoader) contare i core fisici, non le CPU.
Nei job non scrivere un file per campione: il job 29414 ha creato 168 mila file in pochi minuti e la
cancellazione su GlusterFS ha richiesto più di un'ora.

Procedura per ogni job:

1. Script in `slurm/`, con percorsi letterali nelle direttive `#SBATCH` e il prologo
   (`source slurm/prologue.sh; lagrange_prologue gpu|cpu`). I launcher in `archive/` non si usano.
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

Sequenza (pipeline §2): F0 dati, protocollo e piloti di encoder/proiettore/integrazione → F1 allineamento →
F2 adattamento congiunto con ViT addestrabile → F3 specializzazione → F3-M interpolazione facoltativa → F4 RL
verificabile → F4-M facoltativa → valutazione finale. **Oggi il progetto è in F0 preparatorio**: nessun pilota,
training o audit è approvato. Il primo perimetro di lavoro è in `docs/F0_IMPLEMENTATION_PLAN.md`.

- Claude Code **non passa alla fase successiva da solo** e non avvia senza ok esplicito dell'utente: training,
  piloti GPU, generazioni teacher, audit con agenti, download, scansioni pesanti. Stesso per: adottare un
  ambiente, superare 1 TB di spazio totale, lanciare job più lunghi di 24 ore, cambiare dati, architettura o
  iperparametri approvati. A fine fase riporta le misure per capacità (pipeline §15) e aspetta.
- **Decisioni aperte**, da chiudere col protocollo della pipeline (§§3.3-3.5, 6.2-6.5, 16, 18) e non adottate per
  default: checkpoint Qwen da cui estrarre il ViT; budget in ore GPU o euro; contratto di risposta diretta/ragionata;
  layer del ViT, proiettore e fusione; DeepStack; rami visivi verso MUDD e value embeddings; precisione; risoluzione e
  token per immagine; mix, volumi e durate; metriche primarie, margini e soglie; budget degli audit per sorgente.
- La baseline implementata (SigLIP2 so400m NaFlex, ultimo hidden state, pixel shuffle 2x2 e MLP 4608 → 2560 → 1280,
  feature visive negli embedding prima della norm iniziale, value embeddings azzerati sulle posizioni immagine) è il
  riferimento dei confronti, non una scelta approvata.
- Controlli dei dati (pipeline §5): prima ciò che è vero per costruzione e i controlli del codice sul 100%, agenti solo
  su campioni e casi selezionati. Ogni audit con agenti ha budget di chiamate e token dichiarato e approvato prima; il
  consumo va nel ledger a ogni giro; al tetto ci si ferma e si riferisce. Il consenso fra agenti non è verità. Gli
  audit precedenti non contano come audit (decisione del 2026-10-26): orientano i controlli nuovi, non giustificano
  decisioni.
- Le soglie numeriche e le ricette della strategia precedente (≤ 1% sulle risposte e ≤ 2% sui fatti della regola del
  2026-10-19, 300M token di F1-clean, picchi NorMuon al 15%) non valgono più (confermato dall'utente il 2026-10-28):
  soglie, volumi e iperparametri si fissano prima dei piloti, con l'utente.

## Fatti verificati sul codice di Limite (`modeling_limite.py` @ b1f3d572)

Evidenza tecnica sul checkpoint fissato e sulla baseline. Non decide l'architettura; dove la pipeline chiede una
nuova verifica (precisione, controlli di §3.2) i vecchi job orientano ma non bastano.

- `LimiteModel.forward` rifiuta `inputs_embeds` (ValueError): servono `input_ids` perché i
  value embeddings sono un secondo lookup per token. Il modello VL
  (`src/limite_vl/modeling.py`, scelta confermata dall'utente il 2026-10-09) non copia il ciclo
  dei layer. Usa due forward hook sul `LimiteForCausalLM` originale:
  - su `embed_tokens`: le posizioni `<|image_pad|>` ricevono l'output del proiettore prima della
    `rms_norm` iniziale; `<|vision_start|>` e `<|vision_end|>` ricevono le righe addestrabili
    `vision_token_embeds`;
  - su `value_embeds`: il contributo sulle posizioni `<|image_pad|>` viene azzerato
    (`VLConfig.image_value_embeds = "zero"`; `"token"` usa il vettore comune di `<|image_pad|>`).
  Ciclo dei layer, MUDD, cache e `generate()` restano quelli originali. Su solo testo gli
  hidden state sono identici bit per bit a Violetto, in train e in eval (job 29400).
- Lo stato iniziale `rms_norm(embed_tokens(input_ids))` è anche il tap x(0) del MUDD: per questo
  le feature visive entrano prima della norm.
- Nessun supporto al gradient checkpointing (`gradient_checkpointing_enable` dà ValueError):
  wrapper per layer con `torch.utils.checkpoint` (non rientrante), validato da
  `scripts/f0_checks.py` (job 29378: stessa loss, gradienti con diff. relativa 8·10⁻⁸).
- La loss ufficiale materializza tutti i logit in fp32: usare la loss a blocchi con
  `model._softcapped_logits` (vedi `chunked_ce` in `scripts/f0_checks.py`; job 29378:
  identica alla loss ufficiale).
- `forward` va chiamato con `use_cache=False` esplicito: senza, legge `config.use_cache`, che
  la config di Limite non definisce, e con transformers 5.6.2 dà AttributeError (job 29377).
  La generazione lo passa da sé.
- Token visivi già nel vocabolario: `<|vision_start|>` 151652, `<|image_pad|>` 151655,
  `<|vision_end|>` 151653.
- Il codice del modello si importa dalla copia in `third_party/limite-modeling/`, creata da
  `scripts/vendor_limite_modeling.py` e caricata da `limite_vl.limite_code.load()` solo se gli
  SHA-256 coincidono. Niente `trust_remote_code` nel training. Il tokenizer si carica come
  `Qwen2Tokenizer`: `AutoTokenizer` legge l'`auto_map` della config e chiede se eseguire codice
  remoto.
- In eval il MUDD usa costanti bf16 (`_inference_dense1` ecc.): con il decoder in fp32 i confronti
  numerici vanno fatti in modalità train. Limite non ha dropout.
- Contesto dello snapshot: 131.072 posizioni nominali, 48 layer (12 globali, 36 locali con finestra 1024). Il
  processor impone a parte un tetto software di 768 token per immagine (`processing.MAX_IMAGE_TOKENS`): sono tre
  vincoli distinti (pipeline §3.4).
- Packing: con `position_ids` che ripartono a ogni esempio e `attention_mask=None`, transformers
  costruisce la maschera a blocchi. Job 29402, in fp32: loss per token di due esempi
  impacchettati uguali a quelle degli esempi da soli entro 2.5·10⁻⁵; senza reset delle
  posizioni la differenza è 0.85. Le pagine di uno stesso esempio non vanno separate così (pipeline §3.4).
- Attenzione nel training: usare `flex_attention`, impostata a runtime con
  `config._attn_implementation`.
  - Con sdpa, finestra locale e packing diventano maschere dense.
  - Job 29402: in fp32 flex coincide con sdpa entro 4.7·10⁻⁵.
  - Riga da 32k, decoder addestrabile, gradient checkpointing: 15.4 mila token/s con flex
    contro 4.6 mila con sdpa.
  - Righe da 4k: nessun guadagno (12.9 mila contro 12.6 mila).
- `flash-attn` non è nell'ambiente; la copia dentro vLLM ha solo il forward.
- flex_attention e dynamo: transformers compila `create_block_mask` e dynamo ricompila per ogni tipo di
  maschera (righe impacchettate, layer globali e a finestra, prefill e decodifica). Con il limite di default
  (8) la validazione a tre sorgenti lo supera e le maschere tornano in eager: 2 mila token/s invece di
  16 mila (job 29441). `train_utils.allow_flex_recompiles()` lo alza a 64; se nel log compare
  `recompile_limit`, il run va fermato.
- Encoder SigLIP2 so400m: in bf16 si allontana molto da fp32 (job 29446: differenza relativa media 26%
  sull'ultimo hidden state, coseno 0.96; 13% dopo il proiettore di f1-mid) e l'output cambia del 5-13% con
  la composizione del batch (job 29445). In fp32 non dipende dal padding (1e-5). TF32: 1.9% (0.7% dopo il
  proiettore) a circa 4 volte il costo di bf16 sull'encoder. Default del codice `VLConfig.vision_precision = "tf32"`;
  misure con encoder congelato: con il ViT addestrabile la precisione va riverificata.
- SigLIP2 NaFlex è addestrato con sequenze fino a 1024 patch (articolo di SigLIP 2); la baseline usa fino a 2304
  patch per i diagrammi (`processing.DIAGRAM_MAX_PATCHES`). Risoluzione e token per immagine sono decisioni aperte.
- `encode_images` passa le immagini nell'encoder a gruppi di lunghezza simile (al più 32768 patch con il
  padding): con formule, foto e diagrammi nello stesso micro-batch il padding fino a 2304 patch sarebbe
  quasi tutto il lavoro. In fp32 coincide con il gruppo unico entro 7.6·10⁻⁷ (job 29447). Con il ViT non
  congelato il ramo autograd è già presente (niente `no_grad`).
- Gradient checkpointing sul modello VL (job 29402): in fp32 i gradienti del proiettore
  differiscono di 4·10⁻⁷. In bf16 la differenza (circa 1%) è uguale a quella fra due backward
  identici senza checkpointing, cioè è il non determinismo dei kernel.

## Formato delle risposte di Violetto

Fatti osservati sul checkpoint; il processor oggi li segue. Il contratto di risposta della nuova pipeline (risposta
diretta o ragionata, template, label, stop, prefill) va versionato prima del primo aggiornamento del decoder
(pipeline §3.3): il template dello snapshot impone sempre ragionamento passo passo e `\boxed{}` e rifiuta system
prompt diversi.

- Violetto chiude la risposta con `<|endoftext|>` (151643) subito dopo il testo, senza
  `<|im_end|>` (generazione greedy, job 29400). Il chat template invece scrive
  `<|im_end|>\n` dopo ogni turno dell'assistente.
- Il ragionamento è `<think>...</think>` senza a capo, seguito dalla risposta (soluzioni del
  valutatore, job 29393). Il template, con `reasoning_content`, scriverebbe
  `<think>\n...\n</think>\n\n`.
- `LimiteVLProcessor` segue il formato generato, non il template:
  - ultimo turno: contenuto + `<|endoftext|>`, con label;
  - turni intermedi: label solo sul contenuto; `<|im_end|>\n` resta nell'input senza label;
  - immagini sempre prima del testo nel messaggio utente (niente interleaving).
- `<think>` (151665) e `</think>` (151666) sono token aggiunti non speciali. Il processor li
  ammette solo nei testi dell'assistente e nel prefill.
- Il prefill `<think></think>` senza label delle didascalie (`F1_PREFILL`, opzione B del 2026-10-09) è uno strumento
  della F1 precedente. La pipeline lo ammette come strumento dichiarato di F1, ma la modalità d'uso finale si valuta
  senza prefill nascosti.

## Fatti sui dati locali

Struttura e provenienza, non qualità: nessuna sorgente è ammessa o gold. Stato per sorgente in
`downloads/catalog_status.tsv` e `docs/ARTIFACT_INVENTORY.tsv`.

- LLaVA-Pretrain @ 70f9d1e5: 29 shard in `data/shards/llava-pretrain/70f9d1e5/` (558 mila didascalie,
  circa 312 token per esempio con l'immagine a 1024 patch).
- MAVIS-Caption non esiste su Hugging Face; le didascalie vengono da CaraJ/MAVIS-Geometry @ 9c80fff3
  (`RuleBaseGeo.zip`), una parte di MAVIS-Instruct.
  - Struttura dello zip (job 29415): 199.603 cartelle `RuleBaseGeo/depthK/<i>/`, 50 mila per profondità
    0-3. Ogni cartella contiene `<n>_annotation.json` (`text_en`/`text_zh`, `choices`, `answer_index`) e tre
    immagini: `caption.jpg` (text_dominant, misure della didascalia disegnate), `text.jpg` (text_lite),
    `vision_dominant.jpg`. Nel 18% dei campioni `caption.jpg` e `vision_dominant.jpg` sono identiche. Le
    didascalie vanno solo con `caption.jpg`.
  - Shard (job 29419): 199.050 campioni in 11 tar in `data/shards/mavis-geometry/9c80fff3-caption/`, PNG già a
    2304 patch e JSON con `text_en`, scelte e dimensioni; esclusi `d3-3228` e 552 didascalie duplicate.
  - `src/limite_vl/f1_schema.py` e `f1_mavis.py` costruiscono descrizione canonica e domande da
    `diagram_generation` (job 30033, 30052: andata e ritorno su 199.050). La coerenza fra due descrizioni non prova
    la fedeltà ai pixel (`mavis_gen.py`). I valori razionali (denominatori 2 e 3) sono disegnati come decimali a due
    cifre (job 30049): valore esatto, visualizzato e richiesto vanno separati (pipeline §5.6).
- UniMER @ 2343ddd9 (job 29436, 29438): l'immagine `images/<indice>.png` ha come formula la riga `<indice>` di
  `train.txt`; le righe vuote sono HME100K (tolto per copyright) e le formule a mano di CROHME (meno dell'1%) non si
  separano per indice. Shard in `data/shards/unimer/2343ddd9/` (985.450 campioni, test spe/cpe/sce/hwe in `test/`),
  varianti `2343ddd9-x4` e `2343ddd9-dense` (con `lines/`). Formula senza gli spazi del formato im2latex
  (`latex.despace`), immagine al più 2x e 1024 patch (`processing.capped_size`). Controlli e rietichettature in
  `outputs/unimer_check/run4`, sovrapposizioni col test in `outputs/unimer_overlap/run2`.
- MMFineReason @ 7c93f724: le descrizioni di Qwen3-VL-235B sono testo di un modello, non fatti (circa 10% di
  affermazioni false nel campione del 2026-10-19, che non conta come audit); contiene Geo3K e la famiglia WeMath.
  Caption, risposta e CoT sono oggetti distinti (pipeline §5.7).
- Evidenze di decontaminazione storiche: `data/decontam/p6/` (benchmark visivi, regola del 2026-10-14),
  `data/decontam/f1eval/` (Geoperception e VisOnlyQA, job 30034-30039), `data/decontam/text/replay/` (replay testuale,
  regola del 2026-10-16), `outputs/split_duplicates/run1`. Le liste `exclude.txt` grezze sono solo candidati. Nel
  nuovo manifest si importano come evidenze con motivo, versione e stato (pipeline §5.8), senza applicarle alla
  cieca e senza cancellarle.

## Strumenti congelati

Restano importabili e testati, non si sviluppano più e non guidano l'ammissione dei nuovi dati: `scripts/f2_pilot.py`
(congela per sha256 sé stesso, `src/limite_vl/f2_claims.py`, `src/limite_vl/stem_audit.py` e le istruzioni usate),
`scripts/f2_certify.py`, `scripts/f2_build.py`, `src/limite_vl/f2_records.py` (gate fail-closed di `MixedRowStream` per
le sorgenti `f2_solution`), `scripts/stem_audit_*.py`, le istruzioni e la politica in `config/audit/` (indice e sha256 in
`config/audit/INDEX.md`). Non modificare quei file né `config/audit/f2_policy.json`: i manifest e i certificati storici
li citano per percorso e hash. `tests/test_f2_gate.py` importa torch: solo in un job.

## Ottimizzatore

Punto di partenza della pipeline (§14), da calibrare nei piloti: AdamW in tutte le fasi, con gruppi disgiunti per
decoder, ViT, collegamenti e righe visive da F2. NorMuon (`src/limite_vl/optim/`, Algoritmo 1 del report; coefficienti
di Polar Express da modded-nanogpt @ 4ea6b937, MIT; test in `tests/test_normuon.py`) resta un'alternativa. Nel
confronto degli encoder stesso ottimizzatore e stessa procedura per tutti i candidati. Le regole sui nomi dei
parametri in `groups.py` sono ipotesi da verificare con `f0_checks.py` (sezione `7_param_groups`).

## Struttura del repository

- `CLAUDE.md`, `LAGRANGE.md`, `README.md` — istruzioni; `PIPELINE_ADDESTRAMENTO_LIMITE_VL.md` — riferimento corrente;
  `ISTRUZIONI_MIGRAZIONE_LAGRANGE.md` — incarico della migrazione del 2026-10-28
- `docs/` — stato del repository, piano F0, inventario degli artefatti, indice dei file, rilievi e rapporti delle revisioni
- `config/paths.env` — percorsi e variabili offline; `config/runs/` — vuota finché non ci sono ricette collaudate;
  `config/audit/` — istruzioni e politica degli audit storici (congelate)
- `downloads/` — `sources.tsv`, `catalog_status.tsv`, `build_manifests.sh`, `manifests/`
- `scripts/` — ambiente, download, verifica, ispezione e shard dei dati, controlli deterministici, decontaminazione,
  valutazione, codice di Limite, F0; anche i trainer e gli strumenti F2/F1c storici, importati da altri strumenti
  (stato di ogni file in `docs/REPOSITORY_INDEX.tsv`)
- `slurm/` — `prologue.sh` e i launcher correnti; quelli storici sono in `archive/pipeline-precedente/slurm/`
- `env/`, `env-evals/` — pyproject e lock dell'ambiente principale e del client del valutatore
- `third_party/` — `limite-violetto` (plugin vLLM e valutatore @ 7ddad34f), `verifiers` @ d30a3f48,
  `limite-modeling` (copia verificata del codice del modello)
- `src/limite_vl/` — codice del progetto:
  - `modeling.py`: modello VL baseline (SigLIP2, proiettore, hook);
  - `processing.py`: immagini in patch a griglia pari, chat in token e label;
  - `collator.py`: packing e collate;
  - `data.py`: lettura degli shard tar ed esempi per tipo di sorgente (un'immagine per `Sample`);
  - `stream.py`: flusso misto e riprendibile di micro-batch, esclusioni e rietichettature;
  - `data_lists.py`: liste obbligatorie e loro provenienza;
  - `latex.py`, `latex_render.py`: formule e confronto fra trascrizioni;
  - `synth.py`, `mavis_gen.py`, `f1_schema.py`, `f1_mavis.py`, `mavis.py`: sintetici e MAVIS;
  - `decontam.py`, `transcripts.py`, `mmfine.py`, `replay.py`: decontaminazione, PixMo, MMFineReason, replay testuale;
  - `train_utils.py`: loss a blocchi e checkpointing per layer; `generation.py`: generazione;
  - `limite_code.py`: import verificato del codice di Limite; `optim/`: NorMuon e gruppi;
  - `provenance.py`: hash di testi e file, identità del codice, file nuovi senza sovrascrivere;
  - storici ma importati: `f1c.py`, `facts.py`, `f2.py`, `f2_records.py`, `f2_claims.py`, `stem_audit.py`.
- `tests/` — test CPU, lanciati tutti da `slurm/tests_cpu.sbatch`
- `archive/pipeline-precedente/` — ricette, launcher e istruzioni della strategia precedente
- `runs/ledger.tsv` — registro dei job (testo piccolo, versionato)
