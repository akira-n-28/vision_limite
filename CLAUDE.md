# vision_limite (Limite-VL) su Lagrange — istruzioni per Claude Code

Progetto: aggiungere la visione a Limite 1B - Violetto (encoder SigLIP2 + proiettore +
decoder Limite) su una H100 NVL del cluster Lagrange. Il piano operativo, con fasi,
budget e criteri di avanzamento, è qui:
https://claude.ai/code/artifact/d3fe8c7e-5458-49c5-88e3-d113ce6d2483

Il progetto è **solo di ricerca** (decisione dell'utente, 2026-10-12): la licenza di un dataset non è un
motivo per escluderlo (es. MAVIS-Geometry senza licenza dichiarata, `images.zip` di LLaVA-Pretrain "per la
ricerca"). Si citano sempre le fonti, i dati non si ridistribuiscono e si rispettano le richieste di
rimozione. Se l'utente parla di pubblicare i pesi o di uso commerciale, riaprire la questione.

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
| Generazione con vLLM (teacher o Violetto) | 1 | 8 | 96G | ≤ 12:00:00, a shard con ripresa |
| Training F1/F2/F4 | 1 | 8 | 120G | ≤ 24:00:00, con ripresa dal checkpoint |
| CPU: hash, estrazione, shard, render | 0 | 2-4 | 4-24G | ≤ 12:00:00 |

Le CPU di Slurm su lagrange0 sono thread, non core: `--cpus-per-task=4` dà due coppie di thread
sorelli, cioè 2 core fisici (job 29418, CPU 0, 1, 48, 49). `srun` le vede tutte anche senza
`--cpus-per-task`. Per i pool di processi (shard, DataLoader) contare i core fisici, non le CPU.
Nei job non scrivere un file per campione: il job 29414 ha creato 168 mila file in pochi minuti e la
cancellazione su GlusterFS ha richiesto più di un'ora.

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
  value embeddings sono un secondo lookup per token. Il modello VL
  (`src/limite_vl/modeling.py`, scelta confermata dall'utente il 2026-10-09) non copia il ciclo
  dei layer. Usa due forward hook sul `LimiteForCausalLM` originale:
  - su `embed_tokens`: le posizioni `<|image_pad|>` ricevono l'output del proiettore prima della
    `rms_norm` iniziale; `<|vision_start|>` e `<|vision_end|>` ricevono le righe addestrabili
    `vision_token_embeds`;
  - su `value_embeds`: il contributo sulle posizioni `<|image_pad|>` viene azzerato.
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
- Packing: con `position_ids` che ripartono a ogni esempio e `attention_mask=None`, transformers
  costruisce la maschera a blocchi. Job 29402, in fp32: loss per token di due esempi
  impacchettati uguali a quelle degli esempi da soli entro 2.5·10⁻⁵; senza reset delle
  posizioni la differenza è 0.85.
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
  proiettore) a circa 4 volte il costo di bf16 sull'encoder. Default `VLConfig.vision_precision = "tf32"`;
  le prove 29421, 29423 e il run 29425 (f1-mid) avevano l'encoder in bf16.
- `encode_images` passa le immagini nell'encoder a gruppi di lunghezza simile (al più 32768 patch con il
  padding): con formule, foto e diagrammi nello stesso micro-batch il padding fino a 2304 patch sarebbe
  quasi tutto il lavoro. In fp32 coincide con il gruppo unico entro 7.6·10⁻⁷ (job 29447); il controllo
  gira sul primo micro-batch di ogni job di training.
- Gradient checkpointing sul modello VL (job 29402): in fp32 i gradienti del proiettore
  differiscono di 4·10⁻⁷. In bf16 la differenza (circa 1%) è uguale a quella fra due backward
  identici senza checkpointing, cioè è il non determinismo dei kernel.

## Formato delle risposte di Violetto (vale per tutti i dati di training)

- Violetto chiude la risposta con `<|endoftext|>` (151643) subito dopo il testo, senza
  `<|im_end|>` (generazione greedy, job 29400). Il chat template invece scrive
  `<|im_end|>\n` dopo ogni turno dell'assistente.
- Il ragionamento è `<think>...</think>` senza a capo, seguito dalla risposta (soluzioni del
  valutatore, job 29393). Il template, con `reasoning_content`, scriverebbe
  `<think>\n...\n</think>\n\n`.
- `LimiteVLProcessor` segue il formato generato, non il template:
  - ultimo turno: contenuto + `<|endoftext|>`, con label;
  - turni intermedi: label solo sul contenuto; `<|im_end|>\n` resta nell'input senza label;
  - immagine sempre prima del testo nel messaggio utente.
- `<think>` (151665) e `</think>` (151666) sono token aggiunti non speciali. Il processor li
  ammette solo nei testi dell'assistente e nel prefill.
- Didascalie di F1 (opzione B, scelta dall'utente il 2026-10-09): l'assistente riceve
  `<think></think>` come prefill senza label (`F1_PREFILL`), seguito dalla didascalia +
  `<|endoftext|>` con label. Così il proiettore non impara a far saltare il `<think>` iniziale.
  Nelle verifiche di F1 lo stesso prefill va dato in generazione.

## Dati di F1

- LLaVA-Pretrain @ 70f9d1e5: 29 shard in `data/shards/llava-pretrain/70f9d1e5/` (558 mila didascalie,
  circa 312 token per esempio con l'immagine a 1024 patch).
- MAVIS-Caption non esiste su Hugging Face. Gli autori l'hanno pubblicato solo come versione provvisoria
  su Google Drive, e la cartella risponde 404. Per i diagrammi si usa CaraJ/MAVIS-Geometry @ 9c80fff3
  (`RuleBaseGeo.zip`), cioè una parte di MAVIS-Instruct.
  - Struttura dello zip (job 29415): 199.603 cartelle `RuleBaseGeo/depthK/<i>/`, 50 mila per profondità
    0-3 (numero di forme concatenate oltre la prima). Ogni cartella contiene `<n>_annotation.json`
    (`text_en`/`text_zh`, `choices`, `answer_index`) e tre immagini.
  - Corrispondenza con le varianti del parquet di esempio (job 29410, 29415):
    - `caption.jpg` = text_dominant: tutte le misure citate nella didascalia sono disegnate;
    - `text.jpg` = text_lite: figura quasi spoglia, i valori sono nella domanda;
    - `vision_dominant.jpg` = vision_dominant: i valori tolti dalla domanda sono disegnati.
    Nel 18% dei campioni `caption.jpg` e `vision_dominant.jpg` sono identiche.
  - Le didascalie vanno solo con `caption.jpg`: con le altre varianti citano misure che non si vedono.
  - Servono 2304 patch: a 1024 le etichette diventano alte pochi pixel.
- Shard di MAVIS-Geometry (job 29419, `scripts/shard_mavis_geometry.py`): 199.050 campioni in 11 tar in
  `data/shards/mavis-geometry/9c80fff3-caption/`. Esclusi il campione incompleto `d3-3228` e 552
  didascalie duplicate. Per ogni campione:
  - un PNG già alla dimensione esatta di 2304 patch: il processor non lo ridimensiona di nuovo;
  - un JSON con `text_en` completo, scelte e dimensioni.
  In media 568.6 token immagine, 113 milioni in tutto; circa 710 token per esempio con la didascalia.
- UniMER @ 2343ddd9 (job 29436, 29438): l'immagine `images/<indice>.png` ha come formula la riga
  `<indice>` di `train.txt`; il dataset è mescolato, le righe vuote sono HME100K (tolto per copyright) e le
  formule scritte a mano di CROHME (meno dell'1%) non si separano per indice. Shard in
  `data/shards/unimer/2343ddd9/`: 985.450 campioni in 5 tar e i test spe/cpe/sce/hwe in `test/`.
  - Formula senza gli spazi fra token del formato im2latex (`latex.despace`, 31% di token in meno);
    esclusi i 672 casi in cui cambierebbe i token LaTeX.
  - Immagine ingrandita al più 2x e al più 1024 patch (`processing.capped_size`), già alla dimensione
    finale: in media 71 token immagine.
  - Criterio di F1: `scripts/eval_f1_formulas.py` su spe e cpe (uguaglianza a meno degli spazi e
    accuratezza sui token LaTeX).
- Cambi di dati approvati dall'utente il 2026-10-12 (ricerca sui dataset, in scaricamento):
  - F1: didascalie dense di PixMo-Cap (`anthracite-org/pixmo-cap-images`, 25 parquet su 75) e didascalie
    di Qwen3-VL-235B delle immagini STEM di `OpenDataArena/MMFineReason-1.8M`; LLaVA-Pretrain ridotto.
  - F2: MMFineReason come bacino principale, con le sue descrizioni come ponte testuale.
  - F4: `skylenage/DeepVision-103K` e `TIGER-Lab/ViRL39K` (gruppo p5-rl).
  MMFineReason contiene Geo3K e la famiglia WeMath: decontaminare prima del training.
- Decontaminazione contro i benchmark p6 (regola approvata dall'utente il 2026-10-14): usare solo le liste
  `data/decontam/p6/<sorgente>/exclude_final.txt` (MMFineReason 16.934, 0.94%; PixMo, MAVIS, UniMER, LLaVA nessuna).
  Le `exclude.txt` grezze di `scripts/decontam.py scan` sono solo candidati: il pHash confonde le immagini quasi
  vuote (simboli di UniMER contro figure di DynaMath) e un 10-gram in comune unisce problemi diversi. La regola di
  `scripts/decontam_review.py`: somiglianza dei pixel >= 0.90 con rapporto d'aspetto entro 1.2, oppure copertura
  degli n-gram >= 0.8 in entrambe le direzioni con almeno 15 parole (job 29577).

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
  (`verify_manifest.py`), dati (`shard_llava_pretrain.py`, `shard_mavis_geometry.py`; ispezione con
  `peek_parquet.py`, `mavis_variants.py`, `inspect_zip.py`, `mavis_zip_survey.py`, `peek_shard.py`,
  `unimer_survey.py`; shard di UniMER con `shard_unimer.py`), F1 (`train_f1.py`, `eval_f1_val.py`,
  `eval_f1_formulas.py`, `stream_stats.py`, `check_vision_padding.py`, `check_vision_precision.py`),
  valutatore (`link_eval_datasets.py`),
  codice di Limite (`vendor_limite_modeling.py`), F0 (`f0_checks.py`, `f0_vl_checks.py`,
  `vllm_smoke.py`)
- `slurm/` — `prologue.sh` e gli script `.sbatch`
- `env/`, `env-evals/` — pyproject e lock dell'ambiente principale e del client del valutatore
- `third_party/` — `limite-violetto` (plugin vLLM e valutatore), `verifiers`, `limite-modeling`
  (copia verificata del codice del modello)
- `src/limite_vl/` — codice del progetto:
  - `modeling.py`: modello VL e proiettore;
  - `processing.py`: immagini in patch a griglia pari, chat in token e label;
  - `collator.py`: packing e collate;
  - `data.py`: lettura degli shard tar ed esempi per tipo di sorgente;
  - `stream.py`: flusso misto e riprendibile di micro-batch per il training;
  - `latex.py`: formule im2latex in LaTeX naturale e confronto fra trascrizioni;
  - `train_utils.py`: loss a blocchi e checkpointing per layer;
  - `limite_code.py`: import verificato del codice di Limite;
  - `optim/`: NorMuon e gruppi della Tabella 1.
- `tests/` — test CPU (`test_normuon.py`, `test_vl_data.py`, `test_f1_data.py`), lanciati tutti da `slurm/tests_cpu.sbatch`
- `config/runs/` — config dei run di training (`f1.json` è il run completo, da aggiornare con il nuovo mix)
- `runs/ledger.tsv` — registro dei job (testo piccolo, versionato)
