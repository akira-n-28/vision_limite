# vision_limite — kit di avvio per Lagrange

Repository di partenza per aggiungere la visione a Limite 1B - Violetto sul cluster Lagrange,
lavorando con Claude Code sul frontend. Le regole operative per Claude Code sono in
`CLAUDE.md`; la guida del cluster è `LAGRANGE.md`.

## Primi passi (sul frontend `lagrangectl`, una volta sola)

```bash
# 1. Un'unica cartella di progetto: codice (in git) e cartelle pesanti (escluse da .gitignore)
mkdir -p /home/prignano/vision_limite
#    copiare qui il contenuto del kit (scp dal portatile), poi:
cd /home/prignano/vision_limite
mkdir -p data/logs hf wheels tools venvs cache outputs/logs secrets third_party
chmod 700 secrets
git init && git add -A && git status --short   # controllare: solo codice, niente data/ hf/ ecc.
git commit -m "Kit di avvio vision_limite"

# 2. Codice ufficiale di Limite (plugin vLLM e valutatore), repository piccolo
git clone https://github.com/paradigma-inc/limite-violetto.git third_party/limite-violetto
git -C third_party/limite-violetto checkout 7ddad34f5e3d25ddea68c6745fc00373b82bca21   # commit verificato il 2026-10-07
#    verifiers al commit fissato dal valutatore (limite-evals), per il client in env-evals/:
#    git init + remote, poi scripts/guarded_run.sh verifiers-fetch -- git -C third_party/verifiers \
#      fetch --depth 1 origin d30a3f48e5f14b06b3081b2102ec32cc3149b849 && checkout FETCH_HEAD

# 3. Token Hugging Face (solo per repository con accesso riservato), mai nel repository
#    scrivere il token in secrets/hf_token con un editor, poi:
chmod 600 secrets/hf_token

# 4. Claude Code in una sessione che sopravvive alla disconnessione (se tmux c'è)
command -v tmux && tmux new -s vision_limite
cd /home/prignano/vision_limite && claude
```

Prima richiesta suggerita a Claude Code: *"Leggi LAGRANGE.md e CLAUDE.md, poi esegui il
giorno 1 del piano: sonda GPU, manifest p1 e avvio dei download. Poi prepara il venv con
`scripts/setup_env.sh`, fermandoti a chiedermi conferma prima di scegliere il pyproject."*

## Cosa contiene

| Percorso | A cosa serve |
| --- | --- |
| `scripts/setup_env.sh` | Crea il venv in `venvs/` per fasi (`status`, `uv`, `lock`, `wheels`, `install`) senza lavoro pesante sul frontend |
| `scripts/guarded_run.sh` | Esegue `uv lock` sul frontend con le soglie della guida applicate a tutto l'albero di processi (e tetto cgroup se disponibile) |
| `scripts/fetch.sh` | Scarica un file rispettando le soglie della guida (1 GiB per partire, stop sotto 768 MiB o sopra 128 MiB di RSS, 10 MiB/s, un trasferimento alla volta) |
| `scripts/fetch_manifest.sh` | Scarica in sequenza i file di un manifest; si ferma al primo errore |
| `scripts/hf_manifest.py` | Manifest di un repository Hugging Face a una revisione fissata, nel layout della cache HF |
| `scripts/link_hf_snapshot.py` | Crea gli snapshot della cache HF dopo il download (uso offline) |
| `scripts/uvlock_manifest.py` | Elenco delle wheel per Python 3.12 / Linux x86_64 da un `uv.lock` |
| `scripts/verify_manifest.py` | Verifica SHA-256 e hash git in un job CPU |
| `scripts/f0_checks.py` | Verifiche F0 su GPU: `inputs_embeds`, gradient checkpointing, loss a blocchi, throughput, nomi dei parametri e gruppi dell'ottimizzatore |
| `src/limite_vl/optim/` | NorMuon (Algoritmo 1 del report) e gruppi della Tabella 1; test in `tests/test_normuon.py` |
| `slurm/*.sbatch` | Sonda GPU, verifiche F0, verifica hash, estrazione di uv, installazione offline dell'ambiente |
| `downloads/sources.tsv` | Modelli e dataset per priorità, con le revisioni del valutatore ufficiale |
| `env/` | Come costruire l'ambiente in base al driver (CUDA 13 oppure 12) |

Test eseguiti prima della consegna: sintassi e `shellcheck` di tutti gli script bash,
compilazione degli script Python, download reale da un server HTTPS locale (limite di
banda, stop per memoria con `.part` conservato, ripresa con Range, lock, segnale TERM,
dimensione errata), manifest da un'API Hugging Face simulata, link degli snapshot e
verifica degli hash, invio del token Hugging Face solo all'host di Hugging Face.
`uvlock_manifest.py` è stato provato sul `uv.lock` ufficiale (202 wheel, circa 4 GB con
torch). `uv lock` sotto `guarded_run.sh` su un progetto con vLLM e transformers: 226
pacchetti, circa 45 MB, RSS massima 116-127 MiB. Catena completa di `setup_env.sh` con
`sbatch`/`srun` simulati: uv scaricato e verificato, lock, wheel, venv offline con hash
verificati e plugin Limite compilato offline. NorMuon: direzione di aggiornamento identica a modded-nanogpt (diff. relativa 5·10⁻⁸),
update sui pesi entro l'arrotondamento fp32, convergenza su un problema lineare.
`f0_checks.py` e i job GPU non sono stati eseguiti: richiedono il cluster.
