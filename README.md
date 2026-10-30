# vision_limite — Limite-VL su Lagrange

Aggiungere la visione a Limite 1B - Violetto (`paradigma-inc/limite-1b-violetto` @ b1f3d572): leggere immagini,
documenti e formule, collegare più immagini e risolvere problemi di matematica e scienze. Il lavoro gira sul cluster
Lagrange (frontend `lagrangectl`, nodo `lagrange0` con H100 NVL) con Claude Code sul frontend.

## Documenti di riferimento

| Documento | Ruolo |
| --- | --- |
| [`PIPELINE_ADDESTRAMENTO_LIMITE_VL.md`](PIPELINE_ADDESTRAMENTO_LIMITE_VL.md) | **Riferimento corrente** (v1.5, 2026-10-30; perimetro della prima versione): fasi F0-F4, protocollo dei dati, decisioni aperte. Progetto, non ricetta eseguibile |
| [`CLAUDE.md`](CLAUDE.md) | Regole operative per Claude Code sul cluster |
| [`LAGRANGE.md`](LAGRANGE.md) | Guida del cluster (frontend, storage, SLURM, download) |
| [`docs/F0_IMPLEMENTATION_PLAN.md`](docs/F0_IMPLEMENTATION_PLAN.md) | Stato reale del codice rispetto alla pipeline e interventi F0 in ordine |
| [`docs/PANNELLO_VALUTAZIONE.md`](docs/PANNELLO_VALUTAZIONE.md), [`docs/F0_CAMPIONE_CANDIDATO.md`](docs/F0_CAMPIONE_CANDIDATO.md) | Pannello di valutazione e decontaminazione (D6, ruoli in `pannello-v2`); campione candidato di F0, lacune e prossimo lotto |
| [`docs/REPOSITORY_STATUS.md`](docs/REPOSITORY_STATUS.md), [`docs/REVIEW_ACTIONS.md`](docs/REVIEW_ACTIONS.md) | Stato e tracciabilità dei rilievi delle revisioni indipendenti (`docs/reviews/`, immutati) |
| [`docs/MIGRATION_REPORT.md`](docs/MIGRATION_REPORT.md) | Migrazione del 2026-10-28: base git, spostamenti, controlli e limiti |
| [`docs/ARTIFACT_INVENTORY.tsv`](docs/ARTIFACT_INVENTORY.tsv), [`docs/REPOSITORY_INDEX.tsv`](docs/REPOSITORY_INDEX.tsv) | Artefatti fuori da git e stato di ogni file versionato |
| [`archive/pipeline-precedente/`](archive/pipeline-precedente/README.md) | Ricette, launcher e istruzioni della strategia precedente: storia, non comandi |

## Dove si lavora

- **Portatile dell'utente:** pipeline e revisioni, consegnate tramite GitHub; eventuale `uv lock`.
- **Frontend `lagrangectl`:** editing, git, invio e controllo dei job, download con `scripts/fetch.sh`. Niente calcolo,
  test, import di torch o parsing di dati grandi (3.8 GiB di RAM condivisi).
- **Job SLURM:** tutto il calcolo, offline, con le risorse della QoS `castelnuovo_qos` (1 GPU, 12 CPU, 150 GB in totale).

## Stato (2026-10-28)

- **Fase:** F0 preparatorio. Nessun training, pilota GPU, audit con agenti o download è approvato.
- **Codice presente:** modello VL baseline (SigLIP2 so400m NaFlex, pixel shuffle 2x2 + MLP verso i 1280 canali di
  Limite, hook sugli embedding), processor con liste di immagini, packing, flusso dati riprendibile con esclusioni e
  rietichettature, controlli deterministici di MAVIS, UniMER, PixMo e sintetici, decontaminazione, valutatore testuale
  ufficiale, NorMuon. I trainer (`scripts/train_f1.py`, `scripts/train_f2.py`) sono della strategia precedente: encoder
  sempre congelato, adapter salvato senza ViT, conteggi di parametri fissi.
- **Manca rispetto alla pipeline:** contratto di risposta versionato, manifest per compito/variante con split
  globali, caricatori con più immagini, encoder Qwen, gruppi e checkpoint per il ViT addestrabile, pannelli di
  valutazione visivi. Dettagli e ordine in `docs/F0_IMPLEMENTATION_PLAN.md`.
- **Dati:** tutti i download verificati per hash (`outputs/verify/`), circa 540 GiB in tutto (cache HF 274 GiB, `data/` 172 GiB, `outputs/` 68 GiB, ambienti e cache 24 GiB).
  Nessuna sorgente è ammessa alla nuova pipeline: `downloads/catalog_status.tsv`.

## Contenuto

| Percorso | A cosa serve |
| --- | --- |
| `scripts/setup_env.sh`, `scripts/guarded_run.sh` | Ambiente in `venvs/` per fasi senza lavoro pesante sul frontend; `uv lock` con le soglie della guida |
| `scripts/fetch.sh`, `scripts/fetch_manifest.sh`, `scripts/hf_manifest.py`, `scripts/link_hf_snapshot.py` | Download controllati (1 GiB per partire, stop sotto 768 MiB o sopra 128 MiB di RSS, 10 MiB/s) e cache HF offline |
| `scripts/verify_manifest.py`, `slurm/verify_hashes.sbatch` | Verifica SHA-256 e hash git in un job CPU |
| `scripts/f0_checks.py`, `scripts/f0_vl_checks.py` | Verifiche F0 di Violetto e del modello VL baseline |
| `scripts/*_check.py`, `scripts/decontam*.py`, `scripts/split_duplicates.py` | Controlli dei dati senza LLM e decontaminazione |
| `scripts/shard_*.py` | Conversione delle sorgenti in shard tar da circa 1 GB |
| `src/limite_vl/` | Modello, processor, dati, flusso, controlli, ottimizzatori |
| `slurm/*.sbatch` | Launcher correnti; `slurm/prologue.sh` per la provenienza |
| `downloads/` | Catalogo delle sorgenti con revisioni fissate e stato |
| `env/`, `env-evals/` | pyproject e lock dell'ambiente (torch 2.11.0+cu129, transformers 5.6.2, vLLM 0.26.0) e del client del valutatore |
| `runs/ledger.tsv` | Registro dei job |

## Ricostruire la cartella da zero

La cartella del cluster esiste già. Per ricrearla: clonare il repository in `/home/prignano/vision_limite`, creare
`data/logs hf wheels tools venvs cache outputs/logs secrets third_party` (con `secrets/` a 700 e il token in
`secrets/hf_token` a 600), clonare `paradigma-inc/limite-violetto` @ 7ddad34f e `verifiers` @ d30a3f48 in
`third_party/`, poi seguire `CLAUDE.md` (Ambiente, Download). I passi originali del kit sono in
`archive/pipeline-precedente/README.pre-migrazione.md`.
