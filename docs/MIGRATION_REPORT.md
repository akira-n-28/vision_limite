# Rapporto della migrazione alla pipeline v1.4

Eseguita il 2026-10-28 da Claude Code sul frontend `lagrangectl` (utente `prignano`, cartella
`/home/prignano/vision_limite`), secondo [`ISTRUZIONI_MIGRAZIONE_LAGRANGE.md`](../ISTRUZIONI_MIGRAZIONE_LAGRANGE.md).

## 1. Base Git e stato iniziale

| Voce | Valore |
| --- | --- |
| Branch | `main`, che segue `origin/main` (`git@github.com:akira-n-28/vision_limite.git`) |
| HEAD prima del recupero | `9d977b6` (= `origin/main` prima del fetch), lo snapshot analizzato sul portatile |
| Recuperato | `origin/main` = `6e888d5` ("docs: publish Limite-VL pipeline and Lagrange migration handoff"): 8 file di documentazione, nessun file di codice |
| Modo | `git merge --ff-only origin/main` |
| Albero prima del recupero | pulito: nessuna modifica non committata e nessun file non tracciato fuori da `.gitignore` |
| Base della migrazione | tag annotato locale `pre-migrazione-v1.4` su `6e888d5` |
| Commit della migrazione | locali, `git log pre-migrazione-v1.4..main`; non pubblicati |
| Job e processi | nessuno ([elenco](ACTIVE_WORK_AND_UNTOUCHED_PATHS.md)) |
| Spazio | `/home` GlusterFS 14T, 12T liberi (19% usato), inode all'1%; progetto circa 537 GiB, sotto la soglia di 1 TB |

Lo stato riferito dall'utente (repository aggiornata, artefatti fuori da Git, nessun job) è confermato. Non c'era lavoro
locale non pubblicato da integrare.

## 2. Stato preservato e recupero

- Il lavoro precedente è tutto nel tag `pre-migrazione-v1.4`: nessuno stash, reset, clean o checkout forzato.
- Copia dei piccoli file toccati: `outputs/migration-v1.4/pre-migration-small-files.tar` (124 voci: `CLAUDE.md`,
  `README.md`, `config/runs`, `config/audit`, `slurm`, `downloads/sources.tsv`, `downloads/build_manifests.sh`, `docs`,
  `runs/ledger.tsv`), con sha256 accanto. Fuori da git, nessun segreto né dato pesante.
- Recupero di un file archiviato: `git mv archive/pipeline-precedente/<percorso> <percorso>`. Recupero completo:
  `git checkout pre-migrazione-v1.4 -- config/runs slurm CLAUDE.md README.md downloads/sources.tsv` e rimozione di
  `archive/`, `config/runs/README.md`, `config/audit/INDEX.md`, `downloads/catalog_status.tsv` e dei documenti nuovi.

## 3. Riferimento corrente

- `PIPELINE_ADDESTRAMENTO_LIMITE_VL.md`: versione 1.4 del 28 ottobre 2026 (riga 3), 798 righe, sha256
  `cc8cc71106be2178d85aa67c05fee19f9d9c085a29f32a5e40a6e1a574053317`. Integrata così com'è.
- Revisioni indipendenti in `docs/reviews/`, immutate: `REVISIONE_INDIPENDENTE_PIPELINE_LIMITE_VL.md` (sha256
  `dbfa868a…`, riferita alla v1.0, hash dichiarato `f17af2e4…`) e `REVISIONE_INDIPENDENTE_GOLD_E_DECONTAMINAZIONE.md`
  (`4b035b3d…`, riferita alla v1.2, `b49e6299…`). Le versioni 1.0 e 1.2 della pipeline non sono nel repository: i
  rapporti restano verificabili solo per i loro hash. Nessun rapporto approva la v1.4 o il codice.
- `README.md` e `CLAUDE.md` riscritti: rimandano alla pipeline, distinguono portatile, frontend e job, dicono che il
  progetto è in F0 preparatorio, elencano le decisioni aperte e dichiarano superate le soglie numeriche della strategia
  precedente. Tolti il piano esterno (`artifact d3fe8c7e…`), il piano dati del 2026-10-26 (`PMFgabBMybVYD4Hr8z3Sq4`),
  l'ordine di priorità dei download p1-p7, le sezioni su F1-clean, record e gate F2, audit STEM e ottimizzatore di F2,
  e l'invito del README a "eseguire il giorno 1 del piano". Restano invariati i vincoli del cluster, la protezione del
  token e i fatti tecnici verificati sul codice di Limite e sui dati, con la nota che non decidono l'architettura.
  Copie integrali delle versioni precedenti in `archive/pipeline-precedente/CLAUDE.pre-migrazione.md` e
  `README.pre-migrazione.md`.

## 4. Spostamenti e file nuovi

| Elemento | Prima | Dopo | Motivo |
| --- | --- | --- | --- |
| 13 config dei run (`f1`, `f1b`, `f1c`, `f1c-control`, `f1-clean`, `f1-diag-llava`, `f1-mid`, `f1-smoke`, `f1-smoke3`, `f1-smoke-mix`, `f2-smoke`, `f2-abl-adamw`, `f2-abl-normuon`) | `config/runs/` | `archive/pipeline-precedente/config/runs/` | ricette della strategia precedente; nessuna diventa una ricetta v1.4 |
| 4 launcher di training (`train_f1`, `train_f1_smoke`, `train_f2`, `f1_diag`) | `slurm/` | `archive/pipeline-precedente/slurm/` | nessuna ricetta corrente da lanciare |
| 6 launcher F1c | `slurm/` | idem | fase F1c abbandonata, adapter non ereditati |
| 10 launcher del ponte teacher (`f2_generate`, `f2_plan`, `f2_select`, `f2_inspect`, `f2_layout`, `f2_leak`, `f2_phrases`, `f2_qonly_compare`, `f2_sources`, `f2_yield`) | `slurm/` | idem | ponte testuale di Violetto superato (pipeline §9) |
| 7 launcher di audit e gate (`f2_backfill`, `f2_build`, `f2_certify`, `f2_pilot`, `f2_questions`, `stem_audit_export`, `stem_audit_round`) | `slurm/` | idem | strumenti congelati il 2026-10-26 |
| 2 launcher del replay generato (`replay_generate`, `replay_select`) | `slurm/` | idem | il replay v1.4 parte dai riferimenti originali |

Per ogni file, `archive/pipeline-precedente/MOVES.tsv` riporta sha256, blob git, categoria, motivo e ultimo commit.
Contenuti identici byte per byte: blob confrontati con `pre-migrazione-v1.4` per tutti i 42 file.

I candidati del portatile erano 13 config e 28 launcher; qui i launcher sono 29. Il conteggio del portatile non ha un
elenco nominativo, quindi la differenza non si può attribuire con certezza: è probabile che riguardi `f1_diag`
(diagnostica dei run F1 storici) o uno dei due launcher del replay. Restano in `slurm/` 50 launcher e `prologue.sh`:
8 correnti (con `prologue.sh`), 33 riutilizzabili e 10 da adeguare (`docs/REPOSITORY_INDEX.tsv`).

File nuovi: `archive/pipeline-precedente/{README.md, MOVES.tsv, CLAUDE.pre-migrazione.md, README.pre-migrazione.md}`,
`config/runs/README.md`, `config/audit/INDEX.md` (stato e sha256 delle 15 istruzioni congelate e manifest che le citano),
`downloads/catalog_status.tsv` (disponibilità, verifica hash, ruolo v1.4, ammissione per 51 sorgenti, incluse quelle
della pipeline non ancora nel catalogo), due righe di commento in testa a `downloads/sources.tsv` (righe dei dati
invariate; `build_manifests.sh` salta i commenti), `docs/MIGRATION_REPORT.md`, `docs/ARTIFACT_INVENTORY.tsv`,
`docs/REPOSITORY_INDEX.tsv`, `docs/F0_IMPLEMENTATION_PLAN.md`, `docs/ACTIVE_WORK_AND_UNTOUCHED_PATHS.md`, una riga di
nota in `runs/ledger.tsv`. Aggiornati `docs/REPOSITORY_STATUS.md` e `docs/REVIEW_ACTIONS.md` (solo rimandi alle evidenze).

## 5. Cosa non è stato spostato, e perché

- **Script Python e moduli storici** (`train_f1.py`, `train_f2.py`, `f2_*.py`, `f1c_*.py`, `stem_audit_*.py`, `replay_*.py`;
  `f1c.py`, `f2.py`, `f2_records.py`, `f2_claims.py`, `facts.py`, `mavis.py`, `stem_audit.py`): importati da altri
  strumenti e dai test (`train_f1` da 13 script; `data.py` importa `f1c`, `f2`, `replay`, `synth`, `f1_mavis`;
  `mavis_gen` importa `mavis` e `facts`). `scripts/f2_pilot.py` congela per percorso e sha256 sé stesso,
  `f2_claims.py`, `stem_audit.py` e le istruzioni. Stato di ciascuno in `docs/REPOSITORY_INDEX.tsv`.
- **Docstring** di 19 script che citano `slurm/<nome>.sbatch` (17) o solo `config/runs/<nome>.json` (2) ora archiviati: non corrette,
  per non cambiare l'identità del codice congelato; il nuovo percorso è in `MOVES.tsv`.
- **`config/audit/`**: invariata (citata per percorso e hash da `outputs/f2_pilot`, `outputs/f2_cert`).
- **Test storici**: invariati; restano in `slurm/tests_cpu.sbatch`.
- **Artefatti pesanti**: nessuno spostato o copiato; l'archiviazione è logica, nell'inventario.

## 6. Inventario degli artefatti

`docs/ARTIFACT_INVENTORY.tsv`: 119 gruppi (31 repository della cache HF, 12 cartelle di shard, liste, 55 cartelle di
output, ambienti, cache, repository esterni), con origine, stato, dipendenze, dimensione, provenienza, azione
proposta e verifica necessaria. Totale misurato 537 GiB: cache HF 274 GiB, `data/` 172 GiB, `outputs/` 68 GiB, ambienti,
wheel e cache 24 GiB. Tutti gli 11 report di `verify_hashes` (456 file) sono `ok`; ogni manifest in
`downloads/manifests/` ha tutti i suoi file nella cache.

Nessuna proposta di cancellazione in questa consegna. Candidati da discutere in futuro, se servisse spazio, con
motivo, dimensione e recuperabilità: `data/shards/unimer/2343ddd9-x4` (14 GiB, rigenerabile), `data/shards/synth/v2`
(0.8 GiB, superato da v3), `outputs/f2-abl-*` (36 GiB, checkpoint completi di un'ablation storica), GeoSym127K (0.5 GiB,
riscaricabile). Nessuna sorgente è promossa a gold o ammessa: lo stato "riutilizzabile" riguarda l'integrità e la
provenienza, non la qualità.

## 7. Controlli effettuati

| Controllo | Esito |
| --- | --- |
| Identità dei file archiviati (blob contro `pre-migrazione-v1.4`) | 42/42 identici |
| sha256 della copia tar e di un file estratto a stdout | coerenti |
| Codice cambiato (`git diff pre-migrazione-v1.4 -- scripts src tests`) | nessuna differenza |
| Sintassi dei launcher rimasti (`bash -n`) | 51/51 ok |
| Riferimenti a launcher archiviati fuori da `archive/`, docstring degli script, ledger e documenti consegnati | nessuno |
| Riferimenti a `config/runs/*.json` nei file correnti | solo esempi d'uso in 3 docstring (`f1c_audit_export.py`, `stream_stats.py`, `train_f1.py`) |
| Test che leggono config o launcher spostati | nessuno (ricerca statica) |
| Codice che calcola l'hash di tutta `config/audit/` (l'aggiunta di `INDEX.md` lo cambierebbe) | nessuno: si calcolano solo file nominati |
| Link `venvs/current`, `venvs/evals-current`, `data/eval-datasets/*` | validi |
| Colonne dei TSV nuovi | uniformi (catalogo 9, inventario 10, indice 5, `MOVES.tsv` 7) |
| Link relativi dei documenti nuovi e aggiornati | risolti |

Dopo l'approvazione dell'utente, `slurm/tests_cpu.sbatch` (job 30255, 2026-10-28, 7 min 38 s): 22 file, 775 controlli,
nessun fallimento, stderr vuoto, come il job 30050 prima della migrazione. Non eseguiti, perché non sono di questa fase:
hash completi degli shard, controlli della qualità dei dati, collaudi GPU, valutazioni.
Elenco con scopo e risorse in `docs/F0_IMPLEMENTATION_PLAN.md`, §3.

## 8. Limiti

- Le dimensioni vengono da `du` sui metadati di GlusterFS (MiB, 2026-10-28); gli output sono inventariati per cartella,
  non per file.
- La classificazione di script e moduli viene da lettura statica di docstring, import e riferimenti; non da esecuzione.
- I fatti tecnici in `CLAUDE.md` vengono da job storici e valgono per la baseline e il checkpoint fissati.
- Le memorie di Claude Code fuori dal repository sono state aggiornate per indicare la pipeline v1.4 come riferimento.

## 9. Decisioni aperte

Le sei decisioni D1-D6 di `docs/F0_IMPLEMENTATION_PLAN.md` (checkpoint Qwen, metodo del budget, contratto di risposta,
metriche e margini, budget degli audit, ruoli del pannello). Per questa migrazione:

1. **Chiusa il 2026-10-28:** l'utente conferma che le soglie della regola del 2026-10-19 (≤ 1% sulle risposte, ≤ 2% sui
   fatti), i 300M token di F1-clean e la ricetta NorMuon di F2 non valgono più.
2. **Approvato e fatto il 2026-10-28:** `slurm/tests_cpu.sbatch` (job 30255): 22 file, 775 controlli, nessun fallimento.
3. **Approvato il 2026-10-28:** push dei tre commit della migrazione su `origin/main`.
4. **Approvato il 2026-10-28:** scrittura delle quattro specifiche S1-S4. Bozze in `docs/specs/`, da approvare.

## 10. Cosa è pronto

- **Pronta:** la migrazione documentale e il riordino reversibile del repository. La pipeline v1.4 è il riferimento
  unico; la storia è separata dai percorsi attivi; inventario e piano F0 sono scritti.
- **Non pronti:** codice per F0 (nessuna modifica: valgono gli stati di `docs/F0_IMPLEMENTATION_PLAN.md`), dati (nessuna
  sorgente ammessa, nessun manifest), configurazioni di esperimento (`config/runs/` è vuota). Nessun training, pilota,
  audit o pulizia massiva è autorizzato da questa consegna.
