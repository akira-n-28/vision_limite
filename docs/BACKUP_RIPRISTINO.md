# Backup e ripristino di Limite-VL

Incarico del 2026-11-12: mettere al sicuro il progetto per un cambio di utente su Lagrange o un trasferimento su un
altro server, senza toccare la pipeline scientifica. Generazione: **`g1-20261112`**, backup **selezionato**.

I numeri definitivi stanno negli elenchi generati in [`docs/backup/`](backup/): volumi, byte, hash ed esito delle
verifiche.

## 1. Sintesi

| Voce | Valore |
| --- | --- |
| Modalità | **backup selezionato** (il nucleo) sul Google Drive istituzionale Sapienza (`studenti.uniroma1.it`); decisione dell'utente del 2026-11-13 |
| Storia | partito il 2026-11-12 come backup completo; dopo l'interruzione notturna ridotto al nucleo nella stessa generazione; nessuna divisione fra account |
| Spazio | 800 GB per utente, 5,16 GB usati prima del backup (pagina web di Drive); l'API mostra solo lo spazio dell'istituzione |
| Accesso | rclone v1.76.0 dal frontend, ambito OAuth `drive.file` (solo i file creati dall'app), client OAuth proprio dell'utente (progetto Google Cloud `limite-vl-backup`, app desktop in prova) |
| Posizione | cartella privata `LimiteVL-backup/g1-20261112/`; unico permesso il proprietario, nessun link |
| Dimensione | 43 volumi del nucleo, 138,2 GiB di dati, più il volume finale del codice; esclusi 404,2 GiB, documentati per il recupero (§7) |
| Stato del progetto | dati all'inventario 30943 (commit `f10413d`), confermati invariati dall'inventario 30972 del 2026-11-13; il codice si salva per ultimo, dopo i documenti del backup |

## 2. Contenuto

**Nel backup**, cioè il nucleo (colonna `nucleo` di [`backup/g1-20261112-categorie.tsv`](backup/g1-20261112-categorie.tsv)):
- **Codice e storia:** repository con `.git` intero (tag locale `pre-migrazione-v1.4` non pubblicato, reflog),
  documenti, revisioni umane, `third_party/` con i suoi `.git`, manifest dei download (`downloads/manifests/*.tsv`,
  fuori da git), `runs/ledger.tsv`.
- **Metadati dei dati:** `data/` tranne gli shard esclusi: tutte le versioni dei manifest delle ammissioni,
  decontaminazione, pannello, lingua, specifiche degli snapshot, log dei download, link ai dataset di valutazione.
- **Modelli:** gli snapshot iniziali con tokenizer e processor (Violetto, SigLIP2 so400m e base).
- **Benchmark:** quelli già scaricati con ruoli e split (`hf-piccoli`, MathVista; ruoli in `data/panel/`), più i
  piccoli dataset di testo.
- **Output:** tutto `outputs/` (checkpoint, generazioni, audit, valutazioni, report, log).
- **Shard:** UniMER `2343ddd9-dense`, MAVIS `9c80fff3-caption`, PixMo-Cap `3e42775e`, sintetici v2-v4, replay e
  shard F2.
- **Ambiente:** `wheels/` e `tools/` (uv, rclone).

**Fuori dal backup** (404,2 GiB), con il modo di recuperarli in §7:
- sorgenti grezze su Hugging Face: PixMo-Cap (116,3 GiB, 25 parquet), MMFineReason (77,3), MAVIS-Geometry (24,6),
  LLaVA-Pretrain (25,7), DeepVision (9,9), MAVIS-Function (3,6), UniMER (2,0), DeepMath (2,0), ViRL39K (1,7);
- shard derivati di LLaVA, MMFineReason sft e rl, UniMER base e x4 (122,4 GiB);
- `venvs/` con `cache/uv` (12,7 GiB, ricostruibili da `wheels/` e dai lock, §8) e le altre cache (5,3 GiB).

**Escluso sempre**, solo materiale del trasferimento: `trasferimento/`, cioè staging, stato, inventari, piani, log dei
job del backup, configurazione e token di rclone, `client.env`. È anche in `.gitignore`. Gli inventari e i piani sono
comunque su Drive in `indice/`.

**Segreti.** Nel progetto non ce n'è nessuno. `secrets/` è vuota: tutti i download erano pubblici. La ricerca su file di
testo e oggetti git (job 30938 e 30943) ha trovato solo segnaposto di test e di documentazione, e non scrive mai il
testo trovato. Per questo non c'è un archivio cifrato: se `secrets/` avesse un file, il piano si fermerebbe da solo.
Le credenziali del trasferimento restano solo in `trasferimento/rclone/`, con permessi 600.

**Dipendenze fuori dalla cartella**, non salvate e da ricreare:
- l'interprete `/usr/bin/python3.12` (3.12.3), a cui puntano i venv;
- la chiave SSH per GitHub e `~/.gitconfig`;
- il driver NVIDIA del cluster;
- il progetto Google Cloud del client.

Non ci sono hard link verso l'esterno, worktree, alternates o submodule. L'elenco dei link è in
[`backup/collegamenti_assoluti_o_esterni.tsv`](backup/collegamenti_assoluti_o_esterni.tsv).

## 3. Formato

- **Categorie** (`scripts/backup.py`, funzione `category`): formano una partizione dell'albero e ognuna si ripristina
  da sola.
  - `codice`: repository, `third_party/`, `data/download-log.tsv` e `data/.fetch.lock`.
  - `data-metadati` e una categoria `data-shards-<sorgente>-<versione>` per ogni cartella di shard.
  - `hf-<repository>` per i repository sopra 1 GiB, `hf-piccoli` per gli altri.
  - `outputs-<cartella>` per le cartelle sopra 1 GiB, `outputs-resto` per le altre.
  - `ambienti`, cioè `venvs/` più `cache/uv`, che sono legati da hard link; poi `cache-altre`, `wheels`, `strumenti`.
- **Volumi:**
  - tar POSIX (PAX) standard, circa 5 GiB di dati ciascuno; gzip solo dove ci sono testi (codice, generazioni F2);
  - un file sopra 5 GiB ha un volume a sé; un volume sopra 5 GiB è diviso in parti `.partNNN`, da riunire con `cat`;
  - i link restano link; i nomi di uno stesso hard link stanno nello stesso volume; le date sono al secondo.
- **Indici:** per ogni volume, `indice/volumi/<volume>.jsonl.gz` (percorso, tipo, permessi, data, link e sha256 di ogni
  file) e `<volume>.sha256` nel formato di `sha256sum`. Le parti, con byte, sha256 e md5, sono in
  [`backup/g1-20261112-parti.tsv`](backup/g1-20261112-parti.tsv).
- **Su Drive**, dentro `LimiteVL-backup/g1-20261112/`:
  - `volumi/<categoria>/` con i volumi;
  - `indice/piano/` (piano, membri, categorie, checkpoint) e `indice/piano-codice/`;
  - `indice/volumi/`, con gli indici dei singoli volumi;
  - `indice/indice-volumi.tsv`, con lo stato di ogni volume;
  - `LEGGIMI/`, con questo documento e `docs/backup/`.

  Fuori da quella cartella restano le prove: `LimiteVL-prova-client-condiviso/` e `LimiteVL-backup/prove/`.
- **Aggiunte future.** Il piano contiene anche le categorie escluse, con le stesse firme. Per aggiungerle più tardi
  basta creare e caricare senza `--solo-nucleo` / `nucleo`.

## 4. Procedura eseguita

Ogni calcolo si fa in un job CPU. Sul frontend, per deroga dell'utente, si fanno solo tre cose: configurare il client,
caricare file già pronti e una piccola prova di recupero.

Ogni comando rclone passa da `scripts/backup_rclone.sh`, quindi da `guarded_run.sh`:
- un solo trasferimento alla volta;
- avvio solo con almeno 1 GiB di memoria libera;
- stop sotto 768 MiB liberi o sopra 128 MiB di RSS dell'albero di processi, controllati ogni 2 s;
- banda di 10 MiB/s;
- nessun riavvio automatico, salvo la regola A dell'utente: dopo uno stop per memoria si riparte solo con 10 minuti
  di fila sopra 1 GiB, al più 5 stop, portati a 20 dall'utente il 2026-11-13 (`trasferimento/stato/<gen>/max-stop-memoria`).

Il tetto cgroup facoltativo della guardia è spento nel caricamento in background (`GUARD_CGROUP=off`). L'account non ha
il "linger": alla disconnessione systemd chiude il suo ambito, e il 2026-11-12 alle 20:07 questo ha interrotto un
caricamento. RSS osservata di rclone: 65-81 MB.

1. **rclone:** scaricato con `fetch.sh`; il job 30939 ne controlla impronta della chiave, firma PGP e sha256.
2. **Inventario** (job 30943; controllo del 2026-11-13 col job 30972): niente collisioni di nomi, quindi sicuro su
   macOS.
3. **Piano** (job 30944): categorie, volumi, prima il nucleo.
4. **Creazione** (`slurm/backup_crea.sbatch … --attendi --solo-nucleo`):
   - sha256 per file, per parte e per flusso;
   - rilettura completa del volume ed elenco di GNU tar;
   - staging limitato a 30 GiB.
5. **Caricamento** (`scripts/backup_upload.sh lvl-ist g1-20261112 <piano> nucleo`):
   - controlla dimensione, md5 e sha256 calcolati da Drive contro quelli del job;
   - dopo la verifica libera lo staging;
   - non sincronizza e non cancella nulla su Drive.
6. **Codice per ultimo:** inventario finale, piano della sola categoria `codice`, volume nella stessa generazione.
7. **Completezza** (`backup.py completezza --solo-nucleo`): controlla volumi, parti e indici verificati, e i membri
   dei piani contro gli inventari.

```bash
sbatch --parsable slurm/backup_crea.sbatch trasferimento/piani/g1-20261112 --attendi --limite-staging-gib 30 --solo-nucleo
GUARD_CGROUP=off nohup setsid scripts/backup_upload.sh lvl-ist g1-20261112 trasferimento/piani/g1-20261112 nucleo \
  > trasferimento/logs/upload-g1-$(date +%Y%m%d-%H%M%S).log 2>&1 < /dev/null &
touch trasferimento/stato/STOP-UPLOAD    # arresto gentile del caricamento
touch trasferimento/stato/STOP           # arresto gentile della creazione
```

## 5. Verifiche

| Livello | Come | Esito |
| --- | --- | --- |
| Integrità dei file trasferiti | md5 e sha256 di Drive uguali a quelli calcolati dal job, su ogni parte e ogni indice | [`backup/g1-20261112-completezza.json`](backup/g1-20261112-completezza.json) |
| Completezza rispetto all'inventario | membri dei piani contro gli inventari 30943 (nucleo) e finale (codice); indici contro membri | idem |
| Ripristino di archivi, struttura e link | ogni volume riletto per intero ed elencato da GNU tar prima del caricamento; volume di prova riscaricato da Drive ed estratto con `cat \| tar -xpz` (job 30947: 5.273 file e 513 cartelle uguali all'indice, `sha256sum -c`); prova dal Mac (2026-11-12); prove su volumi reali riscaricati, nel rapporto di completezza | idem |
| Ambiente | **da fare** su un nuovo server: i venv non sono nel backup selezionato e vanno ricostruiti (§8) | aperto |

Un caricamento finito non basta per dire che il backup è verificato. Lo stato di ogni volume è in
`indice/indice-volumi.tsv`.

## 6. Checkpoint e ripresa dei run

[`backup/checkpoint.tsv`](backup/checkpoint.tsv) dice, cartella per cartella, se ci sono pesi, `optimizer.pt` e
`trainer_state.json`.

- **`f2-abl-adamw` e `f2-abl-normuon`:** in `checkpoints/step-*` ci sono pesi, stato dell'ottimizzatore e
  `trainer_state.json`. Quest'ultimo registra passo, token visti, posizione dei dati per worker e digest della config.
- **`snapshots/` e `final/`:** solo pesi.
- **RNG:** manca uno stato separato. Una ripresa riparte da pesi, ottimizzatore e posizione dei dati, ma non è
  dimostrato che sia identica bit per bit.

## 7. Dati esclusi e loro recupero

- **Dataset pubblici.** Per ogni file [`backup/catalogo_recupero.tsv`](backup/catalogo_recupero.tsv) riporta:
  repository, commit completo, revisione richiesta, nome, byte, hash di Hugging Face (sha256 per i file LFS, sha1 git
  per gli altri), se è stato scaricato (quali parquet, per esempio i primi 25 su 75 di PixMo), ruolo e accesso.
  - **Accesso:** tutti pubblici. Sono stati scaricati senza token.
  - **Recupero:** `scripts/fetch_manifest.sh downloads/manifests/<gruppo>.tsv`. I manifest sono nel volume del codice
    e usano URL con il commit esatto, mai `main`. Dopo si eseguono `link_hf_snapshot.py` e `slurm/verify_hashes.sbatch`.
  - **Se un repository sparisse:** l'elenco serve a riconoscere una copia equivalente per hash.
- **Copie sul Mac.** Sul Mac ci sono circa 152 GiB, soprattutto PixMo, MAVIS e UniMER: probabilmente proprio le
  sorgenti grezze escluse.
  - Non contano come verificate finché il contenuto non coincide. Il controllo si fa con
    `sh scripts/mac_confronta_hash.sh <cartella> hash_riferimento_dati.tsv catalogo_recupero.tsv`.
  - Il confronto è per sha256, anche con nomi diversi. Usa i file LFS del catalogo e i file di `hf/` e `data/shards/`
    salvati.
- **Shard derivati esclusi.** Per ricostruirli:
  - sorgente, job, launcher e nota del ledger stanno in [`backup/derivati.tsv`](backup/derivati.tsv);
  - parametri, conteggi e sha256 degli shard stanno nei manifest copiati in [`backup/derivati_esclusi/`](backup/derivati_esclusi/).

  | Shard | Ricostruzione |
  | --- | --- |
  | `llava-pretrain/70f9d1e5` | `slurm/shard_llava_pretrain.sbatch` (job 29388), shard da circa 1 GB, da `images.zip` e dalle didascalie |
  | `mmfinereason/7c93f724-sft`, `-rl` | `slurm/shard_parquet.sbatch` (job 29556, 29466), 4 parquet per shard |
  | `unimer/2343ddd9`, `2343ddd9-x4` | `slurm/shard_unimer.sbatch` (job 29438, 29651), scala 2 o 4, al più 1024 patch |

  Servono codice del repository, ambiente `venvs/build-29382` ricostruito e sorgente alla revisione indicata. La
  riproduzione byte per byte non è verificata per questi shard: è nota solo per PixMo-Cap (job 30813) e per i sintetici
  v3 (job 30393), che sono comunque nel backup.

## 8. Ripartire con un altro utente o server

1. **Radice.** Conviene tenere la stessa, `/home/prignano/vision_limite`, anche come link. Così valgono senza
   modifiche:
   - i percorsi letterali di `config/paths.env`, delle direttive `#SBATCH` e dei default degli script
     ([`backup/percorsi_assoluti.tsv`](backup/percorsi_assoluti.tsv));
   - i 6 link assoluti interni (`data/eval-datasets/*` e `outputs/eval-models/*/snapshots/*`).
2. **Altrimenti, rimappatura controllata,** in un solo commit a parte:
   - sostituire la radice nei file di `config/`, `scripts/`, `slurm/`, `src/`, `tests/`, `downloads/` ed `env*/`
     elencati, e ricreare i 6 link;
   - **non** riscrivere manifest, indici, report o log: registrano la vecchia radice come provenienza e i loro sha256
     sono citati altrove;
   - i reader risolvono gli shard rispetto alla radice del repository (`manifest.shard_path`); un percorso di
     provenienza si legge sostituendo la radice, senza toccare il file.
3. **Estrazione.** Prima `codice`, poi gli altri volumi nell'ordine del piano. Su un cluster si fa in un job, con
   `scripts/ripristina_volumi.sh … nucleo` oppure con `backup.py ripristino --esistente` volume per volume.
4. **Ambiente.** Si ricostruisce offline da `wheels/`, dai lock e da uv in `tools/` con `scripts/setup_env.sh install`.
   Servono Linux x86_64, Python 3.12, wheel compatibili con glibc e driver NVIDIA ≥ 570 per torch cu129. Poi si lancia
   `slurm/f0_checks.sbatch`. L'identità dei build originali è in [`backup/ambienti.tsv`](backup/ambienti.tsv).
5. **Git.** Serve una nuova chiave SSH registrata su GitHub; il remote è
   `git@github.com:akira-n-28/vision_limite.git`.
6. **Istruzioni.** Su un altro cluster vanno riscritte le parti di `LAGRANGE.md` e `CLAUDE.md` che riguardano
   Lagrange.

## 9. Scaricare il backup sul Mac

Il backup selezionato occupa circa 125 GiB di archivi, perché i testi sono compressi, e si estrae in 138 GiB. Con
406 GiB liberi si possono tenere archivi ed estratto; con `--elimina-parti` serve meno spazio.

1. **Scaricare:**
   - `LEGGIMI/backup/` (gli elenchi);
   - `indice/volumi/*.sha256`;
   - le cartelle `volumi/<categoria>/`.

   Va bene il browser, un file alla volta (se Drive li unisce in uno zip, aprirlo prima), oppure l'app Google Drive
   per desktop, copiando da `~/Library/CloudStorage/GoogleDrive-…`. Con rclone si usa lo stesso client OAuth:
   `drive.file` vede i file creati da quel client.
2. **Ripristinare:**
   `sh ripristina_volumi.sh g1-20261112-volumi.tsv g1-20261112-parti.tsv ~/Downloads/volumi ~/Downloads/indici ~/limite-vl nucleo`.
   Per ogni volume controlla lo sha256 delle parti, estrae con `cat | tar -xp` e verifica i file con
   `shasum -a 256 -c`. Per riprendere basta rilanciarlo.
3. **Ambiente:** quello Linux/CUDA non si esegue sul Mac.

## 10. Aggiornare il backup prima del cambio di account

1. **Rinnovare l'autorizzazione** se il token è scaduto: l'app è in prova, quindi dura 7 giorni (fino al 19 novembre
   2026), a meno di pubblicarla.
   - togliere il vecchio token: `scripts/backup_rclone.sh config -- config delete lvl-ist`;
   - avviare `scripts/backup_auth.sh avvia lvl-ist`;
   - rifare il login con `ssh -N -L 53682:127.0.0.1:53682 …`.
2. **Nuovo inventario e piano** con `--precedente trasferimento/piani/g1-20261112/piano.json`. Le categorie con la
   stessa firma si riusano; solo quelle cambiate hanno volumi nuovi (`riusata_da` in `categorie.tsv`). Il codice cambia
   sempre.
3. **Creazione, caricamento e completezza** come in §4, con le stesse opzioni del nucleo. Per ripristinare, ogni
   categoria si prende dalla generazione indicata in `riusata_da`.
4. **Prima che l'account istituzionale venga disattivato**, cioè alla fine degli studi, copiare il backup su un'altra
   destinazione.

## 11. Limiti residui

- **Account e client.** Il backup dipende dall'account istituzionale e dal progetto Google Cloud del client. Con
  `drive.file` rclone vede solo i file creati da quel client; il sito di Drive li mostra sempre.
- **Dati esclusi.** Restano recuperabili solo finché i repository Hugging Face esistono, o tramite copie verificate
  per hash, per esempio quelle sul Mac.
- **Cosa si conserva.** Permessi, date al secondo e link sì. Proprietari e gruppi solo estraendo da root. ACL e
  attributi estesi no.
- **Verifiche su Drive.** Confrontano gli hash calcolati da Google. Sono stati riscaricati e ripristinati per intero
  solo il volume di prova e quelli indicati nel rapporto.
- **Ricerca dei segreti.** È fatta con regole sui file di testo fino a 16 MiB, fuori da dati, pesi e ambienti di
  terzi.
- **Ambiente.** Non è stato ancora ricostruito su un altro server.

## 12. Stato del progetto e lavoro da riprendere

Il progetto è in **F0 preparatorio** (pipeline v1.5): non c'è nessun pilota, training o audit approvato.
- **Stato e prossimi interventi:** [`F0_IMPLEMENTATION_PLAN.md`](F0_IMPLEMENTATION_PLAN.md) e i fatti in `CLAUDE.md`.
- **Ultimo lavoro prima del backup:** `synth-v4-candidato-3` verificato (job 30923), con la config di collaudo dei
  sintetici e il rapporto della scorciatoia di function/exists (commit `f10413d`).
- **Decisioni aperte:** D1-D6 del piano F0.
- **Registro dei job:** `runs/ledger.tsv`.
