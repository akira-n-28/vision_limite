# Backup e ripristino di Limite-VL

Incarico del 2026-11-12: mettere al sicuro il progetto per un cambio di utente su Lagrange o un trasferimento su un
altro server, senza toccare la pipeline scientifica.

Ci sono due generazioni, da ripristinare in quest'ordine:
- **`g1-20261112`**, il backup **selezionato** (il nucleo);
- **`s1-20261113`**, un **supplemento autonomo** che dipende da g1. Aggiunge o sostituisce il codice e la storia Git,
  compresi i risultati della revisione MAVIS-C9, più `UniMER-Test.zip` (§10).

I numeri definitivi stanno negli elenchi generati in [`docs/backup/`](backup/): volumi, byte, hash ed esito delle
verifiche.

**Correzioni del 2026-11-13**, dopo la revisione dei push fino a `f1aa291`:
- un file atteso ma non archiviato, un inventario con errori di visita, un marcatore di un altro piano o un file non
  volatile cambiato dopo l'inventario impediscono di dichiarare un volume pronto o un backup completo;
- la copertura si calcola dalle voci davvero archiviate;
- il percorso incrementale (`--precedente`) è rifiutato.

Con queste regole g1 è stato riesaminato (§5): risulta completo.

## 1. Sintesi

| Voce | Valore |
| --- | --- |
| Modalità | **backup selezionato** (il nucleo) sul Google Drive istituzionale Sapienza (`studenti.uniroma1.it`); decisione dell'utente del 2026-11-13 |
| Storia | partito il 2026-11-12 come backup completo; dopo l'interruzione notturna ridotto al nucleo nella stessa generazione; nessuna divisione fra account |
| Spazio | 800 GB per utente, 5,16 GB usati prima del backup (pagina web di Drive); l'API mostra solo lo spazio dell'istituzione |
| Accesso | rclone v1.76.0 dal frontend, ambito OAuth `drive.file` (solo i file creati dall'app), client OAuth proprio dell'utente (progetto Google Cloud `limite-vl-backup`, app desktop in prova) |
| Posizione | cartella privata `LimiteVL-backup/g1-20261112/`; unico permesso il proprietario, nessun link |
| Dimensione | 44 volumi (43 del nucleo più il codice) in 45 file; 138,2 GiB di dati (18.383 voci), 121,8 GiB caricati perché i testi sono compressi; esclusi 404,2 GiB, documentati per il recupero (§7) |
| Esito | completezza senza problemi (job 31104, 2026-11-13): 18.383 voci attese, 18.383 nei volumi; ogni parte e ogni indice con dimensione, md5 e sha256 di Drive uguali a quelli del job; ripristini di prova superati (job 30947, 31101, 31102) |
| Stato del progetto in g1 | dati all'inventario 30943 (commit `f10413d`), invariati all'inventario 30972 del 2026-11-13; volume del codice dall'inventario 31097 (commit `674e3cd` e ledger) |
| Supplemento | `s1-20261113` nella cartella privata `LimiteVL-backup/s1-20261113/`: codice e storia Git al commit dichiarato in [`backup/s1-20261113-stato.json`](backup/s1-20261113-stato.json), risultati MAVIS-C9 compresi, e `UniMER-Test.zip` (§10) |

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
- **Aggiunte future.** Il piano di g1 contiene anche le categorie escluse. Si possono aggiungere nella stessa
  generazione creando e caricando senza `--solo-nucleo` / `nucleo`, ma solo finché i loro file sono uguali
  all'inventario 30943: la creazione si ferma su ogni file mancante o cambiato. Altrimenti serve una generazione nuova.

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
7. **Completezza** (`backup.py completezza`, `slurm/backup_completezza.sbatch GEN "PIANI" [--solo-nucleo]`). Esito
   ed exit code dicono se il backup è completo. Si controlla che:
   - ogni piano sia integro e il suo inventario sia quello registrato, senza errori di visita o lettura;
   - ogni volume abbia il marcatore di questa generazione, di questo piano e di questo volume;
   - parti e indici siano verificati su Drive e ogni indice contenga tutti e soli i membri del volume;
   - nessun avviso sia bloccante;
   - le voci attese (inventario, categorie del piano, meno le esclusioni dichiarate) coincidano con quelle davvero
     archiviate nei volumi verificati.

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
| Integrità dei file trasferiti | md5 e sha256 di Drive uguali a quelli calcolati dal job, su ogni parte (45) e ogni indice (88) | superata: [`backup/g1-20261112-completezza.json`](backup/g1-20261112-completezza.json) (job 31104) |
| Completezza rispetto all'inventario | membri dei piani contro gli inventari 30943 (nucleo, tranne codice) e 31097 (codice); indici contro membri | superata: 18.383 voci su 18.383, nessuna mancante; un avviso atteso, il ledger cambiato dopo l'inventario finale |
| Ripristino di archivi, struttura e link | ogni volume riletto per intero ed elencato da GNU tar prima del caricamento; volumi riscaricati da Drive ed estratti con `cat \| tar -xp`, confronto con l'indice e `sha256sum -c`: prova del codice (job 30947: 5.273 file e 513 cartelle), `outputs-f2-abl-adamw-002` (job 31101: 2 parti, `optimizer.pt` da 7,8 GiB), `outputs-f2_select-001` (job 31102, gzip: 48 file e 14 cartelle); prova dal Mac del 2026-11-12 | superata: [`backup/prove-ripristino/`](backup/prove-ripristino/) |
| Riesame del 2026-11-13 con le regole corrette | inventari 30943 e 31097 senza errori; marcatori legati ai rispettivi piani; avvisi tutti motivati (il ledger); copertura dalle voci archiviate | completo: 44 volumi su 44, 18.383 voci attese e archiviate, 0 mancanti, 0 in più ([`backup/riesame/`](backup/riesame/), job 31107) |
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
3. **Estrazione.** Prima g1 (il codice e poi gli altri volumi, nell'ordine dell'elenco), poi il supplemento con
   l'elenco dei rimossi (§9). Su un cluster si fa in un job, con `scripts/ripristina_volumi.sh` presa da una copia
   esterna alla radice, oppure con `backup.py ripristino --esistente` volume per volume.
4. **Ambiente.** Si ricostruisce offline da `wheels/`, dai lock e da uv in `tools/` con `scripts/setup_env.sh install`.
   Servono Linux x86_64, Python 3.12, wheel compatibili con glibc e driver NVIDIA ≥ 570 per torch cu129. Poi si lancia
   `slurm/f0_checks.sbatch`. L'identità dei build originali è in [`backup/ambienti.tsv`](backup/ambienti.tsv).
5. **Git.** Serve una nuova chiave SSH registrata su GitHub; il remote è
   `git@github.com:akira-n-28/vision_limite.git`.
6. **Istruzioni.** Su un altro cluster vanno riscritte le parti di `LAGRANGE.md` e `CLAUDE.md` che riguardano
   Lagrange.

## 9. Ripristinare base e supplemento (sul Mac o su un altro computer)

Il ripristino si fa in due passaggi, ognuno con i propri elenchi e le proprie cartelle: prima g1, poi il supplemento.
I volumi del supplemento hanno il prefisso `s1-`. I marcatori di ripristino registrano nome e sha256 del flusso,
quindi nessun volume viene saltato o confuso con uno della base.

**Spazio.** g1 sono 121,8 GiB di archivi in 45 file e si estrae in 138,2 GiB; il supplemento è di circa 0,2 GiB. Con
406 GiB liberi si possono tenere archivi ed estratto; con `--elimina-parti` serve meno spazio.

1. **Scaricare**, in cartelle separate per generazione:
   - `g1-20261112/LEGGIMI/backup/` e `s1-20261113/LEGGIMI/` (elenchi e script aggiornati);
   - `<generazione>/indice/volumi/*.sha256`;
   - `<generazione>/volumi/<categoria>/`.

   Va bene il browser, un file alla volta (se Drive li unisce in uno zip, aprirlo prima), oppure l'app Google Drive
   per desktop, copiando da `~/Library/CloudStorage/GoogleDrive-…`. Con rclone si usa lo stesso client OAuth:
   `drive.file` vede i file creati da quel client.
2. **Usare lo script aggiornato,** `s1-20261113/LEGGIMI/script/ripristina_volumi.sh`. Va tenuto fuori dalla cartella
   che si estrae, perché il volume del codice la sovrascrive. Si lancia con `bash`.
3. **Primo passaggio, g1:**
   ```sh
   bash ~/Downloads/s1/LEGGIMI/script/ripristina_volumi.sh \
     ~/Downloads/g1/LEGGIMI/backup/g1-20261112-volumi.tsv ~/Downloads/g1/LEGGIMI/backup/g1-20261112-parti.tsv \
     ~/Downloads/g1/volumi ~/Downloads/g1/indici ~/limite-vl nucleo
   ```
4. **Secondo passaggio, il supplemento**, con l'elenco dei file del codice rimossi o rinominati rispetto a g1:
   ```sh
   bash ~/Downloads/s1/LEGGIMI/script/ripristina_volumi.sh \
     ~/Downloads/s1/LEGGIMI/backup/s1-20261113-volumi.tsv ~/Downloads/s1/LEGGIMI/backup/s1-20261113-parti.tsv \
     ~/Downloads/s1/volumi ~/Downloads/s1/indici ~/limite-vl tutti \
     --rimuovi-elenco ~/Downloads/s1/LEGGIMI/backup/s1-20261113-rimossi.txt
   ```

Per ogni volume lo script fa queste operazioni:
- controlla che nell'elenco risulti "verificato";
- controlla lo sha256 di ogni parte;
- estrae con `cat | tar -xp`;
- verifica i file con `shasum -a 256 -c`.

Un errore ferma tutto: niente marcatore, e nessuna parte cancellata. Per riprendere basta rilanciare. Funzionano anche i
percorsi relativi.

L'ambiente Linux/CUDA non si esegue sul Mac.

## 10. Aggiornare il backup: supplemento autonomo

Il percorso incrementale (`piano --precedente`) è **rifiutato**: la completezza non risolve le categorie riusate da
un'altra generazione. Un aggiornamento è una generazione nuova e autonoma, con la dipendenza dalla base dichiarata.

1. **Rinnovare l'autorizzazione** se il token è scaduto: l'app è in prova, quindi dura 7 giorni a partire dal login,
   a meno di pubblicarla.
   - togliere il vecchio token: `scripts/backup_rclone.sh config -- config delete lvl-ist`;
   - avviare `scripts/backup_auth.sh avvia lvl-ist`;
   - rifare il login con `ssh -N -L 53682:127.0.0.1:53682 …`.
2. **Fissare lo stato:** fare il commit di tutto ciò che va conservato e non cambiare nulla durante la raccolta.
   Fanno eccezione i file volatili (ledger, log dei download, indice di git): un loro cambiamento resta un avviso
   motivato.
3. **Nuovo inventario** (`slurm/backup_inventario.sbatch`). Poi `backup.py firme --piano-base <piano di g1>` dice quali
   categorie sono cambiate. Nel supplemento vanno tutte quelle cambiate del nucleo, più eventuali aggiunte con un
   sottoinsieme dichiarato.
4. **Piano del supplemento**, in una cartella e una generazione nuove:
   ```bash
   sbatch slurm/backup_piano.sbatch <inventario> <generazione> <generazione> --solo-categorie codice,<altre> \
     --sottoinsieme <categoria>=<elenco> --prefisso-volumi s<N>- --dipende-da g1-20261112
   ```
5. **Creazione, caricamento e completezza** come in §4, con lo stato della generazione nuova. Poi
   `backup.py rimossi --base <indici del codice della base> --nuovo <indici del codice del supplemento>` per l'elenco
   dei file da togliere dopo il secondo passaggio.
6. **Ripristino:** prima la base, poi il supplemento (§9).
7. **Prima che l'account istituzionale venga disattivato**, cioè alla fine degli studi, copiare base e supplementi su
   un'altra destinazione.

### Supplemento s1-20261113

| Voce | Contenuto |
| --- | --- |
| Dipende da | `g1-20261112`, da ripristinare prima |
| Sostituisce | la categoria `codice`: repository con `.git` intero, configurazioni, documenti, revisioni umane (compresi gli esiti MAVIS-C9 in `revisioni/mavis-c9-campione-1/risultati/`, TSV e JSON di avanzamento), script, rapporti e istruzioni del backup, `third_party/`, `runs/ledger.tsv`, `data/download-log.tsv`; i file assenti rispetto a g1 sono in `s1-20261113-rimossi.txt` |
| Aggiunge | `UniMER-Test.zip` originale (revisione `2343ddd9`, 146.727.503 byte, sha256 `9bf370b8…`, come nel catalogo) con la scheda README e i collegamenti dello snapshot ([`backup/supplemento/sottoinsieme-unimer.txt`](backup/supplemento/sottoinsieme-unimer.txt)); `UniMER-1M.zip` resta escluso, dichiarato nel piano |
| Restano in g1 | tutte le altre categorie del nucleo, invariate (confronto delle firme in [`backup/s1-20261113-stato.json`](backup/s1-20261113-stato.json)) |
| Stato conservato | il commit indicato in `backup/s1-20261113-stato.json`. Il rapporto conclusivo, che contiene gli elenchi del supplemento, è un commit successivo distribuito a parte su GitHub e in `s1-20261113/LEGGIMI/` |

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
- **Supplemento.** Contiene solo ciò che è cambiato dopo g1, cioè il codice, più `UniMER-Test.zip`. Le categorie dei
  dati sono uguali a g1 secondo il confronto delle firme. Un cambiamento futuro dei dati richiede un nuovo supplemento.
- **Verifiche precedenti alle correzioni.** I rapporti di completezza del 2026-11-12/13 (job 31104) usavano le regole
  vecchie. Il riesame (job 31107) le sostituisce, senza cambiare volumi né indici.

## 12. Stato del progetto e lavoro da riprendere

Il progetto è in **F0 preparatorio** (pipeline v1.5): non c'è nessun pilota, training o audit approvato.
- **Stato e prossimi interventi:** [`F0_IMPLEMENTATION_PLAN.md`](F0_IMPLEMENTATION_PLAN.md) e i fatti in `CLAUDE.md`.
- **Ultimo lavoro prima del backup:** `synth-v4-candidato-3` verificato (job 30923), con la config di collaudo dei
  sintetici e il rapporto della scorciatoia di function/exists (commit `f10413d`).
- **Dopo il backup:** consegnati dal portatile gli esiti della revisione visiva MAVIS-C9
  (`revisioni/mavis-c9-campione-1/risultati/`, commit `d24560a`), conservati nel supplemento. Non sono ancora integrati
  nei manifest: è il prossimo lavoro scientifico.
- **Decisioni aperte:** D1-D6 del piano F0.
- **Registro dei job:** `runs/ledger.tsv`.
