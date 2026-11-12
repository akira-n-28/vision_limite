# Backup e ripristino di Limite-VL

Incarico del 2026-11-12: mettere al sicuro il progetto per un cambio di utente su Lagrange o un trasferimento su un
altro server. Non tocca la pipeline scientifica. Generazione corrente: **`g1-20261112`**. I numeri definitivi
(volumi, byte, hash, esito delle verifiche) sono negli elenchi generati in [`docs/backup/`](backup/).

## 1. Sintesi

| Voce | Valore |
| --- | --- |
| Modalità | **backup completo** su Google Drive istituzionale (Sapienza, dominio `studenti.uniroma1.it`); nessuna divisione fra account |
| Spazio | per utente 800 GB, 5,16 GB usati prima del backup (pagina web di Drive, 2026-11-12); l'API mostra solo lo spazio dell'istituzione (2,9 PiB) |
| Accesso | rclone v1.76.0 dal frontend, ambito OAuth `drive.file` (solo i file creati dall'app), client OAuth proprio dell'utente (progetto Google Cloud `limite-vl-backup`, app desktop in prova) |
| Posizione | cartella privata `LimiteVL-backup/g1-20261112/` nel Drive dell'account istituzionale; unico permesso il proprietario, nessun link |
| Dimensione | 542,4 GiB di dati in 38 categorie e 134 volumi, più il volume finale del codice; nucleo 138,2 GiB (§7) |
| Stato del progetto | commit `f10413d` all'inventario dei dati (job 30943); il codice si salva per ultimo, dopo i documenti del backup |

## 2. Cosa contiene e cosa no

**Tutta la cartella** `/home/prignano/vision_limite`, file nascosti, ignorati e non tracciati compresi: codice e
documenti, `.git` (storia, riferimenti, tag locale `pre-migrazione-v1.4` non pubblicato, reflog), revisioni umane,
`data/` (manifest di tutte le versioni, decontaminazione, pannello, lingua, specifiche degli snapshot, shard), `hf/`
(modelli, benchmark e sorgenti alle revisioni fissate), `outputs/` (checkpoint, generazioni, audit, valutazioni,
report, log), `wheels/`, `tools/`, `venvs/`, `cache/`, `third_party/` (cloni puliti, con `.git`), i manifest dei
download (`downloads/manifests/*.tsv`, fuori da git).

**Escluso**, solo materiale creato per il trasferimento: `trasferimento/` (staging dei volumi, stato del caricamento,
inventari e piani, log dei job del backup, configurazione e token di rclone, `client.env` del client OAuth). È in
`.gitignore`; gli inventari e i piani sono comunque su Drive in `indice/`.

**Segreti.** Nessuno nel progetto: `secrets/` è vuota (il token Hugging Face non è mai stato salvato: tutti i download
erano pubblici). La ricerca sui file di testo e sugli oggetti git (job 30938, 30943; regole in
`trasferimento/inventari/*/segreti.json`, mai il testo trovato) ha dato solo segnaposto di test e documentazione
(`tests/test_review_results.py`, `tests/test_backup.py`, `third_party/verifiers`). Per questo **non c'è un archivio
cifrato**: se in futuro `secrets/` contenesse qualcosa, il piano si ferma da solo (§9). Le credenziali del
trasferimento (token di rclone, segreto del client OAuth) restano solo in `trasferimento/rclone/` (permessi 600).

**Dipendenze fuori dalla cartella** (non salvate, da ricreare): l'interprete di sistema `/usr/bin/python3.12` (3.12.3)
a cui puntano i venv; la chiave SSH per GitHub e `~/.gitconfig` dell'utente; il driver NVIDIA del cluster; il
progetto Google Cloud del client OAuth. Nessun hard link verso l'esterno; nessun worktree, alternates o submodule.
Elenco dei link in [`backup/collegamenti_assoluti_o_esterni.tsv`](backup/collegamenti_assoluti_o_esterni.tsv).

## 3. Formato

- **Categorie** (`scripts/backup.py`, `category`): una partizione dell'albero, ognuna ripristinabile da sola:
  `codice` (repository, `third_party/`, `data/download-log.tsv`, `data/.fetch.lock`), `data-metadati`,
  `data-shards-<sorgente>-<versione>`, `hf-<repository>` (sopra 1 GiB) e `hf-piccoli`, `outputs-<cartella>` (sopra
  1 GiB) e `outputs-resto`, `ambienti` (`venvs/` e `cache/uv` insieme: uv installa con hard link alla sua cache, 11 GiB
  condivisi), `cache-altre`, `wheels`, `strumenti`. Elenco in [`backup/g1-20261112-categorie.tsv`](backup/g1-20261112-categorie.tsv).
- **Volumi**: tar POSIX (PAX) standard, circa 5 GiB di dati ciascuno, gzip solo dove i file sono testo (codice,
  ambienti, generazioni F2). Un file più grande di 5 GiB ha un volume a sé. Un volume sopra 5 GiB è diviso in parti
  `.partNNN` da 5 GiB, da concatenare con `cat`. Link simbolici conservati come link, mai dereferenziati; i nomi di un
  hard link stanno nello stesso volume. Le date dei file sono al secondo.
- **Indici**: per ogni volume `indice/volumi/<volume>.jsonl.gz` (percorso, tipo, permessi, data, uid, gid, link,
  sha256 di ogni file) e `<volume>.sha256` nel formato di `sha256sum`. Per ogni parte: dimensione, sha256 e md5 in
  [`backup/g1-20261112-parti.tsv`](backup/g1-20261112-parti.tsv).
- **Su Drive**: `LimiteVL-backup/g1-20261112/volumi/<categoria>/<volume>.tar[.gz][.partNNN]`, `indice/piano/` (piano,
  membri di ogni volume, categorie, checkpoint), `indice/volumi/` (indici), `indice/indice-volumi.tsv` (stato:
  pianificato, pronto, verificato), `LEGGIMI/` (questo documento e `docs/backup/`). Restano due cartelle delle prove:
  `LimiteVL-prova-client-condiviso/` (prova col client condiviso di rclone) e `LimiteVL-backup/prove/`; si possono
  cancellare dal sito.

## 4. Procedura eseguita

Ogni calcolo in un job CPU; sul frontend, per deroga dell'utente del 2026-11-12, solo configurazione del client,
caricamento dei file pronti e la piccola prova di recupero. Ogni comando rclone passa da `scripts/backup_rclone.sh`,
quindi da `scripts/guarded_run.sh`: un trasferimento alla volta (lock di `fetch.sh`), avvio con almeno 1 GiB di
MemAvailable, stop sotto 768 MiB o sopra 128 MiB di RSS dell'intero albero di processi (più il tetto cgroup
`MemoryMax=128M`), banda 10 MiB/s (`--bwlimit 10M` e controllo della guardia), nessun riavvio automatico. Verificato
che la sessione sorvegliata contenga il processo di rclone; RSS osservata 65-80 MB.

1. **rclone**: `fetch.sh` di `version.txt`, zip, `SHA256SUMS` firmato, pagina `release_signing` e chiave;
   `slurm/backup_rclone.sbatch` (job 30939) controlla impronta `FBF737EC…54FA`, firma PGP e sha256 dello zip ed estrae
   `tools/rclone/rclone`.
2. **Inventario** (`slurm/backup_inventario.sbatch`, job 30943): ogni voce con tipo, dimensione, inode, link, permessi,
   date; git, link, hard link, nomi problematici (nessuna collisione per maiuscole: sicuro su macOS), segreti.
3. **Piano** (`slurm/backup_piano.sbatch`, job 30944): categorie, volumi, ordine (prima il nucleo), compressione, firme.
4. **Creazione** (`slurm/backup_crea.sbatch`, job 30948 e seguenti): ogni file letto una volta (sha256 per file, per
   parte e per flusso), rilettura completa del volume e confronto con l'indice, elenco di GNU tar uguale all'indice;
   staging al massimo 30 GiB; il job aspetta che il caricamento verificato liberi spazio.
5. **Caricamento** (`scripts/backup_upload.sh`, in background sul frontend): una parte alla volta, poi elenco su Drive
   con dimensione, md5 e sha256 calcolati da Google, confrontati con quelli del job; solo dopo la verifica la parte si
   cancella dallo staging. Nessuna sincronizzazione, nessuna cancellazione su Drive. Ripresa: si rilancia lo stesso
   comando.
6. **Codice per ultimo**: dopo il commit dei documenti, nuovo inventario, piano della sola categoria `codice` e suo
   volume nella stessa generazione.
7. **Completezza** (`backup.py completezza`): ogni volume pronto e verificato su Drive parte per parte, ogni indice su
   Drive con gli stessi hash, i membri dei piani coprono l'inventario e gli indici coincidono coi membri.

Comandi tipici (dal frontend; `jobid` nel ledger a ogni invio):

```bash
sbatch --parsable slurm/backup_inventario.sbatch
sbatch --parsable slurm/backup_piano.sbatch trasferimento/inventari/inv-<job> <gen> <gen> --escludi-categorie codice
sbatch --parsable slurm/backup_crea.sbatch trasferimento/piani/<gen> --attendi --limite-staging-gib 30
nohup setsid scripts/backup_upload.sh lvl-ist <gen> trasferimento/piani/<gen> \
  > trasferimento/logs/upload-<gen>-$(date +%Y%m%d-%H%M%S).log 2>&1 < /dev/null &
touch trasferimento/stato/STOP-UPLOAD    # arresto gentile del caricamento
touch trasferimento/stato/STOP           # arresto gentile della creazione
```

## 5. Verifiche

| Livello | Come | Esito |
| --- | --- | --- |
| Integrità dei file trasferiti | md5 e sha256 di Drive uguali a quelli calcolati dal job su ogni parte e ogni indice | [`backup/g1-20261112-completezza.json`](backup/g1-20261112-completezza.json) |
| Completezza rispetto all'inventario | membri dei piani contro inventari 30943 (dati) e finale (codice); indici contro membri | idem |
| Ripristino di archivi, struttura e link | ogni volume riletto per intero ed elencato da GNU tar prima del caricamento; prova completa sul volume di prova riscaricato da Drive (job 30947: `cat \| tar -xpz`, 5.273 file e 513 cartelle uguali all'indice, `sha256sum -c`); prova dal Mac (somma del flusso e `shasum -c`, 2026-11-12); prove di ripristino di volumi reali riscaricati (§5 di completezza) | idem e `trasferimento/stato/*/ripristino/` |
| Ambiente | **da eseguire** su un nuovo server: i venv sono Linux x86_64 con CUDA 12.9 e non sono rilocabili (§8) | aperto |

Un caricamento finito non è un backup verificato: lo stato di ogni volume è in `indice/indice-volumi.tsv` e nel
rapporto di completezza.

## 6. Checkpoint e ripresa dei run

`backup/checkpoint.tsv` elenca, per ogni cartella di checkpoint in `outputs/`, se ci sono pesi, `optimizer.pt` e
`trainer_state.json`. I checkpoint `checkpoints/step-*` di `f2-abl-adamw` e `f2-abl-normuon` hanno pesi, stato
dell'ottimizzatore e `trainer_state.json` (passo, token visti, posizione dei dati per worker, digest della config); le
cartelle `snapshots/` e `final/` hanno solo i pesi. Non c'è uno stato RNG separato: una ripresa riparte da pesi,
ottimizzatore e posizione dei dati, ma che sia identica bit per bit non è dimostrato. Il progetto non riprende comunque
i run storici (CLAUDE.md).

## 7. Scaricare il nucleo sul Mac

Il nucleo (colonna `nucleo` = 1 di `backup/g1-20261112-volumi.tsv`, circa 138 GiB più il codice) contiene: codice e
Git, `data/` tranne gli shard non scelti, modelli iniziali con tokenizer e processor, benchmark con ruoli e split,
tutto `outputs/`, gli shard di UniMER dense, MAVIS caption, PixMo-Cap, sintetici v2-v4, replay e F2, `wheels/`,
`tools/`. Restano fuori dal nucleo, ma nel backup completo: sorgenti grezze grandi (PixMo, MMFineReason, LLaVA,
MAVIS, DeepVision, …), shard di LLaVA, MMFineReason e UniMER base/x4, `venvs/` e cache.

Spazio: con 406 GiB liberi si può tenere sia il nucleo estratto sia gli archivi; con `--elimina-parti` serve solo il
volume più grande in più (circa 8 GiB).

1. Scaricare `LEGGIMI/backup/` (elenchi) e `indice/volumi/*.sha256` dei volumi del nucleo, poi i volumi del nucleo da
   `volumi/<categoria>/`: dal browser un file alla volta (se Drive li unisce in uno zip, aprirlo), con l'app Google
   Drive per desktop copiandoli dalla cartella `~/Library/CloudStorage/GoogleDrive-…`, o con rclone e lo stesso client
   OAuth (l'ambito `drive.file` vede i file creati da quel client).
2. Ripristino e controlli, nel Terminale:

   ```sh
   sh ripristina_volumi.sh g1-20261112-volumi.tsv g1-20261112-parti.tsv ~/Downloads/volumi ~/Downloads/indici \
      ~/limite-vl nucleo            # aggiungere --elimina-parti per cancellare le parti dopo il controllo
   ```

   Per ogni volume: sha256 di ogni parte, `cat | tar -xp`, `shasum -a 256 -c` di ogni file. Si riprende rilanciando
   (i volumi fatti sono in `~/limite-vl.ripristino-fatti`). Su APFS non ci sono collisioni di nomi (inventario).
3. Copie già presenti sul Mac (circa 152 GiB, soprattutto PixMo, MAVIS e UniMER): non sono verificate né considerate
   disponibili su Lagrange finché il contenuto non coincide.

   ```sh
   sh mac_confronta_hash.sh <cartella dei dataset sul Mac> hash_riferimento_dati.tsv catalogo_recupero.tsv
   ```

   Riconosce per sha256 i file uguali a quelli di `hf/` e `data/shards/` (e ai file LFS del catalogo), anche con nomi
   diversi, ed elenca cosa c'è solo sul Mac e solo su Lagrange.

L'ambiente Linux/CUDA salvato non si esegue sul Mac: sul Mac servono solo per consultare codice, dati e risultati.

## 8. Ripartire con un altro utente o server

1. **Radice.** Se possibile la stessa: `/home/prignano/vision_limite` (anche come link verso la nuova cartella). Così
   valgono senza modifiche i percorsi letterali di `config/paths.env`, delle direttive `#SBATCH`, dei default degli
   script ([`backup/percorsi_assoluti.tsv`](backup/percorsi_assoluti.tsv)) e dei 6 link assoluti interni
   (`data/eval-datasets/*`, `outputs/eval-models/*/snapshots/*`).
2. **Altrimenti, rimappatura controllata.** Una sola modifica del codice, in un commit a parte: sostituire la vecchia
   radice nei file di `config/`, `scripts/`, `slurm/`, `src/`, `tests/`, `downloads/` e `env*/` elencati e ricreare i 6
   link (relativi o con la nuova radice). **Non** riscrivere manifest, indici, report o log: registrano la vecchia
   radice come provenienza e i loro sha256 sono citati altrove. I reader dei manifest risolvono gli shard rispetto
   alla radice del repository (`manifest.shard_path`), quindi funzionano dopo lo spostamento; un percorso assoluto di
   provenienza si legge sostituendo la vecchia radice con la nuova, senza modificare il file.
3. **Estrazione.** Sul nuovo cluster in un job, con `scripts/ripristina_volumi.sh … tutti`, oppure con
   `backup.py ripristino --esistente` volume per volume (confronto completo con l'indice). Prima `codice`, poi gli
   altri nell'ordine del piano; `ambienti` richiede tutti i suoi volumi (hard link).
4. **Ambiente.** I venv (`venvs/build-29382`, `evals-build-29390`) puntano a `/usr/bin/python3.12` e hanno percorsi
   assoluti negli script: funzionano solo con la stessa radice e lo stesso Python di sistema. Altrimenti si ricostruisce
   offline da `wheels/` e dai lock: `scripts/setup_env.sh install` (Linux x86_64, glibc compatibile con le wheel,
   Python 3.12, driver NVIDIA ≥ 570 per torch cu129), poi `slurm/f0_checks.sbatch`. Identità in
   [`backup/ambienti.tsv`](backup/ambienti.tsv).
5. **Git.** Nuova chiave SSH registrata su GitHub; `git remote -v` resta `git@github.com:akira-n-28/vision_limite.git`.
   Il bundle non serve: `.git` è salvato intero, tag locale compreso.
6. **Dati pubblici.** Ogni file è in [`backup/catalogo_recupero.tsv`](backup/catalogo_recupero.tsv) con repository,
   commit, nome, dimensione e hash di Hugging Face; si riscaricano con `scripts/fetch_manifest.sh` dai manifest di
   `downloads/manifests/` (nel volume del codice), mai da `main`. I derivati (shard) hanno sorgente e job in
   [`backup/derivati.tsv`](backup/derivati.tsv); la riproduzione byte per byte non è verificata, salvo PixMo-Cap
   (JPEG rifatti uguali, job 30813) e i sintetici v3 (PNG ridisegnati uguali, job 30393): per questo sono nel backup.
7. **Istruzioni di progetto.** `LAGRANGE.md` e `CLAUDE.md` descrivono Lagrange: su un altro server vanno riscritte le
   parti di cluster (host, QoS, memoria del frontend, percorsi).

## 9. Aggiornare il backup prima del cambio di account

1. Rinnovare l'autorizzazione se il token è scaduto (app in prova: 7 giorni): togliere la sezione `[lvl-ist]` di
   `trasferimento/rclone/rclone.conf` con `scripts/backup_rclone.sh config -- config delete lvl-ist`, poi
   `scripts/backup_auth.sh avvia lvl-ist` e il login con l'inoltro SSH (`ssh -N -L 53682:127.0.0.1:53682 …`).
2. Nuovo inventario, poi il piano della nuova generazione con
   `--precedente trasferimento/piani/g1-20261112/piano.json`: le categorie con la stessa firma (stessi percorsi,
   tipi, dimensioni, date, permessi, link) si riusano dalla generazione precedente, solo le cambiate hanno volumi
   nuovi (`riusata_da` in `categorie.tsv`). Il codice cambia sempre.
3. Creazione, caricamento, completezza come in §4. Per ripristinare una generazione aggiornata: per ogni categoria i
   volumi della generazione indicata in `riusata_da`, altrimenti quelli nuovi.
4. Prima che l'account istituzionale venga disattivato (fine degli studi), spostare il backup o almeno il nucleo su
   un'altra destinazione: i file appartengono a quell'account.

## 10. Limiti residui

- Il backup dipende dall'account istituzionale (che può essere disattivato) e dal progetto Google Cloud del client:
  con `drive.file`, rclone vede solo i file creati da quel client; il sito di Drive li mostra sempre.
- Permessi, date al secondo e link sono conservati; proprietari e gruppi si applicano solo estraendo da root; ACL e
  attributi estesi no. Dopo un ripristino a più volumi le date delle cartelle possono cambiare.
- Le verifiche su Drive confrontano gli hash calcolati da Google, senza riscaricare tutto: riscaricati e ripristinati
  per intero sono il volume di prova e i volumi indicati nel rapporto di completezza.
- La ricerca dei segreti usa regole su file di testo fino a 16 MiB, fuori da dati, pesi e ambienti di terzi.
- L'ambiente salvato non è stato riprovato su un altro server (§5).

## 11. Stato del progetto e lavoro da riprendere

Il progetto è in **F0 preparatorio** (pipeline v1.5): nessun pilota, training o audit approvato. Stato e prossimi
interventi in [`F0_IMPLEMENTATION_PLAN.md`](F0_IMPLEMENTATION_PLAN.md) e nella sezione dei fatti di `CLAUDE.md`;
ultimo lavoro prima del backup: `synth-v4-candidato-3` verificato (job 30923), config di collaudo dei sintetici,
rapporto della scorciatoia di function/exists (commit `f10413d`). Decisioni aperte: D1-D6 del piano F0. Il registro
dei job è `runs/ledger.tsv`.
