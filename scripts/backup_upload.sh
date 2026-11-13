#!/usr/bin/env bash
# backup_upload.sh — carica su Google Drive i volumi pronti di una generazione del backup, uno alla volta e
# nell'ordine del piano; verifica ogni file con dimensione, md5 e sha256 calcolati da Drive contro quelli del job
# di creazione; libera lo staging (solo le parti create da questo incarico) dopo la verifica.
# Deroga dell'utente del 2026-11-12: sul frontend solo caricamento di file già preparati; nessun hash calcolato
# qui (--ignore-checksum: il confronto usa gli hash del job e quelli di Drive). Ogni chiamata a rclone passa da
# scripts/backup_rclone.sh, quindi da guarded_run.sh (lock, memoria, RSS, banda). Poche chiamate: una per parte,
# un elenco con gli hash per volume, gli indici dei volumi a lotti.
#
# Uso (sopravvive alla disconnessione):
#   nohup setsid scripts/backup_upload.sh REMOTE GENERAZIONE PIANO [nucleo|tutti] \
#     > /home/prignano/vision_limite/trasferimento/logs/upload-$(date +%Y%m%d-%H%M%S).log 2>&1 < /dev/null &
# Arresto gentile, alla fine del file in corso: touch /home/prignano/vision_limite/trasferimento/stato/STOP-UPLOAD
# Ripresa: rilanciare lo stesso comando; salta i volumi con <volume>.verificato.tsv e non ricarica le parti già su
# Drive con dimensione e md5 giusti. Nessuna cancellazione su Drive, nessuna sincronizzazione.
set -euo pipefail
ROOT=/home/prignano/vision_limite
T=$ROOT/trasferimento
RC=$ROOT/scripts/backup_rclone.sh
die() { echo "[$(date --iso-8601=seconds)] backup_upload.sh: $*" >&2; exit "${2:-1}"; }
log() { echo "[$(date --iso-8601=seconds)] $*"; }
[[ $# -eq 3 || $# -eq 4 ]] || die "uso: backup_upload.sh REMOTE GENERAZIONE PIANO [nucleo|tutti]" 2
REMOTE=$1; GEN=$2; PLAN=$3; MODE=${4:-tutti}
[[ $MODE == nucleo || $MODE == tutti ]] || die "quarto argomento: nucleo (backup selezionato) o tutti" 2
[[ $REMOTE =~ ^lvl-(ist|pers)$ ]] || die "remote sconosciuto: $REMOTE (lvl-ist o lvl-pers)" 2
[[ -f $PLAN/volumi.tsv && -f $PLAN/piano.json ]] || die "piano non trovato: $PLAN" 2
grep -q "\"generazione\": \"$GEN\"" "$PLAN/piano.json" || die "il piano $PLAN non è della generazione $GEN" 2
S=$T/stato/$GEN
BASE="LimiteVL-backup/$GEN"
STOP=$T/stato/STOP-UPLOAD
RLOG=$T/logs/rclone-$GEN.log
BATCH=${INDEX_BATCH:-10}   # indici dei volumi caricati ogni BATCH volumi verificati e alla fine
# Piano principale (cartella = generazione): indice/piano e indice/indice-volumi.tsv; un piano aggiuntivo della
# stessa generazione (es. g1-20261112-codice): indice/piano-codice e indice/indice-volumi-codice.tsv.
TAG=$(basename "$PLAN"); TAG=${TAG#"$GEN"}; TAG=${TAG#-}
SUF=${TAG:+-$TAG}
mkdir -p "$S/vol" "$T/logs" "$T/tmp"
exec 8> "$T/stato/.upload.lock"
flock -n 8 || die "un altro backup_upload.sh è attivo" 13
TMPL="$T/tmp/elenco-$GEN-$$.json"
trap 'rm -f -- "$TMPL" "$TMPL.err" "$T/tmp/volumi-$GEN-$$.txt"' EXIT

# Ripresa dopo uno stop per memoria (opzione A dell'utente, 2026-11-12): soglie invariate; si riparte solo dopo
# 10 minuti di fila con MemAvailable >= 1 GiB, al più N stop in tutto per la generazione: N = 5, portato a 20
# dall'utente il 2026-11-13 col file $S/max-stop-memoria; registro in $S/riprese-memoria.tsv, che vale anche fra
# un'esecuzione e l'altra. Lo stop per RSS di rclone (21) è definitivo: vorrebbe dire che il client non rispetta i
# limiti.
MEMLOG=$S/riprese-memoria.tsv
MAX_MEM_STOPS=${MAX_MEM_STOPS:-5}   # tetto di partenza; $S/max-stop-memoria (un numero) lo sostituisce, riletto a ogni stop
STABLE_SECONDS=600
mem_avail_kb() { awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo; }
wait_memory() {  # attende 10 minuti di fila sopra 1 GiB (si ferma con STOP-UPLOAD)
  local stable=0 avail
  log "attendo $((STABLE_SECONDS / 60)) minuti di fila con MemAvailable >= 1 GiB"
  while (( stable < STABLE_SECONDS )); do
    [[ -f $STOP ]] && { rm -f -- "$STOP"; die "trovato STOP-UPLOAD durante l'attesa della memoria" 0; }
    avail=$(mem_avail_kb)
    if (( avail >= 1048576 )); then stable=$((stable + 30)); else stable=0; fi
    sleep 30
  done
  log "memoria stabile sopra 1 GiB da $((STABLE_SECONDS / 60)) minuti (MemAvailable $(mem_avail_kb) kB): riprendo"
}
max_mem_stops() { if [[ -s $S/max-stop-memoria ]]; then tr -dc 0-9 < "$S/max-stop-memoria"; else echo "$MAX_MEM_STOPS"; fi; }
mem_stops() { if [[ -f $MEMLOG ]]; then grep -c $'\tstop\t' "$MEMLOG" || true; else echo 0; fi; }

# Una chiamata a rclone sotto guardia; i messaggi della guardia vanno nel log di questo script.
# Lock occupato (13): attesa e nuovo tentativo. MemAvailable sotto 1 GiB all'avvio (10, nessun trasferimento
# interrotto): attesa della memoria stabile, al più 20 volte. Stop per MemAvailable (20): registrato, attesa della
# memoria stabile e nuova chiamata, al più 5 stop in tutto. Stop per RSS (21) e ogni altro errore: ci si ferma.
rc_call() {  # etichetta argomenti...
  local label=$1; shift
  local tries_lock=0 tries_start=0 rc n
  while true; do
    set +e
    "$RC" "$label" -- "$@" 8>&-
    rc=$?
    set -e
    case $rc in
      0) return 0 ;;
      13) (( ++tries_lock <= 40 )) || die "lock dei trasferimenti occupato da troppo tempo" 13
          sleep 30 ;;
      10) (( ++tries_start <= 20 )) || die "MemAvailable sotto 1 GiB all'avvio per 20 volte: mi fermo" 10
          log "MemAvailable sotto 1 GiB all'avvio ($label)"; wait_memory ;;
      20) printf '%s\tstop\t%s\tMemAvailable %s kB dopo lo stop\n' "$(date --iso-8601=seconds)" "$label" \
            "$(mem_avail_kb)" >> "$MEMLOG"
          n=$(mem_stops)
          (( n <= $(max_mem_stops) )) || die "stop per memoria numero $n, oltre i $(max_mem_stops) ammessi: mi fermo" 20
          log "rclone fermato dalla guardia per memoria (stop $n di $(max_mem_stops) ammessi, $label)"
          wait_memory
          printf '%s\tripresa\t%s\tMemAvailable %s kB\n' "$(date --iso-8601=seconds)" "$label" "$(mem_avail_kb)" \
            >> "$MEMLOG" ;;
      21) die "rclone fermato dalla guardia per RSS sopra 128 MiB (codice 21): il client non rispetta i limiti" 21 ;;
      *) return "$rc" ;;
    esac
  done
}

# Elenco con dimensione, md5, sha256 e id dei file di una cartella su Drive (metadati di Drive, nessun download).
# Una riga JSON per file; le righe della guardia non hanno "Name". Cartella assente: elenco vuoto.
list_remote() {  # cartella_remota
  rc_call "elenco" lsjson --hash --files-only "$REMOTE:$1" --log-level ERROR > "$TMPL" 2> "$TMPL.err" \
    || { grep -q -i -E "not found|directory not found" "$TMPL.err" && : > "$TMPL" && return 0; \
         die "elenco di $1 non riuscito: $(head -c 300 "$TMPL.err")"; }
}
entry() { grep -F "\"Name\":\"$1\"," "$TMPL" | head -n 1; }   # nome
field() { grep -o "\"$1\":\"\\?[^,\"}]*" <<< "$2" | head -n 1 | sed -E "s/^\"$1\":\"?//"; }  # chiave riga

# Controlla un file contro l'elenco; scrive la riga di verifica. Codice 1 se manca o non coincide.
check_entry() {  # nome byte sha256 md5 remoto uscita
  local line got_size got_md5 got_sha
  line=$(entry "$1")
  [[ -n $line ]] || return 1
  got_size=$(field Size "$line"); got_md5=$(field md5 "$line"); got_sha=$(field sha256 "$line")
  [[ $got_size == "$2" && $got_md5 == "$4" ]] || return 1
  if [[ -n $got_sha && $got_sha != "$3" ]]; then return 1; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$5" "$got_size" "$got_md5" "${got_sha:-(non fornito da Drive)}" \
    "$(field ID "$line")" "$(date --iso-8601=seconds)" >> "$6"
}

copy_file() {  # locale remoto byte etichetta
  [[ -f $1 ]] || die "manca il file locale $1"
  [[ $(stat -c %s -- "$1") == "$3" ]] || die "dimensione locale di $1 diversa dal volume pronto"
  log "  carico $2 ($(($3 / 1024 / 1024)) MiB)"
  rc_call "$4" copyto "$1" "$REMOTE:$2" --bwlimit 10M --transfers 1 --checkers 1 --drive-chunk-size 8M \
    --buffer-size 0 --use-mmap --drive-stop-on-upload-limit --ignore-checksum --retries 3 --low-level-retries 10 \
    --contimeout 60s --timeout 5m --stats 60s --stats-one-line --log-level INFO --log-file "$RLOG" \
    || die "caricamento non riuscito: $2 (vedi $RLOG)"
}

# Indici dei volumi verificati: copia della cartella (solo i file nuovi), elenco con gli hash e confronto.
publish_indices() {
  local out="$S/indici-verificati.tsv" tmp="$S/indici-verificati.tsv.tmp" missing=0
  rc_call "indici" copy "$S/indici" "$REMOTE:$BASE/indice/volumi" --bwlimit 10M --transfers 1 --checkers 1 \
    --ignore-checksum --retries 3 --low-level-retries 10 --log-level INFO --log-file "$RLOG" \
    || die "indici dei volumi non caricati"
  list_remote "$BASE/indice/volumi"
  printf 'remoto\tbyte\tmd5_drive\tsha256_drive\tid_drive\tverificato\n' > "$tmp"
  for ver in "$S"/vol/*.verificato.tsv; do
    [[ -e $ver ]] || continue
    local p=${ver%.verificato.tsv}.parti.tsv
    while IFS=$'\t' read -r kind file size sha md5 _loc rem; do
      [[ $kind == indice ]] || continue
      check_entry "$file" "$size" "$sha" "$md5" "$BASE/$rem" "$tmp" || { missing=$((missing + 1)); log "  indice non verificato: $file"; }
    done < "$p"
  done
  mv -f "$tmp" "$out"
  (( missing == 0 )) || die "$missing indici non coincidono con Drive"
  publish_status
}

publish_status() {  # stato di ogni volume del piano: pianificato, pronto, verificato
  local out="$S/indice-volumi$SUF.tsv"
  {
    printf 'ordine\tvolume\tcategoria\tnucleo\tcompressione\tbyte_dati\tstato\tparti\tsha256_flusso\n'
    tail -n +2 "$PLAN/volumi.tsv" | while IFS=$'\t' read -r ord vid cat nuc comp _n bytes _g; do
      st=pianificato; parts=-; sha=-
      if [[ $MODE == nucleo && $nuc != 1 ]]; then st="fuori dalla selezione"
      elif [[ -f $S/vol/$vid.verificato.tsv ]]; then st=verificato
      elif [[ -f $S/vol/$vid.pronto.json ]]; then st=pronto
      fi
      [[ -f $S/vol/$vid.parti.tsv ]] && parts=$(grep -c '^volume' "$S/vol/$vid.parti.tsv" || true)
      if [[ -f $S/vol/$vid.pronto.json ]]; then
        sha=$(grep -A3 '"flusso"' "$S/vol/$vid.pronto.json" | grep -o '"sha256": "[0-9a-f]*"' | cut -d'"' -f4)
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ord" "$vid" "$cat" "$nuc" "$comp" "$bytes" "$st" "$parts" "$sha"
    done
  } > "$out.tmp"
  mv -f "$out.tmp" "$out"
  rc_call "stato" copyto "$out" "$REMOTE:$BASE/indice/indice-volumi$SUF.tsv" --ignore-checksum --log-level ERROR \
    --log-file "$RLOG" || die "stato dei volumi non caricato"
}

creator_alive() { [[ -n $(squeue -h -u "$USER" -n lvl-bk-crea -o %i 2> /dev/null) ]]; }

log "inizio: $REMOTE:$BASE dal piano $PLAN (stop per memoria finora: $(mem_stops) di $(max_mem_stops))"
if [[ -f $MEMLOG ]] && tail -n 1 "$MEMLOG" | grep -q $'\tstop\t'; then  # ultimo stop non ancora seguito da una ripresa
  (( $(mem_stops) <= $(max_mem_stops) )) || die "già $(mem_stops) stop per memoria, oltre i $(max_mem_stops) ammessi" 20
  wait_memory
  printf '%s\tripresa\t%s\tMemAvailable %s kB\n' "$(date --iso-8601=seconds)" "avvio" "$(mem_avail_kb)" >> "$MEMLOG"
fi
rc_call "piano" copy "$PLAN" "$REMOTE:$BASE/indice/piano$SUF" --include "/*.json" --include "/*.jsonl.gz" --include "/*.tsv" \
  --include "/report.txt" --include "/SHA256SUMS" --ignore-checksum --log-level ERROR --log-file "$RLOG" \
  || die "file del piano non caricati"
tail -n +2 "$PLAN/volumi.tsv" | awk -F'\t' -v m="$MODE" 'm == "tutti" || $4 == 1 { print $2 }' > "$T/tmp/volumi-$GEN-$$.txt"
since_index=0
while read -r vid; do
  [[ -f $S/vol/$vid.verificato.tsv ]] && continue
  if [[ -f $STOP ]]; then rm -f -- "$STOP"; log "trovato STOP-UPLOAD: mi fermo prima di $vid"; break; fi
  waited=0
  while [[ ! -f $S/vol/$vid.parti.tsv || ! -f $S/vol/$vid.pronto.json ]]; do
    [[ -f $S/vol/$vid.errore.json ]] && die "$vid: la creazione è fallita ($S/vol/$vid.errore.json)"
    if ! creator_alive; then
      (( waited < 3 )) || die "$vid non è pronto e nessun job lvl-bk-crea è attivo o in coda"
    fi
    (( waited++ == 0 )) && log "attendo che $vid sia pronto"
    sleep 60
  done
  cat=$(awk -F'\t' -v v="$vid" '$2 == v { print $3 }' "$PLAN/volumi.tsv")
  log "volume $vid"
  : > "$TMPL"
  [[ -f $S/vol/$vid.tentativo ]] && list_remote "$BASE/volumi/$cat"   # ripresa: le parti già su Drive
  date --iso-8601=seconds > "$S/vol/$vid.tentativo"
  while IFS=$'\t' read -r kind file size sha md5 loc rem; do
    [[ $kind == volume ]] || continue
    if [[ -n $(entry "$file") ]] && check_entry "$file" "$size" "$sha" "$md5" "$BASE/$rem" /dev/null; then
      log "  già su Drive: $file"
      continue
    fi
    copy_file "$loc" "$BASE/$rem" "$size" "$vid"
  done < "$S/vol/$vid.parti.tsv"
  list_remote "$BASE/volumi/$cat"
  ver="$S/vol/$vid.verificato.tsv"
  printf 'remoto\tbyte\tmd5_drive\tsha256_drive\tid_drive\tverificato\n' > "$ver.tmp"
  while IFS=$'\t' read -r kind file size sha md5 loc rem; do
    [[ $kind == volume ]] || continue
    check_entry "$file" "$size" "$sha" "$md5" "$BASE/$rem" "$ver.tmp" \
      || die "$vid: $file su Drive non coincide (dimensione, md5 o sha256) o manca"
  done < "$S/vol/$vid.parti.tsv"
  mv -f "$ver.tmp" "$ver"
  printf '%s\tverificato\t%s\t%s\n' "$(date --iso-8601=seconds)" "$vid" "$REMOTE:$BASE" >> "$S/registro.tsv"
  while IFS=$'\t' read -r kind _file _size _sha _md5 loc _rem; do  # solo parti di volume dentro lo staging
    [[ $kind == volume && $loc == "$T/staging/$GEN/"* && -f $loc ]] && rm -f -- "$loc"
  done < "$S/vol/$vid.parti.tsv"
  rm -f -- "$S/vol/$vid.tentativo"
  log "  $vid verificato su Drive; staging liberato"
  if (( ++since_index >= BATCH )); then publish_indices; since_index=0; fi
done < "$T/tmp/volumi-$GEN-$$.txt"
publish_indices
log "fine: $(grep -c $'\tverificato\t' "$S/indice-volumi$SUF.tsv" || true) volumi verificati su $(wc -l < "$T/tmp/volumi-$GEN-$$.txt") da caricare ($MODE)"
