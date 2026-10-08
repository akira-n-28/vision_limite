#!/usr/bin/env bash
# fetch.sh — download monitorato di UN file sul frontend lagrangectl.
#
# Implementa la "User-authorized download clarification — 2026-10-06" di LAGRANGE.md:
#   - un solo trasferimento curl alla volta (lock condiviso con hf_manifest.py)
#   - al massimo 10 MiB/s
#   - non parte con MemAvailable < 1 GiB
#   - controlla MemAvailable e RSS di curl ogni 2 secondi; si ferma con
#     MemAvailable < 768 MiB o RSS > 128 MiB, conserva il .part e registra il motivo
#   - nessun riavvio automatico: in caso di stop esce con errore e decide l'utente
#   - promozione atomica (rename) solo dopo verifica della dimensione attesa
#   - nessun hash completo qui: la verifica SHA-256 si fa in un job SLURM CPU
#
# Uso:   scripts/fetch.sh URL DEST [EXPECTED_SIZE_BYTES]
# Token: se HF_TOKEN_FILE punta a un file (permessi 600) il token viene passato a curl
#        via stdin, quindi non compare mai nella riga di comando né nei log.
#
# Codici di uscita: 0 ok/già presente · 2 uso · 10 memoria insufficiente all'avvio
#   11 spazio/inode insufficienti · 12 resume impossibile · 13 lock occupato
#   20 stop per MemAvailable · 21 stop per RSS · 22 interrotto · 30 errore curl
#   31 dimensione errata · 32 destinazione non valida
set -euo pipefail

# Soglie della guida: costanti, NON abbassarle.
readonly MIN_START_KB=$((1024 * 1024))   # 1 GiB
readonly STOP_AVAIL_KB=$((768 * 1024))   # 768 MiB
readonly MAX_RSS_KB=$((128 * 1024))      # 128 MiB
readonly RATE="10M"                      # curl usa multipli di 1024: 10 MiB/s
readonly POLL_SECONDS=2
readonly HEADROOM_BYTES=$((1024 * 1024 * 1024))

# Configurazione (su Lagrange lasciare i default).
ALLOWED_ROOT="${FETCH_ALLOWED_ROOT:-/home/prignano/vision_limite/}"
STATE_DIR="${FETCH_STATE_DIR:-/home/prignano/vision_limite/data}"
LOCK_FILE="${STATE_DIR}/.fetch.lock"
LOG_FILE="${STATE_DIR}/download-log.tsv"
MEMINFO="${FETCH_MEMINFO:-/proc/meminfo}"   # solo per i test: non cambiare su Lagrange

die() { local code=$1; shift; echo "fetch.sh: $*" >&2; exit "$code"; }

[[ $# -ge 2 && $# -le 3 ]] || die 2 "uso: fetch.sh URL DEST [EXPECTED_SIZE_BYTES]"
URL=$1
DEST=$2
EXPECTED=${3:-}

[[ $URL == https://* ]] || die 2 "solo URL https"
[[ $DEST == /* ]] || die 32 "DEST deve essere un percorso assoluto"
[[ -z $EXPECTED || $EXPECTED =~ ^[0-9]+$ ]] || die 2 "EXPECTED_SIZE_BYTES non numerico"

case "$DEST" in
  /dev/shm/*|/run/*|/proc/*) die 32 "destinazione vietata: $DEST" ;;
esac
DEST_DIR=$(dirname -- "$DEST")
mkdir -p -- "$STATE_DIR"
[[ "$(realpath -m -- "$DEST_DIR")/" == "${ALLOWED_ROOT%/}/"* ]] \
  || die 32 "DEST fuori da ${ALLOWED_ROOT}: $DEST"
mkdir -p -- "$DEST_DIR"
PART="${DEST}.part"

mem_kb() { awk -v k="$1:" '$1 == k { print $2; found = 1 } END { if (!found) print 0 }' "$MEMINFO"; }
now() { date --iso-8601=seconds; }
log_row() {  # status reason bytes
  local etag=${X_ETAG:--}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(now)" "$1" "$2" "$3" "$etag" "$URL" "$DEST" >> "$LOG_FILE"
}
[[ -f $LOG_FILE ]] || printf 'time\tstatus\treason\tbytes\tetag\turl\tdest\n' > "$LOG_FILE"

# Già presente?
if [[ -f $DEST ]]; then
  have=$(stat -c %s -- "$DEST")
  if [[ -z $EXPECTED || $have == "$EXPECTED" ]]; then
    echo "già presente: $DEST ($have byte)"
    exit 0
  fi
  die 31 "esiste già con dimensione $have invece di $EXPECTED: $DEST (non lo tocco)"
fi

# Un solo trasferimento alla volta per questo progetto.
exec 9> "$LOCK_FILE"
flock -n 9 || die 13 "un altro trasferimento è attivo (lock $LOCK_FILE)"

# Memoria all'avvio.
total_kb=$(mem_kb MemTotal)
avail_kb=$(mem_kb MemAvailable)
echo "MemTotal=${total_kb} kB MemAvailable=${avail_kb} kB"
if (( avail_kb < MIN_START_KB )); then
  log_row refused "MemAvailable ${avail_kb}kB < 1GiB" 0
  die 10 "MemAvailable ${avail_kb} kB sotto 1 GiB: non parto"
fi

# Il token Hugging Face va SOLO a huggingface.co (o a HF_ENDPOINT), mai ad altri host.
URL_HOST=$(printf '%s' "$URL" | sed -E 's#^https://([^/:]+).*#\1#')
HF_HOST=$(printf '%s' "${HF_ENDPOINT:-https://huggingface.co}" | sed -E 's#^https?://([^/:]+).*#\1#')
SEND_TOKEN=0
if [[ -n ${HF_TOKEN_FILE:-} && -r ${HF_TOKEN_FILE} && $URL_HOST == "$HF_HOST" ]]; then
  SEND_TOKEN=1
fi
auth_header() {
  if (( SEND_TOKEN )); then
    printf 'Authorization: Bearer %s\n' "$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
  fi
}

# Richiesta HEAD: dimensione finale, supporto ai range, etag Hugging Face.
headers=$(auth_header | curl -sS -I -L --max-time 60 --connect-timeout 30 -H @- -- "$URL") \
  || die 30 "HEAD fallita per $URL"
headers=$(printf '%s' "$headers" | tr -d '\r')
hdr_last() { printf '%s\n' "$headers" | awk -v k="$(tr '[:upper:]' '[:lower:]' <<< "$1"):" 'tolower($1) == k { v = $2 } END { print v }'; }
hdr_first() { printf '%s\n' "$headers" | awk -v k="$(tr '[:upper:]' '[:lower:]' <<< "$1"):" 'tolower($1) == k && !done { print $2; done = 1 }'; }
X_ETAG=$(hdr_first x-linked-etag | tr -d '"')
linked_size=$(hdr_first x-linked-size)
final_len=$(hdr_last content-length)
ranges=$(hdr_last accept-ranges)
if [[ -z $EXPECTED ]]; then
  EXPECTED=${linked_size:-$final_len}
elif [[ -n $linked_size && $linked_size != "$EXPECTED" ]]; then
  die 31 "il server dichiara $linked_size byte, il manifest $EXPECTED"
fi
[[ -n $EXPECTED ]] || echo "attenzione: dimensione attesa sconosciuta, mi baso sull'esito di curl"

part_size=0
[[ -f $PART ]] && part_size=$(stat -c %s -- "$PART")
if [[ -n $EXPECTED ]] && (( part_size > EXPECTED )); then
  die 31 ".part più grande dell'atteso ($part_size > $EXPECTED): controllarlo a mano"
fi
if (( part_size > 0 )) && [[ ${ranges,,} != *bytes* ]]; then
  log_row refused "no range support, .part present" "$part_size"
  die 12 "il server non accetta Range e c'è già un .part: rivederlo a mano prima di riscaricare"
fi

# Spazio e inode sulla destinazione.
avail_bytes=$(df -B1 --output=avail -- "$DEST_DIR" | tail -n 1 | tr -d ' ')
inodes=$(df --output=iavail -- "$DEST_DIR" | tail -n 1 | tr -d ' ')
need=$(( ${EXPECTED:-0} - part_size + HEADROOM_BYTES ))
if (( avail_bytes < need )); then
  die 11 "spazio insufficiente: liberi $avail_bytes, servono $need (file + 1 GiB di margine)"
fi
if [[ $inodes =~ ^[0-9]+$ ]] && (( inodes < 100 )); then
  die 11 "inode liberi insufficienti ($inodes)"
fi

if [[ -n $EXPECTED ]] && (( part_size == EXPECTED )); then
  echo ".part già completo, salto il trasferimento"
else
  echo "scarico $URL -> $PART (già presenti $part_size byte, attesi ${EXPECTED:-?}, limite $RATE/s)"
  start=$(date +%s)
  # HTTP/1.1: con HTTP/2 il CDN di Hugging Face ha annullato lo stream di LLaVA-Pretrain
  # images.zip due volte appena sotto i 128 MiB (curl 92, 2026-10-08).
  auth_header | curl --http1.1 --fail --location --silent --show-error \
      --limit-rate "$RATE" --continue-at - --connect-timeout 30 \
      --speed-time 300 --speed-limit 1024 \
      -H @- -o "$PART" -- "$URL" 9>&- &
  CURL_PID=$!

  stop_curl() {  # reason status exitcode
    kill -TERM "$CURL_PID" 2> /dev/null || true
    wait "$CURL_PID" 2> /dev/null || true
    local got=0; [[ -f $PART ]] && got=$(stat -c %s -- "$PART")
    log_row "$2" "$1" "$got"
    die "$3" "fermato: $1 (conservati $got byte in $PART)"
  }
  trap 'stop_curl "interrotto da segnale" interrupted 22' INT TERM HUP

  while kill -0 "$CURL_PID" 2> /dev/null; do
    avail_kb=$(mem_kb MemAvailable)
    rss_kb=$(awk '/^VmRSS:/ { print $2 }' "/proc/$CURL_PID/status" 2> /dev/null || echo 0)
    rss_kb=${rss_kb:-0}
    if (( avail_kb < STOP_AVAIL_KB )); then
      stop_curl "MemAvailable ${avail_kb}kB < 768MiB" stopped_mem 20
    fi
    if (( rss_kb > MAX_RSS_KB )); then
      stop_curl "curl RSS ${rss_kb}kB > 128MiB" stopped_rss 21
    fi
    sleep "$POLL_SECONDS"
  done
  set +e; wait "$CURL_PID"; rc=$?; set -e
  trap - INT TERM HUP
  got=$(stat -c %s -- "$PART" 2> /dev/null || echo 0)
  if (( rc != 0 )); then
    log_row failed "curl exit $rc" "$got"
    die 30 "curl è uscito con codice $rc (conservati $got byte in $PART)"
  fi
  echo "trasferiti in $(( $(date +%s) - start )) s"
fi

got=$(stat -c %s -- "$PART")
if [[ -n $EXPECTED && $got != "$EXPECTED" ]]; then
  log_row failed "size $got != $EXPECTED" "$got"
  die 31 "dimensione $got diversa dall'attesa $EXPECTED: .part conservato"
fi
mv -n -- "$PART" "$DEST"
[[ -f $DEST && ! -e $PART ]] || die 31 "promozione non riuscita per $DEST"
log_row ok "-" "$got"
echo "ok: $DEST ($got byte)"
