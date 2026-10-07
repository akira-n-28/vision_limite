#!/usr/bin/env bash
# guarded_run.sh — esegue sul frontend un comando che usa la rete (es. `uv lock`, `git clone`)
# con le stesse protezioni di fetch.sh, applicate a TUTTO l'albero di processi del comando:
#   - parte solo con MemAvailable >= 1 GiB; condivide il lock di fetch.sh (un trasferimento alla volta)
#   - ogni 2 s: stop se MemAvailable < 768 MiB o se la RSS totale del comando > 128 MiB
#   - banda media limitata a 10 MiB/s: se il comando legge più in fretta viene sospeso
#     (SIGSTOP) per il tempo necessario e poi ripreso (SIGCONT)
#   - nessun riavvio automatico; esito e motivo nel download-log.tsv
#
# Uso:  scripts/guarded_run.sh ETICHETTA -- comando [argomenti...]
# Esempio:
#   UV_CONCURRENT_DOWNLOADS=1 scripts/guarded_run.sh uv-lock -- \
#       /home/prignano/vision_limite/tools/uv lock --directory env --no-build
#
# Non usarlo per lavoro di calcolo: compilazioni, estrazioni, installazioni e import
# numerici restano nei job SLURM. Serve solo per trasferimenti piccoli e controllati.
set -euo pipefail

readonly MIN_START_KB=$((1024 * 1024))   # 1 GiB
readonly STOP_AVAIL_KB=$((768 * 1024))   # 768 MiB
readonly MAX_RSS_KB=$((128 * 1024))      # 128 MiB
readonly RATE_BPS=$((10 * 1024 * 1024))  # 10 MiB/s
readonly POLL_SECONDS=2

STATE_DIR="${FETCH_STATE_DIR:-/home/prignano/vision_limite/data}"
LOCK_FILE="${STATE_DIR}/.fetch.lock"
LOG_FILE="${STATE_DIR}/download-log.tsv"
MEMINFO="${FETCH_MEMINFO:-/proc/meminfo}"   # solo per i test

die() { local code=$1; shift; echo "guarded_run.sh: $*" >&2; exit "$code"; }
[[ $# -ge 3 && $2 == "--" ]] || die 2 "uso: guarded_run.sh ETICHETTA -- comando [argomenti...]"
LABEL=$1; shift 2
CMD_STR="$*"

mkdir -p -- "$STATE_DIR"
[[ -f $LOG_FILE ]] || printf 'time\tstatus\treason\tbytes\tetag\turl\tdest\n' > "$LOG_FILE"
mem_kb() { awk -v k="$1:" '$1 == k { print $2; found = 1 } END { if (!found) print 0 }' "$MEMINFO"; }
log_row() { printf '%s\t%s\t%s\t%s\t-\tguarded:%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" "$LABEL" "${CMD_STR:0:200}" >> "$LOG_FILE"; }

exec 9> "$LOCK_FILE"
flock -n 9 || die 13 "un altro trasferimento è attivo (lock $LOCK_FILE)"

avail_kb=$(mem_kb MemAvailable)
echo "MemTotal=$(mem_kb MemTotal) kB MemAvailable=${avail_kb} kB"
(( avail_kb >= MIN_START_KB )) || { log_row refused "MemAvailable ${avail_kb}kB < 1GiB" 0; die 10 "MemAvailable sotto 1 GiB: non parto"; }

# Tetto rigido facoltativo: un cgroup utente con MemoryMax=128M (la guida lo ammette come
# protezione aggiuntiva). Se il comando lo supera, il kernel termina solo il comando.
WRAP=()
if [[ ${GUARD_CGROUP:-auto} != off ]] && command -v systemd-run > /dev/null \
   && systemd-run --user --scope --quiet -p MemoryMax=128M -p MemorySwapMax=0 true 2> /dev/null; then
  WRAP=(systemd-run --user --scope --quiet -p MemoryMax=128M -p MemorySwapMax=0)
  echo "tetto cgroup attivo: MemoryMax=128M"
else
  echo "systemd-run --user non disponibile: solo controllo RSS ogni ${POLL_SECONDS} s"
fi

# Il comando diventa leader di una nuova sessione: SID = PGID = PID.
setsid "${WRAP[@]}" "$@" 9>&- &
PID=$!

session_pids() { ps -o pid= --sid "$PID" 2> /dev/null | tr -d ' '; }
tree_rss_kb() { ps -o rss= --sid "$PID" 2> /dev/null | awk '{ s += $1 } END { print s + 0 }'; }
tree_read_bytes() {  # rchar: byte letti dai processi vivi (rete + file), stima prudente
  local p total=0 r
  for p in $(session_pids); do
    r=$(awk '/^rchar:/ { print $2 }' "/proc/$p/io" 2> /dev/null || true)
    total=$((total + ${r:-0}))
  done
  echo "$total"
}
STOPPED=0
stop_tree() {  # motivo stato codice
  (( STOPPED )) && kill -CONT -- "-$PID" 2> /dev/null || true
  kill -TERM -- "-$PID" 2> /dev/null || true
  sleep 2
  kill -KILL -- "-$PID" 2> /dev/null || true
  wait "$PID" 2> /dev/null || true
  log_row "$2" "$1" "$(( peak_bytes ))"
  die "$3" "fermato: $1"
}
trap 'stop_tree "interrotto da segnale" interrupted 22' INT TERM HUP

start=$(date +%s)
peak_bytes=0   # massimo osservato di byte letti (i processi già usciti non contano più)
max_rss=0
while kill -0 "$PID" 2> /dev/null; do
  avail_kb=$(mem_kb MemAvailable)
  rss=$(tree_rss_kb)
  (( rss > max_rss )) && max_rss=$rss
  (( avail_kb < STOP_AVAIL_KB )) && stop_tree "MemAvailable ${avail_kb}kB < 768MiB" stopped_mem 20
  (( rss > MAX_RSS_KB )) && stop_tree "RSS del comando ${rss}kB > 128MiB" stopped_rss 21
  bytes=$(tree_read_bytes)
  (( bytes > peak_bytes )) && peak_bytes=$bytes
  elapsed=$(( $(date +%s) - start + 1 ))
  excess=$(( peak_bytes - RATE_BPS * elapsed ))
  if (( excess > 0 )); then
    pause=$(( excess / RATE_BPS + 1 ))
    (( pause > 30 )) && pause=30
    kill -STOP -- "-$PID" 2> /dev/null || true; STOPPED=1
    sleep "$pause"
    kill -CONT -- "-$PID" 2> /dev/null || true; STOPPED=0
    continue
  fi
  sleep "$POLL_SECONDS"
done
set +e; wait "$PID"; rc=$?; set -e
trap - INT TERM HUP
secs=$(( $(date +%s) - start ))
echo "fine '$LABEL': codice $rc, ${secs}s, RSS massima ${max_rss} kB, letti circa $((peak_bytes / 1024 / 1024)) MiB"
if (( rc == 0 )); then log_row ok "rss_max=${max_rss}kB" "$peak_bytes"; else log_row failed "exit $rc" "$peak_bytes"; fi
exit "$rc"
