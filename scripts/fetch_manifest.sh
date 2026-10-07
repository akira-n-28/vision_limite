#!/usr/bin/env bash
# fetch_manifest.sh — scarica in sequenza i file di un manifest TSV con fetch.sh.
#
# Formato (tab-separated, righe che iniziano con # ignorate):
#   url  dest  size_bytes  hash  label
# `hash` è "sha256:<hex>" (file LFS), "gitsha1:<hex>" (file piccoli Hugging Face) o "-".
#
# Un file alla volta, nessun worker in parallelo. Al primo errore si ferma
# (niente riavvii automatici sotto pressione di memoria). Per fermarlo con calma
# alla fine del file in corso: touch /home/prignano/vision_limite/data/.fetch.stop
#
# Avvio consigliato sul frontend, sopravvive alla disconnessione SSH:
#   nohup setsid scripts/fetch_manifest.sh downloads/manifests/priorita1.tsv \
#     > /home/prignano/vision_limite/data/logs/fetch-$(date +%Y%m%d-%H%M%S).log 2>&1 &
set -euo pipefail

[[ $# -eq 1 && -r $1 ]] || { echo "uso: fetch_manifest.sh MANIFEST.tsv" >&2; exit 2; }
MANIFEST=$1
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STATE_DIR="${FETCH_STATE_DIR:-/home/prignano/vision_limite/data}"
STOP_FILE="${STATE_DIR}/.fetch.stop"
PID_FILE="${STATE_DIR}/.fetch_manifest.pid"
mkdir -p -- "$STATE_DIR"

if [[ -f $PID_FILE ]] && kill -0 "$(cat "$PID_FILE")" 2> /dev/null; then
  echo "fetch_manifest già in esecuzione (PID $(cat "$PID_FILE"))" >&2
  exit 13
fi
echo $$ > "$PID_FILE"
trap 'rm -f -- "$PID_FILE"' EXIT

total=0; done_n=0; skipped=0; bytes=0
while IFS=$'\t' read -r url dest size _hash label; do
  [[ -z ${url// } || $url == \#* ]] && continue
  total=$((total + 1))
  if [[ -f $STOP_FILE ]]; then
    echo "trovato $STOP_FILE: mi fermo prima di $label"
    rm -f -- "$STOP_FILE"
    break
  fi
  [[ $size == "-" ]] && size=""
  if [[ -f $dest && ( -z $size || $(stat -c %s -- "$dest") == "$size" ) ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  echo "[$(date --iso-8601=seconds)] $label (${size:-?} byte)"
  if ! "$HERE/fetch.sh" "$url" "$dest" ${size:+"$size"} < /dev/null; then
    echo "errore su $label: mi fermo. Vedi download-log.tsv; rilanciare a mano dopo aver capito la causa." >&2
    exit 1
  fi
  done_n=$((done_n + 1))
  bytes=$((bytes + ${size:-0}))
done < "$MANIFEST"
echo "fine: $done_n scaricati ($((bytes / 1024 / 1024)) MiB), $skipped già presenti, $total righe"
