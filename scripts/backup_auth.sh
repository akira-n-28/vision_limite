#!/usr/bin/env bash
# backup_auth.sh — avvia l'autorizzazione OAuth di rclone per Google Drive con l'ambito drive.file (solo i file
# creati da rclone) e stampa SOLO l'indirizzo locale da aprire nel browser, attraverso un inoltro SSH:
#   sul computer Linux:  ssh -N -L 53682:127.0.0.1:53682 <utente>@lagrangectl.mat.uniroma1.it
#   nel browser di quel computer: l'indirizzo http://127.0.0.1:53682/auth?state=... stampato qui.
# rclone resta in attesa sotto guarded_run.sh; il token finisce in trasferimento/rclone/rclone.conf (600), mai a
# schermo, nei log del progetto o in git. Password e MFA si inseriscono solo nel browser.
#
# Uso:  scripts/backup_auth.sh avvia lvl-ist|lvl-pers     avvia e stampa l'indirizzo
#       scripts/backup_auth.sh stato lvl-ist|lvl-pers     esito, senza mostrare il token
set -euo pipefail
ROOT=/home/prignano/vision_limite
T=$ROOT/trasferimento
CONF=$T/rclone/rclone.conf
die() { echo "backup_auth.sh: $*" >&2; exit 2; }
[[ $# -eq 2 && $2 =~ ^lvl-(ist|pers)$ ]] || die "uso: backup_auth.sh avvia|stato lvl-ist|lvl-pers"
CMD=$1; R=$2
LOG=$T/rclone/auth-$R.log
umask 077
mkdir -p "$T/rclone"

configured() { [[ -f $CONF ]] && grep -q "^\[$R\]$" "$CONF" && awk -v r="[$R]" '$0 == r { s = 1; next } /^\[/ { s = 0 }
  s && /^token = / { found = 1 } END { exit !found }' "$CONF"; }

case $CMD in
  avvia)
    configured && die "$R è già configurato: per rifarlo togliere a mano la sezione [$R] da $CONF"
    if pgrep -u "$USER" -f "rclone config create $R " > /dev/null; then die "un'autorizzazione di $R è già in corso"; fi
    nohup setsid "$ROOT/scripts/backup_rclone.sh" "auth-$R" -- config create "$R" drive scope=drive.file \
      > "$LOG" 2>&1 < /dev/null &
    for _ in $(seq 1 30); do
      url=$(grep -o 'http://127\.0\.0\.1:53682/auth?state=[A-Za-z0-9_-]*' "$LOG" | head -n 1 || true)
      [[ -n $url ]] && break
      sleep 1
    done
    grep -E '^(MemTotal|tetto cgroup|systemd-run)' "$LOG" || true
    [[ -n ${url:-} ]] || { grep -i -E 'error|fail|refus|fermato' "$LOG" | head -5; die "indirizzo non trovato"; }
    echo "rclone attende l'autorizzazione di $R su 127.0.0.1:53682 (frontend)."
    echo "Indirizzo da aprire nel browser del computer Linux, con l'inoltro SSH attivo:"
    echo "  $url"
    ;;
  stato)
    if configured; then
      echo "$R: configurato (token presente in $CONF, permessi $(stat -c %a "$CONF"))"
      [[ -f $LOG ]] && rm -f -- "$LOG" && echo "log dell'autorizzazione rimosso"
    else
      echo "$R: non ancora configurato"
      if [[ -f $LOG ]]; then
        grep -i -o -E 'Got code|Waiting for code|error[^,]{0,120}|failed[^,]{0,120}|fermato[^,]{0,120}' "$LOG" | sort | uniq -c
      fi
      pgrep -u "$USER" -f "rclone config create $R " > /dev/null && echo "rclone è ancora in attesa" || echo "rclone non è attivo"
    fi
    ;;
  *) die "comando sconosciuto: $CMD" ;;
esac
