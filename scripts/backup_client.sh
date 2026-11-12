#!/usr/bin/env bash
# backup_client.sh — da eseguire DALL'UTENTE nel proprio terminale SSH su lagrangectl (non in Claude Code):
# chiede ID e segreto del client OAuth "App desktop" creato nella Google Cloud Console e li scrive in
# trasferimento/rclone/client.env (permessi 600), che scripts/backup_rclone.sh legge come variabili d'ambiente.
# I valori non compaiono a schermo (il segreto), nella cronologia della shell, sulla riga di comando o in git:
# read e printf sono comandi interni di bash.
set -euo pipefail
F=/home/prignano/vision_limite/trasferimento/rclone/client.env
umask 077
mkdir -p "$(dirname "$F")"
if [[ -f $F ]]; then
  read -r -p "client.env esiste già: sostituirlo? [s/N] " ok
  [[ $ok == s || $ok == S ]] || { echo "lasciato invariato"; exit 0; }
fi
read -r -p "ID client (finisce con .apps.googleusercontent.com): " cid
[[ $cid == *.apps.googleusercontent.com ]] || { echo "l'ID non ha la forma attesa" >&2; exit 1; }
read -r -s -p "Segreto client (non viene mostrato): " sec
echo
[[ -n $sec ]] || { echo "segreto vuoto" >&2; exit 1; }
printf 'RCLONE_DRIVE_CLIENT_ID=%s\nRCLONE_DRIVE_CLIENT_SECRET=%s\n' "$cid" "$sec" > "$F.tmp"
chmod 600 "$F.tmp"
mv -f "$F.tmp" "$F"
unset cid sec
echo "scritto $F (permessi $(stat -c %a "$F")); torna in chat e scrivi: client pronto"
