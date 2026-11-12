#!/usr/bin/env bash
# backup_rclone.sh — esegue rclone sul frontend lagrangectl sotto scripts/guarded_run.sh (deroga dell'utente del
# 2026-11-12, solo per configurare il client Drive, caricare i volumi già preparati e una piccola prova di recupero).
#
# Protezioni di guarded_run.sh, sull'intero albero di processi: un trasferimento alla volta (lock di fetch.sh),
# avvio solo con MemAvailable >= 1 GiB, stop sotto 768 MiB o sopra 128 MiB di RSS, tetto cgroup MemoryMax=128M se
# disponibile, banda media 10 MiB/s; nessun riavvio automatico. In più qui: GOMEMLIMIT per il runtime di Go,
# configurazione (con il token), cache e file temporanei solo in trasferimento/ (escluso dal backup e da git).
#
# Uso:  scripts/backup_rclone.sh ETICHETTA -- argomenti di rclone...
# Un client OAuth proprio, se serve, si legge da trasferimento/rclone/client.env (RCLONE_DRIVE_CLIENT_ID e
# RCLONE_DRIVE_CLIENT_SECRET, file 600 scritto dall'utente): mai sulla riga di comando, nei log o in git.
set -euo pipefail
ROOT=/home/prignano/vision_limite
T=$ROOT/trasferimento
die() { echo "backup_rclone.sh: $*" >&2; exit 2; }
[[ $# -ge 3 && $2 == "--" ]] || die "uso: backup_rclone.sh ETICHETTA -- argomenti di rclone..."
LABEL=$1; shift 2
[[ $(hostname -s) == lagrangectl ]] || die "solo sul frontend lagrangectl (host attuale: $(hostname -s))"
[[ -x $ROOT/tools/rclone/rclone ]] || die "manca tools/rclone/rclone (slurm/backup_rclone.sbatch)"
umask 077
mkdir -p "$T/rclone" "$T/tmp/rclone-cache" "$T/tmp/xdg-config" "$T/tmp/xdg-cache" "$T/logs"
chmod 700 "$T" "$T/rclone"
export RCLONE_CONFIG=$T/rclone/rclone.conf
export RCLONE_CACHE_DIR=$T/tmp/rclone-cache TMPDIR=$T/tmp
export XDG_CONFIG_HOME=$T/tmp/xdg-config XDG_CACHE_HOME=$T/tmp/xdg-cache
export GOMEMLIMIT=${GOMEMLIMIT:-80MiB} GOGC=${GOGC:-50} GOMAXPROCS=${GOMAXPROCS:-2}
# Con il livello DEBUG rclone stampa le opzioni lette dall'ambiente, segreto del client compreso (2026-11-12):
# mai -vv, DEBUG o --dump, qualunque client si usi.
for a in "$@"; do
  case $a in
    -vv|-vvv|DEBUG|--log-level=DEBUG|--dump|--dump=*|--dump-headers|--dump-bodies|--dump-auth)
      die "argomento non ammesso con le credenziali del client: $a" ;;
  esac
done
if [[ -f $T/rclone/client.env && -z ${BACKUP_CLIENT_CONDIVISO:-} ]]; then  # BACKUP_CLIENT_CONDIVISO=1: solo per il vecchio token
  [[ $(stat -c %a "$T/rclone/client.env") == 600 ]] || die "trasferimento/rclone/client.env deve avere permessi 600"
  set -a
  # shellcheck source=/dev/null
  source "$T/rclone/client.env"
  set +a
fi
exec "$ROOT/scripts/guarded_run.sh" "rclone-$LABEL" -- "$ROOT/tools/rclone/rclone" "$@"
