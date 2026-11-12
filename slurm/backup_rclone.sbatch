#!/usr/bin/env bash
# Verifica la provenienza del binario di rclone scaricato con fetch.sh e lo installa in tools/rclone/rclone.
#   1. impronta della chiave di firma = quella pubblicata in https://rclone.org/release_signing/ (scaricata accanto);
#   2. firma PGP di SHA256SUMS con quella chiave (gpgv, portachiavi temporaneo dentro il progetto, nessun demone);
#   3. sha256 dello zip uguale alla riga del testo firmato; 4. estrazione e `rclone version`.
# Uso: sbatch --parsable slurm/backup_rclone.sbatch v1.76.0
#SBATCH --job-name=lvl-bk-rclone
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/prignano/vision_limite/trasferimento/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/trasferimento/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
V=${1:?versione, es. v1.76.0}
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu

FPR=FBF737ECE9F8AB18604BD2AC93935E02FF3B54FA
D="$TOOLS_DIR/rclone/$V"
ZIP="rclone-$V-linux-amd64.zip"
report="$D/VERIFICA-$SLURM_JOB_ID.txt"
fail() { echo "ERRORE: $*" | tee -a "$report" >&2; ledger_note failed "rclone $V: $*"; exit 1; }
for f in "$ZIP" SHA256SUMS rclone-signing-key.asc release_signing.html; do test -f "$D/$f" || fail "manca $D/$f"; done
: > "$report"
export GNUPGHOME="$PROJECT_ROOT/trasferimento/tmp/gnupg-$SLURM_JOB_ID"
mkdir -p "$GNUPGHOME"; chmod 700 "$GNUPGHOME"
trap 'rm -rf -- "$GNUPGHOME"' EXIT

grep -q "$FPR" "$D/release_signing.html" || fail "l'impronta $FPR non compare nella pagina release_signing"
got_fpr=$(gpg --batch --with-colons --show-keys "$D/rclone-signing-key.asc" | awk -F: '$1 == "fpr" { print $10; exit }')
[[ $got_fpr == "$FPR" ]] || fail "impronta della chiave scaricata $got_fpr diversa da $FPR"
echo "chiave: $FPR (uguale alla pagina release_signing)" >> "$report"
gpg --batch --dearmor --output "$GNUPGHOME/rclone.gpg" "$D/rclone-signing-key.asc"
status=$(gpgv --keyring "$GNUPGHOME/rclone.gpg" --status-fd 1 --output "$GNUPGHOME/signed.txt" "$D/SHA256SUMS" 2> /dev/null) \
  || fail "firma di SHA256SUMS non valida"
grep -q "^\[GNUPG:\] VALIDSIG $FPR " <<< "$status" || fail "firma valida ma non della chiave $FPR"
echo "firma: $(grep '^\[GNUPG:\] VALIDSIG' <<< "$status" | cut -d' ' -f3-5)" >> "$report"
want=$(awk -v z="$ZIP" '$2 == z { print $1 }' "$GNUPGHOME/signed.txt")
got=$(sha256sum "$D/$ZIP" | awk '{ print $1 }')
[[ -n $want && $want == "$got" ]] || fail "sha256 dello zip $got diverso dal testo firmato ($want)"
echo "zip: $ZIP sha256 $got (uguale al testo firmato)" >> "$report"

x="$D/extract-$SLURM_JOB_ID"
mkdir -p "$x"
unzip -q "$D/$ZIP" -d "$x"
bin="$x/rclone-$V-linux-amd64/rclone"
test -f "$bin" || fail "binario non trovato nello zip"
install -m 755 "$bin" "$TOOLS_DIR/rclone/rclone.new"
mv -f "$TOOLS_DIR/rclone/rclone.new" "$TOOLS_DIR/rclone/rclone"
echo "binario: $TOOLS_DIR/rclone/rclone sha256 $(sha256sum "$TOOLS_DIR/rclone/rclone" | awk '{ print $1 }')" >> "$report"
export RCLONE_CONFIG=/dev/null
"$TOOLS_DIR/rclone/rclone" version | tee -a "$report"
"$TOOLS_DIR/rclone/rclone" help backend drive > "$D/help-backend-drive.txt"
cat "$report"
ledger_note ok "rclone $V verificato (chiave $FPR, firma di SHA256SUMS, sha256 dello zip) e installato in $TOOLS_DIR/rclone/rclone"
