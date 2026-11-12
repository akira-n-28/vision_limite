#!/bin/sh
# ripristina_volumi.sh — ripristino dei volumi del backup con i soli strumenti standard, su macOS o su Linux
# (NON sul frontend di Lagrange: su un cluster l'estrazione va in un job). Per ogni volume, nell'ordine dell'elenco:
#   1. sha256 di ogni parte uguale al catalogo (sha256sum o shasum -a 256);
#   2. cat parti | tar -xp[z]f - -C RADICE (permessi, date, link e hard link come nell'archivio);
#   3. controllo di ogni file con il suo <volume>.sha256 (sha256sum -c o shasum -a 256 -c).
# Ripresa: i volumi riusciti sono annotati in RADICE.ripristino-fatti e si saltano. Nessun file viene cancellato,
# tranne le parti scaricate se si passa --elimina-parti (dopo il controllo riuscito, per risparmiare spazio).
#
# Uso: sh ripristina_volumi.sh VOLUMI_TSV PARTI_TSV CARTELLA_PARTI CARTELLA_INDICI RADICE [nucleo|tutti] [--elimina-parti]
#   VOLUMI_TSV       docs/backup/<generazione>-volumi.tsv (ordine, volume, categoria, nucleo, compressione, ...)
#   PARTI_TSV        docs/backup/<generazione>-parti.tsv (volume, file, byte, sha256, md5, remoto, id_drive)
#   CARTELLA_PARTI   dove stanno le parti scaricate: CARTELLA/<categoria>/<file> (come su Drive) o CARTELLA/<file>
#   CARTELLA_INDICI  i file <volume>.sha256 scaricati da indice/volumi/ su Drive
#   RADICE           cartella di destinazione (nuova o con un ripristino parziale)
set -eu
die() { echo "ripristina_volumi.sh: $*" >&2; exit 1; }
[ $# -ge 5 ] || die "uso: sh ripristina_volumi.sh VOLUMI_TSV PARTI_TSV CARTELLA_PARTI CARTELLA_INDICI RADICE [nucleo|tutti] [--elimina-parti]"
VOL=$1; PARTS=$2; SRC=$3; IDX=$4; ROOT=$5; WHICH=${6:-nucleo}; DEL=${7:-}
[ -f "$VOL" ] && [ -f "$PARTS" ] && [ -d "$SRC" ] && [ -d "$IDX" ] || die "elenchi o cartelle non trovati"
case $WHICH in nucleo|tutti) ;; *) die "sesto argomento: nucleo o tutti" ;; esac
if command -v sha256sum > /dev/null 2>&1 && [ -z "${FORZA_SHASUM:-}" ]; then HASH="sha256sum"; CHECK="sha256sum -c"
else HASH="shasum -a 256"; CHECK="shasum -a 256 -c"; fi
mkdir -p "$ROOT"
DONE="$ROOT.ripristino-fatti"
touch "$DONE"
TAB=$(printf '\t')
n=0
tail -n +2 "$VOL" | while IFS="$TAB" read -r ord vid cat nuc comp _rest; do
  [ "$WHICH" = tutti ] || [ "$nuc" = 1 ] || continue
  if grep -qx "$vid" "$DONE"; then continue; fi
  echo "== $vid ($cat)"
  list=""
  for row in $(awk -F'\t' -v v="$vid" '$1 == v { print $2 "|" $4 }' "$PARTS"); do
    file=${row%%|*}; want=${row#*|}
    if [ -f "$SRC/$cat/$file" ]; then p="$SRC/$cat/$file"; elif [ -f "$SRC/$file" ]; then p="$SRC/$file"
    else die "manca la parte $file di $vid in $SRC"; fi
    got=$($HASH "$p" | awk '{ print $1 }')
    [ "$got" = "$want" ] || die "$file: sha256 $got diverso dal catalogo $want"
    list="$list$p$TAB"
  done
  [ -n "$list" ] || die "nessuna parte elencata per $vid"
  z=""; [ "$comp" = gzip ] && z=z
  OLDIFS=$IFS; IFS=$TAB
  # shellcheck disable=SC2086
  cat $list | tar -xp${z}f - -C "$ROOT"
  IFS=$OLDIFS
  sums="$IDX/$vid.sha256"
  [ -f "$sums" ] || die "manca $sums"
  if [ -s "$sums" ]; then
    out=$(cd "$ROOT" && $CHECK "$sums" 2>&1) || { echo "$out" | grep -v ': OK$' | head -20; die "$vid: controllo dei file fallito"; }
  fi
  echo "$vid" >> "$DONE"
  if [ "$DEL" = --elimina-parti ]; then
    OLDIFS=$IFS; IFS=$TAB
    for p in $list; do rm -f -- "$p"; done
    IFS=$OLDIFS
  fi
  n=$((n + 1))
  echo "   ok: parti integre, estratto, file controllati ($n in questa esecuzione)"
done
echo "fatto: volumi ripristinati elencati in $DONE"
