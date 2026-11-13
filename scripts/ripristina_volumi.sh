#!/usr/bin/env bash
# ripristina_volumi.sh — ripristino dei volumi del backup con i soli strumenti standard, su macOS o su Linux
# (NON sul frontend di Lagrange: su un cluster l'estrazione va in un job). Eseguirlo da una copia esterna alla
# cartella che si estrae (per esempio da LEGGIMI/script/), perché il volume del codice la sovrascrive.
# Per ogni volume dell'elenco, nell'ordine:
#   1. lo stato del volume nell'elenco deve essere "verificato";
#   2. sha256 di ogni parte uguale al catalogo (sha256sum o shasum -a 256);
#   3. cat parti | tar -xp[z]f - -C RADICE (permessi, date, link e hard link come nell'archivio);
#   4. controllo di ogni file con il suo <volume>.sha256 (sha256sum -c o shasum -a 256 -c).
# Un errore in qualunque passo ferma tutto (pipefail): niente marcatore, niente parti cancellate.
# Ripresa: i volumi riusciti sono annotati in RADICE.ripristino-fatti con nome e sha256 del flusso, quindi un volume
# con lo stesso nome ma contenuto diverso (per esempio il codice di un supplemento dopo quello della base) si applica.
#
# Uso: bash ripristina_volumi.sh VOLUMI_TSV PARTI_TSV CARTELLA_PARTI CARTELLA_INDICI RADICE [nucleo|tutti]
#                                [--elimina-parti] [--rimuovi-elenco FILE]
#   VOLUMI_TSV       elenco dei volumi della generazione (docs/backup/<generazione>-volumi.tsv)
#   PARTI_TSV        elenco delle parti (docs/backup/<generazione>-parti.tsv)
#   CARTELLA_PARTI   dove stanno le parti scaricate: CARTELLA/<categoria>/<file> (come su Drive) o CARTELLA/<file>
#   CARTELLA_INDICI  i file <volume>.sha256 scaricati da indice/volumi/ su Drive
#   RADICE           cartella di destinazione (nuova, o con il ripristino della generazione precedente)
#   --elimina-parti  cancella le parti di un volume solo dopo che il suo controllo è riuscito
#   --rimuovi-elenco FILE  dopo i volumi, toglie i percorsi elencati (relativi alla radice): file rimossi o
#                    rinominati fra la base e un supplemento (docs/backup/<supplemento>-rimossi.txt)
set -euo pipefail
die() { echo "ripristina_volumi.sh: $*" >&2; exit 1; }
abs() { case $1 in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
[ $# -ge 5 ] || die "uso: bash ripristina_volumi.sh VOLUMI_TSV PARTI_TSV CARTELLA_PARTI CARTELLA_INDICI RADICE [nucleo|tutti] [--elimina-parti] [--rimuovi-elenco FILE]"
VOL=$(abs "$1"); PARTS=$(abs "$2"); SRC=$(abs "$3"); IDX=$(abs "$4"); ROOT=$(abs "$5"); shift 5
WHICH=nucleo; DEL=""; RMLIST=""
while [ $# -gt 0 ]; do
  case $1 in
    nucleo|tutti) WHICH=$1 ;;
    --elimina-parti) DEL=1 ;;
    --rimuovi-elenco) [ $# -ge 2 ] || die "--rimuovi-elenco vuole un file"; RMLIST=$(abs "$2"); shift ;;
    *) die "argomento sconosciuto: $1" ;;
  esac
  shift
done
[ -f "$VOL" ] && [ -f "$PARTS" ] || die "elenchi non trovati: $VOL, $PARTS"
[ -d "$SRC" ] && [ -d "$IDX" ] || die "cartelle non trovate: $SRC, $IDX"
[ -z "$RMLIST" ] || [ -f "$RMLIST" ] || die "elenco dei percorsi da togliere non trovato: $RMLIST"
if command -v sha256sum > /dev/null 2>&1 && [ -z "${FORZA_SHASUM:-}" ]; then HASH="sha256sum"; CHECK="sha256sum -c"
else HASH="shasum -a 256"; CHECK="shasum -a 256 -c"; fi
TAB=$(printf '\t')
col() {  # numero della colonna di nome $1 nell'intestazione di $2
  head -n 1 "$2" | tr "$TAB" '\n' | grep -n -x -F "$1" | cut -d: -f1 | head -n 1
}
CV=$(col volume "$VOL"); CC=$(col categoria "$VOL"); CN=$(col nucleo "$VOL"); CZ=$(col compressione "$VOL")
CS=$(col sha256_flusso "$VOL"); CT=$(col stato "$VOL")
[ -n "$CV" ] && [ -n "$CC" ] && [ -n "$CN" ] && [ -n "$CZ" ] && [ -n "$CS" ] && [ -n "$CT" ] \
  || die "$VOL: intestazione senza le colonne volume, categoria, nucleo, compressione, sha256_flusso, stato"
mkdir -p "$ROOT"
DONE="$ROOT.ripristino-fatti"
touch "$DONE"
n=0
while IFS="$TAB" read -r vid cat nuc comp sflux st; do
  [ "$WHICH" = tutti ] || [ "$nuc" = 1 ] || continue
  if grep -q -x -F "$vid$TAB$sflux" "$DONE"; then continue; fi
  [ "$st" = verificato ] || die "$vid: stato \"$st\" nell'elenco, non verificato: mi fermo"
  echo "== $vid ($cat)"
  list=()
  while IFS="$TAB" read -r file want; do
    if [ -f "$SRC/$cat/$file" ]; then p="$SRC/$cat/$file"; elif [ -f "$SRC/$file" ]; then p="$SRC/$file"
    else die "manca la parte $file di $vid in $SRC"; fi
    got=$($HASH "$p" | awk '{ print $1 }')
    [ "$got" = "$want" ] || die "$file: sha256 $got diverso dal catalogo $want"
    list+=("$p")
  done < <(awk -F'\t' -v v="$vid" 'NR > 1 && $1 == v { print $2 "\t" $4 }' "$PARTS")
  [ ${#list[@]} -gt 0 ] || die "nessuna parte elencata per $vid"
  z=""; [ "$comp" = gzip ] && z=z
  cat -- "${list[@]}" | tar -xp${z}f - -C "$ROOT" || die "$vid: estrazione non riuscita"
  sums="$IDX/$vid.sha256"
  [ -f "$sums" ] || die "manca $sums"
  if [ -s "$sums" ]; then
    out=$(cd "$ROOT" && $CHECK "$sums" 2>&1) || { echo "$out" | grep -v ': OK$' | head -20; die "$vid: controllo dei file fallito"; }
  fi
  printf '%s\t%s\n' "$vid" "$sflux" >> "$DONE"
  if [ -n "$DEL" ]; then rm -f -- "${list[@]}"; fi
  n=$((n + 1))
  echo "   ok: parti integre, estratto, file controllati ($n in questa esecuzione)"
done < <(awk -F'\t' -v a="$CV" -v b="$CC" -v c="$CN" -v d="$CZ" -v e="$CS" -v f="$CT" \
           'NR > 1 { print $a "\t" $b "\t" $c "\t" $d "\t" $e "\t" $f }' "$VOL")
if [ -n "$RMLIST" ]; then
  removed=0
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    case "/$rel/" in */../*|*/./*) die "percorso non ammesso in $RMLIST: $rel" ;; esac
    case $rel in /*) die "percorso assoluto non ammesso in $RMLIST: $rel" ;; esac
    t="$ROOT/$rel"
    if [ -L "$t" ] || [ -f "$t" ]; then rm -f -- "$t"; removed=$((removed + 1))
    elif [ -d "$t" ]; then rmdir -- "$t" 2> /dev/null && removed=$((removed + 1)) || echo "   cartella non vuota, lasciata: $rel"
    fi
  done < "$RMLIST"
  echo "tolti $removed percorsi dell'elenco $RMLIST"
fi
echo "fatto: volumi ripristinati elencati in $DONE"
