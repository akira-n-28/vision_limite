#!/usr/bin/env bash
# mac_confronta_hash.sh — riconosce per contenuto (sha256) le copie dei dataset già presenti sul Mac rispetto ai file
# di Lagrange descritti dal backup. Da eseguire sul Mac (o su un altro computer), non sul frontend di Lagrange.
#   1. controlla gli ingressi: cartella leggibile, riferimento e catalogo presenti, non vuoti e col formato atteso;
#   2. sha256 di ogni file sotto CARTELLA_MAC (find + shasum -a 256 o sha256sum): un errore di lettura o di calcolo,
#      o un numero di hash diverso dal numero di file, rende il confronto NON VALIDO (codice d'uscita 1);
#   3. confronto con hash_riferimento_dati.tsv (file di hf/ e data/shards/ salvati nel backup, sha256 dei volumi) e
#      con catalogo_recupero.tsv (hash dei file LFS di Hugging Face, anche di quelli esclusi dal backup);
#   4. tre elenchi nella cartella corrente: uguali (copia equivalente, col percorso su Lagrange), solo sul Mac, solo su
#      Lagrange. Un nome uguale non basta: conta solo il contenuto. Con un errore gli elenchi non vengono scritti.
# Uso: bash mac_confronta_hash.sh CARTELLA_MAC hash_riferimento_dati.tsv catalogo_recupero.tsv
set -euo pipefail
die() { echo "mac_confronta_hash.sh: $*" >&2; echo "CONFRONTO NON VALIDO: nessuna equivalenza è dimostrata." >&2; exit 1; }
[ $# -eq 3 ] || die "uso: bash mac_confronta_hash.sh CARTELLA_MAC hash_riferimento_dati.tsv catalogo_recupero.tsv"
DIR=$1; REF=$2; CAT=$3
[ -d "$DIR" ] && [ -r "$DIR" ] && [ -x "$DIR" ] || die "cartella non leggibile: $DIR"
for f in "$REF" "$CAT"; do
  [ -f "$f" ] && [ -r "$f" ] || die "file non leggibile: $f"
  [ -s "$f" ] || die "file vuoto: $f"
done
TAB=$(printf '\t')
[ "$(head -n 1 "$REF")" = "sha256${TAB}byte${TAB}percorso${TAB}volume" ] || die "$REF: intestazione inattesa"
head -n 1 "$CAT" | awk -F'\t' '$7 == "tipo_hash" && $8 == "hash_hf" && $2 == "repository" { ok = 1 } END { exit !ok }' \
  || die "$CAT: intestazione inattesa (servono repository, tipo_hash e hash_hf nelle colonne 2, 7 e 8)"
nref=$(awk -F'\t' 'NR > 1 && length($1) == 64 && $1 !~ /[^0-9a-f]/' "$REF" | wc -l | tr -d ' ')
ncat=$(awk -F'\t' 'NR > 1 && $7 == "sha256" && length($8) == 64 && $8 !~ /[^0-9a-f]/' "$CAT" | wc -l | tr -d ' ')
[ "$nref" -gt 0 ] || die "$REF: nessun sha256 valido"
[ "$ncat" -gt 0 ] || die "$CAT: nessun hash sha256 valido"
bad=$(awk -F'\t' 'NR > 1 && (length($1) != 64 || $1 ~ /[^0-9a-f]/)' "$REF" | wc -l | tr -d ' ')
[ "$bad" -eq 0 ] || die "$REF: $bad righe senza sha256 valido"
if command -v sha256sum > /dev/null 2>&1 && [ -z "${FORZA_SHASUM:-}" ]; then HASH=(sha256sum); else HASH=(shasum -a 256); fi
OUT=confronto-mac-sha256.txt
rm -f confronto-uguali.tsv confronto-solo-mac.tsv confronto-solo-lagrange.tsv confronto-riferimento.tsv "$OUT"
echo "calcolo gli sha256 sotto $DIR (può richiedere qualche minuto)..."
nfiles=$(find "$DIR" -type f ! -name '.DS_Store' -print0 | tr -cd '\0' | wc -c | tr -d ' ') \
  || die "elenco dei file sotto $DIR non riuscito (cartelle illeggibili?)"
[ "$nfiles" -gt 0 ] || die "nessun file sotto $DIR"
find "$DIR" -type f ! -name '.DS_Store' -print0 | xargs -0 "${HASH[@]}" > "$OUT.tmp" \
  || { rm -f "$OUT.tmp"; die "lettura o calcolo degli hash non riuscito per almeno un file"; }
nhash=$(wc -l < "$OUT.tmp" | tr -d ' ')
[ "$nhash" -eq "$nfiles" ] || { rm -f "$OUT.tmp"; die "$nhash hash per $nfiles file: risultato parziale"; }
mv "$OUT.tmp" "$OUT"
{ awk -F'\t' 'NR > 1 { print $1 "\t" $3 "\t" $2 }' "$REF"
  awk -F'\t' 'NR > 1 && $7 == "sha256" { print $8 "\t" $2 "@" substr($3, 1, 8) "/" $5 "\t" $6 }' "$CAT"
} | sort -u > confronto-riferimento.tsv
awk -F'\t' 'NR == FNR { ref[$1] = (ref[$1] ? ref[$1] " | " : "") $2; next }
  { h = substr($0, 1, 64); p = substr($0, 67)
    if (h in ref) { print h "\t" p "\t" ref[h] > "confronto-uguali.tsv"; seen[h] = 1 }
    else print h "\t" p > "confronto-solo-mac.tsv" }
  END { for (h in ref) if (!(h in seen)) print h "\t" ref[h] > "confronto-solo-lagrange.tsv" }' \
  confronto-riferimento.tsv "$OUT"
touch confronto-uguali.tsv confronto-solo-mac.tsv confronto-solo-lagrange.tsv
echo "file sotto la cartella: $nfiles, tutti letti e calcolati"
echo "uguali per contenuto: $(wc -l < confronto-uguali.tsv | tr -d ' ') file del Mac"
echo "solo sul Mac:         $(wc -l < confronto-solo-mac.tsv | tr -d ' ') file"
echo "solo su Lagrange:     $(wc -l < confronto-solo-lagrange.tsv | tr -d ' ') contenuti del riferimento"
echo "elenchi: confronto-uguali.tsv, confronto-solo-mac.tsv, confronto-solo-lagrange.tsv"
