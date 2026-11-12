#!/bin/sh
# mac_confronta_hash.sh — riconosce per contenuto (sha256) le copie dei dataset già presenti sul Mac rispetto ai file
# di Lagrange salvati nel backup. Da eseguire sul Mac (o su un altro computer), non sul frontend di Lagrange.
#   1. sha256 di ogni file sotto CARTELLA_MAC (find + shasum -a 256 o sha256sum);
#   2. confronto con docs/backup/hash_riferimento_dati.tsv (file di hf/ e data/shards/ con sha256 calcolato nella
#      creazione dei volumi) e con docs/backup/catalogo_recupero.tsv (hash dei file LFS di Hugging Face);
#   3. tre elenchi nella cartella corrente: uguali (copia equivalente, con il percorso su Lagrange), solo sul Mac,
#      solo su Lagrange. Un nome uguale non basta: conta solo il contenuto.
# Uso: sh mac_confronta_hash.sh CARTELLA_MAC hash_riferimento_dati.tsv catalogo_recupero.tsv
set -eu
[ $# -eq 3 ] || { echo "uso: sh mac_confronta_hash.sh CARTELLA_MAC hash_riferimento_dati.tsv catalogo_recupero.tsv" >&2; exit 1; }
DIR=$1; REF=$2; CAT=$3
if command -v sha256sum > /dev/null 2>&1 && [ -z "${FORZA_SHASUM:-}" ]; then HASH="sha256sum"; else HASH="shasum -a 256"; fi  # FORZA_SHASUM: prova del ramo macOS
echo "calcolo gli sha256 sotto $DIR (può richiedere qualche minuto)..."
find "$DIR" -type f ! -name '.DS_Store' -print0 | xargs -0 $HASH > confronto-mac-sha256.txt
# riferimento: sha256 -> percorso su Lagrange (dai volumi) e file LFS del catalogo (sha256 = contenuto)
{ tail -n +2 "$REF" | awk -F'\t' '{ print $1 "\t" $3 "\t" $2 }'
  tail -n +2 "$CAT" | awk -F'\t' '$7 == "sha256" { print $8 "\t" $2 "@" substr($3, 1, 8) "/" $5 "\t" $6 }'
} | sort -u > confronto-riferimento.tsv
rm -f confronto-uguali.tsv confronto-solo-mac.tsv confronto-solo-lagrange.tsv
awk -F'\t' 'NR == FNR { ref[$1] = (ref[$1] ? ref[$1] " | " : "") $2; next }
  { h = substr($0, 1, 64); p = substr($0, 67)
    if (h in ref) { print h "\t" p "\t" ref[h] > "confronto-uguali.tsv"; seen[h] = 1 }
    else print h "\t" p > "confronto-solo-mac.tsv" }
  END { for (h in ref) if (!(h in seen)) print h "\t" ref[h] > "confronto-solo-lagrange.tsv" }' \
  confronto-riferimento.tsv confronto-mac-sha256.txt
touch confronto-uguali.tsv confronto-solo-mac.tsv confronto-solo-lagrange.tsv
echo "uguali per contenuto: $(wc -l < confronto-uguali.tsv) file del Mac"
echo "solo sul Mac:         $(wc -l < confronto-solo-mac.tsv) file"
echo "solo su Lagrange:     $(wc -l < confronto-solo-lagrange.tsv) contenuti del riferimento"
echo "elenchi: confronto-uguali.tsv, confronto-solo-mac.tsv, confronto-solo-lagrange.tsv"
