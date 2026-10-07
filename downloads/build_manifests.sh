#!/usr/bin/env bash
# Genera i manifest di un gruppo di downloads/sources.tsv (solo chiamate all'API, nessun file grande).
# Uso: downloads/build_manifests.sh p1        -> downloads/manifests/p1.tsv
# Le righe con "VERIFICARE" nella nota o nell'include vengono saltate finché non sono corrette.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "uso: build_manifests.sh GRUPPO" >&2; exit 2; }
GROUP=$1
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname -- "$HERE")
OUT="$HERE/manifests/$GROUP.tsv"
mkdir -p -- "$HERE/manifests"
: > "$OUT"
n=0
while IFS=$'\t' read -r group kind repo rev include exclude note; do
  [[ -z ${group// } || $group == \#* || $group != "$GROUP" ]] && continue
  if [[ $note == *VERIFICARE* || $include == *VERIFICARE* ]]; then
    echo "salto $repo: da verificare ($note)"
    continue
  fi
  args=(--repo "$repo" --type "$kind" --revision "$rev" --out "$OUT" --append)
  if [[ $include != "-" ]]; then IFS=';' read -ra inc <<< "$include"; for g in "${inc[@]}"; do args+=(--include "$g"); done; fi
  if [[ $exclude != "-" ]]; then IFS=';' read -ra exc <<< "$exclude"; for g in "${exc[@]}"; do args+=(--exclude "$g"); done; fi
  /usr/bin/python3 -I "$ROOT/scripts/hf_manifest.py" "${args[@]}"
  n=$((n + 1))
  sleep 1  # una richiesta alla volta, senza raffiche verso l'API
done < "$HERE/sources.tsv"
echo "$n sorgenti -> $OUT ($(grep -vc '^#' "$OUT") file)"
