#!/usr/bin/env bash
# Prova di ripristino in due passaggi, base poi supplemento, con lo script per il Mac (scripts/ripristina_volumi.sh,
# copiato fuori dalla cartella che si estrae) e verifica della composizione (scripts/backup.py composizione): stato
# finale uguale agli indici applicati in ordine meno i percorsi rimossi, commit di Git, cartelle confrontate con il
# progetto e con il commit, file attesi per sha256. Copia di prova in trasferimento/tmp/composizione-<job>, rimossa
# alla fine; il rapporto va in trasferimento/stato/<supplemento>/ripristino/.
#
# Uso: sbatch --parsable slurm/backup_composizione.sbatch BASE SUPPLEMENTO SCARICATI VOLUMI_BASE COMMIT \
#        [--confronta CARTELLA]... [--atteso PERCORSO=SHA256]...
#   BASE, SUPPLEMENTO  generazioni (elenchi in docs/backup/<generazione>-{volumi,parti}.tsv, <supplemento>-rimossi.txt)
#   SCARICATI          cartella con <generazione>/volumi/<categoria>/<parte> e <generazione>/indici/<volume>.{sha256,jsonl.gz}
#   VOLUMI_BASE        volumi della base da ripristinare, separati da virgole (la prova non riscarica tutta la base)
#   COMMIT             commit atteso nel codice ripristinato
#SBATCH --job-name=lvl-bk-comp
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/trasferimento/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/trasferimento/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
base=${1:?generazione di base}; sup=${2:?supplemento}; dl=${3:?cartella degli scaricati}
vols=${4:?volumi della base}; commit=${5:?commit atteso}; shift 5
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
B=docs/backup
for f in "$B/$base-volumi.tsv" "$B/$base-parti.tsv" "$B/$sup-volumi.tsv" "$B/$sup-parti.tsv" "$B/$sup-rimossi.txt"; do
  [[ -f $f ]] || { echo "manca $f" >&2; exit 2; }
done
tmp=$PROJECT_ROOT/trasferimento/tmp/composizione-$SLURM_JOB_ID
mkdir -p "$PROJECT_ROOT/trasferimento/stato/$sup/ripristino" "$tmp/script" "$tmp/elenchi"
report=$PROJECT_ROOT/trasferimento/stato/$sup/ripristino/composizione-$SLURM_JOB_ID.json
cp scripts/ripristina_volumi.sh "$tmp/script/"
echo "script: $(sha256sum "$tmp/script/ripristina_volumi.sh")"
# Elenco della base ridotto ai volumi della prova; le parti si cercano per nome di volume.
awk -F'\t' -v want="$vols" 'BEGIN { n = split(want, a, ","); for (i = 1; i <= n; i++) w[a[i]] = 1 }
  NR == 1 { for (i = 1; i <= NF; i++) if ($i == "volume") c = i; print; next } ($c in w)' \
  "$B/$base-volumi.tsv" > "$tmp/elenchi/$base-volumi.tsv"
[[ $(($(wc -l < "$tmp/elenchi/$base-volumi.tsv") - 1)) -eq $(tr ',' '\n' <<< "$vols" | wc -l) ]] \
  || { echo "volumi della base non trovati tutti nell'elenco: $vols" >&2; exit 2; }
indici=()
for v in ${vols//,/ }; do indici+=("$dl/$base/indici/$v.jsonl.gz"); done
for v in $(awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "volume") c = i; next } { print $c }' \
    "$B/$sup-volumi.tsv"); do
  indici+=("$dl/$sup/indici/$v.jsonl.gz")
done
ok=1
echo "== primo passaggio: $base ($vols)"
srun bash "$tmp/script/ripristina_volumi.sh" "$tmp/elenchi/$base-volumi.tsv" "$B/$base-parti.tsv" \
  "$dl/$base/volumi" "$dl/$base/indici" "$tmp/radice" nucleo || ok=0
if [[ $ok == 1 ]]; then
  echo "== secondo passaggio: $sup"
  srun bash "$tmp/script/ripristina_volumi.sh" "$B/$sup-volumi.tsv" "$B/$sup-parti.tsv" \
    "$dl/$sup/volumi" "$dl/$sup/indici" "$tmp/radice" tutti --rimuovi-elenco "$B/$sup-rimossi.txt" || ok=0
fi
echo "== marcatori"; cat "$tmp/radice.ripristino-fatti" || true
if [[ $ok == 1 ]] && srun /usr/bin/python3 -I -B scripts/backup.py composizione --radice "$tmp/radice" \
    --indici "${indici[@]}" --rimossi "$B/$sup-rimossi.txt" --git-commit "$commit" --progetto "$PROJECT_ROOT" \
    "$@" --report "$report"; then
  status=ok
else
  status=failed
fi
rm -rf -- "$tmp"
ledger_note "$status" "backup ripristino in due passaggi $base ($vols) + $sup da $dl -> $report"
[[ $status == ok ]]
