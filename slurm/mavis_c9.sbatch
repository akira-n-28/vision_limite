#!/usr/bin/env bash
# C9 di MAVIS-Geometry (scripts/mavis_c9.py, src/limite_vl/mavis_c9.py; S2 §3). Job solo CPU; shard, manifest e pannello
# in sola lettura, uscite in cartelle nuove. Si leggono solo i JSON degli shard, legati a mavis-geometry-candidato-3.
# Uso:
#   sbatch slurm/mavis_c9.sbatch sonda NOME
#     ricognizione dei fatti nelle annotazioni, nessun manifest -> outputs/mavis_c9/NOME/report.json
#   sbatch [--mem 48G --time 01:30:00] slurm/mavis_c9.sbatch costruzione NOME VERSIONE
#     prima tests/test_mavis_c9.py (se fallisce non si legge nulla); poi scansione dei 199.050 record -> outputs/mavis_c9/
#     NOME/report.json, versione nuova data/manifests/mavis-geometry-candidato-VERSIONE derivata da candidato-3, verifica
#     -> outputs/manifest_check/mavis-geometry-candidato-VERSIONE/verifica.json, riepilogo
#     docs/manifests/mavis-geometry-candidato-VERSIONE.json
#   sbatch [--mem 48G] slurm/mavis_c9.sbatch verifica NOME VERSIONE SHA256
#     prima tests/test_mavis_c9.py; poi verifica del candidato C9 già scritto (sha256 dato), coi valori confrontati coi
#     JSON originali degli shard -> outputs/manifest_check/mavis-geometry-candidato-VERSIONE-NOME/verifica.json, copia
#     docs/manifests/NOME/mavis-geometry-candidato-VERSIONE-<sha256 breve>.json; il candidato non cambia
#   sbatch [--mem 32G] slurm/mavis_c9.sbatch lettore NOME VERSIONE SHA256
#     collegamento col reader sul primo shard (candidato C9 con lo sha256 dato e candidato-3) -> outputs/mavis_c9/NOME/
#     lettore.json, copia in docs/reader_smoke/NOME.json
#SBATCH --job-name=lvl-mavis-c9
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
usage() { echo "uso: mavis_c9.sbatch sonda NOME | costruzione NOME VERSIONE | verifica NOME VERSIONE SHA256" \
  "| lettore NOME VERSIONE SHA256" >&2; exit 2; }
(( $# >= 2 )) || usage
MODE=$1 NAME=$2
case $MODE in
  sonda) (( $# == 2 )) || usage ;;
  costruzione) (( $# == 3 )) && [[ $3 =~ ^[0-9]+$ ]] || usage ;;
  verifica|lettore) (( $# == 4 )) && [[ $3 =~ ^[0-9]+$ && $4 =~ ^[0-9a-f]{64}$ ]] || usage ;;
  *) usage ;;
esac
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nome non valido" >&2; exit 2; }
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
MANIFEST="$DATA_ROOT/manifests/mavis-geometry-candidato-3"
MANIFEST_SHA256=f3cca47367799ccd8a7c8f4d6b63c5c37e0cd06e22d921daa5507c6aa442b9c6  # job 30676
INDEX="$DATA_ROOT/decontam/pannello-v3"
INDEX_SHA256=02fb3cbfc36cbfd17bba92e4511c18a5760d330bc0145855fff2e7aa5130d127  # job 30469
SHARDS="$DATA_ROOT/shards/mavis-geometry/9c80fff3-caption/mavis-geometry-caption-*.tar"
OUT="$OUTPUT_ROOT/mavis_c9/$NAME"
cd "$PROJECT_ROOT"
test -x "$RUN_PYTHON"
source slurm/prologue.sh
lagrange_prologue cpu
echo "=== file modificati o nuovi rispetto a HEAD: id git del contenuto eseguito"
git ls-files -m -o --exclude-standard | while IFS= read -r f; do
  if [[ -f $f ]]; then printf '%s  %s\n' "$(git hash-object -- "$f")" "$f"; fi
done
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
test -s "$MANIFEST/manifest.json" || { echo "manca $MANIFEST/manifest.json" >&2; exit 1; }
[[ ! -e $OUT ]] || { echo "$OUT esiste già" >&2; exit 1; }
t0=$(date +%s)
case $MODE in
  sonda)
    if srun "$RUN_PYTHON" -u -B scripts/mavis_c9.py sonda --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA256" \
        --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-4}"; then
      ledger_note ok "mavis_c9 sonda $NAME -> $OUT/report.json ($(( $(date +%s) - t0 )) s)"
    else
      ledger_note failed "mavis_c9 sonda $NAME: vedi il log"; exit 1
    fi ;;
  costruzione)
    VERSION=$3
    NEW="$DATA_ROOT/manifests/mavis-geometry-candidato-$VERSION"
    CHECK="$OUTPUT_ROOT/manifest_check/mavis-geometry-candidato-$VERSION"
    SUMMARY="docs/manifests/mavis-geometry-candidato-$VERSION.json"
    for p in "$NEW" "$CHECK" "$SUMMARY"; do [[ ! -e $p ]] || { echo "$p esiste già" >&2; exit 1; }; done
    for f in "$INDEX/index.json"; do test -s "$f" || { echo "manca $f" >&2; exit 1; }; done
    if ! srun "$RUN_PYTHON" -B tests/test_mavis_c9.py; then
      ledger_note failed "mavis_c9 costruzione $NAME: tests/test_mavis_c9.py fallito, nessuna lettura"; exit 1
    fi
    if srun "$RUN_PYTHON" -u -B scripts/mavis_c9.py costruzione --manifest "$MANIFEST" \
        --manifest-sha256 "$MANIFEST_SHA256" --new "$NEW" --out "$OUT" --report "$CHECK" --index "$INDEX" \
        --index-sha256 "$INDEX_SHA256" --summary "$SUMMARY" --version "$VERSION" --workers "${SLURM_CPUS_PER_TASK:-4}"; then
      ledger_note ok "mavis_c9 costruzione $NAME: scansione -> $OUT/report.json, manifest -> $NEW, verifica -> $CHECK/verifica.json, riepilogo -> $SUMMARY ($(( $(date +%s) - t0 )) s)"
    else
      ledger_note failed "mavis_c9 costruzione $NAME: vedi il log"; exit 1
    fi ;;
  verifica)
    VERSION=$3 NEW_SHA256=$4
    NEW="$DATA_ROOT/manifests/mavis-geometry-candidato-$VERSION"
    CHECK="$OUTPUT_ROOT/manifest_check/mavis-geometry-candidato-$VERSION-$NAME"
    DOC="docs/manifests/$NAME/mavis-geometry-candidato-$VERSION-${NEW_SHA256:0:8}.json"
    for p in "$CHECK" "$DOC"; do [[ ! -e $p ]] || { echo "$p esiste già" >&2; exit 1; }; done
    test -s "$NEW/manifest.json" || { echo "manca $NEW/manifest.json" >&2; exit 1; }
    if ! srun "$RUN_PYTHON" -B tests/test_mavis_c9.py; then
      ledger_note failed "mavis_c9 verifica $NAME: tests/test_mavis_c9.py fallito, nessuna lettura"; exit 1
    fi
    if srun "$RUN_PYTHON" -u -B scripts/mavis_c9.py verifica --manifest "$MANIFEST" \
        --manifest-sha256 "$MANIFEST_SHA256" --new "$NEW" --new-sha256 "$NEW_SHA256" --report "$CHECK" \
        --workers "${SLURM_CPUS_PER_TASK:-4}"; then
      mkdir -p "$(dirname "$DOC")"
      cp "$CHECK/verifica.json" "$DOC"
      ledger_note ok "mavis_c9 verifica $NAME: $NEW -> $CHECK/verifica.json, copia $DOC ($(( $(date +%s) - t0 )) s)"
    else
      if [[ -s $CHECK/verifica.json ]]; then mkdir -p "$(dirname "$DOC")"; cp "$CHECK/verifica.json" "$DOC"; fi
      ledger_note failed "mavis_c9 verifica $NAME: vedi il log e $CHECK/verifica.json"; exit 1
    fi ;;
  lettore)
    VERSION=$3 NEW_SHA256=$4
    NEW="$DATA_ROOT/manifests/mavis-geometry-candidato-$VERSION"
    DOC="docs/reader_smoke/$NAME.json"
    [[ ! -e $DOC ]] || { echo "$DOC esiste già" >&2; exit 1; }
    if srun "$RUN_PYTHON" -u -B scripts/mavis_c9.py lettore --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA256" \
        --new "$NEW" --new-sha256 "$NEW_SHA256" --shards "$SHARDS" --out "$OUT"; then
      cp "$OUT/lettore.json" "$DOC"
      ledger_note ok "mavis_c9 lettore $NAME -> $OUT/lettore.json, copia $DOC ($(( $(date +%s) - t0 )) s)"
    else
      ledger_note failed "mavis_c9 lettore $NAME: vedi il log"; exit 1
    fi ;;
esac
