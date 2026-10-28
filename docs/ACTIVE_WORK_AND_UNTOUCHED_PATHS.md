# Job, processi e percorsi non toccati dalla migrazione

Rilevato il 2026-10-28 fra le 14:37 e le 15:40 (ora del frontend), sul frontend `lagrangectl`, utente `prignano`.

## Job e processi in corso

| Controllo | Esito |
| --- | --- |
| `squeue -u prignano` | nessun job attivo o in coda |
| Ultimo job del ledger | 30052 (`lvl-f1-mavis`, ok, 2026-10-26 14:47); ultime note del 2026-10-26 14:55 |
| Processi dell'utente sul frontend | solo la sessione SSH, la shell e Claude Code; nessun `fetch.sh`, `fetch_manifest.sh` o `uv` |
| `data/.fetch.lock` | file vuoto del 2026-10-26 12:01, usato da `flock`: nessun download attivo. Lasciato com'è |
| `data/.fetch.stop` | assente |

Non c'era quindi nessun job che potesse leggere i file spostati. I launcher e le config archiviati non sono letti da
nessun job registrato dopo il 2026-10-26.

## Percorsi non toccati

Nessun file è stato cancellato. Fuori da git è stata creata solo `outputs/migration-v1.4/` (copia dei piccoli file e
sha256). Tutto il resto è stato solo letto con `ls`, `stat`, `readlink`, `du -s` e `du --inodes` (metadati), o con
`grep` su piccoli file di testo:

| Percorso | Stato verificato |
| --- | --- |
| `data/` (shard, liste di decontaminazione, lingua, link dei valutatori, specifiche, log) | invariato; link di `data/eval-datasets` validi |
| `hf/` (31 repository nella cache, `hf/datasets`, `hf/modules`) | invariato; una revisione per repository, nessun blob `.incomplete` |
| `outputs/` (55 cartelle di run, controlli, audit, log) | invariato |
| `venvs/` | invariato; `current` → `build-29382`, `evals-current` → `evals-build-29390`, interprete presente |
| `cache/`, `wheels/`, `tools/` | invariati |
| `secrets/` | elencato soltanto (cartella 700); contenuto non letto |
| `third_party/limite-violetto`, `third_party/verifiers` | HEAD `7ddad34f`, `d30a3f48`, alberi puliti |
| `third_party/limite-modeling` | invariato |
| `downloads/manifests/` | invariato |

## File versionati non modificati

- Tutto il codice: `scripts/`, `src/`, `tests/`, `slurm/prologue.sh` e i 50 launcher rimasti in `slurm/`
  (`git diff pre-migrazione-v1.4 -- scripts src tests` vuoto).
- `config/audit/*` (aggiunto solo `INDEX.md`), `config/paths.env`, `env/`, `env-evals/`, `downloads/build_manifests.sh`.
- `LAGRANGE.md`, `PIPELINE_ADDESTRAMENTO_LIMITE_VL.md`, `ISTRUZIONI_MIGRAZIONE_LAGRANGE.md`, `docs/reviews/*`.
- `runs/reports/`; `runs/ledger.tsv` riceve solo una riga di nota della migrazione.

## Dipendenze da rispettare prima di spostare altro

- `outputs/f1/final` → init di `f1b`; `outputs/f1b/final` → init di `f1c`, `f1c-control`, `f2-smoke`, `f2-abl-*`.
- `outputs/f2-abl-*` → `outputs/export/*-step540` → `outputs/eval-models` (stessi file, stesso inode) → `outputs/eval`.
- `outputs/f2_gen` → `outputs/f2_select` → `outputs/f2_backfill`, `data/shards/f2-mmfinereason`, `outputs/f2_cert`.
- `outputs/f2_pilot/*/manifest.json` e `outputs/f2_cert/*/certificate.json` citano per sha256 `scripts/f2_pilot.py`,
  `src/limite_vl/f2_claims.py`, `src/limite_vl/stem_audit.py` e file di `config/audit/`.
- La bozza archiviata `f1-clean.json` cita liste in `data/decontam/p6`, `data/decontam/f1eval`, `outputs/unimer_check/run4`,
  `outputs/unimer_overlap/run2`, `outputs/split_duplicates/run1`.
