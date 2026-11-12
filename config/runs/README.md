# Config dei run

Vuota di proposito dal 2026-10-28. Le config della strategia precedente sono in
`archive/pipeline-precedente/config/runs/` (elenco e hash in `archive/pipeline-precedente/MOVES.tsv`).

Una config entra qui solo quando sono implementati e collaudati il manifest dei dati, il trainer e i contratti che
usa (`docs/F0_IMPLEMENTATION_PLAN.md`) e l'utente ha approvato dati, architettura e budget del run. Una config di
prova di un collaudo F0 dichiara nel campo `notes` che non è una ricetta di training.

Config di collaudo presenti, nessuna è una ricetta:
- `f0-c7-reader-unimer-mavis.json`: le sorgenti di UniMER (formule e righe ritagliate) e di MAVIS coi manifest candidati
  identificati dal loro sha256, per lo smoke test di C7 (`scripts/reader_smoke.py`, `slurm/reader_smoke.sbatch`).
- `f0-c7-reader-pixmo.json`: la sorgente di PixMo-Cap col manifest candidato, per lo smoke test del reader
  (`slurm/pixmo_manifest.sbatch lettore`).
- `f0-c7-reader-synth.json`: le quattro sorgenti dei sintetici v4 (train, dev, test, coppie) con `synth-v4-candidato-3`
  identificato dal suo sha256, per la verifica del reader col filtro delle domande (`scripts/manifest_check.py --config`,
  `slurm/synth_qa_policy.sbatch verifica`; S2 §2.14).
