# Archivio della strategia precedente (fino al 2026-10-26)

Migrazione del 2026-10-28 alla [pipeline v1.4](../../PIPELINE_ADDESTRAMENTO_LIMITE_VL.md). Qui ci sono le ricette e i
launcher della vecchia sequenza F1 / F1b / F1c / F2 (ponte teacher, gate f2-cert, replay generato da Violetto). Il
contenuto è identico byte per byte a quello del tag `pre-migrazione-v1.4` (commit `6e888d5`): `MOVES.tsv` riporta, per
ogni file, percorso precedente, percorso attuale, sha256, blob git, categoria, motivo e ultimo commit che lo ha toccato.

**Non sono ricette o comandi correnti.** Non si lanciano, non si copiano in `config/runs/` cambiando solo il nome e non
si usano per riprendere i run storici. Le nuove ricette nascono dopo inventario, specifiche e collaudi F0
(`docs/F0_IMPLEMENTATION_PLAN.md`).

| Cartella | Contenuto |
| --- | --- |
| `config/runs/` | Le 13 config dei run storici, inclusa la bozza `f1-clean.json` (liste di decontaminazione e rietichettature di MAVIS, UniMER e sintetici, da importare come evidenze secondo la pipeline §5.8, non da applicare alla cieca) |
| `slurm/` | 29 launcher: training (4), F1c (6), ponte teacher (10), audit e gate (7), replay generato da Violetto (2) |
| `CLAUDE.pre-migrazione.md`, `README.pre-migrazione.md` | Copie integrali di `CLAUDE.md` e `README.md` al tag `pre-migrazione-v1.4` (sha256 `0a2238e0…`, `980f3d84…`): vecchio piano esterno, dati e mix di F1, record e gate F2, regole degli audit, ottimizzatore di F2. Le istruzioni correnti ne conservano solo i vincoli operativi e i fatti tecnici |

## Cosa resta dove era

- Gli script Python (`scripts/train_f1.py`, `scripts/train_f2.py`, `scripts/f2_*.py`, `scripts/f1c_*.py`,
  `scripts/stem_audit_*.py`, `scripts/replay_*.py`) e i moduli (`src/limite_vl/f1c.py`, `f2.py`, `f2_records.py`,
  `f2_claims.py`, `facts.py`, `stem_audit.py`) restano nei loro percorsi: altri strumenti e i test li importano, e
  `scripts/f2_pilot.py` congela per percorso e sha256 `scripts/f2_pilot.py`, `src/limite_vl/f2_claims.py`,
  `src/limite_vl/stem_audit.py` e le istruzioni in `config/audit/`. Lo stato di ciascuno è in `docs/REPOSITORY_INDEX.tsv`.
- Le docstring di quegli script citano ancora `slurm/<nome>.sbatch`: il nuovo percorso è in `MOVES.tsv`. Non sono state
  corrette per non cambiare l'identità del codice congelato.
- I manifest in `outputs/<run>/manifest.json` registrano la config per contenuto (`config_digest`) e la config intera:
  lo spostamento non li invalida.

## Recupero

Rimettere un file al suo posto: `git mv archive/pipeline-precedente/<percorso> <percorso>`. Tornare allo stato
precedente alla migrazione: `git checkout pre-migrazione-v1.4 -- config/runs slurm CLAUDE.md README.md`. Una copia dei
piccoli file toccati è anche in `outputs/migration-v1.4/pre-migration-small-files.tar` (fuori da git, sha256 accanto).
