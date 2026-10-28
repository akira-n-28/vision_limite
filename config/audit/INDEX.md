# Istruzioni e politica degli audit storici — indice

Stato al 2026-10-28 (migrazione alla pipeline v1.4): **congelate**. Sono le istruzioni degli agenti e la politica della
strategia precedente (screening STEM di MMFineReason, audit delle soluzioni F2 di Violetto, piloti v2a-v3a). Restano
qui, invariate, perché manifest, lotti e certificati in `outputs/` le citano per percorso e sha256:

- `scripts/f2_pilot.py` registra in `manifest.json` lo sha256 delle istruzioni usate e si ferma se cambiano
  (`outputs/f2_pilot/mavis-v2a`, `-v2b`, `-v2c`: `_rules.md`, `f2_extract.md`, `f2_claim_verify.md`;
  `mavis-v3a`: `_rules.md`, `f2_classify.md`, `f2_claim_verify2.md`);
- `scripts/f2_certify.py` incorpora `f2_policy.json`, `f2_solution_v2.md`, `f2_arbiter_v2.md`
  (`outputs/f2_cert/mavis-f2sol-retro`, `-retro2`); `scripts/stem_audit_export.py --f2-v2` registra le istruzioni v2;
- le istruzioni v1 (`stem_*.md`, `f2_solution.md`, `f2_arbiter.md`, con `_rules.md`) sono quelle date agli agenti negli
  audit `outputs/stem_audit/screening`, `mavis-f2`, `-f2b`, `-f2c`, `-f2sol` (istruzioni precedenti, in
  `archive/pipeline-precedente/CLAUDE.pre-migrazione.md`); il codice non ne registra l'hash, quindi l'identità di quella
  versione è il commit `pre-migrazione-v1.4` e la tabella sotto.

Non si modificano, non si spostano e non si riusano per ammettere dati della nuova pipeline. In `f2_policy.json` soglie,
`frame_counts_as_error` e decisioni sulle segnalazioni sono `null` per scelta dell'utente: ogni certificato esce
`rejected`. Gli audit fatti con queste istruzioni non contano come audit (decisione del 2026-10-26); nuove istruzioni,
se servono, avranno nomi nuovi e un budget approvato (pipeline §§5.4, 5.9).

| File | sha256 | Compito |
| --- | --- | --- |
| `_rules.md` | `b502eed5369d4c7987742189af795b9171bed9f912a47cba7cc4dacf7f1a5c21` | regole comuni degli agenti |
| `stem_solve.md` | `5a139ae22c16559bb1b73f952c7885a864d60414f04a854734fa3727e7bd3af5` | risolvere senza descrizione (screening STEM) |
| `stem_describe.md` | `705ca77b6d689ecf1985fc8c99a5d87fae23fce3d9171f45451c98211217ede9` | fatti della descrizione |
| `stem_gold.md` | `239cd44e5acf21bba06c3cdc0edd6be0149652fe3b0c3f4b1241a010a987f3d0` | revisione della risposta di riferimento |
| `stem_arbiter.md` | `73c794e76021fffdb854c27db2f369c9c6f1bf6e404e2a3639fd2ac2a66e0349` | arbitro dello screening |
| `f2_solution.md` | `27cc822c4768c2dace73bc4f13c1fa0d6dded7e3dcaea09d4b8a6f10ecde3a00` | soluzioni F2, v1 |
| `f2_arbiter.md` | `8e13acf6eda4221c4ca6c428f2e9580e2ba49c01f7b132aedbcc8e758fc76173` | arbitro F2, v1 |
| `f2_solution_v2.md` | `4488692ce17b7200fae086924bf7ec42f54f008ab30958a626283981a8865bd2` | soluzioni F2, v2 (`frame_ok`) |
| `f2_arbiter_v2.md` | `41fae2f547852c2ee2eb7953a1aeb42b6c2abbbe5f8c6521f38721d9f2cb29f4` | arbitro F2, v2 |
| `f2_extract.md` | `d0e6c950e13f400f5443b7ac8363bf609aded4f3140f41990e46128f7e7d9dd0` | pilota f2-pilot/1: estrazione delle affermazioni |
| `f2_claim_verify.md` | `f90b4ab717f7fc5e41d7a4b6b817a88226623bbded32901a9f74248e26927940` | pilota f2-pilot/1: verifica contro l'immagine |
| `f2_classify.md` | `4660fc9dbf4636771e7b574419dc20a1bf6e6022f72672c082e065b6692e379f` | pilota f2-pilot/2: classificazione per frase |
| `f2_claim_verify2.md` | `61295cadd2fda2627bdff1bf70c7a11fe19dad555b5e2df3134b92587a1ff661` | pilota f2-pilot/2: verifica |
| `f2_policy_definitions.md` | `72c56ea760316fb1fe3a1e3adc4496d82c86031c9bbc063b72781e557032f4b5` | definizioni per una politica futura |
| `f2_policy.json` | `fcca954a2d56eb4423efcf516cde14bfe510506d4946b8e00b17cf884eb9d832` | politica di certificazione F2 (valori aperti `null`) |
