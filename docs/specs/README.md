# Specifiche F0

Le quattro specifiche brevi della pipeline v1.4 §5.10, da approvare prima di scrivere codice
(`docs/F0_IMPLEMENTATION_PLAN.md`, §1). Bozze del 2026-10-28: nessuna è approvata, nessun codice le implementa ancora.

| Specifica | Tema | Decisioni dell'utente che contiene |
| --- | --- | --- |
| [S1](S1_RECORD_MANIFEST.md) | Identità dell'esempio, record e manifest, gruppi e split, indice globale, esclusioni storiche, uso nei reader e contabilità | stato proposto per ogni lista storica (§5) |
| [S2](S2_AMMISSIONE_PRIME_FAMIGLIE.md) | Regole di ammissione di sintetici, MAVIS e UniMER: proprietà, livelli di evidenza, stato gold, problemi da correggere | riformulazione di `shape_type`; soglia di leggibilità del renderer |
| [S3](S3_CONTRATTO_ESTRAZIONE.md) | Contratto di risposta `vl-contract/1`, estrazione, verificatori per compito, esiti e politica | D3: template, modalità, forma della risposta diretta, prefill |
| [S4](S4_CAMPIONAMENTO_BUDGET.md) | Campioni diagnostico e di stima, quantità e calcolo, attendibilità del riferimento, budget degli audit | D4 (soglie), D5 (budget); proposta per le prime famiglie (§6) |

Ordine di implementazione dopo l'approvazione: S3 (contratto) e S1 (record) per primi, perché S2 e il punto 3 di F0 li
usano; poi S2 sulle tre famiglie; S4 serve prima di qualsiasi campione di stima o audit.
