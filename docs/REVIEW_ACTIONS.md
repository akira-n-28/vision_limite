# Rilievi delle revisioni: recepimento e verifiche residue

Stato del 28 ottobre 2026, [pipeline v1.4](../PIPELINE_ADDESTRAMENTO_LIMITE_VL.md).
Le indicazioni sono recepite nel documento. La colonna finale specifica lavoro da
implementare o verificare: non è un elenco di correzioni al codice già completate.
Le decisioni sperimentali aperte restano tali; non si inventano risultati o budget.

## Prima revisione della pipeline

[Rapporto originale, v1.0](reviews/REVISIONE_INDIPENDENTE_PIPELINE_LIMITE_VL.md).

| Rilievo | Sezioni della pipeline | Verifica o decisione residua |
| --- | --- | --- |
| R1 — Template e comportamento | 3.3, 5.10, 9 | Versionare template/label/stop e misurare risposte dirette e ragionate |
| R2 — Contesto e immagini multiple | 3.4, 8 | Caricatori, ordine delle pagine, dipendenze lunghe e leggibilità |
| R3 — Estrazione Qwen | 3.5, 6.3 | Scegliere checkpoint e verificare estrazione/preprocessing contro percorso nativo |
| R4 — Aggiornamento ViT | 3.2, 8 | Gruppi, LR/precisione, variazioni pesi, salvataggio e ripresa |
| R5 — Replay | 5.8, 8 | Collegare mix e misurare target-token, lunghezze e scarti |
| R6 — Semantica dei target | 5.6–5.7 | Regole per compito e corrispondenza con input effettivo |
| R7 — Split trasversali | 5.5, 5.8, 6.1 | Gruppi globali e pannelli riservati prima dell'ammissione |
| R8 — Stima e certificazione | 5.9, 15.3 | Campione probabilistico separato, denominatori e riferimento calibrato |
| R9 — Nucleo iniziale | 6.1, 7 | Minimo insieme per copertura/costo; adattatori prioritari |
| R10 — Piloti comparabili | 6.2, 6.5 | Curve intermedie, convergenza e ripetizioni mirate se incerte |
| R11 — Integrazione RL | 11 | Replica immagini, log-probabilità, maschere e riferimento multimodale congelato |
| R12 — Specifica GRPO | 11 | Versione/formulazione, clipping, normalizzazione, KL, troncamenti e gruppi informativi |
| R13 — Verificatori | 5.10, 11 | Contratti per famiglia, collaudo indipendente e dipendenza visiva |
| R14 — Interpolazioni | 10, 12 | Registro tensori, pesi legati, round-trip e metriche di comportamento |

## Revisione di gold e decontaminazione

[Rapporto originale, v1.2](reviews/REVISIONE_INDIPENDENTE_GOLD_E_DECONTAMINAZIONE.md).
G1–G7 corrispondono ai sette rilievi numerati del rapporto.

| Rilievo | Sezioni della pipeline | Verifica o decisione residua |
| --- | --- | --- |
| G1 — Manifest delle QA effettive | 5.2, 5.8, 5.10 | Identità variante, filtraggio loader e contabilità degli scarti |
| G2 — Estrazione/selezione | 5.10, 11 | Answer-only, risposte finali incompatibili e stati del verificatore |
| G3 — Promozione a gold | 5.6 | Distinguere prove individuali, invarianti e stime campionarie |
| G4 — Ambiguità dei sintetici | 5.7, 5.10 | Classe più specifica o alternative ammesse, con collaudo |
| G5 — Esclusioni storiche | 5.8 | Riesame dall'inventario originale senza perdere sovrapposizioni confermate |
| G6 — Quantità statistica | 5.9, 15.3 | Popolazione, denominatore, gruppi/pesi e gestione degli incerti |
| G7 — Fallibilità del riferimento | 5.4, 5.9 | Calibrazione indipendente e limiti della doppia revisione |

Prima dell'esecuzione si aggiorna questa tabella con evidenze delle verifiche effettive,
senza modificare retroattivamente i rapporti indipendenti.

## Corrispondenza con gli interventi F0 (2026-10-28)

Nessun rilievo è ancora chiuso da un'implementazione collaudata. Lo stato del codice per ciascun requisito, con file,
righe e job, è in [`F0_IMPLEMENTATION_PLAN.md`](F0_IMPLEMENTATION_PLAN.md):

| Interventi F0 | Rilievi |
| --- | --- |
| Punto 1 — contratto di risposta | R1, G2 |
| Punto 2 — gold per compito, ambiguità, answer-only | R6, R13, G2, G3, G4 |
| Punto 3 — split globali, manifest, esclusioni storiche, contabilità | R5, R7, G1, G5 |
| Punto 4 — più immagini e contesto | R2 |
| Punto 5 — encoder Qwen, proiettore, layer, ingressi visivi | R3, R10 |
| Punto 6 — ViT addestrabile, gruppi, checkpoint | R4 |
| Punto 7 — pannelli e preventivo | R9, R10 |
| Specifica S4 — campionamento e budget | R8, G6, G7 |
| Fuori dal primo perimetro | R11, R12, R14 |
