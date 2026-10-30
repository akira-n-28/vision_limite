# Stato del repository rispetto alla pipeline v1.4

**Documento storico** della migrazione del 2026-10-28 (pipeline v1.4), non aggiornato dopo. Lo stato corrente è in
[`F0_IMPLEMENTATION_PLAN.md`](F0_IMPLEMENTATION_PLAN.md) (pipeline v1.5).

Preparato il 28 ottobre 2026 sul portatile per la migrazione alla [pipeline v1.4](../PIPELINE_ADDESTRAMENTO_LIMITE_VL.md);
**aggiornato lo stesso giorno sul cluster** dopo la [migrazione](MIGRATION_REPORT.md), con le evidenze del checkout reale.

## Esito della verifica sul cluster

- Checkout reale: `main` a `6e888d5` dopo il fast-forward da `9d977b6`, albero pulito, nessun job attivo, nessun lavoro
  locale non pubblicato. Il codice coincide con lo snapshot analizzato sul portatile.
- I 13 JSON di run storici sono archiviati come previsto; i launcher legati alla vecchia sequenza sono 29 (il portatile ne
  contava 28: [rapporto](MIGRATION_REPORT.md), §4). Elenco e hash in `archive/pipeline-precedente/MOVES.tsv`.
- Gli artefatti fuori da Git (`data/`, `hf/`, `outputs/`, `venvs/`, cache) sono inventariati in
  [`ARTIFACT_INVENTORY.tsv`](ARTIFACT_INVENTORY.tsv); i file versionati in [`REPOSITORY_INDEX.tsv`](REPOSITORY_INDEX.tsv).

La pipeline recepisce progettualmente i rilievi delle due revisioni ([matrice](REVIEW_ACTIONS.md)). I rapporti originali
restano riferiti alle versioni esaminate; nessuno di questi documenti prova che i requisiti siano già implementati.

## Verifica dell'implementazione

| Area | Componenti presenti sul cluster | Stato rispetto alla pipeline | Interventi |
| --- | --- | --- | --- |
| Formato delle risposte | Template Violetto, `F1_PREFILL`, estrattore storico (`f2.grade`) | Formato generato verificato; contratto diretto/ragionato mancante | [F0 punto 1](F0_IMPLEMENTATION_PLAN.md) |
| Encoder e collegamento | SigLIP2, pixel shuffle 2x2 + MLP, hook su embedding e value embeddings | Baseline verificata; encoder Qwen, proiettori alternativi, layer intermedi, DeepStack, rami MUDD/value mancanti | punto 5 |
| ViT addestrabile | Ramo autograd presente; `set_trainable` congela sempre l'encoder; conteggi di parametri fissi; checkpoint senza ViT | Mancante | punto 6 |
| Dati e gold | Parser, generatori, controlli deterministici, rietichettature, provenienza delle liste | Errore confermato nei sintetici v3 (`intersections`) e domande di classificazione ambigue; candidato v4 generato e controllato (S2 §2.6), con manifest e dev split nei loader (S2 §2.7), ancora incompleto per gli esperimenti (audit e decontaminazione esterna, S2 §2.8); gold tipizzato, valore esatto/visualizzato, answer-only mancanti | punto 2, [S2](specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md) |
| Split e manifest | Reader con esclusioni per chiave, validazione dal prefisso dell'ultimo shard, liste storiche; per i sintetici v4 manifest per record, split per gruppo e validazione dal dev nei loader ([S1 §9](specs/S1_RECORD_MANIFEST.md)) | Manifest per compito/variante, gruppi globali, altre sorgenti e contabilità del consumato per compito mancanti | punto 3 |
| Replay e più immagini | Loader testuale; processor con liste di immagini; `Sample` con una sola immagine | Caricatori multi-immagine e mix del replay mancanti | punti 3, 4 |
| Valutazione | Valutatore testuale ufficiale, trascrizione UniMER-Test (spe, cpe, sce, hwe), immagine vera/bianca/assente | Dati dei benchmark visivi presenti ma valutatore VL mancante; dati di documenti, pagine multiple e tabelle mancanti | punto 7 |
| F3 | Componenti SFT riutilizzabili | Fuori dal primo perimetro | — |
| F4 | Helper e verificatori parziali | Fuori dal primo perimetro | — |
| Interpolazioni | `export_decoder.py` (round trip di Violetto) | Fuori dal primo perimetro | — |

Non creare nuove ricette quantitative o dichiararle pronte soltanto perché i file storici sono stati archiviati. Mantenere
import e prove di provenienza necessari; correggere le dipendenze prima di eliminare codice.

## Prossimo lavoro

Le decisioni D1-D6 e le quattro specifiche brevi S1-S4 di [`F0_IMPLEMENTATION_PLAN.md`](F0_IMPLEMENTATION_PLAN.md), poi
gli interventi 1-7 in ordine. Training, piloti GPU, audit con agenti e download richiedono l'ok esplicito dell'utente.
