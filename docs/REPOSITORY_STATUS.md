# Stato da verificare su Lagrange

Preparato il 28 ottobre 2026 per la migrazione alla [pipeline v1.4](../PIPELINE_ADDESTRAMENTO_LIMITE_VL.md).

**Questo documento non dichiara eseguita la migrazione sul cluster.** L'analisi del codice ha usato lo snapshot `9d977b691190256d25fd0f31c0969fc24e1356ad`; Claude deve riconciliare queste informazioni con il checkout reale, le modifiche locali e gli artefatti fuori Git. L'utente riferisce repository aggiornata e assenza di job attivi.

## Base disponibile

La pipeline recepisce progettualmente i rilievi delle due revisioni. La [matrice dei rilievi](REVIEW_ACTIONS.md) distingue requisiti e verifiche residue. I rapporti originali restano riferiti alle versioni esaminate; nessuno di questi documenti prova che i requisiti siano già implementati.

Nel riordino locale preparatorio sono stati individuati 13 JSON di run storici e 28 launcher legati alla vecchia sequenza, inclusa la bozza F1-clean. Si tratta di candidati da verificare sul cluster. Il contenuto di `data/`, `hf/`, `outputs/`, `venvs/` e delle altre directory ignorate da Git non è stato inventariato da questa consegna.

## Verifica dell'implementazione

| Area | Componenti individuati nello snapshot | Requisito da verificare sul cluster |
| --- | --- | --- |
| Formato delle risposte | Template Violetto ed estrattori storici | Contratto diretto/ragionato, label, stop e verificatori per compito |
| Encoder e collegamento | SigLIP2, pixel shuffle/MLP e hook visivi | Scelta Qwen, estrazione/preprocessing e confronto di proiettori, layer e ingressi |
| ViT addestrabile | Ramo autograd e trainer di riferimento | Gruppi distinti, precisione, gradienti, salvataggio e ripresa completi |
| Dati e gold | Parser, generatori, controlli e provenienza | Regole per QA, ambiguità, osservabilità e evidenza indipendente |
| Split e manifest | Reader, esclusioni e controlli di overlap | Gruppi trasversali, varianti/crop ammessi e contabilità dei dati consumati |
| Replay e più immagini | Loader testuale e liste nel processor | Mix effettivo, lunghezze, loader multi-immagine e contesto validato |
| F3 | Componenti SFT riutilizzabili | Curriculum, copertura e supervisione coerente |
| F4 | Helper e verificatori parziali | Trainer multimodale, rollout/log-probabilità, riferimento e GRPO definiti |
| Interpolazioni | Componenti dei checkpoint | Tensori compatibili, pesi legati e valutazioni testo/visione/comportamento |

Non creare nuove ricette quantitative o dichiararle pronte soltanto perché i file storici sono stati archiviati. Mantenere import e prove di provenienza necessari; correggere le dipendenze prima di eliminare codice.

## Prima consegna richiesta

Eseguire la [migrazione](../ISTRUZIONI_MIGRAZIONE_LAGRANGE.md): rendere corrente la pipeline, ritirare i percorsi superati, inventariare gli artefatti e preparare gli interventi F0. Non servono ancora implementazione completa del RL o certificazione generale dei dati.

Claude aggiornerà lo stato con evidenze reali e produrrà `docs/MIGRATION_REPORT.md`, `docs/ARTIFACT_INVENTORY.tsv` e `docs/F0_IMPLEMENTATION_PLAN.md`. Distinguere controlli effettivamente eseguiti, controlli da pianificare in allocazione e decisioni ancora aperte. La pulizia massiva degli artefatti e il training non fanno parte di questa prima consegna.
