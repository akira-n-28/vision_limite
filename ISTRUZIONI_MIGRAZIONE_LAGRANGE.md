# Limite-VL — incarico di migrazione per Claude Code su Lagrange

Data: 28 ottobre 2026. Documento da consegnare insieme a `PIPELINE_ADDESTRAMENTO_LIMITE_VL.md`, versione 1.4. Il testo seguente è l'incarico operativo iniziale; le informazioni sul cluster devono essere accertate sul posto.

**Consegna tramite GitHub:** questi documenti sono aggiunti alla repository senza applicare il riordino del codice o sostituire `README.md` e `CLAUDE.md`. Dopo aver aggiornato il checkout in modo compatibile con il lavoro locale, usa questo file come incarico e la pipeline allegata come riferimento. Le modifiche operative al progetto restano da eseguire su Lagrange.

**Divisione del lavoro confermata dall'utente:** Claude Code esegue tutta la migrazione su Lagrange; il pacchetto preparato sul portatile fornisce pipeline, revisioni e istruzioni. L'utente riferisce che la repository sul cluster è aggiornata, contiene artefatti esclusi da Git e non ha job attivi. Verifica brevemente che lo stato sia ancora questo; non occorre chiedere di nuovo all'utente le stesse informazioni. Branch esatto, eventuali modifiche locali e dipendenze degli artefatti si accertano sul posto.

## Obiettivo e perimetro

La nuova pipeline sostituisce la precedente strategia di addestramento e pulizia dati di Limite-VL. Integra il documento nel progetto e rendi coerenti repository, istruzioni e percorsi operativi, tenendo conto della **cartella completa su Lagrange**, incluse le parti escluse da Git.

La prima consegna è una migrazione reversibile del progetto, un inventario degli artefatti e un elenco concreto dei requisiti ancora da implementare. Non è un incarico a lanciare training, piloti GPU, nuove generazioni teacher, audit con agenti o download. Non cancellare dataset, checkpoint o ambienti e non interrompere job esistenti. Se servono scansioni pesanti, prepara il lavoro e le risorse richieste, senza eseguirlo in questa fase.

Procedi autonomamente con letture, aggiornamenti documentali e riordino reversibile dei piccoli file di progetto per i quali hai accertato dipendenze e stato. Non chiedere conferma per ogni rinomina ordinaria. Segnala invece decisioni scientifiche aperte, informazioni mancanti o interferenze concrete con attività in corso.

Il riordino non dimostra che i dati siano puliti, che i riferimenti siano gold o che il codice implementi tutte le fasi della nuova pipeline. Mantieni distinti progetto, implementazione, collaudo e risultati.

## 1. Accerta lo stato reale prima di applicare cambiamenti

Leggi le istruzioni locali e `LAGRANGE.md`, rispettando i vincoli effettivi del frontend, dello storage e delle allocazioni. Le istruzioni della nuova pipeline sostituiscono le vecchie scelte scientifiche; i limiti operativi del cluster restano applicabili. Non avviare automaticamente i vecchi comandi suggeriti dai documenti da migrare.

Rileva e registra, senza segreti:

- Directory del progetto, host, branch, HEAD, stato Git e differenze rispetto al remoto noto.
- Modifiche non committate e file non tracciati, distinguendo il lavoro corrente dagli artefatti generati.
- Job propri attivi o pendenti, eventuali download/processi del progetto e loro directory di lavoro/configurazioni quando disponibili.
- Spazio disponibile e organizzazione delle directory principali; usa metadati e manifest esistenti prima di scansioni ricorsive.

Preserva il lavoro corrente. Non usare `reset --hard`, `git clean`, checkout forzati, stash automatici non concordati o copie che sovrascrivano file diversi. Un branch da solo non salva i file non committati: prima del riordino conserva una copia recuperabile delle modifiche e dei piccoli file che saranno toccati, escludendo segreti e dati pesanti. Registra base e metodo di recupero.

Se un job può ancora leggere un file o un percorso, rimanda il suo spostamento e marca la dipendenza. Aggiornare codice nella directory di un job attivo può influire anche su import successivi: non dedurre che sia sicuro solo perché il processo è già partito.

L'analisi precedente ha usato il commit `9d977b691190256d25fd0f31c0969fc24e1356ad`. Questo identifica l'evidenza storica, **non impone di riportare il cluster a quel commit**. Il codice e le modifiche più recenti sul cluster vanno confrontati e conservati.

## 2. Introduci un unico riferimento corrente

Integra in repository `PIPELINE_ADDESTRAMENTO_LIMITE_VL.md` v1.4, verificandone versione e contenuto. Conserva i rapporti indipendenti con la versione/hash cui si riferiscono; non riscriverli come se approvassero codice non ancora implementato.

Aggiorna `README.md` e `CLAUDE.md` affinché:

- rimandino alla pipeline come riferimento corrente;
- distinguano lavoro locale, frontend e job di calcolo;
- indichino il prossimo lavoro preparatorio F0 e lo stato reale del codice;
- eliminino vecchi piani esterni, ricette automatiche, budget/soglie non più approvati e inviti a riprendere run storici;
- trattino scelta Qwen, proiettore, layer intermedi, DeepStack, MUDD/value embeddings, precisione, mix e budget come decisioni da chiudere con il protocollo previsto;
- mantengano le restrizioni operative pertinenti del cluster e la protezione dei segreti.

Esiste sul portatile una bozza di pulizia Git separata, non pubblicata: non è il risultato della migrazione su Lagrange e non va applicata automaticamente. La divisione del lavoro scelta lascia a te l'applicazione delle modifiche sulla cartella reale. Non sovrapporre integralmente una copia del portatile alla directory di lavoro sul cluster. Il pacchetto di consegna contiene documenti, non file di codice/configurazione da sostituire in blocco.

## 3. Riordina la parte versionata preservando dipendenze e provenienza

Classifica ogni elemento coinvolto come corrente/riutilizzabile, storico, da adeguare oppure irrisolto. Per spostamenti e sostituzioni conserva percorso precedente, nuovo percorso, motivo, origine e impronta dei piccoli file quando appropriato.

Nello snapshot analizzato tutte le 13 configurazioni di `config/runs/` appartenevano alla precedente strategia, inclusa `f1-clean.json`. Anche 28 launcher di training, F1c, ponte teacher, audit/gate e vecchio replay risultavano storici. Questi conteggi sono **candidati da riconciliare con la repository reale**, non una regola per spostare qualunque file con un certo prefisso.

Azioni previste:

- Archiviare le ricette superate fuori dal percorso operativo corrente, conservandone contenuto e origine. Nessun vecchio JSON diventa una ricetta della nuova pipeline cambiando soltanto nome.
- Archiviare o ritirare i launcher associati e correggere i rimandi operativi. Mantenere gli strumenti generici utili per ispezione, preprocessing, controlli ed eval.
- Conservare ledger e rapporti come storia degli esperimenti. Non usare i vecchi esiti come gold né come prova che le nuove scelte siano già validate.
- Aggiornare il catalogo download distinguendo disponibilità, candidatura e ammissione dei dati. Preservare identificativi, revisioni e filtri salvo correzioni motivate; non avviare nuovi trasferimenti.
- Conservare lock e configurazioni di ambiente come artefatti riproducibili, indicando cosa deve essere ricollaudato per i nuovi candidati.

Non eliminare in blocco moduli `f1c`, `f2`, `facts` o `stem_audit`: parser, loader, verificatori e test possono importarli. `scripts/train_f1.py` contiene helper usati da altri strumenti; il vecchio trainer F2 conserva dipendenze dal gate storico. Mantenere un modulo importabile non significa adottarne il vecchio flusso come strategia corrente.

I prompt e le policy in `config/audit/` possono avere percorsi e hash incorporati nei manifest storici. Conservane l'identità dove serve e dichiarane lo stato con un indice. Non modificare retroattivamente soglie o prompt per far apparire certificati vecchi dati.

Non aggiungere configurazioni fittizie che sembrino pronte al training. I nuovi manifest, trainer e contratti devono essere implementati e collaudati prima di dichiarare un run eseguibile.

## 4. Inventaria la cartella completa senza spostare alla cieca gli artefatti pesanti

Esamina i percorsi effettivi definiti dal progetto, non soltanto quelli in Git. Copri almeno dataset originali/shard/derivati, esclusioni e rietichettature, checkpoint, generazioni teacher/replay, report, cache e snapshot Hugging Face, ambienti, log e file temporanei rilevanti.

Per ciascun artefatto o gruppo omogeneo registra:

| Campo | Contenuto |
| --- | --- |
| Identità | Percorso, tipo, origine/revisione o run quando disponibili |
| Stato | In uso, riutilizzabile, storico, rigenerabile oppure da chiarire |
| Dipendenze | Job, configurazioni, manifest, symlink, script e altri artefatti che lo referenziano |
| Dimensione | Stima da metadati disponibili, oppure non ancora misurata |
| Qualità/provenienza | Evidenza disponibile, controlli eseguiti e limiti; nessuna promozione automatica a gold |
| Azione proposta | Conservare, escludere dal percorso corrente, riesaminare, archiviare o proporre una futura eliminazione |
| Verifica necessaria | Controllo leggero immediato o scansione da pianificare in allocazione |

In questa prima migrazione l'archiviazione dei grandi artefatti è soprattutto **logica**, tramite inventario e riferimenti correnti: evita copie o spostamenti di terabyte soltanto per cambiare struttura. Un dataset scaricato può essere utile anche se il run che lo usava è stato abbandonato.

Particolare attenzione a snapshot e blob condivisi della cache HF, link `venvs/current`, checkpoint referenziati da config e output di job attivi. L'età, un nome di fase o l'assenza da Git non dimostrano che un artefatto sia eliminabile. Eventuali proposte di cancellazione devono indicare motivo, dimensione, dipendenze e recuperabilità; non eseguirle in questo incarico.

Non hashare o caricare l'intero corpus sul frontend per completare l'inventario. Annota l'informazione mancante e il controllo CPU necessario secondo `LAGRANGE.md`.

## 5. Traduci la pipeline in un elenco di interventi verificabili

Confronta il codice reale, incluse le modifiche non pubblicate, con i requisiti della pipeline. Classifica ciascuno come già presente e verificato, presente da collaudare, mancante oppure decisione aperta. Cita file e controlli, senza assumere che una descrizione nel README equivalga a un'implementazione.

Il primo perimetro F0 deve coprire:

1. Contratto di risposta diretta/ragionata, template, label e stop coerenti.
2. Gold per compito/variante, equivalenze e osservabilità; ambiguità dei sintetici; recupero answer-only.
3. Split globali, riesame delle esclusioni storiche, manifest applicato ai loader e contabilità degli esempi realmente consumati.
4. Caricatori con più immagini, preprocessing e ordine delle pagine, senza confondere packing e contesto di un singolo problema.
5. Encoder candidati, confine ViT–merger e protocollo limitato per proiettore, layer e punti di iniezione.
6. Gruppi di ottimizzazione e checkpoint completi per il ViT addestrabile, con collaudo di salvataggio/ripresa.
7. Pannelli di sviluppo/valutazione e preventivo dei piloti.

F4, interpolazioni e certificazione statistica generale non devono diventare prerequisiti per preparare il nucleo F0. Le nuove configurazioni quantitative vengono dopo inventario e specifiche; budget GPU/euro e checkpoint Qwen esatto non sono ancora concordati.

## 6. Verifica e consegna

Esegui soltanto controlli compatibili con questa fase e con l'host: stato/diff Git, integrità dei piccoli file archiviati, link correnti, riferimenti dei launcher rimasti, sintassi e controlli statici leggeri quando consentiti. Non eseguire script di training per provarne l'import; alcune importazioni caricano lo stack numerico o hanno dipendenze pesanti.

I test da eseguire in allocazione vanno elencati con scopo e risorse; non dichiararli superati se non eseguiti. Non modificare test storici soltanto per mascherare dipendenze rotte.

Consegna:

- Repository con riferimento corrente unico e storia distinta dai percorsi attivi.
- `docs/MIGRATION_REPORT.md`: base Git, stato preservato, modifiche, spostamenti, controlli effettuati e limiti.
- `docs/ARTIFACT_INVENTORY.tsv`: inventario degli artefatti esterni a Git, senza segreti e senza contenuti dei dataset.
- `docs/F0_IMPLEMENTATION_PLAN.md`: lista ordinata di interventi minimi, evidenza dello stato, dipendenze, collaudi e decisioni mancanti.
- Elenco separato di job/processi in corso e percorsi che non sono stati toccati per non interferire.

Puoi organizzare le modifiche in commit locali piccoli e revisionabili se coerente con lo stato del progetto e le istruzioni correnti. Non pubblicare o fare merge sul remoto senza un incarico che lo comprenda.

Concludi dichiarando esattamente che cosa è pronto: migrazione documentale, codice, dati o configurazione di un esperimento. Il risultato atteso di questo incarico è la migrazione e un piano F0 concreto; nessun training o pulizia massiva viene considerato implicitamente autorizzato.

## Messaggio breve da accompagnare ai file

> La nuova pipeline allegata sostituisce quella precedente. Leggi `PIPELINE_ADDESTRAMENTO_LIMITE_VL.md` e `ISTRUZIONI_MIGRAZIONE_LAGRANGE.md`, poi esegui la migrazione descritta sulla cartella completa del progetto. Accerta prima branch, modifiche locali e job attivi; integra il lavoro non ancora pubblicato. Riordina repository e istruzioni in modo reversibile, inventaria anche gli artefatti fuori Git e prepara il piano concreto di implementazione F0. Preserva dati, checkpoint, ambienti e job esistenti. In questa fase non avviare training, GPU, nuovi audit con agenti, download o scansioni pesanti. Riporta modifiche, verifiche effettuate, stato degli artefatti e decisioni mancanti.
