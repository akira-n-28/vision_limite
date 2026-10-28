# Revisione indipendente — Gold, pulizia e decontaminazione Limite-VL

**Data:** 27 ottobre 2026.  
**Documento esaminato:** `PIPELINE_ADDESTRAMENTO_LIMITE_VL.md`, versione **1.2**, 725 righe.  
**SHA-256:** `b49e62997077c151f8f4c44616938c3576ac0e4091f72c752f047a6c4cb8c42a`.  
**Codice esaminato:** `origin/main` al commit **`9d977b691190256d25fd0f31c0969fc24e1356ad`**, letto con `git show`, senza usare come riferimento il working tree modificato. I collegamenti al codice sotto sono fissati a questo commit.

La revisione riguarda §5 completo, soprattutto §§5.6–5.10, e la coerenza con §§9, 11, 13, 15.3, 16 e 17. Non assume corrette le conclusioni di precedenti autori o revisori, né usa i risultati dei vecchi audit come prova. Non è stato effettuato un audit dei dataset: nessun download, rendering di dataset, addestramento, job GPU o SLURM. Sono stati eseguiti soltanto piccoli esempi artificiali in memoria sulle funzioni dello snapshot, senza modificare il repository. Non sono state necessarie fonti online ulteriori.

## Verdetto

**Pronto come progetto metodologico, con integrazioni operative circoscritte; non pronto all'esecuzione e non certificante.** Non emerge un difetto concettuale bloccante nella nuova definizione del gold o nel protocollo di decontaminazione. Il documento dichiara correttamente di essere un progetto e di non avere già pulito o certificato alcun dataset (rr. 3–11).

La separazione fra risposta corretta, informazione disponibile nell'input e spiegazione valida è solida. Sono corretti anche il rifiuto del consenso come oracolo, l'ammissione di target controllati nella SFT, il recupero answer-only e la distinzione fra campione diagnostico e campione di stima. Queste scelte rendono realistica una prima versione economica, senza agenti per ogni esempio.

Restano sette rilievi. Alcuni sono **scarti verificati fra codice attuale e comportamento richiesto**, già in parte anticipati dal piano; altri sono **decisioni aperte o rischi condizionati**, non errori dimostrati dei dati. Le priorità indicano quando risolverli: **P1** prima del percorso interessato; **P2** prima di estenderne il perimetro o sostenere la relativa certificazione. Nessun rilievo giustifica un grande refactoring dell'audit.

## Rilievi prioritizzati

### 1. P1 — Il manifest deve identificare gli esempi effettivamente costruiti e governare tutti i percorsi

**Tipo:** lacuna esecutiva verificata, già riconosciuta dal protocollo; precisazione necessaria dell'interfaccia minima.

**Evidenza.** §§5.5 e 5.8 richiedono split per gruppo e consumo del manifest (rr. 180–182, 235–236); §5.10 li considera un requisito di avanzamento (rr. 259, 265). Oggi [`Source` e `SourceReader`](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/stream.py#L39) leggono glob di shard e liste di esclusione; la validazione resta il prefisso dell'ultimo shard (`stream.py`, rr. 158–166 e 265–273). Inoltre [`mavis_f1_sample`](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/data.py#L212) costruisce descrizione o QA durante il caricamento; `unimer_line_sample` sceglie e ritaglia una riga (rr. 244–270). Un ID dell'immagine da solo non identifica il claim e l'input effettivamente ammessi.

**Conseguenza.** Un'ammissione a livello immagine può includere una QA non verificata; un cambio del codice può modificare il target senza modificare la chiave. Il flusso può anche cambiare la popolazione effettiva dopo l'audit: `MixedRowStream` conta e salta errori di costruzione (`stream.py`, rr. 243–248), mentre il [collator scarta gli esempi troppo lunghi](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/collator.py#L68). Questo non prova contaminazione attuale, ma impedisce di equiparare automaticamente manifest e mix consumato.

**Correzione minima.** Definire una chiave composta sorgente/revisione/record/compito-variante, gruppo globale e split; associare hash del target e versione della trasformazione/preprocessing. Un piccolo indice di ammissione può filtrare i reader esistenti e scegliere esplicitamente la validazione. Le trasformazioni deterministiche possono restare al volo se sono fissate e controllate. Registrare gli scarti successivi all'ammissione per sorgente/compito e aggiornare copertura e denominatori quando sono materiali. Verificare lo stesso vincolo nel replay e nel futuro percorso RL, oltre alla SFT.

### 2. P1 per il percorso interessato — Adattare anche estrazione e selezione, non soltanto `answers_match`

**Tipo:** incompatibilità verificata con il nuovo contratto; non un'accusa al vecchio contratto F2.

**Evidenza.** §§5.7 e 5.10 ammettono answer-only e richiedono equivalenze, gestione delle ambiguità e stati distinti del verificatore (rr. 225, 262). Il codice [`grade`](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/f2.py#L183) richiede `</think>` prima di estrarre l'ultimo `boxed`, poi può dichiarare `usable` una soluzione senza verificarne i passaggi. Due esempi artificiali, eseguiti sullo snapshot:

| Output, con riferimento `2` e fine regolare | Esito attuale |
| --- | --- |
| `\boxed{2}` | `answer=None`, `correct=False`, `usable=False` |
| `<think>1+1=2.</think>\boxed{1}; \boxed{2}` | `answer=2`, `correct=True`, `usable=True` |

Il primo esito dipende dal formato; il secondo mostra che la selezione dell'ultimo box non risolve risposte finali incompatibili. Non è stata eseguita Math-Verify: questi esempi attraversano il confronto esatto del modulo.

**Conseguenza.** Modificare soltanto il matcher lascerebbe fuori risposte dirette corrette e potrebbe riutilizzare impropriamente `usable` come ammissione della spiegazione. Per RL, distinguere stati senza decidere il loro effetto sul reward lascia aperta la possibilità di penalizzare risposte corrette non riconosciute o di favorire fallimenti del parser.

**Correzione minima.** Aggiungere un adattatore per compito che separi estrazione, formato, correttezza finale e ammissione della spiegazione; applicare le regole sulle risposte incompatibili alla parte finale, senza scambiare tentativi esplicitamente abbandonati nel ragionamento per contraddizioni. Per RL fissare prima del rollout la politica: errore tecnico con retry limitato/esclusione controllata; risposta non conforme a un formato dichiarato distinta dall'incapacità semantica del verificatore. Un task non verificabile in modo affidabile resta fuori dal primo RL, ma può restare SFT.

**Limite importante.** [`geometry_scores`](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/f1_schema.py#L137) confronta valori normalizzati come stringhe: `1.5` contro `3/2` produce `wrong_value=1`. **Non è di per sé un difetto della trascrizione**: il modulo descrive ciò che è scritto nella figura. Diventerebbe un falso errore se riusato senza adattamento per una domanda che ammette equivalenza numerica. La distinzione della v1.2 su questo punto va conservata, non sostituita con equivalenza algebrica universale.

### 3. P1 per chiamare un record “gold” — Esplicitare la regola di promozione dalle prove di famiglia alle proprietà del record

**Tipo:** decisione operativa aperta; non circolarità dimostrata del documento.

**Evidenza.** La definizione del gold richiede controllo della semantica e dell'input per le proprietà richieste (§5.6, r. 200), mentre la tabella per sorgente ammette controlli sulle famiglie e audit campionari dei renderer (§5.7, r. 214). Le due cose sono compatibili, ma manca una regola esplicita che dica quali proprietà vengono garantite per record e quali rimangono stimate. Nel codice [`mavis_truth`](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/f1_mavis.py#L23) ricava fatti dopo il confronto fra caption e specifica; non riceve pixel. [`latex_value`](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/f1_schema.py#L35) applica anche una convenzione di rendering, trasformando `44/3` in `14.67`; il commento cita osservazioni storiche, che non convalidano nuove configurazioni.

**Conseguenza.** Un implementatore potrebbe promuovere automaticamente tutta una famiglia dopo poche immagini corrette. L'interpretazione opposta — verifica visiva individuale con agenti per ogni record — annullerebbe il vantaggio economico cercato.

**Correzione minima.** Per ciascun compito iniziale registrare una breve regola di ammissione: proprietà verificata automaticamente sul record; proprietà coperta da invarianti della famiglia e relative assunzioni; proprietà soltanto campionata; versione e perimetro della prova. Un controllo dei pixel soltanto campionario non diventa prova individuale universale. Se il claim richiede una proprietà ancora incerta, usare lo stato target controllato o recuperare le sole QA coperte. Per un answer-only la spiegazione è “non applicabile”, non “fallita”. Nessuna richiesta di una seconda implementazione completa o di un agente per ogni esempio.

### 4. P2, P1 se ammesso a reward — Lo stato esatto non elimina l'ambiguità della domanda: c'è già un caso concreto nei sintetici

**Tipo:** ambiguità verificata del codice riutilizzabile; la v1.2 contiene già il principio corretto per risolverla.

**Evidenza.** §5.6 richiede alternative realmente valide e semantica del compito (rr. 200–208). In [`_polygon_qa`](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/synth.py#L290), per un quadrato, il piccolo esempio artificiale restituisce:

- Domanda: `What type of quadrilateral is ABCD?`
- Opzioni: `square`, `rectangle`, `parallelogram`, `trapezoid`.
- Unico target: `square`.

Il commento dichiara la scelta del nome più specifico, ma la domanda non la chiede. Un quadrato è anche un rettangolo e un parallelogramma; non serve decidere la convenzione sui trapezi per mostrare l'ambiguità. La famiglia MAVIS genera analogamente domande aperte sul tipo di figura (`f1_mavis.py`, rr. 82–83).

**Conseguenza.** Un riferimento computato senza errori dallo stato può comunque rifiutare una risposta matematica corretta. Sarebbe particolarmente dannoso usarlo come ricompensa binaria.

**Correzione minima.** Per queste QA chiedere esplicitamente la classe più specifica fra quelle previste, con criterio di specificità definito, oppure ammettere l'insieme delle risposte corrette. Inserire questo caso fra i collaudi della famiglia insieme a unità, arrotondamenti, domini e soluzioni multiple. Non occorre scartare tutte le domande della scena né trasformare la questione in un filtro generico sui formalismi.

### 5. P2 — La migrazione deve rivalutare le vecchie esclusioni, altrimenti i falsi match restano irrevocabili nella pratica

**Tipo:** rischio di migrazione fondato su configurazione verificata; nessun falso positivo reale quantificato.

**Evidenza.** §5.8 tratta correttamente pHash e similarità come candidati e vieta fusioni deboli e scarti per sola formula comune (rr. 233–235). §5.10 identifica correttamente i limiti degli script (r. 261): [`split_duplicates.py`](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/scripts/split_duplicates.py#L99) unisce le condizioni immagine identica o target identico, per sorgente; [`normalize_words`](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/decontam.py#L39) perde gli operatori, così frammenti come `x+1` e `x-1` hanno gli stessi token. Quest'ultimo esempio descrive la perdita d'informazione, non un match effettivo di una domanda corta: l'indice richiede n-gram sufficientemente lunghi.

La [bozza F1-clean](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/config/runs/f1-clean.json#L12) importa già più liste storiche di esclusione, oltre a rietichettature. `make_sources` le unisce senza reinterpretarne le ragioni (`stream.py`, rr. 74–86).

**Conseguenza.** Un nuovo indicizzatore corretto, eseguito soltanto sui sopravvissuti delle vecchie liste, non recupererebbe i dati validi esclusi in passato. Viceversa, eliminare indiscriminatamente le liste riammetterebbe eventuali sovrapposizioni vere.

**Correzione minima.** Costruire il nuovo manifest sull'inventario originale fissato e importare le vecchie decisioni come evidenze con motivo e versione: confermate, da riesaminare o irrisolte. Rivalutare prioritariamente le esclusioni fondate su target comune o sola similarità, senza obbligare a riesaminare manualmente le copie esatte confermate. Conservare gruppi globali per problema/documento e propagare ogni esclusione riservata a tutte le sorgenti e fasi. È una procedura di migrazione, non un nuovo sistema di decontaminazione.

### 6. P2 per la stima; necessario prima di certificare — Definire quale tasso si misura, oltre all'unità di campionamento

**Tipo:** specifica statistica ancora aperta; i numeri pubblicati nel documento sono corretti.

**Evidenza.** §5.9 richiede probabilità note e unità per problema/documento (rr. 244–249); §15.3 distingue clustering, strati e confronti multipli (rr. 638–644). Non è però fissato se la soglia riguardi la quota di documenti con almeno un errore, la quota di QA errate o il peso degli errori nel mix di training. Sono quantità diverse quando un documento produce molte QA.

Per esempio, campionare uniformemente 100 documenti e una QA per documento stima una media per documento; non stima senza pesi opportuni la quota di tutte le QA. Né l'esame di una sola QA certifica l'assenza di errori nelle altre QA dello stesso documento.

**Conseguenza.** Anche un campione probabilistico può essere interpretato sulla popolazione sbagliata. Inoltre, eliminare dal denominatore i casi non giudicabili selezionerebbe il campione a posteriori e renderebbe troppo ottimistica una certificazione dei soli errori dimostrati.

**Correzione minima.** Prima dell'estrazione fissare proprietà, popolazione/versione, denominatore, unità primaria, selezione delle QA all'interno del gruppo e pesi. Dichiarare la gestione degli incerti: risoluzione indipendente, limite conservativo o certificazione esplicitamente parziale; non contarli implicitamente come corretti. I conteggi diagnostici sugli scarti restano diagnostici; per stimare un tasso generale di scarto erroneo serve un disegno probabilistico anche su quella popolazione.

**Verifica numerica.** Con zero errori, n prove Bernoulli indipendenti e classificazione corretta, il limite superiore unilaterale esatto al 95% è `1 − 0.05^(1/n)`: **0,9936% per n=300** e **0,09981% per n=3.000**. §§5.9 e 15.3 sono coerenti; non c'è un errore da correggere nella regola approssimata. Non si trasferisce automaticamente a campioni a cluster o stratificati.

### 7. P2 per la certificazione — Il riferimento indipendente è una condizione sostanziale, non soddisfatta dal solo 10% di doppia revisione

**Tipo:** rischio condizionato e decisione di calibrazione; il documento riconosce già esplicitamente l'assunzione.

**Evidenza.** §§5.4 e 5.9 propongono un secondo verificatore su dubbi/disaccordi e inizialmente sul 10% casuale delle approvazioni (rr. 170, 251), senza forzare il consenso. La validità della certificazione è correttamente subordinata all'accuratezza del riferimento (r. 249). Manca ancora il criterio operativo con cui giudicare adeguata tale accuratezza rispetto alla soglia scelta, soprattutto per obiettivi dello 0,1%.

**Conseguenza.** Due giudici possono condividere un errore, essere influenzati dalla stessa annotazione o non riconoscere una classe di difetti. L'assenza di disaccordi non misura da sola i falsi negativi. La regola dei 300/3.000 limita gli errori rilevati da un riferimento adeguato: non corregge automaticamente gli errori del riferimento stesso.

**Correzione minima.** Sul piccolo nucleo usato per collaudare i verificatori, scegliere prove matematiche o fonti realmente indipendenti per la proprietà rilevante; includere negativi corretti ed errori noti senza affidarsi soltanto a perturbazioni artificiali. Dove serve revisione competente, prevedere una prima valutazione non condizionata dal target candidato per una quota mirata e un'adjudication documentata dei conflitti. Se non si può sostenere l'accuratezza richiesta, dichiarare un risultato condizionato al metodo di revisione o un audit informativo, non una garanzia più forte. Il 10% resta un punto di partenza economico, non un requisito universale né una prova statistica di affidabilità.

Non occorre applicare questa procedura a ogni record SFT. Il nucleo di collaudo può essere development; eventuali esempi del test finale non devono diventare casi su cui adattare verificatore, regole di selezione o modello. È sufficiente esplicitare i ruoli dei piccoli insiemi già previsti, senza creare un'altra infrastruttura.

## Cosa conservare senza irrigidire il piano

- **Le tre proprietà separate** (§5.1). Un solver dimostra conseguenze delle sue premesse; caption, programma e pixel non sono prove automaticamente indipendenti. La v1.2 lo dice correttamente.
- **I tre stati di evidenza** (§5.6), inclusa una quota SFT di target controllati. Non serve pretendere una prova formale individuale per ogni caption o risposta documentale. Per il primo RL è ragionevole chiedere sia riferimento affidabile sia reward validato.
- **Equivalenze guidate dal compito.** Valore esatto, testo visualizzato e risposta richiesta sono distinti. Unità, dominio, arrotondamenti e alternative vanno conservati. La compilabilità LaTeX e il confronto con un rendering sono controlli utili, non oracoli della trascrizione.
- **Il recupero selettivo.** Una spiegazione dubbia non invalida necessariamente la risposta; una QA dubbia non invalida necessariamente l'immagine. Un dubbio non diventa un esempio “impossibile rispondere”.
- **La decontaminazione globale** (§5.8). Sono adeguati l'inclusione di replay/caption/aggregati, il raggruppamento dei derivati, il controllo dei falsi match e il limite dichiarato sull'esposizione pregressa dei modelli. Non è emersa una contraddizione fra split ufficiali e esclusione dei gruppi collegati ai riservati. La separazione delle famiglie generative in §5.2 va letta secondo la precisazione di §§5.5 e 5.8: template disgiunti nel test di generalizzazione, non divieto generale di usare la stessa regola matematica.
- **Controlli prima, audit mirati dopo.** §16 distingue correttamente CPU/storage, token degli agenti, generazione dei target e training. Non promette risparmi non misurati. Non è giustificato sostituire questo impianto con una pipeline multiagente per esempio.

## Cosa serve prima di implementare e prima di eseguire

Prima di implementare bastano quattro specifiche brevi: contratto del record/manifest e dei derivati; regole di ammissione per le prime famiglie; contratto dei verificatori per compito; disegno del primo audit con budget. Non occorre definire subito ogni dataset del catalogo. L'ordine sintetici, MAVIS supportato e formule è ragionevole, purché la selezione finale preservi la copertura documentale e percettiva prevista e non venga decisa dalla sola facilità del verificatore.

Il budget dell'audit deve indicare tetto di chiamate/token, costo della revisione competente, dimensione del campione probabilistico e regola per fermare l'escalation su una sorgente. Al raggiungimento del limite si restringe il claim, si mantiene target controllato quando ammissibile o si rinvia il sottoinsieme: non si forza l'accordo. Le soglie e le quote SFT ancora da scegliere sono **decisioni aperte riconosciute dal documento**, non difetti da colmare inventando numeri. La certificazione più severa può attendere; il primo esperimento richiede invece controlli informativi e limiti espliciti.

Prima del primo esperimento serve una versione realmente ammessa con split globali applicati dal loader, input/preprocessing fissati, evidenza per claim e campione informativo. Il collaudo minimo deve includere una copia trasversale fra sorgenti, un pHash simile con numero decisivo diverso, derivati nello stesso gruppo, una QA non ammessa su immagine ammessa, equivalenza numerica distinta dalla trascrizione e risposta diretta corretta. Sono prove mirate del contratto, non un nuovo audit esteso dei dataset.

Prima del primo RL servono inoltre politica del reward per esiti non determinati/tecnici e verificatori validati sulle famiglie ammesse. Prima di usare “certificato” servono estimando e soglia dichiarati, campione adeguato e attendibilità del riferimento sufficientemente sostenuta. La chiusura di questi punti rende il progetto eseguibile nel perimetro scelto; il solo testo della v1.2 non dimostra che lo siano già dati, codice o configurazioni.

## Esito della verifica del rapporto

I riferimenti alle righe riguardano esclusivamente la v1.2 identificata dall'hash in apertura e lo snapshot Git dichiarato. I risultati numerici riportati provengono da esempi artificiali e dal calcolo del limite binomiale; non sono stime di qualità o prevalenza nei dataset. Il working tree e il documento della pipeline non sono stati modificati. Questo rapporto è l'unico artefatto scritto dalla revisione.
