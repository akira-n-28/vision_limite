# Limite-VL — Pipeline completa di addestramento

**Versione:** 1.5 — 30 ottobre 2026

**Stato:** progetto di addestramento; non è una configurazione esecutiva né un'autorizzazione a lanciare job.

**Repository esaminata:** [akira-n-28/vision_limite](https://github.com/akira-n-28/vision_limite), main al commit [9d977b691190256d25fd0f31c0969fc24e1356ad](https://github.com/akira-n-28/vision_limite/commit/9d977b691190256d25fd0f31c0969fc24e1356ad), del 26 ottobre 2026.

Questo documento sostituisce, per la proposta futura, la sequenza descritta nelle configurazioni storiche. Non assume che i risultati dei vecchi run dimostrino la qualità di una scelta. Le informazioni del repository servono a identificare codice, dati e strumenti riutilizzabili.

**Aggiornamento 1.1:** F0 include esplicitamente la scelta del proiettore, dei layer del ViT e dell'integrazione tramite DeepStack, MUDD e value embeddings. Il [rapporto indipendente](./docs/reviews/REVISIONE_INDIPENDENTE_PIPELINE_LIMITE_VL.md) riguarda la versione 1.0 identificata dal suo hash: questa integrazione non equivale alla risoluzione degli altri rilievi del rapporto.

**Aggiornamento 1.2:** §§5.6–5.10 definiscono costruzione dei riferimenti gold, pulizia per sorgente, decontaminazione globale e criteri di ammissione. Si tratta del protocollo da implementare: nessun dataset è dichiarato pulito o certificato da questo documento.

**Aggiornamento 1.3:** recepiti nel progetto i sette rilievi della [revisione indipendente su gold e decontaminazione](./docs/reviews/REVISIONE_INDIPENDENTE_GOLD_E_DECONTAMINAZIONE.md): identità delle QA e applicazione del manifest, estrazione delle risposte, ammissione al gold, ambiguità dei sintetici, recupero delle vecchie esclusioni, quantità statistiche da stimare e attendibilità dei riferimenti. Il rapporto resta riferito alla v1.2 e al relativo hash; codice, dataset e correttori devono ancora essere adeguati e collaudati.

**Aggiornamento 1.4:** completata la risposta progettuale ai rilievi R1–R14 della prima revisione: template, contesto, estrazione Qwen, aggiornamento/salvataggio del ViT, replay, nucleo dati, confronto encoder, integrazione RL e interpolazioni. La pipeline è ora il riferimento nella repository; [stato e tracciabilità](./docs/REPOSITORY_STATUS.md) distinguono requisiti recepiti, decisioni aperte e implementazione da eseguire. I rapporti originali restano immutati e non costituiscono una nuova approvazione della v1.4.

**Aggiornamento 1.5 (2026-10-30, decisione dell'utente):** perimetro definitivo della prima versione. Comprende matematica/STEM, geometria e formule stampate; grafici, tabelle, OCR e documenti a pagina singola; visione generale di supporto; un perimetro limitato di compiti con poche immagini, legati alla matematica e al confronto visivo; inglese come lingua principale. Rimanda i documenti a più pagine e il ragionamento fra pagine, il training dedicato al manoscritto (MathWriting) e i video. Il manoscritto non è un requisito di F0, una quota del mix né un criterio di avanzamento; gli esempi manoscritti incidentali nelle raccolte miste restano, senza classificazioni costose per toglierli. Delle raccolte documentali miste si usano solo esempi autonomi a pagina singola, senza troncare domande che richiedono altre pagine. Pannello e campione candidato di F0: [`docs/PANNELLO_VALUTAZIONE.md`](./docs/PANNELLO_VALUTAZIONE.md), [`docs/F0_CAMPIONE_CANDIDATO.md`](./docs/F0_CAMPIONE_CANDIDATO.md).

## 1. Obiettivo e decisioni

Costruire un modello basato su Limite 1B — Violetto che sappia leggere immagini, documenti a pagina singola e formule stampate, collegare poche immagini e risolvere problemi matematici e scientifici. La visione generale serve a sostenere queste capacità (perimetro della v1.5).

| Decisione | Stato |
| --- | --- |
| Matematica/STEM, geometria, formule stampate; grafici, tabelle, OCR e documenti a pagina singola; visione generale di supporto | Confermata dall'utente (perimetro definitivo del 2026-10-30) |
| Immagini singole e poche immagini per compiti di matematica e confronto visivo | Confermata dall'utente (2026-10-30) |
| Documenti a più pagine e ragionamento fra pagine | Rimandati (2026-10-30) |
| Training dedicato al manoscritto (MathWriting) | Rimandato (2026-10-30); il manoscritto non è un criterio di F0 |
| Video | Esclusi dalla prima versione |
| Lingua principale | Inglese |
| Risposta diretta per compiti semplici; spiegazione proporzionata per problemi complessi | Confermata dall'utente |
| Rifare l'addestramento multimodale da F1, senza ereditare i vecchi adapter | Confermata dall'utente |
| Confrontare SigLIP2 SO400M e un ViT estratto da Qwen, poi addestrare il vincitore | Confermata dall'utente |
| Valutare più layer del ViT, proiettore, DeepStack e integrazione con MUDD/value embeddings | Richiesto dall'utente; valutazione esplicita in F0, adozione da decidere |
| Qualità dei dati con il minimo costo ragionevole in token di agenti | Priorità confermata |
| RL con ricompense verificabili | Parte del piano proposto |
| Interpolazione dei pesi con Violetto e con il checkpoint precedente al RL | Esperimenti proposti; adozione condizionata alle valutazioni |
| Checkpoint Qwen esatto | Da scegliere |
| Budget di training in ore GPU o euro | Non ancora concordato |

Ripartire da F1 significa riutilizzare i checkpoint preaddestrati del modello linguistico e dell'encoder, inizializzando di nuovo il collegamento multimodale. Non significa preaddestrare da zero il modello linguistico o il ViT.

Audio, video, controllo di interfacce, navigazione autonoma e uso generalizzato di strumenti non rientrano nella prima versione. Le competenze visive tridimensionali possono essere introdotte tramite immagini statiche, senza promettere una copertura generale della robotica.

## 2. Sequenza e corrispondenza con il progetto

**Dati e protocollo → piloti di encoder, proiettore e integrazione visiva → F1 → F2 → F3 → selezione/interpolazione → F4 → selezione/interpolazione → valutazione finale.**

| Fase | Obiettivo | Parametri addestrabili | Risultato |
| --- | --- | --- | --- |
| F0 | Dati, controlli, confronto di encoder/proiettore/integrazione e costi | Nei soli piloti | Dataset ammessi, architettura scelta, protocollo e budget |
| F1 | Allineamento percettivo | Proiettore/merger e nuovi parametri dell'interfaccia visiva | Collegamento visione-linguaggio utilizzabile |
| F2 | Adattamento multimodale ampio | ViT, proiettore e decoder | Capacità multimodali, documentali e su immagini multiple |
| F3 | Specializzazione matematica/scientifica e comportamento | Decoder e proiettore; ViT inizialmente congelato | Checkpoint supervisionato completo |
| F3-M, opzionale | Interpolazione con Violetto originale | Nessun training | Possibile miglioramento del compromesso testo/visione |
| F4 | RL verificabile | Decoder e proiettore; ViT congelato nella ricetta iniziale | Possibile miglioramento del ragionamento verificabile |
| F4-M, opzionale | Interpolazione con il checkpoint da cui è iniziato il RL | Nessun training | Possibile recupero delle capacità attenuate dal RL |
| Valutazione finale | Selezione finale e descrizione dei limiti | Nessuno | Artefatto riproducibile e rapporto delle prestazioni |

Questa è una nuova nomenclatura operativa: il vecchio f1b.json continua prevalentemente l'allineamento, mentre la nuova F2 è un adattamento congiunto ampio. Manteniamo F4 come nome del RL, coerentemente con il catalogo del progetto.

La ricetta si ispira alla separazione fra allineamento, adattamento multimodale e post-training di sistemi come [Qwen3-VL](https://arxiv.org/html/2511.21631v1#S3.SS1). Volumi, iperparametri e beneficio atteso per Limite devono essere misurati; non vengono trasferiti automaticamente da modelli più grandi.

## 3. Punto di partenza architetturale

### 3.1 Componenti

- **Decoder:** checkpoint originale paradigma-inc/limite-1b-violetto, con revisione fissata. Il riferimento del repository è b1f3d572ccacb6919f4d64c321b70ba034ddaef2.
- **Encoder A:** SigLIP2 SO400M NaFlex, con revisione fissata.
- **Encoder B:** ViT estratto dal checkpoint Qwen da scegliere, con estrazione, revisione e preprocessing documentati.
- **Collegamento:** compressione spaziale e proiezione verso i 1280 canali del decoder Limite.

Il proiettore attuale usa pixel shuffle 2×2 e MLP 4608 → 2560 → 1280 per SigLIP2. È una baseline implementata, non una scelta definitiva già approvata per entrambi gli encoder. Cambiando ViT possono cambiare la dimensione in ingresso e il preprocessing; il numero di parametri e il costo del proiettore vanno riportati.

Il confronto iniziale usa l'uscita finale del ViT e un collegamento comparabile. Questo è solo il riferimento iniziale: la scelta definitiva comprende layer singolo o più layer del ViT, proiettore, DeepStack e canali visivi verso MUDD/value embeddings, secondo §§6.3–6.5. Prima si isolano gli effetti, poi si verifica una combinazione motivata; non si presume che i contributi siano indipendenti. L'architettura viene fissata prima del training principale.

Nella baseline attuale le feature visive entrano negli embedding iniziali prima della normalizzazione, raggiungendo anche il tap iniziale di MUDD. Il contributo dei value embeddings nelle posizioni immagine è azzerato: **non vengono azzerate le feature dell'immagine**. Questa scelta resta una baseline da confrontare, non una conclusione sulla migliore integrazione possibile.

### 3.2 Controlli prima di addestrare

- Correttezza di normalizzazione, griglie, positional embeddings e compressione per ciascun encoder.
- Corrispondenza fra ordine delle immagini, segnaposti nel testo e token visivi.
- Gradienti presenti nei moduli dichiarati addestrabili e assenti in quelli congelati.
- Coerenza fra percorsi di training e generazione.
- Precisione numerica verificata nuovamente contro un riferimento appropriato.
- Comportamento su testo puro del modello iniziale confrontato con Violetto originale.
- Separazione dell'attenzione fra esempi indipendenti impacchettati.
- Checkpoint completi: includere il ViT quando viene aggiornato.

Il processor contiene già supporto per liste di immagini, mentre i caricatori correnti rappresentano prevalentemente un'immagine per esempio. Il supporto a immagini multiple va completato e verificato lungo tutto il percorso, non riscritto indiscriminatamente.

Fonti implementative: [modello](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/modeling.py), [processor](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/processing.py), [caricatori](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/data.py).

### 3.3 Contratto di risposta e template

Il template dello snapshot Violetto impone ragionamento passo passo e risposta in `\boxed{}`, rifiutando system prompt diversi; il processor lo riusa. Questo è incompatibile con l'obiettivo di scegliere risposte dirette per i compiti semplici senza un adattamento esplicito. Prima del primo aggiornamento del decoder versionare template, messaggi, delimitatori, label supervisionate, stop token e modalità diretta/ragionata, mantenendo coerenti training, inferenza ed estrattore (§5.10). [Template originale fissato](https://huggingface.co/paradigma-inc/limite-1b-violetto/blob/b1f3d572ccacb6919f4d64c321b70ba034ddaef2/chat_template.jinja).

Il prefill `<think></think>` può essere uno strumento dichiarato di F1; non prova che il modello scelga autonomamente quando essere breve. Valutare la modalità destinata all'uso finale senza prefill nascosti e conservare una baseline di Violetto col suo template canonico. Violetto è già post-addestrato e orientato al ragionamento matematico a turno singolo: istruzioni generali, risposta proporzionata e dialogo vanno misurati e insegnati, non presunti. Il dialogo esteso può attendere; il contratto di risposta serve già nei piloti con decoder addestrabile.

### 3.4 Contesto e rappresentazione di più immagini

Distinguere i limiti dello snapshot: 131.072 posizioni nominali, 48 layer di cui 12 globali e 36 locali con finestra configurata a 1024; il processor impone separatamente un tetto software di 768 token per immagine. La finestra locale non è il contesto totale e le posizioni nominali non dimostrano capacità multimodale a quella lunghezza. [Configurazione del decoder](https://huggingface.co/paradigma-inc/limite-1b-violetto/blob/b1f3d572ccacb6919f4d64c321b70ba034ddaef2/config.json).

Definire ordine e identità delle immagini, token per immagine e totali, lunghezza della domanda e spazio per la risposta. Il processor accetta liste ma non testo/immagini alternati arbitrariamente nello stesso turno; il caricatore corrente conserva una sola immagine per `Sample`. Il primo percorso può usare immagini all'inizio del turno e riferimenti espliciti, senza implementare subito l'interleaving generale. Nella prima versione gli esempi con più immagini sono pochi e brevi (matematica e confronto visivo, di solito 2-4 immagini): non azzerare le posizioni fra le immagini dello stesso esempio, e testare uso della prima e dell'ultima immagine, ordine scambiato e dettaglio piccolo. Le sequenze di pagine e le dipendenze documentali lunghe sono rimandate (v1.5). Ogni cambio del tetto 768 o della risoluzione richiede una prova mirata di leggibilità, costo e uso dell'informazione.

### 3.5 Artefatto Qwen da confrontare

Prima del confronto fissare famiglia, ID e revisione del checkpoint Qwen, chiavi estratte, preprocessing e tensore scelto prima del nuovo collegamento. Dichiarare se il merger originale viene scartato o riutilizzato ed evitare doppia compressione. Validare ordine delle patch, griglie, normalizzazione e feature contro il percorso nativo dello stesso checkpoint. Il loader attuale è specifico di SigLIP2: cambiare il solo percorso dei pesi non implementa un encoder Qwen. Confrontare la sola uscita finale di un sistema che usa anche feature intermedie è un riferimento iniziale dichiarato, non una riproduzione del sistema originale.

## 4. Catalogo dei dati e ruolo di ciascuna sorgente

“Da integrare” significa che il percorso dedicato non è presente nelle configurazioni esaminate. Non è una dichiarazione sullo stato dei download locali o sul cluster. Ogni sorgente nuova rimane candidata fino ai controlli di F0.

| Sorgente | Stato nel progetto esaminato | Impiego proposto | Limite da rispettare |
| --- | --- | --- | --- |
| [MAVIS-Geometry](https://huggingface.co/datasets/CaraJ/MAVIS-Geometry) | Integrato; target canonici nella bozza F1-clean | F1–F2; problemi selezionati in F3/F4 | Coerenza delle specifiche non prova leggibilità o fedeltà del rendering |
| [UniMER](https://github.com/opendatalab/UniMERNet) | Integrato, con controlli e rietichettature | F1–F2; mantenimento percettivo in F3 | Separare trascrizione, equivalenza di notazione ed equivalenza matematica |
| Sintetici del progetto | Generatore e caricatore esistenti | F1–F4 secondo il compito | Lo stato esatto deve corrispondere all'immagine e alle informazioni disponibili |
| [MathWriting](https://github.com/google-research/google-research/blob/master/mathwriting/README.md) | Rimandato (v1.5) | Manoscritto: fuori dalla prima versione | I tratti richiedono rasterizzazione; mantenere gli split e le identità disponibili |
| [HierText](https://github.com/google-research-datasets/hiertext) | Da integrare | OCR e localizzazione in F1–F2 | Non inventare un ordine di lettura completo dalle sole regioni annotate |
| [PixMo-Cap](https://docs.allenai.org/training_data/pixmo) | Presente nei mix precedenti; fuori dalla bozza F1-clean | Scene reali in F1–F2; mantenimento in F3 | Caption e trascrizioni dell'annotatore possono differire; occorre controllo visivo campionario |
| [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | Presente nei mix precedenti | Complemento eventuale per F1 | Ammesso se aggiunge copertura utile; non obbligatorio |
| PixMo-Points / PixMo-AskModelAnything | Da integrare | Grounding e domande varie in F2–F3 | Distinguere annotazioni di punti, risposte approvate e descrizioni generate |
| [MAVIS-Function](https://huggingface.co/datasets/CaraJ/MAVIS-Function) | Nel catalogo; non nel mix F1-clean | Funzioni in F2–F3; F4 dove verificabile | Non trattarlo come un dataset di caption già pronto |
| [ChartQA](https://github.com/vis-nlp/ChartQA), [PlotQA](https://github.com/NiteshMethani/PlotQA) | Nessun percorso dedicato nel mix esaminato | Lettura in F1; comprensione e calcolo in F2–F3; F4 selezionato | Verificare domanda, dati sottostanti, unità e leggibilità |
| [TabMWP](https://github.com/lupantech/PromptPG) | Da integrare esplicitamente | Tabelle e matematica in F2–F4 | La tabella strutturata serve al verificatore, non deve rivelare nell'input ciò che va letto |
| [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet), [DocLayNet](https://github.com/DS4SD/DocLayNet) | Da integrare | Struttura di tabelle e di pagine singole in F1–F2 | Layout annotato non equivale automaticamente a QA o ordine di lettura certificati |
| [Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) | Da integrare | QA documentale a pagina singola in F2–F3: solo esempi autonomi, senza troncare domande che richiedono altre pagine (v1.5) | Le QA generate richiedono controlli; non sono verità formale |
| [M4-Instruct](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/) | Da integrare, selezionando i sottoinsiemi | Confronto fra poche immagini in F2–F3, limitato a matematica e confronto visivo (v1.5) | Escludere i compiti video dalla prima versione |
| [PGPS9K](https://nlpr.ia.ac.cn/databases/CASIA-PGPS9K/), [FormalGeo](https://formalgeo.github.io/) | Candidati da verificare | Geometria strutturata in F2–F4 | Sovrapposizioni con Geometry3K/Geoperception; solver condizionato alle premesse |
| [MMFineReason](https://github.com/LHL3341/MMFineReason) | Bacino già integrato | Problemi selezionati in F2–F3; F4 dove ammissibile | Separare affidabilità di domanda, riferimento, caption e CoT; release e subset possono sovrapporsi |
| [AI2D](https://registry.opendata.aws/allenai-diagrams/), [ScienceQA](https://github.com/lupantech/ScienceQA), [VisualWebInstruct](https://github.com/TIGER-AI-Lab/VisualWebInstruct) | Alcune sorgenti già rappresentate indirettamente in MMFineReason | Copertura scientifica in F2–F3 | Inventario e deduplicazione prima di aggiungere gli originali; ScienceQA ha anche esempi senza immagini |
| [VizWiz](https://vizwiz.org/tasks-and-datasets/vqa/) | Da integrare | Immagini difficili e risposta impossibile in F2–F3 | Separare illeggibilità, ambiguità e informazione mancante |
| DeepScaleR / [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K) | Replay implementato, da collegare ai mix definitivi | Testo matematico in F2–F4 | Qualità delle soluzioni distinta dalla correttezza del risultato; non importare automaticamente tutte le tracce |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Candidato da integrare | Istruzioni e dialogo in F2–F3 | Selezionare i compiti utili; controllare stile e sovrapposizioni |
| [DeepVision-103K](https://huggingface.co/datasets/skylenage-ai/DeepVision-103K), [ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) | Già previsti per F4 | RL visuale verificabile | I correttori vanno validati; la difficoltà misurata su altri modelli non è quella di Limite |

### Sorgenti complementari e fonti riservate

- **MAVIS-Caption:** non è il nome del dataset separato effettivamente integrato nella repository. Le caption correnti provengono da MAVIS-Geometry. Non renderlo una dipendenza finché non è disponibile e controllato.
- **FineVision e Nemotron-VLM-Dataset-v2:** possibili raccolte da cui selezionare sottoinsiemi per lacune specifiche. Non concatenarle indiscriminatamente agli originali.
- **OpenMathInstruct-2:** complemento eventuale al testo matematico. Distinguere risposte originali dai riferimenti ottenuti per voto di maggioranza sui problemi aumentati.
- **MathV360K e Geo170K:** candidati nel catalogo storico; verificare identificativi, sovrapposizioni e contributo prima di integrarli.
- **GeoSym127K:** sorgente ancora esplorativa; non è un target F1 già ammesso né un pool RL già validato.
- **Geoperception, VisOnlyQA e i benchmark matematici:** valutazione e decontaminazione.
- **CLEVR:** resta fuori dal training secondo la separazione decisa nel progetto. Per questi compiti si privilegiano inizialmente i sintetici propri.

Fonti del confronto: [catalogo corrente](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/downloads/sources.tsv), [bozza F1-clean](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/config/runs/f1-clean.json).

## 5. Qualità dei dati, split e verificabilità

### 5.1 Tre proprietà indipendenti

Per ogni target distinguiamo:

1. **Correttezza del riferimento:** valore, trascrizione, struttura o risposta corretti.
2. **Disponibilità nell'input:** tutte le informazioni necessarie sono presenti e leggibili.
3. **Validità della spiegazione:** i passaggi non introducono fatti falsi o inferenze invalide.

Una risposta finale corretta non certifica la spiegazione. Un programma corretto sulle annotazioni non dimostra che le annotazioni descrivano i pixel. Due agenti concordi non costituiscono automaticamente una verità indipendente.

### 5.2 Record e ciclo minimo

Ogni esempio conserva sorgente, revisione, ID originale, immagini, domanda, target, split, controlli superati e motivazione dell'ammissione. Dove disponibili, si conservano formula, tabella, programma o stato della scena. Le trasformazioni e le rietichettature devono essere riproducibili.

L'identità dell'esempio è composta da **sorgente/revisione/record/compito/variante**, distinta dal gruppo globale del problema o documento. Associare hash di domanda e target, immagini ordinate e riferimenti a crop/pagine, versione della trasformazione e del preprocessing. L'ammissione riguarda questa coppia input–compito: un'immagine ammessa non autorizza automaticamente tutte le QA o i crop derivati. Se una trasformazione cambia input, domanda o target, verificare la validità dell'ammissione prima di riutilizzarla. Le trasformazioni deterministiche possono restare nel loader, purché codice, parametri ed eventuale seed siano fissati e il risultato corrisponda al record ammesso.

Il ciclo è:

1. Controllo di integrità, decodifica, campi e formato.
2. Deduplicazione e separazione per problema, immagine e documento; separazione aggiuntiva di template/stili nel test di generalizzazione dichiarato.
3. Controlli specifici della sorgente.
4. Riparazioni deterministiche giustificate.
5. Campione stratificato, includendo una quota casuale degli esempi apparentemente validi.
6. Audit mirato dei dubbi residui e registrazione della copertura dei controlli.
7. Manifest immutabile della versione ammessa.
8. Controllo che i loader usino il manifest e contabilizzino gli scarti successivi all'ammissione per errore di costruzione, lunghezza o altri vincoli.

Classi operative: valido, problema minore correggibile, incerto, errore grave. Gli incerti non diventano automaticamente scarti definitivi o esempi validi.

### 5.3 Controlli senza token LLM

- Hash, schema, file mancanti, immagini vuote, dimensioni, duplicati e conflitti di etichetta.
- Parsing LaTeX e controlli di struttura; confronto con la sorgente dove disponibile.
- Parser delle specifiche geometriche e confronto fra rappresentazioni disponibili.
- Esecuzione dei programmi di soluzione e verifica delle condizioni dichiarate.
- Calcolo su tabelle, funzioni e scene sintetiche.
- Coerenza di opzioni, risposta finale, unità, dominio e arrotondamenti.
- Controlli di clipping e visibilità disponibili dal renderer.
- Controlli delle trasformazioni: una rotazione può richiedere di aggiornare risposte su destra/sinistra.
- Decontaminazione contro gli split riservati, con revisione dei candidati ambigui.

Il modulo MAVIS esistente specifica già che il confronto fra caption e diagram_generation prova coerenza fra due descrizioni, non la fedeltà dell'immagine. Va mantenuta questa distinzione. [mavis_gen.py](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/mavis_gen.py).

### 5.4 Audit e misura degli errori

Gli agenti vengono usati solo quando aggiungono informazione rispetto ai controlli esistenti. Risposte brevi e strutturate, contesto essenziale, budget di chiamate e token dichiarato prima dell'esecuzione. Secondo verificatore su incerti, disaccordi e una quota casuale delle approvazioni; il 10% è un'ipotesi iniziale da calibrare, non una prova dell'accuratezza dei revisori o un requisito universale di certificazione (§5.9).

Misurare falsi positivi e falsi negativi con casi indipendenti quando disponibili. Le alterazioni introdotte devono avere una verità definita; includere controlli negativi corretti con coordinate alternative, rotazioni lecite, formulazioni equivalenti e ipotesi esplicitamente scartate.

Un parser può rilevare bene un tipo di errore e non coprirne altri: riportare copertura per categoria, non solo una sensibilità aggregata. Le alterazioni artificiali misurano la sensibilità sulle categorie costruite, non il tasso naturale di errore del dataset.

### 5.5 Split

Usare training, development e test finale riservato. I benchmark pubblici consultati ripetutamente per scegliere configurazioni sono strumenti di sviluppo; non diventano test indipendenti per semplice denominazione.

Tutte le domande della stessa immagine o dello stesso documento appartengono allo stesso gruppo. Per i sintetici, separare stati, istanze e, nel test di generalizzazione, anche template o stili. Le coppie controfattuali restano nello stesso split.

La validazione basata sui primi elementi di uno shard, attualmente usata dal flusso dati, non basta da sola a garantire questa separazione. Occorrono liste esplicite per gruppo, senza rifare l'intera infrastruttura.

### 5.6 Costruire il gold, non soltanto confrontare una risposta con l'etichetta

Una golden answer è un riferimento sostenuto da evidenza per **quel compito e quell'input**. Il campo `answer` di un dataset è inizialmente un candidato; né l'origine pubblica né l'accordo fra modelli lo rendono automaticamente corretto. Il controllo della risposta finale non convalida la spiegazione che l'accompagna.

Per ogni record conservare almeno:

- Sorgente e revisione, ID originale, origine a monte, identità del compito/variante (§5.2), gruppo del problema/documento e genitori delle trasformazioni.
- Domanda, opzioni complete quando presenti, immagini ordinate, hash e versione del preprocessing.
- Risposta originale e riferimento ammesso separati; tipo di risposta, forma canonica, unità, dominio, tolleranze e regola per le alternative equivalenti.
- Evidenza della risposta: programma/calcolo/annotazione, versione del verificatore ed esito; ipotesi necessarie e loro disponibilità nell'input.
- Esiti separati per correttezza, osservabilità e spiegazione; stato operativo e motivazione di ogni riparazione o esclusione.

Usare tre stati di evidenza, senza attribuire percentuali di affidabilità inventate:

| Stato | Evidenza | Uso |
| --- | --- | --- |
| **Gold verificato nel perimetro dichiarato** | Riferimento ricalcolato o controllato indipendentemente; semantica del compito e input verificati per le proprietà richieste | Nucleo supervisionato, development/test; RL solo con verificatore del reward a sua volta validato |
| **Target controllato** | Annotazione o pseudolabel con controlli e qualità residua stimata, ma senza prova individuale sufficiente | Eventuale quota SFT dichiarata, secondo le soglie della sorgente; non chiamarlo gold verificato né usarlo come verità per certificare altri dati |
| **Incerto o in conflitto** | Riferimenti discordanti, input insufficiente, controlli inconcludenti | Quarantena e recupero mirato; escluso dal nucleo gold e dal primo RL |

Ogni compito iniziale deve avere una breve **regola di ammissione** che elenchi le proprietà necessarie di risposta e input, con il livello di evidenza per ciascuna:

| Evidenza della proprietà | Condizione per utilizzarla |
| --- | --- |
| Verifica sul singolo record | Controllo, versione, input ed esito identificati; gestione esplicita dei casi fuori dominio |
| Invariante della famiglia | Proprietà giustificata per quel generatore/renderer e versione, con assunzioni esplicite e precondizioni controllate sull'istanza |
| Controllo soltanto campionario | Qualità stimata della popolazione dichiarata; non costituisce una prova individuale per ogni record |

La promozione a gold richiede che le proprietà necessarie siano coperte nel perimetro dichiarato. Poche immagini corrette non dimostrano un'invariante di rendering. Se una proprietà necessaria è soltanto campionata, mantenere il target controllato oppure recuperare le sole QA con copertura sufficiente; non introdurre per questo una revisione con agenti di ogni record. Registrare separatamente ciò che è verificato, assunto e stimato. Per un esempio answer-only lo stato della spiegazione è **non applicabile**, non «fallita».

Gli errori dimostrati vengono corretti con una trasformazione tracciata oppure esclusi. Non risolvere un conflitto sostituendo l'etichetta con la risposta più votata. Se più risposte sono realmente valide, rappresentare l'insieme o la regola di accettazione invece di imporre un'unica stringa.

Per i dati programmabili, ricavare la risposta dallo stato strutturato, controllare precondizioni, degenerazioni, dominio e completezza. Usare un controllo matematico indipendente quando disponibile e testare casi limite con verità nota; ripetere lo stesso algoritmo con le stesse assunzioni non dà una seconda prova indipendente. Eseguire codice proveniente dai dataset solo in un ambiente isolato con limiti di risorse.

Separare sempre **valore esatto**, **valore visualizzato** e **risposta richiesta**. Se il renderer mostra `14.67` per un valore interno `44/3`, una trascrizione deve riportare il testo visibile; una soluzione esatta richiede che il problema fornisca informazioni sufficienti per ricostruirla. Analogamente, leggere `x+x` e rispondere `2x` può essere corretto in un compito di semplificazione, ma non in una trascrizione fedele. Preservare notazioni equivalenti senza normalizzazioni che cancellino segni, unità o dominio.

### 5.7 Pulizia specifica per sorgente

| Sorgente/famiglia | Come ottenere o controllare il riferimento | Controllo visivo e limite |
| --- | --- | --- |
| Sintetici propri | Stato della scena, formule e programmi; calcolo esatto, vincoli e controlli indipendenti sulle famiglie | Controllare clipping, occlusioni, label e leggibilità dopo il preprocessing. Una variabile nascosta del generatore non può diventare una risposta visiva non inferibile. Audit dei renderer e campione casuale delle uscite. |
| MAVIS-Geometry / MAVIS-Caption | Riusare parser e fatti supportati; ricalcolare le QA quando la specifica basta; tenere separati descrizione canonica e soluzione | `diagram_generation` è una specifica testuale, non sempre un programma eseguibile. Coerenza con la caption non prova fedeltà ai pixel. Verificare famiglia del disegno, misure stampate e casi ambigui. |
| UniMER / OCR (MathWriting rimandato, v1.5) | Etichette e sorgenti originali; parsing e normalizzazione che preservano la notazione; confronto con rasterizzazione quando disponibile | Un LaTeX compilabile può descrivere un'altra formula. Rendering/OCR aggiuntivo aiutano a trovare candidati errati, non sono un oracolo. Separare matematica equivalente e trascrizione; controllare crop, indici e simboli piccoli. |
| Grafici e tabelle | Calcoli sulle tabelle sorgente: selezione, somma, differenza, rapporto, unità e arrotondamenti | Le celle interrogate devono essere visibili o deducibili. Non usare precisione interna non rappresentata nel grafico; ammettere una tolleranza motivata dal compito. |
| Documenti a pagina singola, PubTabNet/DocLayNet (QA multipagina rimandata, v1.5) | Strutture e annotazioni disponibili; pagine/celle di evidenza per la risposta; programmi aritmetici quando presenti | Layout e HTML non certificano automaticamente ogni QA. Controllare che tutte le pagine necessarie siano fornite, in ordine, leggibili e con un riferimento non ambiguo. |
| PixMo, descrizioni generali, Docmatix | Provenienza, copertura delle annotazioni e campioni stratificati; conservare solo trasformazioni giustificate | Testo libero senza verità strutturata non diventa interamente gold con parser o consenso. Separare target controllati da nucleo gold; evitare riscritture automatiche che introducano nuovi dettagli visivi. |
| MMFineReason, DeepVision/ViRL e aggregati | Risalire alla sorgente originale; ricontrollare risposta, opzioni e problemi risolvibili con strumenti | Caption del teacher, risposta e CoT sono tre oggetti distinti. Se è verificata solo la risposta, usarla senza la spiegazione dubbia; il gold non si ricava dalla stessa traccia che si vuole validare. |
| Replay matematico testuale | Problema originale, dominio, risposta e soluzione breve verificabile quando disponibile | Nessun controllo visivo, ma stessi gruppi di provenienza e decontaminazione. La concordanza con Violetto non dimostra la correttezza del riferimento. |

Le verifiche visive riguardano l'immagine effettivamente vista dal modello. Se una nuova risoluzione o un crop toglie informazione necessaria, aggiornare l'ammissione per quella configurazione; non imputare automaticamente l'errore al dataset originale. Negli esperimenti sugli encoder distinguere guasti di preprocessing e perdita di leggibilità dai problemi del riferimento, senza scegliere a posteriori esempi favorevoli a un candidato.

Controllare anche l'univocità della domanda e delle opzioni. Il generatore attuale può chiedere il tipo di quadrilatero offrendo `square`, `rectangle` e `parallelogram`, con il solo quadrato come target: più descrizioni sono matematicamente vere. Per queste QA chiedere esplicitamente la **classe più specifica fra quelle previste**, con criterio definito, oppure accettare tutte le alternative valide. Applicare la regola anche alle domande aperte analoghe di MAVIS; non scartare l'intera scena. Lo stato esatto del generatore non risolve da solo un'ambiguità del testo. [Generatore dei poligoni](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/synth.py#L290).

Per le spiegazioni preferire derivazioni brevi ottenute da passaggi verificati. Risposte finali giuste con CoT dubbi possono diventare esempi answer-only se domanda e input restano validi. Non trasformare un dubbio in una falsa etichetta «impossibile rispondere»: questa richiede una prova dell'insufficienza dell'input.

### 5.8 Decontaminazione globale e riproducibile

**Deduplicazione, correttezza del gold e decontaminazione sono controlli distinti.** La prima gestisce copie e sovrappeso; la seconda stabilisce il riferimento; la terza impedisce che i dati riservati, o loro derivati, entrino nell'addestramento. Un dataset decontaminato può ancora contenere etichette sbagliate.

Costruire il nuovo manifest sull'**inventario originale fissato**, non solo sugli esempi sopravvissuti ai vecchi filtri. Importare esclusioni e rietichettature storiche come evidenze con motivo, versione e stato: confermate, da riesaminare o irrisolte. Riesaminare prima le esclusioni basate su solo target comune o similarità; conservare le copie e le sovrapposizioni riservate già confermate senza imporre una nuova verifica manuale. Non applicare ciecamente né cancellare indiscriminatamente le vecchie liste della bozza F1-clean. [Configurazione da migrare](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/config/runs/f1-clean.json#L12).

1. **Fissare i dati riservati prima della selezione.** Versionare benchmark, development e test finale, con ID, immagini, domande e provenienza. Le etichette del test restano nel percorso di valutazione/decontaminazione e non vengono usate come esempi, teacher prompt o dimostrazioni. Le verifiche sul gold del test non usano gli errori di Limite per scegliere le correzioni.
2. **Indicizzare tutte le sorgenti insieme.** Includere F1–F4, replay testuale, caption, soluzioni, aggregati e derivati. Conservare legami a problema, immagine, documento, tabella e generatore. Gli split ufficiali sono vincoli da rispettare, ma non garantiscono assenza di copie in un'altra raccolta.
3. **Trovare corrispondenze certe e candidati simili.** ID/provenienza; SHA dei file e dei pixel normalizzati; pHash; testo e formule con normalizzazione conservativa. Riusare gli n-gram esistenti come segnale. Per ritagli, traduzioni, ridisegni e riformulazioni aggiungere ricerche mirate soltanto dove la provenienza indica rischio. Le soluzioni lunghe possono aiutare, ma risposta breve identica o teorema comune non identificano un problema.
4. **Confermare i legami prima di fondere i gruppi.** Hash percettivo e similarità non sono prove di identità: due diagrammi possono differire solo per una cifra decisiva. Preservare operatori, numeri e condizioni nel confronto; non eliminare un esempio solo perché contiene la stessa formula. Una collisione ambigua resta un candidato da risolvere o in quarantena. Evitare che catene di somiglianze deboli colleghino intere famiglie di dati.
5. **Assegnare lo split al gruppo confermato.** Stesso problema, documento o immagine e tutte le sue QA, caption, traduzioni, crop e varianti controfattuali restano insieme. Un gruppo collegato a un riferimento riservato viene escluso dal training in tutte le fasi. Per i sintetici distinguere separazione delle istanze e test intenzionale su template/stili mai visti; non vietare indiscriminatamente ogni uso della stessa regola matematica.
6. **Ricontrollare il manifest finale e il suo uso.** Dopo riparazioni, nuove QA o trasformazioni, ereditare gruppo/split e rifare i controlli pertinenti. Produrre elenco degli esempi ammessi per compito/variante, esclusioni motivate, collisioni irrisolte e statistiche per sorgente. Filtrare i reader esistenti con un indice di ammissione e selezionare esplicitamente la validazione dal manifest; una nuova variante senza ammissione non entra automaticamente. Applicare lo stesso vincolo a SFT, replay e RL. Registrare esempi e target-token effettivamente consumati, ripetizioni e scarti successivi all'ammissione: se cambiano materialmente la popolazione o i pesi, aggiornare copertura, denominatori e stime. Una misura sul manifest non descrive automaticamente la distribuzione realmente consumata.

Non è possibile rimuovere con questi filtri un'eventuale esposizione pregressa di ViT/decoder ai benchmark. Documentare questo limite e includere test nuovi riservati. Nessun match trovato significa «nessuna sovrapposizione rilevata dai controlli dichiarati», non assenza universale di contaminazione. La deduplicazione e la misura dell'overlap hanno motivazioni sperimentali anche nei corpus testuali, ma i risultati quantitativi non si trasferiscono automaticamente a Limite-VL. [Lee et al., ACL 2022](https://aclanthology.org/2022.acl-long.577/).

### 5.9 Verifica mirata e qualità residua

Il nucleo di riferimento per **valutazione e collaudo dei verificatori** deve essere piccolo, con evidenza controllata indipendentemente e revisione competente dei casi che non ammettono verifica automatica. Assegnare ruoli espliciti: i casi usati per sviluppare o calibrare il correttore sono di development; il test finale resta separato e non diventa materiale per adattare verificatore, filtri o modello. Non usare il vecchio audit o il modello in addestramento come unica verità. Il pool RL richiede sia riferimenti affidabili sia un correttore adatto al compito; i target aperti senza verifica sufficiente restano fuori dal primo RL.

Mantenere due campioni separati:

- **Diagnostico:** stratificato e adattivo, concentrato su conflitti, crop, figure dense, annotazioni anomale e famiglie nuove; includere anche approvazioni e scarti automatici per trovare falsi negativi e falsi positivi dei filtri.
- **Di stima:** estratto probabilisticamente dalla versione finale ammessa, con probabilità note e unità per problema/documento. Non riusare come conferma indipendente gli stessi esempi sui quali sono state progettate le riparazioni. Se la versione cambia, aggiornare il disegno e la stima.

Prima dell'estrazione fissare **quale quantità si vuole stimare**: quota di QA errate, quota di documenti con almeno un errore, oppure peso degli errori nella distribuzione di training. Sono quantità diverse. Dichiarare proprietà, soglia quando pertinente, popolazione/versione, denominatore, selezione dei gruppi e delle QA al loro interno, probabilità di inclusione e pesi. Una QA per documento campionato uniformemente non stima automaticamente la quota di tutte le QA; non dimostra neppure che le altre QA del documento siano corrette. Per stimare il tasso generale di scarti erronei serve un disegno probabilistico anche sulla popolazione degli scarti; i conteggi diagnostici restano diagnostici.

Definire in anticipo il trattamento dei casi non giudicabili: risoluzione indipendente, limite conservativo che li consideri potenzialmente errati, oppure dichiarazione esplicitamente limitata alla popolazione valutabile. Gli incerti non vengono eliminati dal denominatore a posteriori né contati implicitamente come corretti. Il rapporto riporta comunque la quota non risolta.

Per sorgente e compito riportare copertura dei controlli, tasso di errore grave, quota incerta, riparazioni, scarti corretti/errati e intervalli appropriati. Per un semplice campione indipendente con zero errori, circa 300 casi danno un limite superiore unilaterale al 95% vicino all'1%; per circa 0,1% ne servono circa 3.000. Questo non dimostra zero errori e non si estende automaticamente a ciascuno strato. La certificazione riguarda la proprietà, popolazione e versione dichiarate, assumendo adeguata accuratezza del riferimento indipendente.

Gli agenti vedono solo l'input necessario al claim: domanda, immagini o ritagli pertinenti e informazioni essenziali; producono esito, breve evidenza e motivo. Secondo verificatore per dubbi/disaccordi e quota casuale delle approvazioni, inizialmente 10% come ipotesi operativa da calibrare. Per una quota mirata del nucleo di riferimento, effettuare la prima valutazione senza mostrare il target candidato, così da ridurre il condizionamento; seguono confronto e risoluzione documentata dei conflitti. Conflitti importanti senza verità indipendente richiedono revisione competente oppure restano irrisolti; non si forza il consenso. Riusare la stessa revisione quando copre lo stesso claim e input, non estenderla automaticamente a nuove domande sulla stessa immagine.

L'accuratezza del riferimento deve essere sostenuta rispetto alla soglia dichiarata: usare prove o fonti indipendenti per la proprietà rilevante, casi corretti difficili ed errori noti, senza limitarsi a perturbazioni artificiali. Registrare classi di errore coperte e limiti della calibrazione. Il 10% di doppia revisione e l'assenza di disaccordi non misurano da soli i falsi negativi condivisi. Se l'accuratezza necessaria non è sostenibile, presentare un audit informativo o un risultato condizionato al metodo di revisione, senza trasformare la regola dei 300/3.000 in una garanzia più forte. Questi requisiti riguardano il piccolo nucleo e la dichiarazione di qualità; non impongono controlli multiagente su ogni record SFT.

Controlli deterministici: zero token LLM, ma costo CPU/storage registrato. Chiamate agli agenti: solo campioni e casi selezionati, con budget per sorgente e output brevi. Loggare chiamate, token visivi/testuali e costo reale; senza numero di dubbi e prezzi effettivi non dichiarare un risparmio percentuale come già misurato.

### 5.10 Riutilizzo del codice e ordine minimo di implementazione

Il codice attuale è utile come base, ma richiede queste correzioni circoscritte:

1. **Manifest e gruppi globali:** riutilizzare lettura degli shard e provenienza; applicare ammissione e split all'identità del compito/variante, incluse QA e crop costruiti al volo. Usare validazione esplicita e contabilizzare scarti e distribuzione consumata anche nel replay e nel futuro RL.
2. **Gold tipizzato per compito:** separare riferimento originale, esatto e visualizzato; fissare la regola di ammissione di §5.6 e risolvere le domande ambigue. Partire da sintetici, MAVIS supportato e formule, poi tabelle/documenti, preservando la copertura prevista. Salvare la prova e la trasformazione, non solo `passed=true`.
3. **Deduplicazione come evidenza graduata:** `split_duplicates.py` lavora per sorgente e usa anche target identici; `decontam.py` rimuove simboli dal testo normalizzato. Riusare questi segnali per trovare candidati, non come prova universale di contaminazione. Migrare le vecchie esclusioni dall'inventario originale con la procedura di §5.8. [Script duplicati](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/scripts/split_duplicates.py), [decontaminazione](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/decontam.py).
4. **Estrazione, selezione e verifica per compito:** adattare l'intero percorso, non solo `f2.answers_match`. Il vecchio `grade` richiede `</think>` ed estrae l'ultimo `boxed`: non implementa il recupero answer-only né il controllo delle risposte finali incompatibili. Separare estrazione, conformità del formato, correttezza finale e ammissione della spiegazione; `usable` non certifica tutti i passaggi. Applicare il controllo delle contraddizioni alla risposta finale senza penalizzare tentativi esplicitamente abbandonati nel ragionamento. Distinguere `corretto`, `errato`, `non determinato` ed errore tecnico; unità, dominio, opzioni ed equivalenze dipendono dal compito. `1.5` e `3/2` sono equivalenti in una risposta numerica, ma non necessariamente in una trascrizione fedele. [Correttore attuale](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/f2.py), [contratto e limiti di Math-Verify](https://github.com/huggingface/Math-Verify/blob/main/README.md).
5. **Collaudo e campioni:** verificare copia trasversale fra sorgenti, pHash simile con cifra decisiva diversa, split ereditato dai derivati, QA non ammessa su immagine ammessa, risposta diretta corretta, risposte finali incompatibili, tentativo abbandonato lecito, equivalenza numerica distinta dalla trascrizione e classificazione geometrica ambigua. Sono piccoli collaudi del contratto; poi audit diagnostico della prima versione, campione di stima e rilascio del manifest. Se una sorgente resta rumorosa e costosa da verificare, ridurne il perimetro o rimandarla anziché introdurre controlli LLM per ogni esempio.

Prima di implementare il primo lotto, fissare quattro specifiche brevi: record/manifest e derivati; regole di ammissione delle prime famiglie; estrazione e verificatori per compito; campionamento e budget. Non è necessario progettare subito gli adattatori di ogni sorgente. Prima di un run serve la loro applicazione verificata sul perimetro scelto; la revisione del documento non sostituisce quel collaudo.

Il criterio per l'esperimento iniziale è un nucleo gold motivato, nessuna collisione confermata train/riservato nel manifest, incertezze isolate e limiti misurati. «Tutte le golden answer sono giuste» non è una dichiarazione supportata da un campione finito: per la parte programmabile si dichiarano proprietà verificate e assunzioni; per la parte annotata si dichiara qualità residua stimata. La correttezza del manifest e il suo uso effettivo nel training sono entrambi requisiti di avanzamento.

## 6. F0 — Preparazione e scelta dell'architettura visiva

### 6.1 Materiale necessario

Preparare un pannello rappresentativo di formule stampate, geometria, grafici, tabelle, documenti a pagina singola e immagini reali (perimetro v1.5). Non usare solo la geometria sintetica per scegliere l'encoder destinato a tutta la pipeline.

Partire da sintetici propri, MAVIS supportato e UniMER verificati; aggiungere il minor insieme che copra le capacità mancanti: uno di scene reali, grafici/tabelle e documenti a pagina singola, più un piccolo pannello con poche immagini per esempio. Il manoscritto non è richiesto (v1.5). PixMo e le raccolte documentali della tabella sono candidati, non prerequisiti da integrare tutti. Scegliere gli adattatori iniziali in base a disponibilità, verificabilità e lacune di copertura, senza dichiarare che siano già implementati o scaricati. Nuove grandi raccolte entrano quando colmano una lacuna documentata.

Fissare anche il pannello documentale (pagina singola) e quello con poche immagini prima dell'ammissione definitiva, per poterne decontaminare i dati. Raggruppare le varianti dello stesso problema, incluse quelle di MathVerse/DynaMath, nei confronti statistici; non contarle come prove indipendenti.

Fissare prima della prova:

- ID degli esempi e target.
- Split e metriche di development.
- Budget di supervisione e protocollo di arresto.
- Griglia limitata degli iperparametri.
- Criteri di scelta, includendo qualità, memoria e costo.
- Budget di valutazione uguale fra candidati.

### 6.2 Due piloti comparabili

Per ciascun encoder eseguire:

1. Un allineamento breve del collegamento visivo.
2. Un breve adattamento con decoder e ViT addestrabili.
3. La stessa valutazione per capacità.

Il confronto usa gli stessi problemi e budget di supervisione, con un numero comparabile di token visivi forniti a Limite. Risoluzione nativa, preprocessing e costo possono differire: registrarli e confrontare anche la qualità a costo simile.

La scelta non si basa soltanto sulla loss del proiettore. Un encoder può allinearsi rapidamente e risultare meno efficace dopo l'adattamento.

Conservare curve e checkpoint intermedi per distinguere velocità iniziale e qualità dopo adattamento. Prima dei risultati fissare la regola che combina capacità, regressioni e costo. Se il vantaggio è piccolo o dipende ancora dalla convergenza del collegamento, ripetere un'inizializzazione o estendere solo il tratto discriminante con una regola comparabile fra candidati; nessun numero universale di seed è imposto a priori.

Il ViT estratto da Qwen proviene da un sistema già addestrato: il confronto valuta punti di partenza concretamente utilizzabili, non isola causalmente il valore della sola architettura ViT. Un test nuovo con verità indipendente aiuta a limitare l'effetto di esposizioni pregresse ai benchmark.

Se la differenza resta incerta, estendere la prova in modo limitato o ripetere il confronto pertinente; non dichiarare un vincitore da oscillazioni trascurabili. A parità di qualità documentata, privilegiare costo e semplicità.

### 6.3 Scelte architetturali da confrontare

**Tutte le opzioni richieste devono ricevere una valutazione esplicita; nessuna è adottata per default.** Per ciascuna registrare ipotesi, fattibilità, prova proposta, costo e decisione. Un'opzione non provata per ragioni di budget resta «non valutata sperimentalmente», non «inefficace». Le configurazioni sotto sono candidati di progetto, non risultati né istruzioni per lanciare job.

Separare tre domande: **quali feature estrarre dal ViT**, **come comprimerle/proiettarle**, **dove introdurle in Limite**.

| Asse | Riferimento | Alternative da valutare | Domanda sperimentale |
| --- | --- | --- | --- |
| Layer del ViT | Ultimo hidden state, con punto di estrazione e normalizzazione dichiarati | Un singolo layer intermedio; fusione di 2–3 profondità | Le feature intermedie migliorano lettura fine, geometria o documenti? |
| Proiettore | Merge spaziale 2×2, MLP `4C → 2560 → 1280` | Proiezione lineare `4C → 1280`; MLP più largo `4C → 4C → 1280` | Quale capacità del collegamento serve realmente? |
| Fusione iniziale | Una sola rappresentazione introdotta negli embedding | Fusione di feature di più layer, allineate sulla stessa griglia | Basta arricchire l'ingresso senza nuovi punti di iniezione? |
| DeepStack | Nessuna iniezione visiva aggiuntiva nel decoder | Feature intermedie proiettate e aggiunte alle posizioni immagine in 2–3 punti del decoder | Reinserire informazione durante l'elaborazione dà un vantaggio sulla fusione iniziale? |
| MUDD | Percorso nativo, già alimentato anche dall'ingresso visivo | Un ramo visivo dedicato nel percorso di aggregazione, con gate | Un accesso esplicito aggiunge informazione utile rispetto al percorso esistente? |
| Value embeddings | Contributo nullo nelle posizioni immagine | Valori dipendenti dalle feature dell'immagine, tramite proiezione dedicata | Un secondo ingresso visivo migliora l'attenzione senza peggiorare stabilità e testo? |

Qui `C` è la dimensione effettiva delle feature dell'encoder scelto; 1280 è la dimensione del decoder. Le dimensioni proposte non sono una copia dichiarata del merger Qwen. Registrare anche normalizzazione, attivazione, bias, inizializzazione e numero di parametri: la sola larghezza non definisce un proiettore. Nel primo confronto mantenere uguali compressione spaziale e token al decoder. Se viene riutilizzato un merger Qwen, dichiararlo come variante distinta e impedire una seconda compressione involontaria.

Per i layer intermedi fissare gli indici dopo aver scelto i checkpoint. Confrontare prima un'uscita intermedia singola con quella finale; poi una fusione semplice di due profondità, estendibile a tre se motivato. Una somma pesata di feature normalizzate e allineate mantiene costante il numero di token; una concatenazione cambia dimensioni e costo e va contabilizzata come tale. Non attribuire al numero di layer un guadagno dovuto soltanto a più parametri.

DeepStack e fusione iniziale sono varianti diverse. Come precedente, Qwen3-VL usa feature a tre profondità del ViT, merger dedicati e aggiunte ai primi tre layer del decoder; questo non dimostra che gli stessi punti siano ottimali per Limite. Il nostro candidato usa una piccola mappa esplicita `layer ViT → punto nel decoder`, mantenendo un ingresso visivo principale. Precisare se l'aggiunta precede o segue il blocco, la normalizzazione e il prelievo dello stato da parte di MUDD. [Qwen3-VL, §2.2](https://arxiv.org/html/2511.21631v1#S2.SS2).

### 6.4 Integrazione con MUDD e value embeddings

MUDD non va trattato come se oggi non vedesse le immagini: il percorso iniziale già le contiene. Il candidato aggiuntivo deve specificare **quale tensore visivo entra in quale operazione di aggregazione** e con quali forme, maschere e gate. Questa specifica richiede la lettura del codice del checkpoint fissato; «collegare a MUDD» non è da sola un'architettura implementabile. Iniziare da un solo ramo dedicato, senza modificare contemporaneamente il routing testuale o tutti i tap del decoder.

Per i value embeddings distinguere tre casi: zero attuale; vettore comune del token segnaposto, già disponibile come controllo; proiezione realmente dipendente dall'immagine, da implementare. Il secondo caso non fornisce dettagli visivi e non sostituisce il terzo. La proiezione dedicata deve rispettare la forma reale del percorso dei valori del checkpoint, senza assumere che coincida con i 1280 canali del residual stream. All'inizio usare le stesse feature del percorso principale, così da isolare l'effetto del nuovo ingresso.

DeepStack, MUDD e value embeddings possono interagire: duplicazione e variazione della scala sono rischi da misurare, non prove di incompatibilità. Valutare prima ciascun ramo rispetto al riferimento, poi almeno un incrocio motivato se costi e segnali lo consentono. Non sommare a priori i benefici misurati separatamente. XSA rimane un controllo di compatibilità esplicito; un nuovo ingresso diretto in XSA richiede una propria ipotesi e non viene aggiunto automaticamente a tutte le varianti.

Controlli minimi per i nuovi rami:

- Maschere limitate alle immagini e corrispondenza esatta fra patch, immagini e posizioni, anche con più immagini e packing.
- Inizializzazione che permetta di riprodurre la baseline con i nuovi contributi disattivati; verificare il flusso dei gradienti evitando combinazioni di gate e proiezioni che restino entrambe senza gradiente.
- Per testo puro, equivalenza prima del training con i rami disattivati; dopo l'adattamento, misurazione delle regressioni senza promettere identità dei logit.
- Parità fra forward completo, prefill e generazione con cache; i contributi non devono essere applicati di nuovo ai token testuali generati.
- Norme delle attivazioni, gradienti, saturazione dei gate e variazioni dei parametri; l'interpretazione dei gate non sostituisce le metriche di capacità.
- Salvataggio/ricaricamento dei nuovi moduli e compatibilità con le interpolazioni opzionali, lasciando separati i parametri assenti da Violetto originale.

### 6.5 Ordine delle prove e limite alla complessità

1. **Confronto iniziale dei due encoder:** usare il riferimento finale-layer/MLP di §6.2; la graduatoria resta provvisoria rispetto alle integrazioni successive.
2. **Proiettore e feature:** sul candidato iniziale confrontare le alternative economiche, una scelta alla volta. Riutilizzare il riferimento e provare un incrocio layer/proiettore solo se necessario per distinguere le cause del guadagno.
3. **Punti di ingresso:** confrontare DeepStack, ramo MUDD e value visivi separatamente con un riferimento comune; una configurazione per famiglia nella prima ricognizione, senza una ricerca su ogni layer possibile.
4. **Interazioni:** confrontare una combinazione motivata con i suoi componenti isolati. Dare priorità a DeepStack×MUDD se entrambi risultano utili o mostrano interferenza; altrimenti scegliere l'incrocio sostenuto dai risultati. Un risultato nullo isolato non esclude una sinergia, ma non giustifica esplorare tutte le combinazioni.
5. **Conferma dell'encoder e della configurazione:** verificare sui due encoder la configurazione finale comparabile, adattandone correttamente dimensioni e preprocessing. Se il budget non consente questo controllo, dichiarare la scelta condizionata alle configurazioni effettivamente provate. Ripetere o prolungare soltanto i confronti ancora incerti.

Ogni confronto decisivo comprende allineamento breve dei nuovi collegamenti e adattamento breve: la sola loss con decoder congelato non seleziona l'architettura finale. Usare gli stessi problemi, target, split, regola di tuning e budget di supervisione; riportare anche qualità a costo confrontabile. Se si dirama da un checkpoint già adattato, dare al riferimento lo stesso training aggiuntivo e non confrontare direttamente questo risultato con una variante partita da zero.

Misurare lettura di simboli piccoli, OCR/formule, geometria, grafici/tabelle, documenti, uso di più immagini e mantenimento delle capacità testuali. Usare confronti appaiati per problema/documento e controlli controfattuali delle immagini. Riportare parametri aggiunti, memoria, throughput, latenza e costo di prefill/generazione; riusare più layer nello stesso forward non è privo di costo, anche a lunghezza della sequenza invariata.

Il numero di prove e il loro tetto GPU/euro vanno fissati dopo un profilo tecnico e prima dell'esecuzione: non viene autorizzata una griglia completa encoder × layer × proiettore × ingressi. A parità di qualità documentata scegliere la soluzione più semplice. Un miglioramento deve superare l'incertezza e rispettare i margini di regressione e costo concordati; non basta una loss inferiore o un singolo benchmark favorevole.

### 6.6 Uscita di F0

Encoder e revisione scelti; layer estratti, proiettore e punti di ingresso definiti; preprocessing verificato; dati iniziali ammessi; protocollo fissato; costo misurato delle configurazioni implementate. Documentare anche opzioni respinte, non provate e motivi. I costi di F3/F4 verranno misurati prima delle rispettive fasi, senza renderne l'implementazione una dipendenza iniziale.

Il run principale riparte dai checkpoint preaddestrati scelti, inizializzando i nuovi collegamenti secondo la configurazione selezionata. In F1 si addestrano anche eventuali nuovi merger DeepStack, proiezioni visive e gate; i parametri originali di ViT e decoder restano congelati. In F2 diventano addestrabili anche i moduli originali previsti dalla fase. In F3/F4, «proiettore» comprende i nuovi collegamenti visivi adottati; i loro gruppi, LR e salvataggi devono essere espliciti. Nessun nuovo percorso viene introdotto a metà del run principale senza una nuova verifica.

## 7. F1 — Allineamento percettivo

### Scopo e parametri

Collegare le rappresentazioni visive a Limite mantenendo congelati encoder e decoder. Addestrare proiettore/merger e nuovi parametri dell'interfaccia visiva.

### Dati e target

Nucleo: MAVIS canonico, UniMER e sintetici esistenti. Integrare un campione controllato di OCR generale, lettura di grafici e tabelle, struttura di pagina singola e scene reali (il manoscritto è rimandato, v1.5).

Compiti: trascrivere, identificare, localizzare, estrarre e descrivere brevemente. Esempi: leggere una formula, indicare una misura scritta, riconoscere una legenda, estrarre una cella, individuare un oggetto.

Non usare lunghe tracce di ragionamento come target ordinario di F1. Le proprietà note soltanto al generatore non diventano automaticamente descrizioni di ciò che si vede.

### Esecuzione proposta

- Contesti prevalentemente brevi, con 4k token come punto di partenza da verificare.
- Risoluzione e token visivi proporzionati alla leggibilità richiesta.
- Campionamento per capacità e limite alle ripetizioni della stessa immagine.
- Nessuna quota di puro testo per aggiornare un decoder congelato: il testo puro resta utile come controllo.
- Mantenere coerenti formato dei target, token speciali e generazione.

### Criterio di passaggio

Miglioramento dei compiti percettivi di development; vantaggio dell'immagine sui problemi realmente visivi; risposte corrette alle modifiche controfattuali; nessun problema sistematico nell'associazione fra immagine e target. Arrestare quando il collegamento è utilizzabile e i miglioramenti marginali rallentano, senza aspettare che F1 risolva tutti i compiti complessi.

## 8. F2 — Adattamento multimodale congiunto

### Scopo e parametri

Addestrare insieme ViT, proiettore e decoder. Dopo un breve avvio stabile, rendere addestrabile il ViT con learning rate inferiore agli altri moduli. Il congelamento permanente dell'encoder non è la ricetta principale proposta.

L'attuale set_trainable congela sempre l'encoder: questa fase richiede un intervento esplicito su gruppi dell'ottimizzatore, salvataggi e controlli dei gradienti.

Il trainer attuale raccoglie i parametri fuori dal decoder come adapter: sbloccare soltanto `requires_grad` assegnerebbe al ViT il gruppo del proiettore. Creare gruppi disgiunti per decoder, ViT, collegamenti e righe visive, con LR, precisione e stati master espliciti; rimuovere i conteggi rigidi di parametri. Salvare tutti i moduli aggiornati e lo stato necessario alla ripresa. Prima del secondo stadio F0 collaudare gradienti, variazioni effettive dei pesi, salvataggio/ricaricamento e ripresa. Il ramo autograd visivo esistente è riutilizzabile; non serve riscrivere il forward. Un LR basso non garantisce da solo aggiornamenti effettivi nella precisione scelta.

### Copertura

- OCR generale e formule stampate (manoscritto rimandato, v1.5).
- Geometria, coordinate e funzioni.
- Grafici statistici e tabelle.
- Layout e comprensione documentale.
- Localizzazione, conteggio e relazioni spaziali.
- Diagrammi e domande scientifiche.
- Scene reali e domande varie.
- Confronto fra poche immagini, in matematica e confronto visivo (collegamento fra pagine rimandato, v1.5).
- Illeggibilità, ambiguità e informazione insufficiente.
- Istruzioni, chiarimenti e dialogo.

Mantenere esempi di F1. Da F2 entra stabilmente il testo puro: matematica tramite DeepScaleR/DeepMath selezionati e istruzioni generaliste controllate. Il replay va collegato alla miscela effettiva, non soltanto disponibile nel codice.

Il campionamento corrente pesa esempi, mentre la loss media token supervisionati: la quota attesa di target-token dipende dal peso di campionamento moltiplicato per la lunghezza del target. Non è una misura diretta del contributo ai gradienti, ma impedisce di confondere percentuale di problemi e peso effettivo della supervisione. Loggare quote di target-token, problemi unici, ripetizioni, lunghezze e scarti per sorgente. Il collator esclude gli esempi oltre il limite: se il replay da preservare non entra nel contesto iniziale, usare un bucket compatibile o limitare la dichiarazione di conservazione, senza troncare soluzioni alla cieca o rigenerare tutto. Separare replay matematico e apprendimento di istruzioni/risposte brevi.

### Immagini multiple e contesto

Cominciare da pochi elementi leggibili, di solito due immagini: confronto fra figure, figura e dettaglio, tabella collegata a grafico, problemi di matematica con più figure. Le sequenze di pagine sono rimandate (v1.5).

Le domande devono richiedere davvero il collegamento fra gli input. Aggiungere distrattori controllati e riferimenti espliciti alle immagini. Non fornire nell'input la tabella o la scena di riferimento quando il compito richiede di ricavarla dai pixel.

Un contesto iniziale intorno a 8k è un'ipotesi di lavoro. Espandere verso 16k–32k solo per compiti che lo richiedono e dopo verifiche dell'uso effettivo del contesto. Il budget visivo si controlla sull'intero esempio: aggiungere immagini riducendole fino a renderle illeggibili non costituisce addestramento utile.

### Criterio di passaggio

Prestazioni di development documentate in tutte le famiglie centrali; uso corretto delle immagini multiple; generalizzazione a stili e documenti nuovi; conservazione delle capacità testuali entro margini predefiniti. Nessuna singola media complessiva sostituisce il controllo delle categorie.

## 9. F3 — Specializzazione nel ragionamento e nel comportamento

### Scopo e parametri

Addestrare decoder e proiettore; congelare inizialmente il ViT già adattato in F2. Eventuali ulteriori aggiornamenti del ViT richiedono una lacuna percettiva documentata e un confronto mirato.

### Programma matematico e scientifico

Verificare copertura e difficoltà per:

- Aritmetica, frazioni, percentuali, proporzioni e unità.
- Algebra, equazioni, disequazioni e sistemi.
- Geometria euclidea e analitica, trigonometria e geometria solida.
- Funzioni, limiti, derivate e integrali.
- Probabilità, statistica e combinatoria.
- Vettori, matrici e algebra lineare.
- Logica e teoria dei numeri.
- Fisica, chimica, ingegneria e altri domini scientifici selezionati.

Un grande dataset di matematica non prova una copertura bilanciata, soprattutto per le versioni visuali dei problemi. La matrice di copertura distingue argomento, difficoltà, tipo di immagine e necessità dell'informazione visiva.

### Formati della supervisione

| Compito | Target preferito |
| --- | --- |
| Lettura, identificazione, calcolo immediato | Risposta diretta |
| Problema ordinario | Informazioni pertinenti, passaggi sufficienti, risultato |
| Problema difficile o dimostrazione | Sviluppo esteso quando necessario |
| Informazione insufficiente o illeggibile | Indicazione precisa di ciò che manca o richiesta di chiarimento |

Usare l'inglese come lingua principale delle istruzioni e delle risposte. Non imporre che ogni problema semplice generi un ragionamento lungo. Non troncare automaticamente una soluzione corretta per renderla breve.

### Scelta delle soluzioni

Priorità a programmi e derivazioni strutturate, poi a soluzioni disponibili con controlli adeguati, infine a generazioni aggiuntive mirate.

MMFineReason viene selezionato per sorgente e difficoltà, mantenendo problemi facili e intermedi. I subset difficili rispetto a un modello di riferimento non sono automaticamente il curriculum giusto per Limite.

Il ponte testuale di Violetto può restare uno strumento: va usato solo quando la descrizione non introduce informazioni privilegiate assenti dall'input finale. La correttezza della risposta non certifica i passaggi prodotti. Non è obbligatorio rigenerare una soluzione lunga per ogni esempio.

Mantenere quote esplicite di percezione, documenti, immagini multiple, testo matematico e istruzioni. Determinare i pesi dopo l'inventario di esempi ammessi e lunghezze; imporre limiti per problema/documento e controllare i token supervisionati per sorgente.

### Criterio di passaggio

Miglioramento del ragionamento sul development a budget di inferenza controllato; spiegazioni valide nel campione indipendente; mantenimento delle capacità percettive e testuali; assenza di un aumento sistematico di affermazioni visive false. Il formato diretto deve funzionare anche in inferenza, senza dipendere da un prefill usato soltanto durante training.

## 10. F3-M — Interpolazione opzionale con Violetto originale

### Ipotesi e formula

Provare se una piccola quota dei pesi originali recupera capacità testuali mantenendo quelle visuali. È un'ipotesi da verificare, non un miglioramento garantito.

Per i parametri condivisi del decoder:

**θ_mix = (1 − λ) · θ_F3 + λ · θ_Violetto**

λ indica la quota di Violetto originale. È un'interpolazione verso l'inizializzazione, vicina a [WiSE-FT](https://arxiv.org/abs/2109.01903). Le [model soups](https://proceedings.mlr.press/v162/wortsman22a.html) classiche combinano più modelli fine-tuned compatibili.

| Componente | Regola |
| --- | --- |
| Parametri condivisi del decoder | Interpolare |
| MUDD, value embeddings, gate e altri parametri già presenti in Violetto | Includere nel primo confronto uniforme |
| ViT | Conservare quello F3 |
| Proiettore/merger | Conservare quello F3 |
| Nuovi parametri esclusivamente visivi | Conservare quelli F3 |

Compatibilità richiesta: stessa identità e significato dei parametri, forme corrispondenti, tokenizer allineato, corretta gestione di embedding e testa di output condivisi. Nuovi canali o modifiche architetturali non si interpolano con parametri inesistenti.

Non combinare direttamente i rami SigLIP e Qwen. Ciascun encoder deve rimanere associato al proprio collegamento.

### Protocollo minimo

1. Conservare F3 non modificato come controllo.
2. Provare λ = 0, 0.05, 0.10, 0.20.
3. Valutare testo e visione sullo stesso development.
4. Raffinare l'intervallo soltanto se emerge un beneficio credibile.
5. Adottare la miscela solo se migliora il compromesso entro i margini prestabiliti.

Rischio principale: proiettore e decoder si sono adattati insieme; riavvicinare il decoder a Violetto può peggiorare l'interpretazione delle feature visive. Nessuna garanzia che interpolare il 10% dei pesi produca un cambiamento del 10% nel comportamento.

Misurare anche rispetto delle istruzioni, brevità appropriata e documenti: l'interpolazione può ripristinare lo stile lungo di Violetto. Selezionare nella modalità di inferenza destinata all'uso finale. Conservare un registro dei tensori interpolati e mantenuti, inclusi MUDD, gate, value embeddings e righe visive nuove; usare precisione adeguata e verificare `λ=0`, pesi legati, identità dei moduli congelati e salvataggio/ricaricamento. Lo stesso contratto vale per F4-M sui moduli aggiornati dal RL.

Se la miscela viene adottata prima del RL, diventa l'inizializzazione e il riferimento del RL. Gli stati dell'ottimizzatore non si mediano come i pesi: il nuovo training inizializza stati coerenti con il checkpoint scelto.

## 11. F4 — Reinforcement learning con ricompense verificabili

### Inizializzazione e parametri

Partire dal checkpoint F3 scelto, eventualmente interpolato. Usare GRPO come algoritmo iniziale proposto, con ViT congelato e aggiornamento di decoder e proiettore. Validare prima compatibilità del modello con rollout, log-probabilità, maschere e aggiornamenti.

Il wrapper attuale non espone direttamente l'interfaccia completa richiesta da molti trainer RL e usa hook per le feature visive. Definire e collaudare replica/riordino delle immagini nei gruppi di generazioni, log-probabilità dei soli token generati, maschere, stop e versioni della policy di rollout, della policy aggiornata e del riferimento. `num_return_sequences=4` non garantisce da solo la corretta espansione delle feature. Prima dell'update verificare la parità delle log-probabilità fra rollout e ricalcolo a parità di policy e input.

Il riferimento congelato deve conservare anche il proiettore pre-RL e gli altri collegamenti visivi aggiornabili: non condividerli come moduli mutabili con la policy. Si possono riusare feature del ViT congelato se immagini e preprocessing sono identici; le feature già proiettate non possono restare costanti quando si aggiorna il proiettore. Misurare il costo effettivo del percorso multimodale; la compatibilità con un motore testuale o TRL non è automatica.

La supervisione precedente deve aver costruito le capacità di base. Il RL non sostituisce la copertura dei dati o la qualità dei riferimenti. La sequenza SFT iniziale e RL ha precedenti come [DeepSeek-R1](https://arxiv.org/html/2501.12948v1), ma gli effetti su Limite devono essere misurati.

### Dati ammessi

- DeepVision-103K e ViRL39K, dopo i controlli.
- Geometria con programmi o verificatori affidabili.
- Grafici e tabelle con risposte calcolabili dai dati sorgente.
- Sintetici con stato esatto e corrispondenza visiva controllata.
- Matematica testuale con riferimenti affidabili.
- Altri compiti percettivi soltanto dove il correttore è sufficientemente definito.

Non utilizzare un giudice LLM come fondamento generale della verità per questa prima fase RL.

### Verificatori

Ogni problema porta riferimento, verificatore e versione, tolleranze, unità, dominio e provenienza.

- Per soluzioni matematiche: equivalenza nel dominio corretto, insiemi di soluzioni e condizioni.
- Per numeri: unità, arrotondamenti e tolleranze coerenti con il problema.
- Per scelta multipla: corrispondenza fra opzione e risposta, anche quando cambia l'ordine.
- Per OCR: fedeltà di trascrizione, senza sostituirla con la sola equivalenza matematica.
- Per punti o strutture: associazione, tolleranze e completezza delle annotazioni.
- Per risposte impossibili: ammissione solo quando l'insufficienza è realmente stabilita.

Il verificatore deve gestire parsing fallito, timeout e ambiguità senza assegnare premi ingiustificati. Testarlo con risposte equivalenti, risposte quasi corrette, unità errate, soluzioni estranee al dominio e output con più risposte incompatibili. L'estrattore deve rispettare il contratto di risposta diretta/ragionata, senza richiedere sempre un CoT (§5.10).

Prima dei rollout fissare la politica per ciascun esito. Un errore tecnico ammette retry limitati e, se persiste, esclusione controllata o sospensione del lotto; registrare l'effetto sui gruppi GRPO e sulla distribuzione, evitando selezione opportunistica delle sole risposte verificabili. Una violazione di un formato semplice, dichiarato e collaudato segue una regola deterministica senza premio di correttezza; è distinta da una risposta valida che il correttore non sa giudicare. Non assegnare automaticamente `errato` a ogni timeout o limite semantico del verificatore. Se i casi non determinati dipendono sistematicamente dalle risposte del modello o la politica è aggirabile, correggere il verificatore o sospendere quella famiglia dal RL; può restare nella SFT se soddisfa i relativi criteri. Verificare questo comportamento prima di ottimizzare il reward.

### Ricompensa e campionamento

Come base usare correttezza verificata, con 1 per risposta corretta e 0 per risposta errata sui compiti binari appropriati. Qualunque ricompensa continua per compiti strutturati deve avere una definizione e una calibrazione proprie.

Il formato da solo non produce il premio di una risposta corretta. Evitare inizialmente penalità generali sulla lunghezza: possono scoraggiare ragionamenti necessari. Una spiegazione falsa con risultato corretto può ancora superare il reward finale; controllare questo limite con valutazioni indipendenti.

Partire con 4 generazioni per problema; aumentare solo quando utile al segnale di apprendimento. Stimare la difficoltà su Limite. Con reward identico nel gruppo, l'advantage relativo centrato è zero; possono restare i termini KL o supervisionati. Questo non dimostra che il dato sia inutile o errato; quattro fallimenti non provano l'impossibilità di apprendere il problema. Registrare gruppi informativi, token generati, troncamenti, tempo dei verificatori e costo per aggiornamento utile.

Prima del pilota fissare libreria/versione e variante GRPO: normalizzazione di advantage e loss, clipping, riuso dei rollout, aggiornamento della policy precedente, uso e coefficiente della KL, peso della componente supervisionata, gruppi a varianza zero e trattamento delle risposte troncate. La scelta di quattro campioni e reward 0/1 non specifica l'intero algoritmo. Una politica di troncamento è necessaria anche senza una penalità generale sulla lunghezza; i suoi parametri restano da calibrare, non vengono ereditati da default mobili.

Mantenere matematica testuale e una componente supervisionata di conservazione delle capacità. Controllare deriva rispetto al checkpoint iniziale, KL, entropia, lunghezze e fallimenti del verificatore. La regolarizzazione non garantisce da sola l'assenza di regressioni.

### Criterio di adozione

Miglioramento di development non limitato al reward di training, a budget di inferenza confrontabile; mantenimento di OCR, documenti, immagini multiple e testo; controllo delle scorciatoie testuali con problemi che richiedono l'immagine.

Per misurare la dipendenza visiva usare controfattuali e rimozione selettiva di immagini, oltre all'immagine bianca che introduce anche una distribuzione anomala. Sviluppare il correttore su casi separati da quelli che ne misurano la qualità. F4 deve dimostrare un vantaggio rispetto a F3 a costo di inferenza comparabile; la sua implementazione e la misura dei suoi costi non bloccano la preparazione F0–F3.

Se F4 non migliora il compromesso, il checkpoint F3 rimane un candidato valido al modello finale.

## 12. F4-M — Interpolazione opzionale dopo RL

Prima di tornare a Violetto originale, confrontare F4 con il checkpoint multimodale esatto da cui è iniziato il RL:

**θ_mix = (1 − λ) · θ_F4 + λ · θ_preRL**

Interpolare coerentemente i moduli aggiornati nel RL, quindi decoder e proiettore nella ricetta iniziale. Il ViT congelato deve risultare identico; nuovi parametri visuali condivisi seguono la stessa regola di compatibilità.

Usare inizialmente la stessa piccola griglia λ = 0, 0.05, 0.10, 0.20, conservando anche i checkpoint puri come riferimenti. Decidere sul development, poi valutare una volta il candidato finale sul test riservato.

L'interpolazione verso l'inizializzazione ha precedenti nel post-training dei modelli linguistici, fra cui [WARP](https://arxiv.org/abs/2406.16768). Questi risultati non dimostrano il beneficio nel RL multimodale di Limite.

Non costruire inizialmente una ricerca combinatoria di pesi per layer, moduli e tre checkpoint contemporaneamente. L'interpolazione è un esperimento di selezione, non una nuova fase obbligatoria di training.

## 13. Dati sintetici propri

Riutilizzare il generatore esistente prima di ampliarlo. Sono già presenti coordinate, poligoni, rette/parabole, figure separate e coppie controfattuali. [synth.py](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/synth.py).

Ordine di estensione:

1. Variazioni controllate di notazione, stile, etichette, risoluzione e impaginazione.
2. Tabelle e grafici statistici da dati sorgente.
3. Funzioni e problemi quantitativi con soluzioni eseguibili.
4. Documenti a pagina singola con testo, formule e tabelle (riferimenti fra pagine rimandati, v1.5).
5. Solidi, proiezioni e sezioni da scene controllate.
6. Altri diagrammi scientifici solo dove esiste un riferimento verificabile.

Ogni generazione deve conservare stato, seed, versione del renderer, domanda, risposta e verificatore. Generatore e verificatore non devono condividere ciecamente lo stesso errore: introdurre controlli indipendenti e casi con risultato noto.

Per insegnare l'insufficienza dei dati, verificare che l'informazione rimossa non sia deducibile da ciò che rimane. Per funzioni e geometria, evitare domande la cui risposta è nota al generatore ma non identificabile dall'immagine.

I dati reali restano necessari per fotografie, scansioni, rumore e impaginazioni non prodotte dai renderer. Ogni gruppo di varianti della stessa istanza rimane nello stesso split.

## 14. Impostazioni tecniche iniziali

Questa tabella contiene ipotesi da calibrare nei piloti, non valori finali.

| Fase | Ottimizzatore di partenza | Learning rate indicativo | Contesto |
| --- | --- | --- | --- |
| F1 | AdamW | Proiettore nell'ordine di 1e-3 | Prevalentemente 4k |
| F2 | AdamW, gruppi separati | Decoder 1e-5; proiettore 1e-4; ViT 1e-6 | Circa 8k, poi più lungo dove necessario |
| F3 | AdamW | Inferiori o uguali a F2, da calibrare per modulo | Bucket brevi e lunghi, fino al contesto validato |
| F4 | AdamW | Aggiornamenti piccoli, da calibrare su KL, reward e regressioni | Budget separati per input e generazione |

NorMuon è già disponibile nel progetto e rimane un'alternativa. Per il confronto degli encoder usare lo stesso ottimizzatore e la stessa procedura di selezione.

Altri requisiti:

- Warmup e schedule espliciti; stati dell'ottimizzatore coerenti quando cambiano i moduli addestrabili.
- Gradient clipping, monitoraggio per modulo e gestione delle anomalie numeriche.
- Precisione scelta dopo verifiche nuove, senza dedurla dai vecchi risultati.
- Checkpointing e loss a blocchi quando necessari.
- Packing senza contaminazione fra esempi.
- Loss sui target appropriati, non su descrizioni privilegiate o riferimenti nascosti.
- Contatori separati per problemi unici, immagini, token di input, token visivi e token supervisionati.
- Salvataggio di modello, configurazione, processor, tokenizer, stato di training e posizione nei dati.

Il budget visivo va espresso in token forniti al decoder oltre che in patch del ViT. Le due quantità non sono intercambiabili; il numero fisso di patch del codice attuale non è un obiettivo scientifico.

## 15. Valutazione e criteri di avanzamento

### 15.1 Misure

| Dimensione | Misure richieste |
| --- | --- |
| Formule e OCR | Fedeltà strutturale, errori di simboli e trascrizioni; formati equivalenti quando appropriato |
| Geometria | Estrazione dei fatti distinta dalla soluzione matematica |
| Grafici e tabelle | Lettura dei dati distinta dai calcoli |
| Documenti | Struttura, ricerca dell'informazione e collegamento fra regioni della pagina (più pagine rimandate, v1.5) |
| Immagini multiple | Associazione di riferimenti e uso congiunto degli input |
| Ragionamento | Correttezza finale e validità dei passaggi su campione indipendente |
| Dipendenza visiva | Controfattuali, immagine rimossa o alterata, sottogruppi che richiedono visione |
| Robustezza | Stili, risoluzioni, scansioni e impaginazioni non viste |
| Testo puro | Matematica e istruzioni rispetto a una nuova baseline di Violetto |
| Comportamento | Risposta diretta, spiegazione proporzionata, chiarimenti e gestione dell'insufficienza |
| Efficienza | Ore GPU, memoria, latenza, token generati e costo per problema |

Mantenere la suite matematica prevista: MathVista, MathVerse, MathVision, We-Math e DynaMath, con la decontaminazione appropriata. Affiancare Geoperception, VisOnlyQA, UniMER-Test (formule stampate; la parte manoscritta hwe è solo diagnostica, v1.5) e valutazioni documentali a pagina singola e percettive coerenti con le nuove capacità ([pannello](./docs/PANNELLO_VALUTAZIONE.md)).

Nessun benchmark aggregato viene considerato indipendente dagli originali senza controllo della provenienza. Conservare una componente nuova riservata con verità indipendente dove possibile. Generarla alla fine non la rende automaticamente nuova: la novità si misura sugli stati e sui componenti delle combinazioni con più immagini, distinguendo contenuti nuovi da nuove combinazioni di contenuti già usati ([pannello, §3.5](./docs/PANNELLO_VALUTAZIONE.md)).

### 15.2 Decisioni

Prima dei piloti fissare metriche primarie, margini accettabili di regressione, budget di inferenza e regole di arresto. Le soglie numeriche finali devono essere registrate prima di guardare i risultati comparativi; non sono ancora concordate in questo documento.

Confrontare checkpoint sugli stessi problemi e con lo stesso protocollo. Riportare intervalli d'incertezza e risultati per capacità. Distinguere pass@1, selezione fra più generazioni e voto di maggioranza: non sono la stessa prestazione né hanno lo stesso costo.

Il test finale non sceglie learning rate, durata, encoder o coefficienti della soup. Le decisioni usano il development.

### 15.3 Pronto per un esperimento e certificato

**Pronto per un esperimento:** provenienza e split controllati, manifest applicato ai compiti/varianti effettivamente caricati, preprocessing fissato, controlli automatici completati, casi incerti gestiti, copertura documentata e audit campionario informativo. Scarti successivi all'ammissione e distribuzione consumata sono contabilizzati. Rimangono esplicite le proprietà non verificate.

**Certificato rispetto a una soglia dichiarata:** proprietà e quantità da stimare fissate, popolazione/versione e denominatore dichiarati, trattamento degli incerti prestabilito, campione probabilistico adeguato e riferimento la cui accuratezza sia sostenuta rispetto alla soglia. Usare unità, probabilità e pesi corretti, con limite di confidenza sotto la soglia. Più domande sulla stessa immagine non valgono automaticamente come osservazioni indipendenti; la quota di QA errate non coincide con la quota di documenti con almeno un errore. Se l'attendibilità del riferimento resta un'assunzione non sufficientemente verificata, presentare una stima condizionata o un audit informativo (§5.9).

Per un semplice campione indipendente con zero errori, circa 300 esempi possono sostenere un limite superiore unilaterale vicino all'1% al 95%. Non confondere questo valore con l'estremo di un intervallo bilaterale. Questo non certifica automaticamente tutte le categorie, tutti gli strati o l'intero mix. Campionamento stratificato, pesi e confronti multipli richiedono il calcolo appropriato; il campione di stima è distinto da quello diagnostico, come in §5.9.

Integrità degli artefatti, qualità delle etichette e prestazioni del modello sono tre verifiche diverse. Un certificato di integrità non sostituisce le altre.

## 16. Budget e contabilità

Non esiste ancora un tetto concordato di ore GPU, euro, durata o token di training. L'infrastruttura disponibile non viene interpretata come un budget economico deciso.

La priorità economica già definita riguarda soprattutto i token degli agenti per pulire i dati. I costi devono essere separati:

| Attività | Contabilità |
| --- | --- |
| Controlli deterministici | Ore CPU, RAM, storage; zero token LLM |
| Audit mirati | Chiamate, token di input/output e costo del modello usato; separatamente tempo/costo della revisione competente |
| Generazione di nuovi target | Token generati, ore GPU o costo API; distinta dall'audit |
| F1/F2/F3 | Ore GPU, token elaborati e supervisionati, problemi unici |
| F4 | Rollout, verifica, aggiornamenti; costi misurati separatamente |
| Interpolazioni | Lettura/scrittura dei pesi e valutazioni; nessun training completo aggiuntivo |
| Valutazione | Ore GPU, numero di problemi, generazioni per problema e lunghezze |

Il preventivo di ogni fase nasce da un benchmark rappresentativo della relativa configurazione. Non estrapolare il throughput di F1 con ViT congelato a F2 con ViT addestrabile, né il costo di SFT a quello del RL.

Il preventivo dell'audit fissa per sorgente tetto di chiamate/token, revisione competente prevista, dimensione e disegno del campione di stima e regola di arresto delle verifiche aggiuntive. Se si raggiunge il limite, restringere la proprietà dichiarata, mantenere target controllati quando ammissibile oppure rinviare il sottoinsieme; non forzare il consenso o ridurre il campione mantenendo invariata una promessa di certificazione. Le soglie numeriche restano da concordare prima dell'esecuzione, senza trasformare gli esempi statistici di §5.9 in obiettivi già approvati.

Una prima stima usa:

**ore della fase ≈ passi previsti × secondi misurati per passo / 3600 + valutazioni + salvataggi**

Per RL aggiungere separatamente il tempo dei rollout e dei verificatori. Per il costo monetario servono la tariffa effettiva e le altre voci pertinenti.

Quantità di esempi, numero di epoche, quote per capacità e durata vengono fissati dopo inventario e benchmark. I 300M token della bozza F1-clean non diventano automaticamente il budget della nuova pipeline.

## 17. Implementazione minima e ordine dei lavori

1. **Inventario e manifest:** identificare sorgenti effettive, sovrapposizioni, annotazioni disponibili e capacità mancanti; fissare identità dei compiti/derivati e importare le vecchie decisioni come evidenze riesaminabili.
2. **Split e controlli:** riusare parser, controlli UniMER/MAVIS, deduplicazione, provenienza e sintetici; fissare regole di ammissione, verificatori per compito e disegno dell'audit con budget (§§5.6–5.10).
3. **Caricatori:** applicare il manifest e gli split globali a QA/crop e validazione, registrare scarti e mix consumato; integrare le sorgenti prioritarie e completare esempi con più immagini, riutilizzando il processor.
4. **Trainer con ViT addestrabile:** gruppi dei parametri, precisione, checkpoint e rimozione dei conteggi di parametri rigidi incompatibili con i candidati.
5. **Piloti F0:** confronto progressivo di encoder, proiettore, layer e ingressi visivi secondo §§6.2–6.5; controlli delle interazioni, profili di costo e scelta del piano quantitativo.
6. **Run F1 e F2:** checkpoint e valutazioni per capacità, con replay esplicitamente collegato.
7. **F3:** selezione delle soluzioni e mantenimento delle altre capacità.
8. **Interpolazione F3-M:** piccolo confronto sul development, solo se i checkpoint sono compatibili.
9. **RL:** integrazione del trainer e dei verificatori, prova breve, poi eventuale estensione.
10. **Interpolazione F4-M e valutazione finale:** selezione sul development, test riservato e rapporto conclusivo.

La vecchia infrastruttura di audit congelata può restare disponibile. Non deve diventare una dipendenza da ampliare per ammettere qualsiasi nuovo esempio. I controlli di ammissione devono distinguere chiaramente dati sperimentali, certificazione statistica e integrità dei file.

Non sono previsti un grande refactoring dell'audit, un agente per esempio, generazioni sistematiche di nuovi CoT o un allargamento indiscriminato del catalogo.

## 18. Decisioni ancora aperte e deliverable

### Domande già poste all'utente

1. **Quale checkpoint Qwen usare per estrarre il ViT?** Servono famiglia, ID e revisione, oppure la delega a scegliere e motivare il candidato.
2. **Come fissare il budget?** Proposta: benchmark breve per fase, preventivo e decisione sull'investimento. In alternativa, tetto totale esplicito in ore GPU o euro.

### Parametri da decidere con l'inventario e i piloti

- Sottoinsiemi ammessi, quote per capacità e volumi.
- Risoluzione, token visivi e numero di immagini per esempio.
- Layer del ViT, fusione, compressione e dimensioni del proiettore; adozione o esclusione motivata di DeepStack e rami visivi MUDD/value embeddings.
- Learning rate, batch effettivo, durata e precisione.
- Metriche primarie, margini di non inferiorità e soglie operative.
- Quantità e popolazione delle stime di qualità, trattamento degli incerti, calibrazione dei riferimenti e budget dell'audit per sorgente.
- Verificatori ammessi al RL e relativo budget.

Questi punti non richiedono tutti una nuova domanda all'utente: le scelte tecniche ordinarie possono essere proposte sulla base delle misure. Le decisioni economiche e di ambito restano esplicite.

### Artefatti da produrre prima del run principale

- Manifest dei dati e degli split per compito/variante, con provenienza delle esclusioni e controllo del suo uso nei loader.
- Matrice capacità × difficoltà × tipo di immagine × affidabilità.
- Regole di ammissione e specifica dei controlli per compito, con copertura, falsi positivi e falsi negativi; collaudo di estrazione, selezione e verifica delle risposte.
- Disegno dei campioni diagnostico e di stima, popolazione/denominatore/pesi, gestione degli incerti, attendibilità dei riferimenti e budget; rapporto sugli scarti successivi all'ammissione.
- Rapporto del confronto di encoder, proiettori, layer e ingressi visivi, con prove isolate e interazioni effettivamente misurate.
- Specifica dell'architettura scelta: tensori, forme, normalizzazioni, maschere, punti di iniezione, gate, inizializzazione e parametri addestrabili per fase.
- Configurazioni per fase e preventivo dei costi.
- Protocollo di valutazione e registro delle decisioni.

### Artefatti finali

- Checkpoint selezionato, con encoder, proiettore e decoder coerenti.
- Processor, tokenizer, configurazione e istruzioni per l'inferenza.
- Provenienza dei dati e delle trasformazioni, senza ridistribuzioni non previste.
- Coefficienti e provenienza delle eventuali interpolazioni.
- Rapporto con prestazioni per capacità, regressioni, limiti e costo effettivo.

Una breve fase aggiuntiva di consolidamento supervisionato resta possibile soltanto se emerge una regressione specifica e un confronto ne dimostra l'utilità. Non viene aggiunta automaticamente dopo il RL.
