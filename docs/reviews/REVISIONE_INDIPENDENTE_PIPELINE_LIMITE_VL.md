# Revisione indipendente della pipeline di Limite-VL

**Data:** 27 ottobre 2026. **Oggetto:** versione 1.0 di [PIPELINE_ADDESTRAMENTO_LIMITE_VL.md](./PIPELINE_ADDESTRAMENTO_LIMITE_VL.md), letta integralmente, 582 righe. **Revisore:** agente distinto dall'autore del piano.

Il confronto con il codice riguarda esclusivamente `origin/main` al commit [`9d977b691190256d25fd0f31c0969fc24e1356ad`](https://github.com/akira-n-28/vision_limite/commit/9d977b691190256d25fd0f31c0969fc24e1356ad), letto mediante `git show`, e i file testuali pubblici dello snapshot Violetto `b1f3d572ccacb6919f4d64c321b70ba034ddaef2`. Non ho usato risultati di vecchi esperimenti o audit come prova, eseguito audit sui dataset, scaricato dati o pesi, avviato training o modificato la repository. Le osservazioni sui difetti dei caricatori derivano dall'ispezione statica: non sono misure della loro frequenza nei dati.

Hash SHA-256 del documento esaminato: `f17af2e4e9f3f59f57a9beade85b80952b7ba0fdb345114532e8cf60e0df00c3`.

## Verdetto

**La pipeline è una base scientificamente sensata per iniziare l'implementazione di F0; non è ancora una ricetta eseguibile né una giustificazione per avviare il run principale.** Conservare la sequenza allineamento → adattamento congiunto → specializzazione, senza introdurre nuove grandi fasi. Le lacune principali sono contratti operativi non ancora fissati: formato delle risposte di Violetto, estrazione Qwen, dati effettivamente consumati dal training, aggiornamento e salvataggio del ViT, uso di più immagini e specifica del RL.

Non emerge un errore che imponga di cambiare backbone, rinunciare al fine-tuning del ViT o rifare l'infrastruttura di audit. Emergono invece incompatibilità concrete fra l'obiettivo e il codice riutilizzabile. Alcune sono già riconosciute dal piano: questa revisione precisa dove diventano bloccanti e quale verifica minima serve.

Il documento è particolarmente valido nel distinguere correttezza, percepibilità e spiegazione; nel non considerare concordanza fra agenti una verità; nel limitare gli audit costosi; nel riservare il test finale; nel rendere facoltative le interpolazioni e nell'ammettere F3 come risultato finale se il RL non conviene.

**Priorità usate:** P0 = da risolvere prima del confronto che decide l'encoder o del relativo run principale; P1 = da risolvere prima della fase interessata; P2 = precisazione o miglioramento che non impedisce l'avvio della preparazione. Non sono soglie economiche o statistiche aggiuntive.

## Rilievi

### R1 — Il contratto di risposta contrasta con il template originale

**P0 per il primo adattamento del decoder. Tipo: incompatibilità confermata e lacuna del piano.** Riferimenti: §§7–9, righe 218–234, 259–261, 296–319.

Il [template dello snapshot Violetto](https://huggingface.co/paradigma-inc/limite-1b-violetto/blob/b1f3d572ccacb6919f4d64c321b70ba034ddaef2/chat_template.jinja) inserisce sempre l'istruzione di ragionare passo passo e chiudere con una risposta in `\boxed{}`; rifiuta system prompt diversi. Il processor lo usa in `apply_chat_template`. F1 inserisce `<think></think>` come prefill senza label, mentre i percorsi F2/replay chiedono sistematicamente ragionamento. Il [processor, righe 43–46 e 197–247](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/processing.py#L43) e il [replay, righe 20–38](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/replay.py#L20) lo confermano.

**Conseguenza:** target di trascrizione o risposta diretta possono contraddire il prompt; il buon comportamento osservato con prefill non dimostra che il modello sappia scegliere autonomamente quando essere breve. Non è sufficiente aggiungere SmolTalk al mix. Inoltre Violetto è già post-addestrato con SFT e RL e progettato principalmente per problemi matematici a turno singolo: istruzioni generali e dialogo sono capacità da misurare e adattare, non preservazioni automatiche di capacità già consolidate. [Scheda del checkpoint](https://huggingface.co/paradigma-inc/limite-1b-violetto/blob/b1f3d572ccacb6919f4d64c321b70ba034ddaef2/README.md).

**Correzione minima:** specificare e versionare template, delimitatori, stop token, label e politica di risposta diretta/ragionata. F1 può mantenere il prefill come strumento esplicito di allineamento. Prima degli aggiornamenti del decoder scegliere una modalità coerente di training e inferenza e includere esempi che insegnino davvero la scelta del formato. Confrontare senza prefill nascosti risposte brevi, problemi complessi e rispetto dell'istruzione; mantenere la valutazione di Violetto originale col suo template canonico. Il dialogo a più turni può essere una successiva estensione circoscritta, senza bloccare il nucleo a turno singolo con più immagini.

### R2 — Esplicitare i limiti reali di contesto e immagini multiple

**P0 per il pannello multi-immagine; P1 per F2 estesa. Tipo: limiti confermati, capacità da testare.** Riferimenti: §3.2 righe 74–87; §8 righe 263–269; §14 righe 442–462.

La [configurazione esatta di Limite](https://huggingface.co/paradigma-inc/limite-1b-violetto/blob/b1f3d572ccacb6919f4d64c321b70ba034ddaef2/config.json) dichiara 131.072 posizioni, 48 layer, 12 layer globali senza RoPE e 36 locali con `sliding_window=1024` — convenzione inclusiva, fino a 1025 chiavi. Il [processor, righe 28–61](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/processing.py#L28) impone invece un tetto software di 768 token per immagine. Questi sono tre vincoli distinti: 768 non è un limite dimostrato dell'architettura; 1024 non è il massimo contesto; 131k non certifica la comprensione multimodale a quella lunghezza.

Il [caricatore, righe 66–70 e 343–377](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/data.py#L66) conserva una sola immagine per `Sample`. Il processor accetta liste ma vieta immagini dopo testo nello stesso turno: non implementa un formato interleaved generale. Il [packing, righe 94–107](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/collator.py#L94) azzera le posizioni a ogni esempio: una riga da 32k piena di esempi brevi non allena dipendenze lunghe nello stesso documento.

**Correzione minima:** definire identità e ordine delle immagini, budget totale, budget per immagine e lunghezza dell'intero esempio inclusa la risposta. Nel primo pannello bastano coppie di immagini e pagine; misurare uso della prima/ultima pagina, scambio dell'ordine, riferimenti espliciti e dettagli piccoli. Nessun reset di posizione fra pagine dello stesso esempio. Per documenti densi, testare un'allocazione leggibile di risoluzione/crop prima di aumentare il numero di pagine. L'eventuale rimozione del tetto 768 richiede verifica mirata; non richiede automaticamente modificare l'attenzione di Limite.

### R3 — «ViT estratto da Qwen» deve diventare un artefatto precisamente definito

**P0 per il confronto encoder. Tipo: decisione legittimamente aperta, interfaccia assente.** Riferimenti: §§3.1 e 6, righe 56–68 e 182–212; §18, riga 551.

Il loader attuale istanzia esclusivamente `Siglip2VisionModel`, usa normalizzazione SigLIP2 e assume il suo formato di griglia. Non basta cambiare `vision_path`. [Modello, righe 60–77 e 190–235](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/modeling.py#L60).

Qwen2.5-VL e Qwen3-VL non sono sinonimi: cambiano encoder, patch, posizioni e merger. Per esempio [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json) specifica patch 14, merge spaziale 2 e patch temporale 2 anche nell'implementazione che gestisce immagini statiche. Qwen3-VL riparte da SigLIP2 e impiega anche DeepStack; confrontarne solo l'ultima uscita è una scelta valida ma non riproduce il sistema Qwen completo. [Rapporto Qwen3-VL, §2](https://arxiv.org/html/2511.21631v1#S2).

**Correzione minima:** fissare famiglia, ID e revisione; elencare chiavi estratte; dichiarare se il merger originale viene scartato; individuare il tensore prima del nuovo collegamento; evitare una doppia compressione 2×2. Validare preprocessing e ordine delle feature contro il percorso nativo. Confrontare problemi, supervisione, regola di tuning e token al decoder; riportare separatamente patch del ViT, parametri del collegamento, memoria e tempo. La pipeline già distingue correttamente confronto fra punti di partenza e confronto causale fra architetture: conservare questa formulazione.

Non occorre introdurre altre famiglie di encoder o una ricerca su DeepStack/MUDD prima della scelta richiesta dall'utente. Il checkpoint Qwen resta una scelta aperta, non un errore da colmare arbitrariamente in questa revisione.

### R4 — Sbloccare il ViT richiede più di `requires_grad=True`

**P0 per il secondo stadio dei piloti e per F2. Tipo: incompatibilità confermata, già riconosciuta ma da dettagliare.** Riferimenti: §8 righe 244–246; §14 righe 442–460; §17 riga 535.

`set_trainable` congela sempre il ViT. Nel trainer, tutti i parametri esterni al decoder vengono raccolti come adapter: rendendo semplicemente addestrabile il ViT, questi entrerebbe nel gruppo a LR del proiettore, oltre a violare il conteggio rigido dei parametri. I salvataggi conservano decoder e adapter, non il ViT. [Modello, righe 140–172](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/modeling.py#L140), [trainer, righe 72–98 e 192–216](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/scripts/train_f2.py#L72).

**Correzione minima:** gruppi espliciti e disgiunti per decoder, ViT, proiettore e righe dei token visivi; precisione e stati master coerenti; checkpoint completo e prova salva/ricarica/riprendi. Verificare variazioni dei pesi e gradienti per modulo, non solo la diminuzione della loss. Il forward ha già un ramo con autograd quando il ViT non è congelato: è una base riutilizzabile, non serve riscriverlo integralmente.

Gli LR indicativi `1e-5 / 1e-4 / 1e-6` sono ipotesi ragionevoli, **non errori scientifici dimostrati**. A LR molto piccolo controllare il cambiamento effettivo del ViT rispetto alla precisione utilizzata; a LR maggiore controllare perdita di capacità preaddestrate. L'obbligo «F3 ≤ F2» va inteso come punto di partenza prudente, non legge generale. Un confronto breve e mirato col congelamento è utile se il contributo dell'aggiornamento visivo resta incerto; non occorre aggiungere automaticamente una terza grande campagna.

### R5 — Il replay va definito per token effettivi, lunghezza e comportamento

**P1 prima di F2; P0 se fa parte del confronto pilota. Tipo: comportamento del codice confermato, rischio di conservazione.** Riferimenti: §8 riga 261; §9 righe 305–315; §14 righe 445–459.

La [miscela, righe 204–208 e 238–250](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/stream.py#L204) campiona per esempio; la [loss, righe 287–310](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/scripts/train_f2.py#L287) è mediata sui token supervisionati. Quindi una percentuale di problemi testuali non corrisponde alla stessa percentuale di loss: in prima approssimazione il contributo dipende dal peso di campionamento moltiplicato per la lunghezza media del target. Un lungo CoT può dominare numerose risposte percettive brevi.

Inoltre il [collator, righe 68–72](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/collator.py#L68) scarta interamente esempi oltre `max_len`. Il replay esistente [genera anche tracce lunghe](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/scripts/replay_generate.py#L37) e le [seleziona per limite di lunghezza e risposta finale](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/scripts/replay_select.py#L44). Non certifica il ragionamento e non costituisce un dataset generale di istruzioni. Nei file `config/runs` esaminati non compare una sorgente `text_replay`.

**Correzione minima:** registrare quote effettive di target-token, problemi unici, lunghezze, scarti e ripetizioni per sorgente; aggiungere esplicitamente il replay ai mix. Se il limite iniziale 8k esclude la componente da preservare, usare un bucket compatibile o ridurre la promessa di conservazione finché misurata. Non troncare soluzioni corrette alla cieca e non rigenerare tutto. Separare la conservazione del ragionamento matematico dall'apprendimento di istruzioni e risposte brevi; valutare entrambe contro una nuova baseline.

### R6 — I target devono avere una semantica controllabile per ciascun compito

**P0 per l'ammissione del pannello F0/F1; P1 per dati nuovi. Tipo: rischio noto, contratto incompleto.** Riferimenti: §§4–5, righe 93–171; §7 righe 222–226; §13 righe 432–436.

Il piano formula correttamente le tre proprietà indipendenti. L'implementazione MAVIS, tuttavia, [deriva la verità dalla concordanza fra caption e specifica, righe 23–54](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/f1_mavis.py#L23); non controlla i pixel. Lo [schema, righe 35–42](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/f1_schema.py#L35), converte alcuni razionali in decimali a due cifre. Questo non dimostra un errore nei dati, ma impone di distinguere valore esatto della costruzione, testo effettivamente stampato e target richiesto.

**Conseguenza:** una domanda su una misura visibile, una trascrizione e un problema che richiede il valore esatto possono avere risposte diverse. Una proprietà del generatore non leggibile, una tabella sorgente con precisione maggiore del grafico o un'equivalenza matematica che cambia la notazione possono produrre supervisione impossibile o una metrica sbagliata.

**Correzione minima:** per ogni adattatore sorgente→compito documentare input disponibile, target, trasformazioni e criterio di correttezza. Conservare separatamente stato esatto e contenuto reso. Distinguere dati espliciti, convenzioni del disegno e conclusioni inferite. Applicare lo stesso principio a LaTeX, coordinate, ordine di lettura, unità e arrotondamenti. Verificare campioni anche dopo il resize/crop effettivamente usato dal modello; la leggibilità dell'originale non basta. È lavoro mirato sui compiti prioritari, non una richiesta di audit LLM esaustivo.

### R7 — Gli split vanno applicati attraverso tutte le sorgenti prima della selezione

**P0 per un confronto interpretabile. Tipo: carenza del codice confermata; contaminazione specifica non dimostrata.** Riferimenti: §5.5 righe 173–179; §15 righe 482–492.

Lo [stream, righe 135–167 e 265–273](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/stream.py#L135) riserva un prefisso di uno shard. Lo [script dei duplicati](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/scripts/split_duplicates.py#L81) lavora per sorgente su hash esatti di immagini/target; la [decontaminazione visiva](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/decontam.py#L1) combina pHash e n-gram. Sono strumenti utili, non garanzie per ritagli, ridisegni, traduzioni o lo stesso problema arrivato da raccolte diverse.

**Correzione minima:** assegnare un'identità comune di problema/immagine/documento e propagare lo split a caption, QA, soluzioni, varianti e replay; eseguire controlli fra sorgenti, non solo dentro ciascuna. Dare precedenza agli ID/provenienze disponibili e usare similarità come ricerca di candidati. Non confondere l'assenza di contaminazione aggiunta dal progetto con l'assenza di esposizione pregressa dei backbone.

Prima dell'ammissione definitiva dei dati scegliere anche i pannelli documentali e multi-immagine: «valutazioni coerenti» è troppo generico per decontaminare preventivamente. Il test nuovo riservato è utile proprio per l'esposizione non ricostruibile dei backbone. Per MathVerse, DynaMath e più QA sulla stessa immagine, gli intervalli e i confronti appaiati devono raggruppare per problema originario; il numero di varianti non è il numero di osservazioni indipendenti.

### R8 — Un campione intenzionale per trovare errori non certifica il residuo

**P1 per dichiarazioni di certificazione; P2 per un normale pilota. Tipo: buona impostazione, raccordo statistico da precisare.** Riferimenti: §5.4 righe 163–171; §15.3 righe 494–502.

Conservare la distinzione «pronto per un esperimento»/«certificato». Per trovare errori conviene sovracampionare casi difficili; per stimare il tasso residuo servono probabilità di inclusione note nel campione della popolazione finale ammessa. Riparare e filtrare in base al campione e poi usare gli stessi casi come conferma indipendente può dare una stima ottimistica.

**Correzione minima:** tenere separati campione diagnostico e campione di stima, oppure usare un disegno che tenga esplicitamente conto della selezione; precisare denominatore, unità, strati e trattamento degli incerti. Il secondo verificatore sul 10% delle approvazioni è un riferimento operativo, non una dimostrazione automatica di accuratezza; calibrazione e volume vanno motivati rispetto alla proprietà cercata e al costo.

Precisazione numerica: con zero errori su 300 prove indipendenti, il limite binomiale **unilaterale** al 95% è `1 − 0,05^(1/300) ≈ 0,994%`. La funzione [Clopper–Pearson del progetto, righe 276–283](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/stem_audit.py#L276) è bilaterale: il suo estremo superiore è circa 1,22%. Non è necessario scegliere qui l'una o l'altra convenzione, ma non vanno confuse.

Non riutilizzare automaticamente `f2_certify.py` come certificatore generale: contiene correzioni per sensibilità su errori artificiali e un'approssimazione per cluster. La trasferibilità agli errori naturali e la copertura congiunta delle incertezze richiedono giustificazione. La pipeline già lo evita concettualmente; mantenerlo fuori dal percorso obbligatorio di F0 è appropriato.

### R9 — Il catalogo è utile, ma manca un nucleo iniziale ordinato per copertura e costo

**P1 per l'organizzazione di F0; non blocca l'inventario. Tipo: decisione operativa aperta.** Riferimenti: §4 righe 90–128; §7 riga 222; §17 righe 532–536.

La tabella non scambia layout con QA né dataset generati con verità formale: bene. La [lista dei loader](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/data.py#L320) mostra però che molto del nuovo catalogo richiede ancora adattatori. Se «integrare» significa completare tutte le sorgenti prima dei piloti, F0 può diventare il progetto più grande dell'intera pipeline.

**Correzione minima:** scegliere dall'inventario il minor insieme che copra formule stampate/manoscritte, scene, geometria, grafici/tabelle e documenti, più un piccolo pannello a due immagini. Introdurre nuovi dati quando colmano una lacuna misurabile. PubTabNet/DocLayNet non coprono da soli il ragionamento documentale; Docmatix non certifica automaticamente l'uso congiunto delle pagine; M4-Instruct include famiglie non tutte in ambito. Le rispettive [scheda Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) e [descrizione M4-Instruct](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/) giustificano selezione per sottoinsieme.

Nel manifest registrare origine a monte e condizioni d'uso disponibili, non soltanto la licenza dichiarata da un aggregatore. Il [catalogo del repository](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/downloads/sources.tsv#L18) segnala già incertezze per alcune sorgenti: non risolverle per inferenza. Questo riguarda l'ammissione della singola sorgente e il rilascio, non giustifica fermare tutto l'inventario.

### R10 — Nel breve confronto encoder controllare anche il grado di adattamento

**P1 per la decisione di F0. Tipo: rischio sperimentale; il disegno di base è corretto.** Riferimenti: §6.2 righe 194–208.

I due stadi e il controllo del costo sono appropriati. Un pilota troppo breve, però, può scegliere il ramo che si allinea più rapidamente al nuovo proiettore. La stessa durata e gli stessi problemi non garantiscono che entrambi abbiano raggiunto uno stadio comparabile di adattamento; una singola inizializzazione del proiettore può contare quando la differenza è piccola. È un rischio, non prova che il confronto proposto sia distorto.

**Esperimento minimo:** conservare curve e checkpoint intermedi, verificare che la scelta non sia determinata soltanto dalla velocità iniziale di convergenza, usare confronti appaiati per capacità. Quando la differenza è incerta ripetere o estendere soltanto il tratto discriminante, come già previsto. Esplicitare in anticipo la regola che combina qualità, regressioni e costo. Non serve imporre ora un numero universale di seed, esempi o ore GPU.

### R11 — La F4 richiede una vera integrazione multimodale, non solo un trainer GRPO

**P1 bloccante prima di F4; non blocca F0–F3. Tipo: incompatibilità confermata e verifiche da eseguire.** Riferimenti: §11 righe 359–361; §17 riga 540.

Il [wrapper attuale](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/modeling.py#L248) restituisce hidden state e loss, non l'interfaccia standard completa con logits richiesta da molti trainer; delega `generate()` al decoder dopo aver predisposto le feature mediante hook. Non è automaticamente compatibile con TRL o con il plugin vLLM testuale di Limite.

Un caso concreto da coprire: le feature vengono predisposte una volta e inserite con `masked_scatter`; il wrapper non espande le immagini quando il generatore espande il batch per più sequenze. Chiamare direttamente `num_return_sequences=4` non equivale a un rollout multimodale corretto. Anche se si risolve ripetendo le chiamate, resta da misurare il costo. [Hook e generazione, righe 175–184 e 271–279](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/modeling.py#L175).

**Correzione minima:** contratto esplicito per gruppi di generazioni, immagini replicate/riordinate, log-probabilità dei soli token generati, maschere, stop, policy precedente e riferimento congelato. Verificare la parità fra log-probabilità del rollout e del ricalcolo prima dell'update. Se il proiettore viene aggiornato, il riferimento deve conservare il suo proiettore pre-RL: non può condividere una proiezione mutabile con la policy. Le feature del ViT congelato possono essere riusate quando preprocessing e input sono identici; quelle proiettate non vanno riusate come costanti durante un aggiornamento del proiettore.

### R12 — «GRPO, quattro generazioni, reward 0/1» non definisce ancora la ricetta

**P1 prima del pilota RL. Tipo: specifica incompleta, non errore nella scelta dell'algoritmo.** Riferimenti: §11 righe 389–403; §16 righe 516–526.

Mancano normalizzazione degli advantage e della loss, clipping, numero di riusi dei rollout, coefficiente/uso della KL, trattamento dei gruppi a varianza zero e delle risposte troncate, peso della componente supervisionata. Queste scelte incidono sia sul ragionamento lungo sia sul costo. La documentazione [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer) e il lavoro [DAPO](https://arxiv.org/html/2503.14476v2) descrivono differenze sostanziali fra formulazioni e trattamento delle sequenze: fissare versione e opzioni, senza ereditare default mobili.

Con reward binario identico nel gruppo, il termine di vantaggio relativo centrato è **zero**, non solo «piccolo»; possono restare i termini KL e SFT. Con quattro campioni indipendenti e probabilità di successo `p`, la probabilità di un gruppo misto è `1 − p^4 − (1−p)^4`: circa 34% a `p=0,1`, circa 4% a `p=0,01`. È un calcolo illustrativo, non una stima di Limite; correlazione e distribuzione dei problemi vanno misurate.

**Correzione minima:** mantenere quattro generazioni come punto di partenza proposto, registrare gruppi informativi, token generati utili e sprecati, troncamenti e tempo dei verificatori. Definire esplicitamente la gestione dei timeout: errore d'infrastruttura e risposta matematicamente errata non sono la stessa osservazione. Niente penalità generale di lunghezza iniziale è sensato, ma non sostituisce una politica per le risposte che raggiungono il limite. Estendere F4 soltanto se produce valore oltre F3 a costo di inferenza confrontabile.

### R13 — Il verificatore esistente non soddisfa ancora la semantica prevista per F4

**P1 prima di usare una famiglia di reward. Tipo: limite confermato, già coperto dal requisito del piano.** Riferimenti: §11 righe 376–393.

Il correttore [attuale, righe 109–154](https://github.com/akira-n-28/vision_limite/blob/9d977b691190256d25fd0f31c0969fc24e1356ad/src/limite_vl/f2.py#L109) estrae l'ultimo `\boxed{}` e confronta lettere, stringhe o espressioni tramite `math_verify`. Non riceve come contratto generale unità, dominio, tutte le risposte proposte o condizioni del problema. La selezione del replay verifica il risultato, non la spiegazione.

**Conseguenza:** riutilizzare tale funzione come reward universale non implementerebbe le promesse della pipeline. La correttezza dell'ultimo riquadro può convivere con alternative incompatibili e una spiegazione falsa; un solver su annotazioni perfette può premiare una risposta non ricavabile dall'immagine.

**Correzione minima:** piccoli verificatori per famiglie ammesse, con casi corretti/equivalenti, quasi corretti, unità e domini errati, più risposte, fallimenti del parser. Per la dipendenza visiva usare coppie controfattuali e rimozione di singole immagini, oltre al confronto con immagine bianca: quest'ultimo introduce anche una distribuzione anomala. Non chiedere al reward di certificare tutte le spiegazioni; misurarle separatamente e mantenere esclusi dal primo RL i compiti senza verifica affidabile. I benchmark del reward devono essere separati dai casi con cui lo si sviluppa.

### R14 — Interpolazioni corrette come ipotesi; servono artefatti e criteri coerenti

**P2; P1 quando si esegue l'interpolazione. Tipo: proposta valida con precisazioni.** Riferimenti: §§10 e 12, righe 321–355 e 405–417.

Le formule hanno il verso corretto: `λ` è la quota del checkpoint di origine. È sensato interpolare con Violetto solo i parametri condivisi del decoder e, dopo RL, usare prima il checkpoint multimodale esatto pre-RL interpolando i moduli effettivamente aggiornati. È corretto non combinare direttamente rami SigLIP e Qwen, non mediare gli stati dell'ottimizzatore e conservare i controlli puri. [WiSE-FT](https://arxiv.org/abs/2109.01903) e [WARP](https://arxiv.org/abs/2406.16768) sono precedenti, non garanzie per questo sistema.

**Rischio aggiuntivo:** tornare verso Violetto può recuperare ragionamento matematico ma anche la tendenza originaria a risposte lunghe e a reinterpretare istruzioni. Le metriche della soup devono quindi includere comportamento e documenti, non soltanto matematica testuale e accuracy visiva. Dopo una modifica del template, valutare entrambe le componenti nella modalità di inferenza effettivamente destinata al prodotto.

**Correzione minima:** registro dei tensori interpolati/conservati, inclusa una decisione esplicita su `vision_token_embeds`, MUDD, gate, value embeddings e pesi condivisi embedding/testa. Calcolare la miscela in precisione adeguata; verificare `λ=0`, identità dei moduli congelati, salvataggio/ricaricamento e corrispondenza dei pesi legati. La piccola griglia proposta è sufficiente per un primo test e resta facoltativa. Non aumentarla prima che emerga un beneficio credibile.

## Cosa conservare, modificare, rendere facoltativo o rimandare

| Elemento | Valutazione |
| --- | --- |
| F1 con encoder/decoder congelati e nuovo collegamento | Conservare; obiettivo percettivo e criteri d'arresto sono appropriati. |
| F2 con ViT addestrabile e LR distinto | Conservare come ricetta principale proposta; completare gruppi, checkpoint e verifica degli aggiornamenti. |
| F3 con ViT inizialmente congelato | Conservare; rappresenta un cambio di curriculum, non richiede un nuovo trainer o un nuovo formato incompatibile. |
| Dati deterministici prima degli audit | Conservare come priorità economica; audit selettivi e misura del residuo hanno ruoli diversi. |
| Replay da F2 | Conservare, collegarlo alle configurazioni e misurarlo per target-token e lunghezze. |
| Matematica, documenti, immagini multiple, inglese prevalente | Conservare come obiettivi; limitare il primo insieme operativo senza promettere tutta la lista STEM. |
| Istruzioni e risposta proporzionata | Modificare il contratto operativo di template/supervisione; non rimandarlo alla fine di F3. |
| Dialogo esteso, interleaving generale, 3D e nuove grandi raccolte | Rimandare finché una lacuna del nucleo ne giustifica l'integrazione. |
| F4 | Conservare come esperimento condizionato; F3 può restare il prodotto finale. |
| F3-M e F4-M | Facoltativi; controllo del compromesso, senza ricerca combinatoria. |
| Certificazione statistica generale | Facoltativa per avviare un esperimento; obbligatoria soltanto se si vuole fare la specifica dichiarazione di qualità. |
| Nuove iniezioni visive in MUDD/XSA/value embeddings | Rimandare alle ablation motivate dopo la baseline; nessuna evidenza qui obbliga a introdurle. |

La sequenza del documento è compatibile con il punto di partenza già post-addestrato, purché F2/F3 siano trattate come adattamento e conservazione di una policy esistente. Non occorre ricreare da zero una capacità di ragionamento che Violetto possiede già, né presumere che un dataset con CoT più lunghi la migliori.

## Blocchi e ordine pratico

**Si può iniziare subito l'implementazione preparatoria di F0:** inventario, specifica dei record e degli split, contratto di formato, adattatori minimi e test del percorso dati. La scelta Qwen e il budget totale non impediscono queste attività. L'assenza di un totale GPU/euro concordato non autorizza a inventarlo.

**Prima di eseguire il confronto F0** servono: candidato Qwen fissato, estrazione e preprocessing funzionanti, pannello e split, contratto comune di supervisione/inferenza, trainer con ViT addestrabile e checkpoint completo, metriche e budget del pilota dichiarati. Il solo ramo SigLIP può essere preparato tecnicamente prima, ma non può essere dichiarato vincitore senza il confronto richiesto.

**Prima del run principale F1/F2** servono scelta dell'encoder, dati iniziali ammessi, configurazioni effettive, verifiche di salvataggio/ripresa e costo misurato della configurazione. Le percentuali del mix, i margini di regressione e la durata possono essere proposti dalle misure; non richiedono tutti una nuova decisione d'ambito dell'utente.

**F4 non deve bloccare F0–F3.** Il requisito di §6.3 di avere «misura dei costi per fase» va interpretato progressivamente: prima dei piloti sono misurabili le configurazioni implementate; F3 e soprattutto F4 richiederanno un proprio benchmark prima dell'investimento. Pretendere già in F0 una misura esatta del RL implicherebbe implementarlo in anticipo, in conflitto con l'ordine minimo di §17.

Non sono blocchi iniziali: un certificato universale dei dati, tutta la tassonomia STEM già coperta, 32k multimodali già dimostrati, tutti i sottoinsiemi del catalogo integrati, né il successo delle soup. Sono blocchi della relativa conclusione le prove che mancano: non chiamare «conservato», «multi-immagine», «certificato» o «migliore» ciò che il protocollo non ha ancora misurato.

## Checklist breve delle decisioni e validazioni

- [ ] Fissare checkpoint/revisione Qwen e confine ViT–merger; validare la conversione contro il percorso nativo.
- [ ] Versionare template, formato diretto/ragionato, label e stop; verificare il comportamento senza prefill nascosti.
- [ ] Chiudere pannello F0, gruppi train/dev/test e provenienza trasversale; scegliere metriche e regola di confronto prima dei risultati.
- [ ] Verificare gradienti, LR effettivi, precisione e round-trip dei checkpoint di tutti i moduli, incluso il ViT.
- [ ] Collegare replay e dati prioritari; contare target-token, scarti per lunghezza e uso reale di più immagini nello stesso esempio.
- [ ] Misurare costo dei piloti e definire l'investimento della fase successiva, separando audit, target generation, training e valutazione.
- [ ] Solo prima di F4: fissare verificatori e variante GRPO; validare rollout/log-probabilità/riferimento, troncamenti e costo dei gruppi informativi.
- [ ] Se si provano interpolazioni: usare checkpoint immutabili, verificare i tensori e selezionare sul development prima dell'unica valutazione finale riservata.

**Esito della revisione:** procedere con la preparazione e con correzioni circoscritte; subordinare i piloti comparativi e il run principale ai contratti e alle verifiche indicati. Nessuna evidenza raccolta giustifica una riprogettazione generale o audit sistematici con un agente per esempio.
