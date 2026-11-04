# Revisione offline delle golden answer dei sintetici v4 · campione-2

Pacchetto per la revisione umana dei fogli `outputs/synth_gold_review/campione-2` (job 30556, codice `6328a29`;
docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md §§2.8, 2.11-2.12). Funziona aprendo `index.html` con un doppio clic: niente
server, installazioni o connessione. I dati sono file `.js` caricati dalla pagina, le immagini file locali.

Contenuto, tutto ricavato da campione-2 senza rigenerare nulla:
- **380 unità del campione casuale**, negli strati e nell'ordine fissati (coord, function, polygon, shapes: 75 QA
  ciascuno; test: 50 QA; coppie: 30 coppie);
- **175 casi mirati**, nelle loro categorie;
- **264 record candidati della decontaminazione**, con tutte le loro **755 corrispondenze** e le immagini del pannello.

Versione e impronta del pacchetto, sha256 degli ingressi (report.json di campione-2, manifest
`synth-v4-candidato-2`, file del checker, shard) e conteggi sono in `pacchetto.json`; gli sha256 di ogni file generato
in `SHA256SUMS`; l'esito della verifica sul cluster in `verifica_pacchetto.json`.

## 1. Scaricare e aprire

1. Su GitHub, nel repository `akira-n-28/vision_limite`, ramo `main`: **Code → Download ZIP**. Estrai lo ZIP: la
   cartella del pacchetto è `vision_limite-main/revisioni/synth-v4-campione-2/`. Puoi spostarla dove vuoi, ma tutta
   intera: `index.html` ha bisogno di `dati/`, `immagini/`, `logica.js`, `app.js` e `stile.css` accanto a sé.
2. Facoltativo, per controllare l'integrità: nel Terminale, dentro la cartella, `shasum -a 256 -c SHA256SUMS` deve dare
   `OK` per ogni riga.
3. **Prima di iniziare apri `verifica.html`** (doppio clic). È un autotest nel tuo browser, sui dati veri: logica del
   primo giudizio, unità collegate ed esposizioni, coppie, decontaminazione, esportazione, importazione e ripresa, TSV,
   caricamento di tutte le 972 immagini. Non tocca il tuo lavoro: usa una memoria finta al posto di quella del browser
   e non scarica file. Non mostra soluzioni: nei risultati ci sono solo identificativi e conteggi, e un controllo
   finale cerca nell'output risposte attese, opzioni ausiliarie e motivi. Deve dire "Tutti i … controlli superati"; se
   c'è una riga rossa, fermati e segnalala.
   - **Non usare la `verifica.html` delle copie scaricate fino al commit `a9252b5`:** in un controllo mostrava il target
     di un'unità del campione. Scarica di nuovo lo ZIP e usa solo questa versione.
4. Apri `index.html`. Se Safari blocca qualcosa, apri il file da *File → Apri file*; anche Firefox e Chrome vanno bene.
   Usa sempre lo stesso browser: il salvataggio automatico vale solo per quel browser.

## 2. Come si lavora

Le schede in alto sono **Campione casuale**, **Casi mirati**, **Decontaminazione** e **Riepilogo e file**, ognuna con il
proprio avanzamento. A sinistra c'è l'elenco dei casi, con lo stato di ognuno:
- ○ non iniziato;
- ◔ primo giudizio in bozza;
- ◑ primo giudizio salvato, soluzione ancora nascosta;
- ◕ soluzione vista, giudizio finale incompleto;
- ● completo;
- `?` segnala un valore incerto, `!` uno non giudicabile.

Per muoverti hai *precedente*, *successivo* e *prossimo da completare*; il filtro "da completare" serve a riprendere i
casi lasciati a metà. L'ordine dei casi è quello del campione e non va cambiato: con un tempo limitato si rivede un
prefisso fissato prima di guardare (S2 §2.8).

**Immagini.** Si aprono al 100% dei pixel, senza riduzioni. I pulsanti 200-400% ingrandiscono a pixel netti, "adatta"
le fa stare nella finestra, "schermo intero" (o un clic sull'immagine) le apre sopra la pagina. Esc chiude.

**Prompt.** È il testo esatto che il modello riceve dopo l'immagine, con gli a capo. Per train e dev è quello del loader
di training, con da 1 a 5 domande numerate: la domanda da giudicare è evidenziata, e il suo numero è scritto sotto.
Per test e coppie è la domanda da sola. Le opzioni compaiono solo quando fanno parte della domanda (per esempio
`shape_type` e `label_type`).

### Primo passaggio: senza la soluzione

Per ogni QA (per le coppie, per la variante a e per la b):
1. **Risposta:** la tua risposta, oppure la spunta "non riesco a rispondere". Nei casi sulla scena intera è facoltativa.
2. **Leggibilità e rispondibilità:** il giudizio iniziale.
3. **Note**, facoltative.
4. **Salva il primo giudizio.** Finché non lo salvi, e finché ci sono modifiche non salvate, la soluzione resta
   nascosta.

Fino a qui non vedi target, opzioni ausiliarie, esito del checker, motivo e misura dei casi mirati.

### Secondo passaggio: soluzione e giudizio finale

**Mostra soluzione e materiale di confronto** chiede conferma e registra data e ora. Da quel momento il primo giudizio
resta fissato, anche dopo un salvataggio e una ripresa: è la tua risposta indipendente. Se cambi idea, scrivilo nel
giudizio finale, anche nel campo "risposta dopo la soluzione", che è segnato come non indipendente.

Il materiale di confronto comprende:
- risposte attese e opzioni, per variante;
- esito del checker e stato del record;
- per ogni record, le domande con le risposte, la specifica della scena, il layout disegnato e la riga del checker,
  presi dagli shard e dal file del candidato senza rigenerarli.

### Unità collegate: perché alcune soluzioni aspettano altre unità

Il materiale di un record rivela le risposte di **tutte** le sue domande, non solo di quella dell'unità: la tabella
delle QA, la specifica, il layout e il JSON completo. Lo stesso vale per un record diverso con **la stessa scena**
(stessa specifica a meno di chiave, versione del generatore e stile). Nel pacchetto succede per 23 unità, in gruppi di
al più 4:
- 7 record compaiono in più unità, per esempio `synth-test-00038` in `C-test-008` e `C-test-012`;
- 13 scene compaiono in più record, per esempio `synth-test-00038`, `synth-test-00014` e `synth-pair-00094-a`.

Per questo, nella pagina di un'unità con collegamenti:
- il riquadro **Unità collegate** elenca le altre unità di cui la soluzione rivelerebbe la risposta, con il loro stato
  e un pulsante per raggiungerle. Nel primo passaggio non dice se il legame è lo stesso record o la stessa scena,
  perché nei casi «stessa istanza?» lo anticiperebbe; lo dice dopo la soluzione;
- **Mostra soluzione** resta disattivato finché ogni unità collegata non ha un primo giudizio salvato, senza modifiche
  in sospeso. Sotto il pulsante c'è l'elenco di quelle che mancano;
- quando mostri la soluzione, il primo giudizio delle unità collegate resta **fissato**: non si modifica più, come il
  tuo. Nella loro pagina compare da quando e per quale unità. La loro soluzione la mostri poi normalmente.

Le coppie espongono i record di entrambe le varianti; un caso mirato con due immagini espone entrambi i record.

### Registro delle esposizioni

Ogni volta che mostri una soluzione si registrano i record esposti, con data e ora e l'unità da cui li hai visti
(Riepilogo → *Record esposti*). Il registro sta nel file di avanzamento e non si cancella: né importando un file più
vecchio, né sostituendo lo stato, né riaprendo la pagina.

Se un record risulta esposto prima che un'unità collegata avesse un primo giudizio salvato, l'unità è segnata **non
indipendente** (`~` nell'elenco, avviso rosso nella pagina). Può succedere solo con lavoro fatto altrove, per esempio
importando un file di avanzamento del formato 1 in cui una soluzione era già stata mostrata. Puoi compilarne il primo
giudizio, che resta fissato al salvataggio. Nel TSV la colonna `risposta_revisore` vale `non_indipendente`, e la
risposta con il motivo va nelle note: una risposta data dopo l'esposizione non conta mai come indipendente.

Nel giudizio finale, per ogni QA o variante:

| Campo | Domanda |
| --- | --- |
| Correttezza | Il target è la risposta vera per ciò che si vede e per la specifica? |
| Coerenza con le annotazioni | Il PNG mostra ciò che dicono specifica e layout? |
| Rispondibilità | Si risponde con la sola immagine e il prompt? |
| Gravità | grave = errore; minore = difetto che non cambia la risposta né la lettura; nessun difetto |
| Tipo di difetto | obbligatorio con gravità minore o grave; scegli dall'elenco o scrivilo |
| Note | libere, anche su più righe |

Nelle coppie c'è anche il **contratto della coppia**: la variante b cambia la risposta, e solo il dettaglio dichiarato?
Ha la sua gravità e le sue note.

I valori dei giudizi sono:

| Valore | Significato |
| --- | --- |
| sì / no | il giudizio |
| incerto | hai guardato ma non sai decidere |
| non giudicabile | il caso non permette un giudizio, per esempio immagine assente o domanda incomprensibile |
| non verificabile | il materiale del pacchetto non basta per quella proprietà |
| non applicabile | la proprietà non riguarda il caso |
| non ancora compilato | il valore iniziale |

Nessun campo vuoto vale come approvazione: un caso è completo solo quando ogni campo è compilato, e incerti e non
giudicabili contano come errori nel limite (S2 §2.8).

### Decontaminazione

Il record sintetico sta a sinistra e tutte le immagini candidate del pannello a destra, ciascuna con il suo
identificativo (`D-001-001` …), elemento, posizione dell'immagine e ruolo nel pannello. Le corrispondenze che non erano
nell'anteprima del manifest sono segnalate. La somiglianza dei pixel serve solo a orientarsi e non compila nulla.

Per ogni corrispondenza scegli l'esame: stesso contenuto, contenuto diverso, incerto, non giudicabile. Poi l'esito del
record:
- **sì** solo indicando la corrispondenza decisiva, che deve essere esaminata come stesso contenuto. Le altre restano
  con il loro esame, anche "non ancora esaminata";
- **no** solo quando tutte le corrispondenze sono esaminate come contenuto diverso e il materiale è completo. Se dopo
  un "no" cambi un esame, il "no" si annulla da solo;
- **incerto** o **non giudicabile**.

Nessun esito di questa interfaccia cambia ammissioni, quarantene o gruppi. Gli esiti entreranno, dopo verifica, in
una versione nuova del manifest.

## 3. Salvare e riprendere

- La pagina salva da sola nel browser a ogni modifica, ma è solo una comodità: può sparire, per esempio con la
  pulizia dei dati di Safari o in una finestra privata. Se all'apertura compare "Nessun lavoro salvato in questo
  browser" e hai già lavorato, importa subito l'ultimo file di avanzamento prima di continuare.
- **La copia che conta è il file di avanzamento:** *Scarica il file di avanzamento* (in alto, o nella scheda Riepilogo)
  scarica nella cartella Download `synth-v4-campione-2_avanzamento_<data-ora>.json`. Scaricalo spesso, almeno a fine
  sessione. Contiene tutto:
  - primi giudizi con data e ora, primi giudizi fissati con la loro provenienza;
  - registro dei record esposti e archivio delle versioni scartate;
  - giudizi finali, risposte dopo la soluzione ed esami delle corrispondenze;
  - tempi ed eventi.
- Lavora in **una sola scheda**. Se la revisione è aperta in due schede, a ogni salvataggio le esposizioni e i primi
  giudizi fissati dell'altra si uniscono, e compare un avviso; per il resto del lavoro vale l'ultima modifica.
- Se lo scaricamento non funziona, Riepilogo → *copia il JSON a mano*.

### Importare e ripristinare

Riepilogo → *Importa il file JSON*. Il file si accetta solo se pacchetto, versione, impronta, identificativi e valori
coincidono. Poi:
1. se nel browser c'è già del lavoro, prima di cambiare qualcosa si scarica una **copia di sicurezza** del browser
   (`…_copia_del_browser_<data-ora>.json`);
2. la **conoscenza** del file si unisce subito e non si perde mai:
   - le esposizioni del file si aggiungono a quelle del browser;
   - un primo giudizio fissato vince su una versione non fissata della stessa unità, sia che venga dal file sia che
     stia nel browser; fra due primi giudizi fissati diversi vale quello fissato prima;
   - la data della soluzione mostrata resta la più antica;
   - i casi presenti solo nel file si aggiungono;
3. il **lavoro in conflitto**, cioè la stessa unità diversa nel browser e nel file senza un primo giudizio fissato, o
   con lo stesso primo giudizio fissato ma un giudizio finale diverso, resta come nel browser finché non scegli:
   - *Tieni le versioni del browser*;
   - *Usa le versioni del file per i casi in conflitto*;
   - *Sostituisci tutto con il file*: vale il file anche per i casi presenti solo nel browser, ma esposizioni e primi
     giudizi fissati restano.

Ogni versione scartata, da una scelta o perché un primo giudizio fissato ha vinto, va nell'**archivio** del file di
avanzamento, con origine e motivo; niente sparisce in silenzio. Una versione salvata dopo l'esposizione è marcata come
non indipendente.

Così importare un file salvato prima di aver visto una soluzione non riapre il primo giudizio, e importare un file
più recente porta con sé le sue esposizioni.

**Formato dei file.** Questa versione scrive il formato 2 (`synth-v4-revisione-avanzamento/2`). Un file o uno stato del
browser nel formato 1, quello del pacchetto fino ad `a9252b5`, si legge con una migrazione esplicita, annunciata in un
messaggio e registrata nel file:
- ogni soluzione già mostrata diventa un'esposizione dei suoi record;
- le unità collegate sono fissate se avevano un primo giudizio salvato prima, altrimenti segnate come non
  indipendenti.

Nel browser resta anche una copia del formato 1. Uno stato del browser illeggibile non si sovrascrive: si copia in una
chiave di riserva e la pagina chiede di importare il file di avanzamento.
- Nella scheda Riepilogo trovi i conteggi di casi completi, in corso, non iniziati, incerti, non giudicabili e non
  verificabili, per sezione e per strato o categoria.

## 4. Restituire gli esiti

1. Riepilogo → scarica il **file di avanzamento** e, se vuoi, `esiti_<data-ora>.tsv` ed
   `esiti_decontaminazione_<data-ora>.tsv`. I TSV hanno le colonne dei moduli originali (`moduli/esiti.tsv` e
   `moduli/esiti_decontaminazione.tsv`) e le stesse righe nello stesso ordine.
   - Nelle celle gli a capo sono `\n`, le tabulazioni `\t` e la barra rovesciata `\\`.
   - Un campo non compilato vale `non_compilato`.
   - Nelle coppie le righe sono a, b e `coppia`.
   - In `corrispondenze_guardate` c'è l'esame di ogni corrispondenza.
2. Su GitHub, apri la cartella `revisioni/synth-v4-campione-2/risultati/` → **Add file → Upload files**, trascina
   questi file (il JSON è obbligatorio, i TSV facoltativi) e fai commit su `main`. Se ricarichi più volte, carica sempre
   l'ultimo JSON: i precedenti si possono lasciare.
3. **Non modificare né ricaricare** gli altri file del pacchetto: dati, immagini, moduli e interfaccia restano come
   sono. Sul cluster recupererò i file di `risultati/` e li verificherò contro questo pacchetto.

## 5. Cartelle e file

| Percorso | Contenuto |
| --- | --- |
| `index.html`, `app.js`, `logica.js`, `stile.css` | interfaccia, senza framework né risorse esterne |
| `verifica.html`, `test.js` | autotest nel browser |
| `verifica_js.json` | esito dell'autotest eseguito sul cluster con Duktape, senza DOM e su un DOM simulato |
| `dati/pacchetto.js` | unità come si vedono al primo passaggio |
| `dati/soluzioni.js`, `dati/evidenze.js` | soluzioni e materiale del secondo passaggio. Non aprirli prima di aver finito |
| `immagini/` | 870 PNG v4 e 102 immagini del pannello, byte per byte come in campione-2 |
| `moduli/` | file originali di campione-2, compresi `soluzioni.tsv`, `opzioni.tsv` e `mirati.tsv`, che rivelano target e motivi: consultali solo dopo |
| `pacchetto.json`, `SHA256SUMS`, `verifica_pacchetto.json` | versione, ingressi, sha256 e verifica |
| `risultati/` | dove ricaricare i tuoi file |

## 6. Limiti

Sul cluster sono stati eseguiti tre tipi di controllo, distinti:
1. **Controlli statici** (`verifica_pacchetto.json`), in Python:
   - dati identici a campione-2 e immagini con sha256 identici;
   - prompt esatti e soluzioni assenti dai dati del primo passaggio;
   - collegamenti fra unità, con la scena calcolata dal JSON dei record uguale allo stato registrato nel manifest;
   - nessuna richiesta di rete e controllo lessicale dei file JavaScript.
2. **Esecuzione della logica e dell'interfaccia** (`verifica_js.json`): `logica.js`, `app.js` e `test.js` girano con
   Duktape, il motore JavaScript ES5.1 installato come libreria di sistema su `lagrange0`. È lo stesso autotest di
   `verifica.html`, una volta senza DOM e una volta su un DOM simulato (`scripts/review_package_dom.js`). Il DOM
   simulato costruisce le pagine, preme i pulsanti e usa una memoria finta al posto di localStorage. I collegamenti
   calcolati da `logica.js` si confrontano con quelli calcolati in Python, e un controllo indipendente cerca le
   soluzioni nelle righe dell'autotest.
3. **Prova in un browser: non eseguita.** Disegno della pagina, eventi veri, localStorage, scaricamenti e caricamento
   delle immagini si provano solo con `verifica.html` sul tuo Mac. Duktape non è il motore di Safari, Chrome o Firefox.
- Nessun record è promosso a gold da questa revisione, che non avvia nessuna valutazione automatica.
