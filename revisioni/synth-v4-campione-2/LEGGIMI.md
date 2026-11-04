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
   primo giudizio, coppie, decontaminazione, esportazione e importazione, TSV, caricamento di tutte le 972 immagini. Non
   tocca il tuo lavoro. Deve dire "Tutti i … controlli superati"; se c'è una riga rossa, fermati e segnalala.
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
  pulizia dei dati di Safari o in una finestra privata.
- **La copia che conta è il file di avanzamento:** *Scarica il file di avanzamento* (in alto, o nella scheda Riepilogo)
  scarica nella cartella Download `synth-v4-campione-2_avanzamento_<data-ora>.json`. Scaricalo spesso, almeno a fine
  sessione. Contiene tutto: primo giudizio con data e ora, momento in cui hai visto la soluzione, giudizi finali,
  risposte dopo la soluzione, esami delle corrispondenze, tempi ed eventi.
- **Per riprendere** su un altro browser o dopo una pulizia: Riepilogo → *Importa il file JSON*. Il file si accetta
  solo se pacchetto, versione, impronta, identificativi e valori coincidono. I casi presenti solo nel file si
  aggiungono; quelli uguali restano; se un caso è diverso nel browser e nel file non si importa niente, e scegli tu se
  tenere il browser o sostituire tutto con il file. In quel caso viene scaricata prima una copia di sicurezza.
- Se lo scaricamento non funziona, Riepilogo → *copia il JSON a mano*.
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
| `dati/pacchetto.js` | unità come si vedono al primo passaggio |
| `dati/soluzioni.js`, `dati/evidenze.js` | soluzioni e materiale del secondo passaggio. Non aprirli prima di aver finito |
| `immagini/` | 870 PNG v4 e 102 immagini del pannello, byte per byte come in campione-2 |
| `moduli/` | file originali di campione-2, compresi `soluzioni.tsv`, `opzioni.tsv` e `mirati.tsv`, che rivelano target e motivi: consultali solo dopo |
| `pacchetto.json`, `SHA256SUMS`, `verifica_pacchetto.json` | versione, ingressi, sha256 e verifica |
| `risultati/` | dove ricaricare i tuoi file |

## 6. Limiti

- L'interfaccia è stata scritta e verificata sul cluster solo staticamente: dati, riferimenti e assenza di richieste
  di rete. Non è stata aperta in un browser. Il primo controllo reale è `verifica.html` sul tuo Mac.
- Nessun record è promosso a gold da questa revisione, che non avvia nessuna valutazione automatica.
