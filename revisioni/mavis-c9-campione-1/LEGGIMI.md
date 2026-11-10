# Revisione visiva mirata di MAVIS-Geometry · C9 · campione-1

Pacchetto per la tua revisione visiva di un piccolo campione diagnostico di MAVIS-Geometry dopo C9
(`docs/specs/S2_AMMISSIONE_PRIME_FAMIGLIE.md` §3). Funziona aprendo `index.html` con un doppio clic: niente server,
installazioni o connessione. I dati sono file `.js` caricati dalla pagina, le immagini sono file locali.

Le schede vengono dal manifest `mavis-geometry-candidato-5`, senza chiamate a modelli e senza giudizi già compilati.
Le quote sono quelle che hai fissato:
- 1 controllo noto;
- 19 letture di valori: decimali arrotondati, decimali esatti, radici, interi e angoli;
- 30 classificazioni C9 recuperabili;
- 15 classificazioni omesse da C9;
- 15 domande degli altri compiti: describe, direction, count_points, removed_side, shared_side.

Il campione è **diagnostico, non rappresentativo**: serve a trovare difetti, non a stimarne la frequenza. Dai suoi esiti
non si calcolano tassi, intervalli di confidenza o certificazioni. Gli esiti non ammettono record, non promuovono nulla
a gold e non attivano il training.

Versione, impronta, sha256 degli ingressi e conteggi sono in `pacchetto.json`; gli sha256 di ogni file in
`SHA256SUMS`; gli esiti delle verifiche sul cluster in `verifica_pacchetto.json` (dati e selezione) e
`verifica_js.json` (JavaScript eseguito fuori dal browser).

## 1. Scaricare e aprire

1. Su GitHub, nel repository `akira-n-28/vision_limite`, ramo `main`: **Code → Download ZIP**. Estrai lo ZIP: la
   cartella del pacchetto è `vision_limite-main/revisioni/mavis-c9-campione-1/`. Puoi spostarla dove vuoi, ma tutta
   intera: `index.html` ha bisogno di `dati/`, `immagini/`, `logica.js`, `app.js` e `stile.css` accanto a sé.
2. Facoltativo, per controllare l'integrità: nel Terminale, dentro la cartella, `shasum -a 256 -c SHA256SUMS` deve dare
   `OK` per ogni riga.
3. **Prima di iniziare apri `verifica.html`** (doppio clic). È un autotest nel tuo browser, sui dati veri: primo giudizio
   separato dalla soluzione, controllo noto, collegamenti ed esposizioni, esportazione, importazione e ripresa, TSV,
   caricamento di tutte le immagini alla risoluzione registrata. Non tocca il tuo lavoro: usa una memoria finta al posto
   di quella del browser e non scarica file. Non mostra soluzioni: un controllo finale cerca nell'output target,
   categorie, motivi, didascalie e valori. Deve dire "Tutti i … controlli superati"; se c'è una riga rossa, fermati e
   segnalala.
4. Apri `index.html`. Se Safari blocca qualcosa, apri il file da *File → Apri file*; anche Firefox e Chrome vanno bene.
   Usa sempre lo stesso browser: il salvataggio automatico vale solo per quel browser.

**Non aprire prima del tempo** `dati/soluzioni.js`, `moduli/` e `pacchetto.json`: contengono target, categorie e motivi
della selezione di ogni scheda.

## 2. Come si lavora

In alto ci sono **Schede** (con l'avanzamento) e **Riepilogo e file**. A sinistra l'elenco delle schede, con lo stato:
- ○ non iniziata; ◔ primo giudizio in bozza; ◑ primo giudizio salvato, soluzione nascosta;
- ◕ soluzione vista, giudizio finale incompleto; ● completa;
- `?` valore incerto, `!` non giudicabile, `~` primo giudizio non indipendente.

Le schede sono 80, da M01 a M80, in un ordine mescolato in modo deterministico (sha256 col seme 0): categoria e titolo
non si vedono e non anticipano la risposta. Puoi muoverti con *precedente*, *successiva* e *prossima da completare*.

**Immagini.** Sono i `caption.jpg` originali di `RuleBaseGeo.zip`, gli stessi byte dello snapshot (sha256 controllato),
non le immagini ridimensionate del training: si valuta l'originale, non una pre-elaborazione futura. Si aprono al 100%
(risoluzione nativa); 200-400% ingrandiscono a pixel netti, "adatta" le fa stare nella finestra, "schermo intero" le apre
sopra la pagina. Esc chiude.

### Primo passaggio: senza la soluzione

Vedi l'immagine, la domanda e, solo quando appartengono davvero alla domanda, le sue opzioni. Compili:
- **Risposta**: libera. Per un valore scritto nella figura riporta la scrittura che vedi, senza trasformarla (una radice
  resta una radice, un decimale resta come è scritto). Oppure spunta **non determinabile**.
- **Leggibilità**: ciò che serve per rispondere è visibile, leggibile e univoco?
- **Sufficienza delle informazioni**: immagine e domanda bastano?
- **Note**, facoltative.

Poi **Salva il primo giudizio**. Puoi correggerlo e salvarlo di nuovo finché non mostri la soluzione.

**Richieste diagnostiche.** Le classificazioni omesse da C9 non hanno una golden answer. Le loro schede sono segnate
"Richiesta diagnostica": non sono domande del training. Indica tutte le opzioni vere (una, più di una o nessuna), oppure
"non determinabile", e scrivi nelle note che cosa manca. Servono anche a cercare omissioni troppo prudenti del filtro.

**Controllo noto.** Una scheda è segnata "Controllo noto": è il caso √841 di `d3-44919` (lato EH), già descritto nei
rapporti di C9. Il suo primo giudizio si registra come non indipendente.

### Secondo passaggio: soluzione e giudizio finale

Con **Mostra soluzione** il primo giudizio resta fissato e compaiono:
- target candidato (o "nessuna golden answer" per le richieste diagnostiche);
- annotazioni del record: didascalia, `diagram_generation`, valori esatti e visualizzati previsti, tipo dichiarato e
  misure della figura;
- esiti automatici di C9: opzioni, confutazioni, motivo dell'omissione, verità delle opzioni secondo le annotazioni;
- categoria, regola e motivo della selezione; stato del record e del compito nel manifest.

Compili poi:
- **correttezza rispetto ai pixel** (per le richieste diagnostiche: annotazioni coerenti con i pixel);
- **rispondibilità** dalla sola immagine;
- per le letture di valori, **scrittura visibile uguale al visualizzato previsto**, nella stessa forma. Le equivalenze
  matematiche (per esempio √841 = 29) vanno nel campo **equivalenze**, a parte: non correggono la trascrizione;
- per le richieste diagnostiche, **filtro troppo prudente** e, facoltative, le classi che ritieni vere;
- **giudizio finale**: nessun problema, errore grave, problema minore di formato o metadato, ambiguità, non giudicabile.
  Con errore grave, problema minore o ambiguità scrivi anche il **tipo di difetto** (ci sono suggerimenti);
- note e, facoltativa, la risposta dopo la soluzione, che resta separata e non è indipendente.

L'accordo fra la tua risposta e il target **non** rende il target "corretto" o "gold": è solo un'osservazione.

### Collegamenti ed esposizioni

Ogni scheda ha un record e un gruppo diversi, quindi nessuna soluzione rivela la risposta di un'altra scheda. La
protezione resta comunque: mostrare una soluzione registra record e gruppo esposti, con data e scheda; una scheda di
un record o gruppo già esposto avrebbe il primo giudizio segnato come non indipendente. Il registro non si cancella.

## 3. Salvare e riprendere

Il lavoro si salva da solo nel browser a ogni modifica. Il salvataggio del browser però si può perdere: con la pulizia
dei dati del browser, in una finestra privata o cambiando browser. Scarica spesso il **file di avanzamento**, con il
pulsante in alto o da *Riepilogo e file*: è un JSON con tutto il lavoro.

Per riprendere su un altro computer o browser, apri `index.html` e in *Riepilogo e file → Importa* scegli l'ultimo file
di avanzamento. Si controllano pacchetto, versione, impronta e identificativi. Esposizioni e primi giudizi fissati non si
perdono mai. Se il browser e il file hanno lavoro diverso sulla stessa scheda, nulla si sovrascrive in silenzio: scegli
se tenere le versioni del browser, usare quelle del file o sostituire tutto col file; le versioni scartate restano
nell'archivio del file di avanzamento. Prima di importare, se nel browser c'è lavoro, se ne scarica una copia.

Lavora in una sola scheda del browser: se la revisione è aperta due volte, la pagina lo segnala e unisce le esposizioni.

## 4. Restituire gli esiti

In *Riepilogo e file* scarica il file di avanzamento (JSON) e, se vuoi, `esiti_….tsv`. Mettili in `risultati/` e
caricali su GitHub (*Add file → Upload files*), senza toccare gli altri file del pacchetto. Il TSV ha una riga per
scheda; nei campi gli a capo sono `\n`, le tabulazioni `\t` e la barra rovesciata `\\`; un campo non compilato vale
`non_compilato`, uno che non riguarda la scheda `non_applicabile`.

Sul cluster gli esiti si verificano prima di qualsiasi uso: impronta e identificativi, registro delle esposizioni,
indipendenza dei primi giudizi. Un esito non cambia da solo ammissioni, quarantene, gruppi o compiti.

Tempo indicativo: circa 2-4 minuti per scheda (di più per le tre descrizioni), cioè 3-4 ore in tutto, anche in più
sessioni.

## 5. Cartelle e file

- `index.html`, `app.js`, `logica.js`, `stile.css`: l'interfaccia; `verifica.html` e `test.js`: l'autotest.
- `dati/pacchetto.js`: le schede come si vedono al primo passaggio; `dati/soluzioni.js`: il secondo passaggio.
- `immagini/`: un `caption.jpg` originale per scheda, col nome del record.
- `moduli/selezione.tsv`: l'inventario delle schede (manifest e sha256, record, gruppo, split, compito, figura o
  elemento, domanda, target, ranghi, motivo e regola della selezione, immagine). `moduli/piano.json`: quote,
  disponibilità per strato e cella, ordine di elaborazione, deficit e regole.
- `pacchetto.json`, `SHA256SUMS`, `verifica_pacchetto.json`, `verifica_js.json`.
- `risultati/`: qui metti i file scaricati.

## 6. Limiti

- Le verifiche sul cluster sono statiche (dati, selezione, hash, riferimenti) e un'esecuzione del JavaScript con Duktape,
  senza DOM e su un DOM simulato. Non sono una prova in un browser: disegno della pagina, eventi veri, salvataggio nel
  browser e caricamento delle immagini si provano solo con `verifica.html` sul tuo computer.
- Il visualizzato previsto dei valori viene dalla regola del renderer, non dai pixel: è proprio ciò che questa revisione
  confronta con l'immagine.
- L'affollamento usato per variare le letture (misure stampate del record rispetto alla mediana della sua profondità) è
  un indicatore della densità di etichette, non una misura della leggibilità.
