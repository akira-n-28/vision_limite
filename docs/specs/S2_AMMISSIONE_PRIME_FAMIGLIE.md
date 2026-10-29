# S2 — Regole di ammissione delle prime famiglie

**Stato:** bozza 2 del 2026-10-28, rivista dopo la revisione esterna dei push fino a `e81f884`; da approvare. Attua la
pipeline v1.4 §§5.1, 5.6, 5.7, 5.10.2 (rilievi G3, G4, R6). Le famiglie sono le tre del nucleo F1 (§7): sintetici
propri, MAVIS-Geometry e UniMER. Le altre sorgenti avranno regole proprie quando entrano.

## 1. Regola comune

**Corretto** significa vero per l'immagine che il modello vede, dopo il preprocessing. Una risposta calcolata dalla
specifica del generatore resta un candidato finché la semantica del suo tipo di domanda non è verificata contro il
disegno: il controesempio della §2.1 mostra che specifica e immagine possono divergere.

Per ogni **tipo di domanda** (o compito) di una famiglia si considerano cinque evidenze:

1. **Definizione semantica scritta**: che cosa deve mostrare l'immagine perché la risposta sia vera, in quale dominio
   visibile e con quali margini.
2. **Verifica indipendente sul 100% delle istanze**: un secondo calcolo che non riusa la funzione del generatore.
   Vale il ricalcolo dalle primitive effettivamente disegnate (spezzate, centri, testi con i loro riquadri in pixel) o
   da predicati geometrici scritti a parte. Ripetere lo stesso algoritmo con le stesse assunzioni non conta (pipeline
   §5.6).
3. **Collaudi**: controesempi noti, casi ai bordi del dominio e test di mutazione (una risposta alterata viene
   trovata).
4. **Invarianti del renderer** (§2.4): riquadri dentro l'immagine, nessuna sovrapposizione che nasconda un elemento
   decisivo, altezza minima del testo alla risoluzione data al modello.
5. **Unicità**: una domanda aperta ha una sola risposta corretta solo se lo dimostra un predicato sull'istanza. Le
   categorie del generatore non lo dimostrano. Altrimenti si sceglie una di queste forme:
   - opzioni nel prompt, con esattamente un'opzione vera, verificata per istanza;
   - un criterio esplicito ("the most specific type among …"), solo se le classi formano una catena;
   - l'insieme delle risposte vere nel gold, per valutazione e RL.

Quali evidenze servono dipende dalla famiglia:

- **tutte le famiglie**: 1, 3 e 5, e nessun errore sistematico confermato;
- **famiglie con generatore e renderer nostri** (sintetici): anche 2 e 4. Solo allora i record possono essere
  `gold_verificato`;
- **famiglie esterne** (MAVIS, UniMER): la correttezza rispetto all'immagine e l'osservabilità si stimano su un
  campione (S4) al posto delle evidenze 2 e 4. I record restano al più `target_controllato`.

**Stato del tipo di domanda**, per famiglia e versione degli shard; i record lo ereditano:

| Stato | Condizione | Record |
| --- | --- | --- |
| `ammissibile` | evidenze richieste per la famiglia presenti e superate | `gold_verificato` se anche l'osservabilità è verificata per record o per invariante, altrimenti `target_controllato` |
| `quarantena` | evidenza mancante o ambiguità non risolta | nessun record di training |
| `escluso` | errore confermato | nessun record di training fino a correzione del generatore, nuova versione degli shard e collaudi superati |

Un errore confermato non si declassa a `target_controllato`. L'esclusione vale per tutto il tipo nella versione
colpita, non solo per le istanze già trovate. Un errore è sistematico quando nasce dal processo: generatore, renderer,
regola di costruzione delle domande. Nelle sorgenti annotate gli errori trovati in singoli record (per esempio i 51
di UniMER) escludono quei record; se la stessa causa ricorre, vale la regola del tipo. Oggi nessun tipo è
`ammissibile`, perché le verifiche indipendenti e i campioni non esistono ancora: dove non è noto un errore lo stato è
`quarantena` per evidenza mancante.

Per i record valgono sempre tre proprietà separate: **corretto**, **osservabile** e **spiegazione** (non applicabile
ai compiti answer-only). Gli stati del gold (pipeline §5.6):

- `gold_verificato`: tutte le proprietà necessarie sono coperte da una verifica per record (R) o da un'invariante della
  famiglia (I);
- `target_controllato`: almeno una proprietà necessaria è coperta solo da un campione (C);
- `incerto`: controlli discordi o inconcludenti; va in quarantena.

Si registrano separatamente il valore **esatto**, quello **visualizzato** e la **risposta richiesta**.

## 2. Sintetici (`src/limite_vl/synth.py`, shard `data/shards/synth/v3`)

Evidenza del 2026-10-28: le 60.000 scene di training di v3 sono state rigenerate dalle loro chiavi con il codice
attuale (`synth-NNNNNNN`, famiglie a rotazione come in `scripts/synth_render.py`). Il calcolo usa solo la libreria
standard e non legge gli shard. Gli istogrammi per famiglia, tipo e risposta coincidono esattamente con quelli del
manifest del job 29756: il codice attuale riproduce le domande di v3.

**Stato** (C1-C2): implementati il 2026-10-28 (`38af82a`) e corretti il 2026-10-29 dopo la revisione di quel commit,
che ha trovato figure tagliate, intersezioni contate nei buchi del disegno e un errore di sintassi nei test. Lo stesso
giorno la revisione di `f7d32d3` ha trovato che il report contava come verificato un campione senza aver dimostrato la
corrispondenza col PNG dello shard; ora gli esiti sono separati (§2.4). La suite CPU è passata il 2026-10-29 (job
30392, `b544b97`): `tests/test_synth.py` esegue 96 controlli col disegno vero, nessun fallimento. Due scarti rispetto
alla §5:
- intersezioni e `shape_type` si collaudano su 4.000 scene e `label_type` su 400, invece delle 10.000 indicate;
- i collaudi MAVIS appartengono a C9 e non sono implementati.

Il controllo sugli shard (job 30393, conteggi nel piano F0 §4) ha dato tre risultati:
- i 60.900 PNG di v3, ridisegnati, hanno byte identici;
- il controllo ritrova i difetti noti di v3;
- sulle 20.000 scene v4 in memoria il checker non segnala nessuna risposta errata.

Le segnalazioni nuove, presenti anche nel codice v4, sono state analizzate nel job 30400 (§2.5). Si separano in tre
gruppi:
- errori reali: 9 coppie polygon contraddittorie e le collisioni fra misure ed etichette dei punti nei polygon;
- riquadri sovrapposti senza contatto dei caratteri, all'origine degli assi;
- un controllo più severo del necessario, sulle coordinate.

Vanno decise prima del render di v4. I test e il controllo collaudano il codice, non i dati: nessun dato è
certificato.
- `synth.py`: `version=4` è il default e `version=3` riproduce v3; sono nuovi `visible_bound`, `intersection_points`
  (conteggio geometrico), `_near_miss`, le domande di classificazione con le opzioni, `_shapes_overlap`,
  `_shapes_fit` e `render(spec, layout=True)`;
- `synth_check.py`: la verifica indipendente;
- `scripts/synth_semantics_check.py`: il controllo sugli shard e i casi singoli (`--cases`);
- `scripts/synth_case_svg.py`: schema SVG di un caso shapes, dalla specifica e senza matplotlib;
- `tests/test_synth.py`: i collaudi della §5.

### 2.1 Errore confermato: `intersections`

- **Codice.** `_intersections` conta le radici con |x|, |y| ≤ R = 6 e scarta quelle a meno di 0.4 da R
  (`synth.py:388-414`, righe 407-408). `_draw_function` disegna x ∈ [−7, 7] e maschera solo |y| > 7 (righe 462 e 466):
  il tratto fra 6 e 7 è visibile ma non contato.
- **Controesempio** (verificato eseguendo `make_spec("audit-68", "function")`): f = −0.5x − 4, g = −2x + 4,
  intersezione in (16/3, −20/3). È nel tratto disegnato, ma il target di "How many times do the graphs of f and g
  intersect in the figure?" è `0`. Con la correzione il conteggio geometrico è 1. L'intersezione però dista 1/3 dal
  bordo V = 7 (|y| = 20/3), dentro il margine di 0.4: la domanda non si pone.
- **Estensione in v3**:
  - 7.460 scene con due grafici e 6.307 domande `intersections`;
  - 37 target errati: il conteggio nel dominio disegnato è diverso;
  - 364 domande con un'intersezione a meno di 0.4 dal bordo disegnato, che non andavano poste;
  - 401 scene colpite in tutto; in 146 esempi di training la domanda è fra quelle scelte (`f1c.pick_questions`,
    deterministica per chiave);
  - il conteggio analitico e quello numerico sulle spezzate disegnate (600 campioni) coincidono su tutte le altre
    domande.
- **Correzione minima.** Un solo dominio visibile `V = R + 1`, usato da `_draw_function` (campioni e maschera) e da
  `_intersections` (conteggio e margine di 0.4 dal bordo disegnato). Restano escluse dalle domande la tangenza
  (discriminante < 0.5), le radici a meno di 0.6 fra loro e le rette parallele. Le immagini non cambiano, cambiano
  solo le domande: serve una nuova versione degli shard, v4, con un job CPU da approvare (v3 ha richiesto circa 12
  minuti). In v3 il tipo `intersections` è **`escluso`**.
- **Grafici che si sfiorano.** La verifica indipendente segnala anche le domande con risposta `0` in cui i due grafici,
  senza intersezioni reali, si avvicinano nel tratto disegnato fin quasi a toccarsi: il discriminante non tiene conto
  della distanza visibile. In v3 sono 17 domande su 1.842 senza intersezioni reali sotto 0.3, e 9 sotto 0.25. v4
  non pone la domanda sotto 0.3 (`synth._near_miss`); la verifica segnala sotto 0.25, così le due regole non
  divergono sul confine.
- **Tratti visibili** (revisione di `38af82a`). La verifica contava un'intersezione se cadeva nel dominio x e sotto il
  livello di taglio, senza guardare i tratti effettivamente disegnati (`visible` del layout). Con f = x + 1,
  g = −x + 1 e tratti [−7, −2] e [2, 7] l'incrocio in x = 0 non è disegnato, ma il conteggio era 1. Ora conta solo se
  l'incrocio cade in un tratto disegnato di entrambi i grafici: la spezzata di matplotlib si interrompe al primo
  campione tolto. Le estremità dei tratti si classificano:
  - estremo del dominio;
  - taglio a |y| = 7: il campione successivo è oltre il livello;
  - buco non spiegato, con un margine di 0.4 in x.

  Un incrocio entro un passo di campionamento da un'estremità non è ammissibile: lì disegnato o no non si decide. Sui
  dati di v3 e v4 i tratti seguono il taglio, quindi gli esiti non cambiano (le stesse 5.897 ok, 37 errate e 373 non
  ammissibili su v3).

### 2.2 Ambiguità di classificazione

Le domande sono aperte: il prompt non mostra le opzioni (`f1c.QA_ONE`). Conteggi su v3, con predicati geometrici
indipendenti calcolati sui vertici:

| Tipo | Problema | Con la correzione proposta |
| --- | --- | --- |
| `shape_type`, triangoli (`synth.py:290-293`) | tutti i 2.128 triangoli rettangoli sono anche scaleni; i 2.105 isosceli sono anche acutangoli (1.808) o ottusangoli (297) | opzioni nel prompt (right, isosceles, equilateral): esattamente una vera in 4.233/4.233 |
| `shape_type`, quadrilateri (`synth.py:294-296`) | con la definizione inclusiva un quadrato soddisfa 4 classi (ed è un rombo), un rettangolo 3, un parallelogramma 2 | opzioni e "the most specific type among square, rectangle, parallelogram, trapezoid" (catena): coincide con `kind` in 8.614/8.614 |
| `label_type`, shapes (`synth.py:500-501`) | 9.465 risposte `square` sono vere anche come `rectangle` | opzioni e classe più specifica fra circle, triangle, square, rectangle |

`has_type` e `count_type` gestiscono già quadrato ⊂ rettangolo (`synth.py:493-499`). In v3 `shape_type` e
`label_type` sono **`quarantena`**. Il predicato di unicità entra nei collaudi; non basta la categoria del generatore.

### 2.3 Gli altri tipi: semantica, condizioni del generatore, verifica richiesta

In v3 tutti sono `quarantena` per evidenza mancante. Nessun errore è noto: diventano `ammissibili` in v4 con la
verifica indipendente e le invarianti del renderer.

| Famiglia | Tipo | Semantica e condizione nel codice | Verifica indipendente richiesta |
| --- | --- | --- | --- |
| coord | `quadrant`, `side_x`, `side_y` | segno delle coordinate; punti a distanza ≥ 2 dagli assi (righe 62-66) | segni ricalcolati dai centri dei marcatori disegnati |
| coord | `coords` | lettura da tacche ed etichette; solo con le tacche, punti su multipli del passo delle etichette (righe 62, 102-107, 145) | ogni coordinata ha un'etichetta di tacca disegnata e leggibile |
| coord | `pair_lr`, `pair_ud` | confronto dei centri con scarto ≥ 2 (righe 112-117) | dai centri disegnati |
| coord | `count`, `exists` | punti etichettati senza O; lettere senza I, J, O, X, Y, Z (riga 26) | testi presenti nel render, marcatori non sovrapposti |
| coord | `line_on` | punto sul reticolo della retta, oppure a distanza ≥ 1 (righe 68-83) | distanza in pixel fra marcatore e retta disegnata |
| polygon | `side_len` | valore scritto presso il lato (righe 264-271, 315-323) | l'etichetta è più vicina al suo lato che agli altri e non si sovrappone |
| polygon | `angle_val` | valore nell'arco, solo triangoli non rettangoli (righe 242-244, 272-277) | arco e testo dentro l'angolo |
| polygon | `right_vertex` | segno dell'angolo retto disegnato (righe 278-279, 324-333) | marcatore presente al vertice giusto |
| polygon | `highest`, `lowest`, `leftmost`, `rightmost` | vertice estremo con margine 0.15·lato del riquadro (righe 280-286); aspetto 1:1 | ricalcolo dai vertici disegnati in pixel |
| polygon | `m_side` | M sul lato con t ∈ [0.3, 0.7] (riga 247) | etichetta di M non sovrapposta; il verificatore accetta AB = BA |
| polygon | `count`, `exists` | numero di lunghezze scritte; etichette dei vertici e di M | testi del render |
| function | `opens`, `vertex_ud`, `vertex_lr`, `slope`, `intercept` | segno di a, \|k\| ≥ 1, \|h\| ≥ 1, \|f(0)\| ∈ [1, R − 0.5] (righe 383-385, 433-445) | dalla spezzata disegnata (vertice e intercetta nel dominio V) |
| function | `exists` | etichette f, g; assenti fra h, p, q | testi del render |
| shapes | `count`, `has_type`, `count_type`, `exists`, `label_type` | celle 3x3 con spostamenti casuali (`_shapes_spec`): 19 scene su 15.000 hanno riquadri sovrapposti; **589 hanno una figura che esce dagli assi [0, 9]², 880 contando mezzo tratto e 1 px** (revisione di `38af82a`: in `synth-0000043` il rettangolo L parte da x = −0.119 e il lato sinistro non si vede) | contorni non sovrapposti e dentro l'area effettiva degli assi nel render; vincoli nel generatore v4 (`_shapes_overlap`, `_shapes_fit`) |
| pair | controfattuali (`counterfactual`) | un dettaglio cambia la risposta (righe 581-637) | come il tipo di domanda corrispondente, per entrambe le immagini |

### 2.4 Invarianti del renderer

Si calcolano durante il render dagli oggetti di matplotlib (`get_window_extent` di testi, marcatori e figure; area
degli assi `axes_px`; spessore del tratto `lw_px`) e si salvano nel JSON di v4. Il layout porta anche `png_sha256`,
lo sha256 dei byte del PNG scritto nello stesso campione: se in lettura coincide con il PNG dello shard, il layout si
ricontrolla senza ridisegnare. Le invarianti:

- figure e marcatori, col loro tratto, dentro l'area effettiva degli assi, dove matplotlib li taglia: è il controllo
  che mancava in `38af82a`, che guardava solo i bordi dell'immagine. Uscire di più di 1 px è una figura tagliata;
- riquadri dei testi dentro l'immagine;
- nessuna sovrapposizione fra testi, né fra un testo e il marcatore di un altro oggetto; nelle shapes, contorni senza
  sovrapposizioni;
- altezza dei testi in pixel, alla risoluzione data al modello, non sotto una soglia (proposta: 10 px).

**Violazioni bloccanti**: figura tagliata, testo fuori dall'immagine, figure sovrapposte, layout senza l'area degli
assi. Una scena che ne ha una non è verificata e le sue domande "ok" diventano `non_ammissibili`. Una scena è
verificata solo senza violazioni e con tutte le domande ok. Superare la verifica non certifica la scena: la
certificazione segue S4.

**Esiti per campione** (`scripts/synth_semantics_check.py`, revisione di `f7d32d3`). Per un record letto da uno shard
si tengono separati tre esiti; il report li conta per origine, e per famiglia, e registra semantica e immagine
insieme:

- **semantica della scena**: `verificata` se `synth_check.check_sample` sul layout usato non trova violazioni e tutte
  le domande sono ok, altrimenti `non_verificata`;
- **immagine**, cioè la corrispondenza fra il rendering descritto da quel layout e il PNG presente nello shard:
  - `digest_corrispondente`: il layout del record porta `png_sha256` uguale allo sha256 del PNG dello shard;
  - `byte_identici`: senza questa prova la scena si ridisegna dalla specifica, e il ridisegno ha gli stessi byte del
    PNG;
  - `pixel_identici`: byte diversi, ma il PNG decodificato da Pillow ha dimensioni e valori RGBA uguali al ridisegno;
    è un confronto deterministico, distinto da quello dei byte, e non guarda i metadati del PNG (gamma, profili);
  - `diversa_da_verificare`: pixel diversi, oppure il file non si decodifica come PNG. Non prova un'immagine diversa
    a vista né una risposta sbagliata: il campione va guardato;
  - `png_mancante`;
- **campione**: `verificato` solo se la semantica è verificata e l'immagine è `digest_corrispondente`,
  `byte_identici` o `pixel_identici`; altrimenti `non_verificato`, con i motivi in una riga di `problems.jsonl`.
  Una scena semanticamente valida con il PNG mancante o diverso non è un campione verificato.

`verificato` vuol dire *approvato dal checker*: i predicati sul layout non trovano violazioni né risposte diverse. Non
è *correttezza dimostrata* e non è *qualità certificata* (S4). Il checker legge il layout, non i pixel: una domanda
può risultare ok anche se il suo testo è coperto nell'immagine (§2.5).

Si registra anche da dove viene il layout usato:

- `salvato_con_digest`;
- `ridisegnato_layout_assente`: record di v3;
- `ridisegnato_senza_digest`: record con un layout ma senza digest, come quelli che `synth_render.py` avrebbe scritto
  fra `38af82a` e questa correzione (nessuno shard v4 è stato generato). Il layout salvato non prova il legame col PNG
  e non si usa;
- `ridisegnato_digest_diverso`: il PNG non è quello per cui il layout era stato calcolato;
- `salvato_png_mancante`.

Una scena generata in memoria (`--generate`) è un'altra origine. Ha immagine `in_memoria` ed esito `scena_verificata`
o `scena_non_verificata`: collauda generatore e renderer, non è un campione di dati. Il report verifica che i
conteggi tornino: ogni campione compare una volta in ogni gruppo, e i verificati sono esattamente quelli con
semantica verificata e immagine corrispondente. Se non tornano, il report riporta le incoerenze e il job fallisce.

Generatore v4: le shapes con due figure a meno di `SHAPE_GAP` o con una figura che, col tratto e 1 px, esce dagli assi
si estraggono di nuovo dallo stesso generatore (`_shapes_overlap`, `_shapes_fit`). Si riestraggono 969 scene shapes di
v3 su 15.000 (6.5%); le altre restano identiche. Schema del caso prima e dopo, ricostruito dalla specifica:
[v3](../img/synth-0000043-v3.svg), [v4](../img/synth-0000043-v4.svg). I PNG del renderer si ottengono nel job con
`--cases`.

### 2.5 Scarti del controllo sugli shard (job 30393), analizzati nel job 30400

L'analisi è in `outputs/synth_review/scarti-run1`, prodotta da `scripts/synth_review.py` (codice `8cd2c22`):
- legge il controllo e lo shard delle coppie v3 in sola lettura;
- ridisegna in memoria;
- non modifica né il generatore né il checker.

I fogli di immagini sono in [`docs/img/synth-review/`](../img/synth-review/). Le 25 scene del campione ridisegnate oggetto
per oggetto coincidono pixel per pixel con `synth.render`. Restano distinti *approvato dal checker*, *correttezza
dimostrata* e *qualità certificata* (§2.4).

**Coppie controfattuali polygon (75 in v3).** La variante `-b` cambia la misura stampata di un lato, aggiungendo 2, 3
o 4, e lascia invariato il disegno. Viola quindi l'invariante di scala del generatore (`lunghezza_incoerente` in
tutte le 75), ma questo da solo non rende errata la risposta. La versione precedente di questa sezione le dava tutte
per errori confermati: era sbagliato.
- Si sono usati solo i vincoli dichiarati nel disegno: misure stampate, angolo scritto, angolo retto marcato.
- Non si sono usate le proporzioni, come i lati paralleli o uguali a vista.
- Arrotondamento: intero ±0.5, decimale ±0.05, angolo ±0.5°. "Interi esatti" legge gli interi come valori esatti.

| Esito | Coppie | Stato proposto (v3) |
| --- | --- | --- |
| risposta uguale alla misura stampata, in `a` e in `b` | 75 su 75 | vale per tutte |
| contraddizione dimostrabile: tre lati stampati e un angolo dichiarato incompatibili anche con l'arrotondamento | 9: 00033, 00037, 00113, 00125, 00137, 00153, 00221, 00265, 00273 | `escluso` |
| contraddizione solo con gli interi esatti: 00285, angolo retto in K, 17² + 48² ≠ 50² | 1 | `quarantena` |
| compatibile con una riscalatura uniforme: una sola misura stampata | 26 | `quarantena` fino alla scelta del contratto |
| nessun vincolo dichiarato violato, ma la figura non è più simile al disegno: due o tre misure, per esempio 00009 con i lati opposti di un parallelogramma apparente a 7.6 e 11.6 | 39 | `quarantena` |
| misura cambiata coperta dall'etichetta del punto sul lato: 00209 illeggibile ("9" sotto "K"), 00169 e 00269 coperte in parte, 00017 e 00205 attaccate ma leggibili | 5, già contate sopra | `quarantena` |

Esempio di contraddizione, 00033: con ST = 5.9, TU = 11.7 e 47° in T il teorema del coseno dà US ≈ 8.8, il valore di
`a`, non 12.8. Nessuna coppia si riammette in automatico.

Controlli dell'analisi stessa:
- le 75 varianti `a` risultano fattibili in entrambe le letture;
- `a` e `b` differiscono solo nelle misure;
- i PNG ridisegnati sono identici a quelli dello shard.

Immagini: [esempi 1](../img/synth-review/coppie-esempi-1.png), [esempi 2](../img/synth-review/coppie-esempi-2.png), le
misure cambiate di tutte le coppie [1](../img/synth-review/coppie-misure-1.png),
[2](../img/synth-review/coppie-misure-2.png) e [3](../img/synth-review/coppie-misure-3.png).

**Contratto proposto per v4: geometria coerente in scala.**
- Ogni misura stampata è quella disegnata, entro l'arrotondamento mostrato; il checker lo richiede già
  (`lunghezza_incoerente`, `angolo_incoerente`). `side_len` legge un numero che coincide col disegno.
- La coppia polygon tiene stampata una sola misura, quella della domanda. La variante `-b` si ottiene riscalando tutto il
  poligono del rapporto fra la misura nuova e quella vecchia. Angoli, angolo retto e punto sul lato restano uguali.
  Gli assi sono adattati alla figura, quindi `b` differisce da `a` solo per il numero.
- È la classe "riscalatura uniforme", l'unica senza contraddizioni.

L'alternativa, "lettura della misura scritta" con la figura dichiarata non in scala, avrebbe due costi: verificare caso
per caso i vincoli dichiarati, come fatto qui, e lasciare figure incoerenti a vista come 00009.

**Riquadri di testo sovrapposti (v4, 20.000 scene).** Si sono censite tutte le 2.702 violazioni, in 1.797 scene; i
conteggi coincidono col report. Per ciascuna i due oggetti sono stati disegnati da soli, misurando i pixel di
inchiostro in comune e la distanza fra i tratti pieni.

Questi conteggi descrivono le segnalazioni del checker in queste scene, non altro. Sostituiscono le percentuali date
prima a partire dalle righe conservate in `problems.jsonl`, che non sono un campione rappresentativo.

| Categoria (famiglia) | Violazioni | Inchiostro in comune | A contatto (≤ 1.5 px) | Separati |
| --- | --- | --- | --- | --- |
| `O` e tacche `-1` all'origine (coord) | 566 | 23 | 27 | 516 |
| `O` e tacche `-1` all'origine (function) | 1.232 | 84 | 64 | 1.084 |
| etichetta di un punto e tacca (coord) | 25 | 1 | 0 | 24 |
| misura di un lato ed etichetta del punto segnato su un lato (polygon) | 840 | 692 | 37 | 111 |
| misura sul punto segnato su un lato (polygon, `testo_su_marcatore`) | 39 | 12 | 3 | 24 |

Correzione: in polygon l'etichetta coinvolta non è quella di un vertice, come scritto prima, ma quella del punto
segnato su un lato (`spec["m"]`). Quell'etichetta sta a 0.07 della figura in direzione opposta al centro; la misura
sta a 0.06 dal punto medio del lato. Se il punto è sul lato con la misura, i due testi cadono quasi nello stesso posto.

Campione stratificato: 20 casi, 2 lievi e 2 marcati per strato, estratti in modo deterministico; lieve vuol dire che
il lato minore dell'intersezione dei riquadri è al più 3 px. In più 5 scene approvate di controllo.

| Casi (`synth-check-…`) | Immagine | Diagnosi | Evidenza |
| --- | --- | --- | --- |
| origine, coord: 0007872, 0014476 lievi; 0008048, 0005100 marcati | [foglio](../img/synth-review/campione-origine-coord.png) | solo riquadri: caratteri separati e leggibili. Nessuna domanda usa le tacche `-1`; `O` compare solo nella domanda di conteggio | tratti a 2.0-7.2 px, nessun pixel in comune |
| origine, function: 0012430, 0010810 lievi; 0001146, 0006562 marcati | [foglio](../img/synth-review/campione-origine-function.png) | solo riquadri; nessuna domanda function usa le etichette di tacca. In 0012430 e 0006562 un grafico attraversa un'etichetta di tacca: il checker non lo controlla | 2.8-4.0 px |
| misura ed etichetta del punto, polygon: 0006537, 0019365 lievi; 0011165, 0003037 marcati | [foglio](../img/synth-review/campione-etichetta_misura-polygon.png) | lievi: separati ma attaccati ("25H"). Marcati: collisione reale, la misura è coperta in parte ("7.8" con "K", "12.1" con "E") | 0 e 11 px in comune |
| misura sul punto, polygon: 0000489, 0006757 lievi; 0005853, 0016169 marcati | [foglio](../img/synth-review/campione-misura_su_punto-polygon.png) | il punto tocca la misura. In tre casi su quattro copre la misura anche l'etichetta del punto: "7.8" si legge "7W8", e su "10.0" la "E" copre lo zero | 0-13 px in comune |
| etichetta di punto e tacca, coord: 0009020, 0005328, 0009800, 0006912 | [foglio](../img/synth-review/campione-altro.png) | solo riquadri. In 0005328 la tacca "8" accanto a "R" si legge "R8": associazione ambigua | 4.0-8.1 px |
| controlli approvati: 0000356, 0000106, 0000677 (testi più vicini), 0000336, 0000073 (a caso) | [foglio](../img/synth-review/campione-controllo.png) | i tre più vicini hanno riquadri sovrapposti di 1.0 px e sono approvati. 0000356 è uguale a vista al caso segnalato 0007872: 1.2 px, stessi 6.0 px fra i tratti | 6.0-6.3 px |

All'origine la regola sui riquadri segnala soprattutto lo spazio vuoto sopra e sotto i caratteri. Nei polygon invece
segnala collisioni reali, che possono cambiare la lettura di una misura. Il checker verifica `side_len` sul layout:
in quelle scene la risposta è "ok" anche quando il numero è coperto.

**Coordinate (58 domande v4, 169 v3).** In tutte le 227 il punto è quello di prova della retta (`line.test`).
- Il punto viene spostato su un punto della retta. Quando le etichette vanno di 2 in 2 (R = 8), una sua coordinata
  può risultare dispari.
- La tacca senza numero è disegnata, sporge dall'asse di 2.5-4.0 px per lato e sta fra due tacche numerate. In 77
  domande mancano le etichette di entrambe le coordinate.
- La coordinata si legge contando una tacca: l'informazione c'è e il checker, che chiede l'etichetta esatta, è più
  severo del necessario.
- Nessun caso ha l'informazione mancante. [Foglio](../img/synth-review/coords.png).

**Proposta minima per v4.** Non è implementata; le regressioni previste andrebbero in `tests/test_synth.py`.

1. **Generatore, coppie polygon:** una sola misura stampata, e `-b` per riscalatura uniforme.
   - Regressioni su 300 coppie v4: `shown` è il solo lato della domanda in `a` e `b`; i punti di `b` sono quelli di
     `a` moltiplicati per il fattore.
   - Nessuna `lunghezza_incoerente` né `angolo_incoerente`; la risposta è la misura stampata.
   - I pixel di `a` e `b` differiscono solo nel riquadro della misura.
   - Con `version=3` lo shard delle coppie si riproduce identico.
2. **Generatore, polygon:** da v4 il punto segnato non sta su un lato con la misura stampata. Si sceglie un altro lato
   con un generatore separato, perché il resto della scena resti come in v3; se non ce ne sono, il punto si toglie.
   - Regressioni su 5.000 scene v4: nessuna sovrapposizione fra misure ed etichette o punti segnati.
   - Le domande `m_side` restano; v3 è invariata.
3. **Generatore, coord e function:** da v4 niente etichette sulle tacche `-1`; le tacche restano. Nessuna domanda le
   usa, perché i punti hanno coordinate di valore assoluto almeno 2.
   - Regressioni su 5.000 scene per famiglia: nessuna sovrapposizione all'origine.
   - Le domande `coords` non cambiano; v3 è invariata.
4. **Checker, coordinate:** una coordinata senza etichetta si accetta se la sua tacca è disegnata e le due tacche vicine
   sono numerate. Le tacche vanno registrate nel layout, con asse e valore nel gid: cambiano i metadati, non i pixel.
   - Regressioni: i 58 casi v4 diventano ok.
   - Senza la tacca, o senza un'etichetta vicina, tornano non ammissibili.
   - Una risposta alterata risulta errata.

Cosa resta com'è:
- la soglia delle sovrapposizioni (`OVERLAP_PX`) non cambia;
- le 25 sovrapposizioni fra etichette di punti e tacche in coord restano segnalate.

Dopo le correzioni 1-3, prima del render di v4, un nuovo controllo su 20.000 scene v4 deve dare zero sovrapposizioni
nelle categorie corrette; quelle residue vanno guardate.

## 3. MAVIS-Geometry (`f1_mavis.py`, `f1_schema.py`, shard `9c80fff3-caption`)

La verità viene da `diagram_generation`, confermata dalla didascalia (`mavis_gen.compare`: 0 segnalazioni su 199.050,
job 30025, 30052). Questo prova la coerenza fra due descrizioni, non che i pixel le rispettino (`mavis_gen.py`). Il
renderer di MAVIS non è nostro, quindi non abbiamo invarianti: l'osservabilità resta un controllo campionario (S4).

| Tipo (`mavis_questions`) | Semantica | Stato proposto |
| --- | --- | --- |
| `describe` | punti, figure, misure e angoli di `render_geometry` | `quarantena`; `target_controllato` dopo il campione di S4 |
| `side_len`, `angle_val` | lettura: valore visualizzato (decimali a due cifre per i razionali, job 30049, 30052); l'esatto (`44/3`) resta nel record | idem |
| `removed_side`, `direction` | lato sostituito; triangolo o semicerchio dentro o fuori | idem, priorità nel campione |
| `shared_side` | posta solo se il proprietario è unico (`f1_mavis.py:84-89`); il verificatore accetta AB = BA | idem |
| `count_points` | tutte le etichette visibili | idem, priorità nel campione |
| `exists_point`, `exists_shape` | il nome finto usa una lettera successiva a tutti i punti (`f1_mavis.py:95`): non coincide con un nome reale | idem |
| `shape_type` (`f1_mavis.py:83`) | aperta, con classi che non formano una catena: quadrato ⊂ rettangolo ∩ rombo, rombo ⊂ aquilone e parallelogramma, un triangolo rettangolo può essere isoscele | **`quarantena`**: opzioni costruite per istanza con esattamente una vera secondo le misure note; se le misure non bastano, la domanda non si pone |

Se in F0 cambia la risoluzione (oggi 2304 patch, oltre le 1024 di addestramento di NaFlex), l'osservabilità va
riverificata per la nuova configurazione (pipeline §5.7).

## 4. UniMER (`data/shards/unimer/2343ddd9-dense`, `latex.py`, `outputs/unimer_check/run4`)

| Compito | Corretto | Osservabile | Stato atteso |
| --- | --- | --- | --- |
| `transcribe` (formula intera) | C sulla sorgente (arXiv, Pix2tex); R per la struttura (`latex.structure_problems`, `unrendered_commands`); il confronto col disegno (`latex_render`) trova candidati e non è un oracolo | C: simboli piccoli, indici, ritaglio | `target_controllato`; `gold_verificato` solo per un piccolo nucleo con revisione competente |
| `transcribe_line` (riga ritagliata, `data.unimer_line_sample`) | come sopra, sulla riga | R: la fascia non contiene pezzi delle righe vicine (già nel codice); C sul ritaglio | idem |

Esiti di run4 (job 30047, su 985.450 campioni):

- 51 errori certi: `escluso`;
- 1.119 incerti, di cui 1.116 in 402 gruppi in conflitto: `quarantena`;
- 18.827 rietichettature: trasformazioni tracciate, con genitore e regola.

Il target è la formula senza gli spazi del formato im2latex (`latex.despace`). L'equivalenza ammessa è l'uguaglianza
dei token LaTeX a meno degli spazi (`latex.same_latex`), non l'equivalenza matematica: `x+x` non vale `2x`. UniMER-Test
(spe, cpe, sce, hwe) resta solo valutazione; hwe è manoscritto.

## 5. Collaudi (CPU, `tests_cpu`)

- **Intersezioni.** Conteggio geometrico e ammissibilità si collaudano separatamente.
  - `audit-68`: conteggio geometrico nel dominio disegnato 1; ammissibilità no, perché è a 1/3 dal bordo, dentro il
    margine di 0.4, quindi nessuna domanda. Con `version=3` la domanda esiste con target `0`, come negli shard v3.
  - Domanda ammessa con target 1, lontana dal bordo, su una scena reale: `synth-0049798`, due parabole con
    intersezione in (4.17, 6.51), a 0.49 dal bordo. Target `0` in v3, `1` in v4.
  - Caso costruito: y = x − 6 e y = −x + 7 si incontrano in (6.5, 0.5). Il conteggio di v3 dà `0` (|x| > R), quello di
    v4 dà `1`, ammesso perché a 0.5 dal bordo.
  - Intersezione con \|y\| ∈ (7.4, 7.5): non disegnata, quindi non contata.
  - y = x² + 0.2 e y = 0: nessuna intersezione, minimo 0.2 nel tratto disegnato. v3 pone la domanda (target `0`),
    v4 no, e la verifica la segnala.
  - Intersezioni a meno di 0.4 da \|x\| = 7 o \|y\| = 7: niente domanda.
  - Tangenza, due radici vicine, rette parallele: niente domanda.
  - Su 10.000 scene il conteggio analitico coincide con quello numerico sulle spezzate.
- **Unicità.** Su 10.000 specifiche per famiglia, esattamente un'opzione vera per ogni domanda di classificazione,
  calcolata con predicati scritti a parte. Un quadrato con le opzioni square/rectangle/parallelogram/trapezoid accetta
  solo `square`.
- **MAVIS.** Per un triangolo rettangolo con i cateti uguali, right e isosceles sono entrambe vere e non compaiono
  insieme fra le opzioni; se non si può costruire un insieme con una sola opzione vera, la domanda non si pone. Un rombo
  che dalle misure stampate è un quadrato dà `square`, oppure la domanda non si pone.
- **Tratti visibili.**
  - f = x + 1 e g = −x + 1 con tratti [−7, −2] e [2, 7]: conteggio 0, ammissibile. Il codice di `38af82a` dava 1.
  - Gli stessi grafici con un buco [−0.2, 0.2]: conteggio 0, non ammissibile (a 0.2 dal buco).
  - Grafico ripido che esce dal taglio a meno di un passo di campionamento dall'incrocio: non ammissibile.
- **Figure tagliate.**
  - `synth-0000043` in v3: il rettangolo L parte da x = −0.118847. Il verificatore sul disegno vero trova una figura
    tagliata di più di 1 px, il campione non è verificato e nessuna domanda resta ok.
  - La stessa chiave in v4: scena riestratta, tutte le figure dentro gli assi col tratto, domande ok.
  - Su 15.000 scene v4 nessuna figura vicina a un'altra o fuori dagli assi.
- **Mutazioni.** Una risposta alterata per ogni tipo viene trovata dalla verifica indipendente. Una scena costruita con
  un'etichetta fuori dall'immagine, due testi sovrapposti o due figure sovrapposte viene rifiutata dalle invarianti.
- **Esiti per campione** (§2.4; `test_sample_evidence`), su una scena shapes v4 semanticamente verificata:
  - PNG corrispondente: `byte_identici`, campione verificato;
  - PNG mancante: campione non verificato, con semantica verificata;
  - PNG di un'altra scena, con o senza layout nel record, oppure un pixel cambiato, o un file che non è un PNG:
    `diversa_da_verificare`, campione non verificato;
  - layout senza digest: si ridisegna. Col PNG giusto il campione è verificato, con un PNG sostituito no; un layout
    salvato svuotato non cambia la semantica;
  - layout con digest corrispondente: si usa quello salvato, e se gli manca l'area degli assi il campione non è
    verificato;
  - PNG ricodificato (byte diversi, pixel identici): `pixel_identici`, verificato; un pixel reso trasparente non lo è;
  - scena in memoria: `in_memoria`, nessun conteggio degli shard; senza layout, o con un PNG, la chiamata è rifiutata;
  - shard di cinque record, scritti con le funzioni di `synth_render.py` e letti da `shard_task`: tre verificati su
    cinque semanticamente validi, una riga col motivo per ciascuno dei due non verificati, conteggi coerenti. Un
    conteggio alterato viene segnalato;
  - `--cases` e `synth_render.py` v4 salvano lo sha256 del PNG scritto; v3 resta senza layout.
- **Regressione v3 → v4.** I vincoli delle shapes riestraggono solo le scene che li violano, e le altre specifiche
  restano identiche: su 8.000 chiavi sono riestratte esattamente le 122 scene shapes da correggere. Le domande dei tipi
  non toccati restano identiche, per tipo e per chiave. Cambiano solo le risposte di `intersections` e i testi di
  `shape_type` e `label_type`.
- **Valori.** `3/2` disegnato come `1.5`: in lettura la risposta è `1.5`, in calcolo entrambe le forme sono
  equivalenti.
- **UniMER.** Una rietichettatura non coerente con l'etichetta attesa dà errore (`stream.apply_relabel`, già presente).
