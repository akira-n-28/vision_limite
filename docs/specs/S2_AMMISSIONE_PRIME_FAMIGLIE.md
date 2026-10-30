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

## 2. Sintetici (`src/limite_vl/synth.py`, shard `data/shards/synth/v3` e `v4`)

Evidenza del 2026-10-28: le 60.000 scene di training di v3 sono state rigenerate dalle loro chiavi con il codice
attuale (`synth-NNNNNNN`, famiglie a rotazione come in `scripts/synth_render.py`). Il calcolo usa solo la libreria
standard e non legge gli shard. Gli istogrammi per famiglia, tipo e risposta coincidono esattamente con quelli del
manifest del job 29756: il codice attuale riproduce le domande di v3.

**Stato** (C1-C2): implementati il 2026-10-28 (`38af82a`) e corretti il 2026-10-29 dopo la revisione di quel commit,
che ha trovato figure tagliate, intersezioni contate nei buchi del disegno e un errore di sintassi nei test. Lo stesso
giorno la revisione di `f7d32d3` ha trovato che il report contava come verificato un campione senza aver dimostrato la
corrispondenza col PNG dello shard; ora gli esiti sono separati (§2.4). La suite CPU è passata il 2026-10-29 (job
30392, `b544b97`): `tests/test_synth.py` esegue 96 controlli col disegno vero, nessun fallimento. Fino al 2026-10-29
c'erano due scarti rispetto alla §5:
- intersezioni e `shape_type` si collaudavano su 4.000 scene e `label_type` su 400, invece delle 10.000 indicate. Per
  `label_type` il collaudo controllava solo il testo della domanda, non l'unicità con un predicato, e per le
  intersezioni mancava il conteggio numerico sulle spezzate. Il requisito resta valido: dal 2026-10-30 (`284e8f1`,
  job 30448) i tre collaudi girano su 10.000 scene ciascuno, con il conteggio numerico e il predicato (§5);
- i collaudi MAVIS appartengono a C9 e non sono implementati. Questo scarto resta.

Il controllo sugli shard (job 30393, conteggi nel piano F0 §4) ha dato tre risultati:
- i 60.900 PNG di v3, ridisegnati, hanno byte identici;
- il controllo ritrova i difetti noti di v3;
- sulle 20.000 scene v4 in memoria il checker non segnala nessuna risposta errata.

Le segnalazioni nuove, presenti anche nel codice v4, sono state analizzate nei job 30400 e 30420 (§2.5). Si separano
in tre gruppi:
- errori reali: 9 coppie polygon contraddittorie, provate con limiti esatti, e le collisioni fra misure ed etichette
  dei punti nei polygon;
- riquadri sovrapposti senza contatto dei caratteri, all'origine degli assi;
- un controllo più severo del necessario, sulle coordinate.

Le correzioni minime sono in `b0b90c5` (§2.5). Dopo di esse:
- su 20.000 scene v4 il checker approva 19.976 scene, e restano 25 sovrapposizioni in coord;
- le 300 coppie v4 sono tutte approvate;
- v3 resta identica.

I test e il controllo collaudano il codice, non i dati: nessun dato è certificato. Il 2026-10-30 il render degli shard
v4 è stato eseguito su incarico dell'utente (job 30447), con i controlli sugli artefatti salvati (job 30448-30449):
esiti e stato del candidato nella §2.6.
- `synth.py`: `version=4` è il default e `version=3` riproduce v3; sono nuovi `visible_bound`, `intersection_points`
  (conteggio geometrico), `_near_miss`, le domande di classificazione con le opzioni, `_shapes_overlap`,
  `_shapes_fit` e `render(spec, layout=True)`;
- `synth_check.py`: la verifica indipendente;
- `scripts/synth_semantics_check.py`: il controllo sugli shard e i casi singoli (`--cases`);
- `scripts/synth_case_svg.py`: schema SVG di un caso shapes, dalla specifica e senza matplotlib;
- `tests/test_synth.py`: i collaudi della §5;
- `scripts/synth_candidate.py`: i controlli del candidato sugli shard salvati (§2.6).

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
  fra `38af82a` e questa correzione (allora nessuno shard v4 era stato generato; quelli del 2026-10-30 portano il
  digest). Il layout salvato non prova il legame col PNG
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

### 2.5 Scarti del controllo sugli shard (job 30393): analisi e correzioni di v4

Due analisi con `scripts/synth_review.py`, uno strumento di analisi e non un passaggio della pipeline:
- job 30400, codice `8cd2c22`, uscite in `outputs/synth_review/scarti-run1`: prima delle correzioni;
- job 30420, codice `b0b90c5`, uscite in `outputs/synth_review/scarti-run2`: dopo le correzioni, con le prove esatte
  delle contraddizioni e le immagini prima/dopo.

Entrambe leggono controllo e shard delle coppie v3 in sola lettura e ridisegnano in memoria. Le scene del campione
ridisegnate oggetto per oggetto coincidono pixel per pixel con `synth.render`. I fogli di immagini sono in
[`docs/img/synth-review/`](../img/synth-review/).

Restano distinti tre livelli (§2.4):
- *approvato dal checker*: l'esito `verificato` del report;
- *correttezza dimostrata*: qui le 9 contraddizioni, dimostrate impossibili;
- *certificazione statistica* (S4): non fatta, nessun dato è certificato.

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
| contraddizione dimostrabile: tre lati stampati e un angolo dichiarato incompatibili anche con l'arrotondamento, con limiti esatti (tabella sotto) | 9: 00033, 00037, 00113, 00125, 00137, 00153, 00221, 00265, 00273 | `escluso` |
| contraddizione solo con gli interi esatti: 00285, angolo retto in K, 17² + 48² ≠ 50² | 1 | `quarantena` |
| compatibile con una riscalatura uniforme: una sola misura stampata | 26 | `quarantena` fino alla scelta del contratto |
| nessun vincolo dichiarato violato, ma la figura non è più simile al disegno: due o tre misure, per esempio 00009 con i lati opposti di un parallelogramma apparente a 7.6 e 11.6 | 39 | `quarantena` |
| misura cambiata coperta dall'etichetta del punto sul lato: 00209 illeggibile ("9" sotto "K"), 00169 e 00269 coperte in parte, 00017 e 00205 attaccate ma leggibili | 5, già contate sopra | `quarantena` |

**Prova delle contraddizioni (job 30420).** Nel job 30400 l'impossibilità veniva dal non aver trovato soluzioni su una
griglia di valori: non bastava a dimostrarla. Ora si prova con limiti esatti.
- Con tre lati stampati e un angolo dichiarato, il lato opposto all'angolo è funzione degli altri due lati e
  dell'angolo (teorema del coseno).
- I suoi limiti sugli intervalli d'arrotondamento sono esatti (`opposite_range`):
  - cresce con l'angolo;
  - a angolo fisso il suo quadrato è convesso nei due lati, quindi il massimo è in un vertice dei loro intervalli e il
    minimo sul bordo, dove ogni tratto è una parabola;
  - i limiti si allargano di 10⁻⁹ per la virgola mobile.
- Il margine è la distanza fra l'intervallo possibile del lato e quello stampato: se è positivo, il triangolo non
  esiste.

| Coppia | Vincolo dichiarato | Lato opposto possibile (interi ±0.5) | Scritto | Margine, interi ±0.5 | Margine, interi esatti |
| --- | --- | --- | --- | --- | --- |
| 00033 | 47° in T fra ST = 5.9 e TU = 11.7 | US fra 8.700 e 8.911 | 12.8 | 3.84 | 3.84 |
| 00037 | 68° in H fra LH = 10.8 e HK = 11 | KL fra 11.808 e 12.590 | 10.8 | 0.96 | 1.24 |
| 00113 | 66° in B fra DB = 12.1 e BC = 13 | CD fra 13.286 e 14.109 | 12.1 | 1.14 | 1.42 |
| 00125 | angolo retto in T fra ST = 8 e TR = 15 | RS fra 16.325 e 17.678 | 21 | 2.82 | 4.00 |
| 00137 | 61° in K fra MK = 10.6 e KL = 11 | LM fra 10.605 e 11.349 | 9.4 | 1.15 | 1.41 |
| 00153 | angolo retto in U fra WU = 11 e UV = 6 | VW fra 11.853 e 13.210 | 10 | 1.35 | 2.53 |
| 00221 | 41° in T fra ST = 8.5 e TR = 11.5 | RS fra 7.438 e 7.656 | 6 | 0.94 | 1.44 |
| 00265 | 54° in S fra US = 8.6 e ST = 14 | TU fra 10.865 e 11.807 | 8.6 | 2.21 | 2.61 |
| 00273 | 76° in B fra AB = 6 e BC = 15.4 | CA fra 14.953 e 15.299 | 12.4 | 2.50 | 2.57 |
| 00285 | angolo retto in K fra HK = 17 e KG = 48 | GH fra 50.284 e 51.561 | 50 | −0.22: compatibile | 0.92 |

Tutte e 9 sono confermate, con margini fra 0.94 e 3.84 unità. Nessun caso resta inconcludente e il conteggio non
cambia. 00285 resta in `quarantena`: è contraddittoria solo leggendo gli interi come esatti.

Per le altre 65 coppie la fattibilità è dimostrata:
- nei triangoli, da un punto della griglia che soddisfa tutti i vincoli dichiarati (un testimone);
- nei quadrilateri perché, senza angoli dichiarati, tre lati stampati al più si completano sempre in un quadrilatero.

Nessuna coppia si riammette in automatico.

Controlli dell'analisi stessa:
- le 75 varianti `a` risultano fattibili in entrambe le letture, e nessuna è contraddetta dai limiti esatti;
- `a` e `b` differiscono solo nelle misure;
- i PNG ridisegnati sono identici a quelli dello shard;
- `opposite_range` contiene ogni valore campionato su 300 scatole a caso, e la griglia arriva a meno di 4·10⁻⁵ dai
  suoi limiti (`test_review_bounds`).

Immagini: [esempi 1](../img/synth-review/coppie-esempi-1.png), [esempi 2](../img/synth-review/coppie-esempi-2.png), le
misure cambiate di tutte le coppie [1](../img/synth-review/coppie-misure-1.png),
[2](../img/synth-review/coppie-misure-2.png) e [3](../img/synth-review/coppie-misure-3.png).

**Contratto di v4: geometria coerente in scala** (adottato, `b0b90c5`).
- Ogni misura stampata è quella disegnata, entro l'arrotondamento mostrato; il checker lo richiede già
  (`lunghezza_incoerente`, `angolo_incoerente`). `side_len` legge un numero che coincide col disegno.
- La coppia polygon tiene stampata una sola misura, quella della domanda. La variante `-b` è il poligono riscalato:
  - fattore = valore nuovo desiderato / lunghezza esatta del lato, non il numero arrotondato;
  - le misure si ricalcolano dalla geometria nuova e si formattano come sempre;
  - angoli, angolo retto e punto sul lato restano uguali;
  - gli assi si adattano alla figura, quindi `b` differisce da `a` solo per il numero.
- È la classe "riscalatura uniforme", l'unica senza contraddizioni.

L'alternativa, "lettura della misura scritta" con la figura dichiarata non in scala, avrebbe due costi: verificare caso
per caso i vincoli dichiarati, come fatto qui, e lasciare figure incoerenti a vista come 00009.

**Riquadri di testo sovrapposti (v4 prima delle correzioni, 20.000 scene, job 30400).** Si sono censite tutte le
2.702 violazioni, in 1.797 scene; i
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

Regola nuova del checker (`synth_check.axis_value`). Il layout registra le tacche disegnate, solo come riquadri: il
gid non porta valori. La coordinata si legge alla tacca che sta sulla proiezione del punto:
- se la tacca ha un numero centrato sopra, vale quello;
- altrimenti la coordinata si accetta solo se valgono tutte queste condizioni:
  - le due tacche vicine hanno un numero;
  - sono alla stessa distanza;
  - tutti i numeri dell'asse stanno su una scala lineare (scarto ≤ 0.05 unità);
  - il valore che ne risulta è intero e coincide col punto.

Le posizioni vengono dai riquadri disegnati e i valori dal testo scritto. In ogni altro caso la domanda resta non
ammissibile, col motivo: tacca mancante, scala insufficiente o scala incoerente.

**Correzioni di v4** (`b0b90c5`). Valgono solo da v4: v3 resta identica. Le regressioni sono in `tests/test_synth.py`.

1. **Coppie polygon** (`synth.counterfactual`): una sola misura stampata e la variante per riscalatura uniforme, come
   nel contratto. Regressioni (`test_pairs_v4`):
   - su 60 coppie, `shown` è il solo lato della domanda in `a` e `b`, e i punti di `b` sono quelli di `a` per il
     fattore;
   - le misure sono ricalcolate dalla geometria; angolo, angolo retto e punto segnato sono uguali;
   - la risposta è la misura disegnata; nessuna `lunghezza_incoerente` né `angolo_incoerente`; il punto segnato sta
     sul suo lato;
   - fra `a` e `b` cambiano solo i pixel nel riquadro della misura;
   - le prime quattro coppie v3 dello shard si riproducono identiche, PNG e JSON.
2. **Punto segnato dei polygon** (`synth._place_mark`): mai su un lato con la misura stampata.
   - Il lato nuovo si sceglie con un generatore a parte, così il resto della scena resta quello di v3.
   - Se tutti i lati hanno la misura (triangoli con tre lati scritti) il punto si toglie, e con lui la sola domanda
     `m_side`.
   - Regressioni (`test_marked_point_v4`, `test_v3_v4_regression`): su 2.500 scene v4 il punto non sta mai su un lato
     con la misura e si toglie solo senza lati liberi; scelta deterministica. Su 300 scene disegnate nessuna
     sovrapposizione fra misure, punti segnati e loro etichette, e `m_side` e `side_len` sono ok. Da v3 a v4
     cambiano solo le shapes da riestrarre e il lato del punto segnato.
3. **Tacche -1 in coord e function** (`synth._draw_axes`): senza numero, e le tacche restano.
   - Regressioni (`test_origin_labels_v4`): su 166 scene con tacche, nessun numero -1, tacche -1 disegnate su
     entrambi gli assi e nessuna sovrapposizione all'origine; domande coord e function tutte ok.
   - In v3 i numeri -1 restano.
4. **Coordinate da tacche vicine** (`synth_check.axis_value`): regola sopra. Il layout registra le tacche: cambiano i
   metadati, non i pixel.
   - Regressioni (`test_coords_from_ticks`): positivi con una coordinata e con entrambe senza numero, e il caso v3
     `synth-0010596`.
   - Negativi con tacca mancante o spostata, tacca vicina senza numero, un solo numero sull'asse, numero vicino
     sbagliato, tacca vicina spostata: tutti non ammissibili. Una risposta alterata è errata.
   - Un gid sbagliato col testo giusto resta ok: il gid non conta.

`OVERLAP_PX` non cambia. `scripts/synth_semantics_check.py --pairs N` controlla in memoria le coppie come le
scriverebbe il render.

**Dopo le correzioni (job 30417-30420, codice `b0b90c5`).**
- **Suite CPU (30417).** 23 file, 892 controlli, nessun fallimento; `tests/test_synth.py` ne ha 117 (21 nuovi).
- **Controllo (30418, `outputs/synth_check/c1c2-run2`).** Conteggi coerenti.
  - v3: i 60.900 PNG hanno ancora byte identici agli shard. 38.325 campioni sono approvati dal checker: 164 in più,
    perché le 169 domande `coords` si leggono ora dalle tacche. Il resto è identico: 37 errate, stesse violazioni.
  - v4: 19.976 scene approvate su 20.000. Le 24 non approvate sono coord con 25 sovrapposizioni fra etichetta di un
    punto e numero di tacca, la categoria non toccata. Nessuna domanda non ammissibile, nessuna segnalata errata.
- **Coppie (30419, `outputs/synth_check/c1c2-pairs-run1`).** 300 coppie v4, 600 record tutti approvati dal checker,
  600 domande ok, nessuna violazione.
- **Analisi (30420).** Restano 25 violazioni in 24 scene: 1 con inchiostro in comune, 24 separate. Le prove esatte
  sono nella tabella sopra. Le 16 scene del campione corrette non hanno più sovrapposizioni.
- **Copertura delle domande** sulle stesse 20.000 scene v4, prima e dopo:
  - i conteggi di tutti i tipi restano uguali, tranne `m_side`: da 2.471 a 2.110 (−361, −14.6%), per il punto tolto
    nei triangoli con tutti e tre i lati misurati;
  - `coords` passa da 10.579 ok e 58 non ammissibili a 10.637 ok.

  Nessun altro compito è stato tolto.
- **Immagini prima/dopo:** [polygon](../img/synth-review/prima-dopo-polygon.png),
  [origine](../img/synth-review/prima-dopo-origine.png), [coppie](../img/synth-review/prima-dopo-coppie.png),
  [coordinate](../img/synth-review/prima-dopo-coords.png). Le scene residue sono nel
  [foglio](../img/synth-review/campione-altro.png) della prima analisi, con la stessa diagnosi: riquadri sovrapposti,
  caratteri leggibili, una possibile associazione ambigua ("R8").

Livelli:
- l'approvazione del checker riguarda 19.976 scene e 600 record di coppie;
- la coerenza fra misura stampata e disegno nelle coppie v4 vale per costruzione, ed è controllata dai test su 60
  coppie e dal checker su 300; non è una certificazione statistica, che resta da fare (S4).

Le tre questioni rimaste aperte (le 25 sovrapposizioni residue, la perdita di copertura di `m_side`, 4.000 o 10.000
scene nei collaudi) sono chiuse dalle decisioni del 2026-10-30 (§2.6).

### 2.6 Candidato v4 (job 30447-30449, codice `284e8f1`, 2026-10-30)

**Decisioni dell'utente (2026-10-30)** per il primo candidato:
- la perdita di copertura di `m_side` è accettata, con i conteggi per compito visibili nel rapporto;
- le scene con sovrapposizioni segnalate vanno in quarantena temporanea. Non sono contate come errori matematici: le
  24 scene del campione non sono tutte errate;
- nessuna soglia si allenta: `OVERLAP_PX`, `MIN_TEXT_PX` e le altre sono quelle del checker (`report.json`, campo
  `thresholds`);
- gli stessi criteri valgono per tutte le scene generate, non solo per gli ID già osservati;
- il requisito delle 10.000 scene nei collaudi resta (§5).

**Generazione** (job 30447, `slurm/synth_render.sbatch v4 --generator-version 4`). Si sono usati dimensioni, chiavi e
split di v3, cioè il manifest del job 29756 e i default di `synth_render.py`; nessun parametro è stato aggiunto.
- **Train:** 60.000 scene da `synth-0000000` a `synth-0059999`, con le famiglie a rotazione (`i % 4`). Sono 6 shard da
  10.000.
- **Test:** 300 scene `synth-test-NNNNN`.
- **Coppie:** 300 coppie `synth-pair-NNNNN-a`/`-b`, cioè 600 record. Test e coppie sono gli insiemi di valutazione di
  `scripts/f1c_eval.py`.
- **Semi:** `synth.rng_for(chiave)` (sha256 della chiave). Le domande usano `chiave + "/qa"`, il punto segnato dei
  polygon `chiave + "/m"`, le coppie `chiave + "/cf"`, con i tentativi da `chiave/0` a `chiave/19`.
- **Uscita:** `data/shards/synth/v4`, 892 MiB fuori da git. Il manifest registra gli sha256 degli shard e l'identità del
  codice. All'avvio del job era modificato solo `runs/ledger.tsv`, per le righe di invio. v3 non è stata toccata.
- **Tempo:** 6 min 18 s.

Il dev split non è definito. S1 lo sceglie dal manifest (C5-C7, non implementato); la validazione storica, i primi 512
campioni dell'ultimo shard, non si usa per i run nuovi (S1 §3).

**Controllo degli shard salvati** (job 30448, `outputs/synth_check/v4-run1`). Prima si esegue `tests/test_synth.py`:
128 controlli, nessun fallimento (§5). Poi `synth_semantics_check.py` legge in sola lettura gli 8 shard di train, test
e coppie:
- 60.900 record, tutti con immagine `digest_corrispondente`: il layout salvato porta lo sha256 del PNG dello shard;
- 60.833 sono approvati dal checker. Gli altri 67 hanno solo testi sovrapposti. Nessuna domanda risulta errata, non
  ammissibile o non verificabile, e non ci sono violazioni bloccanti.

**Candidato** (job 30449, `scripts/synth_candidate.py`, uscite in `outputs/synth_candidate/v4-run1`). I controlli
riguardano ogni record degli shard salvati:

| Controllo | Che cosa | Esito |
| --- | --- | --- |
| integrità degli shard | sha256 e dimensione uguali al manifest del render, nessun altro file nelle cartelle; nel tar un PNG e un JSON per chiave, nessun altro membro, le chiavi attese nell'ordine del render | 8 shard su 8, nessun problema |
| integrità dei record | campi del JSON; risposta fra le opzioni, opzioni distinte; PNG che si decodifica, RGB, alle dimensioni del record; lati multipli di 32; `max_patches` uguale alle patch dell'immagine; layout con `png_sha256` uguale allo sha256 del PNG | 60.900 su 60.900 |
| immagine vista dal modello | `processing.target_size` lascia invariata ognuna delle 49 dimensioni, tutte entro `DIAGRAM_MAX_PATCHES` | 49 su 49 |
| riproducibilità | specifica e domande rigenerate dalla chiave; PNG ridisegnato con byte identici, layout identico; coppie rigenerate intere come in `synth_render.scene` | 60.900 su 60.900 |
| verifica indipendente | `synth_check.check_sample` sul layout salvato; i non approvati devono coincidere con quelli del job 30448 | 67 non approvati, coincidenti |
| coppie | stessa domanda e stesse opzioni in a e b, risposte diverse, immagini diverse; un solo dettaglio cambiato: un punto riflesso (coord), il segno di `a` di un grafico (function), la figura riscalata con una sola misura scritta (polygon), una figura in più o in meno (shapes) | 300 coppie su 300 |
| formato | domande ripetute nella scena, spazi in testa, in coda o doppi | nessun problema |

**Duplicati e split.** Si applicano le regole già previste:
- pipeline §5.5: per i sintetici si separano stati e istanze, e le coppie restano nello stesso split;
- pipeline §5.8.5 e S1 §§3-4: legami confermati, e un gruppo legato alla valutazione è escluso dal training;
- `scripts/split_duplicates.py`: il training che ripete la valutazione si toglie, i duplicati interni si contano
  soltanto.

Come legami confermati valgono solo identità esatte:
- **pixel:** sha256 dei valori RGBA decodificati;
- **stato della scena:** la specifica senza chiave, versione e stile (spessore, carattere, dimensioni dell'immagine),
  cioè la stessa istanza disegnata con un altro stile.

Nessun candidato per somiglianza. Esiti:
- **pixel:** nessun duplicato, né dentro uno split né fra split.
- **stato, dentro il train:** 51.995 stati distinti su 60.000 scene. I duplicati sono quasi tutti in `function`, con
  7.074 stati per 15.000 scene: i grafici hanno coefficienti interi da pochi valori e R = 6, quindi con un solo grafico
  ci sono circa 600 stati possibili. In polygon 79 scene ripetono lo stato di un'altra (14.921 stati), in coord e
  shapes nessuna. Si contano soltanto.
- **stato, fra split:** 79 gruppi, di cui 36 test+train, 33 coppie+train e 10 coppie+test+train. 2.792 scene di train
  `function` hanno lo stesso stato di almeno un record di valutazione: sono escluse dal training con il motivo
  `contaminazione`. Test e coppie restano interi.
- **coppie:** ogni coppia a/b è un solo gruppo nello split delle coppie.
- **valutazione:** test e coppie condividono stati `function` (10 gruppi); dentro il test ci sono 6 record con lo
  stato ripetuto, dentro le coppie 48. Per `function` i due insiemi di valutazione non sono indipendenti fra loro.

**Stato delle scene.** Per le coppie vale lo stato del gruppo a/b.

| Split | Record | `candidato` | `quarantena` | `escluso` | Motivi |
| --- | --- | --- | --- | --- | --- |
| train | 60.000 | 57.141 | 67 | 2.792 | quarantena: `incerto`, testi sovrapposti (65 coord, 2 function); escluso: `contaminazione` (function) |
| test | 300 | 300 | 0 | 0 | — |
| coppie | 600 (300 coppie) | 600 | 0 | 0 | — |

Per categoria, su tutti i 60.900 record:
- `errore_matematico` (domanda errata, misura o angolo scritti incoerenti col disegno): 0;
- `visivo_integrita` (violazione bloccante, artefatto non integro o non riproducibile, contratto della coppia): 0;
- `formato_minore`: 0;
- `incerto`, in quarantena: 67;
- `contaminazione`: 2.792.

Le scene in quarantena hanno 77 sovrapposizioni:
- 73 fra l'etichetta di un punto e il numero di una tacca, in coord;
- 2 fra la `O` dell'origine e l'etichetta di un punto, in coord (`synth-0011772`, `synth-0048520`);
- 2 fra il nome del grafico `g` e la tacca `-3`, in function (`synth-0004682`, `synth-0024890`). È una combinazione
  nuova rispetto alle 20.000 scene in memoria.

In tutte e 67 le domande risultano ok sul layout: la quarantena dice che l'immagine può essere ambigua, non che la
risposta sia sbagliata. Elenchi con i motivi: `excluded.tsv` e `quarantine.tsv`; lo stato, i motivi e gli hash di ogni
record sono in `records.jsonl`.

**Copertura del train per compito.** Colonne: le domande di v3 (manifest del job 29756), quelle generate in v4 prima
dei filtri e quelle delle scene `candidato`.

| Famiglia | Tipo | v3 | v4 prima | v4 dopo | dopo / prima |
| --- | --- | --- | --- | --- | --- |
| coord | `coords` | 31.576 | 31.576 | 31.328 | −0.8% |
| coord | `count`, `exists` | 15.000, 30.000 | 15.000, 30.000 | 14.935, 29.870 | −0.4% |
| coord | `line_on` | 3.819 | 3.819 | 3.793 | −0.7% |
| coord | `pair_lr`, `pair_ud` | 122.238, 121.624 | uguali | 121.510, 120.898 | −0.6% |
| coord | `quadrant`, `side_x`, `side_y` | 52.485 ciascuno | uguali | 52.231 ciascuno | −0.5% |
| function | `exists` | 30.000 | 30.000 | 24.412 | −18.6% |
| function | `intercept` | 22.460 | 22.460 | 19.634 | −12.6% |
| function | `intersections` | 6.307 | 6.436 | 6.411 | −0.4% |
| function | `opens`, `vertex_ud` | 8.949 | 8.949 | 8.696 | −2.8% |
| function | `slope` | 13.511 | 13.511 | 10.938 | −19.0% |
| function | `vertex_lr` | 7.409 | 7.409 | 7.196 | −2.9% |
| polygon | `m_side` | 7.634 | 6.552 | 6.552 | 0 (−14.2% da v3) |
| polygon | gli altri 10 tipi (sotto) | uguali | uguali | uguali | 0 |
| shapes | `has_type` | 28.743 | 28.707 | 28.707 | 0 |
| shapes | `label_type` | 37.522 | 37.046 | 37.046 | 0 |
| shapes | `count`, `count_type`, `exists` | 15.000, 15.000, 30.000 | uguali | uguali | 0 |

Scene di train per famiglia, prima e dopo: coord 15.000 → 14.935, function 15.000 → 12.206, polygon e shapes 15.000 →
15.000.
- In polygon i tipi invariati sono `angle_val` 3.014, `count` 15.000, `exists` 30.000, `highest` 9.313, `leftmost`
  9.869, `lowest` 5.870, `right_vertex` 2.128, `rightmost` 9.839, `shape_type` 12.847 e `side_len` 30.060.
- `m_side` perde 1.082 domande da v3 a v4 (−14.2%), come le −14.6% sulle 20.000 scene in memoria.
- Nelle shapes le differenze con v3 vengono dalle scene riestratte (§2.4).
- In function la contaminazione colpisce soprattutto le scene con una sola retta: `slope` perde il 19%, `opens` il 2.8%.

Test e coppie sono interi:
- test: 75 scene per famiglia, con le domande del manifest del render;
- coppie, per famiglia e domanda: coord 35 `side_x` e 40 `side_y`; function 22 `opens` e 53 `slope`; polygon 75
  `side_len`; shapes 75 `count`.

Le distribuzioni delle risposte dopo i filtri sono nel campo `answers_after` del report.

**Costo.** Tre job CPU con 8 CPU (4 core fisici) e 16G ciascuno:
- 30447 (render): 6 min 18 s;
- 30448 (test e controllo): 1 min 50 s;
- 30449 (candidato, con il ridisegno di tutte le scene): 9 min 38 s.

In tutto 17 min 46 s di tempo reale e 2 h 22 min di CPU allocate (`CPUTime` di `sacct`); l'accounting non registra
`TotalCPU` né `MaxRSS`. Su disco: 892 MiB di shard e 59 MiB di uscite del candidato, quasi tutte in `records.jsonl`.

**Che cosa è verificato** su tutti i 60.900 record:
- integrità e riproducibilità: dalla chiave si riottengono specifica, domande, PNG e layout identici;
- approvazione del checker: semantica delle domande sul layout e invarianti del renderer;
- contratto delle coppie;
- immagine lasciata invariata dal processor;
- regole degli split, per identità esatta.

**Che cosa non è coperto:**
- **checker sul layout.** Il checker legge il layout, non i pixel. Non controlla un grafico che attraversa un'etichetta di
  tacca (job 30400) né le vicinanze ambigue senza sovrapposizione dei riquadri, come "R8". La soglia di 10 px per
  l'altezza dei testi è una proposta.
- **indipendenza parziale.** La verifica prende dalla specifica le formule dei grafici. Predicati di classificazione,
  contratto delle coppie e definizione di stato sono scritti dallo stesso progetto: errori concettuali condivisi non
  sono esclusi.
- **nessuna stima campionaria.** Non ci sono certificazione statistica (S4), audit informativo su un campione
  (pipeline §15.3) né audit con agenti, che non è autorizzato. Superare il checker non dimostra che ogni risposta sia
  corretta.
- **decontaminazione esterna.** Mancano i controlli contro i benchmark riservati, che richiedono l'indice globale
  (C5-C7). L'evidenza storica `data/decontam/f1eval/synth` riguarda v3 (6 candidati pHash, 0 confermati) e non vale
  per v4.
- **legami solo esatti.** Due scene che differiscono solo per un elemento che non cambia le risposte (per esempio le
  tacche in function) non sono legate.
- **stati S2.** Le evidenze 1-5 della §1 ora ci sono per i tipi v4, ma S2 è una bozza e i tipi non sono promossi: i
  record sono `candidato`, non `gold_verificato` né `target_controllato`.

**Stato del candidato: ancora incompleto per gli esperimenti** (pipeline §15.3). I dati sono generati e superano tutti
i controlli automatici dichiarati, su tutto il candidato. Mancano tre condizioni di "pronto per un esperimento":
- **manifest nei loader.** Il manifest non è applicato ai compiti caricati: `excluded.tsv` e `quarantine.tsv` sono TSV
  con intestazione, mentre `Source.exclude` vuole una chiave per riga, e il manifest S1 non esiste (C5-C7).
- **dev split.** Non è definito.
- **audit campionario.** Manca l'audit informativo su un campione.

Problemi residui da decidere:
1. **Stati `function` ripetuti.** Lo spazio degli stati è piccolo, soprattutto con un solo grafico. Per questo la
   regola degli split toglie il 18.6% delle scene `function` di train, e test e coppie condividono stati. Le opzioni:
   - tenere l'esclusione, come in questo candidato;
   - in una versione futura del generatore ampliare lo spazio degli stati, per esempio con più valori dei
     coefficienti;
   - riservare in anticipo alla valutazione un insieme di stati.

   Proposta: la prima opzione per questo candidato e la seconda per la versione successiva.
2. **Le 67 scene in quarantena.** Si possono guardare, correggere nel renderer in una versione futura (etichette dei
   punti lontane dai numeri delle tacche) o escludere.
3. **Uso nei loader.** Per usare il candidato servono una lista delle chiavi per `Source.exclude`, o il manifest S1, e
   la scelta del dev split.

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
  - Su 10.000 scene il conteggio analitico coincide con quello numerico sulle spezzate. Il conteggio numerico conta i
    cambi di segno della differenza fra campioni consecutivi disegnati per entrambi i grafici, senza le formule delle
    radici. Dal 2026-10-30 (job 30448): 4.449 domande confermate da entrambi; prima erano 4.000 scene e solo la
    verifica analitica.
- **Unicità.** Su 10.000 specifiche per famiglia, esattamente un'opzione vera per ogni domanda di classificazione,
  calcolata con predicati scritti a parte: `shape_type` sui vertici dei polygon, `label_type` sui lati delle figure
  delle shapes (24.486 domande nel job 30448; prima 400 scene e solo il testo della domanda). Un quadrato con le opzioni
  square/rectangle/parallelogram/trapezoid accetta
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
  `shape_type` e `label_type`. Da `b0b90c5` nei polygon cambia anche il lato del punto segnato, in 591 scene su 2.000:
  il punto si sposta, o si toglie se tutti i lati hanno la misura. Il resto della scena resta uguale.
- **Correzioni dopo l'analisi degli scarti** (§2.5):
  - coppie polygon per riscalatura uniforme (`test_pairs_v4`);
  - punto segnato lontano dalle misure (`test_marked_point_v4`);
  - tacche -1 senza numero (`test_origin_labels_v4`);
  - coordinate dalle tacche vicine, con casi positivi e negativi (`test_coords_from_ticks`);
  - limiti esatti delle contraddizioni dell'analisi (`test_review_bounds`).
- **Candidato** (§2.6, `test_candidate`):
  - categorie e stati su esiti costruiti: testi sovrapposti, testo piccolo, domanda non ammissibile e violazione
    sconosciuta in quarantena; domanda errata, misura incoerente, figura tagliata e artefatto non riproducibile
    esclusi;
  - coppie vere delle quattro famiglie conformi al contratto; coppie alterate segnalate (due punti cambiati, figura non
    riscalata, due figure in più, stessa risposta, nome del grafico cambiato);
  - legami e split su righe costruite: il training con lo stato di un test o di una variante di coppia è escluso, i
    pixel uguali dentro il train si contano e basta, la coppia è un gruppo con lo stato peggiore;
  - un PNG sostituito: digest e ridisegno diversi, escluso solo quel record; chiavi in un altro ordine: problema di
    struttura;
  - la catena render → controllo → candidato su 12 scene di train, 4 di test e 4 coppie: nessun errore, controllo
    coincidente, immagini lasciate invariate dal processor.
- **Valori.** `3/2` disegnato come `1.5`: in lettura la risposta è `1.5`, in calcolo entrambe le forme sono
  equivalenti.
- **UniMER.** Una rietichettatura non coerente con l'etichetta attesa dà errore (`stream.apply_relabel`, già presente).
