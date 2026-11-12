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
- i collaudi MAVIS appartengono a C9: implementati il 2026-11-10 in `tests/test_mavis_c9.py` (job 30775, §§3 e 5).

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

Aggiornamento dello stesso giorno: il manifest nei loader e il dev split sono fatti (§2.7). Restano l'audit campionario
e la decontaminazione esterna (§2.8).

### 2.7 Manifest delle ammissioni e dev split (job 30450-30451, codice `630d00a`, 2026-10-30)

**Decisioni dell'utente (2026-10-30)** per questo candidato:
- restano escluse dal training le 2.792 scene `function` legate alla valutazione;
- restano in quarantena le 67 scene con sovrapposizioni;
- le riduzioni di copertura della §2.6 sono accettate;
- shard v3 e v4, test e coppie restano come sono, e il generatore non cambia ora;
- dev: circa il 5% delle scene ammesse, stratificato per famiglia, seme 0, gruppi indivisibili. S1 non fissava né quota
  né seme (S1 §9).

**Manifest** `data/manifests/synth-v4-candidato-1` (45 MiB, fuori da git; sottoinsieme di S1, S1 §9).
- **Costruzione** (`scripts/synth_manifest.py`). Parte dalle uscite del candidato (job 30449) e non ricontrolla gli
  shard. Verifica gli sha256 del render, del candidato, degli shard e di ogni JSON e PNG.
- **Gruppi.** Ricalcolati dai legami esatti, coincidono con quelli del candidato: 52.457. Ogni gruppo ha una sola
  famiglia.
- **Record.** Una riga per record, 60.900, con `record_id`, gruppo, split, stato, motivi, hash di immagine, JSON e QA,
  e le evidenze (job 30448, 30449) per riferimento.
- **File derivati** in `keys/`, una chiave per riga: `train` 54.271, `dev` 2.870, `test` 300, `coppie` 600,
  `non_ammessi` 2.859.
- **Identità.** sha256 di `manifest.json` `975dfe42…`. All'avvio del job erano modificati solo documenti e ledger.

| Insieme | Split | Stato | Record |
| --- | --- | --- | --- |
| train | train | `ammesso` | 54.271 |
| train | train | `quarantena` (testi sovrapposti) | 67 |
| train | dev | `ammesso` | 2.870 |
| train | test | `escluso` (contaminazione: gruppo con un record di test o delle coppie) | 2.792 |
| test | test | `ammesso` | 300 |
| coppie | test | `ammesso` | 600 (300 coppie) |

I gruppi sono 49.277 in train, 2.608 in dev e 572 in test. Questi ultimi comprendono i record di train esclusi.

**Dev per famiglia** (obiettivo: 5% delle scene ammesse fuori dalla valutazione):
- coord: 747 su 14.935 (obiettivo 747);
- function: 623 su 12.206 (obiettivo 610). L'ultimo gruppo preso ha 50 scene, uno stato frequente con una sola retta;
- polygon e shapes: 750 su 15.000 ciascuna.

In tutto 2.870 scene, il 5.02% delle 57.141 ammesse.

**Copertura per compito.** Per ogni compito il dev ha fra il 4.7% (polygon `lowest`, 276 su 5.870) e il 5.4%
(function `vertex_lr`, 391 su 7.196) delle QA ammesse. I compiti più piccoli in dev sono polygon `right_vertex` (110),
`angle_val` (151) e coord `line_on` (187). I conteggi completi per split, famiglia e compito sono in `manifest.json` e
nel report delle verifiche.

Il loader costruisce per ogni scena una sola variante, da 1 a 5 domande scelte da `f1c.pick_questions`:

| Famiglia | Domande costruite, train | Domande costruite, dev |
| --- | --- | --- |
| coord | 42.692 | 2.369 |
| function | 34.326 | 1.895 |
| polygon | 42.735 | 2.303 |
| shapes | 42.846 | 2.212 |
| totale | 162.599 | 8.779 |

**Verifiche CPU** (job 30451, `outputs/manifest_check/synth-v4-candidato-1`, `scripts/manifest_check.py`): 32
controlli, nessun fallimento.
- **Integrità.**
  - Il manifest si legge con tutte le verifiche: sha256 dei record, `record_id` ricalcolati, un solo split per gruppo,
    conteggi per shard.
  - Gli shard su disco hanno gli sha256 del manifest.
  - I file derivati coincidono con le chiavi del manifest e con i loro sha256.
- **Riproducibilità dello split.** Ricalcolato due volte con quota e seme del manifest, è identico. Con seme 1 cambia
  per 4.948 gruppi.
- **Gruppi.**
  - I legami ricalcolati dagli hash del manifest danno gli stessi 52.457 gruppi.
  - Nessun gruppo è in due split.
  - Nessun gruppo legato a test o coppie è in train o dev.
- **Loader.**
  - Per train, dev, test e coppie, un'epoca con 2 worker di `SourceReader` consegna esattamente le chiavi ammesse dello
    split (54.271, 2.870, 300 e 600), ciascuna una volta.
  - Non arriva nessuna chiave esclusa, in quarantena o di un altro split, e nessun gruppo compare in due split.
  - Le domande costruite hanno i digest delle QA ammesse e i conteggi per compito attesi dal manifest.
  - `validation_samples` restituisce tutto il dev.
  - `MixedRowStream` col processor di Violetto: in 30 micro-batch 638 esempi caricati, solo chiavi di train, nessun
    errore del manifest.
- **Arresti espliciti.** Ognuno dei casi seguenti ferma il caricamento con `ManifestError`:
  - manifest mancante, o senza split;
  - `records.jsonl.gz` alterato;
  - sha256 di uno shard diverso;
  - gruppo spostato in un altro split;
  - `record_id` non ricalcolabile;
  - shard v3 col manifest v4, o shard v4 senza manifest;
  - split senza record ammessi, o split sconosciuto.

Prima del job la suite CPU (job 30450) ha eseguito 24 file e 937 controlli, nessun fallimento.
`tests/test_manifest.py` ne aggiunge 34 su shard piccoli costruiti nel test: forma canonica, collisione forzata,
prefissi, regola degli split, reader, validazione, contatori, arresti, record alterato contato nel flusso.

**Valutazione.** Test e coppie restano riservati, nello split `test`. Condividono 10 gruppi `function`, quindi le
metriche dei due insiemi non sono conferme statisticamente indipendenti. Anche dentro ciascun insieme ci sono stati
ripetuti: 6 record nel test e 48 nelle coppie (§2.6).

**Uso in una config.** La validazione viene dallo split dev senza altri campi. La famiglia `synth` richiede ancora la
lista v3 vuota `data/decontam/f1eval/synth/exclude_final.txt` (`data_lists.REQUIRED`), che non è un'evidenza su v4
(§2.8).

```json
{"name": "synth-v4", "kind": "synth_qa", "weight": 1.0,
 "shards": "/home/prignano/vision_limite/data/shards/synth/v4/train/*.tar",
 "manifest": "/home/prignano/vision_limite/data/manifests/synth-v4-candidato-1", "split": "train",
 "exclude": ["/home/prignano/vision_limite/data/decontam/f1eval/synth/exclude_final.txt"]}
```

**Costo:**
- job 30450 (suite CPU): 10 min 46 s, 4 CPU;
- job 30451: 4 min, 8 CPU (manifest 23 s, verifiche 128 s);
- in tutto 1 h 15 min di CPU allocate; su disco 45 MiB di manifest e 21 KB di report.

**Stato.** Il filtro nei loader e il dev split sono applicati, con nessun errore rilevato dai controlli eseguiti. Il
candidato resta **ancora incompleto per gli esperimenti** (pipeline §15.3), per tre motivi:
- manca l'audit campionario informativo (§2.8);
- la decontaminazione esterna di v4 non è fatta (§2.8);
- il manifest copre record e gruppi, ma non le varianti né il contratto (S1 §9).

Il candidato non è certificato e non è pronto per il training.

Aggiornamento del 2026-11-02: la decontaminazione esterna è eseguita, con il manifest nuovo
`data/manifests/synth-v4-candidato-2` (§2.9). L'audit campionario è preparato ma non eseguito (§2.10). Restano
l'audit, la verifica dei 264 candidati, le varianti e il contratto.

### 2.8 Audit visivo e decontaminazione esterna di v4: piano (2026-10-30)

Sono le due condizioni che, dopo il manifest (§2.7), mancano per "pronto per un esperimento" (pipeline §15.3; S4
§1). Il 2026-10-30 c'era solo il piano. Il 2026-11-02 la decontaminazione esterna è eseguita (§2.9) e il campione
dell'audit è preparato (§2.10). Il 2026-11-04 i fogli sono rifatti dagli stessi artefatti, con la stessa selezione
(§2.11); l'audit non è iniziato.

**Audit visivo.** È una revisione umana delle immagini come le vede il modello, cioè il PNG, che il processor lascia
invariato (§2.6). Non si usano agenti LLM. I due campioni restano separati (S4 §2).

1. **Campione casuale, di stima (informativo).**
   - **Proprietà:** la risposta è quella che si vede (correttezza) e l'informazione necessaria è visibile e leggibile
     (osservabilità), S2 §1.
   - **Quantità e unità:** la quota di QA errate fra le QA che il loader costruisce (`f1c.pick_questions`) per le scene
     ammesse di train e dev. L'unità è la QA costruita: record, posizione, digest.
   - **Strati e popolazioni** (manifest `synth-v4-candidato-1`, §2.7):
     - le quattro famiglie, con le QA costruite di train e dev: coord N = 45.061, function 36.221, polygon 45.038,
       shapes 45.058;
     - il test, con tutte le QA delle 300 scene, come in `f1c_eval.py`: N = 4.038;
     - le coppie, N = 300. Una coppia è un'unità sola: le due varianti si giudicano insieme.
   - **Numerosità proposta:** 75 QA per famiglia, 50 QA del test e 30 coppie, 380 unità in tutto. Con 0 errori il limite
     superiore al 95% (ipergeometrico, S4 §3) è circa:
     - 3.9% per famiglia;
     - 5.8% per il test;
     - 9.0% per le coppie.

     Si fa un'affermazione per strato, senza unirle.
   - **Selezione riproducibile:** in ogni strato le n unità con lo sha256 più piccolo di `{"audit": "synth-v4-1",
     "seme": 0, "record_id", "posizione", "qa"}`. È un'estrazione senza reinserimento con probabilità uguali n/N. Il
     file del campione registra seme, N, n, probabilità d'inclusione e sha256 del manifest; la numerosità non si
     estende dopo l'estrazione.
   - **Procedura** (corretta il 2026-11-04, §2.11). Nel primo passaggio il revisore vede l'immagine e il prompt
     effettivo, senza target e senza opzioni aggiunte, e risponde prima di vedere il target (S4 §5):
     - **train e dev:** il testo dell'utente costruito dal loader di training (`data.synth_qa_sample`). Sono da 1 a 5
       domande numerate; la QA campionata è la domanda `numero_nel_prompt`. Le opzioni si vedono solo quando fanno
       parte della domanda, come in `shape_type` e `label_type`;
     - **test e coppie:** ogni QA da sola, col prefisso di `f1c.qa_messages` per una domanda. È il protocollo del foglio,
       perché quello di valutazione di F0 non è fissato (D3). `f1c_eval.py`, storico, usava lo stesso prefisso e dava un
       punteggio alle opzioni senza mostrarle. Il loader di training non costruisce il prompt delle coppie.

     Dopo il primo giudizio si consultano le opzioni (`opzioni.tsv`) e il target (`soluzioni.tsv`), poi si confronta. Un
     disaccordo si risolve con layout e specifica, e si documenta. Incerti e casi non giudicabili contano come errori
     nel limite, e la loro quota si riporta sempre. Nelle coppie si registra il giudizio su ciascuna variante (a e b) e
     sul contratto della coppia.
   - **Risultato:** condizionato alla sensibilità del revisore, che non è misurata (S4 §4).
2. **Casi mirati, diagnostici.** Contano errori e falsi allarmi, ma non stimano un tasso. Sono circa 170:
   - le 67 scene in quarantena, tutte. Per ciascuna: leggibile e univoca (proposta di riammissione in una versione
     nuova del manifest) oppure ambigua (esclusione);
   - scene approvate vicino alle soglie, 8 per soglia: riquadri di testo appena sopra `OVERLAP_PX`, testi poco sopra
     `MIN_TEXT_PX`, intersezioni vicine a `EDGE`, grafici vicini a `TANGENT`, coordinate lette dalle tacche vicine,
     vertici estremi vicini a `EXTREME`;
   - 20 coppie, 5 per famiglia, con le due immagini affiancate;
   - 10 esclusioni per contaminazione accanto al loro record di valutazione, per confermare che sono la stessa istanza;
   - 10 domande per ciascuna regola nuova di v4: `intersections` nel dominio disegnato, `shape_type` e `label_type` con
     le opzioni.
3. **Rubrica,** per ogni QA o coppia:
   - **grave** (errore):
     - il target non è la risposta visibile;
     - la domanda non si risponde dall'immagine, perché un elemento è nascosto, tagliato, coperto o illeggibile alla
       risoluzione data;
     - nessuna opzione vera, o più di una;
     - nella coppia, la variante b non cambia la risposta, o cambia più del dettaglio dichiarato;
   - **minore** (non è un errore, si conta a parte): difetti che non cambiano la risposta né la lettura, come testi
     attaccati ma leggibili, un'etichetta un po' lontana ma univoca, la forma della domanda;
   - **incerto:** il revisore non decide. Si risolve con un'evidenza indipendente, altrimenti resta incerto e conta come
     errore nel limite.

   Per ogni caso si registrano anche il tipo di difetto e il tempo.
4. **Prerequisiti, da approvare (D5):**
   - il disegno e le numerosità sopra;
   - il limite ipergeometrico con i collaudi di S4 §8 (C8, `estimate.py`: implementato e collaudato il 2026-11-05, S4
     §9; non ancora applicato agli esiti);
   - uno script che estrae i campioni e prepara i fogli di revisione, in un job CPU di pochi minuti;
   - un foglio TSV per gli esiti.

   Costo previsto: 0 token LLM, circa 4-5 ore di revisione umana (circa 30 s per caso su circa 550 casi), qualche
   minuto di CPU. Un errore trovato non si corregge nel manifest v1: produce una versione nuova.

**Decontaminazione esterna di v4: inventario.**

| Elemento | Stato | Uso per v4 |
| --- | --- | --- |
| Benchmark visivi riservati del catalogo: MathVista, MathVerse, MathVision, We-Math, DynaMath_Sample (p6); Geoperception, VisOnlyQA Real e Synthetic (p8-f1clean) | scaricati, hash verificati (job 29555, 30022) | ingressi riutilizzabili |
| Indici pHash e testo: `data/decontam/p6/index.json` (job 29565, 20.963 elementi), `data/decontam/f1eval/index.json` (job 30034, 13.257) | presenti | ingressi riutilizzabili, dopo averne registrato sha256 e revisioni. Non hanno lo sha256 dei pixel, che serve a un legame confermato (S1 §4) |
| `scripts/decontam.py scan` | esiste | immagini sì (pHash entro 4 bit). Per il tipo `synth` non legge le domande (`texts_of` restituisce una lista vuota), quindi il testo non è coperto |
| `scripts/decontam_review.py` | esiste | filtro dei candidati: somiglianza dei pixel ≥ 0.90 con rapporto d'aspetto entro 1.2, oppure n-gram ≥ 0.8. Dà candidati, non legami confermati |
| Esiti su v3 (`data/decontam/f1eval/synth`: 6 candidati, 0 confermati) | — | non valgono per v4 |
| Lista obbligatoria `data/decontam/f1eval/synth/exclude_final.txt` (`data_lists.REQUIRED`) | vuota, chiavi di v3 | va ancora indicata nelle config della famiglia `synth`, ma non è un'evidenza su v4 |
| Benchmark testuali riservati (p1) | scaricati | rischio basso, perché le domande sono a modello su scene generate. Gli n-gram sulle domande richiedono di estendere `texts_of` |
| Pannelli riservati del progetto (D6) | decisione aperta | da fissare prima dell'ammissione definitiva (pipeline §5.8.1) |
| Indice globale (S1 §4, C6) | non implementato | non serve alla decontaminazione di v4, che è un confronto con i riservati |
| Liste nel manifest | `liste_importate` vuota | da importare in una versione nuova del manifest |

Passi proposti, non eseguiti:
1. decidere D6;
2. registrare sha256 e revisioni degli indici, oppure ricostruirli con `decontam.py index`;
3. estendere `texts_of` alle domande dei sintetici, con un test;
4. scansionare tutti i 60.900 record v4 in un job CPU (su v3 sono serviti 152 s);
5. rivedere i candidati: un legame è confermato solo con pixel identici dopo la normalizzazione; gli altri candidati
   vanno nei casi mirati dell'audit o in quarantena;
6. importare le evidenze in una versione nuova del manifest, con le esclusioni per gruppo.

Costo previsto: pochi minuti di CPU, nessun download, nessun token.

### 2.9 Decontaminazione esterna di v4 contro `pannello-v3` (job 30517-30518, codice `fe85a97`, 2026-11-02)

Esegue i passi 2-6 della §2.8 per la decontaminazione. Il pannello è `pannello-v3` (D6 resta da approvare nei
dettagli, ma i ruoli non cambiano l'esito: ogni elemento, in qualunque ruolo, protegge dal training).

**Strumenti**, riusati con aggiunte minime:
- `scripts/decontam.py scan --kind synth` confronta ogni record con tutte le 99.713 voci dell'indice, senza filtri
  per ruolo. I segnali sono tre:
  - **identità dei pixel:** lo stesso sha256 dei pixel normalizzati (`rgb-su-bianco/1`) di una voce dell'indice. È
    l'unico legame confermato;
  - **pHash entro 4 bit:** candidato;
  - **10-gram di `normalize_words`** fra le domande della scena (o della coppia) e i testi dell'indice, senza i 712
    n-gram di rito: candidato.

  Per ogni record la scansione scrive shard, posizione, sha256 dei byte del PNG, controlli eseguiti e corrispondenze,
  con elemento e posizione del pannello (`records.jsonl.gz`).
- `scripts/synth_decontam.py` fa la verifica di compatibilità e scrive il manifest derivato.
- `scripts/manifest_check.py` verifica anche le versioni derivate.

**Compatibilità verificata**: 23 controlli, superati in entrambi i job.
- **Indice:**
  - convenzione `rgb-su-bianco/1` dichiarata, schema `panel-index/2`, sha256 `02fb3cbf…`;
  - `verifica.json` con 92.658 voci su 92.658 ritrovate nelle sorgenti, e sha256 dei pixel per ogni immagine.
- **Pannello e perimetro:** il pannello è costruito dallo stesso indice. Ogni elemento ha voci nell'indice e ogni voce
  ha il suo elemento, compresi i ruoli fuori dalla metrica primaria.
- **Manifest v1:** letto con le sue verifiche; gli shard hanno gli sha256 registrati.
- **Scansione:** stesso indice e stessa convenzione, identità dei pixel attiva, solo shard del manifest.
- **Identificativi:**
  - ogni riga della scansione ritrova un solo record per chiave, shard, posizione e sha256 dei byte del PNG;
  - ogni corrispondenza ritrova il suo elemento del pannello;
  - nessun elenco di corrispondenze è tagliato.

| Ruolo nel pannello | Elementi confrontati |
| --- | --- |
| sviluppo | 30.189 |
| test ammesso alla metrica primaria | 35.701 |
| test inammissibile (gruppo irrisolto) | 1.368 |
| solo decontaminazione | 12.401 |
| diagnostica | 6.332 |

**Controllo limitato** (job 30517, 1 min 57 s):
- test mirati: `tests/test_panel.py`, `test_decontam.py`, `test_synth_decontam.py`, `test_manifest.py`;
- scansione delle 300 scene di test e verifica parziale.

Risultato: 23 controlli su 23 e un candidato pHash. Nessun problema operativo: indice in memoria e confronto in 10 s.

**Scansione completa** (job 30518): 60.900 record in 93 s, tutte le immagini lette.

| Esito per record | Record | Che cosa vuol dire |
| --- | --- | --- |
| corrispondenza confermata | 0 | nessuna immagine con i pixel identici a una del pannello |
| candidato | 264 | tutti per pHash: 755 corrispondenze, a distanza 0 (47), 2 (241) e 4 (467). Nessun candidato testuale |
| non verificabile o controllo incompleto | 0 | ogni immagine letta; identità dei pixel e pHash eseguiti su tutti |
| nessuna corrispondenza | 60.636 | nei controlli eseguiti. Non certifica la risposta né l'assenza di ogni contaminazione |

Testo: 844.309 domande, di cui 308.237 con almeno 10 parole confrontate; nessun 10-gram in comune. 520 record non
hanno domande confrontabili, quasi tutti varianti di coppie con la domanda breve: per loro decide l'immagine.

| Benchmark del pannello | Record con candidati | Corrispondenze | Ruoli degli elementi |
| --- | --- | --- | --- |
| DynaMath | 178 | 583 | test ammesso |
| UniMER-Test | 26 | 26 | sviluppo 8, test ammesso 18 (25 in `sce`, 1 in `spe`) |
| MV-MATH | 24 | 24 | test ammesso |
| MMStar | 20 | 20 | sviluppo 17, test inammissibile 3 |
| MathVision | 16 | 22 | test ammesso |
| MathVista | 15 | 19 | solo decontaminazione 14, sviluppo 5 |
| ChartQA | 10 | 24 | sviluppo 15, test ammesso 3, test inammissibile 6 |
| We-Math | 5 | 6 | sviluppo |
| Geoperception | 3 | 12 | sviluppo |
| MUIRBench | 3 | 10 | test ammesso |
| OlympiadBench | 2 | 6 | solo decontaminazione |
| VisOnlyQA Synthetic | 2 | 2 | test ammesso |
| DocVQA | 1 | 1 | sviluppo |

Un record può avere candidati in più benchmark. I 264 record si dividono così: train 191, dev 6, train già escluso per
contaminazione interna 61, test 1, coppie 5.
- **DynaMath** genera grafici di funzioni e figure con matplotlib, come i nostri. Un pHash vicino indica
  un'impaginazione simile, non lo stesso contenuto: lo dice solo la verifica visiva.
- **UniMER-Test.** Nel pannello i vicini pHash delle formule non contano, perché le immagini sono quasi bianche (job
  30039). Qui restano candidati, per prudenza.

**Stati: regola per gruppo.** Il gruppo del manifest (stessa scena, stato della scena o coppia) è l'unità:
- una corrispondenza confermata esclude il gruppo;
- un candidato o un controllo incompleto lo mette in quarantena;
- la gravità non diminuisce mai: nessun record già in quarantena o escluso per altri motivi torna ammesso;
- split e gruppi sono quelli della versione 1.

I gruppi con un candidato sono 199. Cambiano stato, tutti da ammesso a quarantena: train 274, dev 6, test 4, coppie 16
(cioè 8 coppie).

**Manifest** `data/manifests/synth-v4-candidato-2` (job 30518): sha256 di `manifest.json` `9a8104cd…`, dei record
`5863b06a…`.

| Insieme | Split e stato | Versione 1 | Versione 2 |
| --- | --- | --- | --- |
| train | train ammesso | 54.271 | 53.997 |
| train | train in quarantena | 67 | 341 (67 + 274) |
| train | dev ammesso | 2.870 | 2.864 (coord 747, function 623, polygon 748, shapes 746) |
| train | dev in quarantena | 0 | 6 |
| train | test escluso | 2.792 | 2.792 |
| test | test ammesso | 300 | 296, più 4 in quarantena |
| coppie | test ammesso | 600 | 584, più 16 in quarantena |

**Evidenze nel manifest.** Ogni record ha l'evidenza `decontaminazione_pannello`, con:
- esito proprio e del gruppo;
- controlli eseguiti: pixel, pHash, e testo con il numero di domande confrontabili;
- corrispondenze per tipo, e un'anteprima di al più 10 elementi del pannello con posizione, distanza, ruolo, gruppo e
  ammissione alla metrica primaria;
- il riferimento a `records.jsonl.gz` della scansione, con lo sha256. L'elenco completo delle corrispondenze sta solo
  lì: chi le esamina segue il riferimento (§2.11).

Chi cambia stato ha anche il motivo `decontaminazione_pannello`, con i membri del gruppo che lo causano. Il manifest
registra, con i loro sha256, indice, verifica, pannello, scansione, versione precedente e identità del codice.

**Verifiche** (job 30518, `outputs/manifest_check/synth-v4-candidato-2`): 37 controlli su 37.
- I 32 della versione 1, sul manifest nuovo. Fra questi i loader: un'epoca consegna esattamente 53.997 record di
  train, 2.864 di dev, 296 di test e 584 delle coppie.
- 5 controlli nuovi:
  - versione precedente letta, con lo sha256 registrato;
  - stessi record, gruppi, insiemi, split e posizioni;
  - nessuno stato meno grave;
  - una sola evidenza di decontaminazione per record;
  - un motivo per ogni cambio di stato.

  La regola degli split si ricalcola sulla versione 1, da cui gli split sono ereditati. Dal 2026-11-04 il validatore
  risale la catena fino alla versione che assegnò gli split, anche per le derivazioni successive (§2.11).

**Identificativi.** Ogni decisione si riconduce al record originale. Chiave, `record_id` ricalcolato da
`manifest.load`, shard, posizione e sha256 dei byte del PNG coincidono fra manifest e riga della scansione. Ogni
corrispondenza si riconduce all'elemento del pannello (`benchmark/item` e posizione), con ruolo e gruppo da
`elements.jsonl.gz`.

**Che cosa non è coperto:**
- nessun candidato è risolto. La verifica visiva dei 264, con il foglio della §2.10, decide quali riammettere in una
  versione nuova;
- il pHash non riconosce la stessa figura disegnata in un altro stile, né lo stesso contenuto con un'impaginazione
  diversa;
- i sintetici sono generati, quindi non hanno legami di provenienza con i benchmark;
- l'esposizione pregressa di SigLIP2 e Violetto resta;
- il testo copre solo le domande di almeno 10 parole, che sono a modello;
- il confronto riguarda solo i benchmark scaricati del pannello.

**Conseguenza per il pannello**, non applicata a `pannello-v3`. 12 record di sviluppo sintetico (dev 6, test 1, coppie
5) hanno candidati verso elementi del pannello, anche di test (§2.10). Per la regola del pannello un candidato fra
sviluppo e test è un'evidenza diretta sull'elemento di test (`docs/PANNELLO_VALUTAZIONE.md` §3.1). L'indice del
pannello non contiene i sintetici, quindi `pannello-v3` non le vede: vanno considerate in `pannello-v4`.

**Requisiti ancora mancanti per il loader di F0:**
1. la config deve indicare `data/manifests/synth-v4-candidato-2` con lo split; senza manifest gli shard v4 non si
   leggono;
2. `data_lists.REQUIRED` chiede ancora, per la famiglia `synth`, la lista v3 `data/decontam/f1eval/synth/exclude_final.txt`.
   È vuota, ha chiavi di v3 e non è un'evidenza su v4. Va indicata nella config finché il controllo non passa
   all'evidenza del manifest; il codice non è cambiato qui;
3. mancano `variant_id`, `uid`, contratto e hash dei token (C3, C5; D3);
4. mancano i contatori per compito e `too_long` per sorgente (C7);
5. il campo gold è vuoto: nessuna stima campionaria (S4) e nessun audit.

### 2.10 Golden answer di v4: inventario dei controlli e campione per la revisione (job 30522, 2026-11-02)

Preparazione dell'audit della §2.8 sul manifest `synth-v4-candidato-2`, con `scripts/synth_gold_sample.py` (codice
`fe85a97`, script poi committati in `87c4108`) e uscite in `outputs/synth_gold_review/campione-1`. Nessuna revisione è
avviata: niente agenti, nessun dato o spiegazione rigenerati.

**Superato per la revisione da `campione-2` (§2.11)**, con la stessa selezione. I fogli di `campione-1` restano come
erano, ma hanno tre difetti: i candidati della decontaminazione mostrano solo l'anteprima di 10 corrispondenze
dell'evidenza; il primo passaggio mostra sempre le opzioni, che il modello di norma non vede; le coppie mirate hanno la
soluzione della sola variante a.

**Quattro proprietà, tenute separate** (§1):
1. **correttezza matematica:** il target è vero per la scena descritta dalla specifica;
2. **coerenza fra annotazioni e immagine:** il PNG mostra ciò che dicono specifica e layout;
3. **rispondibilità con la sola immagine e la domanda:** l'informazione è visibile, leggibile e univoca;
4. **spiegazioni:** i record v4 hanno risposte brevi, senza spiegazioni. La proprietà non si applica.

**Inventario dei controlli disponibili** (`inventario.tsv`: 38 righe per famiglia, tipo di domanda e unità, con QA e
esiti del checker):

| Proprietà | Che cosa esiste, su ogni record | Indipendenza | Che cosa manca |
| --- | --- | --- | --- |
| correttezza | il generatore calcola il target dalla specifica. Il checker (`synth_check`) lo ricalcola dagli oggetti disegnati del layout, con predicati propri: su 60.900 record tutte le QA sono ok (job 30448-30449). Prove esatte solo per le 9 coppie polygon contraddittorie (job 30420) | parziale: stesso progetto; per function le formule vengono dalla specifica | una verifica dai pixel; la revisione su campione |
| coerenza con l'immagine | il layout porta lo sha256 del PNG dello stesso render (60.900 digest corrispondenti); dalla chiave si riottengono byte identici (job 30449) | il legame layout-PNG è verificato; il contenuto dei pixel non è letto e dipende dal renderer | lettura dei pixel o revisione |
| rispondibilità | invarianti del renderer sul layout: figure dentro gli assi, testi dentro l'immagine, non sovrapposti e alti almeno 10 px (proposta), margini SIDE, EDGE, TANGENT ed EXTREME, una sola opzione vera | indicatori sul layout, non sui pixel | leggibilità alla risoluzione del modello e vicinanze ambigue senza sovrapposizione: solo la revisione |
| spiegazioni | — | — | non applicabile |

**Stato del gold:** nessun record è `gold_verificato` né `target_controllato`. Il checker approva le risposte, ma è
dello stesso progetto del generatore e legge il layout. Inoltre S2 non è approvata e manca la stima campionaria (S4).

**Campione casuale** (`campione.tsv`, 380 unità), con la regola della §2.8 sugli stati del manifest v2:

| Strato | Unità | N | n | Probabilità d'inclusione | Limite con 0 errori |
| --- | --- | --- | --- | --- | --- |
| coord | QA costruite dal loader per le scene ammesse di train e dev | 45.056 | 75 | 0,17% | 3,9% |
| function | idem | 35.961 | 75 | 0,21% | 3,9% |
| polygon | idem | 44.941 | 75 | 0,17% | 3,9% |
| shapes | idem | 44.614 | 75 | 0,17% | 3,9% |
| test | tutte le QA delle scene ammesse | 4.019 | 50 | 1,24% | 5,8% |
| coppie | coppia con entrambe le varianti ammesse | 292 | 30 | 10,3% | 8,9% |

Il limite è il superiore esatto al 95%, ipergeometrico, ed è solo informativo:
- vale con zero errori osservati, contando incerti e casi non giudicabili come errori;
- resta condizionato alla sensibilità del revisore (S4 §4);
- C8 (`estimate.py`) non è implementato; qui il limite viene da `synth_gold_sample.upper_zero`, collaudato sui valori
  della §2.8. Dal 2026-11-05 il limite esatto è in C8 (S4 §9), che ritrova i valori della §2.8; questa tabella resta
  quella calcolata con `upper_zero`.

Una numerosità minore va fissata prima di guardare: è il prefisso nell'ordine del rango.

**Casi mirati** (`mirati.tsv`, 175 casi): servono a cercare errori, non a stimare un tasso.

| Categoria | Casi | Scelta |
| --- | --- | --- |
| scene in quarantena per testi sovrapposti | 67 | tutte |
| testi più bassi | 8 | altezza minima 11,1 px (soglia 10) |
| riquadri di testo più vicini | 8 | 0,9-1,0 px (violazione oltre 1) |
| intersezioni vicine al bordo del tratto | 8 | 0,402-0,455 (EDGE 0,4) |
| grafici vicini senza incrociarsi | 8 | 0,333-0,375 (il checker segnala sotto 0,25, il generatore non chiede sotto 0,3) |
| vertici estremi | 8 | stacco 0,150 (EXTREME 0,15) |
| coordinate lette contando le tacche | 8 | per rango |
| coppie per famiglia | 20 | 5 per famiglia, per rango, fuori dal campione |
| esclusioni per contaminazione | 10 | per rango, accanto al record di valutazione con lo stesso stato della scena |
| regole nuove di v4 | 30 | 10 per `intersections`, `shape_type` e `label_type` |

I margini si calcolano dal layout salvato, senza ridisegnare.

**Candidati della decontaminazione** (`candidati_decontaminazione.tsv`): tutti i 264 della §2.9. Per ognuno:
- il PNG v4 e le immagini del pannello citate nell'evidenza: 131 immagini distinte, ritrovate con i lettori
  dell'indice, senza errori;
- ruolo e ammissione alla metrica primaria degli elementi.

`scripts/decontam_review.py` (in `outputs/decontam_review/campione-1/synth-v4`) misura la somiglianza dei pixel a
64×64. Nessun candidato supera 0,80: 198 stanno sotto 0,50 e 66 fra 0,50 e 0,80. La misura serve solo a ordinare la
verifica e non risolve un candidato.

Sette record di sviluppo sintetico hanno candidati verso elementi di test del pannello. Sono materiale per
`pannello-v4` (§2.9):

| Record v4 | Insieme | Elementi di test del pannello |
| --- | --- | --- |
| `synth-0026407` | dev | DynaMath, problema 329 |
| `synth-test-00007` | test | DynaMath, problema 329 |
| `synth-pair-00050-b` | coppie | DynaMath, problema 96 |
| `synth-pair-00074-b` | coppie | DynaMath, problema 96 |
| `synth-0017751` | dev | DynaMath, problema 44 (varianti 2 e 8) |
| `synth-0011211` | dev | UniMER-Test `sce-0000116`; MathVision test 1274 |
| `synth-pair-00115-b` | coppie | UniMER-Test `sce-0000116`; MathVision test 1274 |

Gli altri cinque record di sviluppo con candidati toccano solo elementi di sviluppo o solo di decontaminazione.

**Fogli e moduli:**
- `immagini/`: 870 PNG v4 copiati byte per byte, con lo sha256 controllato, e 737 file del pannello; 17 MiB;
- `soluzioni.tsv`: target, esito del checker e stato della scena, a parte, per rispondere prima senza vederli;
- `esiti.tsv`: 555 righe, una per unità, con una colonna per ciascuna delle quattro proprietà, più gravità, tipo di
  difetto e tempo;
- `esiti_decontaminazione.tsv`: 264 righe;
- `report.json`: disegno, inventario, limiti e sha256 dei file.

La procedura resta quella della §2.8: risposta prima senza il target, poi il confronto; un disaccordo si risolve con
layout e specifica.

**Costo:** job 30522, 5 min 4 s, 8 CPU e 48G, 40 min di CPU allocate; nessun token. Il job 30521, uguale, si era
fermato in 1 s su un errore del test (etichette dei vertici uguali nel poligono di prova), prima di scrivere uscite.

**Restano da decidere e da fare:**
- D5: budget degli audit e numerosità;
- la revisione umana dei 380 + 175 casi e dei 264 candidati;
- C8, per il limite con collaudi: implementato e collaudato il 2026-11-05 (S4 §9), da applicare agli esiti;
- una versione nuova del manifest con gli esiti: riammissioni o esclusioni, mai correzioni della v2.

### 2.11 Correzione dei fogli e del validatore delle derivazioni (job 30555-30556, codice `6328a29`, 2026-11-04)

Intervento circoscritto dopo la revisione dei push fino a `cfe29f4`. Non ripete la scansione e non crea versioni del
manifest. Restano invariati scansione completa, `synth-v4-candidato-2`, `pannello-v3` e `campione-1`. I fogli nuovi
sono in `outputs/synth_gold_review/campione-2`.

**1. Corrispondenze complete nel foglio della decontaminazione.**
- **Difetto.** L'evidenza del manifest conserva un'anteprima di al più 10 corrispondenze (`MAX_EVIDENCE`), e
  `campione-1` usava solo quella per immagini e foglio.
- **Correzione.** Il foglio segue il riferimento dell'evidenza a `records.jsonl.gz` della scansione e ne verifica:
  - lo sha256, uguale a quello dell'evidenza e del manifest; `summary.json`, `matches.jsonl` ed `elements.jsonl.gz`
    hanno gli sha256 del manifest, e l'indice quello registrato;
  - per ogni record chiave, shard, posizione, sha256 del PNG, conteggi e anteprima, uguali all'evidenza;
  - per ogni corrispondenza la voce dell'indice (elemento, posizione dell'immagine, id).

  Un identificativo incoerente ferma il job prima di scrivere. Una riga o un dettaglio mancante, una voce assente o
  un'immagine non ritrovata rendono il record `incompleto`.
- **Fogli.** `candidati_decontaminazione.tsv` ha una riga per record. `corrispondenze_decontaminazione.tsv` ha una riga
  per corrispondenza, con record e `record_id`, elemento del pannello, posizione dell'immagine, voce dell'indice, ruolo,
  gruppo, `metrica_primaria`, presenza nell'anteprima del manifest e file dell'immagine.

**Riconciliazione** (5 controlli, tutti superati):

| Conteggio | Valore |
| --- | --- |
| record con corrispondenze nella scansione = candidati del manifest | 264 = 264 |
| corrispondenze: righe della scansione = `summary.json` = `matches.jsonl` = esportate | 755 |
| corrispondenze oltre l'anteprima, assenti in `campione-1` | 18, in 3 record |
| duplicati di evidenza (stessa coppia record-voce) | 0 |
| voci ed elementi del pannello distinti | 133 |
| immagini del pannello distinte per pixel (file scritti una volta in `immagini/pannello/`) | 102 |
| record con materiale incompleto | 0 |

I tre record oltre l'anteprima sono di train, già in quarantena: `synth-0013615` e `synth-0009643` hanno 18
corrispondenze con DynaMath, `synth-0035515` ne ha 12 con DynaMath, UniMER-Test e DocVQA. I conteggi della §2.9 venivano
già dalla scansione completa e non cambiano, compresi i sette elementi di test della §2.10. Le 133 voci contro le 131
voci di `campione-1` vengono da quelle 18 corrispondenze. Alcune voci hanno gli stessi pixel: le 133 voci corrispondono a
102 immagini distinte, e ciascuna si copia una volta.

**2. Prompt del primo passaggio** (procedura in §2.8). Per train e dev il foglio mostra il testo costruito dal loader
di training (`data.synth_qa_sample`), confrontato per ogni unità con `qa_messages(pick_questions)` dalle posizioni del
manifest. La QA campionata è la domanda `numero_nel_prompt`. Le opzioni sono in `opzioni.tsv` e i target in
`soluzioni.tsv`; nel primo passaggio si vedono solo se fanno parte della domanda. Il testo esatto, con sha256 e domande
contenute, sta in `prompts.jsonl`; nelle celle gli a capo sono `\n`. Unità statistica e QA selezionate non cambiano.

| Origine del prompt | Unità | Chi |
| --- | --- | --- |
| loader, QA nel prompt | 337 | 300 del campione, 30 delle regole nuove, 7 casi delle soglie |
| loader, scena intera | 83 | quarantena 67, testi e riquadri vicini alle soglie 16 |
| QA che il loader non sceglie, come domanda singola | 25 | casi delle soglie: il modello non le vede mai in training |
| test, domanda singola | 50 | campione del test |
| coppie, domanda singola | 50 | 30 del campione e 20 casi mirati |
| nessuno | 10 | esclusioni per contaminazione |

**Test e coppie.** Il protocollo verificato è quello del foglio, la domanda da sola col prefisso di `qa_messages`. Il
protocollo di valutazione di F0 non è fissato (D3). Il loader di training non costruisce il prompt delle coppie: il job
lo prova su `synth-pair-00140-a` (ValueError, nessuna domanda). Le opzioni stanno nel materiale separato, da
consultare dopo il primo giudizio.

`opzioni_nel_testo_della_domanda` è un indicatore testuale: vale sì quando tutte le opzioni compaiono come parole nella
domanda. Lo sono `shape_type` e `label_type`, e anche domande come "above or below"; non lo sono `count`, `coords`,
`exists`, `side_len` e `intersections`.

**3. Entrambe le soluzioni delle coppie.** `soluzioni.tsv` e `opzioni.tsv` hanno una riga per variante, con etichetta a
o b (dalla variante del JSON, uguale al suffisso della chiave), record e `record_id`. Vale per le 30 coppie del campione
e per le 20 mirate, che prima avevano solo la variante a.
- `soluzioni.tsv` ha 512 righe: 412 fuori dalle coppie, più 50 varianti a e 50 b.
- `esiti.tsv` ha 655 righe: per ogni coppia le righe a e b e una riga `coppia` per il contratto.

**4. Split delle derivazioni successive** (`scripts/manifest_check.py`, `verify_splits`).
- **Difetto.** La regola degli split si ricalcolava sui record del genitore immediato. Da una terza versione in poi, un
  gruppo di dev tutto in quarantena nel genitore ha zero scene ammesse, e `assign_splits` lo riassegnerebbe a train.
- **Correzione:** due verifiche separate.
  - **Rispetto al genitore immediato:** stessi record, gruppi, insiemi, split e posizioni; nessuno stato meno grave.
    Con la decontaminazione si controlla anche che:
    - evidenze e motivi del genitore siano conservati;
    - ci sia una sola evidenza nuova per record;
    - ogni cambio di stato abbia un motivo nuovo;
    - lo stato sia il più grave fra quello del genitore e quello dato dall'esito del gruppo, cioè nessuna riammissione;
    - ci sia un solo esito per gruppo.
  - **Rispetto all'assegnazione originale:** la catena `ereditato_da` si risale fino alla versione che assegnò gli
    split. Ogni anello è letto con `manifest.load` e ha lo sha256 di `manifest.json` registrato dal figlio, anche in
    `precedente`. Regola, identità e split sono uguali lungo la catena, nessun anello migliora uno stato, e la regola si
    ricalcola sui record dell'origine.
- **Motivi.** `synth_decontam.py` aggiunge il motivo nuovo a quelli del genitore invece di sostituirli. Su
  `candidato-2`, derivato da una versione senza quel motivo, il risultato non cambia.
- **Su `synth-v4-candidato-2`** (job 30556, in sola lettura, `outputs/manifest_check/synth-v4-candidato-2-campione-2`):
  44 controlli su 44, con la catena 2 → 1 (`9a8104cd…`, `975dfe42…`).

**Collaudi** (job 30555, suite CPU sul codice `6328a29`): 27 file e 1.086 controlli, nessun fallimento, 10 min 2 s.
- `tests/test_synth_gold.py`, +22 controlli:
  - prompt uguale a quello del loader su scene vere del generatore, con opzioni incorporate (`label_type` fra più
    domande) e senza (`count` da sola);
  - test, coppie e loader che rifiuta una coppia;
  - una coppia con risposte diverse (9 e 11), con righe a e b;
  - 12 corrispondenze, l'undicesima esportata oltre l'anteprima di 10; la riconciliazione non passa con le sole 10;
  - materiale incompleto segnalato, posizione o anteprima alterate che fermano il foglio.
- `tests/test_synth_decontam.py`, +12 controlli: due derivazioni consecutive scritte da `synth_decontam.py`.
  - Un gruppo di dev va tutto in quarantena nella versione 2 e resta in dev nella 3. Sul genitore immediato la regola
    lo riporterebbe in train, sull'origine no.
  - Si rifiutano un anello alterato, una riammissione e un'evidenza tolta.

**Confronto con `campione-1`** (`confronto.json`, il job esce con errore se la selezione cambia): nessuna differenza.

| Che cosa | Esito |
| --- | --- |
| Estrazione | stessi seme, etichetta `synth-v4-1`, manifest e popolazioni (N, n e limiti per strato) |
| Unità | 380 del campione e 175 mirate, uguali per unità, strato, rango, record, `record_id`, shard, indice, posizione, ordine nel loader, tipo, domanda e digest della QA |
| Candidati | 264, stessi ranghi ed esiti |
| Risposte | 462 unità con risposta uguali; 20 varianti b aggiunte |
| Opzioni | uguali, ma spostate in `opzioni.tsv` per 462 unità |
| Immagini v4 | 870, con byte identici |
| Anteprima | in `campione-1` era il prefisso dell'elenco completo; mancavano 18 corrispondenze in 3 record |

**Costo.** Job 30555: 10 min 2 s, 4 CPU. Job 30556: 4 min, 8 CPU e 48G. In questo job `manifest_check` ha preso 2 min
30 s e i fogli 1 min 9 s. In tutto circa 14 minuti di tempo reale e 72 minuti di CPU allocate; niente GPU, download,
token o agenti. Su disco `campione-2` occupa circa 11 MiB (870 PNG v4 e 102 immagini del pannello), il controllo del
manifest 24 KiB.

**Limiti:**
- nessuna revisione è fatta e nessun record è promosso a gold;
- il prompt di test e coppie è quello del foglio, non un protocollo di valutazione approvato;
- il primo passaggio mostra il testo dell'utente, non il system prompt fisso del template né il prefill `<think></think>`
  di F1, che il processor aggiunge;
- `opzioni_nel_testo_della_domanda` è un indicatore testuale, non una regola del generatore;
- la verifica della catena prova integrità e coerenza delle versioni, non la correttezza delle decisioni.

**Restano aperti:** revisione visiva del campione e dei 264 candidati (D5), requisiti del loader di F0 della §2.9, C8
(implementato il 2026-11-05, S4 §9, da applicare agli esiti).

**Nota del 2026-11-05 sul confronto.** Il confronto qui sopra va dal formato 1 di `campione-1` al formato attuale, il
caso che il codice di allora trattava correttamente, e resta valido per ciò che confrontava. Non certifica tutto: le 20
risposte b delle coppie mirate e le opzioni b di tutte le 50 coppie non c'erano nel formato 1, quindi sono state
aggiunte senza confronto. `confronto.json` conta le prime come `seconde_varianti_aggiunte` e non nomina le seconde. Lo
stesso codice falliva invece fra due fogli del formato attuale: aggregava le coppie casuali e prendeva la prima
variante nelle altre. È corretto in `2a2002a` (job 30616): confronto per (unita, variante) col formato del precedente
esplicito e le aggiunte del formato 1 elencate come non certificate. Non è stato rieseguito sui fogli reali.
I 743 gruppi esclusi dal pannello possono restare esclusi: risolverli tutti non è un prerequisito per avanzare
(`docs/PANNELLO_VALUTAZIONE.md` §7, punto 2).

### 2.12 Pacchetto offline per la revisione umana (job 30557-30566, 2026-11-04)

L'utente fa la revisione sul proprio portatile: scarica il pacchetto da GitHub, compila gli esiti e li ricarica in
`risultati/`. Il pacchetto è `revisioni/synth-v4-campione-2/`, tracciato in git; le istruzioni sono nel suo
`LEGGIMI.md`. Non è una piattaforma di audit: è una pagina statica senza framework, backend o servizi esterni. Si apre
`index.html` con un doppio clic da `file://`; i dati sono file `.js` caricati con `<script>`, senza `fetch`.

**Contenuto**, ricavato da `campione-2` senza rigenerare nulla e senza `compare_previous` (`scripts/review_package.py`,
codice `5a4b4bb`):
- le stesse 380 unità casuali e 175 casi mirati, nello stesso ordine, e i 264 candidati con tutte le 755
  corrispondenze. Restano identici rango, identificativi, prompt esatto, numero della domanda campionata, risposte
  attese e immagini;
- **primo passaggio** (`dati/pacchetto.js`): unità, prompt e immagini, senza target, opzioni ausiliarie, esiti del
  checker, motivi o misure. La verifica lo controlla sulle chiavi;
- **secondo passaggio** (`dati/soluzioni.js`, `dati/evidenze.js`):
  - risposte, opzioni dal JSON del record, esito del checker e stato della scena, per variante;
  - per i 607 record delle unità, il testo esatto del JSON dello shard (specifica, QA e layout), con lo sha256 del
    manifest, e la riga del checker. Si leggono solo quelli dei record selezionati, dagli artefatti esistenti;
- 870 PNG v4 e 102 immagini del pannello copiati byte per byte, i moduli originali di `campione-2` in `moduli/`,
  `pacchetto.json` (impronta `18a8926f…`, sha256 di `report.json` di `campione-2`, del manifest, del file del checker e
  degli shard) e `SHA256SUMS`, che si controlla sul Mac con `shasum -a 256 -c`.

Su disco: `dati/` 6,1 MB, immagini 9,8 MB, moduli 1,7 MB; circa 18 MB in git.

**Interfaccia** (`index.html`, `app.js`, `logica.js`, `stile.css`):
- **immagini:** al 100% dei pixel, ingrandibili a 200-400% con pixel netti, adattabili e a schermo intero;
- **prompt:** con gli a capo veri, e la domanda campionata evidenziata col suo numero;
- **navigazione:** avanti e indietro, prossimo da completare, filtro per stato; avanzamento distinto per campione
  casuale, casi mirati e decontaminazione, per strato o categoria;
- **primo giudizio:** risposta libera o "non riesco a rispondere", giudizio iniziale di leggibilità e note. La
  soluzione si mostra solo dopo averlo salvato e senza modifiche pendenti; l'ora resta registrata e da quel momento il
  primo giudizio non si modifica più, anche dopo una ripresa. Un ripensamento va nel campo "risposta dopo la
  soluzione", segnato come non indipendente;
- **giudizio finale:** correttezza, coerenza con le annotazioni, rispondibilità, gravità, tipo di difetto e note, per
  QA o variante, più il contratto della coppia. I valori distinguono incerto, non giudicabile, non verificabile, non
  applicabile e non ancora compilato; nessun campo vuoto vale come approvazione;
- **coppie:** varianti a e b affiancate, con risposte e giudizi distinti;
- **decontaminazione:**
  - il record accanto a tutte le immagini candidate, con gli identificativi delle corrispondenze;
  - la somiglianza dei pixel solo come orientamento;
  - un esame per corrispondenza;
  - "sì" solo con una corrispondenza decisiva, "no" solo con tutte esaminate come contenuto diverso e materiale
    completo;
  - un cambio successivo annulla il "no". Nessun esito cambia ammissioni, quarantene o gruppi;
- **salvataggio:**
  - nel browser come comodità; la copia che conta è il file di avanzamento JSON, con primo giudizio, ora della
    soluzione, giudizi, esami, tempi ed eventi;
  - l'importazione controlla schema, pacchetto, versione, impronta, identificativi, record e valori, e non
    sovrascrive in silenzio i casi in conflitto;
  - i TSV hanno le colonne e le righe dei moduli originali, con gli a capo come `\n`.

**Verifiche** (job CPU via `slurm/review_package.sbatch`):

| Job | Che cosa | Esito |
| --- | --- | --- |
| 30557 | costruzione e verifica statica | 25 s; 27 controlli su 27 |
| 30558 | verifica ripetuta, più l'esecuzione della logica con `jjs` | verifica 27 su 27; `jjs` (Nashorn, JDK 8) non esiste su `lagrange0`, c'è solo sul frontend, dove non si eseguono programmi: passo tolto |
| 30559 | verifica con il controllo lessicale aggiunto ai file JS | 28 su 28, in 9 s |

La verifica statica controlla:
- conteggi, ordine e campi contro `campione-2`;
- prompt, soluzioni e opzioni contro `prompts.jsonl`, `soluzioni.tsv` e `opzioni.tsv`;
- motivi solo nel secondo passaggio, e le coppie con a, b e `coppia`;
- tutte le corrispondenze;
- sha256 dei JSON dei record, delle immagini, dei moduli e di `SHA256SUMS`;
- riferimenti degli script e nessuna richiesta di rete;
- commenti, stringhe, regex e parentesi bilanciate nei file JS.

Fino a `a9252b5` il JavaScript non era stato eseguito: si credeva che nei job non ci fosse un motore JavaScript. Per
questo il pacchetto contiene `verifica.html` (`test.js`), un autotest da aprire sul Mac prima di iniziare. Gira sui dati
veri e prova:
- primo giudizio separato dalla soluzione;
- coppie e casi sulla scena intera;
- esportazione e reimportazione con caratteri speciali e note su più righe;
- importazioni rifiutate e conflitti;
- regole della decontaminazione;
- TSV;
- costruzione delle pagine;
- caricamento di tutte le 972 immagini.

**Correzioni della protezione del primo giudizio** (revisione dell'utente fino ad `a9252b5`; job 30562-30566). Campione,
ordine, identificativi, prompt, target, immagini, dati del pacchetto e impronta non cambiano; cambiano interfaccia,
autotest e formato del file di avanzamento.
1. **Autotest.** La `verifica.html` di `a9252b5` stampava il target di un'unità nel controllo del DOM. Ora nomi e
   dettagli contengono solo identificativi e conteggi. Un controllo finale cerca nell'output le risposte attese, le
   opzioni, i motivi e le misure di almeno 4 caratteri assenti dal testo visibile dell'unità (184 stringhe). Se ne
   trova, nasconde la riga e fallisce; una prova interna verifica che riconosca un target vero. Sul cluster lo stesso
   controllo è ripetuto in Python sulle righe prima di scrivere il rapporto.
2. **Unità collegate.** Il materiale di un record (tutte le QA con le risposte, spec, layout, checker) rivela anche le
   risposte delle altre unità dello stesso record. Lo stesso vale per i record con la stessa scena: specifica senza
   chiave, versione e stile, come `synth_candidate.scene_state`. La scena è calcolata in `logica.js` dal JSON dei
   record, e la partizione coincide con lo stato del manifest. Nel pacchetto:
   - 7 record compaiono in più unità (15 unità), per esempio `synth-test-00038` in `C-test-008` e `C-test-012`;
   - 13 scene ricorrono in 28 record;
   - le unità collegate sono 23, in componenti di al più 4.

   La soluzione si mostra solo quando le unità collegate hanno un primo giudizio salvato, senza modifiche in sospeso.
   La pagina le elenca con un pulsante per raggiungerle, senza dire nel primo passaggio se il legame è record o scena,
   perché nei casi «stessa istanza?» lo anticiperebbe. Alla rivelazione:
   - i loro primi giudizi restano fissati (`prima_fissata`: quando, tramite quale unità, quale record, che legame);
   - ogni record esposto entra nel registro `esposizioni` (record, quando, tramite, origine).

   Un'unità esposta prima di avere un primo giudizio salvato è segnata `esposta_senza_primo_giudizio`: il suo primo
   giudizio resta fissato al salvataggio. Nel TSV vale `non_indipendente`, con la risposta nelle note. Coppie e casi
   mirati con più immagini espongono tutti i loro record. Nessun giudizio si compila da solo.
3. **Importazione, sostituzione e ripresa.** `Logica.unisci` sostituisce l'assegnazione diretta dello stato:
   - le esposizioni si uniscono sempre;
   - un primo giudizio fissato vince su una versione non fissata; fra due fissati diversi vale quello fissato prima;
   - la soluzione mostrata tiene la data più antica;
   - il lavoro in conflitto resta quello del browser finché il revisore non sceglie browser, file o «sostituisci tutto»
     (che tiene comunque la conoscenza);
   - ogni versione scartata va in `archivio`, con origine e motivo; una salvata dopo l'esposizione è marcata non
     indipendente.

   La stessa unione vale per l'importazione e per il salvataggio nel browser, che prima di scrivere rilegge lo stato
   salvato da un'altra scheda. Prima di un'importazione si scarica una copia del browser. Uno stato del browser
   illeggibile si copia in una chiave di riserva invece di essere sovrascritto.
4. **Formato.** Il file di avanzamento passa a `synth-v4-revisione-avanzamento/2`. Il formato 1 si legge con una
   migrazione esplicita, registrata in `migrazioni` e annunciata: le soluzioni mostrate diventano esposizioni, e le
   unità collegate sono fissate (salvate prima) o non indipendenti. Nel browser resta una copia del formato 1.
5. **Indipendenza delle unità già fissate** (revisione dell'utente fino a `9ea9af1`; job 30568-30571). In `9ea9af1`,
   `applicaEsposizioni` dava `prima_fissata` alle unità con la soluzione mostrata e poi le saltava. Si perdeva così
   un'esposizione precedente. Esempio di un salvataggio del formato 1, con A e B sullo stesso record:
   - la soluzione di A è mostrata alle 10;
   - B è salvata alle 11 e mostrata alle 12;
   - B risultava fissata alle 12 e indipendente.

   Lo stesso valeva quando un'unione portava un'esposizione più antica per un'unità già fissata, perché `unisci`
   copiava l'annotazione dell'altra versione senza riclassificare.

   Ora blocco e indipendenza sono separati:
   - l'indipendenza confronta, per ogni unità anche già fissata, l'ora del salvataggio con la prima esposizione che la
     riguarda, nel registro o già annotata;
   - il contenuto e il blocco non cambiano. La provenienza del blocco passa all'evento più antico: l'esposizione se il
     giudizio è indipendente, altrimenti il suo salvataggio;
   - ogni cambio è un evento, e `unisci` unisce le cronologie delle due versioni;
   - un'annotazione smentita dalle date torna indipendente.

   Gli stati del formato 2 scritti da `9ea9af1` si riconciliano all'importazione e alla ripresa: la riconciliazione
   finisce in `migrazioni` e nel messaggio, e nel browser resta una copia dello stato precedente.

   Test nuovi, eseguiti con Duktape, tutti superati:
   - formato 1 con entrambe le soluzioni mostrate;
   - caso opposto, con B salvata alle 9:30: indipendente;
   - annotazione smentita dalle date;
   - esposizione più antica emersa dall'unione, nei cinque modi;
   - formato 2 classificato male;
   - coerenza fra stato, riepilogo, TSV e blocco;
   - ripresa nel browser e pagina dell'unità.

   Senza DOM 112 superati e 0 falliti, su DOM simulato 136 e 0 (job 30568, 30569). Prova contraria: lo stesso autotest
   con il `logica.js` di `9ea9af1` (job 30570, `slurm/review_package.sbatch … regressione 9ea9af1`, rapporto in
   `outputs/review_package/regressione-9ea9af1/`) fallisce esattamente nei controlli nuovi, 11 senza DOM e 14 su DOM
   simulato, e supera tutti gli altri.

**Esecuzione del JavaScript.** Una sonda di sola lettura su `lagrange0` (`slurm/js_engine_probe.sbatch`, job 30562) ha
trovato tre candidati:
- `libduktape.so.207`: Duktape 2.7, motore ES5.1, libreria di sistema installata con il sistema operativo;
- Node in `~/.nvm`, fuori dalla cartella del progetto: non usato;
- Firefox snap, che scriverebbe in `~/snap/`: non usato.

`review_package.py esegui-js` carica Duktape con ctypes e valuta i dati, `logica.js`, `app.js` e `test.js` con l'eval
globale, come gli `<script>` della pagina. Lo fa due volte:
- **senza DOM:** si eseguono logica e regressioni;
- **su un DOM simulato** (`scripts/review_package_dom.js`): si costruiscono le pagine, si premono i pulsanti, si
  usa una memoria finta al posto di localStorage.

Confronta poi i collegamenti calcolati da `logica.js` con quelli calcolati in Python e scrive
`revisioni/synth-v4-campione-2/verifica_js.json`. Esiti:
- job 30563, sul codice pubblicato `a9252b5`: 58 controlli superati su 59, saltata la parte DOM. Era la prima
  esecuzione di quel codice;
- job 30564, sulle correzioni: un fallimento, in un controllo del test troppo largo;
- job 30565: senza DOM 97 superati, 0 falliti e 1 saltato; su DOM simulato 118, 0 e 1 (le immagini); collegamenti
  uguali a Python; nessuna soluzione nell'output; verifica statica 29 su 29.

Sono tre livelli distinti:
- controllo statico;
- esecuzione della logica e dell'interfaccia con Duktape;
- prova in un browser, **non eseguita**. Disegno, eventi veri, localStorage, scaricamenti e immagini si provano solo
  con `verifica.html` sul Mac.

**Restituzione:** il file di avanzamento JSON (obbligatorio) e i due TSV (facoltativi) vanno in
`revisioni/synth-v4-campione-2/risultati/`; il resto del pacchetto non si modifica. Sul cluster i file si verificheranno
contro pacchetto, versione e impronta prima di qualsiasi uso; nessun esito diventa automaticamente un cambio del
manifest o un gold.

### 2.13 Esiti della revisione umana di `campione-2`: integrazione verificata e proposta (job 30912-30915, 2026-11-12)

**Consegna e integrazione sono due fatti distinti.**
- **Consegna.** L'utente ha caricato il 2026-11-11 i tre file in `revisioni/synth-v4-campione-2/risultati/` (commit
  `201839b` per i TSV, `777e30c` per il JSON).
- **Integrazione verificata.** È il job 30915, sul codice `dd50a4f`, col riesame tecnico di `d9ce960`.
- **I file caricati** restano byte per byte com'erano: il launcher controlla che siano uguali ai blob di `777e30c` e
  che il pacchetto non abbia modifiche locali, prima e dopo il job.
- **Nessun esito cambia** manifest, gruppi, split, target, configurazioni dei reader o ruoli del pannello. La revisione
  MAVIS (`revisioni/mavis-c9-campione-1`) è un'altra cosa e non è toccata.

**Strumento.** `scripts/review_results.py`, lanciato da `slurm/review_results.sbatch`, col test
`tests/test_review_results.py`. È un adattatore minimo sugli strumenti esistenti:
- da `review_package.py` prende la lettura del pacchetto, la verifica statica (ripetuta, col rapporto in `outputs/`), i
  collegamenti per record e per scena e Duktape;
- da `logica.js` del pacchetto prende `controllaImport`, `tsvEsiti` e `tsvDecontaminazione`;
- non ricostruisce il pacchetto e non scrive in `revisioni/`.

**Rapporti.**
- Per git: `docs/revisioni/synth-v4-campione-2-integrazione-2.json` (controlli, conteggi, materiale del riesame,
  riepilogo della proposta) e `docs/revisioni/synth-v4-campione-2-integrazione-2-proposta.tsv` (proposta per record dei
  gruppi toccati).
- Completi in `outputs/review_results/integrazione-2/`: `report.json`, `esiti_per_unita.tsv` (prima risposta, target e
  giudizi per unità e variante) e `proposta_ammissioni.tsv` (tutti i 3.159 record non ammessi).

**Ingressi:**

| File | sha256 | Byte |
| --- | --- | --- |
| `synth-v4-campione-2_avanzamento_2026-11-11T20-12-26.json` | `4a88d60cb1111ea6a7b4027f58f70865306a4245e4da3398d7c14036d8035b7f` | 1.533.650 |
| `esiti_2026-11-11T20-12-29.tsv` | `57ec0275bf2931c8021cd50b25a6473c4728d06b800c2e68fe5ff229679f8cb5` | 72.421 |
| `esiti_decontaminazione_2026-11-11T20-12-30.tsv` | `a694cfad5114f317e38598d3f5f21f71bf00e9d3cb9473620d2fe047e606438b` | 38.045 |
| riesame tecnico (`docs/revisioni/synth-v4-campione-2-riesame-incerti.tsv`) | `cfb9651faaa883c4cbca149ba2b155da829c4c59080786c7d52c711bb7d83217` | 10.634 |

Pacchetto `synth-v4-campione-2` versione 1, impronta `18a8926f…`. Manifest `synth-v4-candidato-2`, con `manifest.json`
`9a8104cd…` e record `5863b06a…`.

**Verifiche** (31 controlli su 31; test mirato 37 su 37), ricalcolate dai dati registrati senza fidarsi del riepilogo
esportato né dei flag del file:
- **identità:**
  - sha256 attesi dei tre file;
  - formato `synth-v4-revisione-avanzamento/2`, con pacchetto, versione e impronta del pacchetto;
  - verifica statica del pacchetto ripetuta e superata;
  - `logica.js` uguale a quello eseguito dal job 30571;
- **collegamento:**
  - manifest letto e verificato da `manifest.load`, con gli sha256 registrati nel pacchetto;
  - le 555 unità (380 + 175), con varianti e record del pacchetto, e i loro `record_id` nel manifest;
  - i 264 candidati con le 755 corrispondenze, nessuno mancante, ripetuto o in più;
  - i candidati del pacchetto sono esattamente i record con esito `candidato` nel manifest;
- **JSON e TSV:**
  - `logica.js`, eseguito con Duktape, importa il file senza errori, migrazioni, riclassificazioni né cambi di stato;
  - i due TSV che rigenera sono uguali byte per byte a quelli caricati;
  - in Python ogni riga dei TSV (655 e 264) si ricostruisce dai campi del JSON;
- **cronologia:**
  - il registro delle 615 esposizioni coincide coi record esposti dalle soluzioni mostrate;
  - ogni primo giudizio è salvato prima della propria soluzione, e ogni giudizio finale è registrato dopo;
  - la prima esposizione pertinente, cioè dello stesso record o di un record con la stessa scena, è ricalcolata per ogni
    unità: 555 primi giudizi indipendenti su 555. Delle 23 unità collegate, 13 avevano la prima esposizione pertinente
    tramite un'altra unità, comunque dopo il loro salvataggio;
  - le annotazioni del file (esposizione senza primo giudizio, provenienza del blocco) coincidono col ricalcolo;
  - nessuna bozza diversa dal primo giudizio fissato, archivio vuoto;
- **completezza:**
  - nessun `non_compilato`;
  - `non_applicabile` compare solo nella correttezza dei casi sulla scena intera (quarantena, soglie dei testi e dei
    riquadri, esclusioni) e nella rispondibilità dei 10 casi di esclusione per contaminazione;
  - esiti della decontaminazione coerenti con le regole: "no" solo con tutte le corrispondenze diverse e il materiale
    completo.

**Conteggi della revisione statica: confermati, nessuna differenza.**

| Conteggio | Valore |
| --- | --- |
| Unità del campione casuale | 380 |
| Casi mirati | 175 |
| Righe di `esiti.tsv` (varianti a e b e contratto per le 50 coppie) | 655 |
| Candidati della decontaminazione / corrispondenze | 264 / 755 |
| Esito dei candidati: diverso (no) / incerto / stesso contenuto (sì) | 249 / 15 / 0 |
| Corrispondenze: contenuto diverso / incerte | 727 / 28 |

**Lettura, per sezione.** Le 655 righe non sono 655 osservazioni indipendenti: le coppie hanno tre righe, le 23 unità
collegate condividono record o scene, e le corrispondenze di un record o di un gruppo sono legate. "Nessun difetto
segnalato" descrive le unità guardate, non il dataset.

- **Campione casuale** (380 unità; per strato coord, function, polygon, shapes 75 QA ciascuno, test 50 QA, coppie 30):
  - tutti i giudizi finali sono "sì" (correttezza, coerenza con le annotazioni, rispondibilità), con gravità "nessun
    difetto", e i 30 contratti delle coppie sono "sì";
  - leggibilità iniziale "sì" ovunque;
  - prime risposte, scritte prima di vedere il target: delle 350 varianti non di coppia, 345 uguali al target, una
    equivalente nella scrittura e 4 classificate diverse dal confronto, ognuna con un'autocorrezione del revisore
    (sotto); le 60 varianti di coppia sono uguali.
- **Casi mirati** (175, per trovare errori, non per stimare un tasso):
  - giudizi finali tutti positivi;
  - nei 67 casi di quarantena per testi sovrapposti: leggibilità "sì", coerenza e rispondibilità "sì", nessun difetto,
    correttezza `non_applicabile` (caso sulla scena intera);
  - nelle 20 coppie per famiglia, varianti e contratti "sì";
  - nei 10 casi di esclusione per contaminazione, coerenza "sì" in tutti e la nota "stesso stato della scena" in uno
    (`M-esclusioni_contaminazione-009`): nessuno smentisce l'esclusione, che resta;
  - una prima risposta classificata diversa, con un'autocorrezione (`M-coordinate_dalle_tacche-001`).
- **Coppie e contratti:** 50 coppie (30 casuali, 20 mirate), 100 varianti con giudizi positivi e 50 contratti "sì". La
  coppia è giudicata per intero; le sue righe non sono tre prove.
- **Decontaminazione:** 264 record e 755 corrispondenze.
  - Il "no" vuol dire che le corrispondenze candidate sono state esaminate come contenuto diverso. Non certifica
    l'assenza di ogni contaminazione: il pHash non vede la stessa figura in un altro stile.
  - I 15 incerti hanno un riesame tecnico separato (sotto).

**Prime risposte e autocorrezioni.** Sei prime risposte differiscono come stringa dal target:
- `C-coord-022` scrive `two` per `2`: è un'equivalenza, ora riconosciuta dal confronto (numeri in lettere);
- le altre cinque sono **prime risposte classificate diverse dal confronto**, ognuna con un'autocorrezione: una nota
  del revisore che dichiara sbagliata la propria risposta e giusto il target, e la risposta corretta nel campo
  successivo alla soluzione:
  - `C-coord-049`: left contro right;
  - `C-polygon-053`: 2 contro 3;
  - `C-shapes-009`: 0 contro no, alla domanda "Is there a circle in the figure?";
  - `C-shapes-053`: 2 contro no;
  - `M-coordinate_dalle_tacche-001`: (5, -5) contro (-5, 5).

Non vanno lette come cinque errori semantici dimostrati.
- **`C-shapes-009`:** "0" (nessun cerchio) può essere equivalente a "no". Da integrazione-3 (§2.14) il confronto la
  classifica come possibile equivalenza semantica, distinta da "uguale" e da "diversa".
- **Note:** restano quelle originali, con le loro parole.
- **Sensibilità del revisore:** il conteggio non la misura.
- **Golden answer:** nessuna è smentita, e il giudizio finale conferma il target.
- **Conservazione:** prima risposta e giudizio finale restano entrambi, in `esiti_per_unita.tsv` e nel file caricato.
- **Indipendenza:** un disaccordo iniziale risolto dal revisore stesso non è una verifica indipendente del target.

**C8 non si applica in questo lotto.** La corrispondenza fra i valori del pacchetto e gli esiti di C8 era una proposta
(S4 §9) e non è stata approvata prima della revisione: qui non diventa approvata retroattivamente. I conteggi sono
descrittivi. Per una stima mancano:
- la corrispondenza approvata dall'utente, sapendo che gli esiti sono già noti;
- le affermazioni per proprietà e strato;
- D5;
- una misura della sensibilità del revisore. Senza, ogni conclusione resta condizionata.

Nessuna soglia nuova, nessuna certificazione, nessun record promosso a gold.

**Riesame tecnico dei 15 incerti** (dettagli in `docs/revisioni/synth-v4-campione-2-riesame-incerti.md`; giudizio umano
invariato).
- **Le 14 scene y = x + 1:** 13 record di train e la coppia riservata `synth-pair-00074-b`, nello stesso gruppo della
  scena. Le corrispondenze sono DynaMath 96 variante 8 e MV-MATH 1008, immagine 2, entrambe di test. Conclusione:
  contenuto diverso.
  - In comune c'è la formula elementare, generata in modo indipendente.
  - Le domande dei benchmark (distanza fra le intercette, 1.4142; quale grafico non è una funzione) non sono quelle dei
    record, e le risposte dei record non le implicano.
- **D-188 (`synth-0016025`):** triangolo isoscele con l'angolo annotato di 74° al vertice, orientato come la figura di
  Geoperception 7600-7604 e 102362 (ruolo sviluppo). La domanda del record sull'angolo ha la stessa risposta della 7600.
  Conclusione: irrisolto; resta in quarantena.

**Proposta di aggiornamento delle ammissioni: non applicata.** Per record e gruppo, nella `-proposta.tsv`. Un motivo si
toglie solo con un'evidenza della revisione; split, gruppi e target non cambiano; l'ereditarietà dal gruppo si verifica
su tutti i membri.

| Motivo del manifest | Esito per i record che lo portano |
| --- | --- |
| `decontaminazione_pannello` (680 record in 199 gruppi con candidati) | **risolto dalla revisione** in 197 gruppi (570 record), dove tutti i candidati sono "no". **Irrisolto** in 2 gruppi (110 record): il gruppo y = x + 1, con 14 incerti, e `synth-0016025`. Il primo diventa risolvibile col riesame tecnico, da approvare |
| `incerto` (quarantena visiva, 67 record; tutti i 77 dettagli sono `testi_sovrapposti`) | **riesaminabile dai giudizi** per tutti i 67. La regola, fissata prima di applicarla, vuole leggibilità iniziale "sì" con primo giudizio indipendente, coerenza "sì", rispondibilità "sì" e nessun difetto. Vale per questi 67 casi, non come allentamento del filtro dei testi sovrapposti per dati nuovi |
| `contaminazione` (2.792 esclusi: stato della scena uguale a un record di valutazione) | **valido**: blocco indipendente dalla revisione, che resta |

| Categoria proposta | Record | Composizione |
| --- | --- | --- |
| Riammissione proposta | 362 | train 340 (273 per la decontaminazione, 67 per la quarantena visiva), dev 6, test 4, coppie 12. Per motivo tolto: 201 candidati "no", 94 membri dei loro gruppi senza candidato proprio, 67 quarantene visive (nel riepilogo `incerto (membro del gruppo)`, cioè non candidati) |
| Riammissione dopo il riesame tecnico (da approvare) | 4 | `synth-pair-00074-a` e `-b`, `synth-pair-00198-a` e `-b`: coppie riservate del gruppo y = x + 1, il cui unico motivo è la decontaminazione |
| Irrisolto | 1 | `synth-0016025`, in quarantena |
| Resta bloccato | 2.792 | esclusi per contaminazione interna, compresi 48 candidati "no" e 13 incerti y = x + 1 |

**I 249 "no" non sono 249 riammissioni.** 201 candidati "no" stanno in gruppi tutti risolti e si possono riammettere. Gli
altri 48 restano esclusi per contaminazione interna. Fra le riammissioni ci sono anche 94 record che non sono candidati:
erano in quarantena solo perché un altro membro del gruppo lo era.

**Conseguenze per il pannello.** Nessun ruolo di `pannello-v3` cambia.
- Dei sette record sintetici di sviluppo o valutazione con candidati verso elementi di test del pannello (§2.10), sei
  sono "no": nessuna evidenza diretta per `pannello-v4` da quei legami.
  - `synth-0026407` e `synth-test-00007`: DynaMath 329;
  - `synth-pair-00050-b`: DynaMath 96;
  - `synth-0017751`: DynaMath 44;
  - `synth-0011211` e `synth-pair-00115-b`: MathVision 1274 e UniMER-Test `sce-0000116`.
- Resta `synth-pair-00074-b` verso DynaMath 96 variante 8, risolto solo se l'utente accetta il riesame tecnico.
- Gli altri cinque record di sviluppo con candidati toccano solo elementi di sviluppo o di sola decontaminazione, e
  sono tutti "no".

**Costo:** job 30912-30915, 22-23 s ciascuno, 2 CPU. In tutto 1 min 28 s di job su 20 minuti; RSS non registrata da
`sacct`. Nessuna GPU, generazione, download o agente.

**Limiti:**
- **Revisore:** uno solo, con la sensibilità non misurata. Delle 512 prime risposte confrontabili, 5 differiscono dal
  target anche dopo le equivalenze di scrittura, tutte con un'autocorrezione. Una (0 per no) da integrazione-3 è una
  possibile equivalenza semantica. Non è una misura degli errori del revisore, e la revisione guarda per lo più casi
  facili da leggere.
- **Riesame tecnico:** è un controllo di Claude Code, non una seconda revisione umana.
- **Decontaminazione:** risolve i candidati pHash, non i quasi-duplicati che il pHash non vede.
- **Uso degli esiti:** gli esiti descrivono le unità guardate; senza C8 e D5 non danno quote per la popolazione.

**Prossimo passo minimo**, su decisione dell'utente:
1. approvare in tutto o in parte la proposta. Si decidono a parte:
   - i 362 record con evidenza della revisione;
   - i 4 che dipendono dal riesame tecnico;
   - per `synth-0016025`, quarantena o esclusione;
2. con l'ok, una versione derivata `synth-v4-candidato-3` che toglie solo i motivi approvati, con le evidenze della
   revisione nel record, e la sua verifica lungo la catena (`manifest_check.py`). Il reader e la config di collaudo
   passano al manifest nuovo solo dopo la verifica;
3. se servono stime, approvare la corrispondenza verso C8 e le affermazioni di S4 §9 prima di calcolarle.

### 2.14 Scorciatoia testuale di function/exists, politica delle domande e `synth-v4-candidato-3` (job 30916-30923, 2026-11-12)

**Il problema.** In v3 e v4 `_function_spec` disegna sempre il grafico `f` e, in metà delle scene, `g`. `_function_qa`
sceglie l'etichetta della exists positiva fra le presenti e quella della negativa fra h, p, q.
- **Le golden answer sono vere,** ma il nome chiesto dà la risposta senza guardare l'immagine: f, g → yes; h, p, q → no.
  Anche g è sempre yes, perché si chiede solo quando c'è.
- **Il checker non lo vede:** `synth_check._exists` verifica la risposta sul disegno, non la distribuzione delle domande.
- **La revisione visiva non lo esclude:** ha giudicato risposte e leggibilità, non l'assenza di scorciatoie.
- **Che cosa non è:** non è un errore delle risposte e non invalida gli altri sintetici.

**Impatto misurato senza modelli** (`scripts/synth_qa_policy.py impatto`, job 30916;
`docs/inventari/synth-v4-scorciatoia-exists-lotto-1.json`). Ingresso: synth-v4-candidato-2, 60.900 JSON letti con lo
sha256 del manifest e pool di ogni record uguale ai digest registrati.

| function/exists | Pool: yes / no | Scelte dal loader: yes / no |
| --- | --- | --- |
| train, ammessi | 11.491 / 11.491 | 5.214 / 5.201 |
| dev | 623 / 623 | 292 / 280 |
| test, ammessi | 72 / 72 | 34 / 33 |
| test in quarantena; train in quarantena; train esclusi | 3 / 3; 94 / 94; 2.792 / 2.792 | non letti |
| coppie | 0 (la domanda della coppia non è mai exists) | 0 |

- **Etichetta per risposta** sul pool di 30.150 QA: f 11.332 e g 3.743 sempre yes; h 5.019, p 5.006 e q 5.050 sempre no.
  Nessuna etichetta ha entrambe le risposte.
- **Regola testuale** f, g → yes; h, p, q → no: 30.150 su 30.150 nel pool, 11.054 su 11.054 fra le domande scelte dal
  loader (f1c.pick_questions sui record ammessi, come data.synth_qa_sample).
- **Prompt toccati:** hanno almeno una function/exists 7.814 prompt function di train su 11.491, 434 di dev su 623 e 48
  di test su 72.
- **Misura storica:** `scripts/f1c_eval.py` (storico) valuta tutte le QA delle 300 scene di test senza manifest. Su
  4.038 QA, 150 sono function/exists (75 yes, 75 no), tutte risolte dalla regola testuale. Rapporti e benchmark storici
  restano come sono.
- **QA che restano** nei record function ammessi, togliendo le exists:
  - train 11.491 record: 22.982 tolte, 58.223 restano;
  - dev 623 record: 1.246 tolte, 3.162 restano;
  - test 72 record: 144 tolte, 315 restano.

  Nessun record resta senza domande: da 2 a 9 QA per scena. Le 140 coppie function tengono la loro domanda (opens o
  slope).
- **Effetto sulla scelta del loader:** il pool più corto cambia numero e scelta delle altre domande in 10.230 record
  function di train su 11.491, 553 di dev su 623 e 62 di test su 72. Le altre famiglie e le coppie non cambiano. Nei
  prompt function, contando train, dev e test:

  | | exists | intercept | slope | opens | vertex_ud | vertex_lr | intersections |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | prima | 11.054 | 8.313 | 5.089 | 3.435 | 3.392 | 2.763 | 2.263 |
  | dopo | 0 | 11.059 | 6.872 | 4.613 | 4.477 | 3.587 | 3.073 |
- **Campione-2:** i 420 prompt storici del loader (unità `loader` e `loader_scena`) sono riprodotti dal codice corrente.
  - Delle 107 unità function, 19 hanno come domanda una function/exists: 18 in un prompt del loader, una domanda singola
    del test.
  - Col filtro 66 hanno un prompt diverso e 6 lo stesso; le altre sono domande singole.
  - Le 60 unità con più record (coppie e confronti di contaminazione, scritte `a|b`) non si confrontano: le domande
    delle coppie non sono exists, e i record esclusi restano esclusi.
  - Il giudizio umano vale per i prompt storici e non si attribuisce ai prompt nuovi.

**Le altre domande di presenza** (controllo deterministico circoscritto; misurate, non corrette in questo lotto).
- **Vocabolario:** coord, polygon e shapes exists e shapes has_type usano lo stesso vocabolario per sì e no, e nessuna
  etichetta determina la risposta. La regola per etichetta (la risposta più frequente di ogni etichetta) dà sul pool
  0,508 (coord), 0,557 (polygon), 0,511 (shapes), 0,559 (has_type); sulle scelte del loader 0,531, 0,560, 0,519, 0,559.
- **Sbilanciamenti, dimostrati e parziali:**
  - polygon/exists: le lettere agli estremi sono più spesso no (A: 853 no e 324 yes; W: 803 e 293), perché i vertici
    hanno lettere vicine;
  - has_type: rectangle è più spesso yes (3.544 contro 1.428), perché con un quadrato non si chiede del rettangolo.
- **Contesto del prompt, dimostrato per costruzione:** un'etichetta assente non è mai nominata da un'altra domanda.
  Quando un'altra domanda del prompt nomina l'etichetta chiesta, la risposta è sempre yes. Sulle scelte del loader:

  | Famiglia | exists nominate da un'altra domanda (tutte yes, nessun no) | Regola "nominata → yes, altrimenti no" |
  | --- | --- | --- |
  | coord | 1.385 su 3.704 | 3.124 (84%) |
  | polygon | 2.929 su 10.574 | 7.861 (74%) |
  | shapes | 2.008 su 10.929 | 7.109 (65%) |
- **Coppie sì/no obbligatorie:** le due exists di una scena sono una yes e una no; stanno nello stesso prompt 490 volte
  in coord, 3.726 in polygon e 3.920 in shapes. Le due has_type stanno insieme 3.556 volte.
- **Ipotesi, non scorciatoia senza immagine:** has_type e count_type dello stesso tipo stanno nello stesso prompt 1.031
  volte, 524 col conteggio prima. Lì la risposta precedente del target dice la presenza.
- La correzione (domande di presenza da sole, come in v5 per i function) va decisa a parte.

**Correzione dei dati v4: politica delle domande** (S1 §6; `manifest.TASKED_BY_POLICY`, `data.synth_qa_sample`).
- **Record:** compiti famiglia/tipo (`data.synth_task`) con stato e numero di domande. `function/exists` è escluso,
  motivo `scorciatoia_testuale`, in entrambe le polarità; tutti gli altri compiti sono ammessi.
- **Reader:** un manifest dei sintetici con `politica_domande` (`synth-domande/1`) rende il tipo synth_qa una sorgente
  per compito, su train, test e coppie: manifest_sha256 obbligatorio, liste della config rifiutate, decisione del record
  allegata al campione.
- **Loader:**
  - lega il pool (QA e domanda della coppia) al digest `qa_sha256` del manifest, e la famiglia a quella registrata;
  - toglie le domande dei compiti non ammessi prima di `pick_synth_questions`;
  - senza domande ammesse si ferma con NoAdmittedTask, senza ripiegare su quelle bloccate, e il reader salta il record
    col motivo `nessun_compito_ammesso`.
- **Senza politica** (candidato-1 e -2) la lettura resta per record, come prima: col codice nuovo il candidato-2 passa
  45 controlli su 45 (job 30922). Con la politica la lista storica dei sintetici di `data_lists.REQUIRED` non serve.
- **Valutazione:** la misura nuova passa dal manifest con politica; `scripts/f1c_eval.py` resta storico.

**Generatore v5** (`synth.py`, solo con `version=5`; il default resta 4, e v3 e v4 sono riprodotti da test_synth e
test_synth_policy). Nessun corpus rigenerato.
- **Nomi dei grafici:** da `FUNCTION_NAMES` (f, g, h, p, q), lo stesso vocabolario delle domande sui grafici assenti,
  estratti dopo la geometria con un generatore a parte.
- **exists:** una sola per scena, sì o no con probabilità 1/2, marcata `da_sola`. Il loader la mette in un esempio
  senza altre domande: entra solo quando la scelta è di una domanda.
- **Intersezioni:** la domanda nomina i due grafici disegnati, e ora il checker lo controlla.
- **Invarianti:** geometria, stile e altre domande uguali a v4 per la stessa chiave; coord, polygon e shapes uguali a
  v4.
- **Diagnostica,** fuori dai manifest (job 30916, `docs/inventari/synth-v5-diagnostica-lotto-1.json`):
  - su 4.000 chiavi non disegnate, la regola per etichetta dà 1,000 in v4 e 0,518 in v5. In v5 ogni etichetta ha sì e
    no: 1.954 yes e 2.046 no; con un grafico 993 e 1.055, con due 961 e 991;
  - dopo la scelta del loader, 133 esempi su 4.000 hanno la exists, sempre da sola. La tabella è piccola (yes/no: f
    10/13, g 9/20, h 15/15, p 14/11, q 9/17) e dà 0,594 per etichetta: uno sbilanciamento del campione piccolo. La quota
    di esempi con la exists (3,3%) viene dalla regola di scelta e va decisa coi mix;
  - insieme disegnato di 60 scene, 6 per etichetta e risposta: per ogni etichetta la stessa domanda esatta, con risposte
    opposte su immagini diverse. Tutte le exists sono verificate dal layout e tutte le altre domande sono ok. Ogni scena
    è il suo gruppo;
  - l'assenza della scorciatoia deterministica dell'etichetta non prova l'assenza di ogni bias.

**Tracciabilità del riesame** (`scripts/review_results.py`, integrazione-3, job 30917; 37 controlli su 37).
- **Ingressi verificati prima dell'uso, contro le identità registrate:**
  - index.json (`02fb3cbf…`) ed elements.jsonl.gz (`dacbea54…`) contro la sezione decontaminazione del candidato-2;
  - i parquet consultati, DynaMath `sample8` (`d886452b…`) e Geoperception (`30ac27c3…`), contro i manifest di download
    (`p6.tsv`, `p8-f1clean.tsv`).

  Una differenza ferma l'esecuzione; gli ingressi sono nel rapporto.
- **Esito:** materiale del riesame e proposta uguali byte per byte a integrazione-2. Il riesame tecnico già scritto vale
  per gli ingressi verificati, e la revisione visiva non si ripete.
- **Prime risposte:** `C-shapes-009` (0 per no) ora è una possibile equivalenza semantica (§2.13).

**`synth-v4-candidato-3`** (job 30921; `data/manifests/synth-v4-candidato-3`, manifest.json `95a4ac34…`, records
`fb8fca14…`). Derivato da candidato-2, che resta immutato.

| Record | candidato-2 | candidato-3 |
| --- | --- | --- |
| train ammessi | 53.997 | 54.337 |
| dev ammessi | 2.864 | 2.870 |
| test ammessi | 296 | 300 |
| coppie ammesse | 584 | 600 |
| in quarantena | 367 | 1 |
| esclusi | 2.792 | 2.792 |

- **Riammissioni (366):**
  - 362 dalla proposta: 340 train, 6 dev, 4 test, 12 coppie;
  - 4 coppie dopo il riesame tecnico (`synth-pair-00074-a` e `-b`, `synth-pair-00198-a` e `-b`), accettato perché
    integrazione-3 ne conferma ingressi e materiale.
- **Restano bloccati:** `synth-0016025` in quarantena; i 2.792 esclusi per contaminazione interna, 380 dei quali perdono
  i soli motivi risolti dalla revisione (la contaminazione resta).
- **Evidenza:** una per ognuno dei 747 record della proposta, con categoria, esito umano, riesame, motivi risolti coi
  dettagli del genitore e sha256 della riga.
- **Record e domande separati:** function/exists resta esclusa anche nei record riammessi: 23.170 QA in train, 1.246 in
  dev, 150 nel test; le coppie non perdono domande.
- **Invarianti:** record, gruppi, split, insiemi, immagini, JSON e QA.
- **Verifica** (job 30923, `manifest_check.py` con `config/runs/f0-c7-reader-synth.json`, collaudo e non ricetta; 70
  controlli su 70, `docs/manifests/synth-v4-candidato-3-lotto-1.json`):
  - catena 3 → 2 → 1, e le riammissioni sono esattamente le 366 previste, tutte con l'evidenza della revisione;
  - compiti dei record secondo la politica;
  - reader reale su train 54.337, dev 2.870, test 300 e coppie 600: per ogni campione prompt e target del loader sono
    quelli della scelta attesa sul pool filtrato, senza function/exists nei prompt e senza yes o no nei target function;
  - MixedRowStream col processor di Violetto e 13 arresti espliciti.
- **Pannello:** col riesame accettato il legame `synth-pair-00074-b` → DynaMath 96 non è più materiale per `pannello-v4`.
  Nessun ruolo di `pannello-v3` cambia.

**Costo.** Job 30916-30925 (30918 e 30924 non sono nostri): 12 min 5 s di job CPU su 20, con al più 4 CPU e 16 GiB per
job, compreso un test finale di 15 s dopo due ritocchi alle docstring (30925, test_synth_policy 54 su 54).
Comprende due tentativi falliti in 21 s in tutto (30919, 30920): il blocco della revisione si chiamava `revisione`,
che nel meta dei manifest è la revisione della sorgente; ora si chiama `revisione_umana`, con un test. RSS non registrata
da sacct. Nessuna GPU, training, download o agente.

**Limiti.**
- **Stato:** "ammesso" non è gold né certificazione; C8 non si applica.
- **Portata del filtro:** toglie domande dalla supervisione e dalla misura, non corregge immagini né risposte; vale per
  compito (famiglia/tipo), non per singola QA.
- **Altre domande di presenza:** contesto e sbilanciamenti sono misurati, non corretti.
- **v5:** è un generatore diagnostico, non un corpus; quota delle exists e mix sono da decidere.
- **Revisione umana:** copre i prompt storici, non quelli nuovi dei record function.

**Prossimo passo minimo**, su decisione dell'utente:
1. approvare il candidato-3 come manifest operativo dei sintetici per i collaudi;
2. decidere se trattare come i function anche le altre domande di presenza (da sole, in un generatore futuro), e con
   quale quota;
3. registrare in `pannello-v4` l'esito del legame `synth-pair-00074-b`.

## 3. MAVIS-Geometry (`f1_mavis.py`, `f1_schema.py`, shard `9c80fff3-caption`)

La verità viene da `diagram_generation`, confermata dalla didascalia (`mavis_gen.compare`: 0 segnalazioni su 199.050,
job 30025, 30052). Questo prova la coerenza fra due descrizioni, non che i pixel le rispettino (`mavis_gen.py`). Il
renderer di MAVIS non è nostro, quindi non abbiamo invarianti: l'osservabilità resta un controllo campionario (S4).

| Tipo (`mavis_questions`) | Semantica | Stato proposto |
| --- | --- | --- |
| `describe` | punti, figure, misure e angoli di `render_geometry` | `quarantena`; `target_controllato` dopo il campione di S4 |
| `side_len`, `angle_val` | lettura: valore visualizzato (decimali a due cifre per i razionali, job 30049, 30052); dal C9 l'esatto (`44/3`), il visualizzato previsto e la risposta di lettura sono nel record (`valori` di candidato-4), i target non cambiano | idem |
| `removed_side`, `direction` | lato sostituito; triangolo o semicerchio dentro o fuori | idem, priorità nel campione |
| `shared_side` | posta solo se il proprietario è unico (`f1_mavis.py:84-89`); il verificatore accetta AB = BA | idem |
| `count_points` | tutte le etichette visibili | idem, priorità nel campione |
| `exists_point`, `exists_shape` | il nome finto usa una lettera successiva a tutti i punti (`f1_mavis.py:95`): non coincide con un nome reale | idem |
| `shape_type` (`f1_mavis.py:83`) | aperta, con classi che non formano una catena: quadrato ⊂ rettangolo ∩ rombo, rombo ⊂ aquilone e parallelogramma, un triangolo rettangolo può essere isoscele | **`quarantena`**: opzioni costruite per istanza con esattamente una vera secondo le misure note; se le misure non bastano, la domanda non si pone. Dal C9 (sotto) la costruzione `mavis-c9/1` esiste ed è controllata: 118.118 domande recuperabili su 383.592, ancora escluse dal percorso operativo |

Se in F0 cambia la risoluzione (oggi 2304 patch, oltre le 1024 di addestramento di NaFlex), l'osservabilità va
riverificata per la nuova configurazione (pipeline §5.7).

**Manifest candidato** (2026-11-05, job 30626 e 30630, `scripts/mavis_manifest.py`). Cartella
`data/manifests/mavis-geometry-candidato-1`: `manifest.json` ha sha256 `dc0f264c…`, `records.jsonl.gz` ha
`f2a1c212…` (148 MiB). Il riepilogo versionato è `docs/manifests/mavis-geometry-candidato-1.json`.
- **Ingressi.** Gli 11 tar `9c80fff3-caption` (job 29419) sono letti e hashati una volta, con sha256 uguali al loro
  manifest; il blob dello snapshot ha lo sha256 dichiarato.
- **Record.** 199.050, uno per caption.jpg, con compiti, QA e digest. Le QA sono quelle che il loader costruisce con la
  chiave del tar (`data.mavis_f1_sample`). Il record porta anche i digest della descrizione canonica, della verità, di
  diagram_generation e della didascalia.
- **Gruppi.** 158.018, per stessi byte del PNG o stesso diagram_generation. 10.240 hanno più di un record (51.272 in
  tutto), e in nessuno la stessa domanda ha risposte diverse o le descrizioni differiscono.
- **Stati.**
  - Tutti i record sono ammessi come candidati: nessuna verità manca, l'andata e ritorno è perfetta, nessuna lista storica
    li segnala e nessun gruppo è in conflitto.
  - `shape_type` è in quarantena come compito in 177.944 record (383.592 domande), senza togliere gli altri compiti.
  - Gli altri otto tipi di domanda e la descrizione sono ammessi.
- **Due proprietà separate per record.** La coerenza con le annotazioni è verificata dal codice: didascalia contro
  diagram_generation, andata e ritorno della descrizione. La fedeltà all'immagine resta da verificare per tutti i
  record.
- **Dev.** 9.951 record, il 5% di ogni profondità d0-d3, seme 0; rango del campione candidato per ogni gruppo.
- **Riconciliazione.** run2 è ricalcolato con la chiave `mavis-<id>` che usava `f1_mavis_check.py`, diversa da quella del
  loader. Coincidono motivi, record pronti per profondità, domande per tipo, risposte yes/no e domande per esempio. I 60
  esempi di `sample.jsonl` sono riprodotti esattamente.
  - Caption check e geometry check non hanno segnalazioni sui 199.050.
  - Il geometry check non è ricalcolato.
  - Gli esclusi dallo sharding (1 incompleto e 552 didascalie duplicate) non sono negli shard.
  - Le 108 chiavi della vecchia validazione con l'immagine ripetuta stanno in gruppi di più record.
- **Valori.** 22.712 record hanno misure razionali disegnate come decimali. La risposta è il decimale disegnato;
  l'esatto non è nel record (C9). Dal 2026-11-10 è nel record della versione 4 (sotto, C9): 22.506 di questi record
  hanno solo decimali esatti (1.5 = 3/2), 206 hanno 249 lati arrotondati (14.67 per 44/3).
- **Limiti.**
  - Manca la fedeltà all'immagine: revisione visiva e stima di S4.
  - La decontaminazione esterna contro `pannello-v3`, che comprende le geometrie di MathVerse, degli stessi autori, è
    nella versione 2 (sotto, 2026-11-06).
  - C9 non è implementato (fatto il 2026-11-10 nella versione 4, sotto).
  - Il reader col manifest (C7, 2026-11-06, S1 §9) esclude `shape_type`: `mavis_f1` toglie dal pool le domande dei
    compiti non ammessi prima di scegliere. Chiamato direttamente, fuori dal reader, il loader sceglie ancora come prima.
- **Riverifica in sola lettura** (2026-11-05, job 30635, `scripts/manifest_recheck.py`). Il manifest è stato
  riverificato senza ricostruirlo, con la verifica corretta in `c271d6e`, che non chiede più che un altro seme cambi
  il dev.
  - `manifest.json` è quello atteso e i 10 file di evidenze sono uguali a quelli registrati.
  - Esito: 15 controlli su 15, nessuna anomalia.
  - Rapporto: `docs/manifests/riverifica-1/mavis-geometry-candidato-1-dc0f264c.json`.
- **Riverifica corretta** (2026-11-05, test nel job 30638). Le evidenze obbligatorie sono quelle di
  `mavis_manifest.evidence_files`: 5 liste e 10 file. Ognuna dev'essere registrata in `liste_importate` con un digest
  sha256 completo, uguale a quello su disco prima della verifica.
  - Si ferma se `liste_importate` è assente o vuota, se una lista o un file non sono registrati, se un digest manca o
    non è valido: un insieme incompleto non vale più come verifica riuscita.
  - **Esito sul manifest** (job 30640, `riverifica-2`). Le 10 evidenze sono registrate e uguali su disco, con gli stessi
    digest di `riverifica-1`; tre sono file vuoti, perché run2 non ha fallimenti e i due check non hanno segnalazioni.
    15 controlli su 15, nessuna anomalia.
    Rapporto: `docs/manifests/riverifica-2/mavis-geometry-candidato-1-dc0f264c.json`.
  - Non prova la fedeltà all'immagine e non completa la decontaminazione.
- **Decontaminazione esterna contro `pannello-v3`** (2026-11-06, `scripts/panel_decontam.py`, codice `861aba8`; regole
  in `docs/PANNELLO_VALUTAZIONE.md` §6.2, versione derivata in S1 §9). Ogni record contro tutte le voci dell'indice, in
  ogni ruolo, con attenzione alle geometrie di MathVerse, degli stessi autori.
  - **Immagini.** Il PNG dello shard è caption.jpg portato a 2304 patch, quindi l'identità dei pixel si controlla
    sull'originale `<cartella>/<n>_caption.jpg` (zip con sha256 ricalcolato, dimensioni uguali al JSON) e anche sul PNG.
    Il pHash entro 4 bit si calcola su entrambi.
  - **Testi.** 10-gram di `normalize_words` (max_df 25 dell'indice) per caption, diagram_generation, le tre domande e il
    CoT di `text_en`, e per la descrizione canonica e le domande costruite dal loader. text_zh, text.jpg e
    vision_dominant.jpg non sono negli shard: non si confrontano e non si dichiarano coperti.
  - **Corrispondenze testuali senza effetto** (regole esplicite `panel-decontam/2`, dal riesame del 2026-11-06; testo
    della regola in `docs/PANNELLO_VALUTAZIONE.md` §6.2). Una corrispondenza resta senza effetto solo se tutti i suoi
    10-gram sono coperti da una regola del suo campo, verificata sul testo del record. L'unica regola,
    `enumerazione_punti_loader`, copre nel solo campo `descrizione_del_loader` le sequenze di dieci etichette
    consecutive della riga `Points:` della descrizione canonica, ricalcolata dal JSON e legata al manifest da
    `descrizione_sha256`.
    - La stessa sequenza in un altro campo resta un candidato, e così ogni altro 10-gram del record.
    - La frequenza nella sorgente serve solo a ordinare i casi.
  - **Regola abolita.** Dopo la prova limitata del job 30647 (1.705 record su 16.500 candidati per la sola sequenza
    `a b c d e f g h i j` della descrizione canonica, in comune con le dodici lettere A-L di un problema di AIME 2025 I),
    la versione 2 (job 30649-30652, `panel-decontam/1`) ignorava in ogni campo i 10-gram presenti in più di 25 record di
    MAVIS. Era sufficiente la frequenza, senza un motivo specifico: così un problema specifico del benchmark ripetuto in
    più di 25 varianti perdeva tutti i candidati testuali.
  - **Provenienza.** I record di RuleBaseGeo non dichiarano una provenienza. Delle fonti dichiarate da MathVerse
    (colonna `metadata`) nessuna cita MAVIS o RuleBaseGeo. Le stesse impaginazioni o gli stessi autori non sono prove.
  - **Candidati storici.** Le scansioni `p6` e `f1eval` (convenzione `grigio-diretto/0`, sugli shard) hanno liste
    finali vuote. I loro candidati grezzi si confrontano coi nuovi solo per diagnostica, e non si importano.
  - **Scansione completa** (job 30652): tutti i 199.050 record in 53 min con 8 CPU. Gli 11 shard e lo sha256 di
    `RuleBaseGeo.zip` (26 GB) sono uguali a quelli registrati, e tutti gli originali e i PNG sono letti. I testi
    confrontati sono tutti presenti: 199.050 per ogni campo di `text_en` e per la descrizione, 3.139.028 domande del
    loader (228.994 di almeno 10 parole). Nessun controllo manca.

    Esiti con le regole esplicite (`panel-decontam/2`, riesame del job 30676, versione 3), dalle stesse evidenze:

    | Esito | Record | Che cosa |
    | --- | --- | --- |
    | corrispondenza confermata | 0 | nessun pixel identico, né dell'originale né del PNG, nemmeno verso MathVerse |
    | candidato | 714, in 156 gruppi | pHash entro 4 bit: MathVision 587 (test ammesso), MV-MATH 119 (test ammesso), We-Math 12, MathVerse 8 e MathVista 1. Testo: 50 record verso MathVision 753 (test ammesso) e 5 verso UniMER-Test |
    | evidenza senza effetto | 20.066 | sole enumerazioni dei punti, coperte da `enumerazione_punti_loader` |
    | nessuna corrispondenza | gli altri | nei controlli eseguiti |

    Nella decisione di candidato-2 (`panel-decontam/1`, regola di frequenza abolita) i candidati erano 664 in 109
    gruppi: i 50 record verso MathVision 753 erano ignorati, e i record con sole corrispondenze ignorate 20.116.

    - **Corrispondenze testuali.** La scansione ha 5 10-gram distinti in comune col pannello; la frequenza serve solo
      a ordinarli.
      - `a b c d e f g h i j` (20.071 record) e `b c d e f g h i j k` (7.612) verso l'elemento `aime2025-I:6` (dodici
        lettere A-L, sviluppo): sempre e solo nella descrizione del loader, sempre dentro la riga `Points:`. Sono coperte
        tutte e restano senza effetto.
      - `is equal to 1 what is the area of the` verso MathVision 753 (test ammesso, metrica primaria): 50 record e 47
        gruppi, in 54 domande (36 `text_lite_question`, 18 `text_dominant_question`). In ogni caso la sequenza comune
        più lunga è il 10-gram stesso, di 10 parole: soggetto e figura cambiano (per esempio "AB is equal to 1. What
        is the area of the entire shape ABCD?" contro "The shaded area is equal to 1. What is the area of the square
        ABCD?"). La frase sembra un modello del generatore di MAVIS, ma il generatore non è nel repository e la regola
        non si può legare al suo formato: nessuna regola deterministica adeguata, quindi restano candidati.
    - **I 5 candidati testuali** hanno sequenze di cifre ripetute nel CoT (`3 3 3 3 3 3 3 3 3 3` verso `hwe-0003515`,
      `2 2 2 2 2 2 2 2 2 2` verso `cpe-0001708`): sono somiglianze di forma, non prove, e restano da verificare.
    - **MathVerse:** 8 record con candidati pHash verso elementi di sviluppo; nessun pixel, nessun testo. Le sue fonti
      dichiarate sono GeoQA 1.182, Functions_collected 966, Geo2D_collected 882, Geometry3K 774, Geo3D_collected 714 e
      GEOS 210: nessuna è MAVIS o RuleBaseGeo.
    - **Candidati storici:** 598 dei 648 di `p6` sono di nuovo candidati nella decisione di candidato-2. Gli altri 50
      sono i candidati testuali verso MathVision 753, che candidato-2 ignorava per frequenza e che in candidato-3 sono
      di nuovo candidati.
  - **Versione 2** `data/manifests/mavis-geometry-candidato-2` (job 30652): `manifest.json` `8c549a23…`,
    `records.jsonl.gz` `adf9b28c…` (154 MiB), riepilogo `docs/manifests/mavis-geometry-candidato-2.json`. È decisa con
    la regola di frequenza, ora abolita. Resta immutata come genitore di candidato-3, che ne riesamina le decisioni
    testuali.
    - **Stati.** 693 record passano da ammesso a quarantena, cioè i 109 gruppi dei candidati: 683 di train, 10 di dev.
      Per profondità sono d0 643, d1 33, d2 15, d3 2. Nessuna esclusione e nessuna riammissione.
    - **Compiti.** Seguono il record; `shape_type` resta in quarantena in tutti i 177.944 record che l'hanno.
    - **Ammessi.** Train 188.416 e dev 9.941 (d0 2.466, d1 2.493, d2 2.493, d3 2.489).
    - Ogni record ha l'evidenza `decontaminazione_pannello`, che rimanda alla scansione completa
      (`data/decontam/pannello-v3/mavis-geometry/records.jsonl.gz`, `f953c323…`, 49 MiB).
    - Verifica di compatibilità 24 su 24. Verifica della versione nuova 21 su 21: il genitore con
      `mavis_manifest.verify` (15 su 15), poi le transizioni e i compiti ricalcolati dalla scansione.
    - Il job dura 56 min 9 s in tutto, con 8 CPU; la RSS arriva a 8,8 GiB.
  - **Versione 3** `data/manifests/mavis-geometry-candidato-3` (job 30676, `panel_decontam.py riesame`, codice
    `85b835a`):
    `manifest.json` `f3cca473…`, `records.jsonl.gz` `8d199336…` (162 MiB), riepilogo
    `docs/manifests/mavis-geometry-candidato-3.json`.
    - **Derivazione.** Viene da candidato-2: stessi record, gruppi e split; evidenze e motivi di candidato-2
      conservati. La catena è candidato-1 → candidato-2 → candidato-3 (S1 §9).
    - **Riesame.** Decisioni ricalcolate sulla scansione originale, senza rileggere immagini:
      - lette solo le intestazioni dei tar (398.100) e i JSON dei 20.121 record con corrispondenze testuali, 251 MiB
        su 9,2 GiB degli shard, in 133 s;
      - le 20.071 descrizioni ricalcolate hanno tutte lo sha256 del manifest.

      Uscite in `data/decontam/pannello-v3/mavis-geometry-riesame-1`: regole, decisioni per record (`6ba9eddb…`),
      riepilogo coi contesti dei casi.
    - **Cambi rispetto a candidato-2**, tutti per i 50 record verso MathVision 753, ora candidati:
      - i loro 47 gruppi passano da nessuna corrispondenza a candidato;
      - 1.591 record vanno da ammesso a quarantena: train 1.577, dev 14; per profondità d0 1.524, d1 40, d2 16, d3 11;
      - 1.406 di questi stanno in due gruppi di figure identiche di profondità 0 (763 e 643 record), con 4 e 1 record
        che hanno la domanda.

      Nessun esito diventa meno grave, nessuna riammissione, nessuno stato dipende dalla sola decisione precedente.
      L'elenco completo, con le cause, è nel riepilogo.
    - **Motivi.** Ognuno dei 1.591 record ha il motivo nuovo del riesame, accanto a quelli di candidato-2.
    - **Compiti.** Seguono il record; `shape_type` resta in quarantena.
    - **Ammessi.** Train 186.839 e dev 9.927 (d0 2.454, d1 2.491, d2 2.493, d3 2.489); quarantena 2.284.
    - **Verifica della catena 25 su 25.** Comprende:
      - candidato-2 con la sua regola storica (23 su 23) e candidato-1 con `mavis_manifest.verify`;
      - decisioni, evidenze, motivi, stati e compiti ricalcolati;
      - l'elenco dei cambi uguale a quello ricalcolato.
    - Il job dura 7 min 25 s con 4 CPU; la RSS arriva a 10,6 GiB.
  - **Da rivedere a vista** (nessun candidato è risolto): i 156 gruppi in quarantena di candidato-3, a partire dai
    pHash verso il test ammesso di MathVision e MV-MATH e dai 47 gruppi della domanda verso MathVision 753.
    - Una regola che declassi quella domanda va motivata e approvata prima: per esempio legata al modello delle domande
      di MAVIS, sequenza comune di sole 10 parole, soggetto diverso.
    - Senza una regola approvata, quei gruppi restano in quarantena.
  - **Limiti.**
    - Il pHash non riconosce la stessa figura disegnata in un altro stile o con un'altra impaginazione.
    - text_zh, text.jpg e vision_dominant.jpg non sono confrontati.
    - L'esposizione pregressa resta.
    - La fedeltà all'immagine resta da verificare (S4).
    - La decontaminazione non dice nulla sulla correttezza delle risposte e non è una certificazione statistica.

### C9: valori esatti e `shape_type` con opzioni (2026-11-10, `src/limite_vl/mavis_c9.py`, `scripts/mavis_c9.py`)

Job 30773 (sonda), 30775 (test), 30776 (costruzione di candidato-4), 30777 (regressione di C7), 30779 (reader col
candidato C9). La costruzione storica (`f1_mavis.mavis_questions`) resta quella predefinita e non cambia; la nuova,
`mavis-c9/1`, si usa solo per un record che la dichiara (`costruzione_qa`).

- **Tre livelli di fatti, tenuti separati.**
  - *Annotazioni* (nascoste al modello): tipo dichiarato da diagram_generation (`mavis_gen.figures`), misure esatte
    (`mavis_gen.measures`, confermate dalla didascalia), vertice degli isosceli dalla frase di costruzione.
  - *Stampato* (al modello, per la regola del renderer): le misure della didascalia scritte come
    `f1_schema.latex_value` (razionali senza radice come decimali con al più due cifre; il 90° come segno). È la
    rappresentazione prevista, non una lettura dei pixel.
  - *Da verificare a vista*: fedeltà e leggibilità, segni non dichiarati come misure (il segno dell'angolo retto di un
    triangolo rettangolo, visto in contact_run1 ma non garantito), forma disegnata. Non entrano in nessuna prova.
- **Sonda** (job 30773, `outputs/mavis_c9/sonda-1`, 1 min 54 s). Le 383.592 domande `shape_type` storiche riguardano
  sei tipi dichiarati:

  | Tipo | Figure | Misure stampate |
  | --- | --- | --- |
  | quadrato | 78.733 | angolo retto; un lato (35.665), due adiacenti (28.594) o due opposti (14.474) |
  | isoscele | 61.813 | angolo al vertice (30, 45, 60, 90, 120); due lati al vertice (43.051) o uno (18.762); vertice sempre nella frase di costruzione, al centro del nome |
  | settore | 61.785 | angolo al centro (30-120, mai 180), uno o due raggi |
  | parallelogramma | 60.936 | angolo di 30, 45 o 60 (mai 90), due lati adiacenti sempre |
  | rettangolo | 60.542 | angolo retto, due lati adiacenti sempre |
  | triangolo rettangolo | 59.783 | angolo acuto (30, 45, 60) al terzo vertice; l'angolo retto non è mai una misura |

  Nessun rombo, aquilone, trapezio, equilatero o triangolo generico dichiarato; un solo angolo stampato per figura.
- **Valori.** Ogni record di candidato-4 ha `valori`: elemento, esatto (`44/3`), visualizzato previsto (`14.67`),
  rappresentazione e risposta di lettura; le domande di lettura del pool C9 dicono la misura che chiedono (`elemento`).
  - I target non cambiano: la risposta di lettura è il visualizzato, ricalcolata dal valore registrato per ogni domanda
    nella verifica.
  - Angoli tutti interi (498.089). Lati: 597.103 interi, 111.856 radici, 24.856 decimali esatti (x.5), 249 decimali
    arrotondati (p/3) in 206 record. I 22.712 record con razionali non interi non sono 22.712 errori: in 22.506 il
    decimale stampato è esatto; nei 206 l'esatto differisce dal target di lettura, che resta il visualizzato.
  - Un esatto non si ricostruisce da un decimale arrotondato. Un decimale arrotondato stampato vale l'intervallo di
    ±0,005: due 14.67 non dimostrano un'uguaglianza, 14.67 e 15 una differenza sì.
- **`shape_type`: insieme fisso di opzioni per famiglia**, mescolato per chiave (`f1c.key_rng`):
  - triangoli: acute, right, obtuse triangle (una partizione: esattamente una è vera per ogni triangolo);
  - quadrilateri: rectangle, rhombus, square.

  La risposta è l'unica opzione vera secondo le annotazioni (tipo dichiarato e misure esatte, mai i decimali); ogni
  altra opzione dev'essere confutata dalle sole misure stampate; una verità ignota non è mai trattata come falsa.
  Altrimenti la domanda è omessa col motivo. Le classi escluse lo sono per i fatti della sonda: un'opzione mai
  confutabile comparirebbe solo come risposta, e un insieme scelto per istanza ne rivelerebbe la presenza.
  - `parallelogram`: vera per ogni quadrilatero di MAVIS. `sector`: vera per ogni figura circolare.
  - `isosceles triangle`: mai confutabile dalle misure stampate; `equilateral` e `scalene` non formano una partizione
    con le classi degli angoli (definizione inclusiva).
  - `kite`, `trapezoid`: definizioni inclusiva ed esclusiva in conflitto.
- **Controllo indipendente** (`check_question`, scritto a parte dal generatore): la verità esatta da un modello numerico
  della figura (coordinate dal tipo e dalle misure esatte; una lunghezza non data si prova con due testimoni, e una
  classe che cambia fra i due è ignota); le confutazioni da un problema di ammissibilità sugli angoli e da intervalli
  sui lati stampati; testo, ordine e risposta. Riverifica ogni domanda emessa, nella scansione e nella verifica.
- **Esiti sui 199.050 record** (job 30776, scansione in 145 s; per profondità in `outputs/mavis_c9/c9-1/report.json`):

  | Tipo | Recuperabili | Omesse e motivo |
  | --- | --- | --- |
  | isoscele | 50.550: acute 25.737, right 12.458, obtuse 12.355 | 11.263: un solo lato e angolo al vertice di 30, 45 o 60°, che nella figura può essere alla base |
  | rettangolo | 57.310 rectangle | 3.232: lati adiacenti uguali, rettangolo, rombo e quadrato tutti veri |
  | parallelogramma | 3.223 rhombus (lati adiacenti uguali) | 57.713: nessuna opzione vera (parallelogram esclusa) |
  | triangolo rettangolo | 7.035 right (cateti stampati uguali e 45°) | 52.748: richiedono il segno dell'angolo retto, che non è una misura stampata |
  | quadrato | — | 78.733: tre opzioni vere |
  | settore | — | 61.785: nessun insieme informativo |
  | totale | 118.118 (30,8%) | 265.474 |

  - Nessuna misura contraddittoria, nessuna verità ignota, nessun valore non analizzabile.
  - Profondità 0: nessun isoscele acute e nessun triangolo rettangolo recuperabile (un solo lato stampato).
  - Posizione della risposta fra il 33,0% e il 33,4% per famiglia.
  - Risposte: quadrilateri rectangle 57.310 e rhombus 3.223 (l'insieme dà quasi sempre rectangle: square non è mai la
    risposta); triangoli acute 25.737, right 19.493, obtuse 12.355.
  - Le altre domande sono identiche in ogni record; il pool storico e la descrizione ricalcolati hanno i digest di
    candidato-3 in tutti i record.
- **Versione 4** `data/manifests/mavis-geometry-candidato-4` (job 30776): `manifest.json` `77e1849e…`,
  `records.jsonl.gz` `0c5a7d0a…` (193 MiB), riepilogo `docs/manifests/mavis-geometry-candidato-4.json`.
  - Derivata da candidato-3: stessi record, gruppi, split, stati, motivi ed evidenze. Ogni record dichiara
    `costruzione_qa`, ha il pool C9 (`qa`), `valori`, `shape_type_c9` (per figura: esito, motivo, opzioni, risposta,
    confutazioni e i fatti usati) e un'evidenza `mavis-c9`.
  - `shape_type` resta in quarantena (compito presente solo nei record con domande recuperabili).
  - Le domande nuove (721 testi distinti) contro le 99.713 voci di `pannello-v3`, coi 10-gram e le regole di
    `panel_decontam` (nessuna regola esplicita per questo campo): nessuna corrispondenza, nessun cambio di stato. Le
    evidenze sulle immagini, invariate, sono ereditate.
  - Ammessi train 186.839 e dev 9.927, come candidato-3. Verifica 7 su 7: campi ereditati, stati, motivi ed evidenze;
    domande diverse da `shape_type` uguali al genitore; ogni domanda di lettura ricalcolata dal visualizzato
    registrato; ogni domanda `shape_type` ricalcolata dalle opzioni e riverificata coi fatti registrati; conteggi e
    `keys/`.
  - Nessuna configurazione lo usa: `config/runs/f0-c7-reader-unimer-mavis.json` resta su candidato-3.
- **Verifica dei valori contro le annotazioni** (correzione della verifica, 2026-11-10, job 30780-30781; rapporto
  `docs/manifests/verifica-valori-1/mavis-geometry-candidato-4-77e1849e.json`).
  - *Lacuna.* La verifica del job 30776 controllava solo l'autocoerenza dei valori: risposta di lettura uguale al
    visualizzato, digest delle domande di lettura. Un esatto sbagliato scritto con hash coerenti l'avrebbe superata.
  - *Correzione.* `verify_c9` rilegge i soli JSON originali degli shard (`shard_jsons`: posizione e json_sha256 del
    manifest, immagini saltate). Ricava le misure con la normalizzazione della costruzione (`measure_values`:
    `mavis_gen.measures` e `facts.norm_value`) e confronta elemento, esatto, rappresentazione, visualizzato previsto e
    risposta di lettura, contando gli elementi mancanti, ripetuti o in più. Un controllo scritto a parte guarda la
    coerenza di ogni voce con la regola di rappresentazione. Restano i controlli sui digest delle domande storiche.
  - *Test* (job 30780, `tests/test_mavis_c9.py` 93 su 93). Validi: un decimale esatto (3/2, 1.5) e uno arrotondato
    (44/3, 14.67). Con gli hash interni aggiornati la verifica trova esatto alterato, rappresentazione alterata,
    esatto alternativo con lo stesso visualizzato e voce coerente (44/3 -> 4402/300), elemento mancante, ripetuto o
    in più.
  - *Valori rispetto alle annotazioni* (job 30781, candidato-4 invariato, 11 controlli su 12). Su 199.050 record e
    1.232.153 misure nessuna discrepanza.
  - *Rappresentazione prevista dal renderer.* Un'incoerenza reale, in `d3-44919` `side:EH`:

    | Campo | Osservato | Atteso | Rispetto alle annotazioni |
    | --- | --- | --- | --- |
    | esatto | `√841` (= 29) | `√841` | uguale |
    | visualizzato previsto | `\sqrt{841}` | `\sqrt{841}` | uguale alla regola del renderer |
    | risposta di lettura | `\sqrt{841}` | `\sqrt{841}` | uguale al target storico |
    | rappresentazione | `intero` | `radice` | la regola scrive l'esatto come radice |

    `mavis_c9.representation` classifica sul valore normalizzato (√841 = 29) invece che sulla scrittura. È l'unica
    misura con un radicando quadrato perfetto. Non tocca i target di lettura né le decisioni di `shape_type`, che
    usano il valore esatto 29. Non è corretta in candidato-4, che resta immutato: la correzione è la versione 5
    (sotto).
  - *Fedeltà ai pixel*: non verificata da nessuno di questi controlli; resta alla verifica a vista.
- **Versione 5: correzione della rappresentazione** (2026-11-10, job 30783 e 30785; `scripts/mavis_c9.py correzione`;
  riepilogo `docs/manifests/mavis-geometry-candidato-5.json`, rapporto
  `outputs/manifest_check/mavis-geometry-candidato-5/verifica.json`).
  - *Regola corretta.* `mavis_c9.representation` descrive la scrittura prevista dal renderer
    (`f1_schema.latex_value`), non il valore: una scrittura con una radice è `radice` anche quando il valore è
    razionale (`√841` = 29, scritto `\sqrt{841}`). Il valore matematico dei confronti (`parse_exact`), l'esatto, il
    visualizzato previsto, la risposta di lettura, le domande, le opzioni e le decisioni geometriche non cambiano:
    `shown_of` guarda solo i decimali arrotondati.
  - *Test* (job 30783, `tests/test_mavis_c9.py` 102 su 102, 9 nuovi): `√841`, `√16` e `3√4/2` (radici di quadrati
    perfetti) e `√2` registrati come `radice`; `42` intero; `3/2` decimale esatto; `44/3` decimale arrotondato; valore
    29 nei confronti e target `\sqrt{841}` invariati. Su un fixture: un solo cambio da un genitore con l'etichetta
    storica, record senza cambi byte per byte uguali, un'ulteriore differenza della regola documentata e non
    corretta, una differenza non prevista nel figlio trovata dal confronto.
  - *Derivazione.* `data/manifests/mavis-geometry-candidato-5` (`manifest.json` `314a0a52…`, `records.jsonl.gz`
    `a67aed7a…`) da candidato-4 (`77e1849e…`, invariato coi suoi rapporti). Le righe senza cambi sono quelle del
    genitore, byte per byte; nel record cambiato si ricalcola solo la rappresentazione e si aggiunge l'evidenza
    `mavis-c9-rappresentazione`. Nessuna QA, immagine o decontaminazione rigenerata. `manifest.json` registra regola,
    funzione e cambi (`correzione_rappresentazione`), `precedente`, `split.ereditato_da` e la catena delle versioni.
  - *Esito* (job 30785, 3 min 46 s): un solo cambio, `d3-44919` `side:EH` `intero` → `radice`; nessuna ulteriore
    differenza della regola.
  - *Confronto sistematico col genitore* (verifica 13 su 13). 199.049 record byte per byte uguali e 1 cambiato nei
    soli campi `valori` (la rappresentazione) ed `evidenze` (l'evidenza della correzione). Restano uguali gruppi,
    split, stati, motivi ed evidenze pregresse, esatti, visualizzati, risposte di lettura, QA e digest, compiti ed
    esiti di `shape_type`. `manifest.json` differisce solo nei metadati di versione, provenienza e conteggio; `keys/`
    ricalcolati uguali.
  - *Valori contro i JSON originali* degli 11 shard (posizione e json_sha256 del manifest, immagini non lette):
    1.232.153 misure attese, registrate e confrontate, nessuna discrepanza e nessuna incoerenza. Rappresentazioni:
    intero 1.095.191, radice 111.857, decimale esatto 24.856, decimale arrotondato 249.
  - Nessuna configurazione la usa; `shape_type` resta esclusa dal percorso operativo.
- **Revisione visiva mirata, campione-1** (2026-11-10, `scripts/mavis_review_package.py`,
  `slurm/mavis_review_package.sbatch`, job 30787-30788; pacchetto `revisioni/mavis-c9-campione-1`).
  - *Natura.* Campione diagnostico, non rappresentativo della frequenza naturale degli errori (S4): niente intervalli
    di confidenza, stime di tasso o certificazioni; gli esiti non ammettono record, non promuovono a gold e non
    attivano il training. Nessuna chiamata a modelli, nessun giudizio compilato.
  - *Quote dell'utente* (80 schede): 1 controllo noto (`d3-44919` `side:EH`, √841, non indipendente); 19 letture di
    valori (4 decimali arrotondati, 4 esatti, 4 radici, 4 interi, 3 angoli); 30 classificazioni recuperabili (6 per
    acute, right e obtuse triangle, rectangle, rhombus); 15 omesse; 15 altri compiti (3 ciascuno per describe,
    direction, count_points, removed_side, shared_side). Le omesse:
    - 4 `piu_opzioni_vere`;
    - 4 misure insufficienti: `non_deciso_dalle_misure_stampate` senza uguaglianze apparenti;
    - 4 uguaglianze apparenti dovute all'arrotondamento: `non_deciso_dalle_misure_stampate` con due lati scritti
      uguali, di cui uno decimale arrotondato;
    - 3 `richiede_segno_angolo_retto`.
  - *Selezione* (fissata prima della revisione, dai soli metadati, mai guardando le immagini). Record di train ammessi
    di candidato-5; il controllo noto per nome. Una scheda per gruppo. In ogni strato celle a giro con d3 per prima: per
    le letture profondità e affollamento alternato (misure stampate sopra o non sopra la mediana della profondità: 3, 5,
    7 e 10 per d0-d3); per gli altri strati la profondità. Nella cella, gruppi in ordine di
    sha256({campione, seme 0, strato, gruppo}); nel gruppo l'unità con lo sha256 minore. Strati dal meno disponibile al
    più disponibile; celle esaurite saltate e registrate; un deficit si registra, mai compensato con un altro strato.
    Ordine delle schede per sha256({campione, seme 0, ordine: unità}), con identificativi neutri M01-M80.
  - *Esito.* Nessun deficit. Profondità: d0 9, d1 18, d2 24, d3 29 schede. Limiti della disponibilità, registrati in
    `moduli/piano.json`: le uguaglianze apparenti sono esattamente 4 gruppi, tutti in d3; i decimali arrotondati
    (198 gruppi) stanno solo in d2 e d3 con affollamento basso; nessun acute triangle recuperabile in d0.
  - *Domande e target* ricalcolati dai JSON originali (posizione e json_sha256) col codice di C9 e confrontati coi digest
    del manifest: pool, valori ed esiti di `shape_type` di ogni record scelto. Le omesse non hanno golden answer: una
    richiesta diagnostica esplicita con l'insieme della famiglia, più risposte ammesse.
  - *Immagini*: i `caption.jpg` originali di `RuleBaseGeo.zip` (blob `d5b7bd32…`), coi `byte_sha256`, membro e
    dimensioni della scansione registrata nel manifest (`f953c323…`) e del JSON; nessun ridimensionamento.
  - *Interfaccia* (dall'impianto di `synth-v4-campione-2`, senza modificarlo). Primo passaggio senza target,
    annotazioni, esiti, categorie e motivi; secondo passaggio dopo il salvataggio, col primo giudizio fissato. Campi per
    scheda: scrittura visibile ed equivalenze per le letture, prudenza del filtro e classi vere per le diagnostiche.
    Restano registro delle esposizioni, unione degli stati, archivio, esportazione e importazione.
  - *Verifiche* (job 30787-30788): `tests/test_mavis_review_package.py` 24 su 24; verifica statica 27 su 27, con
    selezione e schede ricalcolate dal manifest e dagli shard e condizioni degli strati riverificate con codice scritto
    a parte; JavaScript eseguito con Duktape: senza DOM 48 superati e 1 saltato (l'interfaccia), su DOM simulato 69 e
    1 saltato (le immagini), nessuna soluzione nell'output. Nel job 30787 due righe dell'autotest nominavano la scheda
    di prova, scelta per un target lungo: tolte, firme e verifiche rifatte nel 30788. Non è una prova in un browser: il
    collaudo sul portatile è `verifica.html`.
- **Reader.**
  - Regressione di C7 sul candidato corrente (job 30777, `docs/reader_smoke/c7-3.json`): 22 controlli su 22, uguale a
    c7-2 tolti job, tempi e codice.
  - Candidato C9 col reader (job 30779, `docs/reader_smoke/c9-lettore-1.json`), 8 controlli su 8. Sul primo shard,
    letto per intero col candidato-4 e col candidato-3, le chiavi consegnate sono quelle ammesse dai record grezzi, in
    train (18.060) e in dev (933).
  - Ogni campione costruito col manifest C9 ha il digest del pool C9, solo compiti ammessi, mai `shape_type`, e gli
    stessi messaggi del genitore. Le domande `shape_type` bloccate scendono da 29.154 a 9.472 in train.
  - Senza la costruzione dichiarata il pool C9 è rifiutato in tutti i record con `shape_type` storiche.
- **Limiti.**
  - Le prove riguardano annotazioni e trasformazione, non i pixel: misure stampate e figure vanno verificate a vista
    prima di usare le domande; una confutazione poggia sulla misura stampata prevista dalla regola del renderer.
  - L'insieme dei quadrilateri ha risposte molto sbilanciate e `square` non è mai la risposta.
  - Senza un'approvazione dell'utente le domande non diventano gold né entrano nel percorso operativo.

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

**Manifest candidato** (2026-11-05, job 30624-30625, `scripts/unimer_manifest.py`). Cartella
`data/manifests/unimer-candidato-1`: `manifest.json` ha sha256 `bcc85198…`, `records.jsonl.gz` ha `c183b955…` (262
MiB). Il riepilogo versionato è `docs/manifests/unimer-candidato-1.json`.
- **Ingressi.** Gli shard densi di `2343ddd9` (job 29729) sono letti e hashati una volta, con sha256 uguali al loro
  manifest. I blob dello snapshot di Hugging Face hanno lo sha256 dichiarato dallo sharding.
- **Record.** 985.450 formule e 30.685 righe ritagliate. Ogni riga ritagliata ha il genitore e sta nel gruppo e nello
  split della sua formula. UniMER-Test non entra nel manifest: serve solo a riconoscere le contaminazioni.
- **Gruppi.** 933.870, per stessi byte del PNG: 50.866 duplicati con la stessa formula e 402 gruppi in conflitto.
  Formule uguali con immagini diverse non si uniscono, perché una formula comune non identifica un problema (S1 §5).
- **Stati delle formule.**
  - 983.875 ammesse, di cui 49.195 in dev, cioè il 5%, con seme 0;
  - 1.523 in quarantena: 1.119 incerti di run4 e 404 formule di UniMER-Test con più di 20 token visibili e un'immagine
    diversa, da riesaminare;
  - 52 escluse: 51 errori certi di run4 e 1 con la stessa immagine di un campione di UniMER-Test, escluso col suo
    gruppo.
- **Stati delle righe.** 30.588 ammesse; 91 in quarantena (89 dalla formula, 2 perché la formula è rietichettata e le
  righe vengono dall'etichetta originale); 6 escluse.
- **Rietichettature.** Le 18.827 sono trasformazioni tracciate, con regola e genitore, non correzioni certificate. Le
  formule ammesse sono candidate, non gold.
- **Conteggi documentati, verificati invece che copiati.** run4 e run2 sono ricalcolati dalle regole correnti e coincidono
  coi loro file:
  - tutti i 19.997 motivi e le 18.827 rietichettature;
  - 51 errori certi, 1.119 incerti, di cui 1.116 in 402 gruppi in conflitto;
  - 152.781 corrispondenze col test e 405 contaminazioni.

  Solo `immagine_vuota`, che decodifica i pixel, non è ricalcolata: in run4 non ha casi.
- **Limiti.**
  - La decontaminazione esterna contro `pannello-v3` è nella versione 2 (sotto, 2026-11-06).
  - 152.376 formule del training uguali a una formula breve del test restano un'evidenza senza cambio di stato: la
    metrica senza contaminazione usa i casi mai visti di run2.
  - Il dev misura immagini nuove, non formule nuove.
  - I reader applicano il manifest (C7, 2026-11-06, S1 §9) con `unimer-candidato-2`: formule (transcribe) e righe
    ritagliate (transcribe_line, transcribe_line_select) con decisioni proprie, rietichettature dalla lista registrata.
- **Verificatore corretto** (2026-11-05, `c271d6e`, test nel job 30633).
  - La riga ritagliata deve avere il PNG della formula, con gli stessi byte e le stesse dimensioni. Righe e fasce devono
    essere strutturalmente valide (`band_problems`): coppie di interi con 0 ≤ y0 < y1 ≤ altezza, in ordine, senza
    sovrapposizioni, una per riga, con le dimensioni del JSON uguali a quelle del PNG.
  - Costruzione e verifica lo richiedono prima di gruppi e ammissioni. Il PNG della riga non crea legami.
  - La verifica ricalcola lo stato di ogni record dalle evidenze, e vince il più grave.
  - Non chiede più che un altro seme cambi il dev.
- **Riverifica in sola lettura** (job 30634, `scripts/manifest_recheck.py`).
  - **Ingressi.** `manifest.json` è quello atteso (`bcc85198…`). Gli 8 file di run4 e run2 sono uguali a quelli
    registrati.
  - **Verifica corrente.** 23 controlli su 23:
    - le 30.685 righe ritagliate hanno il PNG e la griglia della loro formula;
    - lo stato di 1.016.135 record su 1.016.135 è quello richiesto dalle evidenze;
    - il gruppo con l'immagine di UniMER-Test (2 record) è escluso;
    - i 404 gruppi con una formula lunga del test (493 record) non hanno record ammessi.
  - **Righe contro lo shard.** Lo shard delle righe è stato riletto e hashato: lo sha256 `43c06874…` è uguale.
    - Tutte le 30.685 righe hanno JSON, PNG, etichetta, posizione e numero di righe del manifest, con fasce valide.
    - Il genitore torna coi digest del manifest. Gli shard di train non sono stati riletti.
    - Le fasce per riga vanno da 2 a 28; la più bassa è alta 15 px.
    - In 255 record due fasce si toccano, con spazio 0: è ammesso perché gli intervalli sono semiaperti, ma lì il
      loader non aggiunge margine verso la riga vicina.
  - **Esito.** 28 controlli su 28, nessuna anomalia, nessuna nuova versione necessaria.
  - **Rapporto.** `docs/manifests/riverifica-1/unimer-candidato-1-bcc85198.json`.
  - **Limite.** La validità delle fasce è strutturale: che ogni fascia contenga la sua riga resta da verificare
    guardando.
- **Riverifica corretta** (2026-11-05, test nel job 30638). Due controlli in più, che fermano la riverifica prima di
  scrivere:
  - **Classificazione degli shard.** L'`insieme` di ogni shard in `manifest.json` dev'essere quello di ciascuno dei suoi
    record. Prima, uno shard delle righe classificato `train` saltava la rilettura dei ritagli, e il controllo passava
    con zero righe attese e zero controllate. Ora le righe attese si ricavano dai record e vanno coperte tutte e sole.
    Zero righe resta valido se il manifest non ne ha.
  - **Evidenze obbligatorie.** Sono gli 8 file di run4 e run2 che la costruzione registra, compresi quelli che la
    verifica rilegge. Devono essere in `liste_importate` con digest sha256 completi e uguali su disco, prima dell'uso.
    `split_duplicates/run1` è registrata ma non riletta (serve solo alla costruzione), ed è elencata a parte.
  - **Esito sul manifest** (job 30639, `riverifica-2`). La classificazione è coerente: 5 shard `train` con 985.450
    formule, 1 shard `righe` con 30.685 righe. Le 8 evidenze sono registrate e uguali su disco, con gli stessi digest di
    `riverifica-1`. 28 controlli su 28, nessuna anomalia: le 30.685 righe ricavate dai record sono tutte controllate, e
    le fasce danno gli stessi numeri di `riverifica-1`.
    Rapporto: `docs/manifests/riverifica-2/unimer-candidato-1-bcc85198.json`.
  - Non prova la corrispondenza visiva delle fasce e non completa la decontaminazione.
- **Decontaminazione esterna contro `pannello-v3`** (2026-11-06, `scripts/panel_decontam.py`, codice `861aba8`; regole
  in `docs/PANNELLO_VALUTAZIONE.md` §6.2, versione derivata in S1 §9). Ogni record contro tutte le 99.713 voci
  dell'indice, in ogni ruolo.
  - **Identità.** Formula e riga ritagliata hanno la stessa chiave e lo stesso record_id. La scansione le distingue per
    insieme, shard e posizione; la verifica riconduce ogni riga al suo record coi digest di JSON e PNG del manifest.
  - **Immagini.** Gli shard densi sono ingranditi fino a 4x, e i loro pixel non possono coincidere con quelli di un
    originale. L'identità dei pixel si controlla quindi sull'originale `UniMER-1M/images/<indice>.png` (zip con sha256
    ricalcolato, dimensioni uguali al JSON), sul PNG dello shard e, per le righe, sui ritagli di tutte le fasce,
    costruiti come `data.unimer_line_sample`. Il ritaglio si confronta solo per identità: la stessa riga tagliata con
    margini diversi non si riconosce.
  - **Formule.** La formula visibile di run2 contro UniMER-Test (dal pannello) e contro le 146 risposte delle 100
    domande HME100K di OCRBench. Oltre 20 token visibili è un candidato; le formule più brevi restano evidenze senza
    effetto. Le righe ritagliate confrontano le righe del LaTeX, cioè i loro target. Il pHash delle formule non si usa.
  - **Riconciliazione con run2.** UniMER-Test di run2 è quello del pannello (23.757 chiavi). Per ogni formula, la
    corrispondenza ritrovata dal pannello deve essere quella di run2 (breve, lunga o nessuna). Il record con
    l'immagine del test di run2 deve essere ritrovato dall'identità dei pixel degli originali; il dato è riportato, e lo
    stato dato da run2 resta in ogni caso.
  - **Prova limitata** (job 30646): 24.000 record in 44 s con 6 CPU, nessun problema; verifica parziale 25 su 25.
  - **Scansione completa** (job 30651): tutti i 1.016.135 record (985.450 formule, 30.685 righe) in 19 min con 4 CPU.
    I 6 shard hanno sha256, byte e campioni del manifest; lo sha256 di `UniMER-1M.zip` è ricalcolato e uguale. Tutti
    gli originali e i PNG sono letti, e i ritagli coprono tutte le 81.606 fasce. Nessun controllo manca.

    | Esito | Record | Che cosa |
    | --- | --- | --- |
    | corrispondenza confermata | 2, un gruppo | `train-0281221` e la sua riga: pixel dell'originale identici a un elemento di UniMER-Test (test ammesso). È l'immagine del test di run2, già esclusa |
    | candidato | 404 | formula lunga uguale a una di UniMER-Test: le quarantene di run2. Nessuna formula lunga di OCRBench, nessuna riga ritagliata con una formula lunga |
    | evidenza senza effetto | 152.413 e 4 | formule brevi uguali a UniMER-Test (152.376 formule, come in run2, e 37 righe) e a OCRBench |
    | nessuna corrispondenza | tutti gli altri | nei controlli eseguiti: nessun pixel identico verso gli altri benchmark, nessun ritaglio identico |

    Il record con l'immagine di run2 ha anche una formula lunga del test: i due casi restano distinti nell'evidenza
    (pixel, confermato; formula lunga, candidato). La riconciliazione per record coincide con run2: 152.376 brevi, 404
    lunghe, 1 immagine, 832.669 nessuna.
  - **Versione 2** `data/manifests/unimer-candidato-2` (job 30651): `manifest.json` `97af1033…`, `records.jsonl.gz`
    `7b33bc25…` (285 MiB), riepilogo `docs/manifests/unimer-candidato-2.json`.
    - **Nessun cambio di stato.** I 405 gruppi toccati (495 record) erano già esclusi o in quarantena per run2. Hanno il
      motivo nuovo, ma conteggi e chiavi ammesse restano quelli della versione 1.
    - Ogni record ha l'evidenza `decontaminazione_pannello`, che rimanda alla scansione completa
      (`data/decontam/pannello-v3/unimer/records.jsonl.gz`, `2de700c4…`, 234 MiB).
    - Verifica di compatibilità 26 su 26. Verifica della versione nuova 20 su 20: il genitore con
      `unimer_manifest.verify` (23 su 23), poi le transizioni ricalcolate dalla scansione.
    - Il job dura 27 min 47 s in tutto, con 4 CPU; la RSS arriva a 20,3 GiB nella verifica della versione nuova.
  - **Limiti.**
    - Il pHash delle formule non si usa.
    - Le formule dentro i testi dei problemi degli altri benchmark non si confrontano.
    - Il ritaglio si confronta solo per identità: non riconosce lo stesso contenuto con altri margini.
    - L'esposizione pregressa di SigLIP2 e Violetto resta.
    - Le 404 formule lunghe restano da riesaminare (S1 §5): la decontaminazione non le risolve.
    - La decontaminazione non dice nulla sulla correttezza delle trascrizioni e non è una certificazione statistica.

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
  Implementato in `tests/test_mavis_c9.py` (job 30775, 84 controlli, §3 C9): il rettangolo isoscele dà `right` fra
  acute/right/obtuse, senza `isosceles` fra le opzioni; il rombo che è un quadrato per le misure, il quadrato e il
  rettangolo coi lati adiacenti uguali sono omessi (tre opzioni vere); insufficienze, contraddizioni, uguaglianze
  apparenti da arrotondamento e mutazioni del controllo indipendente come in §3.
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
  equivalenti. In `tests/test_mavis_c9.py` anche `44/3` disegnato `14.67`: target di lettura `14.67`, esatto `44/3`
  conservato, i due valori non equivalenti; due `14.67` non dimostrano un'uguaglianza.
- **UniMER.** Una rietichettatura non coerente con l'etichetta attesa dà errore (`stream.apply_relabel`, già presente).
