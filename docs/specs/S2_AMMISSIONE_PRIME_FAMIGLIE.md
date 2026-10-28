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

### 2.1 Errore confermato: `intersections`

- **Codice.** `_intersections` conta le radici con |x|, |y| ≤ R = 6 e scarta quelle a meno di 0.4 da R
  (`synth.py:388-414`, righe 407-408). `_draw_function` disegna x ∈ [−7, 7] e maschera solo |y| > 7 (righe 462 e 466):
  il tratto fra 6 e 7 è visibile ma non contato.
- **Controesempio** (verificato eseguendo `make_spec("audit-68", "function")`): f = −0.5x − 4, g = −2x + 4,
  intersezione in (16/3, −20/3). È nel tratto disegnato, ma il target di "How many times do the graphs of f and g
  intersect in the figure?" è `0`.
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
| shapes | `count`, `has_type`, `count_type`, `exists` | celle 3x3 con spostamenti casuali (righe 478-485): 19 scene su 15.000 hanno riquadri sovrapposti (candidati) | sovrapposizione esatta dei contorni nel render; vincolo nel generatore |
| pair | controfattuali (`counterfactual`) | un dettaglio cambia la risposta (righe 581-637) | come il tipo di domanda corrispondente, per entrambe le immagini |

### 2.4 Invarianti del renderer

Si calcolano durante il render dagli oggetti di matplotlib (`get_window_extent` di testi e marcatori) e si salvano nel
JSON di v4, così si possono ricontrollare senza ridisegnare:

- riquadri di testi e marcatori dentro l'immagine;
- nessuna sovrapposizione fra testi, né fra un testo e il marcatore di un altro oggetto; nelle shapes, contorni senza
  sovrapposizioni;
- altezza dei testi in pixel, alla risoluzione data al modello, non sotto una soglia (proposta: 10 px).

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

- **Intersezioni.**
  - `audit-68`: con la correzione, il target è `1`.
  - Intersezione con x ∈ (6.4, 6.6) e \|y\| < 6: contata, mentre oggi non lo è.
  - Intersezione con \|y\| ∈ (7.4, 7.5): non disegnata, quindi non contata.
  - Intersezioni a meno di 0.4 da \|x\| = 7 o \|y\| = 7: niente domanda.
  - Tangenza, due radici vicine, rette parallele: niente domanda.
  - Su 10.000 scene il conteggio analitico coincide con quello numerico sulle spezzate.
- **Unicità.** Su 10.000 specifiche per famiglia, esattamente un'opzione vera per ogni domanda di classificazione,
  calcolata con predicati scritti a parte. Un quadrato con le opzioni square/rectangle/parallelogram/trapezoid accetta
  solo `square`.
- **MAVIS.** Per un triangolo rettangolo con i cateti uguali, right e isosceles sono entrambe vere e non compaiono
  insieme fra le opzioni; se non si può costruire un insieme con una sola opzione vera, la domanda non si pone. Un rombo
  che dalle misure stampate è un quadrato dà `square`, oppure la domanda non si pone.
- **Mutazioni.** Una risposta alterata per ogni tipo viene trovata dalla verifica indipendente. Una scena costruita con
  un'etichetta fuori dall'immagine o due testi sovrapposti viene rifiutata dalle invarianti.
- **Regressione v3 → v4.** Il vincolo di non sovrapposizione riestrae solo le scene che lo violano, quindi le altre
  specifiche restano identiche. Le domande dei tipi non toccati restano identiche, per tipo e per chiave. Cambiano solo
  le risposte di `intersections` e i testi di `shape_type` e `label_type`.
- **Valori.** `3/2` disegnato come `1.5`: in lettura la risposta è `1.5`, in calcolo entrambe le forme sono
  equivalenti.
- **UniMER.** Una rietichettatura non coerente con l'etichetta attesa dà errore (`stream.apply_relabel`, già presente).
