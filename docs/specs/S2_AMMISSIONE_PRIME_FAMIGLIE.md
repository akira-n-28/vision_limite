# S2 — Regole di ammissione delle prime famiglie

**Stato:** bozza del 2026-10-28, da approvare. Attua la pipeline v1.4 §§5.1, 5.6, 5.7, 5.10.2 (rilievi G3, G4, R6). Le
famiglie sono le tre del nucleo F1 (§7): sintetici propri, MAVIS-Geometry e UniMER. Le altre sorgenti avranno regole
proprie quando entrano.

## 1. Regola comune

Ogni compito dichiara le proprietà necessarie e, per ciascuna, il livello di evidenza:

- **R**: verifica sul singolo record (controllo, versione, esito salvati nel record, S1 §2);
- **I**: invariante della famiglia (generatore o renderer, versione, assunzioni scritte, precondizioni controllate
  sull'istanza);
- **C**: solo controllo campionario (qualità stimata, S4).

Le proprietà sono sempre tre: **corretto** (il riferimento), **osservabile** (l'informazione è leggibile nell'immagine
che il modello vede, dopo il preprocessing) e **spiegazione** (non applicabile ai compiti answer-only). Lo stato gold
(pipeline §5.6) si assegna così:

- `gold_verificato`: tutte le proprietà necessarie sono coperte da R o I;
- `target_controllato`: almeno una proprietà necessaria è coperta solo da C;
- `incerto`: controlli discordi o inconcludenti; va in quarantena.

Si registrano separatamente il valore **esatto**, quello **visualizzato** e la **risposta richiesta**. Un compito di
lettura chiede il valore visualizzato; un compito di calcolo chiede quello esatto, solo se il problema basta a
ricostruirlo.

## 2. Sintetici (`data/shards/synth/v3`, `src/limite_vl/synth.py`)

| Compito | Corretto | Osservabile | Stato atteso |
| --- | --- | --- | --- |
| `qa` (coord, polygon, function, shapes) | I: la risposta viene dalla specifica (`synth.questions`) | I da costruire: niente clipping, niente occlusioni delle etichette, testo leggibile al budget di patch; più C sul render | `gold_verificato` dopo i collaudi del renderer, altrimenti `target_controllato` |
| `pair` (controfattuali) | I: `synth.counterfactual` cambia un solo dettaglio | come sopra, per entrambe le immagini | idem; lo stesso gruppo per `-a` e `-b` |

Problemi da correggere prima dell'ammissione:

- **`shape_type` dei quadrilateri** (`synth.py:294-296`): la domanda non mostra le opzioni (`f1c.QA_ONE`) e la risposta è
  solo `spec["kind"]`, ma un quadrato è anche rettangolo e parallelogramma, e un rettangolo è un parallelogramma.
  Proposta: domanda con le opzioni e la richiesta della classe più specifica ("Which is the most specific type of
  quadrilateral ABCD among: square, rectangle, parallelogram, trapezoid?"). Il target resta `kind`; per la valutazione
  si registra anche l'insieme delle risposte vere.
- **Triangoli**: il generatore esclude l'isoscele rettangolo e l'isoscele quasi equilatero, e le terne pitagoriche
  danno solo triangoli rettangoli scaleni (`synth.py:191-205`); lo scaleno non ha domanda. Per costruzione la risposta
  è unica: lo conferma un collaudo che verifica sulle specifiche che una sola classe sia vera.
- **Invarianti del renderer** (nuove, controllate sul 100% dal codice): bounding box di punti ed etichette dentro
  l'immagine; distanza minima fra le etichette; altezza del testo in pixel dopo il ridimensionamento non sotto una
  soglia da fissare (≥ 1 token di 32 px per carattere è un'ipotesi).
- Le domande sulle coordinate esistono solo se ci sono le tacche (già nel codice): va controllato con un test.

## 3. MAVIS-Geometry (`data/shards/mavis-geometry/9c80fff3-caption`, `f1_mavis.py`, `f1_schema.py`)

| Compito | Corretto | Osservabile | Stato atteso |
| --- | --- | --- | --- |
| `describe` (`f1_schema.render_geometry`) | R: verità da `diagram_generation`, confermata dalla didascalia (`mavis_gen.compare`, 0 segnalazioni su 199.050, job 30025, 30052) | C: misure e nomi disegnati in `caption.jpg` (40 immagini guardate, `outputs/f1_mavis/contact_run1`) | `target_controllato` finché l'osservabilità resta solo C |
| `qa` (`f1_mavis.mavis_questions`) | R, come sopra | C, per tipo di domanda | idem |

La coerenza fra didascalia e `diagram_generation` prova che due descrizioni concordano, non che i pixel le rispettino
(`mavis_gen.py`). Per passare a `gold_verificato` serve un'invariante del renderer di MAVIS, che oggi non abbiamo. In
alternativa, un campione stratificato per profondità, forma e tipo di domanda (S4) dà una qualità stimata.

Problemi da correggere:

- **`shape_type`** (`f1_mavis.py:83`): "What type of shape is ABCD?" con risposta il nome del generatore (square,
  rectangle, parallelogram, rhombus, kite, varianti di triangolo). Quadrato ⊂ rettangolo, rombo ⊂ parallelogramma,
  equilatero ⊂ isoscele. Stessa correzione dei sintetici: opzioni e classe più specifica. Inoltre si controlla che la
  classe si possa ricavare dal disegno e dalle misure stampate; dove non si può, la domanda non si costruisce.
- **Valori**: `side_len` e `angle_val` sono letture, quindi chiedono il valore visualizzato (decimali a due cifre per i
  razionali, job 30049, 30052). Il valore esatto (`44/3`) resta nel record per i compiti di calcolo futuri.
- **`direction`** (dentro/fuori) e **`count_points`**: richiedono che triangolo, semicerchio e tutte le etichette siano
  visibili. Vanno nel campione con la massima priorità.
- **Risoluzione**: i diagrammi sono a 2304 patch, oltre le 1024 di addestramento di NaFlex. Se in F0 cambia la
  risoluzione, l'osservabilità va riverificata per la nuova configurazione (pipeline §5.7).

## 4. UniMER (`data/shards/unimer/2343ddd9-dense`, `latex.py`, `outputs/unimer_check/run4`)

| Compito | Corretto | Osservabile | Stato atteso |
| --- | --- | --- | --- |
| `transcribe` (formula intera) | C sulla sorgente (arXiv, Pix2tex); R per struttura (`latex.structure_problems`, `unrendered_commands`) e confronto col disegno (`latex_render`), che trova candidati e non è un oracolo | C: simboli piccoli, indici, ritaglio | `target_controllato`; `gold_verificato` solo per un piccolo nucleo con revisione competente |
| `transcribe_line` (riga ritagliata, `data.unimer_line_sample`) | come sopra, sulla riga | R: la fascia non contiene pezzi delle righe vicine (già nel codice); C sul ritaglio | idem |

Esiti già disponibili (run4, job 30047, su 985.450 campioni): 51 errori certi (esclusi), 1.119 incerti (quarantena; 1.116
in 402 gruppi in conflitto), 18.827 rietichettature tracciate (trasformazioni con genitore e regola). Il target è la
formula senza gli spazi del formato im2latex (`latex.despace`): l'equivalenza ammessa è l'uguaglianza dei token LaTeX
a meno degli spazi (`latex.same_latex`), non l'equivalenza matematica (`x+x` non vale `2x`). Le sovrapposizioni col test
seguono S1 §5. UniMER-Test (spe, cpe, sce, hwe) resta solo valutazione.

## 5. Collaudi (CPU, `tests_cpu`)

- Domanda ambigua: quadrato con le opzioni square/rectangle/parallelogram; la domanda nuova chiede la classe più
  specifica, e la valutazione accetta solo `square`.
- Triangoli: su 10.000 specifiche, una sola classe vera per ogni domanda `shape_type`.
- MAVIS: un rombo che è anche un quadrato dalle misure stampate dà risposta `square`, oppure non genera la domanda.
- Valore visualizzato contro esatto: `3/2` disegnato come `1.5` (lettura: `1.5`; calcolo: entrambi equivalenti).
- Invarianti del renderer: una scena costruita con un'etichetta fuori dall'immagine viene rifiutata (test di mutazione).
- UniMER: una rietichettatura non coerente con l'etichetta attesa dà errore (`stream.apply_relabel`, già presente).
