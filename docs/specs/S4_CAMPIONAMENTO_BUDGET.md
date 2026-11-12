# S4 — Campionamento, stima della qualità e budget degli audit

**Stato:** bozza 2 del 2026-10-28, rivista dopo la revisione esterna dei push fino a `e81f884`; da approvare. Attua la
pipeline v1.4 §§5.4, 5.9, 15.3, 16 (rilievi R8, G6, G7). Soglie e budget numerici sono decisioni dell'utente (D4, D5):
qui ci sono il disegno e i valori proposti, segnati come tali.

Il 2026-11-05 l'utente ha adottato il nucleo tecnico: limiti esatti (§3), compatibilità fra disegno e quantità stimata,
verifiche di copertura (§8). È implementato e collaudato in C8 (§9), non ancora applicato a esiti reali. Soglie di
ammissione, numerosità e budget degli audit proposti (§§6-7) restano decisioni aperte.

## 1. Principi

- Una stima vale solo per la quantità, la popolazione e il disegno dichiarati **prima** dell'estrazione.
- Un metodo non si adotta perché esiste già nella vecchia pipeline. `stem_audit.clustered_rate` e
  `stem_audit.corrected_upper` non si usano per le stime nuove (§4).
- Dove manca una garanzia di copertura, il risultato si dichiara **esplorativo** (conteggi diagnostici, nessun tasso
  generale) o **condizionato** (vale sotto un'assunzione nominata, per esempio la sensibilità del revisore).
- La certificazione non blocca F0. Per "pronto per un esperimento" (pipeline §15.3) basta un audit informativo con le
  proprietà non verificate dichiarate. La certificazione rispetto a una soglia serve solo dove l'utente la chiede.

## 2. Due campioni separati

| | Diagnostico | Di stima |
| --- | --- | --- |
| Scopo | trovare errori e falsi allarmi dei filtri, progettare riparazioni | stimare la qualità residua della versione ammessa |
| Disegno | stratificato e adattivo: conflitti, incerti, ritagli, figure dense, famiglie nuove, più una quota casuale di ammessi e di scarti automatici | probabilistico, fissato prima (§3) |
| Riuso | si può ripetere a ogni versione | mai sugli esempi usati per progettare le riparazioni; se la versione cambia, si ridisegna |
| Esito | conteggi, sempre esplorativi | limite superiore con copertura dichiarata |

## 3. Disegno del primo nucleo

Ogni affermazione dichiara: proprietà (correttezza, osservabilità o spiegazione, S2 §1), quantità e popolazione
(sorgente × compito × versione del manifest, S1). L'unità di campionamento è l'unità della quantità:

| Quantità | Estrazione | Limite superiore unilaterale al 95% |
| --- | --- | --- |
| quota di QA errate | QA (`uid`) estratte senza reinserimento con probabilità uguali n/N | esatto ipergeometrico: D_U/N (definizione e casi limite sotto) |
| quota di gruppi con almeno un errore | gruppi senza reinserimento con probabilità uguali; ogni gruppo estratto si controlla per intero | ipergeometrico sui gruppi |
| quota dei target-token errati | QA con probabilità proporzionale ai target-token, con reinserimento | Clopper-Pearson binomiale esatto: ogni estrazione è errata con probabilità pari alla quota pesata |

- **Limite ipergeometrico.** Con N unità nella popolazione, n estratte e k errate:
  - D_U = max{D intero con k ≤ D ≤ N − n + k : P(X ≤ k | N, D, n) > α}, con α = 0.05; il limite è D_U/N. È il più
    grande numero di errori nella popolazione non escluso dal campione. P(X ≤ k) non cresce con D, quindi basta una
    ricerca monotona.
  - k = n (tutti gli estratti errati): P(X ≤ n) = 1 per ogni D, quindi la ricerca non ha un primo D escluso. Il limite
    si definisce esplicitamente uguale a 1 (D_U = N), senza ricerca.
  - k = 0: stessa definizione, con P(X = 0) = C(N − D, n)/C(N, n).
  - n = N (censimento): X = D, quindi D_U = k e il limite è esatto, k/N; con k = n = N vale 1.
  - Input non validi (N, n, k non interi, n < 1, n > N, k < 0, k > n, α fuori da (0, 1)) danno errore, mai un
    valore corretto in silenzio.
  - Clopper-Pearson binomiale con le stesse regole: k = n dà 1; k = 0 dà 1 − α^(1/n); n ≥ 1 e 0 ≤ k ≤ n.
- **Copertura.** Il numero di errori nel campione ha una distribuzione nota, ipergeometrica o binomiale, qualunque sia
  la correlazione fra le QA dello stesso gruppo. Per questo il limite ha copertura garantita per la sua quantità, e non
  si usano dimensioni efficaci stimate.
- **Registrazione.** Il file del campione registra seme, N, n, probabilità d'inclusione e sha256 del manifest.
- **Numerosità fissata prima.** Nessuna estensione adattiva del campione di stima: cambierebbe la distribuzione su cui
  si basa il limite.
- **Ordini di grandezza.** Con 0 errori, n = 300 dà circa l'1.0% (1 − 0.05^(1/300) per N grande, un po' meno con N
  finito); n = 3.000 dà circa lo 0.1%. Il limite bilaterale al 95% (1.2% su 300) è un'altra cosa.
- **Incerti.** Incerti e casi non giudicabili contano come errori nel limite. La quota non risolta si riporta sempre.
- **Strati.** Nel primo nucleo non si combinano: un'affermazione per strato. Un limite per l'unione degli strati è la
  somma dei limiti, ciascuno al livello 0.05/S (Bonferroni): valido, ma prudente.
- **Più affermazioni.** Ciascuna ha il suo 95%. Un'affermazione congiunta su più quantità richiede la correzione.

## 4. Metodi storici non adottati

### `clustered_rate`

Controesempio, verificato il 2026-10-28 eseguendo `stem_audit.clustered_rate` e `clopper_pearson`:

- popolazione di 1.000 gruppi: 999 con una QA corretta e uno con 1.000 QA errate, quindi il 50.03% delle QA è errato;
- estraendo 300 gruppi senza reinserimento, il gruppo errato manca con probabilità 0.700;
- in quei campioni `clustered_rate` restituisce 0 errori su 300, deff 1 ("caso peggiore") e l'intervallo (0, 1.22%);
  il limite unilaterale al 95% è 0.99%;
- la copertura effettiva del limite per la quota di QA errate è quindi 0.30, non 0.95.

Con un campione di gruppi, la quota di QA dipende dalle dimensioni dei gruppi non campionati. Una dimensione efficace
n/deff stimata dal campione non può garantire la copertura, soprattutto con code pesanti o pochi errori. Con lo stesso
budget, 300 QA estratte uniformemente fra le 1.999 sono tutte corrette con probabilità 1.2·10⁻¹⁰².

### `corrected_upper`

Divide il limite superiore del tasso osservato per il limite inferiore della sensibilità del revisore. Ha due problemi:

1. **Copertura congiunta.** Se ciascun limite vale al 95%, il quoziente vale almeno al 90%, non al 95%. Per
   un'affermazione congiunta al 95% ciascun limite va preso al 97.5% (Bonferroni), oppure serve un metodo esatto per
   la coppia. L'incertezza della calibrazione entra quindi nel risultato, non si somma a parte.
2. **Rappresentatività.** La sensibilità misurata su alterazioni introdotte, o su un altro tipo di errori, non è quella
   sugli errori reali della popolazione. Senza errori reali noti e rappresentativi la correzione poggia su
   un'assunzione non verificabile, quindi il risultato è condizionato.

Proposta per il primo nucleo: gli errori si confermano con evidenza indipendente (codice, sorgente originale,
revisione competente) prima del conteggio, così i falsi positivi non entrano. Il risultato si dichiara **condizionato
alla sensibilità del revisore, non misurata**. La correzione per la sensibilità si usa solo con una calibrazione su
errori reali noti della sorgente, al livello congiunto.

## 5. Attendibilità del riferimento

- Per una quota del nucleo di riferimento la prima valutazione si fa senza vedere il target candidato; seguono il
  confronto e la risoluzione documentata.
- Calibrazione con errori noti delle sorgenti e casi corretti difficili (coordinate alternative, rotazioni lecite,
  formulazioni equivalenti, ipotesi scartate), oltre alle alterazioni introdotte. Si riportano le classi coperte.
- Secondo verificatore solo su incerti, disaccordi e una quota casuale delle approvazioni. Il 10% è un'ipotesi da
  calibrare, e da solo non misura gli errori che entrambi i revisori non vedono.
- I conflitti importanti senza verità indipendente vanno a revisione competente, oppure restano irrisolti. Il voto di
  maggioranza non decide.

## 6. Budget e contabilità

Per ogni sorgente, prima di qualsiasi chiamata e con approvazione dell'utente (D5):

| Voce | Contenuto |
| --- | --- |
| Tetto | numero di chiamate; token di input testuali e visivi; token di output; modello; costo stimato con la tariffa effettiva |
| Revisione competente | numero di casi e tempo previsti |
| Campioni | dimensione e disegno del diagnostico e di quello di stima |
| Regola d'arresto | al tetto ci si ferma e si riferisce. Si restringe la proprietà, si tengono target controllati o si rinvia il sottoinsieme; non si riduce il campione mantenendo la stessa promessa |
| Registro | una riga nel ledger a ogni giro: chiamate, token, costo, casi aperti |

I controlli del codice costano zero token ma non zero CPU: si registrano ore CPU e spazio.

## 7. Proposta per le prime famiglie (da approvare)

| Famiglia | Correttezza | Osservabilità | Agenti |
| --- | --- | --- | --- |
| Sintetici v4 | verificata per record dal codice, con la verifica indipendente di S2 §2.3: nessuna stima | invarianti del renderer sul 100%; diagnostico di 20 immagini per tipo di domanda, guardate | nessuno |
| MAVIS | rispetto alla specifica, per record (coerenza con `diagram_generation`); rispetto all'immagine, nella stima a destra | stima di correttezza e osservabilità sull'immagine: 300 QA ammesse estratte uniformemente, limite ipergeometrico, risultato condizionato al revisore; diagnostico con priorità a `direction`, `count_points` e profondità 3 | solo se la revisione a vista non basta, con budget da approvare |
| UniMER | stima sulle formule ammesse (unità: la formula), 300 estratte uniformemente; incerti di run4 in quarantena fuori dalla popolazione, con la loro quota dichiarata | nella stessa revisione | nessuno; un piccolo nucleo con revisione competente |

I numeri della tabella sono proposte di lavoro, non soglie approvate.

Per i sintetici v4 il piano concreto del 2026-10-30 è in S2 §2.8, ed è anch'esso da approvare. Rispetto alla riga della
tabella aggiunge un campione casuale informativo (75 QA per famiglia, 50 del test, 30 coppie), oltre ai casi mirati,
perché "pronto per un esperimento" chiede un audit campionario (pipeline §15.3; §1). Il limite ipergeometrico è in C8
(§9), collaudato su dati fittizi e non ancora applicato agli esiti di quella revisione.

## 8. Collaudi (CPU, `tests_cpu`)

- **Copertura esatta sul controesempio.** Il campione di gruppi con `clustered_rate` copre nel 30% dei casi. Il
  disegno a QA uniformi con limite ipergeometrico copre almeno al 95%: calcolo esatto con la distribuzione
  ipergeometrica, senza simulazione.
- **Limite ipergeometrico.**
  - Su popolazioni piccole coincide con l'enumerazione completa di tutti i D.
  - Per N → ∞ tende a Clopper-Pearson.
  - k = n dà 1 senza ricerca; k = 0 con n = N dà 0; n = N dà k/N.
  - N = n = 1 e k ∈ {0, 1} danno 0 e 1.
  - Gli input non validi sollevano un errore.
- **Estrazione.** Le probabilità d'inclusione dichiarate (uniforme, proporzionale ai token) coincidono con le frequenze
  su molte estrazioni con seme.
- **Affermazioni non ammesse.** Una richiesta di limite sulla quota di QA a partire da un campione di gruppi viene
  rifiutata: il codice controlla che unità e quantità coincidano.
- **Bonferroni.** Due strati al livello 0.025 ciascuno danno un limite per l'unione con copertura congiunta di almeno
  0.95, su popolazioni costruite apposta.

Eseguiti per C8 tutti tranne quello dell'estrazione, perché C8 non contiene campionatori (§9). Il campionatore dei
sintetici (`scripts/synth_gold_sample.py`) è collaudato solo per l'ordine riproducibile e per il prefisso
(`tests/test_synth_gold.py`).

## 9. Nucleo implementato (C8, 2026-11-05)

Il nucleo tecnico adottato dall'utente il 2026-11-05 è in `src/limite_vl/estimate.py`, che usa solo la libreria
standard. Lo stato va letto su tre livelli separati:
- **Implementazione:** fatta, nel commit di C8.
- **Collaudo:** `tests/test_estimate.py` su dati fittizi, 44 controlli su 44 (job 30616 e 30617).
- **Applicazione agli esiti reali:** non fatta. La revisione di `synth-v4-campione-2` è in corso, e i suoi esiti
  parziali non si usano per scegliere soglie, corrispondenze degli esiti o disegno.

**Funzioni.**

| Funzione | Uso |
| --- | --- |
| `hypergeometric_upper(N, n, k, alpha)` | D_U della §3, con interi esatti; il limite della quota è D_U / N |
| `clopper_pearson_upper(n, k, alpha)` | limite binomiale esatto per estrazioni con reinserimento, arrotondato per eccesso |
| `bonferroni(alpha, m)` | livello alpha / m di ciascuna di m affermazioni simultanee |
| `Claim`, `Design`, `check_compatible` | affermazione e disegno dello strato, con il rifiuto se non sono compatibili |
| `tally`, `Counts`, `describe` | esiti per estrazione, contati senza perdite; conteggi descrittivi |
| `estimate`, `estimate_simultaneous` | limite per uno strato; più affermazioni con Bonferroni, ciascuna col suo risultato |

Alpha vale 1/20 se non è indicato, ed è tenuto come frazione esatta.

**Quantità ammesse,** ciascuna con la sua unità estratta:

| Quantità | Unità estratta | Estrazione | Limite |
| --- | --- | --- | --- |
| `quota_qa_errate` | QA | uniforme senza reinserimento | ipergeometrico |
| `quota_gruppi_con_errore` | gruppo, controllato per intero | uniforme senza reinserimento | ipergeometrico |
| `quota_coppie_con_errore` | coppia, giudicata per intero | uniforme senza reinserimento | ipergeometrico |
| `quota_target_token_errati` | QA | con reinserimento, probabilità proporzionale ai target-token | Clopper-Pearson |

Con n = N il limite ipergeometrico è un censimento e dà la quota esatta.

**Esiti,** uno per estrazione:

| Esito | Significato | Nel limite |
| --- | --- | --- |
| `approvato` | corretto per la proprietà, con primo giudizio indipendente (§5) | approvato |
| `approvato_non_indipendente` | corretto, ma il primo giudizio non era indipendente dal target | errore |
| `errato` | errore confermato | errore |
| `incerto` | il revisore non decide | errore |
| `non_giudicabile` | il caso non permette un giudizio | errore |
| `mancante` | estratto e non giudicato | errore |

Nessun esito si toglie. Tutti compaiono nei conteggi, e la quota non risolta (incerti, non giudicabili, mancanti) si
riporta sempre. Contare come errori gli esiti non approvati lascia valido il limite e lo allarga.

**Il calcolo si rifiuta** (eccezione `Refused`) nei casi seguenti:
- casi mirati, che non stimano un tasso: restano solo i conteggi (`describe`);
- unità estratta diversa da quella della quantità, per esempio la quota di QA da un campione di gruppi o di coppie
  (§4);
- estrazione diversa da quella della quantità;
- popolazione, strato o N diversi fra affermazione e disegno;
- numero di esiti diverso da n, perché la numerosità non si estende né si riduce dopo l'estrazione;
- unità estratta senza esito (va segnata `mancante`), esito di un'unità non estratta, esito ripetuto o sconosciuto,
  unità ripetuta in un'estrazione senza reinserimento;
- la stessa affermazione due volte nella stessa famiglia.

Gli input numerici non validi danno `ValueError` (§3).

**Risultato,** in tre parti separate:
1. `conteggi`, descrittivi: valgono per le unità guardate, non per la popolazione.
2. `limite`, valido per il disegno dichiarato: con probabilità almeno 1 − alpha sull'estrazione, la quota della
   popolazione che la revisione non approverebbe non supera il limite.
3. `conclusione`, condizionata all'attendibilità del revisore: lo stesso numero limita la quota di unità errate solo se
   il revisore non approva unità errate. La sua sensibilità non è misurata e non si corregge (§4).

`certificazione` vale sempre "nessuna": nessun record diventa gold e nessuna soglia si applica.

**Esempio minimo** con dati fittizi, ripreso da `test_example` in `tests/test_estimate.py`:

```python
from limite_vl import estimate as est

pop = "esempio-fittizio"
qa = est.Design(pop, "A", "qa", est.UNIFORM, 10_000, 300)
coppie = est.Design(pop, "B", "coppia", est.UNIFORM, 500, 30)
voci = [(est.Claim("correttezza", "quota_qa_errate", pop, "A", 10_000), qa,
         est.Counts(approvato=294, errato=2, incerto=1, non_giudicabile=1, mancante=1, approvato_non_indipendente=1)),
        (est.Claim("correttezza", "quota_coppie_con_errore", pop, "B", 500), coppie, est.Counts(approvato=30))]
uno = est.estimate(*voci[0])               # un'affermazione al 95%
insieme = est.estimate_simultaneous(voci)  # due affermazioni simultanee, ciascuna a 1/40
```

| Strato | Esiti | k nel limite | Limite al 95% | Con Bonferroni (1/40) |
| --- | --- | --- | --- | --- |
| A: 300 QA su 10.000 | 294 approvati, 2 errati, 1 incerto, 1 non giudicabile, 1 mancante, 1 approvazione non indipendente | 6 | 388 su 10.000 (3,88%) | 426 (4,26%) |
| B: 30 coppie su 500 | 30 approvate | 0 | 46 su 500 (9,2%) | 56 (11,2%) |

- Nello strato A gli errori confermati sono 2, ma il limite usa k = 6; la quota non risolta è 3/300.
- La conclusione condizionata di A: se il revisore non approva QA errate, la quota di QA errate per la correttezza è al
  più il 3,88%, al 95%.
- Una stima della quota di QA dal disegno B è rifiutata: le coppie non stimano la quota di QA.

**Collaudi** (job 30616 e 30617; 2 CPU, 2-3 s di tempo reale e 4-6 s di CPU allocate ciascuno):
- **Enumerazione esatta:** per N ≤ 9 P(X ≤ k) e D_U coincidono con l'enumerazione di tutti i sottoinsiemi, in 630 casi
  con alpha 1/20, 1/5 e 1/2.
- **Copertura esatta:** almeno 1 − alpha per ogni D con N ≤ 12. Il minimo è 1 − alpha esatto, 0,95 (0,80 con alpha
  1/5), dove P(X ≤ k) vale proprio alpha: per esempio N = 6, n = 3, D = 3, con P(X = 0) = 1/20. Con la disuguaglianza
  stretta della §3 quel D è escluso, e la copertura resta al livello nominale, non sotto.
- **Casi limite:** k = n, censimento, N = n = 1, k = 0 contro la scansione di tutti i D, e i valori della S2 §2.8
  (3,91%, 5,77%, 9,00%); il limite cresce con k e cala con n.
- **Convergenza a Clopper-Pearson:** scarto relativo 1,6·10⁻³, 2,1·10⁻⁴ e 1,8·10⁻⁵ per N = 10⁵, 10⁶ e 10⁷.
- **Clopper-Pearson:**
  - con k = 0 coincide con 1 − alpha^(1/n): 0,99% su 300;
  - la probabilità esatta al limite è ≤ alpha, e > alpha poco sotto;
  - coincide con `stem_audit.clopper_pearson` (storico) entro 1,2·10⁻¹⁵;
  - la copertura esatta è ≥ 0,95 su una griglia di p.
- **Controesempio della §4,** con calcolo esatto:
  - col campione di gruppi `clustered_rate` copre nel 30% dei casi, e C8 rifiuta la stessa stima;
  - 300 QA uniformi coprono al 95,55%, e sono tutte corrette con probabilità 1,24·10⁻¹⁰²;
  - dallo stesso campione di gruppi si stima la quota di gruppi con errore: limite 0,8%, copertura 1.
- **Bonferroni,** su due strati costruiti apposta (N = 400 con n = 40, N = 300 con n = 30), con campioni indipendenti:
  - senza correzione la copertura simultanea scende a 0,903 per i D peggiori;
  - con alpha/2 per strato resta almeno 0,9507 per ogni coppia di D;
  - il limite dell'unione copre a 0,9988.
- **Il resto:** input non validi, rifiuti per compatibilità e per esiti, tre parti del risultato, censimento,
  Clopper-Pearson col disegno dei token.

Niente GPU, download o token.

**Limiti rimasti:**
- C8 non contiene campionatori, quindi il collaudo dell'estrazione della §8 non è fatto.
- Non c'è correzione per la sensibilità del revisore né calibrazione: le conclusioni restano condizionate.
- Non c'è un limite automatico per l'unione di strati. È la somma dei D_U ciascuno a alpha/S, da calcolare a parte se
  serve.
- La corrispondenza fra i valori del pacchetto di revisione (`revisioni/synth-v4-campione-2`) e gli esiti di C8 non è
  implementata. Va fissata con l'utente prima di guardare gli esiti. Proposta:
  - `si` → `approvato`, oppure `approvato_non_indipendente` se il primo giudizio non è indipendente;
  - `no` → `errato` se confermato con evidenza indipendente (§4), altrimenti `incerto`;
  - `incerto` e `non_verificabile` → `incerto`; `non_giudicabile` → `non_giudicabile`; `non_compilato` → `mancante`;
  - `non_applicabile` → rifiuto per quella proprietà;
  - la gravità `minore` non è un errore e si conta a parte (S2 §2.8);
  - una coppia è approvata solo se le due varianti e il contratto lo sono.
- Per applicare C8 al campione dei sintetici, `Design` e `Claim` si costruiscono dal suo `report.json`: N, n, unità,
  sha256 del manifest come popolazione, un'affermazione per proprietà e strato.
- **Nota del 2026-11-12.** Gli esiti della revisione dei sintetici sono consegnati e integrati (S2 §2.13). La
  corrispondenza qui sopra non era stata approvata prima della revisione e resta una proposta. Nel lotto
  dell'integrazione ci sono solo conteggi descrittivi. Una stima chiede che l'utente approvi corrispondenza e
  affermazioni sapendo che gli esiti sono noti, e resta condizionata alla sensibilità del revisore.
