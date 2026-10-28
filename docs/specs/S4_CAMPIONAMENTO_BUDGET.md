# S4 — Campionamento, stima della qualità e budget degli audit

**Stato:** bozza 2 del 2026-10-28, rivista dopo la revisione esterna dei push fino a `e81f884`; da approvare. Attua la
pipeline v1.4 §§5.4, 5.9, 15.3, 16 (rilievi R8, G6, G7). Soglie e budget numerici sono decisioni dell'utente (D4, D5):
qui ci sono il disegno e i valori proposti, segnati come tali.

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
| quota di QA errate | QA (`uid`) estratte senza reinserimento con probabilità uguali n/N | esatto ipergeometrico: D/N, con D il più piccolo numero di QA errate per cui P(X ≤ k \| N, D, n) ≤ 0.05 |
| quota di gruppi con almeno un errore | gruppi senza reinserimento con probabilità uguali; ogni gruppo estratto si controlla per intero | ipergeometrico sui gruppi |
| quota dei target-token errati | QA con probabilità proporzionale ai target-token, con reinserimento | Clopper-Pearson binomiale esatto: ogni estrazione è errata con probabilità pari alla quota pesata |

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

## 8. Collaudi (CPU, `tests_cpu`)

- **Copertura esatta sul controesempio.** Il campione di gruppi con `clustered_rate` copre nel 30% dei casi. Il
  disegno a QA uniformi con limite ipergeometrico copre almeno al 95%: calcolo esatto con la distribuzione
  ipergeometrica, senza simulazione.
- **Limite ipergeometrico.** Su popolazioni piccole coincide con l'enumerazione completa, e per N → ∞ tende a
  Clopper-Pearson.
- **Estrazione.** Le probabilità d'inclusione dichiarate (uniforme, proporzionale ai token) coincidono con le frequenze
  su molte estrazioni con seme.
- **Affermazioni non ammesse.** Una richiesta di limite sulla quota di QA a partire da un campione di gruppi viene
  rifiutata: il codice controlla che unità e quantità coincidano.
- **Bonferroni.** Due strati al livello 0.025 ciascuno danno un limite per l'unione con copertura congiunta di almeno
  0.95, su popolazioni costruite apposta.
