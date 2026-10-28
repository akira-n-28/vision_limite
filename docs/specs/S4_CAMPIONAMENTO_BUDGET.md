# S4 — Campionamento, stima della qualità e budget degli audit

**Stato:** bozza del 2026-10-28, da approvare. Attua la pipeline v1.4 §§5.4, 5.9, 15.3, 16 (rilievi R8, G6, G7). Le
soglie e i budget numerici sono decisioni dell'utente (D4, D5): qui ci sono la struttura e i valori proposti, segnati
come tali.

## 1. Due campioni separati

| | Diagnostico | Di stima |
| --- | --- | --- |
| Scopo | trovare errori e falsi allarmi dei filtri, progettare riparazioni | stimare la qualità residua della versione ammessa |
| Disegno | stratificato e adattivo: conflitti, incerti, ritagli, figure dense, famiglie nuove, più una quota casuale di ammessi e di scarti automatici | probabilistico sulla versione finale del manifest (S1), con probabilità d'inclusione note |
| Unità | libera | il gruppo (problema, immagine, documento); dentro il gruppo una regola dichiarata per le QA |
| Riuso | si può ripetere a ogni versione | mai sugli esempi usati per progettare le riparazioni; se la versione cambia, si ridisegna |
| Esito | conteggi diagnostici, senza tassi generali | tasso con intervallo, per la popolazione e la versione dichiarate |

## 2. Prima del campione di stima si fissa

1. **Proprietà**: correttezza, osservabilità o spiegazione (S2 §1).
2. **Quantità**, una sola per stima:
   - quota di QA ammesse errate;
   - quota di gruppi con almeno un errore;
   - quota dei target-token di training con errori.
3. **Popolazione e denominatore**: sorgente, compito, versione del manifest; strati e loro pesi.
4. **Incerti**: risoluzione indipendente, oppure limite conservativo che li conta come errori. In ogni caso la quota
   non risolta si riporta e non esce dal denominatore.
5. **Soglia**, se si vuole una certificazione (D4).

Proposta per le prime famiglie: quantità 1, per compito, con unità il gruppo; incerti contati come errori nel limite
conservativo.

## 3. Calcolo

- Limite superiore **unilaterale** al 95%: Clopper-Pearson con `conf = 0.90` bilaterale
  (`stem_audit.clopper_pearson(k, n, 0.90)[1]`; il modulo è congelato, si importa senza modificarlo). Con 0 errori su
  300 dà circa 1.0%, su 3.000 circa 0.1%. Con `conf = 0.95` si otterrebbe l'estremo di un intervallo bilaterale (1.2%
  su 300): non vanno confusi.
- Più QA dello stesso gruppo non sono osservazioni indipendenti: si usa `stem_audit.clustered_rate` (dimensione
  efficace n/deff, caso peggiore con meno di 5 errori), con lo stesso parametro di confidenza.
- Con strati di peso diverso: stima pesata per strato e intervallo per il disegno stratificato. Un risultato per strato
  non si estende alla popolazione, e viceversa.
- La correzione per la sensibilità del revisore (`stem_audit.corrected_upper`) vale solo se la sensibilità è misurata
  su errori reali o su casi difficili, non solo su perturbazioni artificiali (§4).

## 4. Attendibilità del riferimento

- Per una quota del nucleo di riferimento la prima valutazione si fa senza vedere il target candidato; poi confronto e
  risoluzione documentata.
- Calibrazione con errori noti delle sorgenti e casi corretti difficili (coordinate alternative, rotazioni lecite,
  formulazioni equivalenti, ipotesi scartate), oltre alle alterazioni introdotte. Si riportano le classi coperte.
- Secondo verificatore solo su incerti, disaccordi e una quota casuale delle approvazioni. Il 10% è un'ipotesi da
  calibrare, e da solo non misura gli errori che entrambi i revisori non vedono.
- I conflitti importanti senza verità indipendente vanno a revisione competente (l'utente o una persona indicata) oppure
  restano irrisolti. Il voto di maggioranza non decide.
- Se l'accuratezza del riferimento non è sostenibile rispetto alla soglia, il risultato è un audit informativo o una
  stima condizionata al metodo, non una certificazione.

## 5. Budget e contabilità

Per ogni sorgente, prima di qualsiasi chiamata e con approvazione dell'utente (D5):

| Voce | Contenuto |
| --- | --- |
| Tetto | numero di chiamate; token di input testuali e visivi; token di output; modello; costo stimato con la tariffa effettiva |
| Revisione competente | numero di casi e tempo previsti |
| Campioni | dimensione e disegno del diagnostico e di quello di stima |
| Regola d'arresto | al tetto ci si ferma e si riferisce. Si restringe la proprietà, si tengono target controllati o si rinvia il sottoinsieme; non si riduce il campione mantenendo la stessa promessa |
| Registro | una riga nel ledger a ogni giro: chiamate, token, costo, casi aperti |

I controlli del codice (S2) costano zero token ma non zero CPU: si registrano ore CPU e spazio.

## 6. Proposta per le prime famiglie (da approvare)

| Famiglia | Codice sul 100% | Diagnostico | Di stima | Agenti |
| --- | --- | --- | --- | --- |
| Sintetici | invarianti del renderer (S2 §2) | 20 immagini per famiglia × tipo di domanda, guardate | dopo la correzione di `shape_type`, solo se lo stato resta `target_controllato` | nessuno previsto |
| MAVIS | coerenza già fatta; costruibilità di `shape_type` | priorità a `direction`, `count_points`, profondità 3 | per tipo di domanda, unità la figura; dimensione secondo la soglia D4 | solo se la revisione a vista non basta; budget da approvare |
| UniMER | controlli di run4; conflitti in quarantena | incerti e rietichettature, per regola | sul training ammesso, unità la formula | nessuno previsto; un piccolo nucleo con revisione competente |

I numeri della tabella sono proposte di lavoro, non soglie approvate.
