# Definizioni per la politica F2 (dal pilota f2-pilot/2)

Documento per l'utente, non per gli agenti: nessuno script lo passa a un agente. Traduce le definizioni dell'utente del
2026-10-22 nei controlli del pilota (`src/limite_vl/f2_claims.py`: `CATEGORIES`, `fact_category`; `scripts/f2_pilot.py`)
e propone come portarle in `config/audit/f2_policy.json`. Quel file resta invariato finché l'utente non decide.

## Definizioni dell'utente (2026-10-22)

1. Un sistema di assi ruotato, traslato o riflesso non è un errore se rappresenta soltanto una scelta di coordinate
   coerente con il problema.
2. Una configurazione geometrica realmente falsa è un errore, indipendentemente dall'orientamento del disegno.
3. Le affermazioni sull'orientamento assoluto si valutano separatamente, quando sono presentate come proprietà visibili
   della figura.
4. Un'ipotesi esplicitamente scartata non è un errore visivo da contare. Se è usata nella deduzione finale, va valutata
   come possibile errore di ragionamento.
5. `frame_counts_as_error` deve riguardare solo contraddizioni effettive della figura, non cambi di coordinate leciti.

## Come le applica il pilota

L'agente di classificazione (`config/audit/f2_classify.md`, solo testo) dà per ogni fatto:
- `stance`: `asserted`, `hypothesis` (non risolta) o `rejected` (scartata esplicitamente);
- `basis`: `given` (dalla domanda), `figure` (presentato come visibile), `chosen` (coordinate o orientamento scelti per
  calcolare, o una posizione che segue solo da quella scelta), `computed` (dedotto);
- `used_in_answer`, solo per ipotesi e scarti.

I verificatori (`config/audit/f2_claim_verify2.md`, immagine e affermazione, mai la soluzione) danno due giudizi separati:
- `config`: la geometria a meno di rotazioni, traslazioni e riflessioni dell'intera figura (valori, lati, vertici, da che
  parte di una retta sta una forma, dentro/fuori, angoli retti; per le coordinate: esiste un movimento rigido o una
  riflessione della figura che le rende vere). Classi: errore (`false`, `not_visible`), giusto (`true`, `deduction`),
  incerto;
- `page`: le direzioni nella pagina (orizzontale, verticale, sinistra, destra, sopra, sotto, orario).

Categoria di ogni occorrenza di un fatto (`fact_category`):

| Categoria | Quando | Conta come errore visivo |
|---|---|---|
| `visual_error` | `config` errore e `stance` asserted (anche un riferimento scelto che contraddice la configurazione) | sì (definizioni 2 e 5) |
| `orientation_error` | `config` giusto, `page` contraddetta, `basis` given o figure, asserito | conteggio separato (definizione 3) |
| `frame_change` | `config` giusto, `page` contraddetta, `basis` chosen o computed | mai (definizione 1) |
| `discarded_hypothesis` | `config` errore, `stance` rejected, non usata per la risposta | no (definizione 4) |
| `reasoning_flag` | `config` errore, ipotesi o scarto usato per la risposta | va alla revisione del ragionamento (definizione 4) |
| `open_hypothesis` | `config` errore, ipotesi né scartata né usata | conteggio separato, scelta aperta |
| `code_match` | solo misure uguali alla didascalia, non viste dagli agenti (scorciatoia) | no, se il campione di controllo regge; non prova il ragionamento |
| `ok` | configurazione giusta e nessuna direzione che conti contraddetta | no |
| `quarantine` | nessuna maggioranza su `config` (o su `page`, se conta) | incerto: nel certificato pesa come errore finché non risolto |

Verifiche del metodo legate alle definizioni:
- controlli sugli assi per i verificatori (`frame_control`): coordinate lecite girate o specchiate (attesa: giusto) e
  coordinate rotte con lunghezze scambiate o ordine dei vertici rotto (attesa: errore), entrambe con verità certa;
- le alterazioni di calibrazione con verità certa (misura, costruzione, configurazione) stanno solo in frasi asserite
  (`asserted_at`): niente ipotesi, condizioni, domande, negazioni, casi, frasi corrette subito dopo, citazioni del prompt;
- l'orientamento (scambio di parole di direzione) ha verità non garantita: categoria a parte, mai usata per dire
  certificabile una sensibilità.

## Proposta di traduzione in `f2_policy.json` (da decidere; oggi nulla è cambiato)

- `frame_counts_as_error`: con la definizione 5 il campo non può più voler dire "un orientamento assunto che contraddice
  il disegno esclude la soluzione" (la lettura letterale degli audit v1, scritta oggi nelle opzioni). Un riferimento
  scelto che contraddice la configurazione è già `visual_error`; un cambio di coordinate lecito è `frame_change` e non
  conta per costruzione. Proposta: sostituire il campo con le scelte esplicite qui sotto.
- Nuovo `count_categories`: `["visual_error"]` per i tassi visivi certificati (`thresholds.solution_visual`,
  `thresholds.claim_visual`, che restano null).
- Nuovo `orientation_error`: `"separate"` (tasso riportato, con una soglia propria da decidere o nessuna) oppure
  `"count"`.
- Nuovo `reasoning_flag`: `"separate"`, verso la revisione del ragionamento (`thresholds.reasoning_invalid`).
- Nuovo `open_hypothesis`: scelta aperta fra `"count"` (prudente) e `"separate"`.
- Nuovo `discarded_hypothesis`: `"ignore"` (definizione 4).
- Nuovo `code_match`: ammesso solo se il disaccordo del campione di controllo della scorciatoia resta sotto una soglia (il
  pilota propone 2%); altrimenti quei fatti tornano agli agenti.
- `calibration.required_kinds`: da `["measure", "layout"]` a `["measure", "construction", "configuration"]` più i
  controlli sugli assi; `layout`/orientamento resta calibrato solo a titolo informativo.
- Copertura: il certificato dovrebbe riportare la copertura stimata del generatore di candidati (frasi scartate con un
  fatto asserito, dal campione alla cieca) e trattarla come limite: ciò che non viene estratto non è corretto per
  assenza di segnalazioni.
