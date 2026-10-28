# S3 — Contratto di risposta, estrazione e verificatori

**Stato:** bozza del 2026-10-28, da approvare. Attua la pipeline v1.4 §§3.3, 5.10.4, 11 (Verificatori) (rilievi R1, R13,
G2). Le scelte segnate **D3** sono la decisione aperta del contratto: qui c'è una proposta con le alternative.

## 1. Fatti di partenza

- Il template dello snapshot (`chat_template.jinja` @ b1f3d572) scrive sempre il system prompt
  `You are a helpful assistant.\nPlease reason step by step, and put your final answer within \boxed{}.` e rifiuta
  system prompt diversi; il processor lo usa (`processing.py:197-199`).
- Violetto genera `<think>...</think>` senza a capo, poi la risposta, poi `<|endoftext|>` (job 29393, 29400).
- In F1 il prefill `<think></think>` senza label (`F1_PREFILL`) evita che il proiettore impari a saltare il `<think>`
  iniziale, ma non insegna al modello a scegliere una risposta breve.
- Il vecchio estrattore (`f2.grade`) richiede `</think>`, prende l'ultimo `\boxed{}` e tratta come errata ogni
  eccezione di `answers_match` (`f2.py:140-157, 183-193`).

## 2. Contratto `vl-contract/1` (proposta, D3)

| Voce | Proposta | Alternative |
| --- | --- | --- |
| Template | Copia versionata del template di Violetto (`limite-vl/1`) nel repository, con due system prompt ammessi: quello **canonico** (identico ai token di Violetto) e uno **VL** neutro (`You are a helpful assistant.`) | (a) solo il canonico per tutto, con risposte dirette in contrasto con il system prompt; (b) solo il VL neutro, perdendo la compatibilità con la baseline |
| Modalità `auto` (uso finale) | system VL; il modello sceglie: risposta diretta per lettura, identificazione e calcolo immediato, ragionamento per i problemi | — |
| Modalità `math` | system canonico; sempre ragionamento e `\boxed{}`: per la baseline di Violetto e per il mantenimento del testo matematico | — |
| Forma della risposta diretta | `<think></think>` vuoto con label, poi la risposta breve, poi `<|endoftext|>` | senza `<think>`: formato nuovo per il decoder e un estrattore con due casi |
| Forma della risposta ragionata | `<think>` ragionamento `</think>` risposta; la risposta matematica finale in `\boxed{}` | — |
| Prefill | ammesso solo in F1 (decoder congelato) e dichiarato nel record; mai nelle valutazioni della modalità d'uso finale | nessun prefill nemmeno in F1 |
| Label | tutto il turno dell'assistente, compresi `<think>`, `</think>` e `<|endoftext|>`; in F1 il prefill senza label | — |
| Stop | `<|endoftext|>` (151643) | — |
| Immagini | all'inizio del turno utente; con più immagini il testo le nomina in ordine ("Image 1", "Image 2"), come stabilito dal punto 4 di F0 | interleaving generale, rimandato |

Il contratto è un oggetto con nome e versione. Il processor lo riceve esplicitamente, lo scrive in ogni `Example`, nel
record (S1) e nel manifest del run. L'estrattore rifiuta un testo prodotto con un contratto diverso.

## 3. Estrazione

`extract(testo, contratto, compito) -> {forma_ok, modalità, finale, risposte_finali, ragionamento, troncato}`

- Il **finale** è il testo dopo l'ultimo `</think>`. Se `</think>` manca, tutto il testo è il finale (recupero
  answer-only) e `forma_ok = false`: la correttezza si giudica comunque, la forma si registra a parte.
- **Risposte finali**: tutte le risposte candidate nel finale (`\boxed{}`, "the answer is X", lettera di scelta, riga
  breve). Se sono incompatibili fra loro, l'esito è `non_determinato` e non c'è premio di correttezza. I `\boxed{}`
  dentro il ragionamento sono tentativi e non si penalizzano.
- Testo troncato (niente `<|endoftext|>`, oppure `finish_reason != stop`): `troncato = true`. La politica per fase è
  nella §5.

## 4. Verificatori per compito

| Tipo | Regola | Riusa |
| --- | --- | --- |
| Numerico | equivalenza con `math_verify` nel dominio del compito, tolleranza dichiarata per compito (assoluta o relativa, coerente con il valore visualizzato) e unità; `1.5` ≡ `3/2` | `f2.answers_match` (senza la conversione degli errori in falso) |
| Espressione | equivalenza simbolica con `math_verify`; insiemi di soluzioni e condizioni | idem |
| Trascrizione | uguaglianza dei token LaTeX a meno degli spazi; nessuna equivalenza matematica (`x+x` ≠ `2x`) | `latex.same_latex`, `latex.latex_tokens` |
| Scelta multipla | lettera, oppure testo dell'opzione mappato sulla lettera; indipendente dall'ordine delle opzioni | `stem_audit.mc_options`, `option_text` (solo lettura: modulo congelato) |
| Categoria breve | stringa normalizzata contro l'insieme delle risposte ammesse (alias, "classe più specifica", S2) | — |
| Sì/no | `yes`/`no` normalizzati | — |
| Descrizione strutturata | precisione e richiamo per voce | `f1_schema.geometry_scores` |
| Insufficienza dell'input | ammessa solo se l'insufficienza è stabilita nel record (S2) | — |

Ogni esito porta nome e versione del verificatore.

## 5. Esiti e politica

| Esito | Quando | SFT (selezione di soluzioni) | Valutazione | RL |
| --- | --- | --- | --- | --- |
| `corretto` | il verificatore conferma | ammissibile se anche le altre proprietà lo sono | conta come corretto | premio 1 |
| `errato` | il verificatore smentisce | scartato | errato | premio 0 |
| `non_determinato` | parsing fallito, timeout, forma non supportata, finali incompatibili | scartato | riportato a parte, nel denominatore | nessun premio; famiglia sospesa se la quota dipende dal modello |
| `errore_tecnico` | eccezione del codice o dell'infrastruttura | ritentato (al più 2 volte), poi escluso e contato | escluso e contato | retry limitati, poi lotto sospeso |

Una risposta corretta non certifica la spiegazione: `corretto` riguarda solo il finale.

## 6. Collaudi (CPU, `tests_cpu`)

- Risposta diretta corretta con `<think></think>` vuoto e senza `\boxed{}`.
- Finale con due risposte incompatibili: `non_determinato`.
- Tentativo abbandonato nel ragionamento (`\boxed{3}`, poi correzione) e finale `\boxed{5}` corretto: `corretto`.
- `1.5` contro `3/2`: numerico `corretto`; trascrizione `errato`.
- Classificazione geometrica ambigua: con la richiesta della classe più specifica solo `square` è corretto.
- Testo senza `</think>` ma con la risposta giusta: `corretto`, `forma_ok = false`.
- Timeout del parser: `non_determinato`, mai `errato`.
- Tokenizzazione delle due modalità: label, stop, system prompt canonico identico ai token di Violetto.
