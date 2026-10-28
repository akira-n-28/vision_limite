# S3 — Contratto di risposta, estrazione e verificatori

**Stato:** bozza 2 del 2026-10-28, rivista dopo la revisione esterna dei push fino a `e81f884`; da approvare. Attua la
pipeline v1.4 §§3.3, 5.10.4, 11 (Verificatori) (rilievi R1, R13, G2). Le voci segnate **D3** sono la decisione aperta
del contratto: qui ci sono una proposta e le alternative.

## 1. Fatti di partenza

- **Template.** Il template dello snapshot (`chat_template.jinja` @ b1f3d572) scrive sempre il system prompt
  `You are a helpful assistant.\nPlease reason step by step, and put your final answer within \boxed{}.` e rifiuta
  system prompt diversi. Il processor lo usa (`processing.py:197-199`).
- **Formato di Violetto.** Genera `<think>...</think>` senza a capo, poi la risposta, poi `<|endoftext|>` (job 29393,
  29400). `generation_config.json` dello snapshot ha `eos_token_id = 151643` soltanto: `generate` si ferma solo su
  `<|endoftext|>`.
- **Metadati di terminazione.** `generation.generate_batch` toglie EOS dal testo e restituisce a parte
  `{text, eos, new_tokens}` (`generation.py:56-60`), senza `max_new_tokens`. Le generazioni con vLLM registrano `finish_reason`
  (`scripts/f2_backfill.py:63`). Dal solo testo non si può sapere se una risposta è stata troncata.
- **Valutatore ufficiale** (`third_party/limite-violetto/limite-evals` @ 7ddad34f, `SCORER_VERSION
  2026.10-limite-evals-v1`):
  - tre lenti, strict, lenient e permissive;
  - una tassonomia dei completamenti, casi A-T (`limite_evals/taxonomy.py`);
  - `think_unclosed` e `format_not_evaluable`, vero solo per un `<think>` aperto e troncato;
  - una politica di troncamento dichiarata, `score` o `fail`: mai mescolata, troncati sempre nel denominatore, tasso
    di troncamento sempre riportato;
  - il timeout del confronto conta come errato, quindi il risultato è un limite inferiore.
- **Estrattore storico** (`f2.grade`): richiede `</think>`, prende l'ultimo `\boxed{}` e tratta come errata ogni
  eccezione di `answers_match` (`f2.py:140-157, 183-193`). Non si usa più per i dati nuovi.
- **Prefill in F1.** `<think></think>` senza label (`F1_PREFILL`) evita che il proiettore impari a saltare il `<think>`
  iniziale, ma non insegna al modello a scegliere una risposta breve.

## 2. Contratto `vl-contract/1` (proposta, D3)

| Voce | Proposta | Alternative |
| --- | --- | --- |
| Template | copia versionata del template di Violetto (`limite-vl/1`, sha256 nel manifest) con due system prompt ammessi: quello **canonico**, identico ai token di Violetto, e uno **VL** neutro (`You are a helpful assistant.`) | (a) solo il canonico, con risposte dirette in contrasto col system prompt; (b) solo il VL, senza baseline compatibile |
| Modalità `auto` (uso finale) | system VL; il modello sceglie fra risposta diretta (lettura, identificazione, calcolo immediato) e ragionata (problemi) | — |
| Modalità `math` | system canonico; sempre ragionamento e `\boxed{}`: baseline di Violetto e mantenimento del testo matematico | — |
| Risposta diretta | `<think></think>` vuoto con label, poi la risposta breve, poi `<|endoftext|>` | senza `<think>`: formato nuovo per il decoder |
| Risposta ragionata | `<think>` ragionamento `</think>` risposta; la risposta matematica in `\boxed{}` | — |
| Prefill | solo in F1 (decoder congelato), dichiarato nel record; mai nelle valutazioni della modalità d'uso finale | nessun prefill anche in F1 |
| Label | tutto il turno dell'assistente, compresi `<think>`, `</think>` e `<|endoftext|>`; in F1 il prefill senza label | — |
| Stop | `<|endoftext|>` (151643) | — |
| Immagini | all'inizio del turno utente; con più immagini il testo le nomina in ordine ("Image 1", "Image 2") | interleaving generale, rimandato |

Il contratto è un oggetto con nome e versione. Il processor lo riceve esplicitamente e lo scrive in ogni `Example`. Il
record (S1 §2: contratto, modalità, `system_prompt_id`, prefill) e il manifest del run lo registrano.

## 3. Estrazione

`extract(testo, terminazione, contratto, compito)` restituisce tre esiti separati (§5) più il finale e le risposte
candidate. `terminazione` è obbligatoria, perché il troncamento non si deduce dalla stringa:

```
terminazione = {eos: bool | None, finish_reason: "stop" | "length" | None, new_tokens: int | None, max_new_tokens: int | None}
```

### 3.1 Forma

Si legge dagli id dei token `<think>` (151665) e `</think>` (151666) generati. Se gli id non sono disponibili, si usano le stringhe decodificate (`generation.py` decodifica con `skip_special_tokens=False`) e lo si registra.

| Forma | Riconoscimento | Finale |
| --- | --- | --- |
| `answer_only` | nessun `<think>` e nessun `</think>` | tutto il testo |
| `think_completo` | un `<think>` in apertura, a meno di spazi, e un `</think>` dopo | il testo dopo `</think>` |
| `think_aperto` | `<think>` senza `</think>` | **nessuno**: un ragionamento incompleto non è una risposta finale, anche se contiene un numero o un `\boxed{}` (casi B, D, L della tassonomia ufficiale) |
| `malformato` | `</think>` senza `<think>`, più blocchi, testo prima di `<think>`, ordine invertito | nessuno, solo diagnostico |

### 3.2 Completezza

Si legge dai metadati, mai dal testo.

| Completezza | Condizione |
| --- | --- |
| `completa` | `eos = true` oppure `finish_reason = "stop"` |
| `troncata` | `finish_reason = "length"`, oppure `eos = false` con `new_tokens = max_new_tokens` |
| `sconosciuta` | metadati assenti (generazioni storiche, output esterni) o incoerenti (`eos = false` con `new_tokens < max_new_tokens`); si registra come tale, non si sceglie una delle due |

`think_aperto` con `completa` vuol dire che il modello ha chiuso la generazione senza rispondere: è un errore del
modello. `think_aperto` con `troncata` vuol dire che il budget è finito mentre ragionava: il formato non è valutabile,
come `format_not_evaluable` del valutatore ufficiale.

### 3.3 Risposte candidate nel finale

| Modalità | Candidati |
| --- | --- |
| `math` | i `\boxed{...}` bilanciati del finale |
| `auto`, compito con risposta breve | il finale ripulito, oppure i `\boxed{}` se presenti; per le scelte multiple, la lettera o il testo dell'opzione |
| `auto`, compito ragionato | i `\boxed{}` e le frasi di risposta dichiarate ("the answer is …", "Answer: …") |

- Candidati equivalenti fra loro secondo il verificatore del compito danno una sola risposta: una formulazione ripetuta
  o equivalente non è una contraddizione.
- Candidati non equivalenti danno `non_determinato`.
- Il ragionamento dentro `<think>` non si valuta mai, e le ipotesi abbandonate lì non si penalizzano. Nel finale un
  valore discusso e scartato in prosa non è un candidato: lo sono solo `\boxed{}` e le frasi dichiarate.

**Differenza voluta dal valutatore ufficiale.** Con più box in disaccordo nel finale (casi M e N) la lente strict
prende l'ultimo, mentre qui l'esito è `non_determinato`: la pipeline (§5.10.4) chiede il controllo delle risposte finali
incompatibili. Per la matematica testuale i numeri da riportare restano quelli del valutatore ufficiale, con la
politica di troncamento dichiarata; l'estrattore di questo progetto serve per i compiti VL, la selezione SFT e il RL.
Le divergenze sui casi della tassonomia si contano (§6).

## 4. Verificatori per compito

| Tipo | Regola | Riusa |
| --- | --- | --- |
| Numerico | equivalenza con `math_verify` nel dominio del compito; tolleranza dichiarata per compito, coerente col valore visualizzato; unità; `1.5` ≡ `3/2` | `f2.answers_match`, senza convertire gli errori in falso |
| Espressione | equivalenza simbolica con `math_verify`; insiemi di soluzioni e condizioni | idem |
| Trascrizione | uguaglianza dei token LaTeX a meno degli spazi; nessuna equivalenza matematica (`x+x` ≠ `2x`) | `latex.same_latex`, `latex.latex_tokens` |
| Scelta multipla | lettera, oppure testo dell'opzione mappato sulla lettera; indipendente dall'ordine delle opzioni | `stem_audit.mc_options`, `option_text` (solo lettura: modulo congelato) |
| Categoria breve | stringa normalizzata contro l'insieme delle risposte ammesse (alias, classe più specifica: S2 §2.2); nomi di lati e segmenti senza ordine (AB = BA) | — |
| Sì/no | `yes`/`no` normalizzati | — |
| Descrizione strutturata | precisione e richiamo per voce | `f1_schema.geometry_scores` |
| Insufficienza dell'input | ammessa solo se l'insufficienza è stabilita nel record (S2) | — |

Ogni esito porta nome e versione del verificatore.

## 5. Tre esiti separati e politica per uso

- **Formato:** `conforme` (forma prevista dal contratto, per esempio `think_completo` in `math` e in `auto`),
  `answer_only`, `think_aperto`, `malformato`.
- **Completezza:** `completa`, `troncata`, `sconosciuta` (§3.2).
- **Correttezza:**
  - `corretto`, `errato`;
  - `non_determinato`: parsing fallito, timeout, forma non supportata, candidati incompatibili;
  - `assente`: nessun finale, cioè `think_aperto` o `malformato`, oppure finale senza candidati;
  - `errore_tecnico`: eccezione del codice o dell'infrastruttura.

| Uso | Regola |
| --- | --- |
| **SFT** (selezione di testi generati come target) | solo `corretto` + `conforme` + `completa`, e le proprietà di S2. Tutto il resto non si seleziona e si conta per motivo. I target costruiti da noi non passano per la selezione, ma devono superare il giro di prova: `extract(target)` restituisce il gold |
| **Valutazione** | denominatore pieno, nessun completamento escluso. Metrica principale: `corretto` sul totale; `errato`, `assente` e `non_determinato` contano come non corretti (limite inferiore, come i timeout del valutatore ufficiale). Si riportano sempre a parte: quote di formato, troncamento, `non_determinato` e `sconosciuta`. La politica di troncamento (`fail` qui, perché un troncato non ha finale) è dichiarata e non si mescola. Con completezza `sconosciuta` il risultato è condizionato e non decide confronti fra checkpoint |
| **RL** (futuro) | premio di correttezza 1 solo con `corretto` + `conforme` + `completa`; 0 con `errato` o `assente`. `answer_only` dove il contratto chiede `<think>` è una violazione di formato con regola deterministica, senza premio di correttezza (pipeline §11). `troncata`: premio 0 e politica di troncamento come parametro dichiarato da calibrare, nessuna penalità di lunghezza generale. `non_determinato`: nessun segnale di premio, campione mascherato e monitorato; se la quota dipende dalle risposte del modello si corregge il verificatore o si sospende la famiglia. `errore_tecnico`: al più 2 tentativi, poi esclusione contata; lotto sospeso se persiste. `sconosciuta` non è ammessa: i rollout hanno sempre i metadati |

Una risposta corretta non certifica la spiegazione: `corretto` riguarda solo il finale.

## 6. Collaudi (CPU, `tests_cpu`)

- **Le quattro forme**, ciascuna con terminazione `completa` e `troncata`:
  - `answer_only` corretto: correttezza `corretto`, formato `answer_only`;
  - `think_completo` corretto: `corretto`, `conforme`;
  - `think_aperto` con un `\boxed{}` giusto dentro il ragionamento: `assente`, mai `corretto`;
  - `malformato` (due `</think>`): `assente`.
- **Terminazione.**
  - Lo stesso testo con `eos = true` e con `eos = false, new_tokens = max_new_tokens` dà `completa` e `troncata`.
  - Senza metadati dà `sconosciuta`.
  - Un testo che finisce con `<|im_end|>` senza EOS si classifica dai metadati, non dalla stringa.
- **Candidati.**
  - Finale con `x = 3, so \boxed{3}`: un solo candidato, `corretto`.
  - Finale con `\boxed{3}` e `\boxed{5}`: `non_determinato`.
  - `\boxed{3}` scartato nel ragionamento e `\boxed{5}` giusto nel finale: `corretto`.
  - "It is not 4; the answer is 5": il 4 non è un candidato.
- **Verificatori.**
  - `1.5` contro `3/2`: numerico `corretto`, trascrizione `errato`.
  - `AB` contro `BA` per un lato: `corretto`.
  - Timeout del parser: `non_determinato`, mai `errato`.
- **Tassonomia ufficiale.** Un esempio per ciascuno dei casi A-T di `limite_evals/taxonomy.py`, con l'esito atteso
  documentato e le divergenze dalla lente strict elencate (M, N).
- **Contratto.**
  - Tokenizzazione delle due modalità: label, stop, system prompt canonico identico ai token di Violetto.
  - Il giro di prova `extract(render(target)) = gold` riesce per ogni compito di S2.
