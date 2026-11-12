# S3 — Contratto di risposta, estrazione e verificatori

**Stato:** bozza 3 del 2026-11-12, da approvare. Attua la pipeline v1.5 §§3.3, 5.10.4, 11 (Verificatori) (rilievi R1,
R13, G2). La bozza 2 del 2026-10-28 aveva una proposta senza verifiche; la bozza 3 confronta tre alternative (§2.2-2.3),
fa una proposta per fase (§2.4) e riporta le proprietà meccaniche verificate dalla sonda CPU (§2.5, job 30910,
`scripts/contract_probe.py`, `docs/contratto/sonda-1.md`). **D3 resta aperta** (§2.7): nessun contratto è attivato, e
processor, loader e config operativi restano quelli storici (alternativa A).

## 1. Fatti di partenza

- **Template.** Il template dello snapshot (`chat_template.jinja` @ b1f3d572, sha256 `f6bd2b6b…`) scrive sempre il
  system prompt `You are a helpful assistant.\nPlease reason step by step, and put your final answer within \boxed{}.`
  dopo un `<|endoftext|>` iniziale e rifiuta system prompt diversi, anche quello VL `You are a helpful assistant.`
  (sonda). Il processor lo usa (`processing.py:199-201`), e il tokenizer caricato usa proprio quel file (sonda).
- **Intestazioni** (sonda). Quella canonica è di 27 token, quella con il system VL di 12. Il system VL non è un prefisso
  in token del canonico: diverge al sesto token (`.` nel VL, `.\n` nel canonico).
- **Processor operativo** (sonda). Rifiuta i token speciali nel testo e il prefill nei turni intermedi; scrive
  `reasoning_content` inline come lo genera Violetto (`<think>r</think>`), mentre il template lo scriverebbe con gli a
  capo. Accetta in silenzio combinazioni incoerenti: prefill `<think></think>` più tag nel contenuto, `reasoning_content`
  più `<think>` nel contenuto, `</think>` senza `<think>`, due blocchi `<think>`. Le regole del contratto vanno quindi
  applicate prima del processor (§2.8).
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
  - il timeout del confronto conta come errato, quindi il risultato è un limite inferiore;
  - manda solo il messaggio utente col template del checkpoint (system canonico implicito), con l'istruzione di
    MATH-500 o di AIME nel turno utente; `strip_think` prende il testo dopo l'ultimo `</think>`, o tutto il testo se non
    c'è, e la lente strict vuole un `\boxed{}` fuori da un think aperto (`limite_evals_core/ladder.py`).
- **Estrattore storico** (`f2.grade`): richiede `</think>`, prende l'ultimo `\boxed{}` e tratta come errata ogni
  eccezione di `answers_match` (`f2.py:140-157, 183-193`). Non si usa più per i dati nuovi.
- **Prefill in F1.** `<think></think>` senza label (`F1_PREFILL`) evita che il proiettore impari a saltare il `<think>`
  iniziale, ma non insegna al modello a scegliere una risposta breve.

## 2. Contratto di risposta (D3): alternative, proposta e verifiche

### 2.1 Obiettivo già deciso

Inglese come lingua principale; risposte dirette ai compiti semplici; spiegazioni proporzionate ai problemi complessi;
una baseline matematica confrontabile con Violetto originale (pipeline §§1, 3.3, 9).

### 2.2 Alternative confrontate

| | A, storico | B | C |
| --- | --- | --- | --- |
| System | canonico (template dello snapshot) | VL: `You are a helpful assistant.` | VL |
| Risposta diretta | prefill `<think></think>` senza label, poi risposta ed EOS con label | `<think></think>` generato, poi risposta ed EOS, tutto con label | risposta ed EOS con label, senza tag |
| Risposta ragionata | `<think>…</think>`, finale ed EOS con label (replay e F2 storici) | uguale | uguale |
| In F1 (decoder congelato) | così com'è | **B-F1**: gli stessi token di B, col `<think></think>` come prefill dichiarato senza label | nessun prefill esprime "niente `<think>`": in F1 servirebbe il formato di A o di B-F1 |
| Scelta autonoma della risposta breve | no: la impone il prefill | sì, da imparare col decoder addestrabile | sì, da imparare col decoder addestrabile |

**Baseline matematica M**, separata in tutte le alternative: template canonico, nessun system o quello canonico,
istruzione di MATH-500 o di AIME del valutatore nel turno utente, sempre ragionamento e `\boxed{}`. È anche la modalità
del replay matematico.

**Risposte ragionate nella modalità d'uso finale (B e C).** `<think>` ragionamento `</think>`, poi la parte visibile:
una spiegazione proporzionata e la risposta finale, in `\boxed{}` per le risposte matematiche. Violetto fa già così: dopo
`</think>` scrive una soluzione breve e il `\boxed{}` (job 29400). Il blocco `<think>` non si valuta mai (§3.3). In B una
spiegazione breve senza ragionamento nascosto è `<think></think>` seguito da pochi passaggi e dalla risposta; in C è il
testo senza tag.

### 2.3 Valutazione

- **Formato appreso da Violetto.** Violetto apre sempre con `<think>` (job 29393, 29400).
  - **B** lo conserva in ogni modalità: la scelta fra diretta e ragionata cade sul secondo token (`</think>` o il
    ragionamento).
  - **C** cambia il primo token delle risposte dirette: il modello deve imparare a saltare `<think>` col system VL e a
    non saltarlo col canonico.
  - **A** conserva il formato solo perché il prefill lo impone.
- **Decoder congelato (F1).** La loss di un token di formato si può ridurre solo attraverso il proiettore, che
  imparerebbe a pilotare il formato del decoder.
  - A e B-F1 tolgono i due tag dalla loss. In C il primo token del target porta la decisione "niente `<think>`", e
    nessun prefill la toglie.
  - Il system canonico chiede ragionamento e `\boxed{}`, in contrasto coi target di F1 (trascrizioni, didascalie,
    risposte brevi): col decoder congelato anche questo contrasto può finire nel proiettore. È un'ipotesi da misurare,
    non un fatto.
- **Decoder addestrabile (da F2).** B e C insegnano la scelta senza prefill. A no: le label cominciano dopo il prefill,
  quindi il modello non impara mai a chiudere da solo il `<think>`, e senza prefill ragiona come Violetto. Contraddice il
  criterio di F3: il formato diretto deve funzionare senza un prefill usato solo in training (pipeline §9).
- **Prefill, apprendimento e valutazione.**
  - **Apprendimento.** I token dati come prefill non hanno label: non si impara la scelta che sostituiscono.
  - **Valutazione.** Una valutazione col prefill è condizionata alla forzatura: va dichiarata e non misura la modalità
    d'uso finale. Il comportamento ottenuto col prefill non prova che il modello sappia scegliere lo stile.
  - **Forzatura in B.** La forzatura dichiarata `<think></think>` coincide coi token che il modello genera in B: una
    valutazione forzata, per separare la percezione dalla scelta dello stile, e una libera hanno la stessa sequenza.
  - **Forzatura in C.** Per forzare la diretta bisogna vietare in decodifica il token `<think>`: è un vincolo, non un
    prefill.
- **Tag ed EOS** (verificati, §2.5).
  - **A:** nelle dirette i tag non hanno label, nelle ragionate sì. Nella cronologia la diretta precedente ha i tag con
    label: lo stesso `<think></think>` è senza label nell'ultimo turno e con label in un turno precedente.
  - **B:** tag sempre con label, in ogni turno.
  - **C:** tag solo nelle ragionate.
  - **In tutte:** il `<|endoftext|>` iniziale del template non ha label; l'EOS finale ha label; `<|im_end|>\n` della
    cronologia non ha label; dopo l'ultimo turno non c'è `<|im_end|>`; lo stop è solo `<|endoftext|>`.
- **Costo in token** (sonda).
  - B costa 2 token in più di C per risposta diretta: `<think></think>12` sono 5 token di target contro 3, la
    trascrizione dell'esempio 18 contro 16.
  - L'intestazione VL costa 15 token meno di quella canonica (12 contro 27).
  - Il prefill di A e di B-F1 sono 2 token senza label.
  - Davanti ai 256-576 token reali di un'immagine le differenze sono trascurabili.
- **Cronologia.** In B ogni turno dell'assistente comincia con un token aggiunto (`<think>`): la ritokenizzazione della
  cronologia coincide sempre coi token generati. In C un turno precedente che comincia con un a capo non si conserva,
  perché `\n\n` diventa un token, e il processor lo rifiuta (sonda). È un rischio piccolo, ma meccanico.
- **Effetti per famiglia.**
  - **Replay matematico:** M è invariata in tutte. In B il primo token resta `<think>` anche nelle dirette VL. In C il
    modello deve legare il primo token al system: va misurato il rischio di risposte senza ragionamento nella modalità
    canonica.
  - **Trascrizione e didascalie:** il target è lo stesso, dopo `</think>` in B; per una didascalia di PixMo (circa 250
    token) 2 token in più non contano.
  - **Scelta multipla:** la lettera viene dopo `</think>` in B e da sola in C; l'estrattore delle opzioni legge il finale
    in entrambi i casi.

### 2.4 Proposta (non approvata)

| Fase o uso | Proposta |
| --- | --- |
| F1, decoder congelato | B-F1: system VL; `<think></think>` come prefill dichiarato e registrato, senza label; target ed EOS con label. Gli stessi token di B. Le valutazioni percettive di F1 usano lo stesso prefill dichiarato e non misurano la scelta dello stile |
| Da F2, decoder addestrabile, modalità `auto` (uso finale) | B: system VL; dirette `<think></think>` più risposta, ragionate `<think>…</think>` più spiegazione e risposta, tutto con label; nessun prefill nel training e nelle valutazioni d'uso finale |
| Modalità `math`: baseline e replay | M: template canonico, nessun system o quello canonico, istruzione del valutatore nel turno utente; sempre ragionamento e `\boxed{}` |
| Template | `limite-vl/1`: quello dello snapshot con un system iniziale canonico o VL, come il template della sonda. Senza system rende il canonico identico allo snapshot, quindi il valutatore ufficiale, che manda solo il messaggio utente, resta confrontabile con Violetto |
| Cronologia | i turni precedenti come generati, senza EOS, chiusi da `<|im_end|>\n` senza label; label sul contenuto solo dove il turno è un target dichiarato |
| A | solo per riprodurre i run storici |

**Perché B.**
- Conserva il primo token di Violetto in ogni modalità.
- Ha gli stessi token in F1 e nelle fasi successive: cambiano solo le label di due token.
- Rende la scelta di stile un solo token misurabile.
- Permette una forzatura dichiarata identica alla sequenza libera.
- Non ha il rischio di confine della cronologia.

**Il costo** è di 2 token per risposta diretta.

**C resta un'alternativa valida per l'uso finale.** Il formato è più corto e l'output diretto non ha tag, ma in F1 non
ha un formato coerente e chiede al modello di cambiare il primo token appreso.

### 2.5 Proprietà verificate dalla sonda (job 30910, `docs/contratto/sonda-1.md`)

329 controlli su 329, col tokenizer e il template dello snapshot e sei esempi scritti a mano: trascrizione LaTeX,
didascalia breve, risposta numerica, scelta multipla, problema con spiegazione (solo testo), dialogo con un turno
precedente. Tutti hanno un segnaposto visivo di 64 token, tranne il problema.
- **Esiti attesi espliciti.** Per ogni caso e alternativa, il testo dell'esempio, il prompt d'inferenza e gli intervalli
  con label sono uguali a stringhe scritte nella sonda, cioè i pezzi del formato concatenati senza il template. I tag
  sono 151665 e 151666 e l'EOS 151643.
- **Parità del prefisso** fra training e inferenza, nello stesso contratto, a ogni turno.
- **Duplicazioni.** Nessun tag o terminatore duplicato: un solo EOS in apertura e uno in chiusura, `<|im_end|>` e
  `<|im_start|>` mai con label.
- **Label.** Il prefill è senza label, il contenuto e l'EOS hanno label, il turno precedente ha label e `<|im_end|>\n` no.
- **Cronologia.** I turni precedenti si ritokenizzano coi token del testo generato.
- **Confronti fra alternative:**
  - B e B-F1 hanno gli stessi input, con label diverse solo sui due tag;
  - C è uguale a B tolti i due tag del target;
  - A e B-F1 sono uguali dopo l'intestazione;
  - nelle risposte ragionate B e C coincidono.
- **Baseline canonica conservata.** Il template della sonda, senza system o col system canonico, dà testo, token e label
  identici allo snapshot. Il prompt di M è uguale alla resa del valutatore (solo il messaggio utente) e l'istruzione di
  MATH-500 di `replay.py` è quella di `suites.py`.
- **Rifiuti** (42 combinazioni):
  - system non ammessi o fuori posto;
  - prefill e tag duplicati;
  - prefill nella modalità d'uso finale, in training e nel prompt;
  - think vuoto in C o in A con label;
  - diretta senza `<think>` in B;
  - M senza ragionamento o senza `\boxed{}`;
  - B-F1 ragionato o a più turni;
  - tag malformati in ogni alternativa;
  - token speciali nel testo e prefill nei turni intermedi, che già il processor rifiuta.
- **Fatti per C3:** le combinazioni incoerenti accettate dal processor operativo (§1) e il confine della cronologia di
  C (§2.3).
- **Controlli dei controlli.** Le verifiche si accorgono di un carattere in più nell'atteso e di una label in più.

### 2.6 Cosa la sonda non prova

- Quale alternativa il modello seguirà meglio, né se Violetto si comporta bene col system VL. Servirà un esperimento GPU
  autorizzato: per esempio la verifica già prevista, Violetto col system VL senza prefill su un piccolo pannello
  (punto 1 del piano F0).
- La sonda usa pochi esempi scritti a mano e un segnaposto visivo di 64 token.
- I token generati davvero da un modello possono avere una segmentazione diversa da quella del tokenizer.
- Il template della sonda e le regole del contratto esistono solo nella sonda. La baseline canonica è verificata solo
  sul dominio condiviso: un system canonico in una posizione successiva lo snapshot lo accetta, la sonda lo rifiuta.

### 2.7 Decisioni da sottoporre all'utente

1. **Forma della risposta diretta nell'uso finale:** B, con `<think></think>` generato e con label (raccomandata), o C,
   senza tag.
2. **F1:** system VL con prefill dichiarato, cioè i token di B (B-F1, raccomandata), o il canonico storico (A). In
   entrambi i casi, prima del primo pilota conviene la verifica GPU di Violetto col system VL.

Restano aperte anche, ma non sono parte di D3: istruzioni per più immagini oltre a "Image 1", "Image 2"; politica della
cronologia nei dialoghi lunghi, cioè se togliere il ragionamento dei turni precedenti; quote delle risposte dirette e
ragionate nel mix.

### 2.8 Contratto `vl-contract/1` (proposta, con B)

| Voce | Proposta |
| --- | --- |
| Template | `limite-vl/1`, versionato con sha256 nel manifest: lo snapshot più un system iniziale canonico o VL; senza system, il canonico identico ai token di Violetto |
| Modalità `auto` (uso finale) | system VL; il modello sceglie fra diretta (lettura, identificazione, calcolo immediato) e ragionata (problemi) |
| Modalità `math` | system canonico; sempre ragionamento e `\boxed{}`: baseline di Violetto e replay matematico |
| Risposta diretta | `<think></think>` con label, poi la risposta, poi `<|endoftext|>` |
| Risposta ragionata | `<think>` ragionamento `</think>`, spiegazione proporzionata e risposta finale; la risposta matematica in `\boxed{}` |
| Prefill | solo in F1 (decoder congelato) e solo `<think></think>` davanti a una diretta, dichiarato nel record; mai nelle valutazioni della modalità d'uso finale |
| Label | tutto il turno dell'assistente, compresi `<think>`, `</think>` e `<|endoftext|>`; in F1 il prefill senza label; `<|im_end|>\n` della cronologia senza label |
| Regole prima del processor | un solo blocco `<think>` in apertura, chiuso; niente tag nel testo dei campi del record; niente prefill con tag nel contenuto; system coerente con la modalità |
| Stop | `<|endoftext|>` (151643) |
| Immagini | all'inizio del turno utente; con più immagini il testo le nomina in ordine ("Image 1", "Image 2"); interleaving generale rimandato |

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

La completezza non decide la correttezza del contenuto. Una generazione troncata può contenere un `<think>` chiuso e
una risposta finale completa, e poi continuare fino al limite dei token: quel finale si valuta come gli altri. Solo il
`<think>` aperto non ha finale (§3.1).

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

### 3.4 Conseguenze della proposta per C4 (da implementare dopo la decisione)

- **Dirette e ragionate.**
  - **Con B** c'è una sola forma conforme in ogni modalità, `think_completo`: think vuoto = diretta, non vuoto =
    ragionata. `answer_only` è una violazione di formato in `auto` come in `math`. La forma si legge dagli id 151665 e
    151666, sempre presenti in un turno conforme.
  - **Con C** `answer_only` è la diretta conforme in `auto` e una violazione in `math`: per dire se una risposta senza
    tag è conforme servono la modalità e il compito.
  - In entrambe un think vuoto in `math` è formato non conforme, con la correttezza del finale valutata a parte.
- **Formato, completezza e correttezza** restano separati (§§3.1-3.2, 5). La proposta cambia solo che cosa vuol dire
  `conforme`, per contratto e modalità, che l'estrattore riceve esplicitamente.
- **Prefill.** Il testo generato dopo un prefill non contiene i token del prefill. L'estrattore riceve il prefill
  registrato nel record e ricostruisce il turno: `extract(prefill + testo, terminazione, contratto, compito)`.
  - **Senza il prefill**, una diretta di F1 (`12`) sembrerebbe `answer_only`.
  - **Una valutazione col prefill** porta nel rapporto il prefill dichiarato e non conta come valutazione d'uso finale.
- **Terminazione.** I metadati sono quelli di §3.2. `generate_batch` deve restituire anche `max_new_tokens` e il motivo
  d'arresto. `generation_config.json` si ferma solo su 151643, quindi un `<|im_end|>` generato non è un terminatore: con
  `eos = false` la completezza è `troncata` o `sconosciuta`, mai `completa`.
- **Valutatore ufficiale.**
  - **In `math` è invariato:** stessi token del prompt, lenti, tassonomia e politica di troncamento.
  - **Con B** il suo `strip_think` restituisce il finale (`<think></think>12` dà `12`); con C restituisce tutto il
    testo.
  - **Usi separati.** Resta lo strumento della matematica testuale in `math`. L'estrattore del contratto serve ai compiti
    VL, alla selezione SFT e al RL; non si scarta C perché `f2.grade` richiede `</think>`.

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

## 5. Esiti separati, punteggio e politica per uso

- **Formato:** `conforme` (forma prevista dal contratto, per esempio `think_completo` in `math` e in `auto`),
  `answer_only`, `think_aperto`, `malformato`.
- **Completezza:** `completa`, `troncata`, `sconosciuta` (§3.2).
- **Correttezza:**
  - `corretto`, `errato`;
  - `non_determinato`: parsing fallito, timeout, forma non supportata, candidati incompatibili;
  - `assente`: nessun finale, cioè `think_aperto` o `malformato`, oppure finale senza candidati;
  - `errore_tecnico`: eccezione del codice o dell'infrastruttura.
- **Punteggio**, per politica di troncamento, come nel valutatore ufficiale:
  - `score`: 1 se la correttezza è `corretto`, qualunque sia la completezza;
  - `fail`: 1 solo se `corretto` e `completa`.

  Il punteggio non cambia la correttezza. Una risposta corretta ma troncata resta `corretto`: vale 0 solo nel punteggio
  `fail`, non diventa `errato`.

| Uso | Regola |
| --- | --- |
| **SFT** (selezione di testi generati come target) | solo `corretto` + `conforme` + `completa`, e le proprietà di S2. Tutto il resto non si seleziona e si conta per motivo. I target costruiti da noi non passano per la selezione, ma devono superare il giro di prova: `extract(target)` restituisce il gold |
| **Valutazione** | denominatore pieno, nessun completamento escluso. Metrica principale: il punteggio medio con la politica dichiarata, `fail` o `score`; le due politiche non si mescolano. `errato`, `assente` e `non_determinato` valgono 0 (limite inferiore, come i timeout del valutatore ufficiale). Si riportano sempre a parte: quote di formato, troncamento (e i troncati corretti), `non_determinato` e `sconosciuta`. Con completezza `sconosciuta` il risultato è condizionato e non decide confronti fra checkpoint |
| **RL** (futuro) | politica `fail`: premio di correttezza 1 solo con `corretto` + `conforme` + `completa`; 0 con `errato` o `assente`. Un troncato corretto vale 0 ma resta registrato come `corretto`. `answer_only` dove il contratto chiede `<think>` è una violazione di formato con regola deterministica, senza premio di correttezza (pipeline §11). `troncata`: premio 0 e politica di troncamento come parametro dichiarato da calibrare, nessuna penalità di lunghezza generale. `non_determinato`: nessun segnale di premio, campione mascherato e monitorato; se la quota dipende dalle risposte del modello si corregge il verificatore o si sospende la famiglia. `errore_tecnico`: al più 2 tentativi, poi esclusione contata; lotto sospeso se persiste. `sconosciuta` non è ammessa: i rollout hanno sempre i metadati |

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
- **Troncato con finale completo.** `<think>…</think>\boxed{5}` corretto, poi testo fino al limite dei token
  (`eos = false`, `new_tokens = max_new_tokens`): correttezza `corretto`, completezza `troncata`, punteggio `fail` 0,
  punteggio `score` 1. La stessa risposta con `eos = true` vale 1 con entrambe le politiche.
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
  - Tokenizzazione delle due modalità: label, stop, system prompt canonico identico ai token di Violetto. Le proprietà
    meccaniche sono già verificate dalla sonda (§2.5) per le alternative; in C3 i suoi controlli diventano test del
    contratto adottato.
  - Il giro di prova `extract(render(target)) = gold` riesce per ogni compito di S2.

## 7. Piano minimo di C3, dopo la decisione D3

1. **Contratto.** `src/limite_vl/contract.py` contiene l'oggetto `vl-contract/1`: nome e versione; modalità `auto` e
   `math` con `system_prompt_id`; regola della risposta diretta; prefill ammessi per fase; label; stop. Contiene anche le
   regole della sonda:
   - composizione del turno da (ragionamento, risposta);
   - controllo del system;
   - controllo del turno e del prompt.
2. **Template.** `limite-vl/1` è un file versionato nel repository, col testo del template della sonda e lo sha256
   registrato. Non si scrive nello snapshot né nel tokenizer salvato.
3. **Processor.** `LimiteVLProcessor` riceve il contratto in modo esplicito (template, modalità, fase) e lo applica prima
   del rendering. Scrive contratto, modalità, system e prefill in `Example`. Senza contratto resta il comportamento
   storico (A), solo per riprodurre i run vecchi.
4. **Loader.** Restituiscono ragionamento, risposta e metadati del compito; tag e prefill li compone il contratto.
   `F1_PREFILL` resta la costante storica. I target non cambiano: nessuna migrazione in C3.
5. **Test CPU.** I controlli della sonda diventano test in `slurm/tests_cpu.sbatch`: identità col template dello
   snapshot, rifiuti, parità del prefisso e cronologia. Restano le regressioni dei reader (`tests/test_reader_manifest.py`)
   e il collaudo C7 con le config esistenti.
6. **Record e manifest.** Si aggiungono i campi contratto, modalità, `system_prompt_id` e prefill (S1 §2). `variant_id`
   e i digest dei token vengono dopo, in C5.
7. **Verifica GPU, dopo l'ok dell'utente.** Violetto col system VL, senza prefill nascosti, su un piccolo pannello
   (punto 1 del piano F0).

Fuori da C3: estrattore generale e verificatori (C4), reward RL, `variant_id`, migrazione dei target.
