# Piano di implementazione F0

Stato al 2026-10-28, dopo la migrazione alla [pipeline v1.4](../PIPELINE_ADDESTRAMENTO_LIMITE_VL.md) (§§3, 5, 6, 17).
Il confronto è fra il codice del repository sul cluster (identico a `9d977b6` per codice e test: la migrazione ha
spostato solo config, launcher e documenti) e i requisiti della pipeline. **Nessun intervento di questo piano è stato
eseguito e nessun test è stato lanciato durante la migrazione.**

Stati: **verificato** = presente e verificato da un job citato, per la proprietà indicata; **da collaudare** = presente,
ma senza collaudo rispetto al requisito nuovo; **mancante**; **decisione aperta** = serve una scelta dell'utente o del
protocollo prima di implementare.

## 0. Decisioni da chiudere prima o durante F0

| Id | Decisione | Blocca | Stato |
| --- | --- | --- | --- |
| D1 | Checkpoint Qwen da cui estrarre il ViT (famiglia, ID, revisione) oppure delega a scegliere e motivare (pipeline §18, domanda 1). La ricerca del 2026-10-26 aveva proposto Qwen3-VL-2B-Instruct (circa 4 GB), mai approvato | Punto 5B, download, ambiente | aperta |
| D2 | Metodo del budget: benchmark breve per fase e preventivo, oppure tetto in ore GPU o euro (§18, domanda 2) | Punto 7, piloti | aperta |
| D3 | Contratto di risposta: come si chiede o si sceglie la risposta diretta, system prompt (quello canonico di Violetto impone ragionamento e `\boxed{}` e rifiuta gli altri), `<think>` vuoto o assente nella modalità diretta, stop, prefill di F1 | Punto 1 e ogni target con decoder addestrabile | aperta |
| D4 | Metriche primarie, margini di regressione, budget di inferenza e regole di arresto dei piloti (§15.2) | Punto 7, piloti | aperta |
| D5 | Budget degli audit con agenti per sorgente e disegno dei campioni diagnostico e di stima (§§5.9, 16) | Ammissione dei dati annotati | aperta |
| D6 | Ruoli del pannello: quali benchmark pubblici sono sviluppo e quale componente nuova resta test riservato (§§5.5, 15.1) | Punti 3 e 7 | aperta |

Le scelte tecniche ordinarie (formati dei file, nomi, struttura dei moduli) non richiedono una domanda: si propongono
nelle specifiche del passo 1 e si mostrano all'utente prima di scrivere codice.

## 1. Quattro specifiche brevi, prima del codice (pipeline §5.10, ultimo paragrafo)

Documenti in `docs/specs/`, uno per tema, da approvare. Non serve progettare subito gli adattatori di tutte le sorgenti.

| Specifica | Contenuto minimo | Riusa |
| --- | --- | --- |
| S1 Record e manifest | Chiave sorgente/revisione/record/compito/variante; gruppo globale (problema, immagine, documento, istanza del generatore); split; hash di domanda, target e immagini ordinate; versione di trasformazione e preprocessing; stato e motivo; derivati e loro genitori | `provenance.py`, `data_lists.py`, chiavi degli shard |
| S2 Regole di ammissione delle prime famiglie | Sintetici, MAVIS (descrizione canonica e QA), UniMER (trascrizione): proprietà necessarie, livello di evidenza (record, invariante della famiglia, campione), valore esatto/visualizzato/richiesto | `f1_mavis.py`, `f1_schema.py`, `synth.py`, `latex.py`, esiti di `unimer_check` |
| S3 Contratto di risposta, estrazione e verificatori | Modalità diretta/ragionata, template, label, stop; estrazione per compito; esiti `corretto`/`errato`/`non determinato`/errore tecnico; answer-only; risposte finali incompatibili; tentativi abbandonati | `processing.py`, `f2.py` (`last_boxed`, `answers_match`), `latex.py` |
| S4 Campionamento e budget | Proprietà e quantità da stimare, popolazione e denominatore, incerti, campione diagnostico separato da quello di stima, tetto di chiamate e token per sorgente | regole di `config/audit/` solo come esperienza, non come istruzioni |

## 2. Interventi F0 in ordine

L'ordine segue la pipeline §17: prima contratto e dati (punti 1-4), poi modello e trainer (5-6), poi pannelli e
preventivo (7). I punti 5 e 6 possono procedere in parallelo ai punti 2-4 dopo le specifiche.

### Punto 1 — Contratto di risposta diretta/ragionata (pipeline §3.3, R1)

| Requisito | Stato | Evidenza |
| --- | --- | --- |
| Formato che Violetto genera (`<think>...</think>` senza a capo, `<|endoftext|>` finale) | verificato | job 29393, 29400; `processing.py:10-18` |
| Template, label e stop versionati, coerenti fra training, inferenza ed estrattore | mancante | `processing.py:197-199` usa il chat template dello snapshot, che fissa il system prompt `Please reason step by step, and put your final answer within \boxed{}` e solleva eccezione con system diversi (`chat_template.jinja` @ b1f3d572, righe 1, 5 e 13-16) |
| Modalità diretta per compiti semplici | mancante, decisione aperta (D3) | l'unico strumento è il prefill `F1_PREFILL = "<think></think>"` (`processing.py:46`); `replay.py:20-38` impone ragionamento e `\boxed{}` |
| Baseline di Violetto col template canonico | verificato (testo) | `slurm/eval_text_baseline.sbatch`, job 29391-29398 |

**Intervento minimo.** Un modulo del contratto con identificativo e versione (modalità, istruzione o system prompt,
posizione di `<think>`, label, stop, prefill ammessi); il processor lo riceve esplicitamente e lo scrive in ogni
`Example` e nel manifest del run. Il template canonico resta per la baseline di Violetto. Estrattori per modalità nel
punto 2 (S3).
**Collaudo.** CPU: tokenizzazione e label per ogni modalità, stop, rifiuto delle combinazioni non ammesse (estensione di
`tests/test_vl_data.py`, in `slurm/tests_cpu.sbatch`). GPU, dopo l'ok: rispetto della modalità diretta da parte di
Violetto su un piccolo pannello senza prefill nascosti (1 GPU, 8 CPU, 48G, circa 1 ora).
**Dipende da:** D3, S3.

### Punto 2 — Gold per compito e variante, equivalenze, osservabilità, ambiguità, answer-only (pipeline §§5.6-5.7, 5.10.2, 5.10.4; G2-G4, R6, R13)

| Requisito | Stato | Evidenza |
| --- | --- | --- |
| MAVIS: verità e QA ricavate da `diagram_generation` | verificato per coerenza fra descrizioni | `f1_mavis.py`, `f1_schema.py`; job 30033, 30052 (andata e ritorno su 199.050); `mavis_caption_check` run1 |
| MAVIS: fedeltà ai pixel e leggibilità alla risoluzione scelta | da collaudare | solo 40 immagini guardate (`outputs/f1_mavis/contact_run1`); nessuna invariante del renderer dimostrata |
| Valore esatto, visualizzato e richiesto separati | mancante | job 30049: i razionali (denominatori 2 e 3) sono disegnati come decimali a due cifre; oggi un solo valore per campo |
| UniMER: trascrizione distinta dall'equivalenza matematica | da collaudare | `latex.py` (`despace`, `effective_label`); `outputs/unimer_check/run4`: 51 errori certi, 1119 incerti, rietichettature |
| Ambiguità dei sintetici | mancante | `synth.py:290-297`: `shape_type` dei quadrilateri ha come risposta solo `spec["kind"]` fra square/rectangle/parallelogram/trapezoid (un quadrato è anche rettangolo e parallelogramma); per i triangoli right/isosceles/equilateral non si esclude che più classi siano vere |
| Estrazione answer-only e risposte finali incompatibili | mancante | `f2.py:109-135, 183`: l'estrazione richiede `</think>` e prende l'ultimo `\boxed{}`; `grade` non ha un esito `non determinato` |
| Evidenza salvata con la trasformazione, non solo `passed` | mancante | gli esiti dei controlli sono liste di esclusione e JSON per run, non campi del record |

**Intervento minimo.** Record di gold tipizzato (S1/S2) con riferimento originale e ammesso, forma canonica, regola delle
alternative, evidenza (programma o controllo, versione, esito), esiti separati di correttezza, osservabilità e
spiegazione. Costruttori per le tre prime famiglie: sintetici (stato della scena; domande `shape_type` riformulate come
"classe più specifica fra quelle date" o con insieme di risposte ammesse), MAVIS (dalla specifica già parsata, con
valore visualizzato), UniMER (etichetta originale, rietichettatura tracciata, incerti in quarantena). Estrattore e
verificatori per compito secondo S3, con i quattro esiti.
**Collaudo.** CPU, in `tests_cpu`: i casi di §5.10.5 (risposta diretta corretta, finali incompatibili, tentativo
abbandonato lecito, `1.5` contro `3/2` in risposta numerica e in trascrizione, classificazione geometrica ambigua) e test
di mutazione per famiglia. Poi un job CPU che costruisce i record su tutti gli shard delle tre famiglie (0 GPU, 4 CPU,
16G, ≤ 2 ore, da stimare su uno shard).
**Dipende da:** S2, S3; punto 1 per i target con ragionamento.

### Punto 3 — Split globali, esclusioni storiche, manifest nei loader, contabilità del consumato (pipeline §§5.2, 5.5, 5.8, 5.10.1, 5.10.3; G1, G5, R5, R7)

| Requisito | Stato | Evidenza |
| --- | --- | --- |
| Validazione per gruppo esplicito | mancante | `stream.py:5-8`, `158-166`, `265-273`: validazione = primi `val_count` campioni dell'ultimo shard di ogni sorgente |
| Esclusioni per identità di compito/variante | mancante | `stream.py:45`, `74-77`: insieme di chiavi del campione (immagine); per `f2_solution` anche la chiave del problema (`164-165`) |
| QA costruite al volo legate al record ammesso | mancante | `data.py:191-283` (`mavis_qa_sample`, `mavis_f1_sample`, `unimer_line_sample`, …) scelgono domanda o riga durante il caricamento |
| Rietichettature tracciate | da collaudare | `stream.py:79-97` (`apply_relabel`, solo se l'etichetta coincide); `unimer_check/run4/relabel.jsonl` |
| Liste obbligatorie e provenienza | verificato per la bozza F1-clean | `data_lists.py`; job 30051 |
| Deduplicazione trasversale fra sorgenti | mancante | `scripts/split_duplicates.py` lavora per sorgente e usa target identici; `decontam.py` normalizza il testo togliendo simboli: segnali, non prove |
| Esclusioni storiche come evidenze riesaminabili | mancante | liste in `data/decontam/p6`, `data/decontam/f1eval`, `data/decontam/text/replay`, `outputs/unimer_check/run4`, `outputs/unimer_overlap/run2`, `outputs/split_duplicates/run1`, `outputs/pixmo_check/run4` (riferite dalla bozza archiviata `f1-clean.json`) |
| Contabilità per sorgente di esempi consumati, target-token, ripetizioni, scarti dopo l'ammissione | parziale | `stream.py:243-248` conta gli errori per sorgente; `collator.py:68-72` conta `too_long` solo in totale; nessun conteggio di target-token per sorgente |

**Intervento minimo.** (a) Indice globale costruito dall'inventario fissato (`docs/ARTIFACT_INVENTORY.tsv`): ID e
provenienza, sha256 dei file, pixel normalizzati, pHash, testo e formule con normalizzazione conservativa; i
candidati si confermano prima di fondere i gruppi (niente catene di somiglianze deboli). (b) Import delle vecchie liste
con motivo, versione e stato (confermata, da riesaminare, irrisolta). (c) Manifest immutabile delle ammissioni (S1) e
filtro dei reader esistenti con l'indice: un `Source` legge solo le coppie ammesse e la validazione viene dal manifest.
(d) Contatori per sorgente e compito nel log e nel manifest del run: esempi, problemi unici, immagini, token di input,
token visivi, target-token, ripetizioni, scarti per errore e per lunghezza. Stesso vincolo per replay e RL.
**Collaudo.** CPU: copia trasversale fra due sorgenti, pHash simile con cifra decisiva diversa, split ereditato dai
derivati, QA non ammessa su immagine ammessa, validazione esplicita, contatori; poi `stream_stats` adeguato su un
manifest di prova. Costruzione dell'indice: job CPU (0 GPU, 4 CPU, 24G); la durata va misurata prima su uno shard per
sorgente: si leggono circa 170 GiB di shard e i benchmark riservati.
**Dipende da:** S1; D6 per i pannelli riservati, che vanno fissati prima dell'ammissione definitiva (§5.8.1).

### Punto 4 — Più immagini, preprocessing e ordine delle pagine (pipeline §§3.2, 3.4, 8; R2)

| Requisito | Stato | Evidenza |
| --- | --- | --- |
| Processor con liste di immagini | da collaudare | `processing.py:166-196, 204-216`: più immagini ammesse, solo prima del testo; segnaposti concatenati senza riferimenti espliciti |
| Encoder e proiettore per immagine, in ordine | da collaudare | `modeling.py:89-101` (shuffle per immagine, token in ordine), `190-235` (gruppi per lunghezza, risultato in ordine); job 29447 con una immagine per esempio |
| Caricatori con più immagini | mancante | `data.py:67-70`: `Sample.image` singola; `build_example` (`data.py:343-361`) passa `[sample.image]` |
| Posizioni continue fra le pagine dello stesso esempio | da collaudare | `collator.py:92-107` azzera le posizioni a ogni `Example`: le pagine dentro lo stesso esempio restano contigue se l'esempio è unico |
| Tetto per immagine e budget dell'intero esempio | decisione aperta | `processing.py:31` `MAX_IMAGE_TOKENS = 768`; `DIAGRAM_MAX_PATCHES = 2304` oltre le 1024 patch di training di NaFlex |

**Intervento minimo.** `Sample.images` (lista ordinata con identità) mantenendo i loader esistenti a un'immagine;
riferimenti espliciti nel testo secondo il contratto (punto 1); budget di token per immagine e per esempio registrato
nel record; prima versione con immagini all'inizio del turno, senza interleaving generale.
**Collaudo.** CPU: 2-4 immagini, conteggio dei segnaposti, ordine, budget. GPU (1 GPU, 8 CPU, 24G, 1 ora, estensione di
`f0_vl_checks.py`): corrispondenza patch → posizioni con più immagini e con packing, parità forward completo / prefill /
generazione con cache, scambio d'ordine e dettaglio piccolo su un pannello minimo di coppie.
**Dipende da:** punto 1 (riferimenti), punto 3 (record con più immagini).

### Punto 5 — Encoder candidati, confine ViT-merger, proiettore, layer e punti di iniezione (pipeline §§3.1, 3.5, 6.2-6.5; R3, R10)

| Requisito | Stato | Evidenza |
| --- | --- | --- |
| Encoder A SigLIP2 so400m NaFlex, revisione fissata | verificato (baseline) | `modeling.py:60-78`; snapshot cc24074f (job 29348); parità su solo testo bit per bit (job 29400) |
| Encoder B dal checkpoint Qwen | mancante, decisione aperta (D1) | `load_vision_tower` costruisce solo `Siglip2VisionModel` dalle chiavi `vision_model.*`; normalizzazione fissa `PIXEL_MEAN = PIXEL_STD = 0.5` (`modeling.py:33-34`) |
| Proiettore baseline 2x2 + MLP 4608 → 2560 → 1280 | verificato (forme e training F1 storico) | `modeling.py:80-101`; `VLConfig.projector_hidden = 2560` |
| Alternative del proiettore (lineare, `4C → 4C → 1280`) | mancante | dimensione nascosta configurabile, struttura fissa |
| Feature di layer intermedi | mancante | `modeling.py:232` usa solo `last_hidden_state` |
| DeepStack, ramo MUDD, value embeddings dipendenti dall'immagine | mancante | un solo punto di ingresso (hook su `embed_tokens`); `VLConfig.image_value_embeds` ammette solo `zero` e `token` (vettore comune, già un controllo di §6.4) |
| Precisione dell'encoder | da collaudare | bf16 lontano da fp32, TF32 entro il 2% con encoder congelato (job 29445, 29446); con il ViT addestrabile va rimisurata |

**Intervento minimo.** (a) Interfaccia dell'encoder (preprocess → patch e griglia; forward → feature ai layer
richiesti; dimensione `C`; compressione) con SigLIP2 come prima implementazione, a parità bit per bit con la baseline
attuale. (b) Encoder B dopo D1 e il download approvato: estrazione delle chiavi, preprocessing nativo, scelta del tensore
prima del merger, nessuna doppia compressione; parità delle feature con il percorso nativo dello stesso checkpoint;
verifica che transformers 5.6.2 lo supporti, altrimenti nuovo build dell'ambiente (con ok). (c) Proiettore
parametrizzato (lineare, MLP di riferimento, MLP largo) con conteggio dei parametri. (d) Specifiche scritte, una per
famiglia, di DeepStack, ramo MUDD e value embeddings visivi dopo la lettura di `third_party/limite-modeling/modeling_limite.py`
(quale tensore entra in quale aggregazione, forme, maschere, gate, inizializzazione che riproduce la baseline). Le
implementazioni seguono l'ordine di §6.5, una alla volta.
**Collaudo.** GPU, 1 GPU, 8 CPU, 24G, ≤ 1 ora ciascuno: refactoring di SigLIP2 identico alla baseline; parità
dell'encoder B col percorso nativo; per ogni ramo nuovo con contributo disattivato, uguaglianza col riferimento su
testo e immagini, flusso dei gradienti, parità prefill/generazione con cache. CPU: forme dei proiettori.
**Dipende da:** D1 per (b); punti 1-4 per i piloti, non per (a), (c), (d).

### Punto 6 — Gruppi dell'ottimizzatore e checkpoint completi col ViT addestrabile (pipeline §§3.2, 8, 14; R4)

| Requisito | Stato | Evidenza |
| --- | --- | --- |
| Ramo autograd dell'encoder | presente | `modeling.py:201, 229`: `no_grad` solo se l'encoder è congelato |
| ViT addestrabile | mancante | `modeling.py:164-172`: `set_trainable` congela sempre l'encoder; fasi solo `f1` e `f2` |
| Gruppi disgiunti per decoder, ViT, collegamenti, righe visive | mancante | `train_f2.py:202`: tutti i parametri addestrabili fuori dal decoder vanno nel gruppo `adapter` del proiettore (il ViT finirebbe lì) |
| Niente conteggi di parametri fissi | mancante | `train_f1.py:55` `F1_PARAMS`; `train_f2.py:60-61, 204-205` `DECODER_PARAMS`, `ADAPTER_PARAMS` |
| Salvataggio di tutti i moduli aggiornati e ripresa | parziale | `modeling.py:146-153` salva proiettore e righe; `train_f2.py:93-106` decoder master fp32 + adapter + ottimizzatori, senza ViT |
| Stati master e precisione per modulo | da collaudare | `train_utils.MasterWeights` (`train_utils.py:51`) per il decoder; un LR di 1e-6 sul ViT in bf16 può non produrre aggiornamenti |
| Regole sui nomi dei parametri | da collaudare | `optim/groups.py` (ipotesi, `f0_checks.py` sezione `7_param_groups`) |

**Intervento minimo.** Specifica dei moduli addestrabili per fase (decoder, ViT, collegamenti, righe visive) con
gruppo, LR, weight decay, precisione e stato master per ciascuno; conteggi attesi derivati dalla config e registrati nel
manifest, non costanti; checkpoint con ViT (se aggiornato), collegamenti, ottimizzatori, stato dei dati e config;
`train_f1.py` come base del nuovo trainer, con i suoi helper.
**Collaudo.** GPU (1 GPU, 8 CPU, 64G, 1 ora): gradienti presenti solo nei moduli dichiarati; variazione effettiva dei
pesi del ViT dopo pochi passi nella precisione scelta; salvataggio e ricaricamento identici; ripresa uguale alla corsa
continua. CPU: partizione dei gruppi senza sovrapposizioni.
**Dipende da:** punto 5 (a) per l'interfaccia dell'encoder.

### Punto 7 — Pannelli di sviluppo e valutazione, preventivo dei piloti (pipeline §§6.1, 6.2, 15, 16; R9, R10)

| Requisito | Stato | Evidenza |
| --- | --- | --- |
| Testo puro col valutatore ufficiale | verificato | `eval_text_baseline`, `eval_text_local` (job 29391-29398, 29682-29686) |
| Trascrizione di formule (UniMER-Test spe/cpe) | da collaudare | `scripts/eval_f1_formulas.py`; esiti storici in `outputs/eval_f1_formulas` |
| Immagine vera, bianca, assente | presente | `data.build_example(image_mode=…)` (`data.py:343-361`), `scripts/eval_f1_val.py` |
| Benchmark visivi (MathVista, MathVerse, MathVision, We-Math, DynaMath, Geoperception, VisOnlyQA) | mancante | dati e indici di decontaminazione presenti (`hf/hub`, `data/decontam/p6`, `data/decontam/f1eval`); nessun valutatore del modello VL |
| Pannelli di documenti, pagine multiple, grafici e tabelle, manoscritto, scene reali | mancante | nessuna sorgente scaricata per documenti, grafici, tabelle o manoscritto (`downloads/catalog_status.tsv`) |
| Controfattuali e varianti raggruppate | mancante | coppie controfattuali dei sintetici in `data/shards/synth/v3/pairs`; varianti di MathVerse e DynaMath non raggruppate |
| Profilo dei costi | parziale | 15.4 mila token/s con decoder addestrabile a 32k, 12.9 mila a 4k (job 29402, ViT congelato); ViT addestrabile ed encoder B non misurati |

**Intervento minimo.** Manifest del pannello (ID, ruolo sviluppo o test, metrica, gruppo delle varianti) fissato prima
dell'ammissione definitiva dei dati; valutatore per capacità del modello VL, con generazione a batch
(`generation.py`) e gli estrattori di S3, risultati appaiati per problema e intervalli; scelta delle sorgenti minime
per le capacità mancanti (§6.1: un sottoinsieme manoscritto, uno di scene reali, grafici e tabelle, documenti, coppie
di immagini), con download solo dopo approvazione. Preventivo: un job di profilo per ciascuna configurazione
implementata (token/s, memoria, secondi per passo) e la formula di §16.
**Collaudo.** CPU: costruzione del manifest del pannello e decontaminazione contro il training. GPU: una valutazione di
prova del valutatore visivo su Violetto con proiettore casuale e con un adapter storico, solo per validare il codice
(1 GPU, 8 CPU, 48G, ≤ 2 ore).
**Dipende da:** D2, D4, D6; punto 2 (estrattori).

## 3. Collaudi in allocazione da pianificare

Nessuno è stato eseguito in questa migrazione e nessuno è dichiarato superato.

| Collaudo | Scopo | Risorse | Quando |
| --- | --- | --- | --- |
| `slurm/tests_cpu.sbatch` | Suite CPU completa dopo la migrazione: conferma che nessun test dipendeva dai file spostati (la ricerca statica non ha trovato riferimenti) | 0 GPU, 4 CPU, 8G, 20 min | subito, con l'ok dell'utente |
| Contratto (punto 1) | Tokenizzazione, label e stop per modalità; risposta diretta di Violetto | CPU 20 min; GPU 1 ora | dopo D3 e S3 |
| Record di gold (punto 2) | Test di contratto e di mutazione; costruzione dei record delle prime famiglie | CPU 20 min; CPU 4 CPU 16G ≤ 2 ore | dopo S2, S3 |
| Indice globale e manifest (punto 3) | Collisioni, split ereditati, filtro dei reader, contatori | CPU 4 CPU 24G, durata da misurare su uno shard | dopo S1, D6 |
| Più immagini (punto 4) | Ordine, budget, parità con cache | CPU 20 min; GPU 1 ora | dopo punti 1, 3 |
| Encoder (punto 5) | Refactoring SigLIP2 identico; parità dell'encoder B col percorso nativo; rami disattivati uguali al riferimento | GPU 1 ora ciascuno | (a) subito dopo l'ok; (b) dopo D1 |
| ViT addestrabile (punto 6) | Gradienti, variazione dei pesi, salvataggio, ripresa | GPU 64G 1 ora | dopo punto 5 (a) |
| Valutatore visivo (punto 7) | Correttezza del codice di valutazione | GPU ≤ 2 ore | dopo punto 2 |

## 4. Fuori dal primo perimetro

F4 (trainer RL, rollout, log-probabilità, GRPO), interpolazioni F3-M/F4-M, certificazione statistica generale e nuove
generazioni di target non sono prerequisiti di F0 (pipeline §5.10, incarico §5). Le implementazioni di DeepStack, MUDD e
value embeddings visivi seguono le loro specifiche e l'ordine di §6.5, dopo il confronto iniziale degli encoder.
