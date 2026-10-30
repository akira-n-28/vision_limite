# Piano di implementazione F0

Stato al 2026-10-28, dopo la migrazione alla [pipeline v1.4](../PIPELINE_ADDESTRAMENTO_LIMITE_VL.md) (§§3, 5, 6, 17).
Aggiornato lo stesso giorno dopo la revisione esterna dei push fino a `e81f884`. Il confronto è fra il codice del
repository sul cluster e i requisiti della pipeline. La migrazione ha spostato solo config, launcher e documenti: per
codice e test il repository è identico a `9d977b6`.

**Nessun intervento di questo piano è stato eseguito.** Dopo la migrazione è girato un solo collaudo,
`slurm/tests_cpu.sbatch` (job 30255: 22 file, 775 controlli, nessun fallimento). Riguarda il codice esistente, non i
requisiti nuovi (§4).

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
| D6 | Ruoli del pannello: quali benchmark pubblici sono sviluppo e quale componente nuova resta test riservato (§§5.5, 15.1) | Punti 3 e 7 | proposta pronta, da approvare: [`docs/PANNELLO_VALUTAZIONE.md`](PANNELLO_VALUTAZIONE.md) (2026-10-30) |

Le scelte tecniche ordinarie (formati dei file, nomi, struttura dei moduli) non richiedono una domanda: si propongono
nelle specifiche del passo 1 e si mostrano all'utente prima di scrivere codice.

## 1. Quattro specifiche brevi, prima del codice (pipeline §5.10, ultimo paragrafo)

Seconde bozze del 2026-10-28, rivedute dopo la revisione esterna, in [`docs/specs/`](specs/README.md); da approvare.
Non serve progettare subito gli adattatori di tutte le sorgenti.

| Specifica | Contenuto minimo | Riusa |
| --- | --- | --- |
| S1 Record e manifest | Identificatori sha256 completi con controllo delle collisioni (i prefissi solo per la lettura); variante con contratto, modalità e prefill; gruppo globale e split; hash dei token di input e target e delle immagini ordinate; esclusioni storiche risolte su un solo record; derivati e loro genitori | `provenance.py`, `data_lists.py`, chiavi degli shard |
| S2 Regole di ammissione delle prime famiglie | Correttezza rispetto all'immagine; per ogni tipo di domanda semantica scritta, verifica indipendente, collaudi, invarianti del renderer e unicità; stati `escluso` e `quarantena` per tipo; errore confermato di `intersections` e ambiguità di classificazione dei sintetici; MAVIS e UniMER | `f1_mavis.py`, `f1_schema.py`, `synth.py`, `latex.py`, esiti di `unimer_check` |
| S3 Contratto di risposta, estrazione e verificatori | Contratto `vl-contract/1` (D3); quattro forme (answer-only, `<think>` completo, `<think>` aperto, malformato); completezza dai metadati di terminazione; formato, completezza e correttezza separati; politica per SFT, valutazione e RL; allineamento col valutatore ufficiale | `processing.py`, `generation.py`, `f2.py`, `latex.py`, `limite-evals` (tassonomia) |
| S4 Campionamento e budget | Unità di campionamento uguale all'unità della quantità, limite esatto (ipergeometrico o binomiale); `clustered_rate` e `corrected_upper` non adottati; risultati esplorativi o condizionati dove manca la garanzia; budget degli audit | `stem_audit.clopper_pearson` (solo lettura); le regole di `config/audit/` solo come esperienza |

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
| Semantica delle domande dei sintetici rispetto all'immagine | **errore confermato** | `intersections` conta entro ±R (`synth.py:404-411`), il disegno arriva a ±(R+1) (`synth.py:462, 466`). Controesempio `audit-68`: target `0` con un'intersezione disegnata. In v3: 37 target errati e 364 domande al bordo disegnato, cioè 401 scene; in 146 esempi di training la domanda è fra quelle scelte (S2 §2.1). Gli altri tipi non hanno una verifica indipendente (S2 §2.3) |
| Ambiguità di classificazione | mancante | domande aperte, senza opzioni nel prompt: tutti i 2.128 triangoli rettangoli sono anche scaleni; un quadrato soddisfa 4 classi; 9.465 risposte `label_type` = square sono vere anche come rectangle. Con le opzioni nel prompt, o la classe più specifica di una catena, la risposta è unica in tutti i casi di v3 (S2 §2.2). MAVIS `shape_type` (`f1_mavis.py:83`): classi non in catena |
| Estrazione answer-only e risposte finali incompatibili | mancante | `f2.py:109-135, 183`: l'estrazione richiede `</think>` e prende l'ultimo `\boxed{}`; `grade` non ha un esito `non determinato` |
| Evidenza salvata con la trasformazione, non solo `passed` | mancante | gli esiti dei controlli sono liste di esclusione e JSON per run, non campi del record |

**Intervento minimo.** Record di gold tipizzato (S1/S2) con riferimento originale e ammesso, forma canonica, regola delle
alternative, evidenza (programma o controllo, versione, esito), esiti separati di correttezza, osservabilità e
spiegazione. Costruttori per le tre prime famiglie:
- sintetici: correzione di `intersections`, domande di classificazione con opzioni, verifica semantica indipendente,
  nuova versione v4 degli shard;
- MAVIS: dalla specifica già parsata, con valore visualizzato ed esatto, `shape_type` con opzioni costruite dalle
  misure;
- UniMER: etichetta originale, rietichettatura tracciata, incerti in quarantena.

Estrattore e verificatori per compito secondo S3.
**Collaudo.** CPU, in `tests_cpu`: i casi di §5.10.5 (risposta diretta corretta, finali incompatibili, tentativo
abbandonato lecito, `1.5` contro `3/2` in risposta numerica e in trascrizione, classificazione geometrica ambigua) e test
di mutazione per famiglia. Poi un job CPU che costruisce i record su tutti gli shard delle tre famiglie (0 GPU, 4 CPU,
16G, ≤ 2 ore, da stimare su uno shard).
**Dipende da:** S2, S3; punto 1 per i target con ragionamento.

### Punto 3 — Split globali, esclusioni storiche, manifest nei loader, contabilità del consumato (pipeline §§5.2, 5.5, 5.8, 5.10.1, 5.10.3; G1, G5, R5, R7)

| Requisito | Stato | Evidenza |
| --- | --- | --- |
| Validazione per gruppo esplicito | implementata per i sintetici v4 (2026-10-30, `630d00a`, S1 §9); mancante per le altre sorgenti | con un manifest (`manifest.py`, `stream.py`) la validazione è lo split dev del manifest, assegnato per gruppo; senza, resta quella di prima: i primi `val_count` campioni dell'ultimo shard di ogni sorgente |
| Esclusioni per identità di compito/variante | mancante | `stream.py:45`, `74-77`: insieme di chiavi del campione (immagine); per `f2_solution` anche la chiave del problema (`164-165`) |
| QA costruite al volo legate al record ammesso | mancante | `data.py:191-283` (`mavis_qa_sample`, `mavis_f1_sample`, `unimer_line_sample`, …) scelgono domanda o riga durante il caricamento |
| Rietichettature tracciate | da collaudare | `stream.py:79-97` (`apply_relabel`, solo se l'etichetta coincide); `unimer_check/run4/relabel.jsonl` |
| Liste obbligatorie e provenienza | verificato per la bozza F1-clean | `data_lists.py`; job 30051 |
| Deduplicazione trasversale fra sorgenti | mancante | `scripts/split_duplicates.py` lavora per sorgente e usa target identici; `decontam.py` normalizza il testo togliendo simboli: segnali, non prove |
| Esclusioni storiche come evidenze riesaminabili | mancante | liste in `data/decontam/p6`, `data/decontam/f1eval`, `data/decontam/text/replay`, `outputs/unimer_check/run4`, `outputs/unimer_overlap/run2`, `outputs/split_duplicates/run1`, `outputs/pixmo_check/run4` (riferite dalla bozza archiviata `f1-clean.json`) |
| Contabilità per sorgente di esempi consumati, target-token, ripetizioni, scarti dopo l'ammissione | parziale | `stream.py` conta errori, errori del manifest ed esempi caricati per sorgente, e i reader contano letti, saltati e consegnati (`630d00a`); `collator.py` conta `too_long` solo in totale; nessun conteggio per compito né di target-token nel flusso (i conteggi per compito sono nel controllo `manifest_check.py`) |

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
| Trascrizione di formule su UniMER-Test | dati presenti; valutatore da collaudare | shard spe, cpe, sce e hwe (hwe = manoscritto) in `data/shards/unimer/2343ddd9/test/`; `scripts/eval_f1_formulas.py` usa spe e cpe di default, gli altri con `--subsets`; esiti storici in `outputs/eval_f1_formulas` |
| Immagine vera, bianca, assente | presente | `data.build_example(image_mode=…)` (`data.py:343-361`), `scripts/eval_f1_val.py` |
| Benchmark visivi (MathVista, MathVerse, MathVision, We-Math, DynaMath, Geoperception, VisOnlyQA) | dati presenti; valutatore mancante | dati e indici di decontaminazione in `hf/hub`, `data/decontam/p6`, `data/decontam/f1eval`; nessun valutatore del modello VL |
| Dati presenti per i pannelli percettivi | presenti, da selezionare | scene reali: PixMo-Cap (25 parquet su 75) e LLaVA-Pretrain; manoscritto: UniMER-Test hwe (solo valutazione) e le formule CROHME nel training di UniMER (meno dell'1%, non separabili per indice); grafici e tabelle: solo dentro alcuni benchmark visivi riservati, da individuare per ID |
| Dati mancanti | mancante; pannello e download proposti in [`docs/PANNELLO_VALUTAZIONE.md`](PANNELLO_VALUTAZIONE.md) | documenti a più pagine, struttura di tabelle e pagine, OCR generale, manoscritto per il training (MathWriting), coppie di immagini: nessuna sorgente scaricata (`downloads/catalog_status.tsv`) |
| Valutatori mancanti | mancante | nessun valutatore del modello VL per i benchmark visivi, per più immagini o per i documenti; per le formule c'è solo `eval_f1_formulas.py` |
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

## 3. Prossimi interventi al codice, in ordine (elenco minimo)

Ognuno è piccolo, ha i suoi collaudi CPU in `slurm/tests_cpu.sbatch` e non tocca dataset o checkpoint esistenti. I job
segnati vanno approvati uno per uno. Gli interventi dei punti 4-7 (più immagini, encoder, trainer, valutatore) vengono
dopo questi.

| # | Intervento | File | Collaudi | Job necessario | Dipende da |
| --- | --- | --- | --- | --- | --- |
| C1 | Sintetici: un solo dominio visibile per `intersections` (conteggio geometrico e ammissibilità separati); niente domanda per grafici che si sfiorano senza incrociarsi; domande di classificazione con opzioni (triangoli) o classe più specifica (quadrilateri, `label_type`); shapes senza figure vicine né fuori dagli assi (col tratto), con nuova estrazione solo per le scene che violano i vincoli; layout degli oggetti disegnati nel JSON da v4, con lo sha256 del PNG del campione (`png_sha256`); `version=3` riproduce v3. **Implementato il 2026-10-28 (`38af82a`), corretto il 2026-10-29 (figure tagliate; digest del PNG dopo la revisione di `f7d32d3`); suite CPU passata (job 30392), controllo sugli shard eseguito (job 30393, §4), scarti analizzati (job 30400, S2 §2.5); correzioni minime di v4 in `b0b90c5` (coppie per riscalatura, punto segnato lontano dalle misure, tacche -1 senza numero), con suite e controlli passati (job 30417-30420); render di v4 eseguito il 2026-10-30 su incarico dell'utente (job 30447, `data/shards/synth/v4`), candidato controllato (job 30448-30449, S2 §2.6)** | `src/limite_vl/synth.py`, `scripts/synth_render.py` (`--generator-version` obbligatorio), `slurm/synth_render.sbatch` | `tests/test_synth.py` (S2 §5) | render di v4: CPU, 8 CPU, 16G, 6 min 18 s (job 30447) | S2 |
| C2 | Verifica semantica indipendente dei sintetici: ricalcolo per tipo dagli oggetti disegnati (layout) con predicati propri, intersezioni solo nei tratti disegnati; invarianti del renderer, compreso il taglio sull'area effettiva degli assi; violazioni bloccanti che impediscono la verifica del campione; nel report esiti separati per semantica della scena, corrispondenza col PNG dello shard e campione (S2 §2.4). **Implementato il 2026-10-28 (`38af82a`), corretto il 2026-10-29, esiti per campione dopo la revisione di `f7d32d3`; suite CPU passata (job 30392), controllo sugli shard eseguito (job 30393, §4); coordinate lette dalle tacche vicine (`b0b90c5`, S2 §2.5), controlli passati (job 30417-30420); collaudi a 10.000 scene e controllo del candidato v4 sugli shard salvati (`284e8f1`, job 30448-30449: integrità, riproducibilità, checker, coppie, duplicati e split)** | `src/limite_vl/synth_check.py`, `scripts/synth_semantics_check.py` (anche `--cases`), `slurm/synth_semantics_check.sbatch`, `scripts/synth_case_svg.py` (schemi dei casi), `scripts/synth_candidate.py`, `slurm/synth_candidate.sbatch` | mutazioni ed esiti per campione (`test_sample_evidence`) e candidato (`test_candidate`) in `tests/test_synth.py`; su v3 deve ritrovare i difetti noti | controllo di v3 e di scene v4 in memoria (CPU, 8 CPU, 16G, ≤ 1 h 30); candidato v4: 8 CPU, 16G, 9 min 38 s | C1 |
| C3 | Contratto `vl-contract/1`: oggetto versionato, template `limite-vl/1`, processor che lo riceve e lo registra | `src/limite_vl/contract.py` (nuovo), `processing.py` | S3 §6 (tokenizzazione delle due modalità, system prompt canonico identico ai token di Violetto) | — | D3 |
| C4 | Estrattore e verificatori: forma, completezza dai metadati, candidati, tre esiti separati; `generate_batch` restituisce anche `max_new_tokens` e il motivo d'arresto | `src/limite_vl/answers.py` (nuovo), `generation.py` | S3 §6, compresi i casi A-T della tassonomia ufficiale | — | C3 |
| C5 | Identità e manifest: JSON canonico, digest completi, mappa delle collisioni, record, scrittura immutabile; i loader restituiscono la costruzione canonica. **In parte, per i sintetici v4 (2026-10-30, `630d00a`, S1 §9): JSON canonico, digest e collisioni, `record_id`, `group_id`, record, split per gruppo, scrittura immutabile (`data/manifests/synth-v4-candidato-1`, job 30451); mancano `variant_id`, `uid` e la costruzione canonica dai loader, che dipendono da C3** | `src/limite_vl/manifest.py`, `scripts/synth_manifest.py`, `data.py` | S1 §8 (collisione forzata, canonicalizzazione, prefissi, contratto): `tests/test_manifest.py` tranne il contratto | — | C3 (per la costruzione canonica) |
| C6 | Indice globale e import delle esclusioni storiche | `scripts/build_index.py` (nuovo) | S1 §8 (gruppi, chiavi storiche) | costruzione dell'indice (CPU, 4 CPU, 24G, durata da misurare su uno shard per sorgente) | C5, D6 |
| C7 | Reader col manifest, validazione dal manifest, contatori per sorgente e compito (`too_long` compreso). **In parte (2026-10-30, `630d00a`): solo i record ammessi dello split, validazione dal manifest, sha256 di JSON e PNG controllati, contatori dei reader e degli esempi caricati per sorgente, shard v4 leggibili solo col manifest; verifiche sui dati reali nel job 30451 (`scripts/manifest_check.py`); mancano i contatori per compito e `too_long` per sorgente** | `stream.py`, `collator.py`, `data_lists.py` | S1 §8 (reader): `tests/test_manifest.py` | verifiche del manifest (CPU, 8 CPU, 16G, 4 min) | C5 |
| C8 | Stime: limite ipergeometrico e binomiale, estrazione con probabilità registrate, Bonferroni, rifiuto se unità e quantità non coincidono | `src/limite_vl/estimate.py` (nuovo) | S4 §8 (copertura esatta sul controesempio) | — | S4 |
| C9 | MAVIS: `shape_type` con opzioni dalle misure note; valore visualizzato ed esatto nel record | `src/limite_vl/f1_mavis.py` | S2 §5 (MAVIS) | controllo su tutti gli shard (CPU, come il job 30052) | C5 |

## 4. Collaudi

### Già eseguiti

| Job | Collaudo | Esito | Che cosa dimostra |
| --- | --- | --- | --- |
| 30255 (2026-10-28) | `slurm/tests_cpu.sbatch`, suite CPU completa dopo la migrazione | 22 file, 775 controlli, nessun fallimento, stderr vuoto, come il job 30050 prima della migrazione | che nessun test dipendeva dai file archiviati; nulla sui requisiti nuovi |
| 30392 (2026-10-29) | `slurm/tests_cpu.sbatch` su `b544b97`, prima esecuzione di `tests/test_synth.py` | 23 file, 871 controlli, nessun fallimento, stderr vuoto, 7 min 42 s. I 22 file del job 30255 sono invariati e danno gli stessi 775 controlli; `test_synth.py` ne aggiunge 96 | che il codice di C1-C2 compila e supera le regressioni scritte, col disegno vero. Coperti: PNG, specifica e domande di 8 scene v3 identici allo shard; taglio di `synth-0000043` (v3 fuori dagli assi di 7.3 px, v4 dentro); tratti visibili; mutazioni; esiti per campione su uno shard di prova. Nulla sugli shard interi né sui dati |
| 30393 (2026-10-29) | `slurm/synth_semantics_check.sbatch c1c2-run1` su `253d505`: gli 8 shard v3 in sola lettura, 20.000 scene v4 in memoria, casi `synth-0000043` v3 e v4 | 8 min 8 s, conteggi coerenti (`incoherent` vuoto). v3: 60.900 PNG ridisegnati con byte identici, 38.161 campioni verificati. v4: 18.146 scene approvate dal checker su 20.000, nessuna risposta segnalata errata. Dettagli sotto | che v3 si riproduce byte per byte e che il controllo ritrova i difetti noti di v3; segnalazioni nuove, presenti anche in v4, analizzate nel job 30400. Non certifica dati |
| 30400 (2026-10-29) | `slurm/synth_review.sbatch scarti-run1` su `8cd2c22`: analisi degli scarti del job 30393 (coppie polygon v3, riquadri di testo sovrapposti v4, coordinate) | 1 min 11 s. Censimento uguale al report (2.702 violazioni). Coppie: 9 contraddizioni dimostrabili su 75. Riquadri: collisioni reali fra misure ed etichette dei punti nei polygon (692 su 840), all'origine quasi solo riquadri. Coordinate: 227 su 227 leggibili da scala. S2 §2.5 | quali segnalazioni sono errori, quali ambiguità e quali controlli troppo severi; base della proposta minima. Non modifica generatore né checker, non certifica dati. Le 9 contraddizioni vengono da una griglia: la prova esatta è nel job 30420 |
| 30417 (2026-10-29) | `slurm/tests_cpu.sbatch` su `b0b90c5`, correzioni minime di v4 | 23 file, 892 controlli, nessun fallimento, 7 min 25 s; `tests/test_synth.py` 117 (21 nuovi) | che le correzioni e la regola delle coordinate superano le regressioni scritte, v3 compresa |
| 30418 (2026-10-29) | `slurm/synth_semantics_check.sbatch c1c2-run2` su `b0b90c5`: 8 shard v3 e 20.000 scene v4 in memoria | 8 min 35 s, conteggi coerenti. v3: 60.900 PNG con byte identici, 38.325 campioni approvati dal checker (+164: le 169 `coords`). v4: 19.976 scene approvate su 20.000; 25 sovrapposizioni residue (etichetta di punto e tacca, coord); nessuna domanda non ammissibile o errata | che v3 è invariata e che le correzioni tolgono le segnalazioni corrette; copertura invariata tranne `m_side` (2.471 → 2.110). Non certifica dati |
| 30419 (2026-10-29) | `slurm/synth_semantics_check.sbatch c1c2-pairs-run1 --pairs 300` su `b0b90c5` | 43 s: 600 record di coppie v4 approvati dal checker, 600 domande ok, nessuna violazione | che le coppie che il render scriverebbe sono coerenti per il checker. Non certifica dati |
| 30420 (2026-10-29) | `slurm/synth_review.sbatch scarti-run2 --before scarti-run1` su `b0b90c5` | 22 s. Prove esatte: 9 contraddizioni confermate (margini 0.94-3.84), 00285 solo con interi esatti. Residui: 25 violazioni in 24 scene. 16 scene del campione senza più sovrapposizioni; immagini prima/dopo | correttezza dimostrata per le 9 esclusioni; il resto è approvazione del checker. Non certifica dati |
| 30447 (2026-10-30) | `slurm/synth_render.sbatch v4 --generator-version 4` su `284e8f1`: 60.000 scene train, 300 test, 300 coppie, chiavi e semi di v3 | 6 min 18 s; 8 shard in `data/shards/synth/v4` (892 MiB), manifest con sha256 e identità del codice | che il candidato v4 esiste con le dimensioni, le chiavi e gli split di v3; v3 intatta |
| 30448 (2026-10-30) | `slurm/synth_semantics_check.sbatch v4-run1` su `284e8f1`: prima `tests/test_synth.py`, poi il checker sugli 8 shard v4 salvati | 1 min 50 s. Test: 128 controlli, nessun fallimento (collaudi a 10.000: 4.449 `intersections` confermate anche dal conteggio numerico, 24.486 `label_type` uniche). Checker: 60.900 record con digest corrispondente, 60.833 approvati, 67 con testi sovrapposti, nessuna domanda non ok | che le regressioni a 10.000 e i collaudi del candidato passano, e che il checker approva gli artefatti salvati salvo 67 scene. Non certifica dati |
| 30449 (2026-10-30) | `slurm/synth_candidate.sbatch v4-run1` su `284e8f1`: integrità, riproducibilità, checker confrontato col 30448, coppie, formato, duplicati e split | 9 min 38 s. Nessun problema d'integrità, di riproducibilità, di coppie o di formato; processor invariato su 49 dimensioni. Train: 57.141 `candidato`, 67 `quarantena` (testi sovrapposti), 2.792 `escluso` (`function` con lo stato di un record di valutazione); test e coppie interi. S2 §2.6 | che il candidato v4 è riproducibile e integro e che le regole degli split sono applicate. Non certifica dati: niente stima campionaria, checker sul layout |
| 30450 (2026-10-30) | `slurm/tests_cpu.sbatch` su `630d00a` (manifest delle ammissioni e filtro dei reader) | 24 file, 937 controlli, nessun fallimento, stderr vuoto, 10 min 46 s; `tests/test_manifest.py` 34 controlli | che il sottoinsieme di S1 supera i collaudi su shard piccoli e che i test esistenti di stream e loader non cambiano |
| 30451 (2026-10-30) | `slurm/synth_manifest.sbatch synth-v4-candidato 1` su `630d00a`: manifest dal candidato v4-run1 e verifiche CPU | 4 min. Manifest `data/manifests/synth-v4-candidato-1`: train 54.271, dev 2.870 (5.02%, per famiglia 747, 623, 750, 750), quarantena 67, esclusi 2.792, test 300, coppie 600; 52.457 gruppi. Verifiche: 32 su 32 (integrità, split riproducibile, nessun gruppo in due split, ogni split letto esattamente dai reader, domande costruite uguali alle QA ammesse, validazione dal dev, flusso col processor, 10 arresti espliciti). S2 §2.7 | che i loader consumano solo i record ammessi dello split richiesto e che lo split è riproducibile. Non certifica dati |

Calcoli di sola libreria standard sul frontend, il 2026-10-28, a supporto delle specifiche (non sono collaudi):
- riproduzione delle domande di synth v3 dalle chiavi, con istogrammi identici al manifest del job 29756;
- conteggio delle intersezioni nel dominio disegnato e controllo delle classificazioni (S2 §2);
- controesempio di `clustered_rate` (S4 §4).

Dopo l'implementazione di C1-C2, una verifica leggera del codice nuovo sul frontend (solo `synth` e `synth_check`,
in streaming: 18 s, RSS 20 MB). Niente disegno, niente shard, non è la suite di test:
- con `version=3` gli istogrammi delle 60.000 scene coincidono ancora con il manifest del job 29756;
- la verifica indipendente sulle 6.307 domande `intersections` di v3 dà 5.897 ok, 37 errate e 373 non ammissibili
  (364 al bordo e 9 grafici che si sfiorano);
- casi del collaudo: `audit-68` ha conteggio geometrico 1 e nessuna domanda in v4; `synth-0049798` ha target 1 in v4 e
  0 in v3; i casi costruiti danno i valori attesi;
- `shape_type`: in v4 una sola risposta vera su 3.459 domande; in v3 sono aperte con più risposte vere, tranne i
  trapezi;
- v3 → v4 su 8.000 chiavi: 7.992 scene con domande identiche nei tipi non toccati e stesse risposte di
  classificazione, 8 shapes riestratte;
- 1.739 domande `intersections` di v4 tutte confermate dalla verifica; 0 shapes v4 con figure vicine su 15.000.

Revisione di `38af82a` (2026-10-29): tre difetti riproducibili, corretti.
- `tests/test_synth.py` non compilava: un `:=` nell'iterabile di una comprehension. Il controllo precedente usava
  `ast.parse`, che non costruisce la tabella dei simboli e non lo vede.
- Le figure delle shapes potevano uscire dagli assi (`synth-0000043`), e il verificatore guardava solo i bordi
  dell'immagine.
- Le intersezioni ignoravano i tratti effettivamente disegnati.

Stato delle correzioni, tenendo distinti i tre livelli:
- **Implementazione.**
  - `synth._shapes_fit` col margine del tratto; il generatore v4 riestrae le scene che lo violano.
  - Nel layout: `axes_px`, `lw_px`, `clip_on`, `samples`.
  - `synth_check`: taglio sull'area degli assi, violazioni bloccanti che impediscono la verifica del campione,
    intersezioni nei soli tratti disegnati con le estremità classificate.
  - `synth_semantics_check.py --cases`, `scripts/synth_case_svg.py`, regressioni in `tests/test_synth.py`.
- **Verifica locale sul frontend** (2026-10-29; sola libreria standard, `synth` e `synth_check` caricati da soli,
  nessun disegno né shard; 20 s, RSS 22 MB):
  - `compile()` in memoria di tutti i 130 file Python tracciati e dei nuovi, senza errori; prima della correzione
    falliva solo `tests/test_synth.py:222`;
  - 15 controlli su 15 riusciti:
    - v3 riprodotta: istogrammi delle 60.000 scene identici al manifest del job 29756, specifiche identiche a `38af82a`;
    - `synth-0000043`: tagliata in v3 e nella v4 di `38af82a`, corretta nella nuova v4;
    - nessuna figura vicina o fuori dagli assi nelle 15.000 scene shapes v4;
    - su 8.000 chiavi riestratte esattamente le 122 shapes da correggere, le altre identiche;
    - il controesempio dei tratti visibili dà (0, []) col codice nuovo e (1, []) con quello di `38af82a`;
    - casi del buco e del campionamento non ammissibili;
    - 1.739 domande v4 confermate;
    - su v3 gli stessi 5.897 ok, 37 errate e 373 non ammissibili.
  - `tests/test_synth.py` compila ma **non è stato eseguito**: importa numpy e disegna, quindi va in un job.
- **Suite CPU in allocazione: eseguita il 2026-10-29** (job 30392, tabella sopra), col disegno vero. Coperti: layout
  e invarianti del renderer, taglio di `synth-0000043`, PNG, specifica e domande di 8 scene v3 identici allo shard,
  script del controllo su uno shard di prova.
- **Controllo sugli shard: eseguito il 2026-10-29** (job 30393, sotto). Nessun dato è certificato da questi controlli.

Revisione di `f7d32d3` (2026-10-29): il report di `scripts/synth_semantics_check.py` contava come verificato un campione
di uno shard guardando solo la semantica della scena ricostruita, anche con il PNG diverso o mancante. Inoltre, con il
layout già nel record, il PNG non si confrontava affatto. Stato della correzione:
- **Implementazione.**
  - Esiti separati per semantica della scena, immagine e campione, più la provenienza del layout; il significato degli
    stati è in S2 §2.4.
  - Il layout di `synth_render.py` v4 e di `--cases` porta `png_sha256`. I record senza digest si ridisegnano e si
    confrontano: prima i byte, poi i pixel decodificati.
  - Le scene di `--generate` hanno un'origine propria.
  - In `problems.jsonl` una riga per ogni campione non verificato, coi motivi. Il report (schema `synth-check/3`)
    controlla la coerenza dei conteggi.
  - `tests_cpu.sbatch` tiene configurazione e cache di matplotlib dentro il progetto.
  - Regressioni: `test_sample_evidence` in `tests/test_synth.py`.
- **Verifica locale: nessuna.** Dal 2026-10-29 anche `compile()` va in un job.
- **Suite CPU: passata il 2026-10-29** (job 30392, tabella sopra), compreso `test_sample_evidence`. `test_synth.py`
  carica `synth_semantics_check.py` e `synth_render.py`, quindi tutti i file toccati sono stati compilati ed eseguiti.
- **Controllo sugli shard: eseguito il 2026-10-29** (job 30393, sotto).

Controllo sugli shard (job 30393, 2026-10-29, codice `253d505`, uscite in `outputs/synth_check/c1c2-run1`):
- **Coerenza.** `incoherent` vuoto. Il report risulta `dirty` solo per la riga del ledger scritta all'invio.
- **v3: 60.900 campioni** (60.000 di training, 300 di test e 600 delle coppie).
  - Immagini: tutte `byte_identici`. Nessun campione ha il layout, quindi la provenienza è `ridisegnato_layout_assente`.
  - Campioni: 38.161 verificati e 22.739 no. Verificati per famiglia: coord 14.101, function 13.083, polygon 3.557,
    shapes 7.420 su 15.225 ciascuna.
  - Domande: 819.701 ok, 26.041 non ammissibili e 37 errate.
  - Difetti noti ritrovati:
    - `intersections`: 37 errate, 374 non ammissibili e 5.920 ok. Il totale è coerente col calcolo sul frontend per il
      solo training (5.897, 37 e 373) più le 24 domande dello shard di test; il report però non separa gli shard;
    - `shape_type`: 10.758 non ammissibili (le aperte; ok 2.153);
    - `label_type`: 11.050 non ammissibili;
    - 625 scene shapes bloccate, con 623 violazioni `figura_tagliata` e 15 `figure_sovrapposte`; fra queste
      `synth-0000043`.
- **v4 in memoria: 20.000 scene.**
  - Esiti: 18.146 verificate. Per famiglia: coord 4.628, function 4.358, polygon 4.160, shapes 5.000 su 5.000.
  - Domande: 281.164 ok e 58 non ammissibili, tutte `coords` senza l'etichetta di tacca che serve a leggere la
    coordinata (169 anche in v3). Il checker non segnala nessuna risposta errata. `intersections` (2.155),
    `shape_type` (4.301) e `label_type` (12.345) sono tutte ok.
  - 1.796 scene non sono approvate dal checker soltanto per violazioni non bloccanti.
  - "Verificata" qui vuol dire approvata dal checker: non è correttezza dimostrata né qualità certificata (S2 §2.4).
- **Casi.** `synth-0000043` v3 è bloccata da `figura_tagliata`: le 8 domande sono non ammissibili e il PNG mostra il
  rettangolo L senza il lato sinistro. In v4 la scena ha un solo quadrato dentro gli assi e 7 domande ok. Il PNG v4
  ha lo stesso sha256 del job 30392: il disegno è deterministico fra job.
- **Segnalazioni nuove**, presenti in v3 e nel codice v4 attuale. Sono analizzate nel job 30400 (sotto e S2 §2.5),
  che corregge la prima lettura data qui:
  1. *Coppie controfattuali polygon*: `synth.counterfactual` cambia la lunghezza scritta di un lato senza cambiare
     il disegno, e tutte le 75 varianti `-b` hanno `lunghezza_incoerente`. Questo viola l'invariante di scala, ma non
     basta a rendere errata la risposta: le contraddizioni dimostrabili coi vincoli dichiarati sono 9.
  2. *Riquadri di testo sovrapposti* (più di 1 px): violazioni in v3 coord 1.808, function 3.396, polygon 2.636; in
     v4 591, 1.232 e 840.
     - Le percentuali date qui in un primo momento venivano dalle righe conservate in `problems.jsonl`, che non sono
       un campione rappresentativo. Valgono i conteggi del censimento del job 30400.
     - In polygon l'etichetta coinvolta è quella del punto segnato su un lato, non quella di un vertice.

Analisi degli scarti (job 30400, 2026-10-29, codice `8cd2c22`, `scripts/synth_review.py`, uscite in
`outputs/synth_review/scarti-run1`). Il job legge il controllo e lo shard delle coppie in sola lettura e ridisegna in
memoria; generatore e checker non cambiano. Dettagli, tabelle e immagini in S2 §2.5.
- **Coppie polygon v3 (75).**
  - Tutte e 75 hanno la risposta uguale alla misura stampata.
  - 9 contraddicono un angolo dichiarato (scritto o retto) con tre lati stampati: `escluso`.
  - 1 è contraddittoria solo leggendo gli interi come esatti.
  - 26 sono compatibili con una riscalatura uniforme, cioè hanno una sola misura.
  - 39 non violano vincoli dichiarati ma cambiano la forma.
  - In 5 la misura cambiata è coperta dall'etichetta del punto sul lato.
  - Tutte tranne le 9 restano in `quarantena`.
  - Contratto proposto per v4: geometria coerente in scala, con la coppia ottenuta per riscalatura uniforme.
- **Riquadri sovrapposti v4.** Il censimento copre tutte le 2.702 violazioni e torna col report.
  - All'origine (1.798 violazioni) i tratti si toccano o condividono inchiostro in 198 casi; negli altri resta
    spazio fra i caratteri. Nessuna domanda usa quelle etichette, salvo `O` nel conteggio.
  - Nei polygon la misura e l'etichetta del punto sul lato condividono inchiostro in 692 casi su 840: sono collisioni
    reali, che possono cambiare la lettura di una misura.
  - Il campione di 20 casi più 5 controlli approvati lo conferma. Un controllo approvato, con riquadri sovrapposti di
    1.0 px, è uguale a vista a un caso segnalato con 1.2 px.
- **Coordinate.** Tutte le 227 domande non ammissibili (58 v4, 169 v3) si leggono contando una tacca. La tacca è
  disegnata fra due tacche numerate e sporge di 2.5-4.0 px: il checker è più severo del necessario.
- **Proposta minima** (S2 §2.5), approvata dall'utente e attuata in `b0b90c5`:
  1. coppie polygon con una sola misura e riscalatura uniforme;
  2. punto segnato mai su un lato con la misura;
  3. niente etichette sulle tacche `-1`;
  4. nel checker, coordinata leggibile da una tacca disegnata fra due tacche numerate, con le tacche nel layout.

  La soglia delle sovrapposizioni non cambia.

Correzioni di v4 e controlli dopo (job 30417-30420, 2026-10-29, codice `b0b90c5`; dettagli in S2 §2.5):
- **Prova delle contraddizioni.** Le 9 coppie contraddittorie sono confermate con limiti esatti del lato opposto,
  invece della griglia, con margini fra 0.94 e 3.84 unità. 00285 resta contraddittoria solo con gli interi esatti.
- **Suite (30417).** 892 controlli, nessun fallimento; 117 in `tests/test_synth.py`.
- **Controllo (30418).** v3 identica: 60.900 PNG con byte identici, 169 domande `coords` ora leggibili, il resto uguale.
  v4: 19.976 scene approvate dal checker su 20.000; restano 25 sovrapposizioni fra etichette di punti e tacche in coord.
- **Coppie (30419).** 300 coppie v4: 600 record approvati, nessuna violazione.
- **Copertura.** Uguale per tutti i tipi tranne `m_side`: da 2.471 a 2.110 su 20.000 scene, per il punto tolto nei
  triangoli con tre lati misurati.
- **Analisi (30420).** Immagini prima/dopo in `docs/img/synth-review/prima-dopo-*.png`.

Candidato v4 (job 30447-30449, 2026-10-30, codice `284e8f1`; dettagli in S2 §2.6). Le decisioni dell'utente: `m_side`
accettato, sovrapposizioni in quarantena su tutte le scene, nessuna soglia allentata, collaudi a 10.000.
- **Render (30447).** Dimensioni, chiavi, semi e split di v3, in `data/shards/synth/v4`.
- **Controllo (30448).** Collaudi a 10.000 superati; checker sugli shard salvati: 67 scene non approvate, tutte per
  testi sovrapposti.
- **Candidato (30449).** Artefatti integri e riproducibili; coppie conformi; nessun errore matematico.
  - Train: 57.141 candidati, 67 in quarantena, 2.792 esclusi per contaminazione.
  - Le scene `function` hanno pochi stati, cioè coefficienti interi: 7.074 stati per 15.000 scene. 2.792 scene di
    train hanno lo stesso stato di un test o di una coppia.
  - Copertura dopo i filtri: `slope` −19%, `exists` di function −18.6%, `intercept` −12.6%; coord −0.4/−0.8%;
    polygon e shapes invariati.
- **Stato.** Ancora incompleto per gli esperimenti (pipeline §15.3): mancano il manifest nei loader, il dev split e un
  audit campionario.

Manifest e dev split (job 30450-30451, 2026-10-30, codice `630d00a`; dettagli in S2 §2.7 e S1 §9). Le decisioni
dell'utente: esclusioni e quarantena mantenute, coperture accettate, generatore invariato; dev al 5% per famiglia, seme
0, gruppi indivisibili.
- **Manifest** `data/manifests/synth-v4-candidato-1`: 60.900 record, 52.457 gruppi.
  - train 54.271 ammessi, dev 2.870, quarantena 67, esclusi 2.792 (nello split test);
  - test 300 e coppie 600, riservati, che condividono 10 gruppi `function`.
- **Loader.** Le sorgenti con `manifest` e `split` leggono solo i record ammessi di quello split, e la validazione è lo
  split dev. Gli shard v4 senza manifest non si leggono. Ogni incoerenza ferma il caricamento.
- **Verifiche (30451).** 32 su 32, comprese le letture reali di ogni split e 10 arresti espliciti.
- **Stato.** Restano incompleti per gli esperimenti l'audit campionario e la decontaminazione esterna, pianificati in S2
  §2.8 e da approvare.

### Da pianificare

Nessuno dei controlli che restano in questa tabella è stato eseguito o è dichiarato superato: quelli eseguiti sono
nella tabella sopra. Dal 2026-10-29 anche le verifiche leggere, che finora sono state eseguite sul frontend, vanno in
un job CPU (CLAUDE.md).

| Collaudo | Scopo | Risorse | Quando |
| --- | --- | --- | --- |
| Sintetici (C1, C2) | Eseguiti: test (job 30392, 30417, 30448), controllo sugli shard (job 30393, 30418, 30448), coppie v4 (job 30419), analisi degli scarti (job 30400, 30420), render e candidato v4 (job 30447-30449), manifest e dev split (job 30450-30451). Da fare, con l'ok dell'utente: audit visivo (piano in S2 §2.8: campione casuale di 380 unità e circa 170 casi mirati, rubrica, C8 prima del limite) e decontaminazione esterna di v4 (inventario e passi in S2 §2.8, dopo D6) | estrazione dell'audit e scansione: pochi minuti di CPU ciascuna | con l'ok dell'utente |
| Contratto ed estrattore (C3, C4; punto 1) | Tokenizzazione e label per modalità; quattro forme e terminazione; tassonomia ufficiale; risposta diretta di Violetto | CPU 20 min; GPU 1 ora | dopo D3 e S3 |
| Manifest, indice e reader (C5-C7; punto 3) | Per i sintetici v4 eseguiti (job 30450-30451, S1 §9). Restano: indice globale e chiavi storiche (C6), varianti e contratto (dopo C3), contatori per compito, le altre sorgenti | CPU 20 min; indice CPU 4 CPU 24G, durata da misurare | dopo S1, D6 |
| Stime (C8) | Copertura esatta, limite ipergeometrico, estrazioni | CPU 20 min | dopo S4 |
| MAVIS (C9; punto 2) | `shape_type` con opzioni; valori visualizzati ed esatti | CPU 20 min; controllo CPU su tutti gli shard | dopo C5 |
| Più immagini (punto 4) | Ordine, budget, parità con cache | CPU 20 min; GPU 1 ora | dopo i punti 1, 3 |
| Encoder (punto 5) | Refactoring SigLIP2 identico; parità dell'encoder B col percorso nativo; rami disattivati uguali al riferimento | GPU 1 ora ciascuno | (a) dopo l'ok; (b) dopo D1 |
| ViT addestrabile (punto 6) | Gradienti, variazione dei pesi, salvataggio, ripresa | GPU 64G 1 ora | dopo il punto 5 (a) |
| Valutatore visivo (punto 7) | Correttezza del codice di valutazione | GPU ≤ 2 ore | dopo C4 |

## 5. Fuori dal primo perimetro

F4 (trainer RL, rollout, log-probabilità, GRPO), interpolazioni F3-M/F4-M, certificazione statistica generale e nuove
generazioni di target non sono prerequisiti di F0 (pipeline §5.10, incarico §5). Le implementazioni di DeepStack, MUDD e
value embeddings visivi seguono le loro specifiche e l'ordine di §6.5, dopo il confronto iniziale degli encoder.
