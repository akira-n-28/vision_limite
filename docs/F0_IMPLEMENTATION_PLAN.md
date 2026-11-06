# Piano di implementazione F0

Stato al 2026-11-05, [pipeline v1.5](../PIPELINE_ADDESTRAMENTO_LIMITE_VL.md) (§§3, 5, 6, 17). Il confronto è fra il
codice del repository sul cluster e i requisiti della pipeline.

**Stato attuale degli interventi (§3):**
- **C1-C2, sintetici:** implementati e collaudati. Il candidato v4 è generato e controllato (job 30447-30449), e il
  manifest è applicato nei loader con il dev split (job 30450-30451; S2 §§2.6-2.7). Il 2026-11-02 v4 è decontaminata
  contro `pannello-v3`, con il manifest `synth-v4-candidato-2` (job 30517-30518). Il campione per l'audit delle golden
  answer è preparato (job 30522; S2 §§2.9-2.10). Il 2026-11-04 i fogli sono corretti con la stessa selezione, e il
  validatore verifica gli split lungo la catena delle derivazioni (job 30555-30556, `campione-2`, S2 §2.11). La
  revisione non è fatta. Il 2026-11-05 il confronto con un campione precedente (`compare_previous`) lavora per unità e
  variante, col formato del precedente esplicito (`2a2002a`, job 30616, sezione "Confronto dei fogli e stime" in §4).
- **C5 e C7:** implementati in parte, nel sottoinsieme di S1 per i sintetici v4 (S1 §9). Il 2026-11-05 lo stesso
  sottoinsieme dà i manifest candidati di UniMER-1M (`unimer-candidato-1`, job 30624-30625, S2 §4) e di
  MAVIS-Geometry (`mavis-geometry-candidato-1`, job 30626 e 30630, S2 §3), con le evidenze storiche importate e
  ricalcolate; i loader non li applicano ancora.
- **C8, stime:** nucleo implementato e collaudato su dati fittizi il 2026-11-05 (job 30616-30617, S4 §9); non applicato
  agli esiti reali.
- **C3, C4, C6, C9:** non iniziati.
- **Pannello di valutazione (punto 7):** benchmark scaricati e verificati (job 30463). Indice, gruppi e ruoli sono
  fissati in `pannello-v3` (job 30469), prima di qualsiasi valutazione del modello, e descritti in
  `docs/PANNELLO_VALUTAZIONE.md`. v3 corregge sei problemi di v2 (collaudi nel job 30468). Da approvare (D6).
- **Campione candidato di F0:** `docs/F0_CAMPIONE_CANDIDATO.md`. Non è ancora caricabile: mancano contratto,
  varianti, manifest delle altre sorgenti, decontaminazione e verifica visiva (§5 di quel documento).

**Perimetro della prima versione** (pipeline v1.5, decisione dell'utente del 2026-10-30). Comprende:
- matematica/STEM, geometria e formule stampate;
- grafici, tabelle, OCR e documenti a pagina singola;
- visione generale di supporto;
- pochi compiti con più immagini, legati alla matematica e al confronto visivo.

Sono rimandati i documenti a più pagine, il training dedicato al manoscritto (MathWriting) e i video. Il manoscritto
non è un requisito di F0.

**Evidenze storiche:**
- Stato al 2026-10-28, dopo la migrazione alla pipeline v1.4 e la revisione esterna dei push fino a `e81f884`. La
  migrazione aveva spostato solo config, launcher e documenti: per codice e test il repository era identico a
  `9d977b6`.
- Allora nessun intervento era eseguito, e l'unico collaudo era `slurm/tests_cpu.sbatch` (job 30255: 22 file, 775
  controlli, nessun fallimento).

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
| D6 | Ruoli del pannello: quali benchmark pubblici sono sviluppo e quale componente nuova resta test riservato (§§5.5, 15.1) | Punti 3 e 7 | perimetro definitivo e download dei benchmark mancanti decisi dall'utente (2026-10-30); ruoli e gruppi fissati in `pannello-v3` (job 30469) e descritti in [`docs/PANNELLO_VALUTAZIONE.md`](PANNELLO_VALUTAZIONE.md), da approvare nei dettagli (quote, stratificazione, irrisolti) |

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
| Validazione per gruppo esplicito | implementata per i sintetici v4 (2026-10-30, `630d00a`, S1 §9); per UniMER dev per gruppo nel manifest candidato (2026-11-05, job 30625), non ancora letto dai loader; mancante per le altre sorgenti | con un manifest (`manifest.py`, `stream.py`) la validazione è lo split dev del manifest, assegnato per gruppo; senza, resta quella di prima: i primi `val_count` campioni dell'ultimo shard di ogni sorgente |
| Esclusioni per identità di compito/variante | mancante | `stream.py:45`, `74-77`: insieme di chiavi del campione (immagine); per `f2_solution` anche la chiave del problema (`164-165`) |
| QA costruite al volo legate al record ammesso | mancante | `data.py:191-283` (`mavis_qa_sample`, `mavis_f1_sample`, `unimer_line_sample`, …) scelgono domanda o riga durante il caricamento |
| Rietichettature tracciate | da collaudare | `stream.py:79-97` (`apply_relabel`, solo se l'etichetta coincide); `unimer_check/run4/relabel.jsonl` |
| Liste obbligatorie e provenienza | verificato per la bozza F1-clean | `data_lists.py`; job 30051 |
| Deduplicazione trasversale fra sorgenti | mancante | `scripts/split_duplicates.py` lavora per sorgente e usa target identici; `decontam.py` normalizza il testo togliendo simboli: segnali, non prove |
| Esclusioni storiche come evidenze riesaminabili | in parte: `unimer_check/run4` e `unimer_overlap/run2` importate, risolte e ricalcolate nel manifest candidato di UniMER (2026-11-05, job 30625; `split_duplicates/run1` solo come controllo incrociato); mancanti le altre | liste in `data/decontam/p6`, `data/decontam/f1eval`, `data/decontam/text/replay`, `outputs/unimer_check/run4`, `outputs/unimer_overlap/run2`, `outputs/split_duplicates/run1`, `outputs/pixmo_check/run4` (riferite dalla bozza archiviata `f1-clean.json`) |
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

### Punto 4 — Poche immagini e preprocessing (pipeline §§3.2, 3.4, 8; R2). Ordine delle pagine rimandato (v1.5)

| Requisito | Stato | Evidenza |
| --- | --- | --- |
| Processor con liste di immagini | da collaudare | `processing.py:166-196, 204-216`: più immagini ammesse, solo prima del testo; segnaposti concatenati senza riferimenti espliciti |
| Encoder e proiettore per immagine, in ordine | da collaudare | `modeling.py:89-101` (shuffle per immagine, token in ordine), `190-235` (gruppi per lunghezza, risultato in ordine); job 29447 con una immagine per esempio |
| Caricatori con più immagini | mancante | `data.py:67-70`: `Sample.image` singola; `build_example` (`data.py:343-361`) passa `[sample.image]` |
| Posizioni continue fra le immagini dello stesso esempio (le pagine sono rimandate, v1.5) | da collaudare | `collator.py:92-107` azzera le posizioni a ogni `Example`: le pagine dentro lo stesso esempio restano contigue se l'esempio è unico |
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
| Benchmark visivi (MathVista, MathVerse, MathVision, We-Math, DynaMath, Geoperception, VisOnlyQA, OlympiadBench, OCRBench, DocVQA val, ChartQA, MUIRBench, MV-MATH, MMStar) | dati presenti e indicizzati; valutatore mancante | snapshot in `hf/hub`; indice `data/decontam/pannello-v3` con sha256 dei pixel nella convenzione dichiarata e una voce per immagine, verificato voce per voce, gruppi e ruoli in `data/panel/pannello-v3` (job 30469); nessun valutatore del modello VL |
| Dati presenti per i pannelli percettivi | presenti, da selezionare | scene reali: PixMo-Cap (25 parquet su 75) e LLaVA-Pretrain; manoscritto (non richiesto dalla v1.5): UniMER-Test hwe solo diagnostica e le formule CROHME nel training di UniMER (meno dell'1%, non separabili per indice, restano); grafici e tabelle per il training: nessuna sorgente scaricata (`docs/F0_CAMPIONE_CANDIDATO.md`) |
| Dati mancanti | pannello scaricato e verificato (job 30463, [`docs/PANNELLO_VALUTAZIONE.md`](PANNELLO_VALUTAZIONE.md)); training mancante | per il training: grafici e tabelle, OCR e documenti a pagina singola, poche immagini (proposte minime in [`docs/F0_CAMPIONE_CANDIDATO.md`](F0_CAMPIONE_CANDIDATO.md)); documenti a più pagine e manoscritto rimandati (v1.5) |
| Valutatori mancanti | mancante | nessun valutatore del modello VL per i benchmark visivi, per più immagini o per i documenti; per le formule c'è solo `eval_f1_formulas.py` |
| Controfattuali e varianti raggruppate | raggruppate nel pannello; confronti appaiati mancanti | coppie controfattuali dei sintetici in `data/shards/synth/v4/pairs`; varianti di MathVerse, DynaMath e MUIRBench (coppie rispondibile/non rispondibile) nello stesso gruppo in `pannello-v3`, inammissibili insieme; nessun valutatore le appaia ancora |
| Profilo dei costi | parziale | 15.4 mila token/s con decoder addestrabile a 32k, 12.9 mila a 4k (job 29402, ViT congelato); ViT addestrabile ed encoder B non misurati |

**Intervento minimo.**
- **Manifest del pannello,** fissato prima dell'ammissione definitiva dei dati. ID, ruolo e gruppo sono in
  `pannello-v3`, con l'ammissione alla metrica primaria; mancano la metrica (D4) e le varianti del contratto (C3).
- **Valutatore per capacità del modello VL:** generazione a batch (`generation.py`), estrattori di S3, risultati
  appaiati per problema e intervalli.
- **Sorgenti minime per le capacità mancanti** (§6.1): scene reali, grafici e tabelle, documenti a pagina singola,
  poche immagini; il manoscritto non è richiesto (v1.5). Le proposte sono in `docs/F0_CAMPIONE_CANDIDATO.md` §4, e i
  download vanno approvati.
- **Preventivo:** un job di profilo per ciascuna configurazione implementata (token/s, memoria, secondi per passo) e la
  formula di §16.
**Collaudo.** CPU: costruzione del manifest del pannello e decontaminazione contro il training. GPU: una valutazione di
prova del valutatore visivo su Violetto con proiettore casuale e con un adapter storico, solo per validare il codice
(1 GPU, 8 CPU, 48G, ≤ 2 ore).
**Dipende da:** D2, D4, D6; punto 2 (estrattori).

## 3. Prossimi interventi al codice, in ordine (elenco minimo)

Ognuno è piccolo, ha i suoi collaudi CPU in `slurm/tests_cpu.sbatch` e non tocca dataset o checkpoint esistenti. I job
segnati vanno approvati uno per uno. Gli interventi dei punti 4-7 (più immagini, encoder, trainer, valutatore) vengono
dopo questi.

Il prossimo lotto minimo, indipendente dalle decisioni aperte, è in `docs/F0_CAMPIONE_CANDIDATO.md` §6:
1. verifica visiva dei gruppi inammissibili del pannello;
2. decontaminazione di v4 contro `pannello-v3`: fatta il 2026-11-02 (job 30517-30518); resta la verifica visiva dei
   264 candidati;
3. manifest delle ammissioni di UniMER e MAVIS (estensione di C5 e C7), il lotto indicato dall'utente il 2026-11-05:
   fatti tutti e due (`unimer-candidato-1`, job 30624-30625; `mavis-geometry-candidato-1`, job 30626 e 30630);
4. scansione delle sorgenti locali contro il pannello.

C3 e C4 seguono D3.

| # | Intervento | File | Collaudi | Job necessario | Dipende da |
| --- | --- | --- | --- | --- | --- |
| C1 | Sintetici: un solo dominio visibile per `intersections` (conteggio geometrico e ammissibilità separati); niente domanda per grafici che si sfiorano senza incrociarsi; domande di classificazione con opzioni (triangoli) o classe più specifica (quadrilateri, `label_type`); shapes senza figure vicine né fuori dagli assi (col tratto), con nuova estrazione solo per le scene che violano i vincoli; layout degli oggetti disegnati nel JSON da v4, con lo sha256 del PNG del campione (`png_sha256`); `version=3` riproduce v3. **Implementato il 2026-10-28 (`38af82a`), corretto il 2026-10-29 (figure tagliate; digest del PNG dopo la revisione di `f7d32d3`); suite CPU passata (job 30392), controllo sugli shard eseguito (job 30393, §4), scarti analizzati (job 30400, S2 §2.5); correzioni minime di v4 in `b0b90c5` (coppie per riscalatura, punto segnato lontano dalle misure, tacche -1 senza numero), con suite e controlli passati (job 30417-30420); render di v4 eseguito il 2026-10-30 su incarico dell'utente (job 30447, `data/shards/synth/v4`), candidato controllato (job 30448-30449, S2 §2.6)** | `src/limite_vl/synth.py`, `scripts/synth_render.py` (`--generator-version` obbligatorio), `slurm/synth_render.sbatch` | `tests/test_synth.py` (S2 §5) | render di v4: CPU, 8 CPU, 16G, 6 min 18 s (job 30447) | S2 |
| C2 | Verifica semantica indipendente dei sintetici: ricalcolo per tipo dagli oggetti disegnati (layout) con predicati propri, intersezioni solo nei tratti disegnati; invarianti del renderer, compreso il taglio sull'area effettiva degli assi; violazioni bloccanti che impediscono la verifica del campione; nel report esiti separati per semantica della scena, corrispondenza col PNG dello shard e campione (S2 §2.4). **Implementato il 2026-10-28 (`38af82a`), corretto il 2026-10-29, esiti per campione dopo la revisione di `f7d32d3`; suite CPU passata (job 30392), controllo sugli shard eseguito (job 30393, §4); coordinate lette dalle tacche vicine (`b0b90c5`, S2 §2.5), controlli passati (job 30417-30420); collaudi a 10.000 scene e controllo del candidato v4 sugli shard salvati (`284e8f1`, job 30448-30449: integrità, riproducibilità, checker, coppie, duplicati e split)** | `src/limite_vl/synth_check.py`, `scripts/synth_semantics_check.py` (anche `--cases`), `slurm/synth_semantics_check.sbatch`, `scripts/synth_case_svg.py` (schemi dei casi), `scripts/synth_candidate.py`, `slurm/synth_candidate.sbatch` | mutazioni ed esiti per campione (`test_sample_evidence`) e candidato (`test_candidate`) in `tests/test_synth.py`; su v3 deve ritrovare i difetti noti | controllo di v3 e di scene v4 in memoria (CPU, 8 CPU, 16G, ≤ 1 h 30); candidato v4: 8 CPU, 16G, 9 min 38 s | C1 |
| C3 | Contratto `vl-contract/1`: oggetto versionato, template `limite-vl/1`, processor che lo riceve e lo registra | `src/limite_vl/contract.py` (nuovo), `processing.py` | S3 §6 (tokenizzazione delle due modalità, system prompt canonico identico ai token di Violetto) | — | D3 |
| C4 | Estrattore e verificatori: forma, completezza dai metadati, candidati, tre esiti separati; `generate_batch` restituisce anche `max_new_tokens` e il motivo d'arresto | `src/limite_vl/answers.py` (nuovo), `generation.py` | S3 §6, compresi i casi A-T della tassonomia ufficiale | — | C3 |
| C5 | Identità e manifest: JSON canonico, digest completi, mappa delle collisioni, record, scrittura immutabile; i loader restituiscono la costruzione canonica. **In parte, per i sintetici v4 (2026-10-30, `630d00a`, S1 §9): JSON canonico, digest e collisioni, `record_id`, `group_id`, record, split per gruppo, scrittura immutabile (`data/manifests/synth-v4-candidato-1`, job 30451); mancano `variant_id`, `uid` e la costruzione canonica dai loader, che dipendono da C3. Il 2026-11-05 manifest candidato di UniMER-1M con lo stesso sottoinsieme, più trasformazioni e genitori (righe ritagliate, rietichettature), compiti, evidenze per lista e rango del campione candidato (`data/manifests/unimer-candidato-1`, job 30624-30625, S1 §9, S2 §4); poi quello di MAVIS-Geometry, con stato per compito (`shape_type` in quarantena) e coerenza con le annotazioni separata dalla fedeltà all'immagine (`data/manifests/mavis-geometry-candidato-1`, job 30626 e 30630, S2 §3)** | `src/limite_vl/manifest.py`, `scripts/synth_manifest.py`, `scripts/unimer_manifest.py`, `scripts/mavis_manifest.py`, `data.py` | S1 §8 (collisione forzata, canonicalizzazione, prefissi, contratto): `tests/test_manifest.py` tranne il contratto | — | C3 (per la costruzione canonica) |
| C6 | Indice globale e import delle esclusioni storiche | `scripts/build_index.py` (nuovo) | S1 §8 (gruppi, chiavi storiche) | costruzione dell'indice (CPU, 4 CPU, 24G, durata da misurare su uno shard per sorgente) | C5, D6 |
| C7 | Reader col manifest, validazione dal manifest, contatori per sorgente e compito (`too_long` compreso). **In parte (2026-10-30, `630d00a`): solo i record ammessi dello split, validazione dal manifest, sha256 di JSON e PNG controllati, contatori dei reader e degli esempi caricati per sorgente, shard v4 leggibili solo col manifest; verifiche sui dati reali nel job 30451 (`scripts/manifest_check.py`); mancano i contatori per compito e `too_long` per sorgente** | `stream.py`, `collator.py`, `data_lists.py` | S1 §8 (reader): `tests/test_manifest.py` | verifiche del manifest (CPU, 8 CPU, 16G, 4 min) | C5 |
| C8 | Stime: limite ipergeometrico e binomiale, estrazione con probabilità registrate, Bonferroni, rifiuto se unità e quantità non coincidono. **Nucleo implementato il 2026-11-05, dopo l'adozione del nucleo tecnico di S4 da parte dell'utente: limiti esatti, Bonferroni, compatibilità fra disegno, quantità e popolazione, esiti contati senza perdite, tre livelli del risultato (S4 §9). Collaudato su dati fittizi (job 30616-30617), non applicato agli esiti reali. Senza campionatori: l'estrazione con probabilità registrate resta quella di `synth_gold_sample.py`** | `src/limite_vl/estimate.py` (nuovo) | S4 §8 tranne l'estrazione: `tests/test_estimate.py` | test mirati: CPU, 2 CPU, 4G, 2-3 s (job 30616-30617) | S4 |
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
| 30463 (2026-10-30) | `slurm/verify_hashes.sbatch downloads/manifests/p10-pannello.tsv`, dopo `fetch_manifest.sh` sul frontend (kit, limiti della guida): OCRBench, DocVQA val, ChartQA val e test, MUIRBench, MV-MATH, MMStar, GSM8K test, IFEval | download: 30 file, 1.680 MiB, nessuno già presente, nessun errore; verifica: 30 file su 30 ok | che i benchmark del perimetro sono presenti e integri alle revisioni fissate. Nulla sulla loro qualità |
| 30464 (2026-10-30) | `slurm/panel_inventory.sbatch run2` su `59c27cd`: inventario dei benchmark riservati, compresi i nuovi | 58 s; 26 repository, tutti i file del manifest presenti e verificati; schema e righe in `outputs/panel_inventory/run2` | struttura e campi dei benchmark su cui si fondano gruppi e ruoli |
| 30465 (2026-10-30) | `slurm/panel_groups.sbatch pannello-v1` su `59c27cd` | 2 min 24 s; superato dal job 30466: le pagine dello stesso documento UCSF di DocVQA non erano raggruppate | niente: mai usato, i file restano |
| 30466 (2026-10-30) | `slurm/panel_groups.sbatch pannello-v2` su `e084cba`: indice del pannello e gruppi e ruoli con seme 0 | 2 min 30 s. Indice: 99.713 voci, 92.658 immagini, nessuna illeggibile. 85.991 elementi in 51.363 gruppi. DocVQA per documento UCSF, senza documenti in due ruoli; 1.300 coppie di MUIRBench insieme; 1.299 irrisolti nel test, 191 forti (`docs/PANNELLO_VALUTAZIONE.md` §§3, 5) | che criteri, seme e assegnazione sono fissati prima di qualsiasi valutazione e che i legami confermati sono applicati. Non dice quali candidati siano duplicati: serve la verifica visiva |
| 30468 (2026-10-30) | `slurm/tests_cpu.sbatch` prima di `pannello-v3` (codice poi committato in `c9765ca`) | 25 file, 990 controlli, nessun fallimento, 11 min 3 s; `tests/test_panel.py` 53 controlli: irrisolti per gruppo, testo conservativo, vicini pHash asimmetrici e dello stesso gruppo, PNG trasparente, recupero dal nuovo indice, train di ChartQA escluso | che le sei correzioni del pannello superano le regressioni scritte e che i 937 controlli precedenti non cambiano |
| 30469 (2026-10-30) | `slurm/panel_groups.sbatch pannello-v3 --precedente pannello-v2` su `c9765ca` | 5 min 34 s. Indice con convenzione `rgb-su-bianco/1`, verifica di 92.658 voci su 92.658. 51.386 gruppi; pHash completo, senza troncamenti; 1.300 evidenze dirette in 743 gruppi inammissibili; riesame delle 26 unioni testuali di v2 (3 confermate, 23 candidate); confronto con v2 (`docs/PANNELLO_VALUTAZIONE.md` §8) | che l'assegnazione corretta è fissata con gli stessi criteri e lo stesso seme. Non dice quali candidati siano duplicati |
| 30517 (2026-11-02) | `slurm/synth_decontam.sbatch controllo controllo-1` su `8786afb` più le modifiche poi committate in `fe85a97`: test mirati, scansione delle 300 scene di test v4 contro `pannello-v3`, verifica di compatibilità parziale | 1 min 57 s. Test passati: `tests/test_panel.py`, con 11 controlli nuovi sul confronto fra versioni, `test_decontam.py`, `test_synth_decontam.py` (31) e `test_manifest.py`. Compatibilità 23 su 23; un candidato pHash; nessun problema operativo | che indice, convenzione, perimetro e identificativi sono compatibili prima della scansione completa |
| 30518 (2026-11-02) | `slurm/synth_decontam.sbatch completo completo-1 2` su `fe85a97`: scansione dei 60.900 record v4, verifica completa, manifest derivato, `manifest_check.py` | 5 min 45 s. Nessun pixel identico, nessun candidato testuale, 264 record con candidati pHash. Manifest `synth-v4-candidato-2`: train 53.997, dev 2.864, quarantena 367, test 296, coppie 584. Verifiche 37 su 37 (S2 §2.9) | che la decontaminazione esterna di v4 è applicata nel manifest e nei loader, con split ereditati e nessuna riammissione. Non dimostra contaminazioni, e non ne esclude oltre i controlli eseguiti |
| 30521 (2026-11-02) | `slurm/synth_gold_sample.sbatch campione-1` | fermato in 1 s da `tests/test_synth_gold.py`, per un errore del test, prima di scrivere uscite; ripetuto nel 30522 | niente |
| 30522 (2026-11-02) | `slurm/synth_gold_sample.sbatch campione-1`: `decontam_review.py` sui candidati e `synth_gold_sample.py` sul manifest v2 | 5 min 4 s. Somiglianza dei pixel dei 264 candidati sempre sotto 0,80. Campione di 380 unità e 175 casi mirati, foglio dei candidati con 131 immagini del pannello, inventario dei controlli per proprietà (S2 §2.10) | che il materiale per la revisione è pronto e riproducibile. Nessuna revisione e nessuna stima |
| 30523 (2026-11-02) | `slurm/tests_cpu.sbatch` su `87c4108` | 27 file, 1.052 controlli, nessun fallimento, stderr vuoto, 10 min 54 s. Nuovi: `tests/test_panel.py` +11, `test_synth_decontam.py` 31, `test_synth_gold.py` 20 | che le modifiche di questo lotto superano le regressioni scritte e che i 990 controlli precedenti non cambiano |
| 30555 (2026-11-04) | `slurm/tests_cpu.sbatch` su `6328a29` | 27 file, 1.086 controlli, nessun fallimento, stderr vuoto, 10 min 2 s. Nuovi: `test_synth_gold.py` +22 (prompt del loader con e senza opzioni incorporate, coppia con risposte diverse, undicesima corrispondenza, riconciliazione), `test_synth_decontam.py` +12 (due derivazioni consecutive) | che le quattro correzioni superano le regressioni scritte e che i controlli precedenti non cambiano |
| 30557-30559 (2026-11-04) | `slurm/review_package.sbatch campione-2 synth-v4-campione-2` (30557), poi con `verifica` (30558, 30559) | Pacchetto offline `revisioni/synth-v4-campione-2` (S2 §2.12): 380 + 175 unità, 264 candidati con 755 corrispondenze, 972 immagini con sha256 identici; verifica statica 28 su 28 con il controllo lessicale dei file JS. `jjs` non c'è su `lagrange0` (30558) | che i dati del pacchetto sono quelli di `campione-2` e completi. Non prova l'interfaccia in un browser né esegue il JavaScript: il collaudo è `verifica.html` sul portatile |
| 30562-30566 (2026-11-04) | `slurm/js_engine_probe.sbatch` (30562); `slurm/review_package.sbatch campione-2 synth-v4-campione-2 verifica` (30563-30566) | Sonda: su `lagrange0` c'è Duktape 2.7 come libreria di sistema; Node in `~/.nvm` e Firefox snap restano fuori dal progetto e non si usano. Il JavaScript pubblicato (`a9252b5`), eseguito per la prima volta con Duktape: 58 su 59 controlli, saltata la parte DOM. Correzioni del primo giudizio (S2 §2.12): autotest senza target, unità collegate per record e scena (23 unità), registro delle esposizioni, unione degli stati, formato 2 con migrazione. Esito: senza DOM 97 superati e 0 falliti; su DOM simulato 118 e 0; collegamenti uguali a Python; verifica statica 29 su 29 | che logica e interfaccia si eseguono con Duktape, con le regressioni scritte, sui dati veri. Non è una prova in un browser: il collaudo resta `verifica.html` sul portatile |
| 30568-30571 (2026-11-04) | `slurm/review_package.sbatch campione-2 synth-v4-campione-2 verifica` (30568, 30569, 30571 sul commit) e `… regressione 9ea9af1` (30570) | Correzione dell'indipendenza delle unità già fissate (S2 §2.12, punto 5): blocco e classificazione separati, cronologie unite, stati del formato 2 di `9ea9af1` riconciliati. Senza DOM 112 superati e 0 falliti; DOM simulato 136 e 0; verifica statica 29 su 29. Con il `logica.js` di `9ea9af1` falliscono esattamente i controlli nuovi (11 senza DOM, 14 su DOM simulato) | che i nuovi test rilevano il difetto e che la correzione lo elimina, con Duktape. Nessuna prova in un browser |
| 30556 (2026-11-04) | `slurm/synth_gold_sample.sbatch campione-2 campione-1 campione-1`, dopo il 30555 | 4 min. Test mirati; `manifest_check` di `synth-v4-candidato-2` col validatore corretto, 44 su 44 con la catena 2 → 1; fogli `campione-2`: 755 corrispondenze su 755 riconciliate (18 oltre l'anteprima, in 3 record), 102 immagini del pannello, prompt del loader per 337 unità, soluzioni per variante; selezione identica a `campione-1` (S2 §2.11) | che i fogli sono completi e riproducibili con la stessa estrazione. Nessuna revisione e nessuna stima |
| 30616 (2026-11-05) | `slurm/tests_cpu.sbatch tests/test_synth_gold.py tests/test_estimate.py`, test mirati sulle dipendenze modificate, prima dei commit | 3 s, 2 CPU, 6 s di CPU allocate, stderr vuoto. `test_synth_gold.py` 60 controlli su 60, 18 nuovi su `compare_previous`; `test_estimate.py` 44 su 44. Gli id git dei file provati coincidono con quelli di `2a2002a` | che il confronto per unità e variante e il nucleo di C8 superano le regressioni scritte, su dati fittizi. I campioni reali non sono rigenerati, e C8 non è applicato agli esiti |
| 30617 (2026-11-05) | `slurm/tests_cpu.sbatch tests/test_estimate.py`, sul contenuto del commit di C8 | 2 s, 2 CPU, 4 s di CPU allocate: 44 su 44, compresi i numeri dell'esempio minimo (S4 §9) | come sopra, per la versione committata del test di C8 |
| 30622 (2026-11-05) | `slurm/tests_cpu.sbatch tests/test_synth_gold.py`, correzione delle tabelle vuote (`1002c94`) | 2 s, 2 CPU, 4 s di CPU allocate: 66 su 66, 6 nuovi | che scrittura e confronto usano le stesse colonne canoniche anche senza righe, e che i file vuoti del vecchio scrittore si accettano solo senza righe attese |
| 30624 (2026-11-05) | `slurm/unimer_manifest.sbatch prova unimer-candidato`: test, poi un tar per insieme degli shard densi | 42 s, 4 CPU, 2 min 48 s di CPU allocate. Test 15 su 15. 200.000 formule in 34,8 s (5.747 al secondo), sha256 uguali allo sharding; RSS 487 MiB; stima della lettura completa circa 50 s | il dimensionamento della costruzione completa |
| 30625 (2026-11-05) | `slurm/unimer_manifest.sbatch costruzione unimer-candidato 1` (24G): test, lettura verificata dei 10 tar, riconciliazione, scrittura e verifica | 3 min 53 s, 4 CPU, 15 min 32 s di CPU allocate, RSS massima 7,6 GiB nella rilettura. `data/manifests/unimer-candidato-1` (262 MiB più 14 MiB di chiavi): 985.450 formule e 30.685 righe ritagliate in 933.870 gruppi; run4 e run2 riprodotti; verifica 21 su 21 (S2 §4) | che il candidato è riproducibile, con evidenze riconciliate, gruppi indivisibili e trasformazioni coi genitori. Non certifica le formule, non è letto dai loader, la decontaminazione esterna manca |
| 30626 (2026-11-05) | `slurm/mavis_manifest.sbatch prova mavis-geometry-candidato`: test, poi un tar | 37 s, 4 CPU, 2 min 28 s di CPU allocate. Test 12 su 12, col run2 del fixture scritto da `f1_mavis_check.py`. 19.000 record in 26 s, sha256 uguale allo sharding, 22,6 domande per record; RSS 212 MiB; stima della lettura completa 70-80 s | il dimensionamento della costruzione completa |
| 30630 (2026-11-05) | `slurm/mavis_manifest.sbatch costruzione mavis-geometry-candidato 1`: test, lettura verificata degli 11 tar, riconciliazione, scrittura e verifica | 2 min 23 s, 4 CPU, 9 min 32 s di CPU allocate, RSS massima 3,9 GiB. `data/manifests/mavis-geometry-candidato-1` (148 MiB): 199.050 record ammessi in 158.018 gruppi, nessun conflitto di verità, `shape_type` in quarantena in 177.944 record; run2 riprodotto coi 60 esempi; verifica 16 su 16 (S2 §3) | che il candidato è riproducibile e coerente con le annotazioni, con gruppi indivisibili e stato per compito. Non dice nulla sulla fedeltà all'immagine, non è letto dai loader, la decontaminazione esterna manca |
| 30633 (2026-11-05) | `slurm/tests_cpu.sbatch` con `test_unimer_manifest.py`, `test_mavis_manifest.py` e `test_manifest_recheck.py`, prima del commit `c271d6e` | 22 s, 4 CPU, 1 min 28 s di CPU allocate, stderr vuoto: 29 su 29, 14 su 14, 8 su 8. Gli id git del contenuto provato coincidono col commit | che le lacune del verificatore sono chiuse sui fixture: PNG della riga scambiato rifiutato prima dei gruppi, fasce non valide rifiutate (20 casi), stati alterati rifiutati dai soli controlli attesi, quota zero e semi consecutivi con lo stesso dev accettati |
| 30634 (2026-11-05) | `slurm/manifest_recheck.sbatch unimer unimer-candidato-1 bcc85198… riverifica-1`: test mirati, poi riverifica in sola lettura | 1 min 43 s, 2 CPU, 3 min 26 s di CPU allocate, RSS 7,5 GiB. 28 controlli su 28, nessuna anomalia: verifica corrente 23 su 23 (stato atteso per 1.016.135 record su 1.016.135); 30.685 righe su 30.685 contro lo shard delle righe e la formula genitore, coi digest del manifest; 8 file di evidenze e 5 shard di train (solo dimensione) uguali a quelli registrati (S2 §4) | che il manifest scritto rispetta i controlli nuovi: stesso PNG della formula, fasce strutturalmente valide, stati delle contaminazioni. Non prova la corrispondenza visiva delle fasce |
| 30635 (2026-11-05) | `slurm/manifest_recheck.sbatch mavis mavis-geometry-candidato-1 dc0f264c… riverifica-1` | 42 s, 2 CPU, 1 min 24 s di CPU allocate, RSS 3,8 GiB: 15 controlli su 15, nessuna anomalia, 10 file di evidenze uguali a quelli registrati (S2 §3) | che il manifest scritto supera la verifica senza il controllo sul cambio di seme |
| 30638 (2026-11-05) | `slurm/tests_cpu.sbatch tests/test_manifest_recheck.py`, prima del commit della correzione di `scripts/manifest_recheck.py` | 7 s, 4 CPU, stderr vuoto: 25 controlli su 25, di cui 17 nuovi. Gli id git del contenuto provato coincidono col commit | che sui fixture la riverifica rifiuta lo shard delle righe classificato `train` coi record invariati, e le evidenze incomplete: `liste_importate` assente o vuota, lista o file non registrati, digest mancante o non valido, per UniMER e MAVIS. Accetta i manifest completi e quello senza righe ritagliate. Il job non legge i manifest reali (S2 §§3-4) |
| 30639 (2026-11-05) | `slurm/manifest_recheck.sbatch unimer unimer-candidato-1 bcc85198… riverifica-2`, coi controlli di `aed2e2b` | 1 min 45 s, 2 CPU, 3 min 30 s di CPU allocate, RSS 7,3 GiB. Test mirati 54 su 54. Classificazione degli shard coerente coi record: 5 shard `train` con 985.450 formule, 1 shard `righe` con 30.685 righe. 8 evidenze obbligatorie registrate e uguali su disco, con gli stessi digest di `riverifica-1`. 28 controlli su 28, nessuna anomalia: 30.685 righe su 30.685, ricavate dai record (S2 §4) | che il manifest scritto supera anche i controlli nuovi. Non prova la corrispondenza visiva delle fasce e non completa la decontaminazione |
| 30640 (2026-11-05) | `slurm/manifest_recheck.sbatch mavis mavis-geometry-candidato-1 dc0f264c… riverifica-2` | 44 s, 2 CPU, 1 min 28 s di CPU allocate, RSS 3,8 GiB. Test mirati 39 su 39. 10 evidenze obbligatorie registrate e uguali su disco, con gli stessi digest di `riverifica-1`. 15 controlli su 15, nessuna anomalia (S2 §3) | come sopra. Non prova la fedeltà all'immagine |
| 30643-30645 (2026-11-06) | `slurm/tests_cpu.sbatch tests/test_panel_decontam.py`, test nuovo della decontaminazione esterna di UniMER e MAVIS | 30643 fermato da un errore del test stesso, 30644 con un'aspettativa sbagliata del test (50 su 51), 30645 superato (50 controlli). 8-13 s, 4 CPU | che sui fixture identità dei pixel, candidati, formule, identità fra formula e riga ritagliata, gruppi, compiti, arresti e verifica della versione nuova funzionano |
| 30646 (2026-11-06) | `slurm/panel_decontam.sbatch prova unimer prova-unimer-1 4000` (6 CPU): primi 4.000 campioni per shard | 1 min 42 s. Test 50 su 50. 24.000 record in 44 s, RSS 1,2 GiB; nessun pixel identico; 5 formule lunghe uguali a UniMER-Test (già quarantene di run2), 3.070 record con formule brevi uguali (evidenza senza effetto); verifica parziale 25 su 25, run2 riconciliato per record | il dimensionamento della scansione completa di UniMER (circa 13 min con 6 CPU) |
| 30647 (2026-11-06) | `slurm/panel_decontam.sbatch prova mavis prova-mavis-1 1500` (6 CPU) | 6 min 30 s. Test 50 su 50. 16.500 record in 353 s, RSS 3,0 GiB; verifica parziale 22 su 22. **Esplosione di candidati poco informativi:** 1.705 record legati per un 10-gram a un solo problema di AIME 2025 | che il metodo andava corretto prima della scansione completa. Allora con una regola di frequenza della sorgente, abolita nel job 30676 (S2 §3, `docs/PANNELLO_VALUTAZIONE.md` §6.2) |
| 30648 (2026-11-06) | `slurm/tests_cpu.sbatch tests/test_panel_decontam.py` con la regola di frequenza della sorgente (abolita) | 53 su 54: un caso del test costruito senza dettagli testuali, corretto | che la regola di allora supera i controlli scritti |
| 30649 (2026-11-06) | `slurm/panel_decontam.sbatch prova mavis prova-mavis-2 1500` (8 CPU), stessi record di 30647 | 5 min 2 s. Test 54 su 54. 16.500 record in 269 s; verifica parziale 23 su 23. 10-gram ignorati per frequenza: 2 sequenze di lettere dei punti della descrizione del loader (`a b c d e f g h i j`), senza contenuto in comune col problema di AIME 2025. Candidati da 1.767 a 62 record (58 pHash, 8 testo verso MathVision 753) | il ritmo con 8 CPU (circa 1 ora per tutta la sorgente). Non prova che la frequenza distingua le frasi generiche: la regola è abolita (job 30676) |
| 30651 (2026-11-06) | `slurm/panel_decontam.sbatch completo unimer completo-unimer-1 2` (4 CPU, 48G) su `861aba8` | 27 min 47 s; scansione di 1.016.135 record in 19 min, RSS massima 20,3 GiB nella verifica della versione nuova. Shard e zip con sha256 uguali, nessun controllo mancante. Confermate: 2 record, l'immagine di UniMER-Test di run2; candidati: le 404 formule lunghe di run2; formule brevi senza effetto 152.413 (UniMER-Test) e 4 (OCRBench). run2 riconciliato per record. `unimer-candidato-2` senza cambi di stato; compatibilità 26 su 26, versione nuova 20 su 20 (S2 §4) | che la decontaminazione esterna di UniMER, nei controlli dichiarati, non trova altro oltre run2. Non dimostra l'assenza di ogni contaminazione |
| 30652 (2026-11-06) | `slurm/panel_decontam.sbatch completo mavis completo-mavis-1 2` (8 CPU, 48G) su `861aba8` | 56 min 9 s; scansione di 199.050 record in 53 min, RSS massima 8,8 GiB. Nessun pixel identico; candidati in 664 record e 109 gruppi (pHash MathVision 587, MV-MATH 119, We-Math 12, MathVerse 8, MathVista 1; testo 5); 3 10-gram ignorati per la sola frequenza, 20.116 record con sole corrispondenze ignorate. `mavis-geometry-candidato-2`: 693 record in quarantena per gruppo, `shape_type` invariata; compatibilità 24 su 24, versione nuova 21 su 21 (S2 §3) | che i candidati sono in quarantena per gruppo, con split ereditati e nessuna riammissione. Non li risolve e non prova la fedeltà all'immagine. Le decisioni testuali, prese con la regola di frequenza, sono riesaminate nel job 30676 |
| 30674-30675 (2026-11-06) | `slurm/tests_cpu.sbatch tests/test_panel_decontam.py`, correzione del filtro testuale di MAVIS (regole esplicite `panel-decontam/2`, comando `riesame`) | 30674: 89 su 90, un'aspettativa sbagliata del test (riammissione provata su un record ammesso anche nel genitore); 30675: 90 su 90. 15 s | che sui fixture un problema specifico ripetuto in più di 25 record resta candidato, con duplicati e varianti nello stesso gruppo, e che lo stesso test fallisce col filtro di `f163a81`. Inoltre: l'enumerazione dei punti è coperta solo nella descrizione del loader legata al manifest; il riesame di un candidato-2 costruito col codice vecchio dà una versione verificata; le alterazioni sono rifiutate |
| 30676 (2026-11-06) | `slurm/panel_decontam.sbatch riesame mavis riesame-1 3` (4 CPU, 48G) | 7 min 25 s, RSS massima 10,6 GiB. Test 90 su 90; candidato-2 riverificato con la sua regola storica (23 su 23). Lette solo intestazioni e 20.121 JSON (251 MiB su 9,2 GiB, 133 s); 20.071 descrizioni legate al manifest. Enumerazioni coperte; 50 record verso MathVision 753 candidati (47 gruppi). `mavis-geometry-candidato-3`: 1.591 record da ammesso a quarantena (1.406 in due gruppi di figure identiche); verifica della catena 25 su 25 (S2 §3) | che la frequenza non decide più nulla e che le decisioni nuove sono ricalcolabili dalla scansione e dalle descrizioni. Non risolve i candidati |

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

Decontaminazione esterna e preparazione dell'audit (job 30517-30523, 2026-11-02, codice `fe85a97` e `87c4108`;
dettagli in S2 §§2.9-2.10):
- **Scansione** dei 60.900 record contro `pannello-v3`, con identità dei pixel, pHash entro 4 bit e 10-gram delle
  domande. Esiti:
  - 0 pixel identici, 0 candidati testuali, 0 record non verificabili;
  - 264 record con candidati pHash, tutti con somiglianza dei pixel sotto 0,80.
- **Manifest** `data/manifests/synth-v4-candidato-2`, con gli split ereditati e la quarantena per gruppo dei candidati.
  - train 53.997 ammessi, dev 2.864, quarantena 367, esclusi 2.792;
  - test 296 e coppie 584 ammessi.

  Verifiche 37 su 37, comprese le 5 nuove sulle versioni derivate.
- **Campione** delle golden answer: 380 unità e 175 casi mirati, inventario dei controlli per proprietà, foglio dei
  264 candidati. Nessuna revisione è avviata.
- **Correzioni del 2026-11-04** (job 30555-30556, codice `6328a29`, S2 §2.11), con la stessa selezione:
  - il foglio dei candidati prende tutte le corrispondenze dalla scansione completa, 755 su 755 riconciliate;
  - il primo passaggio mostra il prompt del loader;
  - le coppie hanno soluzioni ed esiti per variante;
  - `manifest_check` verifica gli split risalendo la catena delle derivazioni fino alla versione che li assegnò.

  I fogli da usare sono quelli di `campione-2`. Per la revisione sul portatile c'è il pacchetto offline
  `revisioni/synth-v4-campione-2` (job 30557-30566, S2 §2.12), con l'autotest `verifica.html`; gli esiti tornano in
  `revisioni/synth-v4-campione-2/risultati/`.
  - La soluzione di un'unità si mostra solo dopo i primi giudizi delle unità sullo stesso record o sulla stessa scena,
    che restano fissati.
  - Le esposizioni si registrano e non si perdono in importazioni e ripristini; il file di avanzamento è nel formato 2,
    con migrazione dal formato 1.
  - Logica e interfaccia sono eseguite con Duktape su un DOM simulato; nessuna prova in un browser. L'indipendenza
    dipende dalle date di salvataggio e di prima esposizione anche per le unità già fissate (job 30568-30571).
- **Stato.** Ancora incompleto per gli esperimenti: mancano l'audit visivo, la verifica dei candidati, le varianti e il
  contratto. Decontaminazione superata, golden answer verificate e dati pronti per F0 sono tre condizioni diverse, e
  nessuna delle ultime due è raggiunta.

Confronto dei fogli e stime (job 30616-30617, 2026-11-05), mentre l'utente fa la revisione di `synth-v4-campione-2`.
La revisione, il pacchetto, i fogli, la selezione e il manifest non sono toccati.
- **`compare_previous`** (`scripts/synth_gold_sample.py`, commit `2a2002a`). Il difetto: il confronto assumeva sempre
  il formato di `campione-1`, con le coppie casuali aggregate in una riga e le coppie mirate con la sola variante a.
  Due fogli del formato attuale, che ha una riga per variante, risultavano quindi diversi anche se identici, e la
  variante b non si confrontava.
  - *Implementazione:* il formato del precedente viene dallo schema del suo `report.json`, controllato con le colonne
    dei file. Nel formato attuale soluzioni e opzioni si confrontano per (unita, variante), con record e record_id
    associati, risposta, esito del checker, stato e stato della scena. Ogni foglio si controlla anche da solo: righe
    ripetute, varianti mancanti o inattese, record sbagliato. Dal formato 1 sono ammesse solo le trasformazioni
    previste: le risposte aggregate si separano per variante, e le varianti b assenti si dichiarano aggiunte senza
    confronto, quindi non certificate (`confronto_completo` falso). Restano i controlli su selezione, ordine, ranghi,
    manifest e immagini.
  - *Collaudo:* 18 controlli nuovi su fogli fittizi (job 30616).
  - *Tabelle vuote* (correzione residua dopo la revisione di `50ef2c6`, job 30622). Senza righe e senza colonne date,
    `write_table` scriveva la sola intestazione `unita`, mentre il confronto chiede quelle complete: due campioni
    identici senza soluzioni né opzioni risultavano incompatibili. Ora scrittura e confronto usano le stesse colonne
    canoniche (`SHEET_COLUMNS`, scritte da `write_sheet` anche per le tabelle vuote), e un campo fuori dalle colonne è
    un errore. Un file vuoto del vecchio scrittore si accetta, e si dichiara in `formati.tabelle_vuote_senza_colonne`,
    solo se le unità non attendono righe; altrimenti, o se ha righe, è un file incompleto. 6 controlli nuovi scrivono
    i fogli col percorso di `main` (66 su 66).
  - *Applicazione ai fogli reali:* nessuna; il confronto nuovo si userà alla prossima generazione di fogli. Il
    confronto già scritto fra `campione-1` e `campione-2` (job 30556) andava dal formato 1 a quello attuale, il caso che
    il codice vecchio trattava correttamente, e resta valido per ciò che confrontava. Le 20 risposte b delle coppie
    mirate e le opzioni b delle 50 coppie non erano nel formato 1, quindi non sono certificate invariate (S2 §2.11).
- **C8** (`src/limite_vl/estimate.py`, S4 §9).
  - *Implementazione:* limite ipergeometrico esatto e Clopper-Pearson, Bonferroni, compatibilità fra popolazione, unità
    estratta e quantità, esiti contati senza perdite, tre livelli del risultato. Non ci sono campionatori, aggregazioni
    di strati, soglie né certificazioni.
  - *Collaudo:* 44 controlli su dati fittizi (job 30616-30617): enumerazione esatta, copertura esatta, casi limite,
    controesempio della S4 §4, Bonferroni, rifiuti.
  - *Applicazione agli esiti reali:* non fatta. Servono la revisione completa e la corrispondenza fra i valori del
    pacchetto e gli esiti di C8, da fissare con l'utente prima di guardare gli esiti (proposta in S4 §9). Gli esiti
    parziali non scelgono soglie né cambiano il disegno.
  - *Esempio minimo,* con dati fittizi (S4 §9):

    ```python
    from limite_vl import estimate as est

    d = est.Design("esempio-fittizio", "A", "qa", est.UNIFORM, 10_000, 300)  # N = 10.000 QA, n = 300 estratte
    c = est.Claim("correttezza", "quota_qa_errate", "esempio-fittizio", "A", 10_000)
    k = est.Counts(approvato=294, errato=2, incerto=1, non_giudicabile=1, mancante=1, approvato_non_indipendente=1)
    r = est.estimate(c, d, k)  # r["limite"]: k = 6, D_U = 388, valore 0.0388; r["conteggi"]: quota non risolta 3/300
    ```

    Con 2 errori confermati il limite usa k = 6, perché incerti, non giudicabili, mancanti e approvazioni non
    indipendenti contano come errori. Il risultato: al più 388 QA su 10.000 non sarebbero approvate (3,88%, al 95%,
    valido per il disegno). Lo stesso numero vale per le QA errate solo se il revisore non approva errori. Con un
    secondo strato simultaneo (Bonferroni, 1/40 ciascuno) il limite sale a 426. Una stima della quota di QA da un
    campione di coppie è rifiutata.

Manifest candidati di UniMER e MAVIS (lotto di `docs/F0_CAMPIONE_CANDIDATO.md` §6, punto 3), con il sottoinsieme di S1
dei sintetici. Organizzano candidati ed evidenze: non certificano la qualità, non assegnano gold e non autorizzano il
training; i loader non li applicano ancora (C7).
- **UniMER-1M** (`scripts/unimer_manifest.py`, job 30624-30625, S1 §9, S2 §4).
  - *Ingressi verificati:* shard densi di `2343ddd9` letti e hashati una volta, con sha256 uguali allo sharding (job
    29729) e allo snapshot di Hugging Face.
  - *Record:* 985.450 formule e 30.685 righe ritagliate. Le righe hanno lo stesso record_id della formula,
    trasformazione `righe` e genitore. UniMER-Test non entra: serve solo a riconoscere le contaminazioni.
  - *Gruppi:* stessi byte del PNG, con le righe nel gruppo della loro formula.
  - *Stati:* da run4 per record (errori certi esclusi, incerti in quarantena, rietichettature tracciate) e da run2 per
    gruppo (stessa immagine del test: esclusione; formula del test con più di 20 token e immagine diversa: quarantena).
  - *Dev:* 5% delle formule ammesse, seme 0; rango del campione candidato per ogni gruppo.
  - *Collaudo:* run4 e run2 ricalcolati dalle regole correnti e uguali ai loro file; 15 test su fixture; verifica dopo
    la scrittura 21 su 21.
  - *Verificatore corretto* (`c271d6e`, job 30633): la riga ritagliata deve avere il PNG della formula e fasce
    strutturalmente valide, prima dei gruppi; lo stato di ogni record è il più grave richiesto dalle evidenze; tolto il
    controllo sul cambio di seme.
  - *Riverifica in sola lettura* (job 30634, `scripts/manifest_recheck.py`): 28 controlli su 28, nessuna anomalia. Il
    rapporto è `docs/manifests/riverifica-1/unimer-candidato-1-bcc85198.json`.
  - *Riverifica corretta* (job 30638, test su fixture): si ferma se la classificazione degli shard non è quella dei
    record, o se le evidenze obbligatorie non sono registrate con digest validi (S2 §4). Sul manifest, coi controlli
    nuovi (job 30639, `riverifica-2`): 28 su 28, nessuna anomalia.
  - *Decontaminazione esterna* (job 30651, versione 2 `unimer-candidato-2`, S2 §4): pixel sugli originali, sui PNG e
    sui ritagli, formule contro UniMER-Test e OCRBench. Ritrova solo l'immagine e le formule lunghe di run2: nessun cambio
    di stato.
  - *Lacune:* revisione campionaria (S4); le 404 formule lunghe da riesaminare; `immagine_vuota` di run4 non
    ricalcolata; formule uguali con immagini diverse non unite (il dev misura immagini nuove); reader (C7).
- **MAVIS-Geometry** (`scripts/mavis_manifest.py`, job 30626 e 30630, S1 §9, S2 §3).
  - *Ingressi verificati:* 11 tar `9c80fff3-caption` letti e hashati una volta, con sha256 uguali allo sharding (job
    29419) e allo snapshot.
  - *Record:* 199.050, con compiti, QA del loader e digest di descrizione, verità, diagram_generation e didascalia.
  - *Gruppi:* stessi byte del PNG o stesso diagram_generation: 158.018, di cui 10.240 con più record. Una stessa
    immagine o figura con verità diverse andrebbe in quarantena: nessun caso.
  - *Stati:* `shape_type` in quarantena per compito, senza togliere gli altri compiti; tutti i record ammessi come
    candidati. Coerenza con le annotazioni verificata, fedeltà all'immagine da verificare.
  - *Dev:* 5% per profondità, seme 0; rango del campione candidato per ogni gruppo.
  - *Collaudo:* run2 ricalcolato con la sua chiave (`mavis-<id>`, diversa da quella del loader) e uguale al report e ai
    60 esempi; caption e geometry check senza segnalazioni; 12 test su fixture con annotazioni vere, il cui run2 lo
    scrive lo script storico; verifica dopo la scrittura 16 su 16.
  - *Riverifica in sola lettura* (job 30635), senza il controllo sul cambio di seme: 15 su 15, nessuna anomalia.
  - *Riverifica corretta* (job 30638, test su fixture): le evidenze obbligatorie di `evidence_files` devono essere
    registrate con digest validi e uguali su disco prima dell'uso (S2 §3). Sul manifest, coi controlli nuovi (job
    30640, `riverifica-2`): 15 su 15, nessuna anomalia.
  - *Decontaminazione esterna* (job 30652, versione 2 `mavis-geometry-candidato-2`, S2 §3): nessun pixel identico; 693
    record in quarantena per gruppo, per candidati pHash e testuali. Le decisioni testuali usavano una regola di
    frequenza, ora abolita.
  - *Riesame* (job 30676, versione 3 `mavis-geometry-candidato-3`, S2 §3): regole testuali esplicite. Le enumerazioni
    dei punti nella descrizione del loader restano senza effetto; i 50 record verso MathVision 753 sono candidati. In
    tutto 2.284 record in quarantena in 156 gruppi, nessuna riammissione.
  - *Lacune:* fedeltà all'immagine (revisione visiva, S4); verifica a vista dei 156 gruppi candidati; C9 (opzioni di
    `shape_type`, valore visualizzato ed esatto, 22.712 record con razionali disegnati come decimali); geometry check
    non ricalcolato; reader (C7), mentre il loader attuale può ancora scegliere `shape_type`.

### Da pianificare

Nessuno dei controlli che restano in questa tabella è stato eseguito o è dichiarato superato: quelli eseguiti sono
nella tabella sopra. Dal 2026-10-29 anche le verifiche leggere, che finora sono state eseguite sul frontend, vanno in
un job CPU (CLAUDE.md).

| Collaudo | Scopo | Risorse | Quando |
| --- | --- | --- | --- |
| Sintetici (C1, C2) | Eseguiti: test (job 30392, 30417, 30448), controllo sugli shard (job 30393, 30418, 30448), coppie v4 (job 30419), analisi degli scarti (job 30400, 30420), render e candidato v4 (job 30447-30449), manifest e dev split (job 30450-30451), decontaminazione esterna contro `pannello-v3` e manifest `synth-v4-candidato-2` (job 30517-30518), campione dell'audit preparato (job 30522, S2 §2.10), fogli corretti con la stessa selezione (job 30556, `campione-2`, S2 §2.11), confronto per unità e variante (job 30616). In corso, fatta dall'utente: revisione visiva del campione (380 unità e 175 casi mirati, rubrica della S2 §2.8) e dei 264 candidati della decontaminazione. Poi: limite con C8 sugli esiti, dopo aver fissato con l'utente la corrispondenza degli esiti (S4 §9) | revisione umana; C8 CPU pochi secondi | con l'ok dell'utente (D5) |
| Contratto ed estrattore (C3, C4; punto 1) | Tokenizzazione e label per modalità; quattro forme e terminazione; tassonomia ufficiale; risposta diretta di Violetto | CPU 20 min; GPU 1 ora | dopo D3 e S3 |
| Manifest, indice e reader (C5-C7; punto 3) | Per i sintetici v4 eseguiti (job 30450-30451, S1 §9). Manifest candidati di UniMER (job 30624-30625) e di MAVIS (job 30626 e 30630) eseguiti, e le loro versioni 2 con la decontaminazione esterna (job 30651-30652). Restano: reader che applichino i manifest di UniMER e MAVIS (C7), indice globale e chiavi storiche delle altre sorgenti (C6), varianti e contratto (dopo C3), contatori per compito nel flusso, PixMo e LLaVA | CPU 20 min; indice CPU 4 CPU 24G, durata da misurare | dopo S1, D6 |
| Stime (C8) | Eseguiti: copertura esatta, limite ipergeometrico e binomiale, Bonferroni, rifiuti (job 30616-30617, S4 §9). Da fare: collaudo dell'estrazione solo se si aggiunge un campionatore; applicazione agli esiti reali dopo la revisione completa | CPU pochi secondi | applicazione con l'ok dell'utente |
| MAVIS (C9; punto 2) | `shape_type` con opzioni; valori visualizzati ed esatti | CPU 20 min; controllo CPU su tutti gli shard | dopo C5 |
| Più immagini (punto 4) | Ordine, budget, parità con cache | CPU 20 min; GPU 1 ora | dopo i punti 1, 3 |
| Encoder (punto 5) | Refactoring SigLIP2 identico; parità dell'encoder B col percorso nativo; rami disattivati uguali al riferimento | GPU 1 ora ciascuno | (a) dopo l'ok; (b) dopo D1 |
| ViT addestrabile (punto 6) | Gradienti, variazione dei pesi, salvataggio, ripresa | GPU 64G 1 ora | dopo il punto 5 (a) |
| Valutatore visivo (punto 7) | Correttezza del codice di valutazione | GPU ≤ 2 ore | dopo C4 |
| Pannello e campione (punto 7; D6) | Eseguiti: download e verifica (job 30463), inventario (job 30464), indice, gruppi e ruoli `pannello-v2` (job 30466), collaudi e `pannello-v3` (job 30468-30469). Il 2026-11-02 il confronto fra versioni usa `metrica_primaria` (job 30517), e v4 è decontaminata (job 30518). Da fare, con l'ok dell'utente: i gruppi dei 7 elementi di test con candidati dallo sviluppo sintetico (S2 §2.10), con esito in `pannello-v4`; la verifica visiva dei 743 gruppi inammissibili è facoltativa, perché possono restare esclusi dalla metrica primaria e risolverli tutti non è un prerequisito per avanzare (chiarimento dell'utente del 2026-11-04); decontaminazione contro `pannello-v3` di MMFineReason, PixMo e LLaVA (UniMER e MAVIS fatti nei job 30651-30652, con la verifica a vista dei loro candidati ancora da fare) | CPU, pochi minuti per job; revisione umana per la verifica visiva | dopo l'approvazione di D6 |

## 5. Fuori dal primo perimetro

F4 (trainer RL, rollout, log-probabilità, GRPO), interpolazioni F3-M/F4-M, certificazione statistica generale e nuove
generazioni di target non sono prerequisiti di F0 (pipeline §5.10, incarico §5). Le implementazioni di DeepStack, MUDD e
value embeddings visivi seguono le loro specifiche e l'ordine di §6.5, dopo il confronto iniziale degli encoder.
