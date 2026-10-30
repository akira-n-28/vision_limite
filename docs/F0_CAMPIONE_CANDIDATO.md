# Campione candidato di F0: selezione riproducibile

**Stato al 2026-10-30.** È una proposta.
- Nessuna sorgente è ammessa o gold.
- Il campione non è ancora caricabile per un pilota: mancano i passi di §5.
- Avere i file non vuol dire che F0 sia completo.
- Per preparare il campione non sono stati avviati training, GPU, audit con agenti né download.

Collegamenti: pipeline v1.5 §6.1 (materiale di F0), §5 (dati), §15.3 (pronto per un esperimento); piano F0
(`docs/F0_IMPLEMENTATION_PLAN.md`), pannello e ruoli (`docs/PANNELLO_VALUTAZIONE.md`), specifiche S1-S4.

## 1. Regole della selezione

**Tre ruoli, mai mescolati:**
- **training dei piloti:** solo sorgenti di training locali. Nessun elemento del pannello, in nessun ruolo, e nessun
  benchmark riservato usato per colmare una lacuna;
- **sviluppo:** gli elementi `sviluppo` di `pannello-v2` e lo split dev dei sintetici v4. Servono a scegliere le
  configurazioni. Test e coppie v4 sono già consultati, quindi valgono come sviluppo e diagnostica;
- **test finale riservato:** gli elementi `test` di `pannello-v2`. In F0 non si consultano: servono solo alla
  decontaminazione.

**Ingressi fissati.**
- Revisione della sorgente e manifest degli shard: ogni manifest ha il job di sharding e lo sha256 di ogni tar.
- Manifest delle ammissioni, dove esiste: `data/manifests/synth-v4-candidato-1`, sha256 `975dfe42…`.
- File del pannello: `data/panel/pannello-v2/elements.jsonl.gz`, sha256 `31dd636e…`.

**Ordine di estrazione.**
- Dentro ogni strato (sorgente × famiglia o compito) i gruppi si ordinano per lo sha256 del JSON canonico
  `{"campione": "f0-candidato-1", "seme": 0, "gruppo": <group_id>}`. Un pilota con N esempi per strato prende i gruppi
  interi a partire dal rango più piccolo, fra i record ammessi. Un N più grande contiene sempre quello più piccolo.
- Lo stesso ordine, con `"ruolo": "sviluppo"` nel JSON, sceglie il sottoinsieme di sviluppo quando il budget di
  valutazione è minore dell'intero sviluppo.
- È lo schema già usato per l'estrazione dell'audit di v4 (S2 §2.8).
- I volumi sono una decisione aperta (§6).

**Lingua.** Solo inglese. Le domande non inglesi delle sorgenti miste si tolgono con gli elenchi esistenti
(`data/lang/mmfinereason`); i sottoinsiemi non inglesi dei benchmark sono `solo_decontaminazione`.

**Perimetro (v1.5).**
- Nessuna quota né criterio per il manoscritto. Gli esempi manoscritti incidentali restano, e non si avviano
  classificazioni per toglierli.
- Delle raccolte documentali miste si usano solo esempi autonomi a pagina singola.

## 2. Training dei piloti: sorgenti locali

Il minimo della pipeline (§6.1) parte da sintetici, MAVIS e UniMER e aggiunge una sorgente di scene reali. Le
capacità senza sorgente locale sono in §4.

| Capacità | Sorgente e shard | Selezione e numerosità | Provenienza del target | Controlli disponibili | Incertezze | Adattamenti mancanti |
| --- | --- | --- | --- | --- | --- | --- |
| Geometria, coordinate e grafici di funzioni (sintetici) | `src/limite_vl/synth.py` v4 (`284e8f1`), `data/shards/synth/v4` (job 30447), manifest `synth-v4-candidato-1` (job 30451) | split `train`, stato `ammesso`: 54.271 scene. Il loader ne costruisce 162.599 QA (coord 42.692, function 34.326, polygon 42.735, shapes 42.846) | calcolata dal codice del generatore e approvata dal checker sul layout (C2) | integrità, riproducibilità, checker su tutti i 60.900 record, gruppi e split, filtro dei loader (job 30447-30451) | il checker legge il layout, non i pixel; 67 scene in quarantena; audit visivo e decontaminazione esterna non fatti (S2 §2.8) | `variant_id` e contratto (C3, C5); audit (D5); scansione contro `pannello-v2` |
| Geometria (MAVIS-Geometry) | `CaraJ/MAVIS-Geometry@9c80fff3`, `data/shards/mavis-geometry/9c80fff3-caption` (11 tar, job 29419) | 199.050 immagini `caption.jpg`; esclusi `d3-3228` e 552 didascalie duplicate | regole del motore di MAVIS (`diagram_generation`): descrizione canonica e domande costruite dal codice (`f1_schema.py`, `f1_mavis.py`) | andata e ritorno su 199.050 (job 30033, 30052); valori razionali disegnati come decimali (job 30049) | la coerenza fra due descrizioni non prova la fedeltà ai pixel; solo 40 immagini guardate; `shape_type` ambigua, in quarantena; renderer non nostro; stessi autori di MathVerse | C9 (opzioni di `shape_type`, valore visualizzato ed esatto); manifest delle ammissioni (C5, C7); controllo visivo campionario (S4); scansione contro le geometrie del pannello |
| Formule stampate (UniMER-1M) | `wanderkid/UniMER_Dataset@2343ddd9`, `data/shards/unimer/2343ddd9-dense` (5 tar, job 29729; più `lines/` con 30.685 righe ritagliate) | 985.450 formule, meno 51 errori certi (`escluso`) e 1.119 incerti (`quarantena`), di cui 1.116 in 402 gruppi in conflitto. Le 18.827 rietichettature sono tracciate (S2 §4) | LaTeX della sorgente, senza gli spazi del formato im2latex (`latex.despace`); uguaglianza dei token, non equivalenza matematica | `outputs/unimer_check/run4` (job 30047: struttura, rendering, conflitti); sovrapposizioni col test (`outputs/unimer_overlap/run2`) | le formule CROHME manoscritte (meno dell'1%) non si separano per indice: restano (v1.5); un LaTeX compilabile può descrivere un'altra formula | import di esclusioni, quarantena e rietichettature come evidenze (C5, C6); manifest (C7); confronto con HME100K di OCRBench |
| Scene reali, descrizioni dettagliate (PixMo-Cap) | `anthracite-org/pixmo-cap-images@3e42775e`, 25 parquet su 75; `data/shards/pixmo-cap/3e42775e` (25 tar, job 29560) | 235.896 esempi | descrizione riscritta da un modello a partire dalle trascrizioni degli annotatori, che non sono negli shard | `outputs/pixmo_check/run4` (job 30044): segnali di rischio su dettagli assenti dalle trascrizioni, non esclusioni. Righe con almeno un segnale: conteggi 2-12 10.181, colori 8.803, testi citati 3.564 | testo libero senza verità strutturata: è un target controllato, non gold (pipeline §5.7); un terzo del dataset | audit campionario (D5); manifest (C5, C7); scansione contro MMStar e MUIRBench |
| Scene reali, didascalie brevi (LLaVA-Pretrain), solo complemento | `liuhaotian/LLaVA-Pretrain@70f9d1e5`, `data/shards/llava-pretrain/70f9d1e5` (29 tar, job 29388) | 558.128 esempi | didascalie BLIP generate su immagini di LAION, CC e SBU | nessun controllo di qualità; sha256 degli shard | testo di un modello, spesso generico | come PixMo, se entra nel mix |

**Sorgenti locali fuori dal minimo di F0:**
- **MMFineReason** (`OpenDataArena/MMFineReason-1.8M@7c93f724`): 1.770.926 righe sft e 39.961 rl. È un bacino per
  F2-F3.
  - Caption, risposta e CoT sono oggetti distinti, e caption e CoT vengono da Qwen3-VL-235B.
  - Contiene Geometry3K e WeMath2, e AI2D e ScienceQA, da cui vengono domande di MMStar.
  - La composizione interna di MMR1 non è inventariata.
- **MAVIS-Function:** 6 zip, non ancora in shard.
- **DeepScaleR, DeepMath:** testo per il mantenimento, con i riferimenti originali; le tracce di Violetto sono solo
  storiche.
- **DeepVision-103K, ViRL39K:** RL in F4.

## 3. Sviluppo e test per capacità

Le numerosità sono di `pannello-v2` (job 30466). "Senza irrisolti" è la base della metrica primaria del test (pannello
§3.1). La verità è quella ufficiale del benchmark; metriche e margini sono D4.

| Capacità | Sviluppo | Test riservato (senza irrisolti) | Provenienza della verità | Incertezze |
| --- | --- | --- | --- | --- |
| Matematica e geometria | MathVista testmini in inglese 936; MathVerse 4.728 (788 problemi in 6 versioni); We-Math 1.740; MathVision testmini 304 più 306 | MathVision 2.734 (2.731); DynaMath 5.000 su 500 problemi (4.968); OlympiadBench 606 multimodali e 674 solo testo; MV-MATH 2.009 (2.003) | annotazioni dei benchmark, in parte dalle sorgenti d'origine (MathVista ne ha 31) | figure di Geometry3K condivise fra benchmark (pannello §1); DynaMath_Sample non è la distribuzione completa |
| Percezione dei diagrammi | Geoperception 11.657; dev dei sintetici v4 2.870 scene (8.779 QA); test e coppie v4 | VisOnlyQA Real 896 (633), Synthetic 700 | annotazioni; codice per i sintetici | 263 geometrie di VisOnlyQA Real hanno candidati nello sviluppo |
| Formule stampate | UniMER-Test 1.261 | UniMER-Test 16.164 | etichette LaTeX | hwe (6.332) è solo diagnostica |
| OCR | OCRBench 525 | OCRBench 475 (447) | annotazioni delle sorgenti; corretto se la risposta è contenuta nell'uscita | metrica permissiva; 150 domande su manoscritto non fanno criterio |
| Documenti a pagina singola | DocVQA 2.681 (468 documenti) | DocVQA 2.668 (2.583; 470 documenti) | risposte estrattive annotate a mano, più risposte ammesse (ANLS) | 81 candidati verso la componente `docVQA` di OCRBench |
| Grafici | ChartQA 2.033 | ChartQA 2.387 (1.761) | domande `human` scritte da annotatori, `augmented` generate; etichette dai dati del grafico | 626 irrisolti, soprattutto grafici con la stessa impaginazione |
| Tabelle | DocVQA `table/list` 934; MathVista `table` 70 | DocVQA `table/list` 846 | come DocVQA | nessun test di tabelle pure |
| Poche immagini | MUIRBench 1.134 | MUIRBench 1.466 (1.335); MV-MATH; OlympiadBench, 271 problemi con più immagini | annotazioni curate | `Ordered_Pages` e `Video` fuori perimetro (pannello §3.4) |
| Visione generale | MMStar 855; MUIRBench | MMStar 645 (520) | selezione rivista a mano da benchmark esistenti | sviluppo sbilanciato verso le domande da MathVista |
| Domande senza risposta | MUIRBench 567 varianti | MUIRBench 733 varianti | per costruzione (opzione "None of the choices provided") | nessuna con un'immagine sola |
| Testo | AIME 2024-2026, HMMT feb 2025 e 2026, GSM8K, IFEval | MATH-500, BeyondAIME, APEX 44, OlympiadBench `OE_TO_maths_en_COMP` | risposte ufficiali; controlli a regole per IFEval | un problema di APEX in comune con la suite, da individuare; i controlli di IFEval mancano |

## 4. Lacune del training e aggiunte minime proposte

Nessun download è stato avviato. Revisioni e dimensioni vanno fissate con l'API prima della proposta di download, come
per `p10-pannello`.

| Capacità | Lacuna | Aggiunta minima proposta | Perché | Controlli prima dell'uso |
| --- | --- | --- | --- | --- |
| Grafici | nessuna sorgente locale | train di ChartQA, `ahmed-masry/ChartQA@af8b6f5c` `data/train-*.parquet` (circa 854 MB, `downloads/catalog_status.tsv`) | grafici reali, domande umane e generate, stessa revisione e formato del pannello | `imgname`, pixel e pHash contro val e test, contro MathVista (ChartQA è una sua sorgente) e contro le componenti ChartQA di OCRBench; `human` e `augmented` separati |
| Documenti a pagina singola e OCR di documenti | nessuna sorgente locale | train di DocVQA, `lmms-lab-encoder/DocVQA@539088ef` `DocVQA/train-*.parquet`: 12 file, 7.392.672.068 byte (API di Hugging Face, 2026-10-30) | pagine singole per costruzione, domande umane, stessi campi del pannello (documento UCSF compreso) | gruppi per documento UCSF contro val, pixel e pHash; contro MathVista (DocVQA è una sua sorgente) e la componente `docVQA` di OCRBench. Un sottoinsieme per documento è possibile, ma richiede comunque il confronto con val |
| Tabelle | nessuna sorgente locale | TabMWP: tabelle come immagini, con problemi a risposta verificabile. La fonte ufficiale è `lupantech/PromptPG`; copia, revisione e dimensioni sono da fissare | risposte numeriche verificabili a regole, insieme piccolo | contro MathVista, che lo usa come sorgente (62 domande in sviluppo, 338 in `solo_decontaminazione`) |
| Poche immagini | nessuna sorgente locale con più immagini | (a) un sottoinsieme di M4-Instruct, limitato a confronto visivo e matematica, senza video; componenti e dimensioni da fissare sulla scheda. (b) Senza download: coppie composte da record a un'immagine già ammessi, con la verità dal codice. Richiede un generatore nuovo, da non costruire ora, e dà nuove combinazioni di contenuti già usati | serve a provare il percorso con più immagini (punto 4 del piano) | (a) contro MUIRBench, costruito da dataset esistenti. (b) solo training, mai test |
| OCR di scene | solo segnali indiretti (testi citati nelle descrizioni di PixMo) | facoltativa, dopo il primo pilota: TextVQA train (dimensione da verificare) | OCRBench misura la VQA sulle scene | contro OCRBench (`textVQA`, `STVQA`) e MMStar |
| Domande senza risposta con un'immagine sola | nessuna | fuori dal minimo: VizWiz | foto con "unanswerable" | contro MMStar e MathVista (`VizWiz` è una sua sorgente) |

## 5. Cosa manca perché il campione sia caricabile e utilizzabile

1. **Contratto e target (D3 → C3, C4).** Senza `vl-contract/1` non si costruiscono target con risposta diretta o
   ragionata, né l'estrattore e i verificatori che separano formato, completezza e correttezza.
2. **Identità delle varianti.** `variant_id` e `uid` (S1) mancano per tutte le sorgenti, anche per v4: il manifest
   copre record e gruppi, non varianti né contratto.
3. **C5 e C7 per le altre sorgenti.** Oggi solo i sintetici v4 hanno un manifest delle ammissioni e i reader filtrati.
   Mancano per MAVIS, UniMER, PixMo e LLaVA:
   - record e gruppi;
   - stati, esclusioni e quarantene importate;
   - split di sviluppo per gruppo;
   - contatori per compito e `too_long` per sorgente.
4. **C9 per MAVIS:** `shape_type` con le opzioni, e valore visualizzato ed esatto nel record.
5. **Esclusioni storiche come evidenze (C6):** `unimer_check/run4`, `pixmo_check/run4`, `split_duplicates/run1`, liste di
   `p6` e `f1eval`, ciascuna con motivo, versione e stato.
6. **Decontaminazione contro `pannello-v2`.** Non è fatta per nessuna sorgente di training. Per v4 è possibile subito
   (S2 §2.8, `texts_of` per i sintetici). Per MAVIS, UniMER, PixMo e LLaVA serve una scansione con pixel, pHash e
   testo, e poi la risoluzione dei candidati.
7. **Verifica visiva:**
   - dei 1.299 irrisolti del pannello (191 forti), prima di qualsiasi uso del test;
   - l'audit di v4 (380 unità e circa 170 casi, S2 §2.8, D5);
   - campioni di MAVIS (fedeltà ai pixel) e di PixMo (dettagli aggiunti);
   - l'osservabilità alla risoluzione scelta.
8. **Più immagini (punto 4 del piano).** `Sample.images`, loader e riferimenti nel testo servono per MUIRBench, MV-MATH,
   OlympiadBench e per una sorgente di training con più immagini.
9. **Valutatore del modello VL (punto 7).** Metriche per capacità (D4) ed estrattori (C4).
10. **Sorgenti delle lacune (§4).** Download da approvare.

## 6. Prossimo lotto minimo di implementazione

Il lotto è indipendente dalle decisioni aperte: si può fare dopo l'ok, con job CPU di pochi minuti ciascuno, senza GPU
né agenti.

1. **Verifica visiva degli irrisolti del pannello.**
   - Fogli di coppie e un TSV degli esiti, a partire dai 191 forti.
   - La correlazione dei pixel a 64×64 di `decontam_review.py` serve a ordinarli.
   - Revisione umana; l'esito va in `pannello-v3`.
2. **Decontaminazione esterna di v4 contro `pannello-v2`** (passi 2-6 di S2 §2.8). L'esito va in una versione nuova
   del manifest v4.
3. **Manifest delle ammissioni di UniMER-1M e MAVIS-Geometry,** col sottoinsieme di S1 già usato per v4:
   - record, gruppi e stati;
   - esclusioni e quarantena importate da `unimer_check/run4`;
   - per MAVIS, `shape_type` in quarantena;
   - dev al 5% per gruppo con seme 0;
   - il rango di estrazione di §1.
4. **Scansione di MAVIS, UniMER, PixMo e LLaVA contro `pannello-v2`** (pixel e pHash, testo dove c'è). Dà candidati da
   mettere in quarantena, non esclusioni automatiche. La durata va misurata prima su uno shard per sorgente.

Dopo D3 vengono C3, C4 e le varianti (`variant_id`), poi il punto 4 (più immagini) e il valutatore (punto 7).

**Decisioni aperte, dell'utente:**
- D1, checkpoint Qwen;
- D2, budget;
- D3, contratto;
- D4: metriche, margini, sottoinsieme di MUIRBench nel perimetro, OCRBench senza manoscritto;
- D5, budget degli audit;
- D6, approvazione di `pannello-v2`: quote, eventuale stratificazione, regola degli irrisolti;
- i download di §4;
- volumi e mix dei piloti;
- risoluzione e token per immagine.
