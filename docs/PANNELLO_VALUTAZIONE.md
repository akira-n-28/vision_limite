# Pannello di valutazione e decontaminazione di Limite-VL (proposta per D6)

**Stato:** proposta del 2026-10-30, da approvare. Nessun download, scansione, audit o valutazione è stato avviato.
Collegamenti: piano F0 (`docs/F0_IMPLEMENTATION_PLAN.md`, decisione D6, punti 3 e 7), pipeline §§5.5, 5.8, 6.1, 15,
S1 §§3-4.

**Evidenze di questo documento:**
- inventario locale: job 30457-30458, `outputs/panel_inventory/run1` (`inventory.json`, `sintesi.md`);
- sovrapposizioni fra i benchmark indicizzati: job 30459, `outputs/panel_overlaps/run1`;
- stati dei sintetici v4 già presenti in v3: job 30460, `outputs/synth_eval_reuse/run1`;
- manifest dei download, verifiche degli hash (`outputs/verify/`), ledger;
- fonti primarie dei candidati: API e schede di Hugging Face, articoli e repository ufficiali, consultati il
  2026-10-30.

Nel documento:
- **scaricato** vuol dire che i file del manifest del download sono presenti con hash verificato;
- **indicizzato** vuol dire presente negli indici di decontaminazione `data/decontam/p6` o `data/decontam/f1eval`;
- **usato** vuol dire consultato per una scelta.

## 1. Inventario verificato

Per tutte le sorgenti locali qui sotto, tutti i file del manifest sono presenti e gli hash sono verificati. Nessuna
lettura ha dato errori (job 30457).

| Benchmark (repo@revisione) | Split e varianti presenti | Contenuto utile | Indici | Usi precedenti | Stato |
| --- | --- | --- | --- | --- | --- |
| MathVista `AI4Math/MathVista@2b6ad694` | testmini 1.000 con risposta (936 in inglese, 62 in cinese, 2 in persiano); test 5.141 con la risposta vuota | immagini, `pid`, scelte, tipo di risposta, `metadata.source` con 31 sorgenti d'origine (fra cui Geometry3K, ChartQA, TabMWP, AI2D) | p6: 6.141 | solo le dimensioni delle immagini (job 29649) | scaricato, indicizzato. Il test serve solo alla decontaminazione |
| MathVerse `AI4Math/MathVerse@3bc86196` | testmini: 788 problemi in 5 versioni con immagine (3.940; la Vision Only è senza testo) più 788 Text Only senza immagine | `problem_index`, versione, risposta, sorgente (GeoQA, Geometry3K, GEOS, raccolte proprie) | p6: 4.728, di cui 788 solo testo | 29649 | scaricato, indicizzato |
| MathVision `MathLLMs/MathVision@2837ddb3` | test 3.040 con risposta; testmini 304, contenuto per intero nel test (304 ID su 304, job 30459) | immagini, opzioni, livello 1-5, 16 materie | p6: 3.344 | 29649 | scaricato, indicizzato |
| We-Math `We-Math/We-Math@527c44d4` | testmini 1.740 domande su 525 problemi (`ID`), scomposti in 2 o 3 passi | scelta multipla A-E, concetti | p6: 1.740 | 29649 | scaricato, indicizzato |
| DynaMath_Sample `DynaMath/DynaMath_Sample@bc7fcd1a` | 10 varianti già generate per 501 problemi (5.010). Il `.7z` dentro `10VariationSamples.zip` non è stato aperto | `id` 1-501 uguale nelle 10 varianti, risposta, materia | p6: 5.010 | 29649 | scaricato, indicizzato. Non è la distribuzione DynaMath completa: il generatore dei problemi non è locale |
| Geoperception `euclid-multimodal/Geoperception@c34c5956` | 11.657 domande (split `train`), 7 predicati | figure di Geometry3K, risposta breve (1 vuota) | f1eval: 11.657 | struttura letta (job 30026) | scaricato, indicizzato |
| VisOnlyQA Real `ryokamoi/VisOnlyQA_Eval_Real_v1.1@3747d137` | 900 domande in 10 file: grafici, chimica, geometria | scelta multipla | f1eval: 900 | 30026 | scaricato, indicizzato |
| VisOnlyQA Synthetic `ryokamoi/VisOnlyQA_Eval_Synthetic@ef0b5fbc` | 700 domande in 7 file: 3D, geometria sintetica | scelta multipla | f1eval: 700 | 30026 | scaricato, indicizzato |
| OlympiadBench `Hothan/OlympiadBench@91184b52` | 18 sottoinsiemi; in inglese con immagini: `OE_MM_maths_en_COMP` 150 (19 con 2-4 immagini), `OE_MM_physics_en_COMP` 456, `TP_MM_*_en_COMP` 62 + 19 (dimostrazioni); solo testo: `OE_TO_maths_en_COMP` 674 | `final_answer`, tipo di risposta, unità, materia | nessuno per le immagini; il testo è nel controllo del replay (job 29699-29700) | nessuno | scaricato, non indicizzato |
| UniMER-Test (`wanderkid/UniMER_Dataset@2343ddd9`, `UniMER-Test.zip`) | spe 6.762, cpe 5.921, sce 4.742, hwe 6.332 immagini, con le etichette | formule LaTeX | nessuno: c'è il confronto col training di UniMER (`outputs/unimer_overlap/run2`) | i primi 200 di spe e i primi 200 di cpe: criterio di F1 e valutazioni F1c (job 29448-29761) | scaricato; in parte usato |
| Suite testuale (job 29348) | AIME 2024 30, AIME 2025 30, AIME 2026 30, HMMT feb 2025 30, HMMT feb 2026 33, BeyondAIME 100, APEX shortlist 47, MATH-500 500, OlympiadBench `OE_TO_maths_en_COMP` 674 | problema e risposta | confronto col replay testuale (job 29699-29700) | AIME 2026: baseline e ablation dell'ottimizzatore, usate per decidere (job 29685-29697). AIME 2025 e HMMT 2026: baseline (job 29393) | scaricata. AIME 2026 è già usata per scegliere |
| GSM8K (`openai/gsm8k`) | — | — | — | — | non scaricato. È nella suite ufficiale del valutatore (`MATH_STANDARD`), senza revisione fissata |

**Sintetici v4.** Test (300) e coppie (300) del manifest v4 hanno le stesse chiavi di v3, e test e coppie v3 sono stati
consultati nelle valutazioni F1c (job 29741-29761). Con la stessa chiave lo stato è identico:
- nel test, per 277 scene su 300 (coord e function tutte, polygon 54 su 75, shapes 73 su 75);
- nelle coppie, per 217 su 300, con a e b entrambe uguali (job 30460).

Restano protetti dal training, ma non sono un test finale indipendente.

**Sovrapposizioni fra i benchmark locali** (job 30459). Sono candidati per pHash, salvo l'identità degli ID:
- **MathVision:** il testmini è un sottoinsieme del test (304 ID su 304).
- **MathVista:** testmini e test non condividono ID, ma 382 coppie hanno lo stesso pHash. Sono la stessa figura con
  domande diverse.
- **Figure di Geometry3K e affini:** attraversano più benchmark. Elementi con un vicino entro 4 bit in un altro
  insieme:
  - verso Geoperception: 269 di MathVerse, 199 del test e 38 del testmini di MathVista;
  - verso le geometrie di VisOnlyQA Real: 632 di Geoperception, 227 del testmini di MathVista, 167 di MathVerse.
- **Grafici:** quelli di VisOnlyQA Real sono vicini a quelli di MathVista (46 elementi del test e 13 del testmini).
- **DynaMath:** pochi vicini (fino a 30 per coppia di insiemi).

La conferma richiede il confronto dei pixel (§5).

**File fuori da Git che riguardano il pannello.** Snapshot in `hf/hub/datasets--*`, link in `data/eval-datasets` verso
gli snapshot del valutatore, indici e liste in `data/decontam/{p6,f1eval,text}`, esiti storici in `outputs/eval`,
`outputs/eval_f1_formulas`, `outputs/f1c`. Sono già in `docs/ARTIFACT_INVENTORY.tsv`. Le sorgenti di training locali
(MAVIS, MMFineReason, LLaVA, PixMo, DeepScaleR, DeepMath, DeepVision, ViRL39K, MAVIS-Function, GeoSym127K) entrano solo
nella decontaminazione (§5).

## 2. Copertura per capacità e candidati confrontati

| Capacità | Copertura locale | Lacuna | Candidati confrontati | Scelta |
| --- | --- | --- | --- | --- |
| Matematica, STEM, geometria | MathVista testmini, MathVerse, MathVision, We-Math, DynaMath_Sample, OlympiadBench multimodale (150 di matematica, 456 di fisica) | nessuna | — | nessun download |
| Percezione e dipendenza dall'immagine | Geoperception, VisOnlyQA, versioni di MathVerse, immagine bianca o assente nel valutatore, sintetici | nessuna sui diagrammi | — | nessun download |
| Formule stampate e manoscritte | UniMER-Test spe, cpe, sce, hwe | nessuna | MathWriting (test) solo se entra nel training; le 100 HMER di OCRBench vengono da HME100K | nessun download dedicato |
| OCR generale | — | testo nelle scene, estrazione di campi, OCR di documenti | **OCRBench v1** (1.000 domande: riconoscimento 300, VQA su scene 200, VQA su documenti 200, estrazione di campi 200, formule manoscritte 100; corretto se la risposta è contenuta nell'uscita); OCRBench v2 (circa 10.000, bilingue, con una parte privata) | OCRBench v1 |
| Documenti, pagina singola | — | assente | **DocVQA val** (5.349 domande con `question_types`: tabelle e liste, moduli, layout, scrittura a mano); InfographicVQA val | DocVQA val |
| Documenti, più pagine | — | assente | **MMLongBench-Doc** (1.091 domande su 135 PDF: 22.8% non rispondibili, 33.2% fra più pagine; serve un renderer PDF, e la valutazione ufficiale estrae le risposte con GPT-4o); **MP-DocVQA val** (5.187 domande, fino a 20 pagine, 4,7 GB: sono le domande di DocVQA con l'evidenza su una pagina) | MMLongBench-Doc, oppure MP-DocVQA (decisione 2) |
| Grafici e tabelle | grafici e tabelle dentro MathVista e VisOnlyQA | grafici reali con risposta verificabile | **ChartQA** (val 1.920, test 2.500; accuratezza rilassata, a regole); CharXiv (val con risposte, valutazione ufficiale con GPT-4o) | ChartQA. Le tabelle vengono da DocVQA (tabelle e liste), MathVista (TabMWP) e MMLongBench-Doc |
| Più immagini | 19 problemi multimodali di OlympiadBench | assente | **MUIRBench** (2.600 domande su 11.264 immagini, 12 compiti, ogni domanda con la sua variante non rispondibile); **MV-MATH** (2.009 problemi di matematica in inglese con 2-8 immagini) | MUIRBench e MV-MATH |
| Visione generale e domande non rispondibili | solo sorgenti di training | assente | **MMStar** (1.500 domande a scelta multipla, 6 capacità, scelte perché richiedono l'immagine; l'origine di ogni domanda è in `meta_info`); VizWiz val (4.319, foto reali con "unanswerable", candidata anche di training) | MMStar. Le non rispondibili vengono da MUIRBench, MMLongBench-Doc e dalla componente nuova (§3.3); VizWiz resta fuori dal minimo |
| Mantenimento del testo | suite math-extended locale | GSM8K (nella suite ufficiale), istruzioni | **GSM8K** test (1.319); **IFEval** (541 istruzioni verificabili a regole) | GSM8K e IFEval |

Restano fuori dal minimo, perché la capacità è già coperta o il costo non torna: CharXiv, InfographicVQA, OCRBench v2,
VizWiz, MathWriting come benchmark separato.

## 3. Ruoli proposti (D6)

### 3.1 Regole

- **Sviluppo:** tutto ciò che serve a scegliere encoder, proiettore, risoluzione, mix, durate e soglie. Si consulta
  più volte, e i suoi risultati non sono una prova finale.
- **Test finale riservato:** non si consulta prima della valutazione finale. Si usa una volta, con un protocollo fissato
  prima, e dopo non si prendono decisioni.
- **Diagnostica:** analisi mirate (versioni, varianti, immagine tolta, tipi di domanda, non rispondibili). Non è una
  conferma indipendente; sui dati del test si fa solo dopo la valutazione finale.
- **Gruppi:** problemi, immagini, documenti e varianti collegati da un legame confermato (provenienza dichiarata,
  pixel identici, stesso problema) formano un gruppo, anche fra benchmark diversi. Un gruppo che tocca lo sviluppo va
  tutto nello sviluppo, e il test perde quegli elementi. Le dimensioni finali si contano dopo la conferma (§5).
- **Già consultato:** un dato già consultato non diventa test cambiando etichetta. Oggi lo sono:
  - AIME 2026, AIME 2025 e HMMT 2026;
  - i primi 200 di spe e i primi 200 di cpe di UniMER-Test;
  - test e coppie dei sintetici v3, e quindi gli stati condivisi da v4.
- **Protezione dal training:** ogni elemento del pannello, in qualunque ruolo, entra nell'indice di decontaminazione, e
  un record di training legato a uno di essi è escluso (S1 §3).
- **Esposizione pregressa:** la decontaminazione del nuovo training non toglie quello che SigLIP2 e Violetto possono
  aver visto prima. Per questo serve la componente nuova riservata (§3.3).

### 3.2 Assegnazione

| Capacità | Sviluppo | Test finale riservato | Diagnostica |
| --- | --- | --- | --- |
| Matematica e geometria | MathVista testmini in inglese (936); MathVerse testmini, un gruppo per `problem_index`; We-Math, un gruppo per `ID`; MathVision testmini (304) | MathVision test senza il testmini (2.736); DynaMath_Sample (501 gruppi da 10 varianti); OlympiadBench `OE_MM_maths_en_COMP` (150) e `OE_MM_physics_en_COMP` (456) | versioni di MathVerse (dipendenza dall'immagine); varianti di DynaMath (caso peggiore per gruppo) |
| Percezione | Geoperception, un gruppo per figura | VisOnlyQA Real e Synthetic, meno i gruppi legati allo sviluppo | predicati di Geoperception; immagine bianca o assente |
| Formule | UniMER-Test: i 400 già consultati più il 5% di ogni sottoinsieme, estratto per gruppo con seme 0 | il resto di UniMER-Test (sce e hwe mai usati) | errori di struttura (`latex.structure_problems`) |
| OCR e documenti | OCRBench (1.000); DocVQA val (5.349, un gruppo per documento) | MMLongBench-Doc (1.091, un gruppo per documento), oppure MP-DocVQA val | tipi di domanda di DocVQA; componenti di OCRBench |
| Grafici e tabelle | ChartQA val (1.920) | ChartQA test (2.500), meno i grafici legati al testmini di MathVista | tabelle di DocVQA e MathVista |
| Più immagini | metà dei gruppi di MUIRBench (coppie indivisibili, seme 0) | l'altra metà di MUIRBench; MV-MATH (2.009) | coppie rispondibile/non rispondibile di MUIRBench |
| Visione generale e non rispondibili | MMStar (1.500) | non rispondibili di MMLongBench-Doc e della componente nuova | MMStar per capacità |
| Testo | AIME 2024, 2025 e 2026; HMMT feb 2025 e 2026; GSM8K; IFEval | MATH-500; OlympiadBench `OE_TO_maths_en_COMP`; BeyondAIME; APEX shortlist meno i 4 problemi presenti anche nella suite (commento in `limite_evals/suites.py`) | — |
| Sintetici | dev del manifest v4 (2.870); test e coppie v4 | componente nuova (§3.3) | coppie controfattuali v4 |

Note:
- MathVista test e i sottoinsiemi non inglesi servono solo alla decontaminazione.
- Il manifest v4 non cambia: test e coppie restano nello split `test`, cioè fuori dal training, e il pannello li usa per
  sviluppo e diagnostica.
- Il ruolo di ogni elemento va in un manifest del pannello (punto 7 del piano F0), con il gruppo.

### 3.3 Componente nuova riservata

Ha la verità dal codice e non è mai stata pubblicata, quindi nessun encoder o decoder l'ha vista. Qui si fissa come
costruirla; non si genera in questo incarico.

1. **Sintetici riservati.** Si generano con il generatore v4 congelato (`284e8f1`) e chiavi nuove
   (`synth-riservato-NNNNNN`): 1.000 scene per famiglia e 300 coppie, tenendo solo gli stati assenti da tutto il
   manifest v4 (controllo per `stato_scena_sha256`). Le scene `function` con un solo grafico hanno circa 600 stati,
   quasi tutti già nel training (S2 §2.6): servono un generatore v5 o le sole scene con due grafici. Si aggiunge uno
   split di stili mai visti
   (S1 §3): caratteri, spessori e dimensioni fuori dagli intervalli di `synth._style`, che richiede un'opzione del
   generatore.
2. **Grafici, tabelle e documenti generati.** Un generatore nuovo produce tabelle casuali e, con matplotlib (già
   nell'ambiente), grafici a barre, linee e torte e pagine di rapporto su più pagine. Le domande sono di lettura, di
   calcolo, di collegamento fra pagine, e non rispondibili (dato assente). La verità è per costruzione, con un
   verificatore come `synth_check`.
3. **Più immagini sintetiche.** Due o tre scene v4 con domande di confronto (quale ha più figure, dove sta lo stesso
   punto) e varianti non rispondibili.

Per tutte e tre:
- chiavi, semi e numerosità si fissano nel manifest del pannello prima di qualsiasi addestramento;
- la generazione avviene solo quando si prepara la valutazione finale, e il risultato si consulta una volta;
- gruppi: uno per scena, coppia, documento o insieme di immagini.

Un piccolo insieme di pagine reali annotate a mano, con revisione competente, è possibile ma non fa parte del minimo.

## 4. Download mancanti

Revisioni, file e dimensioni vengono dall'API di Hugging Face (2026-10-30). Le righe del gruppo `p9-pannello` in
`downloads/sources.tsv` sono una proposta: nessun manifest è stato generato e nessun file è stato scaricato.

| Benchmark | Repository | Revisione | File | Byte |
| --- | --- | --- | --- | --- |
| OCRBench v1 | `echo840/OCRBench` | `92a54bd1384387c178d5a07140a2d85e0a3d12e1` | `data/test-00000-of-00001.parquet` | 67.576.988 |
| DocVQA val | `lmms-lab-encoder/DocVQA` (prima `lmms-lab/DocVQA`) | `539088ef8a8ada01ac8e2e6d4e372586748a265e` | `DocVQA/validation-0000[0-5]-of-00006.parquet` | 1.055.835.028 |
| MMLongBench-Doc | `yubo2333/MMLongBench-Doc` | `2ff6aa9237fc777b6627dc57a486e9225ac5fb86` | `data/train-00000-of-00001.parquet` e 135 PDF in `documents/` (due identici) | 85.239 + 662.135.983 |
| MP-DocVQA val (alternativa) | `lmms-lab-encoder/MP-DocVQA` | `e776eeade5a631d9ae18230ed3ab6213cd063dac` | `data/val-*.parquet` (29 file) | 4.690.012.441 |
| ChartQA | `ahmed-masry/ChartQA` | `af8b6f5c08c95085271561c2a3f9d15f2b5a9031` | `data/val-00000-of-00001.parquet`, `data/test-00000-of-00001.parquet` | 50.238.778 + 68.916.402 |
| MUIRBench | `MUIRBENCH/MUIRBENCH` | `4c393cffc985c77d28de3b9045e2e5186920df80` | `data/test-0000[0-4]-of-00005.parquet` | 429.440.985 |
| MV-MATH | `PeijieWang/MV-MATH` | `b22144b7d9315ce30c7ee8ce1bbaaa7b52fa8674` | `MV-MATH.json`, `images.zip` | 3.977.469 + 43.215.035 |
| MMStar | `Lin-Chen/MMStar` | `bc98d668301da7b14f648724866e57302778ab27` | `mmstar.parquet` | 41.798.712 |
| GSM8K | `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | `main/test-00000-of-00001.parquet` | 419.088 |
| IFEval | `google/IFEval` | `966cd89545d6b6acfd7638bc708b98261ca58e84` | `ifeval_input_data.jsonl` | 207.111 |

A questi si aggiungono i `README.md`, di pochi KB.
- **Totale:** circa 2,42 GB con MMLongBench-Doc, circa 6,45 GB con MP-DocVQA al suo posto.
- **Licenze dichiarate:**
  - ChartQA GPL-3.0; MUIRBench CC-BY-4.0; MV-MATH MIT; GSM8K MIT; IFEval Apache-2.0;
  - MMLongBench-Doc: dati CC BY-NC 4.0;
  - la copia di DocVQA porta `apache-2.0`, ma i documenti vengono dalla UCSF Industry Documents Library;
  - OCRBench e MMStar non dichiarano una licenza nella scheda.

  Il progetto è solo di ricerca: le fonti si citano e i dati non si ridistribuiscono.

**Codice da aggiungere:**
- IFEval: i controlli delle istruzioni del repository ufficiale (google-research `instruction_following_eval`) non sono
  nel valutatore vendorizzato. Vanno clonati in `third_party/` a un commit fissato, oppure riscritti.
- MMLongBench-Doc: serve un renderer PDF, come `pypdfium2` o PyMuPDF, assente dall'ambiente adottato. È un build nuovo
  e richiede l'ok dell'utente.
- GSM8K: è già nella suite del valutatore; va fissata la revisione e collegata in `data/eval-datasets`.

## 5. Piano minimo di decontaminazione

### 5.1 Per benchmark

| Benchmark | Materiale per il confronto | Strumenti riutilizzabili | Adattamenti indispensabili | Copertura |
| --- | --- | --- | --- | --- |
| MathVista (testmini e test), MathVerse, MathVision, We-Math, DynaMath_Sample | immagini e domande degli snapshot | indice p6 (pHash e testo), `decontam.py scan`, `decontam_review.py` | sha256 dei pixel normalizzati nell'indice; ID di gruppo (`problem_index`, `ID`, `id`) | provenienza (`metadata.source` di MathVista e MathVerse), pixel, testo |
| Geoperception, VisOnlyQA | indice f1eval | come sopra | sha256 dei pixel | provenienza (Geometry3K), pixel, testo |
| OlympiadBench multimodale | colonne `image_1`-`image_9` e domande | `decontam.py index` | leggere più immagini per elemento; indice nuovo | pixel, testo |
| UniMER-Test | immagini e formule | `scripts/unimer_test_overlap.py`, `latex.despace` e `visible_tokens` | indice delle formule normalizzate, da confrontare con tutte le sorgenti con formule (MMFineReason, OCRBench HMER, MathWriting) | formule (stessi token normalizzati), pixel |
| Suite testuale, GSM8K, IFEval | testi | `replay_decontam.py` (copertura degli n-gram) | estendere il confronto a MMFineReason e alle domande delle altre sorgenti | testo |
| OCRBench, DocVQA val, ChartQA, MMStar | parquet da scaricare | `decontam.py index` (immagini struct già lette) | provenienza da `dataset` (OCRBench), `meta_info.source` (MMStar), `docId` (DocVQA), `imgname` (ChartQA) | provenienza, pixel, testo |
| MUIRBench, MV-MATH | liste di immagini (`image_list`, `input_image` con `images.zip`) | `decontam.py index` | leggere liste di immagini e JSON con zip | pixel per immagine, testo |
| MMLongBench-Doc | PDF da renderizzare | nessuno | rendering delle pagine in un job, indice per pagina | pixel per pagina, testo delle domande |
| Componente nuova | generata | manifest dei sintetici, controllo per stato | — | per costruzione |

**Somiglianze non confermate.** pHash e n-gram danno solo candidati (S1 §4).
- **Conferma:** pixel identici dopo la normalizzazione, provenienza dichiarata, oppure lo stesso problema verificato dal
  codice. `decontam_review.py`, con la somiglianza dei pixel ≥ 0.90 e il rapporto d'aspetto entro 1.2, decide quali
  candidati guardare.
- **Candidato irrisolto fra training e pannello:** resta in quarantena, fuori dal training.
- **Candidato irrisolto fra sviluppo e test:** l'elemento del test si conta a parte, finché non si risolve.
- **Nessuna catena:** nessuna fusione passa per una catena di candidati.
- **Frasi di rito:** una domanda a modello comune ("How many …") non è un legame.

### 5.2 Subito per v4 e dopo con le altre sorgenti

- **Subito, senza download:**
  - immagini v4 contro gli indici p6 e f1eval;
  - pixel identici, aggiungendo lo sha256 all'indice;
  - le domande, estendendo `texts_of` di `decontam.py` al tipo `synth` (oggi restituisce una lista vuota).

  v4 non ha legami di provenienza: è generata.
- **Dopo i download approvati:**
  - v4 contro gli indici nuovi (OCRBench, DocVQA, ChartQA, MUIRBench, MV-MATH, MMStar, OlympiadBench multimodale,
    pagine di MMLongBench-Doc);
  - indice delle formule di UniMER-Test.
- **Quando entrano le altre sorgenti:**
  - MAVIS (stessi autori di MathVerse) contro tutti i benchmark di geometria;
  - UniMER contro UniMER-Test (già fatto, run2), OCRBench HMER e CROHME;
  - MMFineReason, un aggregato che contiene Geometry3K, WeMath e varianti di DynaMath, contro tutto il pannello;
  - PixMo e LLaVA contro MMStar e MUIRBench;
  - le raccolte documentali (Docmatix) contro DocVQA e MMLongBench-Doc;
  - il train di ChartQA contro val e test;
  - M4-Instruct contro MUIRBench;
  - MathWriting contro hwe e HMER;
  - il replay testuale contro GSM8K e IFEval.
- **Fra i benchmark del pannello:** gli stessi controlli formano i gruppi fra benchmark (§3.1) prima di fissare i ruoli.

## 6. Decisioni dell'utente

1. Approvare il pannello e i ruoli delle §§2-3, oppure indicare le modifiche (D6).
2. Più pagine: MMLongBench-Doc, che richiede un renderer PDF in un ambiente nuovo, oppure MP-DocVQA val (4,7 GB, nessuna
   dipendenza nuova, ma senza domande fra pagine né non rispondibili).
3. Approvare i download della §4, compreso il codice di IFEval.
4. Approvare la costruzione della componente nuova riservata (§3.3): generatore di grafici, tabelle e documenti, ed
   eventualmente un generatore v5 per le scene `function`.
