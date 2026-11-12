# Campione candidato di F0: selezione riproducibile

**Stato al 2026-11-11.** È una proposta. PixMo-Cap ha l'inventario tecnico e la decontaminazione contro `pannello-v3`
(§7, job 30812-30813), il manifest candidato e il reader col manifest (§8, job 30814-30817). Il confronto delle
identità esatte fra sintetici v4, UniMER, MAVIS e PixMo non trova sovrapposizioni né conflitti train/dev (§9, job
30818-30819).
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
- **sviluppo:** gli elementi `sviluppo` di `pannello-v3` e lo split dev dei sintetici v4. Servono a scegliere le
  configurazioni. Test e coppie v4 sono già consultati, quindi valgono come sviluppo e diagnostica;
- **test finale riservato:** gli elementi `test` di `pannello-v3`. In F0 non si consultano: servono solo alla
  decontaminazione.

**Ingressi fissati.**
- Revisione della sorgente e manifest degli shard: ogni manifest ha il job di sharding e lo sha256 di ogni tar.
- Manifest delle ammissioni, dove esiste: `data/manifests/synth-v4-candidato-2`, sha256 `9a8104cd…`. È la
  versione 1 (`975dfe42…`) con la decontaminazione esterna contro `pannello-v3`; split e gruppi non cambiano.
- File del pannello: `data/panel/pannello-v3/elements.jsonl.gz`, sha256 `dacbea54…`.

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
| Geometria, coordinate e grafici di funzioni (sintetici) | `src/limite_vl/synth.py` v4 (`284e8f1`), `data/shards/synth/v4` (job 30447), manifest `synth-v4-candidato-2` (job 30518) | split `train`, stato `ammesso`: 53.997 scene; dev 2.864. Per train e dev il loader costruisce 170.572 QA (coord 45.056, function 35.961, polygon 44.941, shapes 44.614) | calcolata dal codice del generatore e approvata dal checker sul layout (C2) | integrità, riproducibilità, checker su tutti i 60.900 record, gruppi e split, filtro dei loader (job 30447-30451); decontaminazione esterna contro `pannello-v3`: pixel, pHash e domande (job 30518, S2 §2.9) | il checker legge il layout, non i pixel; 367 scene in quarantena, cioè 67 per testi sovrapposti e 300 per i 264 candidati pHash del pannello, propagati al gruppo; audit visivo non fatto, campione preparato (S2 §2.10) | `variant_id` e contratto (C3, C5); audit e verifica dei candidati (D5) |
| Geometria (MAVIS-Geometry) | `CaraJ/MAVIS-Geometry@9c80fff3`, `data/shards/mavis-geometry/9c80fff3-caption` (11 tar, job 29419) | 199.050 immagini `caption.jpg`; esclusi `d3-3228` e 552 didascalie duplicate. Manifest candidato `mavis-geometry-candidato-1` (job 30630): tutti i record ammessi come candidati, 9.951 in dev; `shape_type` in quarantena come compito | regole del motore di MAVIS (`diagram_generation`): descrizione canonica e domande costruite dal codice (`f1_schema.py`, `f1_mavis.py`) | andata e ritorno su 199.050 (job 30033, 30052); valori razionali disegnati come decimali (job 30049) | la coerenza fra due descrizioni non prova la fedeltà ai pixel; solo 40 immagini guardate; `shape_type` ambigua, in quarantena; renderer non nostro; stessi autori di MathVerse | C9 (opzioni di `shape_type`, valore visualizzato ed esatto); reader col manifest (C7); controllo visivo campionario (S4); scansione contro le geometrie del pannello |
| Formule stampate (UniMER-1M) | `wanderkid/UniMER_Dataset@2343ddd9`, `data/shards/unimer/2343ddd9-dense` (5 tar, job 29729; più `lines/` con 30.685 righe ritagliate) | 985.450 formule, meno 51 errori certi (`escluso`) e 1.119 incerti (`quarantena`), di cui 1.116 in 402 gruppi in conflitto. Le 18.827 rietichettature sono tracciate (S2 §4). Manifest candidato `unimer-candidato-1` (job 30625): 983.875 formule ammesse, di cui 49.195 in dev; 1.523 in quarantena e 52 escluse, contando anche le contaminazioni con UniMER-Test; 30.588 righe ritagliate ammesse | LaTeX della sorgente, senza gli spazi del formato im2latex (`latex.despace`); uguaglianza dei token, non equivalenza matematica | `outputs/unimer_check/run4` (job 30047: struttura, rendering, conflitti); sovrapposizioni col test (`outputs/unimer_overlap/run2`) | le formule CROHME manoscritte (meno dell'1%) non si separano per indice: restano (v1.5); un LaTeX compilabile può descrivere un'altra formula | reader col manifest (C7); decontaminazione esterna contro `pannello-v3`; confronto con HME100K di OCRBench |
| Scene reali, descrizioni dettagliate (PixMo-Cap) | `anthracite-org/pixmo-cap-images@3e42775e`, 25 parquet su 75 (`train-00000`-`00024`: il prefisso dei file, non un campione casuale); `data/shards/pixmo-cap/3e42775e` (25 tar, job 29560) | 235.897 righe dei parquet locali, il 33,3% dei 707.647 esempi della scheda; 235.896 negli shard (scartata solo `pixmo-00024-007342`, immagine illeggibile) | descrizione riscritta da un modello (Claude, secondo la scheda) dalle trascrizioni degli annotatori di Ai2, che non sono negli shard; immagini riscaricate da anthracite-org dagli URL | inventario e decontaminazione del 2026-11-11 (§7, job 30813): collegamento originale → shard riga per riga, trasformazione dello sharding riprodotta byte per byte, `pixmo_check/run4` riprodotto riga per riga (25.578 righe con segnali: conteggi 2-12 10.181, colori 8.803, testi citati 3.564); nessun pixel identico al pannello, 248 candidati | testo libero senza verità strutturata: target controllato, non gold (pipeline §5.7); probabili segnaposto (297 righe con lo stesso file di `i.imgur.com`, 146 con quello di `i.ebayimg.com`, gruppi minori); 59 immagini con un orientamento EXIF che lo shard non applica | manifest `pixmo-cap-candidato-1` e reader col manifest fatti (§8): 235.098 record ammessi come candidati per i controlli operativi (11.755 in dev), 798 in quarantena; restano la verifica a vista di duplicati e candidati, l'audit campionario delle didascalie (D5) e la decisione sul training |
| Scene reali, didascalie brevi (LLaVA-Pretrain), solo complemento | `liuhaotian/LLaVA-Pretrain@70f9d1e5`, `data/shards/llava-pretrain/70f9d1e5` (29 tar, job 29388) | 558.128 esempi | didascalie BLIP generate su immagini di LAION, CC e SBU | sha256 degli shard; decontaminazione storica p6 (761 candidati pHash, lista finale vuota); confronto preliminare in §7.8 | testo di un modello, spesso generico; pixel mai decodificati dallo sharding | inventario e decontaminazione come per PixMo (§7.8), se entra nel mix |

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

Le numerosità sono di `pannello-v3` (job 30469). Fra parentesi il test utilizzabile, cioè senza gruppi inammissibili
(pannello §3.1), la base della metrica primaria del test. La verità è quella ufficiale del benchmark; metriche e
margini sono D4.

| Capacità | Sviluppo | Test riservato (utilizzabile) | Provenienza della verità | Incertezze |
| --- | --- | --- | --- | --- |
| Matematica e geometria | MathVista testmini in inglese 936; MathVerse 4.728 (788 problemi in 6 versioni); We-Math 1.740; MathVision testmini 304 più 306 | MathVision 2.734 (2.731); DynaMath 5.000 su 500 problemi (4.960); OlympiadBench 606 multimodali e 674 solo testo; MV-MATH 2.009 (2.003) | annotazioni dei benchmark, in parte dalle sorgenti d'origine (MathVista ne ha 31) | figure di Geometry3K condivise fra benchmark (pannello §1); DynaMath_Sample non è la distribuzione completa |
| Percezione dei diagrammi | Geoperception 11.657; dev dei sintetici v4 2.870 scene (8.779 QA); test e coppie v4 | VisOnlyQA Real 896 (631), Synthetic 700 | annotazioni; codice per i sintetici | 263 geometrie di VisOnlyQA Real hanno candidati nello sviluppo |
| Formule stampate | UniMER-Test 1.261 | UniMER-Test 16.164 | etichette LaTeX | hwe (6.332) è solo diagnostica |
| OCR | OCRBench 525 | OCRBench 475 (447) | annotazioni delle sorgenti; corretto se la risposta è contenuta nell'uscita | metrica permissiva; 150 domande su manoscritto non fanno criterio |
| Documenti a pagina singola | DocVQA 2.681 (468 documenti) | DocVQA 2.668 (2.569; 450 documenti su 470) | risposte estrattive annotate a mano, più risposte ammesse (ANLS) | 81 candidati verso la componente `docVQA` di OCRBench |
| Grafici | ChartQA 2.033 | ChartQA 2.387 (1.730) | domande `human` scritte da annotatori, `augmented` generate; etichette dai dati del grafico | 626 evidenze dirette, soprattutto grafici con la stessa impaginazione |
| Tabelle | DocVQA `table/list` 934; MathVista `table` 70 | DocVQA `table/list` 846 (814) | come DocVQA | nessun test di tabelle pure |
| Poche immagini | MUIRBench 1.134 | MUIRBench 1.466 (1.324, 662 coppie intere); MV-MATH; OlympiadBench, 271 problemi con più immagini | annotazioni curate | `Ordered_Pages` e `Video` fuori perimetro (pannello §3.4) |
| Visione generale | MMStar 855; MUIRBench | MMStar 645 (518) | selezione rivista a mano da benchmark esistenti | sviluppo sbilanciato verso le domande da MathVista |
| Domande senza risposta | MUIRBench 567 varianti | MUIRBench 733 varianti (662) | per costruzione (opzione "None of the choices provided") | nessuna con un'immagine sola |
| Testo | AIME 2024-2026, HMMT feb 2025 e 2026, GSM8K, IFEval | MATH-500, BeyondAIME, APEX 45 (44), OlympiadBench `OE_TO_maths_en_COMP` | risposte ufficiali; controlli a regole per IFEval | un problema di APEX candidato verso AIME 2025 e uno in comune non individuato; i controlli di IFEval mancano |

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
3. **C5 e C7 per le altre sorgenti.** I sintetici v4, UniMER e MAVIS hanno un manifest delle ammissioni e i reader
   filtrati. Per UniMER (`unimer-candidato-2`) e MAVIS (`mavis-geometry-candidato-3`) l'ammissione vale per record,
   split e compito, con la configurazione di collaudo `config/runs/f0-c7-reader-unimer-mavis.json` (C7 del 2026-11-06,
   job 30686-30688, S1 §9). Dal 2026-11-11 anche PixMo-Cap: manifest `pixmo-cap-candidato-1` dall'inventario, senza
   rileggere le immagini, e reader per il compito `pixmo_caption` con la configurazione di collaudo
   `config/runs/f0-c7-reader-pixmo.json` (§8, job 30814-30817). Mancano per LLaVA:
   - record e gruppi;
   - stati, esclusioni e quarantene importate;
   - split di sviluppo per gruppo;
   - contatori per compito e `too_long` per sorgente.
4. **C9 per MAVIS:** `shape_type` con le opzioni, e valore visualizzato ed esatto nel record.
5. **Esclusioni storiche come evidenze (C6):** `unimer_check/run4`, `pixmo_check/run4`, `split_duplicates/run1`, liste di
   `p6` e `f1eval`, ciascuna con motivo, versione e stato. `unimer_check/run4` e `unimer_overlap/run2` sono importate e
   ricalcolate nel manifest di UniMER (2026-11-05); `split_duplicates/run1` vi serve solo come controllo incrociato.
   `pixmo_check/run4` è collegato alle righe correnti e riprodotto riga per riga, e le liste p6 e f1eval di PixMo sono
   risolte per chiave e confrontate con la scansione nuova (2026-11-11, §7.3).
6. **Decontaminazione contro `pannello-v3`.**
   - **Fatta per v4** (job 30517-30518, S2 §2.9): nessuna corrispondenza confermata e 264 candidati pHash, in
     quarantena per gruppo nel manifest `synth-v4-candidato-2`.
   - **Fatta per UniMER e MAVIS** (job 30651-30652, S2 §§3-4), con le versioni nuove `unimer-candidato-2` e
     `mavis-geometry-candidato-2`: corrispondenze confermate escluse e candidati in quarantena per gruppo, da risolvere.
     Per MAVIS la versione corrente è `mavis-geometry-candidato-3` (job 30676), con le decisioni testuali ricalcolate
     senza la regola di frequenza.
   - **Fatta per PixMo** (job 30813, §7.4): nessun pixel identico, 248 candidati da verificare, proposti in quarantena.
   - Per LLaVA serve ancora una scansione con pixel, pHash e testo, e poi la risoluzione dei candidati.
7. **Verifica visiva:**
   - l'audit di v4 (S2 §2.8, D5): campione di 380 unità e 175 casi mirati, con i fogli corretti di `campione-2` (S2
     §2.11), revisione non fatta;
   - i 264 candidati della decontaminazione di v4, con tutte le loro corrispondenze nel foglio di `campione-2`;
   - non è un prerequisito la verifica dei 743 gruppi inammissibili del pannello (1.300 evidenze dirette, 191 forti):
     restano fuori dalla metrica primaria, che usa solo il test utilizzabile, e si verificano solo se servono a un
     compito preciso (chiarimento dell'utente del 2026-11-04);
   - campioni di MAVIS (fedeltà ai pixel) e di PixMo (dettagli aggiunti);
   - per PixMo anche un'immagine per ciascun gruppo di duplicati con didascalie diverse (probabili segnaposto) e i 248
     candidati della decontaminazione (§7.7);
   - l'osservabilità alla risoluzione scelta.
8. **Più immagini (punto 4 del piano).** `Sample.images`, loader e riferimenti nel testo servono per MUIRBench, MV-MATH,
   OlympiadBench e per una sorgente di training con più immagini.
9. **Valutatore del modello VL (punto 7).** Metriche per capacità (D4) ed estrattori (C4).
10. **Sorgenti delle lacune (§4).** Download da approvare.

## 6. Prossimo lotto minimo di implementazione

Il lotto è indipendente dalle decisioni aperte: si può fare dopo l'ok, con job CPU di pochi minuti ciascuno, senza GPU
né agenti.

1. **Verifica visiva dei gruppi inammissibili del pannello** (`groups_excluded.jsonl.gz` di `pannello-v3`),
   facoltativa: i 743 gruppi possono restare esclusi dalla metrica primaria (chiarimento del 2026-11-04). Si fa solo
   per i gruppi che bloccano un compito.
   - Fogli di coppie e un TSV degli esiti per gruppo, a partire dai 191 candidati forti.
   - Le immagini si ritrovano con `panel_index.resolve_entries`, già verificato sull'indice.
   - La correlazione dei pixel a 64×64 di `decontam_review.py` serve a ordinarli.
   - Revisione umana; l'esito va in `pannello-v4`.
2. **Decontaminazione esterna di v4 contro `pannello-v3`.** Fatta il 2026-11-02 (job 30517-30518, S2 §2.9), con il
   manifest nuovo `synth-v4-candidato-2`. Il campione per l'audit delle golden answer è preparato (job 30522, S2
   §2.10), e i fogli sono corretti il 2026-11-04 con la stessa selezione (job 30556, `campione-2`, S2 §2.11). Restano la
   revisione visiva e la verifica dei candidati.
3. **Manifest delle ammissioni di UniMER-1M e MAVIS-Geometry,** col sottoinsieme di S1 già usato per v4:
   - record, gruppi e stati;
   - esclusioni e quarantena importate da `unimer_check/run4`;
   - per MAVIS, `shape_type` in quarantena;
   - dev al 5% per gruppo con seme 0;
   - il rango di estrazione di §1.

   MAVIS fatto il 2026-11-05: `data/manifests/mavis-geometry-candidato-1` (job 30626 e 30630, S2 §3), con `shape_type` in
   quarantena per compito; la fedeltà all'immagine resta da verificare.
   UniMER fatto il 2026-11-05: `data/manifests/unimer-candidato-1` (job 30624-30625, S2 §4). È un candidato, non letto
   dai loader.
4. **Scansione di MAVIS, UniMER, PixMo e LLaVA contro `pannello-v3`** (pixel e pHash, testo dove c'è). Dà candidati da
   mettere in quarantena, non esclusioni automatiche. La durata va misurata prima su uno shard per sorgente.

   UniMER e MAVIS fatti il 2026-11-06 (`scripts/panel_decontam.py`, S2 §§3-4), dopo prove limitate (job 30646, 30647,
   30649): pixel sugli originali dello snapshot, sui PNG e sui ritagli; formule contro UniMER-Test e OCRBench; pHash e
   10-gram per MAVIS. Le decisioni testuali di MAVIS sono state rifatte (job 30676, `mavis-geometry-candidato-3`) con
   regole esplicite al posto della frequenza nella sorgente. PixMo fatto il 2026-11-11 (job 30812-30813, §7). Resta
   LLaVA.

Dopo D3 vengono C3, C4 e le varianti (`variant_id`), poi il punto 4 (più immagini) e il valutatore (punto 7).

**Decisioni aperte, dell'utente:**
- D1, checkpoint Qwen;
- D2, budget;
- D3, contratto;
- D4: metriche, margini, sottoinsieme di MUIRBench nel perimetro, OCRBench senza manoscritto;
- D5, budget degli audit;
- D6, approvazione di `pannello-v3`: quote, eventuale stratificazione, regola degli irrisolti;
- i download di §4;
- volumi e mix dei piloti;
- risoluzione e token per immagine;
- le proposte di trattamento di PixMo (§7.7).

## 7. PixMo-Cap: preparazione deterministica (2026-11-11)

Lotto chiesto dall'utente il 2026-11-11: inventario, collegamento, evidenze storiche e decontaminazione dei dati
locali di PixMo-Cap, riusando gli strumenti esistenti. Senza modelli, audit con agenti, GPU, download o nuovi pacchetti
di revisione.
- **Strumenti** (`afe2b27`): `scripts/pixmo_inventory.py`, che legge ogni parquet e il suo shard una sola volta, il
  launcher `slurm/pixmo_inventory.sbatch` e `tests/test_pixmo_inventory.py` (49 controlli, job 30811).
- **Job:** sonda su uno shard (job 30812, 32 controlli su 32), lavoro completo (job 30813, 36 controlli su 36).
- **Rapporto della verifica:** `docs/inventari/pixmo-cap-completo-1.json`.
- Nessun record è ammesso, escluso o promosso, e nessuna lista è collegata ai loader.

### 7.1 Inventario verificato e copertura

- **Sorgente.** `anthracite-org/pixmo-cap-images@3e42775eea79dd4874c41379230c7f871b10218e`: è la ripubblicazione di
  `allenai/pixmo-cap` con le immagini scaricate dagli URL. La scheda dichiara 707.647 esempi in 75 parquet (371,8 GB).
- **Sottoinsieme locale.** Sono i parquet `train-00000`-`00024` (filtro di `downloads/sources.tsv`):
  124.853.885.806 byte, con gli sha256 ricalcolati nella scansione uguali ai nomi dei blob e al job 29554. È il prefisso
  dei file, non un campione casuale.
- **Righe.** 235.897, cioè il 33,34% degli esempi dichiarati, con 95 gruppi di righe per file.
- **Shard.** 235.896 record nei 25 tar del job 29560, con sha256, byte e campioni uguali a `shards.jsonl`.
- **Uscite** (fuori da git):
  - inventario `outputs/pixmo_inventory/completo-1/inventario/records.jsonl.gz`, una riga per riga dei parquet (120 MB);
  - scansione `data/decontam/pannello-v3/pixmo-cap/records.jsonl.gz` (55 MB);
  - verifica `outputs/pixmo_inventory/completo-1/verifica`: rapporto, `duplicati.jsonl.gz` e
    `proposte_di_trattamento.jsonl.gz`.
- **Riga dell'inventario.**
  - Identità: file, riga e gruppo di righe; shard, posizione e membri, oppure il motivo dello scarto; `record_id` di S1
    §1 (`pixmo-cap`, revisione, chiave).
  - Immagini: digest dell'originale (byte, pixel e pHash nella convenzione `rgb-su-bianco/1`, dimensioni, formato,
    modo, orientamento EXIF, fotogrammi) e del JSON e del JPEG dello shard.
  - Testi: digest della didascalia originale e di quella del loader, delle trascrizioni e dei metadati; URL; richiesta
    del loader.
  - Esiti: segnali e difetti, in campi separati.
- **Composizione.**
  - Formati: JPEG 200.436, PNG 34.192, WebP 1.125, GIF 111, MPO 32.
  - Modi: RGB 214.340, RGBA 14.869, P 5.434, L 1.079, CMYK 142, LA 20, 1 11, I;16 1.
  - Lato lungo: 8 immagini sotto 64 px, 1.545 sotto 256, 268 da 8192 in su.
  - Didascalie: 174.990 fra 100 e 199 parole, 75 da 500 in su, una sotto 10.
  - Trascrizioni per riga: una 137.719, tre 86.248, due 5.736, quattro 1.948, cinque o più 4.246.

### 7.2 Collegamento e difetti strutturali

- **Chiave → parquet → shard.**
  - Le chiavi sono uniche e uguali a `pixmo-<file>-<riga>`.
  - Ogni record dello shard ritrova la sua riga, nello stesso ordine, e non ci sono record senza riga.
  - I JSON sono coerenti con la riga: id, didascalia, URL, dimensioni originali e finali, `max_patches`, formato,
    sorgente.
- **Trasformazione.**
  - `shard_parquet.convert` (`to_rgb`, 1024 patch, BILINEAR, JPEG 95) rifatta sugli originali dà gli stessi byte in
    235.896 record su 235.896.
  - L'immagine dello shard è ridimensionata e ricodificata, quindi non ha i pixel dell'originale. La distanza pHash fra
    originale e shard è di 0-2 bit in 235.490 righe, 4 bit in 264, 6 o più in 142.
- **Difetti strutturali dimostrati.** Uno solo: `pixmo-00024-007342`, con l'immagine originale illeggibile
  (`UnidentifiedImageError`). Lo sharding l'aveva già scartata, quindi è fuori dal training per costruzione. Nessuna
  didascalia vuota, nessuna riga senza trascrizioni, nessuna chiave ripetuta, nessuna riga mancante.
- **Collegamenti ambigui per contenuto,** risolti dalla chiave:
  - 116 gruppi con gli stessi byte dell'originale (692 righe), 193 con gli stessi pixel (876), 194 con lo stesso JPEG
    nello shard (878);
  - 72 coppie di righe con lo stesso URL e didascalie diverse;
  - nessuna didascalia ripetuta.
- **Segnali strutturali forti, non dimostrati: probabili segnaposto.** Sono gruppi di righe con la stessa immagine e
  URL e didascalie tutti diversi.
  - 297 righe con il file di 503 byte e 161×81 px di `i.imgur.com`: la misura sembra quella dell'immagine che imgur
    mostra per le immagini rimosse, ma è da confermare a vista.
  - 146 righe con un file di 80×80 px di `i.ebayimg.com`.
  - 8 righe di `pixmo.s3.us-west-2.amazonaws.com` (512×512) e 8 di `www.japantimes.co.jp` (800×600).
  - Altri gruppi da 5, 5, 4 e 3 righe, e 10 gruppi con gli stessi pixel e byte diversi, fino a 21 righe: 18 gruppi da 3
    righe in su, 526 righe.

  Se l'immagine è un segnaposto, la didascalia descrive un'altra immagine. Per deciderlo basta guardare un'immagine per
  gruppo.
- **Altri segnali da verificare, non difetti.**
  - 59 immagini con l'orientamento EXIF 3, 6 o 8: lo shard e l'indice non lo applicano, un browser sì. Altre 198 hanno
    il valore non valido 0 e 2 valori fuori scala.
  - 135 immagini con più fotogrammi (GIF e WebP animate, MPO): si usa il primo.
  - 142 immagini CMYK convertite senza profilo colore; 14.869 RGBA e 20 LA composte su bianco.

### 7.3 Segnali storici delle didascalie (run4) e loro limiti

- **Collegamento.** `outputs/pixmo_check/run4` (job 30044, schema `pixmo-transcript-check/4`) copre gli stessi 25
  parquet della revisione, con gli stessi byte di oggi: 235.897 righe, nessuna senza trascrizioni. Le chiavi di
  `flagged.jsonl` si risolvono ciascuna in una sola riga.
- **Provenienza.** run4 viene dal codice `00d80e3` con modifiche non salvate, quindi la provenienza non bastava. I
  segnali sono stati ricalcolati nello stesso passaggio sugli stessi ingressi, col codice corrente (`transcripts.py` di
  `25b86c8`).
- **Esito.** Uguali riga per riga e negli stessi conteggi: 25.578 righe con almeno un segnale (10,8%). Conteggi 2-12
  10.181, colori 8.803, testi citati 3.564, altri numeri 1.978, anni 1.730, orari 715.
- **Limiti.** Sono segnali di rischio, non difetti.
  - Un numero, un colore o un testo assente dalle trascrizioni può essere vero nell'immagine.
  - Il controllo non vede oggetti, posizioni, relazioni e attributi aggiunti dal riscrittore, né gli errori percettivi
    degli annotatori.
  - Restano nell'inventario come evidenze, senza esclusioni automatiche.
- **Liste storiche di decontaminazione.**
  - Sono p6 e f1eval: `decontam.py scan` sui JPEG degli shard, convenzione `grigio-diretto/0`, solo pHash, contro
    indici diversi dal pannello. Contano 183 e 15 candidati, con le liste finali vuote.
  - Ogni chiave si risolve in una riga, e tutti i candidati storici restano candidati nella scansione nuova.
  - Il confronto è diagnostico: le liste non si importano.

### 7.4 Decontaminazione contro `pannello-v3`

- **Indice.** `data/decontam/pannello-v3/index.json` (sha256 `02fb3cbf…`, convenzione `rgb-su-bianco/1`, 99.713 voci),
  in tutti i ruoli: sviluppo 30.189, test ammesso 35.701, test non ammesso 1.368, diagnostica 6.332,
  solo_decontaminazione 12.401 elementi.
- **Controlli,** eseguiti su ogni record dello shard, senza mancanti:
  - pixel normalizzati dell'originale e del JPEG dello shard;
  - pHash entro 4 bit delle due immagini;
  - 10-gram della didascalia del loader, con la regola di `TextIndex`: dal lato dell'indice si tolgono gli n-gram in
    più di 25 testi diversi del pannello. Nessuna regola copre le didascalie, e la frequenza in PixMo non decide nulla;
  - le sei richieste fisse del loader: nessun 10-gram in comune e nessun testo identico, ma quattro sono più corte di
    10 parole.

| Esito | Righe | Nota |
| --- | --- | --- |
| Identità d'immagine confermata (pixel identici) | 0 | né sull'originale né sul JPEG dello shard |
| Somiglianza da verificare (candidato) | 248 | 226 pHash, 22 testo |
| Controllo non eseguibile | 1 | `pixmo-00024-007342`, fuori dagli shard |
| Nessuna corrispondenza nei controlli effettuati | 235.648 | non certifica la didascalia né l'assenza di contaminazione |

- **pHash per benchmark** (righe): MathVision 93, DynaMath 85, MMStar 23, MathVista 21, ChartQA 20, OCRBench 16,
  VisOnlyQA Real 13, UniMER-Test 9, MUIRBench 7, MV-MATH 6, We-Math 5, VisOnlyQA Synthetic 2.
  - Sono toccati 357 elementi del pannello.
  - Alcune righe hanno molti vicini: le 20 righe verso ChartQA hanno 785 corrispondenze col solo sviluppo. Succede alle
    immagini con poca struttura, ma qui non è verificato.
- **Testo** (22 righe, 18 elementi): AIME 2024 8, OlympiadBench 5, MATH-500 3, MathVision 3, UniMER-Test 3, AIME 2025 2,
  MMStar 1, MV-MATH 1.
  - I 10-gram in comune sono sequenze di cifre o di lettere (`1 2 3 4 5 6 7 8 9 10`, `a b c d e f g h i j`), frasi
    generiche (`from the lower left corner to the upper right corner`, 8 righe) e una formula della gaussiana.
  - Restano candidati.
- Tutte le corrispondenze sono nella scansione, senza tagli. Un'identità d'immagine non dimostrerebbe l'identità della
  domanda, e l'assenza di corrispondenze vale solo per i controlli dichiarati.

### 7.5 Controlli non eseguiti e proprietà non verificate

- **Non eseguiti per costruzione:**
  - trascrizioni contro il pannello, perché il modello non le vede;
  - URL e provenienza, perché il pannello non ha URL;
  - ritagli, ridisegni, rotazioni e riformulazioni, oltre ai pixel identici e al pHash entro 4 bit;
  - la sola didascalia sotto 10 parole.
- **Fuori dal sottoinsieme:** i 50 parquet non scaricati, con 471.750 esempi dichiarati.
- **Non verificate:**
  - fedeltà delle didascalie all'immagine;
  - corrispondenza fra l'immagine riscaricata e quella vista dagli annotatori;
  - rappresentatività del prefisso dei file;
  - orientamento visto dagli annotatori;
  - leggibilità a 1024 patch;
  - duplicati quasi identici dentro PixMo (qui solo identità esatte) e fra sorgenti (niente indice globale in questo
    lotto).

### 7.6 Costi

- **Sonda** (job 30812): un parquet, 9.436 righe in 107 s con 8 CPU. Preventivo: 25 × 107 s ≈ 45 min.
- **Lavoro completo** (job 30813): 48 min in tutto.
  - Tempi: test 2 min, scansione 45 min 26 s, verifica 27 s; fase dell'indice 63 s; unità fra 103 e 112 s.
  - Memoria: RSS massima 7,5 GiB nella scansione e 3,3 GiB nella verifica.
  - CPU dei processi: 20.845 s, di cui 14.420 per decodifica, pixel e pHash degli originali, 4.942 per la
    trasformazione, 1.122 per lo shard, 274 per i segnali e 88 per la decontaminazione.
  - Byte letti: 124,85 GB per gli sha256 (in parallelo, senza attese), 124,84 GB di colonne e 20,57 GB di shard.
  - Spazio: 175 MB di uscite, più 7 MB della sonda.

### 7.7 Proposte di trattamento e verifiche visive eventuali

**Proposte**, da approvare. Stanno in `proposte_di_trattamento.jsonl.gz`, che nessun loader legge.
- **248 candidati:** quarantena fino alla verifica, a vista per il pHash e leggendo il testo in comune per i 10-gram.
- **Gruppi di immagini identiche con didascalie diverse:** si guarda un'immagine per gruppo. Se è un segnaposto, si
  escludono tutte le righe del gruppo (proposta, non applicata). Nel manifest i duplicati esatti stanno comunque nello
  stesso gruppo.
- **Orientamento EXIF 3, 6 o 8** (59 immagini): quarantena, oppure uno shard nuovo con l'orientamento applicato. È da
  decidere.
- **Riga illeggibile:** è già fuori.

**Verifiche visive eventualmente necessarie,** dopo quelle in corso:
1. un'immagine per ciascuno dei 18 gruppi di duplicati da 3 righe in su;
2. le coppie dei 226 candidati pHash, ordinate con `decontam_review.py`;
3. i 22 candidati testuali, leggendo solo i testi;
4. un campione stratificato per le didascalie, coi segnali di run4 (D5);
5. le 59 immagini ruotate.

### 7.8 LLaVA-Pretrain: confronto preliminare, senza scansioni nuove

- **Revisione.** `liuhaotian/LLaVA-Pretrain@70f9d1e5e1a697fe35830875cfc7de1dd590d727`: `blip_laion_cc_sbu_558k.json`
  (180.506.011 byte) e `images.zip` (27.356.108.382 byte), con gli sha256 verificati (job 29387, 29554).
- **Shard.** `data/shards/llava-pretrain/70f9d1e5` (job 29388, `scripts/shard_llava_pretrain.py`): 29 tar da circa 1 GB
  (27 GiB), con 558.128 campioni, tanti quanti i file dello zip e le didascalie.
  - Le immagini hanno i byte originali dello zip, senza ricodifica.
  - Il JSON ha id, membro dello zip, richiesta, didascalia, dimensioni e formato.
  - Il lato corto è quasi sempre di 336 px: mediana 336×336, massimo 12192×5177.
  - Lo sharding ha letto solo le intestazioni: i pixel non sono mai stati decodificati.
- **Didascalie.** Sono brevi e vengono da BLIP, su immagini di LAION, CC e SBU (secondo il nome del file e il catalogo).
  Le richieste vengono dal JSON del dataset, non dal nostro codice.
  - È testo di un modello, senza riferimento umano: manca l'equivalente delle trascrizioni di PixMo, quindi non c'è un
    segnale deterministico sulle didascalie.
  - Immagini e didascalie sono distribuite insieme, nello zip e nel JSON. Non è verificato che le didascalie siano state
    generate proprio su quelle immagini.
- **Evidenze.**
  - Scansione storica p6 (`decontam.py`, `grigio-diretto/0`, solo pHash, contro MathVista, MathVerse, MathVision,
    We-Math e DynaMath_Sample): 761 candidati, lista finale vuota (`data_lists.REQUIRED`).
  - Nessuna scansione f1eval, nessun controllo dei duplicati.
- **Riusabili:**
  - `data.llava_pretrain_sample` e la lettura degli shard;
  - `shard_llava_pretrain.py` per il collegamento membro dello zip ↔ record;
  - gli helper di `panel_decontam.py` e `limite_vl.decontam`;
  - lo schema di `pixmo_inventory.py`: inventario per riga, verifica, proposte.
- **Lavoro necessario.**
  - Un adattamento della lettura: zip e JSON invece dei parquet. Non c'è una trasformazione da rifare, perché lo shard
    ha i byte dell'originale: basta confrontare gli sha256.
  - La decodifica completa delle immagini e i duplicati.
  - La decontaminazione contro `pannello-v3`: pixel e pHash, e 10-gram di didascalie e richieste, in gran parte più
    corte di 10 parole e quindi poco confrontabili.
  - Manifest e reader.
  - Stima per analogia con PixMo (immagini da circa 0,1 MP invece di 1-4 MP): 20-30 minuti con 8 CPU, da misurare con
    una sonda.
- **Confronto con PixMo.**
  - PixMo ha descrizioni dense (in maggioranza 100-300 parole), nate da annotatori, un controllo deterministico contro
    le trascrizioni e immagini grandi.
  - LLaVA ha didascalie brevi di un modello e immagini piccole, con 2,4 volte gli esempi, poco costosi in token (circa
    312 per esempio a 1024 patch).
  - Come alternativa per le descrizioni dettagliate non serve, perché è un altro compito. Come complemento può dare
    didascalie brevi per l'allineamento (F1), solo se un pilota lo richiede e dopo la stessa preparazione.

### 7.9 Prossimo passo minimo

1. Decisioni dell'utente sulle proposte di §7.7: segnaposto, candidati, orientamento. *Prese il 2026-11-11 come
   trattamento operativo: tutto in quarantena (§8).*
2. Poi, senza rileggere le immagini, il manifest candidato `pixmo-cap-candidato-1` dall'inventario (C5):
   - record, gruppi per pixel dell'originale e per URL, stati dalle proposte approvate;
   - evidenze per riferimento: scansione, run4, liste storiche;
   - dev per gruppo col seme 0.

   Quindi il reader di `pixmo_caption` col manifest (C7). *Fatti il 2026-11-11 (§8), con i gruppi estesi anche
   all'immagine dello shard.*
3. LLaVA solo se il mix lo richiede, con una sonda e la scansione con lo stesso schema.

## 8. PixMo-Cap: manifest candidato e reader (2026-11-11)

Lotto chiesto dall'utente il 2026-11-11: rendere il sottoinsieme locale caricabile per i collaudi di F0, partendo dalla
preparazione di §7.
- **Non autorizza il training** e non certifica le didascalie.
- **Strumenti** (`8eb716c`):
  - `scripts/pixmo_manifest.py` costruisce e verifica il manifest dall'inventario, dalla verifica e dalla scansione del
    job 30813, senza rileggere parquet né immagini;
  - il reader di C7 è esteso al compito `pixmo_caption`;
  - `slurm/pixmo_manifest.sbatch`, `tests/test_pixmo_manifest.py`.
- **Job:** test (30814), costruzione e verifica (30815), collaudo del reader (30816), regressione di UniMER e MAVIS
  (30817): 17 min 34 s di job CPU in tutto.

### 8.1 Percorsi, digest e ingressi

- **Manifest** `data/manifests/pixmo-cap-candidato-1`: `manifest.json` `794787ab…`, `records.jsonl.gz` `d3eee365…`
  (235.896 righe, 92 MB), `keys/` (train, dev, non ammessi).
- **Verifica** `outputs/manifest_check/pixmo-cap-candidato-1/verifica.json`: 15 controlli su 15.
- **Riepilogo** versionato `docs/manifests/pixmo-cap-candidato-1.json`.
- **Ingressi verificati prima di derivare:**
  - inventario (`records.jsonl.gz` `ade3e0c2…`) e scansione (`767658e0…`) del job 30813, con gli sha256 di summary.json,
    quelli ritrovati dalla verifica e quelli attesi dal launcher;
  - verifica senza controlli falliti, con duplicati e proposte coi digest del rapporto;
  - indice `pannello-v3` `02fb3cbf…`;
  - sharding del job 29560, coi tar presenti e la dimensione registrata. Gli sha256 dei tar vengono dal registro, già
    ritrovati nel job 30813, e il reader li verifica a ogni caricamento.
- run4 e liste storiche sono registrati solo per riferimento.

### 8.2 Trattamento applicato

Sono le scelte dell'utente. I conteggi autorizzati sono passati al launcher, e la costruzione si ferma se non tornano.
- **Candidati della decontaminazione:** 248 righe in quarantena.
- **Possibili segnaposto:** i 18 gruppi da almeno 3 righe per pixel dell'immagine originale, presi dall'elenco completo
  dei duplicati, 526 righe. È una quarantena cautelativa: non un errore dimostrato.
- **Orientamento EXIF 3, 6 o 8:** 59 righe in quarantena, senza rotazioni né shard nuovi.
- **Riga illeggibile:** `pixmo-00024-007342` non ha record ed è contata in `scarti_dello_sharding`.
- **Segnali di run4:** restano evidenze nei record (25.578), senza effetto sullo stato.
- **Altri duplicati:** si raggruppano senza scarti.

I blocchi si propagano al gruppo con la politica già adottata (`panel_decontam.decide`): lo stato del gruppo è il più
grave fra quelli dei suoi record, senza riammissioni. Ogni record registra i blocchi diretti (`blocchi.diretti`) e quelli
ereditati dagli altri record del gruppo (`blocchi.ereditati`).

### 8.3 Gruppi: prima e dopo le unioni

Le unioni sono transitive, per uguaglianza esatta.

| Relazione | Legami | Gruppi con più record | Record coinvolti |
| --- | --- | --- | --- |
| Pixel dell'immagine originale | 683 | 193 | 876 |
| Pixel del JPEG dello shard | 685 | 194 | 879 |
| Byte del JPEG dello shard | 684 | 194 | 878 |
| Stesso URL non vuoto | 72 | 72 | 144 |

- **Gruppi cumulativi:** 235.213 col solo originale, 235.211 dopo il pixel dello shard. Le immagini finali identiche
  uniscono 3 record con originali diversi; i byte del JPEG e gli URL non aggiungono altro.
- **Esito:** 235.211 gruppi, 194 con più record (879 record), il più grande di 297.
- Ogni legame resta dentro un solo gruppo e un solo split: gli input finali identici non attraversano train e dev
  (controllo della verifica, zero eccezioni per relazione).

### 8.4 Stati, quarantene e split

| Split | Ammessi | Quarantena | Esclusi | Gruppi |
| --- | --- | --- | --- | --- |
| train | 223.343 | 798 | 0 | 223.461 (288 in quarantena) |
| dev | 11.755 | 0 | 0 | 11.750 |

- **Quarantene dirette:** 798 record.
  - Solo candidati 213; candidati che sono anche possibili segnaposto 35; solo segnaposto 491; solo EXIF 59.
  - EXIF e segnaposto non si sovrappongono.
- **Quarantene ereditate:** nessun record è in quarantena solo per eredità. Un record ha, oltre al proprio, anche il
  motivo segnaposto ereditato da un gruppo unito per l'immagine dello shard.
- **Split:** dev per gruppo col seme 0 e la quota dei manifest esistenti (5% dei record ammessi): 11.755 su 235.098, cioè
  il 5,000%. I gruppi in quarantena restano in train, dove nessun record si legge.
- **Segnali di run4:** 25.492 record ammessi li hanno come evidenza (1.256 in dev), più 86 in quarantena per altri
  motivi.
- **"Ammesso"** riguarda i controlli operativi effettuati: collegamento, integrità, decontaminazione nei controlli
  dichiarati, compito `pixmo_caption`. La fedeltà delle didascalie ai pixel resta da valutare, e il training richiede
  una decisione successiva.

### 8.5 Reader

- **Codice condiviso:**
  - `pixmo-cap` entra in `manifest.TASKED`, col compito `pixmo_caption` sull'insieme train, e in `MANIFEST_REQUIRED`:
    gli shard locali di PixMo non si leggono senza manifest;
  - `Admission.check` controlla il JPEG quando lo shard non ha PNG;
  - la decisione allegata porta lo sha256 della didascalia, che `data.pixmo_caption_sample` verifica senza cambiare
    didascalia, immagine o richiesta;
  - `data_lists.MANIFEST_FAMILIES` comprende `pixmo-cap`.
- **Invariati:** tokenizer, template, prefill, contratto e config storiche.
- **Config di collaudo:** `config/runs/f0-c7-reader-pixmo.json`. È solo collaudo CPU, non una ricetta.
- **Collaudo sui dati reali** (job 30816, `docs/reader_smoke/c7-pixmo-1.json`): 8 controlli su 8.
  - Il manifest è identificato dallo sha256 della config, e i 25 shard sono verificati.
  - Il primo shard è letto per intero. In train consegna esattamente gli 8.928 ammessi e salta 471 record di dev e 37
    in quarantena, coi motivi attesi; consegna 976 ammessi coi segnali di run4. In dev consegna 471 su 471.
  - 318 esempi col processor senza errori, con didascalia, richiesta e JPEG invariati e coi digest del manifest.
  - Validazione dal dev: 200; flusso misto: 630 esempi, tutti ammessi di train.
- **Regressione** (job 30817): i reader di UniMER e MAVIS danno 22 controlli su 22. `docs/reader_smoke/c7-4.json` è
  uguale a `c7-3`, tolti job, tempi, memoria e identità del codice.

### 8.6 Test, costi e limiti

- **Test** (job 30814): `tests/test_pixmo_manifest.py` 33 su 33. Copre:
  - unioni transitive, comprese immagini finali identiche con originali diversi;
  - blocchi sovrapposti, propagati ed ereditati;
  - run4 senza effetto;
  - manifest e verifica su una fixture vera, arresti, manifest alterato;
  - reader.

  Regressioni: `test_reader_manifest` 48 su 48, `test_manifest` e `test_data_lists` superati.
- **Costi:**
  - costruzione e verifica in 97 s, con RSS di 5,0 GiB. Si rileggono solo l'inventario (120 MB), la scansione (55 MB) e
    i file piccoli della verifica: nessun parquet né immagine;
  - collaudo del reader in 96 s, con RSS di 2,4 GiB. Rilegge i 25 shard (20,6 GB) per gli sha256, come vuole il
    contratto del reader, e il primo shard due volte;
  - spazio: 92 MB del manifest.
- **Limiti aperti:**
  - la fedeltà visiva delle didascalie (oggetti, posizioni, relazioni, attributi, testi) non è valutata;
  - i 248 candidati e i gruppi di possibili segnaposto non sono risolti, e restano in quarantena;
  - non si sa quale orientamento abbiano visto gli annotatori, né se l'immagine riscaricata dagli URL sia quella
    annotata;
  - la leggibilità a 1024 patch non è controllata;
  - mancano `variant_id`, `uid` e i digest dei token, che dipendono dal contratto (D3).

## 9. Identità esatte fra le sorgenti: primo sottoinsieme di C6 (2026-11-11)

Lotto chiesto dall'utente il 2026-11-11: sovrapposizioni esatte fra sintetici v4, UniMER, MAVIS e PixMo-Cap e possibili
conflitti train/dev, dai manifest e dalle scansioni già fatti, senza rileggere immagini e senza cambiare split.
- **Non è** l'indice globale di C6 né la decontaminazione globale. Nessun manifest, gruppo, split o stato è cambiato, e
  nessuna lista va ai loader.
- **Strumenti** (`26cb82d`): `scripts/exact_overlap.py`, `slurm/exact_overlap.sbatch`, `tests/test_exact_overlap.py`.
- **Job:** test e prova limitata (30818, 12 s), confronto completo (30819, 1 min 36 s). In tutto 1 min 48 s di job CPU su
  30, con 2 CPU e RSS massima di 2,2 GiB.
- **Evidenze:** fuori da git in `outputs/exact_overlap/completo-1` (`report.json`; `componenti.jsonl.gz`, vuoto).
  Riepilogo versionato: `docs/indice_globale/identita-esatte-completo-1.json`.

### 9.1 Ingressi e verifiche

| Sorgente | Manifest (`manifest.json`) | Record | Scansione citata dal manifest | Scansione fatta su |
| --- | --- | --- | --- | --- |
| Sintetici v4 | `synth-v4-candidato-2` `9a8104cd…` | `5863b06a…`, 60.900 | `pannello-v3/synth-v4` `38554b53…` (job 30518) | stessi record (il manifest cita la scansione) |
| UniMER-1M | `unimer-candidato-2` `97af1033…` | `7b33bc25…`, 1.016.135 | `pannello-v3/unimer` `2de700c4…` (job 30651) | candidato-1 `bcc85198…`, il genitore |
| MAVIS | `mavis-geometry-candidato-3` `f3cca473…` | `8d199336…`, 199.050 | `pannello-v3/mavis-geometry` `f953c323…` (job 30652) | candidato-1 `dc0f264c…`, nella catena |
| PixMo-Cap | `pixmo-cap-candidato-1` `794787ab…` | `d3eee365…`, 235.896 | `pannello-v3/pixmo-cap` `767658e0…` (job 30813) | ingresso del manifest |

- **MAVIS candidato-5** (`314a0a52…`) sta nella catena di candidato-3, con le stesse righe. Resta il riferimento della
  revisione C9, non una sorgente in più: record, gruppi e split sono ereditati, quindi l'esito vale anche per lui.
- **Controlli:** 28 su 28.
  - sha256 di manifest, record e scansioni uguali a quelli registrati;
  - record_id ricalcolati, sorgente, revisione, shard e conteggi;
  - voci (insieme, chiave) e posizioni uniche, ogni gruppo in un solo split;
  - righe ritagliate di UniMER con la chiave, il record_id e il gruppo della loro formula: sono due voci e un record;
  - ogni riga della scansione ricollegata a una sola voce, coi digest del manifest (byte dell'immagine dello shard; JSON
    per UniMER e MAVIS; pixel, originale, file, riga e didascalia per PixMo), e ogni voce con la sua riga.
- **Convenzione:** tutte le scansioni dichiarano `rgb-su-bianco/1`, e sono state fatte con lo stesso ambiente
  (`build-29382`, quindi lo stesso decodificatore).

### 9.2 Identità confrontate e copertura

Ogni digest è confrontato con tutti quelli dello stesso tipo, di ogni sorgente e rappresentazione: anche l'originale di
una sorgente con lo shard di un'altra. Nessun hash è ricalcolato.

| Sorgente | Originale: byte e pixel | Shard (immagine letta dal loader): byte e pixel | Ritagli delle righe: pixel |
| --- | --- | --- | --- |
| Sintetici v4 | non distinto: il PNG generato è l'originale | 60.900 (train 60.000, test 300, coppie 600) | — |
| UniMER-1M | 985.450 formule e 30.685 righe, che hanno l'originale della formula | 1.016.135 PNG densi | 81.606 ritagli di 30.685 righe |
| MAVIS | 199.050 `caption.jpg` | 199.050 PNG a 2304 patch | — |
| PixMo-Cap | 235.896 | 235.896 JPEG a 1024 patch | — |

- **Copertura:** il 100% delle voci per ogni rappresentazione registrata. Nessuna immagine illeggibile, nessun problema
  registrato dalle scansioni, nessuna voce senza riga. In tutto 1.511.981 voci, 1.379.556 gruppi e 5.509.454 digest
  distinti.
- **Non confrontati:**
  - il pixel RGBA del manifest dei sintetici, che è un'altra convenzione: si usa il pixel rgb della scansione;
  - i byte dei ritagli, che non sono file;
  - la riga illeggibile di PixMo scartata dallo sharding, che non ha una voce.

### 9.3 Esito

- **Nessuna sovrapposizione esatta fra sorgenti.** Per ognuna delle sei coppie: 0 classi di digest uguali, 0 componenti,
  0 record condivisi.
- **Nessun legame nuovo dentro una sorgente.** Ogni identità esatta di byte o pixel sta già dentro un gruppo del suo
  manifest, anche fra rappresentazioni diverse (l'originale di un record uguale allo shard di un altro) e sui ritagli.
  Ogni componente delle identità esatte sta dentro un solo gruppo operativo, che può essere più largo (stesso
  `diagram_generation` in MAVIS, stesso URL in PixMo, stato della scena e coppie nei sintetici): per queste identità i
  gruppi non vanno uniti.
- **Quindi:** 0 conflitti train/dev fra ammessi, 0 duplicazioni ammesse solo nel train fra gruppi diversi, 0 legami con
  record in quarantena o esclusi e 0 legami coi riservati dei sintetici.
- **Ripetizioni già coperte:** 317.293 classi di digest ripetuti.
  - **Dentro un gruppo, 315.034.** UniMER ne ha 68.188 per ciascuna identità di originale e shard, che comprendono le
    30.685 coppie formula e riga ritagliata (lo stesso file); in più 165 classi, per byte e per pixel, che uniscono
    originali e shard di record diversi dello stesso gruppo, e 81 di ritagli. MAVIS ne ha 10.294 per ciascuna identità.
    PixMo ne ha 116 per i byte e 191 per i pixel dell'originale, 194 e 192 per lo shard, e 2 di pixel che uniscono
    originali e shard di record dello stesso gruppo.
  - **Dentro una voce, 2.259.** Sono 1.124 PNG densi di UniMER uguali al file originale, per byte e per pixel (2.248
    classi), e 11 classi di pixel uguali fra ritagli della stessa riga ritagliata. Il contatore `stessa_voce` conta
    classi di digest, non voci o record: le 11 classi possono stare in meno di 11 righe ritagliate, e una classe può
    avere più di due ritagli (precisazione sotto, §9.4).

### 9.4 Proposte e limiti

- **Trattamento:** nessuno da applicare, perché non ci sono conflitti. Proposta: l'indice globale di C6 può ereditare i
  gruppi di queste quattro sorgenti senza fusioni dovute a identità esatte.
- **Ritagli con pixel uguali dentro una riga ritagliata (11 classi): da guardare, non conflitti.** Sono tre fatti
  distinti, e solo il primo è misurato:
  - *ripetizione dei pixel*: in una stessa riga ritagliata di UniMER due o più ritagli hanno gli stessi pixel. Il
    rapporto non dice in quante righe ritagliate stanno le 11 classi né quanti ritagli ha ciascuna;
  - *discordanza delle annotazioni*: le trascrizioni dei ritagli uguali (`rows` dello shard) non sono state confrontate.
    Se coincidono, la formula ripete una riga; se differiscono, almeno una trascrizione o una fascia non corrisponde ai
    pixel;
  - *conflitto nell'esempio che il loader costruisce*: il loader sceglie dalla chiave una sola riga k per record
    (`data._unimer_line`), la stessa per i due compiti. `transcribe_line` riceve il ritaglio della riga k;
    `transcribe_line_select` riceve l'immagine intera e la richiesta `Transcribe only line {k+1} of the formula in the
    image into LaTeX.`, che individua la riga anche quando due righe hanno gli stessi pixel.

  La sola identità dei ritagli non dimostra quindi un compito `transcribe_line_select` ambiguo né risposte sbagliate. Per
  sapere se c'è un problema servono l'elenco delle classi (circa 2 minuti, senza leggere immagini) e il confronto delle
  trascrizioni, che è un confronto testuale fuori da questo lotto. La correzione non richiede quarantene, scansioni o
  revisioni visive, e l'esito del confronto fra sorgenti resta valido.
- **Precisazione sul rapporto del job 30819** (2026-11-11). `outputs/exact_overlap/completo-1/report.json` e il
  riepilogo `docs/indice_globale/identita-esatte-completo-1.json` restano immutati. La loro descrizione della
  rappresentazione `ritaglio` ("l'immagine data al processor per transcribe_line e transcribe_line_select") è sbagliata
  per `transcribe_line_select`, che riceve l'immagine intera (la rappresentazione `shard`) col numero della riga. Allo
  stesso modo "11 righe con due ritagli identici" della nota del ledger va letto come 11 classi di pixel uguali dentro
  una riga ritagliata. Lo script corretto scrive nelle esecuzioni future la descrizione giusta e una nota sui contatori
  (`classi_nota`); conteggi ed esiti non cambiano. Il test del loader (`tests/test_f1c.py`) ora verifica immagine
  intera, numero della riga e riga scelta.
- **Limiti residui:**
  - solo uguaglianza esatta: la stessa immagine ricompressa, ridimensionata o ritagliata con margini diversi non si
    vede. Servono pHash o somiglianze, fuori da questo lotto;
  - l'ingresso del modello dopo il ridimensionamento del processor non è registrato: pixel uguali danno lo stesso
    ingresso solo con lo stesso numero massimo di patch;
  - i pixel non applicano l'orientamento EXIF;
  - non entrano `text.jpg` e `vision_dominant.jpg` di MAVIS, assenti dagli shard, né i 50 parquet di PixMo non scaricati;
  - LLaVA, MMFineReason, le altre sorgenti locali e le liste storiche restano fuori;
  - il confronto col pannello resta quello delle scansioni per sorgente: questo lotto non è la decontaminazione globale.
