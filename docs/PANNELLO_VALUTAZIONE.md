# Pannello di valutazione e decontaminazione di Limite-VL (D6)

**Stato al 2026-10-30, perimetro della pipeline v1.5.**
- I download decisi dall'utente sono eseguiti e verificati (job 30463).
- Indice, gruppi e ruoli sono fissati in `pannello-v2` (job 30466), prima di qualsiasi valutazione del modello.
- Nessun modello è stato valutato su questi ruoli. L'assegnazione resta da approvare nei dettagli (D6, §7).

Collegamenti: piano F0 (`docs/F0_IMPLEMENTATION_PLAN.md`, decisione D6, punti 3 e 7), campione candidato
(`docs/F0_CAMPIONE_CANDIDATO.md`), pipeline §§5.5, 5.8, 6.1, 15, S1 §§3-4.

**Evidenze di questo documento:**
- inventario dei benchmark locali: job 30457-30458 (`outputs/panel_inventory/run1`) e job 30464 (`run2`, con i
  benchmark nuovi);
- download: `data/logs/fetch-20261030-153329-p10-pannello.log` e verifica degli hash del job 30463
  (`outputs/verify/report-30463.tsv`);
- indice e gruppi: job 30466, in `data/decontam/pannello-v2` e `data/panel/pannello-v2` (`summary.json` con conteggi,
  copertura e sha256 dei file), codice `e084cba` (`scripts/panel_index.py`, `scripts/panel_groups.py`,
  `slurm/panel_groups.sbatch`);
- storiche: sovrapposizioni per pHash fra i benchmark già indicizzati (job 30459, `outputs/panel_overlaps/run1`),
  stati dei sintetici v4 già presenti in v3 (job 30460);
- fonti primarie: API e schede di Hugging Face, articoli e repository ufficiali, consultati il 2026-10-30.

`pannello-v1` (job 30465) è superato: non raggruppava le pagine dello stesso documento UCSF di DocVQA. Non è mai stato
usato, e i suoi file restano dove sono.

Nel documento:
- **scaricato** vuol dire che i file del manifest del download sono presenti, con hash verificato;
- **indicizzato** vuol dire presente in `data/decontam/pannello-v2`;
- **usato** o **consultato** vuol dire che è servito a una scelta;
- **legame confermato** vuol dire pixel identici dopo la normalizzazione, oppure provenienza o stesso problema
  dichiarati nei dati;
- **candidato** vuol dire somiglianza: pHash vicino o uguale, testo uguale, stesso nome di file. Un pHash uguale indica
  un candidato duplicato e non dimostra che le immagini siano identiche;
- **irrisolto** è un candidato fra un elemento di sviluppo e uno di test.

## 1. Inventario verificato

Per tutte le sorgenti qui sotto i file del manifest sono presenti e gli hash sono verificati, e nessuna immagine è
illeggibile (job 30464, 30466).

| Benchmark (repo@revisione) | Contenuto | Elementi e immagini indicizzati | Usi precedenti |
| --- | --- | --- | --- |
| MathVista `AI4Math/MathVista@2b6ad694` | testmini 1.000 con risposta (936 in inglese, 62 in cinese, 2 in persiano); test 5.141 con la risposta vuota; `metadata` con 31 sorgenti d'origine, contesto e compito | 6.141 | solo le dimensioni delle immagini (job 29649) |
| MathVerse `AI4Math/MathVerse@3bc86196` | 788 problemi in 5 versioni con immagine (3.940) e 788 Text Only | 4.728 (3.940 immagini) | 29649 |
| MathVision `MathLLMs/MathVision@2837ddb3` | test 3.040 con risposta; testmini 304, contenuto per intero nel test | 3.344 | 29649 |
| We-Math `We-Math/We-Math@527c44d4` | testmini 1.740 domande su 525 problemi (`ID`) | 1.740 | 29649 |
| DynaMath_Sample `DynaMath/DynaMath_Sample@bc7fcd1a` | 10 varianti già generate per 501 problemi | 5.010 | 29649 |
| Geoperception `euclid-multimodal/Geoperception@c34c5956` | 11.657 domande, 7 predicati, figure di Geometry3K | 11.657 | struttura letta (job 30026) |
| VisOnlyQA Real `ryokamoi/VisOnlyQA_Eval_Real_v1.1@3747d137` | 900 domande: grafici, chimica, geometria | 900 | 30026 |
| VisOnlyQA Synthetic `ryokamoi/VisOnlyQA_Eval_Synthetic@ef0b5fbc` | 700 domande: 3D, geometria sintetica | 700 | 30026 |
| OlympiadBench `Hothan/OlympiadBench@91184b52` | 18 sottoinsiemi; in inglese `OE_MM_maths_en_COMP` 150, `OE_MM_physics_en_COMP` 456 (271 problemi su 606 con 2-9 immagini), `OE_TO_maths_en_COMP` 674 solo testo | 8.476 elementi, 5.875 immagini | `OE_TO_maths_en_COMP` nella suite testuale |
| UniMER-Test (`wanderkid/UniMER_Dataset@2343ddd9`, `UniMER-Test.zip`) | spe 6.762, cpe 5.921, sce 4.742, hwe 6.332 (manoscritto) | 23.757 | i primi 200 di spe e di cpe (job 29448-29761) |
| OCRBench v1 `echo840/OCRBench@92a54bd1` | 1.000 domande, 10 tipi e 25 dataset d'origine: VQA su scene 200, su documenti 200, estrazione di campi 200, formule manoscritte (HME100K) 100, 6 tipi di riconoscimento da 50 | 1.000 | nessuno |
| DocVQA val `lmms-lab-encoder/DocVQA@539088ef` | 5.349 domande su 1.286 pagine (`docId`) di 938 documenti UCSF; `question_types` | 5.349 | nessuno |
| ChartQA `ahmed-masry/ChartQA@af8b6f5c` | val 1.920 su 1.056 grafici, test 2.500 su 1.509; metà `human` e metà `augmented` | 4.420 | nessuno |
| MUIRBench `MUIRBENCH/MUIRBENCH@4c393cff` | 2.600 domande su 2-9 immagini (11.264), 12 compiti, 10 relazioni; 1.300 coppie rispondibile/non rispondibile (`counterpart_idx`; la risposta della variante è "None of the choices provided") | 2.600 elementi, 11.264 immagini | nessuno |
| MV-MATH `PeijieWang/MV-MATH@b22144b7` | 2.009 problemi di matematica in inglese con 2-8 immagini; 11 materie; risposta a scelta 1.109, a un passo 800, a più passi 100 | 2.009 elementi, 6.061 immagini | nessuno |
| MMStar `Lin-Chen/MMStar@bc98d668` | 1.500 domande a scelta, 6 capacità da 250; origine in `meta_info`: SEEDBench 424, MMBench 366, MathVista 345, AI2D 196, MMMU 100, ScienceQA 69 | 1.500 | nessuno |
| Suite testuale (job 29348) | AIME 2024 30, AIME 2025 30, AIME 2026 30, HMMT feb 2025 30, HMMT feb 2026 33, BeyondAIME 100, APEX shortlist 47, MATH-500 500 | 800 | AIME 2026 usata per scegliere (job 29685-29697); AIME 2025 e HMMT 2026 nella baseline (job 29393) |
| GSM8K `openai/gsm8k@740312ad` | test 1.319 | 1.319 | nessuno |
| IFEval `google/IFEval@966cd895` | 541 istruzioni verificabili a regole | 541 | nessuno |

**Sintetici v4.** Test (300) e coppie (300) del manifest v4 hanno le chiavi di v3, e test e coppie v3 sono stati
consultati nelle valutazioni F1c (job 29741-29761). Con la stessa chiave lo stato è identico per 277 scene di test su
300 e per 217 coppie su 300 (job 30460). Restano nello split `test` del manifest, quindi fuori dal training, ma sono
sviluppo e diagnostica, non test finale.

**Sovrapposizioni fra benchmark** (job 30466, `elementi_legati_fra_insiemi` di `summary.json`):
- **MathVista.** Testmini e test non condividono ID. Le 382 coppie con lo stesso pHash del job 30459 erano candidati
  duplicati, non una prova. Con i legami confermati, 266 elementi del test stanno nel gruppo di un elemento del
  testmini.
- **MathVision.** Il testmini è un sottoinsieme del test: 304 ID su 304, con gli stessi pixel.
- **Geometry3K e affini.** Pixel identici collegano Geoperception a MathVista e MathVerse: 3.450 elementi di
  Geoperception stanno in un gruppo con il test di MathVista, 1.669 con MathVerse, 756 con il testmini di MathVista.
  Le geometrie di VisOnlyQA Real non hanno pixel identici altrove: per loro restano solo candidati.
- **Grafici e documenti dentro MathVista.** 111 elementi del test di ChartQA e 113 di DocVQA val stanno nel gruppo di
  un elemento del testmini di MathVista; 609 di ChartQA test e 250 di DocVQA val in quello di un elemento del test di
  MathVista.
- **MMStar.** I 345 elementi che vengono da MathVista sono tutti legati per nome del file d'immagine.
- **Testo.** 3 problemi di APEX hanno lo stesso testo di HMMT feb 2025 (2) e AIME 2025 (1). Il commento della suite
  in `limite_evals/suites.py` ne dichiara 4: il quarto non ha testo identico dopo la normalizzazione e va ancora
  individuato.

**File fuori da Git che riguardano il pannello.** Snapshot in `hf/hub/datasets--*`, link in `data/eval-datasets`,
indici e liste in `data/decontam/{p6,f1eval,text,pannello-v1,pannello-v2}`, gruppi in `data/panel/{pannello-v1,pannello-v2}`,
esiti storici in `outputs/eval`, `outputs/eval_f1_formulas`, `outputs/f1c`. Sono in `docs/ARTIFACT_INVENTORY.tsv`.

## 2. Copertura per capacità

| Capacità (v1.5) | Sviluppo | Test finale riservato | Lacuna |
| --- | --- | --- | --- |
| Matematica, STEM, geometria | MathVista testmini in inglese, MathVerse, We-Math, MathVision testmini, Geoperception | MathVision test, DynaMath_Sample, OlympiadBench multimodale in inglese, VisOnlyQA, MV-MATH | nessuna |
| Formule stampate | UniMER-Test, parte di sviluppo | UniMER-Test spe, cpe e sce, resto | nessuna. hwe (manoscritto) è solo diagnostica |
| OCR | metà dei gruppi di OCRBench | l'altra metà | nessuna |
| Documenti a pagina singola | metà dei documenti di DocVQA val | l'altra metà | nessuna |
| Grafici | ChartQA val | ChartQA test | nessuna |
| Tabelle | domande `table/list` di DocVQA, MathVista `table` (TabMWP) | domande `table/list` di DocVQA | ChartQA non misura le tabelle (§2.1) |
| Poche immagini (matematica e confronto) | metà dei gruppi di MUIRBench | l'altra metà; MV-MATH; OlympiadBench con più immagini | nessuna |
| Visione generale di supporto | metà dei gruppi di MMStar; MUIRBench | l'altra metà di MMStar | nessuna |
| Domande senza risposta | varianti di MUIRBench | varianti di MUIRBench | nessuna con un'immagine sola (§2.1) |
| Mantenimento del testo | AIME 2024-2026, HMMT feb 2025 e 2026, GSM8K, IFEval | MATH-500, BeyondAIME, APEX, OlympiadBench `OE_TO_maths_en_COMP` | i controlli di IFEval non sono nel valutatore (§4) |

**Fuori dal perimetro della prima versione** (pipeline v1.5): documenti a più pagine (MMLongBench-Doc e MP-DocVQA non
scaricati, righe commentate in `downloads/sources.tsv`), manoscritto come capacità (MathWriting rimandato; hwe e le
componenti manoscritte di OCRBench e DocVQA restano, ma non sono criteri di avanzamento) e video. Non sono stati scelti
CharXiv, InfographicVQA e OCRBench v2, perché la capacità è già coperta o il costo non torna.

### 2.1 Dove si misurano domande senza risposta, tabelle e visione generale

| Proprietà | Dove (sviluppo / test, domande) | Limiti |
| --- | --- | --- |
| Domande senza risposta | MUIRBench: 567 / 733 varianti non rispondibili, ciascuna nello stesso gruppo della sua rispondibile | solo con più immagini. Con un'immagine sola il pannello non ha domande senza risposta: lacuna dichiarata. VizWiz (foto con "unanswerable") resta un candidato futuro, non scaricato |
| Tabelle | DocVQA `table/list`: 934 / 846; MathVista testmini con contesto `table`: 70 in sviluppo (62 da TabMWP); estrazione di campi da moduli e scontrini in OCRBench: 122 / 78 | ChartQA ha solo grafici. Il test delle tabelle è DocVQA: tabelle dentro pagine di documenti, senza tabelle di calcolo pure. `question_types` è una lista: una domanda può contare in più tipi |
| Visione generale | MMStar: 855 / 645, con le 6 capacità in entrambi i ruoli; MUIRBench su foto, mappe, scene, immagini mediche; OCRBench, VQA su scene: 96 / 104 | MMStar è sbilanciato dai legami con MathVista (§3.2) |

## 3. Ruoli (D6)

### 3.1 Regole

- **Sviluppo:** tutto ciò che serve a scegliere encoder, proiettore, risoluzione, mix, durate e soglie. Si consulta
  più volte, e i suoi risultati non sono una prova finale.
- **Test finale riservato:** non si consulta prima della valutazione finale. Si usa una volta, con un protocollo fissato
  prima, e dopo non si prendono decisioni.
- **Diagnostica:** analisi mirate (versioni, varianti, immagine tolta, tipi di domanda, non rispondibili). Non è una
  conferma indipendente; sui dati del test si fa solo dopo la valutazione finale.
- **Solo decontaminazione:** elementi senza risposta, fuori lingua o non verificabili (MathVista test e non inglese,
  sottoinsiemi di OlympiadBench in cinese o con dimostrazioni). Servono a escludere dati dal training.
- **Gruppi:** elementi collegati da un legame confermato formano un gruppo, anche fra benchmark diversi. Un gruppo
  con un membro di sviluppo, o già consultato, va tutto nello sviluppo; altrimenti, con un membro di test, tutto nel
  test. Nessun gruppo sta in due ruoli.
- **Estrazione:** per OCRBench, DocVQA, MUIRBench e MMStar ogni gruppo va in sviluppo se il suo rango (`split_rank`
  col seme 0 sull'ID del gruppo) è sotto 0,5; per UniMER-Test sotto 0,05. L'ID del gruppo dipende solo dai membri,
  quindi lo stesso gruppo ha la stessa estrazione in ogni run con lo stesso seme.
- **Già consultato:** un dato già consultato non diventa test cambiando etichetta. Oggi lo sono:
  - AIME 2026, AIME 2025 e HMMT 2026;
  - i primi 200 di spe e i primi 200 di cpe di UniMER-Test;
  - test e coppie dei sintetici v3, e quindi gli stati condivisi da v4.
- **Irrisolti:** un elemento di test con un candidato verso lo sviluppo resta nel test, ma si conta a parte. La
  metrica primaria del test usa gli elementi senza irrisolti. La verifica visiva conferma il legame, e allora
  l'elemento esce dal test con tutto il gruppo, oppure lo esclude. Contano come irrisolti:
  - pHash entro 4 bit;
  - lo stesso nome di file di ChartQA in split diversi;
  - testi uguali di almeno 15 parole fra benchmark diversi.

  Un candidato è **forte** se pHash e testo collegano gli stessi due gruppi. Nessuna fusione passa per una catena di
  candidati.
- **Protezione dal training:** ogni elemento del pannello, in qualunque ruolo, sviluppo compreso, resta fuori dal
  training. Un record di training legato a uno di essi è escluso; con un candidato irrisolto va in quarantena (S1 §3).
- **Esposizione pregressa:** la decontaminazione del nuovo training non toglie quello che SigLIP2 e Violetto possono
  aver visto prima.
- **Versioni:** una modifica produce `pannello-v3`, mai una correzione dei file esistenti. Dopo la prima valutazione
  del modello sul test l'assegnazione non cambia più.

### 3.2 Assegnazione (`pannello-v2`, seme 0)

Domande per ruolo; fra parentesi i gruppi. "Senza irrisolti" è la base della metrica primaria del test.

| Capacità | Sviluppo | Test finale riservato | Test senza irrisolti | Diagnostica o solo decontaminazione |
| --- | --- | --- | --- | --- |
| Matematica e geometria | MathVista testmini in inglese 936 (907); MathVerse 4.728 (757); We-Math 1.740 (518); MathVision testmini 304 più i 306 del test con le stesse immagini (304); DynaMath 10 (1 problema, legato a MathVista) | MathVision test 2.734 (2.724); DynaMath 5.000 (495 problemi); OlympiadBench 606 multimodali, di cui 271 con più immagini (in tutto 1.280 col solo testo); MV-MATH 2.009 (2.008) | MathVision 2.731; DynaMath 4.968; OlympiadBench 1.280; MV-MATH 2.003 | MathVista test e non inglese 5.205; OlympiadBench, altri sottoinsiemi 7.196 |
| Percezione | Geoperception 11.657 (902); VisOnlyQA Real 4 (grafici legati al testmini di MathVista) | VisOnlyQA Real 896 (546), Synthetic 700 (609) | Real 633; Synthetic 700 | — |
| Formule | UniMER-Test 1.261 (spe 542, cpe 476, sce 243: i 400 consultati, il 5% estratto e i loro gruppi) | UniMER-Test 16.164 (spe 6.220, cpe 5.445, sce 4.499) | 16.164 | hwe 6.332, diagnostica |
| OCR | OCRBench 525 (485) | OCRBench 475 (445) | 447 | componenti manoscritte (150), riportate a parte |
| Documenti a pagina singola | DocVQA 2.681 domande, 644 pagine, 468 documenti UCSF | DocVQA 2.668 domande, 642 pagine, 470 documenti | 2.583 | `question_types` |
| Grafici | ChartQA val 1.920 più 113 del test, legati al testmini di MathVista (111) o nello stesso gruppo di elementi di val (2); 1.105 gruppi | ChartQA test 2.387 (1.458) | 1.761 | `human` e `augmented` |
| Poche immagini | MUIRBench 1.134 (462; 567 coppie) | MUIRBench 1.466 (458; 733 coppie) | 1.335 | non rispondibili; `Ordered_Pages` e `Video` fuori perimetro |
| Visione generale | MMStar 855 (809) | MMStar 645 (617) | 520 | MMStar per capacità |
| Testo | AIME 2024, 2025, 2026 (90); HMMT feb 2025 e 2026 (63); GSM8K 1.319; IFEval 541; APEX 3 (testo identico a HMMT e AIME) | MATH-500 500; BeyondAIME 100; APEX 44; OlympiadBench `OE_TO_maths_en_COMP` 674 (compreso sopra) | tutti, ma APEX ha ancora un problema in comune da individuare (§1) | — |
| Sintetici | dev del manifest v4 (2.870); test (300) e coppie (600) v4 | nessuno per ora (§3.5) | — | coppie controfattuali v4 |

Irrisolti nel test: 1.299 in tutto, di cui 191 forti.

| Benchmark | Irrisolti (forti) | Controparti principali in sviluppo |
| --- | --- | --- |
| ChartQA | 626 (90) | ChartQA val 529; OCRBench in sviluppo 254; il testmini di MathVista 94 |
| VisOnlyQA Real | 263 (7) | MathVista testmini, MathVerse, Geoperception, MMStar (geometrie); ChartQA val (grafici) |
| MUIRBench | 131 (41) | MUIRBench in sviluppo 101; MMStar 30 |
| MMStar | 125 | MathVista testmini 95; MUIRBench 25 |
| DocVQA | 85 (44) | OCRBench in sviluppo 81 (la componente `docVQA`) |
| DynaMath, OCRBench, MV-MATH, MathVision | 32, 28 (9), 6, 3 | vari |

Un elemento può avere controparti in più insiemi. Molti candidati sono grafici o pagine di documento con la stessa
impaginazione, che il pHash a 64 bit non distingue: solo la verifica visiva dice quali sono duplicati.

Note:
- MathVista test e i sottoinsiemi non inglesi servono solo alla decontaminazione.
- Il manifest v4 non cambia: test e coppie restano nello split `test`, fuori dal training, e il pannello li usa per
  sviluppo e diagnostica.
- `data/panel/pannello-v2/elements.jsonl.gz` è il manifest del pannello: un elemento per riga con gruppo, ruolo fisso
  e finale, motivo e irrisolto. Mancano le metriche per benchmark (D4) e le varianti del contratto (C3).
- Gli sbilanciamenti vengono dai gruppi indivisibili. Si possono correggere solo con una regola nuova, decisa prima di
  qualsiasi valutazione (§7):
  - **MUIRBench:** due gruppi di 232 e 210 domande, legati da immagini riusate, finiscono interi da una parte. Così il
    test ha tutte le 234 domande su immagini mediche e 268 delle 348 `Ordered_Pages`.
  - **MMStar:** il ragionamento logico ha 192 domande in sviluppo e 58 nel test, perché le domande da MathVista vanno
    nello sviluppo.

### 3.3 DocVQA val per documento

- **Unità:** il documento UCSF (`ucsf_document_id`). Tutte le domande sulle pagine dello stesso documento stanno nello
  stesso gruppo, insieme agli elementi con pixel identici di altri benchmark. In `pannello-v1` il gruppo era la
  pagina (`docId`), e 347 unioni mancavano.
- **Legami con altri benchmark:**
  - 31 gruppi con MathVista e 2 con MathVista e MMStar;
  - i documenti con un elemento nel testmini di MathVista vanno nello sviluppo.
- **Candidati con OCRBench.** La componente `docVQA` di OCRBench (50 domande) non ha pixel identici in DocVQA val.
  Nel test di DocVQA ci sono 85 domande irrisolte, 44 forti; 81 hanno un candidato nella metà di sviluppo di
  OCRBench. Se la verifica conferma un legame, il documento esce dal test.
- **Esito:** sviluppo 2.681 domande, 644 pagine, 468 documenti; test 2.668, 642, 470. Nessun documento sta in due ruoli
  (`docvqa_documenti_in_piu_ruoli` = 0).
- **Training futuro:** il train di DocVQA (non scaricato) va confrontato con val per documento UCSF, oltre che per pixel.

### 3.4 MUIRBench: varianti e coppie

- Tutti i 2.600 `counterpart_idx` corrispondono a un elemento, e formano 1.300 coppie rispondibile/non rispondibile.
- Ogni coppia sta in un solo gruppo: 1.262 coppie avevano già immagini identiche, per le altre 38 il legame viene dal
  riferimento.
- I gruppi seguono anche le immagini condivise fra domande diverse. Per questo sono 920, e due superano le 200
  domande.
- **Fuori perimetro:**
  - relazione `Ordered_Pages` (presentazioni, ragionamento fra pagine): 80 in sviluppo, 268 nel test;
  - tipo `Video` (fotogrammi): 52 e 44.

  Si riportano a parte e non sono criteri di avanzamento (D4).

### 3.5 Componente nuova riservata: regole, non costruita

Non si costruiscono ora generatori nuovi né il test sintetico finale. La pipeline chiede di conservare una componente
riservata con verità indipendente dove possibile. Generarla alla fine non basta a garantire contenuti mai visti:
- il generatore è lo stesso codice che produce il training. Una chiave nuova può dare uno stato già presente: le scene
  `function` hanno 7.074 stati per 15.000 scene, e 2.792 scene di train hanno lo stato di un record di test o delle
  coppie (S2 §2.6);
- il renderer è lo stesso. Un contenuto nuovo non è una distribuzione visiva nuova;
- l'esposizione pregressa di SigLIP2 e Violetto a figure simili non si esclude.

Regole per quando si costruirà, con l'ok dell'utente:
1. **La novità si misura.** Lo stato di ogni scena (`stato_scena_sha256`: la specifica senza chiave, versione e stile)
   si confronta con tutti gli stati di tutti i manifest: v3 e v4, train, dev, test e coppie, esclusi e quarantena
   compresi. È contenuto nuovo solo uno stato assente.
2. **Componenti delle combinazioni.** In un esempio con più immagini ogni componente si controlla da solo. L'esempio
   è contenuto nuovo solo se lo sono tutti i suoi componenti. Se i componenti sono già usati e nuova è solo la
   combinazione, è una **nuova combinazione di contenuti già usati**: misura la composizione, non la lettura di
   contenuti mai visti, e si riporta a parte.
3. **Domande.** Una domanda nuova su uno stato già usato è una combinazione (stato, domanda) nuova, non un contenuto
   nuovo.
4. **Spazi di stati piccoli.** Le famiglie con pochi stati, come `function` a un solo grafico, non bastano senza
   cambiare il generatore. Si dichiara, senza forzare.
5. **Tempi.** Chiavi, semi e numerosità si fissano nel manifest del pannello prima di qualsiasi addestramento. Si
   genera solo per la valutazione finale, e il risultato si consulta una volta.

Test e coppie v4 già consultati restano sviluppo o diagnostica, protetti dal training.

## 4. Download eseguiti

Gruppo `p10-pannello` di `downloads/sources.tsv`, autorizzato dall'utente il 2026-10-30. Il manifest è
`downloads/manifests/p10-pannello.tsv`, generato con `build_manifests.sh`. Il trasferimento è passato da
`fetch_manifest.sh`, con i limiti del kit: 30 file, 1.680 MiB, nessuno già presente, nessun errore. Gli hash sono
verificati dal job 30463, 30 file su 30.

| Benchmark | Repository | Revisione | File | Byte |
| --- | --- | --- | --- | --- |
| OCRBench v1 | `echo840/OCRBench` | `92a54bd1384387c178d5a07140a2d85e0a3d12e1` | `data/test-00000-of-00001.parquet` | 67.576.988 |
| DocVQA val | `lmms-lab-encoder/DocVQA` | `539088ef8a8ada01ac8e2e6d4e372586748a265e` | `DocVQA/validation-0000[0-5]-of-00006.parquet` | 1.055.835.028 |
| ChartQA | `ahmed-masry/ChartQA` | `af8b6f5c08c95085271561c2a3f9d15f2b5a9031` | `data/val-00000-of-00001.parquet`, `data/test-00000-of-00001.parquet` | 50.238.778 + 68.916.402 |
| MUIRBench | `MUIRBENCH/MUIRBENCH` | `4c393cffc985c77d28de3b9045e2e5186920df80` | `data/test-0000[0-4]-of-00005.parquet` | 429.440.985 |
| MV-MATH | `PeijieWang/MV-MATH` | `b22144b7d9315ce30c7ee8ce1bbaaa7b52fa8674` | `MV-MATH.json`, `images.zip` | 3.977.469 + 43.215.035 |
| MMStar | `Lin-Chen/MMStar` | `bc98d668301da7b14f648724866e57302778ab27` | `mmstar.parquet` | 41.798.712 |
| GSM8K | `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | `main/test-00000-of-00001.parquet` | 419.088 |
| IFEval | `google/IFEval` | `966cd89545d6b6acfd7638bc708b98261ca58e84` | `ifeval_input_data.jsonl` | 207.111 |

Si aggiungono i `README.md` e i `.gitattributes`, di pochi KB.

Nessuno di questi repository era già nella cache, quindi nessun file è stato scaricato due volte. Il gruppo `p9-pannello`
(MMLongBench-Doc e MP-DocVQA) non è stato scaricato.

**Licenze dichiarate:**
- ChartQA GPL-3.0, MUIRBench CC-BY-4.0, MV-MATH MIT, GSM8K MIT, IFEval Apache-2.0;
- la copia di DocVQA porta `apache-2.0`, ma i documenti vengono dalla UCSF Industry Documents Library;
- OCRBench e MMStar non dichiarano una licenza nella scheda.

Il progetto è solo di ricerca: le fonti si citano e i dati non si ridistribuiscono.

**Codice da aggiungere prima di usarli:**
- IFEval: i controlli delle istruzioni del repository ufficiale (google-research `instruction_following_eval`) non sono
  nel valutatore vendorizzato. Vanno clonati in `third_party/` a un commit fissato, oppure riscritti.
- GSM8K: è nella suite del valutatore senza revisione; va fissata e collegata in `data/eval-datasets`.
- Valutatore del modello VL per capacità, con le metriche ufficiali dei benchmark: punto 7 del piano F0.

## 5. Indice e gruppi: stato e copertura

**Indice** `data/decontam/pannello-v2/index.json` (job 30466, sha256 `e59242f5…`). Ha una voce per immagine e una per
ogni elemento di solo testo: 99.713 voci, cioè 92.658 immagini e 7.055 testi.
- Ogni voce ha pHash, sha256 dei pixel normalizzati (`processing.to_rgb`), dimensioni, testo e i campi per gruppi e
  copertura.
- Gli elementi con più immagini (MUIRBench, MV-MATH, OlympiadBench) hanno una voce per immagine, con la posizione.
- È compatibile con `scripts/decontam.py scan`.
- Copre tutti i 26 repository riservati del catalogo. Nessuna immagine è illeggibile, e nessun percorso di MV-MATH manca.
- Gli indici `p6` e `f1eval` non hanno lo sha256 dei pixel. Restano come evidenze storiche delle scansioni fatte.

**Gruppi** `data/panel/pannello-v2`: 85.991 elementi in 51.363 gruppi.

| File | Contenuto | sha256 |
| --- | --- | --- |
| `elements.jsonl.gz` | elemento, gruppo, ruolo, motivo, irrisolto | `31dd636e…` |
| `links.jsonl.gz` | legami confermati che hanno unito due gruppi | `2b6a7599…` |
| `candidates.jsonl.gz` | candidati, con tipo e distanza | `94ccda76…` |
| `unresolved.jsonl.gz` | irrisolti, con le controparti e la forza | `6aac8532…` |
| `summary.json` | conteggi, copertura per ruolo, regole, seme | — |

Unioni per tipo di legame:

| Legame | Unioni nuove | Conferme di gruppi già uniti |
| --- | --- | --- |
| pixel identici | 26.587 | 4.607 |
| MathVerse `problem_index` | 1.990 | 1.950 |
| We-Math `ID` | 492 | 723 |
| DynaMath, varianti | 3.872 | 637 |
| MathVision testmini-test | 0 | 304 |
| DocVQA `docId` | 0 | 4.063 |
| DocVQA documento UCSF | 347 | 4.064 |
| ChartQA `imgname` | 0 | 1.855 |
| UniMER-Test, stessa formula | 931 | 4 |
| solo testo, stesso testo | 26 | 0 |
| MUIRBench, controparti | 38 | 2.562 |
| MMStar → MathVista | 345 | 0 |

**Limiti.**
- **UniMER-Test.** Le formule sono quasi bianche, e i loro vicini pHash non hanno senso (job 30039). Per loro contano
  solo pixel e formula identici.
- **Cluster densi.** Le immagini con più di 200 vicini registrano solo i più vicini: ChartQA 96, VisOnlyQA 15,
  MathVista 9, OCRBench 4.
- **Domande di rito.** I testi in più di 25 gruppi non contano come candidati: 16 testi in Geoperception, 6 in
  MathVista, 1 in OCRBench.
- **Sorgenti di training.** Nessuna è ancora confrontata con questo indice (§6).

## 6. Piano minimo di decontaminazione

### 6.1 Per benchmark

| Benchmark | Materiale | Strumenti | Adattamenti indispensabili | Copertura |
| --- | --- | --- | --- | --- |
| Tutti i benchmark visivi del pannello | `pannello-v2` | `decontam.py scan` sull'indice nuovo, `decontam_review.py` (correlazione dei pixel a 64×64) | portare lo sha256 dei pixel nella scansione delle sorgenti di training | pixel, pHash, testo, provenienza dichiarata |
| UniMER-Test | immagini e formule | `scripts/unimer_test_overlap.py`, `latex.despace` e `visible_tokens` | indice delle formule normalizzate, da confrontare con tutte le sorgenti con formule (MMFineReason, HME100K di OCRBench) | formule, pixel |
| Suite testuale, GSM8K, IFEval | testi | `replay_decontam.py` (copertura degli n-gram) | estendere il confronto a MMFineReason e alle domande delle altre sorgenti; individuare il quarto problema in comune di APEX | testo |

**Somiglianze non confermate.** pHash e n-gram danno solo candidati (S1 §4).
- **Conferma:** pixel identici dopo la normalizzazione, provenienza dichiarata, oppure lo stesso problema verificato dal
  codice. `decontam_review.py`, con la somiglianza dei pixel ≥ 0.90 e il rapporto d'aspetto entro 1.2, decide quali
  candidati guardare.
- **Candidato irrisolto fra training e pannello:** resta in quarantena, fuori dal training.
- **Candidato irrisolto fra sviluppo e test:** si conta a parte finché non si risolve (§3.1).

### 6.2 Subito per v4 e dopo con le altre sorgenti

- **Possibile subito, senza download (la scansione va approvata):**
  - immagini v4 contro `pannello-v2`, con pixel e pHash;
  - le domande, estendendo `texts_of` di `decontam.py` al tipo `synth` (oggi restituisce una lista vuota).

  v4 non ha legami di provenienza: è generata.
- **Quando entrano le altre sorgenti locali:**
  - MAVIS (stessi autori di MathVerse) contro le geometrie del pannello;
  - UniMER contro UniMER-Test (già fatto, `outputs/unimer_overlap/run2`) e HME100K di OCRBench;
  - MMFineReason contro tutto il pannello. Fra le sue sorgenti dichiarate (`data/lang/mmfinereason/stats.json`) ci sono
    `FineVision-geometry3k`, che tocca le figure di Geometry3K in MathVista, MathVerse e Geoperception, e WeMath2, degli
    autori di We-Math. Ci sono anche AI2D e ScienceQA, da cui vengono 196 e 69 domande di MMStar. MMR1 (1,27 milioni di
    righe) è un aggregato con una composizione interna non inventariata;
  - PixMo e LLaVA contro MMStar (immagini di SEEDBench e MMBench) e MUIRBench.
- **Con le aggiunte proposte** (`docs/F0_CAMPIONE_CANDIDATO.md`, non scaricate):
  - il train di ChartQA contro val, test, MathVista e le componenti ChartQA di OCRBench;
  - il train di DocVQA contro val per documento UCSF, contro MathVista e contro la componente `docVQA` di OCRBench;
  - TabMWP contro MathVista, che lo usa come sorgente;
  - M4-Instruct contro MUIRBench;
  - il replay testuale contro GSM8K e IFEval.

## 7. Decisioni dell'utente

1. Approvare ruoli e assegnazione di `pannello-v2` (D6), oppure indicare le modifiche. Per esempio una quota diversa
   da 0,5, o un'estrazione stratificata per compito in MUIRBench e per capacità in MMStar. Una regola nuova produce
   `pannello-v3` e va decisa prima di qualsiasi valutazione.
2. Irrisolti: approvare la regola di §3.1 (contati a parte finché la verifica visiva non li risolve) e la verifica
   stessa, senza agenti. Si comincia dai 191 forti.
3. Metriche (D4):
   - MUIRBench con e senza `Ordered_Pages` e `Video`;
   - OCRBench con e senza le componenti manoscritte;
   - le metriche ufficiali: accuratezza rilassata di ChartQA, ANLS di DocVQA.
4. Domande senza risposta con un'immagine sola: accettare la lacuna, oppure aggiungere un candidato (VizWiz) con un
   download separato.
5. Componente nuova riservata: costruirla più avanti, con le regole di §3.5.
6. Controlli di IFEval in `third_party/` (clone a un commit fissato) prima di usare IFEval.
