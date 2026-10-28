# S1 — Record, manifest e derivati

**Stato:** bozza 2 del 2026-10-28, rivista dopo la revisione esterna dei push fino a `e81f884`; da approvare. Attua la
pipeline v1.4 §§5.2, 5.5, 5.8, 5.10.1, 5.10.3 (rilievi G1, G5, R5, R7). Nessun codice è ancora scritto.

## 1. Identificatori

Tutti gli identificatori che fanno da chiave sono **digest sha256 completi** (64 caratteri esadecimali) di un JSON
canonico: chiavi ordinate, nessuno spazio, UTF-8, numeri interi o stringhe. Un prefisso breve (`uid8`, `g8`) serve solo
per la lettura umana di log e rapporti. Nessun codice lo usa come chiave, né per unire, filtrare o assegnare split ed
esclusioni.

| Identificatore | Digest di | Uso |
| --- | --- | --- |
| `record_id` | `{sorgente, revisione (40 car.), record}`; la revisione dei sintetici è la versione degli shard più l'identità del codice del generatore | campione originale |
| `variant_id` | costruzione canonica: `{compito, prompt_id, domande, ritaglio, max_patches, image_mode, contratto, modalità, system_prompt_id, prefill}` | variante dell'esempio |
| `uid` | `{record_id, variant_id}` | esempio ammesso: chiave del manifest |
| `group_id` | `{versione dell'indice, membri: lista ordinata dei record_id}` | gruppo di separazione (§3) |

Oggi compito e variante sono scelti dal loader con hash della chiave (`data.mavis_f1_sample`: `key + "/task"`;
`f1c.pick_questions`: `key + "/pick"`; `data._unimer_line`: `key + "/line"`). La regola resta deterministica, ma il
loader deve restituire anche la costruzione canonica, da cui si calcola `variant_id`.

**Controllo delle collisioni.** Ogni costruzione (manifest, indice, import delle liste) tiene una mappa
`digest → JSON canonico`. Se lo stesso digest compare con un contenuto diverso, la costruzione si ferma con errore.
Nessuna fusione e nessuna scelta automatica. Con sha256 una collisione reale è trascurabile. Il controllo protegge da
identificatori calcolati su un input diverso da quello registrato o accorciati per errore. I difetti della forma
canonica (due contenuti diversi con la stessa serializzazione) si prevengono con i suoi collaudi (§8). Due problemi
distinti non condividono mai un identificatore, uno split o un'esclusione
per una collisione: la condivisione nasce solo da un legame confermato (§4).

## 2. Record del manifest

Una riga JSON per `uid`:

```
uid, record_id, variant_id, sorgente, revisione, record, compito,
costruzione {prompt_id, domande, ritaglio, max_patches, image_mode},
contratto {nome, versione}, modalità, system_prompt_id, prefill {testo, con_label} | null,
gruppo (group_id), split, stato, motivo, gold (S2: gold_verificato | target_controllato | incerto),
evidenze [{controllo, versione, esito, rif}],
input_tokens_sha256, target_tokens_sha256, risposta_sha256, immagini [{sha256, ordine, griglia}],
trasformazione {nome, versione}, genitori [uid]
```

- **Contratto, modalità e prefill** (S3 §2) fanno parte della variante: lo stesso campione con la modalità `auto` o
  `math`, o con e senza prefill, dà due `uid` diversi. L'estrattore rifiuta un testo prodotto con un contratto diverso
  da quello del record.
- `input_tokens_sha256`: digest degli id dei token non supervisionati (system prompt, turno utente con i segnaposti
  delle immagini, prefill), nell'ordine.
- `target_tokens_sha256`: digest degli id dei token con label, compresi `<think>`, `</think>` e `<|endoftext|>` secondo
  il contratto.
- `risposta_sha256`: digest della sola risposta, senza gli involucri del contratto; serve a confrontare il gold fra
  contratti diversi.
- L'identità del tokenizer (sha256 di `tokenizer.json`) e quella del template stanno nel `manifest.json`.
- `stato`: `ammesso`, `quarantena`, `escluso`; `split`: `train`, `dev`, `test`. I riservati non hanno record di
  training. Lo stato del tipo di domanda (S2 §1) prevale su quello del record: un tipo `escluso` o in `quarantena` non
  ha record `ammessi`.
- `evidenze` cita gli esiti per riferimento (file e sha256), non copia le liste.

## 3. Gruppi e split

- Il gruppo è l'unità di separazione: stesso problema, immagine, documento o istanza del generatore, con tutte le QA,
  le descrizioni, i ritagli, le traduzioni e le varianti controfattuali.
- Lo split si assegna al gruppo, mai al record. Un gruppo legato a un riferimento riservato è escluso dal training in
  tutte le fasi.
- Sintetici: un gruppo per scena; la coppia controfattuale `-a`/`-b` è un solo gruppo. Il test di generalizzazione su
  template o stili mai visti è uno split dichiarato a parte.
- La validazione si sceglie dal manifest (`split = dev`). Il prefisso dell'ultimo shard (`stream.py:158-166`) non si
  usa più per i run nuovi.
- `group_id` dipende dai membri: se cambia la composizione cambia l'identificatore. Ogni versione dell'indice elenca le
  fusioni, ciascuna col legame che l'ha prodotta.

## 4. Indice globale (prima del manifest)

Cartella `data/index/<versione>/`, fuori da git, costruita in un job CPU sull'inventario fissato, comprese le sorgenti
riservate (`docs/ARTIFACT_INVENTORY.tsv`). Segnali per record:

1. ID e provenienza dichiarata (MAVIS e MMFineReason contengono Geometry3K e WeMath);
2. sha256 dei file e dei pixel normalizzati;
3. pHash (`decontam.phash`, 64 bit) e n-gram di testo e formule (`decontam.normalize_words`, `word_ngrams`).

Regole:

- Solo un legame **confermato** fonde due gruppi. Sono confermati: sha256 uguale dei pixel normalizzati, provenienza
  dichiarata, oppure un candidato verificato dal codice con operatori, numeri e condizioni conservati.
- pHash e n-gram producono solo candidati, perché sono segnali corti (un pHash ha 64 bit) e non identificano nulla. Un
  candidato ambiguo resta in quarantena: non diventa un legame.
- Nessuna fusione transitiva tramite candidati: niente catene di somiglianze deboli.

## 5. Esclusioni e rietichettature storiche

Si importano in `data/index/<versione>/historical.jsonl` con: file d'origine e sha256, chiave, motivo, regola e data,
stato proposto. Le liste restano dove sono.

Ogni riga storica è una chiave nuda (`train-0012637`, `synth-0003949`). Si risolve contro la sorgente e la revisione
della lista d'origine e deve corrispondere a **esattamente un** `record_id`. Una chiave che corrisponde a zero record o
a più di uno è un errore dell'import, mai un'esclusione applicata per somiglianza di nome. Un'esclusione si applica al
record. Si estende al gruppo solo se il motivo riguarda il gruppo, come la contaminazione con un riservato.

| Lista | Stato proposto | Ragione |
| --- | --- | --- |
| `data/decontam/p6/*/exclude_final.txt`, `data/decontam/f1eval/*/exclude_final.txt` | escluse dal training finché non sono riesaminate | sovrapposizioni con benchmark riservati: si conservano, ma la regola usa la somiglianza |
| `data/decontam/text/replay/exclude_final.txt` | idem | regola severa sul testo (copertura ≥ 0.2) |
| `outputs/unimer_overlap/run2/train_contamination_long.txt` | confermata se la causa è la stessa immagine; da riesaminare se è solo la formula | una formula comune non identifica un problema |
| `outputs/split_duplicates/run1/*/val_duplicates.txt` | da riesaminare | regola per sorgente, anche su target identico |
| `outputs/unimer_check/run4/quality_excluded.txt` | confermata (errore trovato dal codice) | — |
| `outputs/unimer_check/run4/uncertain.txt` | quarantena | incerti non diventano scarti o validi |
| `outputs/unimer_check/run4/relabel.jsonl` | trasformazione tracciata (`genitori`, versione) | — |
| `data/decontam/*/exclude.txt` grezze | solo candidati | pHash confonde le immagini quasi vuote |

## 6. Uso nei reader e contabilità

- `Source` riceve il manifest. `SourceReader` legge solo i record con almeno un `uid` ammesso per quella sorgente, e il
  loader costruisce solo i compiti e le varianti ammessi. Se ricalcolando `variant_id`, `input_tokens_sha256` o
  `target_tokens_sha256` il valore non coincide, è un errore contato, mai un passaggio silenzioso.
- Lo stesso filtro vale per SFT, replay e RL.
- Contatori nel log e nel manifest del run, per sorgente e compito:
  - esempi costruiti, `uid` unici, gruppi unici, ripetizioni;
  - token di input, token visivi, target-token;
  - scarti dopo l'ammissione, per errore e per `too_long`.

  Oggi esistono solo `errors/<sorgente>` (`stream.py:247`) e `too_long` in totale (`collator.py:71`).
- Se gli scarti dopo l'ammissione cambiano in modo rilevante popolazione o pesi, si aggiornano copertura e
  denominatori.

## 7. File e immutabilità

`data/manifests/<nome>-<versione>/`: `records.jsonl.gz` e `manifest.json`. Il `manifest.json` contiene:

- versione e indice usato;
- manifest degli shard con sha256;
- liste importate con sha256;
- tokenizer, template e contratti con i loro sha256;
- identità del codice (`provenance.code_identity`);
- conteggi per sorgente, compito, split e stato;
- sha256 di `records.jsonl.gz`.

Una versione si scrive una volta sola (`provenance.write_new_json`); ogni correzione produce una versione nuova.

## 8. Collaudi (CPU, `tests_cpu`)

- **Collisione forzata.** Con la funzione di digest sostituita nel test da una troncata a 8 caratteri, due record
  distinti producono lo stesso digest: la costruzione si ferma con errore e non fonde nulla.
- **Canonicalizzazione.** Lo stesso contenuto con chiavi in ordine diverso dà lo stesso digest; contenuti diversi danno
  digest diversi; la forma canonica si rilegge nello stesso contenuto (andata e ritorno), anche con stringhe Unicode e
  valori vuoti.
- **Prefissi.** Nessuna funzione pubblica accetta un prefisso breve come chiave.
- **Chiavi storiche.** Una chiave nuda che non corrisponde a nessun record della sorgente d'origine, o a più di uno
  (lista senza sorgente univoca, chiave ripetuta fra shard), dà errore all'import.
- **Gruppi e split.**
  - La stessa immagine in due sorgenti dà un solo gruppo e un solo split.
  - Un pHash simile con una cifra decisiva diversa resta un candidato, non una fusione.
  - Lo split si eredita dai derivati (ritaglio, QA, controfattuale).
- **Contratto.** Lo stesso campione in modalità `auto` e `math`, o con e senza prefill, dà `uid` diversi; un testo
  generato con un altro contratto viene rifiutato dall'estrattore.
- **Reader.**
  - Una QA non ammessa su un'immagine ammessa non viene costruita.
  - Un target cambiato dal codice dopo l'ammissione è un errore contato.
  - La validazione viene dal manifest.
  - I contatori sono coerenti su un flusso di prova.
