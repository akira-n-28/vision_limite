# S1 — Record, manifest e derivati

**Stato:** bozza del 2026-10-28, da approvare. Attua la pipeline v1.4 §§5.2, 5.5, 5.8, 5.10.1, 5.10.3 (rilievi G1, G5,
R5, R7). Nessun codice è ancora scritto.

## 1. Identità di un esempio

Un esempio ammesso è una coppia input–compito, non un'immagine:

`uid = <sorgente>@<rev8>/<record>/<compito>/<variante>`

| Campo | Significato | Esempi |
| --- | --- | --- |
| `sorgente` | nome breve del catalogo (`downloads/catalog_status.tsv`) | `mavis-geometry`, `unimer`, `synth` |
| `rev8` | primi 8 caratteri della revisione HF; per i sintetici versione degli shard | `9c80fff3`, `2343ddd9`, `v3` |
| `record` | chiave del campione nello shard, invariata | `d3-3228`, `train-0012637`, `synth-0001234` |
| `compito` | elenco chiuso per famiglia (S2) | `describe`, `qa`, `transcribe`, `transcribe_line` |
| `variante` | `v` + 8 esadecimali dello sha256 del JSON canonico della costruzione: id del prompt, id delle domande scelte, ritaglio, budget di patch, `image_mode` | `v3fa91c20` |

Oggi compito e variante sono scelti dal loader con hash della chiave (`data.mavis_f1_sample`: `key + "/task"`;
`f1c.pick_questions`: `key + "/pick"`; `data._unimer_line`: `key + "/line"`). La regola resta deterministica, ma il
loader deve restituire anche compito e variante, così il record ammesso e l'esempio consumato si possono confrontare.

## 2. Record del manifest

Una riga JSON per `uid`:

```
uid, sorgente, revisione (40 car.), record, compito, variante, costruzione {prompt_id, domande, ritaglio, max_patches},
gruppo, split, stato, motivo, gold (S2: gold_verificato | target_controllato | incerto), evidenze [{controllo,
versione, esito, rif}], input_sha256, target_sha256, immagini [{sha256, ordine, griglia}], trasformazione {nome,
versione}, genitori [uid]
```

- `input_sha256`: sha256 del testo del prompt più gli sha256 delle immagini dopo il preprocessing, in ordine.
- `target_sha256`: sha256 del target supervisionato esatto (senza prefill né token di fine).
- `stato`: `ammesso`, `quarantena`, `escluso`; `split`: `train`, `dev`, `test`; i riservati non hanno record di
  training.
- `evidenze` cita gli esiti per riferimento (file e sha256), non copia le liste.

## 3. Gruppi e split

- Il gruppo è l'unità di separazione: stesso problema, immagine, documento o istanza del generatore, con tutte le QA,
  le descrizioni, i ritagli, le traduzioni e le varianti controfattuali. Formato `g:<sha8>`.
- Lo split si assegna al gruppo, mai al record. Un gruppo legato a un riferimento riservato è escluso dal training in
  tutte le fasi.
- Sintetici: un gruppo per scena (la coppia controfattuale `-a`/`-b` è un solo gruppo). Il test di generalizzazione su
  template o stili mai visti è uno split dichiarato a parte.
- La validazione si sceglie dal manifest (`split = dev`). Il prefisso dell'ultimo shard (`stream.py:158-166`) non si
  usa più per i run nuovi.

## 4. Indice globale (prima del manifest)

Cartella `data/index/<versione>/`, fuori da git, costruita in un job CPU sull'inventario fissato, comprese le sorgenti
riservate (`docs/ARTIFACT_INVENTORY.tsv`). Segnali per record:

1. ID e provenienza dichiarata (MAVIS e MMFineReason contengono Geometry3K e WeMath);
2. sha256 dei file e dei pixel normalizzati;
3. pHash (`decontam.phash`) e n-gram di testo e formule (`decontam.normalize_words`, `word_ngrams`).

Regole:

- Solo un legame **confermato** fonde due gruppi. Sono confermati: sha256 uguale, provenienza dichiarata, oppure un
  candidato verificato dal codice con operatori, numeri e condizioni conservati.
- pHash e n-gram producono solo candidati. Un candidato ambiguo resta in quarantena: non diventa un legame.
- Nessuna fusione transitiva tramite candidati: niente catene di somiglianze deboli.

## 5. Esclusioni e rietichettature storiche

Si importano in `data/index/<versione>/historical.jsonl` con: file d'origine e sha256, chiave, motivo, regola e data,
stato proposto. Le liste restano dove sono.

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
  loader costruisce solo i compiti e le varianti ammessi. Se ricalcolando `target_sha256` o `input_sha256` il valore
  non coincide, è un errore contato, mai un passaggio silenzioso.
- Lo stesso filtro vale per SFT, replay e RL.
- Contatori nel log e nel manifest del run, per sorgente e compito: esempi costruiti, uid unici, gruppi unici,
  ripetizioni, token di input, token visivi, target-token, scarti dopo l'ammissione (errore, `too_long`). Oggi esistono
  solo `errors/<sorgente>` (`stream.py:247`) e `too_long` in totale (`collator.py:71`).
- Se gli scarti dopo l'ammissione cambiano in modo rilevante popolazione o pesi, si aggiornano copertura e denominatori.

## 7. File e immutabilità

`data/manifests/<nome>-<versione>/`: `records.jsonl.gz` e `manifest.json`. Il `manifest.json` contiene: versione,
manifest degli shard con sha256, indice usato, liste importate con sha256, identità del codice
(`provenance.code_identity`), conteggi per sorgente, compito, split e stato, sha256 di `records.jsonl.gz`. Una
versione si scrive una volta sola (`provenance.write_new_json`); ogni correzione produce una versione nuova.

## 8. Collaudi (CPU, `tests_cpu`)

- Copia della stessa immagine in due sorgenti: un solo gruppo e un solo split.
- pHash simile con una cifra decisiva diversa: candidato, non fusione.
- Split ereditato dai derivati (ritaglio, QA, controfattuale).
- QA non ammessa su un'immagine ammessa: il loader non la costruisce.
- Target cambiato dal codice dopo l'ammissione: errore contato.
- Validazione presa dal manifest; contatori coerenti su un flusso di prova.
