# Riesame tecnico dei 15 candidati incerti della decontaminazione (synth-v4-campione-2)

Riesame del 2026-11-12, fatto da Claude Code sul cluster; **non è una seconda revisione umana**. Riguarda solo i 15
record che il revisore ha chiuso con esito `incerto`. Il giudizio umano resta quello dei file caricati in
`revisioni/synth-v4-campione-2/risultati/` e non si modifica. Le conclusioni sono nel TSV
[`synth-v4-campione-2-riesame-incerti.tsv`](synth-v4-campione-2-riesame-incerti.tsv), che `scripts/review_results.py`
legge con `--riesame`. Nella proposta ammissioni stanno in una categoria separata, che chiede l'approvazione
dell'utente (S2 §2.13).

## Evidenze consultate

Per tutti i 15 record, materiale estratto nel job 30915 (`docs/revisioni/synth-v4-campione-2-integrazione-2.json`,
sezione `riesame_incerti`):
- **giudizio del revisore:** esito `incerto` del record ed esame `incerto` per ciascuna delle 28 corrispondenze, con le
  note. Il file di avanzamento è `4a88d60c…`;
- **record:** JSON dallo shard v4 (specifica, QA e, per le coppie, la domanda della coppia), con lo sha256 registrato
  nel manifest `synth-v4-candidato-2` (`9a8104cd…`);
- **immagini:** il PNG del record e le immagini del pannello, nel pacchetto e con gli sha256 ricontrollati dalla
  verifica statica ripetuta. Le ho guardate direttamente;
- **pannello:** la voce di `pannello-v3` (testo della domanda) e la riga del benchmark (domanda e risposta; per MV-MATH
  solo il testo dell'indice), con gruppo e ruoli degli elementi;
- **manifest:** stato, gruppo e motivi del record, e tutti i membri del gruppo;
- **provenienza:** le scene vengono da `synth.py` versione 4 (`generator_version` 4 nella specifica), dalla chiave del
  record. Il generatore non legge dati dei benchmark (S2 §2.9).

**Criterio.** Si confronta l'istanza del benchmark (figura, domanda e risposta) col record. Una formula elementare
comune non dimostra contaminazione, e un rendering diverso non dimostra indipendenza.

## Le 14 scene y = x + 1

**Il record.** Retta `y = x + 1` (curva `line` con a = 1 e b = 1), assi da -6 a 6 con le tacche, etichetta `f`, senza
griglia. Specifica identica nei 14 record, che stanno nello stesso gruppo della scena.
- **Domande dei 13 record di train:** crescente o decrescente (crescente); la retta taglia l'asse y sopra o sotto
  l'origine (sopra); esistenza di `f` (sì) e di un'altra etichetta (no).
- **Coppia `synth-pair-00074-b`:** la sua domanda è crescente o decrescente (crescente).

**Le corrispondenze**, tutte a distanza pHash 4:

| Elemento | Ruolo nel pannello | Figura | Domanda e risposta del benchmark |
| --- | --- | --- | --- |
| DynaMath, problema 96, variante 8 | test, metrica primaria; il gruppo comprende le 10 varianti | retta blu y = x + 1 su griglia da -8 a 8, intercette segnate con due punti | distanza fra l'intercetta x e l'intercetta y: 1.4142 |
| MV-MATH 1008, immagine 2 | test, metrica primaria | retta viola y = x + 1 su assi da -5 a 5, piccola | quale dei quattro grafici non rappresenta y come funzione di x (scelta multipla); l'immagine 2 è un'opzione |

**Confronto.** In comune c'è la funzione elementare. Le domande dei benchmark non sono quelle dei record, e le risposte
dei record non danno né implicano 1.4142 o la lettera della scelta multipla. Le figure hanno intervallo, griglia, colori,
segni ed etichette diversi, e la provenienza è indipendente.

**Conclusione: contenuto diverso.** È la coincidenza di una formula comune, non la riproduzione dell'istanza.

| Unità | Record | Insieme, split, stato | Corrispondenze | Conclusione |
| --- | --- | --- | --- | --- |
| D-022 | synth-0036534 | train, test, escluso | MV-MATH 1008#2, DynaMath 96 v8 | contenuto diverso |
| D-041 | synth-0005122 | train, test, escluso | DynaMath 96 v8 | contenuto diverso |
| D-048 | synth-0045894 | train, test, escluso | DynaMath 96 v8 | contenuto diverso |
| D-069 | synth-0001142 | train, test, escluso | MV-MATH 1008#2, DynaMath 96 v8 | contenuto diverso |
| D-071 | synth-0030226 | train, test, escluso | MV-MATH 1008#2, DynaMath 96 v8 | contenuto diverso |
| D-082 | synth-pair-00074-b | coppie, test, quarantena | DynaMath 96 v8 | contenuto diverso |
| D-094 | synth-0039818 | train, test, escluso | MV-MATH 1008#2, DynaMath 96 v8 | contenuto diverso |
| D-134 | synth-0015650 | train, test, escluso | MV-MATH 1008#2, DynaMath 96 v8 | contenuto diverso |
| D-144 | synth-0028262 | train, test, escluso | DynaMath 96 v8 | contenuto diverso |
| D-163 | synth-0033994 | train, test, escluso | MV-MATH 1008#2, DynaMath 96 v8 | contenuto diverso |
| D-184 | synth-0029486 | train, test, escluso | MV-MATH 1008#2, DynaMath 96 v8 | contenuto diverso |
| D-196 | synth-0036994 | train, test, escluso | MV-MATH 1008#2, DynaMath 96 v8 | contenuto diverso |
| D-200 | synth-0038254 | train, test, escluso | MV-MATH 1008#2, DynaMath 96 v8 | contenuto diverso |
| D-227 | synth-0034434 | train, test, escluso | MV-MATH 1008#2, DynaMath 96 v8 | contenuto diverso |

**Conseguenze.**
- **Training:** nessuna. I 13 record di train sono già esclusi per contaminazione interna (motivo `contaminazione`):
  hanno lo stato della scena delle coppie riservate `synth-pair-00074` e `synth-pair-00198`, e restano esclusi qualunque
  sia l'esito della decontaminazione. Il loro split `test` è quello del gruppo, che contiene record di valutazione.
- **Gruppo:**
  - col solo giudizio umano la decontaminazione del gruppo resta irrisolta;
  - se l'utente accetta il riesame, il motivo `decontaminazione_pannello` si toglie a tutto il gruppo. Le quattro
    coppie riservate (`synth-pair-00074-a` e `-b`, `synth-pair-00198-a` e `-b`), che hanno solo quel motivo,
    tornerebbero nell'insieme di valutazione sintetico.
- **Pannello:**
  - DynaMath 96 (10 varianti di test) riceve un'evidenza diretta da un record di valutazione sintetico solo da
    `synth-pair-00074-b`. Accettato il riesame, quel legame non è più materiale per `pannello-v4`; altrimenti resta
    (S2 §2.10, `docs/PANNELLO_VALUTAZIONE.md` §3.1);
  - MV-MATH 1008 è legato solo a record di train esclusi: nessuna conseguenza;
  - nessun ruolo di `pannello-v3` cambia.

## D-188, synth-0016025

**Il record.** Train, split train, in quarantena per il solo motivo `decontaminazione_pannello`; il gruppo ha solo lui.
Triangolo isoscele PQR con l'angolo al vertice R segnato, 74°, e i lati scritti: PQ = 12, QR = RP = 10, coerenti con
l'angolo. Le sue domande sono:
- misura dell'angolo segnato in R (74°);
- lunghezze dei tre lati;
- tipo di triangolo (isoscele);
- vertici estremi;
- numero di lati con la misura;
- esistenza di due etichette.

**Le corrispondenze.** Quattro elementi di Geoperception con la stessa immagine, a distanza pHash 4. Hanno ruolo
sviluppo e nessuna metrica primaria; il gruppo del pannello è lo stesso. La figura è il triangolo isoscele TUV con
l'angolo al vertice T annotato, 74°, e le tacche di uguaglianza su TU e TV.

| Elemento | Domanda | Risposta |
| --- | --- | --- |
| 7600 | measure of angle VTU as annotated | 74 |
| 7602 | line equal to line TV | TU |
| 7604 | line equal to line TU | TV |
| 102362 | is angle TVU acute or obtuse | acute |

**Confronto.**
- **In comune:** la stessa configurazione (triangolo simile, orientato quasi allo stesso modo), lo stesso valore
  annotato al vertice e una domanda equivalente con la stessa risposta (la misura dell'angolo segnato, 74). Il
  revisore l'aveva già detto: la coincidenza matematica è significativa.
- **Diverso:** le etichette (PQR contro TUV) e le annotazioni (lunghezze contro tacche). La generazione è
  indipendente.
- **Le altre domande del benchmark:** lati uguali e angolo acuto non sono fra quelle del record.

**Conclusione: irrisolto.** Le evidenze non distinguono una coincidenza di configurazione da una copia della figura nel
contenuto. Il record resta nello stato conservativo corrente.

**Conseguenze.**
- **Il record:** resta in quarantena; è un solo record, in un gruppo che ha solo lui, e un'esclusione avrebbe lo stesso
  effetto sul training.
- **Il pannello:** gli elementi sono di sviluppo, senza metrica primaria, e nessun ruolo cambia.

## Che cosa questo riesame non fa

- Non cambia il giudizio umano né il manifest. Le quattro riammissioni che ne dipendono restano proposte da approvare.
- La mia lettura delle immagini è un controllo tecnico, non una seconda revisione umana indipendente.
- Non cerca altri quasi-duplicati oltre ai candidati pHash della scansione.
