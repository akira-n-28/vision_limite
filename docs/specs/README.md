# Specifiche F0

Le quattro specifiche brevi della pipeline v1.4 §5.10, da approvare prima di scrivere codice
(`docs/F0_IMPLEMENTATION_PLAN.md`, §1). Seconde bozze del 2026-10-28, rivedute dopo la revisione esterna dei push
fino a `e81f884`, con le correzioni chieste sul commit `1c1e852`. Nessuna è approvata. Sono implementati C1-C2 e,
in parte, C5 e C7:
- **C1-C2** (S2 §2, sintetici): `38af82a`, corretto il 2026-10-29 dopo la revisione di quel commit (figure tagliate,
  intersezioni nei buchi del disegno, test che non compilava). Test e controlli nei job 30392-30420. Il 2026-10-30 il
  candidato v4 è stato generato e controllato su tutti i record (job 30447-30449, S2 §2.6).
- **C5 e C7, in parte** (S1 §9): il sottoinsieme di S1 che applica il candidato nei loader, con il dev split
  (`630d00a`, job 30450-30451, S2 §2.7).

Il candidato resta incompleto per gli esperimenti, e nessun dato è certificato.

| Specifica | Tema | Decisioni dell'utente che contiene |
| --- | --- | --- |
| [S1](S1_RECORD_MANIFEST.md) | Identificatori sha256 completi e controllo delle collisioni; record con contratto, modalità e prefill; gruppi, split, indice globale; esclusioni storiche risolte su un solo record; reader e contabilità | — |
| [S2](S2_AMMISSIONE_PRIME_FAMIGLIE.md) | Correttezza rispetto all'immagine; evidenze per ogni tipo di domanda; stati `escluso`/`quarantena` per tipo; errore confermato di `intersections` e ambiguità di classificazione dei sintetici; MAVIS e UniMER | soglia di altezza del testo (proposta 10 px) |
| [S3](S3_CONTRATTO_ESTRAZIONE.md) | Contratto `vl-contract/1`; quattro forme; completezza dai metadati di terminazione; formato, completezza e correttezza separati; politica per SFT, valutazione e RL | D3: template, modalità, forma della risposta diretta, prefill |
| [S4](S4_CAMPIONAMENTO_BUDGET.md) | Disegno con unità di campionamento uguale all'unità della quantità e limiti esatti; metodi storici non adottati; budget degli audit | D4 (soglie), D5 (budget); proposta per le prime famiglie (§7) |

## Decisioni tecniche prese nelle specifiche

Sono scelte ordinarie di progetto, motivate nelle specifiche. Si cambiano con una nuova versione, non richiedono una
decisione dell'utente.

1. **Correttezza.** Una risposta è corretta se è vera per l'immagine data al modello. Ogni tipo di domanda richiede
   semantica scritta, verifica indipendente sul 100%, collaudi, invarianti del renderer e unicità. Un errore confermato
   esclude l'intero tipo nella versione colpita, fino alla correzione e a nuovi shard (S2 §1).
2. **`intersections`.** Si corregge con un solo dominio visibile `V = R + 1` per conteggio e disegno, senza cambiare le
   immagini. Le domande di classificazione mostrano le opzioni, o chiedono la classe più specifica se le classi
   formano una catena (S2 §2).
3. **Identificatori.** Tutti sono digest sha256 completi di JSON canonici. Ogni costruzione ha una mappa delle
   collisioni che si ferma con errore. I prefissi brevi servono solo alla lettura. Contratto, modalità e prefill fanno
   parte della variante (S1 §§1-2).
4. **Esclusioni storiche.** Ogni chiave deve corrispondere a un solo record della sorgente d'origine, altrimenti
   l'import fallisce. L'esclusione si estende al gruppo solo per motivi di gruppo (S1 §5).
5. **Estrazione.**
   - Un `<think>` aperto non ha finale.
   - Il troncamento si legge solo dai metadati di terminazione; senza metadati la completezza è `sconosciuta`.
   - Formato, completezza e correttezza restano separati.
   - Candidati finali in disaccordo danno `non_determinato`, con la differenza dalla lente strict del valutatore
     ufficiale dichiarata (S3 §§3, 5).
6. **Stime.** L'unità di campionamento è l'unità della quantità, con estrazione a probabilità note. I limiti sono esatti:
   ipergeometrico, o binomiale per l'estrazione proporzionale ai token. Gli incerti contano come errori e gli strati
   non si combinano salvo Bonferroni. `clustered_rate` e `corrected_upper` non si adottano; senza garanzia il
   risultato è esplorativo o condizionato (S4 §§3-4).

## Ordine di implementazione dopo l'approvazione

Elenco C1-C9 in `docs/F0_IMPLEMENTATION_PLAN.md` §3:

1. sintetici (C1, C2);
2. contratto ed estrattore (C3, C4);
3. manifest, indice e reader (C5-C7);
4. stime (C8);
5. MAVIS (C9).
