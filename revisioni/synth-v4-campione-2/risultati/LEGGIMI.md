# Risultati della revisione

Qui vanno i file scaricati da `index.html` (scheda Riepilogo e file):

- `synth-v4-campione-2_avanzamento_<data-ora>.json`: obbligatorio. È il file completo, nel formato
  `synth-v4-revisione-avanzamento/2`, e contiene:
  - primi giudizi, primi giudizi fissati con la loro provenienza e momento in cui è stata mostrata ogni soluzione;
  - registro dei record esposti (quando e tramite quale unità) e archivio delle versioni scartate;
  - giudizi finali, esami delle corrispondenze, tempi ed eventi;
- `esiti_<data-ora>.tsv` ed `esiti_decontaminazione_<data-ora>.tsv`: facoltativi. Hanno le colonne dei moduli originali
  di `../moduli/`. Una risposta non indipendente vale `non_indipendente` nella colonna `risposta_revisore`, con la
  risposta e il motivo nelle note.

Caricali con GitHub → Add file → Upload files, senza toccare gli altri file del pacchetto.

Sul cluster i file si verificano prima di qualsiasi uso:
- contro pacchetto, versione e impronta (`../pacchetto.json`) e contro gli identificativi;
- con il registro delle esposizioni: un primo giudizio non indipendente non conta come indipendente.

Un esito non modifica da solo ammissioni, quarantene o gruppi.
