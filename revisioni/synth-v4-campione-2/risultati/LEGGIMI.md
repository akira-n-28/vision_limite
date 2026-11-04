# Risultati della revisione

Qui vanno i file scaricati da `index.html` (scheda Riepilogo e file):

- `synth-v4-campione-2_avanzamento_<data-ora>.json`: obbligatorio. È il file completo: primo giudizio, momento in cui
  è stata mostrata la soluzione, giudizi finali, esami delle corrispondenze, tempi ed eventi;
- `esiti_<data-ora>.tsv` ed `esiti_decontaminazione_<data-ora>.tsv`: facoltativi. Hanno le colonne dei moduli originali
  di `../moduli/`.

Caricali con GitHub → Add file → Upload files, senza toccare gli altri file del pacchetto. Sul cluster i file si
verificano contro pacchetto, versione e impronta (`../pacchetto.json`) prima di qualsiasi uso; un esito non modifica da
solo ammissioni, quarantene o gruppi.
