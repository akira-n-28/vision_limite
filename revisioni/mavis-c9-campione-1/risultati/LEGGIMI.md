# Risultati della revisione

Qui vanno i file scaricati da `index.html` (pagina Riepilogo e file):

- `mavis-c9-campione-1_avanzamento_<data-ora>.json`: obbligatorio. È il file completo, nel formato
  `mavis-c9-revisione-avanzamento/1`, e contiene:
  - primi giudizi, primi giudizi fissati con la loro provenienza e momento in cui è stata mostrata ogni soluzione;
  - registro dei record e dei gruppi esposti (quando e tramite quale scheda) e archivio delle versioni scartate;
  - giudizi finali, tempi ed eventi;
- `esiti_<data-ora>.tsv`: facoltativo, una riga per scheda. La colonna `indipendente` dice se il primo giudizio è
  indipendente (il controllo noto non lo è mai); `non_applicabile` segna i campi che non riguardano la scheda.

Caricali con GitHub → Add file → Upload files, senza toccare gli altri file del pacchetto.

Sul cluster i file si verificano prima di qualsiasi uso:
- contro pacchetto, versione e impronta (`../pacchetto.json`) e contro gli identificativi delle schede;
- con il registro delle esposizioni: un primo giudizio non indipendente non conta come indipendente.

Un esito non modifica da solo ammissioni, quarantene, gruppi o compiti, e l'accordo con un target non lo rende gold.
