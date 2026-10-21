# Audit automatico delle soluzioni F2 di Violetto su MAVIS-Geometry

150 soluzioni selezionate da f2_select.py (/home/prignano/vision_limite/outputs/f2_select/f2-mavis2k/selected.jsonl), una per problema, in parti uguali fra le profondità (seed 20261023); 20 soluzioni di calibrazione con una misura della figura cambiata apposta in un punto del ragionamento, mescolate alla cieca. Agenti Haiku: due controlli indipendenti alla cieca (config/audit/f2_solution.md: fatti sulla figura falsi o non visibili, ragionamento valido, informazioni né nella figura né nella domanda), poi terzo controllo e arbitri dove serve.

## Affidabilità degli agenti (calibrazione)

- Soluzioni con una misura cambiata escluse dalla procedura: 10/19 (52.6%, IC 95% 28.9%-75.6%); in quarantena 1. Agente singolo: 20/43 (46.5%).

## Esito

| Controllo | Soluzioni | IC 95% | Limite sup. corretto |
|---|---|---|---|
| Valide (fatti visibili, ragionamento, nessuna informazione esterna) | 97/150 (64.7%) | 56.5%-72.3% | — |
| Con fatti sulla figura falsi o non visibili | 51/148 (34.5%) | 26.8%-42.7% | 100.0% |
| Ragionamento non valido | 1/150 (0.7%) | 0.0%-3.7% | — |
| Informazioni né nella figura né nella domanda | 1/150 (0.7%) | 0.0%-3.7% | — |

Per profondità: MAVIS-Geometry d0 28 valide, 10 escluse, 0 in quarantena; MAVIS-Geometry d1 24 valide, 14 escluse, 0 in quarantena; MAVIS-Geometry d2 21 valide, 15 escluse, 1 in quarantena; MAVIS-Geometry d3 24 valide, 12 escluse, 1 in quarantena.

## Limiti

- Tutti i giudizi vengono da Haiku; la calibrazione misura solo le misure cambiate, non i passaggi di ragionamento sbagliati, che possono sfuggire di più.
- Le soluzioni sono state scelte fra quelle con la risposta finale corretta (f2_select.py): il tasso di soluzioni valide vale per i dati che entrerebbero nel training, non per tutte le generazioni.
- Nessun controllo umano durante l'audit; nessuna modifica ai dataset, ai filtri o ai training.

Dettaglio per soluzione in audit_f2.jsonl; riepilogo in f2_summary.json; soluzioni valide in include_f2_solutions.txt.
