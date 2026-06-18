# Esperimento perturbazioni

Questo codice è stato creato dopo la chiamata del 17/06 con i professori.

## Novità introdotte

Rispetto alla versione precedente, in questa versione:

- non viene più applicata la maschera sulla diagonale;
- train e test vengono eseguiti in batch;
- sono stati inseriti i parametri per STO, EEV e PI su 40 nodi;
- l'entropia è stata disattivata, perché non apportava benefici ai risultati, anzi tendeva a peggiorarli.

## Prossimi passi

Dopo questo push verranno svolte le seguenti modifiche:

- eliminazione del concetto di prenotazione;
- modifica dei costi degli scenari prima della local search, per verificare se la heatmap da sola riesce a guidare il percorso;
- se la heatmap non sarà sufficiente, verrà introdotta una soglia non statica.
