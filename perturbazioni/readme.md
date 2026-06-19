Cosa avviene in questo codice e cosa cambia dalla chiamata del 17:
- aumentato drasticamente scenari di train a 3000 e test a 300
- divisione in batches da 30 scenari di entrambi i gruppi
- spento l'entropia che non apportava nulla
- creato codice per modelli gurobi su 40 nodi
- eliminato il concetto di soglia sulla heatmap, adesso guarda le frequenze di utilizzo dalla LS e poi decide

  Cosa devo fare
  - ottenere risultati su 40 e poi su 25 nodi
  - valutare se ha senso creare un validation set di un paio di batches in cui non calcolo frequenze degli archi in I ma applico quelli prenotati post LS.
