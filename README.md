# Tesi droni — Esperimento B + UTSP

Versione ridotta del progetto: contiene solo il necessario per eseguire
`Esperimento B` e `B_UTSP`.

## File principali

- `main.py`: punto di ingresso.
- `config.py`: parametri del progetto.
- `common.py`: caricamento dati, ambiente Gurobi, utilità generali.
- `tsp_utils.py`: funzioni su archi, tour e costi.
- `gurobi_models.py`: TSP esatto, reservation TSP, STO two-stage.
- `scenarios.py`: generazione scenari e archi frequenti.
- `evaluation.py`: EEV, validazione, riassunti.
- `experiment_B.py`: Esperimento B.
- `two_stage_utsp_loss.py`: loss UTSP two-stage.
- `utsp.py`: rete UTSP e pipeline B_UTSP.

## Avvio

```bash
python main.py --only ALL
```

Oppure solo Esperimento B:

```bash
python main.py --only B
```

Nota: `B_UTSP` richiede i risultati di `B`, quindi `main.py --only B_UTSP`
esegue comunque prima `B`.

## Dati

Il codice cerca il file indicato da `TESI_DATA_FILE`; se non esiste, prova
`nodi_ch_15.json` nella cartella corrente.

```bash
export TESI_DATA_FILE=/percorso/nodi_ch_15.json
```

## Gurobi

Le credenziali non sono nel codice. Usa variabili d'ambiente:

```bash
export GRB_WLSACCESSID="..."
export GRB_WLSSECRET="..."
export GRB_LICENSEID="..."
```

## Nota

I grafici e gli esperimenti A/C/D/E/NNE sono stati esclusi dalla versione
pulita per ridurre il codice al percorso effettivamente usato.
Per slavare quanot prodotto, prova_neur.py ha dentro tutti gli esperimenti che sono stati scartati.
