# Pipeline Parallela — Esperimento B

## Struttura della pipeline

```
config.py  (cambia solo DATA_FILE_PRIMARY per switchare istanza)
    │
    ├─── 15 nodi: nodi_15.json → risultati_15/
    ├─── 25 nodi: nodi_25.json → risultati_25/
    └─── 40 nodi: nodi_40.json → risultati_40/
```

```
                    ┌─────────┐
                    │  SETUP  │  Job 0 — I, b, p, C, calibrazione, perturbazioni
                    └────┬────┘
                         │  setup.pkl
            ┌────────────┼────────────┐
            ▼            ▼            ▼
       ┌────────┐  ┌──────────┐  ┌────────┐
       │   PI   │  │   EEV    │  │  STO   │  Job 1 — in parallelo
       │ 8 TSP  │  │ 1+8 TSP  │  │ 2-stage│
       └───┬────┘  └────┬─────┘  └───┬────┘
           │            │            │
           ▼            ▼            ▼
        pi.pkl      eev.pkl      sto.pkl
            └────────────┼────────────┘
                         ▼
                  ┌────────────┐
                  │  ASSEMBLE  │  Job 2 — unisce tutto
                  └──────┬─────┘
                         │
                         ▼
                  res_B_cached.pkl   (compatibile con main.py → UTSP)
                         │
                         ▼
                  ┌────────────┐
                  │  VALIDATE  │  300 scenari out-of-sample
                  └────────────┘
```

Ogni istanza (15/25/40 nodi) ha la propria cartella:

```
risultati_40/
├── parallel_data/
│   ├── setup.pkl       ← output di setup
│   ├── pi.pkl          ← output di pi
│   ├── eev.pkl         ← output di eev
│   └── sto.pkl         ← output di sto
├── res_B_cached.pkl    ← dizionario finale (26 chiavi)
├── risultati_espB.txt  ← riepilogo testuale
└── *.png               ← grafici
```


## File nuovi

| File | Cosa fa |
|------|---------|
| `gurobi_parallelo.py` | Esecuzione parallela dell'esperimento B. Contiene 6 fasi invocabili da CLI: `setup`, `pi`, `eev`, `sto`, `assemble`, `validate`. Ogni fase salva un pickle in `risultati_XX/parallel_data/`. Non tocca nessun file originale del progetto. |
| `run_0_setup.sh` | Job SLURM per la fase setup (4h, 16G). Costruisce I, b, p, C, calibra archi frequenti, genera perturbazioni. |
| `run_1_parallel.sh` | Lancia 3 job SLURM in parallelo: PI (6h, 8G), EEV (6h, 8G), STO (24h, 32G). Da lanciare dopo che setup è finito. |
| `run_2_assemble.sh` | Job SLURM per assemblare i risultati in `res_B_cached.pkl` e lanciare la validazione out-of-sample (8h, 16G). |
| `run_tutto.sh` | Lancia l'intera pipeline con dipendenze SLURM automatiche. Un solo comando per far partire tutto. |
| `run_40_e_25.sh` | Lancia le pipeline per 40 e 25 nodi contemporaneamente, usando variabili d'ambiente per differenziarle. |


## File modificati

| File | Cosa è cambiato |
|------|-----------------|
| `config.py` | Aggiunto `INSTANCE_TAG` e `N_NODES`, derivati da `DATA_FILE_PRIMARY`. I parametri che dipendono dall'istanza (medoidi, time limit, archi) sono in un blocco `if/elif` su `N_NODES`. `OUTPUT_DIR` punta automaticamente a `risultati_XX/`. Per switchare istanza basta cambiare `DATA_FILE_PRIMARY`. |
| `main.py` | `CACHE_PATH` ora usa `OUTPUT_DIR` da config invece di un path hardcodato, così carica il pickle giusto per l'istanza corrente. Aggiunto `os.makedirs` prima del salvataggio. |
| `experiment_B.py` | Aggiunto sistema di checkpoint con ripresa automatica. Se il job muore a metà, rilanci e riparte dall'ultimo passo completato. Checkpoint in `checkpoints_B/` (perturbazioni) o `checkpoints_B_wind/` (vento ERA5). |


## Come usare

### Lancio rapido (pipeline completa per una istanza)

```bash
# 1. Cambia DATA_FILE_PRIMARY in config.py al JSON desiderato, oppure:
TESI_DATA_FILE=/home/atorre/UTSP/unione/git/UTSP/data/pert/nodi_40.json bash run_tutto.sh
```

### Lancio di 40 e 25 nodi insieme

```bash
bash run_40_e_25.sh
```

### Lancio manuale passo per passo

```bash
# 1. Setup
sbatch run_0_setup.sh

# 2. Aspetta che finisca, poi lancia pi/eev/sto in parallelo
bash run_1_parallel.sh

# 3. Aspetta che finiscano tutti (controlla con squeue -u $USER)
sbatch run_2_assemble.sh
```

### Dopo l'esperimento B → parte neurale UTSP

```bash
# main.py trova res_B_cached.pkl nella cartella giusta e lo carica
python main.py --only B_UTSP_LS
```


## Switchare istanza

Cambia **una sola riga** in `config.py`:

```python
# Per 15 nodi:
DATA_FILE_PRIMARY = os.getenv("TESI_DATA_FILE", ".../nodi_15.json")

# Per 25 nodi:
DATA_FILE_PRIMARY = os.getenv("TESI_DATA_FILE", ".../nodi_25.json")

# Per 40 nodi:
DATA_FILE_PRIMARY = os.getenv("TESI_DATA_FILE", ".../nodi_40.json")
```

Tutto il resto si adatta automaticamente: medoidi, time limit STO, cartella output, nomi pickle.

In alternativa, senza toccare `config.py`, usa la variabile d'ambiente:

```bash
TESI_DATA_FILE=/path/to/nodi_25.json bash run_tutto.sh
```


## Parametri per istanza (configurati in config.py)

| Parametro | 15 nodi | 25 nodi | 40 nodi |
|-----------|---------|---------|---------|
| `K_MEDOID_NODES` | `[70, 101, 84]` | `[16, 12, 30, 44, 19]` | `[34, 26, 20, 10, 7, 51, 18, 6, 33, 14]` |
| `MAX_KMEDOID_I_ARCS` | 7 | 14 | 25 |
| `KMEDOID_ARCS_PER_NODE` | 5 | 5 | 5 |
| `STO_TIME_LIMIT` | 600s (10min) | 3600s (1h) | 43200s (12h) |
| `STO_MIP_GAP` | 0.005 | 0.005 | 0.005 |

Nota: `STO_TIME_LIMIT` per 25 nodi è impostato a 1h come punto di partenza. Da calibrare dopo il primo run.


## Checkpoint (experiment_B.py)

Per l'esecuzione sequenziale (non parallela), `experiment_B.py` salva checkpoint dopo ogni passo in `checkpoints_B/` (o `checkpoints_B_wind/`). Se il job muore:

1. Rilancia lo stesso `sbatch run_b.sh`
2. Riparte dall'ultimo passo completato

Per ripartire da zero: `rm -rf checkpoints_B/`

I checkpoint sono separati dalla pipeline parallela — sono due approcci alternativi allo stesso problema.


## Ripartire da zero

```bash
# Cancella tutto per un'istanza (es. 40 nodi)
rm -rf risultati_40/

# Oppure solo i dati intermedi (mantieni i grafici)
rm -rf risultati_40/parallel_data/ risultati_40/res_B_cached.pkl

# Per i checkpoint sequenziali
rm -rf checkpoints_B/
```
