# -*- coding: utf-8 -*-

import os
import re

# ── Percorso dati: CAMBIA SOLO QUESTA RIGA per switchare istanza ─
DATA_FILE_PRIMARY = os.getenv("TESI_DATA_FILE", "/home/atorre/UTSP/unione/git/UTSP/data/pert/nodi_40.json")

# Tag istanza (derivato automaticamente)
_match = re.search(r'nodi_(\d+)', DATA_FILE_PRIMARY)
INSTANCE_TAG = _match.group(1) if _match else "unknown"
N_NODES = int(INSTANCE_TAG) if INSTANCE_TAG.isdigit() else 0

DATA_FILE_FALLBACK = os.getenv("TESI_DATA_FILE_FALLBACK", f"/home/atorre/UTSP/unione/git/UTSP/data/pert/nodi_{INSTANCE_TAG}.json")
OUTPUT_DIR = os.getenv("TESI_OUTPUT_DIR", f"/home/atorre/UTSP/unione/git/UTSP/PERT/RISULTATI_{INSTANCE_TAG}")

# ── Parametri dipendenti dall'istanza ────────────────────────────
if N_NODES == 15:
    K_MEDOID_NODES       = [70, 101, 84]
    MAX_KMEDOID_I_ARCS   = 7
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT       = 600          # ~10 min
    STO_MIP_GAP          = 0.005

elif N_NODES == 25:
    K_MEDOID_NODES       = [16, 12, 30, 44, 19]
    MAX_KMEDOID_I_ARCS   = 14
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT       = 21600         
    STO_MIP_GAP          = 0.005

elif N_NODES == 40:
    K_MEDOID_NODES       = [34, 26, 20, 10, 7, 51, 18, 6, 33, 14]
    MAX_KMEDOID_I_ARCS   = 25
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT       = 43200        # 12h
    STO_MIP_GAP          = 0.005

else:
    raise ValueError(f"Istanza non configurata: nodi_{INSTANCE_TAG}. "
                     f"Aggiungi un blocco elif N_NODES == {N_NODES} in config.py")

# ── Parametri generali (comuni a tutte le istanze) ───────────────
GLOBAL_SEED = 42
N_TRAINING_SCENARIOS = 8
DO_VALIDATION = True
N_VALIDATION_SCENARIOS = 300
SCENARIO_IDS = list(range(1, N_TRAINING_SCENARIOS + 1))
FINAL_SCENARIO_SEED = GLOBAL_SEED
CALIBRATION_SCENARIO_SEED = 30
VALIDATION_SEED = 99

# Esperimenti da eseguire nel main
ESPERIMENTI_DA_ESEGUIRE = ["B", "B_UTSP"]

# Perturbazioni sintetiche
N_EXTRA_ARCS = 30
MEAN_FRAC = 0.40
SIGMA_FRAC = 0.20

# Costi first-stage / penalità
PRENOTAZIONE_FRAC = 0.25
PENALTY_FRAC = 0.50

# Archi frequenti usati nella generazione scenari di B
N_CALIBRATION_SCENARIOS = 30
N_FREQUENT_ARCS = 6
MIN_FREQ_FREQUENT = 0.80

# ── Iperparametri UTSP 2-stage ───────────────────────────────────
UTSP2_HIDDEN = 64
UTSP2_NLAYERS = 2
UTSP2_LR = 1e-3
UTSP2_EPOCHS = 50
UTSP2_STEP_LR = 10
UTSP2_LOG_FREQ = 5

# Loss weights (modalità SUM)
UTSP2_LS_ALPHA   = 0.01
UTSP2_LAMBDA1    = 5.0
UTSP2_LAMBDA2    = 1.0
UTSP2_LAMBDA_D   = 3.0
UTSP2_LAMBDA_E   = 0.5
UTSP2_TEMP_SCALE = 0.5

# Scalino booking: .sum su loss, .mean su decode
UTSP2_ALPHA_LOSS   = 0.6
UTSP2_ALPHA_DECODE = 4.0

# Temperatura kernel gaussiano adj = exp(-d/T)
UTSP2_TEMP_MODE = "median"
UTSP2_TEMP_FIXED = 1.0

# Normalizzazione distanze interna alla GNN/loss
UTSP2_DIST_SCALE_MODE = "mean_positive"

# Modalità UTSP eseguibili dal main
# "policy"       = x_utsp + secondo stadio Gurobi
# "local_search" = H_avg + decodifica + local search UTSP
# "both"         = entrambe le valutazioni con un solo training
UTSP_RUN_MODE = "local_search"

# Local search UTSP
UTSP_TRAINING_SEED = GLOBAL_SEED
N_TRAINING_SCENARIOS_UTSP = 3000
UTSP_BATCH_SIZE = 30
UTSP_LS_MAX_ACTIONS = 5000
UTSP_LS_ACTIONS_PER_ROUND = 120
UTSP_LS_MAX_RESTARTS = 80
UTSP_LS_M = 8
UTSP_LS_K = 15
UTSP_LS_BETA = 10.0
UTSP_LS_RANDOM_SEED = 12345
UTSP_LS_APPLY_INITIAL_2OPT = True

UTSP2_INCLUDE_PENALTY = True
UTSP2_INCLUDE_ENTROPY = False

# per il .sum
#GRID_SEARCH = {
#    "UTSP2_EPOCHS":     [300, 500],
#    "UTSP2_LS_ALPHA":   [0.001, 0.005, 0.01, 0.05],
#    "UTSP2_LAMBDA1":    [5.0, 10.0, 20.0, 30.0],
#    "UTSP2_LAMBDA2":    [0.1, 0.5, 1.0],
#    "UTSP2_LAMBDA_D":   [1.0, 3.0, 5.0],
#    "UTSP2_LAMBDA_E":   [0.5, 1.0, 2.0],
#    "UTSP2_TEMP_SCALE": [0.2, 0.5, 0.8],
#}

# per il .mean
#GRID_SEARCH = {
#    "UTSP2_EPOCHS":     [300, 500],
#    "UTSP2_LS_ALPHA":   [4.0, 6.0, 8.0, 10.0],
#    "UTSP2_LAMBDA1":    [20.0, 30.0,60.0, 100.0],
#    "UTSP2_LAMBDA2":    [0.5, 1.0, 3.0],
#    "UTSP2_LAMBDA_D":   [5.0, 8.0, 12.0],
#    "UTSP2_LAMBDA_E":   [0.5, 1.0, 2.0],
#    "UTSP2_TEMP_SCALE": [0.2, 0.5, 0.8],
#}
