# -*- coding: utf-8 -*-

import os
import re

# Percorsi
DATA_FILE_PRIMARY = os.getenv("TESI_DATA_FILE", "/home/atorre/UTSP/unione/git/UTSP/data/pert/nodi_15.json")
DATA_FILE_FALLBACK = os.getenv("TESI_DATA_FILE_FALLBACK", "/home/atorre/UTSP/unione/git/UTSP/data/pert/nodi_15.json")
OUTPUT_DIR = os.getenv("TESI_OUTPUT_DIR", "/home/atorre/UTSP/unione/git/UTSP/CVETT/RISULTATI_15")

_match = re.search(r'nodi_(\d+)', DATA_FILE_PRIMARY)
INSTANCE_TAG = _match.group(1) if _match else "unknown"
N_NODES = int(INSTANCE_TAG) if INSTANCE_TAG.isdigit() else 0


# Esperimenti da eseguire nel main
ESPERIMENTI_DA_ESEGUIRE = ["B", "B_UTSP"]

# Parametri generali
GLOBAL_SEED = 42
N_TRAINING_SCENARIOS = 8
DO_VALIDATION = True
N_VALIDATION_SCENARIOS = 48  # in cvett coincide con il test holdout se uso 3000/48
SCENARIO_IDS = list(range(1, N_TRAINING_SCENARIOS + 1))
FINAL_SCENARIO_SEED = GLOBAL_SEED
CALIBRATION_SCENARIO_SEED = 30
VALIDATION_SEED = 99


# VALORI NECESSARI PER CAMPI VETTORIALI E SCENARI CON QUESTI
PERTURBATION_MODE = "wind_nc"   # oppure "random"

WIND_NC_PATH = "/home/atorre/UTSP/unione/git/UTSP/data/cvett/cvett.nc"

WIND_U_VAR = "u100"
WIND_V_VAR = "v100"
WIND_LAT_NAME = "latitude"
WIND_LON_NAME = "longitude"
WIND_TIME_NAME = "valid_time"

WIND_ALPHA = 0.25
WIND_MIN_FACTOR = 0.70
WIND_MAX_FACTOR = 1.50
WIND_TURB_SIGMA = 0.08   # deviazione std turbolenza locale (~8% per arco)
# True se coords sono coordinate locali del TSP, non lat/lon reali
WIND_MAP_COORDS_TO_GRID = True

ERA5_NC_PATH = "/home/atorre/UTSP/unione/git/UTSP/data/cvett/cvett.nc"

UTSP_BATCH_SIZE = int(os.getenv("TESI_UTSP_BATCH_SIZE", "30"))

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

# Parametri dipendenti dall'istanza
if N_NODES == 15:
    K_MEDOID_NODES = [70, 101, 84]
    MAX_KMEDOID_I_ARCS = 7
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT = 2700
    STO_MIP_GAP = 0.0001

elif N_NODES == 25:
    K_MEDOID_NODES = [16, 12, 30, 44, 19]
    MAX_KMEDOID_I_ARCS = 14
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT = 3600
    STO_MIP_GAP = 0.0001

elif N_NODES == 40:
    K_MEDOID_NODES = [34, 26, 20, 10, 7, 51, 18, 6, 33, 14]
    MAX_KMEDOID_I_ARCS = 25
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT = 43200
    STO_MIP_GAP = 0.005

else:
    raise ValueError(f"Istanza non configurata: nodi_{INSTANCE_TAG}")

# Iperparametri UTSP 2-stage
UTSP2_HIDDEN = 64
UTSP2_NLAYERS = 2
UTSP2_LR = 1e-3
UTSP2_EPOCHS = 50       # era 500
UTSP2_STEP_LR = 10      # era 50 (proporzionale)
UTSP2_LOG_FREQ = 5 


#MEAN
#UTSP2_EPOCHS = 500
#UTSP2_LS_ALPHA  = 6.0 
#UTSP2_LAMBDA1  = 60.0
#UTSP2_LAMBDA2 = 1.0 
#UTSP2_LAMBDA_D  = 12.0 
#UTSP2_LAMBDA_E  =  2.0
#UTSP2_TEMP_SCALE =  0.5


#SUM

UTSP2_LS_ALPHA  = 0.01 
UTSP2_LAMBDA1  = 5.0
UTSP2_LAMBDA2 = 1.0 
UTSP2_LAMBDA_D  = 3.0 
UTSP2_LAMBDA_E  =  0.5
UTSP2_TEMP_SCALE =  0.5

# tengo .sum su loss booking e loss penalty e quindi uso un parametro meno aggressivo di 0.4,
#mentre usando .mean su decode booking per gli x tilde mi serve un valore di alpha maggiore altrimenti
# lo scalino desiderato non c'è e un arco prenotabile non supera la treshold
UTSP2_ALPHA_LOSS = 0.6 # post grid era 0.6   
UTSP2_ALPHA_DECODE  = 4.0



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
#3888

GRID_SEARCH = {
    "UTSP2_LAMBDA1":  [5.0, 10.0, 20.0],
    "UTSP2_LAMBDA2":  [0.5, 1.0, 3.0],
    "UTSP2_LAMBDA_D": [1.0, 3.0, 5.0],
} #27 combinazioni

#per il .mean
#GRID_SEARCH = {
#    "UTSP2_EPOCHS":     [300, 500],
#    "UTSP2_LS_ALPHA":   [4.0, 6.0, 8.0, 10.0],
#    "UTSP2_LAMBDA1":    [20.0, 30.0,60.0, 100.0],
#    "UTSP2_LAMBDA2":    [0.5, 1.0, 3.0],
#    "UTSP2_LAMBDA_D":   [5.0, 8.0, 12.0],
#    "UTSP2_LAMBDA_E":   [0.5, 1.0, 2.0],
#    "UTSP2_TEMP_SCALE": [0.2, 0.5, 0.8],
#} # 2592 combinazioni

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

# Local search UTSP da unionev3.py
UTSP_TRAINING_SEED = GLOBAL_SEED
N_WIND_OBSERVATIONS = int(os.getenv("TESI_N_WIND_OBSERVATIONS", "3048"))
N_TRAINING_SCENARIOS_UTSP = int(os.getenv("TESI_N_TRAINING_SCENARIOS_UTSP", "3000"))
N_TEST_SCENARIOS_UTSP = N_WIND_OBSERVATIONS - N_TRAINING_SCENARIOS_UTSP

TRAIN_SCENARIO_IDS_UTSP = list(range(1, N_TRAINING_SCENARIOS_UTSP + 1))
TEST_SCENARIO_IDS_UTSP = list(range(N_TRAINING_SCENARIOS_UTSP + 1, N_WIND_OBSERVATIONS + 1))

# Il training è fatto solo con batch completi; il test può avere ultimo batch incompleto.
DROP_LAST_TRAIN_BATCH = True
DROP_LAST_TEST_BATCH = False
UTSP_LS_MAX_ACTIONS = 5000 # era 2500
UTSP_LS_ACTIONS_PER_ROUND = 120
UTSP_LS_MAX_RESTARTS = 80 #era 40
UTSP_LS_M = 8
UTSP_LS_K = 15 # era 10
UTSP_LS_BETA = 10.0
UTSP_LS_RANDOM_SEED = 12345
UTSP_LS_APPLY_INITIAL_2OPT = True

UTSP2_INCLUDE_PENALTY = True

UTSP2_INCLUDE_ENTROPY = False

#per disabilitare k-medoids e scegliere archi
#K_MEDOID_NODES = []  # disabilita k-medoids


# Batch di training/test UTSP su osservazioni del file .nc
if N_TEST_SCENARIOS_UTSP <= 0:
    raise ValueError(
        f"N_TRAINING_SCENARIOS_UTSP ({N_TRAINING_SCENARIOS_UTSP}) deve essere minore "
        f"di N_WIND_OBSERVATIONS ({N_WIND_OBSERVATIONS})."
    )

if DROP_LAST_TRAIN_BATCH and N_TRAINING_SCENARIOS_UTSP % UTSP_BATCH_SIZE != 0:
    raise ValueError(
        f"Training non divisibile in batch completi: "
        f"{N_TRAINING_SCENARIOS_UTSP} scenari, batch_size={UTSP_BATCH_SIZE}."
    )

N_TRAINING_BATCHES = N_TRAINING_SCENARIOS_UTSP // UTSP_BATCH_SIZE
N_TEST_BATCHES = (N_TEST_SCENARIOS_UTSP + UTSP_BATCH_SIZE - 1) // UTSP_BATCH_SIZE


















