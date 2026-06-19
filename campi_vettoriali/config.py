# -*- coding: utf-8 -*-

import os

# Percorsi
DATA_FILE_PRIMARY = os.getenv("TESI_DATA_FILE", "/home/atorre/UTSP/unione/nodi_ch_15.json")
DATA_FILE_FALLBACK = os.getenv("TESI_DATA_FILE_FALLBACK", "nodi_ch_15.json")
OUTPUT_DIR = os.getenv("TESI_OUTPUT_DIR", ".")

# Esperimenti da eseguire nel main
ESPERIMENTI_DA_ESEGUIRE = ["B", "B_UTSP"]

# Parametri generali
GLOBAL_SEED = 42
N_TRAINING_SCENARIOS = 8
DO_VALIDATION = True
N_VALIDATION_SCENARIOS = 30
SCENARIO_IDS = list(range(1, N_TRAINING_SCENARIOS + 1))
FINAL_SCENARIO_SEED = GLOBAL_SEED
CALIBRATION_SCENARIO_SEED = 30
VALIDATION_SEED = 99


# VALORI NECESSARI PER CAMPI VETTORIALI E SCENARI CON QUESTI
PERTURBATION_MODE = "wind_nc"   # oppure "random"

WIND_NC_PATH = "/home/atorre/UTSP/unione/git/cvett.nc"

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

ERA5_NC_PATH = "/home/atorre/UTSP/unione/git/cvett.nc"   # <-- metti il path reale


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

# Parametri STO Gurobi
STO_TIME_LIMIT = 1200
STO_MIP_GAP = 0.0001

# K-medoids: nodi ottenuti esternamente
#per ch=15
K_MEDOID_NODES = [70, 101, 84]
#per ch=25 
#K_MEDOID_NODES = [16, 12, 30, 44, 19]
MAX_KMEDOID_I_ARCS = 7 # 14 per ch25
KMEDOID_ARCS_PER_NODE = 5 # 10 per ch25

# Iperparametri UTSP 2-stage
UTSP2_HIDDEN = 64
UTSP2_NLAYERS = 2
UTSP2_LR = 1e-3
UTSP2_STEP_LR = 50
UTSP2_LOG_FREQ = 25


#MEAN
#UTSP2_EPOCHS = 500
#UTSP2_LS_ALPHA  = 6.0 
#UTSP2_LAMBDA1  = 60.0
#UTSP2_LAMBDA2 = 1.0 
#UTSP2_LAMBDA_D  = 12.0 
#UTSP2_LAMBDA_E  =  2.0
#UTSP2_TEMP_SCALE =  0.5


#SUM
UTSP2_EPOCHS = 500
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
N_TRAINING_SCENARIOS_UTSP = 300
UTSP_LS_MAX_ACTIONS = 5000 # era 2500
UTSP_LS_ACTIONS_PER_ROUND = 120
UTSP_LS_MAX_RESTARTS = 80 #era 40
UTSP_LS_M = 8
UTSP_LS_K = 15 # era 10
UTSP_LS_BETA = 10.0
UTSP_LS_RANDOM_SEED = 12345
UTSP_LS_APPLY_INITIAL_2OPT = True

UTSP2_INCLUDE_PENALTY = True

UTSP2_INCLUDE_ENTROPY = True

#per disabilitare k-medoids e scegliere archi
#K_MEDOID_NODES = []  # disabilita k-medoids


# Batch di training UTSP
if N_TRAINING_SCENARIOS_UTSP % N_VALIDATION_SCENARIOS != 0:
    raise ValueError(
        f"N_TRAINING_SCENARIOS_UTSP ({N_TRAINING_SCENARIOS_UTSP}) non è divisibile "
        f"per N_VALIDATION_SCENARIOS ({N_VALIDATION_SCENARIOS}). "
        f"Modifica uno dei due valori."
    )
N_TRAINING_BATCHES = N_TRAINING_SCENARIOS_UTSP // N_VALIDATION_SCENARIOS


















