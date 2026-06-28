import os
import importlib.util
from pathlib import Path

# ============================================================
# CONFIG UNICO UTSP
# ============================================================
# Si controlla tutto da variabili d'ambiente:
#
#   TESI_EXPERIMENT=PERT oppure CVETT
#   TESI_N_NODES=15 oppure 25 oppure 40
#
# Esempio:
#   TESI_EXPERIMENT=PERT TESI_N_NODES=40 python main.py --only B
#   TESI_EXPERIMENT=CVETT TESI_N_NODES=25 python main.py --only B_UTSP_LS
# ============================================================

ROOT_DIR = Path("/home/atorre/UTSP/unione/git/UTSP")

EXPERIMENT = os.getenv("TESI_EXPERIMENT", "PERT").upper()
N_NODES = int(os.getenv("TESI_N_NODES", "40"))

if EXPERIMENT not in {"PERT", "CVETT"}:
    raise ValueError(f"TESI_EXPERIMENT non valido: {EXPERIMENT}. Usa PERT oppure CVETT.")

if N_NODES not in {15, 25, 40}:
    raise ValueError(f"TESI_N_NODES non valido: {N_NODES}. Usa 15, 25 oppure 40.")

INSTANCE_TAG = str(N_NODES)

DATA_DIR = ROOT_DIR / "data"
PERT_DATA_DIR = DATA_DIR / "pert"
CVETT_DATA_DIR = DATA_DIR / "cvett"

DATA_FILE_PRIMARY = str(PERT_DATA_DIR / f"nodi_{N_NODES}.json")
DATA_FILE_FALLBACK = DATA_FILE_PRIMARY

EXPERIMENT_DIR = ROOT_DIR / EXPERIMENT
OUTPUT_DIR = str(EXPERIMENT_DIR / f"RISULTATI_{N_NODES}")

WIND_NC_PATH = str(CVETT_DATA_DIR / "cvett.nc")
ERA5_NC_PATH = WIND_NC_PATH

# Creo le cartelle standard, se non esistono.
for subdir in ["output", "grafici", "checkpoint", "pkl"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

# Imposto le variabili usate ancora dai vecchi config.
# Questo serve perché PERT/config.py e CVETT/config.py leggono già
# TESI_DATA_FILE e TESI_OUTPUT_DIR.
os.environ["TESI_DATA_FILE"] = DATA_FILE_PRIMARY
os.environ["TESI_DATA_FILE_FALLBACK"] = DATA_FILE_FALLBACK
os.environ["TESI_OUTPUT_DIR"] = OUTPUT_DIR

# ============================================================
# Caricamento del vecchio config come backend temporaneo
# ============================================================

if EXPERIMENT == "PERT":
    _backend_config_path = ROOT_DIR / "PERT" / "config_backend.py"
else:
    _backend_config_path = ROOT_DIR / "CVETT" / "config_backend.py"

_spec = importlib.util.spec_from_file_location("_backend_config", _backend_config_path)
_backend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend)

# Copio nel config unico tutte le variabili MAIUSCOLE del vecchio config.
# Poi sotto risovrascrivo quelle centrali, così i path restano quelli nuovi.
for _name in dir(_backend):
    if _name.isupper():
        globals()[_name] = getattr(_backend, _name)

# ============================================================
# Override finale: questi valori devono sempre venire dal config unico
# ============================================================

ROOT_DIR = str(ROOT_DIR)
EXPERIMENT = EXPERIMENT
N_NODES = N_NODES
INSTANCE_TAG = INSTANCE_TAG

DATA_FILE_PRIMARY = DATA_FILE_PRIMARY
DATA_FILE_FALLBACK = DATA_FILE_FALLBACK

EXPERIMENT_DIR = str(EXPERIMENT_DIR)
OUTPUT_DIR = OUTPUT_DIR

WIND_NC_PATH = WIND_NC_PATH
ERA5_NC_PATH = ERA5_NC_PATH

# Alias utili, se in futuro servono.
IS_PERT = EXPERIMENT == "PERT"
IS_CVETT = EXPERIMENT == "CVETT"


# ============================================================
# Batch sweep satellite (opzionale)
# ============================================================
# TESI_BATCH_SWEEP, se settata, sposta l'output sotto
# OUTPUT_DIR/batch_sweep/BATCH_<size>/ e forza UTSP_BATCH_SIZE
# a quel valore. Se non settata, nessun cambiamento rispetto a oggi.

_batch_sweep = os.getenv("TESI_BATCH_SWEEP")
if _batch_sweep is not None:
    UTSP_BATCH_SIZE = int(_batch_sweep)
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "batch_sweep", f"BATCH_{UTSP_BATCH_SIZE}")
    for subdir in ["output", "grafici", "checkpoint", "pkl"]:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
    os.environ["TESI_OUTPUT_DIR"] = OUTPUT_DIR
