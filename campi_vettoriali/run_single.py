# -*- coding: utf-8 -*-

import argparse
import csv
import pickle
import itertools
import hashlib
import importlib
import sys

import torch

import config
from common import load_data, load_env, set_seed


def torch_state(label):
    """
    Diagnostica Torch SENZA consumare numeri casuali.
    """
    state = torch.get_rng_state().cpu().numpy().tobytes()
    h = hashlib.sha256(state).hexdigest()[:12]
    print(f"[{label}] torch.initial_seed={torch.initial_seed()} rng_hash={h}")


parser = argparse.ArgumentParser()
parser.add_argument("--combo-index", type=int, required=True)
args = parser.parse_args()

grid = config.GRID_SEARCH
keys = list(grid.keys())
combos = list(itertools.product(*grid.values()))

combo = combos[args.combo_index]
params = dict(zip(keys, combo))

print("\n" + "=" * 80)
print(f"RUN SINGOLA — combo_index = {args.combo_index}")
print(f"Parametri: {params}")
print("=" * 80)

# 1. Imposto i parametri della combinazione
for k, v in params.items():
    setattr(config, k, v)

config.SAVE_PLOTS = False
config.SAVE_RESULTS_TXT = False

# 2. Ricarico i moduli che leggono i valori da config
for mod in ["two_stage_utsp_loss", "utsp"]:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])

# 3. Carico dati e benchmark
env = load_env()
nodes, coords, base_dist, E, root = load_data()

with open("res_B_cached.pkl", "rb") as f:
    res_B = pickle.load(f)

# 4. Reset del seme immediatamente prima della run vera
set_seed()
torch_state("prima della run")

from utsp import run_esperimento_B_UTSP

res = run_esperimento_B_UTSP(
    nodes, coords, base_dist, E, root, env,
    res_B=res_B,
    mode="local_search",
)

utsp_ls_val = res["local_search"]["UTSP_LS_val"]
sto_val = res_B["STO"]
gap = (utsp_ls_val - sto_val) / sto_val * 100

row = {
    "combo_index": args.combo_index,
    **params,
    "UTSP_LS_val": utsp_ls_val,
    "STO_val": sto_val,
    "gap_%": gap,
}

out_file = f"grid_results_combo{args.combo_index}.csv"

with open(out_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)

print(
    f"\nSalvato {out_file} | "
    f"UTSP_LS_val={utsp_ls_val:.4f} | "
    f"STO_val={sto_val:.4f} | "
    f"gap={gap:.2f}%"
)