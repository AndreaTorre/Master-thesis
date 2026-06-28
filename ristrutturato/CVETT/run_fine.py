# -*- coding: utf-8 -*-
import argparse, csv, pickle, sys
import config
from common import load_data, load_env, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--combo-index", type=int, required=True)
args = parser.parse_args()

import itertools

FINE_GRID = {
    "UTSP2_LAMBDA1":     [8.0, 10.0, 15.0, 20.0],
    "UTSP2_LAMBDA2":     [3.0],
    "UTSP2_LAMBDA_D":    [5.0, 7.0, 10.0, 15.0],
    "UTSP2_LAMBDA_E":    [0.5, 1.0, 1.5],
    "UTSP2_TEMP_SCALE":  [0.7],
    "UTSP2_ALPHA_LOSS":  [0.6, 0.7, 0.8],
    "UTSP2_ALPHA_DECODE":[3.0, 4.0, 5.0],
    "UTSP2_EPOCHS":      [300],
}

keys   = list(FINE_GRID.keys())
combos = list(itertools.product(*FINE_GRID.values()))
combo  = combos[args.combo_index]
params = dict(zip(keys, combo))

# Sovrascrive config PRIMA di qualsiasi import da utsp
for k, v in params.items():
    setattr(config, k, v)

config.SAVE_PLOTS = False
config.SAVE_RESULTS_TXT = False

from common.utsp import run_esperimento_B_UTSP

# Stessa sequenza esatta di main.py
set_seed()                                        # 1
env = load_env()                                  # 2
nodes, coords, base_dist, E, root = load_data()   # 3

with open("res_B_cached.pkl", "rb") as f:         # 4
    res_B = pickle.load(f)

print(f"Combo {args.combo_index}: {params}")

res = run_esperimento_B_UTSP(                     # 5
    nodes, coords, base_dist, E, root, env,
    res_B=res_B,
    mode="local_search",
)

utsp_ls_val = res["local_search"]["UTSP_LS_val"]
sto_val     = res_B["STO"]
gap         = (utsp_ls_val - sto_val) / sto_val * 100

row = {**params, "UTSP_LS_val": utsp_ls_val, "STO_val": sto_val, "gap_%": gap}

out_file = f"fine_results_combo{args.combo_index}.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)

print(f"Salvato {out_file}  UTSP_LS_val={utsp_ls_val:.4f}  gap={gap:.2f}%")
