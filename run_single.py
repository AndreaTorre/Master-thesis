import argparse, csv, pickle, sys
import config
from common import load_data, load_env, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--combo-index", type=int, required=True)
args = parser.parse_args()

# Ricostruisce la griglia e prende la combinazione giusta
import itertools
grid   = config.GRID_SEARCH
keys   = list(grid.keys())
combos = list(itertools.product(*grid.values()))
combo  = combos[args.combo_index]
params = dict(zip(keys, combo))

# Sovrascrive config
for k, v in params.items():
    setattr(config, k, v)

# Importa DOPO aver settato config
from utsp import run_esperimento_B_UTSP

with open("res_B_cached.pkl", "rb") as f:
    res_B = pickle.load(f)

set_seed()
env = load_env()
nodes, coords, base_dist, E, root = load_data()

print(f"Combo {args.combo_index}: {params}")

res = run_esperimento_B_UTSP(
    nodes, coords, base_dist, E, root, env,
    res_B=res_B,
    mode="local_search",
)
print("Chiavi disponibili:", list(res.keys()))
utsp_ls_val = res["UTSP_LS_val"]
sto_val     = res_B["STO"]
gap         = (utsp_ls_val - sto_val) / sto_val * 100

row = {**params, "UTSP_LS_val": utsp_ls_val, "STO_val": sto_val, "gap_%": gap}

out_file = f"grid_results_combo{args.combo_index}.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)

print(f"Salvato {out_file}  UTSP_LS_val={utsp_ls_val:.4f}  gap={gap:.2f}%")