# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import pickle

import torch

from common import load_data, load_env, set_seed
from config import OUTPUT_DIR
from experiment_B import run_esperimento_B, run_esperimento_B_wind
from utsp import run_esperimento_B_UTSP

CACHE_PATH = os.path.join(OUTPUT_DIR, "pkl", "res_B_cached.pkl")

def main():
    parser = argparse.ArgumentParser(description="Esegue Esperimento B e le varianti UTSP.")
    parser.add_argument(
        "--only",
        choices=["B", "B_UTSP_LS"],
        default="B_UTSP_LS",
        help=(
            "B = solo esperimento B; "
            "B_UTSP_LS = B + heatmap UTSP + local search."
        ),
    )
    args = parser.parse_args()

    set_seed()
        
    env = load_env()
     
    nodes, coords, base_dist, E, root = load_data()
    from wind_perturbation import load_wind_field
    from config import ERA5_NC_PATH, WIND_ALPHA, OUTPUT_DIR
    wind  = load_wind_field(ERA5_NC_PATH)
    import numpy as np
    u = wind["u100"][0]
    v = wind["v100"][0]
    speed = np.sqrt(u**2 + v**2)
    print(f"[WIND] istanti temporali: {wind['n_times']}")
    print(f"[WIND] griglia: {len(wind['lats'])} lat x {len(wind['lons'])} lon")
    print(f"[WIND] velocita media: {speed.mean():.2f} m/s | max: {speed.max():.2f} m/s")
    alpha = WIND_ALPHA
    risultati = {}

    if os.path.exists(CACHE_PATH):
        print(f"Carico res_B da file .pkl: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            risultati["B"] = pickle.load(f)
    else:
        print("File .pkl non trovato. Eseguo esperimento B...")
        risultati["B"] = run_esperimento_B_wind(nodes, coords, base_dist, E, root, env, wind, alpha)
    

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    with open(CACHE_PATH, "wb") as f:
    	pickle.dump(risultati["B"], f)

    print(f"res_B salvato in: {CACHE_PATH}")
    
        
    if args.only == "B":
        return risultati

    if args.only == "B_UTSP_LS":
        mode = "local_search"
    else:
        mode = "local_search"

    risultati["B_UTSP"] = run_esperimento_B_UTSP(
        nodes, coords, base_dist, E, root, env,
        res_B=risultati["B"],
        mode=mode,
        scenario_kwargs={"wind": wind, "alpha": alpha},
        exp_name="espB_wind_UTSP_LS",
    )

    return risultati

 

if __name__ == "__main__":
    main()
