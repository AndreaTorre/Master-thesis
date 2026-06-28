# -*- coding: utf-8 -*-

import os
import sys
import csv
import pickle
import argparse
import itertools
import importlib
import random
import hashlib

import numpy as np

import config
from common import load_data, load_env


def force_seed(seed=42):
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def torch_state(label):
    import torch

    state = torch.get_rng_state().cpu().numpy().tobytes()
    h = hashlib.sha256(state).hexdigest()[:12]
    print(f"[{label}] torch.initial_seed={torch.initial_seed()} rng_hash={h}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Creo tutte le combinazioni della grid search
    grid = config.GRID_SEARCH
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))

    combo = combos[args.combo_index]
    params = dict(zip(keys, combo))

    print("\n" + "=" * 80)
    print(f"GRID CHECK - combo_index = {args.combo_index}")
    print(f"Parametri: {params}")
    print("=" * 80)

    # Imposto i parametri della combinazione dentro config
    for k, v in params.items():
        setattr(config, k, v)

    config.SAVE_PLOTS = False
    config.SAVE_RESULTS_TXT = False

    # Ricarico i moduli che importano valori da config
    for mod in ["two_stage_utsp_loss", "utsp"]:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])

    # Importo utsp solo dopo avere aggiornato config
    from common.utsp import run_esperimento_B_UTSP

    # Carico dati e benchmark
    env = load_env()
    nodes, coords, base_dist, E, root = load_data()

    with open("res_B_cached.pkl", "rb") as f:
        res_B = pickle.load(f)

    # Fisso il seme subito prima della run vera
    force_seed(args.seed)
    torch_state("prima run")

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
        "seed": args.seed,
    }

    out_file = f"grid_check_combo{args.combo_index}.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print("\nRisultato:")
    print(f"UTSP_LS_val = {utsp_ls_val:.4f}")
    print(f"STO_val     = {sto_val:.4f}")
    print(f"gap_%       = {gap:.2f}")
    print(f"Salvato in  = {out_file}")


if __name__ == "__main__":
    main()