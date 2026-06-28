# -*- coding: utf-8 -*-
import json
import os
import random
import subprocess
import numpy as np
import gurobipy as gp

from config import DATA_FILE_PRIMARY, DATA_FILE_FALLBACK, OUTPUT_DIR, GLOBAL_SEED


def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def out_path(filename):
    ensure_dir(OUTPUT_DIR)
    return os.path.join(OUTPUT_DIR, filename)


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

# Creo ambiente gurobi senza le credenziali nel codice esplicite
def load_env():
    access_id = os.getenv("GRB_WLSACCESSID")
    secret = os.getenv("GRB_WLSSECRET")
    license_id = os.getenv("GRB_LICENSEID")

    if access_id and secret and license_id:
        env = gp.Env(empty=True)
        env.setParam("WLSAccessID", access_id)
        env.setParam("WLSSecret", secret)
        env.setParam("LicenseID", int(license_id))
        env.start()
        return env

    return gp.Env()


def load_data():
    try:
        with open(DATA_FILE_PRIMARY, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        with open(DATA_FILE_FALLBACK, "r", encoding="utf-8") as f:
            data = json.load(f)

    nodes = [int(n) for n in data["node_ids"]]
    coords = {int(k): tuple(v) for k, v in data["coordinates"].items()}
    base_dist = {
        int(i): {int(j): float(v) for j, v in row.items()}
        for i, row in data["distance_matrix"].items()
    }
    E = [(i, j) for i in nodes for j in nodes if i != j]
    root = nodes[0]
    return nodes, coords, base_dist, E, root
