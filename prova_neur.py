# -*- coding: utf-8 -*-


import json
import os
import random

import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import sys 

# Import aggiuntivi per NNE
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from unionev3_loss import run_esperimento_B_UTSP

# Scelta dell espetimento da far girare:
ESPERIMENTI_DA_ESEGUIRE = ["B","B_UTSP"]

# PARAMETRI GENERALI 

# PERCORSO DATI 
DATA_FILE_PRIMARY  = r"/home/atorre/UTSP/unione/nodi_ch_15.json"
DATA_FILE_FALLBACK = "nodi_ch_15.json"

# Tengo la directory attuale per salvare
OUTPUT_DIR = "."  

# Credenziali Gurobi  
GUROBI_WLSACCESSID = "235a9511-a6a5-4853-ac61-0e9cae25f646"
GUROBI_WLSSECRET   = "66a7d4be-3cf0-4604-98d1-147bd12370b3"
GUROBI_LICENSEID   = 2805342

# Parametri condivisi 
GLOBAL_SEED  = 42
N_TRAINING_SCENARIOS   = 8     # numero di scenari su cui girare STO e medione
DO_VALIDATION          = True  # esegui validazione out-of-sample
N_VALIDATION_SCENARIOS = 30     # numero di scenari di validazione (seme diverso)
SCENARIO_IDS = list(range(1, N_TRAINING_SCENARIOS + 1))  # generato automaticamente
FINAL_SCENARIO_SEED = GLOBAL_SEED # Seme usato per gli scenari finali degli esperimenti
CALIBRATION_SCENARIO_SEED = 30 # Seme diverso, usato solo per gli scenari di calibrazione dei k-medoids
N_EXTRA_ARCS = 30          # archi casuali aggiuntivi da perturbare
MEAN_FRAC    = 0.40        # media perturbazione 
SIGMA_FRAC   = 0.20        # deviazione std perturbazione 
PRENOTAZIONE_FRAC = 0.25   # p = 25% del costo di percorrenza b
PENALTY_FRAC = 0.5 # valore della multa
N_CALIBRATION_SCENARIOS = 30 # scenari di calibrazione per scegliere quali I tenere 
MIN_FREQ_FOR_CANDIDATE = 2 # frequenza minima
MAX_FREQ_FOR_CANDIDATE = 0.9 # frequenza massima
N_FREQUENT_ARCS   = 6     # archi ad alta frequenza nei PI di calibrazione
MIN_FREQ_FREQUENT = 0.80  # soglia: arco incluso se appare in ≥ 65% dei PI di calibrazione
VALIDATION_SEED   = 99    # seme indipendente per scenari di validazione


# Parametri STO 
STO_TIME_LIMIT = 600
STO_MIP_GAP    = 0.0001


# Parametri NNE
# Generazione dataset 
NNE_N_SAMPLES       = 1500   # campioni totali (train + val)
NNE_K_PRIME         = 15     # scenari per campione (K' del paper)
NNE_TRAIN_FRAC      = 0.80  # 80% train, 20% val
NNE_DATA_SEED       = 7     # seme per la shuffling del dataset
NNE_SCEN_BASE_SEED  = 5000  # base-seed degli scenari di training
                             # (diverso da FINAL_SCENARIO_SEED!)
#  Architettura (Ψ1, Ψ2, ΦE)
NNE_EMBED_HIDDEN_DIM = 16   # Ψ1: hidden dim dello scenario encoder
NNE_EMBED_DIM1       = 8   # Ψ1: output (dimensione embedding intermedia)
NNE_EMBED_DIM2       = 4    # Ψ2: output (dimensione embedding finale ξ_λ)
NNE_RELU_HIDDEN_DIM  = 16   # ΦE: hidden dim del feed-forward finale
NNE_AGG_TYPE         = "mean"   # aggregazione su K': "mean" oppure "sum"
#  Training 

NNE_BATCH_SIZE  = 32
NNE_N_EPOCHS    = 150
NNE_LOG_FREQ    = 5   # stampa metriche ogni N epoche
NNE_DROPOUT     = 0.10   # 0 = disabilitato
NNE_LR          = 1e-3



# Parametri A
I_INDICES_A    = [(0, 3), (14, 8), (6, 1), (1, 10), (7, 11)]
SIGMA_FRAC_A   = 0.10   

# Parametri D 
N_AREAS_D      = 3
MARGIN_FRAC_D  = 0.25
RADIUS_FRAC_D  = 0.12
MAX_ARCS_D     = 8
MAX_AREA_I_ARCS_D = 7
AREA_ARCS_PER_AREA_D = 2
MIN_FREQ_D = 0.20
MAX_FREQ_D = 0.80

# Conversione vento E
WIND_COST_ALPHA_E      = 0.50   # intensità della correzione vento/costo
WIND_REFERENCE_SPEED_E = 10.0   # scala di normalizzazione del campo vettoriale
WIND_MIN_FACTOR_E      = 0.35   # costo minimo = 35% del costo base
WIND_MAX_FACTOR_E      = 1.80   # costo massimo = 180% del costo base
WIND_ARC_SAMPLES_E     = 9      # punti campionati lungo ogni arco

# K-MEDOIDS  (usato da B, C, E)
# I cluster PAM sono stati generati in uno script esterno sui nodi, non sugli archi.
# I nodi medoidi ottenuti sono:
#   C1 = 70  (lato a destra)
#   C2 = 101 (basso a destra)
#   C3 = 84  (alto a sinistra)
# Per gli esperimenti B, C ed E l'insieme I_k contiene tutte le tratte uscenti da questi nodi.
# Nota: nel modello corrente I viene gestito come tratta non orientata, quindi ogni arco uscente
# (m, j) viene salvato nella forma canonica {m, j}.
K_MEDOID_NODES = [70, 101, 84]
# Parametro per k-medoid, numero massimo di archi uscenti da un nodo medoide da tenere in considerazione in I
# poi di quei 15 archi ne tengo la metà
MAX_KMEDOID_I_ARCS = 7
KMEDOID_ARCS_PER_NODE = 5


# CARICAMENTO GUROBI E DATI

def load_env():
    return gp.Env()


def load_data():
    try:
        with open(DATA_FILE_PRIMARY, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        with open(DATA_FILE_FALLBACK, "r") as f:
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


def out_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


# FUNZIONI BASE: GRAFO, DISTANZE, PERTURBAZIONI

# Non oriento la tratta i,j
def canon_edge(i, j):
    return (i, j) if i < j else (j, i)


def directed_to_undirected_arcs(arcs):
    return sorted({canon_edge(i, j) for (i, j) in arcs if i != j})


def all_undirected_edges(nodes):
    return sorted({
        canon_edge(nodes[a], nodes[b])
        for a in range(len(nodes))
        for b in range(a + 1, len(nodes))
    })


# Costruisco I_k per B, C ed E a partire dai nodi medoidi ottenuti nello script esterno



def build_I_from_medoid_outgoing_nodes(nodes, E, base_dist,
                                        medoid_nodes=K_MEDOID_NODES,
                                        max_arcs=MAX_KMEDOID_I_ARCS,
                                        arcs_per_medoid=KMEDOID_ARCS_PER_NODE):
    nodes_set = set(nodes)
    missing = [m for m in medoid_nodes if m not in nodes_set]
    if missing:
        raise ValueError(f"Nodi medoidi non presenti nei dati: {missing}")

    E_set = set(E)

    selected = []
    all_candidates = []

    for m in medoid_nodes:
        outgoing_directed = [
            (m, j)
            for j in nodes
            if j != m and (m, j) in E_set
        ]

        outgoing_undir = directed_to_undirected_arcs(outgoing_directed)

        outgoing_undir = sorted(
            outgoing_undir,
            key=lambda e: base_cost_undirected(base_dist, e[0], e[1])
        )

        taken_for_medoid = 0

        for edge in outgoing_undir:
            
            if len(selected) >= max_arcs:
                break
            if edge not in selected and taken_for_medoid < arcs_per_medoid:
                selected.append(edge)
                taken_for_medoid += 1

            all_candidates.append(edge)

    # Se non arrivo a max_arcs con la quota per medoide,
    # completo con le tratte k-medoids più corte rimaste.
    all_candidates = sorted(
        set(all_candidates),
        key=lambda e: base_cost_undirected(base_dist, e[0], e[1])
    )



    I_k = sorted(selected)

    info = {
        "source": "limited_medoid_outgoing_nodes",
        "medoid_nodes": list(medoid_nodes),
        "max_arcs": max_arcs,
        "arcs_per_medoid": arcs_per_medoid,
        "selected_I": I_k,
        "n_undirected_tratte": len(I_k),
    }

    return I_k, info

# Costo base della tratta non orientata 
def base_cost_undirected(base_dist, i, j):
    return 0.5 * (base_dist[i][j] + base_dist[j][i])

# Valore di un parametro associato a una tratta non orientata.
# Convenzione dei nomi:
#   p = costo di prenotazione
#   C = multa
#   b = costo di percorrenza
def get_edge_value(values, i, j):
    edge = canon_edge(i, j)
    if isinstance(values, dict):
        return values[edge]
    return values

# Compatibilità con il codice precedente: C indica sempre la multa.
def get_C_value(C, i, j):
    return get_edge_value(C, i, j)

# Genero due tipologie di perturbazioni: una controvento (+delta), l'altra a favore (-delta). Stessa intensità ma verso diverso
def add_directional_wind_perturbation(result, i, j, rng, mean_frac, sigma_frac, base_dist):
    i, j = canon_edge(i, j)
    base_ij = base_dist[i][j]
    base_ji = base_dist[j][i]
    ref_len = base_cost_undirected(base_dist, i, j)
    magnitude = max(0.0, rng.gauss(mean_frac * ref_len, sigma_frac * ref_len))

    if rng.random() < 0.5:
        delta_ij, delta_ji = magnitude, -magnitude
    else:
        delta_ij, delta_ji = -magnitude, magnitude

    result[(i, j)] = max(base_ij + delta_ij, 0.05 * base_ij) - base_ij
    result[(j, i)] = max(base_ji + delta_ji, 0.05 * base_ji) - base_ji


def select_diverse_edges(edges, k, max_degree=2):
    selected = []
    degree = {}

    for (i, j) in edges:
        if degree.get(i, 0) >= max_degree:
            continue
        if degree.get(j, 0) >= max_degree:
            continue

        selected.append((i, j))
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1

        if len(selected) >= k:
            break

    return selected


# Costruzione delle perturbazioni
# Archi perturbati = I ∪ frequent_arcs ∪ random_extra
# I template fissi per scenario sono stati rimossi.
def build_perturbation(
    scenario_id, nodes, base_dist, I,
    n_extra_arcs, mean_frac, sigma_frac, base_seed,
    frequent_arcs=None
):
    if frequent_arcs is None:
        frequent_arcs = []

    rng = random.Random(base_seed + scenario_id)

    # Unione deterministica: I ∪ frequent_arcs
    selected_edges = set(I) | set(frequent_arcs)

    # Estrazione casuale: escludi I e frequent_arcs
    all_edges = all_undirected_edges(nodes)
    candidate_edges = [e for e in all_edges if e not in selected_edges]

    selected_edges.update(
        rng.sample(candidate_edges, min(n_extra_arcs, len(candidate_edges)))
    )

    result = {}
    for (i, j) in selected_edges:
        add_directional_wind_perturbation(
            result, i, j, rng, mean_frac, sigma_frac, base_dist
        )

    return result

# Applico le perturbazioni in base all'orientamento della matrice delle distanze
def build_scenario_dist(base_dist, perturb_dict):
    sd = {i: {j: base_dist[i][j] for j in base_dist[i]} for i in base_dist}
    for (i, j), delta in perturb_dict.items():
        sd[i][j] = base_dist[i][j] + delta
    return sd
   
# Ricostruisco il tour ordinato
def extract_tour_from_arcs(arcs, start):
    succ = {i: j for (i, j) in arcs}
    tour = [start]
    current = start
    while True:
        if current not in succ:
            raise ValueError("Tour non ricostruibile: manca un successore.")
        nxt = succ[current]
        if nxt == start:
            break
        if nxt in tour:
            raise ValueError("Tour non semplice o sottociclo.")
        tour.append(nxt)
        current = nxt
    return tour

# calcolo la lunghezza totale del tour
def tour_length_from_arcs(arcs, dist):
    return sum(dist[i][j] for (i, j) in arcs)


# TSP ESATTO

# Risolvo il TSP esatto , no archi obbligaotri, MTZ per il subtour
def solve_exact_tsp(nodes, E, dist, root, env, fixed_arcs=None, fixed_edges_undir=None, output_flag=0):
    if fixed_arcs is None:
        fixed_arcs = []
    if fixed_edges_undir is None:
        fixed_edges_undir = []

    model = gp.Model("tsp", env=env)
    model.Params.OutputFlag = output_flag
    model.Params.Threads = 1
    model.Params.Seed = 42
    n = len(nodes)

    y = model.addVars(E, vtype=GRB.BINARY, name="y")
    u = model.addVars(nodes, lb=0, ub=n - 1, vtype=GRB.CONTINUOUS, name="u")

    model.setObjective(gp.quicksum(dist[i][j] * y[i, j] for (i, j) in E), GRB.MINIMIZE)

    for i in nodes:
        model.addConstr(gp.quicksum(y[i, j] for j in nodes if j != i) == 1, f"out_{i}")
    for j in nodes:
        model.addConstr(gp.quicksum(y[i, j] for i in nodes if i != j) == 1, f"in_{j}")
    for (i, j) in fixed_arcs:
        model.addConstr(y[i, j] == 1, f"fixed_arc_{i}_{j}")
    for (i, j) in fixed_edges_undir:
        i2, j2 = canon_edge(i, j)
        model.addConstr(y[i2, j2] + y[j2, i2] == 1, f"fixed_edge_{i2}_{j2}")
    for i in nodes:
        for j in nodes:
            if i != j and i != root and j != root:
                model.addConstr(u[i] - u[j] + n * y[i, j] <= n - 1, f"mtz_{i}_{j}")

    model.optimize()

    if model.SolCount == 0:
        return {"status": "NO_SOLUTION", "objective": None, "arcs": [], "tour": [], "length": None}

    arcs         = [(i, j) for (i, j) in E if y[i, j].X > 0.5]
    ordered_tour = extract_tour_from_arcs(arcs, root)
    length       = tour_length_from_arcs(arcs, dist)

    status_map = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INFEASIBLE: "INFEASIBLE"}
    return {
        "status": status_map.get(model.Status, str(model.Status)),
        "objective": model.ObjVal,
        "arcs": arcs,
        "tour": ordered_tour,
        "length": length,
    }



# PIV / WAIT AND SEE

# Risolveo un TSP con possibilità di prenotare tratte in I.
# x[i,j] = 1 se la tratta {i,j} viene prenotata.
# y[i,j] = 1 se l'arco orientato (i,j) viene percorso.
# z[i,j] = 1 se la tratta {i,j} viene percorsa senza prenotazione: in quel caso si paga C.
def solve_reservation_tsp(nodes, E, I, dist, root, p, C, env,
                          fixed_reservations=None, output_flag=0,
                          model_name="reservation_tsp"):
    model = gp.Model(model_name, env=env)
    model.Params.OutputFlag = output_flag
    model.Params.Threads = 1
    model.Params.Seed = 42
    n = len(nodes)

    x = model.addVars(I, vtype=GRB.BINARY, name="x_prenota")
    y = model.addVars(E, vtype=GRB.BINARY, name="y_percorri")
    z = model.addVars(I, vtype=GRB.BINARY, name="z_multa")
    u = model.addVars(nodes, lb=0, ub=n - 1, vtype=GRB.CONTINUOUS, name="u")

    costo_percorrenza = gp.quicksum(dist[i][j] * y[i, j] for (i, j) in E)
    costo_prenotazione = gp.quicksum(get_edge_value(p, i, j) * x[i, j] for (i, j) in I)
    costo_multe = gp.quicksum(get_edge_value(C, i, j) * z[i, j] for (i, j) in I)
    model.setObjective(costo_prenotazione + costo_percorrenza + costo_multe, GRB.MINIMIZE)

    for i in nodes:
        model.addConstr(gp.quicksum(y[i, j] for j in nodes if j != i) == 1, f"out_{i}")
    for j in nodes:
        model.addConstr(gp.quicksum(y[i, j] for i in nodes if i != j) == 1, f"in_{j}")

    for (i, j) in I:
        used_ij = y[i, j] + y[j, i]
        # z = (1 - x) * used_ij, con variabili binarie.
        # Quindi la multa si paga solo se uso una tratta in I senza prenotarla.
        model.addConstr(z[i, j] <= used_ij, f"z_le_used_{i}_{j}")
        model.addConstr(z[i, j] <= 1 - x[i, j], f"z_le_not_reserved_{i}_{j}")
        model.addConstr(z[i, j] >= used_ij - x[i, j], f"z_ge_used_minus_reserved_{i}_{j}")

    if fixed_reservations is not None:
        fixed_set = {canon_edge(*e) for e in fixed_reservations}
        for (i, j) in I:
            model.addConstr(x[i, j] == (1 if canon_edge(i, j) in fixed_set else 0),
                            f"fix_prenotazione_{i}_{j}")

    for i in nodes:
        for j in nodes:
            if i != j and i != root and j != root:
                model.addConstr(u[i] - u[j] + n * y[i, j] <= n - 1, f"mtz_{i}_{j}")

    model.optimize()

    if model.SolCount == 0:
        return {
            "status": "NO_SOLUTION", "objective": None, "arcs": [], "tour": [],
            "length": None, "tour_cost": None, "reservation_paid": None,
            "penalty_paid": None, "total_cost": None,
            "x_used": [], "x_not_used": list(I),
            "reserved_used_directed": [], "reserved_not_used": [], "used_unreserved_directed": [],
        }

    arcs = [(i, j) for (i, j) in E if y[i, j].X > 0.5]
    ordered_tour = extract_tour_from_arcs(arcs, root)
    tour_cost = sum(dist[i][j] for (i, j) in arcs)

    x_used = [(i, j) for (i, j) in I if x[i, j].X > 0.5]
    x_not_used = [(i, j) for (i, j) in I if x[i, j].X < 0.5]
    reservation_paid = sum(get_edge_value(p, i, j) for (i, j) in x_used)
    penalty_paid = sum(get_edge_value(C, i, j) for (i, j) in I if z[i, j].X > 0.5)

    reserved_used_directed = []
    reserved_not_used = []
    used_unreserved_directed = []
    for (i, j) in I:
        used_dir = None
        if y[i, j].X > 0.5:
            used_dir = (i, j)
        elif y[j, i].X > 0.5:
            used_dir = (j, i)

        if x[i, j].X > 0.5 and used_dir is not None:
            reserved_used_directed.append(used_dir)
        elif x[i, j].X > 0.5 and used_dir is None:
            reserved_not_used.append((i, j))
        elif x[i, j].X < 0.5 and used_dir is not None:
            used_unreserved_directed.append(used_dir)

    status_map = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INFEASIBLE: "INFEASIBLE"}
    total_cost = tour_cost + reservation_paid + penalty_paid
    return {
        "status": status_map.get(model.Status, str(model.Status)),
        "objective": model.ObjVal,
        "arcs": arcs,
        "tour": ordered_tour,
        "length": model.ObjVal,
        "tour_cost": tour_cost,
        "reservation_paid": reservation_paid,
        "penalty_paid": penalty_paid,
        "total_cost": total_cost,
        "x_used": x_used,
        "x_not_used": x_not_used,
        "reserved_used_directed": reserved_used_directed,
        "reserved_not_used": reserved_not_used,
        "used_unreserved_directed": used_unreserved_directed,
    }



# MODELLO STOCASTICO A DUE STADI

# Serve se MIPGap o ObjBound non sono disponibili
def safe_gurobi_attr(model, attr_name, default=None):
    try:
        return getattr(model, attr_name)
    except Exception:
        return default
    
# Risolvo lo STO proposto.
# x non dipende dallo scenario: le tratte prenotate dallo stocastico sono quindi identiche
# in tutti gli scenari. y invece è una decisione di ricorso e può cambiare per scenario.
def solve_stochastic(nodes, E, I, b, root, p, C, env, scenario_deltas, scenario_probs=None, force_important=False):

    scenarios = list(scenario_deltas.keys())
    if scenario_probs is None:
        scenario_probs = {s: 1.0 / len(scenarios) for s in scenarios}

    model = gp.Model("stochastic_2stage", env=env)
    model.Params.OutputFlag = 1
    model.Params.TimeLimit  = STO_TIME_LIMIT
    model.Params.MIPGap     = STO_MIP_GAP
    model.Params.Threads = 1
    model.Params.Seed = 42
    n = len(nodes)

    # x = prenotazione, unica per tutti gli scenari.
    # y = arco percorso nello scenario s.
    # z = attivatore della multa nello scenario s.
    x = model.addVars(I, vtype=GRB.BINARY, name="x_prenota")
    y = model.addVars(scenarios, E, vtype=GRB.BINARY, name="y_percorri")
    z = model.addVars(scenarios, I, vtype=GRB.BINARY, name="z_multa")
    u = model.addVars(scenarios, nodes, lb=0, ub=n - 1, vtype=GRB.CONTINUOUS, name="u")

    costo_prenotazione = gp.quicksum(get_edge_value(p, i, j) * x[i, j] for (i, j) in I)
    costo_atteso_ricorso = gp.quicksum(
        scenario_probs[s] * (
            gp.quicksum(
                (b[i][j] + scenario_deltas[s].get((i, j), 0.0)) * y[s, i, j]
                for (i, j) in E
            )
            + gp.quicksum(get_edge_value(C, i, j) * z[s, i, j] for (i, j) in I)
        )
        for s in scenarios
    )
    model.setObjective(costo_prenotazione + costo_atteso_ricorso, GRB.MINIMIZE)

    for s in scenarios:
        for i in nodes:
            model.addConstr(gp.quicksum(y[s, i, j] for j in nodes if j != i) == 1, f"out_{s}_{i}")
        for j in nodes:
            model.addConstr(gp.quicksum(y[s, i, j] for i in nodes if i != j) == 1, f"in_{s}_{j}")

        for (i, j) in I:
            used_ij = y[s, i, j] + y[s, j, i]
            # z[s,i,j] = (1 - x[i,j]) * used_ij
            model.addConstr(z[s, i, j] <= used_ij, f"z_le_used_{s}_{i}_{j}")
            model.addConstr(z[s, i, j] <= 1 - x[i, j], f"z_le_not_reserved_{s}_{i}_{j}")
            model.addConstr(z[s, i, j] >= used_ij - x[i, j], f"z_ge_used_minus_reserved_{s}_{i}_{j}")

        for i in nodes:
            for j in nodes:
                if i != j and i != root and j != root:
                    model.addConstr(u[s, i] - u[s, j] + n * y[s, i, j] <= n - 1, f"mtz_{s}_{i}_{j}")

    if force_important:
        for (i, j) in I:
            model.addConstr(x[i, j] == 1, f"force_prenotazione_{i}_{j}")

    model.optimize()

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }

    solver_info = {
        "status": status_map.get(model.Status, str(model.Status)),
        "status_code": model.Status,
        "sol_count": model.SolCount,
        "objective": model.ObjVal if model.SolCount > 0 else None,
        "obj_bound": safe_gurobi_attr(model, "ObjBound", None),
        "mip_gap": safe_gurobi_attr(model, "MIPGap", None),
        "runtime": safe_gurobi_attr(model, "Runtime", None),
        "node_count": safe_gurobi_attr(model, "NodeCount", None),
    }

    if model.SolCount == 0:
        return {
            "status": solver_info["status"],
            "objective": None,
            "solver_info": solver_info,
            "x_used": [],
            "x_not_used": list(I),
            "scenario_solutions": {},
        }

    x_used = [(i, j) for (i, j) in I if x[i, j].X > 0.5]
    x_not_used = [(i, j) for (i, j) in I if x[i, j].X < 0.5]
    reservation_paid = sum(get_edge_value(p, i, j) for (i, j) in x_used)

    print("\nArchi prenotati dallo stocastico, comuni a tutti gli scenari:")
    if x_used:
        for (i, j) in x_used:
            print(f"  {{{i},{j}}} | p={get_edge_value(p, i, j):.4f}")
    else:
        print("  nessuna tratta prenotata")

    scenario_solutions = {}
    for s in scenarios:
        y_used = [(i, j) for (i, j) in E if y[s, i, j].X > 0.5]
        ordered_tour = extract_tour_from_arcs(y_used, root)
        tour_cost = sum(b[i][j] + scenario_deltas[s].get((i, j), 0.0) for (i, j) in y_used)
        penalty_paid = sum(get_edge_value(C, i, j) for (i, j) in I if z[s, i, j].X > 0.5)

        reserved_used_directed = []
        reserved_not_used = []
        used_unreserved_directed = []
        for (i, j) in I:
            used_dir = None
            if y[s, i, j].X > 0.5:
                used_dir = (i, j)
            elif y[s, j, i].X > 0.5:
                used_dir = (j, i)

            if x[i, j].X > 0.5 and used_dir is not None:
                reserved_used_directed.append(used_dir)
            elif x[i, j].X > 0.5 and used_dir is None:
                reserved_not_used.append((i, j))
            elif x[i, j].X < 0.5 and used_dir is not None:
                used_unreserved_directed.append(used_dir)

        scenario_solutions[s] = {
            "y_used": y_used,
            "tour": ordered_tour,
            "tour_cost": tour_cost,
            "reservation_paid": reservation_paid,
            "penalty_paid": penalty_paid,
            "total_cost": tour_cost + reservation_paid + penalty_paid,
            "x_used": x_used,
            "x_not_used": x_not_used,
            "reserved_used_directed": reserved_used_directed,
            "reserved_not_used": reserved_not_used,
            "used_unreserved_directed": used_unreserved_directed,
        }

    return {
        "status": solver_info["status"],
        "objective": solver_info["objective"],
        "solver_info": solver_info,
        "x_used": x_used,
        "x_not_used": x_not_used,
        "reservation_paid": reservation_paid,
        "scenario_solutions": scenario_solutions,
    }



# ─────────────────────────────────────────────────────────────────────────────
# VALIDAZIONE OUT-OF-SAMPLE
# Applica le politiche x_sto e x_ev trovate in training su scenari indipendenti
# (generati con VALIDATION_SEED, stessa distribuzione).
# Per ogni scenario di validazione: PI libero, costo con x_sto fisso + y ottimale,
# costo con x_ev fisso + y ottimale.
# ─────────────────────────────────────────────────────────────────────────────

def validate_policies(
    nodes, E, base_dist, root, env, I, p, C,
    x_sto, x_ev,
    frequent_arcs, n_validation_scenarios,
    n_extra_arcs, mean_frac, sigma_frac,
    exp_name=""
):
    scenario_ids_val = list(range(1, n_validation_scenarios + 1))
    reservation_sto  = sum(get_edge_value(p, i, j) for (i, j) in x_sto)
    reservation_ev   = sum(get_edge_value(p, i, j) for (i, j) in x_ev)

    print("\n" + "=" * 60)
    print("VALIDAZIONE OUT-OF-SAMPLE")
    print(f"  Scenari: {n_validation_scenarios} | seme: VALIDATION_SEED={VALIDATION_SEED}")
    print(f"  Politica STO: {sorted(x_sto)} ({len(x_sto)} archi prenotati)")
    print(f"  Politica EEV: {sorted(x_ev)} ({len(x_ev)} archi prenotati)")

    # Genera scenari di validazione indipendenti
    results_val, _, _ = generate_scenarios(
        scenario_ids_val, nodes, E, base_dist, I, frequent_arcs,
        n_extra_arcs, mean_frac, sigma_frac, VALIDATION_SEED,
        root=root, env=env, p=p, C=C
    )

    pi_costs  = {}
    sto_costs = {}; sto_tc = {}; sto_pc = {}
    eev_costs = {}; eev_tc = {}; eev_pc = {}
    sto_solutions_val = {}
    eev_solutions_val = {}  

    for sid in scenario_ids_val:
        sd = results_val[sid]["scenario_dist"]
        pi_costs[sid] = results_val[sid]["exact_free"]["length"]

        # Applica x_sto: fissa le prenotazioni, ottimizza y
        r_sto = solve_reservation_tsp(
            nodes, E, I, sd, root, p, C, env,
            fixed_reservations=list(x_sto), output_flag=0,
            model_name=f"val_sto_{sid}"
        )
        sto_tc[sid]    = r_sto["tour_cost"]    if r_sto["tour_cost"]    is not None else 0.0
        sto_pc[sid]    = r_sto["penalty_paid"] if r_sto["penalty_paid"] is not None else 0.0
        sto_costs[sid] = reservation_sto + sto_tc[sid] + sto_pc[sid]
        sto_solutions_val[sid] = r_sto

        # Applica x_ev: fissa le prenotazioni, ottimizza y
        r_ev = solve_reservation_tsp(
            nodes, E, I, sd, root, p, C, env,
            fixed_reservations=list(x_ev), output_flag=0,
            model_name=f"val_ev_{sid}"
        )
        eev_tc[sid]    = r_ev["tour_cost"]    if r_ev["tour_cost"]    is not None else 0.0
        eev_pc[sid]    = r_ev["penalty_paid"] if r_ev["penalty_paid"] is not None else 0.0
        eev_costs[sid] = reservation_ev + eev_tc[sid] + eev_pc[sid]
        eev_solutions_val[sid] = r_ev

    n       = len(scenario_ids_val)
    PI_val  = sum(v for v in pi_costs.values()  if v is not None) / n
    STO_val = sum(sto_costs.values()) / n
    EEV_val = sum(eev_costs.values()) / n
    VSS_val = EEV_val - STO_val
    gap_pi_sto_val = (STO_val - PI_val) / abs(PI_val) * 100 if PI_val else float("nan")
    random_impact_val = compute_random_edge_usage_stats(
    results_val,
    scenario_ids_val,
    stoch_solutions=sto_solutions_val,
    eev_solutions=eev_solutions_val,
    )

    random_impact_lines = format_random_edge_usage_lines(
        random_impact_val,
        "IMPATTO ARCHI CASUALI - VALIDAZIONE"
    )

    print("\n  DETTAGLIO PER SCENARIO DI VALIDAZIONE")
    print(f"  {'Scen':>4} | {'PI':>10} | {'STO_val':>10} [perc, multa] | {'EEV_val':>10} [perc, multa]")
    for sid in scenario_ids_val:
        pi_s = f"{pi_costs[sid]:.4f}" if pi_costs[sid] is not None else "N/A"
        print(f"  {sid:>4} | {pi_s:>10} | {sto_costs[sid]:>10.4f} "
              f"[{sto_tc[sid]:.4f}, {sto_pc[sid]:.4f}] | "
              f"{eev_costs[sid]:>10.4f} [{eev_tc[sid]:.4f}, {eev_pc[sid]:.4f}]")

    print(f"\n  MEDIE VALIDAZIONE OUT-OF-SAMPLE")
    print(f"  PI_val  = {PI_val:.4f}")
    print(f"  STO_val = {STO_val:.4f}  (politica x_sto applicata out-of-sample)")
    print(f"  EEV_val = {EEV_val:.4f}  (politica x_ev  applicata out-of-sample)")
    print(f"  VSS_val = EEV_val - STO_val = {VSS_val:.4f} ({VSS_val / abs(EEV_val) * 100:.4f}%)")
    print(f"  Gap PI-STO_val = {gap_pi_sto_val:.4f}%")
    print("=" * 60)
    print("\n".join(random_impact_lines))
    # Appende i risultati di validazione al txt dell'esperimento
    if exp_name:
        val_lines = [
            "",
            "=" * 60,
            "VALIDAZIONE OUT-OF-SAMPLE",
            f"  Scenari: {n_validation_scenarios} | seme: VALIDATION_SEED={VALIDATION_SEED}",
            f"  Politica STO: {sorted(x_sto)} ({len(x_sto)} archi prenotati)",
            f"  Politica EEV: {sorted(x_ev)} ({len(x_ev)} archi prenotati)",
            "",
            f"  {'Scen':>4} | {'PI':>10} | {'STO_val':>10} [perc, multa] | {'EEV_val':>10} [perc, multa]",
        ]
        for sid in scenario_ids_val:
            pi_s = f"{pi_costs[sid]:.4f}" if pi_costs[sid] is not None else "N/A"
            val_lines.append(
                f"  {sid:>4} | {pi_s:>10} | {sto_costs[sid]:>10.4f} "
                f"[{sto_tc[sid]:.4f}, {sto_pc[sid]:.4f}] | "
                f"{eev_costs[sid]:>10.4f} [{eev_tc[sid]:.4f}, {eev_pc[sid]:.4f}]"
            )
        val_lines += [
            "",
            "  MEDIE VALIDAZIONE OUT-OF-SAMPLE",
            f"  PI_val  = {PI_val:.4f}",
            f"  STO_val = {STO_val:.4f}  (politica x_sto applicata out-of-sample)",
            f"  EEV_val = {EEV_val:.4f}  (politica x_ev  applicata out-of-sample)",
            f"  VSS_val = EEV_val - STO_val = {VSS_val:.4f} ({VSS_val / abs(EEV_val) * 100:.4f}%)",
            f"  Gap PI-STO_val = {gap_pi_sto_val:.4f}%",
            
            "=" * 60,
        ]
        val_lines += random_impact_lines
        fname = out_path(f"risultati_{exp_name}.txt")
        with open(fname, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(val_lines) + "\n")

    return {
        "PI_val":      PI_val,
        "STO_val":     STO_val,
        "EEV_val":     EEV_val,
        "VSS_val":     VSS_val,
        "pi_costs":    pi_costs,
        "sto_costs":   sto_costs,
        "eev_costs":   eev_costs,
        "results_val": results_val,
    }


# CALCOLO PI / EEV / STO  (parte condivisa)

# Trova gli archi ad alta frequenza nei PI di calibrazione.
# Questi vengono aggiunti sistematicamente all'insieme degli archi perturbati.
def find_frequent_arcs(
    nodes, E, base_dist, root, env, I,
    n_calibration_scenarios=None,
    min_freq=None,
    n_frequent=None,
    calibration_seed=None,
    n_extra_arcs=None,
    mean_frac=None,
    sigma_frac=None,
):
    if n_calibration_scenarios is None: n_calibration_scenarios = N_CALIBRATION_SCENARIOS
    if min_freq                is None: min_freq                = MIN_FREQ_FREQUENT
    if n_frequent              is None: n_frequent              = N_FREQUENT_ARCS
    if calibration_seed        is None: calibration_seed        = CALIBRATION_SCENARIO_SEED
    if n_extra_arcs            is None: n_extra_arcs            = N_EXTRA_ARCS
    if mean_frac               is None: mean_frac               = MEAN_FRAC
    if sigma_frac              is None: sigma_frac              = SIGMA_FRAC

    I_set = {canon_edge(i, j) for (i, j) in I}
    usage_counter = Counter()
    n_valid = 0

    print(f"\nCALIBRAZIONE ARCHI FREQUENTI "
          f"({n_calibration_scenarios} scenari, seme={calibration_seed}):")

    for sid in range(1, n_calibration_scenarios + 1):
        pert = build_perturbation(
            sid, nodes, base_dist, I,
            n_extra_arcs, mean_frac, sigma_frac,
            calibration_seed
        )
        scenario_dist = build_scenario_dist(base_dist, pert)
        pi_res = solve_exact_tsp(
            nodes, E, scenario_dist, root, env,
            fixed_arcs=[], fixed_edges_undir=[], output_flag=0
        )
        if pi_res["length"] is None:
            print(f"  Scenario calibrazione {sid:02d}: PI non trovato, saltato")
            continue

        n_valid += 1
        for arc in pi_res["arcs"]:
            usage_counter[canon_edge(*arc)] += 1

    if n_valid == 0:
        print("  Nessun PI valido: ritorno lista vuota.")
        return []

    freq_info = []
    for edge in all_undirected_edges(nodes):
        if edge in I_set:
            continue
        freq = usage_counter[edge] / n_valid
        freq_info.append({
            "edge" : edge,
            "count": usage_counter[edge],
            "freq" : freq,
            "cost" : base_cost_undirected(base_dist, edge[0], edge[1]),
        })

    candidates = [x for x in freq_info if x["freq"] >= min_freq]
    candidates.sort(key=lambda x: (-x["freq"], x["cost"]))
    selected = candidates[:n_frequent]

    print(f"  Scenari validi: {n_valid}/{n_calibration_scenarios}")
    print(f"  Archi frequenti selezionati (freq >= {min_freq:.0%}): {len(selected)}")
    for item in selected:
        print(f"    {item['edge']}  freq={item['freq']:.2%}"
              f"  ({item['count']}/{n_valid})  costo={item['cost']:.4f}")
    if len(selected) == 0:
        print("  Nessun arco supera la soglia: considera di abbassare MIN_FREQ_FREQUENT.")

    return [item["edge"] for item in selected]


# Genera scenari con archi perturbati = I ∪ frequent_arcs ∪ random_extra.
# Sostituisce run_scenarios. Usare base_seed=FINAL_SCENARIO_SEED per training,
# base_seed=VALIDATION_SEED per validazione (stessa distribuzione, seme diverso).
# Ritorna (results, scenario_probs, total_random_uses) dove total_random_uses è
# il numero aggregato di inserimenti di archi random su tutti gli scenari.
def generate_scenarios(
    scenario_ids, nodes, E, base_dist, I, frequent_arcs,
    n_extra_arcs, mean_frac, sigma_frac, base_seed,
    root, env, p, C
):
    scenario_probs = {s: 1.0 / len(scenario_ids) for s in scenario_ids}
    results = {}
    I_set    = {canon_edge(i, j) for (i, j) in I}
    freq_set = {canon_edge(i, j) for (i, j) in frequent_arcs}
    total_random_uses = 0

    for scenario_id in scenario_ids:
        pert = build_perturbation(
            scenario_id, nodes, base_dist, I,
            n_extra_arcs, mean_frac, sigma_frac, base_seed,
            frequent_arcs=frequent_arcs
        )
        # Archi perturbati nel dizionario → forme non orientate canoniche
        perturbed_undir = {canon_edge(i, j) for (i, j) in pert.keys()}
        random_edges    = perturbed_undir - I_set - freq_set
        total_random_uses += len(random_edges)

        scenario_dist = build_scenario_dist(base_dist, pert)

        exact_free = solve_exact_tsp(
            nodes, E, scenario_dist, root, env,
            fixed_arcs=[], fixed_edges_undir=[], output_flag=0
        )

        results[scenario_id] = {
            "pert"         : pert,
            "scenario_dist": scenario_dist,
            "exact_free"   : exact_free,
            "random_edges" : sorted(random_edges),
        }

        pi_str = (f"{exact_free['length']:.4f}"
                  if exact_free["length"] is not None else "N/A")
        print(f"  Scenario {scenario_id} | PI = {pi_str}")

    return results, scenario_probs, total_random_uses

def _solution_arcs_undir(solution):
    if not solution:
        return set()

    arcs = solution.get("arcs", None)
    if arcs is None:
        arcs = solution.get("y_used", [])

    return {canon_edge(i, j) for (i, j) in arcs}


def compute_random_edge_usage_stats(results, scenario_ids, stoch_solutions=None, eev_solutions=None):
    """
    Conta quanti archi casuali perturbati compaiono nei tour PI/STO/EEV.

    Un arco conta solo se:
    1. è stato estratto come arco casuale nello scenario s;
    2. compare nel tour PI, STO o EEV dello stesso scenario s.
    """
    stoch_solutions = stoch_solutions or {}
    eev_solutions = eev_solutions or {}

    policy_sources = {
        "PI":  {sid: results[sid].get("exact_free", {}) for sid in scenario_ids},
        "STO": stoch_solutions,
        "EEV": eev_solutions,
    }

    stats = {
        "n_scenarios": len(scenario_ids),
        "random_occurrences": 0,
        "random_unique": set(),
        "random_by_scenario": {},
        "by_policy": {},
        "any_policy": {
            "used_occurrences": 0,
            "used_unique": set(),
            "by_scenario": {},
        },
    }

    for policy in policy_sources:
        stats["by_policy"][policy] = {
            "used_occurrences": 0,
            "used_unique": set(),
            "by_scenario": {},
        }

    for sid in scenario_ids:
        random_edges = {canon_edge(i, j) for (i, j) in results[sid].get("random_edges", [])}

        stats["random_occurrences"] += len(random_edges)
        stats["random_unique"].update(random_edges)
        stats["random_by_scenario"][sid] = len(random_edges)

        used_any_sid = set()

        for policy, source in policy_sources.items():
            tour_edges = _solution_arcs_undir(source.get(sid, {}))
            used_sid = random_edges & tour_edges

            stats["by_policy"][policy]["used_occurrences"] += len(used_sid)
            stats["by_policy"][policy]["used_unique"].update(used_sid)
            stats["by_policy"][policy]["by_scenario"][sid] = used_sid

            used_any_sid.update(used_sid)

        stats["any_policy"]["used_occurrences"] += len(used_any_sid)
        stats["any_policy"]["used_unique"].update(used_any_sid)
        stats["any_policy"]["by_scenario"][sid] = used_any_sid

    return stats


def format_random_edge_usage_lines(stats, title):
    if not stats:
        return []

    n_s = stats["n_scenarios"]
    random_occ = stats["random_occurrences"]
    random_unique_n = len(stats["random_unique"])

    def pct(num, den):
        return 100.0 * num / den if den else 0.0

    out = [
        "",
        title,
        f"  Scenari analizzati = {n_s}",
        f"  Estrazioni casuali totali scenario-specifiche = {random_occ}",
        f"  Archi casuali distinti = {random_unique_n}",
        "",
        "  Conteggio degli archi casuali che compaiono nei tour",
    ]

    for policy in ["PI", "STO", "EEV"]:
        item = stats["by_policy"][policy]
        used_occ = item["used_occurrences"]
        used_unique_n = len(item["used_unique"])

        out.append(
            f"  {policy:<3}: distinti usati almeno una volta = {used_unique_n}/{random_unique_n} "
            f"({pct(used_unique_n, random_unique_n):.2f}%) | "
            f"occorrenze scenario-specifiche = {used_occ}/{random_occ} "
            f"({pct(used_occ, random_occ):.2f}%)"
        )

    any_item = stats["any_policy"]
    any_occ = any_item["used_occurrences"]
    any_unique_n = len(any_item["used_unique"])

    out.append(
        f"  ANY: distinti usati in almeno un tour PI/STO/EEV = {any_unique_n}/{random_unique_n} "
        f"({pct(any_unique_n, random_unique_n):.2f}%) | "
        f"occorrenze scenario-specifiche = {any_occ}/{random_occ} "
        f"({pct(any_occ, random_occ):.2f}%)"
    )

    out += [
        "",
        "  Dettaglio per scenario: sid | casuali | PI | STO | EEV | ANY",
    ]

    for sid in sorted(stats["any_policy"]["by_scenario"]):
        pi_n  = len(stats["by_policy"]["PI"]["by_scenario"].get(sid, set()))
        sto_n = len(stats["by_policy"]["STO"]["by_scenario"].get(sid, set()))
        eev_n = len(stats["by_policy"]["EEV"]["by_scenario"].get(sid, set()))
        any_n = len(stats["any_policy"]["by_scenario"].get(sid, set()))

        rand_sid_n = stats.get("random_by_scenario", {}).get(sid, None)
        rand_s = str(rand_sid_n) if rand_sid_n is not None else "N/A"

        out.append(
            f"  {sid:>4} | {rand_s:>7} | {pi_n:>2} | {sto_n:>3} | {eev_n:>3} | {any_n:>3}"
        )

    return out

# Creazione del medione. come funziona:
# - Calcolo il costo medio di perturbazione dei 4 scenari sugli archi
# - Genero un tour ottimale 
# - Guardo quali archi in I ha usato e li estrapolo
# - Tengo quegli archi fissati e sui 4 scenari calcolo WS sui singoli scenari
# N.B. il costo degl archi fissi è il costo medio, mentre il costo degli archi che poi vengono scelti negli scenari singoli 
# è il costo perturbato in quello scenario

def compute_eev_medione(nodes, E, root, env, base_dist, I, p, C, results, scenario_ids, return_solutions=False):
    # Step 1: distanza media sui 4 scenari
    all_deltas = [results[s]["pert"] for s in scenario_ids]
    dist_media = {
        i: {
            j: base_dist[i][j] + sum(d.get((i, j), 0.0) for d in all_deltas) / len(all_deltas)
            for j in base_dist[i]
        }
        for i in base_dist
    }

    # Step 2: problema deterministico sul medione.
    # Da qui estraggo x^EV, cioè le prenotazioni decise con informazione media.
    mean_solution = solve_reservation_tsp(
        nodes, E, I, dist_media, root, p, C, env,
        fixed_reservations=None, output_flag=0,
        model_name="eev_medione"
    )

    tour_medio = mean_solution["tour"]
    arcs_medio = mean_solution["arcs"]
    x_ev = set(mean_solution["x_used"])

    # Step 3: valutazione EEV vera.
    # Fisso x^EV, poi per ogni scenario riottimizzo il tour y sui costi reali dello scenario.
    eev_costs = {}
    eev_tour_costs = {}
    eev_penalty_costs = {}
    eev_solutions = {}

    reservation_paid = sum(get_edge_value(p, i, j) for (i, j) in x_ev)

    for sid in scenario_ids:
        scenario_dist = results[sid]["scenario_dist"]

        second_stage = solve_reservation_tsp(
            nodes, E, I, scenario_dist, root, p, C, env,
            fixed_reservations=list(x_ev), output_flag=0,
            model_name=f"eev_secondostadio_{sid}"
        )

        if second_stage["tour_cost"] is not None:
            tour_c = second_stage["tour_cost"]
            penalty_c = second_stage["penalty_paid"]
            arcs_ev = second_stage["arcs"]
            tour_ev = second_stage["tour"]
            reserved_used_directed = second_stage.get("reserved_used_directed", [])
            reserved_not_used = second_stage.get("reserved_not_used", [])
            used_unreserved_directed = second_stage.get("used_unreserved_directed", [])
        else:
            # Fallback prudente: non dovrebbe servire se il modello è corretto.
            tour_c = sum(scenario_dist[i][j] for (i, j) in arcs_medio)
            penalty_c = 0.0
            arcs_ev = arcs_medio
            tour_ev = tour_medio
            reserved_used_directed = []
            reserved_not_used = list(x_ev)
            used_unreserved_directed = []

        total_c = reservation_paid + tour_c + penalty_c

        eev_costs[sid] = total_c
        eev_tour_costs[sid] = tour_c
        eev_penalty_costs[sid] = penalty_c

        eev_solutions[sid] = {
            "arcs": arcs_ev,
            "y_used": arcs_ev,
            "tour": tour_ev,
            "tour_cost": tour_c,
            "reservation_paid": reservation_paid,
            "penalty_paid": penalty_c,
            "total_cost": total_c,
            "x_used": sorted(x_ev),
            "reserved_used_directed": reserved_used_directed,
            "reserved_not_used": reserved_not_used,
            "used_unreserved_directed": used_unreserved_directed,
        }

    EEV = sum(eev_costs.values()) / len(eev_costs)
    PI = sum(
        results[s]["exact_free"]["length"]
        for s in scenario_ids
        if results[s]["exact_free"]["length"] is not None
    ) / len(scenario_ids)

    print("\nControllo EEV:")
    if x_ev:
        print("  x^EV, tratte prenotate sul medione:", sorted(x_ev))
    else:
        print("  x^EV: nessuna tratta prenotata sul medione")

    for sid in scenario_ids:
        red_edges = [a for a in eev_solutions[sid]["arcs"] if canon_edge(*a) in x_ev]
        print(
            f"  Scenario {sid}: EEV={eev_costs[sid]:.6f} | "
            f"percorrenza={eev_tour_costs[sid]:.6f} | "
            f"prenotazione={reservation_paid:.6f} | "
            f"multa={eev_penalty_costs[sid]:.6f} | "
            f"archi prenotati usati={red_edges}"
        )

    if return_solutions:
        return tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs, eev_penalty_costs, eev_solutions, EEV, PI

    # Mantengo la vecchia interfaccia per gli altri esperimenti.
    return tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs, eev_penalty_costs, EEV, PI

# salvataggio e stampa del txt
def print_and_save_summary(
        exp_name, scenario_ids, results,
        eev_costs, eev_tour_costs, eev_penalty_costs,
        stoch_costs, stoch_tour_costs, stoch_penalty_costs,
        PI, STO, EEV,
        I=None, kmedoids_info=None, final_pi_counts=None,
        penalty_weights=None, N_CALIB=None,
        p=None, C=None, b=None,
        stoch_solver_info=None,
        frequent_arcs=None, total_random_uses=None,
        random_impact_stats=None,

        # NN-E / NNE
        nne_costs=None,
        nne_tour_costs=None,
        nne_penalty_costs=None,
        nne_metrics=None,          # output di evaluate_nne_model(...)
        nne_history=None,          # history restituito da train_nne_model(...)
        nne_extra=None,            # dizionario con x_nne, obj_surr, reservation_nne, ecc.
        scenario_probs=None):      # pesi degli scenari, se vuoi media pesata

    def fmt(x, nd=4):
        if x is None:
            return "N/A"
        try:
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return "N/A"
            return f"{x:.{nd}f}"
        except Exception:
            return str(x)

    def fmt_pct(x, nd=4):
        if x is None:
            return "N/A"
        try:
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return "N/A"
            return f"{100 * x:.{nd}f}%"
        except Exception:
            return str(x)

    def media_scenari(values):
        if values is None:
            return None

        valid_sids = [
            sid for sid in scenario_ids
            if sid in values
            and values[sid] is not None
            and not (isinstance(values[sid], float) and (math.isnan(values[sid]) or math.isinf(values[sid])))
        ]

        if not valid_sids:
            return None

        if scenario_probs is not None:
            den = sum(scenario_probs.get(sid, 0.0) for sid in valid_sids)
            if den == 0:
                return None
            return sum(scenario_probs.get(sid, 0.0) * values[sid] for sid in valid_sids) / den

        return sum(values[sid] for sid in valid_sids) / len(valid_sids)

    lines = ["RIASSUNTO PER SCENARIO"]

    for sid in scenario_ids:
        fc = results[sid]["exact_free"]["length"]

        ec = eev_costs.get(sid, 0.0)
        et = eev_tour_costs.get(sid, 0.0)
        ep = eev_penalty_costs.get(sid, 0.0)
        er = ec - et - ep

        sc = stoch_costs.get(sid, 0.0)
        tc = stoch_tour_costs.get(sid, 0.0)
        pc = stoch_penalty_costs.get(sid, 0.0)
        rc = sc - tc - pc

        fc_s = fmt(fc)

        line = (
            f"Scenario {sid} | PI: {fc_s} | "
            f"STO: {fmt(sc)} [perc={fmt(tc)}, pren={fmt(rc)}, multa={fmt(pc)}] | "
            f"EEV: {fmt(ec)} [perc={fmt(et)}, pren={fmt(er)}, multa={fmt(ep)}]"
        )

        if nne_costs is not None and sid in nne_costs:
            nc = nne_costs.get(sid, 0.0)
            ntc = (nne_tour_costs or {}).get(sid, 0.0)
            npc = (nne_penalty_costs or {}).get(sid, 0.0)
            nrc = nc - ntc - npc

            line += (
                f" | NNE: {fmt(nc)} "
                f"[perc={fmt(ntc)}, pren={fmt(nrc)}, multa={fmt(npc)}]"
            )

        lines.append(line)

    lines += [
        "",
        "BENCHMARK MEDI SUI SCENARI",
        f"PI: {fmt(PI)}",
        f"STO: {fmt(STO)}",
        f"EEV: {fmt(EEV)}",
    ]

    NNE_val = None
    if nne_costs is not None:
        NNE_val = media_scenari(nne_costs)
        lines.append(f"NNE: {fmt(NNE_val)}")

    if stoch_solver_info is not None:
        gap = stoch_solver_info.get("mip_gap", None)
        bound = stoch_solver_info.get("obj_bound", None)
        runtime = stoch_solver_info.get("runtime", None)
        node_count = stoch_solver_info.get("node_count", None)

        lines += [
            "",
            "DIAGNOSTICA SOLVER STO GUROBI",
            f"Status        : {stoch_solver_info.get('status', 'N/A')}",
            f"Soluzioni     : {stoch_solver_info.get('sol_count', 'N/A')}",
            f"ObjVal        : {fmt(STO)}" if STO is not None else "ObjVal        : N/A",
            f"ObjBound      : {fmt(bound)}",
            f"MIPGap        : {fmt_pct(gap)}",
            f"Runtime       : {fmt(runtime, 2)}s" if runtime is not None else "Runtime       : N/A",
            f"Nodi esplorati: {fmt(node_count, 0)}" if node_count is not None else "Nodi esplorati: N/A",
        ]

    VSS_abs = EEV - STO
    gap_VSS = VSS_abs / abs(EEV) if EEV != 0 else float("nan")
    gap_PI_STO = (STO - PI) / abs(PI) if PI != 0 else float("nan")

    lines += [
        "",
        "GAP",
        f"VSS = EEV - STO = {fmt(VSS_abs)} ({fmt_pct(gap_VSS)})",
        f"Distanza diagnostica PI-STO = {fmt_pct(gap_PI_STO)}",
    ]

    if NNE_val is not None:
        gap_nne_sto = (NNE_val - STO) / abs(STO) if STO != 0 else float("nan")
        gap_nne_eev = (NNE_val - EEV) / abs(EEV) if EEV != 0 else float("nan")

        lines += [
            f"Gap NNE vs STO = {fmt_pct(gap_nne_sto)}",
            f"Gap NNE vs EEV = {fmt_pct(gap_nne_eev)}",
        ]

    if nne_extra is not None:
        lines += [
            "",
            "RIEPILOGO NN-E",
        ]

        if "x_nne" in nne_extra:
            x_nne = nne_extra["x_nne"]
            lines.append(f"Tratte prenotate da NN-E: {sorted(x_nne)}")
            lines.append(f"Numero tratte prenotate da NN-E: {len(x_nne)}")

        if "reservation_nne" in nne_extra:
            lines.append(f"Costo prenotazioni NN-E: {fmt(nne_extra['reservation_nne'])}")

        if "obj_surr" in nne_extra:
            lines.append(f"Obiettivo surrogato NN-E: {fmt(nne_extra['obj_surr'])}")

        if NNE_val is not None and "reservation_nne" in nne_extra:
            q_reale = NNE_val - nne_extra["reservation_nne"]
            lines.append(f"E[Q(x, ξ)] reale NN-E: {fmt(q_reale)}")

        if "NNE_val" in nne_extra:
            lines.append(f"Costo totale NN-E atteso: {fmt(nne_extra['NNE_val'])}")

    if nne_metrics is not None or nne_history is not None:
        lines += [
            "",
            "METRICHE DI APPRENDIMENTO NN-E",
        ]

        if nne_history is not None:
            val_mae_hist = nne_history.get("val_mae", [])
            val_mape_hist = nne_history.get("val_mape", [])

            if len(val_mae_hist) > 0:
                best_idx = int(np.argmin(val_mae_hist))
                best_mae = val_mae_hist[best_idx]
                best_mape = val_mape_hist[best_idx] if best_idx < len(val_mape_hist) else None

                lines += [
                    f"Miglior epoca registrata: {best_idx + 1}",
                    f"Miglior val_MAE: {fmt(best_mae)}",
                    f"Miglior val_MAPE: {fmt_pct(best_mape)}",
                ]

        if nne_metrics is not None:
            for name in ("tr", "val"):
                if name in nne_metrics:
                    m = nne_metrics[name]
                    label = "TRAIN" if name == "tr" else "VALIDATION"
                    lines.append(
                        f"{label:<10} | "
                        f"MAE={fmt(m.get('mae'))} | "
                        f"RMSE={fmt(m.get('rmse'))} | "
                        f"MAPE={fmt_pct(m.get('mape'), 3)} | "
                        f"r={fmt(m.get('corr'))}"
                    )

    if I is not None and p is not None and C is not None and b is not None:
        lines += [
            "",
            "TRATTE I, PRENOTAZIONE E MULTA",
        ]

        for (i, j) in I:
            cnt_str = ""

            if kmedoids_info is not None and N_CALIB is not None:
                cnt = kmedoids_info["edge_count"].get((i, j), 0)
                cnt_str = f" | freq_calib={cnt}/{N_CALIB} ({cnt / N_CALIB:.2%})"

            if final_pi_counts is not None:
                cnt_f = final_pi_counts.get((i, j), 0)
                w = (penalty_weights or {}).get((i, j), 0.0)
                cnt_str += (
                    f" | freq_PI_finali={cnt_f}/{len(scenario_ids)}"
                    f" | peso={w:.4f}"
                )

            lines.append(
                f"  {{{i},{j}}}{cnt_str} | "
                f"b={fmt(b[i, j])} | p={fmt(p[i, j])} | C={fmt(C[i, j])}"
            )

    if frequent_arcs is not None:
        lines += [
            "",
            f"ARCHI FREQUENTI AGGIUNTI ALLA PERTURBAZIONE: {len(frequent_arcs)}",
        ]

        for edge in frequent_arcs:
            lines.append(f"  {edge}")

    if total_random_uses is not None:
        n_s = len(scenario_ids)

        lines += [
            "",
            f"USO ARCHI RANDOM (aggregato su {n_s} scenari di training)",
            f"  Totale inserimenti = {total_random_uses}",
            f"  Media per scenario = {total_random_uses / n_s:.1f}",
        ]

    if random_impact_stats is not None:
        lines += format_random_edge_usage_lines(
            random_impact_stats,
            "IMPATTO ARCHI CASUALI - ADDESTRAMENTO"
        )

    output_text = "\n".join(lines)
    print("\n" + output_text)

    fname = out_path(f"risultati_{exp_name}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(f"\n  → Risultati salvati in: {fname}")

    return output_text, VSS_abs



# PENALITÀ FREQUENZIALE  (usato da C)

# Calcolo C la calcolo in base a quante volte compare l arco I
def penalty_weight_from_final_pi_count(count, n_scenarios, min_weight=0.20):
    if n_scenarios <= 1:
        return 0.0
    if count == 0:
        return 0.0
    return min_weight + (1 - min_weight) * (count - 1) / (n_scenarios - 1)

# Costruzione delle penalità (sui PI)
def build_frequency_based_penalties(I, results, scenario_ids, base_dist, penalty_frac):
    n_ref  = len(scenario_ids)
    counts = {edge: 0 for edge in I}
    for scenario_id in scenario_ids:
        pi_edges = {canon_edge(i, j) for (i, j) in results[scenario_id]["exact_free"]["arcs"]}
        for edge in I:
            if edge in pi_edges:
                counts[edge] += 1

    weights = {}
    C = {}
    for (i, j) in I:
        count = counts[(i, j)]
        w = penalty_weight_from_final_pi_count(count, n_ref)
        weights[(i, j)] = w
        C[(i, j)] = penalty_frac * w * base_cost_undirected(base_dist, i, j)
    return C, counts, weights




# AREE PROBLEMATICHE  (usato da D)


def make_circle(cx, cy, radius):
    return {"type": "circle", "cx": cx, "cy": cy, "radius": radius}


def make_rect(x_min, y_min, x_max, y_max):
    return {"type": "rect", "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def make_ellipse(cx, cy, rx, ry):
    return {"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry}


def auto_place_areas(coords, n_areas=3, margin_frac=0.18, radius_frac=0.14):
    """
    Genera aree problematiche in posizioni più utili:
    - cerchio: lasciato nella zona in basso a sinistra;
    - ellisse: spostata in alto a sinistra;
    - rettangolo: spostato in basso a destra.

    Le posizioni sono calcolate come frazioni del rettangolo che contiene tutti i nodi,
    così la funzione resta indipendente dalla scala delle coordinate.
    """
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = x_max - x_min
    h = y_max - y_min
    r = radius_frac * min(w, h)

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # 1. Cerchio: area residua, in basso a sinistra.
    circle_cx = x_min + 0.24 * w
    circle_cy = y_min + 0.24 * h
    circle = make_circle(circle_cx, circle_cy, r)

    # 2. Ellisse: alto a sinistra.
    ellipse_rx = 1.65 * r
    ellipse_ry = 0.95 * r
    ellipse_cx = x_min + 0.23 * w
    ellipse_cy = y_max - 0.18 * h

    ellipse_cx = clamp(ellipse_cx, x_min + ellipse_rx, x_max - ellipse_rx)
    ellipse_cy = clamp(ellipse_cy, y_min + ellipse_ry, y_max - ellipse_ry)

    ellipse = make_ellipse(ellipse_cx, ellipse_cy, ellipse_rx, ellipse_ry)

    # 3. Rettangolo: basso a destra.
    rect_w = 2.30 * r
    rect_h = 1.45 * r
    rect_cx = x_max - 0.23 * w
    rect_cy = y_min + 0.20 * h

    rect_cx = clamp(rect_cx, x_min + rect_w / 2, x_max - rect_w / 2)
    rect_cy = clamp(rect_cy, y_min + rect_h / 2, y_max - rect_h / 2)

    rect = make_rect(
        rect_cx - rect_w / 2,
        rect_cy - rect_h / 2,
        rect_cx + rect_w / 2,
        rect_cy + rect_h / 2
    )

    candidate_areas = [circle, ellipse, rect]

    if n_areas is None:
        return candidate_areas

    if n_areas < 1:
        raise ValueError("n_areas deve essere almeno 1.")

    return candidate_areas[:min(n_areas, len(candidate_areas))]


def format_area(area):
    """Stampa leggibile delle aree, ora rappresentate come dizionari."""
    t = area["type"]

    if t == "circle":
        return f"cerchio: centro=({area['cx']:.2f}, {area['cy']:.2f}), raggio={area['radius']:.2f}"

    if t == "rect":
        return (
            f"rettangolo: x=[{area['x_min']:.2f}, {area['x_max']:.2f}], "
            f"y=[{area['y_min']:.2f}, {area['y_max']:.2f}]"
        )

    if t == "ellipse":
        return (
            f"ellisse: centro=({area['cx']:.2f}, {area['cy']:.2f}), "
            f"rx={area['rx']:.2f}, ry={area['ry']:.2f}"
        )

    return f"area di tipo sconosciuto: {t}"


def point_to_segment_dist(px, py, ax, ay, bx, by):
    """
    Distanza minima tra il punto P=(px,py) e il segmento AB.
    Serve per capire se un arco attraversa un cerchio o un'ellisse.
    """
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy

    if denom == 0:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))

    proj_x = ax + t * dx
    proj_y = ay + t * dy

    return math.hypot(px - proj_x, py - proj_y)


def _orientation(ax, ay, bx, by, cx, cy):
    val = (by - ay) * (cx - bx) - (bx - ax) * (cy - by)
    eps = 1e-12

    if abs(val) < eps:
        return 0

    return 1 if val > 0 else 2


def _on_segment(ax, ay, bx, by, cx, cy):
    eps = 1e-12
    return (
        min(ax, cx) - eps <= bx <= max(ax, cx) + eps and
        min(ay, cy) - eps <= by <= max(ay, cy) + eps
    )


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    o1 = _orientation(ax, ay, bx, by, cx, cy)
    o2 = _orientation(ax, ay, bx, by, dx, dy)
    o3 = _orientation(cx, cy, dx, dy, ax, ay)
    o4 = _orientation(cx, cy, dx, dy, bx, by)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _on_segment(ax, ay, cx, cy, bx, by):
        return True
    if o2 == 0 and _on_segment(ax, ay, dx, dy, bx, by):
        return True
    if o3 == 0 and _on_segment(cx, cy, ax, ay, dx, dy):
        return True
    if o4 == 0 and _on_segment(cx, cy, bx, by, dx, dy):
        return True

    return False


def _segment_crosses_rect(ax, ay, bx, by, area):
    """
    Un segmento attraversa un rettangolo se almeno un estremo è dentro,
    oppure se interseca uno dei quattro lati.
    """
    xmn, xmx = area["x_min"], area["x_max"]
    ymn, ymx = area["y_min"], area["y_max"]

    def inside(x, y):
        return xmn <= x <= xmx and ymn <= y <= ymx

    if inside(ax, ay) or inside(bx, by):
        return True

    rect_edges = [
        (xmn, ymn, xmx, ymn),
        (xmx, ymn, xmx, ymx),
        (xmx, ymx, xmn, ymx),
        (xmn, ymx, xmn, ymn),
    ]

    return any(
        _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy)
        for (cx, cy, dx, dy) in rect_edges
    )


def _segment_crosses_ellipse(ax, ay, bx, by, area):
    """
    Porto l'ellisse a un cerchio unitario tramite riscalamento degli assi.
    Poi verifico se la distanza dal centro trasformato al segmento è <= 1.
    """
    cx, cy, rx, ry = area["cx"], area["cy"], area["rx"], area["ry"]

    ax_s = (ax - cx) / rx
    ay_s = (ay - cy) / ry
    bx_s = (bx - cx) / rx
    by_s = (by - cy) / ry

    return point_to_segment_dist(0.0, 0.0, ax_s, ay_s, bx_s, by_s) <= 1.0


def arc_crosses_area(i, j, coords, area):
    xi, yi = coords[i]
    xj, yj = coords[j]

    t = area["type"]

    if t == "circle":
        return point_to_segment_dist(area["cx"], area["cy"], xi, yi, xj, yj) <= area["radius"]

    if t == "rect":
        return _segment_crosses_rect(xi, yi, xj, yj, area)

    if t == "ellipse":
        return _segment_crosses_ellipse(xi, yi, xj, yj, area)

    raise ValueError(f"Tipo area sconosciuto: {t}")


def arc_crosses_any_area(i, j, coords, areas):
    return any(arc_crosses_area(i, j, coords, area) for area in areas)




def select_I_from_problematic_areas_all(nodes, E, coords, base_dist, areas):
    """
    Restituisce TUTTE le tratte non orientate {i,j} il cui segmento geometrico
    attraversa almeno una delle aree problematiche.

    Questa è la versione coerente con l'interpretazione:
        passaggio in area colorata => arco problematico => prenotazione oppure multa.

    Nota:
    - I viene restituito come insieme/lista di archi non orientati canonici.
    - Si controlla che entrambi gli orientamenti (i,j) e (j,i) siano presenti in E,
      perché il modello usa y[s,i,j] e y[s,j,i].
    """
    I_set = set()

    for i in nodes:
        for j in nodes:
            if i >= j:
                continue

            if (i, j) not in E or (j, i) not in E:
                continue

            if arc_crosses_any_area(i, j, coords, areas):
                I_set.add(canon_edge(i, j))

    I = sorted(I_set, key=lambda e: base_cost_undirected(base_dist, e[0], e[1]))

    if not I:
        raise ValueError(
            "Nessuna tratta attraversa le aree problematiche. "
            "Controlla posizione/dimensione delle aree oppure arc_crosses_any_area."
        )

    return I


from collections import Counter





def select_I_from_problematic_areas_by_pi_frequency(
    nodes, E, coords, base_dist, root, env, areas,
    n_calibration_scenarios=N_CALIBRATION_SCENARIOS,
    min_freq=MIN_FREQ_FOR_CANDIDATE,
    max_freq=MAX_FREQ_FOR_CANDIDATE,
    max_arcs=MAX_AREA_I_ARCS_D,
    n_extra_arcs=N_EXTRA_ARCS,
    mean_frac=MEAN_FRAC,
    sigma_frac=SIGMA_FRAC,
    base_seed=FINAL_SCENARIO_SEED + 10000):
    """
    Seleziona I per l'esperimento D così:

    1. prende tutti gli archi che attraversano le aree problematiche;
    2. genera n_calibration_scenarios scenari di calibrazione;
    3. risolve il PI libero su ogni scenario;
    4. conta quante volte ogni arco candidato viene usato dal PI;
    5. tiene gli archi con frequenza in [min_freq, max_freq];
    6. sceglie al massimo max_arcs archi, preferendo quelli con frequenza vicina a 0.5.

    L'idea è selezionare archi né sempre usati né mai usati.
    """

    I_candidates = select_I_from_problematic_areas_all(
        nodes, E, coords, base_dist, areas
    )

    candidate_set = set(I_candidates)
    usage_counter = Counter()

    calibration_ids = list(range(1, n_calibration_scenarios + 1))

    pi_values = {}

    print("\nCALIBRAZIONE D SU PI LIBERO:")
    print(f"Archi candidati dalle aree problematiche: {len(I_candidates)}")
    print(f"Scenari di calibrazione: {n_calibration_scenarios}")

    for sid in calibration_ids:
        pert = build_perturbation(
            sid,
            nodes,
            base_dist,
            I_candidates,      # importante: perturbo tutti gli archi candidati delle aree
            n_extra_arcs,
            mean_frac,
            sigma_frac,
            base_seed
        )

        scenario_dist = build_scenario_dist(base_dist, pert)

        pi_res = solve_exact_tsp(
            nodes, E, scenario_dist, root, env,
            fixed_arcs=[],
            fixed_edges_undir=[],
            output_flag=0
        )

        if pi_res["length"] is None:
            print(f"  Scenario calibrazione {sid}: PI non trovato")
            continue

        pi_values[sid] = pi_res["length"]

        pi_edges_undir = {
            canon_edge(i, j)
            for (i, j) in pi_res["arcs"]
        }

        for edge in pi_edges_undir:
            if edge in candidate_set:
                usage_counter[edge] += 1

        print(f"  Scenario calibrazione {sid:02d} | PI = {pi_res['length']:.4f}")

    n_valid = len(pi_values)

    if n_valid == 0:
        raise ValueError("Nessuno scenario di calibrazione ha prodotto un PI valido.")

    freq_info = []

    for edge in I_candidates:
        freq = usage_counter[edge] / n_valid
        cost = base_cost_undirected(base_dist, edge[0], edge[1])

        freq_info.append({
            "edge": edge,
            "count": usage_counter[edge],
            "freq": freq,
            "cost": cost,
            "score_uncertainty": freq * (1.0 - freq),
            "distance_from_half": abs(freq - 0.5),
        })

    # Prima selezione: solo archi con frequenza intermedia.
    eligible = [
        item for item in freq_info
        if min_freq <= item["freq"] <= max_freq
    ]

    # Preferisco frequenze vicine a 0.5.
    # A parità, preferisco archi più corti.
    eligible = sorted(
        eligible,
        key=lambda x: (x["distance_from_half"], x["cost"])
    )

    selected = eligible[:max_arcs]

    # Se gli archi in [0.2, 0.8] sono meno di 7, completo con quelli più vicini all'intervallo.
    if len(selected) < max_arcs:
        selected_edges = {item["edge"] for item in selected}

        fallback = [
            item for item in freq_info
            if item["edge"] not in selected_edges
        ]

        fallback = sorted(
            fallback,
            key=lambda x: (
                min(abs(x["freq"] - min_freq), abs(x["freq"] - max_freq)),
                x["distance_from_half"],
                x["cost"]
            )
        )

        selected.extend(fallback[:max_arcs - len(selected)])

    I_selected = sorted(
        [item["edge"] for item in selected],
        key=lambda e: base_cost_undirected(base_dist, e[0], e[1])
    )

    info = {
        "source": "problematic_areas_pi_frequency",
        "n_candidates": len(I_candidates),
        "n_calibration_scenarios": n_valid,
        "min_freq": min_freq,
        "max_freq": max_freq,
        "max_arcs": max_arcs,
        "selected_I": I_selected,
        "selected_details": selected,
        "all_frequency_details": sorted(
            freq_info,
            key=lambda x: (-x["score_uncertainty"], x["cost"])
        ),
        "pi_values": pi_values,
        "I_candidates": I_candidates,
    }

    return I_selected, info

def crossing_area_edges_from_arcs(arcs, coords, areas):
    """
    Dato un tour, restituisce gli archi non orientati del tour che attraversano
    almeno una delle aree problematiche.
    """
    return sorted({
        canon_edge(i, j)
        for (i, j) in arcs
        if arc_crosses_any_area(i, j, coords, areas)
    })


def print_area_policy_diagnostics(scenario_ids, I, coords, areas, results, res_stoch=None, eev_solutions=None):
    """
    Diagnostica per verificare che la politica 'area => I' sia rispettata.

    Dopo la patch, per ogni tour dovrebbe valere:
        archi che attraversano aree ⊆ I

    Se compaiono archi in 'attraversano aree ma NON sono in I', allora il legame
    area -> multa/prenotazione è ancora incompleto.
    """
    I_set = {canon_edge(*e) for e in I}

    print("\nDIAGNOSTICA AREE -> I")
    print("=" * 80)
    print(f"Numero tratte in I: {len(I_set)}")

    for sid in scenario_ids:
        print(f"\nScenario {sid}")

        pi_arcs = results[sid]["exact_free"]["arcs"]
        pi_cross = set(crossing_area_edges_from_arcs(pi_arcs, coords, areas))
        print(f"  PI:  attraversano aree = {sorted(pi_cross)}")
        print(f"       attraversano aree ma NON sono in I = {sorted(pi_cross - I_set)}")

        if eev_solutions is not None and sid in eev_solutions:
            eev_arcs = eev_solutions[sid]["arcs"]
            eev_cross = set(crossing_area_edges_from_arcs(eev_arcs, coords, areas))
            print(f"  EEV: attraversano aree = {sorted(eev_cross)}")
            print(f"       attraversano aree ma NON sono in I = {sorted(eev_cross - I_set)}")

        if res_stoch is not None and sid in res_stoch.get("scenario_solutions", {}):
            sto_arcs = res_stoch["scenario_solutions"][sid]["y_used"]
            sto_cross = set(crossing_area_edges_from_arcs(sto_arcs, coords, areas))
            sto_reserved = {canon_edge(*e) for e in res_stoch.get("x_used", [])}
            sto_used_unreserved = sto_cross - sto_reserved

            print(f"  STO: attraversano aree = {sorted(sto_cross)}")
            print(f"       attraversano aree ma NON sono in I = {sorted(sto_cross - I_set)}")
            print(f"       attraversano aree e prenotati = {sorted(sto_cross & sto_reserved)}")
            print(f"       attraversano aree NON prenotati => dovrebbero pagare multa = {sorted(sto_used_unreserved)}")


def _draw_areas(ax, areas):
    """
    Disegna le aree usando colore ed etichetta in base al tipo.
    Così l'ordine delle aree non crea etichette sbagliate.
    """
    from matplotlib.patches import Rectangle, Ellipse

    style = {
        "circle":  {"color": "orange",    "label": "Zona circolare"},
        "ellipse": {"color": "limegreen", "label": "Zona ellittica"},
        "rect":    {"color": "royalblue", "label": "Zona rettangolare"},
    }

    for area in areas:
        t = area["type"]
        col = style[t]["color"]
        lab = style[t]["label"]

        if t == "circle":
            ax.add_patch(
                plt.Circle(
                    (area["cx"], area["cy"]),
                    area["radius"],
                    color=col,
                    alpha=0.25,
                    zorder=2,
                    label=lab
                )
            )
            ax.plot(area["cx"], area["cy"], "x", color=col, markersize=8, zorder=7)

        elif t == "ellipse":
            ax.add_patch(
                Ellipse(
                    (area["cx"], area["cy"]),
                    2 * area["rx"],
                    2 * area["ry"],
                    color=col,
                    alpha=0.25,
                    zorder=2,
                    label=lab
                )
            )
            ax.plot(area["cx"], area["cy"], "x", color=col, markersize=8, zorder=7)

        elif t == "rect":
            ax.add_patch(
                Rectangle(
                    (area["x_min"], area["y_min"]),
                    area["x_max"] - area["x_min"],
                    area["y_max"] - area["y_min"],
                    color=col,
                    alpha=0.25,
                    zorder=2,
                    label=lab
                )
            )



# FUNZIONI GRAFICI (comune a tutti)

# Disegno gli archi
def _draw_arcs(ax, arcs, coords, highlight_undir, base_color, lw_normal=1.8, lw_hi=2.8):
    for (i, j) in arcs:
        xi, yi = coords[i]; xj, yj = coords[j]
        in_I = highlight_undir is not None and canon_edge(i, j) in highlight_undir
        color  = "crimson" if in_I else base_color
        lw     = lw_hi     if in_I else lw_normal
        ms     = 16        if in_I else 13
        ax.annotate("", xy=(xj, yj), xytext=(xi, yi),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=ms), zorder=3)

def _draw_reserved_not_used(ax, reserved_edges, tour_arcs, coords):
    """Mostra le prenotazioni x=1 che non compaiono nel tour y dello scenario.

    Le tratte prenotate e percorse sono già colorate in rosso da _draw_arcs.
    Questa funzione aggiunge solo le prenotazioni non percorse, con linea tratteggiata.
    """
    if not reserved_edges:
        return

    tour_edges = {canon_edge(i, j) for (i, j) in tour_arcs}

    for (i, j) in sorted({canon_edge(*e) for e in reserved_edges}):
        if canon_edge(i, j) in tour_edges:
            continue
        xi, yi = coords[i]
        xj, yj = coords[j]
        ax.plot(
            [xi, xj], [yi, yj],
            color="crimson",
            linestyle="--",
            linewidth=2.2,
            alpha=0.55,
            zorder=2,
        )


# Disegno i nodi
def _draw_nodes(ax, nodes, coords):
    xs = [coords[n][0] for n in nodes]; ys = [coords[n][1] for n in nodes]
    ax.scatter(xs, ys, color="#333333", s=70, zorder=5)
    for n, (cx, cy) in coords.items():
        ax.text(cx + 0.4, cy + 0.4, str(n), fontsize=8, zorder=6)

#Creo un pannello 2x2
def plot_scenario_comparison(
        exp_name, scenario_id, nodes, coords, results,
        tour_medio, arcs_medio, eev_costs,
        res_stoch, stoch_costs, x_ev, x_used_set,
        PI, EEV, STO,
        stoch_tour_costs=None, stoch_penalty_costs=None,
        areas=None, eev_fixed_set=None, eev_solutions=None,
        # ── parametri NNE (opzionali per retrocompatibilità) ──
        nne_solution=None,      # dict restituito da evaluate_nne_solution per scenario
        x_nne=None,             # lista tratte prenotate da NNE
        nne_costs=None,         # dict scenario_id → costo totale NNE
        save=True):
 
    if eev_fixed_set is None:
        eev_fixed_set = x_ev
 
    # Soluzione di ricorso EEV
    if eev_solutions is not None and scenario_id in eev_solutions:
        eev_arcs  = eev_solutions[scenario_id]["arcs"]
        eev_tour  = eev_solutions[scenario_id]["tour"]
        eev_title = f"EEV: ricorso con x^EV fissato (scenario {scenario_id})"
        ev_tc = eev_solutions[scenario_id].get("tour_cost",       0.0)
        ev_pc = eev_solutions[scenario_id].get("penalty_paid",    0.0)
        ev_rc = eev_solutions[scenario_id].get("reservation_paid", 0.0)
    else:
        eev_arcs  = arcs_medio
        eev_tour  = tour_medio
        eev_title = f"Medione / EEV (scenario {scenario_id})"
        ev_tc = ev_pc = ev_rc = 0.0
 
    # Soluzione di ricorso NNE
    nne_arcs  = []
    nne_tour  = []
    nne_cost_s = None
    if nne_solution is not None:
        nne_arcs   = nne_solution.get("arcs",  [])
        nne_tour   = nne_solution.get("tour",  [])
        nne_tc     = nne_solution.get("tour_cost",       0.0)
        nne_pc     = nne_solution.get("penalty_paid",    0.0)
        nne_rc     = nne_solution.get("reservation_paid", 0.0)
        nne_cost_s = (nne_costs or {}).get(scenario_id)
 
    fig, axes = plt.subplots(2, 2, figsize=(26, 22))
    fig.suptitle(
        f"Scenario {scenario_id} — confronto PI / EEV / STO / NNE",
        fontsize=16, fontweight="bold", y=1.01)
 
    panels = [
        {
            "title":     "PI libero",
            "arcs":      results[scenario_id]["exact_free"]["arcs"],
            "tour":      results[scenario_id]["exact_free"]["tour"],
            "cost":      results[scenario_id]["exact_free"]["length"],
            "color":     "#01D80B",
            "highlight": set(),
        },
        {
            "title":     eev_title,
            "arcs":      eev_arcs,
            "tour":      eev_tour,
            "cost":      eev_costs[scenario_id],
            "color":     "#A500CE",
            "highlight": eev_fixed_set,
        },
        {
            "title":     "Stocastico / STO",
            "arcs":      res_stoch["scenario_solutions"][scenario_id]["y_used"],
            "tour":      res_stoch["scenario_solutions"][scenario_id]["tour"],
            "cost":      stoch_costs[scenario_id],
            "color":     "#000000",
            "highlight": x_used_set,
        },
        {
            "title":     f"NN-E (Neur2SP) scenario {scenario_id}",
            "arcs":      nne_arcs,
            "tour":      nne_tour,
            "cost":      nne_cost_s,
            "color":     "#0055CC",
            "highlight": set(x_nne) if x_nne else set(),
        },
    ]
 
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
 
    for panel, (r, c) in zip(panels, positions):
        ax = axes[r][c]
        _draw_nodes(ax, nodes, coords)
 
        if areas:
            _draw_areas(ax, areas)
 
        # pannello NNE vuoto se non disponibile
        if panel["arcs"]:
            _draw_arcs(ax, panel["arcs"], coords, panel["highlight"], panel["color"])
            _draw_reserved_not_used(ax, panel["highlight"], panel["arcs"], coords)
        else:
            ax.text(0.5, 0.5, "NNE non disponibile",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
 
        tour = panel["tour"]
        if tour:
            tour_str = " → ".join(str(n) for n in tour) + f" → {tour[0]}"
            arc_list = [(tour[k], tour[(k + 1) % len(tour)]) for k in range(len(tour))]
            arc_str  = "  →  ".join(f"({i},{j})" for (i, j) in arc_list)
        else:
            tour_str = "n.d."
            arc_str  = "n.d."
 
        cost_str = f"{panel['cost']:.4f}" if panel["cost"] is not None else "N/A"
 
        handles = [mpatches.Patch(color=panel["color"], label="Arco percorso")]
        if panel["highlight"]:
            tour_edges        = {canon_edge(i, j) for (i, j) in panel["arcs"]}
            highlighted_edges = {canon_edge(*e)   for e      in panel["highlight"]}
            if highlighted_edges & tour_edges:
                handles.append(mpatches.Patch(color="crimson",
                                              label="Tratta prenotata e percorsa"))
            if highlighted_edges - tour_edges:
                handles.append(Line2D([0], [0], color="crimson", linestyle="--",
                                      linewidth=2.2,
                                      label="Tratta prenotata non percorsa"))
        if areas:
            handles.append(mpatches.Patch(color="orange", alpha=0.5,
                                          label="Area problematica"))
 
        ax.legend(handles=handles, loc="upper left", fontsize=8)
        ax.set_title(
            f"{panel['title']}\nCosto: {cost_str}\nTour: {tour_str}\nArchi: {arc_str}",
            fontsize=8, loc="left", pad=8, family="monospace")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle="--", alpha=0.35)
 
    plt.tight_layout()
 
    if save:
        fname = out_path(f"{exp_name}_scenario_{scenario_id}_confronto.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Salvato: {fname}")
    else:
        plt.show()

# Genero tutti i grafici per i(l) esperiment/i
def genera_grafici(exp_name, nodes, coords, scenario_ids, results, tour_medio, arcs_medio,
                   eev_costs, res_stoch, stoch_costs, x_ev, x_used_set, PI, STO, EEV,
                   stoch_tour_costs=None, stoch_penalty_costs=None, areas=None,
                   eev_solutions=None, save=True):

    for sid in scenario_ids:
        plot_scenario_comparison( exp_name, sid, nodes, coords, results, tour_medio, arcs_medio, eev_costs,
            res_stoch, stoch_costs, x_ev, x_used_set, PI, EEV, STO, stoch_tour_costs=stoch_tour_costs,
            stoch_penalty_costs=stoch_penalty_costs, areas=areas, eev_fixed_set=x_ev,
            eev_solutions=eev_solutions, save=save )


# ESPERIMENTO A — I manuali, C fisso, pert ±N(40%,10%)

def run_esperimento_A(nodes, coords, base_dist, E, root, env):
    exp_name = "espA"
    scenario_ids = SCENARIO_IDS
    print("ESPERIMENTO A: I scelti manualmente, C fisso, pert ±N(40%,10%)")
    
    # Selezione I
    I_directed = [(nodes[a], nodes[b]) for (a, b) in I_INDICES_A]
    I = directed_to_undirected_arcs(I_directed)
    for (i, j) in I:
        if i == j: raise ValueError(f"Tratta in I non valida: ({i},{j})")
        if (i, j) not in E or (j, i) not in E:
            raise ValueError(f"Tratta ({i},{j}) non ha entrambi i versi in E")

    fixed_set = set(I)
    b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
    p = {(i, j): PRENOTAZIONE_FRAC * b[i, j] for (i, j) in I}
    C = {(i, j): PENALTY_FRAC * b[i, j] for (i, j) in I}

    print("\nTRATTE IMPORTANTI: b percorrenza, p prenotazione, C multa:")
    for (i, j) in I:
        print(f"  Tratta {{{i},{j}}} | b = {b[i,j]:.4f} | p = {p[i,j]:.4f} | C = {C[i,j]:.4f}")

    # PI
    frequent_arcs = find_frequent_arcs(nodes, E, base_dist, root, env, I,
                                        mean_frac=MEAN_FRAC, sigma_frac=SIGMA_FRAC_A)
    print("\nRISOLUZIONE SCENARI:")
    results, scenario_probs, total_random_uses = generate_scenarios(
        scenario_ids, nodes, E, base_dist, I, frequent_arcs,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC_A, FINAL_SCENARIO_SEED,
        root=root, env=env, p=p, C=C)

    # EEV / medione
    tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev,eev_solutions, EEV, PI = compute_eev_medione(
        nodes, E, root, env, base_dist, I, p, C, results, scenario_ids, return_solutions=True)

    # Stocastico
    scenario_deltas = {sid: results[sid]["pert"] for sid in scenario_ids}
    res_stoch = solve_stochastic(nodes, E, I, base_dist, root, p, C, env,
                                  scenario_deltas, scenario_probs, force_important=False)
    STO = res_stoch["objective"]
    stoch_solver_info = res_stoch.get("solver_info", {})
    x_used_set     = set(res_stoch["x_used"])
    stoch_costs    = {sid: res_stoch["scenario_solutions"][sid]["total_cost"] for sid in scenario_ids}
    STO_recomputed = sum(scenario_probs[sid] * stoch_costs[sid] for sid in scenario_ids)
    print(f"\nControllo STO:")
    print(f"  STO da modello     = {STO:.6f}")
    print(f"  STO ricalcolato    = {STO_recomputed:.6f}")
    print(f"  differenza assoluta = {abs(STO - STO_recomputed):.8f}")
    stoch_tour_c   = {sid: res_stoch["scenario_solutions"][sid]["tour_cost"]   for sid in scenario_ids}
    stoch_penalty_c = {sid: res_stoch["scenario_solutions"][sid]["penalty_paid"] for sid in scenario_ids}

    print_and_save_summary(exp_name, scenario_ids, results, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, stoch_costs,
                        stoch_tour_c, stoch_penalty_c, PI, STO, EEV,
                            I=I, p=p, C=C, b=b, stoch_solver_info=stoch_solver_info,
                            frequent_arcs=frequent_arcs, total_random_uses=total_random_uses, random_impact_stats=random_impact_train)
    if DO_VALIDATION:
        validate_policies(
            nodes, E, base_dist, root, env, I, p, C,
            x_used_set, x_ev, frequent_arcs, N_VALIDATION_SCENARIOS,
            N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC_A, exp_name=exp_name
        )
    genera_grafici(exp_name, nodes, coords, scenario_ids, results, tour_medio, arcs_medio,
                   eev_costs, res_stoch, stoch_costs, x_ev, x_used_set, PI, STO, EEV, stoch_tour_costs=stoch_tour_c,
                   stoch_penalty_costs=stoch_penalty_c,
                   eev_solutions=eev_solutions)


# ESPERIMENTO B — I k-medoids, C fisso

def run_esperimento_B(nodes, coords, base_dist, E, root, env):
    exp_name = "espB"
    scenario_ids = SCENARIO_IDS
    
    print("ESPERIMENTO B: I dagli archi uscenti dai nodi medoidi, C fisso, pert N(40%,20%)")
    
    I, medoid_info = build_I_from_medoid_outgoing_nodes(nodes, E, base_dist, K_MEDOID_NODES)
    fixed_set = set(I)
    b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
    p = {(i, j): PRENOTAZIONE_FRAC * b[i, j] for (i, j) in I}
    C = {(i, j): PENALTY_FRAC * b[i, j] for (i, j) in I}

    print(f"\nNODI MEDOIDI FISSATI: {medoid_info['medoid_nodes']}")
    print(f"Tratte k-medoids selezionate in I: {medoid_info['n_undirected_tratte']}")
    print(f"Tratte non orientate in I: {medoid_info['n_undirected_tratte']}")
    print("\nTRATTE SELEZIONATE IN I:")
    for (i, j) in I:
        print(f"  {{{i},{j}}} | b={b[i,j]:.4f} | p={p[i,j]:.4f} | C={C[i,j]:.4f}")

    frequent_arcs = find_frequent_arcs(nodes, E, base_dist, root, env, I)
    print("\nRISOLUZIONE SCENARI:")
    results, scenario_probs, total_random_uses = generate_scenarios(
        scenario_ids, nodes, E, base_dist, I, frequent_arcs,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
        root=root, env=env, p=p, C=C)

    tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, eev_solutions,EEV, PI = compute_eev_medione(
        nodes, E, root, env, base_dist, I, p, C, results, scenario_ids,return_solutions=True)

    scenario_deltas = {sid: results[sid]["pert"] for sid in scenario_ids}
    res_stoch = solve_stochastic(nodes, E, I, base_dist, root, p, C, env,
                                  scenario_deltas, scenario_probs, force_important=False)
    STO = res_stoch["objective"]
    stoch_solver_info = res_stoch.get("solver_info", {})
    x_used_set      = set(res_stoch["x_used"])
    stoch_costs     = {sid: res_stoch["scenario_solutions"][sid]["total_cost"] for sid in scenario_ids}
    STO_recomputed = sum(scenario_probs[sid] * stoch_costs[sid] for sid in scenario_ids)
    print(f"\nControllo STO:")
    print(f"  STO da modello     = {STO:.6f}")
    print(f"  STO ricalcolato    = {STO_recomputed:.6f}")
    print(f"  differenza assoluta = {abs(STO - STO_recomputed):.8f}")
    stoch_tour_c    = {sid: res_stoch["scenario_solutions"][sid]["tour_cost"]  for sid in scenario_ids}
    stoch_penalty_c = {sid: res_stoch["scenario_solutions"][sid]["penalty_paid"] for sid in scenario_ids}
    random_impact_train = compute_random_edge_usage_stats(
    results,
    scenario_ids,
    stoch_solutions=res_stoch.get("scenario_solutions", {}),
    eev_solutions=eev_solutions,)

    print_and_save_summary(exp_name, scenario_ids, results, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, stoch_costs,
                            stoch_tour_c, stoch_penalty_c, PI, STO, EEV,
                            I=I, p=p, C=C, b=b, stoch_solver_info=stoch_solver_info,
                            frequent_arcs=frequent_arcs, total_random_uses=total_random_uses, random_impact_stats=random_impact_train)
    if DO_VALIDATION:
        validate_policies(
            nodes, E, base_dist, root, env, I, p, C,
            x_used_set, x_ev, frequent_arcs, N_VALIDATION_SCENARIOS,
            N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, exp_name=exp_name
        )
    genera_grafici(exp_name, nodes, coords, scenario_ids, results, tour_medio, arcs_medio,
                   eev_costs, res_stoch, stoch_costs, x_ev, x_used_set, PI, STO, EEV,stoch_tour_costs=stoch_tour_c,
                   stoch_penalty_costs=stoch_penalty_c,
                   eev_solutions=eev_solutions)
    return {
        "I":               I,
        "b":               b,
        "p":               p,
        "C":               C,
        "results":         results,          # dict scenario_id → dati
        "scenario_probs":  scenario_probs,
        "frequent_arcs":   frequent_arcs,
        "x_ev":            x_ev,
        "x_used_sto":      x_used_set,
        "eev_costs":       eev_costs,
        "eev_tour_costs":  eev_tour_costs_ev,
        "eev_penalty_costs": eev_penalty_costs_ev,
        "eev_solutions":   eev_solutions,
        "stoch_costs":     stoch_costs,
        "stoch_tour_costs":   stoch_tour_c,
        "stoch_penalty_costs": stoch_penalty_c,
        "stoch_solutions": res_stoch["scenario_solutions"],
        "res_stoch":       res_stoch,
        "stoch_solver_info": stoch_solver_info,
        "tour_medio":      tour_medio,
        "arcs_medio":      arcs_medio,
        "PI":              PI,
        "STO":             STO,
        "EEV":             EEV,
        "total_random_uses": total_random_uses,
        "random_impact_stats": random_impact_train,
    }


# ESPERIMENTO C — I k-medoids, C frequenziale sui PI finali

def run_esperimento_C(nodes, coords, base_dist, E, root, env):
    exp_name = "espC"
    scenario_ids = SCENARIO_IDS
    
    print("ESPERIMENTO C: I dagli archi uscenti dai nodi medoidi, C frequenziale sui PI finali")

    I, medoid_info = build_I_from_medoid_outgoing_nodes(nodes, E, base_dist, K_MEDOID_NODES)
    fixed_set = set(I)
    b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
    p = {(i, j): PRENOTAZIONE_FRAC * b[i, j] for (i, j) in I}

    print(f"\nNODI MEDOIDI FISSATI: {medoid_info['medoid_nodes']}")
    print(f"Tratte k-medoids selezionate in I: {medoid_info['n_undirected_tratte']}")
    print(f"Tratte non orientate in I: {medoid_info['n_undirected_tratte']}")
    print("\nTRATTE SELEZIONATE IN I:")
    for (i, j) in I:
        print(f"  {{{i},{j}}} | b={b[i,j]:.4f} | p={p[i,j]:.4f}")

    # Primo passo: PI liberi per calcolare C frequenziale
    frequent_arcs = find_frequent_arcs(nodes, E, base_dist, root, env, I)
    print("\nRISOLUZIONE PI LIBERI SUI 4 SCENARI FINALI:")
    C_placeholder = {(i, j): 0.0 for (i, j) in I}
    results, scenario_probs, total_random_uses = generate_scenarios(
        scenario_ids, nodes, E, base_dist, I, frequent_arcs,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
        root=root, env=env, p=p, C=C_placeholder)

    # C frequenziale
    C, final_pi_counts, penalty_weights = build_frequency_based_penalties(
        I, results, scenario_ids, base_dist, PENALTY_FRAC)

    print("\nPENALITÀ C FREQUENZIALE:")
    n_scen = len(scenario_ids)
    print(f"  Regola: count 0/4 o 1/4 → peso 0; count {n_scen//2}/4 → intermedio; count 4/4 → 1")
    for (i, j) in I:
        print(f"  {{{i},{j}}} | compare nei PI = {final_pi_counts[(i,j)]}/{n_scen}"
              f" | peso = {penalty_weights[(i,j)]:.4f} | C = {C[i,j]:.4f}")

    # 
    #EEV
    tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev,eev_solutions, EEV, PI = compute_eev_medione(
        nodes, E, root, env, base_dist, I, p, C, results, scenario_ids, return_solutions=True)

    scenario_deltas = {sid: results[sid]["pert"] for sid in scenario_ids}
    res_stoch = solve_stochastic(nodes, E, I, base_dist, root, p, C, env,
                                  scenario_deltas, scenario_probs, force_important=False)
    STO = res_stoch["objective"]
    stoch_solver_info = res_stoch.get("solver_info", {})
    x_used_set      = set(res_stoch["x_used"])
    stoch_costs     = {sid: res_stoch["scenario_solutions"][sid]["total_cost"] for sid in scenario_ids}
    STO_recomputed = sum(scenario_probs[sid] * stoch_costs[sid] for sid in scenario_ids)
    random_impact_train = compute_random_edge_usage_stats(
    results,
    scenario_ids,
    stoch_solutions=res_stoch.get("scenario_solutions", {}),
    eev_solutions=eev_solutions,)

    print(f"\nControllo STO:")
    print(f"  STO da modello     = {STO:.6f}")
    print(f"  STO ricalcolato    = {STO_recomputed:.6f}")
    print(f"  differenza assoluta = {abs(STO - STO_recomputed):.8f}")
    stoch_tour_c    = {sid: res_stoch["scenario_solutions"][sid]["tour_cost"]  for sid in scenario_ids}
    stoch_penalty_c = {sid: res_stoch["scenario_solutions"][sid]["penalty_paid"] for sid in scenario_ids}

    print_and_save_summary(exp_name, scenario_ids, results, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, stoch_costs,
                            stoch_tour_c, stoch_penalty_c, PI, STO, EEV, I=I,
                            final_pi_counts=final_pi_counts, penalty_weights=penalty_weights,
                            p=p, C=C, b=b, stoch_solver_info=stoch_solver_info,
                            frequent_arcs=frequent_arcs, total_random_uses=total_random_uses, random_impact_stats=random_impact_train)
    if DO_VALIDATION:
        validate_policies(
            nodes, E, base_dist, root, env, I, p, C,
            x_used_set, x_ev, frequent_arcs, N_VALIDATION_SCENARIOS,
            N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, exp_name=exp_name
        )
    genera_grafici(exp_name, nodes, coords, scenario_ids, results, tour_medio, arcs_medio,
                   eev_costs, res_stoch, stoch_costs, x_ev, x_used_set, PI, STO, EEV, stoch_tour_costs=stoch_tour_c,
                   stoch_penalty_costs=stoch_penalty_c,
                   eev_solutions=eev_solutions)


# ESPERIMENTO D — I da aree problematiche
def run_esperimento_D(nodes, coords, base_dist, E, root, env):
    exp_name = "espD"
    scenario_ids = SCENARIO_IDS
    print("\n" + "=" * 70)
    print("ESPERIMENTO D: I da aree problematiche, C fisso")
    

    # Aree problematiche
    areas = auto_place_areas(coords, n_areas=N_AREAS_D, margin_frac=MARGIN_FRAC_D, radius_frac=RADIUS_FRAC_D)
    print("\nAREE PROBLEMATICHE:")
    for k_area, area in enumerate(areas):
        print(f"  Area {k_area + 1}: {format_area(area)}")

    #I = select_I_from_problematic_areas_all(nodes, E, coords, base_dist, areas) prima
    I, area_info = select_I_from_problematic_areas_by_pi_frequency( nodes, E, coords, base_dist, root, env, areas, 
                                                                    n_calibration_scenarios=30, min_freq=0.20, max_freq=0.80, max_arcs=7 )


    fixed_set = set(I)
    b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
    p = {(i, j): PRENOTAZIONE_FRAC * b[i, j] for (i, j) in I}
    C = {(i, j): PENALTY_FRAC * b[i, j] for (i, j) in I}

    print(f"\nTRATTE I SELEZIONATE ({len(I)} totali):")
    for (i, j) in I:
        print(f"  {{{i},{j}}} | b = {b[i,j]:.4f} | p = {p[i,j]:.4f} | C = {C[i,j]:.4f}")

    frequent_arcs = find_frequent_arcs(nodes, E, base_dist, root, env, I)
    print("\nRISOLUZIONE SCENARI:")
    results, scenario_probs, total_random_uses = generate_scenarios(
        scenario_ids, nodes, E, base_dist, I, frequent_arcs,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
        root=root, env=env, p=p, C=C)

    tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, eev_solutions, EEV, PI = compute_eev_medione(
    nodes, E, root, env, base_dist, I, p, C, results, scenario_ids,
    return_solutions=True)

    scenario_deltas = {sid: results[sid]["pert"] for sid in scenario_ids}
    res_stoch = solve_stochastic(nodes, E, I, base_dist, root, p, C, env,
                                  scenario_deltas, scenario_probs, force_important=False)
    STO = res_stoch["objective"]
    stoch_solver_info = res_stoch.get("solver_info", {})
    x_used_set      = set(res_stoch["x_used"])
    stoch_costs     = {sid: res_stoch["scenario_solutions"][sid]["total_cost"] for sid in scenario_ids}
    STO_recomputed = sum(scenario_probs[sid] * stoch_costs[sid] for sid in scenario_ids)

    random_impact_train = compute_random_edge_usage_stats(
    results,
    scenario_ids,
    stoch_solutions=res_stoch.get("scenario_solutions", {}),
    eev_solutions=eev_solutions,)

    print(f"\nControllo STO:")
    print(f"  STO da modello     = {STO:.6f}")
    print(f"  STO ricalcolato    = {STO_recomputed:.6f}")
    print(f"  differenza assoluta = {abs(STO - STO_recomputed):.8f}")
    stoch_tour_c    = {sid: res_stoch["scenario_solutions"][sid]["tour_cost"]  for sid in scenario_ids}
    stoch_penalty_c = {sid: res_stoch["scenario_solutions"][sid]["penalty_paid"] for sid in scenario_ids}
    #print_area_policy_diagnostics( scenario_ids=scenario_ids, I=I, coords=coords, areas=areas, results=results, res_stoch=res_stoch, eev_solutions=eev_solutions )


    print_and_save_summary(exp_name, scenario_ids, results, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, stoch_costs,
                            stoch_tour_c, stoch_penalty_c, PI, STO, EEV,
                            I=I, p=p, C=C, b=b, stoch_solver_info=stoch_solver_info,
                            frequent_arcs=frequent_arcs, total_random_uses=total_random_uses, random_impact_stats=random_impact_train)
    if DO_VALIDATION:
        validate_policies(
            nodes, E, base_dist, root, env, I, p, C,
            x_used_set, x_ev, frequent_arcs, N_VALIDATION_SCENARIOS,
            N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, exp_name=exp_name
        )
    genera_grafici(exp_name, nodes, coords, scenario_ids, results, tour_medio, arcs_medio,
                   eev_costs, res_stoch, stoch_costs, x_ev, x_used_set, PI, STO, EEV,
                   stoch_tour_costs=stoch_tour_c, stoch_penalty_costs=stoch_penalty_c,
                   areas=areas, eev_solutions=eev_solutions)


# ESPERIMENTO E — I k-medoids, campi vettoriali nei grafici

try:
    # Carico quanto necessario per grafici e per oggetti nuovi
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    import numpy as np
    from dataclasses import dataclass, field
    from typing import List, Tuple
    
    # Creo oggetti
    @dataclass
    class Cyclone:
        cx: float; cy: float; strength: float; radius: float; sign: int = -1

    @dataclass
    class Gust:
        cx: float; cy: float; angle_deg: float; strength: float; sigma: float

    @dataclass
    class WindScenario:
        name: str
        probability: float
        background: Tuple[float, float]
        vortices: List = field(default_factory=list)
        gusts:    List = field(default_factory=list)

    # Creo il vortice: seleziono un centro e un raggio, poi  distiguo la parte intensa vicina al centro
    # la parte più lontana >0.3R con forma esponenziale che diminuisce intensità
    def _vortex_field(X, Y, vortex):
        dx = X - vortex.cx; dy = Y - vortex.cy
        r  = np.sqrt(dx**2 + dy**2) + 1e-9
        r_core = vortex.radius * 0.3
        speed  = np.where(r <= r_core,
                          vortex.strength * (r / r_core),
                          vortex.strength * np.exp(-((r - r_core) / vortex.radius)**2))
        return vortex.sign * (-dy / r) * speed, vortex.sign * (dx / r) * speed

    # Creo raffica di vento
    def _gust_field(X, Y, gust):
        ang      = np.radians(gust.angle_deg)
        envelope = gust.strength * np.exp(
            -(((X - gust.cx)**2 + (Y - gust.cy)**2) / (2 * gust.sigma**2)))
        return envelope * np.cos(ang), envelope * np.sin(ang)
    
    # Creo i 4 scenari, mischiando gli ogetti che creati
    def _build_wind_scenarios():
        return [
            WindScenario("Calm - scattered gusts", 0.25, (2.0, 1.0),
                         gusts=[Gust(0.25, 0.70, 45, 6.0, 0.12), Gust(0.75, 0.30, 200, 5.0, 0.10)]),
            WindScenario("Cyclone-Anticyclone dipole", 0.25, (1.0, -1.0),
                         vortices=[Cyclone(0.30, 0.60, 9.0, 0.28, -1), Cyclone(0.70, 0.35, 7.0, 0.22, +1)],
                         gusts=[Gust(0.50, 0.80, 270, 4.0, 0.09)]),
            WindScenario("Severe storm - intense cyclone", 0.25, (-3.0, -2.0),
                         vortices=[Cyclone(0.50, 0.50, 15.0, 0.40, -1)],
                         gusts=[Gust(0.15, 0.20, 315, 10.0, 0.08), Gust(0.85, 0.80, 315, 10.0, 0.08)]),
            WindScenario("Multi-vortex + channelled gusts", 0.25, (4.0, 0.5),
                         vortices=[Cyclone(0.20, 0.80, 8.0, 0.20, -1), Cyclone(0.80, 0.20, 7.0, 0.18, -1),
                                   Cyclone(0.50, 0.50, 6.0, 0.15, +1)],
                         gusts=[Gust(0.50, 0.15, 90, 8.0, 0.10), Gust(0.50, 0.85, 270, 8.0, 0.10)]),
        ]

    # Associo scenario numerico al relativo campo vettoriale
    def _wind_scenario_for_id(scenario_id, wind_scenarios):
        return wind_scenarios[(scenario_id - 1) % len(wind_scenarios)]

    #?
    def _coords_bounds(coords):
        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        xr = x_max - x_min or 1.0
        yr = y_max - y_min or 1.0
        return x_min, y_min, xr, yr

    # Normalizzo i punti
    def _to_normalized_point(x, y, bounds):
        x_min, y_min, xr, yr = bounds
        return (x - x_min) / xr, (y - y_min) / yr

    # Studio il campo vettoriale nel punto appena normalizzato
    def _wind_vector_at_norm(x, y, ws):
        U = np.zeros_like(np.asarray(x, dtype=float)) + ws.background[0]
        V = np.zeros_like(np.asarray(y, dtype=float)) + ws.background[1]

        for vort in ws.vortices:
            du, dv = _vortex_field(x, y, vort)
            U += du
            V += dv

        for gust in ws.gusts:
            du, dv = _gust_field(x, y, gust)
            U += du
            V += dv

        return U, V

    # Calcolo la componente media del vento lungo l arco ij
    def _wind_projection_on_arc(i, j, coords, ws, bounds, n_samples=WIND_ARC_SAMPLES_E):
        xi, yi = coords[i]
        xj, yj = coords[j]

        xi_n, yi_n = _to_normalized_point(xi, yi, bounds)
        xj_n, yj_n = _to_normalized_point(xj, yj, bounds)

        dx = xj_n - xi_n
        dy = yj_n - yi_n
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            return 0.0

        ex = dx / length
        ey = dy / length

        ts = np.linspace(0.0, 1.0, n_samples)
        xs = xi_n + ts * dx
        ys = yi_n + ts * dy

        U, V = _wind_vector_at_norm(xs, ys, ws)
        projection = U * ex + V * ey
        return float(np.mean(projection))

    # Sistemo le perturbazioni in base alla direzione del vento e quella del tour
    def build_wind_field_perturbation(nodes, E, coords, base_dist, wind_scenario, alpha=WIND_COST_ALPHA_E,
                                      reference_speed=WIND_REFERENCE_SPEED_E,  min_factor=WIND_MIN_FACTOR_E,
                                      max_factor=WIND_MAX_FACTOR_E, n_samples=WIND_ARC_SAMPLES_E):
        
        bounds = _coords_bounds(coords)
        perturb = {}

        for (i, j) in E:
            base = base_dist[i][j]
            projection = _wind_projection_on_arc(i, j, coords, wind_scenario, bounds, n_samples)

            factor = 1.0 - alpha * (projection / reference_speed)
            factor = max(min_factor, min(max_factor, factor))

            new_cost = max(base * factor, 0.05 * base)
            perturb[(i, j)] = new_cost - base

        return perturb


    def _wind_perturbation_stats(pert):
        values = list(pert.values())
        if not values:
            return "min=0.0000 | media=0.0000 | max=0.0000"
        return f"min={min(values):.4f} | media={sum(values)/len(values):.4f} | max={max(values):.4f}"

    # Faccio andare gli scenari
    def run_scenarios_wind_field(nodes, E, coords, root, env, base_dist, I, p, C, scenario_ids, wind_scenarios):
        
        raw_probs = {sid: _wind_scenario_for_id(sid, wind_scenarios).probability for sid in scenario_ids}
        prob_sum = sum(raw_probs.values()) or 1.0
        scenario_probs = {sid: raw_probs[sid] / prob_sum for sid in scenario_ids}

        b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
        results = {}

        for scenario_id in scenario_ids:
            ws = _wind_scenario_for_id(scenario_id, wind_scenarios)
            pert = build_wind_field_perturbation(nodes, E, coords, base_dist, ws)
            scenario_dist = build_scenario_dist(base_dist, pert)

            exact_free = solve_exact_tsp(nodes, E, scenario_dist, root, env,
                                         fixed_arcs=[], fixed_edges_undir=[], output_flag=0)


            results[scenario_id] = {
                "pert": pert,
                "scenario_dist": scenario_dist,
                "exact_free": exact_free,
                "wind_name": ws.name,
            }

            pi_str = f"{exact_free['length']:.4f}" if exact_free['length'] is not None else "N/A"
            print(f"  Scenario {scenario_id} ({ws.name}) | PI = {pi_str}")
            print(f"    perturbazioni da campo vettoriale: {_wind_perturbation_stats(pert)}")

        return results, b, scenario_probs

    # Per ogni scenario di k-medoids genero dei campi vettoriali con parametri diversi, altrimenti avrei solo i 4 scenari 
    def _randomize_wind_scenario(ws, seed):
        rng = random.Random(seed)
        
        scale_bg = rng.gauss(1.0, 0.20)
        new_bg = (ws.background[0] * scale_bg, ws.background[1] * scale_bg)
        
        new_vortices = [
            Cyclone(v.cx, v.cy,
                    v.strength * max(0.3, rng.gauss(1.0, 0.25)),
                    v.radius   * max(0.5, rng.gauss(1.0, 0.15)),
                    v.sign)
            for v in ws.vortices
        ]
        new_gusts = [
            Gust(g.cx, g.cy,
                g.angle_deg + rng.gauss(0, 20),
                g.strength  * max(0.3, rng.gauss(1.0, 0.25)),
                g.sigma)
            for g in ws.gusts
        ]
        return WindScenario(ws.name, ws.probability, new_bg, new_vortices, new_gusts)

    
    # Disegno il campo vettoriale nel grafico
    def _draw_wind_background(ax, ws, cmap_name, coords, nx=40, ny=40):
        xs = [c[0] for c in coords.values()]; ys = [c[1] for c in coords.values()]
        x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
        xr = x_max - x_min or 1.0; yr = y_max - y_min or 1.0
        x1d = np.linspace(x_min, x_max, nx); y1d = np.linspace(y_min, y_max, ny)
        X_orig, Y_orig = np.meshgrid(x1d, y1d)
        X_norm = (X_orig - x_min) / xr; Y_norm = (Y_orig - y_min) / yr
        U = np.full_like(X_norm, ws.background[0]); V = np.full_like(Y_norm, ws.background[1])
        for vort in ws.vortices:
            du, dv = _vortex_field(X_norm, Y_norm, vort); U += du; V += dv
        for gust in ws.gusts:
            du, dv = _gust_field(X_norm, Y_norm, gust); U += du; V += dv
        speed = np.sqrt(U**2 + V**2)
        ax.contourf(X_orig, Y_orig, speed, levels=16, cmap=cmap_name, alpha=0.15, zorder=0)
        ax.streamplot(x1d, y1d, U, V, color=speed, cmap=cmap_name, linewidth=0.4, density=0.8,
                      arrowsize=0.5, norm=mcolors.Normalize(vmin=float(speed.min()), vmax=float(speed.max())), zorder=1)

    WIND_SCENARIOS_AVAILABLE = True

except ImportError:
    WIND_SCENARIOS_AVAILABLE = False

# Faccio andare esperimento E
def run_esperimento_E(nodes, coords, base_dist, E, root, env):
    exp_name = "espE"
    scenario_ids = SCENARIO_IDS
    print("\n" + "=" * 70)
    print("ESPERIMENTO E: I dagli archi uscenti dai nodi medoidi, C fisso, distanze generate dai campi vettoriali")
    

    if not WIND_SCENARIOS_AVAILABLE:
        raise RuntimeError("Esperimento E non eseguibile: numpy/dataclasses/matplotlib.colors non disponibili.")

    wind_scenarios = _build_wind_scenarios()

    I, medoid_info = build_I_from_medoid_outgoing_nodes(nodes, E, base_dist, K_MEDOID_NODES)
    fixed_set = set(I)
    b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
    p = {(i, j): PRENOTAZIONE_FRAC * b[i, j] for (i, j) in I}
    C = {(i, j): PENALTY_FRAC_E * b[i, j] for (i, j) in I}

    print(f"\nNODI MEDOIDI FISSATI: {medoid_info['medoid_nodes']}")
    print(f"Tratte k-medoids selezionate in I: {medoid_info['n_undirected_tratte']}")
    print(f"Tratte non orientate in I: {medoid_info['n_undirected_tratte']}")
    print("\nTRATTE SELEZIONATE IN I:")
    for (i, j) in I:
        print(f"  {{{i},{j}}} | b={b[i,j]:.4f} | p={p[i,j]:.4f} | C={C[i,j]:.4f}")

    print("\nRISOLUZIONE SCENARI:")
    results, _, scenario_probs = run_scenarios_wind_field(
    nodes, E, coords, root, env, base_dist, I, p, C, scenario_ids,
    wind_scenarios=wind_scenarios)

    tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, eev_solutions,EEV, PI = compute_eev_medione(
        nodes, E, root, env, base_dist, I, p, C, results, scenario_ids, return_solutions=True)

    scenario_deltas = {sid: results[sid]["pert"] for sid in scenario_ids}
    res_stoch = solve_stochastic(nodes, E, I, base_dist, root, p, C, env,
                                  scenario_deltas, scenario_probs, force_important=False)
    STO = res_stoch["objective"]
    stoch_solver_info = res_stoch.get("solver_info", {})
    x_used_set      = set(res_stoch["x_used"])
    stoch_costs     = {sid: res_stoch["scenario_solutions"][sid]["total_cost"] for sid in scenario_ids}
    STO_recomputed = sum(scenario_probs[sid] * stoch_costs[sid] for sid in scenario_ids)

    print(f"\nControllo STO:")
    print(f"  STO da modello     = {STO:.6f}")
    print(f"  STO ricalcolato    = {STO_recomputed:.6f}")
    print(f"  differenza assoluta = {abs(STO - STO_recomputed):.8f}")
    stoch_tour_c    = {sid: res_stoch["scenario_solutions"][sid]["tour_cost"]  for sid in scenario_ids}
    stoch_penalty_c = {sid: res_stoch["scenario_solutions"][sid]["penalty_paid"] for sid in scenario_ids}

    print_and_save_summary(exp_name, scenario_ids, results, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, stoch_costs,
                            stoch_tour_c, stoch_penalty_c, PI, STO, EEV,
                            I=I, p=p, C=C, b=b, stoch_solver_info=stoch_solver_info)

    _genera_grafici_vento(exp_name, nodes, coords, scenario_ids, results, tour_medio,
                           arcs_medio, eev_costs, res_stoch, stoch_costs, x_ev,
                           x_used_set, PI, STO, EEV, stoch_tour_costs=stoch_tour_c,
                           stoch_penalty_costs=stoch_penalty_c, wind_scenarios=wind_scenarios,
                           eev_solutions=eev_solutions)

# Disegno il campo vettoriale di sfondo
def _genera_grafici_vento(exp_name, nodes, coords, scenario_ids, results, tour_medio,
                           arcs_medio, eev_costs, res_stoch, stoch_costs, x_ev,
                           x_used_set, PI, STO, EEV, stoch_tour_costs=None,
                           stoch_penalty_costs=None, wind_scenarios=None,
                           eev_solutions=None):
    if wind_scenarios is None:
        wind_scenarios = _build_wind_scenarios()
    CMAPS = ["YlOrRd", "PuBu", "RdPu", "BuGn"]

    xs_all = [c[0] for c in coords.values()]
    ys_all = [c[1] for c in coords.values()]
    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    x_range = x_max - x_min or 1.0

    def _setup_wind_ax(ax, sid):
        ax.set_facecolor("#0d1117")
        _draw_wind_background(
            ax,
            _wind_scenario_for_id(sid, wind_scenarios),
            CMAPS[(sid - 1) % len(CMAPS)],
            coords
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.grid(True, linestyle="--", alpha=0.20, color="white")

    def _draw_nodes_wind(ax):
        xsn = [coords[n][0] for n in nodes]
        ysn = [coords[n][1] for n in nodes]
        ax.scatter(
            xsn, ysn,
            color="white",
            s=120,
            zorder=6,
            edgecolors="black",
            linewidths=1.2
        )
        for n, (cx, cy) in coords.items():
            ax.text(
                cx + 0.01 * x_range,
                cy + 0.01 * x_range,
                str(n),
                fontsize=9,
                color="yellow",
                fontweight="bold",
                zorder=7
            )

    def _draw_arcs_wind(ax, arcs, highlight_undir, base_color):
        for (i, j) in arcs:
            xi, yi = coords[i]
            xj, yj = coords[j]
            in_I = highlight_undir is not None and canon_edge(i, j) in highlight_undir
            color = "crimson" if in_I else base_color
            lw = 5.0 if in_I else 3.5
            ms = 28 if in_I else 14
            ax.annotate(
                "",
                xy=(xj, yj),
                xytext=(xi, yi),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=lw,
                    mutation_scale=ms
                ),
                zorder=3
            )

    def _draw_reserved_not_used_wind(ax, reserved_edges, tour_arcs):
        if not reserved_edges:
            return

        tour_edges = {canon_edge(i, j) for (i, j) in tour_arcs}

        for (i, j) in sorted({canon_edge(*e) for e in reserved_edges}):
            if canon_edge(i, j) in tour_edges:
                continue
            xi, yi = coords[i]
            xj, yj = coords[j]
            ax.plot(
                [xi, xj], [yi, yj],
                color="crimson",
                linestyle="--",
                linewidth=3.0,
                alpha=0.65,
                zorder=2
            )

    # Confronto per scenario
    for sid in scenario_ids:
        fig, axes = plt.subplots(2, 2, figsize=(26, 22))
        fig.patch.set_facecolor("#0d1117")
        ws_name = results[sid].get("wind_name", _wind_scenario_for_id(sid, wind_scenarios).name)
        fig.suptitle(
            f"Scenario {sid} — {ws_name} — PI / EEV / STO",
            fontsize=16,
            fontweight="bold",
            color="white",
            y=1.01
        )

        if eev_solutions is not None and sid in eev_solutions:
            eev_arcs = eev_solutions[sid]["arcs"]
            eev_tour = eev_solutions[sid]["tour"]
            eev_reserved_not_used = eev_solutions[sid].get("reserved_not_used", [])
            eev_label = f"EEV: ricorso con x^EV fissato (scenario {sid})"
        else:
            eev_arcs = arcs_medio
            eev_tour = tour_medio
            eev_reserved_not_used = []
            eev_label = f"Medione / EEV (scenario {sid})"

        sto_reserved_not_used = res_stoch["scenario_solutions"][sid].get("reserved_not_used", [])

        panels = [
            {
                "label": "PI libero",
                "arcs": results[sid]["exact_free"]["arcs"],
                "tour": results[sid]["exact_free"]["tour"],
                "cost": results[sid]["exact_free"]["length"],
                "color": "#2E7D32",
                "hl": set(),
                "reserved_not_used": [],
            },
            {
                "label": eev_label,
                "arcs": eev_arcs,
                "tour": eev_tour,
                "cost": eev_costs[sid],
                "color": "#6A1B9A",
                "hl": x_ev,
                "reserved_not_used": eev_reserved_not_used,
            },
            {
                "label": "Stocastico / RP",
                "arcs": res_stoch["scenario_solutions"][sid]["y_used"],
                "tour": res_stoch["scenario_solutions"][sid]["tour"],
                "cost": stoch_costs[sid],
                "color": "#E65100",
                "hl": x_used_set,
                "reserved_not_used": sto_reserved_not_used,
            },
        ]

        for panel, (r, c) in zip(panels, [(0, 0), (0, 1), (1, 0)]):
            ax = axes[r][c]
            _setup_wind_ax(ax, sid)
            _draw_nodes_wind(ax)
            _draw_arcs_wind(ax, panel["arcs"], panel["hl"], panel["color"])
            _draw_reserved_not_used_wind(ax, panel.get("reserved_not_used", []), panel["arcs"])

            tour = panel["tour"]
            tour_str = " → ".join(str(n) for n in tour) + f" → {tour[0]}" if tour else "n.d."
            cost_str = f"{panel['cost']:.4f}" if panel["cost"] is not None else "N/A"

            handles = [mpatches.Patch(color=panel["color"], label="Arco tour")]

            if panel["hl"]:
                tour_edges = {canon_edge(i, j) for (i, j) in panel["arcs"]}
                highlighted_edges = {canon_edge(*e) for e in panel["hl"]}

                if highlighted_edges & tour_edges:
                    handles.append(
                        mpatches.Patch(
                            color="crimson",
                            label="Tratta prenotata e percorsa"
                        )
                    )

            if panel.get("reserved_not_used", []):
                handles.append(
                    Line2D(
                        [0], [0],
                        color="crimson",
                        lw=2.5,
                        linestyle="--",
                        label="Tratta prenotata non percorsa"
                    )
                )

            ax.legend(
                handles=handles,
                loc="upper left",
                fontsize=8,
                facecolor="#1a1f2e",
                edgecolor="#555",
                labelcolor="white"
            )

            ax.set_title(
                f"{panel['label']}\nCosto: {cost_str}\nTour: {tour_str}",
                fontsize=8,
                loc="left",
                pad=8,
                family="monospace",
                color="white"
            )
            ax.set_xlabel("x", color="white")
            ax.set_ylabel("y", color="white")

        # Quarto pannello lasciato vuoto: il confronto usa solo PI, EEV e STO.
        axes[1][1].axis("off")
        axes[1][1].set_facecolor("#0d1117")

        plt.tight_layout()
        fname = out_path(f"{exp_name}_scenario_{sid}_confronto.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Salvato: {fname}")

    # Plot overview campi di vento
    fig0, axes0 = plt.subplots(2, 2, figsize=(14, 12))
    fig0.patch.set_facecolor("#0d1117")
    for ax, sid in zip(axes0.flatten(), scenario_ids):
        _setup_wind_ax(ax, sid)
        ws = _wind_scenario_for_id(sid, wind_scenarios)
        ax.set_title(f"Scenario {sid} | {ws.name}", color="white", fontsize=10, pad=8)
    fig0.suptitle("Campi vettoriali di vento", color="white", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fname = out_path(f"{exp_name}_wind_overview.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig0.get_facecolor())
    plt.close()
    print(f"  Salvato: {fname}")


# PROVA NEUR

# DIPENDENZE (già presenti nel notebook):
#   canon_edge, build_perturbation, build_scenario_dist,
#   solve_reservation_tsp, get_edge_value, base_cost_undirected,
#   build_I_from_medoid_outgoing_nodes,
#   N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
#   SCENARIO_IDS, PRENOTAZIONE_FRAC, PENALTY_FRAC,
#   K_MEDOID_NODES, MAX_KMEDOID_I_ARCS, KMEDOID_ARCS_PER_NODE
#
# LOGICA GENERALE:
#   1. Genera dataset: per ogni campione, campiona x ∈ {0,1}^|I|
#      e K' scenari, risolve il secondo stadio, calcola E[Q(x,ξ)]
#   2. Addestra ReLUNetworkExpected su (x, {ξ_k}) → E[Q(x,ξ)]
#   3. Usa la NN come oracolo: enumera le 2^|I| soluzioni x e prende
#      quella che minimizza c^T x + NN-E(x, scenari_test)
#   4. Valuta la soluzione out-of-sample e confronta con STO/EEV




#  Architettura ReLUNetworkExpected (network.py)
#   (adattata per batch senza padding (K fisso per campione))

class ReLUNetworkExpected(nn.Module):
    """
        Mappa:  (x, {ξ_k}_{k=1}^{K'}) ──► E_ξ[Q(x, ξ)]

    Pipeline:
      Ψ1 : ogni ξ_k  →  embedding intermedio   (per-scenario)
      ⊕   : mean-aggregation su K' embeddings
      Ψ2 : aggregato →  embedding finale ξ_λ
      ΦE  : cat(x, ξ_λ) →  valore atteso secondo stadio

    Input:
      x_fs   : (batch, fs_input_dim)        — decisioni primo stadio (binarie)
      x_scen : (batch, K', ss_input_dim)    — feature degli scenari
    Output:
      (batch, 1)  — predizione di E[Q(x,ξ)]
    """

    def __init__(self, fs_input_dim, ss_input_dim, ss_hidden_dim, ss_embed_dim1, ss_embed_dim2,
                 relu_hidden_dim, dropout=0.0, agg_type="mean"):
        super().__init__()
        self.dropout  = dropout
        self.agg_type = agg_type

        # Ψ1: scenario encoder  (ss_input_dim → ss_hidden_dim → ss_embed_dim1)
        self.scen_input  = nn.Linear(ss_input_dim,  ss_hidden_dim,  bias=False)
        self.scen_embed1 = nn.Linear(ss_hidden_dim, ss_embed_dim1,  bias=False)
        # Ψ2: post-aggregation  (ss_embed_dim1 → ss_embed_dim2)
        self.scen_embed2 = nn.Linear(ss_embed_dim1, ss_embed_dim2)

        # ΦE: feed-forward finale  (fs_input_dim + ss_embed_dim2 → 1)
        self.relu_input  = nn.Linear(fs_input_dim + ss_embed_dim2, relu_hidden_dim)
        self.relu_output = nn.Linear(relu_hidden_dim, 1)


    def _embed_scenarios(self, x_scen):
        """
        x_scen : (batch, K', ss_input_dim)
        return  : (batch, ss_embed_dim2)
        """
        x = F.relu(self.scen_input(x_scen))        # (B, K', ss_hidden)
        if self.dropout and self.training:
            x = F.dropout(x, p=self.dropout)

        x = F.relu(self.scen_embed1(x))             # (B, K', ss_embed1)
        if self.dropout and self.training:
            x = F.dropout(x, p=self.dropout)

        # Aggregazione su K' (dim=1)
        x = torch.mean(x, dim=1) if self.agg_type == "mean" else torch.sum(x, dim=1)
        # → (B, ss_embed1)

        x = F.relu(self.scen_embed2(x))             # (B, ss_embed2)
        if self.dropout and self.training:
            x = F.dropout(x, p=self.dropout)
        return x

    def forward(self, x_fs, x_scen):
        """
        x_fs   : (batch, fs_input_dim)
        x_scen : (batch, K', ss_input_dim)
        return : (batch, 1)
        """
        xi_lambda = self._embed_scenarios(x_scen)          # (B, ss_embed2)
        x = torch.cat([x_fs, xi_lambda], dim=1)            # (B, fs + ss_embed2)
        x = F.relu(self.relu_input(x))
        if self.dropout and self.training:
            x = F.dropout(x, p=self.dropout)
        return self.relu_output(x)                          # (B, 1)



# Encoding delle feature, converto soluzioni 1S in vettore binario
def encode_x_nne(x_set, I):
    x_set_c = {canon_edge(u, v) for (u, v) in x_set} # lista di archi prenotati
    return np.array(
        [1.0 if canon_edge(u, v) in x_set_c else 0.0 for (u, v) in I],
        dtype=np.float32
    )

# Codifico uno scenario ξ come vettore di feature normalizzate (levo la scala assoluta delle distanze)
def encode_scenario_nne(pert_dict, scenario_edges, base_dist):
    """
    Per ogni tratta (u,v) in scenario_edges (ordine canonico) include:
      [Δ(u,v) / b(u,v),  Δ(v,u) / b(v,u)]
    → vettore di lunghezza 2*|I|  (ss_dim)
    """
    feats = []

    for (u, v) in scenario_edges:
        b_uv = base_dist[u][v]
        b_vu = base_dist[v][u]

        d_uv = pert_dict.get((u, v), 0.0) / max(b_uv, 1e-9)
        d_vu = pert_dict.get((v, u), 0.0) / max(b_vu, 1e-9)

        feats.extend([d_uv, d_vu])

    return np.array(feats, dtype=np.float32)




# Generazione del dataset come nella sezione 4.3 paper
def generate_nne_dataset(nodes, E, I, base_dist, root, p, C, env, n_samples=NNE_N_SAMPLES,
                         k_prime=NNE_K_PRIME, data_seed=NNE_DATA_SEED, scen_base_seed=NNE_SCEN_BASE_SEED,
                         mean_frac=MEAN_FRAC, sigma_frac=SIGMA_FRAC, verbose=True, scenario_edges=None, frequent_arcs=None):
    """
        Per ogni campione i:
      1. Campiona x_i ∈ {0,1}^|I|  uniformemente
      2. Campiona k_prime scenari  ξ_1, ..., ξ_{K'}
      3. Per ogni k: risolve il secondo stadio → Q(x_i, ξ_k)
         (tour_cost + penalty, escludendo il costo di prenotazione)
      4. Label y_i = (1/K') * Σ_k  Q(x_i, ξ_k)

    Returns dict con:
      "x_fs"   : (n_samples, |I|)           — feature primo stadio
      "x_scen" : (n_samples, k_prime, 2|I|) — feature scenari
      "y"      : (n_samples,)               — E[Q(x,ξ)] stimato
    """
    rng = np.random.RandomState(data_seed)
    if scenario_edges is None:
        scenario_edges = I

    ss_dim = 2 * len(scenario_edges)

    X_fs, X_scen, Y = [], [], []
    n_failed = 0
    i_sample = 0
    attempt  = 0

    print(f"\nGenerazione dataset NN-E:")
    print(f"  n_samples={n_samples}, K'={k_prime}, |I|={len(I)}, "
      f"|scenario_edges|={len(scenario_edges)}, ss_dim={ss_dim}")

    while i_sample < n_samples:
        attempt += 1

        #  Campiono x casuale
        x_bits = rng.randint(0, 2, size=len(I))           # {0,1}^|I|
        x_set  = [I[j] for j in range(len(I)) if x_bits[j] == 1]

        # seed unico per questo tentativo (lontano da FINAL_SCENARIO_SEED)
        base_seed_attempt = scen_base_seed + attempt * 1000

        scen_feats, scen_costs = [], []
        valid = True

        # Genero K' scenari e risolvo il secondo stadio
        for k in range(1, k_prime + 1):
            pert = build_perturbation( scenario_id=k, nodes=nodes, base_dist=base_dist,
                I=I, n_extra_arcs=N_EXTRA_ARCS, mean_frac=mean_frac,
                sigma_frac=sigma_frac, base_seed=base_seed_attempt, frequent_arcs=frequent_arcs)
            scen_dist = build_scenario_dist(base_dist, pert)

            sol = solve_reservation_tsp(nodes, E, I, scen_dist, root, p, C, env,
                fixed_reservations=x_set, output_flag=0)

            if sol["tour_cost"] is None:
                valid = False
                n_failed += 1
                break

            # Q(x, ξ) = costo di ricorso (escluso il costo di prenotazione)
            q = sol["tour_cost"] + sol["penalty_paid"]
            scen_costs.append(q)
            scen_feats.append(encode_scenario_nne(pert, scenario_edges, base_dist))

        if not valid:
            continue

        # Label = E[Q(x,ξ)]
        X_fs.append(x_bits.astype(np.float32))
        X_scen.append(np.stack(scen_feats))    # (k_prime, 2*|I|)
        Y.append(float(np.mean(scen_costs)))
        i_sample += 1

        if verbose and i_sample % 100 == 0:
            print(f"  {i_sample}/{n_samples} campioni  (falliti ignorati: {n_failed})")

    print(f"Dataset generato: {n_samples} campioni, {n_failed} fallimenti saltati.")
    print(f"  Label — media: {np.mean(Y):.3f}  std: {np.std(Y):.3f}  "
          f"min: {np.min(Y):.3f}  max: {np.max(Y):.3f}")

    return {
        "x_fs":   np.stack(X_fs),                            # (N, |I|)
        "x_scen": np.stack(X_scen),                          # (N, K', 2|I|)
        "y":      np.array(Y, dtype=np.float32),             # (N,)
    }


# Divido in train e validation causalmente
def split_nne_dataset(dataset, train_frac=NNE_TRAIN_FRAC, seed=NNE_DATA_SEED):
    n   = len(dataset["y"])
    idx = np.random.RandomState(seed).permutation(n)
    n_tr = int(n * train_frac)
    tr, val = idx[:n_tr], idx[n_tr:]

    out = {}
    for key, arr in dataset.items():
        out[f"{key}_tr"]  = arr[tr]
        out[f"{key}_val"] = arr[val]
    return out



# Training di ReLUNetworkExpected e ottengo in miglior modello (minor MAE sul val)
# Stampo anche lista metriche per epoca
def train_nne_model(split, I, embed_hidden_dim = NNE_EMBED_HIDDEN_DIM,
                    embed_dim1= NNE_EMBED_DIM1, embed_dim2 = NNE_EMBED_DIM2,
                     relu_hidden_dim = NNE_RELU_HIDDEN_DIM, lr = NNE_LR,
                     batch_size = NNE_BATCH_SIZE, n_epochs = NNE_N_EPOCHS,
                     log_freq = NNE_LOG_FREQ, dropout = NNE_DROPOUT, agg_type = NNE_AGG_TYPE):

    fs_dim = split["x_fs_tr"].shape[-1]
    ss_dim = split["x_scen_tr"].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nTraining NN-E  |  device={device}  fs_dim={fs_dim}  ss_dim={ss_dim}")
    print(f"  Architettura: Ψ1({ss_dim}→{embed_hidden_dim}→{embed_dim1}) "
          f"Ψ2(→{embed_dim2}) ΦE({fs_dim}+{embed_dim2}→{relu_hidden_dim}→1)")

    # Costruisce la rete  
    model = ReLUNetworkExpected( fs_input_dim=fs_dim, ss_input_dim=ss_dim,
        ss_hidden_dim=embed_hidden_dim, ss_embed_dim1=embed_dim1,
        ss_embed_dim2=embed_dim2, relu_hidden_dim=relu_hidden_dim,dropout=dropout, agg_type=agg_type, ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn   = nn.MSELoss()

    # Tensori  
    def t(arr):
        return torch.from_numpy(arr).float().to(device)

    x_fs_tr    = t(split["x_fs_tr"])
    x_scen_tr  = t(split["x_scen_tr"])
    y_tr       = t(split["y_tr"]).reshape(-1, 1)
    x_fs_val   = t(split["x_fs_val"])
    x_scen_val = t(split["x_scen_val"])
    y_val      = t(split["y_val"]).reshape(-1, 1)

    loader_tr = DataLoader(
        TensorDataset(x_fs_tr, x_scen_tr, y_tr),
        batch_size=batch_size, shuffle=True
    )

    # Loop di training     
    best_val_mae = float("inf")
    best_state   = None
    history      = {"tr_loss": [], "val_mae": [], "val_mse": [], "val_mape": []}

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_losses = []
        for xfs_b, xsc_b, y_b in loader_tr:
            preds = model(xfs_b, xsc_b)
            loss  = loss_fn(preds, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())
        history["tr_loss"].append(float(np.mean(ep_losses)))

        # Validazione periodica    
        if epoch % log_freq == 0:
            model.eval()
            with torch.no_grad():
                pv   = model(x_fs_val, x_scen_val).cpu().numpy().reshape(-1)
                yv   = y_val.cpu().numpy().reshape(-1)
            mae  = float(np.mean(np.abs(pv - yv)))
            mse  = float(np.mean((pv - yv) ** 2))
            mape = float(np.mean(np.abs((pv - yv) / (np.abs(yv) + 1e-9))))
            history["val_mae"].append(mae)
            history["val_mse"].append(mse)
            history["val_mape"].append(mape)

            marker = ""
            if mae < best_val_mae:
                best_val_mae = mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                marker = "  - miglior modello"

            print(f"  Ep {epoch:4d}/{n_epochs} | "
                  f"loss={history['tr_loss'][-1]:.4f} | "
                  f"val_MAE={mae:.4f} | val_MAPE={mape:.3%}{marker}")

    elapsed = time.time() - t0
    print(f"\nTraining completato in {elapsed:.1f}s | Miglior val_MAE = {best_val_mae:.4f}")

    # Ricarica il miglior modello
    model.load_state_dict(best_state)
    model.eval()
    return model, history


# Calcolo MAE, RMSE, MAPE su train e val

def evaluate_nne_model(model, split, device=None):
    if device is None:
        device = next(model.parameters()).device

    def t(arr):
        return torch.from_numpy(arr).float().to(device)

    model.eval()
    results = {}
    for name in ("tr", "val"):
        with torch.no_grad():
            y_pred = model(t(split[f"x_fs_{name}"]),
                           t(split[f"x_scen_{name}"])).cpu().numpy().reshape(-1)
        y_true = split[f"y_{name}"]
        mae  = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mape = float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-9))))
        corr = float(np.corrcoef(y_pred, y_true)[0, 1])
        results[name] = dict(mae=mae, rmse=rmse, mape=mape, corr=corr)
        print(f"  [{name:3s}]  MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"MAPE={mape:.3%}  r={corr:.4f}")
    return results


# Soluzione surrogata via NN-E 
def solve_surrogate_nne(model, I, p, base_dist, test_perturbations, scenario_probs=None,
                        chunk_size=256, scenario_edges=None):
    """
    Trova x* ∈ {0,1}^|I| che minimizza il modello surrogato:

        min_{x}  c^T x  +  NN-E(x, {ξ_k}_{k=1}^K)

    Enumera tutte le 2^|I| combinazioni (pratico per |I| ≤ ~18).

    test_perturbations : lista di K dizionari perturbazione (scenari di test)
    scenario_probs     : pesi degli scenari (default: uniformi)

    Returns:
      best_x_set   : lista di tratte prenotate
      best_obj     : valore obiettivo surrogato
      all_combos   : lista di (x_bits, booking_cost, nne_pred, obj)
    """
    device  = next(model.parameters()).device
    n_I     = len(I)
    K       = len(test_perturbations)
    probs   = scenario_probs if scenario_probs is not None else [1.0 / K] * K
    if scenario_edges is None:
        scenario_edges = sorted({
            canon_edge(u, v)
            for u in base_dist
            for v in base_dist
            if u != v
        })
    # Feature degli scenari — uguali per tutte le x
    scen_feats = np.stack([
        encode_scenario_nne(pert, scenario_edges, base_dist)
        for pert in test_perturbations
    ])  # (K, ss_dim)

    x_scen_t = torch.from_numpy(scen_feats).float().unsqueeze(0).to(device)
    # shape: (1, K, ss_dim)

    # Tutti i 2^|I| vettori x
    n_combos  = 2 ** n_I
    all_x_bits = np.array(
        [((np.arange(n_I, dtype=np.int32) >> 0) & ((i >> np.arange(n_I, dtype=np.int32)) & 1) != 0)
         if False else ((i >> np.arange(n_I, dtype=np.int32)) & 1)
         for i in range(n_combos)],
        dtype=np.float32
    )  # (n_combos, n_I)

    # Costi di prenotazione per ciascuna combo
    booking_costs = np.array([
        sum(get_edge_value(p, I[j][0], I[j][1]) * all_x_bits[i, j] for j in range(n_I))
        for i in range(n_combos)
    ], dtype=np.float32)

    # Predizione NN-E in batch
    model.eval()
    nne_preds = np.zeros(n_combos, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_combos, chunk_size):
            end    = min(start + chunk_size, n_combos)
            xfs_b  = torch.from_numpy(all_x_bits[start:end]).to(device)
            xsc_b  = x_scen_t.expand(end - start, -1, -1)
            nne_preds[start:end] = model(xfs_b, xsc_b).cpu().numpy().reshape(-1)

    objectives = booking_costs + nne_preds
    best_idx   = int(np.argmin(objectives))

    best_x_bits = all_x_bits[best_idx]
    best_x_set  = [I[j] for j in range(n_I) if best_x_bits[j] > 0.5]
    best_obj    = float(objectives[best_idx])

    all_combos = [
        (all_x_bits[i], float(booking_costs[i]), float(nne_preds[i]), float(objectives[i]))
        for i in range(n_combos)
    ]

    print(f"\nSoluzione NN-E surrogata (su {n_combos} combo, {K} scenari test):")
    print(f"  Tratte prenotate : {best_x_set}")
    booking = float(booking_costs[best_idx])
    print(f"  Costo booking    : {booking:.4f}")
    print(f"  NN-E E[Q(x,ξ)]  : {float(nne_preds[best_idx]):.4f}")
    print(f"  Obiettivo surr.  : {best_obj:.4f}")
    return best_x_set, best_obj, all_combos


# Valutazione out-of-sample della soluzione NN-E 
# Valuto la soluzione x_nne risolvendo il secondo stadio vero per ogni scenario di test
def evaluate_nne_solution(x_nne, nodes, E, I, base_dist, root, p, C, env,
                           scenario_perturbations, scenario_probs=None):
    
    K = len(scenario_perturbations)
    probs = scenario_probs if scenario_probs is not None else [1.0 / K] * K

    booking_cost = sum(get_edge_value(p, u, v) for (u, v) in x_nne)

    recourse = []
    for pert in scenario_perturbations:
        dist = build_scenario_dist(base_dist, pert)
        sol  = solve_reservation_tsp(
            nodes, E, I, dist, root, p, C, env,
            fixed_reservations=x_nne, output_flag=0
        )
        if sol["tour_cost"] is None:
            recourse.append(float("inf"))
        else:
            recourse.append(sol["tour_cost"] + sol["penalty_paid"])

    exp_recourse = sum(probs[k] * recourse[k] for k in range(K))
    total        = booking_cost + exp_recourse

    return {
        "total_expected": total, # costo atteso , primo stadio + valore atteso funzione di ricorso
        "booking_cost": booking_cost, # costo di prenotazione
        "expected_recourse": exp_recourse, # valore atteso della funzione di ricorso
        "recourse_per_scen": recourse, # costi di Q per ogni scenario
        }

# Pipeline completa: run_esperimento_B_NNE
# Uso gli stessi scenari di run_esperimento_B (passati via res_B)
def run_esperimento_B_NNE(nodes, coords, base_dist, E, root, env, res_B, salva_modello=False):
    
    print("\n" + "=" * 70)
    print("ESPERIMENTO B — NN-E  (Neur2SP, NeurIPS 2022)")
    # Controllo di avere tutto per la stampa finale
    if res_B is None:
        raise ValueError(
            "Per eseguire B_NNE devi prima eseguire B. "
            "Imposta ESPERIMENTI_DA_ESEGUIRE = ['B', 'B_NNE']."
        )

    chiavi_necessarie = [
        "I", "b", "p", "C",
        "results", "scenario_probs",
        "stoch_solutions",
        "stoch_costs", "stoch_tour_costs", "stoch_penalty_costs",
        "eev_costs", "eev_tour_costs", "eev_penalty_costs",
        "eev_solutions",
        "tour_medio", "arcs_medio",
        "x_ev", "x_used_sto",
        "PI", "STO", "EEV"
    ]

    mancanti = [k for k in chiavi_necessarie if k not in res_B]

    if mancanti:
        raise KeyError(f"res_B non contiene queste chiavi necessarie: {mancanti}")
    # Recupera dati da res_B  
    I = res_B["I"]
    b = res_B["b"]
    p = res_B["p"]
    C = res_B["C"]
    scenario_edges = sorted({  canon_edge(i, j)
                            for sid in res_B["results"]
                            for (i, j) in res_B["results"][sid]["pert"].keys()
    })
    results = res_B["results"] # stessi scenari di STO/EEV/PI
    scenario_probs = res_B["scenario_probs"]
    scenario_ids = list(results.keys())
    probs = [scenario_probs[sid] for sid in scenario_ids]
 
    # Perturbazioni degli scenari condivisi (usate da encode e valutazione)
    test_perts = [results[sid]["pert"] for sid in scenario_ids]
 
    print(f"\nSetup:  |I|={len(I)}  scenari={len(scenario_ids)}  "
      f"fs_dim={len(I)}  |scenario_edges|={len(scenario_edges)}  "
      f"ss_dim={2 * len(scenario_edges)}")
 
    # Generazione dataset  
    print(" Generazione dataset")
    t0      = time.time()
    dataset = generate_nne_dataset(
        nodes=nodes, E=E, I=I, base_dist=base_dist, root=root,
        p=p, C=C, env=env,scenario_edges=scenario_edges, frequent_arcs=res_B["frequent_arcs"])
    split   = split_nne_dataset(dataset)
    print(f"  {time.time()-t0:.1f}s  |  "
          f"train={len(split['y_tr'])}  val={len(split['y_val'])}")
 
    # Training  
    print("  Training NN-E")
    model, history = train_nne_model(split, I)
 
    if salva_modello:
        path = out_path("nne_model_espB.pt")
        torch.save(model.state_dict(), path)
        print(f"  Modello salvato: {path}")
 
    # Metriche di apprendimento  
    print("Metriche di apprendimento: ")
    metriche = evaluate_nne_model(model, split)
 
    # Soluzione surrogata  
    print(" Soluzione surrogata NN-E sugli scenari condivisi")
    x_nne, obj_surr, _ = solve_surrogate_nne(
        model=model, I=I, p=p,
        base_dist=base_dist,
        test_perturbations=test_perts,
        scenario_probs=probs, scenario_edges=scenario_edges)
 
    # Valutazione out-of-sample per ogni scenario  
    print("Valutazione out-of-sample soluzione NN-E")
    nne_costs, nne_tour_costs, nne_penalty_costs = {}, {}, {}
    nne_solutions = {}   # usato da plot_scenario_comparison
 
    reservation_nne = sum(get_edge_value(p, u, v) for (u, v) in x_nne)
 
    for sid, pert in zip(scenario_ids, test_perts):
        dist = build_scenario_dist(base_dist, pert)
        sol  = solve_reservation_tsp(
            nodes, E, I, dist, root, p, C, env,
            fixed_reservations=x_nne, output_flag=0)
 
        if sol["tour_cost"] is None:
            # caso degenere: imposto infinito
            nne_costs[sid]         = float("inf")
            nne_tour_costs[sid]    = float("inf")
            nne_penalty_costs[sid] = 0.0
            nne_solutions[sid]     = {"arcs": [], "tour": [],
                                      "tour_cost": float("inf"),
                                      "penalty_paid": 0.0,
                                      "reservation_paid": reservation_nne}
        else:
            tc  = sol["tour_cost"]
            pc  = sol["penalty_paid"]
            nne_costs[sid]         = reservation_nne + tc + pc
            nne_tour_costs[sid]    = tc
            nne_penalty_costs[sid] = pc
            nne_solutions[sid]     = {
                "arcs":             sol["arcs"],
                "tour":             sol["tour"],
                "tour_cost":        tc,
                "penalty_paid":     pc,
                "reservation_paid": reservation_nne,
            }
 
    NNE_val = sum(scenario_probs[sid] * nne_costs[sid] for sid in scenario_ids)
 
    #   Riepilogo numerico  
    print("\n" + "─" * 60)
    print("RIEPILOGO ESPERIMENTO B — NN-E")
    print(f"  Tratte prenotate da NN-E  : {x_nne}")
    print(f"  Costo prenotazioni             : {reservation_nne:.4f}")
    print(f"  E[Q(x,ξ)] reale           : {NNE_val - reservation_nne:.4f}")
    print(f"  Costo totale NNE (atteso) : {NNE_val:.4f}")
    print(f"  Costo totale STO          : {res_B['STO']:.4f}")
    print(f"  Costo totale EEV          : {res_B['EEV']:.4f}")
    gap_sto = (NNE_val - res_B["STO"]) / res_B["STO"] * 100
    gap_eev = (NNE_val - res_B["EEV"]) / res_B["EEV"] * 100
    print(f"  Gap NNE vs STO            : {gap_sto:+.2f}%")
    print(f"  Gap NNE vs EEV            : {gap_eev:+.2f}%")
    print("─" * 60)
 
    # Salvo su .txt (aggiunge NNE al file dell'esperimento B) 
    print_and_save_summary(
        "espB_NNE", scenario_ids, results,

        res_B["eev_costs"],
        res_B["eev_tour_costs"],
        res_B["eev_penalty_costs"],

        res_B["stoch_costs"],
        res_B["stoch_tour_costs"],
        res_B["stoch_penalty_costs"],

        res_B["PI"],
        res_B["STO"],
        res_B["EEV"],

        I=I,
        p=p,
        C=C,
        b=b,

        stoch_solver_info=res_B.get("stoch_solver_info"),
        frequent_arcs=res_B.get("frequent_arcs"),
        total_random_uses=res_B.get("total_random_uses"),
        random_impact_stats=res_B.get("random_impact_stats"),

        nne_costs=nne_costs,
        nne_tour_costs=nne_tour_costs,
        nne_penalty_costs=nne_penalty_costs,

        nne_metrics=metriche,
        nne_history=history,
        nne_extra={
            "x_nne": x_nne,
            "obj_surr": obj_surr,
            "reservation_nne": reservation_nne,
            "NNE_val": NNE_val,
        },

        scenario_probs=scenario_probs
    )
 
    # Grafici: un plot per scenario con 4 pannelli  
    x_nne_set = set(canon_edge(u, v) for (u, v) in x_nne)
 
    # Ricreo res_stoch nel formato atteso da plot_scenario_comparison cosi posso metterlo in un pannello
    res_stoch_real = res_B.get("res_stoch")
    if res_stoch_real is None:
        res_stoch_real = {"scenario_solutions": res_B["stoch_solutions"]}
    # usa le soluzioni STO reali se disponibili dentro res_B
    # (run_esperimento_B le salva in results solo se hai run_esperimento_B completo)
    # → passa res_B["stoch_solutions"] se l'hai aggiunto a run_esperimento_B,
    #   altrimenti il pannello STO mostrerà il PI come approssimazione visiva.
    # Per avere il tour STO corretto aggiungi anche "stoch_solutions" al return
    # di run_esperimento_B (vedi nota in fondo al file).
 
    for sid in scenario_ids:
        plot_scenario_comparison(  "espB_NNE", sid, nodes, coords, results, res_B["tour_medio"], res_B["arcs_medio"],
            res_B["eev_costs"], res_stoch_real, res_B["stoch_costs"], res_B["x_ev"],
            res_B["x_used_sto"], res_B["PI"], res_B["EEV"], res_B["STO"], stoch_tour_costs=res_B["stoch_tour_costs"],
            stoch_penalty_costs=res_B["stoch_penalty_costs"], eev_solutions=res_B["eev_solutions"],
            nne_solution=nne_solutions.get(sid), x_nne=list(x_nne_set), nne_costs=nne_costs, save=True)
 
    return {
        "model":        model,
        "history":      history,
        "metriche":     metriche,
        "x_nne":        x_nne,
        "obj_surr":     obj_surr,
        "nne_costs":    nne_costs,
        "nne_solutions": nne_solutions,
        "NNE_val":      NNE_val,
        "split":        split,
        "stoch_solutions": res_B.get("stoch_solutions", {}),
    }



# Esecuzione 
#
# Decommenta la riga che ti serve e passa i risultati STO
# se vuoi il confronto diretto.
#
# Opzione 1: solo NN-E
#   res_B_nne = run_esperimento_B_NNE(nodes, coords, base_dist, E, root, env)
#
# Opzione 2: NN-E + confronto con STO (passa i risultati di exp B già calcolati)
# res_B_nne = run_esperimento_B_NNE(nodes, coords, base_dist, E, root, env,res_sto=res_B["results"])
#
# Opzione 3: salva il modello su disco
#   res_B_nne = run_esperimento_B_NNE(nodes, coords, base_dist, E, root, env,
#                                      salva_modello=True)



# SELEZIONE DEGLI ESPERIEMENTI

ESPERIMENTI = {
    "A": ("I manuali, C fisso (0.50), pert N(40%, 10%)", run_esperimento_A),
    "B": ("I dagli archi uscenti dai nodi medoidi, C fisso (0.50), pert N(40%, 20%)", run_esperimento_B),
    "C": ("I dagli archi uscenti dai nodi medoidi, C frequenziale sui PI finali",     run_esperimento_C),
    "D": ("I da aree problematiche, C fisso (0.50)",                                  run_esperimento_D),
    "E": ("I dagli archi uscenti dai nodi medoidi, C fisso (0.50), grafici vento",   run_esperimento_E),
    "B_NNE": ("prova neur", run_esperimento_B_NNE),
    "B_UTSP": ("prova neur", run_esperimento_B_UTSP),
}


def menu_selezione():
    scelti_raw = ESPERIMENTI_DA_ESEGUIRE
    if any(e.upper() == "ALL" for e in scelti_raw):
        return list(ESPERIMENTI.keys())
    scelti = []
    for e in scelti_raw:
        t = e.upper()
        if t in ESPERIMENTI:
            if t not in scelti:
                scelti.append(t)
        else:
            print(f"  Avviso: '{t}' non è un esperimento valido, ignorato.")
    return scelti


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scelti = menu_selezione()

    if not scelti:
        print("Nessun esperimento valido. Modifica ESPERIMENTI_DA_ESEGUIRE.")
        return {}

    print(f"\nEsperimenti da eseguire: {', '.join(scelti)}")

    env = load_env()
    nodes, coords, base_dist, E, root = load_data()
    print(f"Dati caricati: {len(nodes)} nodi, {len(E)} archi orientati.")

    risultati = {}

    for key in scelti:
        _, run_fn = ESPERIMENTI[key]

        try:
            if key == "B_UTSP":
                # Se B è stato eseguito prima, passo i suoi risultati per il confronto
                risultati[key] = run_fn(
                    nodes, coords, base_dist, E, root, env,
                    res_B=risultati.get("B")
                )
            else:
                risultati[key] = run_fn(nodes, coords, base_dist, E, root, env)

        except Exception as ex:
            print(f"\nERRORE nell'esperimento {key}: {ex}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Esecuzione completata.")

    return risultati


if __name__ == "__main__":
    risultati = main()
