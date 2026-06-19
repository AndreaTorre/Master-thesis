# -*- coding: utf-8 -*-
import os
import pickle
import time
import functools

# flush automatico su ogni print
print = functools.partial(print, flush=True)

from config import (
    SCENARIO_IDS, K_MEDOID_NODES, PRENOTAZIONE_FRAC, PENALTY_FRAC,
    N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
    DO_VALIDATION, N_VALIDATION_SCENARIOS,
)
from tsp_utils import base_cost_undirected
from gurobi_models import build_I_from_medoid_outgoing_nodes, solve_stochastic
from scenarios import find_frequent_arcs, generate_scenarios
from evaluation import (
    compute_eev_medione, compute_random_edge_usage_stats,
    print_and_save_summary, validate_policies,
)

# ── Checkpoint helpers ────────────────────────────────────────────
CKPT_DIR = "checkpoints_B"

def _ckpt_path(step):
    return os.path.join(CKPT_DIR, f"ckpt_{step}.pkl")

def _save_ckpt(step, data):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = _ckpt_path(step)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)          # atomico: evita file corrotti
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [CKPT] Salvato {path} ({size_mb:.1f} MB)")

def _load_ckpt(step):
    path = _ckpt_path(step)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"  [CKPT] Caricato {path}")
        return data
    return None

def _last_completed_step():
    """Trova l'ultimo checkpoint completato (0 = nessuno)."""
    for step in range(6, 0, -1):
        if os.path.exists(_ckpt_path(step)):
            return step
    return 0

# ── Funzione principale ──────────────────────────────────────────
def run_esperimento_B(nodes, coords, base_dist, E, root, env):
    exp_name = "espB"
    scenario_ids = SCENARIO_IDS
    t0 = time.time()

    last = _last_completed_step()
    if last > 0:
        print(f"\n{'='*60}")
        print(f"RIPRESA DA CHECKPOINT {last}")
        print(f"{'='*60}")

    print("ESPERIMENTO B: I dagli archi uscenti dai nodi medoidi, C fisso, pert N(40%,20%)")

    # ── STEP 1: Costruzione I, b, p, C ───────────────────────────
    if last >= 1:
        ckpt = _load_ckpt(1)
        I = ckpt["I"]; b = ckpt["b"]; p = ckpt["p"]; C = ckpt["C"]
        medoid_info = ckpt["medoid_info"]
    else:
        print(f"\n[STEP 1/6] Costruzione I dai medoidi...")
        I, medoid_info = build_I_from_medoid_outgoing_nodes(nodes, E, base_dist, K_MEDOID_NODES)
        b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
        p = {(i, j): PRENOTAZIONE_FRAC * b[i, j] for (i, j) in I}
        C = {(i, j): PENALTY_FRAC * b[i, j] for (i, j) in I}
        _save_ckpt(1, {"I": I, "b": b, "p": p, "C": C, "medoid_info": medoid_info})
        print(f"  Completato in {time.time()-t0:.1f}s")

    print(f"\nNODI MEDOIDI FISSATI: {medoid_info['medoid_nodes']}")
    print(f"Tratte k-medoids selezionate in I: {medoid_info['n_undirected_tratte']}")
    print(f"Tratte non orientate in I: {medoid_info['n_undirected_tratte']}")
    print("\nTRATTE SELEZIONATE IN I:")
    for (i, j) in I:
        print(f"  {{{i},{j}}} | b={b[i,j]:.4f} | p={p[i,j]:.4f} | C={C[i,j]:.4f}")

    # ── STEP 2: Archi frequenti (30 TSP di calibrazione) ─────────
    if last >= 2:
        ckpt = _load_ckpt(2)
        frequent_arcs = ckpt["frequent_arcs"]
    else:
        print(f"\n[STEP 2/6] Calibrazione archi frequenti (30 TSP)...")
        t_step = time.time()
        frequent_arcs = find_frequent_arcs(nodes, E, base_dist, root, env, I)
        _save_ckpt(2, {"frequent_arcs": frequent_arcs})
        print(f"  Completato in {time.time()-t_step:.1f}s (totale: {time.time()-t0:.1f}s)")

    # ── STEP 3: Generazione scenari + PI ─────────────────────────
    if last >= 3:
        ckpt = _load_ckpt(3)
        results = ckpt["results"]; scenario_probs = ckpt["scenario_probs"]
        total_random_uses = ckpt["total_random_uses"]
    else:
        print(f"\n[STEP 3/6] Generazione scenari e calcolo PI...")
        t_step = time.time()
        print("\nRISOLUZIONE SCENARI:")
        results, scenario_probs, total_random_uses = generate_scenarios(
            scenario_ids, nodes, E, base_dist, I, frequent_arcs,
            N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
            root=root, env=env, p=p, C=C)
        _save_ckpt(3, {"results": results, "scenario_probs": scenario_probs,
                        "total_random_uses": total_random_uses})
        print(f"  Completato in {time.time()-t_step:.1f}s (totale: {time.time()-t0:.1f}s)")

    # ── STEP 4: EEV ──────────────────────────────────────────────
    if last >= 4:
        ckpt = _load_ckpt(4)
        tour_medio = ckpt["tour_medio"]; arcs_medio = ckpt["arcs_medio"]
        x_ev = ckpt["x_ev"]; eev_costs = ckpt["eev_costs"]
        eev_tour_costs_ev = ckpt["eev_tour_costs_ev"]
        eev_penalty_costs_ev = ckpt["eev_penalty_costs_ev"]
        eev_solutions = ckpt["eev_solutions"]
        EEV = ckpt["EEV"]; PI = ckpt["PI"]
    else:
        print(f"\n[STEP 4/6] Calcolo EEV...")
        t_step = time.time()
        tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, eev_solutions, EEV, PI = compute_eev_medione(
            nodes, E, root, env, base_dist, I, p, C, results, scenario_ids, return_solutions=True)
        _save_ckpt(4, {"tour_medio": tour_medio, "arcs_medio": arcs_medio,
                        "x_ev": x_ev, "eev_costs": eev_costs,
                        "eev_tour_costs_ev": eev_tour_costs_ev,
                        "eev_penalty_costs_ev": eev_penalty_costs_ev,
                        "eev_solutions": eev_solutions, "EEV": EEV, "PI": PI})
        print(f"  EEV = {EEV:.4f}, PI = {PI:.4f}")
        print(f"  Completato in {time.time()-t_step:.1f}s (totale: {time.time()-t0:.1f}s)")

    # ── STEP 5: STO (il passo più pesante) ───────────────────────
    if last >= 5:
        ckpt = _load_ckpt(5)
        res_stoch = ckpt["res_stoch"]
    else:
        print(f"\n[STEP 5/6] Risoluzione modello stocastico (STO)...")
        print(f"  TimeLimit = {43200}s (12h), MIPGap = 0.005")
        print(f"  Inizio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        t_step = time.time()
        scenario_deltas = {sid: results[sid]["pert"] for sid in scenario_ids}
        res_stoch = solve_stochastic(nodes, E, I, base_dist, root, p, C, env,
                                      scenario_deltas, scenario_probs, force_important=False)
        dt = time.time() - t_step
        print(f"  Fine: {time.strftime('%Y-%m-%d %H:%M:%S')} ({dt:.0f}s = {dt/3600:.2f}h)")
        _save_ckpt(5, {"res_stoch": res_stoch})
        print(f"  STO objective = {res_stoch['objective']:.4f}")

    # ── Post-STO: estrazione risultati (veloce) ──────────────────
    STO = res_stoch["objective"]
    stoch_solver_info = res_stoch.get("solver_info", {})
    x_used_set = set(res_stoch["x_used"])
    stoch_costs = {sid: res_stoch["scenario_solutions"][sid]["total_cost"] for sid in scenario_ids}

    STO_recomputed = sum(scenario_probs[sid] * stoch_costs[sid] for sid in scenario_ids)
    print(f"\nControllo STO:")
    print(f"  STO da modello     = {STO:.6f}")
    print(f"  STO ricalcolato    = {STO_recomputed:.6f}")
    print(f"  differenza assoluta = {abs(STO - STO_recomputed):.8f}")

    stoch_tour_c    = {sid: res_stoch["scenario_solutions"][sid]["tour_cost"]  for sid in scenario_ids}
    stoch_penalty_c = {sid: res_stoch["scenario_solutions"][sid]["penalty_paid"] for sid in scenario_ids}

    random_impact_train = compute_random_edge_usage_stats(
        results, scenario_ids,
        stoch_solutions=res_stoch.get("scenario_solutions", {}),
        eev_solutions=eev_solutions)

    print_and_save_summary(
        exp_name, scenario_ids, results, eev_costs, eev_tour_costs_ev,
        eev_penalty_costs_ev, stoch_costs, stoch_tour_c, stoch_penalty_c,
        PI, STO, EEV, I=I, p=p, C=C, b=b,
        stoch_solver_info=stoch_solver_info, frequent_arcs=frequent_arcs,
        total_random_uses=total_random_uses,
        random_impact_stats=random_impact_train)

    # ── STEP 6: Validazione out-of-sample ────────────────────────
    if last >= 6:
        print("\n[STEP 6/6] Validazione già completata (checkpoint trovato)")
    elif DO_VALIDATION:
        print(f"\n[STEP 6/6] Validazione out-of-sample ({N_VALIDATION_SCENARIOS} scenari)...")
        t_step = time.time()
        validate_policies(
            nodes, E, base_dist, root, env, I, p, C,
            x_used_set, x_ev, frequent_arcs, N_VALIDATION_SCENARIOS,
            N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, exp_name=exp_name)
        _save_ckpt(6, {"validation_done": True})
        print(f"  Completato in {time.time()-t_step:.1f}s (totale: {time.time()-t0:.1f}s)")

    print(f"\n{'='*60}")
    print(f"ESPERIMENTO B COMPLETATO in {time.time()-t0:.1f}s ({(time.time()-t0)/3600:.2f}h)")
    print(f"{'='*60}")

    return {
        "I":               I,
        "b":               b,
        "p":               p,
        "C":               C,
        "results":         results,
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