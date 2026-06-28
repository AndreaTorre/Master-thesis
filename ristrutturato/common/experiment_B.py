# -*- coding: utf-8 -*-
import os
import pickle
import time
import functools

print = functools.partial(print, flush=True)

from config import (
    SCENARIO_IDS, K_MEDOID_NODES, PRENOTAZIONE_FRAC, PENALTY_FRAC,
    N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
    DO_VALIDATION, N_VALIDATION_SCENARIOS, OUTPUT_DIR,
)
from tsp_utils import base_cost_undirected
from gurobi_models import build_I_from_medoid_outgoing_nodes, solve_stochastic
from scenarios import find_frequent_arcs, generate_scenarios
from evaluation import (
    compute_eev_medione, compute_random_edge_usage_stats,
    print_and_save_summary, validate_policies,
)

# ── Checkpoint helpers ────────────────────────────────────────────

def _ckpt_path(ckpt_dir, step):
    return os.path.join(ckpt_dir, f"ckpt_{step}.pkl")

def _save_ckpt(ckpt_dir, step, data):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = _ckpt_path(ckpt_dir, step)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [CKPT] Salvato {path} ({size_mb:.1f} MB)")

def _load_ckpt(ckpt_dir, step):
    path = _ckpt_path(ckpt_dir, step)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"  [CKPT] Caricato {path}")
        return data
    return None

def _last_completed_step(ckpt_dir, max_step=6):
    for step in range(max_step, 0, -1):
        if os.path.exists(_ckpt_path(ckpt_dir, step)):
            return step
    return 0

# ── ESPERIMENTO B (perturbazioni sintetiche) ─────────────────────

CKPT_DIR_PERT = os.path.join(OUTPUT_DIR, "checkpoint", "B")

def run_esperimento_B(nodes, coords, base_dist, E, root, env):
    exp_name = "espB"
    scenario_ids = SCENARIO_IDS
    t0 = time.time()
    ckpt_dir = CKPT_DIR_PERT

    last = _last_completed_step(ckpt_dir)
    if last > 0:
        print(f"\n{'='*60}")
        print(f"RIPRESA DA CHECKPOINT {last}")
        print(f"{'='*60}")

    print("ESPERIMENTO B: I dagli archi uscenti dai nodi medoidi, C fisso, pert N(40%,20%)")

    # ── STEP 1: Costruzione I, b, p, C ───────────────────────────
    if last >= 1:
        ckpt = _load_ckpt(ckpt_dir, 1)
        I = ckpt["I"]; b = ckpt["b"]; p = ckpt["p"]; C = ckpt["C"]
        medoid_info = ckpt["medoid_info"]
    else:
        print(f"\n[STEP 1/6] Costruzione I dai medoidi...")
        I, medoid_info = build_I_from_medoid_outgoing_nodes(nodes, E, base_dist, K_MEDOID_NODES)
        b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
        p = {(i, j): PRENOTAZIONE_FRAC * b[i, j] for (i, j) in I}
        C = {(i, j): PENALTY_FRAC * b[i, j] for (i, j) in I}
        _save_ckpt(ckpt_dir, 1, {"I": I, "b": b, "p": p, "C": C, "medoid_info": medoid_info})
        print(f"  Completato in {time.time()-t0:.1f}s")

    print(f"\nNODI MEDOIDI FISSATI: {medoid_info['medoid_nodes']}")
    print(f"Tratte k-medoids selezionate in I: {medoid_info['n_undirected_tratte']}")
    print(f"Tratte non orientate in I: {medoid_info['n_undirected_tratte']}")
    print("\nTRATTE SELEZIONATE IN I:")
    for (i, j) in I:
        print(f"  {{{i},{j}}} | b={b[i,j]:.4f} | p={p[i,j]:.4f} | C={C[i,j]:.4f}")

    # ── STEP 2: Archi frequenti (30 TSP di calibrazione) ─────────
    if last >= 2:
        ckpt = _load_ckpt(ckpt_dir, 2)
        frequent_arcs = ckpt["frequent_arcs"]
    else:
        print(f"\n[STEP 2/6] Calibrazione archi frequenti (30 TSP)...")
        t_step = time.time()
        frequent_arcs = find_frequent_arcs(nodes, E, base_dist, root, env, I)
        _save_ckpt(ckpt_dir, 2, {"frequent_arcs": frequent_arcs})
        print(f"  Completato in {time.time()-t_step:.1f}s (totale: {time.time()-t0:.1f}s)")

    # ── STEP 3: Generazione scenari + PI ─────────────────────────
    if last >= 3:
        ckpt = _load_ckpt(ckpt_dir, 3)
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
        _save_ckpt(ckpt_dir, 3, {"results": results, "scenario_probs": scenario_probs,
                                   "total_random_uses": total_random_uses})
        print(f"  Completato in {time.time()-t_step:.1f}s (totale: {time.time()-t0:.1f}s)")

    # ── STEP 4: EEV ──────────────────────────────────────────────
    if last >= 4:
        ckpt = _load_ckpt(ckpt_dir, 4)
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
        _save_ckpt(ckpt_dir, 4, {"tour_medio": tour_medio, "arcs_medio": arcs_medio,
                                   "x_ev": x_ev, "eev_costs": eev_costs,
                                   "eev_tour_costs_ev": eev_tour_costs_ev,
                                   "eev_penalty_costs_ev": eev_penalty_costs_ev,
                                   "eev_solutions": eev_solutions, "EEV": EEV, "PI": PI})
        print(f"  EEV = {EEV:.4f}, PI = {PI:.4f}")
        print(f"  Completato in {time.time()-t_step:.1f}s (totale: {time.time()-t0:.1f}s)")

    # ── STEP 5: STO (il passo più pesante) ───────────────────────
    if last >= 5:
        ckpt = _load_ckpt(ckpt_dir, 5)
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
        _save_ckpt(ckpt_dir, 5, {"res_stoch": res_stoch})
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
        _save_ckpt(ckpt_dir, 6, {"validation_done": True})
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


# ── ESPERIMENTO B WIND (campo vettoriale ERA5) ───────────────────

CKPT_DIR_WIND = os.path.join(OUTPUT_DIR, "checkpoint", "B_wind")

def run_esperimento_B_wind(nodes, coords, base_dist, E, root, env, wind, alpha):
    exp_name = "espB_wind"
    scenario_ids = SCENARIO_IDS
    t0 = time.time()
    ckpt_dir = CKPT_DIR_WIND

    last = _last_completed_step(ckpt_dir, max_step=5)
    if last > 0:
        print(f"\n{'='*60}")
        print(f"RIPRESA DA CHECKPOINT {last}")
        print(f"{'='*60}")

    print("ESPERIMENTO B WIND: perturbazioni da campo vettoriale ERA5")

    # ── STEP 1: Costruzione I, b, p, C ───────────────────────────
    if last >= 1:
        ckpt = _load_ckpt(ckpt_dir, 1)
        I = ckpt["I"]; b = ckpt["b"]; p = ckpt["p"]; C = ckpt["C"]
        medoid_info = ckpt["medoid_info"]
    else:
        print(f"\n[STEP 1/5] Costruzione I dai medoidi...")
        I, medoid_info = build_I_from_medoid_outgoing_nodes(nodes, E, base_dist, K_MEDOID_NODES)
        b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
        p = {(i, j): PRENOTAZIONE_FRAC * b[i, j] for (i, j) in I}
        C = {(i, j): PENALTY_FRAC      * b[i, j] for (i, j) in I}
        _save_ckpt(ckpt_dir, 1, {"I": I, "b": b, "p": p, "C": C, "medoid_info": medoid_info})
        print(f"  Completato in {time.time()-t0:.1f}s")

    print(f"\nNODI MEDOIDI FISSATI: {medoid_info['medoid_nodes']}")
    print(f"Tratte non orientate in I: {medoid_info['n_undirected_tratte']}")
    print("\nTRATTE SELEZIONATE IN I:")
    for (i, j) in I:
        print(f"  {{{i},{j}}} | b={b[i,j]:.4f} | p={p[i,j]:.4f} | C={C[i,j]:.4f}")

    # NOTA: niente find_frequent_arcs — col vento tutti gli archi sono perturbati
    frequent_arcs = []

    # ── STEP 2: Generazione scenari + PI ─────────────────────────
    if last >= 2:
        ckpt = _load_ckpt(ckpt_dir, 2)
        results = ckpt["results"]; scenario_probs = ckpt["scenario_probs"]
        total_random_uses = ckpt["total_random_uses"]
    else:
        print(f"\n[STEP 2/5] Generazione scenari e calcolo PI (wind)...")
        t_step = time.time()
        print("\nRISOLUZIONE SCENARI (wind):")
        results, scenario_probs, total_random_uses = generate_scenarios(
            scenario_ids, nodes, E, base_dist, I, frequent_arcs,
            N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
            root=root, env=env, p=p, C=C,
            coords=coords, wind=wind, alpha=alpha)
        _save_ckpt(ckpt_dir, 2, {"results": results, "scenario_probs": scenario_probs,
                                   "total_random_uses": total_random_uses})
        print(f"  Completato in {time.time()-t_step:.1f}s (totale: {time.time()-t0:.1f}s)")

    # ── STEP 3: EEV ──────────────────────────────────────────────
    if last >= 3:
        ckpt = _load_ckpt(ckpt_dir, 3)
        tour_medio = ckpt["tour_medio"]; arcs_medio = ckpt["arcs_medio"]
        x_ev = ckpt["x_ev"]; eev_costs = ckpt["eev_costs"]
        eev_tour_costs_ev = ckpt["eev_tour_costs_ev"]
        eev_penalty_costs_ev = ckpt["eev_penalty_costs_ev"]
        eev_solutions = ckpt["eev_solutions"]
        EEV = ckpt["EEV"]; PI = ckpt["PI"]
    else:
        print(f"\n[STEP 3/5] Calcolo EEV...")
        t_step = time.time()
        tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, eev_solutions, EEV, PI = compute_eev_medione(
            nodes, E, root, env, base_dist, I, p, C, results, scenario_ids,
            return_solutions=True)
        _save_ckpt(ckpt_dir, 3, {"tour_medio": tour_medio, "arcs_medio": arcs_medio,
                                   "x_ev": x_ev, "eev_costs": eev_costs,
                                   "eev_tour_costs_ev": eev_tour_costs_ev,
                                   "eev_penalty_costs_ev": eev_penalty_costs_ev,
                                   "eev_solutions": eev_solutions, "EEV": EEV, "PI": PI})
        print(f"  EEV = {EEV:.4f}, PI = {PI:.4f}")
        print(f"  Completato in {time.time()-t_step:.1f}s (totale: {time.time()-t0:.1f}s)")

    # ── STEP 4: STO (il passo più pesante) ───────────────────────
    if last >= 4:
        ckpt = _load_ckpt(ckpt_dir, 4)
        res_stoch = ckpt["res_stoch"]
    else:
        print(f"\n[STEP 4/5] Risoluzione modello stocastico (STO)...")
        print(f"  Inizio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        t_step = time.time()
        scenario_deltas = {sid: results[sid]["pert"] for sid in scenario_ids}
        res_stoch = solve_stochastic(
            nodes, E, I, base_dist, root, p, C, env,
            scenario_deltas, scenario_probs, force_important=False)
        dt = time.time() - t_step
        print(f"  Fine: {time.strftime('%Y-%m-%d %H:%M:%S')} ({dt:.0f}s = {dt/3600:.2f}h)")
        _save_ckpt(ckpt_dir, 4, {"res_stoch": res_stoch})
        print(f"  STO objective = {res_stoch['objective']:.4f}")

    # ── Post-STO: estrazione risultati (veloce) ──────────────────
    STO = res_stoch["objective"]
    stoch_solver_info = res_stoch.get("solver_info", {})
    x_used_set    = set(res_stoch["x_used"])
    stoch_costs   = {sid: res_stoch["scenario_solutions"][sid]["total_cost"]   for sid in scenario_ids}
    stoch_tour_c  = {sid: res_stoch["scenario_solutions"][sid]["tour_cost"]    for sid in scenario_ids}
    stoch_penalty_c = {sid: res_stoch["scenario_solutions"][sid]["penalty_paid"] for sid in scenario_ids}

    STO_recomputed = sum(scenario_probs[sid] * stoch_costs[sid] for sid in scenario_ids)
    print(f"\nControllo STO:")
    print(f"  STO da modello      = {STO:.6f}")
    print(f"  STO ricalcolato     = {STO_recomputed:.6f}")
    print(f"  differenza assoluta = {abs(STO - STO_recomputed):.8f}")

    random_impact_train = compute_random_edge_usage_stats(
        results, scenario_ids,
        stoch_solutions=res_stoch.get("scenario_solutions", {}),
        eev_solutions=eev_solutions)

    print_and_save_summary(
        exp_name, scenario_ids, results,
        eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev,
        stoch_costs, stoch_tour_c, stoch_penalty_c,
        PI, STO, EEV, I=I, p=p, C=C, b=b,
        stoch_solver_info=stoch_solver_info, frequent_arcs=frequent_arcs,
        total_random_uses=total_random_uses,
        random_impact_stats=random_impact_train)

    # ── STEP 5: Validazione out-of-sample ────────────────────────
    if last >= 5:
        print("\n[STEP 5/5] Validazione già completata (checkpoint trovato)")
    elif DO_VALIDATION:
        print(f"\n[STEP 5/5] Validazione out-of-sample ({N_VALIDATION_SCENARIOS} scenari)...")
        t_step = time.time()
        validate_policies(
            nodes, E, base_dist, root, env, I, p, C,
            x_used_set, x_ev, frequent_arcs, N_VALIDATION_SCENARIOS,
            N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC,
            coords=coords, wind=wind, alpha=alpha,
            exp_name=exp_name)
        _save_ckpt(ckpt_dir, 5, {"validation_done": True})
        print(f"  Completato in {time.time()-t_step:.1f}s (totale: {time.time()-t0:.1f}s)")

    print(f"\n{'='*60}")
    print(f"ESPERIMENTO B WIND COMPLETATO in {time.time()-t0:.1f}s ({(time.time()-t0)/3600:.2f}h)")
    print(f"{'='*60}")

    return {
        "I":                   I,
        "b":                   b,
        "p":                   p,
        "C":                   C,
        "results":             results,
        "scenario_probs":      scenario_probs,
        "frequent_arcs":       frequent_arcs,
        "x_ev":                x_ev,
        "x_used_sto":          x_used_set,
        "eev_costs":           eev_costs,
        "eev_tour_costs":      eev_tour_costs_ev,
        "eev_penalty_costs":   eev_penalty_costs_ev,
        "eev_solutions":       eev_solutions,
        "stoch_costs":         stoch_costs,
        "stoch_tour_costs":    stoch_tour_c,
        "stoch_penalty_costs": stoch_penalty_c,
        "stoch_solutions":     res_stoch["scenario_solutions"],
        "res_stoch":           res_stoch,
        "stoch_solver_info":   stoch_solver_info,
        "tour_medio":          tour_medio,
        "arcs_medio":          arcs_medio,
        "PI":                  PI,
        "STO":                 STO,
        "EEV":                 EEV,
        "total_random_uses":   total_random_uses,
        "random_impact_stats": random_impact_train,
    }