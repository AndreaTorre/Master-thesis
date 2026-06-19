# -*- coding: utf-8 -*-
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

def run_esperimento_B(nodes, coords, base_dist, E, root, env):
    exp_name = "espB"
    scenario_ids = SCENARIO_IDS
    
    print("ESPERIMENTO B: I dagli archi uscenti dai nodi medoidi, C fisso, pert N(40%,20%)")
    
    I, medoid_info = build_I_from_medoid_outgoing_nodes(nodes, E, base_dist, K_MEDOID_NODES)
    #I = [(31, 94), (94, 113), (11,99), (99,11), (99,101), (70,99)]  # canonici: i < j
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
    # Grafici disattivati nel modulo pulito. Se servono, recuperarli da prova_neur.py.
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

# ESPERIMENTO B CON VENTO

# ---------------------------------------------------------------------------
# VERSIONE WIND — aggiungere in fondo a experiment_B.py
# Usa perturbazioni da campo vettoriale ERA5 invece della perturbazione
# sintetica normale. Le vecchie funzioni restano invariate.
# ---------------------------------------------------------------------------

from scenarios import generate_scenarios   # già aggiornata con coords/wind/alpha

def run_esperimento_B_wind(nodes, coords, base_dist, E, root, env, wind, alpha):
    """
    Come run_esperimento_B ma con perturbazioni da campo vettoriale ERA5.
    - Niente find_frequent_arcs: tutti gli archi vengono perturbati dal vento.
    - generate_scenarios riceve coords, wind, alpha.
    - L'interfaccia di ritorno è identica a run_esperimento_B.
    """
    exp_name = "espB_wind"
    scenario_ids = SCENARIO_IDS

    print("ESPERIMENTO B WIND: perturbazioni da campo vettoriale ERA5")

    I, medoid_info = build_I_from_medoid_outgoing_nodes(nodes, E, base_dist, K_MEDOID_NODES)
    b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
    p = {(i, j): PRENOTAZIONE_FRAC * b[i, j] for (i, j) in I}
    C = {(i, j): PENALTY_FRAC      * b[i, j] for (i, j) in I}

    print(f"\nNODI MEDOIDI FISSATI: {medoid_info['medoid_nodes']}")
    print(f"Tratte non orientate in I: {medoid_info['n_undirected_tratte']}")
    print("\nTRATTE SELEZIONATE IN I:")
    for (i, j) in I:
        print(f"  {{{i},{j}}} | b={b[i,j]:.4f} | p={p[i,j]:.4f} | C={C[i,j]:.4f}")

    # Niente find_frequent_arcs: con il vento tutti gli archi sono perturbati
    frequent_arcs = []

    print("\nRISOLUZIONE SCENARI (wind):")
    results, scenario_probs, total_random_uses = generate_scenarios(
        scenario_ids, nodes, E, base_dist, I, frequent_arcs,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
        root=root, env=env, p=p, C=C,
        coords=coords, wind=wind, alpha=alpha,
    )

    tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev, eev_solutions, EEV, PI = compute_eev_medione(
        nodes, E, root, env, base_dist, I, p, C, results, scenario_ids,
        return_solutions=True,
    )

    scenario_deltas = {sid: results[sid]["pert"] for sid in scenario_ids}
    res_stoch = solve_stochastic(
        nodes, E, I, base_dist, root, p, C, env,
        scenario_deltas, scenario_probs, force_important=False,
    )
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
        eev_solutions=eev_solutions,
    )

    print_and_save_summary(
        exp_name, scenario_ids, results,
        eev_costs, eev_tour_costs_ev, eev_penalty_costs_ev,
        stoch_costs, stoch_tour_c, stoch_penalty_c,
        PI, STO, EEV,
        I=I, p=p, C=C, b=b,
        stoch_solver_info=stoch_solver_info,
        frequent_arcs=frequent_arcs,
        total_random_uses=total_random_uses,
        random_impact_stats=random_impact_train,
    )

    if DO_VALIDATION:
        validate_policies(
            nodes, E, base_dist, root, env, I, p, C,
            x_used_set, x_ev, frequent_arcs, N_VALIDATION_SCENARIOS,
            N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC,
            coords=coords, wind=wind, alpha=alpha,
            exp_name=exp_name,
        )

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