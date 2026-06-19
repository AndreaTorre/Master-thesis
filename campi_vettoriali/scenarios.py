# -*- coding: utf-8 -*-
import random
from collections import Counter

from config import (
    N_CALIBRATION_SCENARIOS, MIN_FREQ_FREQUENT, N_FREQUENT_ARCS,
    CALIBRATION_SCENARIO_SEED, N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC,
)
from tsp_utils import canon_edge, all_undirected_edges, base_cost_undirected
from gurobi_models import solve_exact_tsp

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

def build_scenario_dist(base_dist, perturb_dict):
    sd = {i: {j: base_dist[i][j] for j in base_dist[i]} for i in base_dist}
    for (i, j), delta in perturb_dict.items():
        sd[i][j] = base_dist[i][j] + delta
    return sd

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


# ---------------------------------------------------------------------------
# BATCH PER TRAINING UTSP
# ---------------------------------------------------------------------------

def generate_scenario_batches(
    nodes, E, base_dist, I, frequent_arcs,
    n_extra_arcs, mean_frac, sigma_frac, base_seed,
    root, env, p, C,
    n_training_scenarios, n_validation_scenarios,
):
    """
    Genera n_training_scenarios scenari e li divide in batch
    da n_validation_scenarios scenari ciascuno.
    """
    n_batches = n_training_scenarios // n_validation_scenarios

    all_ids = list(range(1, n_training_scenarios + 1))
    print(f"\nGenerazione {n_training_scenarios} scenari di training "
          f"→ {n_batches} batch da {n_validation_scenarios} scenari")

    all_results, all_probs, total_random = generate_scenarios(
        all_ids, nodes, E, base_dist, I, frequent_arcs,
        n_extra_arcs, mean_frac, sigma_frac, base_seed,
        root, env, p, C,
    )

    batches = []
    for b in range(n_batches):
        start = b * n_validation_scenarios
        end   = start + n_validation_scenarios
        batch_ids = all_ids[start:end]

        batch_results = {sid: all_results[sid] for sid in batch_ids}
        batch_probs   = {sid: 1.0 / n_validation_scenarios for sid in batch_ids}

        batches.append({
            "batch_id"      : b,
            "scenario_ids"  : batch_ids,
            "results"       : batch_results,
            "scenario_probs": batch_probs,
        })
        print(f"  Batch {b:02d}: scenari {batch_ids[0]}..{batch_ids[-1]}")

    return batches