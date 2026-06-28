# -*- coding: utf-8 -*-
import random
from collections import Counter

from config import (
    N_CALIBRATION_SCENARIOS, MIN_FREQ_FREQUENT, N_FREQUENT_ARCS,
    CALIBRATION_SCENARIO_SEED, N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC,
)
from tsp_utils import canon_edge, all_undirected_edges, base_cost_undirected
from gurobi_models import solve_exact_tsp
from wind_perturbation import build_wind_perturbation

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
    root, env, p, C,
    solve_pi=True,
    coords=None, wind=None, alpha=None, turb_sigma=0.0,
):
    scenario_probs = {s: 1.0 / len(scenario_ids) for s in scenario_ids}
    results = {}
    I_set    = {canon_edge(i, j) for (i, j) in I}
    freq_set = {canon_edge(i, j) for (i, j) in frequent_arcs}
    total_random_uses = 0

    for scenario_id in scenario_ids:
        if coords is not None and wind is not None and alpha is not None:
            pert = build_wind_perturbation(
                scenario_id=scenario_id,
                nodes=nodes,
                base_dist=base_dist,
                coords=coords,
                wind=wind,
                alpha=alpha,
                turb_sigma=turb_sigma,
            )
        else:
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

        if solve_pi:
            exact_free = solve_exact_tsp(
                nodes, E, scenario_dist, root, env,
                fixed_arcs=[], fixed_edges_undir=[], output_flag=0
            )
        else:
            exact_free = {"length": None, "arcs": [], "tour": []}
            
        results[scenario_id] = {
            "pert"         : pert,
            "scenario_dist": scenario_dist,
            "exact_free"   : exact_free,
            "random_edges" : sorted(random_edges),
        }

        if exact_free["length"] is not None:
            print(f"  Scenario {scenario_id} | PI = {exact_free['length']:.4f}")

    return results, scenario_probs, total_random_uses


# ---------------------------------------------------------------------------
# BATCH PER TRAINING / TEST UTSP
# ---------------------------------------------------------------------------

def _chunk_scenario_ids(scenario_ids, batch_size, drop_last=False):
    """
    Divide una lista di id scenario in batch.
    Se drop_last=False, l'ultimo batch può essere più piccolo.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size deve essere positivo, ricevuto {batch_size}")

    batches = []
    for start in range(0, len(scenario_ids), batch_size):
        batch_ids = scenario_ids[start:start + batch_size]
        if drop_last and len(batch_ids) < batch_size:
            continue
        if batch_ids:
            batches.append(batch_ids)
    return batches


def generate_scenario_batches(
    nodes, E, base_dist, I, frequent_arcs,
    n_extra_arcs, mean_frac, sigma_frac, base_seed,
    root, env, p, C,
    n_scenarios=None, batch_size=30,
    scenario_ids=None,
    start_id=1,
    drop_last=False,
    coords=None, wind=None, alpha=None, turb_sigma=0.0,
):
    """
    Genera scenari e li divide in batch.

    Uso classico, compatibile con pert:
        generate_scenario_batches(..., n_scenarios=3000, batch_size=30)
        -> scenari 1..3000

    Uso per train/test su dataset .nc:
        generate_scenario_batches(..., scenario_ids=TRAIN_SCENARIO_IDS_UTSP, ...)
        generate_scenario_batches(..., scenario_ids=TEST_SCENARIO_IDS_UTSP, ...)

    In cvett è importante passare scenario_ids distinti per train e test,
    altrimenti il test riparte da 1 e riusa osservazioni già viste nel training.
    """
    if scenario_ids is None:
        if n_scenarios is None:
            raise ValueError("Devi passare n_scenarios oppure scenario_ids.")
        scenario_ids = list(range(start_id, start_id + n_scenarios))
    else:
        scenario_ids = list(scenario_ids)
        if n_scenarios is not None and n_scenarios != len(scenario_ids):
            raise ValueError(
                f"n_scenarios={n_scenarios}, ma len(scenario_ids)={len(scenario_ids)}. "
                "Usane uno solo oppure rendili coerenti."
            )

    id_batches = _chunk_scenario_ids(
        scenario_ids=scenario_ids,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    if len(id_batches) == 0:
        raise ValueError("Nessun batch generato: controlla scenario_ids, batch_size e drop_last.")

    n_used = sum(len(b) for b in id_batches)
    n_dropped = len(scenario_ids) - n_used

    print(
        f"\nGenerazione {n_used} scenari "
        f"→ {len(id_batches)} batch da massimo {batch_size} scenari"
    )
    if n_dropped > 0:
        print(f"  Scenari scartati perché batch incompleto: {n_dropped}")

    used_ids = [sid for batch_ids in id_batches for sid in batch_ids]

    all_results, all_probs, total_random = generate_scenarios(
        used_ids, nodes, E, base_dist, I, frequent_arcs,
        n_extra_arcs, mean_frac, sigma_frac, base_seed,
        root, env, p, C,
        solve_pi=False,
        coords=coords, wind=wind, alpha=alpha, turb_sigma=turb_sigma,
    )

    batches = []
    for b, batch_ids in enumerate(id_batches):
        batch_results = {sid: all_results[sid] for sid in batch_ids}
        batch_probs   = {sid: 1.0 / len(batch_ids) for sid in batch_ids}

        batches.append({
            "batch_id"      : b,
            "scenario_ids"  : batch_ids,
            "results"       : batch_results,
            "scenario_probs": batch_probs,
            "batch_size"    : len(batch_ids),
        })

        if len(batch_ids) == 1:
            label = f"scenario {batch_ids[0]}"
        else:
            label = f"scenari {batch_ids[0]}..{batch_ids[-1]}"
        print(f"  Batch {b:03d}: {label}  n={len(batch_ids)}")

    return batches


def generate_train_test_scenario_batches(
    nodes, E, base_dist, I, frequent_arcs,
    n_extra_arcs, mean_frac, sigma_frac, base_seed,
    root, env, p, C,
    train_scenario_ids, test_scenario_ids,
    batch_size,
    coords=None, wind=None, alpha=None, turb_sigma=0.0,
    drop_last_train=False,
    drop_last_test=False,
):
    """
    Genera direttamente i batch separati di training e test.
    Gli id scenario vengono mantenuti distinti: nessuna sovrapposizione train/test.
    """
    train_set = set(train_scenario_ids)
    test_set = set(test_scenario_ids)
    overlap = train_set & test_set
    if overlap:
        sample = sorted(overlap)[:10]
        raise ValueError(f"Train e test condividono scenari: esempi {sample}")

    print("\n=== BATCH TRAIN ===")
    train_batches = generate_scenario_batches(
        nodes, E, base_dist, I, frequent_arcs,
        n_extra_arcs, mean_frac, sigma_frac, base_seed,
        root, env, p, C,
        scenario_ids=train_scenario_ids,
        batch_size=batch_size,
        drop_last=drop_last_train,
        coords=coords, wind=wind, alpha=alpha, turb_sigma=turb_sigma,
    )

    print("\n=== BATCH TEST ===")
    test_batches = generate_scenario_batches(
        nodes, E, base_dist, I, frequent_arcs,
        n_extra_arcs, mean_frac, sigma_frac, base_seed,
        root, env, p, C,
        scenario_ids=test_scenario_ids,
        batch_size=batch_size,
        drop_last=drop_last_test,
        coords=coords, wind=wind, alpha=alpha, turb_sigma=turb_sigma,
    )

    return train_batches, test_batches
