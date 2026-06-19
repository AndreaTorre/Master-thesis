# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

from config import K_MEDOID_NODES, MAX_KMEDOID_I_ARCS, KMEDOID_ARCS_PER_NODE, STO_TIME_LIMIT, STO_MIP_GAP
from tsp_utils import canon_edge, directed_to_undirected_arcs, base_cost_undirected, get_edge_value, extract_tour_from_arcs, tour_length_from_arcs

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

def safe_gurobi_attr(model, attr_name, default=None):
    try:
        return getattr(model, attr_name)
    except Exception:
        return default

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
