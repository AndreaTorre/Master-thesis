# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from config import VALIDATION_SEED
from common import out_path
from tsp_utils import canon_edge, get_edge_value
from scenarios import generate_scenarios
from gurobi_models import solve_reservation_tsp

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

# ═══════════════════════════════════════════════════════════════════
# GRAFICI 4 PANNELLI: PI / EEV / STO / UTSP
# ═══════════════════════════════════════════════════════════════════

def _draw_nodes(ax, nodes, coords):
    xs = [coords[n][0] for n in nodes]
    ys = [coords[n][1] for n in nodes]
    ax.scatter(xs, ys, color="#333333", s=70, zorder=5)
    for n, (cx, cy) in coords.items():
        ax.text(cx + 0.4, cy + 0.4, str(n), fontsize=8, zorder=6)


def _draw_arcs(ax, arcs, coords, highlight_undir, base_color, lw_normal=1.8, lw_hi=2.8):
    for (i, j) in arcs:
        xi, yi = coords[i]
        xj, yj = coords[j]
        in_I = highlight_undir is not None and canon_edge(i, j) in highlight_undir
        color = "crimson" if in_I else base_color
        lw = lw_hi if in_I else lw_normal
        ms = 16 if in_I else 13
        ax.annotate(
            "",
            xy=(xj, yj),
            xytext=(xi, yi),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=ms),
            zorder=3,
        )


def _draw_reserved_not_used(ax, reserved_edges, tour_arcs, coords):
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


def _solution_arcs(solution):
    if not solution:
        return []
    return solution.get("arcs", solution.get("y_used", []))


def _solution_tour(solution):
    if not solution:
        return []
    return solution.get("tour", [])


def plot_scenario_comparison_utsp(
    exp_name,
    scenario_id,
    nodes,
    coords,
    results,
    eev_costs,
    eev_solutions,
    stoch_costs,
    stoch_solutions,
    utsp_costs,
    utsp_solutions,
    x_ev,
    x_sto,
    x_utsp,
    utsp_label="UTSP",
    save=True,
):
    fig, axes = plt.subplots(2, 2, figsize=(26, 22))
    fig.suptitle(
        f"Scenario {scenario_id} — confronto PI / EEV / STO / {utsp_label}",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    pi_sol = results[scenario_id]["exact_free"]
    eev_sol = eev_solutions[scenario_id]
    sto_sol = stoch_solutions[scenario_id]
    utsp_sol = utsp_solutions[scenario_id]

    pi_arcs = _solution_arcs(pi_sol)
    pi_tour = _solution_tour(pi_sol)
    eev_arcs = _solution_arcs(eev_sol)
    eev_tour = _solution_tour(eev_sol)
    sto_arcs = _solution_arcs(sto_sol)
    sto_tour = _solution_tour(sto_sol)
    utsp_arcs = _solution_arcs(utsp_sol)
    utsp_tour = _solution_tour(utsp_sol)

    panels = [
        {
            "title": "PI libero",
            "arcs": pi_arcs,
            "tour": pi_tour,
            "cost": pi_sol.get("length", pi_sol.get("objective", None)),
            "color": "#01D80B",
            "highlight": set(),
        },
        {
            "title": "EEV: ricorso con x^EV fissato",
            "arcs": eev_arcs,
            "tour": eev_tour,
            "cost": eev_costs[scenario_id],
            "color": "#A500CE",
            "highlight": set(x_ev),
        },
        {
            "title": "Stocastico / STO",
            "arcs": sto_arcs,
            "tour": sto_tour,
            "cost": stoch_costs[scenario_id],
            "color": "#000000",
            "highlight": set(x_sto),
        },
        {
            "title": utsp_label,
            "arcs": utsp_arcs,
            "tour": utsp_tour,
            "cost": utsp_costs[scenario_id],
            "color": "#0055CC",
            "highlight": set(x_utsp) if x_utsp else set(),
        },
    ]

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for panel, (r, c) in zip(panels, positions):
        ax = axes[r][c]
        _draw_nodes(ax, nodes, coords)

        if panel["arcs"]:
            _draw_arcs(ax, panel["arcs"], coords, panel["highlight"], panel["color"])
            _draw_reserved_not_used(ax, panel["highlight"], panel["arcs"], coords)
        else:
            ax.text(
                0.5, 0.5, "Soluzione non disponibile",
                ha="center", va="center", transform=ax.transAxes, fontsize=12,
            )

        tour = panel["tour"]
        if tour:
            tour_str = " → ".join(str(n) for n in tour) + f" → {tour[0]}"
            arc_list = [(tour[k], tour[(k + 1) % len(tour)]) for k in range(len(tour))]
            arc_str = "  →  ".join(f"({i},{j})" for (i, j) in arc_list)
        else:
            tour_str = "n.d."
            arc_str = "n.d."

        cost = panel["cost"]
        cost_str = f"{cost:.4f}" if cost is not None else "N/A"

        handles = [mpatches.Patch(color=panel["color"], label="Arco percorso")]
        if panel["highlight"]:
            tour_edges = {canon_edge(i, j) for (i, j) in panel["arcs"]}
            highlighted_edges = {canon_edge(*e) for e in panel["highlight"]}
            if highlighted_edges & tour_edges:
                handles.append(mpatches.Patch(color="crimson", label="Tratta prenotata e percorsa"))
            if highlighted_edges - tour_edges:
                handles.append(Line2D([0], [0], color="crimson", linestyle="--", linewidth=2.2, label="Tratta prenotata non percorsa"))

        ax.legend(handles=handles, loc="upper left", fontsize=8)
        ax.set_title(
            f"{panel['title']}\nCosto: {cost_str}\nTour: {tour_str}\nArchi: {arc_str}",
            fontsize=8,
            loc="left",
            pad=8,
            family="monospace",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()

    if save:
        fname = out_path(f"{exp_name}_scenario_{scenario_id}_confronto.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Salvato grafico 4 pannelli: {fname}")
    else:
        plt.show()


def genera_grafici_utsp(
    exp_name,
    nodes,
    coords,
    scenario_ids,
    results,
    eev_costs,
    eev_solutions,
    stoch_costs,
    stoch_solutions,
    utsp_costs,
    utsp_solutions,
    x_ev,
    x_sto,
    x_utsp,
    utsp_label="UTSP",
    save=True,
):
    for sid in scenario_ids:
        plot_scenario_comparison_utsp(
            exp_name=exp_name,
            scenario_id=sid,
            nodes=nodes,
            coords=coords,
            results=results,
            eev_costs=eev_costs,
            eev_solutions=eev_solutions,
            stoch_costs=stoch_costs,
            stoch_solutions=stoch_solutions,
            utsp_costs=utsp_costs,
            utsp_solutions=utsp_solutions,
            x_ev=x_ev,
            x_sto=x_sto,
            x_utsp=x_utsp,
            utsp_label=utsp_label,
            save=save,
        )
# ═══════════════════════════════════════════════════════════════════
# HEATMAP E GRAFO PESATO — UTSP
# ═══════════════════════════════════════════════════════════════════

def plot_utsp_heatmap(exp_name, nodes, H, title_suffix="", save=True):
    """
    Visualizza la matrice H (n×n) come imshow con etichette dei nodi sugli assi.
    H è tipicamente H_decode (diagonale azzerata, media sugli scenari).
    """
    import numpy as np
    n = len(nodes)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.75), max(5, n * 0.7)))

    h_plot = np.array(H, dtype=float)
    im = ax.imshow(h_plot, cmap="YlOrRd", aspect="auto", vmin=0, vmax=h_plot.max())
    plt.colorbar(im, ax=ax, shrink=0.8, label="H[i→j]")

    labels = [str(v) for v in nodes]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("nodo j  (destinazione)", fontsize=10)
    ax.set_ylabel("nodo i  (origine)", fontsize=10)

    title = f"Heatmap UTSP  {title_suffix}".strip()
    ax.set_title(title, fontsize=12, fontweight="bold")

    h_max = float(h_plot.max()) if h_plot.max() > 0 else 1.0
    for ii in range(n):
        for jj in range(n):
            val = float(h_plot[ii, jj])
            if val > 1e-4:
                txt_color = "white" if val > 0.6 * h_max else "black"
                ax.text(jj, ii, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=7, color=txt_color)

    plt.tight_layout()
    if save:
        fname = out_path(f"{exp_name}_heatmap.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Salvato heatmap: {fname}")
    else:
        plt.show()


def plot_utsp_graph_weights(exp_name, nodes, coords, H,
                            threshold=0.01, title_suffix="", save=True):
    """
    Disegna il grafo con archi la cui larghezza è proporzionale a H[i→j].
    H è tipicamente H_raw (media sugli scenari, con diagonale).
    Gli archi con H[i,j] / H.max() < threshold vengono omessi.
    """
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    idx = {v: k for k, v in enumerate(nodes)}
    H = np.array(H, dtype=float)
    h_max = float(H.max())
    if h_max < 1e-9:
        print("  plot_utsp_graph_weights: H è quasi nulla, grafico saltato.")
        return
    H_norm = H / h_max

    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_nodes(ax, nodes, coords)

    lw_min,    lw_max    = 0.4,  6.0
    alpha_min, alpha_max = 0.12, 0.88
    ms_min,    ms_max    = 6,    18

    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            w = float(H_norm[idx[i], idx[j]])
            if w < threshold:
                continue
            xi, yi = coords[i]
            xj, yj = coords[j]
            lw    = lw_min    + (lw_max    - lw_min)    * w
            alpha = alpha_min + (alpha_max - alpha_min) * w
            ms    = ms_min    + (ms_max    - ms_min)    * w
            color = (0.05, 0.25 + 0.45 * (1 - w), 0.85, alpha)
            ax.annotate(
                "",
                xy=(xj, yj), xytext=(xi, yi),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=lw,
                    mutation_scale=ms,
                ),
                zorder=2,
            )

    sm = ScalarMappable(cmap="Blues", norm=Normalize(vmin=0, vmax=h_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.7, label="H[i→j]  (valore assoluto)")

    title = f"Grafo pesato UTSP  {title_suffix}".strip()
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save:
        fname = out_path(f"{exp_name}_graph_weights.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Salvato grafo pesato: {fname}")
    else:
        plt.show()
