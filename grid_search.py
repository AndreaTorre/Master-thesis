import itertools, csv, pickle, importlib, sys, argparse
import config
from common import load_data, load_env, set_seed

def run_grid_search():
    # ← AGGIUNTA: leggi job-id e n-jobs
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    with open("res_B_cached.pkl", "rb") as f:
        res_B = pickle.load(f)

    
    env = load_env()
    nodes, coords, base_dist, E, root = load_data()

    grid   = config.GRID_SEARCH
    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))

    # ← AGGIUNTA: ogni job prende solo il suo sottoinsieme
    combos_this_job = combos[args.job_id::args.n_jobs]

    print(f"Job {args.job_id}/{args.n_jobs} — combinazioni: {len(combos_this_job)} su {len(combos)} totali\n")

    results_log = []

    for idx, combo in enumerate(combos_this_job):
        
        params = dict(zip(keys, combo))
        for k, v in params.items():
            setattr(config, k, v)
        set_seed()
        for mod in ["two_stage_utsp_loss", "utsp"]:
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])
        from utsp import run_esperimento_B_UTSP

        print(f"\n--- Run {idx+1}/{len(combos_this_job)} ---")
        for k, v in params.items():
            print(f"  {k} = {v}")

        res = run_esperimento_B_UTSP(
            nodes, coords, base_dist, E, root, env,
            res_B=res_B,
            mode="local_search",
        )

        utsp_ls_val = res["local_search"]["UTSP_LS_val"]
        sto_val     = res_B["STO"]
        gap         = (utsp_ls_val - sto_val) / sto_val * 100

        row = {**params, "UTSP_LS_val": utsp_ls_val, "STO_val": sto_val, "gap_%": gap}
        results_log.append(row)
        print(f"  → UTSP_LS_val={utsp_ls_val:.4f}  gap={gap:.2f}%")

    # ← MODIFICA: nome file include job-id
    out_file = f"grid_results_job{args.job_id}.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader()
        writer.writerows(results_log)
    print(f"\nRisultati salvati in {out_file}")

if __name__ == "__main__":
    run_grid_search()