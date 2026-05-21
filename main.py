# -*- coding: utf-8 -*-
import argparse

from config import ESPERIMENTI_DA_ESEGUIRE
from common import load_data, load_env, set_seed
from experiment_B import run_esperimento_B
from utsp import run_esperimento_B_UTSP


def main():
    parser = argparse.ArgumentParser(description="Esegue Esperimento B e/o B_UTSP.")
    parser.add_argument(
        "--only",
        choices=["B", "B_UTSP", "ALL"],
        default="ALL",
        help="Esperimento da eseguire. B_UTSP richiede prima B nella stessa esecuzione.",
    )
    args = parser.parse_args()

    set_seed()
    env = load_env()
    nodes, coords, base_dist, E, root = load_data()

    risultati = {}
    run_list = ESPERIMENTI_DA_ESEGUIRE if args.only == "ALL" else [args.only]

    if "B" in run_list or "B_UTSP" in run_list:
        risultati["B"] = run_esperimento_B(nodes, coords, base_dist, E, root, env)

    if "B_UTSP" in run_list:
        risultati["B_UTSP"] = run_esperimento_B_UTSP(
            nodes, coords, base_dist, E, root, env,
            res_B=risultati["B"],
        )

    return risultati


if __name__ == "__main__":
    main()
