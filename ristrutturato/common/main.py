import os
import sys
import runpy
from pathlib import Path

import config


def _print_context():
    print("=== UTSP COMMON MAIN ===")
    print(f"EXPERIMENT      = {config.EXPERIMENT}")
    print(f"N_NODES         = {config.N_NODES}")
    print(f"INSTANCE_TAG    = {config.INSTANCE_TAG}")
    print(f"DATA_FILE       = {config.DATA_FILE_PRIMARY}")
    print(f"OUTPUT_DIR      = {config.OUTPUT_DIR}")

    if config.IS_CVETT:
        print(f"WIND_NC_PATH    = {config.WIND_NC_PATH}")

    print(f"config file     = {config.__file__}")
    print("========================")


def main():
    _print_context()

    # Modalità di test: non importa né Gurobi né moduli pesanti.
    if "--dry-run" in sys.argv:
        print("Dry run completato: nessun esperimento lanciato.")
        return

    root = Path(config.ROOT_DIR)
    exp_dir = root / config.EXPERIMENT
    target_main = exp_dir / "main.py"

    if not target_main.exists():
        raise FileNotFoundError(f"main.py non trovato per {config.EXPERIMENT}: {target_main}")

    # sys.path[0] resta common/, così `import config` prende common/config.py.
    # La cartella specifica PERT/CVETT viene dopo, così importa i moduli specifici
    # come experiment_B.py, scenarios.py, utsp.py.
    exp_dir_str = str(exp_dir)
    if exp_dir_str not in sys.path:
        sys.path.insert(1, exp_dir_str)

    # Per compatibilità con i vecchi main.py.
    old_argv0 = sys.argv[0]
    sys.argv[0] = str(target_main)

    try:
        runpy.run_path(str(target_main), run_name="__main__")
    finally:
        sys.argv[0] = old_argv0


if __name__ == "__main__":
    main()
