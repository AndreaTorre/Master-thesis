# check_nc_h5.py
# -*- coding: utf-8 -*-

import h5py

nc_path = "/home/atorre/UTSP/unione/git/cvett.nc"

print("Apro file con h5py...", flush=True)

with h5py.File(nc_path, "r") as f:
    print("File aperto correttamente.", flush=True)

    print("\n=== ATTRIBUTI GLOBALI ===")
    for k, v in f.attrs.items():
        print(f"{k}: {v}")

    print("\n=== STRUTTURA FILE ===")

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name} | shape={obj.shape} | dtype={obj.dtype}")
            for ak, av in obj.attrs.items():
                print(f"          attr {ak}: {av}")
        elif isinstance(obj, h5py.Group):
            print(f"[GROUP]   {name}")

    f.visititems(visit)

print("\nFine.")