# -*- coding: utf-8 -*-
import glob, pandas as pd

dfs = []
for f in sorted(glob.glob("/home/atorre/UTSP/unione/git/grid_check_combo*.csv")):
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        print(f"SKIP {f} - {e}")

if dfs:
    df = pd.concat(dfs, ignore_index=True).sort_values("UTSP_LS_val").reset_index(drop=True)
    df.to_csv("grid_results_final.csv", index=False)
    print(f"Raccolte {len(df)} combinazioni")
    print(f"File CSV letti: {len(dfs)}")
    print(f"\n=== TOP 10 per UTSP_LS_val ===")
    print(df.head(10).to_string(index=False))

    if df["gap_%"].notna().any():
        print(f"\n=== TOP 10 per gap_% (minimo) ===")
        print(df.dropna(subset=["gap_%"]).nsmallest(10, "gap_%").to_string(index=False))
else:
    print("Nessun CSV trovato.")