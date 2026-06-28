# -*- coding: utf-8 -*-
import glob, pandas as pd, os

results = []

for f in sorted(glob.glob("/home/atorre/UTSP/unione/git/fine_results_combo*.csv")):
    try:
        df = pd.read_csv(f)
        if not df.empty and "UTSP_LS_val" in df.columns:
            results.append(df)
    except Exception as e:
        print(f"SKIP {f} ({e})")

if results:
    df_all = pd.concat(results, ignore_index=True).sort_values("UTSP_LS_val").reset_index(drop=True)
    # Fix gap NaN
    df_all["gap_%"] = (df_all["UTSP_LS_val"] - df_all["STO_val"]) / df_all["STO_val"] * 100
    df_all.to_csv("/home/atorre/UTSP/unione/git/fine_results_final.csv", index=False)
    print(f"Raccolte {len(df_all)} combinazioni")
    print("\n=== TOP 10 ===")
    print(df_all.head(10).to_string(index=False))

    # Mancanti
    existing = set(int(os.path.basename(f).replace("fine_results_combo","").replace(".csv",""))
                   for f in glob.glob("/home/atorre/UTSP/unione/git/fine_results_combo*.csv"))
    missing = sorted(set(range(432)) - existing)
    print(f"\nCombo mancanti: {len(missing)}")
else:
    print("Nessun risultato trovato.")
