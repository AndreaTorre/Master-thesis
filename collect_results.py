# -*- coding: utf-8 -*-
import glob, re, ast, pandas as pd

results = []

for f in sorted(glob.glob("/home/atorre/UTSP/unione/git/grid_logs/output_*.txt")):
    text = open(f, encoding="utf-8", errors="ignore").read()

    combo_match = re.search(r"Combo \d+: ({.*?})", text)
    utsp_match  = re.search(r"UTSP_LS_val\s*=\s*([\d.]+)", text)

    if combo_match and utsp_match:
        params = ast.literal_eval(combo_match.group(1))
        utsp_ls_val = float(utsp_match.group(1))
        results.append({**params, "UTSP_LS_val": utsp_ls_val})
    else:
        print("SKIP " + f + " - dati mancanti")

if results:
    df = pd.DataFrame(results).sort_values("UTSP_LS_val").reset_index(drop=True)
    df.to_csv("grid_results_final.csv", index=False)
    print("Raccolte " + str(len(results)) + " combinazioni")
    print("\n=== TOP 5 ===")
    print(df.head(72).to_string(index=False))
else:
    print("Nessun risultato trovato.")