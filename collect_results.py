# -*- coding: utf-8 -*-
import glob, re, ast, pandas as pd

results = []

for f in sorted(glob.glob("/home/atorre/UTSP/unione/git/grid_logs/output_*.txt")):
    text = open(f, encoding="utf-8", errors="ignore").read()

    combo_match  = re.search(r"Combo \d+: ({.*?})", text)
    # regex robusta: accetta "UTSP_LS_val=3.14", "UTSP_LS_val = 3.14", "UTSP_LS_val: 3.14"
    utsp_match   = re.search(r"UTSP_LS_val\s*[=:]\s*([\d.]+)", text)
    sto_match    = re.search(r"STO_val\s*[=:]\s*([\d.]+)", text)
    gap_match    = re.search(r"gap\s*[=:]\s*([\d.]+)", text)

    if combo_match and utsp_match:
        params = ast.literal_eval(combo_match.group(1))
        utsp_ls_val = float(utsp_match.group(1))
        row = {**params, "UTSP_LS_val": utsp_ls_val}
        if sto_match:
            row["STO_val"] = float(sto_match.group(1))
        if gap_match:
            row["gap_%"] = float(gap_match.group(1))
        results.append(row)
    else:
        print("SKIP " + f + " - dati mancanti")

if results:
    df = pd.DataFrame(results).sort_values("UTSP_LS_val").reset_index(drop=True)
    df.to_csv("grid_results_final.csv", index=False)
    print("Raccolte " + str(len(results)) + " combinazioni")
    print("\n=== TOP 10 ===")
    print(df.head(10).to_string(index=False))
else:
    print("Nessun risultato trovato.")