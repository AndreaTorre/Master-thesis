# -*- coding: utf-8 -*-
"""
diagnostica_perturbazioni.py
============================
Script di diagnostica per capire l'effetto reale dei campi vettoriali
sulle distanze degli scenari.

Produce:
  1) Tabella riassuntiva dei delta per 5 scenari
  2) Tabella specifica degli archi in I
  3) Istogramma dei delta percentuali
  4) Grafo dei nodi con campo vettoriale sovrapposto (per ogni scenario)
  5) Matrice heatmap |Δ%| per ogni scenario

Uso:
    python diagnostica_perturbazioni.py

Output nella cartella output_diagnostica/
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Import dal progetto ──────────────────────────────────────────────
from config import (
    WIND_ALPHA, WIND_TURB_SIGMA,
    ERA5_NC_PATH, K_MEDOID_NODES, MAX_KMEDOID_I_ARCS,
    PRENOTAZIONE_FRAC, PENALTY_FRAC,
)
from common import load_data, set_seed
from tsp_utils import canon_edge, base_cost_undirected
from wind_perturbation import load_wind_field, build_wind_perturbation
from gurobi_models import build_I_from_medoid_outgoing_nodes

OUT_DIR = "output_diagnostica"
os.makedirs(OUT_DIR, exist_ok=True)

N_DIAG_SCENARIOS = 5


# =====================================================================
# 1. CARICAMENTO DATI
# =====================================================================
print("=" * 70)
print("DIAGNOSTICA PERTURBAZIONI DA CAMPO VETTORIALE")
print("=" * 70)

set_seed()
nodes, coords, base_dist, E, root = load_data()
n = len(nodes)
print(f"\nNodi: {n},  Root: {root}")
print(f"Alpha vento: {WIND_ALPHA},  turb_sigma: {WIND_TURB_SIGMA}")

wind = load_wind_field(ERA5_NC_PATH)
print(f"Campo vettoriale: {wind['n_times']} istanti temporali, "
      f"griglia {len(wind['lats'])}×{len(wind['lons'])}")

# Magnitudine globale del vento (non normalizzata)
all_mag = np.sqrt(wind["u100"]**2 + wind["v100"]**2)
print(f"Magnitudine vento globale: "
      f"media={all_mag.mean():.2f} m/s, "
      f"p50={np.percentile(all_mag, 50):.2f}, "
      f"p90={np.percentile(all_mag, 90):.2f}, "
      f"max={all_mag.max():.2f}")

# Costruisci I
try:
    I, medoid_info = build_I_from_medoid_outgoing_nodes(
        nodes, E, base_dist,
        medoid_nodes=K_MEDOID_NODES,
        max_arcs=MAX_KMEDOID_I_ARCS,
    )
    print(f"Metodo I: {medoid_info.get('source', 'n/a')}")
except Exception as exc:
    print(f"ATTENZIONE: build_I fallita ({exc}), uso I = []")
    I = []
I_set = {canon_edge(i, j) for (i, j) in I}
print(f"Archi in I: {len(I_set)}  →  {sorted(I_set)}")


# =====================================================================
# 2. GENERA PERTURBAZIONI E RACCOGLI STATISTICHE
# =====================================================================

def compute_delta_stats(base_dist, pert_dict, nodes):
    """Calcola statistiche sui delta assoluti e percentuali."""
    deltas_abs = []
    deltas_pct = []
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            d0 = base_dist[i][j]
            delta = pert_dict.get((i, j), 0.0)
            deltas_abs.append(delta)
            if d0 > 1e-9:
                deltas_pct.append(100.0 * delta / d0)
    return np.array(deltas_abs), np.array(deltas_pct)


print(f"\n{'='*70}")
print(f"ANALISI DI {N_DIAG_SCENARIOS} SCENARI")
print(f"{'='*70}")

all_pct = []          # tutti i Δ% di tutti gli scenari
scenario_data = []    # per i grafici

for sid in range(1, N_DIAG_SCENARIOS + 1):
    pert = build_wind_perturbation(
        scenario_id=sid,
        nodes=nodes,
        base_dist=base_dist,
        coords=coords,
        wind=wind,
        alpha=WIND_ALPHA,
        turb_sigma=WIND_TURB_SIGMA,
    )

    d_abs, d_pct = compute_delta_stats(base_dist, pert, nodes)
    all_pct.extend(d_pct.tolist())

    print(f"\n--- Scenario {sid} ---")
    print(f"  Archi perturbati: {len(pert)} / {n*(n-1)}")
    print(f"  Δ assoluto:  min={d_abs.min():+.4f}  max={d_abs.max():+.4f}  "
          f"media={d_abs.mean():+.4f}  std={d_abs.std():.4f}")
    print(f"  Δ%:          min={d_pct.min():+.2f}%  max={d_pct.max():+.2f}%  "
          f"media={d_pct.mean():+.2f}%  std={d_pct.std():.2f}%")

    # Dettaglio archi in I
    if I_set:
        print(f"  Archi in I:")
        for (a, b) in sorted(I_set):
            d0_ab = base_dist[a][b]
            d0_ba = base_dist[b][a]
            delta_ab = pert.get((a, b), 0.0)
            delta_ba = pert.get((b, a), 0.0)
            pct_ab = 100 * delta_ab / d0_ab if d0_ab > 0 else 0
            pct_ba = 100 * delta_ba / d0_ba if d0_ba > 0 else 0
            print(f"    ({a:>3},{b:>3})  base={d0_ab:8.2f}  "
                  f"Δ(a→b)={delta_ab:+8.2f} ({pct_ab:+.1f}%)  "
                  f"Δ(b→a)={delta_ba:+8.2f} ({pct_ba:+.1f}%)")

    scenario_data.append({
        "sid": sid, "pert": pert,
        "d_abs": d_abs, "d_pct": d_pct,
    })


# =====================================================================
# 3. ISTOGRAMMA Δ% AGGREGATO
# =====================================================================
all_pct = np.array(all_pct)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(all_pct, bins=80, edgecolor="black", alpha=0.7)
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Δ% rispetto a base_dist")
ax.set_ylabel("Conteggio archi")
ax.set_title(f"Distribuzione Δ% su {N_DIAG_SCENARIOS} scenari  "
             f"(α={WIND_ALPHA}, n={n} nodi)")
txt = (f"media={all_pct.mean():.2f}%\n"
       f"std={all_pct.std():.2f}%\n"
       f"min={all_pct.min():.1f}%\n"
       f"max={all_pct.max():.1f}%")
ax.text(0.98, 0.95, txt, transform=ax.transAxes,
        va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "istogramma_delta_pct.png"), dpi=150)
plt.close(fig)
print(f"\n→ Salvato istogramma_delta_pct.png")


# =====================================================================
# 4. GRAFO + CAMPO VETTORIALE SOVRAPPOSTO
# =====================================================================

def _get_wind_at_time(wind, t_idx):
    """Restituisce u, v e griglia per l'istante t_idx."""
    u = wind["u100"][t_idx]
    v = wind["v100"][t_idx]
    lats = wind["lats"]
    lons = wind["lons"]
    return u, v, lats, lons


for sd in scenario_data:
    sid = sd["sid"]
    # Ricostruisci quale t_idx viene usato (stessa logica di build_wind_perturbation)
    rng = np.random.default_rng(sid)
    t_idx = int(rng.integers(0, wind["n_times"]))

    u_field, v_field, lats, lons = _get_wind_at_time(wind, t_idx)

    # normalizzazione identica a build_wind_perturbation
    wind_scale = max(float(np.percentile(
        np.sqrt(u_field**2 + v_field**2), 90
    )), 1e-9)

    # coordinate pixel dei nodi
    xs = [coords[i][0] for i in nodes]
    ys = [coords[i][1] for i in nodes]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # griglia vento mappata nello spazio pixel dei nodi
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(x_min, x_max, lons.shape[0]),
        np.linspace(y_max, y_min, lats.shape[0]),   # y invertito
    )

    # Il campo vento: u = componente est (→ +x pixel), v = componente nord (→ -y pixel)
    # Per il quiver in coordinate pixel: u_pixel = u, v_pixel = -v
    u_plot = u_field / wind_scale
    v_plot = -v_field / wind_scale  # inversione per asse y pixel

    fig, ax = plt.subplots(figsize=(10, 8))

    # Campo vettoriale di sfondo
    mag = np.sqrt(u_plot**2 + v_plot**2)
    q = ax.quiver(
        lon_grid, lat_grid, u_plot, v_plot, mag,
        cmap="coolwarm", alpha=0.5, scale=25, width=0.003,
    )
    plt.colorbar(q, ax=ax, label="Magnitudine vento (normalizzata)")

    # Nodi
    node_x = [coords[i][0] for i in nodes]
    node_y = [coords[i][1] for i in nodes]
    ax.scatter(node_x, node_y, c="black", s=80, zorder=5)
    for i in nodes:
        ax.annotate(str(i), coords[i], fontsize=7,
                    xytext=(4, 4), textcoords="offset points", zorder=6)

    # Archi in I (evidenziati)
    for (a, b) in I_set:
        xa, ya = coords[a]
        xb, yb = coords[b]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="->", color="red",
                                    lw=2, connectionstyle="arc3,rad=0.05"),
                    zorder=4)

    ax.set_title(f"Scenario {sid}  (t_idx={t_idx})  —  "
                 f"Campo vettoriale + nodi + archi I (rosso)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"grafo_vento_scen{sid:02d}.png"), dpi=150)
    plt.close(fig)
    print(f"→ Salvato grafo_vento_scen{sid:02d}.png")


# =====================================================================
# 5. HEATMAP |Δ%| PER SCENARIO
# =====================================================================

node_list = sorted(nodes)
node_idx = {n: i for i, n in enumerate(node_list)}

for sd in scenario_data:
    sid = sd["sid"]
    pert = sd["pert"]

    mat = np.zeros((n, n))
    for (i, j), delta in pert.items():
        d0 = base_dist[i][j]
        if d0 > 1e-9:
            mat[node_idx[i], node_idx[j]] = 100 * delta / d0

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(abs(mat.min()), abs(mat.max()), 1.0)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, label="Δ%")
    ax.set_xticks(range(n))
    ax.set_xticklabels(node_list, fontsize=6, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(node_list, fontsize=6)
    ax.set_xlabel("j (destinazione)")
    ax.set_ylabel("i (origine)")
    ax.set_title(f"Scenario {sid} — Δ% per arco orientato  (α={WIND_ALPHA})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"heatmap_delta_scen{sid:02d}.png"), dpi=150)
    plt.close(fig)
    print(f"→ Salvato heatmap_delta_scen{sid:02d}.png")


# =====================================================================
# 6. CONFRONTO TRA SCENARI: gli stessi archi cambiano verso?
# =====================================================================
print(f"\n{'='*70}")
print("CONFRONTO DIREZIONE PERTURBAZIONE TRA SCENARI")
print("(Un arco cambia segno tra scenari? Se no → poca diversità)")
print(f"{'='*70}")

# Per ogni arco orientato, raccogliamo il segno del delta in ogni scenario
arc_signs = defaultdict(list)
for sd in scenario_data:
    pert = sd["pert"]
    for (i, j) in pert:
        arc_signs[(i, j)].append(np.sign(pert[(i, j)]))

n_arcs_total = 0
n_arcs_same_sign = 0
n_arcs_mixed = 0
for arc, signs in arc_signs.items():
    if len(signs) < 2:
        continue
    n_arcs_total += 1
    if all(s == signs[0] for s in signs):
        n_arcs_same_sign += 1
    else:
        n_arcs_mixed += 1

print(f"Archi con almeno 2 scenari: {n_arcs_total}")
print(f"  Sempre stesso segno: {n_arcs_same_sign} "
      f"({100*n_arcs_same_sign/max(n_arcs_total,1):.0f}%)")
print(f"  Segno misto:         {n_arcs_mixed} "
      f"({100*n_arcs_mixed/max(n_arcs_total,1):.0f}%)")

if n_arcs_same_sign / max(n_arcs_total, 1) > 0.80:
    print("\n⚠️  ATTENZIONE: >80% degli archi hanno sempre lo stesso segno.")
    print("   Questo significa che scenari diversi producono perturbazioni")
    print("   quasi parallele → tour molto simili → STO ≈ EEV.")
    print("   Possibili cause:")
    print("   - Il campo ERA5 copre pochi istanti temporali con vento simile")
    print("   - L'area dei nodi è troppo piccola rispetto alla griglia ERA5")
    print("   - Servono più istanti temporali o stagioni diverse nel file .nc")


# =====================================================================
# 7. RIEPILOGO FINALE
# =====================================================================
print(f"\n{'='*70}")
print("RIEPILOGO E RACCOMANDAZIONI")
print(f"{'='*70}")

mean_abs_pct = np.abs(all_pct).mean()
max_abs_pct = np.abs(all_pct).max()

print(f"\n|Δ%| medio su tutti gli archi e scenari: {mean_abs_pct:.2f}%")
print(f"|Δ%| massimo:                             {max_abs_pct:.2f}%")

if mean_abs_pct < 5:
    print(f"\n⚠️  Perturbazione media sotto il 5%: probabilmente troppo debole.")
    print(f"   Con 10 nodi e tour dominante, servono almeno 15-25% di Δ medio")
    print(f"   per far cambiare il tour tra scenari.")
    print(f"   → Prova ad aumentare WIND_ALPHA (attualmente {WIND_ALPHA})")
    suggested = WIND_ALPHA * (20.0 / max(mean_abs_pct, 0.1))
    print(f"   → Alpha suggerito per ~20% medio: {suggested:.2f}")
elif mean_abs_pct < 15:
    print(f"\n⚡ Perturbazione moderata ({mean_abs_pct:.1f}%).")
    print(f"   Potrebbe bastare, ma verifica se i tour PI cambiano tra scenari.")
else:
    print(f"\n✓  Perturbazione significativa ({mean_abs_pct:.1f}%).")
    print(f"   Se i tour restano uguali, il problema è geometrico (tour dominante),")
    print(f"   non l'intensità del vento.")

print(f"\nFile salvati in {OUT_DIR}/")
print("Done.")
