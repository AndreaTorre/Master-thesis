# -*- coding: utf-8 -*-
"""
confronto_n_samples.py
======================
Confronto rapido tra il vecchio midpoint (n_samples=1) e il nuovo
multi-point (n_samples=5) per capire se il campionamento lungo l'arco
introduce variabilità aggiuntiva.

Produce:
  - Scatter plot: Δ% (midpoint) vs Δ% (5 punti) per scenario
  - Statistiche sulla differenza tra i due metodi
  - Verifica se l'antisimmetria i→j / j→i si rompe

Uso:
    python confronto_n_samples.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import WIND_ALPHA, WIND_MIN_FACTOR, WIND_MAX_FACTOR, ERA5_NC_PATH
from common import load_data, set_seed
from wind_perturbation import load_wind_field, build_wind_perturbation

OUT_DIR = "output_diagnostica"
os.makedirs(OUT_DIR, exist_ok=True)

N_SCENARIOS = 5

# ── Caricamento ──────────────────────────────────────────────────────
set_seed()
nodes, coords, base_dist, E, root = load_data()
wind = load_wind_field(ERA5_NC_PATH)
n = len(nodes)

print("=" * 70)
print("CONFRONTO n_samples=1 (midpoint) vs n_samples=5 (multi-point)")
print("=" * 70)

for sid in range(1, N_SCENARIOS + 1):
    # Vecchio metodo: solo punto medio
    pert_1 = build_wind_perturbation(
        scenario_id=sid, nodes=nodes, base_dist=base_dist,
        coords=coords, wind=wind, alpha=WIND_ALPHA,
        min_factor=WIND_MIN_FACTOR, max_factor=WIND_MAX_FACTOR,
        n_samples=1,
    )
    # Nuovo metodo: 5 punti lungo l'arco
    pert_5 = build_wind_perturbation(
        scenario_id=sid, nodes=nodes, base_dist=base_dist,
        coords=coords, wind=wind, alpha=WIND_ALPHA,
        min_factor=WIND_MIN_FACTOR, max_factor=WIND_MAX_FACTOR,
        n_samples=5,
    )

    # Raccogli Δ% per entrambi
    arcs = sorted(pert_1.keys())
    pct_1 = []
    pct_5 = []
    for (i, j) in arcs:
        d0 = base_dist[i][j]
        if d0 > 1e-9:
            pct_1.append(100 * pert_1[(i, j)] / d0)
            pct_5.append(100 * pert_5[(i, j)] / d0)

    pct_1 = np.array(pct_1)
    pct_5 = np.array(pct_5)
    diff = pct_5 - pct_1

    print(f"\n--- Scenario {sid} ---")
    print(f"  |Δ%| medio  midpoint: {np.abs(pct_1).mean():.2f}%   "
          f"multi-point: {np.abs(pct_5).mean():.2f}%")
    print(f"  Differenza (5pt - 1pt):  "
          f"media={diff.mean():+.3f}%  std={diff.std():.3f}%  "
          f"max|diff|={np.abs(diff).max():.3f}%")

    # Quanti archi cambiano segno tra i due metodi?
    sign_flip = np.sum(np.sign(pct_1) != np.sign(pct_5))
    print(f"  Archi con cambio di segno (1pt→5pt): {sign_flip}/{len(arcs)}")

    # Antisimmetria: per ogni coppia (i,j)/(j,i), quanto è |Δ%_ij + Δ%_ji|?
    # Se perfettamente antisimmetrico → 0.  Più alto → più broken.
    asym_1 = []
    asym_5 = []
    for i_node in nodes:
        for j_node in nodes:
            if i_node >= j_node:
                continue
            d_ij = base_dist[i_node][j_node]
            d_ji = base_dist[j_node][i_node]
            if d_ij < 1e-9 or d_ji < 1e-9:
                continue
            p1_ij = 100 * pert_1.get((i_node, j_node), 0) / d_ij
            p1_ji = 100 * pert_1.get((j_node, i_node), 0) / d_ji
            p5_ij = 100 * pert_5.get((i_node, j_node), 0) / d_ij
            p5_ji = 100 * pert_5.get((j_node, i_node), 0) / d_ji
            asym_1.append(abs(p1_ij + p1_ji))
            asym_5.append(abs(p5_ij + p5_ji))

    asym_1 = np.array(asym_1)
    asym_5 = np.array(asym_5)
    print(f"  Antisimmetria |Δ%_ij + Δ%_ji|:  "
          f"midpoint media={asym_1.mean():.3f}%  "
          f"multi-point media={asym_5.mean():.3f}%")
    if asym_5.mean() > asym_1.mean() * 1.1:
        print(f"  → Il multi-point rompe parzialmente l'antisimmetria ✓")
    else:
        print(f"  → Antisimmetria ancora intatta (campo troppo uniforme)")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(pct_1, pct_5, s=8, alpha=0.6)
    lim = max(abs(pct_1).max(), abs(pct_5).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, label="y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Δ% midpoint (n_samples=1)")
    ax.set_ylabel("Δ% multi-point (n_samples=5)")
    ax.set_title(f"Scenario {sid} — Confronto perturbazioni")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(diff, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", ls="--")
    ax.set_xlabel("Δ% (5pt) − Δ% (1pt)")
    ax.set_ylabel("Conteggio archi")
    ax.set_title(f"Scenario {sid} — Distribuzione della differenza")
    txt = f"std={diff.std():.3f}%\nmax|Δ|={np.abs(diff).max():.3f}%"
    ax.text(0.98, 0.95, txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"confronto_samples_scen{sid:02d}.png"), dpi=150)
    plt.close(fig)
    print(f"  → Salvato confronto_samples_scen{sid:02d}.png")


# ── Riepilogo ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("CONCLUSIONE")
print(f"{'='*70}")
print("""
Se la std della differenza è sotto 0.5% e l'antisimmetria resta intatta,
il multi-point da solo NON basta a diversificare i tour.
In quel caso il campo ERA5 è troppo uniforme sull'area dei nodi e serve
l'opzione A (turbolenza locale) o l'opzione C (alpha più alto).

Se invece la std è >1% e l'antisimmetria si rompe, il multi-point
sta già introducendo variabilità utile. Rilancia la pipeline completa
con la nuova wind_perturbation.py e verifica se PI cambia tra scenari.
""")
print(f"File salvati in {OUT_DIR}/")
