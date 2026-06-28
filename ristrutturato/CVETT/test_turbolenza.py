# -*- coding: utf-8 -*-
"""
test_turbolenza.py
==================
Test rapido: verifica che la turbolenza locale rompa l'antisimmetria
e introduca diversità tra scenari.

Uso:  python test_turbolenza.py
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
TURB_VALUES = [0.0, 0.05, 0.08, 0.12]

set_seed()
nodes, coords, base_dist, E, root = load_data()
wind = load_wind_field(ERA5_NC_PATH)

print("=" * 70)
print("TEST TURBOLENZA LOCALE — effetto su antisimmetria e diversità")
print("=" * 70)

for turb in TURB_VALUES:
    asym_all = []
    pct_all = []

    for sid in range(1, N_SCENARIOS + 1):
        pert = build_wind_perturbation(
            scenario_id=sid, nodes=nodes, base_dist=base_dist,
            coords=coords, wind=wind, alpha=WIND_ALPHA,
            min_factor=WIND_MIN_FACTOR, max_factor=WIND_MAX_FACTOR,
            n_samples=5, turb_sigma=turb,
        )

        for i in nodes:
            for j in nodes:
                if i >= j:
                    continue
                d_ij = base_dist[i][j]
                d_ji = base_dist[j][i]
                if d_ij < 1e-9 or d_ji < 1e-9:
                    continue
                p_ij = 100 * pert.get((i, j), 0) / d_ij
                p_ji = 100 * pert.get((j, i), 0) / d_ji
                asym_all.append(abs(p_ij + p_ji))
                pct_all.append(abs(p_ij))

    asym_all = np.array(asym_all)
    pct_all = np.array(pct_all)

    print(f"\n  turb_sigma = {turb:.2f}")
    print(f"    |Δ%| medio:                  {pct_all.mean():.2f}%")
    print(f"    Antisimmetria |Δ%_ij+Δ%_ji|: media={asym_all.mean():.3f}%  "
          f"max={asym_all.max():.3f}%")
    if turb > 0:
        ratio = asym_all.mean() / max(pct_all.mean(), 1e-9) * 100
        print(f"    Rapporto antisym/|Δ%|:       {ratio:.1f}%")


# ── Grafico riepilogativo ────────────────────────────────────────────
print(f"\n{'='*70}")
print("CONFRONTO HEATMAP Δ% — turb_sigma=0.0 vs 0.08, scenario 1")
print(f"{'='*70}")

node_list = sorted(nodes)
node_idx = {n: i for i, n in enumerate(node_list)}
n = len(nodes)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, turb, label in zip(axes, [0.0, 0.08],
                             ["Senza turbolenza", "Con turbolenza (σ=0.08)"]):
    pert = build_wind_perturbation(
        scenario_id=1, nodes=nodes, base_dist=base_dist,
        coords=coords, wind=wind, alpha=WIND_ALPHA,
        min_factor=WIND_MIN_FACTOR, max_factor=WIND_MAX_FACTOR,
        n_samples=5, turb_sigma=turb,
    )
    mat = np.zeros((n, n))
    for (i, j), delta in pert.items():
        d0 = base_dist[i][j]
        if d0 > 1e-9:
            mat[node_idx[i], node_idx[j]] = 100 * delta / d0

    vmax = 30
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, label="Δ%", shrink=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(node_list, fontsize=6, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(node_list, fontsize=6)
    ax.set_title(label)

fig.suptitle(f"Scenario 1 — Effetto della turbolenza locale (α={WIND_ALPHA})",
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confronto_turbolenza.png"), dpi=150)
plt.close(fig)
print(f"→ Salvato confronto_turbolenza.png")

print(f"""
INTERPRETAZIONE:
  - Senza turbolenza: la matrice è perfettamente antisimmetrica
    (Δ%[i,j] = -Δ%[j,i], colori opposti simmetrici rispetto alla diagonale)
  - Con turbolenza: l'antisimmetria si rompe, ogni arco ha una componente
    indipendente → tour diversi reagiscono diversamente → PI divergono
""")
