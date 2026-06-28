# -*- coding: utf-8 -*-
"""
Perturbazioni basate su campo vettoriale di vento ERA5.

Formula additiva (integrale di linea discretizzato, midpoint rule):

    Delta_ij = -alpha * sum_{k=1}^{m} (L_ij / m) * [w(s_k) · d_hat_ij]

    c_ij     = max(L_ij + Delta_ij, eps)

Dipendenze: numpy, h5py  (niente netCDF4, niente scipy)
"""

import math
import numpy as np
import h5py


# ---------------------------------------------------------------------------
# CARICAMENTO
# ---------------------------------------------------------------------------

def load_wind_field(nc_path: str) -> dict:
    """
    Legge il file ERA5 (NetCDF 4 / HDF5) e restituisce un dizionario con
    tutti i dati in memoria.  Chiamare una volta sola all'avvio, poi passare
    l'oggetto alle funzioni.
    """
    with h5py.File(nc_path, "r") as ds:
        u100 = ds["u100"][:].astype(float)   # (T, lat, lon)
        v100 = ds["v100"][:].astype(float)
        lats = ds["latitude"][:].astype(float)
        lons = ds["longitude"][:].astype(float)
        n_times = u100.shape[0]

    return {
        "u100":    u100,
        "v100":    v100,
        "lats":    lats,
        "lons":    lons,
        "n_times": n_times,
    }


# ---------------------------------------------------------------------------
# MAPPATURA COORDINATE PIXEL → GRIGLIA ERA5
# ---------------------------------------------------------------------------

def _make_geo_coords(nodes, coords, n_lat, n_lon):
    """
    Mappa coordinate pixel (x, y) in indici frazionari sulla griglia ERA5.
    Restituisce dict: node_id -> (lat_frac, lon_frac).

    Convenzione:
      x cresce verso destra  → longitudine cresce verso est   (stessa direzione)
      y cresce verso il basso → latitudine cresce verso nord  (direzione invertita)
    """
    xs = [coords[i][0] for i in nodes]
    ys = [coords[i][1] for i in nodes]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    geo = {}
    for i in nodes:
        x, y = coords[i]
        lon_frac = (x - x_min) / (x_max - x_min + 1e-12) * (n_lon - 1)
        lat_frac = (1.0 - (y - y_min) / (y_max - y_min + 1e-12)) * (n_lat - 1)
        geo[i] = (lat_frac, lon_frac)
    return geo


def _bilinear(field_2d, lat_frac, lon_frac):
    """Interpolazione bilineare su griglia 2D (lat, lon)."""
    n_lat, n_lon = field_2d.shape
    r0 = max(0, min(int(math.floor(lat_frac)), n_lat - 2))
    c0 = max(0, min(int(math.floor(lon_frac)), n_lon - 2))
    dr = lat_frac - r0
    dc = lon_frac - c0
    return (field_2d[r0,     c0    ] * (1 - dr) * (1 - dc) +
            field_2d[r0 + 1, c0    ] *      dr  * (1 - dc) +
            field_2d[r0,     c0 + 1] * (1 - dr) *      dc  +
            field_2d[r0 + 1, c0 + 1] *      dr  *      dc)


# ---------------------------------------------------------------------------
# PERTURBAZIONE COMPLETA PER UNO SCENARIO  (formula additiva)
# ---------------------------------------------------------------------------

def build_wind_perturbation(
    scenario_id,
    nodes,
    base_dist,
    coords,
    wind,
    alpha=0.01,
    n_samples=5,
    eps=1e-6,
    turb_sigma=0.0,
):
    """
    Costruisce la perturbazione additiva dal vento per tutti gli archi i->j.

    Formula per ogni arco orientato i -> j:

        Delta_ij = -alpha * (L_ij / m) * sum_k [w(s_k) · d_hat_ij]
                   + turb_ij

    dove d_hat_ij è la direzione normalizzata (con y pixel invertita),
    e la distanza perturbata è  max(L_ij + Delta_ij, eps).

    turb_ij ~ N(0, turb_sigma * L_ij) è una turbolenza locale indipendente
    per ogni arco e scenario.  Rompe l'antisimmetria perfetta Delta(i->j)
    = -Delta(j->i) che altrimenti limita la diversificazione dei tour.

    NOTA: nessuna normalizzazione del vento (vedi documento di formulazione).
    Il parametro alpha assorbe la conversione dimensionale [m/s] -> [distanza].
    Calibrare alpha con perturbation_diagnostics().

    Parametri
    ---------
    scenario_id : int   — determina l'istante temporale (scenario_id % n_times)
    nodes       : list  — id nodo
    base_dist   : dict  — base_dist[i][j] = L_ij
    coords      : dict  — coords[i] = (px, py)  coordinate pixel
    wind        : dict  — restituito da load_wind_field()
    alpha       : float — forza della perturbazione
    n_samples   : int   — punti di campionamento lungo l'arco (midpoint rule)
    eps         : float — distanza minima ammessa (evita c_ij <= 0)
    turb_sigma  : float — dev. std relativa della turbolenza locale
                  (0.0 = nessuna; 0.05 = ~5% noise per arco)

    Restituisce
    -----------
    dict  (i, j) -> delta   dove  c_ij = max(L_ij + delta, eps)
    """
    rng = np.random.default_rng(scenario_id)
    t_idx = int(rng.integers(0, wind["n_times"]))
    n_lat = len(wind["lats"])
    n_lon = len(wind["lons"])

    u_field = wind["u100"][t_idx]   # (lat, lon) — componente est  [m/s]
    v_field = wind["v100"][t_idx]   # (lat, lon) — componente nord [m/s]

    geo = _make_geo_coords(nodes, coords, n_lat, n_lon)

    # frazioni midpoint: n_samples=5 → [0.1, 0.3, 0.5, 0.7, 0.9]
    sample_fracs = [(k + 0.5) / n_samples for k in range(n_samples)]

    pert = {}
    for i in nodes:
        xi, yi = coords[i]
        for j in nodes:
            if i == j:
                continue

            xj, yj = coords[j]
            L_ij = float(base_dist[i][j])

            if L_ij < eps:
                pert[(i, j)] = 0.0
                continue

            # direzione i→j in pixel (normalizzata, y invertita per nord)
            dx = xj - xi
            dy = yj - yi
            length = math.hypot(dx, dy)
            ux =  dx / length          # componente est
            uy = -dy / length          # componente nord (y pixel invertita)

            # coordinate griglia ERA5 dei nodi
            lat_i, lon_i = geo[i]
            lat_j, lon_j = geo[j]

            # campionamento del vento in m punti lungo l'arco
            proj_sum = 0.0
            for frac in sample_fracs:
                lat_s = lat_i + frac * (lat_j - lat_i)
                lon_s = lon_i + frac * (lon_j - lon_i)
                u = _bilinear(u_field, lat_s, lon_s)
                v = _bilinear(v_field, lat_s, lon_s)
                proj_sum += u * ux + v * uy

            # --- formula additiva ---
            # delta = -alpha * (L_ij / m) * sum_k proj_k
            #       = -alpha * L_ij * (proj_sum / m)
            delta = -alpha * L_ij * (proj_sum / n_samples)

            # turbolenza locale indipendente per arco
            if turb_sigma > 0:
                delta += float(rng.normal(0.0, turb_sigma * L_ij))

            pert[(i, j)] = delta

    return pert


# ---------------------------------------------------------------------------
# DIAGNOSTICHE
# ---------------------------------------------------------------------------

def perturbation_diagnostics(nodes, base_dist, all_pert, print_output=True):
    """Statistiche sulle perturbazioni relative Delta_ij / L_ij.

    Parametri
    ---------
    all_pert : dict[omega -> dict[(i,j) -> delta]]

    Restituisce dict con 'per_scenario' e 'global'.
    """
    all_rel = []
    per_scenario = []

    for omega, pert in all_pert.items():
        rel_deltas = []
        n_negative = 0
        asym_vals = []

        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                L = float(base_dist[i][j])
                if L < 1e-12:
                    continue
                d = pert[(i, j)]
                rel_deltas.append(d / L)
                all_rel.append(d / L)

                if L + d <= 0:
                    n_negative += 1

                d_rev = pert[(j, i)]
                asym_vals.append(abs(d + d_rev) / L)

        rd = np.array(rel_deltas)
        sc = {
            "omega": omega,
            "mean": rd.mean(), "std": rd.std(),
            "min": rd.min(), "max": rd.max(),
            "p1": np.percentile(rd, 1), "p5": np.percentile(rd, 5),
            "p50": np.percentile(rd, 50),
            "p95": np.percentile(rd, 95), "p99": np.percentile(rd, 99),
            "n_negative_dist": n_negative,
            "mean_asym": np.mean(asym_vals),
        }
        per_scenario.append(sc)

        if print_output:
            print(f"--- Scenario {omega} ---")
            print(f"  rel delta:  mean={sc['mean']:+.4f}  std={sc['std']:.4f}")
            print(f"  range:      [{sc['min']:+.4f}, {sc['max']:+.4f}]")
            print(f"  percentili: 1%={sc['p1']:+.4f}  5%={sc['p5']:+.4f}"
                  f"  50%={sc['p50']:+.4f}  95%={sc['p95']:+.4f}"
                  f"  99%={sc['p99']:+.4f}")
            print(f"  archi c<=0: {sc['n_negative_dist']}")
            print(f"  asimmetria: {sc['mean_asym']:.6f}")

    ar = np.array(all_rel)
    glob = {
        "mean": ar.mean(), "std": ar.std(),
        "min": ar.min(), "max": ar.max(),
        "p1": np.percentile(ar, 1), "p99": np.percentile(ar, 99),
    }
    if print_output:
        print("=== GLOBALE ===")
        print(f"  mean={glob['mean']:+.4f}  std={glob['std']:.4f}"
              f"  range=[{glob['min']:+.4f}, {glob['max']:+.4f}]"
              f"  1%={glob['p1']:+.4f}  99%={glob['p99']:+.4f}")

    return {"per_scenario": per_scenario, "global": glob}