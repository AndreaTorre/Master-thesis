# -*- coding: utf-8 -*-
"""
Perturbazioni basate su campo vettoriale di vento ERA5.
Tutti gli archi orientati vengono perturbati per ogni scenario/istante.

Dipendenze: numpy, scipy (niente netCDF4)
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

    Usa h5py invece di scipy.io.netcdf_file perché ERA5 produce file
    NetCDF 4 (basati su HDF5), non compatibili con il lettore NetCDF 3
    di scipy.
    """
    with h5py.File(nc_path, "r") as ds:
        u100 = ds["u100"][:].astype(float)   # (T, lat, lon)
        v100 = ds["v100"][:].astype(float)
        lats = ds["latitude"][:].astype(float)
        lons = ds["longitude"][:].astype(float)
        n_times = u100.shape[0]

    wind = {
        "u100"   : u100,
        "v100"   : v100,
        "lats"   : lats,
        "lons"   : lons,
        "n_times": n_times,
    }
    return wind


# ---------------------------------------------------------------------------
# MAPPATURA COORDINATE PIXEL → GRIGLIA
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
# PERTURBAZIONE COMPLETA PER UNO SCENARIO
# ---------------------------------------------------------------------------

def build_wind_perturbation(
    scenario_id,
    nodes,
    base_dist,
    coords,
    wind,
    alpha=0.01,
    min_factor=0.70,
    max_factor=1.50,
    n_samples=5,
    turb_sigma=0.0,
):
    """
    Costruisce scenario_dist perturbata dal vento per tutti gli archi i->j.

    Il vento viene campionato in n_samples punti equidistanti lungo ogni
    arco (non solo nel punto medio).

    Una componente di turbolenza locale ε_ij ~ N(0, turb_sigma²),
    indipendente per ogni arco orientato e scenario, viene sommata al
    fattore di perturbazione.  Fisicamente rappresenta raffiche, wind shear
    e incertezza residua nella stima del vento.  Poiché ε(i→j) e ε(j→i)
    sono indipendenti, la turbolenza rompe l'antisimmetria perfetta
    Δ(i→j) = −Δ(j→i) che impedisce ai tour di diversificarsi tra scenari.

    Parametri
    ---------
    scenario_id : int  — determina l'istante temporale (scenario_id % n_times)
    nodes       : lista degli id nodo
    base_dist   : dict base_dist[i][j]
    coords      : dict coords[i] = (px, py)
    wind        : dict restituito da load_wind_field()
    alpha       : forza della perturbazione (0.01 = ~1% per m/s di vento)
    min_factor  : fattore moltiplicativo minimo (es. 0.70 → -30% max)
    max_factor  : fattore moltiplicativo massimo (es. 1.50 → +50% max)
    n_samples   : int — numero di punti di campionamento lungo l'arco
                  (1 = solo punto medio, come prima; 5 = default consigliato)
    turb_sigma  : float — deviazione std della turbolenza locale
                  (0.0 = nessuna turbolenza, retrocompatibile;
                   0.05 = ~5% di rumore indipendente per arco)

    Restituisce
    -----------
    dict  (i, j) -> delta   dove delta = scenario_dist[i][j] - base_dist[i][j]
    """
    rng = np.random.default_rng(scenario_id)
    t_idx = int(rng.integers(0, wind["n_times"]))
    n_lat = len(wind["lats"])
    n_lon = len(wind["lons"])

    u_field = wind["u100"][t_idx]  # (lat, lon)
    v_field = wind["v100"][t_idx]

    # normalizzazione del vento: usa il 90° percentile per robustezza
    wind_scale = max(float(np.percentile(
        np.sqrt(u_field**2 + v_field**2), 90
    )), 1e-9)

    geo = _make_geo_coords(nodes, coords, n_lat, n_lon)

    # punti di campionamento: frazioni equidistanti lungo l'arco,
    # escludendo gli estremi (0 e 1) per evitare bias ai nodi
    # n_samples=1 → [0.5] (solo punto medio, retrocompatibile)
    # n_samples=5 → [0.1, 0.3, 0.5, 0.7, 0.9]
    sample_fracs = [(k + 0.5) / n_samples for k in range(n_samples)]

    scenario_dist = {i: {} for i in nodes}

    for i in nodes:
        xi, yi = coords[i]
        for j in nodes:
            if i == j:
                scenario_dist[i][j] = 0.0
                continue

            xj, yj = coords[j]

            # direzione i→j in pixel (normalizzata)
            dx = xj - xi
            dy = yj - yi
            length = math.hypot(dx, dy)
            ux =  dx / length
            uy = -dy / length  # y pixel invertita rispetto a nord

            # coordinate sulla griglia ERA5 dei nodi i e j
            lat_i, lon_i = geo[i]
            lat_j, lon_j = geo[j]

            # campionamento del vento in n_samples punti lungo l'arco
            proj_sum = 0.0
            for frac in sample_fracs:
                lat_s = lat_i + frac * (lat_j - lat_i)
                lon_s = lon_i + frac * (lon_j - lon_i)

                u = _bilinear(u_field, lat_s, lon_s) / wind_scale
                v = _bilinear(v_field, lat_s, lon_s) / wind_scale

                # proiezione: >0 = tailwind (accorcia), <0 = headwind
                proj_sum += u * ux + v * uy

            proj_avg = proj_sum / n_samples

            # turbolenza locale: ε_ij indipendente per ogni arco orientato
            eps = float(rng.normal(0.0, turb_sigma)) if turb_sigma > 0 else 0.0

            factor = float(np.clip(1.0 - alpha * proj_avg + eps,
                                   min_factor, max_factor))
            scenario_dist[i][j] = float(base_dist[i][j]) * factor

    return {(i, j): scenario_dist[i][j] - float(base_dist[i][j])
        for i in scenario_dist
        for j in scenario_dist[i]
        if i != j}