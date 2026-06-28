# -*- coding: utf-8 -*-
"""
Plot geospaziale per CVETT: mappa + campo vettoriale ERA5 + punti TSP.

Versione Montréal, senza cartopy e senza pyproj.

Caratteristiche:
- usa h5py tramite wind_perturbation.load_wind_field;
- usa Natural Earth tramite pyshp, se gli shapefile sono presenti;
- usa l'estensione effettiva del file ERA5 come area del grafico;
- densifica il campo vettoriale solo per la visualizzazione;
- mappa i nodi TSP nello stesso rettangolo geografico del campo ERA5.

Uso:
    cd /home/atorre/UTSP/unione/git/UTSP/CVETT

    source /home/atorre/UTSP/unione/git/UTSP/venv/bin/activate
    export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH

    TESI_EXPERIMENT=CVETT TESI_N_NODES=25 python plot_mappa_vento_tsp_montreal.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_data
from config import ERA5_NC_PATH, OUTPUT_DIR
from wind_perturbation import load_wind_field


# =============================================================================
# AREA RICHIESTA A ERA5: MONTRÉAL
# =============================================================================
# Richiesta sul sito:
# N = 46
# S = 45
# W = -74.5
# E = -72.9
#
# Nota: ERA5 usa una griglia a passo 0.25°. Se chiedi E=-72.9,
# il file può fermarsi a -73.0. Per questo il grafico usa i limiti effettivi
# letti dal file dopo il ritaglio, non i limiti teorici della richiesta.
REQ_LAT_MIN = 45.0
REQ_LAT_MAX = 46.0
REQ_LON_MIN = -74.5
REQ_LON_MAX = -72.9


# =============================================================================
# OUTPUT
# =============================================================================
OUT_DIR = Path(OUTPUT_DIR) / "grafici" / "mappe_vento_tsp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PARAMETRI GRAFICI
# =============================================================================
SCENARIOS_TO_PLOT = [1, 2, 3]

# QUIVER_STEP = 1 usa tutti i punti della griglia plottata.
QUIVER_STEP = 1

# Densifica solo la visualizzazione del vento.
# Il file ERA5 non viene modificato e il modello non cambia.
# Con griglia ERA5 5x7:
#   1 -> 5x7 = 35 frecce
#   3 -> 15x21 = 315 frecce
#   4 -> 20x28 = 560 frecce
#   5 -> 25x35 = 875 frecce
QUIVER_DENSIFY = 4

# Valori più grandi = frecce più corte.
# Se vuoi autoscaling matplotlib, metti None.
QUIVER_SCALE = 70

FIGSIZE = (9.5, 7.3)
DPI = 220
SHOW_NODE_LABELS = True
DRAW_IMPORTANT_ARCS = False

# Piccolo margine visivo intorno all'estensione ERA5.
# 0.00 = assi esattamente uguali al campo vento.
AXIS_PAD_FRAC = 0.00


# =============================================================================
# FUNZIONI PER LAT/LON E CAMPO VENTO
# =============================================================================
def _normalize_lons(lons: np.ndarray) -> np.ndarray:
    """Porta le longitudini in [-180, 180] se ERA5 le salva in 0..360."""
    lons = np.asarray(lons, dtype=float)
    if np.nanmax(lons) > 180.0:
        lons = ((lons + 180.0) % 360.0) - 180.0
    return lons


def _ensure_lat_lon_shape(
    u: np.ndarray,
    v: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Allinea u/v alla forma (lat, lon)."""
    expected = (len(lats), len(lons))

    if u.shape == expected:
        return u, v

    if u.T.shape == expected:
        return u.T, v.T

    raise ValueError(
        f"Forma vento non compatibile con lat/lon: u={u.shape}, "
        f"v={v.shape}, expected={expected}"
    )


def _crop_and_sort_field(
    u: np.ndarray,
    v: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ordina lat/lon in senso crescente e ritaglia il campo alla finestra richiesta.
    Restituisce u, v, lats, lons con forma u/v = (lat, lon).
    """
    lats = np.asarray(lats, dtype=float)
    lons = _normalize_lons(np.asarray(lons, dtype=float))

    u, v = _ensure_lat_lon_shape(u, v, lats, lons)

    lat_order = np.argsort(lats)
    lon_order = np.argsort(lons)

    lats = lats[lat_order]
    lons = lons[lon_order]
    u = u[np.ix_(lat_order, lon_order)]
    v = v[np.ix_(lat_order, lon_order)]

    lat_low, lat_high = sorted([lat_min, lat_max])
    lon_low, lon_high = sorted([lon_min, lon_max])

    lat_mask = (lats >= lat_low) & (lats <= lat_high)
    lon_mask = (lons >= lon_low) & (lons <= lon_high)

    if not np.any(lat_mask):
        raise ValueError(
            f"Nessuna latitudine ERA5 dentro [{lat_low}, {lat_high}]. "
            f"Lat disponibili: min={lats.min()}, max={lats.max()}"
        )

    if not np.any(lon_mask):
        raise ValueError(
            f"Nessuna longitudine ERA5 dentro [{lon_low}, {lon_high}]. "
            f"Lon disponibili: min={lons.min()}, max={lons.max()}"
        )

    lats_c = lats[lat_mask]
    lons_c = lons[lon_mask]
    u_c = u[np.ix_(lat_mask, lon_mask)]
    v_c = v[np.ix_(lat_mask, lon_mask)]

    return u_c, v_c, lats_c, lons_c


def densify_wind_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    factor: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpola il campo vento su una griglia lon/lat più fitta.

    Serve solo per il grafico:
    - non modifica il file ERA5;
    - non modifica gli scenari;
    - non modifica la perturbazione usata dal modello.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if factor <= 1:
        X, Y = np.meshgrid(lon, lat)
        speed = np.sqrt(u ** 2 + v ** 2)
        return X, Y, u, v, speed

    if lon[0] > lon[-1]:
        lon = lon[::-1]
        u = u[:, ::-1]
        v = v[:, ::-1]

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u = u[::-1, :]
        v = v[::-1, :]

    n_lat_new = max(len(lat) * factor, len(lat))
    n_lon_new = max(len(lon) * factor, len(lon))

    lat_new = np.linspace(float(lat.min()), float(lat.max()), n_lat_new)
    lon_new = np.linspace(float(lon.min()), float(lon.max()), n_lon_new)

    # 1) interpolazione lungo la longitudine
    u_lon = np.empty((len(lat), len(lon_new)))
    v_lon = np.empty((len(lat), len(lon_new)))

    for r in range(len(lat)):
        u_lon[r, :] = np.interp(lon_new, lon, u[r, :])
        v_lon[r, :] = np.interp(lon_new, lon, v[r, :])

    # 2) interpolazione lungo la latitudine
    u_new = np.empty((len(lat_new), len(lon_new)))
    v_new = np.empty((len(lat_new), len(lon_new)))

    for c in range(len(lon_new)):
        u_new[:, c] = np.interp(lat_new, lat, u_lon[:, c])
        v_new[:, c] = np.interp(lat_new, lat, v_lon[:, c])

    X_new, Y_new = np.meshgrid(lon_new, lat_new)
    speed_new = np.sqrt(u_new ** 2 + v_new ** 2)

    return X_new, Y_new, u_new, v_new, speed_new


def time_index_from_scenario(scenario_id: int, n_times: int) -> int:
    """Replica la scelta dell'istante temporale usata in build_wind_perturbation."""
    rng = np.random.default_rng(scenario_id)
    return int(rng.integers(0, n_times))


# =============================================================================
# MAPPATURA PUNTI TSP
# =============================================================================
def tsp_pixel_to_lonlat(
    nodes,
    coords,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
):
    nodes = list(nodes)

    xs = np.array([coords[i][0] for i in nodes], dtype=float)
    ys = np.array([coords[i][1] for i in nodes], dtype=float)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    out = {}

    for i in nodes:
        x, y = coords[i]

        x_frac = (x - x_min) / (x_max - x_min + 1e-12)
        y_frac = (y - y_min) / (y_max - y_min + 1e-12)

        lon = lon_min + x_frac * (lon_max - lon_min)

        # NON invertire y: mantiene la disposizione dei grafici TSP originali
        lat = lat_min + y_frac * (lat_max - lat_min)

        out[i] = (lon, lat)

    return out


# =============================================================================
# SFONDO GEOGRAFICO SENZA CARTOPY/PYPROJ
# =============================================================================
def _natural_earth_candidate_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        here.parents[1] / "data" / "naturalearth",  # UTSP/data/naturalearth
        here.parent / "naturalearth",                # CVETT/naturalearth
        Path.cwd() / "data" / "naturalearth",
        Path.cwd() / "naturalearth",
    ]


def _find_natural_earth_dir() -> Optional[Path]:
    for d in _natural_earth_candidate_dirs():
        if d.exists() and any(d.glob("*.shp")):
            return d
    return None


def _bbox_intersects(
    bbox,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> bool:
    x0, y0, x1, y1 = bbox
    return not (x1 < lon_min or x0 > lon_max or y1 < lat_min or y0 > lat_max)


def _iter_shape_parts(shape):
    pts = shape.points
    parts = list(shape.parts) + [len(pts)]

    for a, b in zip(parts[:-1], parts[1:]):
        part = pts[a:b]
        if len(part) >= 2:
            xs = [p[0] for p in part]
            ys = [p[1] for p in part]
            yield xs, ys


def _draw_shapefile_lines(
    ax,
    shp_path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    *,
    color: str = "0.35",
    linewidth: float = 0.5,
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    import shapefile

    reader = shapefile.Reader(str(shp_path))

    for shape in reader.shapes():
        if not _bbox_intersects(shape.bbox, lon_min, lon_max, lat_min, lat_max):
            continue

        for xs, ys in _iter_shape_parts(shape):
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _draw_shapefile_polygons(
    ax,
    shp_path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    *,
    facecolor: str = "0.94",
    edgecolor: str = "none",
    linewidth: float = 0.0,
    alpha: float = 1.0,
    zorder: int = 0,
) -> None:
    import shapefile

    reader = shapefile.Reader(str(shp_path))

    for shape in reader.shapes():
        if not _bbox_intersects(shape.bbox, lon_min, lon_max, lat_min, lat_max):
            continue

        for xs, ys in _iter_shape_parts(shape):
            ax.fill(
                xs,
                ys,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )


def _make_geo_axis(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    lon_pad = (lon_max - lon_min) * AXIS_PAD_FRAC
    lat_pad = (lat_max - lat_min) * AXIS_PAD_FRAC

    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    ax.grid(alpha=0.25, linewidth=0.5)

    ne_dir = _find_natural_earth_dir()

    if ne_dir is None:
        print("Sfondo geografico non trovato: continuo senza Natural Earth.")
        return fig, ax

    try:
        land = ne_dir / "ne_10m_land.shp"
        lakes = ne_dir / "ne_10m_lakes.shp"
        rivers = ne_dir / "ne_10m_rivers_lake_centerlines.shp"
        coastline = ne_dir / "ne_10m_coastline.shp"
        borders = ne_dir / "ne_10m_admin_0_boundary_lines_land.shp"

        if land.exists():
            _draw_shapefile_polygons(
                ax,
                land,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                facecolor="0.94",
                edgecolor="none",
                zorder=0,
            )

        if lakes.exists():
            _draw_shapefile_polygons(
                ax,
                lakes,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                facecolor="0.985",
                edgecolor="0.55",
                linewidth=0.3,
                zorder=1,
            )

        if rivers.exists():
            _draw_shapefile_lines(
                ax,
                rivers,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                color="0.55",
                linewidth=0.35,
                alpha=0.8,
                zorder=1,
            )

        if coastline.exists():
            _draw_shapefile_lines(
                ax,
                coastline,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                color="0.25",
                linewidth=0.65,
                zorder=2,
            )

        if borders.exists():
            _draw_shapefile_lines(
                ax,
                borders,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                color="0.35",
                linewidth=0.55,
                alpha=0.8,
                zorder=2,
            )

        print(f"Sfondo geografico letto da: {ne_dir}")

    except ImportError:
        print("Pacchetto 'pyshp' non installato. Installa con: pip install pyshp")
    except Exception as exc:
        print("Sfondo geografico non disegnato:", exc)

    return fig, ax


# =============================================================================
# PLOT PRINCIPALE
# =============================================================================
def plot_scenario_map(scenario_id: int) -> Path:
    nodes, coords, base_dist, E, root = load_data()
    wind = load_wind_field(ERA5_NC_PATH)

    t_idx = time_index_from_scenario(scenario_id, wind["n_times"])

    u_raw = wind["u100"][t_idx]
    v_raw = wind["v100"][t_idx]
    lats_raw = wind["lats"]
    lons_raw = wind["lons"]

    u, v, lats, lons = _crop_and_sort_field(
        u_raw,
        v_raw,
        lats_raw,
        lons_raw,
        REQ_LAT_MIN,
        REQ_LAT_MAX,
        REQ_LON_MIN,
        REQ_LON_MAX,
    )

    # Limiti effettivi del file ERA5: questi governano TUTTO il grafico.
    plot_lat_min = float(np.min(lats))
    plot_lat_max = float(np.max(lats))
    plot_lon_min = float(np.min(lons))
    plot_lon_max = float(np.max(lons))

    print("")
    print(f"Scenario {scenario_id} | t_idx={t_idx}")
    print("ERA5_NC_PATH:", ERA5_NC_PATH)
    print("Griglia ERA5 ritagliata:", u.shape)
    print("Latitudini usate:", lats)
    print("Longitudini usate:", lons)
    print(
        "LIMITI PLOT USATI:",
        f"lon=[{plot_lon_min:.4f}, {plot_lon_max:.4f}]",
        f"lat=[{plot_lat_min:.4f}, {plot_lat_max:.4f}]",
    )

    X_full, Y_full, u_plot, v_plot, speed_plot = densify_wind_grid(
        lons,
        lats,
        u,
        v,
        factor=QUIVER_DENSIFY,
    )

    X_q = X_full[::QUIVER_STEP, ::QUIVER_STEP]
    Y_q = Y_full[::QUIVER_STEP, ::QUIVER_STEP]
    u_q = u_plot[::QUIVER_STEP, ::QUIVER_STEP]
    v_q = v_plot[::QUIVER_STEP, ::QUIVER_STEP]
    speed_q = speed_plot[::QUIVER_STEP, ::QUIVER_STEP]

    print("Griglia vento plottata:", u_q.shape)
    print("Numero frecce:", u_q.size)

    tsp_geo = tsp_pixel_to_lonlat(
        nodes,
        coords,
        plot_lat_min,
        plot_lat_max,
        plot_lon_min,
        plot_lon_max,
    )

    node_lons = [tsp_geo[i][0] for i in nodes]
    node_lats = [tsp_geo[i][1] for i in nodes]

    fig, ax = _make_geo_axis(plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)

    q = ax.quiver(
        X_q,
        Y_q,
        u_q,
        v_q,
        speed_q,
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=QUIVER_SCALE,
        width=0.0020,
        headwidth=3.3,
        headlength=4.2,
        headaxislength=3.8,
        pivot="mid",
        alpha=0.82,
        zorder=3,
    )

    cbar = fig.colorbar(q, ax=ax, shrink=0.78, pad=0.03)
    cbar.set_label("Intensità vento (m/s)")

    ax.scatter(
        node_lons,
        node_lats,
        s=52,
        c="black",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
        label="Nodi TSP",
    )

    if SHOW_NODE_LABELS:
        for i in nodes:
            lon_i, lat_i = tsp_geo[i]
            ax.text(
                lon_i,
                lat_i,
                str(i),
                fontsize=7,
                ha="left",
                va="bottom",
                zorder=6,
            )

    if DRAW_IMPORTANT_ARCS:
        try:
            from config import K_MEDOID_NODES, MAX_KMEDOID_I_ARCS
            from gurobi_models import build_I_from_medoid_outgoing_nodes
            from tsp_utils import canon_edge

            I = build_I_from_medoid_outgoing_nodes(
                K_MEDOID_NODES,
                nodes,
                base_dist,
                MAX_KMEDOID_I_ARCS,
            )

            I_set = sorted({canon_edge(i, j) for (i, j) in I})

            for a, b in I_set:
                lon_a, lat_a = tsp_geo[a]
                lon_b, lat_b = tsp_geo[b]

                ax.plot(
                    [lon_a, lon_b],
                    [lat_a, lat_b],
                    color="red",
                    linewidth=1.4,
                    alpha=0.8,
                    zorder=4,
                )

        except Exception as exc:
            print("Archi importanti non disegnati:", exc)

    ax.set_title(
        f"Mappa + campo vettoriale ERA5 + punti TSP | "
        f"scenario={scenario_id}, t_idx={t_idx}"
    )
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left")

    fig.tight_layout()

    out_file = OUT_DIR / f"mappa_vento_tsp_montreal_scen{scenario_id:03d}_t{t_idx}.png"
    fig.savefig(out_file, dpi=DPI)
    plt.close(fig)

    return out_file


def main() -> None:
    print("=" * 72)
    print("PLOT MAPPA + VENTO + TSP | VERSIONE MONTRÉAL")
    print("=" * 72)
    print("Coordinate richiesta ERA5:")
    print(f"  N={REQ_LAT_MAX}, S={REQ_LAT_MIN}, W={REQ_LON_MIN}, E={REQ_LON_MAX}")
    print("Coordinate effettive del plot: lette dal file ERA5 dopo il ritaglio.")

    for sid in SCENARIOS_TO_PLOT:
        out = plot_scenario_map(sid)
        print("Salvato:", out)


if __name__ == "__main__":
    main()
