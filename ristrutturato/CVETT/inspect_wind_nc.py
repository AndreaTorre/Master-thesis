import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================
# CONFIGURAZIONE
# ==========================

NC_FILE = "cvett.nc"

U_VAR = "u100"
V_VAR = "v100"
LAT_VAR = "latitude"
LON_VAR = "longitude"

OUT_DIR = "plots_cvett_clean"

# numero di istanti da plottare
N_PLOTS = 4

# più piccolo = più frecce
QUIVER_STEP = 1

# se None usa autoscaling di matplotlib
QUIVER_SCALE = None

FIGSIZE = (9, 7)
DPI = 220


# ==========================
# FUNZIONI
# ==========================

def stampa_struttura(h5):
    print("\n=== STRUTTURA FILE ===")

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:30s} shape={obj.shape} dtype={obj.dtype}")

    h5.visititems(visitor)


def leggi(h5, nome):
    return np.asarray(h5[nome][()])


def prendi_2d(arr, t_idx):
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[t_idx, :, :]
    if arr.ndim == 4:
        return arr[t_idx, 0, :, :]
    raise ValueError(f"Forma non gestita: {arr.shape}")


def statistiche(nome, arr):
    x = np.asarray(arr)
    x = x[np.isfinite(x)]

    print(f"\nStatistiche {nome}")
    print(f"min  = {np.min(x):.4f}")
    print(f"mean = {np.mean(x):.4f}")
    print(f"max  = {np.max(x):.4f}")
    print(f"std  = {np.std(x):.4f}")
    print(f"p05  = {np.percentile(x, 5):.4f}")
    print(f"p50  = {np.percentile(x, 50):.4f}")
    print(f"p95  = {np.percentile(x, 95):.4f}")


def plot_quiver_clean(X, Y, u, v, speed, out_file, title):
    plt.figure(figsize=FIGSIZE)

    q = plt.quiver(
        X, Y, u, v, speed,
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=QUIVER_SCALE,
        width=0.0022,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
        pivot="mid"
    )

    cbar = plt.colorbar(q)
    cbar.set_label("Intensità vento")

    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI)
    plt.close()


# ==========================
# MAIN
# ==========================

Path(OUT_DIR).mkdir(exist_ok=True)

with h5py.File(NC_FILE, "r") as h5:

    stampa_struttura(h5)

    print("\n=== VARIABILI USATE ===")
    print("u   =", U_VAR)
    print("v   =", V_VAR)
    print("lat =", LAT_VAR)
    print("lon =", LON_VAR)

    u_all = leggi(h5, U_VAR)
    v_all = leggi(h5, V_VAR)
    lat = leggi(h5, LAT_VAR)
    lon = leggi(h5, LON_VAR)

    print("\n=== SHAPE ===")
    print("u:", u_all.shape)
    print("v:", v_all.shape)
    print("lat:", lat.shape)
    print("lon:", lon.shape)

    if u_all.ndim == 2:
        n_times = 1
    else:
        n_times = u_all.shape[0]

    time_indices = np.linspace(0, n_times - 1, min(N_PLOTS, n_times), dtype=int)

    for t_idx in time_indices:

        print("\n==============================")
        print(f"ISTANTE TEMPORALE {t_idx}")
        print("==============================")

        u = prendi_2d(u_all, t_idx)
        v = prendi_2d(v_all, t_idx)

        # se necessario, trasponi per allineare a lat/lon
        if lat.ndim == 1 and lon.ndim == 1:
            expected_shape = (len(lat), len(lon))
            if u.shape != expected_shape and u.T.shape == expected_shape:
                u = u.T
                v = v.T
            X_full, Y_full = np.meshgrid(lon, lat)
        else:
            X_full, Y_full = lon, lat

        speed = np.sqrt(u**2 + v**2)

        statistiche("u", u)
        statistiche("v", v)
        statistiche("intensità vento", speed)

        # sottocampionamento per il quiver
        X = X_full[::QUIVER_STEP, ::QUIVER_STEP]
        Y = Y_full[::QUIVER_STEP, ::QUIVER_STEP]
        u_sub = u[::QUIVER_STEP, ::QUIVER_STEP]
        v_sub = v[::QUIVER_STEP, ::QUIVER_STEP]
        speed_sub = speed[::QUIVER_STEP, ::QUIVER_STEP]

        out_file = f"{OUT_DIR}/vento_quiver_t{t_idx}.png"
        title = f"Campo vettoriale del vento - t={t_idx}"

        plot_quiver_clean(X, Y, u_sub, v_sub, speed_sub, out_file, title)

        print("Salvato:", out_file)