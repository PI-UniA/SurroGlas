# plot_main_fem.py  —  Aronen & Karvinen (2018) 1D validation
# Figures 4, 5, 6, 7, 8 styled to match the paper exactly:
#   - Black & white, no color
#   - Solid = Surface, Dashed = Mid-plane, Dotted = Difference
#   - Horizontal grid lines only, light gray
#   - Log time axis 0.001 – 100 s

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Global style matching paper ──────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.linewidth":    1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  5,
    "ytick.major.size":  5,
    "legend.frameon":    False,
    "legend.fontsize":   10,
    "figure.dpi":        150,
})

LW_MAIN = 2.0    # main curve line width
LW_DIFF = 1.5    # difference curve line width

# ============================================================
# Settings
# ============================================================
DATA_DIR = "results/fem_aronen2018"
PLOT_DIR  = "plots_aronen2018"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Simulation settings — loaded automatically from meta.json ────────────
# These defaults are overridden by meta.json if it exists.
t_start      = 0.0
t_end        = 400.0
dt           = 0.1
Nt           = int(round((t_end - t_start) / dt))
THICKNESS_MM = 4.0
N_NODES      = 29

def _load_meta():
    """Load simulation metadata saved by main_fem.py."""
    import json
    meta_path = os.path.join(DATA_DIR, "meta.json")
    global t_start, t_end, dt, Nt, THICKNESS_MM, N_NODES
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            m = json.load(f)
        t_start      = float(m.get("t_start",      t_start))
        t_end        = float(m.get("t_end",         t_end))
        dt           = float(m.get("dt",            dt))
        Nt           = int(m.get("Nt",              Nt))
        THICKNESS_MM = float(m.get("thickness_mm",  THICKNESS_MM))
        N_NODES      = int(m.get("n_nodes",         N_NODES))
        print(f"  Loaded meta.json: t_end={t_end}s  dt={dt}s  "
              f"Nt={Nt}  N_NODES={N_NODES}  b={THICKNESS_MM}mm")
    else:
        # Fallback: infer from temperature file size
        p = os.path.join(DATA_DIR, "temperature_all.txt")
        if not os.path.exists(p):
            return
        total = int(np.loadtxt(p).size)
        if total % N_NODES == 0:
            Nt = total // N_NODES
            dt = (t_end - t_start) / Nt
            print(f"  meta.json not found — inferred: Nt={Nt}, dt={dt:.5f}s")
        else:
            print(f"  WARNING: meta.json missing and file size {total} "
                  f"not divisible by N_NODES={N_NODES}. "
                  f"Set t_end and N_NODES manually in plot_main_fem.py.")

_load_meta()
t_array = np.linspace(t_start + dt, t_end, Nt)

# ============================================================
# Load
# ============================================================

def load_fields():
    temp_path  = os.path.join(DATA_DIR, "temperature_all.txt")
    stress_path= os.path.join(DATA_DIR, "stress_all.txt")
    ftmp_path  = os.path.join(DATA_DIR, "fictive_temp_all.txt")

    for p in (temp_path, stress_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    arr_T = np.loadtxt(temp_path)
    arr_s = np.loadtxt(stress_path)

    # ── DOF counts ────────────────────────────────────────────────────────
    global N_NODES
    nx_T = arr_T.size // Nt      # temperature DOFs per step
    nx_s = arr_s.size // Nt      # stress DOFs per step (always CG nodes)

    # ── Temperature: sort DG DOFs into spatial order ───────────────────────
    coord_path = os.path.join(DATA_DIR, "T_dof_coords.txt")
    if os.path.exists(coord_path) and nx_T != N_NODES:
        # DG mode: load DOF x-coordinates and sort into spatial order
        x_dofs  = np.loadtxt(coord_path)
        sort_idx = np.argsort(x_dofs)          # left → right
        arr_T_s  = arr_T.reshape(Nt, nx_T)     # (Nt, nx_T)
        arr_T_sorted = arr_T_s[:, sort_idx].T  # (nx_T, Nt) spatially ordered
        temperature  = arr_T_sorted
        # x-coordinates of DG DOFs (sorted)
        x_T = x_dofs[sort_idx] * 1000.0        # m → mm
        print(f"  DG mode: {nx_T} temperature DOFs sorted by coordinate")
    else:
        # CG mode: DOFs already in spatial order
        temperature = arr_T.reshape(Nt, nx_T).T
        x_T = np.linspace(-THICKNESS_MM/2, THICKNESS_MM/2, nx_T)

    # ── Stress: always CG, already in order ────────────────────────────────
    dofs_s     = arr_s.size // Nt
    nc         = dofs_s // N_NODES if N_NODES > 0 else 1
    nx_s_nodes = N_NODES
    if dofs_s % N_NODES != 0:
        nx_s_nodes = dofs_s
        nc = 1
    stress = (arr_s.reshape(Nt, nx_s_nodes, nc)[:, :, 0].T
              if nc > 1 else arr_s.reshape(Nt, nx_s_nodes).T)

    # ── Fictive temperature ────────────────────────────────────────────────
    fictive = None
    if os.path.exists(ftmp_path):
        arr_f   = np.loadtxt(ftmp_path)
        nx_f    = arr_f.size // Nt
        if os.path.exists(coord_path) and nx_f != N_NODES:
            arr_f_s  = arr_f.reshape(Nt, nx_f)
            fictive  = arr_f_s[:, sort_idx].T
        else:
            fictive  = arr_f.reshape(Nt, nx_f).T

    # ── Spatial axes ────────────────────────────────────────────────────────
    nx = nx_s_nodes                                           # stress nodes
    z  = np.linspace(-THICKNESS_MM/2, THICKNESS_MM/2, nx)   # stress z-axis
    print(f"  Loaded: T_dofs={nx_T}, σ_nodes={nx}, Nt={Nt}, "
          f"dt={dt:.5f}s, σ components/node={nc}")
    return temperature, stress, fictive, nx, z, x_T


def surf_mid(nx):
    """
    Return (surface_idx, midplane_idx) for the stress array (always CG).
    Node 0 = surface (x = -h/2), node nx//2 = midplane (x = 0).
    Works correctly because stress is always stored in CG (N_NODES nodes).
    """
    return 0, nx // 2


def _save(fig, name):
    p = os.path.join(PLOT_DIR, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {p}")


def _hgrid(ax):
    """Horizontal grid lines only, light gray — matching paper style."""
    ax.yaxis.grid(True, color="lightgray", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def _log_taxis(ax):
    ax.set_xscale("log")
    # x-axis: from first saved timestep to t_end, with small margins
    x_max = 10 ** (np.ceil(np.log10(t_end)))   # round up to next power of 10
    ax.set_xlim(1e-3, x_max)
    ax.set_xlabel("Time $t$ [s]", fontsize=12)


# ============================================================
# Paper digitised data (Aronen & Karvinen 2018)
# ============================================================

def _load_paper_data():
    """
    Load digitised stress and temperature data from Aronen & Karvinen (2018).
    CSV files must be in the same directory as this script.
    Returns stress and temperature arrays as (N,2) [t_s, value],
    or None if files not found.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _load_csv(filename, label):
        path = os.path.join(script_dir, filename)
        if not os.path.exists(path):
            print(f"  NOTE: {label} CSV not found — skipping overlay.")
            print(f"  Expected: {path}")
            return None
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        arr = arr[arr[:, 0].argsort()]
        print(f"  Loaded {label}: {len(arr)} points")
        return arr

    ps  = _load_csv("Aronen_data/aronen2018_surface_stress.csv",      "stress surface")
    pm  = _load_csv("Aronen_data/aronen2018_midplane_stress.csv",     "stress mid-plane")
    ts  = _load_csv("Aronen_data/aronen2018_surface_temperature.csv", "temperature surface")
    tm  = _load_csv("Aronen_data/aronen2018_midplane_temperature.csv","temperature mid-plane")

    # Temperature profiles through thickness at fixed times
    prof = {}
    for t_label in ["t0", "t5", "t10", "t20", "t100"]:
        fname = f"Aronen_data/aronen2018_temp_profile_{t_label}.csv"
        arr   = _load_csv(fname, f"T profile {t_label}")
        if arr is not None:
            arr = arr[arr[:, 0].argsort()]   # sort by z
        prof[t_label] = arr
    # Stress profiles through thickness at fixed times
    sprof = {}
    for t_label in ["t0", "t5", "t10", "t20", "t100"]:
        fname = f"Aronen_data/aronen2018_stress_profile_{t_label}.csv"
        arr   = _load_csv(fname, f"σ profile {t_label}")
        if arr is not None:
            arr = arr[arr[:, 0].argsort()]
        sprof[t_label] = arr

    return ps, pm, ts, tm, prof, sprof

(PAPER_SURF, PAPER_MID, PAPER_T_SURF, PAPER_T_MID,
 PAPER_T_PROF, PAPER_S_PROF) = _load_paper_data()


# ============================================================
# Fig. 4 — Temperature vs time + ΔT right axis
# ============================================================

def plot_fig4(temperature, nx, x_T=None):
    s, m = surf_mid(nx)
    # Map temperature DOFs to surface/midplane positions using coordinates
    nx_T = temperature.shape[0]
    if nx_T != nx and x_T is not None:
        # DG: find DOF closest to surface (x=-h/2) and midplane (x=0)
        idx_surf = np.argmin(np.abs(x_T - (-THICKNESS_MM/2)))
        idx_mid  = np.argmin(np.abs(x_T))
    else:
        idx_surf, idx_mid = s, m
    T_s = temperature[idx_surf] - 273.15
    T_m = temperature[idx_mid]  - 273.15
    dT  = T_m - T_s

    # Prepend IC at t=1e-3 (log axis left edge): T0=650°C, ΔT=0
    t_plot   = np.concatenate([[1e-3], t_array])
    T_s_plot = np.concatenate([[650.0], T_s])
    T_m_plot = np.concatenate([[650.0], T_m])
    dT_plot  = np.concatenate([[0.0],   dT])

    fig, ax1 = plt.subplots(figsize=(7.0, 5.0))
    _hgrid(ax1)

    # Temperature curves — Surface solid, Mid-plane dashed (matching paper)
    l1, = ax1.plot(t_plot, T_s_plot, color="black", lw=LW_MAIN, ls="-",
                   label="Surface")
    l2, = ax1.plot(t_plot, T_m_plot, color="black", lw=LW_MAIN, ls="--",
                   label="Mid-plane")
    ax1.set_ylabel("Temperature $T$ [°C]", fontsize=12)
    ax1.set_ylim(0, 700)
    ax1.set_yticks(range(0, 701, 100))
    _log_taxis(ax1)

    # Right axis: temperature difference ΔT
    ax2 = ax1.twinx()
    l3, = ax2.plot(t_plot, dT_plot, color="black", lw=LW_DIFF, ls=":",
                   label="Difference")
    ax2.set_ylabel(r"Temperature difference $\Delta T$ [°C]", fontsize=12)
    dT_max = float(dT.max())
    # Right axis: 0 to 140°C (matching paper Fig. 4)
    ax2.set_ylim(0, 140)
    ax2.set_yticks(range(0, 141, 20))
    ax2.spines["right"].set_visible(True)

    # Single combined legend, upper-left, no frame
    ax1.legend(handles=[l1, l2, l3], loc="upper left",
               frameon=False, fontsize=10)

    fig.tight_layout()
    _save(fig, "fig4_temperature_vs_time.png")


# ============================================================
# Fig. 5 — Stress vs time
# ============================================================

def plot_fig5(stress, nx):
    s, m = surf_mid(nx)
    sig_s = stress[s] / 1e6
    sig_m = stress[m] / 1e6

    # Prepend IC: at t→0, stress = 0
    t_plot   = np.concatenate([[1e-3], t_array])
    sig_s_pl = np.concatenate([[0.0],  sig_s])
    sig_m_pl = np.concatenate([[0.0],  sig_m])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _hgrid(ax)

    ax.plot(t_plot, sig_s_pl, color="black", lw=LW_MAIN, ls="-",  label="Surface")
    ax.plot(t_plot, sig_m_pl, color="black", lw=LW_MAIN, ls="--", label="Mid-plane")
    ax.axhline(0, color="black", lw=0.8)

    ax.set_ylabel(r"Stress $\sigma$ [MPa]", fontsize=12)
    ax.set_ylim(-140, 80)
    ax.set_yticks(range(-140, 81, 20))
    _log_taxis(ax)
    ax.legend(loc="lower left")

    # Annotate residual values
    n_res = max(1, Nt // 20)
    s_res = float(np.mean(sig_s[-n_res:]))
    m_res = float(np.mean(sig_m[-n_res:]))
    ax.text(0.97, 0.08,
            f"Residual:  surface = {s_res:.0f} MPa\n"
            f"           mid-plane = +{m_res:.0f} MPa",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="lightgray"))

    fig.tight_layout()
    _save(fig, "fig5_stress_vs_time.png")


# ============================================================
# Fig. 6 — Temperature profiles through thickness
# ============================================================

def plot_fig6(temperature, z, x_T=None):
    time_slots = [0.0, 0.2, 2.0, 5.0, 10.0, 20.0, 100.0]
    labels     = ["0", "0.2", "2", "5", "10", "20", "100"]
    # Line styles cycling through solid/dashed/dashdot to distinguish without color
    styles = ["-", "--", "-.", ":", (0,(5,1)), (0,(3,1,1,1)), (0,(1,1))]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    _hgrid(ax)

    nx_T = temperature.shape[0]
    use_x_T = (x_T is not None and nx_T != len(z))
    plot_x   = x_T if use_x_T else z    # DG: use DOF coords; CG: use z grid
    for ts, lbl, ls in zip(time_slots, labels, styles):
        if ts <= 0:
            y = np.full(nx_T if use_x_T else len(z), 650.0)
        else:
            idx = max(0, min(int(round(ts / dt)) - 1, Nt - 1))
            y   = temperature[:, idx] - 273.15
        ax.plot(plot_x, y, color="black", lw=1.6, ls=ls, label=lbl)
    ax.set_xlim(-THICKNESS_MM/2, THICKNESS_MM/2)

    ax.set_xlabel("$z$ [mm]", fontsize=12)
    ax.set_ylabel("Temperature $T$ [°C]", fontsize=12)
    ax.set_xlim(-THICKNESS_MM/2, THICKNESS_MM/2)
    ax.set_ylim(0, 700)
    ax.set_yticks(range(0, 701, 100))
    ax.legend(title="Time $t$ [s]", fontsize=9, title_fontsize=9,
              ncol=2, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig6_temperature_profiles.png")


# ============================================================
# Fig. 7 — Stress profiles through thickness
# ============================================================

def plot_fig7(stress, z):
    time_slots = [0.0, 0.2, 2.0, 5.0, 10.0, 20.0, 100.0]
    labels     = ["0", "0.2", "2", "5", "10", "20", "100"]
    styles     = ["-", "--", "-.", ":", (0,(5,1)), (0,(3,1,1,1)), (0,(1,1))]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    _hgrid(ax)

    for ts, lbl, ls in zip(time_slots, labels, styles):
        if ts <= 0:
            y = np.zeros_like(z)
        else:
            idx = max(0, min(int(round(ts / dt)) - 1, Nt - 1))
            y   = stress[:, idx] / 1e6
        ax.plot(z, y, color="black", lw=1.6, ls=ls, label=lbl)

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("$z$ [mm]", fontsize=12)
    ax.set_ylabel(r"Stress $\sigma$ [MPa]", fontsize=12)
    ax.set_xlim(-THICKNESS_MM/2, THICKNESS_MM/2)
    ax.legend(title="Time $t$ [s]", fontsize=9, title_fontsize=9,
              ncol=2, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig7_stress_profiles.png")


# ============================================================
# Fig. 8 — Fictive temperature vs time
# ============================================================

def plot_fig8(temperature, fictive, nx, x_T=None):
    s, m = surf_mid(nx)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _hgrid(ax)

    nx_T = temperature.shape[0]
    if nx_T != nx and x_T is not None:
        idx_surf = np.argmin(np.abs(x_T - (-THICKNESS_MM/2)))
        idx_mid  = np.argmin(np.abs(x_T))
    else:
        idx_surf, idx_mid = s, m

    if fictive is not None:
        Tf_s = fictive[idx_surf] - 273.15
        Tf_m = fictive[idx_mid]  - 273.15
    else:
        print("  WARNING: fictive temperature not saved — showing real T as proxy")
        Tf_s = temperature[idx_surf] - 273.15
        Tf_m = temperature[idx_mid]  - 273.15

    # Prepend IC: at t→0, Tf = T0 = 650°C
    t_plot   = np.concatenate([[1e-3], t_array])
    Tf_s_pl  = np.concatenate([[650.0], Tf_s])
    Tf_m_pl  = np.concatenate([[650.0], Tf_m])

    ax.plot(t_plot, Tf_s_pl, color="black", lw=LW_MAIN, ls="-",  label="Surface")
    ax.plot(t_plot, Tf_m_pl, color="black", lw=LW_MAIN, ls="--", label="Mid-plane")
    ax.set_ylabel("Fictive temperature $T_f$ [°C]", fontsize=12)
    ax.set_ylim(580, 660)
    ax.set_yticks(range(580, 661, 10))
    _log_taxis(ax)
    ax.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, "fig8_fictive_temperature.png")


# ============================================================
# Error metrics — FEM vs Paper
# ============================================================

def compute_error_metrics(stress, nx):
    """
    Compute quantitative error metrics between FEM and digitised paper data.

    Metrics computed for both Surface and Mid-plane curves:
      - Residual error     : |FEM_final - Paper_final| / |Paper_final| × 100%
      - MAE                : Mean Absolute Error over common time range
      - RMSE               : Root Mean Square Error over common time range
      - Max absolute error : worst-case pointwise error
      - Peak stress error  : error at peak stress value

    The FEM curve is interpolated onto the paper's time points for comparison.
    Results are printed to console and returned as a dict.
    """
    if PAPER_SURF is None or PAPER_MID is None:
        print("  Cannot compute errors — paper CSV files not found.")
        return {}

    from scipy.interpolate import interp1d

    s, m  = surf_mid(nx)
    sig_s = stress[s] / 1e6   # MPa
    sig_m = stress[m] / 1e6

    # Build FEM interpolators (log time for accuracy on log-spaced paper data)
    t_full   = np.concatenate([[1e-4], t_array])
    fs_full  = np.concatenate([[0.0],  sig_s])
    fm_full  = np.concatenate([[0.0],  sig_m])

    interp_s = interp1d(t_full, fs_full, kind="linear",
                        bounds_error=False, fill_value="extrapolate")
    interp_m = interp1d(t_full, fm_full, kind="linear",
                        bounds_error=False, fill_value="extrapolate")

    # Evaluate FEM at paper time points
    t_ps  = PAPER_SURF[:, 0];  sigma_ps  = PAPER_SURF[:, 1]
    t_pm  = PAPER_MID[:, 0];   sigma_pm  = PAPER_MID[:, 1]

    fem_at_ps = interp_s(t_ps)
    fem_at_pm = interp_m(t_pm)

    err_s = fem_at_ps - sigma_ps   # signed error [MPa]
    err_m = fem_at_pm - sigma_pm

    # ── Residual (final value) error ──────────────────────────────────────
    n_res   = max(1, Nt // 20)
    res_fem_s = float(np.mean(sig_s[-n_res:]))
    res_fem_m = float(np.mean(sig_m[-n_res:]))
    res_pap_s = float(PAPER_SURF[-3:, 1].mean())
    res_pap_m = float(PAPER_MID[-3:, 1].mean())
    res_err_s = (res_fem_s - res_pap_s) / abs(res_pap_s) * 100
    res_err_m = (res_fem_m - res_pap_m) / abs(res_pap_m) * 100

    # ── Peak stress error ─────────────────────────────────────────────────
    peak_pap_s = sigma_ps[np.argmax(np.abs(sigma_ps))]
    peak_pap_m = sigma_pm[np.argmax(np.abs(sigma_pm))]
    peak_fem_s = fem_at_ps[np.argmax(np.abs(sigma_ps))]
    peak_fem_m = fem_at_pm[np.argmax(np.abs(sigma_pm))]
    peak_err_s = (peak_fem_s - peak_pap_s) / abs(peak_pap_s) * 100
    peak_err_m = (peak_fem_m - peak_pap_m) / abs(peak_pap_m) * 100

    # ── MAE, RMSE, Max — absolute [MPa] and normalised [%] ───────────────
    # Normalised by the paper's stress range (max-min) for each curve
    range_s = float(np.ptp(sigma_ps)) if np.ptp(sigma_ps) != 0 else 1.0
    range_m = float(np.ptp(sigma_pm)) if np.ptp(sigma_pm) != 0 else 1.0

    mae_s      = float(np.mean(np.abs(err_s)))
    mae_m      = float(np.mean(np.abs(err_m)))
    mae_s_pct  = mae_s  / range_s * 100
    mae_m_pct  = mae_m  / range_m * 100

    rmse_s     = float(np.sqrt(np.mean(err_s**2)))
    rmse_m     = float(np.sqrt(np.mean(err_m**2)))
    rmse_s_pct = rmse_s / range_s * 100
    rmse_m_pct = rmse_m / range_m * 100

    max_s      = float(np.max(np.abs(err_s)))
    max_m      = float(np.max(np.abs(err_m)))
    max_s_pct  = max_s  / range_s * 100
    max_m_pct  = max_m  / range_m * 100

    metrics = {
        "surface": {
            "residual_fem_MPa":   res_fem_s,
            "residual_paper_MPa": res_pap_s,
            "residual_error_%":   res_err_s,
            "peak_fem_MPa":       peak_fem_s,
            "peak_paper_MPa":     peak_pap_s,
            "peak_error_%":       peak_err_s,
            "MAE_MPa":            mae_s,
            "MAE_%":              mae_s_pct,
            "RMSE_MPa":           rmse_s,
            "RMSE_%":             rmse_s_pct,
            "max_abs_error_MPa":  max_s,
            "max_abs_error_%":    max_s_pct,
        },
        "midplane": {
            "residual_fem_MPa":   res_fem_m,
            "residual_paper_MPa": res_pap_m,
            "residual_error_%":   res_err_m,
            "peak_fem_MPa":       peak_fem_m,
            "peak_paper_MPa":     peak_pap_m,
            "peak_error_%":       peak_err_m,
            "MAE_MPa":            mae_m,
            "MAE_%":              mae_m_pct,
            "RMSE_MPa":           rmse_m,
            "RMSE_%":             rmse_m_pct,
            "max_abs_error_MPa":  max_m,
            "max_abs_error_%":    max_m_pct,
        },
    }

    # ── Print report ──────────────────────────────────────────────────────
    sep = "-" * 60
    print(f"\n{sep}")
    print("  FEM vs Aronen & Karvinen (2018) — Error Report")
    print(sep)
    print(f"  {'Metric':<30} {'Surface':>12} {'Mid-plane':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12}")
    rows = [
        ("Residual FEM [MPa]",      f"{res_fem_s:>12.1f}", f"{res_fem_m:>12.1f}"),
        ("Residual Paper [MPa]",    f"{res_pap_s:>12.1f}", f"{res_pap_m:>12.1f}"),
        ("Residual error [%]",      f"{res_err_s:>12.1f}", f"{res_err_m:>12.1f}"),
        ("Peak FEM [MPa]",          f"{peak_fem_s:>12.1f}", f"{peak_fem_m:>12.1f}"),
        ("Peak Paper [MPa]",        f"{peak_pap_s:>12.1f}", f"{peak_pap_m:>12.1f}"),
        ("Peak error [%]",          f"{peak_err_s:>12.1f}", f"{peak_err_m:>12.1f}"),
        ("MAE [MPa]",               f"{mae_s:>12.2f}", f"{mae_m:>12.2f}"),
        ("MAE [%]",                 f"{mae_s_pct:>12.1f}", f"{mae_m_pct:>12.1f}"),
        ("RMSE [MPa]",              f"{rmse_s:>12.2f}", f"{rmse_m:>12.2f}"),
        ("RMSE [%]",                f"{rmse_s_pct:>12.1f}", f"{rmse_m_pct:>12.1f}"),
        ("Max abs error [MPa]",     f"{max_s:>12.2f}", f"{max_m:>12.2f}"),
        ("Max abs error [%]",       f"{max_s_pct:>12.1f}", f"{max_m_pct:>12.1f}"),
    ]
    for label, vs, vm in rows:
        print(f"  {label:<30} {vs} {vm}")
    print(sep)

    # Save report to file
    report_path = os.path.join(PLOT_DIR, "error_report.txt")
    with open(report_path, "w") as f:
        f.write("FEM vs Aronen & Karvinen (2018) — Error Report\n")
        f.write(f"b={THICKNESS_MM}mm  T0=650C  h=450 W/m2K\n")
        f.write(f"Nt={Nt}  dt={dt}s  N_NODES={N_NODES}\n\n")
        f.write(f"{'Metric':<30} {'Surface':>12} {'Mid-plane':>12}\n")
        f.write(f"{'─'*56}\n")
        for key in ["residual_fem_MPa","residual_paper_MPa","residual_error_%",
                    "peak_fem_MPa","peak_paper_MPa","peak_error_%",
                    "MAE_MPa","RMSE_MPa","max_abs_error_MPa"]:
            f.write(f"  {key:<28} {metrics['surface'][key]:>12.2f}"
                    f" {metrics['midplane'][key]:>12.2f}\n")
    print(f"  Report saved: {report_path}")

    return metrics


# ============================================================
# Fig. 10 — FEM vs Paper (Aronen 2018) temperature comparison
# ============================================================

def plot_fig10_vs_paper(temperature, nx, x_T=None):
    """
    Overlay FEM surface and mid-plane temperature against
    digitised Aronen & Karvinen (2018) Fig. 4 data.
    FEM: solid/dashed black lines
    Paper: open circle/square markers
    """
    if PAPER_T_SURF is None or PAPER_T_MID is None:
        print("  Skipping fig10 — temperature paper CSV files not available.")
        return

    s, m = surf_mid(nx)
    nx_T = temperature.shape[0]
    if nx_T != nx and x_T is not None:
        idx_surf = np.argmin(np.abs(x_T - (-THICKNESS_MM/2)))
        idx_mid  = np.argmin(np.abs(x_T))
    else:
        idx_surf, idx_mid = s, m

    T_s = temperature[idx_surf] - 273.15
    T_m = temperature[idx_mid]  - 273.15

    # Prepend IC — use t slightly before first timestep to avoid duplicates
    t_ic     = t_array[0] * 0.1
    t_plot   = np.concatenate([[t_ic], t_array])
    T_s_plot = np.concatenate([[650.0], T_s])
    T_m_plot = np.concatenate([[650.0], T_m])

    # Deduplicate time axis for interpolation
    _, uniq  = np.unique(t_plot, return_index=True)
    t_uniq   = t_plot[uniq]
    Ts_uniq  = T_s_plot[uniq]
    Tm_uniq  = T_m_plot[uniq]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _hgrid(ax)

    ax.plot(t_plot, T_s_plot, color="black", lw=LW_MAIN, ls="-",
            label="FEM --- Surface")
    ax.plot(t_plot, T_m_plot, color="black", lw=LW_MAIN, ls="--",
            label="FEM --- Mid-plane")
    ax.plot(PAPER_T_SURF[:, 0], PAPER_T_SURF[:, 1],
            color="black", lw=0, marker="o", ms=5,
            mfc="white", mew=1.3, label="Aronen 2018 --- Surface")
    ax.plot(PAPER_T_MID[:, 0],  PAPER_T_MID[:, 1],
            color="black", lw=0, marker="s", ms=5,
            mfc="white", mew=1.3, label="Aronen 2018 --- Mid-plane")

    ax.set_ylabel("Temperature $T$ [degC]", fontsize=12)
    ax.set_ylim(0, 700)
    ax.set_yticks(range(0, 701, 100))
    _log_taxis(ax)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    from scipy.interpolate import interp1d
    interp_s = interp1d(t_uniq, Ts_uniq, bounds_error=False,
                        fill_value="extrapolate")
    interp_m = interp1d(t_uniq, Tm_uniq, bounds_error=False,
                        fill_value="extrapolate")

    fem_at_ts = interp_s(PAPER_T_SURF[:, 0])
    fem_at_tm = interp_m(PAPER_T_MID[:, 0])
    err_s = fem_at_ts - PAPER_T_SURF[:, 1]
    err_m = fem_at_tm - PAPER_T_MID[:, 1]

    # Normalise by paper temperature range (max-min) for % metrics
    range_ts = float(np.ptp(PAPER_T_SURF[:, 1])) or 1.0
    range_tm = float(np.ptp(PAPER_T_MID[:, 1]))  or 1.0

    mae_s      = float(np.mean(np.abs(err_s)))
    mae_m      = float(np.mean(np.abs(err_m)))
    mae_s_pct  = mae_s / range_ts * 100
    mae_m_pct  = mae_m / range_tm * 100

    rmse_s     = float(np.sqrt(np.mean(err_s**2)))
    rmse_m     = float(np.sqrt(np.mean(err_m**2)))
    rmse_s_pct = rmse_s / range_ts * 100
    rmse_m_pct = rmse_m / range_tm * 100

    max_s      = float(np.max(np.abs(err_s)))
    max_m      = float(np.max(np.abs(err_m)))
    max_s_pct  = max_s / range_ts * 100
    max_m_pct  = max_m / range_tm * 100

    ax.text(0.97, 0.08,
            (f"MAE:  surf={mae_s:.1f}\u00b0C ({mae_s_pct:.1f}%)  mid={mae_m:.1f}\u00b0C ({mae_m_pct:.1f}%)\n"
             f"RMSE: surf={rmse_s:.1f}\u00b0C ({rmse_s_pct:.1f}%)  mid={rmse_m:.1f}\u00b0C ({rmse_m_pct:.1f}%)"),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      alpha=0.9, ec="lightgray"))

    ax.set_title(
        "FEM vs Aronen & Karvinen (2018) — Temperature\n"
        rf"$b$ = {THICKNESS_MM} mm,  $T_0$ = 650 °C,  $h$ = 450 W m$^{{-2}}$ K$^{{-1}}$",
        fontsize=11)

    fig.tight_layout()
    _save(fig, "fig10_temperature_vs_paper.png")

    # Print error report
    sep = "-" * 60
    print(f"\n{sep}")
    print("  Temperature error — FEM vs Aronen 2018")
    print(sep)
    print(f"  {'Metric':<30} {'Surface':>12} {'Mid-plane':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    rows_T = [
        ("MAE [degC]",          f"{mae_s:>12.2f}",     f"{mae_m:>12.2f}"),
        ("MAE [%]",             f"{mae_s_pct:>12.1f}", f"{mae_m_pct:>12.1f}"),
        ("RMSE [degC]",         f"{rmse_s:>12.2f}",    f"{rmse_m:>12.2f}"),
        ("RMSE [%]",            f"{rmse_s_pct:>12.1f}",f"{rmse_m_pct:>12.1f}"),
        ("Max abs error [degC]",f"{max_s:>12.2f}",     f"{max_m:>12.2f}"),
        ("Max abs error [%]",   f"{max_s_pct:>12.1f}", f"{max_m_pct:>12.1f}"),
    ]
    for label, vs, vm in rows_T:
        print(f"  {label:<30} {vs} {vm}")
    print(sep)


# ============================================================
# Fig. 9 — FEM vs Paper (Aronen 2018) stress comparison
# ============================================================

def plot_fig9_vs_paper(stress, nx):
    """
    Overlay FEM stress results against digitised Aronen 2018 paper data.
    FEM: solid/dashed black lines
    Paper: open circle/square markers (no connecting line)
    """
    if PAPER_SURF is None or PAPER_MID is None:
        print("  Skipping fig9 — paper CSV files not available.")
        return

    s, m = surf_mid(nx)
    sig_s = stress[s] / 1e6
    sig_m = stress[m] / 1e6

    # Prepend IC at t=1e-3
    t_plot   = np.concatenate([[1e-3], t_array])
    sig_s_pl = np.concatenate([[0.0],  sig_s])
    sig_m_pl = np.concatenate([[0.0],  sig_m])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _hgrid(ax)

    # FEM lines
    ax.plot(t_plot, sig_s_pl, color="black", lw=LW_MAIN, ls="-",
            label="FEM — Surface")
    ax.plot(t_plot, sig_m_pl, color="black", lw=LW_MAIN, ls="--",
            label="FEM — Mid-plane")

    # Paper markers — open symbols, no line
    ax.plot(PAPER_SURF[:, 0], PAPER_SURF[:, 1],
            color="black", lw=0, marker="o", ms=5,
            mfc="white", mew=1.3, label="Aronen 2018 — Surface")
    ax.plot(PAPER_MID[:, 0],  PAPER_MID[:, 1],
            color="black", lw=0, marker="s", ms=5,
            mfc="white", mew=1.3, label="Aronen 2018 — Mid-plane")

    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel(r"Stress $\sigma$ [MPa]", fontsize=12)
    ax.set_ylim(-140, 80)
    ax.set_yticks(range(-140, 81, 20))
    _log_taxis(ax)
    ax.legend(loc="lower left", fontsize=9, ncol=2)

    # Residual annotation
    n_res = max(1, Nt // 20)
    s_res = float(np.mean(sig_s[-n_res:]))
    m_res = float(np.mean(sig_m[-n_res:]))
    p_s   = float(PAPER_SURF[-3:, 1].mean())   # last 3 paper points
    p_m   = float(PAPER_MID[-3:, 1].mean())
    err_s = abs((s_res - p_s) / p_s) * 100 if p_s != 0 else float("nan")
    err_m = abs((m_res - p_m) / p_m) * 100 if p_m != 0 else float("nan")
    ax.text(0.97, 0.08,
            (f"FEM:   surface={s_res:.0f} MPa   mid-plane=+{m_res:.0f} MPa\n"
             f"Paper: surface={p_s:.0f} MPa   mid-plane=+{p_m:.0f} MPa\n"
             f"Error: surface={err_s:.1f}%      mid-plane={err_m:.1f}%"),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      alpha=0.9, ec="lightgray"))

    ax.set_title(
        "FEM vs Aronen & Karvinen (2018)\n"
        rf"$b$ = {THICKNESS_MM} mm,  $T_0$ = 650 °C,  $h$ = 450 W m$^{{-2}}$ K$^{{-1}}$",
        fontsize=11)

    fig.tight_layout()
    _save(fig, "fig9_fem_vs_paper.png")


# ============================================================
# Fig. 11 — FEM vs Paper temperature profiles through thickness
# ============================================================

def plot_fig11_vs_paper(temperature, z):
    """
    Overlay FEM temperature profiles vs digitised Aronen 2018 Fig. 6 data.
    FEM: black lines with different linestyles per time
    Paper: open markers per time snapshot
    """
    if not PAPER_T_PROF or all(v is None for v in PAPER_T_PROF.values()):
        print("  Skipping fig11 — temperature profile CSVs not available.")
        return

    # Time slots, matching labels, linestyles, and marker shapes
    slots = [
        ("t0",   0.0,   "-",              "o", "0 s"),
        ("t5",   5.0,   "--",             "s", "5 s"),
        ("t10",  10.0,  "-.",             "^", "10 s"),
        ("t20",  20.0,  ":",              "D", "20 s"),
        ("t100", 100.0, (0, (5, 1)),      "v", "100 s"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.yaxis.grid(True, color="lightgray", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    mae_all = []

    for key, ts, ls, mk, lbl in slots:
        # FEM line
        if ts <= 0:
            y_fem = np.full_like(z, 650.0)
        else:
            idx = max(0, min(int(round(ts / dt)) - 1, Nt - 1))
            y_fem = temperature[:, idx] - 273.15

        ax.plot(z, y_fem, color="black", lw=1.8, ls=ls, label=f"FEM {lbl}")

        # Paper markers
        prof = PAPER_T_PROF.get(key)
        if prof is not None:
            ax.plot(prof[:, 0], prof[:, 1],
                    color="black", lw=0, marker=mk, ms=5,
                    mfc="white", mew=1.2, label=f"Paper {lbl}")

            # Compute MAE at this time
            from scipy.interpolate import interp1d
            f_fem = interp1d(z, y_fem, bounds_error=False,
                             fill_value="extrapolate")
            fem_at_paper = f_fem(prof[:, 0])
            mae = float(np.mean(np.abs(fem_at_paper - prof[:, 1])))
            mae_pct = mae / (float(np.ptp(prof[:, 1])) or 1.0) * 100
            mae_all.append((lbl, mae, mae_pct))

    ax.set_xlabel("$z$ [mm]", fontsize=12)
    ax.set_ylabel("Temperature $T$ [°C]", fontsize=12)
    ax.set_xlim(-THICKNESS_MM/2, THICKNESS_MM/2)
    ax.set_ylim(0, 700)
    ax.set_yticks(range(0, 701, 100))

    # Two-column legend: FEM lines left, Paper markers right
    handles, labels = ax.get_legend_handles_labels()
    n = len(slots)
    ax.legend(handles[:n] + handles[n:], labels[:n] + labels[n:],
              fontsize=8, ncol=2,
              title="FEM (lines)  vs  Paper (markers)", title_fontsize=8,
              loc="upper right")

    ax.set_title(
        "FEM vs Aronen & Karvinen (2018) — Temperature profiles\n"
        rf"$b$ = {THICKNESS_MM} mm,  $T_0$ = 650 \u00b0C,  $h$ = 450 W m$^{{-2}}$ K$^{{-1}}$",
        fontsize=11)
    fig.tight_layout()
    _save(fig, "fig11_temp_profiles_vs_paper.png")

    # Print error table
    sep = "-" * 50
    print(f"\n{sep}")
    print("  Temperature profile MAE — FEM vs Aronen 2018")
    print(sep)
    print(f"  {'Time':<12} {'MAE [degC]':>12} {'MAE [%]':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*10}")
    for lbl, mae, mae_pct in mae_all:
        print(f"  {lbl:<12} {mae:>12.2f} {mae_pct:>10.1f}")
    print(sep)


# ============================================================
# Fig. 12 — FEM vs Paper stress profiles through thickness
# ============================================================

def plot_fig12_vs_paper(stress, z):
    """
    Overlay FEM stress profiles vs digitised Aronen 2018 Fig. 7.
    FEM: black lines, different linestyles per time
    Paper: open markers per time snapshot
    """
    if not PAPER_S_PROF or all(v is None for v in PAPER_S_PROF.values()):
        print("  Skipping fig12 — stress profile CSVs not available.")
        return

    slots = [
        ("t0",   0.0,   "-",           "o", "0 s"),
        ("t5",   5.0,   "--",          "s", "5 s"),
        ("t10",  10.0,  "-.",          "^", "10 s"),
        ("t20",  20.0,  ":",           "D", "20 s"),
        ("t100", 100.0, (0, (5, 1)),   "v", "100 s"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.yaxis.grid(True, color="lightgray", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    mae_all = []
    from scipy.interpolate import interp1d

    for key, ts, ls, mk, lbl in slots:
        # FEM
        if ts <= 0:
            y_fem = np.zeros_like(z)
        else:
            idx   = max(0, min(int(round(ts / dt)) - 1, Nt - 1))
            y_fem = stress[:, idx] / 1e6

        ax.plot(z, y_fem, color="black", lw=1.8, ls=ls, label=f"FEM {lbl}")

        # Paper markers
        prof = PAPER_S_PROF.get(key)
        if prof is not None:
            ax.plot(prof[:, 0], prof[:, 1],
                    color="black", lw=0, marker=mk, ms=5,
                    mfc="white", mew=1.2, label=f"Paper {lbl}")

            # MAE
            f_fem = interp1d(z, y_fem, bounds_error=False,
                             fill_value="extrapolate")
            err   = f_fem(prof[:, 0]) - prof[:, 1]
            mae   = float(np.mean(np.abs(err)))
            rng   = float(np.ptp(prof[:, 1])) or 1.0
            mae_all.append((lbl, mae, mae / rng * 100))

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("$z$ [mm]", fontsize=12)
    ax.set_ylabel(r"Stress $\sigma$ [MPa]", fontsize=12)
    ax.set_xlim(-THICKNESS_MM/2, THICKNESS_MM/2)

    handles, labels = ax.get_legend_handles_labels()
    n = len(slots)
    ax.legend(handles[:n] + handles[n:], labels[:n] + labels[n:],
              fontsize=8, ncol=2,
              title="FEM (lines)  vs  Paper (markers)", title_fontsize=8,
              loc="lower right")

    ax.set_title(
        "FEM vs Aronen & Karvinen (2018) — Stress profiles\n"
        rf"$b$ = {THICKNESS_MM} mm,  $T_0$ = 650 \u00b0C,  $h$ = 450 W m$^{{-2}}$ K$^{{-1}}$",
        fontsize=11)
    fig.tight_layout()
    _save(fig, "fig12_stress_profiles_vs_paper.png")

    # Print error table
    sep = "-" * 50
    print(f"\n{sep}")
    print("  Stress profile MAE — FEM vs Aronen 2018")
    print(sep)
    print(f"  {'Time':<12} {'MAE [MPa]':>12} {'MAE [%]':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*10}")
    for lbl, mae, mae_pct in mae_all:
        print(f"  {lbl:<12} {mae:>12.2f} {mae_pct:>10.1f}")
    print(sep)


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  Aronen & Karvinen (2018) — 1D validation")
    print(f"  b={THICKNESS_MM}mm | T₀=650°C | h=450 W/m²K | T∞=20°C")
    print("=" * 60 + "\n")

    temperature, stress, fictive, nx, z, x_T = load_fields()
    s, m = surf_mid(nx)
    print(f"  Surface node={s},  Midplane node={m}\n")

    plot_fig4(temperature, nx, x_T)
    plot_fig5(stress, nx)
    plot_fig6(temperature, z, x_T)
    plot_fig7(stress, z)
    plot_fig8(temperature, fictive, nx, x_T)
    plot_fig10_vs_paper(temperature, nx, x_T)
    plot_fig11_vs_paper(temperature, z)
    plot_fig12_vs_paper(stress, z)
    plot_fig9_vs_paper(stress, nx)
    compute_error_metrics(stress, nx)

    print(f"\nAll plots saved to: {os.path.abspath(PLOT_DIR)}")
    n_res = max(1, Nt//20)
    print(f"\nResidual stresses (last {n_res} steps averaged):")
    print(f"  Surface  = {stress[s,-n_res:].mean()/1e6:.1f} MPa  (paper: ~-120 MPa)")
    print(f"  Midplane = {stress[m,-n_res:].mean()/1e6:+.1f} MPa  (paper: ~+55 MPa)")


if __name__ == "__main__":
    main()