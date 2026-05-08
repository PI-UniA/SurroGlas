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
t_end        = 100.0
dt           = 0.001
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
    ax1.set_ylim(0, 800)
    ax1.set_yticks(range(0, 801, 100))
    _log_taxis(ax1)

    # Right axis: temperature difference ΔT
    ax2 = ax1.twinx()
    l3, = ax2.plot(t_plot, dT_plot, color="black", lw=LW_DIFF, ls=":",
                   label="Difference")
    ax2.set_ylabel(r"Temperature difference $\Delta T$ [°C]", fontsize=12)
    dT_max = float(dT.max())
    # Right axis: 0 to 140°C (matching paper Fig. 4)
    ax2.set_ylim(0, 150)
    ax2.set_yticks(range(0, 151, 20))
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

    print(f"\nAll plots saved to: {os.path.abspath(PLOT_DIR)}")
    n_res = max(1, Nt//20)
    print(f"\nResidual stresses (last {n_res} steps averaged):")
    print(f"  Surface  = {stress[s,-n_res:].mean()/1e6:.1f} MPa  (paper: ~-120 MPa)")
    print(f"  Midplane = {stress[m,-n_res:].mean()/1e6:+.1f} MPa  (paper: ~+55 MPa)")


if __name__ == "__main__":
    main()