# plot_case_annealing_multizone.py
#
# Plots one matched annealing-Lehr case (temperature / stress fields, time
# series, and vs-Lehr-distance views).  All run settings (t_start, t_end, dt,
# velocity, node count, zone windows) are read from results/<split>/meta.json,
# which main.py writes, so this script can never drift from the run that
# produced the data.  Missing meta.json falls back to the DEFAULTS below.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# User settings
# ============================================================
# Which split(s) to search: "train", "test_unseen", or "auto"
split = "auto"
# How to choose which case to plot:
#   "index"  -> plot CASE_INDEX directly (always works; params read from the CSV)
#   "params" -> find the case matching target_params (must be one you actually sampled)
SELECT_BY  = "index"
CASE_INDEX = 0

PLOT_DIR = "plots_case_checks"
os.makedirs(PLOT_DIR, exist_ok=True)

# Fallback DEFAULTS (used only if a meta.json is missing).  These are overwritten
# per split by load_meta() so the plots match main.py automatically.
DEFAULTS = {
    "t_start": 0.0,
    "t_end": 328.5,          # full Lehr (was 200.0)
    "dt": 0.1,
    "n_nodes": 29,
    "velocity": 0.16417,
    "zone_time_windows": {
        "A1": [0.0,    54.7],
        "A2": [54.7,   109.6],
        "B1": [109.6,  182.6],
        "B2": [182.6,  255.57],
        "C1": [255.57, 328.5],
    },
}

# These module-level values are set from meta.json in main() before plotting.
t_start           = DEFAULTS["t_start"]
t_end             = DEFAULTS["t_end"]
dt                = DEFAULTS["dt"]
velocity_m_per_s  = DEFAULTS["velocity"]
n_nodes           = DEFAULTS["n_nodes"]
ZONE_TIME_WINDOWS = {z: tuple(v) for z, v in DEFAULTS["zone_time_windows"].items()}

# ------------------------------------------------------------
# Target multi-zone parameter set to plot
# ------------------------------------------------------------
target_params = {
    "htc_A1": 420.0,
    "htc_A2": 435.0,
    "htc_B1": 450.0,
    "htc_B2": 435.0,
    "htc_C1": 435.0,

    "T_amb_A1": 873.0,
    "T_amb_A2": 846.0,
    "T_amb_B1": 812.0,
    "T_amb_B2": 757.0,
    "T_amb_C1": 748.0,

    "epsilon": 0.775,
    "sigma": 1.670e-8,
}


# ============================================================
# Meta / IO helpers
# ============================================================
def load_params_csv(which: str):
    if which == "train":
        return "parameter_combinations_train.csv", os.path.join("results", "train")
    elif which == "test_unseen":
        return "parameter_combinations_test_unseen.csv", os.path.join("results", "test_unseen")
    else:
        raise ValueError('split must be "train" or "test_unseen"')


def load_meta(data_dir: str):
    """Read run settings from meta.json (falling back to DEFAULTS) and push them
    into the module-level globals used by the plot functions."""
    global t_start, t_end, dt, velocity_m_per_s, n_nodes, ZONE_TIME_WINDOWS

    meta_path = os.path.join(data_dir, "meta.json")
    meta = dict(DEFAULTS)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta.update(json.load(f))
        print(f"[meta] loaded {meta_path}")
    else:
        print(f"[meta] {meta_path} not found -> using DEFAULTS")

    t_start          = float(meta["t_start"])
    t_end            = float(meta["t_end"])
    dt               = float(meta["dt"])
    velocity_m_per_s = float(meta["velocity"])
    n_nodes          = int(meta["n_nodes"])
    ZONE_TIME_WINDOWS = {z: tuple(v) for z, v in meta["zone_time_windows"].items()}
    return meta


def find_match(df: pd.DataFrame, target: dict) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col, val in target.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in parameter CSV")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            # relative tolerance handles both htc (~440) and sigma (~1e-8)
            mask &= np.isclose(df[col].astype(float), float(val), rtol=1e-6, atol=1e-12)
        else:
            mask &= (df[col] == val)
    return df[mask]


def load_fields(data_dir: str, case_idx: int):
    """Load temperature/stress histories.  The number of time steps is DERIVED
    from the data using the (reliable) node count, so an off-by-one in the saved
    history length (initial state included or not) is handled automatically."""
    temp_path   = os.path.join(data_dir, f"temperature_all_case{case_idx}.txt")
    stress_path = os.path.join(data_dir, f"stress_all_case{case_idx}.txt")
    if not os.path.exists(temp_path):
        raise FileNotFoundError(f"Missing temperature file: {temp_path}")
    if not os.path.exists(stress_path):
        raise FileNotFoundError(f"Missing stress file: {stress_path}")

    temp_flat   = np.loadtxt(temp_path)
    stress_flat = np.loadtxt(stress_path)

    nx = n_nodes
    if temp_flat.size % nx != 0:
        raise ValueError(
            f"Temperature length {temp_flat.size} not divisible by n_nodes={nx}. "
            f"Check meta.json 'n_nodes' vs the saved field."
        )
    nt = temp_flat.size // nx

    # (nt, nx) -> transpose to (space, time) for imshow/time-series
    temperature = temp_flat.reshape(nt, nx).T
    stress      = stress_flat.reshape(nt, nx).T
    return temperature, stress, nx, nt


def make_tag(which: str, case_idx: int, p: dict):
    sig_str = f"{p['sigma']:.3e}".replace("+", "")
    return (
        f"{which}_case{case_idx}"
        f"_A1{p['htc_A1']}_A2{p['htc_A2']}_B1{p['htc_B1']}_B2{p['htc_B2']}_C1{p['htc_C1']}"
        f"_eps{p['epsilon']}_sig{sig_str}"
    )


# ============================================================
# Zone annotations
# ============================================================
def add_zone_lines_time(ax):
    for zone, (z0, z1) in ZONE_TIME_WINDOWS.items():
        if t_start <= z1 <= t_end:
            ax.axvline(z1, linestyle="--", linewidth=1, color="gray")
        if z0 < t_end:
            center = min(z0 + 0.5 * (z1 - z0), t_end - 1e-6)
            ax.text(center, 1.01, zone, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10)


def add_zone_lines_distance(ax):
    x_max = velocity_m_per_s * t_end
    visible = []
    for zone, (z0, z1) in ZONE_TIME_WINDOWS.items():
        x0 = velocity_m_per_s * z0
        x1 = velocity_m_per_s * z1
        if x0 < x_max:
            visible.append((zone, x0, min(x1, x_max)))
    for zone, x0, x1 in visible:
        if x1 <= x_max:
            ax.axvline(x1, linestyle="--", linewidth=1, color="gray")
    for zone, x0, x1 in visible:
        ax.text(0.5 * (x0 + x1), 1.01, zone, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10)


# ============================================================
# Plot functions
# ============================================================
def _time_axis(nt):
    return np.linspace(t_start, t_end, nt)


def plot_temperature_map(temperature, which, case_idx, tag, nx_now):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    im = ax.imshow(temperature, aspect="auto", origin="lower", cmap="RdYlBu_r",
                   extent=[t_start, t_end, 0, nx_now - 1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Space index")
    ax.set_title(f"FEM Temperature Field ({which}, case {case_idx})", pad=20)
    fig.colorbar(im, ax=ax).set_label("Temperature (K)")
    add_zone_lines_time(ax)
    ax.set_xlim(t_start, t_end)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"temperature_map_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight"); plt.show()
    print(f"[OK] Saved: {out}")


def plot_stress_map(stress, which, case_idx, tag, nx_now):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    im = ax.imshow(stress, aspect="auto", origin="lower", cmap="PuOr",
                   extent=[t_start, t_end, 0, nx_now - 1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Space index")
    ax.set_title(f"FEM Stress Field ({which}, case {case_idx})", pad=20)
    fig.colorbar(im, ax=ax).set_label("Stress (Pa)")
    add_zone_lines_time(ax)
    ax.set_xlim(t_start, t_end)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"stress_map_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight"); plt.show()
    print(f"[OK] Saved: {out}")


def plot_temperature_timeseries(temperature, which, case_idx, tag):
    nx_now = temperature.shape[0]
    nt = temperature.shape[1]
    t_ = _time_axis(nt)
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(t_, temperature[0],            label=f"Surface node (0)")
    plt.plot(t_, temperature[nx_now // 2],  label=f"Mid-plane ({nx_now // 2})")
    plt.plot(t_, temperature[nx_now - 1],   label=f"Last node ({nx_now - 1})")
    plt.xlabel("Time (s)"); plt.ylabel("Temperature (K)")
    plt.title(f"Temperature evolution ({which}, case {case_idx})", pad=20)
    plt.grid(True); plt.legend(); plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"temperature_timeseries_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight"); plt.show()
    print(f"[OK] Saved: {out}")


def plot_stress_timeseries(stress, which, case_idx, tag):
    nx_now = stress.shape[0]
    nt = stress.shape[1]
    t_ = _time_axis(nt)
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(t_, stress[0],            label=f"Surface node (0)")
    plt.plot(t_, stress[nx_now // 2],  label=f"Mid-plane ({nx_now // 2})")
    plt.plot(t_, stress[nx_now - 1],   label=f"Last node ({nx_now - 1})")
    plt.xlabel("Time (s)"); plt.ylabel("Stress (Pa)")
    plt.title(f"Stress evolution ({which}, case {case_idx})", pad=20)
    plt.grid(True); plt.legend(); plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"stress_timeseries_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight"); plt.show()
    print(f"[OK] Saved: {out}")


def plot_surface_temperature_vs_distance(temperature, which, case_idx, tag):
    nt = temperature.shape[1]
    x_ = velocity_m_per_s * _time_axis(nt)
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(x_, temperature[0], label="Surface temperature")
    plt.xlabel("Lehr distance x (m)"); plt.ylabel("Temperature (K)")
    plt.title(f"Surface temperature vs distance ({which}, case {case_idx})", pad=20)
    plt.grid(True); plt.legend()
    add_zone_lines_distance(plt.gca())
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"surface_temperature_vs_distance_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight"); plt.show()
    print(f"[OK] Saved: {out}")


def plot_surface_stress_vs_distance(stress, which, case_idx, tag):
    nt = stress.shape[1]
    x_ = velocity_m_per_s * _time_axis(nt)
    mid = stress.shape[0] // 2
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(x_, stress[0],   label="Surface stress")
    plt.plot(x_, stress[mid], label="Mid-plane stress")
    plt.xlabel("Lehr distance x (m)"); plt.ylabel("Stress (Pa)")
    plt.title(f"Stress vs distance ({which}, case {case_idx})", pad=20)
    plt.grid(True); plt.legend()
    add_zone_lines_distance(plt.gca())
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"stress_vs_distance_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight"); plt.show()
    print(f"[OK] Saved: {out}")


def plot_temperature_map_vs_distance(temperature, which, case_idx, tag, nx_now):
    nt = temperature.shape[1]
    x_ = velocity_m_per_s * _time_axis(nt)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    im = ax.imshow(temperature, aspect="auto", origin="lower", cmap="RdYlBu_r",
                   extent=[x_[0], x_[-1], 0, nx_now - 1])
    ax.set_xlabel("Lehr distance x (m)")
    ax.set_ylabel("Space index / thickness node")
    ax.set_title(f"FEM Temperature Field vs Lehr Distance ({which}, case {case_idx})", pad=20)
    fig.colorbar(im, ax=ax).set_label("Temperature (K)")
    add_zone_lines_distance(ax)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"temperature_map_vs_distance_{tag}.png")
    plt.savefig(out, dpi=800, bbox_inches="tight"); plt.show()
    print(f"[OK] Saved: {out}")


def plot_stress_map_vs_distance(stress, which, case_idx, tag, nx_now):
    nt = stress.shape[1]
    x_ = velocity_m_per_s * _time_axis(nt)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    # stress shown in MPa
    im = ax.imshow(stress / 1.0e6, aspect="auto", origin="lower", cmap="PuOr",
                   extent=[x_[0], x_[-1], 0, nx_now - 1])
    ax.set_xlabel("Lehr distance x (m)")
    ax.set_ylabel("Space index / thickness node")
    ax.set_title(f"FEM Stress Field vs Lehr Distance ({which}, case {case_idx})", pad=20)
    fig.colorbar(im, ax=ax).set_label("Stress (MPa)")
    add_zone_lines_distance(ax)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"stress_map_vs_distance_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight"); plt.show()
    print(f"[OK] Saved: {out}")


# ============================================================
# Main
# ============================================================
def main():
    candidates = ["train", "test_unseen"] if split == "auto" else [split]
    found = False

    for which in candidates:
        param_csv, data_dir = load_params_csv(which)
        if not os.path.exists(param_csv):
            print(f"[skip] CSV not found: {param_csv}")
            continue

        load_meta(data_dir)   # sets t_start/t_end/dt/velocity/n_nodes/zones from the run

        df = pd.read_csv(param_csv)

        if SELECT_BY == "index":
            if CASE_INDEX >= len(df):
                print(f"[skip] case {CASE_INDEX} out of range in {which} ({len(df)} cases)")
                continue
            case_idx = CASE_INDEX
            case_params = df.iloc[case_idx].to_dict()      # guaranteed to exist
        else:  # "params"
            match = find_match(df, target_params)
            if match.empty:
                print(f"[skip] No match in {param_csv}")
                continue
            case_idx = int(match.index[0])
            case_params = target_params

        print(f"[OK] Plotting case index {case_idx} from {which}")

        temperature, stress, nx_now, nt = load_fields(data_dir, case_idx)
        print(f"[OK] Loaded fields: temperature={temperature.shape}, stress={stress.shape} "
              f"(nx={nx_now}, nt={nt})")

        tag = make_tag(which, case_idx, case_params)        

        plot_temperature_map(temperature, which, case_idx, tag, nx_now)
        plot_stress_map(stress, which, case_idx, tag, nx_now)
        plot_temperature_map_vs_distance(temperature, which, case_idx, tag, nx_now)
        plot_stress_map_vs_distance(stress, which, case_idx, tag, nx_now)
        plot_temperature_timeseries(temperature, which, case_idx, tag)
        plot_stress_timeseries(stress, which, case_idx, tag)
        plot_surface_temperature_vs_distance(temperature, which, case_idx, tag)
        plot_surface_stress_vs_distance(stress, which, case_idx, tag)

        found = True
        break

    if not found:
        raise ValueError(
            f"No matching parameter set found in the selected split(s). Target: {target_params}"
        )


if __name__ == "__main__":
    main()