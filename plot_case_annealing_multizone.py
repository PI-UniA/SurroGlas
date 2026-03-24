# plot_case_annealing_multizone.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# User settings
# ============================================================

# Choose where to search:
# "train", "test_unseen", or "auto"
split = "auto"

# Folder where plots will be saved
PLOT_DIR = "plots_case_checks"
os.makedirs(PLOT_DIR, exist_ok=True)

# Time settings (must match the dataset generation settings)
t_start = 0.0
t_end = 330.0
dt = 0.1
Nt = int((t_end - t_start) / dt)   

# Spatial points (must match your saved FEM field shape)
# Adjust if your saved history length indicates a different value
Nx = 20

# Velocity used to convert time -> Lehr distance
velocity_m_per_s = 0.24

# Zone boundaries in time [s]
ZONE_TIME_WINDOWS = {
    "A1": (0.0,    54.7),
    "A2": (54.7,  109.6),
    "B1": (109.6, 182.6),
    "B2": (182.6, 255.57),
    "C1": (255.57, 328.5),
}

# Zone boundaries converted to distance [m]
ZONE_DISTANCE_MARKERS = {
    zone: velocity_m_per_s * t1 for zone, (_, t1) in ZONE_TIME_WINDOWS.items()
}

# ------------------------------------------------------------
# Define target multi-zone parameter set
# ------------------------------------------------------------
target_params = {
    "htc_A1": 40.0,
    "htc_A2": 40.0,
    "htc_B1": 30.0,
    "htc_B2": 30.0,
    "htc_C1": 40.0,

    "T_amb_A1": 868.0,
    "T_amb_A2": 846.0,
    "T_amb_B1": 800.0,
    "T_amb_B2": 770.0,
    "T_amb_C1": 740.0,

    "epsilon": 0.80,
    "sigma": 3.670e-8,
}

# ============================================================
# Helper functions
# ============================================================

def load_params_csv(which: str):
    if which == "train":
        return "parameter_combinations_train.csv", os.path.join("results", "train")
    elif which == "test_unseen":
        return "parameter_combinations_test_unseen.csv", os.path.join("results", "test_unseen")
    else:
        raise ValueError('split must be "train" or "test_unseen"')


def find_match(df: pd.DataFrame, target: dict) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)

    for col, val in target.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in parameter CSV")

        if isinstance(val, float):
            atol = 1e-12
            if col == "sigma":
                atol = 1e-20
            mask &= np.isclose(df[col].astype(float), float(val), atol=atol, rtol=0.0)
        else:
            mask &= (df[col] == val)

    return df[mask]


def infer_nx_from_file(file_path: str, Nt_expected: int):
    arr = np.loadtxt(file_path)
    n_total = arr.size
    if n_total % Nt_expected != 0:
        raise ValueError(
            f"Cannot infer Nx: file length {n_total} is not divisible by Nt={Nt_expected}"
        )
    return n_total // Nt_expected


def load_fields(data_dir: str, case_idx: int):
    temp_path = os.path.join(data_dir, f"temperature_all_case{case_idx}.txt")
    stress_path = os.path.join(data_dir, f"stress_all_case{case_idx}.txt")

    if not os.path.exists(temp_path):
        raise FileNotFoundError(f"Missing temperature file: {temp_path}")
    if not os.path.exists(stress_path):
        raise FileNotFoundError(f"Missing stress file: {stress_path}")

    nx_inferred = infer_nx_from_file(temp_path, Nt)

    temperature = np.loadtxt(temp_path).reshape(Nt, nx_inferred).T
    stress = np.loadtxt(stress_path).reshape(Nt, nx_inferred).T

    return temperature, stress, nx_inferred


def make_tag(which: str, case_idx: int, p: dict):
    sig_str = f"{p['sigma']:.3e}".replace("+", "")
    return (
        f"{which}_case{case_idx}"
        f"_A1{p['htc_A1']}_A2{p['htc_A2']}_B1{p['htc_B1']}_B2{p['htc_B2']}_C1{p['htc_C1']}"
        f"_eps{p['epsilon']}_sig{sig_str}"
    )


'''def add_zone_lines_time(ax):
    for zone, (t0, t1) in ZONE_TIME_WINDOWS.items():
        ax.axvline(t1, linestyle="--", linewidth=1)
        ax.text(t0 + 0.5 * (t1 - t0), 1.01, zone,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom")
'''
def add_zone_lines_time(ax):
    for zone, (t0, t1) in ZONE_TIME_WINDOWS.items():
        if t_start <= t1 <= t_end:
            ax.axvline(t1, linestyle="--", linewidth=1)
        if t0 < t_end:
            center = min(t0 + 0.5 * (t1 - t0), t_end - 1e-6)
            ax.text(center, 1.01, zone,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom")
            
            
'''def add_zone_lines_distance(ax): for zone, x_end in ZONE_DISTANCE_MARKERS.items(): ax.axvline(x_end, linestyle="--", linewidth=1) ax.text(x_end, 1.01, zone, transform=ax.get_xaxis_transform(), ha="center", va="bottom", rotation=90)'''

def add_zone_lines_distance(ax):
    x_max = velocity_m_per_s * t_end

    # Build visible zone intervals in distance
    visible_zones = []
    for zone, (t0, t1) in ZONE_TIME_WINDOWS.items():
        x0 = velocity_m_per_s * t0
        x1 = velocity_m_per_s * t1

        if x0 < x_max:
            visible_zones.append((zone, x0, min(x1, x_max)))

    # Draw boundary lines at zone ends
    for zone, x0, x1 in visible_zones:
        if x1 <= x_max:
            ax.axvline(x1, linestyle="--", linewidth=1, color="gray")

    # Put zone names at the center of each zone
    for i, (zone, x0, x1) in enumerate(visible_zones):
        x_center = 0.5 * (x0 + x1)
        y_offset = 1.01

        ax.text(
            x_center,
            y_offset,
            zone,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10
        )


# ============================================================
# Plot functions
# ============================================================

def plot_temperature_map(temperature: np.ndarray, which: str, case_idx: int, tag: str, nx_now: int):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    im = ax.imshow(
        temperature,
        aspect="auto",
        origin="lower",
        cmap="RdYlBu_r",
        extent=[t_start, t_end, 0, nx_now - 1]
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Space index")
    ax.set_title(f"FEM Temperature Field ({which}, case {case_idx})", pad=20)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature (K)")
    add_zone_lines_time(ax)
    ax.set_xlim(t_start, t_end)   # important
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"temperature_map_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out}")


def plot_stress_map(stress: np.ndarray, which: str, case_idx: int, tag: str, nx_now: int):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    im = ax.imshow(
        stress,
        aspect="auto",
        origin="lower",
        cmap="PuOr",
        extent=[t_start, t_end, 0, nx_now - 1]
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Space index")
    ax.set_title(f"FEM Stress Field ({which}, case {case_idx})", pad=20)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Stress (Pa)")
    add_zone_lines_time(ax)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"stress_map_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out}")


def plot_temperature_timeseries(temperature: np.ndarray, which: str, case_idx: int, tag: str):
    nx_now = temperature.shape[0]
    surface_idx = 0
    mid_idx = nx_now // 2
    far_idx = nx_now - 1

    t_ = np.linspace(t_start, t_end, Nt)

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(t_, temperature[surface_idx], label=f"Surface / first node ({surface_idx})")
    plt.plot(t_, temperature[mid_idx], label=f"Mid-plane ({mid_idx})")
    plt.plot(t_, temperature[far_idx], label=f"Last node ({far_idx})")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title(f"Temperature evolution ({which}, case {case_idx})", pad=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"temperature_timeseries_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out}")


def plot_stress_timeseries(stress: np.ndarray, which: str, case_idx: int, tag: str):
    nx_now = stress.shape[0]
    surface_idx = 0
    mid_idx = nx_now // 2
    far_idx = nx_now - 1

    t_ = np.linspace(t_start, t_end, Nt)

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(t_, stress[surface_idx], label=f"Surface / first node ({surface_idx})")
    plt.plot(t_, stress[mid_idx], label=f"Mid-plane ({mid_idx})")
    plt.plot(t_, stress[far_idx], label=f"Last node ({far_idx})")
    plt.xlabel("Time (s)")
    plt.ylabel("Stress (Pa)")
    plt.title(f"Stress evolution ({which}, case {case_idx})", pad=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"stress_timeseries_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out}")


def plot_surface_temperature_vs_distance(temperature: np.ndarray, which: str, case_idx: int, tag: str):
    t_ = np.linspace(t_start, t_end, Nt)
    x_ = velocity_m_per_s * t_

    surface_idx = 0

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(x_, temperature[surface_idx], label="Surface temperature")
    plt.xlabel("Lehr distance x (m)")
    plt.ylabel("Temperature (K)")
    plt.title(f"Surface temperature vs distance ({which}, case {case_idx})", pad=20)
    plt.grid(True)
    plt.legend()
    add_zone_lines_distance(plt.gca())
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"surface_temperature_vs_distance_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out}")


def plot_surface_stress_vs_distance(stress: np.ndarray, which: str, case_idx: int, tag: str):
    t_ = np.linspace(t_start, t_end, Nt)
    x_ = velocity_m_per_s * t_

    surface_idx = 0
    mid_idx = stress.shape[0] // 2

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(x_, stress[surface_idx], label="Surface stress")
    plt.plot(x_, stress[mid_idx], label="Mid-plane stress")
    plt.xlabel("Lehr distance x (m)")
    plt.ylabel("Stress (Pa)")
    plt.title(f"Stress vs distance ({which}, case {case_idx})", pad=20)
    plt.grid(True)
    plt.legend()
    add_zone_lines_distance(plt.gca())
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"stress_vs_distance_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    candidates = ["train", "test_unseen"] if split == "auto" else [split]

    found = False

    for which in candidates:
        param_csv, data_dir = load_params_csv(which)

        if not os.path.exists(param_csv):
            print(f"⚠️ CSV not found: {param_csv}")
            continue

        df = pd.read_csv(param_csv)
        match = find_match(df, target_params)

        if match.empty:
            print(f"❌ No match in {param_csv}")
            continue

        case_idx = int(match.index[0])
        print(f"✅ Matching case index in {which}: {case_idx}")

        temperature, stress, nx_now = load_fields(data_dir, case_idx)
        print(f"✅ Loaded fields with shape: temperature={temperature.shape}, stress={stress.shape}")

        tag = make_tag(which, case_idx, target_params)

        plot_temperature_map(temperature, which, case_idx, tag, nx_now)
        plot_stress_map(stress, which, case_idx, tag, nx_now)
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
    
    
'''Next Step (Very Important for your project)

Later, when you extend FEM to full Lehr:

time_window = (0.0, 328.5)

👉 Then remove the filter:

if x_end <= x_max:'''