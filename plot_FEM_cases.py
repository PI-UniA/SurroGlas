# plot_case_from_params.py
# ------------------------------------------------------------
# Plot FEM outputs (temperature/stress) for a specific parameter set
# WITHOUT re-running FEM simulations.
#
# Expected files:
#   parameter_combinations_train.csv
#   parameter_combinations_test_unseen.csv
#   results/train/temperature_all_case{i}.txt
#   results/train/stress_all_case{i}.txt
#   results/test_unseen/temperature_all_case{i}.txt
#   results/test_unseen/stress_all_case{i}.txt
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


time_window = (0.0, 50.0)
dt = 0.1

# ============================
# User settings
# ============================
# Define target input parameters
target_params = {
    "htc": 100,
    "epsilon": 0.7,
    "sigma": 1.670e-8,
    "alpha": 15
}

# Choose: "train", "test_unseen", or "auto"
split = "auto"

# Data shape
Nt, Nx = 500, 49

# Time window (only for x-axis labeling)
t_start, t_end = 0.0, 50.0

# Float tolerances for matching (important!)
TOL_EPS = 1e-12
TOL_SIG = 1e-20

# Output folder for plots
PLOT_DIR = os.path.join("results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================
# Helpers
# ============================
def load_params_csv(which: str):
    if which == "train":
        return "parameter_combinations_train.csv", os.path.join("results", "train")
    if which == "test_unseen":
        return "parameter_combinations_test_unseen.csv", os.path.join("results", "test_unseen")
    raise ValueError('which must be "train" or "test_unseen"')


def find_match(param_df: pd.DataFrame, target: dict):
    # exact match for ints, tolerance match for floats
    htc_ok = param_df["htc"].astype(float) == float(target["htc"])
    alpha_ok = param_df["alpha"].astype(float) == float(target["alpha"])
    eps_ok = np.isclose(param_df["epsilon"].astype(float), float(target["epsilon"]), atol=TOL_EPS, rtol=0.0)
    sig_ok = np.isclose(param_df["sigma"].astype(float), float(target["sigma"]), atol=TOL_SIG, rtol=0.0)
    match = param_df[htc_ok & alpha_ok & eps_ok & sig_ok]
    return match


def load_fields(data_dir: str, case_idx: int):
    temp_path = os.path.join(data_dir, f"temperature_all_case{case_idx}.txt")
    stress_path = os.path.join(data_dir, f"stress_all_case{case_idx}.txt")

    if not os.path.exists(temp_path):
        raise FileNotFoundError(f"Missing temperature file: {temp_path}")
    if not os.path.exists(stress_path):
        raise FileNotFoundError(f"Missing stress file: {stress_path}")

    temperature = np.loadtxt(temp_path).reshape(Nt, Nx).T
    stress = np.loadtxt(stress_path).reshape(Nt, Nx).T
    return temperature, stress


def make_tag(which: str, case_idx: int, p: dict):
    # safe string tag for filenames
    sig_str = f"{p['sigma']:.3e}".replace("+", "")
    return f"{which}_case{case_idx}_htc{p['htc']}_eps{p['epsilon']}_sig{sig_str}_a{p['alpha']}"


def plot_temperature_map(temperature: np.ndarray, which: str, case_idx: int, tag: str):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    im = ax.imshow(
        temperature,
        aspect="auto",
        origin="lower",
        cmap="RdYlBu_r", 
        extent=[time_window[0] if "time_window" in globals() else 0, time_window[1] if "time_window" in globals() else 50, 0, Nx - 1]
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Space index (x)")
    ax.set_title(f"FEM Temperature Field ({which}, case {case_idx})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature (K)")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"temperature_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out}")


def plot_stress_map(stress: np.ndarray, which: str, case_idx: int, tag: str):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    im = ax.imshow(
        stress,
    aspect="auto",
    origin="lower",
    cmap="PuOr", 
    extent=[time_window[0] if "time_window" in globals() else 0, time_window[1] if "time_window" in globals() else 50, 0, Nx - 1]
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Space index (x)")
    ax.set_title(f"FEM Stress Field ({which}, case {case_idx})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Stress (Pa)")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"stress_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out}")


def plot_stress_timeseries(stress: np.ndarray, which: str, case_idx: int, tag: str):
    surface_idx = 0
    mid_plane_idx = Nx // 2
    # core_idx = Nx - 1  # optional

    t_ = np.linspace(t_start, t_end, Nt)

    plt.figure(figsize=(5.0, 3.0))
    plt.plot(t_, stress[surface_idx], label=f"Surface (x={surface_idx})")
    plt.plot(t_, stress[mid_plane_idx], label=f"Mid-plane (x={mid_plane_idx})")
    # plt.plot(t_, stress[core_idx], label=f"Opposite surface (x={core_idx})")  # optional

    plt.xlabel("Time (s)")
    plt.ylabel("Stress (Pa)")
    plt.title(f"Stress evolution ({which}, case {case_idx})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"stress_timeseries_{tag}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out}")


# ============================
# Main (plot-only)
# ============================
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

        temperature, stress = load_fields(data_dir, case_idx)
        tag = make_tag(which, case_idx, target_params)

        plot_temperature_map(temperature, which, case_idx, tag)
        plot_stress_map(stress, which, case_idx, tag)
        plot_stress_timeseries(stress, which, case_idx, tag)

        found = True
        break

    if not found:
        raise ValueError(
            f"No matching parameter set found in the selected split(s). Target: {target_params}"
        )


if __name__ == "__main__":
    main()
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Define target input parameters
# ------------------------------------------------------------
target_params = {
    "htc": 150,
    "epsilon": 0.775,
    "sigma": 3.670e-8,
    "alpha": 10
}

# ------------------------------------------------------------
# 2) Choose which dataset to search: "train" or "test_unseen"
# ------------------------------------------------------------
split = "test_unseen"   # <-- change to "train" when needed

if split == "train":
    param_csv = "parameter_combinations_train.csv"
    data_dir = os.path.join("results", "train")
elif split == "test_unseen":
    param_csv = "parameter_combinations_test_unseen.csv"
    data_dir = os.path.join("results", "test_unseen")
else:
    raise ValueError('split must be "train" or "test_unseen"')

# ------------------------------------------------------------
# 3) Load parameter combinations and find matching index
#    (use tolerances for float columns to avoid equality issues)
# ------------------------------------------------------------
param_df = pd.read_csv(param_csv)

tol_eps = 1e-12
tol_sig = 1e-20

match = param_df[
    (param_df["htc"].astype(float) == float(target_params["htc"])) &
    (np.isclose(param_df["epsilon"].astype(float), float(target_params["epsilon"]), atol=tol_eps, rtol=0.0)) &
    (np.isclose(param_df["sigma"].astype(float), float(target_params["sigma"]), atol=tol_sig, rtol=0.0)) &
    (param_df["alpha"].astype(float) == float(target_params["alpha"]))
]

if match.empty:
    raise ValueError(f"❌ No matching parameter set found in {param_csv} for {target_params}")

case_idx = int(match.index[0])
print(f"✅ Matching case index in {split}: {case_idx}")

# ------------------------------------------------------------
# 4) Load and reshape temperature and stress data
# ------------------------------------------------------------
Nt, Nx = 500, 49

temp_path = os.path.join(data_dir, f"temperature_all_case{case_idx}.txt")
stress_path = os.path.join(data_dir, f"stress_all_case{case_idx}.txt")

temperature = np.loadtxt(temp_path).reshape(Nt, Nx).T
stress = np.loadtxt(stress_path).reshape(Nt, Nx).T

# ------------------------------------------------------------
# 5) Plotting (edited: cleaner labels, consistent filenames, no hardcoded colors)
# ------------------------------------------------------------
tag = f"{split}_case{case_idx}_htc{target_params['htc']}_eps{target_params['epsilon']}_sig{target_params['sigma']}_a{target_params['alpha']}"

# === Plot temperature map ===
fig, ax = plt.subplots(figsize=(5.0, 3.0))
im = ax.imshow(
    temperature,
    aspect="auto",
    origin="lower",
    extent=[time_window[0] if "time_window" in globals() else 0, time_window[1] if "time_window" in globals() else 50, 0, Nx - 1]
)
ax.set_xlabel("Time (s)" if "time_window" in globals() else "Time index")
ax.set_ylabel("Space index (x)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Temperature (K)")
ax.set_title(f"FEM Temperature Field ({split}, case {case_idx})")
plt.tight_layout()
plt.savefig(f"temperature_{tag}.png", dpi=600, bbox_inches="tight")
plt.show()

# === Plot stress map ===
fig, ax = plt.subplots(figsize=(5.0, 3.0))
im = ax.imshow(
    stress,
    aspect="auto",
    origin="lower",
    extent=[time_window[0] if "time_window" in globals() else 0, time_window[1] if "time_window" in globals() else 50, 0, Nx - 1]
)
ax.set_xlabel("Time (s)" if "time_window" in globals() else "Time index")
ax.set_ylabel("Space index (x)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Stress (Pa)")
ax.set_title(f"FEM Stress Field ({split}, case {case_idx})")
plt.tight_layout()
plt.savefig(f"stress_{tag}.png", dpi=600, bbox_inches="tight")
plt.show()

# === Plot stress over time at selected spatial indices ===
surface_idx = 0
mid_plane_idx = Nx // 2
core_idx = Nx - 1

t_ = np.linspace(0, 50, Nt)

plt.figure(figsize=(5.0, 3.0))
plt.plot(t_, stress[surface_idx], label=f"Surface (x={surface_idx})")
plt.plot(t_, stress[mid_plane_idx], label=f"Mid-plane (x={mid_plane_idx})")
# plt.plot(t_, stress[core_idx], label=f"Opposite surface (x={core_idx})")  # optional
plt.xlabel("Time (s)")
plt.ylabel("Stress (Pa)")
plt.title(f"Stress evolution ({split}, case {case_idx})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"stress_timeseries_{tag}.png", dpi=600, bbox_inches="tight")
plt.show()
'''