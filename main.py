# main.py  —  annealing-Lehr parametric runner (multi-zone, thermo-visco)
#
# Runs the annealing Lehr across all five temperature zones (A1..C1), each with
# its own convective htc and ambient/wall temperature T_amb, plus per-case
# radiation features (epsilon, sigma).  The full residence time is covered so
# the glass actually cools through the annealing/strain point in the last zones
# and the saved stress is the *annealed* state.
#
# Design:
#   * time_window is DERIVED from the zone table, so it always spans the whole
#     Lehr (start of A1 -> end of the last zone) and can never be truncated.
#   * The zone schedule is defined ONCE (ZONE_TIME_WINDOWS) and reused; per step
#     the active zone's htc / T_amb are interpolated into the solver Functions.
#   * epsilon and sigma are kept as swept, per-case radiation features (constant
#     across zones); a fresh model is built per case so each gets its own values.
#   * The train factorial is sampled WITHOUT materialising the full product.

import os
import json
import time
import random
import logging
import statistics
import itertools

import numpy as np
import pandas as pd
from petsc4py.PETSc import ScalarType

from fem.geometry import create_mesh
from fem.ThermoViscoProblem import ThermoViscoProblem
from OutgoingDto import OutgoingDto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FEM-ANNEALING-RUN")

# ============================================================
# Zone timing  (single source of truth)
# ============================================================
ZONE_ORDER = ["A1", "A2", "B1", "B2", "C1"]

ZONE_TIME_WINDOWS = {
    "A1": (0.0,    54.7),
    "A2": (54.7,   109.6),
    "B1": (109.6,  182.6),
    "B2": (182.6,  255.57),
    "C1": (255.57, 328.5),
}

# ============================================================
# Global domain / solver configuration
# ============================================================
# time_window is derived from the zones so the simulation ALWAYS covers the
# full Lehr.  This is the single change that makes the annealing physically
# complete: the glass now reaches the cold final zones and cools through the
# strain point instead of stopping mid-Lehr.
LEHR_START = ZONE_TIME_WINDOWS[ZONE_ORDER[0]][0]
LEHR_END   = ZONE_TIME_WINDOWS[ZONE_ORDER[-1]][1]     # = 328.5 s
time_window = (LEHR_START, LEHR_END)

dt = 0.1                       # 3285 steps over the Lehr; fine for slow annealing
problem_dim = 1
zone_name = "annealing_lehr_all"
mesh_path = f"mesh{problem_dim}d.msh"
create_vtx_files = False

# Belt speed: time -> Lehr distance.  Single source; also written to meta.json
# so the plot script uses exactly the same value.
VELOCITY = 0.16417             # m/s

# Geometry (through-thickness mesh)
THICKNESS_MM = 4.0
N_THICKNESS_NODES = 29         # must match the mesh actually built below

# Optional geometry values for a future 2D extension
N_LENGTH_NODES = 10
LENGTH_M = 1.0

# ============================================================
# TRAIN parameter space  (wider; covers the test range too)
# ============================================================
train_param_space = {
    "htc_A1":   [420.0, 435.0, 450.0],
    "htc_A2":   [420.0, 435.0, 450.0],
    "htc_B1":   [420.0, 435.0, 450.0],
    "htc_B2":   [420.0, 435.0, 450.0],
    "htc_C1":   [420.0, 435.0, 450.0],

    "T_amb_A1": [873.0, 866.0, 860.0, 853.0],
    "T_amb_A2": [853.0, 846.0, 838.0, 827.0],
    "T_amb_B1": [827.0, 812.0, 800.0, 779.0],
    "T_amb_B2": [779.0, 770.0, 764.0, 757.0],
    "T_amb_C1": [757.0, 748.0, 740.0, 733.0],

    "epsilon":  [0.70, 0.775, 0.85],
    "sigma":    [1.670e-8, 3.670e-8, 5.670e-8],
}

# ============================================================
# TEST / UNSEEN parameter space  (midpoints NOT in train)
# ============================================================
test_param_space = {
    "htc_A1":   [427.0, 443.0],
    "htc_A2":   [427.0, 443.0],
    "htc_B1":   [427.0, 443.0],
    "htc_B2":   [427.0, 443.0],
    "htc_C1":   [427.0, 443.0],

    "T_amb_A1": [869.5, 856.5],
    "T_amb_A2": [849.5, 832.5],
    "T_amb_B1": [819.5, 789.5],
    "T_amb_B2": [774.5, 760.5],
    "T_amb_C1": [752.5, 736.5],

    "epsilon":  [0.737, 0.812],
    "sigma":    [2.670e-8, 4.670e-8],
}

# ============================================================
# Fixed thermo-visco material parameters
# ============================================================
def build_model_params(row):
    """
    Per-case model parameters.  epsilon and sigma are the radiation features
    (kept constant across zones, swept per case).  T_ambient / htc are seeded
    with the FIRST zone's values; they are then overwritten every step by the
    active zone (see run_annealing_case).
    """
    first = ZONE_ORDER[0]
    return {
        "f": 0.0,
        "epsilon": float(row["epsilon"]),           # radiation feature (per case)
        "sigma":   float(row["sigma"]),             # radiation feature (per case)
        "T_ambient": float(row[f"T_amb_{first}"]),  # seed; updated per zone
        "T_0": 923.15,
        "alpha": 10.0,
        "htc": float(row[f"htc_{first}"]),          # seed; updated per zone
        "rho": 2530.0,
        "cp": 1433.0,
        "k": 1.0,
        "HvRg": 76200.0,
        "H": 633527.0,
        "Tb": 869.0,
        "Rg": 8.314,
        "alpha_solid": 9.10e-6,
        "alpha_liquid": 32.10e-6,
        "Tf_init": 923.15,
        "lambda_": 1.25,
        "mu": 1.0,
        "Young's_modulus": 70.0e9,
        "Possion_ratio": 0.22,
        "beta": 0.5,
        "gamma": 0.5,
        "velocity": VELOCITY,
    }

analytical_constants = {
    "a": 0.2957,
    "c": 1.676e3,
    "Tb": 779.9,
    "E0": 70e9,
    "b": 6.937,
    "H": 22.380e3,
    "k": -1.231e8,
    "lambda_": 0.7012,
}

fe_config = {
    "T":     {"element": "CG", "degree": 1},
    "sigma": {"element": "CG", "degree": 1},
    "U":     {"element": "CG", "degree": 1},
}

# ============================================================
# Zone schedule  (single source of truth, used everywhere)
# ============================================================
def zone_schedule(row):
    """List of zones with their time window + this case's htc / T_amb."""
    return [
        dict(name=z,
             t0=ZONE_TIME_WINDOWS[z][0],
             t1=ZONE_TIME_WINDOWS[z][1],
             htc=float(row[f"htc_{z}"]),
             T_amb=float(row[f"T_amb_{z}"]))
        for z in ZONE_ORDER
    ]


def active_zone(t, schedule):
    """Zone active at time t (clamped to the last zone at/after the Lehr end)."""
    for z in schedule:
        if z["t0"] <= t < z["t1"]:
            return z
    return schedule[-1]


# ============================================================
# Case sampling  (memory-safe: does NOT materialise the full product)
# ============================================================
def sample_factorial(param_space, max_cases=None, seed=42):
    """
    Sample up to `max_cases` UNIQUE combinations from the full factorial without
    building the whole Cartesian product in memory.  Falls back to the complete
    product when it is smaller than `max_cases`.
    """
    keys  = list(param_space.keys())
    sizes = [len(param_space[k]) for k in keys]
    total = 1
    for s in sizes:
        total *= s

    rng = random.Random(seed)
    if max_cases is None or total <= max_cases:
        rows = list(itertools.product(*(param_space[k] for k in keys)))
    else:
        rows = []
        for idx in rng.sample(range(total), max_cases):   # unique, cheap even for millions
            combo = []
            for s, k in zip(sizes, keys):
                idx, r = divmod(idx, s)
                combo.append(param_space[k][r])
            rows.append(tuple(combo))

    return pd.DataFrame(rows, columns=keys)


# ============================================================
# Metadata for the plot script (single source of truth)
# ============================================================
def write_meta(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "t_start":        time_window[0],
        "t_end":          time_window[1],
        "dt":             dt,
        "Nt":             int(round((time_window[1] - time_window[0]) / dt)),
        "n_nodes":        N_THICKNESS_NODES,
        "thickness_mm":   THICKNESS_MM,
        "velocity":       VELOCITY,
        "discretisation": "CG",
        "zone_order":     ZONE_ORDER,
        "zone_time_windows": {z: list(ZONE_TIME_WINDOWS[z]) for z in ZONE_ORDER},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Output saving
# ============================================================
def save_case_outputs(model, row, out_dir, case_id, setup_time, solve_time, total_time):
    """Save temperature/stress space-time histories, the parameter row, and timing."""
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(
        os.path.join(out_dir, f"temperature_all_case{case_id}.txt"),
        np.asarray(model.all_temperatures).ravel(),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(out_dir, f"stress_all_case{case_id}.txt"),
        np.asarray(model.all_stresses).ravel(),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(out_dir, f"params_case_{case_id}.txt"),
        row.values[np.newaxis, :],
        delimiter=",",
        header=",".join(row.index.tolist()),
        comments="",
    )
    with open(os.path.join(out_dir, f"time_case{case_id}.json"), "w") as f:
        json.dump(
            {"setup_time_s": float(setup_time),
             "solve_time_s": float(solve_time),
             "total_time_s": float(total_time)},
            f, indent=4,
        )


# ============================================================
# Single annealing-Lehr case
# ============================================================
def _const_field(value):
    """Interpolant that fills a scalar Function with a constant value."""
    v = float(value)
    def f(x):
        return np.full(x.shape[1], v, dtype=ScalarType)
    return f


def run_annealing_case(row):
    """
    Run one annealing-Lehr case with zone-wise htc / T_amb and per-case
    epsilon / sigma.  Before every time step the active zone's htc and T_amb are
    interpolated into the solver Functions, so BOTH the convective and the
    radiative boundary flux follow the zone.

    IMPORTANT (verify in ThermoViscoProblem/ThermalModel): the radiation term
    must use the SAME functions["T_ambient"] as convection, i.e.
        q = htc*(T - T_amb) + epsilon*sigma*(T^4 - T_amb^4)
    with the one T_amb updated here.  Otherwise radiation will not track the
    zone and the late-zone cooling will be wrong.
    """
    schedule = zone_schedule(row)

    model = ThermoViscoProblem(
        mesh_path=mesh_path,
        problem_dim=problem_dim,
        config=fe_config,
        time=time_window,
        dt=dt,
        model_parameters=build_model_params(row),
        analy_parameters=analytical_constants,
    )

    t0_setup = time.perf_counter()
    model.setup(dirichlet_bc_mech=True, create_vtx_files=create_vtx_files)
    t1_setup = time.perf_counter()

    def update_zone_controls(t):
        z = active_zone(t, schedule)
        model.functions["T_ambient"].interpolate(_const_field(z["T_amb"]))
        model.functions["htc"].interpolate(_const_field(z["htc"]))
        return z

    # Wrap solve_timestep so the zone controls are refreshed before each step.
    original_solve_timestep = model.solve_timestep

    def solve_timestep_wrapped(t):
        z = update_zone_controls(t)
        if getattr(model, "_last_zone_printed", None) != z["name"] and model.mesh.comm.rank == 0:
            logger.info(
                f"[ZONE] Enter {z['name']:>3s} at t={t:8.2f}s | "
                f"T_amb={z['T_amb']:8.2f}K | htc={z['htc']:8.2f}"
            )
        model._last_zone_printed = z["name"]
        return original_solve_timestep(t)

    model.solve_timestep = solve_timestep_wrapped

    # If the solver has native zone support, hand it the same schedule.
    if hasattr(model, "set_zones_from_main"):
        model.set_zones_from_main(schedule)

    t0_solve = time.perf_counter()
    _dto = model.solve()
    t1_solve = time.perf_counter()

    # Collect space-time histories for dataset export.
    if hasattr(model, "temperature_field_history"):
        model.all_temperatures = np.asarray(model.temperature_field_history)
    else:
        raise AttributeError("model.temperature_field_history not found")

    if hasattr(model, "stress_field_history"):
        model.all_stresses = np.asarray(model.stress_field_history)
    else:
        raise AttributeError("model.stress_field_history not found")

    setup_time = t1_setup - t0_setup
    solve_time = t1_solve - t0_solve
    total_time = t1_solve - t0_setup
    return model, setup_time, solve_time, total_time


# ============================================================
# Batch runner
# ============================================================
def run_batch_and_save(param_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    create_mesh(
        path=mesh_path,
        dim=problem_dim,
        name=zone_name,
        t_start=time_window[0],
        t_end=time_window[1],
        thickness_mm=THICKNESS_MM,
        n_thickness_nodes=N_THICKNESS_NODES,
        length_m=LENGTH_M,
        n_length_nodes=N_LENGTH_NODES,
    )

    # Write metadata up front so the plot script can auto-detect this run.
    write_meta(out_dir)

    case_times = []
    for i, row in param_df.reset_index(drop=True).iterrows():
        logger.info(f"[{out_dir}] Running case {i + 1}/{len(param_df)}")
        logger.info(f"Parameters = {row.to_dict()}")

        try:
            model, setup_time, solve_time, total_time = run_annealing_case(row)
            case_times.append(solve_time)
            logger.info(
                f"[{out_dir}] Case {i}: setup={setup_time:.4f}s, "
                f"solve={solve_time:.4f}s, total={total_time:.4f}s"
            )
            save_case_outputs(model, row, out_dir, i, setup_time, solve_time, total_time)
        except Exception as e:
            logger.exception(f"Simulation {i} failed in {out_dir}: {e}")

    if case_times:
        timing_summary = {
            "n_cases": len(case_times),
            "solve_time_mean_s":   float(statistics.mean(case_times)),
            "solve_time_median_s": float(statistics.median(case_times)),
            "solve_time_min_s":    float(min(case_times)),
            "solve_time_max_s":    float(max(case_times)),
        }
        with open(os.path.join(out_dir, "timing_summary.json"), "w") as f:
            json.dump(timing_summary, f, indent=4)
        print(f"[OK] Timing summary saved: {os.path.join(out_dir, 'timing_summary.json')}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    logger.info(
        f"Lehr time window = {time_window} s  "
        f"({int(round((LEHR_END - LEHR_START) / dt))} steps, dt={dt}s)"
    )

    # Full factorials are large (train ~2.24M combos); sample without materialising.
    train_df = sample_factorial(train_param_space, max_cases=64, seed=42)
    test_df  = sample_factorial(test_param_space,  max_cases=8,  seed=123)

    train_dir = os.path.abspath("results/train")
    test_dir  = os.path.abspath("results/test_unseen")

    run_batch_and_save(train_df, train_dir)
    train_df.to_csv("parameter_combinations_train.csv", index=False)
    print("[OK] Saved parameter_combinations_train.csv and results/train/*")

    run_batch_and_save(test_df, test_dir)
    test_df.to_csv("parameter_combinations_test_unseen.csv", index=False)
    print("[OK] Saved parameter_combinations_test_unseen.csv and results/test_unseen/*")

    all_params = pd.concat([train_df, test_df], ignore_index=True)
    all_params.to_csv("results/parameters_all.csv", index=False)
    print("[OK] Saved results/parameters_all.csv")