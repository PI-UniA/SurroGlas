# fem_parametric_runner_annealing_multizone.py

import os
import json
import time
import logging
import statistics
import itertools
import numpy as np
import pandas as pd

from geometry import create_mesh
from ThermoViscoProblem import ThermoViscoProblem
from OutgoingDto import OutgoingDto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FEM-ANNEALING-RUN")

# ============================================================
# Global domain / solver configuration
# ============================================================
time_window = (0.0, 330.0)   # adapt to your Lehr total duration
dt = 0.1
problem_dim = 1
zone_name = "annealing_lehr_all"
mesh_path = f"mesh{problem_dim}d.msh"
create_vtx_files = False

# expected output shape
# adapt Nx to your real mesh nodes if needed
Nt = int((time_window[1] - time_window[0]) / dt)
Nx = 49
# thickness and nodes used
THICKNESS_MM = 6.0
N_THICKNESS_NODES = 20

# optional geometry values for future 2D
N_LENGTH_NODES = 10
LENGTH_M = 1.0


# ============================================================
# Zone timing (example only — replace with your real timings)
# ============================================================
ZONE_ORDER = ["A1", "A2", "B1", "B2", "C1"]

ZONE_TIME_WINDOWS = {
    "A1": (0.0,   54.7),
    "A2": (54.7,  109.6),
    "B1": (109.6, 182.6),
    "B2": (182.6, 255.57),
    "C1": (255.57, 328.5),
}

# ============================================================
# TRAIN parameter space (corner values)
# ============================================================
train_param_space = {
    "htc_A1":   [20.0, 50.0],
    "htc_A2":   [20.0, 50.0],
    "htc_B1":   [20.0, 50.0],
    "htc_B2":   [20.0, 50.0],
    "htc_C1":   [20.0, 50.0],

    "T_amb_A1": [873.0, 853.0],
    "T_amb_A2": [853.0, 827.0],
    "T_amb_B1": [827.0, 779.0],
    "T_amb_B2": [779.0, 757.0],
    "T_amb_C1": [757.0, 733.0],

    "epsilon":  [0.70, 0.85],
    "sigma":    [1.670e-8, 5.670e-8],
}

# ============================================================
# TEST / UNSEEN parameter space (midpoint-like values)
# ============================================================
test_param_space = {
    "htc_A1":   [30.0, 40.0],
    "htc_A2":   [30.0, 40.0],
    "htc_B1":   [30.0, 40.0],
    "htc_B2":   [30.0, 40.0],
    "htc_C1":   [30.0, 40.0],

    "T_amb_A1": [868.0, 860.0],
    "T_amb_A2": [846.0, 838.0],
    "T_amb_B1": [812.0, 800.0],
    "T_amb_B2": [770.0, 764.0],
    "T_amb_C1": [748.0, 740.0],

    "epsilon":  [0.75, 0.80],
    "sigma":    [3.670e-8, 4.670e-8],
}

# ============================================================
# Fixed thermo-visco material parameters
# ============================================================
def build_model_params(row):
    return {
        "f": 0.0,
        "epsilon": float(row["epsilon"]),
        "sigma": float(row["sigma"]),
        "T_ambient": float(row["T_amb_A1"]),   # initial value; gets updated by zone
        "T_0": 873.0,
        "alpha": 10.0,                         # keep fixed unless you want to vary it too
        "htc": float(row["htc_A1"]),           # initial value; gets updated by zone
        "rho": 2500.0,
        "cp": 1433.0,
        "k": 1.0,
        "Hv": 457.05e3,
        "H": 627.8e3,
        "Tb": 869.0,
        "Rg": 8.314,
        "alpha_solid": 9.10e-6,
        "alpha_liquid": 25.10e-6,
        "Tf_init": 873.0,
        "lambda_": 1.25,
        "mu": 1.0,
        "Young's_modulus": 70.0e6,
        "Possion_ratio": 0.22,
        "beta": 0.5,
        "gamma": 0.5,
        "velocity": 0.24,                      # fixed, as you requested earlier
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
    "T": {"element": "CG", "degree": 1},
    "sigma": {"element": "CG", "degree": 1},
    "U": {"element": "CG", "degree": 1},
}

# ============================================================
# Utilities
# ============================================================
def build_zone_schedule_from_row(row):
    """
    Returns a list of zones with time windows and zone-specific HTC / T_amb.
    """
    schedule = []
    for z in ZONE_ORDER:
        t0, t1 = ZONE_TIME_WINDOWS[z]
        schedule.append({
            "zone": z,
            "t0": t0,
            "t1": t1,
            "htc": float(row[f"htc_{z}"]),
            "T_amb": float(row[f"T_amb_{z}"]),
        })
    return schedule


def get_active_zone_values(current_time, zone_schedule):
    """
    Returns (htc, T_amb, zone_name) for the active zone at current_time.
    """
    for item in zone_schedule:
        if item["t0"] <= current_time < item["t1"]:
            return item["htc"], item["T_amb"], item["zone"]

    # if exactly at final time, keep last zone
    last = zone_schedule[-1]
    return last["htc"], last["T_amb"], last["zone"]


def make_case_dataframe_from_full_factorial(param_space, max_cases=None, seed=42):
    """
    Builds a dataframe from Cartesian product.
    WARNING: for many dimensions this becomes huge.
    Use max_cases to downsample.
    """
    keys = list(param_space.keys())
    combos = list(itertools.product(*(param_space[k] for k in keys)))
    df = pd.DataFrame(combos, columns=keys)

    if max_cases is not None and len(df) > max_cases:
        df = df.sample(n=max_cases, random_state=seed).reset_index(drop=True)

    return df


def save_case_outputs(model, row, out_dir, case_id, setup_time, solve_time, total_time):
    """
    Saves temperature, stress, params, and timing.
    """
    os.makedirs(out_dir, exist_ok=True)

    # These two assume your solver stores full space-time histories:
    #   model.all_temperatures -> shape like (Nt, Nx)
    #   model.all_stresses     -> shape like (Nt, Nx)
    #
    # If your solver uses different variable names, replace them here.
    np.savetxt(
        os.path.join(out_dir, f"temperature_all_case{case_id}.txt"),
        np.asarray(model.all_temperatures).ravel(),
        fmt="%.6f"
    )

    np.savetxt(
        os.path.join(out_dir, f"stress_all_case{case_id}.txt"),
        np.asarray(model.all_stresses).ravel(),
        fmt="%.6f"
    )

    np.savetxt(
        os.path.join(out_dir, f"params_case_{case_id}.txt"),
        row.values[np.newaxis, :],
        delimiter=",",
        header=",".join(row.index.tolist()),
        comments=""
    )

    with open(os.path.join(out_dir, f"time_case{case_id}.json"), "w") as f:
        json.dump(
            {
                "setup_time_s": float(setup_time),
                "solve_time_s": float(solve_time),
                "total_time_s": float(total_time),
            },
            f,
            indent=4
        )


# ============================================================
# IMPORTANT: connect your annealing Lehr solve logic here
# ============================================================
from petsc4py.PETSc import ScalarType
import numpy as np

def run_annealing_case(row):
    """
    Runs one annealing-Lehr case with zone-wise htc/T_amb and global epsilon/sigma.
    """

    # build zone table for this case
    ZONES = [
        dict(name="A1", t0=0.0,    t1=54.7,   htc=float(row["htc_A1"]), T_amb=float(row["T_amb_A1"])),
        dict(name="A2", t0=54.7,   t1=109.6,  htc=float(row["htc_A2"]), T_amb=float(row["T_amb_A2"])),
        dict(name="B1", t0=109.6,  t1=182.6,  htc=float(row["htc_B1"]), T_amb=float(row["T_amb_B1"])),
        dict(name="B2", t0=182.6,  t1=255.57, htc=float(row["htc_B2"]), T_amb=float(row["T_amb_B2"])),
        dict(name="C1", t0=255.57, t1=328.5,  htc=float(row["htc_C1"]), T_amb=float(row["T_amb_C1"])),
    ]

    model = ThermoViscoProblem(
        mesh_path=mesh_path,
        problem_dim=problem_dim,
        config=fe_config,
        time=time_window,
        dt=dt,
        model_parameters=build_model_params(row),
        analy_parameters=analytical_constants
    )

    t0_setup = time.perf_counter()
    model.setup(dirichlet_bc_mech=True, create_vtx_files=create_vtx_files)
    t1_setup = time.perf_counter()

    def lookup_zone_value(t: float, key: str, default=None):
        for z in ZONES:
            if z["t0"] <= t < z["t1"]:
                return z.get(key, default)
        if ZONES:
            return ZONES[-1].get(key, default)
        return default

    def current_zone_name(t: float) -> str:
        for z in ZONES:
            if z["t0"] <= t < z["t1"]:
                return z["name"]
        return ZONES[-1]["name"] if ZONES else "NA"

    def update_zone_controls(model, t: float):
        Tamb = float(lookup_zone_value(t, "T_amb", default=build_model_params(row)["T_0"]))

        def T_ambient_expr(x):
            return np.full(x.shape[1], Tamb, dtype=ScalarType)

        model.functions["T_ambient"].interpolate(T_ambient_expr)

        htc_val = float(lookup_zone_value(t, "htc", default=0.0))

        def htc_expr(x):
            return np.full(x.shape[1], htc_val, dtype=ScalarType)

        model.functions["htc"].interpolate(htc_expr)

        return Tamb, htc_val

    def patch_model_zone_functions(model):
        original_solve_timestep = model.solve_timestep

        def solve_timestep_wrapped(t: float):
            Tamb, htc_val = update_zone_controls(model, t)

            name = current_zone_name(t)
            if not hasattr(model, "_last_zone_printed"):
                model._last_zone_printed = None

            if model._last_zone_printed != name and model.mesh.comm.rank == 0:
                logger.info(
                    f"[ZONE] Enter {name:>3s} at t={t:8.2f}s | "
                    f"T_amb={Tamb:8.2f}K | htc={htc_val:8.2f}"
                )

            model._last_zone_printed = name
            return original_solve_timestep(t)

        model.solve_timestep = solve_timestep_wrapped

        if hasattr(model, "set_zones_from_main"):
            model.set_zones_from_main(ZONES)

    patch_model_zone_functions(model)

    t0_solve = time.perf_counter()
    _dto = model.solve()
    t1_solve = time.perf_counter()

    # save histories for dataset export
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
def run_batch_and_save(param_df: pd.DataFrame, out_dir: str):
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

    case_times = []

    for i, row in param_df.reset_index(drop=True).iterrows():
        logger.info(f"[{out_dir}] Running case {i+1}/{len(param_df)}")
        logger.info(f"Parameters = {row.to_dict()}")

        try:
            model, setup_time, solve_time, total_time = run_annealing_case(row)

            case_times.append(solve_time)

            logger.info(
                f"[{out_dir}] Case {i}: "
                f"setup={setup_time:.4f}s, solve={solve_time:.4f}s, total={total_time:.4f}s"
            )

            save_case_outputs(
                model=model,
                row=row,
                out_dir=out_dir,
                case_id=i,
                setup_time=setup_time,
                solve_time=solve_time,
                total_time=total_time
            )

        except Exception as e:
            logger.exception(f"Simulation {i} failed in {out_dir}: {e}")

    if case_times:
        timing_summary = {
            "n_cases": len(case_times),
            "solve_time_mean_s": float(statistics.mean(case_times)),
            "solve_time_median_s": float(statistics.median(case_times)),
            "solve_time_min_s": float(min(case_times)),
            "solve_time_max_s": float(max(case_times)),
        }
        with open(os.path.join(out_dir, "timing_summary.json"), "w") as f:
            json.dump(timing_summary, f, indent=4)

        print(f"✅ Timing summary saved: {os.path.join(out_dir, 'timing_summary.json')}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # IMPORTANT:
    # Full factorial for 12 parameters can be huge.
    # Here we cap the number of sampled cases.
    train_df = make_case_dataframe_from_full_factorial(
        train_param_space,
        max_cases=64,   # change as you want
        seed=42
    )

    test_df = make_case_dataframe_from_full_factorial(
        test_param_space,
        max_cases=16,   # change as you want
        seed=123
    )

    train_dir = os.path.abspath("results/train")
    test_dir = os.path.abspath("results/test_unseen")

    run_batch_and_save(train_df, train_dir)
    train_df.to_csv("parameter_combinations_train.csv", index=False)
    print("✅ Saved parameter_combinations_train.csv and results/train/*")

    run_batch_and_save(test_df, test_dir)
    test_df.to_csv("parameter_combinations_test_unseen.csv", index=False)
    print("✅ Saved parameter_combinations_test_unseen.csv and results/test_unseen/*")

    all_params = pd.concat([train_df, test_df], ignore_index=True)
    all_params.to_csv("results/parameters_all.csv", index=False)
    print("✅ Saved results/parameters_all.csv")