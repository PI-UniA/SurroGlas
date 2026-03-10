from geometry import create_mesh
from ThermoViscoProblem import ThermoViscoProblem
import matplotlib.pyplot as plt
import numpy as np
from AnalyticalSoln import AnalyticalSoln
from OutgoingDto import OutgoingDto
import logging
import numpy as np
import os
import psutil
import json
import time
import statistics
import os
import itertools
import logging
import numpy as np
import pandas as pd
from geometry import create_mesh
from ThermoViscoProblem import ThermoViscoProblem
from OutgoingDto import OutgoingDto

# fem_parametric_runner_unseen.py
# Generates two disjoint FEM datasets:
#   - results/train/         (the original 16 corner/grid cases)
#   - results/test_unseen/   (16 midpoints unseen during training)
#
# Each set contains:
#   temperature_all_case{i}.txt  -> (Nt*Nx,) flattened, row-major (time major)
#   stress_all_case{i}.txt       -> (Nt*Nx,) flattened
#   params_case_{i}.txt          -> single CSV row with header: htc,epsilon,sigma,alpha


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FEM-RUN")

# -----------------------------
# Domain / solver configuration
# -----------------------------
time_window = (0.0,50.0)
dt = 0.1
problem_dim = 1
zone_name = "all"
mesh_path = f"mesh{problem_dim}d.msh"
create_vtx_files = False
Nt, Nx = 500, 49  # expected output shape per your pipeline

# -----------------------------
# TRAIN grid (16 cases): original endpoints
# -----------------------------
htc_values_train     = [100, 200]
epsilon_values_train = [0.7, 0.85]
sigma_values_train   = [1.670e-8, 5.670e-8]
alpha_values_train   = [5, 15]

# -----------------------------
# UNSEEN/TEST grid (16 cases): midpoints only (none of these appear in train)
# -----------------------------
htc_values_test     = [120, 180]
epsilon_values_test = [0.6, 0.8]
sigma_values_test   = [3.670e-8, 4.670e-8]
alpha_values_test   = [7, 10]

'''# To reach 16 cases for test, vary each parameter around the midpoint a bit:
# e.g., two-level jitter around midpoint while staying within bounds
def jitter(vals, perc=0.1):
    # ±10% around each midpoint but clipped into the original train bounds
    out = []
    for v in vals:
        out.extend([v*(1.0 - perc), v*(1.0 + perc)])
    return out

htc_values_test     = jitter(htc_values_test)          # 2 values
epsilon_values_test = jitter(epsilon_values_test)      # 2 values
sigma_values_test   = jitter(sigma_values_test)        # 2 values
alpha_values_test   = jitter(alpha_values_test)        # 2 values
# 2*2*2*2 = 16 unseen combos'''

# -----------------------------
# Common FEM parameter builders
# -----------------------------
def build_model_params(row):
    return {
        "f": 0.0,
        "epsilon": float(row["epsilon"]),
        "sigma": float(row["sigma"]),
        "T_ambient": 273.0,
        "T_0": 873.0,
        "alpha": float(row["alpha"]),
        "htc": float(row["htc"]),
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

def run_batch_and_save(param_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    create_mesh(path=mesh_path, dim=problem_dim, name=zone_name,
                t_start=time_window[0], t_end=time_window[1])

    case_times = []  # store solve runtimes (seconds)

    for i, row in param_df.reset_index(drop=True).iterrows():
        logger.info(f"[{out_dir}] Running {i+1}/{len(param_df)} params={row.to_dict()}")
        try:
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

            t0_solve = time.perf_counter()
            _dto = model.solve()
            t1_solve = time.perf_counter()

            setup_time = t1_setup - t0_setup
            solve_time = t1_solve - t0_solve
            total_time = (t1_solve - t0_setup)

            case_times.append(solve_time)

            logger.info(
                f"[{out_dir}] Case {i}: setup={setup_time:.4f}s, solve={solve_time:.4f}s, total={total_time:.4f}s"
            )

            # Save fields
            np.savetxt(os.path.join(out_dir, f"temperature_all_case{i}.txt"),
                       np.asarray(model.all_temperatures).ravel(), fmt="%.6f")
            np.savetxt(os.path.join(out_dir, f"stress_all_case{i}.txt"),
                       np.asarray(model.all_stresses).ravel(), fmt="%.6f")
            np.savetxt(os.path.join(out_dir, f"params_case_{i}.txt"),
                       row.values[np.newaxis, :],
                       delimiter=",", header="htc,epsilon,sigma,alpha", comments="")

            # Save per-case timing too (optional, but nice)
            with open(os.path.join(out_dir, f"time_case{i}.json"), "w") as f:
                json.dump(
                    {"setup_time_s": setup_time, "solve_time_s": solve_time, "total_time_s": total_time},
                    f,
                    indent=4
                )

        except Exception as e:
            logger.error(f"Simulation {i} failed in {out_dir}: {e}")

    # Save summary timing for the folder (mean/median)
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

# -----------------------------
# Build and run TRAIN (16)
# -----------------------------
train_combos = list(itertools.product(htc_values_train,
                                      epsilon_values_train,
                                      sigma_values_train,
                                      alpha_values_train))
train_df = pd.DataFrame(train_combos, columns=["htc", "epsilon", "sigma", "alpha"])
train_dir = os.path.abspath("results/train")
run_batch_and_save(train_df, train_dir)
train_df.to_csv("parameter_combinations_train.csv", index=False)
print("✅ Saved parameter_combinations_train.csv and results/train/*")

# -----------------------------
# Build and run TEST/UNSEEN (16)
# -----------------------------
test_combos = list(itertools.product(htc_values_test,
                                     epsilon_values_test,
                                     sigma_values_test,
                                     alpha_values_test))
test_df = pd.DataFrame(test_combos, columns=["htc", "epsilon", "sigma", "alpha"])
test_dir = os.path.abspath("results/test_unseen")
run_batch_and_save(test_df, test_dir)
test_df.to_csv("parameter_combinations_test_unseen.csv", index=False)
print("✅ Saved parameter_combinations_test_unseen.csv and results/test_unseen/*")

all_params = pd.concat([train_df, test_df], ignore_index=True)
os.makedirs("results", exist_ok=True)
all_params.to_csv("results/parameters_all.csv", index=False)
print("✅ Saved results/parameters_all.csv (train + test_unseen)")
