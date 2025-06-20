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

import numpy as np
import pandas as pd
import itertools
import os

# Define parameter ranges
htc_values = [100, 200]
epsilon_values = [0.7, 0.85]
T_ambient_values = [273.15, 293.15]
T_0_values = [873.15, 923.15]

# Generate all combinations
parameter_combinations = list(itertools.product(htc_values, epsilon_values, T_ambient_values, T_0_values))

# Create DataFrame
param_df = pd.DataFrame(parameter_combinations, columns=["htc", "epsilon", "T_ambient", "T_0"])

# Save to CSV
output_path = "parameter_combinations.csv"
param_df.to_csv(output_path, index=False)


# Write the parametric simulation runner script
import numpy as np
import pandas as pd
import os
from geometry import create_mesh
from ThermoViscoProblem import ThermoViscoProblem
from OutgoingDto import OutgoingDto
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulation config
time = (0.0, 20.0)
dt = 0.1
problem_dim = 1
Zone_name = "all"
mesh_path = f"mesh{problem_dim}d.msh"
create_vtx_files = False

# Load parameter combinations
param_df = pd.read_csv("parameter_combinations.csv")
results_dir = os.path.abspath("results")
os.makedirs(results_dir, exist_ok=True)

# Loop through each parameter set
for i, row in param_df.iterrows():
    logger.info(f"Running simulation {i+1}/{len(param_df)} with parameters: {row.to_dict()}")

    model_params = {
        "f": 0.0,
        "epsilon": row["epsilon"],
        "sigma": 5.670e-8,
        "T_ambient": row["T_ambient"],
        "T_0": row["T_0"],
        "alpha": 15.0,
        "htc": row["htc"],
        "rho": 2500.0,
        "cp": 1433.0,
        "k": 1.0,
        "Hv": 457.05e3,
        "H": 627.8e3,
        "Tb": 869.0,
        "Rg": 8.314,
        "alpha_solid": 9.10e-6,
        "alpha_liquid": 25.10e-6,
        "Tf_init": row["T_0"],
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

    try:
        create_mesh(path=mesh_path, dim=problem_dim, name=Zone_name, t_start=time[0], t_end=time[1])

        model = ThermoViscoProblem(mesh_path=mesh_path, problem_dim=problem_dim,
                                   config=fe_config, time=time, dt=dt,
                                   model_parameters=model_params,
                                   analy_parameters=analytical_constants)

        model.setup(dirichlet_bc_mech=True, create_vtx_files=create_vtx_files)
        dto = model.solve()
        np.savetxt(f"{results_dir}/temperature_all_case{i}.txt", model.all_temperatures, fmt="%.6f")
        np.savetxt(f"{results_dir}/stress_all_case{i}.txt", model.all_stresses, fmt="%.6f")
        np.savetxt(f"{results_dir}/params_case_{i}.txt", row.values[np.newaxis, :], delimiter=",", header="htc,epsilon,T_ambient,T_0", comments="")

    except Exception as e:
        logger.error(f"Simulation {i} failed: {e}")


# now train with FNO and DeepOnet with different dataset