# main_fem.py
# Material parameters and BCs aligned with:
#   Daudeville & Carré, "Thermal Tempering Simulation of Glass Plates:
#   Inner and Edge Residual Stresses", J. Thermal Stresses, 21:667-689, 1998.
#
# Validation case (paper Fig. 11):
#   T_0 = 738 °C = 1011.15 K,  h = 221.5 W/m²K (constant),  T_ext = 20 °C
#   Plate thickness = 6.1 mm  (0.61 cm, thin plate inner-effect case)

import os
import json
import time
import logging
import numpy as np

from petsc4py.PETSc import ScalarType

from fem.geometry import create_mesh
from fem.ThermoViscoProblem import ThermoViscoProblem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FEM-ARONEN2018")

# ============================================================
# Simulation window
# ============================================================
time_window = (0.0, 100.0)
dt           = 0.001
problem_dim  = 1               # 1D: Aronen & Karvinen (2018)

# CG discretisation — standard Lagrange elements, validated against Aronen (2018)
fe_config = {
    "T":     {"element": "CG", "degree": 1},
    "sigma": {"element": "CG", "degree": 1},
    "U":     {"element": "CG", "degree": 1},
}


# ============================================================
# Geometry  —  0.61 cm = 6.1 mm thick plate  (paper Sec. "Inner Effect")
# ============================================================
THICKNESS_MM      = 4.0        # mm — Aronen 2018: b = 4 mm
N_THICKNESS_NODES = 29         # nodes across thickness (odd → midplane node)

N_LENGTH_NODES   = 11          # nodes along y (edge direction)
LENGTH_M         = 0.05        # m — half-length of plate (50 mm half = 100 mm total)

mesh_path        = f"mesh{problem_dim}d.msh"
create_vtx_files = False

# ============================================================
# Zones  —  single uniform quench zone (paper: constant h, constant T_ext)
# ============================================================
ZONES = [
    {
        "name": "Q1",
        "t0":   0.0,
        "t1":   time_window[1],
        "htc":  450.0,          # W/m²K  h = 450 W/m²K (Aronen 2018)
        "htc2": 100.0,          # W/m²K  h₂ — edge HTC (paper Fig. 7/8)
        "T_amb": 293.15,        # K  (20 °C)
    },
]

# ============================================================
# Material / model parameters  —  Daudeville & Carré (1998)
# ============================================================
model_parameters = {
    # --- Boundary condition forcing terms ---
    "f":       0.0,            # volumetric heat source [W/m³]
    "epsilon": 0.0,            # radiation neglected for thin glass (Aronen 2018 Sec 2)
    "sigma":   5.670e-8,       # Stefan-Boltzmann [W/m²K⁴] (kept for completeness)

    # --- Initial / ambient conditions ---
    "T_ambient": ZONES[0]["T_amb"],   # K  (20°C = 293.15K)
    "T_0":       923.15,              # K  (650 °C — Aronen 2018)
    "htc":       ZONES[0]["htc"],     # W/m²K  (Aronen 2018: 450 W/m²K)
    # htc2 not used in 1D

    # --- Thermal material (Table 1 in paper / Saint-Gobain data) ---
    # k(T) and cp(T) are evaluated inline in ThermoViscoProblem._setup_weak_form_T
    "rho":  2530.0,    # kg/m³
    "cp":   1433.0,    # J/kgK  (reference value; temperature-dependent form used)
    "k":    1.0,       # W/mK   (reference value; temperature-dependent form used)

    # --- Narayanaswamy shift-function parameters (Eq. 10) ---
    # H/R = 55 000 K  →  H = 55000 * 8.314 = 457 270 J/mol
    "Hv":  633527.0,   # J/mol  H/R=76200 K → H=76200*8.314 (Aronen 2018 Table 1)
    "H":   633527.0,   # J/mol  same as Hv (Aronen 2018)
    "Tb":  869.0,      # K      reference temperature T_ref  (Table 1 caption)
    "Rg":  8.314,      # J/mol K

    # --- Thermal expansion (Eq. 11) ---
    "alpha_solid":  9.0e-6,    # K⁻¹  β_g  solid glass   (paper p. 675)
    "alpha_liquid": 32.0e-6,   # K⁻¹  αl  liquid glass  (Aronen 2018 Table 1)
    "Tf_init":      923.15,    # K    initial fictive temperature = T_0

    # --- Elastic constants (paper p. 675) ---
    "lambda_":          1.25,          # placeholder Lamé λ (not used directly)
    "mu":               1.0,           # placeholder Lamé μ (not used directly)
    "Young's_modulus":  70.0e9,        # Pa  E = 7×10¹⁰ Pa  ← was 70.0e6 (wrong)
    "Possion_ratio":    0.22,          # ν

    # --- Newmark / time-integration (kept from original) ---
    "beta":    0.5,
    "gamma":   0.5,
    "velocity": 0.16417,   # m/s  (lehr transport speed, not used in pure-quench sim)

    # --- Narayanaswamy structural-relaxation weighting (chi = x in Eq. 10) ---
    # chi = 0.5 is set inside ViscoelasticModel; listed here for documentation only.
    "alpha": 10.0,   # legacy field kept so ThermalModel.__init__ doesn't raise
}

# ============================================================
# Analytical solution constants (kept for surrogate use)
# ============================================================
analytical_constants = {
    "a":       0.2957,
    "c":       1.676e3,
    "Tb":      779.9,
    "E0":      70e9,
    "b":       6.937,
    "H":       22.380e3,
    "k":       -1.231e8,
    "lambda_": 0.7012,
}

# ============================================================
# FE configuration
# ============================================================
# ============================================================
# Zone helpers  (same interface as before)
# ============================================================
def lookup_zone_value(t, key, default=None):
    for z in ZONES:
        if z["t0"] <= t < z["t1"]:
            return z.get(key, default)
    return ZONES[-1].get(key, default)


def current_zone_name(t):
    for z in ZONES:
        if z["t0"] <= t < z["t1"]:
            return z["name"]
    return ZONES[-1]["name"]


def update_zone_controls(model, t):
    Tamb     = float(lookup_zone_value(t, "T_amb",  default=model_parameters["T_0"]))
    htc_val  = float(lookup_zone_value(t, "htc",    default=0.0))
    htc2_val = float(lookup_zone_value(t, "htc2",   default=htc_val))

    model.functions["T_ambient"].interpolate(lambda x: np.full(x.shape[1], Tamb,     dtype=ScalarType))
    model.functions["htc"].interpolate(       lambda x: np.full(x.shape[1], htc_val,  dtype=ScalarType))
    if "htc2" in model.functions:
        model.functions["htc2"].interpolate(  lambda x: np.full(x.shape[1], htc2_val, dtype=ScalarType))
    return Tamb, htc_val


def patch_model_zone_functions(model):
    original_solve_timestep = model.solve_timestep

    def solve_timestep_wrapped(t):
        Tamb, htc_val = update_zone_controls(model, t)
        zone = current_zone_name(t)
        if not hasattr(model, "_last_zone_printed"):
            model._last_zone_printed = None
        if model._last_zone_printed != zone and model.mesh.comm.rank == 0:
            logger.info(
                f"[ZONE] Enter {zone:>3s} at t={t:8.2f}s | "
                f"T_amb={Tamb:8.2f} K | htc={htc_val:8.2f}"
            )
        model._last_zone_printed = zone
        return original_solve_timestep(t)

    model.solve_timestep = solve_timestep_wrapped
    if hasattr(model, "set_zones_from_main"):
        model.set_zones_from_main(ZONES)


# ============================================================
# Output
# ============================================================
def save_outputs(model, out_dir, setup_time, solve_time, total_time):
    os.makedirs(out_dir, exist_ok=True)

    temperature = np.asarray(model.temperature_field_history)
    stress      = np.asarray(model.stress_field_history)

    np.savetxt(os.path.join(out_dir, "temperature_all.txt"), temperature.ravel(), fmt="%.6f")
    np.savetxt(os.path.join(out_dir, "stress_all.txt"),      stress.ravel(),      fmt="%.6f")

    # Save DG temperature DOF coordinates for correct spatial ordering
    # This is needed because DG DOFs are NOT ordered left-to-right spatially
    T_space = model.functionSpaces["T"]
    dof_coords = T_space.tabulate_dof_coordinates()  # (n_dofs, 3)
    np.savetxt(
        os.path.join(out_dir, "T_dof_coords.txt"),
        dof_coords[:, 0],   # x-coordinate only (1D)
        fmt="%.8f",
        header="x-coordinate of each temperature DOF (for DG ordering)"
    )

    # Save simulation metadata for plot script auto-detection
    import json
    meta = {
        "t_start":    time_window[0],
        "t_end":      time_window[1],
        "dt":         dt,
        "Nt":         int(round((time_window[1] - time_window[0]) / dt)),
        "n_nodes":    N_THICKNESS_NODES,
        "thickness_mm": THICKNESS_MM,
        "discretisation": "CG",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as _f:
        json.dump(meta, _f, indent=2)

    # Save fictive temperature history
    if hasattr(model, "fictive_temp_history") and model.fictive_temp_history:
        fictive = np.asarray(model.fictive_temp_history)
        np.savetxt(os.path.join(out_dir, "fictive_temp_all.txt"), fictive.ravel(), fmt="%.6f")

    # Save node coordinates
    coords = np.asarray(model.position_history[-1])
    np.savetxt(os.path.join(out_dir, "node_coords.txt"), coords, fmt="%.8f")

    # Save node coordinates for post-processing — critical for 2D node identification
    coords = np.asarray(model.position_history[-1])   # (n_nodes, dim)
    np.savetxt(os.path.join(out_dir, "node_coords.txt"), coords, fmt="%.8f")

    with open(os.path.join(out_dir, "zones.json"), "w") as f:
        json.dump(ZONES, f, indent=4)
    with open(os.path.join(out_dir, "model_parameters.json"), "w") as f:
        json.dump(model_parameters, f, indent=4)
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(
            {
                "setup_time_s":      float(setup_time),
                "solve_time_s":      float(solve_time),
                "total_time_s":      float(total_time),
                "temperature_shape": list(temperature.shape),
                "stress_shape":      list(stress.shape),
            },
            f, indent=4,
        )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    out_dir = os.path.abspath("results/fem_aronen2018")
    os.makedirs(out_dir, exist_ok=True)

    create_mesh(
        path=mesh_path,
        dim=problem_dim,
        name="aronen2018_validation",
        t_start=time_window[0],
        t_end=time_window[1],
        thickness_mm=THICKNESS_MM,
        n_thickness_nodes=N_THICKNESS_NODES,
    )

    model = ThermoViscoProblem(
        mesh_path=mesh_path,
        problem_dim=problem_dim,
        config=fe_config,
        time=time_window,
        dt=dt,
        model_parameters=model_parameters,
        analy_parameters=analytical_constants,
    )

    t0_setup = time.perf_counter()
    model.setup(
        dirichlet_bc_mech=True,
        create_vtx_files=create_vtx_files,
    )
    t1_setup = time.perf_counter()

    patch_model_zone_functions(model)

    t0_solve = time.perf_counter()
    dto = model.solve()
    t1_solve = time.perf_counter()

    setup_time = t1_setup - t0_setup
    solve_time = t1_solve - t0_solve
    total_time = t1_solve - t0_setup

    save_outputs(model, out_dir, setup_time, solve_time, total_time)

    print("✅ FEM solution finished.")
    print(f"   Setup:  {setup_time:.1f} s")
    print(f"   Solve:  {solve_time:.1f} s")
    print(f"   Total:  {total_time:.1f} s")
    print(f"   Results saved in: {out_dir}")