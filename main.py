import logging
import numpy as np
import matplotlib.pyplot as plt
from geometry import create_mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from ThermoViscoProblem import ThermoViscoProblem
from OutgoingDto import OutgoingDto
from AnalyticalSoln import AnalyticalSoln
from pathlib import Path


def main():
    # ----------------------------
    # 1) Logging
    # ----------------------------
    logger = logging.getLogger(__name__)
    logger.setLevel("DEBUG")
    logger.propagate = False

    formatter = logging.Formatter(
        "{asctime} - {levelname} - {filename} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel("DEBUG")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
        file_handler.setLevel("INFO")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # ----------------------------
    # 2) JIT options
    # ----------------------------
    jit_options = {"cffi_extra_compile_args": ["-O3", "-march=native"]}

    # ----------------------------
    # 3) User controls
    # ----------------------------
    BASE_DIR = Path(__file__).resolve().parent
    mesh_dir = BASE_DIR / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = mesh_dir / "glass_1d.msh"
    mesh_path = str(mesh_path)

    problem_dim = 1

    time = (0.0, 350.0)
    dt = 0.1
    n_steps = int((time[1] - time[0]) / dt)

    # thickness and nodes used
    THICKNESS_MM = 6.0
    N_THICKNESS_NODES = 20

    # optional geometry values for future 2D
    N_LENGTH_NODES = 10
    LENGTH_M = 1.0

    # ----------------------------
    # 4) Zone table
    # ----------------------------
    ZONES = [
        dict(name="A1", t0=0.0,    t1=54.7,   htc=20.0, T_amb=845.0),
        dict(name="A2", t0=54.7,   t1=109.6,  htc=20.0, T_amb=818.0),
        dict(name="B1", t0=109.6,  t1=182.6,  htc=20.0, T_amb=790.0),
        dict(name="B2", t0=182.6,  t1=255.57, htc=20.0, T_amb=763.0),
        dict(name="C1", t0=255.57, t1=328.5,  htc=20.0, T_amb=741.0),
    ]

    # ----------------------------
    # 5) FEM element config
    # ----------------------------
    config = {
        "T":     {"element": "CG", "degree": 1},
        "U":     {"element": "CG", "degree": 1},
        "sigma": {"element": "CG", "degree": 1},
    }

    # ----------------------------
    # 6) Model parameters
    # ----------------------------
    model_params = {
        "f": 0.0,
        "epsilon": 0.87,
        "sigma": 5.670e-8,
        "velocity": 0.24,
        "T_0": 873.0,
        "alpha": 15.0,
        "rho": 2500.0,
        "k": 0.80,
        "Hv": 457.05e3,
        "H": 627.8e3,
        "Tb": 869.0,
        "Rg": 8.314,
        "alpha_solid": 9.0e-6,
        "alpha_liquid": 32.50e-6,
        "Tf_init": 873.0,
        "lambda_": 1.25,
        "mu": 1.0,
        "Young's_modulus": 70.0e6,
        "Possion_ratio": 0.22,
    }

    # ----------------------------
    # 7) Analytical parameters
    # ----------------------------
    analytical_consts = {
        "a": 0.2957,
        "c": 1.676e3,
        "Tb": 779.9,
        "E0": 70e9,
        "b": 6.937,
        "H": 22.380e3,
        "k": -1.231e8,
        "lambda_": 0.7012,
    }

    # ----------------------------
    # 8) Helpers for zone controls
    # ----------------------------
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

    def update_zone_controls(model: ThermoViscoProblem, t: float):
        Tamb = float(lookup_zone_value(t, "T_amb", default=model_params["T_0"]))

        def T_ambient_expr(x):
            return np.full(x.shape[1], Tamb, dtype=ScalarType)

        model.functions["T_ambient"].interpolate(T_ambient_expr)

        htc_val = float(lookup_zone_value(t, "htc", default=0.0))

        def htc_expr(x):
            return np.full(x.shape[1], htc_val, dtype=ScalarType)

        model.functions["htc"].interpolate(htc_expr)

        return Tamb, htc_val

    def patch_model_zone_functions(model: ThermoViscoProblem):
        model.T_ambient_zone = lambda t: float(
            lookup_zone_value(t, "T_amb", default=model_params["T_0"])
        )
        model.htc_zone = lambda t: float(
            lookup_zone_value(t, "htc", default=0.0)
        )

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

    # ----------------------------
    # 9) Plotting helper functions
    # ----------------------------
    def plot_temperature_field_map(model, t_):
        if not hasattr(model, "temperature_field_history"):
            logger.warning("temperature_field_history not found in model. Skipping temperature map.")
            return

        temperature = np.array(model.temperature_field_history).T
        x_coords = model.mesh.geometry.x[:, 0]

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            temperature,
            aspect="auto",
            origin="lower",
            cmap="RdYlBu_r",
            extent=[t_[0], t_[-1], x_coords.min(), x_coords.max()]
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Thickness coordinate x (m)")
        ax.set_title("Temperature Field T(x,t)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Temperature (K)")
        plt.tight_layout()
        plt.savefig("temperature_field_map.png", dpi=600, bbox_inches="tight")
        plt.show()

    def plot_stress_field_map(model, t_):
        if not hasattr(model, "stress_field_history"):
            logger.warning("stress_field_history not found in model. Skipping stress map.")
            return

        stress = np.array(model.stress_field_history).T
        x_coords = model.mesh.geometry.x[:, 0]

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            stress,
            aspect="auto",
            origin="lower",
            cmap="PuOr",
            extent=[t_[0], t_[-1], x_coords.min(), x_coords.max()]
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Thickness coordinate x (m)")
        ax.set_title("Stress Field σ(x,t)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Stress (MPa)")
        plt.tight_layout()
        plt.savefig("stress_field_map.png", dpi=600, bbox_inches="tight")
        plt.show()

    # ----------------------------
    # 10) Run simulation
    # ----------------------------
    REGENERATE_MESH = True
    if REGENERATE_MESH:
        create_mesh(
            path=mesh_path,
            dim=problem_dim,
            name="glass_1d",
            t_start=0,
            t_end=0,
            thickness_mm=THICKNESS_MM,
            n_thickness_nodes=N_THICKNESS_NODES,
            length_m=LENGTH_M,
            n_length_nodes=N_LENGTH_NODES,
        )

    if not Path(mesh_path).exists():
        raise FileNotFoundError(f"Mesh not found after create_mesh: {mesh_path}")

    model = None
    dto = None

    try:
        model = ThermoViscoProblem(
            mesh_path=mesh_path,
            problem_dim=problem_dim,
            time=time,
            dt=dt,
            config=config,
            model_parameters=model_params,
            analy_parameters=analytical_consts,
            jit_options=jit_options,
        )

        model.setup(dirichlet_bc_mech=True, create_vtx_files=True)
        patch_model_zone_functions(model)

        dto = model.solve()

        if model.mesh.comm.rank == 0:
            logger.info(f"Simulation executed. - Execution Time: {np.round(model.execution_time, 6)}s")
            logger.info(f"Number of elements in OutgoingDto: {dto.num_elements()}")
            logger.info("Code executed")

    except Exception as e:
        logger.error(e, exc_info=True)
        dto = OutgoingDto()

    # ----------------------------
    # 11) Plotting
    # ----------------------------
    if model is not None and problem_dim == 1 and model.mesh.comm.rank == 0:
        t_ = np.linspace(start=0.0, stop=time[1], num=len(model.avg_T))

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 15

        # Temperature plot
        #plt.figure(dpi=600)
        plt.plot(t_, model.T_0_edge, label="Simulated results at 1st node", color="b")
        plt.xlabel("Time (s)")
        plt.ylabel("Temperatures (K)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Stress plot
        #plt.figure(dpi=600)
        plt.plot(t_, model.avg_t_sigma_surface, label="Stresses at surface", color="red", linestyle="-")
        plt.plot(t_, model.avg_t_sigma_mid, label="Stresses at center", color="black", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Stress (MPa)")
        plt.legend()
        plt.grid(True)
        plt.xlim(0, t_[-1])
        plt.tight_layout()
        plt.show()

        # Time -> distance plots
        v_m_per_s = 9.85 / 60.0
        x_ = v_m_per_s * np.asarray(t_)
        L = 100.0
        mask = x_ <= L

        #plt.figure(dpi=600)
        plt.plot(x_[mask], np.asarray(model.T_0_edge)[mask], label="Surface temperature", color="b")
        plt.xlabel("Lehr distance x (m)")
        plt.ylabel("Surface temperature (K)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        #plt.figure(dpi=600)
        plt.plot(x_[mask], np.asarray(model.avg_t_sigma_surface)[mask], label="Surface stress", color="red", linestyle="-")
        plt.plot(x_[mask], np.asarray(model.avg_t_sigma_mid)[mask], label="Center stress", color="black", linestyle="--")
        plt.xlabel("Lehr distance x (m)")
        plt.ylabel("Stress (MPa)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # NEW: field maps
        plot_temperature_field_map(model, t_)
        plot_stress_field_map(model, t_)


if __name__ == "__main__":
    main()

'''
#plt.subplot(2, 3, 1)
#Stresses over time
#plt.subplot(2, 3, 6)
#plt.figure(figsize=(6, 4), dpi=600)  # high resolution figure

#plt.xscale('log')                 # logarithmic time axis
#plt.ylim(-20, 10)                 # y-axis limits in MPa

# Shift functions
#plt.subplot(2, 3, 2)
plt.plot(t_, phi_, label='Analytical results', color='r')
plt.plot(t_, model.avg_phi, label='Simulated results', color='b')
>>>>>>> origin/thussein_stress_calculation
plt.xlabel('Time (s)')
plt.ylabel('Shift function')
plt.legend()
plt.grid(True)
plt.show()

#Scaled times
#plt.subplot(2, 3, 3)
plt.plot(t_, xi_, label='Analytical results', color='r')
plt.plot(t_, model.avg_xi, label='Simulated results', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Scaled time (s)')
plt.legend()
plt.grid(True)
plt.show()

#Strains
#plt.subplot(2, 3, 4)
plt.plot(t_, dedt, label='Analytical results', color='r')
plt.plot(t_, model.avg_t_epsilon, label='Simulated results', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Total strain (-)')
plt.legend()
plt.grid(True)
plt.show()

#Stresses
#plt.subplot(2, 3, 5)
plt.plot(t_, sigma_analytical_, label='Analytical results', color='lightgreen', linestyle='--')
plt.plot(t_, model.avg_t_sigma, label='Simulated results', color='darkgreen')
plt.xlabel('Time (s)')
plt.ylabel('Stress (MPa)')
plt.legend()
plt.grid(True)
plt.show()

#Stresses over time
#plt.subplot(2, 3, 6)
plt.plot(t_, model.avg_t_sigma_surface, label='Stresses at surface ', color='b')
plt.plot(t_, model.avg_t_sigma_mid, label='Stresses at mid_plane', color='r')

#plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlabel('Time (s)')
plt.ylabel('Stress (MPa)')
plt.legend()
plt.grid(True)
plt.show()

# Number of available cores
print(f"Number of CPU cores: {os.cpu_count()}")


# Get the current process
process = psutil.Process()

# CPU utilization percentage per core
print(f"CPU usage per core: {psutil.cpu_percent(percpu=True)}")

# Current process's CPU utilization
print(f"Current process CPU usage: {process.cpu_percent(interval=1.0)}")

# idea how to import scaled times into stress equations
# understand what is ([x[0])
# how the position is time-dependent x(t)


# today scaled done and phi done
# epsilon - previous


# --- Plotting block starts here ---
# Extract x and temperature
x = model.mesh.geometry.x[:, 0]  # 1D mesh
T = model.functions_current["T"].x.array[:]

# Sort for smooth plot
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
T_sorted = T[sorted_indices]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x_sorted, T_sorted, label='Temperature Profile', color='red', linewidth=2)

# Zone markers
plt.axvline(x=20, color='black', linestyle='--', label='End of Zone A')
plt.axvline(x=40, color='black', linestyle='--', label='End of Zone B1')

# Shading zones
plt.fill_between(x_sorted, T_sorted.min(), T_sorted.max(), where=(x_sorted <= 20), color='blue', alpha=0.1, label='Zone A')
plt.fill_between(x_sorted, T_sorted.min(), T_sorted.max(), where=((x_sorted > 20) & (x_sorted <= 40)), color='green', alpha=0.1, label='Zone B1')
plt.fill_between(x_sorted, T_sorted.min(), T_sorted.max(), where=(x_sorted > 40), color='orange', alpha=0.1, label='Zone B2')

# Labels and title
plt.xlabel('Position along Lehr [m]')
plt.ylabel('Temperature [K]')
plt.title('Glass Temperature Profile along Lehr (Zones A, B1, B2)')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# step by step cooling and change the distance domain, where zone 1 first, then zone 2 and then zone 3
# apply the actial values from the 
# try to apply different cooling rate in the zones instead of the changing htc and ambient temperatures (multiply by cooling rate=v/t)
# check visco_model.py to correct the error of residual stresses
# how to apply cooling rates in heat equation
# try to change cooling rate import and/or change htc values


# try kth and cp not expressions, try as cooling rate in heat equation
# print the values of k_T and cp_T to know their values are write or wrong

#check the parameters again and again, then apply 3d with the proper dimensions


#check the code and the values properly from the article
#do taylor
# delete stiffnes matrix
# 
# length best spannung crazy
# check some stress calculation
# problem in matching htc numbers with different values of t_ambient
# 
# thickness and cooling plays important role
# self._taylor_exponential(functions,functions_previous,
# continue adapting updates and previous and current
# adjust cooling process
# delete the previous in general
#make the refernce grenzbach grahs and cooling rate controlling by htc
#try to change the htc over zones and print "cooling rates" for each end zone to detect
#put all the variables in main.py all like geometry and others
# reconrd thickness
# adapt htc functions and comment the unneeded lines
# '''
