from geometry import create_mesh
from ThermoViscoProblem import ThermoViscoProblem
import matplotlib.pyplot as plt
import numpy as np
from AnalyticalSoln import AnalyticalSoln
from OutgoingDto import OutgoingDto
import logging
import numpy as np

# Logging
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logger.propagate = False
formatter = logging.Formatter(
    "{asctime} - {levelname} - {filename} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_handler.setLevel("INFO")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)




# Enable full compiler optimizations for generated
# C++ code
jit_options = {
            "cffi_extra_compile_args": ["-O3", "-march=native"]
        }

# Time domain (whole time domain or for each zone)
t_start = 0.0
t_end = 50.0
time = (t_start, t_end)

dt = 0.1
t = t_start

# Problem dimensions (1D, 2D, or 3D)
problem_dim = 1

# Name of the glass zone 
Zone_name = "all"

mesh_path = f"mesh{problem_dim}d.msh"

# Create new mesh for simulation
create_new_mesh = True

# Create VTX Files for visualization in Paraview
create_vtx_files = True

try:
    if create_new_mesh:
        create_mesh(path=mesh_path,dim=problem_dim, name=Zone_name, t_start=t_start, t_end=t_end)
        logger.info("Mesh created")

    fe_config = {
        "T":        {"element": "CG", "degree": 1},
        "sigma":    {"element": "CG", "degree": 1},
        "U":        {"element": "CG", "degree": 1}
    }

    model_params = {
        # Volumetric heat dissipation
        "f": 0.0,
        # Radiative heat emissivity
        "epsilon": 0.93,
        # Boltzmann constant
        "sigma": 5.670e-8,
        # Ambient temperature
        "T_ambient": 293.15,
        # Initial temperature
        "T_0": 923.15,
        "alpha": 15.0,    #ideal for 1d 2, for 2d 0.2
        # Convective heat transfer coefficient (Controlling cooling rate)
        "htc": 280.1,
        # Material density
        "rho": 2500.0,
        # Specific heat capacity
        "cp": 1433.0,
        # Heat conduction coefficient
        "k": 1.0,
        "Hv": 457.05e3,
        "H": 627.8e3,
        "Tb": 869.0,
        "Rg": 8.314,
        "alpha_solid": 9.10e-6,
        "alpha_liquid": 25.10e-6,
        "Tf_init": 923.1,
        "lambda_": 1.25,
        "mu": 1.0,
        "Young's_modulus": 70.0e6, # from GP into MPa
        "Possion_ratio": 0.22,
    }

    analytical_constants = {
            "a":    0.2957,
            "c":    1.676e3,
            "Tb":   779.9,
            "E0":   70e9,
            "b":    6.937,
            "H":    22.380e3,
            "k":    -1.231e8,
            "lambda_":   0.7012,
    }

    model = ThermoViscoProblem(mesh_path=mesh_path,problem_dim=problem_dim,
                            config=fe_config,time=time,dt=dt,model_parameters=model_params, analy_parameters=analytical_constants,
                            jit_options=jit_options)

    model.setup(dirichlet_bc_mech=True,create_vtx_files=create_vtx_files)
    dto = model.solve()
    result = dto.to_json()
    
    logger.info(f"Simulation executed. - Execution Time: {np.round(model.execution_time,6)}s")
    logger.info(f"Number of elements in OutgoingDto: {dto.num_elements()}")
    logger.info("Code executed")

except Exception as e:
    logger.error(e, exc_info=True)
    result = OutgoingDto().to_json()


#t_ = np.logspace(start=-4, stop=2, num=500)
t_ = np.linspace(start=0.0, stop=50, num=500)

#Variables of analytical equations in arrays over time loop

T_ = [AnalyticalSoln .T(t_i, constants=analytical_constants) for t_i in t_]
phi_ = [AnalyticalSoln.phi(t_i, constants=analytical_constants) for t_i in t_]
E_ = [AnalyticalSoln.E(t_i, constants=analytical_constants) for t_i in t_]
xi_ = [AnalyticalSoln.xi(t_i, constants=analytical_constants) for t_i in t_]
epsilon_ = [AnalyticalSoln.epsilon(t_i, constants=analytical_constants) for t_i in t_]
dedt = [AnalyticalSoln.de(t_i, constants=analytical_constants) for t_i in t_]
sigma_ = [AnalyticalSoln.stress(t_i, constants=analytical_constants) for t_i in t_]
sigma_analytical_ = [AnalyticalSoln.sigma_analytical(t_i, constants=analytical_constants) for t_i in t_]


#fig, axs = plt.subplots(2, 3)
#Nx = 50
#x_vals = np.linspace(0, 50, Nx)
#time_index = 100
#avg_t_sigma_array = np.array(model.avg_t_sigma)
#Stresses over distance
#plt.plot(50, model.avg_t_sigma[10])
#plt.xlabel("x-coordinate")
#plt.ylabel("Stresses")
#plt.legend(["$t={:.0f}$".format(ti) for ti in t[1::50//10]], ncol=2)
#plt.show()

# Temperatures
#plt.subplot(2, 3, 1)
# Use Arial font and set font size globally
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 15

plt.plot(t_, T_, label='Analytical results',color='lightgreen', linestyle='--')
plt.plot(t_, model.avg_T, label='Simulated results', color='darkgreen')
plt.xlabel('Time (s)')
plt.ylabel('Tempertures (K)')
plt.legend()
plt.grid(True)
plt.show()

# Shift functions
#plt.subplot(2, 3, 2)
plt.plot(t_, phi_, label='Analytical results',color='lightgreen', linestyle='--')
plt.plot(t_, model.avg_phi, label='Simulated results', color='darkgreen')
plt.xlabel('Time (s)')
plt.ylabel('Shift function')
plt.legend()
plt.grid(True)
plt.show()

#Scaled times
#plt.subplot(2, 3, 3)
plt.plot(t_, xi_, label='Analytical results',color='lightgreen', linestyle='--')
plt.plot(t_, model.avg_xi, label='Simulated results', color='darkgreen')
plt.xlabel('Time (s)')
plt.ylabel('Scaled time (s)')
plt.legend()
plt.grid(True)
plt.show()

#Strains
#plt.subplot(2, 3, 4)
plt.plot(t_, dedt, label='Analytical results',color='lightgreen', linestyle='--')
plt.plot(t_, model.avg_t_epsilon, label='Simulated results', color='darkgreen')
plt.xlabel('Time (s)')
plt.ylabel('Total strain (-)')
plt.legend()
plt.grid(True)
plt.show()

#Stresses
#plt.subplot(2, 3, 5)
#plt.plot(t_, sigma_analytical_, label='Analytical results', color='r')
plt.plot(t_, model.avg_t_sigma, label='Simulated results', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Stress (MPa)')
plt.legend()
plt.grid(True)
plt.show()

#Stresses over time
#plt.subplot(2, 3, 6)
plt.plot(t_, model.avg_t_sigma_surface, label='Stresses at surface ', color='b')
plt.plot(t_, model.avg_t_sigma_mid, label='Stresses at mid_plane', color='r')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlabel('Time (s)')
plt.ylabel('Stress (MPa)')
plt.legend()
plt.grid(True)
plt.show()







