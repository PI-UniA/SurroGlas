from geometry import create_mesh
from ThermoViscoProblem import ThermoViscoProblem

# Enable full compiler optimizations for generated
# C++ code
jit_options = {
            "cffi_extra_compile_args": ["-O3", "-march=native"]
        }

# Time domain
t_start = 0.0
t_end = 50.0
time = (0.0, 50.0)

dt = 0.1
t = t_start
problem_dim = 2

mesh_path = f"mesh{problem_dim}d.msh"


create_new_mesh = True

if create_new_mesh:
    create_mesh(path=mesh_path,dim=problem_dim)

fe_config = {
    "T":        {"element": "DG", "degree": 1},
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
    "T_ambient": 600.0,
    # Initial temperature
    "T_0": 923.1,
    "alpha": 1.0,
    # Convective heat transfer coefficient (Controlling cooling rate)
    "htc": 280.0,
    # Material density
    "rho": 2500.0,
    # Specific heat capacity
    "cp": 1433.0,
    # Heat conduction coefficient
    "k": 1.0,
    "H": 457.05e3,
    "Tb": 869.0e0,
    "Rg": 8.314,
    "alpha_solid": 9.10e-6,
    "alpha_liquid": 25.10e-6,
    "Tf_init": 923.1,
    "lambda_": 1.25,
    "mu": 1.0,
}

model = ThermoViscoProblem(mesh_path=mesh_path,problem_dim=problem_dim,
                           config=fe_config,time=time,dt=dt,model_parameters=model_params,
                           jit_options=jit_options)

model.setup(dirichlet_bc=True)
model.solve()

