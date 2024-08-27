from ufl import Identity
from dolfinx.mesh import Mesh
from dolfinx.fem import Constant
from petsc4py.PETSc import ScalarType

class ThermalModel:
    def __init__(self,mesh: Mesh, model_parameters: dict) -> None:
            """
            Create all parameters of the viscoelastic model
            as FEniCS constants    
            Args:
                - `model_parameters`: dict holding the parameter values
            """
            # Right hand side
            self.f = Constant(mesh,ScalarType(model_parameters["f"]))
            self.epsilon = Constant(mesh,ScalarType(model_parameters["epsilon"])) # view factor
            self.sigma = Constant(mesh,ScalarType(model_parameters["sigma"])) # Stefan Boltzmann constant - W/m^2K^4
            self.alpha = Constant(mesh,ScalarType(model_parameters["alpha"])) 
            self.htc = Constant(mesh,ScalarType(model_parameters["htc"]))     # heat convective coefficent - W/(m^2*K) 
            self.rho = Constant(mesh, ScalarType(model_parameters["rho"]))    # density kg/m^3
            self.cp = Constant(mesh,ScalarType(model_parameters["cp"]))       # specific heat - J/(kg*K)
            self.k = Constant(mesh,ScalarType(model_parameters["k"]))         # thermal conductivity - W/(m*K) 
            # Ambient temperature [K]
            self.T_ambient = Constant(mesh, model_parameters["T_ambient"])
            self.T_0 = Constant(mesh, ScalarType(model_parameters["T_0"]))     # initial temperature [K]

            return