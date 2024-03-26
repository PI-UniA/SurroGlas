from dolfinx.mesh import Mesh
from dolfinx.fem import Constant, Expression
from petsc4py.PETSc import ScalarType
import ufl
from ufl import (inner, tr, sym, Identity)
import numpy as np
from math import factorial
from ufl import (TestFunction,TrialFunction, FiniteElement, TensorElement,
                 VectorElement, grad, inner,
                 CellDiameter, avg, jump,
                 Measure, SpatialCoordinate, FacetNormal,inner, tr, sym, Identity,dot, nabla_div)#, ds, dx


class ViscoelasticModel:
    def __init__(self,mesh: Mesh, model_parameters: dict) -> None:
        """
        Define the material model
        """
        # weighting coefficient for temperature and structural energies, c.f. Nielsen et al. eq. 8
        self.chi = 0.5
        self.tableau_size = 6
        self.dim = mesh.topology.dim
        self.m_n_tableau = Constant(mesh,[
            5.523e-2,
            8.205e-2,
            1.215e-1,
            2.286e-1,
            2.860e-1,
            2.265e-1,
        ])
        self.lambda_m_n_tableau = Constant(mesh,[
            5.965e-4,
            1.077e-2,
            1.362e-1,
            1.505e-1,
            6.747e+0,
            2.963e+1,
        ])
        self.g_n_tableau = Constant(mesh,[
            1.585,
            2.354,
            3.486,
            6.558,
            8.205,
            6.498,
        ])
        self.lambda_g_n_tableau = Constant(mesh,[
            6.658e-5,
            1.197e-3,
            1.514e-2,
            1.672e-1,
            7.497e-1,
            3.292e+0
        ])
        self.k_n_tableau = Constant(mesh,[
            7.588e-1,
            7.650e-1,
            9.806e-1,
            7.301e+0,
            1.347e+1,
            1.090e+1,
            #7.5e+0
            
        ])
        self.lambda_k_n_tableau = Constant(mesh,[
            5.009e-5,
            9.945e-4,
            2.022e-3,
            1.925e-2,
            1.199e-1,
            2.033e+0,
            #1.0e+100
              # instead of Inf
        ])

        # Identity tensor
        self.I = Identity(mesh.topology.dim)
        # Intial (fictive) temperture [K]
        self.T_init = Constant(mesh, model_parameters["T_0"])
        # Activation energy [J/mol]
        self.H = Constant(mesh, model_parameters["H"])
        # Universal gas constant [J/(mol K)]
        self.Rg = Constant(mesh, model_parameters["Rg"])
        # Base temperature [K]
        self.Tb = Constant(mesh, model_parameters["Tb"])
        # Solid thermal expansion coefficient [K^-1]
        self.alpha_solid = Constant(mesh, model_parameters["alpha_solid"])
        # Liquid thermal expansion coefficient [K^-1]
        self.alpha_liquid = Constant(mesh, model_parameters["alpha_liquid"])
        # Lame's elasticity parameters
        self.lambda_= Constant(mesh, model_parameters["lambda_"])
        self.mu = Constant(mesh, model_parameters["mu"])

        
        return
    
    def _init_expressions(self,functions: dict, functions_next: dict,
                          functions_current: dict, functions_previous: dict,
                          functionSpaces: dict, dt: float) -> None:
        """
        Initialize the FEniCS expressions that are needed
        to compute the derived quantities defined in Nielsen et al.
        for the viscoelastic tempering model.
        They are individually called each time step by their respective
        self.__update methods and stored as properties in advance, since they only have to be instantiated once.
        """

        self.expressions = {}

        # Eq. 25
        self.expressions["phi"] = Expression(
            ufl.exp(
            self.H / self.Rg * (
                1.0 / self.Tb
                - self.chi / functions_current["T"]
                - (1.0 - self.chi) / functions_previous["Tf"]
            )),
            functionSpaces["T"].element.interpolation_points()
        )       

        # Eq. 24
        self.expressions["Tf_partial"] = Expression(
            ufl.as_vector([(
                self.lambda_m_n_tableau[i] * functions_previous["Tf_partial"][i]
                + functions_current["T"] * dt * functions["phi"])
                / (self.lambda_m_n_tableau[i] + dt * functions["phi"])
                for i in range(0,self.tableau_size)]
            ),
             functionSpaces["Tf_partial"].element.interpolation_points()
        )

        # Eq. 26
        self.expressions["Tf"] = Expression(
            inner(self.m_n_tableau,functions_current["Tf_partial"]),
            functionSpaces["T"].element.interpolation_points()
        )

        # Eq. 9
        self.expressions["thermal_strain"] = Expression(
            self.I * (self.alpha_solid * (functions_current["T"] - functions_previous["T"])
                      + (self.alpha_liquid - self.alpha_solid) 
                      * (functions_current["Tf"] - functions_previous["Tf"])),
            functionSpaces["sigma"].element.interpolation_points()
        )
    
        # Eq. 28
        self.expressions["total_strain"] = Expression(
            functions["elastic_strain"] - functions["thermal_strain"],
            functionSpaces["sigma"].element.interpolation_points()
        )

        # Eq. 29
        self.expressions["deviatoric_strain"] = Expression(
            # TODO: Find out if 1/3 applies for all dims
            functions["total_strain"] - 1/self.dim * self.I * tr(functions["total_strain"]),
            functionSpaces["sigma"].element.interpolation_points()
        )
        
        self.expressions["elastic_strain"] = Expression(
            self.elastic_epsilon(functions["U"]),
            functionSpaces["sigma"].element.interpolation_points()
        )

        # No eq. specified, extrapolation step
        # T(i+1) = T(i) + dT = T(i) + (T(i) - T(i-1))
        self.expressions["T_next"] = Expression(
            functions_current["T"] + (functions_current["T"] - functions_previous["T"]),
            functionSpaces["T"].element.interpolation_points()
        )

        # Eq. 5
        self.expressions["phi"] = Expression(
            ufl.exp(
                self.H / self.Rg * (1/self.Tb - 1/functions_current["T"])
            ),
            functionSpaces["T"].element.interpolation_points()
        )
        self.expressions["phi_next"] = Expression(
            ufl.exp(
                self.H / self.Rg * (1/self.Tb - 1/functions_next["T"])
            ),
            functionSpaces["T"].element.interpolation_points()
        )

        # Eq. 19
        self.expressions["xi"] = Expression(
            dt/2 * (functions_next["phi"] - functions["phi"]),
            functionSpaces["T"].element.interpolation_points()
        )
            
        # User-subroutine for ABAQUS mentioned in Nielsen et al. Dissertation
        if np.any(functions["xi"].x.array > 1.0e-2):
            
            # Eq. 15a + 20
            self.expressions["ds_partial"] = Expression(
                ufl.as_tensor([
                2.0 * g_n * functions["deviatoric_strain"] * (lambda_g_n/functions["xi"]) * (1 - ufl.exp(- functions["xi"]/lambda_g_n))
                    for (lambda_g_n,g_n) in zip(self.lambda_g_n_tableau,self.g_n_tableau)]),
                functionSpaces["sigma_partial"].element.interpolation_points()
            )

            # Eq. 15b + 20
            self.expressions["dsigma_partial"] = Expression(
                ufl.as_tensor([
                    k_n * (tr(functions["total_strain"])*self.I) * (lam_k_n/functions["xi"]) * (1 - ufl.exp(- functions["xi"]/lam_k_n))
                    for (lam_k_n,k_n) in zip(self.lambda_k_n_tableau,self.k_n_tableau)]),
                functionSpaces["sigma_partial"].element.interpolation_points()
            )
            
        else:
                       
            # Eq. 15a + 20
            self.expressions["ds_partial"] = Expression(
                ufl.as_tensor([
                    2.0 * g_n * functions["deviatoric_strain"] * (1 - (1/2*(functions["xi"]/lambda_g_n)) + (1/6 * (functions["xi"]/lambda_g_n)**2))
                    for (lambda_g_n,g_n) in zip(self.lambda_g_n_tableau,self.g_n_tableau)]),
                functionSpaces["sigma_partial"].element.interpolation_points()
            )

            # Eq. 15b + 20
            self.expressions["dsigma_partial"] = Expression(
                ufl.as_tensor([
                    k_n * (tr(functions["total_strain"])*self.I) * (1 - (1/2*(functions["xi"]/lam_k_n)) + (1/6 * (functions["xi"]/lam_k_n)**2))
                    for (lam_k_n,k_n) in zip(self.lambda_k_n_tableau,self.k_n_tableau)]),
                functionSpaces["sigma_partial"].element.interpolation_points()
            )

        # Eq. 16a
        #_, i, j = ufl.indices(3)
        self.expressions["s_tilde_partial_next"] = Expression(ufl.as_tensor([
            functions_current["s_tilde_partial"][n,:,:] * self._taylor_exponential(
            functions,self.lambda_g_n_tableau[n]) for n in range(0,self.tableau_size)
        ]),
        functionSpaces["sigma_partial"].element.interpolation_points()
        )

        # Eq. 16b
        self.expressions["sigma_tilde_partial_next"] = Expression(
            ufl.as_tensor([
                functions_current["sigma_tilde_partial"][n,:,:] * self._taylor_exponential(
                functions,self.lambda_k_n_tableau[n]) for n in range(0,self.tableau_size)
            ]),
            functionSpaces["sigma_partial"].element.interpolation_points()
        )

        # Eq. 17a
        self.expressions["s_partial_next"] = Expression(
            functions["ds_partial"] + functions_next["s_tilde_partial"],
            functionSpaces["sigma_partial"].element.interpolation_points()
        )

        # Eq. 17b
        self.expressions["sigma_partial_next"] = Expression(
            functions["dsigma_partial"] + functions_next["sigma_tilde_partial"],
            functionSpaces["sigma_partial"].element.interpolation_points()
        )

        # Eq. 18
        self.expressions["sigma_next"] = Expression(
            np.sum([functions_next["s_partial"][n,:,:] + functions_next["sigma_partial"][n,:,:] for
                    n in range(0,self.tableau_size)]),
            functionSpaces["sigma"].element.interpolation_points()
        )
        return

    def _taylor_exponential(self,functions: dict, lambda_value):
        """
        A taylor series expression to replace an exponential
        in order to avoid singularities,
        c.f. Nielsen et al., Eq. 20.
        """
        return  (
            np.sum([1.0/factorial(k)
            * (- functions["xi"]/lambda_value)**k for k in range(0,3)])
            )
        
    def elastic_epsilon(self,ua):
        return sym(grad(ua)) 

    def elastic_sigma(self,ua):
        return self.lambda_ * (nabla_div(ua))* self.I + 2 * self.mu * self.elastic_epsilon(ua)
