from dolfinx.mesh import Mesh, MeshTags
from dolfinx.fem import Constant, Expression
from petsc4py.PETSc import ScalarType
from dolfinx import fem, io
import ufl
from ufl import (inner, tr, sym, Identity)
import numpy as np
from math import factorial
from ufl import (TestFunction,TrialFunction, grad, inner,
                 CellDiameter, avg, jump,
                 Measure, SpatialCoordinate, FacetNormal,inner, tr, sym, Identity,dot, nabla_div, ds, dx)#, ds, dx
from AnalyticalSoln import AnalyticalSoln
from dolfinx.fem import (Constant,Function, FunctionSpace, assemble, Expression, assemble_scalar, assemble_matrix, assemble_vector)
from dolfinx.mesh import create_unit_square
from mpi4py import MPI


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
            7.5e+0
            
        ])
        self.lambda_k_n_tableau = Constant(mesh,[
            5.009e-5,
            9.945e-4,
            2.022e-3,
            1.925e-2,
            1.199e-1,
            2.033e+0,
            1.0e+100
              # instead of Inf
        ])

        # Identity tensor
        self.I = Identity(mesh.topology.dim)
        # Intial (fictive) temperture [K]
        self.T_init = Constant(mesh, model_parameters["T_0"])
        # Activation energy [J/mol]
        self.Hv = Constant(mesh, model_parameters["Hv"])
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
        self.G0= Constant(mesh, model_parameters["Young's_modulus"]/(2*(1+ model_parameters["Possion_ratio"])))
        self.K0 = Constant(mesh, model_parameters["Young's_modulus"]/(3*(1- 2*model_parameters["Possion_ratio"])))
        self.K_inf = Constant(mesh, 0.18*(model_parameters["Young's_modulus"]/(3*(1- 2*model_parameters["Possion_ratio"]))))
        self.stress_zero = Constant(mesh, 0.0)
        self.x = SpatialCoordinate(mesh)
        self.young_modulus = Constant(mesh, model_parameters["Young's_modulus"])
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
        self.expressions["phi_v"] = Expression(
            ufl.exp(
            self.Hv / self.Rg * (
                1.0 / self.Tb
                - self.chi / functions_current["T"]
                - (1.0 - self.chi) / functions_previous["Tf"]
            )),
            functionSpaces["T"].element.interpolation_points()
        )       

        # Eq. 24
        self.expressions["Tf_partial"] = Expression(
            ufl.as_vector([(
                self.lambda_m_n_tableau[n] * functions_previous["Tf_partial"][n]
                + functions_current["T"] * dt * functions["phi_v"])
                / (self.lambda_m_n_tableau[n] + dt * functions["phi_v"])
                for n in range(0,self.tableau_size)]
            ),
             functionSpaces["Tf_partial"].element.interpolation_points()
        )

        # Eq. 26
        self.expressions["Tf"] = Expression(
            inner(self.m_n_tableau,functions_current["Tf_partial"]),
            functionSpaces["T"].element.interpolation_points()
        )   
        
        #elastic strain (assuming linear elasticity)
        self.expressions["elastic_strain"] = Expression(
            self.elastic_epsilon(functions["U"]),
            functionSpaces["sigma"].element.interpolation_points()
        )   
        
        # Summation of elastic loading - curve a in fig. 4
        self.expressions["elastic_stress"] = Expression(
            self.elastic_sigma(functions["U"]),
            functionSpaces["sigma"].element.interpolation_points()
        )
        
        # Eq. 9
        self.expressions["thermal_strain"] = Expression( 
            (self.alpha_solid * (functions_current["T"] - functions_previous["T"])
                      + ((self.alpha_liquid - self.alpha_solid) * (functions_current["Tf"] - functions_previous["Tf"]))),
            functionSpaces["T"].element.interpolation_points()
        )
    
        # Eq. 28
        self.expressions["total_strain"] = Expression(
            self.I*((functions["volumetric_strain"]) - (functions["thermal_strain"]))/10**6,
            functionSpaces["T"].element.interpolation_points()
        )
        # Eq. 29 (diagnoals + off-diagonals of the deviatoric strain)
        self.expressions["deviatoric_strain"] = Expression(
            ((functions["total_strain"]) - (self.I * (functions["volumetric_strain"]))) 
            ,
            functionSpaces["sigma"].element.interpolation_points()
        )

        #volumetric strain (assuming linear elasticity) for 1D exx = [0,0], 2D (exx+eyy)= ([0,0]+[1,1])/2, etc, it is also called sum of direct strains
        if self.dim == 1:
            self.expressions["volumetric_strain"] = Expression(
                (self.elastic_epsilon(functions["U"])[0, 0]),
                functionSpaces["T"].element.interpolation_points()
            )
            
        elif self.dim == 2:
            self.expressions["volumetric_strain"] = Expression(
                (1/2)*(self.elastic_epsilon(functions["U"])[0, 0] + self.elastic_epsilon(functions["U"])[1, 1]),
                functionSpaces["T"].element.interpolation_points()
            )
            
        elif self.dim == 3:
            self.expressions["volumetric_strain"] = Expression(
                (1/3)*(self.elastic_epsilon(functions["U"])[0, 0] + self.elastic_epsilon(functions["U"])[1, 1] + self.elastic_epsilon(functions["U"])[2, 2]),
                functionSpaces["T"].element.interpolation_points()
                )


        # No eq. specified, extrapolation step
        # T(i+1) = T(i) + dT = T(i) + (T(i) - T(i-1))
        self.expressions["T_next"] = Expression(
            functions_current["T"] + (functions_current["T"] - functions_previous["T"]),
            functionSpaces["T"].element.interpolation_points()
        )

        # Eq. 5
        self.expressions["phi_current"] = Expression(
            ufl.exp(
                ((self.H * self.chi)/ self.Rg) * ((1/self.Tb) - (1/functions_current["T"]))
            ),
            functionSpaces["T"].element.interpolation_points()
        )     
        self.expressions["phi_next"] = Expression(
            ufl.exp(
                ((self.H * (1-self.chi))/ self.Rg) * ((1/self.Tb) - (1/functions_current["Tf"]))
            ),
            functionSpaces["T"].element.interpolation_points()
        )
        self.expressions["phi_previous"] = Expression(
            ufl.exp(
                (self.H / self.Rg) * ((1/self.Tb) - (1/functions_previous["T"]))
            ),
            functionSpaces["T"].element.interpolation_points()
        )
        
        # Eq. 19/ here scaled time calculation scheme taken from another paper
        self.expressions["xi"] = Expression(functions_previous["xi"] +
            (0.5*(dt)*(functions_next["phi"] + functions_current["phi"])),
            functionSpaces["T"].element.interpolation_points()
        )
        
        # Eq. 15a + 20
        self.expressions["ds_partial"] = Expression(
            ufl.as_tensor([
            2.0 * self.g_n_tableau[n] * ((self.elastic_epsilon(functions["U"])) - (self.I * (functions["volumetric_strain"]))) * (self.lambda_g_n_tableau[n]/(functions["xi"]-functions_previous["xi"])) * (1 - self._taylor_exponential(functions, functions_previous,self.lambda_g_n_tableau[n]))
                for n in range(0,self.tableau_size)]),
            functionSpaces["sigma_partial"].element.interpolation_points()
        )

        # Eq. 15b + 20
        self.expressions["dsigma_partial"] = Expression(
            ufl.as_vector([
            self.k_n_tableau[n] * tr(functions["stiffness_matrix"])*((functions["volumetric_strain"]) - (functions["thermal_strain"])) * (self.lambda_k_n_tableau[n]/(functions["xi"]-functions_previous["xi"])) * (1 - self._taylor_exponential(functions, functions_previous,self.lambda_k_n_tableau[n]))
                for n in range(0,self.tableau_size)]),
            functionSpaces["sigma_partial"].element.interpolation_points()
        )

        # Summation of viscoelastic part curve b in fig. 4
        self.expressions["total_d_partial"] = Expression(
            np.sum([functions["ds_partial"][n,:,:] + (self.I*functions["dsigma_partial"][n]) for n in range(0,self.tableau_size)
        ]),
        functionSpaces["sigma"].element.interpolation_points()
        )
        
        # Eq. 16a
        #_, i, j = ufl.indices(3)
        self.expressions["s_tilde_partial_next"] = Expression(
            ufl.as_tensor([
            functions_current["s_partial"][n,:,:] * ufl.exp(-(functions["xi"]-functions_previous["xi"])/self.lambda_g_n_tableau[n])  for n in range(0,self.tableau_size) 
        ]),
        functionSpaces["sigma_partial"].element.interpolation_points()
        )

        # Eq. 16b
        self.expressions["sigma_tilde_partial_next"] = Expression(
            ufl.as_vector([
                functions_current["sigma_partial"][n] * ufl.exp(-(functions["xi"]-functions_previous["xi"])/self.lambda_k_n_tableau[n])  for n in range(0,self.tableau_size)
            ]),
            functionSpaces["Tf_partial"].element.interpolation_points()
        )
                
        # Summation of structural relaxation part curve d in fig. 4  
        self.expressions["total_tilde_partial"] = Expression(
            np.sum([functions_next["s_tilde_partial"][n,:,:] + self.I*functions_next["sigma_tilde_partial"][n] for
                    n in range(0,self.tableau_size)]),
            functionSpaces["sigma"].element.interpolation_points()
        )
    
        # Eq. 17a
        self.expressions["s_partial_next"] = Expression(ufl.as_tensor([
            (functions["ds_partial"][n,:,:]) + functions_next["s_tilde_partial"][n,:,:] for
                    n in range(0,self.tableau_size)]),
            functionSpaces["sigma_partial"].element.interpolation_points()
        )

        # Eq. 17b
        self.expressions["sigma_partial_next"] = Expression(ufl.as_vector([
            (functions["dsigma_partial"][n]) + functions_next["sigma_tilde_partial"][n] for
                    n in range(0,self.tableau_size)]),
            functionSpaces["Tf_partial"].element.interpolation_points()
        )
        
        # Eq. 18 - Summation of total stresses curve c in fig. 2, including residual stress
        self.expressions["sigma_next"] = Expression(
            np.sum([functions_next["s_partial"][n,:,:] + (self.I*functions_next["sigma_partial"][n]) for n in range(0,self.tableau_size)])/10**6,
            functionSpaces["sigma"].element.interpolation_points()
        )     
        
        # Eq. 22
        self.expressions["A"] = Expression(
            (1/3) * np.sum([g_n *  (lambda_g_n/dt) * (1 - ufl.exp(-dt/lambda_g_n))
                for (lambda_g_n,g_n) in zip(self.lambda_g_n_tableau,self.g_n_tableau)]),
            functionSpaces["T"].element.interpolation_points()
        )
        self.expressions["B"] = Expression(
            np.sum([k_n * (lambda_k_n/dt) * (1 - ufl.exp(-dt/lambda_k_n))
                for (lambda_k_n,k_n) in zip(self.lambda_k_n_tableau,self.k_n_tableau)]) ,
            functionSpaces["T"].element.interpolation_points()
        )
   
        """
        Eq. 21 -  viscoelastic material stiffness matrix - for 2D 
        Dt = [[B+4A, B-2A, 0], [B-2A, B+4A, 0], [0, 0, 3A]]"""
        if self.dim == 1:
            self.expressions["stiffness_matrix"] = Expression(ufl.as_tensor(     
                [[functions["B"] + 4*functions["A"]]]),
                functionSpaces["sigma"].element.interpolation_points()   
            )
        elif self.dim == 2:
            self.expressions["stiffness_matrix"] = Expression(ufl.as_tensor(     
                [
                [functions["B"] + 4*functions["A"], functions["B"] - 2*functions["A"]],
                [functions["B"] - 2*functions["A"], functions["B"] + 4*functions["A"]]]),
                functionSpaces["sigma"].element.interpolation_points()   
            )     
        elif self.dim == 3:
            self.expressions["stiffness_matrix"] = Expression(ufl.as_tensor(     
                [
                [functions["B"] + 4*functions["A"], functions["B"] - 2*functions["A"], 0],
                [functions["B"] - 2*functions["A"], functions["B"] + 4*functions["A"], 0],
                [0, 0, 3*functions["A"]]]),
                functionSpaces["sigma"].element.interpolation_points()   
            )

        return

    def _taylor_exponential(self,functions:dict, functions_previous:dict, lambda_value):
        """
        A taylor series expression to replace an exponential
        in order to avoid singularities,
        c.f. Nielsen et al., Eq. 20.
        """
        return  (
            np.sum([1.0/factorial(k)
            * (- (functions["xi"] - functions_previous["xi"])/lambda_value)**k for k in range(0,3)])
            )

    def elastic_epsilon(self,ua):
        """linear elasticity with isotropy and hook's law
        """
        return sym(grad(ua)) 

    def elastic_sigma(self,ua):
        return self.elastic_epsilon(ua)
