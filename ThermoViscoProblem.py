from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.nls import petsc
from dolfinx.io import gmshio
from dolfinx.fem import (FunctionSpace, Function, Constant, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import NonlinearProblem
from ufl import (TestFunction,TrialFunction, FiniteElement, TensorElement,
                 VectorElement, grad, inner,
                 CellDiameter, avg, jump,
                 Measure, SpatialCoordinate, FacetNormal,inner, tr, sym, Identity, dot, nabla_div, ds, dx)#
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import numpy as np
from basix.ufl import element
from math import ceil
from time import time
from ViscoelasticModel import ViscoelasticModel
from ThermalModel import ThermalModel
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, Expression


class ThermoViscoProblem:
    def __init__(self,mesh_path: str, problem_dim: int, time: tuple,
                 dt: float, config: dict, model_parameters: dict,
                 jit_options: (dict|None) = None ) -> None:
        self.dim = problem_dim
        self.mesh, self.cell_tags, self.facet_tags = gmshio.read_from_msh(
            mesh_path, MPI.COMM_WORLD, 0, gdim=problem_dim)
        self.__init_boundary_markers()
        self.dt = dt
        # The time domain
        self.time = time
        # The current timestep
        self.t = self.time[0]
        self.n_steps = ceil((self.time[1] - self.time[0]) / self.dt)

        self.material_model = ViscoelasticModel(mesh=self.mesh, 
                                            model_parameters=model_parameters)
        self.physical_model = ThermalModel(
            mesh=self.mesh,
            model_parameters=model_parameters
        )

        self.__init_function_spaces(config=config)
        self.__init_functions()

        self.material_model._init_expressions(
            functionSpaces=self.functionSpaces,
            functions=self.functions,
            functions_current=self.functions_current,
            functions_previous=self.functions_previous,
            functions_next=self.functions_next,
            dt=self.dt)

        self.jit_options = jit_options

        return
    

    def __init_boundary_markers(self) -> None:
        self.bc_markers = {}
        self.bc_markers["left"]     = self.facet_tags.find(10)
        self.bc_markers["right"]    = self.facet_tags.find(12)
        if self.dim == 2:
            self.bc_markers["top"]      = self.facet_tags.find(11)
            self.bc_markers["bottom"]   = self.facet_tags.find(13)

        return
    

    def __init_function_spaces(self,config: dict) -> None:
        """
        Create all necessary finite element data structures
        
        Args:
            - `config`: dictionary holding the types and degrees
                of each finite element
        """
        # Only CG and DG are supported
        assert all(var["element"] in ['CG','DG']
            for var in config.values()), "Only CG and DG elements are supported"
        
        self.finiteElements = {}
        self.functionSpaces = {}

        # Temperature
        self.finiteElements["T"] = FiniteElement(config["T"]["element"],
                                  self.mesh.ufl_cell(),
                                  config["T"]["degree"])
        self.functionSpaces["T"] = FunctionSpace(mesh=self.mesh,element=self.finiteElements["T"])
        # Partial fictive temperature is summarized in a vector (6 x T)
        self.finiteElements["Tf_partial"] = VectorElement(config["T"]["element"],
                                    self.mesh.ufl_cell(),
                                    degree=config["T"]["degree"],
                                    dim=self.material_model.tableau_size)
        self.functionSpaces["Tf_partial"] = FunctionSpace(self.mesh,self.finiteElements["Tf_partial"])

        # Stress / strain
        self.finiteElements["sigma"] = TensorElement(config["sigma"]["element"],
                                      self.mesh.ufl_cell(),
                                      config["sigma"]["degree"],
                                      shape=(self.dim,self.dim))
        self.functionSpaces["sigma"] = FunctionSpace(mesh=self.mesh, element=self.finiteElements["sigma"])
        
        # Partial stresses are summarized in a 3-dim tensor (6 x dim x dim)
        # c.f. https://fenicsproject.discourse.group/t/ways-to-define-vector-elements/12642/8 
        self.finiteElements["sigma_partial"] = element(config["sigma"]["element"],
                                 self.mesh.basix_cell(),
                                 degree=config["sigma"]["degree"],
                                 shape=(self.material_model.tableau_size,self.dim,self.dim))
        self.functionSpaces["sigma_partial"] = FunctionSpace(self.mesh,self.finiteElements["sigma_partial"])
        
        # Displacements
        self.finiteElements["U"] = VectorElement(config["U"]["element"],
                                    self.mesh.ufl_cell(),
                                    degree=config["U"]["degree"])
        self.functionSpaces["U"] = FunctionSpace(mesh=self.mesh, element=self.finiteElements["U"])
        
        return
    

    def __init_functions(self) -> None:
        """
        Create all necessary FEniCS functions to model the viscoelastic
        problem, c.f. Nielsen et al., Fig. 5
        """
        # Time-dependent functions
        self.functions_previous = {}
        self.functions_current = {}
        # Non time-dependent functions
        self.functions = {}
        # Only for extrapolation in the viscoelastic model
        self.functions_next = {}

        # Temperature
        # Heat equation with radiation BC is nonlinear, thus
        # there is no TrialFunction
        # BUG: For C++ forms to compile directly, function names must
        # conform to c++ standards, i.e. no whitespace
        self.functions_current["T"] = Function(self.functionSpaces["T"], name="Temperature")
        # For output
        self.functions_previous["T"] = Function(self.functionSpaces["T"]) # previous time step
        self.functions_next["T"] = Function(self.functionSpaces["T"]) 
        # For building the weak form
        self.v = TestFunction(self.functionSpaces["T"])

        # Partial fictive temperatures
        # Looping through a list causes problems, likely during
        # JIT and AD
        self.functions_previous["Tf_partial"] = Function(self.functionSpaces["Tf_partial"])
        self.functions_current["Tf_partial"] = Function(self.functionSpaces["Tf_partial"], name="Fictive_temperature")

        # Fictive temperature
        self.functions_previous["Tf"] = Function(self.functionSpaces["T"])
        self.functions_current["Tf"] = Function(self.functionSpaces["T"], name="Fictive_Temperature") 

        # Shift function
        self.functions["phi"] = Function(self.functionSpaces["T"], name="Shift_function")
        self.functions["phi"] = Function(self.functionSpaces["T"])
        self.functions_next["phi"] = Function(self.functionSpaces["T"])
        # Shifted time
        self.functions["xi"] = Function(self.functionSpaces["T"], name="Shifted_time")

        # Strains
        self.functions["thermal_strain"] = Function(self.functionSpaces["sigma"], name="thermal_strain")
        self.functions["total_strain"] = Function(self.functionSpaces["sigma"], name="total_strain")
        self.functions["deviatoric_strain"] = Function(self.functionSpaces["sigma"], name="deviatoric_strain")
    
        # Stresses
        self.functions["ds_partial"] = Function(self.functionSpaces["sigma_partial"],
                                   name="Deviatoric_stress_increment")
        self.functions["dsigma_partial"] = Function(self.functionSpaces["sigma_partial"],
                                       name="Hydrostatic_stress_increment")

        self.functions_current["s_tilde_partial"] = Function(self.functionSpaces["sigma_partial"])
        self.functions_next["s_tilde_partial"] = Function(self.functionSpaces["sigma_partial"])

        self.functions_current["sigma_tilde_partial"] = Function(self.functionSpaces["sigma_partial"])
        self.functions_next["sigma_tilde_partial"] = Function(self.functionSpaces["sigma_partial"])

        self.functions_current["s_partial"] = Function(self.functionSpaces["sigma_partial"])
        self.functions_next["s_partial"] = Function(self.functionSpaces["sigma_partial"])

        self.functions_current["sigma_partial"] = Function(self.functionSpaces["sigma_partial"])
        self.functions_next["sigma_partial"] = Function(self.functionSpaces["sigma_partial"])
        #self.functions["f2G"] = Function(self.functionSpaces["f2G"])
        #self.functions["f2K"] = Function(self.functionSpaces["f2K"])

        self.functions_next["sigma"] = Function(self.functionSpaces["sigma"], name="Stress_tensor")
        
        self.functions["U"] = Function(self.functionSpaces["U"], name="Displacement")
        self.u_trial = TrialFunction(self.functionSpaces["U"])
        self.v_test = TestFunction(self.functionSpaces["U"])
        
        self.functions["elastic_strain"] = Function(self.functionSpaces["sigma"], name="Mechanical_strain")

        return
    

    def setup(self, dirichlet_bc: bool = True,
              outfile_T: str = "visco",
              outfile_sigma: str = "stresses") -> None:
        self._set_initial_condition(temp_value=self.material_model.T_init)
        if dirichlet_bc:
            self._set_dirichlet_bc()
        self._write_initial_output(t=self.t)
        self._setup_weak_form_T()
        self._setup_solver_T()
        self._setup_weak_form_u()
        self._setup_solver_u()


    def _set_initial_condition(self, temp_value: float) -> None:
       self.__set_IC_T(temp_value) 
       self.__set_IC_Tf()
       self.__set_IC_Tf_partial()
    

    def __set_IC_T(self, temp_value: float) -> None:
        x = SpatialCoordinate(self.mesh)
        def temp_init(x):
            values = np.full(x.shape[1], temp_value, dtype = ScalarType) 
            return values
        self.functions_previous["T"].interpolate(temp_init)
        self.functions_current["T"].interpolate(temp_init)

        return
    

    def __set_IC_Tf(self) -> None:
        """
        Set the initial condition for fictive temperature.
        For t0, Tf = T (c.f. Nielsen et al., eq. 27)
        """
        self.functions_previous["Tf"].x.array[:] = self.functions_previous["T"].x.array[:]
        self.functions_current["Tf"].x.array[:] = self.functions_current["T"].x.array[:]

        return
    

    def __set_IC_Tf_partial(self) -> None:
        """
        Set the initial condition for the partial fictive temperature
        values.
        For t0, Tf(n) = T (c.f. Nielsen et al., eq. 27)
        """
        temp_value = self.functions_current["T"].x.array[0]
        dim = self.material_model.tableau_size
        def Tf_init(x):
            values = np.full((dim,x.shape[1]), temp_value, dtype = ScalarType) 
            return values

        self.functions_previous["Tf_partial"].interpolate(Tf_init)
        self.functions_current["Tf_partial"].interpolate(Tf_init)

        return
    


    def _write_initial_output(self,t: float = 0.0) -> None:
        self.vtx_files = [
            # Temperature
            io.VTXWriter(self.mesh.comm,"output/T.bp",
                         [self.functions_current["T"]],engine="BP4"),
            # Shift function
            io.VTXWriter(self.mesh.comm,"output/phi.bp",
                         [self.functions["phi"]],engine="BP4"),
            # Fictive temperature
            io.VTXWriter(self.mesh.comm,"output/Tf.bp",
                         [self.functions_current["Tf"]],engine="BP4"),
            # BUG: VTXWriter doesn't support mixed elements
            #io.VTXWriter(self.mesh.comm,"output/Tf_partial.bp",
            #             [self.functions_current["Tf_partial"]],engine="BP4"),
            # thermal strain
            io.VTXWriter(self.mesh.comm,"output/eth.bp",
                         [self.functions["thermal_strain"]],engine="BP4"),
            # Shifted time
            io.VTXWriter(self.mesh.comm,"output/xi.bp",
                         [self.functions["xi"]],engine="BP4"),
            # Displacements
            io.VTXWriter(self.mesh.comm,"output/u.bp",
                         [self.functions["U"]],engine="BP4"),

        ]
        
        for file in self.vtx_files:
            file.write(t)

        # Stresses
        # BUG: VTXWriter doesn't support TensorElement
        self.outfile_sigma = io.XDMFFile(self.mesh.comm, 
                                         "output/sigma.xdmf", "w")
        self.outfile_sigma.write_mesh(self.mesh)
        self.outfile_sigma.write_function(self.functions_next["sigma"], t)


        return

    # Heat diffusion equation #
    
    def _setup_weak_form_T(self) -> None:
        
        #ds = Measure("exterior_facet",domain=self.mesh)
        #dx = Measure("dx",domain=self.mesh)

        element_type = self.finiteElements["T"].family()

        alpha = self.physical_model.alpha
        f = self.physical_model.f
        sigma = self.physical_model.sigma
        epsilon = self.physical_model.epsilon
        T_ambient = self.physical_model.T_ambient
        htc = self.physical_model.htc
        
        self.F = (
            # Mass Matrix
            (self.functions_current["T"] - self.functions_previous["T"]) * self.v * dx
            + self.dt * (
            # Laplacian
            + alpha * inner(grad(self.functions_current["T"]),grad(self.v)) * dx
            # Right hand side
            - f * self.v * dx
            # Radiation
            + 0.001 * (sigma * epsilon) * (self.functions_current["T"]**4 - T_ambient**4) * self.v * ds
            # Convection
            + 0.001 * htc * (self.functions_current["T"] - T_ambient) * self.v * ds
            )
        )

        if element_type == 'Discontinuous Lagrange':
            # (SIP)DG specifics
            dS = Measure("interior_facet",domain=self.mesh)
            n = FacetNormal(self.mesh)
            # penalty parameter to enforce continuity
            penalty = Constant(self.mesh,ScalarType(5.0))
            h = CellDiameter(self.mesh)

            # DG Part of the weak form: additional surface integrals over
            # interior facets
            self.F += self.dt * alpha('+')*(
                # p/h * <[[v]],[[T]]>
                (penalty('+')/h('+')) * inner(jump(self.v,n),jump(self.functions_current["T"],n)) * dS
                # - <{∇v},[[T·n]]>
                - inner(avg(grad(self.v)), jump(self.functions_current["T"], n))*dS
                # - <{v·n},[[∇T]]>
                - inner(jump(self.v, n), avg(grad(self.functions_current["T"])))*dS
            )

        return
    

    def _setup_solver_T(self) -> None:
        self.prob = NonlinearProblem(F=self.F,u=self.functions_current["T"],
                                               jit_options=self.jit_options)

        self.solver = petsc.NewtonSolver(self.mesh.comm, self.prob)
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-12
        self.solver.report = True

        self.ksp = self.solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = self.ksp.getOptionsPrefix()
        # Linear system produced by heat equation is SPD, thus we can use CG
        opts[f"{option_prefix}ksp_type"] = "cg"
        opts[f"{option_prefix}pc_type"] = "gamg"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        self.ksp.setFromOptions()
        
    # linear elasticity equation #
    
    def _set_dirichlet_bc(self) -> None:
          
        facet_dim = self.mesh.topology.dim-1
        
        bottom_bc = locate_dofs_topological(V=self.functionSpaces["U"], entity_dim=facet_dim, entities=self.bc_markers["bottom"])
        
        
        self.bc = [
                   fem.dirichletbc(ScalarType([0.,0.]), bottom_bc, self.functionSpaces["U"])
                   ]
    
    def _setup_weak_form_u(self) -> None:
        
        #ds = Measure("exterior_facet",domain=self.mesh)
        #dx = Measure("dx",domain=self.mesh)
        
        self.ss = Constant(self.mesh,default_scalar_type((0.0,0.0)))          # Body force
        self.traction = Constant(self.mesh,default_scalar_type((0.0,0.0)))      # traction force no matter

        self.a = inner(self.material_model.elastic_sigma(self.u_trial), self.material_model.elastic_epsilon(self.v_test)) * dx 
        self.L = dot(self.ss, self.v_test) * dx + dot(self.traction,self.v_test) * ds
        
        
    def _setup_solver_u(self) -> None:
    
        self.problem = fem.petsc.LinearProblem(self.a, self.L, u=self.functions["U"], bcs=self.bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        
    def _update_values(self,current: Function,previous: Function) -> None:
        # Update ghost values across processes, relevant for MPI computations
        current.x.scatter_forward()
        # assign values
        previous.x.array[:] = current.x.array[:]
        return
    

    def _write_output(self) -> None:
        for file in self.vtx_files:
            file.write(t=self.t)
        
        self.outfile_sigma.write_function(self.functions_next["sigma"], self.t)

        return
    

    def solve_timestep(self,t) -> None:
        print(f"t={self.t}")
        self._solve_T()
        self._solve_u()
        self._solve_Tf()
        self._solve_strains()
        self._solve_shifted_time()
        self._solve_stress()
        self._write_output()
        
        # For some computations, functions_previous["T"] and functions_previous["displacement"] is needed
        # thus, we update only at the end of each timestep
        self._update_values(current=self.functions_current["T"],
                            previous=self.functions_previous["T"])
        #self._update_values(current=self.functions_current["displacement"],previous=self.functions_previous["displacement"])
        
        return
    

    def _solve_T(self) -> None:
        """
        Solve the heat equation for each time step.
        Update values and write current values to file.
        """
        _, converged = self.solver.solve(self.functions_current["T"])
        assert(converged)
        return
    
    def _solve_u(self) -> None:
        """
        Solve the linear elasticity equation for each time step.
        Update values and write current values to file.
        """
        self.problem.solve()

        return
    
    def _solve_Tf(self) -> None:
        """
        Calculate the current fictive temperature,
        c.f. Nielsen et al., Fig. 5.:

        Steps:
        - compute shift function `phi`
        - compute current partial fictive temperatures
        - compute current fictive temperature
        """
        self.__update_shift_function()
        self.__update_partial_fictive_temperature()
        self.__update_fictive_temperature()
        
        return 
    
    def _solve_strains(self) -> None:
        """
        Calculate the current strains,
        c.f. Nielsen et al., Fig. 5.:

        Steps:
        - compute thermal strain
        - compute total strain (normally from thermal and mechanical strain)
        - compute deviatoric strain
        """ 
        self.__update_thermal_strain()
        self.__update_total_strain()
        self.__update_deviatoric_strain()
        self.__update_elastic_strain()

        return
    

    def _solve_shifted_time(self) -> None:
        """
        Calculate the shifted Time,
        c.f. Nielsen et al., Eq. 19
        """
        self.__update_T_next()
        self.__update_phi()
        self.__update_shifted_time()

        return
    
    
    def _solve_stress(self) -> None:
        """
        Calculate the current stresses,
        c.f. Nielsen et al., Fig. 5.:

        Steps:
        - compute deviatoric stress
        - compute hydrostatic stress
        - compute total stress
        """
        self.__update_deviatoric_stress()
        self.__update_hydrostatic_stress()
        self.__update_total_stress()
        

        return


    def __update_shift_function(self) -> None:
        self.functions["phi"].interpolate(self.material_model.expressions["phi"])

        return

    
    def __update_partial_fictive_temperature(self) -> None:
        """
        Update the partial fictive temperature for the current timestep.
        C.f. Nielsen et al., Eq. 24
        """        
        self.functions_current["Tf_partial"].interpolate(
            self.material_model.expressions["Tf_partial"]
        )
        self._update_values(current=self.functions_current["Tf_partial"],
                            previous=self.functions_previous["Tf_partial"])

        return


    def __update_fictive_temperature(self) -> None:
        """
        Update the fictive temperature for the current timestep.
        C.f. Nielsen et al., Eq. 26
        """
        self.functions_current["Tf"].interpolate(self.material_model.expressions["Tf"])
        self._update_values(current=self.functions_current["Tf"],
                            previous=self.functions_previous["Tf"])

        return
    
    
    def __update_thermal_strain(self) -> None:
        """
        Update the thermal strain for the current timestep.
        c.f. Nielsen et al., Eq. 9
        """
        self.functions["thermal_strain"].interpolate(
            self.material_model.expressions["thermal_strain"]
        )

        return
    

    def __update_total_strain(self) -> None:
        """
        Update the total strain for the current timestep.
        c.f. Nielsen et al., Eq. 28
        """
        self.functions["total_strain"].interpolate(
            self.material_model.expressions["total_strain"]
        )

        return
    

    def __update_deviatoric_strain(self) -> None:
        """
        Update the total strain for the current timestep.
        c.f. Nielsen et al., Eq. 28
        """
        self.functions["deviatoric_strain"].interpolate(
            self.material_model.expressions["deviatoric_strain"]
        )

        return

    def __update_T_next(self) -> None:

        self.functions_next["T"].interpolate(
            self.material_model.expressions["T_next"]
        )

        return
    
    def __update_phi(self) -> None:
        self.functions["phi"].interpolate(self.material_model.expressions["phi"])
        self.functions_next["phi"].interpolate(self.material_model.expressions["phi_next"]
        )

        return

    
    def __update_shifted_time(self) -> None:
        self.functions["xi"].interpolate(
            self.material_model.expressions["xi"]
        )
        return


    def __update_deviatoric_stress(self) -> None:
        self.functions["ds_partial"].interpolate(
            self.material_model.expressions["ds_partial"]
        )
        self.functions_next["s_tilde_partial"].interpolate(
            self.material_model.expressions["s_tilde_partial_next"]
        )
        self.functions_next["s_partial"].interpolate(
            self.material_model.expressions["s_partial_next"]
        )

        self._update_values(current=self.functions_next["s_tilde_partial"],
                            previous=self.functions_current["s_tilde_partial"])
        self._update_values(current=self.functions_next["s_partial"],
                            previous=self.functions_current["s_partial"])

        return
    
    
    def __update_hydrostatic_stress(self) -> None:
        self.functions["dsigma_partial"].interpolate(
            self.material_model.expressions["dsigma_partial"]
        )
        self.functions_next["sigma_tilde_partial"].interpolate(
            self.material_model.expressions["sigma_tilde_partial_next"]
        )
        self.functions_next["sigma_partial"].interpolate(
            self.material_model.expressions["sigma_partial_next"]
        )

        self._update_values(
            current=self.functions_next["sigma_tilde_partial"],
            previous=self.functions_current["sigma_tilde_partial"]
        )
        self._update_values(
            current=self.functions_next["sigma_partial"],
            previous=self.functions_current["sigma_partial"]
        )

        return
    

    def __update_total_stress(self) -> None:
        self.functions_next["sigma"].interpolate(
            self.material_model.expressions["sigma_next"]
        )
        return
    
    def __update_elastic_strain(self) -> None:
        self.functions["elastic_strain"].interpolate(
            self.material_model.expressions["elastic_strain"]
        )

        return


    def solve(self) -> None:
        if self.mesh.comm.rank == 0:
            print("Starting solve")
            t_start = time()
        for _ in range(self.n_steps):
            self.t += self.dt
            self.solve_timestep(t=self.t)
        if self.mesh.comm.rank == 0:
            t_end = time()
            print(f"Solve finished in {t_end - t_start} seconds.")

        self._finalize()

        return


    def _finalize(self) -> None:
        for file in self.vtx_files:
            file.close()
        
        self.outfile_sigma.close()

        return