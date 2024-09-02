from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.nls import petsc
from dolfinx.io import gmshio
from dolfinx.geometry import BoundingBoxTree, compute_closest_entity
from dolfinx.fem import (FunctionSpace, Function, Constant, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import NonlinearProblem, apply_lifting, assemble_matrix, assemble_vector, set_bc
from ufl import (TestFunction,TrialFunction, grad, inner,
                 CellDiameter, avg, jump,
                 Measure, SpatialCoordinate, FacetNormal,inner, tr, sym, Identity, dot, nabla_div)#, ds, dx
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
from ufl import conditional, ge, lt
from dolfinx.fem import (Constant,Function, FunctionSpace, assemble, Expression)
import ufl
from ufl import (inner, tr, sym, Identity)
from OutgoingDto import OutgoingDto,Elements
import logging
from geometry import read_from_msh

gmshio.read_from_msh = read_from_msh
logger = logging.getLogger("__main__")

class ThermoViscoProblem:
    def __init__(self,mesh_path: str, problem_dim: int, time: tuple,
                 dt: float, config: dict, model_parameters: dict,analy_parameters:dict,
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
            dt=self.dt, analy_parameters=  analy_parameters, mesh=self.mesh)

        self.jit_options = jit_options
        
        self.create_vtx_files = None
        self.outgoing_dto = OutgoingDto()
        self.execution_time = None


        return
    

    def __init_boundary_markers(self) -> None:
        self.bc_markers = {}
        if self.dim == 1:
            self.bc_markers["left"]     = self.facet_tags.find(10)
            self.bc_markers["right"]    = self.facet_tags.find(12)
        elif self.dim == 2:
            self.bc_markers["left"]     = self.facet_tags.find(10)
            self.bc_markers["top"]      = self.facet_tags.find(11)
            self.bc_markers["right"]    = self.facet_tags.find(12)
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
        self.finiteElements["T"] = element(config["T"]["element"],
                                  self.mesh.basix_cell(),
                                  degree=config["T"]["degree"])
        self.functionSpaces["T"] =fem.functionspace(mesh=self.mesh,element=self.finiteElements["T"])
        # Partial fictive temperature is summarized in a vector (6 x T)
        self.finiteElements["Tf_partial"] = element(config["T"]["element"],
                                    self.mesh.basix_cell(),
                                    degree=config["T"]["degree"],
                                    shape=(self.material_model.tableau_size,))
        self.functionSpaces["Tf_partial"] = fem.functionspace(self.mesh,self.finiteElements["Tf_partial"])

        # Stress / strain
        self.finiteElements["sigma"] = element(config["sigma"]["element"],
                                      self.mesh.basix_cell(),
                                      config["sigma"]["degree"],
                                      shape=(self.dim,self.dim))
        self.functionSpaces["sigma"] = fem.functionspace(mesh=self.mesh, element=self.finiteElements["sigma"])
        
        # Partial stresses are summarized in a 3-dim tensor (6 x dim x dim)
        # c.f. https://fenicsproject.discourse.group/t/ways-to-define-vector-elements/12642/8 
        self.finiteElements["sigma_partial"] = element(config["sigma"]["element"],
                                 self.mesh.basix_cell(),
                                 degree=config["sigma"]["degree"],
                                 shape=(self.material_model.tableau_size,self.dim,self.dim))
        self.functionSpaces["sigma_partial"] = fem.functionspace(self.mesh,self.finiteElements["sigma_partial"])

        # Displacements
        self.finiteElements["U"] = element(config["U"]["element"],
                                    self.mesh.basix_cell(),
                                    degree=config["U"]["degree"], shape=(self.dim,))
        self.functionSpaces["U"] = fem.functionspace(mesh=self.mesh, element=self.finiteElements["U"])

        
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
        self.functions["phi_v"] = Function(self.functionSpaces["T"], name="Shift_function")
        self.functions_previous["phi_v"] = Function(self.functionSpaces["T"], name="Shift_function")
        self.functions_previous["phi"] = Function(self.functionSpaces["T"])
        self.functions_current["phi"] = Function(self.functionSpaces["T"])
        self.functions_next["phi"] = Function(self.functionSpaces["T"])
        # Shifted time
        self.functions["xi"] = Function(self.functionSpaces["T"], name="Shifted_time")
        self.functions_previous["xi"] = Function(self.functionSpaces["T"], name="Shifted_time")

        # Strains
        self.functions["thermal_strain"] = Function(self.functionSpaces["T"], name="thermal_strain")
        self.functions["total_strain"] = Function(self.functionSpaces["sigma"], name="total_strain")
        self.functions["deviatoric_strain"] = Function(self.functionSpaces["sigma"], name="deviatoric_strain")

    
        # Stresses
        self.functions["ds_partial"] = Function(self.functionSpaces["sigma_partial"],
                                   name="Deviatoric_stress_increment")
        self.functions["dsigma_partial"] = Function(self.functionSpaces["Tf_partial"],
                                       name="Hydrostatic_stress_increment")
        self.functions_previous["ds_partial"] = Function(self.functionSpaces["sigma_partial"],
                                   name="Deviatoric_stress_increment")
        self.functions_previous["dsigma_partial"] = Function(self.functionSpaces["Tf_partial"],
                                       name="Hydrostatic_stress_increment")

        self.functions_current["s_tilde_partial"] = Function(self.functionSpaces["sigma_partial"])
        self.functions_next["s_tilde_partial"] = Function(self.functionSpaces["sigma_partial"])

        self.functions_current["sigma_tilde_partial"] = Function(self.functionSpaces["Tf_partial"])
        self.functions_next["sigma_tilde_partial"] = Function(self.functionSpaces["Tf_partial"])

        self.functions_current["s_partial"] = Function(self.functionSpaces["sigma_partial"])
        self.functions_next["s_partial"] = Function(self.functionSpaces["sigma_partial"])

        self.functions_current["sigma_partial"] = Function(self.functionSpaces["Tf_partial"])
        self.functions_next["sigma_partial"] = Function(self.functionSpaces["Tf_partial"])

        self.functions_next["sigma"] = Function(self.functionSpaces["sigma"], name="Stress_tensor")
        self.functions["sigma_xx"] = Function(self.functionSpaces["T"])
        self.functions["equilibrium_force_x"] = Function(self.functionSpaces["T"])
        self.functions_current["sigma"] = Function(self.functionSpaces["sigma"])
        self.functions_next["total_d_partial"] = Function(self.functionSpaces["sigma"], name="Viscoelastic_part")
        self.functions_next["total_tilde_partial"] = Function(self.functionSpaces["sigma"], name="Structural_relaxation")
        self.functions_next["sigma_next_adjusted"] = Function(self.functionSpaces["sigma"])
        
         
        self.functions["U"] = Function(self.functionSpaces["U"], name="Displacement")
        self.functions_previous["U"] = Function(self.functionSpaces["U"])
        self.u_trial = TrialFunction(self.functionSpaces["U"])
        self.v_test = TestFunction(self.functionSpaces["U"])
        
        self.functions["elastic_strain"] = Function(self.functionSpaces["sigma"], name="Mechanical_strain")
        self.functions_previous["elastic_strain"] = Function(self.functionSpaces["sigma"])
        self.functions["elastic_stress"] = Function(self.functionSpaces["sigma"], name="Mechanical_stress")
        self.functions_previous["volumetric_strain"] = Function(self.functionSpaces["T"])
        self.functions["volumetric_strain"] = Function(self.functionSpaces["T"])

        #stiffness matrix parameters
        self.functions["A"] = Function(self.functionSpaces["T"])
        self.functions["B"] = Function(self.functionSpaces["T"])

        return
    

    def setup(self, dirichlet_bc_mech: bool = True,
              outfile_T: str = "visco",
              outfile_sigma: str = "stresses",
              create_vtx_files: bool = True) -> None:
        
        self._set_initial_condition(temp_value=self.material_model.T_init)
        self.create_vtx_files = create_vtx_files
        
        if dirichlet_bc_mech:
            self._set_dirichlet_bc_mech()
        
        if self.create_vtx_files:
            self._write_initial_output(t=self.t)
            logger.debug('Create vtx-Files')
        
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
        self.functions_previous["xi"].x.array[-1] = Constant(self.mesh,ScalarType(0.)) 
        #self.functions_next["sigma"].x.array[-1] = 2 * sigma_center
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
            io.VTXWriter(self.mesh.comm,"output/phi_v.bp",
                         [self.functions["phi_v"]],engine="BP4"),
            io.VTXWriter(self.mesh.comm,"output/phi.bp",
                         [self.functions_current["phi"]],engine="BP4"),
            # Fictive temperature
            io.VTXWriter(self.mesh.comm,"output/Tf.bp",
                         [self.functions_current["Tf"]],engine="BP4"),
            # BUG: VTXWriter doesn't support mixed elements
            #io.VTXWriter(self.mesh.comm,"output/Tf_partial.bp",
            #             [self.functions_current["Tf_partial"]],engine="BP4"),
            # thermal strain
            io.VTXWriter(self.mesh.comm,"output/sigma_xx.bp",
                         [self.functions["sigma_xx"]],engine="BP4"),
            # Shifted time
            io.VTXWriter(self.mesh.comm,"output/xi.bp",
                         [self.functions["xi"]],engine="BP4"),
            # Displacements
            #io.VTXWriter(self.mesh.comm,"output/u.bp",
            #             [self.functions["U"]],engine="BP4"),
            # Viscoelastic part 
            #io.VTXWriter(self.mesh.comm,"output/total_d.bp",
            #             [self.functions_next["total_d_partial"]],engine="BP4"),
            # Structural relaxation part
            #io.VTXWriter(self.mesh.comm,"output/total_tilda.bp",
            #             [self.functions_next["total_tilde_partial"]],engine="BP4"),
            # Elastic loading
            #io.VTXWriter(self.mesh.comm,"output/elastic_strain.bp",
            #             [self.functions["elastic_strain"]],engine="BP4"),

        ]
        
        for file in self.vtx_files:
            file.write(t)

        # Stresses
        # BUG: VTXWriter doesn't support TensorElement
        self.outfile_sigma = io.XDMFFile(self.mesh.comm, 
                                         "output/sigma.xdmf", "w")
        self.outfile_sigma.write_mesh(self.mesh)
        self.outfile_sigma.write_function(self.functions_next["sigma"], t)
        
        self.outfile_elastic_strain= io.XDMFFile(self.mesh.comm, 
                                         "output/elastic_sigma.xdmf", "w")
        self.outfile_elastic_strain.write_mesh(self.mesh)
        self.outfile_elastic_strain.write_function(self.functions["elastic_strain"], t)


        return

    # Heat diffusion equation #
    def _setup_weak_form_T(self) -> None:
        # Define the left boundary (x = 0)
        def left_boundary(x):
            return np.isclose(x[0], -25.0)

        # Define the right boundary (x = L)
        def right_boundary(x):
            return np.isclose(x[0], 25.0)
        
        def top_boundary(x):
            return np.isclose(x[1], 5.0)

        # Define the right boundary (x = L)
        def bottom_boundary(x):
            return np.isclose(x[1], -5.0)

        # Mark boundaries
        facet_indices, facet_markers = [], []
        boundary_facets_left = locate_entities_boundary(self.mesh, dim=0, marker=left_boundary)
        boundary_facets_right = locate_entities_boundary(self.mesh, dim=0, marker=right_boundary)
        boundary_facets_top = locate_entities_boundary(self.mesh, dim=0, marker=top_boundary)
        boundary_facets_bottom = locate_entities_boundary(self.mesh, dim=0, marker=bottom_boundary)

        # Combine facet indices and assign markers
        facet_indices = np.concatenate([boundary_facets_left, boundary_facets_right, boundary_facets_bottom, boundary_facets_top])
        facet_markers = np.concatenate([
            np.full(len(boundary_facets_left), 1, dtype=np.int32),
            np.full(len(boundary_facets_right), 2, dtype=np.int32),
            np.full(len(boundary_facets_bottom), 3, dtype=np.int32),
            np.full(len(boundary_facets_top), 4, dtype=np.int32)
        ])

        # Create MeshTags object for boundary facets (dim=1)
        facet_tags = meshtags(self.mesh, 0, facet_indices, facet_markers)

        # Measure for the left (ds(1)) and right (ds(2)) boundaries
        ds = Measure("exterior_facet", domain=self.mesh)
        dx = Measure("dx",domain=self.mesh)

        element_type = self.finiteElements["T"]

        alpha = self.physical_model.alpha
        f = self.physical_model.f
        sigma = self.physical_model.sigma
        epsilon = self.physical_model.epsilon
        T_ambient = self.physical_model.T_ambient
        htc = self.physical_model.htc
        rho = self.physical_model.rho
        T_0 = self.physical_model.T_0
        #cp = self.physical_model.cp
        #k = self.physical_model.k
        
        " Thermal conductivity - Eq.B1 "
        def kth(T):
            return 0.741 + ((T) * 8.58e-4)
        
        " Heat capacity - Eq.B1 "
        def Cp(T):  
            return conditional(ge(T, 850),
                                1433 + (6.5e-3 * T),
                                893 + (0.4 * T) - (18 * 10e-8 * T**-2))

        "Using implicit euler scheme"
        self.F = (
            # Mass Matrix
            (self.functions_current["T"] - self.functions_previous["T"]) * self.v * dx
            + self.dt * (
            # Laplacian
            + (alpha)*inner(grad(self.functions_current["T"]),grad(self.v)) * dx 
            # Right hand side (heat source)
            - (f) * self.v * dx
            # Radiation
            + 3e-4*((sigma * epsilon)) * (self.functions_current["T"]**4 - T_ambient**4) * self.v * ds
            # Convection
            + 3e-4*(htc) * (self.functions_current["T"] - T_ambient) * self.v * ds
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

    def _set_dirichlet_bc_mech(self) -> None:
          
        facet_dim = self.mesh.topology.dim-1
        
        if self.dim == 1:
            left_bc = locate_dofs_topological(V=self.functionSpaces["U"], entity_dim=facet_dim, entities=self.bc_markers["left"])
            right_bc = locate_dofs_topological(V=self.functionSpaces["U"], entity_dim=facet_dim, entities=self.bc_markers["right"])       
            self.bc = [ 
                        fem.dirichletbc(ScalarType([0.0]), left_bc, self.functionSpaces["U"]),
                        fem.dirichletbc(ScalarType([0.0]), right_bc, self.functionSpaces["U"]),
                    ]
            
        elif self.dim == 2:
            left_bc = locate_dofs_topological(V=self.functionSpaces["U"], entity_dim=facet_dim, entities=self.bc_markers["left"])
            top_bc = locate_dofs_topological(V=self.functionSpaces["U"], entity_dim=facet_dim, entities=self.bc_markers["top"])
            right_bc = locate_dofs_topological(V=self.functionSpaces["U"], entity_dim=facet_dim, entities=self.bc_markers["right"])
            bottom_bc = locate_dofs_topological(V=self.functionSpaces["U"], entity_dim=facet_dim, entities=self.bc_markers["bottom"])
            
            self.bc = [ 
                        fem.dirichletbc(ScalarType([0.0,0.0]), left_bc, self.functionSpaces["U"]),
                        fem.dirichletbc(ScalarType([0.0,0.0]), top_bc, self.functionSpaces["U"]),
                        fem.dirichletbc(ScalarType([0.0,0.0]), right_bc, self.functionSpaces["U"]),
                        fem.dirichletbc(ScalarType([0.0,0.0]), bottom_bc, self.functionSpaces["U"])
                    ]
    
    def _setup_weak_form_u(self) -> None:
        
        ds = Measure("exterior_facet", domain=self.mesh)
        dx = Measure("dx", domain=self.mesh)
        
        x = SpatialCoordinate(self.mesh)
        if self.dim == 1:
            self.ss = ufl.as_vector([(x[0])])  # Using position-dependent body forces
            self.traction = Constant(self.mesh, ScalarType([0.0])) # traction force
            # Weak form: Standard elasticity problem
            self.a = inner(self.material_model.elastic_sigma(self.u_trial), self.material_model.elastic_epsilon(self.v_test)) * dx
            self.L = dot(self.ss, self.v_test) * dx + dot(self.traction,self.v_test) * ds
            
        elif self.dim == 2:
            self.ss = ufl.as_vector((0.0,-x[1]))  # Using position-dependent body forces
            self.traction = Constant(self.mesh, ScalarType([0.0,0.0])) # traction force
            # Weak form: Standard elasticity problem
            self.a = inner(self.material_model.elastic_sigma(self.u_trial), self.material_model.elastic_epsilon(self.v_test)) * dx
            self.L = dot(self.ss, self.v_test) * dx + dot(self.traction,self.v_test) * ds
            
    def _setup_solver_u(self) -> None:
    
        self.u_problem = fem.petsc.LinearProblem(self.a, self.L, u=self.functions["U"], bcs=self.bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
       
        
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
        #self.outfile_e_strain.write_function(self.functions["elastic_strain"], self.t)
        #self.outfile_t_strain.write_function(self.functions["total_strain"], self.t)
        #self.outfile_thermal_strain.write_function(self.functions["thermal_strain"], self.t)
        self.outfile_elastic_strain.write_function(self.functions["elastic_strain"], self.t)

        return
    
    def solve_timestep(self,t) -> None:
        #print(f"t={t}")
        logger.debug(f"t={t}")
        self._solve_T()
        self._solve_Tf()
        self._solve_u()
        self._solve_strains()
        self._solve_shifted_time()
        self._solve_stress()
        self.avg_T.append([np.average(self.functions_current["T"].x.array[:])])
        self.avg_phi_v.append([np.average(self.functions["phi_v"].x.array[:])])
        self.avg_phi.append([np.average(self.functions_current["phi"].x.array[:])])
        self.avg_Tf.append([np.average(self.functions_current["Tf"].x.array[:])])
        self.avg_xi.append([np.average(self.functions["xi"].x.array[:])])
        self.avg_thermal_epsilon.append([np.average(self.functions["thermal_strain"].x.array[:])])
        self.avg_t_epsilon.append([np.average(self.functions["total_strain"].x.array[:])])
        self.avg_t_sigma.append([np.average(self.functions_next["sigma"].x.array[:])])
        
        #self._append_outgoing_dto()
        
        if self.create_vtx_files:
            self._write_output()

        
        
        # For some computations, functions_previous["T"] and functions_previous["displacement"] is needed
        # thus, we update only at the end of each timestep
        self._update_values(current=self.functions_current["T"],previous=self.functions_previous["T"])
        
        #self._update_values(current=self.functions["phi_v"],previous=self.functions_previous["phi_v"])
        
        self._update_values(current=self.functions_current["Tf_partial"],previous=self.functions_previous["Tf_partial"])
        
        self._update_values(current=self.functions_current["Tf"],previous=self.functions_previous["Tf"])
        #self._update_values(current=self.functions["U"],previous=self.functions_previous["U"])
        
        self._update_values(current=self.functions_next["T"],previous=self.functions_current["T"])
        #self._update_values(current=self.functions_current["phi"],previous=self.functions_previous["phi"])
        self._update_values(current=self.functions["xi"], previous=self.functions_previous["xi"])
        
        self._update_values(current=self.functions["ds_partial"],previous=self.functions_previous["ds_partial"])
        self._update_values(current=self.functions["dsigma_partial"],previous=self.functions_previous["dsigma_partial"])
        self._update_values(current=self.functions_next["s_tilde_partial"],previous=self.functions_current["s_tilde_partial"])
        self._update_values(current=self.functions_next["s_partial"],previous=self.functions_current["s_partial"]) 
        self._update_values(current=self.functions_next["sigma_tilde_partial"],previous=self.functions_current["sigma_tilde_partial"])
        self._update_values(current=self.functions_next["sigma_partial"],previous=self.functions_current["sigma_partial"])
        self._update_values(current=self.functions_next["sigma"],previous=self.functions_current["sigma"])

        return
    
    def _to_np_arrays(self,t) -> None:
        """
        Convert the fenics functions into numpy arrays 
        for easier avalibility
        """  
        #number of divisions or elements in the mesh
        nx=50
        ny=10
        
        "stress functions into numpy arrays"
        s_array= self.functions_next["sigma"].x.array[:]
        # 4 tensor components, nodes at x position and y position
        s_array_reshaped = s_array.reshape((4, (nx+1), (ny+1)))
        # Extract sigma_xx as the first component
        self.sigma_xx = s_array_reshaped[0,:,:]
        self.avg_t_sigma_mid.append([np.average(self.sigma_xx[:,5])])
        
        "temperature functions into numpy arrays"
        T_array= self.functions_current["T"].x.array[:]
        T_array_reshaped = T_array.reshape((nx+1), (ny+1))
    
        "position in 3 spatial coordinates"
        position = self.mesh.geometry.x
    
        #extract the mid_plane of x-axis
        #print(self.sigma_xx)
        #print(self.sigma_xx[25,:])
        #print(T_array_reshaped)
        #print(position[:])
        
        
        # the arrangment of functions inside the produced text numpy file
        #self.temperature_time_array.append(f"t={t:.1f}, x={position[:]} T={T_array_reshaped.tolist()} sigma={s_array.tolist()}")
        
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
        self.u_problem.solve()

        return
    
    def _solve_sigma_zero(self) -> None:
        """
        Solve the equilbrium of forces equation for each time step.
        Update values and write current values to file.
        """
        self.s_problem.solve()

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
        self.__update_volumteric_strain()

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
        self.functions["phi_v"].interpolate(self.material_model.expressions["phi_v"])


        return

    
    def __update_partial_fictive_temperature(self) -> None:
        """
        Update the partial fictive temperature for the current timestep.
        C.f. Nielsen et al., Eq. 24
        """        
        self.functions_current["Tf_partial"].interpolate(
            self.material_model.expressions["Tf_partial"]
        )
        
        return


    def __update_fictive_temperature(self) -> None:
        """
        Update the fictive temperature for the current timestep.
        C.f. Nielsen et al., Eq. 26
        """
        self.functions_current["Tf"].interpolate(self.material_model.expressions["Tf"])

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
    
    def __update_elastic_strain(self) -> None:
        self.functions["elastic_strain"].interpolate(
            self.material_model.expressions["elastic_strain"]
        )
        
        self.functions["A"].interpolate(
            self.material_model.expressions["A"]
        )
        self.functions["B"].interpolate(
            self.material_model.expressions["B"]
        )

        return
    
    def __update_volumteric_strain(self) -> None:
        
        self.functions["volumetric_strain"].interpolate(
            self.material_model.expressions["volumetric_strain"]
        )
        

        return

    def __update_T_next(self) -> None:

        self.functions_next["T"].interpolate(self.material_model.expressions["T_next"])

        
        return
    
    def __update_phi(self) -> None:
        self.functions_previous["phi"].interpolate(self.material_model.expressions["phi_previous"])
        self.functions_current["phi"].interpolate(self.material_model.expressions["phi_current"])
        self.functions_next["phi"].interpolate(self.material_model.expressions["phi_next"])

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

        return
    

    def __update_total_stress(self) -> None:
        self.functions_next["sigma"].interpolate(
            self.material_model.expressions["sigma_next"]
        )

        self.functions_next["total_d_partial"].interpolate(
            self.material_model.expressions["total_d_partial"]
        )
        self.functions_next["total_tilde_partial"].interpolate(
            self.material_model.expressions["total_tilde_partial"]
        )
        self.functions["elastic_stress"].interpolate(
            self.material_model.expressions["elastic_stress"]
        )
            

        return
    


    def solve(self) -> None:
        self.avg_T= []
        self.avg_phi_v= []
        self.avg_phi= []
        self.avg_Tf= []
        self.avg_xi= []
        self.avg_thermal_epsilon= []
        self.avg_t_epsilon= []
        self.avg_t_sigma= []
        self.avg_t_sigma_mid= []
        self.temperature_time_array = []
        if self.mesh.comm.rank == 0:
            # print("Starting solve")
            logger.debug("Starting solve")
            t_start = time()
        for _ in range(self.n_steps):
            self.t += self.dt
            self.solve_timestep(t=self.t)
            self._to_np_arrays(t=self.t)
            self._append_outgoing_dto()

        if self.mesh.comm.rank == 0:
            t_end = time()
            # print(f"Solve finished in {t_end - t_start} seconds.")
            logger.debug(f"Solve finished in {t_end - t_start} seconds.")
            self.execution_time = t_end - t_start
        
        # Convert the list to a numpy array
        if self.create_vtx_files:
            self.time_series_array = np.array(self.temperature_time_array)
            with open('temperature_time_series.txt', 'w') as f:
                f.writelines(self.temperature_time_array)
            self._finalize()

        return self.outgoing_dto


    def _finalize(self) -> None:
        for file in self.vtx_files:
            file.close()
        
        self.outfile_sigma.close()
        #self.outfile_e_strain.close()
        #self.outfile_t_strain.close()
        #self.outfile_thermal_strain.close()
        self.outfile_elastic_strain.close()

        return

    def _append_outgoing_dto(self) -> None:
        """ Store the temperature, stress and thickness data in the outgoingDto-"""
        elem = Elements()
        elem.Time = self.t
        elem.Stress = self.functions_next["sigma"].x.array[:].tolist()
        elem.Temperature = self.functions_current["T"].vector.array.tolist()
        elem.Thickness = None
        self.outgoing_dto.append(elem)
        
        return
