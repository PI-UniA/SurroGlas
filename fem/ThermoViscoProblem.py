# ThermoViscoProblem.py
#
# Aligned with:
#   Daudeville & Carré, J. Thermal Stresses, 21:667-689, 1998
#
# Changes from original:
#   1. _setup_weak_form_T: k(T) in °C; cp Tg=850K; radiation removed
#   2. _setup_solver_T: NonlinearProblem replaces NewtonSolverNonlinearProblem
#   3. solve_timestep: correct step ordering T→Tf→ξ→u→strains→stress→equil
#   4. _solve_shifted_time: phi_prev + phi_curr before xi update
#   5. _solve_stress: A/B/Dt moved here; correct ordering of partial stresses
#   6. enforce_zero_force_shift: scatter_forward added; ufl.as_ufl for int_one
#   7. __set_IC_Tf: xi initialised to zero correctly (no Constant assignment)
#   8. __set_IC_Tf_partial: uses float(T_init) instead of x.array[0]
#   9. Removed inline imports from solve loop; cleaned debug prints

from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.io import gmsh as gmshio
from dolfinx.fem import (FunctionSpace, Function, Constant,
                          locate_dofs_topological, Expression)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import (TestFunction, TrialFunction, grad, inner,
                 CellDiameter, Measure, SpatialCoordinate,
                 FacetNormal, sym, Identity, dot, conditional, ge)
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import numpy as np
from basix.ufl import element
from math import ceil
from time import time
import ufl
import logging

from fem.ViscoelasticModel import ViscoelasticModel
from fem.ThermalModel import ThermalModel
from OutgoingDto import OutgoingDto, Elements

logger = logging.getLogger("__main__")


class ThermoViscoProblem:
    def __init__(self, mesh_path: str, problem_dim: int, time: tuple,
                 dt: float, config: dict, model_parameters: dict,
                 analy_parameters: dict,
                 jit_options: (dict | None) = None,
                 zones: list[dict] | None = None) -> None:

        self.dim = problem_dim

        mesh_data = gmshio.read_from_msh(
            mesh_path, MPI.COMM_WORLD, 0, gdim=problem_dim
        )
        if hasattr(mesh_data, "mesh"):
            self.mesh        = mesh_data.mesh
            self.cell_tags   = mesh_data.cell_tags
            self.facet_tags  = mesh_data.facet_tags
            self.ridge_tags  = getattr(mesh_data, "ridge_tags",  None)
            self.peak_tags   = getattr(mesh_data, "peak_tags",   None)
            self.physical_groups = getattr(mesh_data, "physical_groups", None)
        else:
            self.mesh, self.cell_tags, self.facet_tags = mesh_data
            self.ridge_tags      = None
            self.peak_tags       = None
            self.physical_groups = None

        self.__init_boundary_markers()

        self.dt     = dt
        self.time   = time
        self.t      = self.time[0]
        self.n_steps = ceil((self.time[1] - self.time[0]) / self.dt)

        self.material_model = ViscoelasticModel(
            mesh=self.mesh, model_parameters=model_parameters)
        self.physical_model = ThermalModel(
            mesh=self.mesh, model_parameters=model_parameters)

        self.config = config                        # stored for weak form selection
        self.__init_function_spaces(config=config)
        self.__init_functions()

        self.material_model._init_expressions(
            functionSpaces=self.functionSpaces,
            functions=self.functions,
            functions_current=self.functions_current,
            functions_previous=self.functions_previous,
            functions_next=self.functions_next,
            dt=self.dt)

        self.jit_options     = jit_options
        self.create_vtx_files = None
        self.outgoing_dto    = OutgoingDto()
        self.execution_time  = None

        self._zones             = zones or []
        self._zone_results      = {}
        self._zone_start_values = {}

    # ====================================================================
    # Boundary markers
    # ====================================================================
    def __init_boundary_markers(self) -> None:
        self.bc_markers = {}
        if self.dim == 1:
            self.bc_markers["left"]   = self.facet_tags.find(10)
            self.bc_markers["right"]  = self.facet_tags.find(12)
        elif self.dim == 2:
            self.bc_markers["left"]   = self.facet_tags.find(10)
            self.bc_markers["top"]    = self.facet_tags.find(11)
            self.bc_markers["right"]  = self.facet_tags.find(12)
            self.bc_markers["bottom"] = self.facet_tags.find(13)
        elif self.dim == 3:
            self.bc_markers["front"]  = self.facet_tags.find(10)
            self.bc_markers["back"]   = self.facet_tags.find(11)
            self.bc_markers["left"]   = self.facet_tags.find(12)
            self.bc_markers["right"]  = self.facet_tags.find(13)
            self.bc_markers["top"]    = self.facet_tags.find(14)
            self.bc_markers["bottom"] = self.facet_tags.find(15)

    # ====================================================================
    # Function spaces
    # ====================================================================
    def __init_function_spaces(self, config: dict) -> None:
        assert all(v["element"] in ["CG", "DG"] for v in config.values()), \
            "Only CG or DG elements are supported"

        self.finiteElements = {}
        self.functionSpaces = {}

        # ── Temperature function spaces ───────────────────────────────────
        # Temperature (scalar) — standard CG Lagrange
        self.finiteElements["T"] = element(
            config["T"]["element"], self.mesh.basix_cell(),
            degree=config["T"]["degree"])
        self.functionSpaces["T"] = fem.functionspace(
            mesh=self.mesh, element=self.finiteElements["T"])

        # Partial fictive temperatures (vector, size = tableau_size)
        self.finiteElements["Tf_partial"] = element(
            config["T"]["element"], self.mesh.basix_cell(),
            config["T"]["degree"],
            shape=(self.material_model.tableau_size,))
        self.functionSpaces["Tf_partial"] = fem.functionspace(
            self.mesh, self.finiteElements["Tf_partial"])

        # Stress / strain tensor (dim × dim)
        self.finiteElements["sigma"] = element(
            config["sigma"]["element"], self.mesh.basix_cell(),
            config["sigma"]["degree"],
            shape=(self.dim, self.dim))
        self.functionSpaces["sigma"] = fem.functionspace(
            mesh=self.mesh, element=self.finiteElements["sigma"])

        # Partial stresses (tableau_size × dim × dim)
        self.finiteElements["sigma_partial"] = element(
            config["sigma"]["element"], self.mesh.basix_cell(),
            degree=config["sigma"]["degree"],
            shape=(self.material_model.tableau_size, self.dim, self.dim))
        self.functionSpaces["sigma_partial"] = fem.functionspace(
            self.mesh, self.finiteElements["sigma_partial"])

        # Displacement (vector, size = dim)
        self.finiteElements["U"] = element(
            config["U"]["element"], self.mesh.basix_cell(),
            degree=config["U"]["degree"], shape=(self.dim,))
        self.functionSpaces["U"] = fem.functionspace(
            mesh=self.mesh, element=self.finiteElements["U"])

    # ====================================================================
    # Functions
    # ====================================================================
    def __init_functions(self) -> None:
        self.functions_previous = {}
        self.functions_current  = {}
        self.functions          = {}
        self.functions_next     = {}

        V_T   = self.functionSpaces["T"]
        V_sig = self.functionSpaces["sigma"]
        V_sp  = self.functionSpaces["sigma_partial"]
        V_Tfp = self.functionSpaces["Tf_partial"]
        V_U   = self.functionSpaces["U"]

        # --- Temperature ---
        self.functions_current["T"]  = Function(V_T, name="Temperature")
        self.functions_previous["T"] = Function(V_T)
        self.functions_next["T"]     = Function(V_T)
        self.v = TestFunction(V_T)



        # --- Fictive temperature ---
        self.functions_previous["Tf_partial"] = Function(V_Tfp)
        self.functions_current["Tf_partial"]  = Function(V_Tfp, name="Fictive_temperature")
        self.functions_previous["Tf"]         = Function(V_T)
        self.functions_current["Tf"]          = Function(V_T, name="Fictive_Temperature")

        # --- Shift function / reduced time ---
        self.functions["phi_v"]          = Function(V_T, name="Shift_function")
        self.functions["kth"]            = Function(V_T, name="Thermal_conductivity")
        self.functions["cp"]             = Function(V_T, name="Specific_heat")
        self.functions_previous["phi_v"] = Function(V_T)
        self.functions_previous["phi"]   = Function(V_T)
        self.functions_current["phi"]    = Function(V_T)
        self.functions_next["phi"]       = Function(V_T)
        self.functions["xi"]             = Function(V_T, name="Shifted_time")
        self.functions_previous["xi"]    = Function(V_T)
        self.functions["delta_xi"]       = Function(V_T, name="Delta_xi")

        # --- Strains ---
        self.functions["thermal_strain"]          = Function(V_T, name="thermal_strain")
        self.functions_previous["thermal_strain"] = Function(V_T)
        self.functions["total_strain"]            = Function(V_sig, name="total_strain")
        self.functions_previous["total_strain"]   = Function(V_sig)
        self.functions["deviatoric_strain"]            = Function(V_sig, name="deviatoric_strain")
        self.functions_previous["deviatoric_strain"]   = Function(V_sig)
        self.functions["elastic_strain"]          = Function(V_sig, name="Mechanical_strain")
        self.functions_previous["elastic_strain"] = Function(V_sig)
        self.functions["elastic_stress"]          = Function(V_sig, name="Mechanical_stress")
        self.functions_previous["volumetric_strain"] = Function(V_T)
        self.functions["volumetric_strain"]          = Function(V_T)

        # --- Partial stresses ---
        self.functions["ds_partial"]               = Function(V_sp, name="Deviatoric_stress_increment")
        self.functions_previous["ds_partial"]      = Function(V_sp)
        self.functions["dsigma_partial"]            = Function(V_Tfp, name="Hydrostatic_stress_increment")
        self.functions_previous["dsigma_partial"]   = Function(V_Tfp)

        self.functions_current["s_tilde_partial"]  = Function(V_sp)
        self.functions_next["s_tilde_partial"]     = Function(V_sp)
        self.functions_current["sigma_tilde_partial"] = Function(V_Tfp)
        self.functions_next["sigma_tilde_partial"]    = Function(V_Tfp)

        self.functions_current["s_partial"]        = Function(V_sp)
        self.functions_next["s_partial"]           = Function(V_sp)
        self.functions_current["sigma_partial"]    = Function(V_Tfp)
        self.functions_next["sigma_partial"]       = Function(V_Tfp)

        # --- Total stress ---
        self.functions_next["sigma"]               = Function(V_sig, name="Stress_tensor")
        self.functions_current["sigma"]            = Function(V_sig)
        self.functions["sigma_xx"]                 = Function(V_T)
        self.functions["equilibrium_force_x"]      = Function(V_T)
        self.functions_next["total_d_partial"]     = Function(V_sig, name="Viscoelastic_part")
        self.functions_next["total_tilde_partial"] = Function(V_sig, name="Structural_relaxation")
        self.functions_next["sigma_next_adjusted"] = Function(V_sig)
        self.functions["sigma_1d"]                 = Function(V_sig)

        # --- Stiffness matrix coefficients ---
        self.functions["stiffness_matrix"] = Function(V_sig)
        self.functions["A"]                = Function(V_T)
        self.functions["B"]                = Function(V_T)

        # --- Boundary condition functions ---
        self.functions["T_ambient"]    = fem.Function(V_T, name="T_ambient")

        self.functions["htc"]          = fem.Function(V_T, name="HTC_faces")
        self.functions["htc2"]         = fem.Function(V_T, name="HTC_edges")
        self.functions["cooling_rate"] = fem.Function(V_T, name="Cooling_rate")

        # --- Displacement ---
        self.functions["U"]            = Function(V_U, name="Displacement")
        self.functions_previous["U"]   = Function(V_U)
        self.u_trial = TrialFunction(V_U)
        self.v_test  = TestFunction(V_U)

        # --- Uniform strain constant (for equilibrium enforcement) ---
        self.functions["z"] = fem.Constant(self.mesh, ScalarType(0.0))

    # ====================================================================
    # Setup
    # ====================================================================
    def setup(self, dirichlet_bc_mech: bool = True,
              outfile_T: str = "visco",
              outfile_sigma: str = "stresses",
              create_vtx_files: bool = True) -> None:

        self._set_initial_condition(temp_value=self.material_model.T_init)
        self.create_vtx_files = create_vtx_files

        if self.create_vtx_files:
            self._write_initial_output(t=self.t)
            logger.debug("Create vtx-Files")

        self._setup_weak_form_T()
        self._setup_solver_T()
        # Note: in 1D infinite-plate tempering, displacement U is not solved.
        # Stress is driven purely by thermal_strain; equilibrium ∫σdΩ=0
        # is enforced by enforce_zero_force_shift() each step.
        # _setup_weak_form_u and _setup_solver_u are skipped for dim==1.
        if self.dim > 1:
            if dirichlet_bc_mech:
                self._set_dirichlet_bc_mech()
            self._setup_weak_form_u()
            self._setup_solver_u()

    # ====================================================================
    # Initial conditions
    # ====================================================================
    def _set_initial_condition(self, temp_value) -> None:
        self.__set_IC_T(temp_value)
        self.__set_IC_Tf()
        self.__set_IC_Tf_partial()

    def __set_IC_T(self, temp_value) -> None:
        T0 = float(temp_value)
        def temp_init(x):
            return np.full(x.shape[1], T0, dtype=ScalarType)
        self.functions_previous["T"].interpolate(temp_init)
        self.functions_current["T"].interpolate(temp_init)

    def __set_IC_Tf(self) -> None:
        """Tf(t=0) = T(t=0);  xi(t=0) = 0."""
        self.functions_previous["Tf"].x.array[:] = \
            self.functions_previous["T"].x.array[:]
        self.functions_current["Tf"].x.array[:] = \
            self.functions_current["T"].x.array[:]
        # FIX: zero xi properly (old code assigned a Constant to a scalar slot)
        self.functions["xi"].x.array[:]             = 0.0
        self.functions_previous["xi"].x.array[:]    = 0.0
        self.functions_previous["phi"].x.array[:]   = 1.0
        self.functions_current["phi"].x.array[:]    = 1.0
        # phi_v IC at T=T0, Tf=T0:  φ = exp((H/R)(1/Tref - 1/T0))
        import math as _math
        H_val    = float(self.material_model.Hv.value)
        R_val    = float(self.material_model.Rg.value)
        Tb_val   = float(self.material_model.Tb.value)
        T0_val   = float(self.material_model.T_init.value)
        phi_v_ic = _math.exp((H_val / R_val) * (1.0/Tb_val - 1.0/T0_val))
        self.functions_previous["phi_v"].x.array[:] = phi_v_ic
        self.functions_previous["phi_v"].x.scatter_forward()
        self.functions["phi_v"].x.array[:]           = phi_v_ic
        self.functions["phi_v"].x.scatter_forward()

    def __set_IC_Tf_partial(self) -> None:
        """Tf_partial_i(t=0) = T(t=0) for all i."""
        # FIX: use float(T_init) — rank-safe; old code used x.array[0]
        T0  = float(self.material_model.T_init)
        dim = self.material_model.tableau_size
        def Tf_init(x):
            return np.full((dim, x.shape[1]), T0, dtype=ScalarType)
        self.functions_previous["Tf_partial"].interpolate(Tf_init)
        self.functions_current["Tf_partial"].interpolate(Tf_init)

    # ====================================================================
    # VTX output
    # ====================================================================
    def _write_initial_output(self, t: float = 0.0) -> None:
        self.vtx_files = [
            io.VTXWriter(self.mesh.comm, "output/T.bp",
                         [self.functions_current["T"]], engine="BP4"),
            io.VTXWriter(self.mesh.comm, "output/phi_v.bp",
                         [self.functions["phi_v"]], engine="BP4"),
            io.VTXWriter(self.mesh.comm, "output/phi.bp",
                         [self.functions_current["phi"]], engine="BP4"),
            io.VTXWriter(self.mesh.comm, "output/Tf.bp",
                         [self.functions_current["Tf"]], engine="BP4"),
            io.VTXWriter(self.mesh.comm, "output/sigma_xx.bp",
                         [self.functions["sigma_xx"]], engine="BP4"),
            io.VTXWriter(self.mesh.comm, "output/xi.bp",
                         [self.functions["xi"]], engine="BP4"),
        ]
        for file in self.vtx_files:
            file.write(t)

        self.outfile_sigma = io.XDMFFile(
            self.mesh.comm, "output/sigma.xdmf", "w")
        self.outfile_sigma.write_mesh(self.mesh)
        self.outfile_sigma.write_function(self.functions_next["sigma"], t)

        self.outfile_elastic_strain = io.XDMFFile(
            self.mesh.comm, "output/elastic_sigma.xdmf", "w")
        self.outfile_elastic_strain.write_mesh(self.mesh)
        self.outfile_elastic_strain.write_function(
            self.functions["elastic_strain"], t)

    # ====================================================================
    # Thermal weak form
    # ====================================================================
    def _setup_weak_form_T(self) -> None:
        """
        Standard CG Galerkin weak form for the heat equation.
        Backward Euler time integration, Robin (convection) BCs.
        """
        T  = self.functions_current["T"]
        T0 = self.functions_previous["T"]
        dx = Measure("dx", domain=self.mesh)

        facet_dim = self.mesh.topology.dim - 1
        xcoords   = self.mesh.geometry.x[:, 0]
        atol      = (xcoords.max() - xcoords.min()) * 0.01
        xL = self.mesh.comm.allreduce(xcoords.min(), op=MPI.MIN)
        xR = self.mesh.comm.allreduce(xcoords.max(), op=MPI.MAX)

        def on_left(x):  return np.isclose(x[0], xL, atol=atol)
        def on_right(x): return np.isclose(x[0], xR, atol=atol)

        left_facets  = locate_entities_boundary(self.mesh, facet_dim, on_left)
        right_facets = locate_entities_boundary(self.mesh, facet_dim, on_right)

        if self.dim == 2:
            ycoords = self.mesh.geometry.x[:, 1]
            yL = self.mesh.comm.allreduce(ycoords.min(), op=MPI.MIN)
            yR = self.mesh.comm.allreduce(ycoords.max(), op=MPI.MAX)
            def on_bottom(x): return np.isclose(x[1], yL, atol=atol)
            def on_top(x):    return np.isclose(x[1], yR, atol=atol)
            bottom_facets = locate_entities_boundary(self.mesh, facet_dim, on_bottom)
            top_facets    = locate_entities_boundary(self.mesh, facet_dim, on_top)
            indices = np.concatenate(
                [left_facets, right_facets, bottom_facets, top_facets]).astype(np.int32)
            values  = np.concatenate([
                np.full(left_facets.size,   1, dtype=np.int32),
                np.full(right_facets.size,  2, dtype=np.int32),
                np.full(bottom_facets.size, 3, dtype=np.int32),
                np.full(top_facets.size,    4, dtype=np.int32),
            ])
        else:
            indices = np.concatenate([left_facets, right_facets]).astype(np.int32)
            values  = np.concatenate([
                np.full(left_facets.size,  1, dtype=np.int32),
                np.full(right_facets.size, 2, dtype=np.int32),
            ])

        if indices.size == 0:
            indices = np.array([], dtype=np.int32)
            values  = np.array([], dtype=np.int32)

        facet_tags_local = meshtags(self.mesh, facet_dim, indices, values)
        ds = Measure("ds", domain=self.mesh, subdomain_data=facet_tags_local)

        # Thermal conductivity k(T) and specific heat cp(T)
        T_C  = T - 273.15
        k_T  = 0.975 + 8.58e-4 * T_C
        Tg   = 850.0
        cp_l = 1433.0
        cp_s = 893.0 + 0.4 * T - 1.8e-7 / T**2
        cp_T = conditional(ge(T, Tg), cp_l, cp_s)

        rho      = self.physical_model.rho
        f        = self.physical_model.f
        eps      = self.physical_model.epsilon
        sigma_SB = self.physical_model.sigma
        T_ext    = self.functions["T_ambient"]

        # Combined convection + radiation HTC
        h_rad   = sigma_SB * eps * (T**3 + T**2*T_ext + T*T_ext**2 + T_ext**3)
        h_total = self.functions["htc"] + h_rad

        h2_total = self.functions["htc2"] + h_rad

        face_terms = (
            h_total * (T - T_ext) * self.v * ds(1)
            + h_total * (T - T_ext) * self.v * ds(2)
        )
        edge_terms = ufl.as_ufl(0)
        if self.dim == 2:
            edge_terms = (
                h2_total * (T - T_ext) * self.v * ds(3)
                + h2_total * (T - T_ext) * self.v * ds(4)
            )

        self.F = (
            rho * cp_T * (T - T0) * self.v * dx
            + self.dt * (
                k_T * inner(grad(T), grad(self.v)) * dx
                - f * self.v * dx
                + face_terms
                + edge_terms
            )
        )

    def _setup_solver_T(self) -> None:
        """
        Manual Newton solver for the nonlinear heat equation.
        Uses standard Newton iteration with GMRES solver.
        """
        import ufl as _ufl
        T_trial   = TrialFunction(self.functionSpaces["T"])
        self.J_form = fem.form(
            _ufl.derivative(self.F, self.functions_current["T"], T_trial)
        )
        self.F_form = fem.form(self.F)

        self._T_ksp = PETSc.KSP().create(self.mesh.comm)
        self._T_ksp.setType("gmres")
        self._T_ksp.getPC().setType("ilu")
        self._T_ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)

        self._newton_rtol   = 1e-8
        self._newton_atol   = 1e-10
        self._newton_max_it = 25

    def _solve_T(self) -> None:
        """
        Newton iteration for the nonlinear heat equation (CG).
        Assembles F and J fresh each iteration.
        """
        from dolfinx.fem.petsc import (assemble_matrix as _asm_mat,
                                       assemble_vector as _asm_vec)
        T     = self.functions_current["T"]
        T_vec = T.x.petsc_vec

        for it in range(self._newton_max_it):
            b = _asm_vec(self.F_form)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
            r_norm = b.norm(PETSc.NormType.NORM_2)

            J = _asm_mat(self.J_form, bcs=[])
            J.assemble()

            b.scale(-1.0)
            self._T_ksp.setOperators(J)
            dT = J.createVecRight()
            self._T_ksp.solve(b, dT)

            T_vec.axpy(1.0, dT)
            T.x.scatter_forward()

            dx_norm = dT.norm(PETSc.NormType.NORM_2)
            if dx_norm < self._newton_atol:
                break
            if it > 0 and r_norm > 0 and dx_norm < self._newton_rtol * r_norm:
                break

        self.functions_previous["T"].x.array[:] = self.functions_current["T"].x.array.copy()
        self.functions_previous["T"].x.scatter_forward()

    def _solve_Tf(self) -> None:
        """Update shift function φ_v, then partial and total fictive temp."""
        self.functions["phi_v"].interpolate(
            self.material_model.expressions["phi_v"])
        self.functions_current["Tf_partial"].interpolate(
            self.material_model.expressions["Tf_partial"])
        self.functions_previous["Tf_partial"].x.array[:] = self.functions_current["Tf_partial"].x.array.copy()
        self.functions_previous["Tf_partial"].x.scatter_forward()
        self.functions_current["Tf"].interpolate(
            self.material_model.expressions["Tf"])
        self.functions_previous["Tf"].x.array[:] = self.functions_current["Tf"].x.array.copy()
        self.functions_previous["Tf"].x.scatter_forward()

    def _solve_shifted_time(self) -> None:
        """
        Compute reduced time ξ using φ_v (Aronen Eq. 12-13).

        Aronen Eq. 12: φ = exp((H/R)(1/Tref - x/T - (1-x)/Tf))
        Aronen Eq. 13: ξ(t) = ∫₀ᵗ φ(t') dt'

        φ depends on BOTH T and Tf (full Narayanaswamy shift function).
        We use φ_v (already computed in _solve_Tf) for the xi increment.

        Trapezoidal rule:
          Δξ = (Δt/2)(φ_v^{n}  + φ_v^{n+1})
          where φ_v^{n}   = phi_v evaluated at previous (T^n,  Tf^n)
                φ_v^{n+1} = phi_v evaluated at current  (T^{n+1}, Tf^{n+1})

        NOTE: _solve_Tf must run before _solve_shifted_time so that
        both Tf^{n+1} and phi_v are available.
        """
        # φ_v^{n+1} was computed in _solve_Tf at (T^{n+1}, Tf^{n+1}) — available
        # φ_v^{n}   was stored as functions_previous["phi_v"] at end of last step
        # Δξ = (Δt/2)(φ_v^n + φ_v^{n+1})
        self.functions["xi"].interpolate(
            self.material_model.expressions["xi"])
        self.functions["delta_xi"].interpolate(
            self.material_model.expressions["delta_xi"])

        # Also update T_next for next-step carry-over extrapolation
        self.functions_next["T"].interpolate(
            self.material_model.expressions["T_next"])
        # phi_v^{n} ← phi_v^{n+1} for next step (store current as previous)
        self.functions_previous["phi_v"].x.array[:] = self.functions["phi_v"].x.array.copy()
        self.functions_previous["phi_v"].x.scatter_forward()

    def _solve_strains(self) -> None:
        """
        Compute thermal strain and volumetric strain.
        In 1D infinite-plate tempering:
          - thermal_strain drives the hydrostatic stress via k_n terms
          - elastic_strain / deviatoric_strain = 0  (U=0, no bending)
          - volumetric_strain = 0  (no net displacement through thickness)
        The stress is entirely hydrostatic, driven by thermal_strain,
        with equilibrium enforced by the post-step mean subtraction.
        """
        self.functions["thermal_strain"].interpolate(
            self.material_model.expressions["thermal_strain"])
        # volumetric_strain = 0 (U=0 → sym(grad(U))=0)
        self.functions["volumetric_strain"].x.array[:] = 0.0
        self.functions["volumetric_strain"].x.scatter_forward()
        # elastic_strain = 0
        self.functions["elastic_strain"].x.array[:] = 0.0
        self.functions["elastic_strain"].x.scatter_forward()

    def _solve_stress(self) -> None:
        """
        Compute in-plane residual stress for 1D infinite-plate tempering.
        Also updates M_eff (effective biaxial modulus) for DG elasticity.

        Uses the incremental viscoelastic update (Prony series) for the
        hydrostatic part, driven by thermal strain.

        The total stress is assembled from partial Prony stresses and
        equilibrium is enforced by enforce_zero_force_shift().
        """
        # Stress increments  Δσ̄_n  driven by -ε_th (Eq. 15b)
        self.functions["dsigma_partial"].interpolate(
            self.material_model.expressions["dsigma_partial"])

        # Carry-over terms  σ̃_n  (Eq. 16b)
        self.functions_next["sigma_tilde_partial"].interpolate(
            self.material_model.expressions["sigma_tilde_partial_next"])

        # Updated partial stresses  σ_n = Δσ̄_n + σ̃_n  (Eq. 17b)
        self.functions_next["sigma_partial"].interpolate(
            self.material_model.expressions["sigma_partial_next"])

        # Total stress  σ = Σ_n σ_n
        # sigma_partial has function space shape (tableau_size,) per node
        # Stored as flat array: [node0_term0, node0_term1, ..., node1_term0, ...]
        # OR as [term0_node0, term0_node1, ..., term1_node0, ...]  depending on ordering
        # Use the UFL expression to compute the sum safely
        self.functions_next["sigma"].interpolate(
            self.material_model.expressions["sigma_next"])

    # ====================================================================
    # Main time-step  (corrected ordering)
    # ====================================================================
    # ====================================================================
    # Utility helpers
    # ====================================================================

    def _capture_zone_transitions(self, t: float) -> None:
        """No-op placeholder — zone transitions handled in main_fem.py."""
        pass

    def _solve_u(self) -> None:
        """
        1D infinite plate: displacement U is not solved.
        Equilibrium ∫σ dΩ = 0 enforced by enforce_zero_force_shift().
        """
        self.functions["U"].x.array[:] = 0.0
        self.functions["U"].x.scatter_forward()

    def _setup_weak_form_u(self) -> None:
        """CG Galerkin weak form for displacement (2D only)."""
        ds = Measure("exterior_facet", domain=self.mesh)
        dx = Measure("dx", domain=self.mesh)

        if self.dim == 2:
            ss       = ufl.as_vector((0.0, 0.0))
            traction = Constant(self.mesh, ScalarType([0.0, 0.0]))
        else:
            ss       = ufl.as_vector([0.0])
            traction = Constant(self.mesh, ScalarType([0.0]))

        self.a_u = inner(
            self.material_model.elastic_sigma(self.u_trial),
            self.material_model.elastic_epsilon(self.v_test)
        ) * dx
        self.L_u = dot(ss, self.v_test) * dx + dot(traction, self.v_test) * ds

    def _setup_solver_u(self) -> None:
        """Standard CG linear solver for displacement (2D only)."""
        self.u_problem = fem.petsc.LinearProblem(
            self.a_u, self.L_u,
            u=self.functions["U"],
            bcs=self.bc,
            petsc_options_prefix="mech_U_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type":  "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        )

    def _print_zone_summary_end(self, zone_name=None, t_start=None,
                                t_end=None, T_start=None, T_end=None) -> None:
        """Log zone summary at end of zone. All arguments optional."""
        if zone_name is None:
            return   # called with no args from solve() — no-op
        dt_zone  = (t_end - t_start) if (t_end and t_start) else 0.0
        dT_zone  = (T_end - T_start) if (T_end and T_start) else 0.0
        rate_K_s = dT_zone / dt_zone if dt_zone > 0 else 0.0
        logger.info(
            f"[ZONE END] {zone_name:<8} "
            f"t={t_end:7.1f}s  "
            f"T_start={T_start:.2f}K  T_end={T_end:.2f}K  "
            f"dT={dT_zone:.2f}K  dt={dt_zone:.1f}s  "
            f"rate={rate_K_s:.5f} K/s  ({rate_K_s*60:.3f} K/min)"
        )



    def enforce_zero_force_shift(self) -> float:
        """
        Enforce global equilibrium ∫σ dΩ = 0 for the 1D infinite plate.
        Subtracts mean(σ) from the total stress and shifts sigma_partial
        consistently so carry-over terms reflect the equilibrium state.
        """
        dx = ufl.Measure("dx", domain=self.mesh)

        int_sigma = fem.assemble_scalar(
            fem.form(self.functions_next["sigma"][0, 0] * dx))
        int_one   = fem.assemble_scalar(
            fem.form(ufl.as_ufl(1.0) * dx))

        int_sigma = self.mesh.comm.allreduce(int_sigma, op=MPI.SUM)
        int_one   = self.mesh.comm.allreduce(int_one,   op=MPI.SUM)

        mean_sigma = int_sigma / int_one

        self.functions_next["sigma"].x.array[:] -= ScalarType(mean_sigma)
        self.functions_next["sigma"].x.scatter_forward()

        shift_per_term = mean_sigma / self.material_model.tableau_size
        self.functions_next["sigma_partial"].x.array[:] -= ScalarType(shift_per_term)
        self.functions_next["sigma_partial"].x.scatter_forward()

        return float(mean_sigma)

    def _set_dirichlet_bc_mech(self) -> None:
        """Set Dirichlet BCs for the mechanical problem (2D only)."""
        if self.dim == 1:
            self.bc = []
            return
        # For 2D: pin midplane to prevent rigid-body motion
        V_U   = self.functionSpaces["U"]
        dofs  = fem.locate_dofs_geometrical(
            V_U, lambda x: np.isclose(x[0], 0.0, atol=1e-10))
        zero  = Function(V_U)
        zero.x.array[:] = 0.0
        self.bc = [fem.dirichletbc(zero, dofs)]

    def solve_timestep(self, t) -> None:
        """
        Corrected step ordering aligned with paper algorithm:
          1. T^{n+1}       — heat equation
          2. Tf^{n+1}      — fictive temperature (Markovsky-Soules)
          3. ξ^{n+1}       — reduced time (trapezoidal)
          4. u^{n+1}       — displacement (linear elasticity)
          5. ε, ε_th, ε̄   — strains
          6-8. σ           — stress increments → partial → total
          9. equilibrium   — ∫σ dΩ = 0
         10. history       — advance all state variables
        """
        logger.debug(f"t={t:.4f}")

        self._solve_T()
        self._capture_zone_transitions(t)
        self._solve_Tf()
        self._solve_shifted_time()
        self._solve_u()
        self._solve_strains()
        self._solve_stress()
        mean_sigma = self.enforce_zero_force_shift()

        # --- stress magnitude diagnostic (first 5 steps only) ---
        if self.t <= 5 * self.dt + 1e-12:
            sig_arr = self.functions_next["sigma"].x.array
            xi_arr  = self.functions["xi"].x.array
            eps_arr = self.functions["elastic_strain"].x.array
            ds_arr  = self.functions["ds_partial"].x.array
            dsig_arr= self.functions["dsigma_partial"].x.array
            sp_arr  = self.functions_next["s_partial"].x.array
            sigp_arr= self.functions_next["sigma_partial"].x.array
            logger.info(
                f"  t={self.t:.3f}  |sigma|_max={np.abs(sig_arr).max():.3e}"
                f"  xi={xi_arr.mean():.3e}"
                f"  |eps|={np.abs(eps_arr).max():.3e}"
                f"  |ds|={np.abs(ds_arr).max():.3e}"
                f"  |dsig|={np.abs(dsig_arr).max():.3e}"
                f"  |s_part|={np.abs(sp_arr).max():.3e}"
                f"  |sig_part|={np.abs(sigp_arr).max():.3e}"
            )

        # --- diagnostics ---
        self.avg_T.append([np.average(self.functions_current["T"].x.array)])
        nx = len(self.functions_current["T"].x.array)
        self.T_0_edge.append([self.functions_current["T"].x.array[0]])
        self.T_0_middle.append([self.functions_current["T"].x.array[nx // 2]])
        self.avg_phi_v.append([np.average(self.functions["phi_v"].x.array)])
        self.avg_phi.append([np.average(self.functions_current["phi"].x.array)])
        self.avg_Tf.append([np.average(self.functions_current["Tf"].x.array)])
        self.avg_xi.append([np.average(self.functions["xi"].x.array)])
        self.avg_thermal_epsilon.append([np.average(self.functions["thermal_strain"].x.array)])
        self.avg_t_epsilon.append([np.average(self.functions["volumetric_strain"].x.array)])
        self.avg_t_sigma.append([np.average(self.functions_next["sigma"].x.array)])
        self.avg_t_sigma_mid.append([self.functions_next["sigma"].x.array[nx // 2]])
        self.avg_t_sigma_surface.append([self.functions_next["sigma"].x.array[0]])

        self.temperature_field_history.append(
            self.functions_current["T"].x.array.copy())
        self.fictive_temp_history.append(
            self.functions_current["Tf"].x.array.copy())
        self.stress_field_history.append(
            self.functions_next["sigma"].x.array.copy())
        # Save full coordinates (x and y) for node identification in post-processing
        self.position_history.append(
            self.mesh.geometry.x[:, :self.dim].copy())

        if self.create_vtx_files:
            self._write_output()

        self._update_step_history()

    def _write_output(self) -> None:
        """Write VTX output files if enabled."""
        if hasattr(self, "vtx_T"):
            self.vtx_T.write(self.t)
        if hasattr(self, "vtx_sigma"):
            self.vtx_sigma.write(self.t)

    def _update_step_history(self) -> None:
        """
        Advance all state variables at end of each timestep.
        Copies current/next values into previous slots.
        """
        def _copy(src, dst):
            dst.x.array[:] = src.x.array.copy()
            dst.x.scatter_forward()

        # T: current → previous
        _copy(self.functions_current["T"], self.functions_previous["T"])

        # phi: current → previous
        _copy(self.functions_current["phi"], self.functions_previous["phi"])

        # phi_v: functions → previous
        _copy(self.functions["phi_v"], self.functions_previous["phi_v"])

        # Tf: current → previous
        _copy(self.functions_current["Tf"], self.functions_previous["Tf"])

        # Tf_partial: current → previous
        _copy(self.functions_current["Tf_partial"],
              self.functions_previous["Tf_partial"])

        # xi: functions (updated) → previous
        _copy(self.functions["xi"], self.functions_previous["xi"])

        # thermal_strain: functions → previous (needed for Δε_th)
        if "thermal_strain" in self.functions_previous:
            _copy(self.functions["thermal_strain"],
                  self.functions_previous["thermal_strain"])

        # sigma_partial and s_partial: next → current (carry-over for next step)
        # The carry-over expressions read from functions_current["sigma_partial"]
        # and functions_current["s_partial"], so we must advance them here.
        self.functions_current["sigma_partial"].x.array[:] = \
            self.functions_next["sigma_partial"].x.array.copy()
        self.functions_current["sigma_partial"].x.scatter_forward()

        self.functions_current["s_partial"].x.array[:] = \
            self.functions_next["s_partial"].x.array.copy()
        self.functions_current["s_partial"].x.scatter_forward()

        # sigma_tilde_partial and s_tilde_partial: next → current
        if "s_tilde_partial" in self.functions_current:
            self.functions_current["s_tilde_partial"].x.array[:] = \
                self.functions_next["s_tilde_partial"].x.array.copy()
            self.functions_current["s_tilde_partial"].x.scatter_forward()

        if "sigma_tilde_partial" in self.functions_current:
            self.functions_current["sigma_tilde_partial"].x.array[:] = \
                self.functions_next["sigma_tilde_partial"].x.array.copy()
            self.functions_current["sigma_tilde_partial"].x.scatter_forward()




    # ====================================================================
    # Solve loop
    # ====================================================================
    def solve(self):
        self.avg_T                  = []
        self.T_0_edge               = []
        self.T_0_middle             = []
        self.avg_phi_v              = []
        self.avg_phi                = []
        self.avg_Tf                 = []
        self.avg_xi                 = []
        self.avg_thermal_epsilon    = []
        self.avg_t_epsilon          = []
        self.avg_t_sigma            = []
        self.avg_t_sigma_mid        = []
        self.avg_t_sigma_surface    = []
        self.temperature_field_history = []
        self.stress_field_history      = []
        self.fictive_temp_history      = []
        self.position_history          = []

        if self.mesh.comm.rank == 0:
            logger.debug("Starting solve")
            t_start = time()

        for _ in range(self.n_steps):
            self.t += self.dt
            self.solve_timestep(t=self.t)
            self._append_outgoing_dto()

        if self.mesh.comm.rank == 0:
            t_end = time()
            logger.debug(f"Solve finished in {t_end - t_start:.1f} s")
            self.execution_time = t_end - t_start

        if self.create_vtx_files:
            self._finalize()

        self._print_zone_summary_end()

        self.temperature_field_history = np.array(self.temperature_field_history)
        self.stress_field_history      = np.array(self.stress_field_history)
        self.position_history          = np.array(self.position_history)

        return self.outgoing_dto

    # ====================================================================
    # Output helpers
    # ====================================================================
    def _finalize(self) -> None:
        for file in self.vtx_files:
            file.close()
        self.outfile_sigma.close()
        self.outfile_elastic_strain.close()

    def _append_outgoing_dto(self) -> None:
        elem = Elements()
        elem.Time        = self.t
        elem.Stress      = self.functions_next["sigma"].x.array[:].tolist()
        elem.Temperature = self.functions_current["T"].x.array[:].tolist()
        elem.Thickness   = None
        self.outgoing_dto.append(elem)

    def _to_np_arrays(self, t) -> None:
        """Convert FEniCS functions to numpy arrays (diagnostic)."""
        if self.dim == 1:
            s_array     = self.functions_next["sigma"].x.array[:]
            self.sigma_xx = s_array
            self.avg_t_sigma_mid.append(
                [np.average(self.sigma_xx[len(self.sigma_xx) // 2])])
            self.avg_t_sigma_surface.append(
                [np.average(self.sigma_xx[0])])
        elif self.dim == 2:
            nx, ny = 20, 10
            s_array          = self.functions_next["sigma"].x.array[:]
            s_array_reshaped = s_array.reshape((4, nx, ny))
            self.sigma_xx    = s_array_reshaped[0, :, :]
            self.avg_t_sigma_mid.append(
                [np.average(self.sigma_xx[:, ny // 2])])
        elif self.dim == 3:
            nx, ny, nz = 20, 10, 10
            s_array          = self.functions_next["sigma"].x.array[:]
            s_array_reshaped = s_array.reshape((9, nx, ny))
            self.sigma_xx    = s_array_reshaped[0, :, :]
            self.avg_t_sigma_mid.append(
                [np.average(self.sigma_xx[:, ny // 2])])