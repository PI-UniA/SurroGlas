# ViscoelasticModel.py
#
# Material model + rigorous 1D-plate stress solver for glass tempering.
#
#   * ViscoelasticModel : holds the material constants and Prony tableaux and
#     builds the FEniCS (UFL) expressions for the decoupled structural-
#     relaxation problem that ThermoViscoProblem interpolates each step:
#         phi_v         shift function          (Aronen Eq. 11 / Kong Eq. 7)
#         delta_xi      reduced-time increment  (Aronen Eq. 13)
#         Tf_partial,Tf fictive temperature     (Markovsky-Soules, Eq. 9-10)
#         thermal_strain                        (Aronen Eq. 14 / Kong Eq. 10)
#
#   * MTildeStress : the in-plane residual-stress update (pure NumPy), called
#     from ThermoViscoProblem._solve_stress.  For the infinite plate it forms
#     the consistent biaxial tangent  M~ = 18 K~ G~ / (3 K~ + 4 G~), eliminates
#     eps_zz from sigma_zz = 0, and solves the uniform in-plane strain from
#     int sigma dz = 0 (exact equilibrium).  See its own docstring.
#
# Material data (k, G Prony series, expansion coeffs, shift function):
#   Aronen & Karvinen, Glass Struct. Eng. (2018) 3:3-15  (Tables 1 & 2).
# Plate reduction / deviatoric-volumetric split:
#   Kong, Kim & Chung, Met. Mater. Int. 13(1):67-75 (2007).
# -------------------------------------------------------------------------

from dolfinx.mesh import Mesh
from dolfinx.fem import Constant, Expression
from dolfinx import fem
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import ufl
from ufl import sym, Identity, grad, SpatialCoordinate
import numpy as np


class ViscoelasticModel:
    def __init__(self, mesh: Mesh, model_parameters: dict) -> None:
        """
        Material model for soda-lime silicate glass tempering.
        Structural-relaxation parameters: Aronen & Karvinen (2018) Tables 1 & 2.
        Stress: Kong (2007) auxiliary-modulus single-convolution method.
        """
        # ----------------------------------------------------------------
        # Narayanaswamy structural-relaxation weighting  (Aronen Eq. 11)
        #   chi = x = Hg/H = 0.5  (Aronen Table 1)
        # ----------------------------------------------------------------
        self.chi          = 0.5
        self.tableau_size = 6
        self.dim          = mesh.topology.dim

        # ----------------------------------------------------------------
        # Structural relaxation  M_p(xi) = sum C_i exp(-xi/lambda_i)  (Eq. 10)
        #   C_i dimensionless, sum C_i = 1 ; lambda_i in seconds (Table 2)
        # ----------------------------------------------------------------
        self.m_n_tableau = Constant(mesh, [
            5.523e-2, 8.205e-2, 1.215e-1, 2.286e-1, 2.860e-1, 2.266e-1,
        ])
        self.lambda_m_n_tableau = Constant(mesh, [
            5.965e-4, 1.077e-2, 1.362e-1, 1.505e+0, 6.747e+0, 2.963e+1,
        ])

        # ----------------------------------------------------------------
        # Shear relaxation  G(xi) = G0 * sum g_n exp(-xi/lambda_{g,n})  (Eq. 7)
        #   g_n dimensionless, sum g_n = 1 ; physical G_n = g_n * G0
        # ----------------------------------------------------------------
        self.g_n_tableau = Constant(mesh, [
            0.05523, 0.08205, 0.1215, 0.2286, 0.2860, 0.22662,
        ])
        self.lambda_g_n_tableau = Constant(mesh, [
            6.658e-5, 1.197e-3, 1.514e-2, 1.672e-1, 7.497e-1, 3.292e+0,
        ])

        # ----------------------------------------------------------------
        # Bulk relaxation  K(xi) = K_inf + (K0-K_inf) sum k_n exp(-xi/lambda_{k,n})
        #   K_e/K_g = K_inf/K0 = 0.18  (Aronen Table 1 caption, Eq. 6)
        # ----------------------------------------------------------------
        self.k_n_tableau = Constant(mesh, [
            0.0222, 0.0224, 0.0286, 0.2137, 0.394, 0.3191,
        ])
        self.lambda_k_n_tableau = Constant(mesh, [
            5.009e-5, 9.945e-4, 2.022e-3, 1.925e-2, 1.199e-1, 2.033e+0,
        ])

        self.I = Identity(mesh.topology.dim)
        self.T_init = Constant(mesh, ScalarType(model_parameters["T_0"]))

        # Shift-function parameters (Aronen Eq. 11): HvRg = H/R [K]
        self.HvRg = Constant(mesh, ScalarType(model_parameters["HvRg"]))
        self.H  = Constant(mesh, ScalarType(model_parameters["H"]))
        self.Rg = Constant(mesh, ScalarType(model_parameters["Rg"]))
        self.Tb = Constant(mesh, ScalarType(model_parameters["Tb"]))

        # Thermal expansion (Aronen Eq. 14 / Kong Eq. 10)
        self.alpha_solid  = Constant(mesh, ScalarType(model_parameters["alpha_solid"]))
        self.alpha_liquid = Constant(mesh, ScalarType(model_parameters["alpha_liquid"]))

        # ----------------------------------------------------------------
        # Elastic constants  E=70 GPa, nu=0.22
        #   G0 = E/2(1+nu) = 28.69 GPa,  K0 = E/3(1-2nu) = 41.67 GPa
        #   K_inf = 0.18 K0 = 7.50 GPa
        #   biaxial modulus  E/(1-nu) = 18 K0 G0/(3K0+4G0) = 89.74 GPa
        # ----------------------------------------------------------------
        E  = model_parameters["Young's_modulus"]
        nu = model_parameters["Possion_ratio"]
        K0_val    = E / (3.0 * (1.0 - 2.0 * nu))
        G0_val    = E / (2.0 * (1.0 + nu))
        K_inf_val = 0.18 * K0_val
        M_biax0_val = E / (1.0 - nu)

        self.G0      = Constant(mesh, ScalarType(G0_val))
        self.K0      = Constant(mesh, ScalarType(K0_val))
        self.K_inf   = Constant(mesh, ScalarType(K_inf_val))
        self.M_biax0 = Constant(mesh, ScalarType(M_biax0_val))
        self.lambda_ = Constant(mesh, ScalarType(model_parameters["lambda_"]))
        self.mu      = Constant(mesh, ScalarType(model_parameters["mu"]))
        self.stress_zero   = Constant(mesh, ScalarType(0.0))
        self.young_modulus = Constant(mesh, ScalarType(E))
        self.x = SpatialCoordinate(mesh)

        # ----------------------------------------------------------------
        # Physical Prony amplitudes of the relaxation moduli (NumPy arrays),
        # consumed by the stress solver MTildeStress.  Derived ONCE here so
        # the moduli live in a single place:
        #   K(t) = K_inf + sum_p dK_p exp(-t / tau_K_p)
        #   G(t) =          sum_q dG_q exp(-t / tau_G_q)
        # ----------------------------------------------------------------
        self.dK_p  = (K0_val - K_inf_val) * np.array(self.k_n_tableau.value, float)
        self.tau_K = np.array(self.lambda_k_n_tableau.value, float)
        self.dG_q  = G0_val * np.array(self.g_n_tableau.value, float)
        self.tau_G = np.array(self.lambda_g_n_tableau.value, float)

    # ====================================================================
    # Expression initialisation
    # ====================================================================
    def _init_expressions(self,
                          functions:          dict,
                          functions_next:     dict,
                          functions_current:  dict,
                          functions_previous: dict,
                          functionSpaces:     dict,
                          dt:                 float,
                          d_eps,
                          mesh) -> None:
        self.expressions = {}

        ipts_T   = functionSpaces["T"].element.interpolation_points
        ipts_Tfp = functionSpaces["Tf_partial"].element.interpolation_points
        ip = lambda f: f.function_space.element.interpolation_points
        nK,   nG   = len(self.dK_p),  len(self.dG_q)

        dxi  = functions["delta_xi"]
        deth = functions["thermal_strain"] - functions_previous["thermal_strain"]
        
        # --- reduced moduli and history (decayed by the carry-over factor) ---
        psiK = [self._ramp_ufl(dxi, float(self.tau_K[p])) for p in range(nK)]
        psiG = [self._ramp_ufl(dxi, float(self.tau_G[q])) for q in range(nG)]
        Ktil = self.K_inf + sum(float(self.dK_p[p]) * psiK[p] for p in range(nK))
        Gtil =        sum(float(self.dG_q[q]) * psiG[q] for q in range(nG))
        Mtil = 18.0 * Ktil * Gtil / (3.0 * Ktil + 4.0 * Gtil)

        sigH_hat  = [functions["sigma_bulk_partial"][p] * ufl.exp(-dxi / float(self.tau_K[p])) for p in range(nK)]
        sS_hat    = [functions["sigma_shear_partial"][q]   * ufl.exp(-dxi / float(self.tau_G[q])) for q in range(nG)]
        sigH_hist = self.K_inf * functions["mech_dilatation"] + sum(sigH_hat)
        sxx_hist  = sum(sS_hat)
        Hxx = sigH_hist + sxx_hist
        Hzz = sigH_hist - 2.0 * sxx_hist            # s_zz = -2 s_xx (plate)
        Bc  = Ktil - (2.0 / 3.0) * Gtil
        Dc  = Ktil + (4.0 / 3.0) * Gtil             # >= Kinf > 0 always
        Hxx_star = Hxx - (Bc / Dc) * Hzz

        # --- global equilibrium: d_eps = int(Mtil deth - Hxx*)dz / int Mtil dz
        #     (assembled integrals; for CG1 these equal the trapezoidal sums) ---
        dx = ufl.Measure("dx", domain=mesh)
        self.stress_num_form = fem.form((Mtil * deth - Hxx_star) * dx)
        self.stress_den_form = fem.form(Mtil * dx)
        self.stress_len_form = fem.form(Constant(mesh, ScalarType(1.0)) * dx)
        self.stress_len = mesh.comm.allreduce(
            fem.assemble_scalar(self.stress_len_form), op=MPI.SUM)
        self.stress_K0  = float(self.K0.value)
        
        # --- state increments (depend on d_eps, set each step before interpolate) ---
        d_eps_zz = (((4.0 / 3.0) * Gtil - 2.0 * Ktil) * d_eps
                    + 3.0 * Ktil * deth - Hzz) / Dc
        d_eps_mech  = 2.0 * d_eps + d_eps_zz - 3.0 * deth
        d_e_xx   = (1.0 / 3.0) * (d_eps - d_eps_zz)
        epszz_new = functions["out_of_plane_strain"] + d_eps_zz
        sigma_new = Mtil * (d_eps - deth) + Hxx_star
        
        # ----------------------------------------------------------------
        # Shift function  phi  (Aronen Eq. 11 / Kong Eq. 7)
        #   phi = exp( (H/R)(1/T_ref - x/T - (1-x)/Tf) )
        # ----------------------------------------------------------------
        self.expressions["phi_v"] = Expression(
            ufl.exp(
                (self.HvRg) * (
                    1.0 / self.Tb
                    - self.chi         / functions_current["T"]
                    - (1.0 - self.chi) / functions_previous["Tf"]
                )
            ),
            ipts_T
        )

        # ----------------------------------------------------------------
        # Reduced-time increment  Delta_xi = (dt/2)(phi^n + phi^{n+1})  (Eq. 13)
        #   Defined BEFORE Tf so the fictive-temperature recursion can reuse it
        #   (consistency: same Delta_xi drives Tf, ramps and carry-overs).
        # ----------------------------------------------------------------
        delta_xi_expr = (dt / 2.0) * (
            functions_previous["phi_v"] + functions["phi_v"]
        )
        self.expressions["delta_xi"] = Expression(delta_xi_expr, ipts_T)

        # ----------------------------------------------------------------
        # Partial fictive temperatures  (Markovsky-Soules, Aronen Eq. 9-10)
        #   Tf_i^{n+1} = (lam_i Tf_i^n + T^{n+1} Delta_xi)/(lam_i + Delta_xi)
        #   FIX: uses the trapezoidal Delta_xi (was dt*phi before).
        # ----------------------------------------------------------------
        self.expressions["Tf_partial"] = Expression(
            ufl.as_vector([
                (
                    self.lambda_m_n_tableau[n] * functions_previous["Tf_partial"][n]
                    + functions_current["T"] * delta_xi_expr
                ) / (
                    self.lambda_m_n_tableau[n] + delta_xi_expr
                )
                for n in range(self.tableau_size)
            ]),
            ipts_Tfp
        )

        # Total fictive temperature  Tf = sum C_i Tf_i
        self.expressions["Tf"] = Expression(
            sum(
                self.m_n_tableau[n] * functions_current["Tf_partial"][n]
                for n in range(self.tableau_size)
            ),
            ipts_T
        )

        # ----------------------------------------------------------------
        # Thermal strain  (Aronen Eq. 14 / Kong Eq. 10)
        #   eps_th = alpha_g (T - T0) + (alpha_l - alpha_g)(Tf - T0)
        # ----------------------------------------------------------------
        self.expressions["thermal_strain"] = Expression(
            (
                self.alpha_solid * (functions_current["T"] - self.T_init)
                + (self.alpha_liquid - self.alpha_solid)
                  * (functions_current["Tf"] - self.T_init)
            ),
            ipts_T
        )
        self.expressions["sigma_bulk_partial"]  = Expression(          
            ufl.as_vector([sigH_hat[p] + float(self.dK_p[p]) * psiK[p] * d_eps_mech
                                   for p in range(nK)]),  
            ip(functions_next["sigma_bulk_partial"])
        )   
        self.expressions["sigma_shear_partial"] = Expression(
            ufl.as_vector([sS_hat[q] + 2.0 * float(self.dG_q[q]) * psiG[q] * d_e_xx
                                   for q in range(nG)]),    
            ip(functions_next["sigma_shear_partial"])
        )  
        self.expressions["mech_dilatation"] = Expression(
            functions["mech_dilatation"]  + 2.0 * d_eps + d_eps_zz - 3.0 * deth, 
            ip(functions_next["mech_dilatation"])
        )
        self.expressions["out_of_plane_strain"] = Expression(
            epszz_new, 
            ip(functions_next["out_of_plane_strain"])
        )
        # 1D plate: sigma tensor is (1,1) -> interpolate directly (no permutation)
        self.expressions["sigma"] = Expression(
            ufl.as_matrix([[sigma_new]]),
            ip(functions_next["sigma"])
        )
    # ====================================================================
    # Rigorous 1D infinite-plate stress (Aronen Eq.3 + Kong split), as UFL.
    #   sigma_xx = M~ (d_eps - d_eps_th) + Hxx*,  M~ = 18 K~ G~/(3 K~ + 4 G~)
    # eps_zz eliminated pointwise from sigma_zz = 0; the uniform in-plane
    # strain increment d_eps solved from int sigma dz = 0 (assembled forms).
    # All FEniCS objects are created & owned by ThermoViscoProblem and passed
    # in here; this class only assembles the physics into UFL.
    # ====================================================================
    @staticmethod
    def _ramp_ufl(dxi, tau):
        """
        psi(dxi,tau) = (1 - exp(-r))/r,  r = dxi/tau, with psi->1 as r->0.
        The denominator is guarded (safe=1 when r is tiny) because UFL
        conditional() evaluates BOTH branches, so an unguarded (1-e^-r)/r
        would produce 0/0 at dxi=0 even though the series branch is selected.
        """
        r    = dxi / tau
        safe = ufl.conditional(ufl.lt(r, 1.0e-6), 1.0, r)
        return ufl.conditional(ufl.lt(r, 1.0e-6),
                               1.0 - 0.5 * r + r * r / 6.0,
                               (1.0 - ufl.exp(-r)) / safe)
