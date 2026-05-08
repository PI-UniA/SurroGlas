# ViscoelasticModel.py
#
# Aligned with:
#   Aronen & Karvinen, Glass Struct. Eng. (2018) 3:3-15
#   (same Prony parameters as Daudeville & Carré 1998 / Carré & Daudeville 1999)
#
# Equations referenced: Aronen & Karvinen (2018) notation
#   Eq. 3  — hereditary stress integral (K and G Prony series)
#   Eq. 6  — bulk modulus K(t) with K_inf/K0 = 0.18
#   Eq. 7  — shear modulus G(t), G0 at t=0
#   Eq. 10 — structural relaxation response function Mp(t)
#   Eq. 11 — thermal strain εth = αg(T-T0) + (αl-αg)(Tf-Tf0)
#   Eq. 12 — shift function φ(T,Tf) — depends on BOTH T and Tf
#   Eq. 13 — reduced time ξ = ∫φ dt  (uses φ with Tf dependence)
#   Eq. 14 — incremental stress update via Prony carry-overs
#   Eq. 16 — force equilibrium: ∫σ dz = 0
#   Eq. 17 — moment equilibrium: ∫σ·z dz = 0  (auto-satisfied by symmetry)
#
# Key implementation decisions:
#   1. xi uses φv (Eq. 12 — both T and Tf), NOT T-only shift
#   2. Tf update uses Markovsky-Soules explicit algorithm
#   3. Stress uses biaxial modulus M_n = K_n + (4/3)G_n for in-plane 1D
#   4. _ramp uses Δξ (increment), not ξ (total) — critical for frozen glass
#   5. Carry-over exp(-Δξ/λ) → 1 when glass frozen (Δξ→0)

from dolfinx.mesh import Mesh
from dolfinx.fem import Constant, Expression
from petsc4py.PETSc import ScalarType
from dolfinx import fem
import ufl
from ufl import (sym, Identity, grad, SpatialCoordinate, conditional, ge)
import numpy as np
from math import factorial


class ViscoelasticModel:
    def __init__(self, mesh: Mesh, model_parameters: dict) -> None:
        """
        Material model for soda-lime silicate glass tempering.
        Parameters from Aronen & Karvinen (2018) Appendix Tables 1 & 2
        (identical to Carré & Daudeville 1999 / Daudeville & Carré 1998).
        """

        # ----------------------------------------------------------------
        # Narayanaswamy structural-relaxation weighting  (paper Eq. 10)
        #   x in paper == self.chi here
        # ----------------------------------------------------------------
        self.chi          = 0.5
        self.tableau_size = 6
        self.dim          = mesh.topology.dim

        # ----------------------------------------------------------------
        # Table 2 — Structural relaxation / fictive temperature
        #   M_v(ξ) = Σ C_i exp(−ξ/λ_i)   (Eq. 8)
        # ----------------------------------------------------------------
        self.m_n_tableau = Constant(mesh, [
            5.523e-2,
            8.205e-2,
            1.215e-1,
            2.286e-1,
            2.860e-1,
            2.265e-1,
        ])
        # FIX: index 3 was 1.505e-1 in old code; paper Table 2 gives 1.505 s
        self.lambda_m_n_tableau = Constant(mesh, [
            5.965e-4,
            1.077e-2,
            1.362e-1,
            1.505e+0,
            6.747e+0,
            2.963e+1,
        ])

        # ----------------------------------------------------------------
        # Table 1 — Deviatoric relaxation  G(t) = 2 G_g Ψ₁(t)  (Eq. 2)
        #   FIX: multiply by 1e9 — Table 1 lists values in units of 10⁹ Pa
        # ----------------------------------------------------------------
        self.g_n_tableau = Constant(mesh, [
            1.5845e9,
            2.3539e9,
            3.4857e9,
            6.5582e9,
            8.2049e9,
            6.4980e9,
        ])
        self.lambda_g_n_tableau = Constant(mesh, [
            6.658e-5,
            1.197e-3,
            1.514e-2,
            1.672e-1,
            7.497e-1,
            3.292e+0,
        ])

        # ----------------------------------------------------------------
        # Table 1 — Volume relaxation  K(t)  (Eq. 3)
        #   FIX: 6 terms only (old code had spurious 7th); multiply by 1e9
        #   K_e / K_g = 0.18  (Table 1 caption)
        # ----------------------------------------------------------------
        self.k_n_tableau = Constant(mesh, [
            0.7588e9,
            0.7650e9,
            0.9806e9,
            7.301e9,
            13.47e9,
            10.896e9,
        ])
        self.lambda_k_n_tableau = Constant(mesh, [
            5.009e-5,
            9.945e-4,
            2.022e-3,
            1.925e-2,
            1.199e-1,
            2.033e+0,
        ])

        # Identity tensor
        self.I = Identity(mesh.topology.dim)

        # Initial / reference temperature [K]
        self.T_init = Constant(mesh, ScalarType(model_parameters["T_0"]))

        # Narayanaswamy shift-function parameters (Eq. 5 / 10)
        # H/R = 76 200 K  →  H = 76200 × 8.314 = 633 527 J/mol  (Aronen Table 1)
        self.Hv = Constant(mesh, ScalarType(model_parameters["Hv"]))
        self.H  = Constant(mesh, ScalarType(model_parameters["H"]))
        self.Rg = Constant(mesh, ScalarType(model_parameters["Rg"]))
        self.Tb = Constant(mesh, ScalarType(model_parameters["Tb"]))

        # Thermal expansion coefficients (Eq. 11)
        self.alpha_solid  = Constant(mesh, ScalarType(model_parameters["alpha_solid"]))
        self.alpha_liquid = Constant(mesh, ScalarType(model_parameters["alpha_liquid"]))

        # Elastic constants (paper p. 675)
        E  = model_parameters["Young's_modulus"]   # Pa  (must be 70e9)
        nu = model_parameters["Possion_ratio"]      # 0.22
        self.G0    = Constant(mesh, ScalarType(E / (2.0 * (1.0 + nu))))
        self.K0    = Constant(mesh, ScalarType(E / (3.0 * (1.0 - 2.0 * nu))))
        self.K_inf = Constant(mesh, ScalarType(0.18 * E / (3.0 * (1.0 - 2.0 * nu))))
        self.young_modulus = Constant(mesh, ScalarType(E))

        # Lamé parameters kept for backward compatibility
        self.lambda_ = Constant(mesh, ScalarType(model_parameters["lambda_"]))
        self.mu      = Constant(mesh, ScalarType(model_parameters["mu"]))
        self.stress_zero = Constant(mesh, ScalarType(0.0))
        self.x = SpatialCoordinate(mesh)

    # ====================================================================
    # Expression initialisation
    # ====================================================================
    def _init_expressions(self,
                          functions:          dict,
                          functions_next:     dict,
                          functions_current:  dict,
                          functions_previous: dict,
                          functionSpaces:     dict,
                          dt:                 float) -> None:
        """
        Build all UFL expressions used during the time loop.
        Called once after function spaces are set up.
        """
        self.expressions = {}

        # interpolation_points is a property (array), not a method — no ()
        ipts_T   = functionSpaces["T"].element.interpolation_points
        ipts_sig = functionSpaces["sigma"].element.interpolation_points
        ipts_sp  = functionSpaces["sigma_partial"].element.interpolation_points
        ipts_Tfp = functionSpaces["Tf_partial"].element.interpolation_points

        # ----------------------------------------------------------------
        # Shift function φ  (Aronen Eq. 12 — Narayanaswamy form)
        #   φ = exp((H/R)(1/Tref − x/T − (1−x)/Tf))
        #   x = Hg/H  (ratio of activation energies, Table 1: x=0.5)
        #   Used for: (a) Tf update via Markovsky-Soules
        #             (b) reduced time ξ increment (Eq. 13)
        # ----------------------------------------------------------------
        self.expressions["phi_v"] = Expression(
            ufl.exp(
                (self.Hv / self.Rg) * (
                    1.0 / self.Tb
                    - self.chi         / functions_current["T"]
                    - (1.0 - self.chi) / functions_previous["Tf"]
                )
            ),
            ipts_T
        )

        # Shift function at current T only (Eq. 5 — no Tf dependence)
        self.expressions["phi_current"] = Expression(
            ufl.exp(
                (self.Hv / self.Rg) * (1.0 / self.Tb - 1.0 / functions_current["T"])
            ),
            ipts_T
        )
        self.expressions["phi_next"] = Expression(
            ufl.exp(
                (self.Hv / self.Rg) * (1.0 / self.Tb - 1.0 / functions_next["T"])
            ),
            ipts_T
        )
        self.expressions["phi_previous"] = Expression(
            ufl.exp(
                (self.Hv / self.Rg) * (1.0 / self.Tb - 1.0 / functions_previous["T"])
            ),
            ipts_T
        )

        # Linear extrapolation T_{n+1} ≈ 2T^n − T^{n-1}
        self.expressions["T_next"] = Expression(
            functions_current["T"] + (functions_current["T"] - functions_previous["T"]),
            ipts_T
        )

        # ----------------------------------------------------------------
        # Partial fictive temperatures  Tf_i  (Aronen Eq. 9–10)
        #   Explicit algorithm of Markovsky et al. (1984):
        #   Tf_i^{n+1} = (λi·Tf_i^n + T^{n+1}·Δt·φv) / (λi + Δt·φv)
        #   Total: Tf = Σ_i Ci·Tf_i   (Aronen Eq. 10)
        #   φv uses Tf at previous step (explicit, unconditionally stable)
        # ----------------------------------------------------------------
        self.expressions["Tf_partial"] = Expression(
            ufl.as_vector([
                (
                    self.lambda_m_n_tableau[n] * functions_previous["Tf_partial"][n]
                    + functions_current["T"] * dt * functions["phi_v"]
                ) / (
                    self.lambda_m_n_tableau[n] + dt * functions["phi_v"]
                )
                for n in range(self.tableau_size)
            ]),
            ipts_Tfp
        )

        # ----------------------------------------------------------------
        # Total fictive temperature  Tf = Σ C_i · T_{f,i}  (Eq. 8 / Eq. 26)
        # FIX: np.sum → Python sum() for UFL-safe accumulation
        # ----------------------------------------------------------------
        self.expressions["Tf"] = Expression(
            sum(
                self.m_n_tableau[n] * functions_current["Tf_partial"][n]
                for n in range(self.tableau_size)
            ),
            ipts_T
        )

        # ----------------------------------------------------------------
        # Reduced (shifted) time  ξ  (Aronen Eq. 13)
        #   ξ(t) = ∫₀ᵗ φ(t') dt'
        #   φ from Eq. 12 depends on BOTH T and Tf:
        #     φ = exp((H/R)(1/Tref - x/T - (1-x)/Tf))
        #   Trapezoidal rule:  Δξ = (Δt/2)(φ_v^n + φ_v^{n+1})
        #   We use phi_v (T+Tf dependent) for the xi increment,
        #   consistent with Narayanaswamy (1978) and Aronen Eq. 12–13.
        # ----------------------------------------------------------------
        # φ_v^n   = functions_previous["phi_v"]  (stored at end of last step)
        # φ_v^n+1 = functions["phi_v"]             (just computed in _solve_Tf)
        delta_xi_expr = (dt / 2.0) * (
            functions_previous["phi_v"] + functions["phi_v"]
        )
        self.expressions["xi"] = Expression(
            functions_previous["xi"] + delta_xi_expr,
            ipts_T
        )
        # Δξ = increment of reduced time for THIS step (used in ramp)
        self.expressions["delta_xi"] = Expression(
            delta_xi_expr,
            ipts_T
        )

        # ----------------------------------------------------------------
        # Elastic strain tensor  ε = sym(∇u)
        # ----------------------------------------------------------------
        self.expressions["elastic_strain"] = Expression(
            self.elastic_epsilon(functions["U"]),
            ipts_sig
        )
        self.expressions["elastic_stress"] = Expression(
            self.elastic_sigma(functions["U"]),
            ipts_sig
        )

        # ----------------------------------------------------------------
        # Volumetric strain  ε̄
        #   1D: ε_xx   |  2D: (ε_xx + ε_yy)/2  |  3D: tr(ε)/3
        # ----------------------------------------------------------------
        eps = self.elastic_epsilon(functions["U"])
        if self.dim == 1:
            vol_expr = eps[0, 0]
        elif self.dim == 2:
            vol_expr = 0.5 * (eps[0, 0] + eps[1, 1])
        else:
            vol_expr = (1.0 / 3.0) * (eps[0, 0] + eps[1, 1] + eps[2, 2])

        self.expressions["volumetric_strain"] = Expression(vol_expr, ipts_T)

        # ----------------------------------------------------------------
        # Thermal strain  ε_th  (Aronen Eq. 11)
        #   εth(t) = αg·(T(t) − T(0)) + (αl − αg)·(Tf(t) − Tf(0))
        #   Since Tf(0) = T(0) = T0 (glass above Tg at start):
        #   εth = αg·(T − T0) + (αl − αg)·(Tf − T0)
        #   Reference: Narayanaswamy (1978), also Aronen (2018) Appendix
        # ----------------------------------------------------------------
        self.expressions["thermal_strain"] = Expression(
            (
                self.alpha_solid  * (functions_current["T"]  - self.T_init)
                + (self.alpha_liquid - self.alpha_solid)
                  * (functions_current["Tf"] - self.T_init)
            ),
            ipts_T
        )

        # ----------------------------------------------------------------
        # Deviatoric stress increment  Δs_n  (Eq. 15a + Eq. 20)
        #   Δs_n = 2 g_n · e · (λ_n/Δξ)(1 − exp(−Δξ/λ_n))
        #   e = deviatoric strain = ε − (1/dim) tr(ε) I
        # ----------------------------------------------------------------
        deviatoric_strain = (
            self.elastic_epsilon(functions["U"])
            - (1.0 / self.dim) * self.I * functions["volumetric_strain"]
        )
        self.expressions["ds_partial"] = Expression(
            ufl.as_tensor([
                2.0 * self.g_n_tableau[n]
                * deviatoric_strain
                * self._ramp(functions, self.lambda_g_n_tableau[n],
                             float(self.lambda_g_n_tableau.value[n]))
                for n in range(self.tableau_size)
            ]),
            ipts_sp
        )

        # ----------------------------------------------------------------
        # In-plane biaxial stress increment  Δσ̄_n  (Aronen Eq. 3 + 14)
        #
        # For 1D infinite plate (Kirchhoff-Love hypothesis, Aronen Eq. 16-17):
        #   σyy = σzz (biaxial by symmetry)
        #   In-plane biaxial modulus: M_n = K_n + (4/3)G_n
        #   (combines bulk K_n and shear G_n contributions)
        #
        # Incremental update (Chambers 1992, used by Aronen):
        #   Δσ̄_n = M_n · (-Δεth) · ψ(Δξ, λ_n)
        #   ψ(Δξ,λ) = (λ/Δξ)(1 - exp(-Δξ/λ))    — Aronen Eq. 14
        #
        # Uses Δξ (increment per step), NOT total ξ:
        #   When T < Tg: Δξ→0, ψ→1, but Δεth→0 → Δσ→0 (frozen glass ✅)
        #   When T > Tg: Δξ large → ψ→0 → fast relaxation (liquid ✅)
        #
        # Equilibrium (Aronen Eq. 16): ∫σ dz = 0
        #   Enforced by subtracting mean(σ) each step.
        # ----------------------------------------------------------------
        # Δε_th = ε_th^{n+1} - ε_th^n  (increment this step)
        delta_eth = functions["thermal_strain"] - functions_previous["thermal_strain"]
        self.expressions["dsigma_partial"] = Expression(
            ufl.as_vector([
                (self.k_n_tableau[n] + (4.0/3.0) * self.g_n_tableau[n])
                * (-delta_eth)
                * self._ramp(functions, self.lambda_k_n_tableau[n],
                             float(self.lambda_k_n_tableau.value[n]))
                for n in range(self.tableau_size)
            ]),
            ipts_Tfp
        )

        # ----------------------------------------------------------------
        # Carry-over terms  s̃_n, σ̃_n  (Eq. 16a/b)
        #   s̃_n^{n+1} = s_n^n · exp(−Δξ/λ_{g,n})
        #   σ̃_n^{n+1} = σ_n^n · exp(−Δξ/λ_{k,n})
        # ----------------------------------------------------------------
        # Carry-over (Aronen Eq. 14 / Chambers 1992 recursive update):
        #   σ̃_n^{n+1} = σ_n^n · exp(-Δξ/λ_n)
        # Uses Δξ (step increment), not total ξ:
        #   When Δξ→0 (frozen glass): exp→1 → history perfectly preserved ✅
        #   When Δξ large (above Tg): exp→0 → fast relaxation / liquid ✅
        self.expressions["s_tilde_partial_next"] = Expression(
            ufl.as_tensor([
                functions_current["s_partial"][n, :, :]
                * ufl.exp(-functions["delta_xi"] / self.lambda_g_n_tableau[n])
                for n in range(self.tableau_size)
            ]),
            ipts_sp
        )
        self.expressions["sigma_tilde_partial_next"] = Expression(
            ufl.as_vector([
                functions_current["sigma_partial"][n]
                * ufl.exp(-functions["delta_xi"] / self.lambda_k_n_tableau[n])
                for n in range(self.tableau_size)
            ]),
            ipts_Tfp
        )

        # ----------------------------------------------------------------
        # Updated partial stresses  s_n, σ_n  (Eq. 17a/b)
        #   s_n^{n+1}  = Δs_n  + s̃_n^{n+1}
        #   σ_n^{n+1}  = Δσ̄_n + σ̃_n^{n+1}
        # ----------------------------------------------------------------
        self.expressions["s_partial_next"] = Expression(
            ufl.as_tensor([
                functions["ds_partial"][n, :, :]
                + functions_next["s_tilde_partial"][n, :, :]
                for n in range(self.tableau_size)
            ]),
            ipts_sp
        )
        self.expressions["sigma_partial_next"] = Expression(
            ufl.as_vector([
                functions["dsigma_partial"][n]
                + functions_next["sigma_tilde_partial"][n]
                for n in range(self.tableau_size)
            ]),
            ipts_Tfp
        )

        # ----------------------------------------------------------------
        # Total in-plane stress  (Aronen Eq. 3, 1D plate interpretation)
        #   σ = Σ_n σ_n   where σ_n = Δσ̄_n + σ̃_n  (biaxial hydrostatic)
        # Deviatoric contribution (s_partial via G_n) = 0 for 1D plate
        # since the plate remains flat (Kirchhoff-Love, Aronen Eq. 16-17)
        # and there is no through-thickness displacement U to compute.
        # Equilibrium shift (Aronen Eq. 16) applied after assembly.
        # ----------------------------------------------------------------
        self.expressions["sigma_next"] = Expression(
            sum(
                self.I * functions_next["sigma_partial"][n]
                for n in range(self.tableau_size)
            ),
            ipts_sig
        )


        # ----------------------------------------------------------------
        # Stiffness coefficients A, B  (Eq. 22)
        #   A = (1/3) Σ g_n · (λ_{g,n}/Δt) · (1 − exp(−Δt/λ_{g,n}))
        #   B =       Σ k_n · (λ_{k,n}/Δt) · (1 − exp(−Δt/λ_{k,n}))
        # Note: real time Δt used here (not reduced Δξ)
        # ----------------------------------------------------------------
        # A and B: exact formula (λ/dt)(1-exp(-dt/λ)) — already stable
        A_expr = (1.0 / 3.0) * sum(
            self.g_n_tableau[n]
            * (self.lambda_g_n_tableau[n] / dt)
            * (1.0 - ufl.exp(-dt / self.lambda_g_n_tableau[n]))
            for n in range(self.tableau_size)
        )
        B_expr = sum(
            self.k_n_tableau[n]
            * (self.lambda_k_n_tableau[n] / dt)
            * (1.0 - ufl.exp(-dt / self.lambda_k_n_tableau[n]))
            for n in range(self.tableau_size)
        )
        self.expressions["A"] = Expression(A_expr, ipts_T)
        self.expressions["B"] = Expression(B_expr, ipts_T)

        # ----------------------------------------------------------------
        # Viscoelastic stiffness matrix  D_t  (Eq. 21)
        #   1D: [B + 4A]
        #   2D: [[B+4A, B-2A], [B-2A, B+4A]]
        #   3D: full 3×3 with off-diag = B-2A
        # ----------------------------------------------------------------
        A = functions["A"]
        B = functions["B"]
        if self.dim == 1:
            Dt = ufl.as_tensor([[B + 4.0 * A]])
        elif self.dim == 2:
            Dt = ufl.as_tensor([
                [B + 4.0 * A, B - 2.0 * A],
                [B - 2.0 * A, B + 4.0 * A],
            ])
        else:
            Dt = ufl.as_tensor([
                [B + 4.0 * A, B - 2.0 * A, B - 2.0 * A],
                [B - 2.0 * A, B + 4.0 * A, B - 2.0 * A],
                [B - 2.0 * A, B - 2.0 * A, B + 4.0 * A],
            ])
        self.expressions["stiffness_matrix"] = Expression(Dt, ipts_sig)

        # Diagnostic: σ_1D = D_t · (ε̄ − ε_th)
        self.expressions["sigma_1d"] = Expression(
            functions["stiffness_matrix"]
            * (functions["volumetric_strain"] - functions["thermal_strain"]),
            ipts_sig
        )

        # Summation helpers for VTX output
        # FIX: np.sum → Python sum()
        self.expressions["total_d_partial"] = Expression(
            sum(
                functions["ds_partial"][n, :, :]
                + self.I * functions["dsigma_partial"][n]
                for n in range(self.tableau_size)
            ),
            ipts_sig
        )
        self.expressions["total_tilde_partial"] = Expression(
            sum(
                functions_next["s_tilde_partial"][n, :, :]
                + self.I * functions_next["sigma_tilde_partial"][n]
                for n in range(self.tableau_size)
            ),
            ipts_sig
        )

    # ====================================================================
    # Helper: ramp function (λ/Δξ)(1 − e^{−Δξ/λ})
    # ====================================================================
    def _ramp(self, functions, lambda_value, lam_float: float):
        """
        Computes (λ/Δξ)·(1 − exp(−Δξ/λ))  exactly for all Δξ/λ.

        Uses Δξ (the increment per timestep), NOT the total ξ.
        When Δξ → 0 (frozen glass, T << Tg): ramp → 1 but Δε_mech → 0,
        so the product Δσ = M * Δε_mech * ramp → 0 correctly.

        lam_float: plain Python float value of λ.
        """
        dxi      = functions["delta_xi"]
        eps      = 1e-30
        dxi_safe = dxi + eps
        return (lam_float / dxi_safe) * (1.0 - ufl.exp(-dxi_safe / lam_float))

    # ====================================================================
    # Strain / stress helpers
    # ====================================================================
    def elastic_epsilon(self, ua):
        """Symmetric gradient ε = sym(∇u)."""
        from ufl import sym, grad
        return sym(grad(ua))

    def elastic_sigma(self, ua):
        """Alias kept for backward compatibility."""
        return self.elastic_epsilon(ua)