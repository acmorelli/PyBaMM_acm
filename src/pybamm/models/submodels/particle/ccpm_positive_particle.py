#C:\Users\dottavianomo\programming\PyBaMM_acm\src\pybamm\models\submodels\particle\ccpm_positive_particle.py
from .base_particle import BaseParticle
import pybamm


class CCPMPositiveParticle(BaseParticle):
    """
    CCPM positive electrode particle model (Clarke et al. 2026).

    Models hysteresis in LFP cathodes via probability density functions (PDFs)
    on three branches (A=Li-poor, B=mixed-phase, C=Li-rich), each living on
    its own compact concentration subdomain (Clarke 2026, eq 62):
      Branch A: [0, c_sp1]         (Li-poor spinodal limit)
      Branch B: [c1_star, c2_star] (mixed-phase)
      Branch C: [c_sp2, c_max]     (Li-rich spinodal limit)

    Each PDF is discretised on a separate FV mesh; there is no masking.
    """

    def __init__(
        self,
        param,
        domain="positive",
        options=None,
        phase="primary",
        initial_branch="C",
        source_method="flux_form",
        n_pts=(63, 258, 63),
    ):
        super().__init__(
            param,
            domain,
            options=options,
            phase=phase,
        )
        # Three subdomain spatial variables
        self.c_p_a = pybamm.standard_spatial_vars.c_p_a
        self.c_p_b = pybamm.standard_spatial_vars.c_p_b
        self.c_p_c = pybamm.standard_spatial_vars.c_p_c

        self.initial_branch = initial_branch.upper()
        self.source_method = source_method
        # Number of FV cells per branch — must match dfn_ccpm default_var_pts
        self.n_pts_a, self.n_pts_b, self.n_pts_c = n_pts

    def get_fundamental_variables(self):
        # Each PDF lives on its own concentration subdomain
        g_a = pybamm.Variable(
            "Branch A PDF CCPM",
            domains={
                "primary": "CCPM positive particle branch A",
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
        )
        g_b = pybamm.Variable(
            "Branch B PDF CCPM",
            domains={
                "primary": "CCPM positive particle branch B",
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
        )
        g_c = pybamm.Variable(
            "Branch C PDF CCPM",
            domains={
                "primary": "CCPM positive particle branch C",
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
        )

        variables = {
            "Branch A PDF CCPM": g_a,
            "Branch B PDF CCPM": g_b,
            "Branch C PDF CCPM": g_c,
            "X-averaged Branch A PDF CCPM": pybamm.x_average(g_a),
            "X-averaged Branch B PDF CCPM": pybamm.x_average(g_b),
            "X-averaged Branch C PDF CCPM": pybamm.x_average(g_c),
        }
        return variables

    def _branch_boundaries(self):
        """Return the four scalar branch-boundary constants."""
        c_max = self.param.p.prim.c_max
        c1_star = pybamm.Scalar(0.0710) * c_max
        c_sp1   = pybamm.Scalar(0.2113) * c_max
        c_sp2   = pybamm.Scalar(0.7887) * c_max
        c2_star = pybamm.Scalar(0.9290) * c_max
        return c1_star, c_sp1, c_sp2, c2_star

    def get_coupled_variables(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        c_p_a = self.c_p_a
        c_p_b = self.c_p_b
        c_p_c = self.c_p_c
        c_max = self.param.p.prim.c_max

        # Average stoichiometry: sum of integrals over each branch domain
        theta_CCPM = (
            pybamm.Integral(g_a * (c_p_a / c_max), c_p_a)
            + pybamm.Integral(g_b * (c_p_b / c_max), c_p_b)
            + pybamm.Integral(g_c * (c_p_c / c_max), c_p_c)
        )

        # Branch masses
        m_a = pybamm.Integral(g_a, c_p_a)
        m_b = pybamm.Integral(g_b, c_p_b)
        m_c = pybamm.Integral(g_c, c_p_c)
        m_tot = m_a + m_b + m_c

        variables.update({
            "CCPM Branch A mass": m_a,
            "CCPM Branch B mass": m_b,
            "CCPM Branch C mass": m_c,
            "CCPM Total mass": m_tot,
            "Positive CCPM stoichiometry": theta_CCPM,
            "X-averaged positive CCPM stoichiometry": pybamm.x_average(theta_CCPM),
            "X-averaged Branch A PDF CCPM": pybamm.x_average(g_a),
            "X-averaged Branch B PDF CCPM": pybamm.x_average(g_b),
            "X-averaged Branch C PDF CCPM": pybamm.x_average(g_c),
            "X-averaged CCPM Branch A mass": pybamm.x_average(m_a),
            "X-averaged CCPM Branch B mass": pybamm.x_average(m_b),
            "X-averaged CCPM Branch C mass": pybamm.x_average(m_c),
            "X-averaged CCPM Total mass": pybamm.x_average(m_tot),
        })

        # Standard concentration equivalent for downstream submodels
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        variables.update({
            f"R-averaged {domain} {phase_name}"
            "particle concentration [mol.m-3]": theta_CCPM * c_max,
        })

        return variables

    def set_rhs(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        R_a = variables["CCPM Branch A lithiation rate"]
        R_b = variables["CCPM Branch B lithiation rate"]
        R_c = variables["CCPM Branch C lithiation rate"]

        def adv_flux(g, R):
            R_pos = pybamm.Maximum(R, 0)
            R_neg = pybamm.Minimum(R, 0)
            return pybamm.Upwind(g) * R_pos + pybamm.Downwind(g) * R_neg

        F_a = adv_flux(g_a, R_a)
        F_b = adv_flux(g_b, R_b)
        F_c = adv_flux(g_c, R_c)

        # Clean divergences — no masking needed with separate subdomains
        rhs_a = -pybamm.div(F_a)
        rhs_b = -pybamm.div(F_b)
        rhs_c = -pybamm.div(F_c)

        # Compute transition sources and wall corrections together
        # so they share the same R_edge and g_cell values (FV-consistent).
        (S_a, S_b, S_c,
         S_wall_a, S_wall_b, S_wall_c,
         J_A_to_B, J_B_to_A, J_B_to_C, J_C_to_B
        ) = self._get_sources_and_corrections(variables)

        # Mass-balance error: scalar sum of source integrals across all domains
        source_error = (
            pybamm.Integral(S_a, self.c_p_a)
            + pybamm.Integral(S_b, self.c_p_b)
            + pybamm.Integral(S_c, self.c_p_c)
        )

        variables.update({
            "Branch A CCPM Flux": F_a,
            "Branch B CCPM Flux": F_b,
            "Branch C CCPM Flux": F_c,
            "Branch A to B scalar flux": J_A_to_B,
            "Branch B to A scalar flux": J_B_to_A,
            "Branch B to C scalar flux": J_B_to_C,
            "Branch C to B scalar flux": J_C_to_B,
            "Branch A source": S_a,
            "Branch B source": S_b,
            "Branch C source": S_c,
            "Branch A wall correction": S_wall_a,
            "Branch B wall correction": S_wall_b,
            "Branch C wall correction": S_wall_c,
            "Branch A wall correction integral": pybamm.Integral(S_wall_a, self.c_p_a),
            "Branch B wall correction integral": pybamm.Integral(S_wall_b, self.c_p_b),
            "Branch C wall correction integral": pybamm.Integral(S_wall_c, self.c_p_c),
            "CCPM source mass balance error": source_error,
        })

        self.rhs = {
            g_a: rhs_a + S_a + S_wall_a,
            g_b: rhs_b + S_b + S_wall_b,
            g_c: rhs_c + S_c + S_wall_c,
        }

    def _get_sources_and_corrections(self, variables):
        """
        Compute transition sources AND wall corrections in one place so they
        share exactly the same R_edge and g_cell values at each boundary.

        At every Dirichlet-g=0 boundary the FV upwind scheme produces:
          F_edge = g_cell · |R_edge|       (ghost = -g_cell)

        The mass removed from each domain at this edge is g_cell·|R_edge|.
        We split this into:
          • "correct direction" flux: deposited into receiving branch
          • "wrong direction" flux:   added back as wall correction

        Physical walls (A-left, C-right): ALL flux is spurious → full correction.
        Transition edges: only the wrong-direction part is corrected.

        Using the SAME R_edge for both J and wall_corr guarantees:
          J + wall_corr = g_cell · |R_edge| = FV removal
        → exact mass conservation.
        """
        c_p_a = self.c_p_a
        c_p_b = self.c_p_b
        c_p_c = self.c_p_c
        c_max = self.param.p.prim.c_max
        eps_c = pybamm.Scalar(1e-8) * c_max
        c1_star, c_sp1, c_sp2, c2_star = self._branch_boundaries()

        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        R_a = variables["CCPM Branch A lithiation rate"]
        R_b = variables["CCPM Branch B lithiation rate"]
        R_c = variables["CCPM Branch C lithiation rate"]

        n_a, n_b, n_c = self.n_pts_a, self.n_pts_b, self.n_pts_c
        dc_a = (c_sp1 - eps_c) / n_a
        dc_b = (c2_star - c1_star) / n_b
        dc_c = (c_max - eps_c - c_sp2) / n_c
        width_a = c_sp1 - eps_c
        width_b = c2_star - c1_star
        width_c = c_max - eps_c - c_sp2

        half = pybamm.Scalar(0.5)
        three_half = pybamm.Scalar(1.5)

        def _edge_R(R_var, c_cell, c_neighbour):
            """FV node-to-edge linear extrapolation at a boundary."""
            R0 = pybamm.EvaluateAt(R_var, c_cell)
            R1 = pybamm.EvaluateAt(R_var, c_neighbour)
            return three_half * R0 - half * R1

        def regularized_delta_on(c_target, c_domain_var, dc):
            """Normalised Gaussian delta on the given domain variable."""
            sigma = 1.5 * dc
            ker = pybamm.exp(-((c_domain_var - c_target) ** 2) / (2 * sigma ** 2))
            return ker / pybamm.Integral(ker, c_domain_var)

        # ================================================================
        # Domain A boundaries
        # ================================================================
        # Left (physical wall at c_p = eps_c): correct ALL ghost flux
        c_a0 = eps_c + dc_a * half
        c_a1 = eps_c + dc_a * three_half
        g_a0 = pybamm.EvaluateAt(g_a, c_a0)
        R_edge_AL = _edge_R(R_a, c_a0, c_a1)
        corr_AL = g_a0 * (pybamm.Maximum(R_edge_AL, 0)
                          - pybamm.Minimum(R_edge_AL, 0))  # |R_edge|

        # Right (transition A→B at c_sp1)
        c_aN = c_sp1 - dc_a * half       # last cell of A
        c_aN1 = c_sp1 - dc_a * three_half  # second-to-last
        g_aN = pybamm.EvaluateAt(g_a, c_aN)
        R_edge_AR = _edge_R(R_a, c_aN, c_aN1)
        # Correct direction: R>0 (outflow A→B) → transition flux
        J_A_to_B = g_aN * pybamm.Maximum(R_edge_AR, 0)
        # Wrong direction: R<0 (ghost reflux) → wall correction
        corr_AR = g_aN * pybamm.Maximum(-R_edge_AR, 0)

        # Localized corrections: each boundary's correction goes back near
        # that boundary cell to prevent cross-talk between boundaries.
        delta_a_left  = regularized_delta_on(c_a0, c_p_a, dc_a)
        delta_a_right = regularized_delta_on(c_aN, c_p_a, dc_a)
        S_wall_a = corr_AL * delta_a_left + corr_AR * delta_a_right

        # ================================================================
        # Domain B boundaries
        # ================================================================
        # Left (transition B→A at c1_star)
        c_b0 = c1_star + dc_b * half
        c_b1 = c1_star + dc_b * three_half
        g_b0 = pybamm.EvaluateAt(g_b, c_b0)
        R_edge_BL = _edge_R(R_b, c_b0, c_b1)
        # Correct direction: R<0 (outflow B→A) → transition flux
        J_B_to_A = g_b0 * pybamm.Maximum(-R_edge_BL, 0)
        # Wrong direction: R>0 → wall correction
        corr_BL = g_b0 * pybamm.Maximum(R_edge_BL, 0)

        # Right (transition B→C at c2_star)
        c_bN = c2_star - dc_b * half
        c_bN1 = c2_star - dc_b * three_half
        g_bN = pybamm.EvaluateAt(g_b, c_bN)
        R_edge_BR = _edge_R(R_b, c_bN, c_bN1)
        # Correct direction: R>0 (outflow B→C) → transition flux
        J_B_to_C = g_bN * pybamm.Maximum(R_edge_BR, 0)
        # Wrong direction: R<0 → wall correction
        corr_BR = g_bN * pybamm.Maximum(-R_edge_BR, 0)

        delta_b_left  = regularized_delta_on(c_b0, c_p_b, dc_b)
        delta_b_right = regularized_delta_on(c_bN, c_p_b, dc_b)
        S_wall_b = corr_BL * delta_b_left + corr_BR * delta_b_right

        # ================================================================
        # Domain C boundaries
        # ================================================================
        # Left (transition C→B at c_sp2)
        c_c0 = c_sp2 + dc_c * half
        c_c1 = c_sp2 + dc_c * three_half
        g_c0 = pybamm.EvaluateAt(g_c, c_c0)
        R_edge_CL = _edge_R(R_c, c_c0, c_c1)
        # Correct direction: R<0 (outflow C→B) → transition flux
        J_C_to_B = g_c0 * pybamm.Maximum(-R_edge_CL, 0)
        # Wrong direction: R>0 → wall correction
        corr_CL = g_c0 * pybamm.Maximum(R_edge_CL, 0)

        # Right (physical wall at c_max - eps_c): correct ALL ghost flux
        c_right = c_max - eps_c
        c_cN = c_right - dc_c * half
        c_cN1 = c_right - dc_c * three_half
        g_cN = pybamm.EvaluateAt(g_c, c_cN)
        R_edge_CR = _edge_R(R_c, c_cN, c_cN1)
        corr_CR = g_cN * (pybamm.Maximum(R_edge_CR, 0)
                          - pybamm.Minimum(R_edge_CR, 0))  # |R_edge|

        delta_c_left  = regularized_delta_on(c_c0, c_p_c, dc_c)
        delta_c_right = regularized_delta_on(c_cN, c_p_c, dc_c)
        S_wall_c = corr_CL * delta_c_left + corr_CR * delta_c_right

        # ================================================================
        # Deposition kernels on receiving domains
        # ================================================================
        dA_c1  = regularized_delta_on(c1_star, c_p_a, dc_a)
        dB_sp1 = regularized_delta_on(c_sp1,  c_p_b, dc_b)
        dB_sp2 = regularized_delta_on(c_sp2,  c_p_b, dc_b)
        dC_c2  = regularized_delta_on(c2_star, c_p_c, dc_c)

        # Source terms: deposition only (removal handled by FV BCs + wall corr)
        S_a = J_B_to_A * dA_c1
        S_b = J_A_to_B * dB_sp1 + J_C_to_B * dB_sp2
        S_c = J_B_to_C * dC_c2

        return (S_a, S_b, S_c,
                S_wall_a, S_wall_b, S_wall_c,
                J_A_to_B, J_B_to_A, J_B_to_C, J_C_to_B)

    def set_boundary_conditions(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        zero = pybamm.Scalar(0)

        # Clarke eq 62 boundary conditions
        #
        # PHYSICAL WALLS (Fa|cp=0 = 0, Fc|cp=cmax = 0):
        #   Dirichlet g=0 required by PyBaMM's upwind scheme.
        #   Ghost = −g_first creates spurious wall flux.
        #   Corrected via source term in set_rhs (see _wall_correction_sources).
        #
        # TRANSITION EDGES (outflow-only at csp1, c1*, c2*, csp2):
        #   Dirichlet g=0 lets the upwind scheme only advect mass outward.

        self.boundary_conditions = {
            g_a: {"left": (zero, "Dirichlet"), "right": (zero, "Dirichlet")},
            g_b: {"left": (zero, "Dirichlet"), "right": (zero, "Dirichlet")},
            g_c: {"left": (zero, "Dirichlet"), "right": (zero, "Dirichlet")},
        }

    def set_initial_conditions(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        c_max = self.param.p.prim.c_max
        eps_c = pybamm.Scalar(1e-8) * c_max
        c1_star, c_sp1, c_sp2, c2_star = self._branch_boundaries()

        c_p_a = self.c_p_a
        c_p_b = self.c_p_b
        c_p_c = self.c_p_c

        # Gamma(k,λ) shape: g(c) ∝ (c - c_min)^(k-1) * exp(-(c - c_min)/λ)
        # with mean = k·λ = c_init  →  λ = c_init / k.
        # k = 10 gives g ≈ 0 at the first cell centre (wall ghost leak ≈ 0)
        # while keeping the peak near c_init and θ(0) = c_init / c_max.
        c_init = pybamm.Parameter(
            "Initial concentration in positive electrode [mol.m-3]"
        )
        k_shape = pybamm.Scalar(10)
        lam = c_init / k_shape

        def gamma_pdf_on(c_domain_var, c_left, c_domain_spatial_var):
            """Normalised Gamma(k,λ) on the given branch domain."""
            shifted = c_domain_var - c_left
            dist = shifted ** (k_shape - 1) * pybamm.exp(-shifted / lam)
            norm = pybamm.Integral(dist, c_domain_spatial_var)
            return dist / norm

        # Zero ICs retain domain structure via multiplication with spatial variable
        zero_a = pybamm.Scalar(0) * c_p_a
        zero_b = pybamm.Scalar(0) * c_p_b
        zero_c = pybamm.Scalar(0) * c_p_c

        if self.initial_branch == "A":
            init_a = gamma_pdf_on(c_p_a, eps_c,   c_p_a)
            self.initial_conditions = {
                g_a: init_a,
                g_b: zero_b,
                g_c: zero_c,
            }
        elif self.initial_branch == "B":
            init_b = gamma_pdf_on(c_p_b, c1_star, c_p_b)
            self.initial_conditions = {
                g_a: zero_a,
                g_b: init_b,
                g_c: zero_c,
            }
        elif self.initial_branch == "C":
            init_c = gamma_pdf_on(c_p_c, c_sp2, c_p_c)
            self.initial_conditions = {
                g_a: zero_a,
                g_b: zero_b,
                g_c: init_c,
            }
        else:
            raise ValueError(
                f"Invalid CCPM initial_branch='{self.initial_branch}'. "
                "Use 'A', 'B', or 'C'."
            )