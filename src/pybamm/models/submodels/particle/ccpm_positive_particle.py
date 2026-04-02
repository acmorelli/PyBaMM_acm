#C:\Users\dottavianomo\programming\PyBaMM_acm\src\pybamm\models\submodels\particle\ccpm_positive_particle.py
from .base_particle import BaseParticle
import pybamm


class CCPMPositiveParticle(BaseParticle):
    """
    CCPM positive electrode particle model (Clarke et al. 2026).

    Models hysteresis in LFP cathodes via probability density functions (PDFs)
    on three branches (A=Li-poor, B=mixed-phase, C=Li-rich) defined on a
    concentration grid. Replaces Fickian diffusion entirely.
    """

    def __init__(
        self,
        param,
        domain="positive",
        options=None,
        phase="primary",
        initial_branch="A",
        mode='discharge'

    ):
        super().__init__(
            param,
            domain,
            options=options,
            phase=phase,
        )
        
        
        self.c_p_space = pybamm.standard_spatial_vars.c_p
        self.initial_branch = initial_branch.upper()
        self.mode = mode
        
    def get_fundamental_variables(self):
        # 1. Define the 3 PDFs as state variables (Differential Variables)
        # Primary domain: the concentration grid 'c'
        # Secondary domain: the electrode position 'x'
        g_a = pybamm.Variable(
            "Branch A PDF CCPM",
        domains={
            "primary": "CCPM positive particle concentration",
            "secondary": "positive electrode",
            "tertiary": "current collector"
        }
        )
        g_b = pybamm.Variable(
            "Branch B PDF CCPM",
        domains={
            "primary": "CCPM positive particle concentration",
            "secondary": "positive electrode",
            "tertiary": "current collector"
        }
        )
        g_c = pybamm.Variable(
            "Branch C PDF CCPM",
        domains={
            "primary": "CCPM positive particle concentration",
            "secondary": "positive electrode",
            "tertiary": "current collector"
    }
        )

        variables = {
            "Branch A PDF CCPM": g_a,
            "Branch B PDF CCPM": g_b,
            "Branch C PDF CCPM": g_c,
            "CCPM positive particle concentration": self.c_p_space,
            "X-averaged Branch A PDF CCPM": pybamm.x_average(g_a),
            "X-averaged Branch B PDF CCPM": pybamm.x_average(g_b),
            "X-averaged Branch C PDF CCPM": pybamm.x_average(g_c),
        }

        return variables
    def _branch_geometry(self, c_p):
        eps=1e-8*self.param.p.prim.c_max 
        c_min = eps                          # left edge of truncated domain
        c_max = self.param.p.prim.c_max -eps # right edge of truncated domain

        # Snapped to exact cell edges for npts=300 (edge indices 21,63,237,279)
        # Cell edge k is at: c_min + k/npts * (c_max - c_min)
        #
        # Original Clarke2026 θ values  →  Snapped θ (= cell edge / c_max_raw)
        #   c1_star:  θ = 0.0709        →  21/300  = 0.0700
        #   c_sp1:    θ = 0.2113        →  63/300  = 0.2100
        #   c_sp2:    θ = 0.7887        →  237/300 = 0.7900
        #   c2_star:  θ = 0.9291        →  279/300 = 0.9300
        npts = 300
        L = c_max - c_min  # domain length
        c1_star = c_min + pybamm.Scalar(21 / npts) * L
        c_sp1   = c_min + pybamm.Scalar(63 / npts) * L
        c_sp2   = c_min + pybamm.Scalar(237 / npts) * L
        c2_star = c_min + pybamm.Scalar(279 / npts) * L

        # same boolean-mask style you already used
        mask_a = (c_p <= c_sp1)
        mask_b = (c_p >= c1_star) * (c_p <= c2_star)
        mask_c = (c_p >= c_sp2)

        return c1_star, c_sp1, c_sp2, c2_star, mask_a, mask_b, mask_c

    def get_coupled_variables(self, variables):
        # Retrieve our PDFs and the concentration grid
        g_a= variables["Branch A PDF CCPM"]
        g_b= variables["Branch B PDF CCPM"]
        g_c= variables["Branch C PDF CCPM"]
        c_p = variables["CCPM positive particle concentration"]

        eps=1e-8*self.param.p.prim.c_max
        c_max = self.param.p.prim.c_max - eps
        theta_CCPM= pybamm.Integral((g_a + g_b + g_c) * (c_p / c_max), c_p)

        # mass per branch
        m_a = pybamm.Integral(g_a, c_p)
        m_b = pybamm.Integral(g_b, c_p)
        m_c = pybamm.Integral(g_c, c_p)
        m_tot = m_a + m_b + m_c

        # Early partial update: stoichiometry and masses are needed by the OCP
        # submodel and do NOT depend on lithiation rates. By updating variables
        # here (before accessing R_a), the framework's KeyError-retry loop can
        # resolve the circular dependency: Particle → OCP → Interface → Particle.
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        variables.update({
            "Positive CCPM stoichiometry": theta_CCPM,
            "X-averaged positive CCPM stoichiometry": pybamm.x_average(theta_CCPM),
            "CCPM Branch A mass": m_a,
            "CCPM Branch B mass": m_b,
            "CCPM Branch C mass": m_c,
            "CCPM Total mass": m_tot,
            "X-averaged CCPM Branch A mass": pybamm.x_average(m_a),
            "X-averaged CCPM Branch B mass": pybamm.x_average(m_b),
            "X-averaged CCPM Branch C mass": pybamm.x_average(m_c),
            "X-averaged CCPM Total mass": pybamm.x_average(m_tot),
            f"R-averaged {domain} {phase_name}"
            "particle concentration [mol.m-3]": theta_CCPM * c_max,
        })

        # The remainder depends on lithiation rates (from the interface submodel).
        # On the first build pass R_a may not yet be present; the KeyError will be
        # caught by build_coupled_variables and this submodel retried after the
        # interface submodel has run.
        R_a = variables["CCPM Branch A lithiation rate"]
        R_b = variables["CCPM Branch B lithiation rate"]
        R_c = variables["CCPM Branch C lithiation rate"]
        
        c1_star, c_sp1, c_sp2, c2_star, mask_a, mask_b, mask_c = (
            self._branch_geometry(c_p)
        )

        def adv_flux(g, R):
            #R_pos = pybamm.Maximum(R, 0)
            R_pos=pybamm.smooth_max(R, 0, 100)
            #R_neg = pybamm.Minimum(R, 0)
            R_neg=pybamm.smooth_min(R,0,100)
            return pybamm.Upwind(g) * R_pos + pybamm.Downwind(g) * R_neg

        F_a = adv_flux(g_a, R_a)
        F_b = adv_flux(g_b, R_b)
        F_c = adv_flux(g_c, R_c)

        #sources
        S_a, S_b, S_c, J_A_to_B, J_B_to_A, J_B_to_C, J_C_to_B = self._get_transition_sources(variables, F_a, F_b, F_c) # changed the balance

        # R_a is now enforced to be 0 at c_min via the ramp in the interface
        # kinetics, so the ghost-cell flux at the left wall is zero and no
        # wall correction is needed.
        rhs_a = -pybamm.div(F_a) * mask_a
        rhs_b = -pybamm.div(F_b) * mask_b     
        rhs_c = -pybamm.div(F_c) * mask_c
        # track rhs for debug
        rhs_int_a = pybamm.Integral(rhs_a, c_p)
        rhs_int_b = pybamm.Integral(rhs_b, c_p)
        rhs_int_c = pybamm.Integral(rhs_c, c_p)        
        variables.update({
            "Branch A RHS": rhs_a,
            "Branch B RHS": rhs_b,
            "Branch C RHS": rhs_c,
            
            "Branch A PDF CCPM RHS Integrated on c_p": rhs_int_a,
            "Branch B PDF CCPM RHS Integrated on c_p": rhs_int_b,
            "Branch C PDF CCPM RHS Integrated on c_p": rhs_int_c,

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
            "CCPM source mass balance error": pybamm.Integral(S_a + S_b + S_c, c_p),


            "CCPM Total PDF sum": g_a + g_b + g_c, # Should integrate to 1,
            "X-averaged Branch A PDF CCPM": pybamm.x_average(g_a),
            "X-averaged Branch B PDF CCPM": pybamm.x_average(g_b),
            "X-averaged Branch C PDF CCPM": pybamm.x_average(g_c),
        })

        return variables

    def set_rhs(self, variables):

        self.rhs = {
            variables["Branch A PDF CCPM"]: variables["Branch A RHS"] + variables["Branch A source"],
            variables["Branch B PDF CCPM"]: variables["Branch B RHS"] + variables["Branch B source"],
            variables["Branch C PDF CCPM"]: variables["Branch C RHS"] + variables["Branch C source"],
        }
        
    def _get_transition_sources(self, variables, F_a, F_b, F_c):
        """
        Eq 18-20 from Clarke2026: source/sink terms for transitions between branches.

        Transfer magnitudes J equal the exact FV edge flux at the transition
        boundary (via divergence telescoping).  Deposit into the receiving
        branch goes into a single FV cell adjacent to the transition edge.
        """
        c_p = variables["CCPM positive particle concentration"]

        # Single-cell FV deposit: normalized box selecting one cell
        eps_c = pybamm.Scalar(1e-8) * self.param.p.prim.c_max
        c_max_eff = self.param.p.prim.c_max - eps_c
        npts = 300
        dc = (c_max_eff - eps_c) / npts

        def cell_delta(c_edge_left):
            """Normalized box for one FV cell starting at c_edge_left."""
            box = (c_p >= c_edge_left) * (c_p <= c_edge_left + dc)
            return box / pybamm.Integral(box, c_p)

        # Transition points and masks
        c1_star, c_sp1, c_sp2, c2_star, mask_a, mask_b, mask_c = (
            self._branch_geometry(c_p)
        )
        g_a = variables["Branch A PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]

        mask_b_to_c2 = (c_p <= c2_star)  # cells 0..278
        mask_b_to_c1 = (c_p >= c1_star)  # cells 21..299
        mask_c_to_b = (c_p >= c_sp2)  # cells 237..299

        # Default all fluxes to zero; only the active direction is computed
        J_A_to_B = pybamm.Scalar(0)
        J_B_to_A = pybamm.Scalar(0)
        J_B_to_C = pybamm.Scalar(0)
        J_C_to_B = pybamm.Scalar(0)

        if self.mode == "discharge":
            # A -> B -> C
            dB_sp1 = cell_delta(c_sp1)          # A->B: into cell 63 of B
            dC_c2  = cell_delta(c2_star)        # B->C: into cell 279 of C
            J_A_to_B = pybamm.Integral(pybamm.div(F_a) * mask_a, c_p)       # = F_a(c_sp1)
            J_B_to_C = pybamm.Integral(pybamm.div(F_b) * mask_b_to_c2, c_p) # = F_b(c2_star)

            S_a = pybamm.Scalar(0) * g_a
            S_b = J_A_to_B * dB_sp1
            S_c = J_B_to_C * dC_c2

        elif self.mode == "charge":
            dA_c1  = cell_delta(c1_star - dc)   # B->A: into cell 20 of A
            dB_sp2 = cell_delta(c_sp2 - dc)     # C->B: into cell 236 of B
            J_B_to_A = pybamm.Integral(pybamm.div(F_b) * mask_b_to_c1, c_p) # = -F_b(c1_star)
            J_C_to_B = pybamm.Integral(pybamm.div(F_c) * mask_c_to_b, c_p)  # = -F_c(c_sp2)

            S_a = J_B_to_A * dA_c1
            S_b = J_C_to_B * dB_sp2
            S_c = pybamm.Scalar(0) * g_c

        else:
            raise ValueError(f"Unknown mode '{self.mode}', expected 'discharge' or 'charge'")

        return S_a, S_b, S_c, J_A_to_B, J_B_to_A, J_B_to_C, J_C_to_B
    
    def set_boundary_conditions(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        zero = pybamm.Scalar(0)
        self.boundary_conditions = {
            # Physical walls use "zero_flux": ghost nodes for upwind stencil,
            # but divergence zeros the boundary flux → true zero-flux wall.

            g_a: {"left": (zero, "zero_flux"), "right": (zero, "zero_flux")},
            g_b: {"left": (zero, "zero_flux"), "right": (zero, "zero_flux")},
            g_c: {"left": (zero, "zero_flux"), "right": (zero, "zero_flux")},
        }
    
    def set_initial_conditions(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        c_max_raw = self.param.p.prim.c_max
        eps_c = pybamm.Scalar(1e-8) * c_max_raw
        c_min_domain = eps_c
        c_max_domain = c_max_raw - eps_c

        # Snap to exact cell edges on the truncated domain, consistent
        # with _branch_geometry: edge k = c_min + k/npts * (c_max - c_min)
        npts = 300
        L = c_max_domain - c_min_domain
        c_sp1 = c_min_domain + pybamm.Scalar(63 / npts) * L
        c_sp2 = c_min_domain + pybamm.Scalar(237 / npts) * L

        c_p = variables["CCPM positive particle concentration"]
        c_init_particle = self.param.p.prim.c_init  # r domain
        # collapse to electrode (r-average), then broadcast into concentration domain
        c_init_x = pybamm.r_average(c_init_particle)  # positive electrode domain
        variables.update({"Initial CCPM particle concentration": c_init_x})

        # Regularised Dirac delta: Gamma(2,λ) anchored at the domain boundary,
        # with compact support via branch-edge damping and Heaviside mask.
        # g(c) ∝ (c - c_min) * (c_sp1 - c) * exp(-(c - c_min)/λ) * 𝟙(c ≤ c_sp1)
        #
        # Two-pass λ correction: analytical λ₀ gives biased discrete mean on
        # coarse grids. We compute the discrete mean of the trial PDF, then
        # rescale λ so the final discrete mean = c_init exactly.

        def _build_raw_pdf_A(lam_):
            shifted_ = c_p - c_min_domain
            return shifted_ * (c_sp1 - c_p) * pybamm.exp(-shifted_ / lam_) * (c_p <= c_sp1)

        def _build_raw_pdf_C(mu_, sigma_):
            """Tight Gaussian (Dirac-delta approximation) centred on mu_.
            g(c) ∝ exp(-(c - μ)²/(2σ²))  with σ ~ 2·dc so it becomes
            zero within a few cells of c_init."""
            return pybamm.exp(-((c_p - mu_) ** 2) / (2 * sigma_ ** 2))

        if self.initial_branch == "A":
            lam0 = (c_init_x - c_min_domain) / 2
            trial = _build_raw_pdf_A(lam0)
            # discrete mean of trial PDF
            mu_trial = pybamm.Integral(trial * c_p, c_p) / pybamm.Integral(trial, c_p)
            # rescale λ: mean ∝ λ → λ_corrected = λ₀ * target_shift / actual_shift
            lam = lam0 * (c_init_x - c_min_domain) / (mu_trial - c_min_domain)
            raw_pdf = _build_raw_pdf_A(lam)

        elif self.initial_branch == "C":
            # Tight Gaussian approximating a Dirac delta at c_init.
            # σ = 2·dc  →  becomes zero ~4-5 cells from c_init.
            dc = L / npts
            sigma = 2 * dc
            raw_pdf = _build_raw_pdf_C(c_init_x, sigma)

        else:
            raise ValueError(
                f"Invalid CCPM initial_branch='{self.initial_branch}'. "
                "Use 'A' or 'C'."
            )

        # Normalize so the integral over c_p is 1 (Particle Conservation)
        norm = pybamm.Integral(raw_pdf, c_p)
        initial_pdf = raw_pdf / norm

        zero_pdf = 0 * initial_pdf

        if self.initial_branch == "A":
            self.initial_conditions = {
                g_a: initial_pdf,
                g_b: zero_pdf,
                g_c: zero_pdf,
            }
        elif self.initial_branch == "C":
            self.initial_conditions = {
                g_a: zero_pdf,
                g_b: zero_pdf,
                g_c: initial_pdf,
            }
    