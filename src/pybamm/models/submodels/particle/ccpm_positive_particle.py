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
        initial_branch="C",
        source_method="flux_form",
    ):
        super().__init__(
            param,
            domain,
            options=options,
            phase=phase,
        )
        
        
        self.c_p_space = pybamm.standard_spatial_vars.c_p
        self.initial_branch = initial_branch.upper()
        self.source_method = source_method
        
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
        c_max = self.param.p.prim.c_max

        c1_star = pybamm.Scalar(0.0710) * c_max
        c_sp1   = pybamm.Scalar(0.2113) * c_max
        c_sp2   = pybamm.Scalar(0.7887) * c_max
        c2_star = pybamm.Scalar(0.9290) * c_max

        # same boolean-mask style you already used
        mask_a = (c_p <= c_sp1)
        mask_b = (c_p >= c1_star) * (c_p <= c2_star)
        mask_c = (c_p >= c_sp2)

        return c1_star, c_sp1, c_sp2, c2_star, mask_a, mask_b, mask_c
    
    def get_coupled_variables(self, variables):
        # Retrieve our PDFs and the concentration grid
        g_a_unmasked = variables["Branch A PDF CCPM"]
        g_b_unmasked = variables["Branch B PDF CCPM"]
        g_c_unmasked = variables["Branch C PDF CCPM"]
        c_p = variables["CCPM positive particle concentration"]
        c_max = self.param.p.prim.c_max

        # mask density functions
        c1_star, c_sp1, c_sp2, c2_star, mask_a, mask_b, mask_c = self._branch_geometry(c_p)
        g_a = g_a_unmasked * mask_a
        g_b = g_b_unmasked * mask_b
        g_c = g_c_unmasked * mask_c
        # 2. Calculate the average stoichiometry for the CCPM
        # This is the integral of (c * total_PDF) / c_max
        # This replaces the old 'theta_a_ref' logic with real PDF physics
        theta_CCPM= pybamm.Integral((g_a + g_b + g_c) * (c_p / c_max), c_p)

        
        # mass per branch
        m_a = pybamm.Integral(g_a, c_p)
        m_b = pybamm.Integral(g_b, c_p)
        m_c = pybamm.Integral(g_c, c_p)
        m_tot = m_a + m_b + m_c
        
        # include rhs_c
        cbar_c = pybamm.Integral(g_c * c_p, c_p) / pybamm.Maximum(m_c, 1e-16) # average concentration in branch C
        m_a_raw = pybamm.Integral(variables["Branch A PDF CCPM"], c_p)
        m_b_raw = pybamm.Integral(variables["Branch B PDF CCPM"], c_p)
        m_c_raw = pybamm.Integral(variables["Branch C PDF CCPM"], c_p)
        m_tot_raw = m_a_raw + m_b_raw + m_c_raw
        variables.update({
            "CCPM Unmasked Mass Branch A": m_a_raw,
            "CCPM Unmasked Mass Branch B": m_b_raw,
            "CCPM Unmasked Mass Branch C": m_c_raw,
            "CCPM Unmasked Total Mass": m_tot_raw,
            "Branch A PDF CCPM Masked": g_a,
            "Branch B PDF CCPM Masked": g_b,
            "Branch C PDF CCPM Masked": g_c,
            "CCPM Branch A mass": m_a,
            "CCPM Branch B mass": m_b,
            "CCPM Branch C mass": m_c,
            "CCPM Total mass": m_tot,
            "Positive CCPM stoichiometry": theta_CCPM,
            "X-averaged positive CCPM stoichiometry": pybamm.x_average(theta_CCPM),
            "CCPM Total PDF sum": g_a + g_b + g_c, # Should integrate to 1,
            "CCPM Branch C mean concentration": cbar_c,
            "X-averaged Branch A PDF CCPM": pybamm.x_average(g_a),
            "X-averaged Branch B PDF CCPM": pybamm.x_average(g_b),
            "X-averaged Branch C PDF CCPM": pybamm.x_average(g_c),
            "X-averaged CCPM Branch A mass": pybamm.x_average(m_a),
            "X-averaged CCPM Branch B mass": pybamm.x_average(m_b),
            "X-averaged CCPM Branch C mass": pybamm.x_average(m_c),
            "X-averaged CCPM Total mass": pybamm.x_average(m_tot),
        })

        # Provide standard concentration variable equivalents for downstream
        # submodels (TotalConcentration, etc.). CCPM has no radial diffusion;
        # θ_CCPM * c_max is the physically equivalent average concentration.
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        c_s_rav_equiv = theta_CCPM * c_max
        variables.update({
            f"R-averaged {domain} {phase_name}"
            "particle concentration [mol.m-3]": c_s_rav_equiv,
        })

        return variables

    def set_rhs(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        R_a = variables["CCPM Branch A lithiation rate"]
        R_b = variables["CCPM Branch B lithiation rate"]
        R_c = variables["CCPM Branch C lithiation rate"]
        c_p= variables["CCPM positive particle concentration"]

        # Mask PDFs to their branch domains before computing the flux.
        # This prevents advective mass from leaking beyond the branch
        # boundaries; the outgoing flux at each boundary becomes the
        # natural mass-transfer rate to the neighbouring branch.
        c1_star, c_sp1, c_sp2, c2_star, mask_a, mask_b, mask_c = (
            self._branch_geometry(c_p)
        )
        g_a_m = g_a * mask_a
        g_b_m = g_b * mask_b
        g_c_m = g_c * mask_c

        def adv_flux(g, R):
            R_pos = pybamm.Maximum(R, 0)
            R_neg = pybamm.Minimum(R, 0)
            return pybamm.Upwind(g) * R_pos + pybamm.Downwind(g) * R_neg

        F_a = adv_flux(g_a_m, R_a)
        F_b = adv_flux(g_b_m, R_b)
        F_c = adv_flux(g_c_m, R_c)

        # Mask the divergence so cells outside a branch's domain
        # have zero RHS (mass cannot appear beyond the boundary).
        rhs_a = -pybamm.div(F_a) * mask_a
        rhs_b = -pybamm.div(F_b) * mask_b
        rhs_c = -pybamm.div(F_c) * mask_c
        # track rhs for debug
        rhs_int = pybamm.Integral(rhs_a, c_p)

        #sources
        
        S_a, S_b, S_c, J_A_to_B, J_B_to_A, J_B_to_C, J_C_to_B = self._get_transition_sources(variables, F_a, F_b, F_c) # changed the balance
        variables.update({
            "Branch A PDF CCPM RHS": rhs_int,
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
            })

        self.rhs = {
            g_a: rhs_a + S_a,
            g_b: rhs_b + S_b,
            g_c: rhs_c + S_c,
        }
        
    def _get_transition_sources(self, variables, F_a, F_b, F_c):
        """
        Eq 18-20 from Clarke2026: source/sink terms for transitions between branches.

        Uses flux-form extraction: the scalar transfer magnitude J equals the
        actual upwind advective flux evaluated at the transition point (via
        EvaluateAt), ensuring exact consistency with the advection scheme.
        Deposition into the receiving branch uses a regularized Gaussian delta
        to spread mass over a few cells.
        """
        c_p = variables["CCPM positive particle concentration"]

        # Deposition kernel (normalized Gaussian)
        def regularized_delta(c_target, mask):
            eps_c = pybamm.Scalar(1e-8) * self.param.p.prim.c_max
            c_min = eps_c
            c_max_eff = self.param.p.prim.c_max - eps_c
            dc = (c_max_eff - c_min) / (300 - 1)
            sigma = 1.5 * dc
            ker = pybamm.exp(-((c_p - c_target) ** 2) / (2 * sigma ** 2)) * mask
            return ker / pybamm.Integral(ker, c_p)

        # Transition points and masks
        c1_star, c_sp1, c_sp2, c2_star, mask_a, mask_b, mask_c = (
            self._branch_geometry(c_p)
        )
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        R_a = variables["CCPM Branch A lithiation rate"]
        R_b = variables["CCPM Branch B lithiation rate"]
        R_c = variables["CCPM Branch C lithiation rate"]

        if self.source_method == "flux_form":
            # --- Flux-form scalar transfer magnitudes ---
            # Evaluate g and R at the transition point (cell center nearest),
            # forming the upwind flux g * max(R, 0). Matches advection exactly.
            J_A_to_B = (
                pybamm.EvaluateAt(g_a, c_sp1)
                * pybamm.Maximum(pybamm.EvaluateAt(R_a, c_sp1), 0)
            )
            J_B_to_A = (
                pybamm.EvaluateAt(g_b, c1_star)
                * pybamm.Maximum(-pybamm.EvaluateAt(R_b, c1_star), 0)
            )
            J_B_to_C = (
                pybamm.EvaluateAt(g_b, c2_star)
                * pybamm.Maximum(pybamm.EvaluateAt(R_b, c2_star), 0)
            )
            J_C_to_B = (
                pybamm.EvaluateAt(g_c, c_sp2)
                * pybamm.Maximum(-pybamm.EvaluateAt(R_c, c_sp2), 0)
            )
        else:
            # --- Old Gaussian-integral scalar transfer magnitudes ---
            dA_ext = regularized_delta(c_sp1, mask_a)
            dB_ext_c1 = regularized_delta(c1_star, mask_b)
            dB_ext_c2 = regularized_delta(c2_star, mask_b)
            dC_ext = regularized_delta(c_sp2, mask_c)
            J_A_to_B = pybamm.Integral(dA_ext * g_a * pybamm.Maximum(R_a, 0), c_p)
            J_B_to_A = pybamm.Integral(dB_ext_c1 * g_b * pybamm.Maximum(-R_b, 0), c_p)
            J_B_to_C = pybamm.Integral(dB_ext_c2 * g_b * pybamm.Maximum(R_b, 0), c_p)
            J_C_to_B = pybamm.Integral(dC_ext * g_c * pybamm.Maximum(-R_c, 0), c_p)

        # --- Deposition kernels (receiving branch only) ---
        # The advective flux is now truncated at the branch boundary
        # (masked PDFs + masked divergence), so mass that reaches
        # the boundary naturally leaves the donor branch via the
        # divergence term.  Only the *deposition* into the receiving
        # branch needs an explicit source.
        dB_sp1 = regularized_delta(c_sp1, mask_b)  # A->B deposit into B at c_sp1
        dA_c1  = regularized_delta(c1_star, mask_a) # B->A deposit into A at c1*
        dC_c2  = regularized_delta(c2_star, mask_c) # B->C deposit into C at c2*
        dB_sp2 = regularized_delta(c_sp2, mask_b)   # C->B deposit into B at c_sp2

        # --- Source terms (deposition only, no removal) ---
        S_a = J_B_to_A * dA_c1
        S_b = J_A_to_B * dB_sp1 + J_C_to_B * dB_sp2
        S_c = J_B_to_C * dC_c2

        return S_a, S_b, S_c, J_A_to_B, J_B_to_A, J_B_to_C, J_C_to_B
    
    def set_boundary_conditions(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        zero = pybamm.Scalar(0)
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
        c_sp1 = 0.2113*c_max #TODO: cast to scalar?
        c_sp2=0.7887*c_max #TODO
        
        
        c_p = variables["CCPM positive particle concentration"] 
        c_init_particle=self.param.p.prim.c_init # r domain
        # collapse to electrode (r-average), then broadcast into concentration domain
        c_init_x = pybamm.r_average(c_init_particle) # positive electrode domain
        variables.update({ "Initial CCPM particle concentration": c_init_x})

        # Asymmetric initial PDF: Gamma(2,λ) shape ensures g(c_min)=0
        # while keeping the mode at c_init. Avoids left-tail clipping that
        # a symmetric Gaussian would suffer when c_init is near the boundary.
        eps_c = pybamm.Scalar(1e-8) * c_max
        c_min_domain = eps_c
        c_init_cp = pybamm.PrimaryBroadcast(c_init_x, "CCPM positive particle concentration")
        # lam numerically corrected via brentq on 300-pt grid so that the
        # discrete trapezoid mean equals c_init exactly (theta=0.0038, Prada2013).
        # Analytical lam=(c_init-c_min)/2 gives discrete mean ~theta=0.0047 due
        # to sub-grid resolution (c_init ≈ 1.1*dc). Corrected: lam=27.7, dc=76.3.
        lam = pybamm.Scalar(1.214e-3) * c_max
        shifted = c_p - c_min_domain
        initial_distribution_a = (
            shifted * pybamm.exp(-shifted / lam) * (c_p <= c_sp1)
        )
        
        # Normalize so the integral over c_p is 1 (Particle Conservation)
        # We start with everything in Branch C for a lithiated cathode
        # or Branch A for a delithiated one.
        norm = pybamm.Integral(initial_distribution_a, c_p)
        initial_pdf = initial_distribution_a / norm # normalised
        
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
        else:
            raise ValueError(
                f"Invalid CCPM initial_branch='{self.initial_branch}'. "
                "Use 'A' or 'C'."
            )