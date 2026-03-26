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
    ):
        super().__init__(
            param,
            domain,
            options=options,
            phase=phase,
        )
        
        
        self.c_p_space = pybamm.standard_spatial_vars.c_p
        self.initial_branch = initial_branch.upper()
        
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
        
        def adv_flux(g, R):
            R_pos = pybamm.Maximum(R, 0)
            R_neg = pybamm.Minimum(R, 0)
            return pybamm.Upwind(g) * R_pos + pybamm.Downwind(g) * R_neg

        F_a = adv_flux(g_a, R_a)
        F_b = adv_flux(g_b, R_b)
        F_c = adv_flux(g_c, R_c)

        rhs_a = -pybamm.div(F_a)
        rhs_b = -pybamm.div(F_b)
        rhs_c = -pybamm.div(F_c)
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
        Mass leaves branch A and enters B when crossing spinodal point 1, and leaves B and enters C when crossing c2*. 
        Ensure Monodirectionaly and mass conservation by making the source for one branch the negative of the other.
        """
        
        def regularized_delta(c_target, mask):
            eps_c = pybamm.Scalar(1e-8) * self.param.p.prim.c_max
            c_min = eps_c
            c_max_eff = self.param.p.prim.c_max - eps_c
            dc = (c_max_eff - c_min) / (300 - 1)
            sigma = 1.5 * dc  # regularization width (≈9 cells FWHM)
            ker = pybamm.exp(-((c_p - c_target) ** 2) / (2 * sigma ** 2)) * mask
            return ker / pybamm.Integral(ker, c_p)  # normalize: integral = 1
        
        c_p= variables["CCPM positive particle concentration"]
        
        # get transition points
        # Lithiation: A->B at c_sp1, B->C at c_2*
        # Delithiation: C->B at c_sp2, B->A at c_s1*
        c1_star, c_sp1, c_sp2, c2_star, mask_a, mask_b, mask_c = self._branch_geometry(c_p)
        g_a= variables["Branch A PDF CCPM"]
        g_b= variables["Branch B PDF CCPM"]
        g_c= variables["Branch C PDF CCPM"]
        R_a = variables["CCPM Branch A lithiation rate"]
        R_b = variables["CCPM Branch B lithiation rate"]
        R_c = variables["CCPM Branch C lithiation rate"]
        
        # Logic Transition from A to B when crossing spinodal point 1
        # Make all fluxes positive to include signal reasoning in the source term.
        # If R_a > 0 (rate in lithium poor phase is positive-> lithiation), and c=c_sp1, mass should jump to B
        
        ##flux_A_to_B = g_a * pybamm.Maximum(0, R_a)
        # If R_b > 0 (rate in mixed phase is positive-> lithiation), and c=c_2*, mass should jump to C
        ##flux_B_to_C = g_b * pybamm.Maximum(0, R_b)
        
        # Delithiation path
        # If R_c < 0 (rate in lithium rich phase is negative-> delithiation), and c=c_sp2, mass should jump back to B
        ##flux_C_to_B = g_c * pybamm.Maximum(-R_c, 0)
        # If R_b < 0 (rate in mixed phase is negative-> delithiation), and c=c_1*, mass should jump
        ##flux_B_to_A = g_b * pybamm.Maximum(-R_b, 0)

        # localized kernels on the correct source/target branches
        dA_sp1 = regularized_delta(c_sp1, mask_a)
        dB_sp1 = regularized_delta(c_sp1, mask_b)

        dB_c1 = regularized_delta(c1_star, mask_b)
        dA_c1 = regularized_delta(c1_star, mask_a)

        dB_c2 = regularized_delta(c2_star, mask_b)
        dC_c2 = regularized_delta(c2_star, mask_c)

        dC_sp2 = regularized_delta(c_sp2, mask_c)
        dB_sp2 = regularized_delta(c_sp2, mask_b)

        # scalar transfer magnitudes
        J_A_to_B = pybamm.Integral(dA_sp1 * g_a* pybamm.Maximum(R_a, 0), c_p)
        J_B_to_A = pybamm.Integral(dB_c1 * g_b * pybamm.Maximum(-R_b, 0), c_p)
        J_B_to_C = pybamm.Integral(dB_c2 * g_b * pybamm.Maximum(R_b, 0), c_p)
        J_C_to_B = pybamm.Integral(dC_sp2 * g_c * pybamm.Maximum(-R_c, 0), c_p)

        # conservative paired source/sink terms
        S_a = -J_A_to_B * dA_sp1 + J_B_to_A * dA_c1
        S_b = J_A_to_B * dB_sp1 - J_B_to_A * dB_c1 - J_B_to_C * dB_c2 + J_C_to_B * dB_sp2
        S_c = J_B_to_C * dC_c2 - J_C_to_B * dC_sp2

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
        c_init_cp = pybamm.PrimaryBroadcast(c_init_x, "CCPM positive particle concentration")

        # Define a narrow Gaussian to represent the initial state
        #c_init_cp = 0.01*c_max
        sigma_a = 0.003 * c_sp1
        initial_distribution_a = pybamm.exp(-((c_p - c_init_cp) ** 2) / (2 * sigma_a ** 2)) * (c_p <= c_sp1)
        
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