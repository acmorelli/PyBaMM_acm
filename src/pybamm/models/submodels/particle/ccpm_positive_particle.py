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
        source_method='gaussian'

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
        eps=1e-8*self.param.p.prim.c_max 
        c_max = self.param.p.prim.c_max -eps

        # Snapped to exact cell edges for npts=300 (edge indices 21,63,237,279)
        npts = 300
        c1_star = pybamm.Scalar(21 / npts) * c_max
        c_sp1   = pybamm.Scalar(63 / npts) * c_max
        c_sp2   = pybamm.Scalar(237 / npts) * c_max
        c2_star = pybamm.Scalar(279 / npts) * c_max

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
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        R_a = variables["CCPM Branch A lithiation rate"]
        R_b = variables["CCPM Branch B lithiation rate"]
        R_c = variables["CCPM Branch C lithiation rate"]
        dB_sp1 = cell_delta(c_sp1)          # A->B: into cell 63 of B
        dA_c1  = cell_delta(c1_star - dc)   # B->A: into cell 20 of A
        dC_c2  = cell_delta(c2_star)        # B->C: into cell 279 of C
        dB_sp2 = cell_delta(c_sp2 - dc)     # C->B: into cell 236 of B
        if self.source_method == "flux_form":
            # compute scalar fluxes at transition points
            # Evaluate g and R at the transition point (cell center nearest),
            # forming the upwind flux g * max(R, 0). Matches advection exactly.
            J_A_to_B = (
                pybamm.EvaluateAt(g_a, c_sp1)
                * pybamm.smooth_max(pybamm.EvaluateAt(R_a, c_sp1), 0,100)
            )
            J_B_to_A = (
                pybamm.EvaluateAt(g_b, c1_star)
                * pybamm.smooth_max(-pybamm.EvaluateAt(R_b, c1_star), 0,100)
            )
            J_B_to_C = (
                pybamm.EvaluateAt(g_b, c2_star)
                * pybamm.smooth_max(pybamm.EvaluateAt(R_b, c2_star), 0,100)
            )
            J_C_to_B = (
                pybamm.EvaluateAt(g_c, c_sp2)
                * pybamm.smooth_max(-pybamm.EvaluateAt(R_c, c_sp2), 0,100)
            )

            S_a = J_B_to_A * dA_c1
            S_b = J_A_to_B * dB_sp1 + J_C_to_B * dB_sp2
            S_c = J_B_to_C * dC_c2
        else:
            # Integrated-divergence approach: J equals the exact FV edge
            # flux at the transition boundary via divergence telescoping.
            # Removal is already in the RHS via -div(F)*mask; sources only deposit.
            #
            # A: zero_flux left wall => Integral(div(F_a)*(c<=c_sp1)) = F_a[63]-F_a[0] = F_a[63]
            # B: zero_flux left wall => Integral(div(F_b)*(c<=c2*))   = F_b[279]-F_b[0] = F_b[279]
            mask_b_to_c2 = (c_p <= c2_star)  # cells 0..278
            J_A_to_B = pybamm.Integral(pybamm.div(F_a) * mask_a, c_p)
            J_B_to_C = pybamm.Integral(pybamm.div(F_b) * mask_b_to_c2, c_p)
            # Charge direction (future work)
            J_B_to_A = pybamm.Scalar(0)
            J_C_to_B = pybamm.Scalar(0)

            # Deposit only
            S_a = pybamm.Scalar(0) * g_a
            S_b = J_A_to_B * dB_sp1
            S_c = J_B_to_C * dC_c2


        return S_a, S_b, S_c, J_A_to_B, J_B_to_A, J_B_to_C, J_C_to_B
    
    def set_boundary_conditions(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        zero = pybamm.Scalar(0)
        self.boundary_conditions = {
            # Physical walls use "zero_flux": ghost nodes for upwind stencil,
            # but divergence zeros the boundary flux → true zero-flux wall.
            # Transition boundaries use standard "Dirichlet" g=0.
            g_a: {"left": (zero, "zero_flux"), "right": (zero, "Dirichlet")},
            g_b: {"left": (zero, "zero_flux"), "right": (zero, "Dirichlet")},
            g_c: {"left": (zero, "Dirichlet"), "right": (zero, "zero_flux")},
        }
    
    def set_initial_conditions(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        c_max = self.param.p.prim.c_max
        npts = 300
        c_sp1 = (63 / npts) * c_max
        c_sp2 = (237 / npts) * c_max
        
        
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
        lam = pybamm.Scalar(1.3681e-3) * c_max
        shifted = c_p - c_min_domain
        initial_distribution_a = (
            shifted *2* pybamm.exp(-shifted / lam) * (c_p <= c_sp1)
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
    