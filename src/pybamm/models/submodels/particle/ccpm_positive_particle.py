#C:\Users\dottavianomo\programming\PyBaMM_acm\src\pybamm\models\submodels\particle\ccpm_positive_particle.py
from .base_particle import BaseParticle
import numpy as np
from scipy.optimize import brentq
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
        mode='discharge',
        R_p_float=None,
        npts=300,
        c_max_float=None,
        solid_diffusion="none",
        N_shell=3,
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
        self.solid_diffusion = solid_diffusion
        self.N_shell = N_shell  # number of chi intervals (N_shell+1 nodes)

        if R_p_float is not None and c_max_float is not None:
            consts = self.compute_branch_constants(R_p_float, npts, c_max_float)
            self.omega   = consts["omega"]
            self.c1_star = consts["c1_star"]
            self.c_sp1   = consts["c_sp1"]
            self.c_sp2   = consts["c_sp2"]
            self.c2_star = consts["c2_star"]
            self.npts    = consts["npts"]
        else:
            raise ValueError(
                "R_p_float and c_max_float are required for CCPMPositiveParticle"
            )

    @staticmethod
    def compute_branch_constants(R_p_float, npts, c_max_float):
        """Compute Ω, spinodal, and binodal concentrations from particle radius.

        Thermodynamic quantities computed on true domain [0, c_max].
        Results snapped to nearest FV cell edge on truncated domain
        [eps_c, c_max - eps_c].

        Parameters
        ----------
        R_p_float : float
            Particle radius [m].
        npts : int
            Number of FV cells on the concentration grid.
        c_max_float : float
            Maximum lithium concentration [mol/m³].

        Returns
        -------
        dict with omega (float), c1_star, c_sp1, c_sp2, c2_star
        (pybamm.Scalar), npts (int).
        """
        # Ferguson & Bazant 2014, Electrochim. Acta 146, 89-97
        omega = 4.5 - (2.5 / 0.18e9) * 3.0 / R_p_float

        # Spinodals: ∂²G/∂x² = 0 → θ_sp = (1 ± sqrt(1 - 2/Ω)) / 2
        disc = 1.0 - 2.0 / omega
        if disc <= 0:
            raise ValueError(
                f"Ω={omega:.4f} < 2: no miscibility gap (particle too small?)"
            )
        sqrt_disc = np.sqrt(disc)
        theta_sp1 = (1.0 - sqrt_disc) / 2.0
        theta_sp2 = (1.0 + sqrt_disc) / 2.0

        # Binodals: μ(θ) = 0 → ln(θ/(1-θ)) + Ω(1-2θ) = 0
        def mu_eq(x):
            return np.log(x / (1.0 - x)) + omega * (1.0 - 2.0 * x)

        theta_bin1 = brentq(mu_eq, 1e-12, theta_sp1 - 1e-12)
        theta_bin2 = 1.0 - theta_bin1

        # Snap to nearest cell EDGE on truncated FV domain
        eps_c = 1e-8 * c_max_float
        c_min_t = eps_c
        c_max_t = c_max_float - eps_c
        L = c_max_t - c_min_t

        def snap_to_edge(theta):
            c_true = theta * c_max_float
            k = round((c_true - c_min_t) / L * npts)
            k = max(0, min(npts, k))
            return c_min_t + k / npts * L

        c1_star = snap_to_edge(theta_bin1)
        c_sp1   = snap_to_edge(theta_sp1)
        c_sp2   = snap_to_edge(theta_sp2)
        c2_star = snap_to_edge(theta_bin2)

        print(f"[CCPM] Omega={omega:.4f}, R_p={R_p_float*1e9:.1f}nm, npts={npts}")
        print(f"  spinodals: th_sp1={theta_sp1:.6f} -> edge "
              f"{round((snap_to_edge(theta_sp1) - c_min_t) / L * npts)}")
        print(f"             th_sp2={theta_sp2:.6f} -> edge "
              f"{round((snap_to_edge(theta_sp2) - c_min_t) / L * npts)}")
        print(f"  binodals:  th_b1 ={theta_bin1:.6f} -> edge "
              f"{round((snap_to_edge(theta_bin1) - c_min_t) / L * npts)}")
        print(f"             th_b2 ={theta_bin2:.6f} -> edge "
              f"{round((snap_to_edge(theta_bin2) - c_min_t) / L * npts)}")

        return {
            "omega": omega,
            "c1_star": pybamm.Scalar(c1_star),
            "c_sp1":   pybamm.Scalar(c_sp1),
            "c_sp2":   pybamm.Scalar(c_sp2),
            "c2_star": pybamm.Scalar(c2_star),
            "npts":    npts,
        }
        
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

        # ── Core-shell state variables on the c_p grid (nondimensional) ──
        if self.solid_diffusion == "core_shell":
            cp_domains = {
                "primary": "CCPM positive particle concentration",
                "secondary": "positive electrode",
                "tertiary": "current collector",
            }
            # r_hat = r_p / R_p  (O(1))
            shell_r_hat = pybamm.Variable(
                "Shell phase front r_hat", domains=cp_domains
            )
            variables["Shell phase front r_hat"] = shell_r_hat
            # Dimensional output for plotting/diagnostics
            R_p_fund = self.param.p.prim.R
            variables["Shell phase front r_p"] = R_p_fund * shell_r_hat

            c_max_fund = self.param.p.prim.c_max
            for j in range(1, self.N_shell + 1):
                # c_hat_j = c_j / c_max  (O(1))
                var = pybamm.Variable(
                    f"Shell concentration hat chi_{j}", domains=cp_domains
                )
                variables[f"Shell concentration hat chi_{j}"] = var
                # Dimensional output
                variables[f"Shell concentration chi_{j}"] = c_max_fund * var

        return variables
    def _branch_geometry(self, c_p):
        # Branch boundaries computed in __init__ via compute_branch_constants
        # (analytical spinodals/binodals from Ω, snapped to cell edges)
        mask_a = (c_p <= self.c_sp1)
        mask_b = (c_p >= self.c1_star) * (c_p <= self.c2_star)
        mask_c = (c_p >= self.c_sp2)

        return self.c1_star, self.c_sp1, self.c_sp2, self.c2_star, mask_a, mask_b, mask_c

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

        # ── Core-shell RHS: nondimensional Stefan + shell PDE ──
        # State variables: r_hat = r_p/R_p, c_hat_j = c_j/c_max (all O(1))
        # Nondim RHS eliminates ~30 orders of magnitude spread in Jacobian.
        if self.solid_diffusion == "core_shell":
            F_const = pybamm.constants.F
            R_p = self.param.p.prim.R
            c_p_max = self.param.p.prim.c_max
            c_sp1_val = self.c_sp1
            c2_star_val = self.c2_star
            # Nondimensional branch constants
            theta_sp1 = c_sp1_val / c_p_max
            theta_2star = c2_star_val / c_p_max
            N = self.N_shell
            dchi = 1.0 / N

            # Read nondimensional state variables
            shell_r_hat = variables["Shell phase front r_hat"]
            shell_chat = {}  # j=1..N: chat_1 near core, chat_N at surface
            for j in range(1, N + 1):
                shell_chat[j] = variables[f"Shell concentration hat chi_{j}"]

            # Clamp r_hat for FDM geometry to [0.01, 0.99].
            # In bins with g_b=0 the state r_hat can drift; the clamp
            # breaks the positive-feedback runaway (ell_hat->0 => drhat_dt->inf)
            # without coupling to g_b (which would stiffen the Jacobian).
            r_hat_safe = pybamm.maximum(pybamm.minimum(shell_r_hat, pybamm.Scalar(0.99)), pybamm.Scalar(0.01))
            ell_hat = pybamm.Scalar(1.0) - r_hat_safe

            # Diffusivity (constant, evaluated at theta_2* for beta phase)
            T_p = variables["Positive electrode temperature [K]"]
            T_p_bc = pybamm.PrimaryBroadcast(T_p, "CCPM positive particle concentration")
            c_p_max_trunc = c_p_max - pybamm.Scalar(1e-8) * c_p_max
            D_s = self.param.p.prim.D(c2_star_val / c_p_max_trunc, T_p_bc)

            # Characteristic diffusion rate: D_s / R_p^2
            D_hat = D_s / R_p**2

            # j_tr_b from interface submodel
            j_tr_b = variables["CCPM Branch B interfacial current density [A.m-2]"]

            # === Boundary values (nondimensional) ===
            # chi=0 (core-shell interface): Dirichlet chat_0 = theta_2*
            chat_0 = theta_2star

            # chi=1 (surface): ghost node from Neumann BC
            #   D_s*c_max/(R_p*ell_hat) * (chat_{N+1} - chat_{N-1})/(2*dchi) = -j_tr_b/F
            #   chat_{N+1} = chat_{N-1} - 2*dchi*ell_hat*R_p*j_tr_b/(F*D_s*c_max)
            chat_ghost = shell_chat[N - 1] - pybamm.Scalar(2 * dchi) * ell_hat * R_p * j_tr_b / (F_const * D_s * c_p_max)

            # === Stefan condition: dr_hat/dt ===
            # Correct sign: during lithiation chat_1 > theta_2* => core shrinks (drhat < 0)
            # dr_hat/dt = -D_hat * c_max * (chat_1 - theta_2*) / ((c2* - c_sp1) * ell_hat * dchi)
            drhat_dt = -D_hat * c_p_max * (shell_chat[1] - chat_0) / (
                (c2_star_val - c_sp1_val) * ell_hat * pybamm.Scalar(dchi)
            )
            # Suppress shell ODE outside branch-B mask.
            drhat_dt = drhat_dt * mask_b

            # ── Smooth bound enforcement: prevent r_hat leaving [0, 1] ──
            # upper_gate → 0 as r_hat → 1  (kills growth)
            # lower_gate → 0 as r_hat → 0  (kills shrinkage)
            upper_gate = pybamm.Scalar(1.0) - pybamm.sigmoid(
                shell_r_hat, pybamm.Scalar(0.98), 200
            )
            lower_gate = pybamm.sigmoid(
                shell_r_hat, pybamm.Scalar(0.02), 200
            )
            drhat_pos = pybamm.maximum(drhat_dt, pybamm.Scalar(0.0))
            drhat_neg = pybamm.minimum(drhat_dt, pybamm.Scalar(0.0))
            drhat_dt = drhat_pos * upper_gate + drhat_neg * lower_gate

            variables["Shell r_hat RHS"] = drhat_dt
            variables["Shell r_p RHS"] = R_p * drhat_dt  # dimensional, for diagnostics

            # === Interior + surface node ODEs (nondimensional) ===
            for j in range(1, N + 1):
                chi_j = j * dchi
                # r_hat_j = r_hat_safe + chi_j * ell_hat
                r_hat_j = r_hat_safe + pybamm.Scalar(chi_j) * ell_hat

                # Neighbor nondim concentrations
                chat_jm1 = chat_0 if j == 1 else shell_chat[j - 1]
                chat_jp1 = chat_ghost if j == N else shell_chat[j + 1]
                chat_j = shell_chat[j]

                # Second derivative: (chat_{j+1} - 2*chat_j + chat_{j-1}) / dchi^2
                d2c = (chat_jp1 - 2 * chat_j + chat_jm1) / pybamm.Scalar(dchi**2)

                # First derivative: (chat_{j+1} - chat_{j-1}) / (2*dchi)
                dc = (chat_jp1 - chat_jm1) / pybamm.Scalar(2 * dchi)

                # Spherical term: 2*ell_hat/r_hat_j * dc
                r_hat_j_safe = pybamm.smooth_max(r_hat_j, pybamm.Scalar(1e-6), 1e10)
                sph_term = 2 * ell_hat / r_hat_j_safe * dc

                # Diffusion: D_hat/ell_hat^2 * (d2c + sph_term)
                diffusion = D_hat / ell_hat**2 * (d2c + sph_term)

                # Convection from moving boundary:
                # (1 - chi_j) * drhat_dt / ell_hat * dc
                convection = pybamm.Scalar(1 - chi_j) * drhat_dt / ell_hat * dc

                rhs_j = (diffusion + convection) * mask_b

                variables[f"Shell chat_{j} RHS"] = rhs_j

        return variables

    def set_rhs(self, variables):

        self.rhs = {
            variables["Branch A PDF CCPM"]: variables["Branch A RHS"] + variables["Branch A source"],
            variables["Branch B PDF CCPM"]: variables["Branch B RHS"] + variables["Branch B source"],
            variables["Branch C PDF CCPM"]: variables["Branch C RHS"] + variables["Branch C source"],
        }

        # ── Core-shell ODEs (nondimensional) ──
        if self.solid_diffusion == "core_shell":
            self.rhs[variables["Shell phase front r_hat"]] = variables["Shell r_hat RHS"]
            for j in range(1, self.N_shell + 1):
                self.rhs[variables[f"Shell concentration hat chi_{j}"]] = variables[f"Shell chat_{j} RHS"]
        
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
        npts = self.npts
        dc = (c_max_eff - eps_c) / npts

        def cell_delta(c_edge_left):
            """Normalized box for one FV cell starting at c_edge_left."""
            box = (c_p >= c_edge_left) * (c_p <= c_edge_left + dc)
            return box / dc
        
        def cell_delta_charge(c_edge_right):
            """Normalized box for one FV cell starting at c_edge_left."""
            box = (c_p <= c_edge_right) * (c_p >= c_edge_right - dc)
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

        # default all fluxes to zero, so only the active direction is computed
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
            dA_c1  = cell_delta_charge(c1_star)   # B->A: into cell 20 of A
            dB_sp2 = cell_delta_charge(c_sp2)     # C->B: into cell 236 of B
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

        # Branch boundaries from compute_branch_constants (snapped to cell edges)
        c_sp1 = self.c_sp1
        c_sp2 = self.c_sp2

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

        def _build_raw_pdf_C(lam_):
            """Mirror of branch A: Gamma(2,λ) anchored at c_max, support ≥ c_sp2.
            g(c) ∝ (c_max - c) * (c - c_sp2) * exp(-(c_max - c)/λ) * 𝟙(c ≥ c_sp2)
            """
            shifted_ = c_max_domain - c_p
            return shifted_ * (c_p - c_sp2) * pybamm.exp(-shifted_ / lam_) * (c_p >= c_sp2)

        if self.initial_branch == "A":
            lam0 = (c_init_x - c_min_domain) / 2
            trial = _build_raw_pdf_A(lam0)
            # discrete mean of trial PDF
            mu_trial = pybamm.Integral(trial * c_p, c_p) / pybamm.Integral(trial, c_p)
            # rescale λ: mean ∝ λ → λ_corrected = λ₀ * target_shift / actual_shift
            lam = lam0 * (c_init_x - c_min_domain) / (mu_trial - c_min_domain)
            raw_pdf = _build_raw_pdf_A(lam)

        elif self.initial_branch == "C":
            # Mirror of branch A: Gamma(2,λ) anchored at c_max_domain.
            # Two-pass λ correction so discrete mean = c_init exactly.
            lam0 = (c_max_domain - c_init_x) / 2
            trial = _build_raw_pdf_C(lam0)
            mu_trial = pybamm.Integral(trial * c_p, c_p) / pybamm.Integral(trial, c_p)
            lam = lam0 * (c_max_domain - c_init_x) / (c_max_domain - mu_trial)
            raw_pdf = _build_raw_pdf_C(lam)

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

        # ── Core-shell ICs (nondimensional) ──
        if self.solid_diffusion == "core_shell":
            c_sp1 = self.c_sp1
            c2_star = self.c2_star
            c_max_ic = self.param.p.prim.c_max
            # Lever-rule equilibrium r_hat(c_p):
            #   vol_frac_core = (c2* - c_p) / (c2* - c_sp1)
            #   r_hat = vol_frac_core^(1/3)
            frac_eq = (c2_star - c_p) / (c2_star - c_sp1)
            frac_eq = pybamm.maximum(frac_eq, pybamm.Scalar(1e-4))
            frac_eq = pybamm.minimum(frac_eq, pybamm.Scalar(1.0 - 1e-4))
            r_hat_init = frac_eq ** (1.0 / 3.0)

            shell_r_hat = variables["Shell phase front r_hat"]
            self.initial_conditions[shell_r_hat] = r_hat_init

            # Shell nondim concentration: uniform at theta_2* = c2*/c_max
            theta_2star_field = c2_star / c_max_ic + pybamm.Scalar(0) * c_p
            for j in range(1, self.N_shell + 1):
                var = variables[f"Shell concentration hat chi_{j}"]
                self.initial_conditions[var] = theta_2star_field
    