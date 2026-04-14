#C:\Users\dottavianomo\programming\PyBaMM_acm\src\pybamm\models\submodels\interface\kinetics\ccpm_positive_interface.py
import pybamm
from .base_kinetics import BaseKinetics
from ...particle.ccpm_positive_particle import CCPMPositiveParticle


class CCPMPositiveInterface(BaseKinetics):
    def __init__(self, param, domain, reaction, options, phase="primary",
                 R_p_float=None, npts=300, c_max_float=None,
                 solid_diffusion="none", mode="discharge", N_shell=4):
        super().__init__(param, domain, reaction, options=options, phase=phase)
        self.j_tr_a = None
        self.j_tr_b = None
        self.j_tr_c = None
        self.j_tot=None
        self.R_a = None # reaction rate for branch a
        self.R_b = None 
        self.R_c = None
        self.solid_diffusion = solid_diffusion  # "none" or "core_shell"
        self.mode = mode  # "charge" or "discharge"
        self.N_shell = N_shell  # number of chi intervals for shell FDM

        if R_p_float is not None and c_max_float is not None:
            consts = CCPMPositiveParticle.compute_branch_constants(
                R_p_float, npts, c_max_float
            )
            self.omega   = consts["omega"]
            self.c1_star = consts["c1_star"]
            self.c_sp1   = consts["c_sp1"]
            self.c_sp2   = consts["c_sp2"]
            self.c2_star = consts["c2_star"]
            self.npts    = consts["npts"]
        else:
            raise ValueError(
                "R_p_float and c_max_float are required for CCPMPositiveInterface"
            )
        

        
    def _get_ccpm_currents(self,variables):
        # Coordinates and state variables
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        c_p = variables["CCPM positive particle concentration"] # spatial variable
        c_p_max = self.param.p.prim.c_max - (1e-8*self.param.p.prim.c_max )
        theta_p= c_p/c_p_max
        
        # Physical constants and parameters
        F = pybamm.constants.F
        R = pybamm.constants.R   
        U_eq_p0 = pybamm.Scalar(3.397) #3.42 b#3.397 #self.param.p.U_ref  # scalar 3.42V for LFP (Clarke2026)     #TODO
        j_prime_p0 = pybamm.Scalar(0.948) # 2 #0.948 #TODO: reaction rate constant in positive electrode
        c_e_init = self.param.c_e_init
        omega = pybamm.Scalar(self.omega)
        # Branch boundaries from compute_branch_constants (snapped to cell edges)
        c1_star = self.c1_star
        c2_star = self.c2_star
        c_sp1 = self.c_sp1
        c_sp2 = self.c_sp2
        
        # get macroscopic fields
        phi_p = variables["Positive electrode potential [V]"]
        phi_e = variables["Positive electrolyte potential [V]"]
        T_p = variables["Positive electrode temperature [K]"]
        c_e = variables["Positive electrolyte concentration [mol.m-3]"]
        
        # Compute the overpotential η = φ_s − φ_e − U_eq_p0 at macroscopic level
        # BEFORE broadcasting.  η is O(0.01 V), whereas φ_s − φ_e alone is
        # O(3.4 V); broadcasting the difference avoids catastrophic cancellation
        # when F·(φ_s − φ_e)/(2RT) ≈ 66 is subtracted from F·U_eq_p0/(2RT) ≈ 66
        # inside the sinh argument (the result is O(1e-4)).
        eta = phi_p - phi_e - U_eq_p0          # macroscopic, O(0.01 V)

        # broadcast x-domain quantities to the concentration grid
        eta_broadcast = pybamm.PrimaryBroadcast(eta, "CCPM positive particle concentration")
        T_p_broadcast = pybamm.PrimaryBroadcast(T_p, "CCPM positive particle concentration")
        c_e_broadcast = pybamm.PrimaryBroadcast(c_e, "CCPM positive particle concentration")
        
        # chemical potential on the c-grid (single-phase branches A and C)
        mu = R * T_p_broadcast * (pybamm.log(c_p / (c_p_max - c_p)) + omega * (1 - (2 * theta_p)))
        common_factor = j_prime_p0 * (c_e_broadcast / c_e_init) ** 0.5
        overpotential_term_singlePhase = (F * eta_broadcast + mu) / (2 * R * T_p_broadcast)
        overpotential_term_mixedPhase  = (F * eta_broadcast)       / (2 * R * T_p_broadcast)
        
        # signal: from particle to electrolyte, so positive j means lithium leaving the particle (delithiation)
        j_tr_a = j_tr_c = common_factor * (c_p / c_p_max) ** 0.5 * (1-(c_p/c_p_max))**0.5 * pybamm.sinh(overpotential_term_singlePhase)

        if self.solid_diffusion == "core_shell":
            # ── Transient core-shell (S&N 2004): read c_surf from shell PDE ──
            # c_surf = shell concentration at chi=1 (outermost node)
            c_surf = variables[f"Shell concentration chi_{self.N_shell}"]

            # Clamp c_surf to (eps, c_max - eps) for BV prefactor
            eps_cs = pybamm.Scalar(1e-6) * c_p_max
            c_surf = pybamm.smooth_max(c_surf, eps_cs, 100)
            c_surf = pybamm.smooth_min(c_surf, c_p_max - eps_cs, 100)

            theta_surf = c_surf / c_p_max
            j_tr_b = common_factor * theta_surf**0.5 * (
                1 - theta_surf
            ) ** 0.5 * pybamm.sinh(overpotential_term_mixedPhase)
        else:
            # Original: constant binodal surface concentration
            j_tr_b = common_factor * (c1_star / c_p_max) ** 0.5 * (1-(c1_star/c_p_max))**0.5 * pybamm.sinh(overpotential_term_mixedPhase)
        
        
        # integration - total current: couple back macroscopic domain
        # IMPORTANT. integrate only ovr the range of the branch, otherwise numerical leaks lead to big errors.
        # choose one-sided intervals consistently to avoid double counting at boundaries
        mask_a = (c_p <= c_sp1)
        mask_b = (c1_star <= c_p) * (c_p <= c2_star)
        mask_c = (c_sp2 <= c_p)

        j_tot = pybamm.Integral(mask_a*g_a*j_tr_a + mask_b*g_b*j_tr_b + mask_c*g_c*j_tr_c, c_p)
        return j_tr_a, j_tr_b, j_tr_c, j_tot
    
    def _get_ccpm_rates_and_currents(self, variables):
        # For now, just return the currents as "rates" for testing purposes
        j_tr_a, j_tr_b, j_tr_c, j_tot = self._get_ccpm_currents(variables)
        F = pybamm.constants.F

        beta = 3 / (F * self.param.p.prim.R ) #TODO - A/FV: spherical particles DEBUG reduce size

        # reaction rates
        R_a = -beta * j_tr_a
        R_b = -beta * j_tr_b
        R_c = -beta * j_tr_c

        return j_tr_a, j_tr_b, j_tr_c, j_tot, R_a, R_b, R_c
    
    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name
        phase_name = self.phase_name

        # Compute CCPM currents and rates directly (no standard j0/eta_r pathway)
        (
            self.j_tr_a,
            self.j_tr_b,
            self.j_tr_c,
            self.j_tot,
            self.R_a,
            self.R_b,
            self.R_c,
        ) = self._get_ccpm_rates_and_currents(variables)

        # Get surface potential difference for diagnostic eta_r
        delta_phi = variables[
            f"{Domain} electrode surface potential difference [V]"
        ]
        if isinstance(delta_phi, pybamm.Broadcast):
            delta_phi = delta_phi.orphans[0]

        # Read OCP produced by CCPMOpenCircuitPotential submodel
        ocp = variables[
            f"{Domain} electrode {reaction_name}open-circuit potential [V]"
        ]
        if isinstance(ocp, pybamm.Broadcast):
            if delta_phi.domains.get("secondary") == ["current collector"]:
                ocp = ocp.orphans[0]

        # Diagnostic overpotential (not used in CCPM kinetics — built into sinh)
        eta_r = delta_phi - ocp

        # CCPM doesn't use standard exchange current density
        j0 = pybamm.Scalar(0)

        # No SEI film overpotential
        eta_sei = pybamm.Scalar(0)

        # Average total interfacial current density (for SEI resistance estimate)
        j_tot_av, a_j_tot_av = (
            self._get_average_total_interfacial_current_density(variables)
        )

        # Standard formatting helpers from BaseInterface
        variables.update(
            self._get_standard_interfacial_current_variables(self.j_tot)
        )
        variables.update(
            self._get_standard_total_interfacial_current_variables(
                j_tot_av, a_j_tot_av
            )
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        variables.update(
            self._get_standard_volumetric_current_density_variables(variables)
        )
        variables.update(
            self._get_standard_sei_film_overpotential_variables(eta_sei)
        )

        # CCPM-specific variables
        a_s_p = variables[
            f"{Domain} electrode {phase_name}"
            "surface area to volume ratio [m-1]"
        ]
        j_vol = a_s_p * self.j_tot
        variables.update({
            "CCPM Branch A interfacial current density [A.m-2]": self.j_tr_a,
            "CCPM Branch B interfacial current density [A.m-2]": self.j_tr_b,
            "CCPM Branch C interfacial current density [A.m-2]": self.j_tr_c,
            "CCPM Positive electrode interfacial current density [A.m-2]": self.j_tot,
            "CCPM Positive electrode volumetric interfacial current density [A.m-3]": j_vol,
            "CCPM Branch A lithiation rate": self.R_a,
            "CCPM Branch B lithiation rate": self.R_b,
            "CCPM Branch C lithiation rate": self.R_c,
            "X-averaged CCPM Branch A lithiation rate": pybamm.x_average(self.R_a),
            "X-averaged CCPM Branch B lithiation rate": pybamm.x_average(self.R_b),
            "X-averaged CCPM Branch C lithiation rate": pybamm.x_average(self.R_c),
        })

        return variables