#C:\Users\dottavianomo\programming\PyBaMM_acm\src\pybamm\models\submodels\interface\kinetics\ccpm_positive_interface.py
import pybamm
from .base_kinetics import BaseKinetics


class CCPMPositiveInterface(BaseKinetics):
    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options=options, phase=phase)
        self.j_tr_a = None
        self.j_tr_b = None
        self.j_tr_c = None
        self.j_tot = None
        self.R_a = None
        self.R_b = None
        self.R_c = None

    def _get_ccpm_currents(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]

        # Each branch has its own subdomain spatial variable
        c_p_a = pybamm.standard_spatial_vars.c_p_a
        c_p_b = pybamm.standard_spatial_vars.c_p_b
        c_p_c = pybamm.standard_spatial_vars.c_p_c
        c_p_max = self.param.p.prim.c_max
        eps_c = pybamm.Scalar(1e-8) * c_p_max
        # Truncated max to avoid log/sqrt singularity at the domain boundary c_p = c_p_max
        c_max_eff = c_p_max - eps_c

        # Physical constants (TODO: migrate to param)
        F = pybamm.constants.F
        R = pybamm.constants.R
        U_eq_p0   = pybamm.Scalar(3.42)  # LFP plateau potential [V]
        j_prime_p0 = pybamm.Scalar(2)    # reaction rate constant [A m-2]
        omega     = pybamm.Scalar(3)     # regular-solution parameter
        c_e_init  = self.param.c_e_init
        c1_star   = pybamm.Scalar(0.0710) * c_p_max
        c2_star   = pybamm.Scalar(0.9290) * c_p_max

        # Macroscopic fields (positive-electrode domain)
        phi_p = variables["Positive electrode potential [V]"]
        phi_e = variables["Positive electrolyte potential [V]"]
        T_p   = variables["Positive electrode temperature [K]"]
        c_e   = variables["Positive electrolyte concentration [mol.m-3]"]

        # Broadcast macroscopic quantities to each branch subdomain
        def bc(scalar_or_x, domain_name):
            return pybamm.PrimaryBroadcast(scalar_or_x, domain_name)

        phi_p_a = bc(phi_p, "CCPM positive particle branch A")
        phi_e_a = bc(phi_e, "CCPM positive particle branch A")
        T_a     = bc(T_p,   "CCPM positive particle branch A")
        c_e_a   = bc(c_e,   "CCPM positive particle branch A")

        phi_p_b = bc(phi_p, "CCPM positive particle branch B")
        phi_e_b = bc(phi_e, "CCPM positive particle branch B")
        T_b     = bc(T_p,   "CCPM positive particle branch B")
        c_e_b   = bc(c_e,   "CCPM positive particle branch B")

        phi_p_c = bc(phi_p, "CCPM positive particle branch C")
        phi_e_c = bc(phi_e, "CCPM positive particle branch C")
        T_c     = bc(T_p,   "CCPM positive particle branch C")
        c_e_c   = bc(c_e,   "CCPM positive particle branch C")

        # --- Branch A: single-phase, concentration-dependent activity (eq 52) ---
        theta_a = c_p_a / c_max_eff
        mu_a    = R * T_a * (
            pybamm.log(c_p_a / (c_max_eff - c_p_a)) + omega * (1 - 2 * theta_a)
        )
        eta_a = (
            F * (phi_p_a - phi_e_a) - F * U_eq_p0 + mu_a
        ) / (2 * R * T_a)
        j_tr_a = (
            j_prime_p0
            * (c_e_a / c_e_init) ** 0.5
            * theta_a ** 0.5
            * (1 - theta_a) ** 0.5
            * pybamm.sinh(eta_a)
        )

        # --- Branch B: mixed-phase, fixed activity at c1_star (eq 53) ---
        theta_b_ref = c1_star / c_max_eff
        eta_b = (
            F * (phi_p_b - phi_e_b) - F * U_eq_p0
        ) / (2 * R * T_b)
        j_tr_b = (
            j_prime_p0
            * (c_e_b / c_e_init) ** 0.5
            * theta_b_ref ** 0.5
            * (1 - theta_b_ref) ** 0.5
            * pybamm.sinh(eta_b)
        )

        # --- Branch C: single-phase, concentration-dependent activity (eq 54) ---
        theta_c = c_p_c / c_max_eff
        mu_c    = R * T_c * (
            pybamm.log(c_p_c / (c_max_eff - c_p_c)) + omega * (1 - 2 * theta_c)
        )
        eta_c = (
            F * (phi_p_c - phi_e_c) - F * U_eq_p0 + mu_c
        ) / (2 * R * T_c)
        j_tr_c = (
            j_prime_p0
            * (c_e_c / c_e_init) ** 0.5
            * theta_c ** 0.5
            * (1 - theta_c) ** 0.5
            * pybamm.sinh(eta_c)
        )

        # Integrate over each branch subdomain — no masking needed (Clarke eq 51)
        j_a_tot = pybamm.Integral(g_a * j_tr_a, c_p_a)
        j_b_tot = pybamm.Integral(g_b * j_tr_b, c_p_b)
        j_c_tot = pybamm.Integral(g_c * j_tr_c, c_p_c)
        j_tot   = j_a_tot + j_b_tot + j_c_tot

        return j_tr_a, j_tr_b, j_tr_c, j_tot

    def _get_ccpm_rates_and_currents(self, variables):
        j_tr_a, j_tr_b, j_tr_c, j_tot = self._get_ccpm_currents(variables)
        F    = pybamm.constants.F
        R_p  = self.param.p.prim.R
        beta = 3 / (F * R_p)  # surface-area-to-volume for spherical particles

        # Rates on their respective branch domains
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