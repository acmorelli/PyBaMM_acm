#
# CCPM open-circuit potential for positive electrode (Clarke et al. 2026)
#
import pybamm

from . import BaseOpenCircuitPotential


class CCPMOpenCircuitPotential(BaseOpenCircuitPotential):
    """
    Computes an effective OCP from the CCPM-averaged stoichiometry θ_CCPM.

    The CCPM has no standard OCP in the Fickian sense. Instead, the effective
    equilibrium potential at a given stoichiometry θ is:

        U_eq_eff(θ) = U_eq_p0 - (RT/F) * [ln(θ/(1-θ)) + Ω*(1 - 2θ)]

    where U_eq_p0 = 3.42 V is the LFP plateau reference, Ω = 3 is the regular
    solution parameter, and θ_CCPM = ∫(g_a+g_b+g_c)*(c_p/c_max) dc_p is computed
    in the particle model.
    """

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        # CCPM stoichiometry (produced by CCPMPositiveParticle)
        theta_CCPM = variables["Positive CCPM stoichiometry"]
        theta_CCPM_av = variables["X-averaged positive CCPM stoichiometry"]
        T = variables[f"{Domain} electrode temperature [K]"]
        T_bulk = pybamm.xyzs_average(T)

        # Physical constants
        F = pybamm.constants.F
        R = pybamm.constants.R
        U_eq_p0 = pybamm.Scalar(3.397)  # V, LFP plateau reference
        omega = pybamm.Scalar(3)  # regular solution parameter

        # Effective OCP: U_eq_eff = U_eq_p0 - μ_p(θ)/F
        ocp_surf = U_eq_p0 - (R * T / F) * (
            pybamm.log(theta_CCPM / (1 - theta_CCPM))
            + omega * (1 - 2 * theta_CCPM)
        )
        ocp_bulk = U_eq_p0 - (R * T_bulk / F) * (
            pybamm.log(theta_CCPM_av / (1 - theta_CCPM_av))
            + omega * (1 - 2 * theta_CCPM_av)
        )
        dUdT = pybamm.Scalar(0)  # TODO: entropic change for CCPM

        # Equilibrium OCP (same as ocp for CCPM — no hysteresis splitting)
        variables.update({
            f"{Domain} electrode {phase_name}equilibrium "
            "open-circuit potential [V]": ocp_surf,
            f"X-averaged {domain} electrode {phase_name}equilibrium "
            "open-circuit potential [V]": pybamm.x_average(ocp_surf),
        })

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables
