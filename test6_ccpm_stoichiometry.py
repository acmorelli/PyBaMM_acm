from .fickian_diffusion import FickianDiffusion
import pybamm


class CCPMPositiveParticle(FickianDiffusion):
    """
    Step-6 version.

    Keeps the standard Fickian positive-particle model for DFN compatibility,
    while adding CCPM branch states, a CCPM-derived stoichiometry diagnostic,
    and now a first CCPM-derived open-circuit-potential diagnostic.
    """

    def __init__(
        self,
        param,
        domain="positive",
        options=None,
        phase="primary",
        x_average=False,
    ):
        super().__init__(
            param,
            domain,
            options=options,
            phase=phase,
            x_average=x_average,
        )

    def get_fundamental_variables(self):
        variables = super().get_fundamental_variables()

        electrode_domain = f"{self.domain} electrode"

        g_a = pybamm.Variable(
            "Positive CCPM branch a density",
            domain=electrode_domain,
        )
        g_b = pybamm.Variable(
            "Positive CCPM branch b density",
            domain=electrode_domain,
        )
        g_c = pybamm.Variable(
            "Positive CCPM branch c density",
            domain=electrode_domain,
        )

        variables.update(
            {
                "Positive CCPM branch a density": g_a,
                "Positive CCPM branch b density": g_b,
                "Positive CCPM branch c density": g_c,
                "Positive CCPM total density": g_a + g_b + g_c,
            }
        )

        return variables

    def get_coupled_variables(self, variables):
        variables = super().get_coupled_variables(variables)

        g_a = variables["Positive CCPM branch a density"]
        g_b = variables["Positive CCPM branch b density"]
        g_c = variables["Positive CCPM branch c density"]

        # Placeholder representative stoichiometries for the three branches
        theta_a = pybamm.Scalar(0.05)
        theta_b = pybamm.Scalar(0.50)
        theta_c = pybamm.Scalar(0.95)

        theta_ccpm = theta_a * g_a + theta_b * g_b + theta_c * g_c

        # First placeholder CCPM OCP diagnostic
        U0 = pybamm.Scalar(3.40)
        a = pybamm.Scalar(0.20)
        U_ccpm = U0 + a * (1 - theta_ccpm)

        variables.update(
            {
                "Positive CCPM branch fractions sum": g_a + g_b + g_c,
                "Positive CCPM pseudo-stoichiometry": g_b + g_c,
                "Positive CCPM stoichiometry": theta_ccpm,
                "X-averaged positive CCPM stoichiometry": pybamm.x_average(theta_ccpm),
                "Positive CCPM open-circuit potential [V]": U_ccpm,
                "X-averaged positive CCPM open-circuit potential [V]": pybamm.x_average(U_ccpm),
            }
        )

        return variables

    def set_rhs(self, variables):
        super().set_rhs(variables)

        g_a = variables["Positive CCPM branch a density"]
        g_b = variables["Positive CCPM branch b density"]
        g_c = variables["Positive CCPM branch c density"]

        self.rhs[g_a] = pybamm.Scalar(0)
        self.rhs[g_b] = pybamm.Scalar(0)
        self.rhs[g_c] = pybamm.Scalar(0)

    def set_initial_conditions(self, variables):
        super().set_initial_conditions(variables)

        g_a = variables["Positive CCPM branch a density"]
        g_b = variables["Positive CCPM branch b density"]
        g_c = variables["Positive CCPM branch c density"]

        self.initial_conditions[g_a] = pybamm.Scalar(1)
        self.initial_conditions[g_b] = pybamm.Scalar(0)
        self.initial_conditions[g_c] = pybamm.Scalar(0)
