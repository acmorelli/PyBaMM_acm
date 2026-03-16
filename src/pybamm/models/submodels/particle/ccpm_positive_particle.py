from .fickian_diffusion import FickianDiffusion
import pybamm


class CCPMPositiveParticle(FickianDiffusion):
    """
    Step-4 version.

    Still inherits the standard positive-particle Fickian diffusion model so that
    the full DFN remains compatible, but now also introduces the first CCPM
    state variables:
        - g_a
        - g_b
        - g_c

    For now these are only placeholder states with trivial time evolution.
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
        # Keep all standard DFN positive-particle variables
        variables = super().get_fundamental_variables()

        electrode_domain = f"{self.domain} electrode"

        # First CCPM branch-density placeholders
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

        # Simple diagnostic variables for checking that the CCPM states exist
        variables.update(
            {
                "Positive CCPM branch fractions sum": g_a + g_b + g_c,
                "Positive CCPM pseudo-stoichiometry": g_b + g_c,
            }
        )

        return variables

    def set_rhs(self, variables):
        # Keep standard DFN particle dynamics
        super().set_rhs(variables)

        g_a = variables["Positive CCPM branch a density"]
        g_b = variables["Positive CCPM branch b density"]
        g_c = variables["Positive CCPM branch c density"]

        # Trivial placeholder dynamics:
        # start entirely on branch a and stay there
        self.rhs[g_a] = pybamm.Scalar(0)
        self.rhs[g_b] = pybamm.Scalar(0)
        self.rhs[g_c] = pybamm.Scalar(0)

    def set_initial_conditions(self, variables):
        # Keep standard DFN particle initial conditions
        super().set_initial_conditions(variables)

        g_a = variables["Positive CCPM branch a density"]
        g_b = variables["Positive CCPM branch b density"]
        g_c = variables["Positive CCPM branch c density"]

        # Initially all "mass" on branch a
        self.initial_conditions[g_a] = pybamm.Scalar(1)
        self.initial_conditions[g_b] = pybamm.Scalar(0)
        self.initial_conditions[g_c] = pybamm.Scalar(0)
