from .fickian_diffusion import FickianDiffusion
import pybamm


class CCPMPositiveParticle(FickianDiffusion):
    """
    Step-2b compatibility placeholder.

    For now this is still the standard Fickian positive-particle submodel,
    but with one extra dummy CCPM variable added so we can keep the DFN build
    working while preparing the CCPM insertion point.
    """

    def __init__(self, param, domain="positive", options=None, phase="primary", x_average=False):
        super().__init__(
            param,
            domain,
            options=options,
            phase=phase,
            x_average=x_average,
        )

    def get_fundamental_variables(self):
        variables = super().get_fundamental_variables()

        dummy = pybamm.Variable(
            "Positive electrode CCPM dummy variable",
            domain=f"{self.domain} electrode",
        )
        variables.update(
            {
                "Positive electrode CCPM dummy variable": dummy,
            }
        )
        return variables

    def set_rhs(self, variables):
        super().set_rhs(variables)
        dummy = variables["Positive electrode CCPM dummy variable"]
        self.rhs[dummy] = pybamm.Scalar(0)

    def set_initial_conditions(self, variables):
        super().set_initial_conditions(variables)
        dummy = variables["Positive electrode CCPM dummy variable"]
        self.initial_conditions[dummy] = pybamm.Scalar(0)
