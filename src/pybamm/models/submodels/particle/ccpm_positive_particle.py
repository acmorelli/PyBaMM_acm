from .fickian_diffusion import FickianDiffusion
import pybamm


class CCPMPositiveParticle(FickianDiffusion):
    """
    Step-3 debug version.

    Still behaves like standard Fickian diffusion, but prints the variables
    created by the positive-particle submodel so we can inspect the interface.
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

        print("\n=== CCPMPositiveParticle: FUNDAMENTAL VARIABLES ===")
        for key in sorted(variables.keys()):
            if "positive" in key.lower():
                print(key)

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

    def get_coupled_variables(self, variables):
        variables = super().get_coupled_variables(variables)

        print("\n=== CCPMPositiveParticle: COUPLED VARIABLES (positive-related) ===")
        for key in sorted(variables.keys()):
            if "positive" in key.lower():
                print(key)

        return variables

    def set_rhs(self, variables):
        super().set_rhs(variables)
        dummy = variables["Positive electrode CCPM dummy variable"]
        self.rhs[dummy] = pybamm.Scalar(0)

    def set_initial_conditions(self, variables):
        super().set_initial_conditions(variables)
        dummy = variables["Positive electrode CCPM dummy variable"]
        self.initial_conditions[dummy] = pybamm.Scalar(0)
