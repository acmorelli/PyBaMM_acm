#C:\Users\dottavianomo\programming\PyBaMM_acm\src\pybamm\models\submodels\particle\ccpm_positive_particle.py
from .fickian_diffusion import FickianDiffusion
import pybamm


class CCPMPositiveParticle(FickianDiffusion):
    """
    CCPM-extended Fickian diffusion particle model.
    
    Extends the standard Fickian diffusion with CCPM-derived diagnostics:
    - CCPM stoichiometry (weighted combination of branch stoichiometries)
    - CCPM open-circuit potential (weighted by branch fractions)
    - CCPM exchange current density (weighted by branch fractions)
    
    The branch fractions themselves are computed algebraically in the interface kinetics
    based on the DFN stoichiometry.
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
        
        
        self.c_p_space = pybamm.standard_spatial_vars.c_p
        
    def get_fundamental_variables(self):
        # 1. Define the 3 PDFs as state variables (Differential Variables)
        # Primary domain: the concentration grid 'c'
        # Secondary domain: the electrode position 'x'
        g_a = pybamm.Variable(
            "Branch A PDF CCPM",
        domains={
            "primary": "positive particle concentration",
            "secondary": "positive electrode",
            "tertiary": "current collector"
        }
        )
        g_b = pybamm.Variable(
            "Branch B PDF CCPM",
        domains={
            "primary": "positive particle concentration",
            "secondary": "positive electrode",
            "tertiary": "current collector"
        }
        )
        g_c = pybamm.Variable(
            "Branch C PDF CCPM",
        domains={
            "primary": "positive particle concentration",
            "secondary": "positive electrode",
            "tertiary": "current collector"
    }
        )

        variables = {
            "Branch A PDF CCPM": g_a,
            "Branch B PDF CCPM": g_b,
            "Branch C PDF CCPM": g_c,
            "CCPM positive particle concentration": self.c_p_space,
        }

        # Keep standard Fickian variables to avoid breaking DFN dependency
        variables.update(super().get_fundamental_variables())
        return variables
    
    def get_coupled_variables(self, variables):
        #TODO: remove inheritance after full integration
        variables.update(super().get_coupled_variables(variables))
        # Retrieve our PDFs and the concentration grid
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        c_p = variables["CCPM positive particle concentration"]
        c_max = self.param.p.prim.c_max
        
        # 2. Calculate the average stoichiometry for the CCPM 
        # This is the integral of (c * total_PDF) / c_max
        # This replaces the old 'theta_a_ref' logic with real PDF physics
        theta_ccpm = pybamm.Integral((g_a + g_b + g_c) * (c_p / c_max), c_p)

        
        # mass per branch
        m_a = pybamm.Integral(g_a, c_p)
        m_b = pybamm.Integral(g_b, c_p)
        m_c = pybamm.Integral(g_c, c_p)
        m_tot = m_a + m_b + m_c
        
        variables.update({
            "CCPM Branch A mass": m_a,
            "CCPM Branch B mass": m_b,
            "CCPM Branch C mass": m_c,
            "CCPM Total mass": m_tot,
            "Positive CCPM stoichiometry": theta_ccpm,
            "X-averaged positive CCPM stoichiometry": pybamm.x_average(theta_ccpm),
            "CCPM Total PDF sum": g_a + g_b + g_c, # Should integrate to 1
        })

        return variables

    def set_rhs(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]

        # For now, stay stationary so we can test the build
        self.rhs = {
            g_a: pybamm.Scalar(0),
            g_b: pybamm.Scalar(0),
            g_c: pybamm.Scalar(0),
        }
    
    def set_boundary_conditions(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        self.boundary_conditions = {
            g_a: {"left": (pybamm.Scalar(0), "Dirichlet"), "right": (pybamm.Scalar(0), "Dirichlet")},
            g_b: {"left": (pybamm.Scalar(0), "Dirichlet"), "right": (pybamm.Scalar(0), "Dirichlet")},
            g_c: {"left": (pybamm.Scalar(0), "Dirichlet"), "right": (pybamm.Scalar(0), "Dirichlet")},
        }
    
    def set_initial_conditions(self, variables):
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        c_max = self.param.p.prim.c_max
        c_sp1 = 0.211*c_max #TODO: cast to scalar?
        c_sp2=0.789*c_max #TODO
        
        
        c_p = variables["CCPM positive particle concentration"] 
        c_init_particle=self.param.p.prim.c_init-1 # r domain
        # collapse to electrode (r-average), then broadcast into concentration domain
        c_init_x = pybamm.r_average(c_init_particle) # positive electrode domain
        c_init_cp = pybamm.PrimaryBroadcast(c_init_x, "positive particle concentration")

        # Define a narrow Gaussian to represent the initial state
        sigma = 0.02 * c_max 
        initial_distribution = pybamm.exp(-((c_p - c_init_cp) ** 2) / (2 * sigma ** 2))
        
        # Normalize so the integral over c_p is 1 (Particle Conservation)
        # We start with everything in Branch C for a lithiated cathode
        # or Branch A for a delithiated one.
        norm = pybamm.Integral(initial_distribution, c_p)
        initial_pdf = initial_distribution / norm # normalised
        
        # boolean-like masks on x-domain
        on_a = pybamm.sigmoid(c_init_x, c_sp1, 1e6) #TODO logic switch?
        on_c = pybamm.sigmoid(c_sp2, c_init_x, 1e6)
        #print('Non mixed state initial condition if sum is 1: ', on_a.evaluate() * on_c.evaluate())
        on_a_cp = pybamm.PrimaryBroadcast(on_a, "positive particle concentration")
        on_c_cp = pybamm.PrimaryBroadcast(on_c, "positive particle concentration")
        
        """ self.initial_conditions = {
            g_a: on_a_cp * initial_pdf,
            g_b: pybamm.Scalar(0),
            g_c: on_c_cp * initial_pdf,
        } """ 
        self.initial_conditions = {
            g_a: initial_pdf,
            g_b: 0 * initial_pdf,
            g_c: 0 * initial_pdf,
        }