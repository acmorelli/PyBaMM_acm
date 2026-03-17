#C:\Users\dottavianomo\programming\PyBaMM_acm\src\pybamm\models\submodels\interface\kinetics\ccpm_positive_interface.py
import pybamm
from .base_kinetics import BaseKinetics


class CCPMPositiveInterface(BaseKinetics):
    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options=options, phase=phase)
        self.j_tr_a = None
        self.j_tr_b = None
        self.j_tr_c = None
        self.j_tot=None
        self.R_a = None # reaction rate for branch a
        self.R_b = None 
        self.R_c = None
        

        
    def _get_ccpm_currents(self,variables):
        # Coordinates and state variables
        g_a = variables["Branch A PDF CCPM"]
        g_b = variables["Branch B PDF CCPM"]
        g_c = variables["Branch C PDF CCPM"]
        c_p = variables["CCPM positive particle concentration"] # spatial variable
        c_p_max = self.param.p.prim.c_max 
        theta_p= c_p/c_p_max
        
        # Physical constants and parameters
        F = pybamm.constants.F
        R = pybamm.constants.R   
        U_eq_p0 = pybamm.Scalar(3.42) #self.param.p.U_ref  # scalar 3.42V for LFP (Clarke2026)     #TODO
        j_prime_p0 = pybamm.Scalar(2) #TODO: reaction rate constant in positive electrode
        c_e_init = self.param.c_e_init
        omega = pybamm.Scalar(3) #TODO self.param?? Eq 13 from Clarke2026
        c1_star = 0.071* c_p_max # from Clarke
        c2_star = 0.929* c_p_max # from Clarke
        c_sp1 = 0.2113 * c_p_max # from Clarke
        c_sp2 = 0.7887 * c_p_max # from Clarke
        
        # get macroscopic fields
        phi_p = variables["Positive electrode potential [V]"]
        phi_e = variables["Positive electrolyte potential [V]"]
        T_p = variables["Positive electrode temperature [K]"]
        c_e = variables["Positive electrolyte concentration [mol.m-3]"]
        
        # broadcast x-domain to the concentration grid (make it available for every c point)
        phi_e_broadcast=pybamm.PrimaryBroadcast(phi_e, "CCPM positive particle concentration")
        phi_p_broadcast = pybamm.PrimaryBroadcast(phi_p, "CCPM positive particle concentration" )
        T_p_broadcast = pybamm.PrimaryBroadcast(T_p, "CCPM positive particle concentration")
        c_e_broadcast= pybamm.PrimaryBroadcast(c_e, "CCPM positive particle concentration")
        
        # calculate branch specfic current j_tr_branchName on the c-grid
        # for this, we define the chemical potential as a function of the spatial variable c_p
        mu= R*T_p_broadcast * ( pybamm.log(c_p/(c_p_max-c_p)) + omega*(1-(2*theta_p)))
        common_factor = j_prime_p0 * (c_e_broadcast / c_e_init) ** 0.5
        overpotential_term_singlePhase = (F * (phi_p_broadcast - phi_e_broadcast) - F * U_eq_p0 + mu) / (2 * R * T_p_broadcast)
        overpotential_term_mixedPhase= (F * (phi_p_broadcast - phi_e_broadcast) - F * U_eq_p0) / (2 * R * T_p_broadcast)
        
        # signal: from particle to electrolyte, so positive j means lithium leaving the particle (delithiation)
        j_tr_a = common_factor * (c_p / c_p_max) ** 0.5 * (1-(c_p/c_p_max))**0.5 * pybamm.sinh(overpotential_term_singlePhase)
        j_tr_b = common_factor * (c1_star / c_p_max) ** 0.5 * (1-(c1_star/c_p_max))**0.5 * pybamm.sinh(overpotential_term_mixedPhase)
        j_tr_c = common_factor * (c_p / c_p_max) ** 0.5 * (1-(c_p/c_p_max))**0.5 * pybamm.sinh(overpotential_term_singlePhase)
        
        
        # integration - total current: couple back macroscopic domain
        # IMPORTANT. integrate only ovr the range of the branch, otherwise numerical leaks lead to big errors.
        # choose one-sided intervals consistently to avoid double counting at boundaries
        mask_a = (c_p <= c_sp1)
        mask_b = (c1_star < c_p) * (c_p <= c2_star)
        mask_c = (c_sp2 < c_p)

        j_a_tot = pybamm.Integral(mask_a * g_a * j_tr_a, c_p)
        j_b_tot = pybamm.Integral(mask_b * g_b * j_tr_b, c_p)
        j_c_tot = pybamm.Integral(mask_c * g_c * j_tr_c, c_p)
        j_tot = j_a_tot + j_b_tot + j_c_tot
        return j_tr_a, j_tr_b, j_tr_c, j_tot
    
    def _get_ccpm_rates_and_currents(self, variables):
        # For now, just return the currents as "rates" for testing purposes
        j_tr_a, j_tr_b, j_tr_c, j_tot = self._get_ccpm_currents(variables)
        F = pybamm.constants.F
        R_p = self.param.p.prim.R   # or self.param.p.prim.R
        beta = 3 / (F * R_p) #TODO - A/FV: spherical particles

        R_a = -beta * j_tr_a
        R_b = -beta * j_tr_b
        R_c = -beta * j_tr_c

        return j_tr_a, j_tr_b, j_tr_c, j_tot, R_a, R_b, R_c
    
    def _get_kinetics(self, j0, ne, eta_r, T, u):
        (
            self.j_tr_a,
            self.j_tr_b,
            self.j_tr_c,
            self.j_tot,
            self.R_a,
            self.R_b,
            self.R_c,
        ) = self._get_ccpm_rates_and_currents(self.variables)
        
        #Butler Voler from butler_volmer.py:
        Feta_RT = self.param.F * eta_r / (self.param.R * T)
        j_bv= 2 * u * j0 * pybamm.sinh(ne * 0.5 * Feta_RT)

        return self.j_tot 
        #return j_bv #TODO not coupling CCPM

                                                               
    def get_coupled_variables(self, variables):
        self.variables = variables #should call _get_kinetics
        variables = super().get_coupled_variables(variables)
        a_s_p = variables["Positive electrode surface area to volume ratio [m-1]"]
        j_vol = a_s_p * self.j_tot
        variables.update({
            # Standard PyBaMM expects this variable for the DFN:
            "Positive electrode interfacial current density [A.m-2]": self.j_tot,
            
            "Positive electrode volumetric interfacial current density [A.m-3]": j_vol,
                
            # CCPM
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