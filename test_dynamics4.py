import numpy as np
import matplotlib.pyplot as plt
import pybamm

from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM


def choose_physical_initial_branch(parameter_values):
    c_max = float(parameter_values["Maximum concentration in positive electrode [mol.m-3]"])
    c_init = float(parameter_values["Initial concentration in positive electrode [mol.m-3]"])

    c_sp1 = 0.2113 * c_max
    c_sp2 = 0.7887 * c_max

    print(f"c_init = {c_init:.6e}")
    print(f"c_sp1  = {c_sp1:.6e}")
    print(f"c_sp2  = {c_sp2:.6e}")
    print(f"theta_init = {c_init / c_max:.6f}")

    if c_init < c_sp1:
        return "A"
    elif c_init > c_sp2:
        return "C"
    else:
        raise ValueError(
            "Initial concentration lies inside the spinodal window. "
            "A physical single-branch initialization is not unique there; "
            "you need history/preconditioning."
        )
        
parameter_values = pybamm.ParameterValues("Prada2013")   # or your set
initial_branch = choose_physical_initial_branch(parameter_values)

print("Chosen physical CCPM initial branch:", initial_branch)   



