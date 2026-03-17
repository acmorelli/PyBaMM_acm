import pybamm
from pybamm.models.submodels.interface.kinetics import ccpm_positive_interface
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

import numpy as np

model = DFN_CCPM()
print("Model built.")


# 1. Create the model
model = DFN_CCPM()

""" print("=== keys containing CCPM ===")
for k in sorted(model.variables.keys()):
    if "CCPM" in k:
        print(k) """
        
# solve
""" 
""" 
param = pybamm.ParameterValues("Chen2020")
param.update({"Current function [A]": 0})
solver = pybamm.CasadiSolver(mode="safe")
t_eval = np.linspace(0, 100, 21)  # 0 to 100 s

sim = pybamm.Simulation(model, solver=solver, parameter_values=param)
solution = sim.solve(t_eval=t_eval)

# Retrieve CCPM mass variables
m_a = solution["CCPM Branch A mass"].entries
m_b = solution["CCPM Branch B mass"].entries
m_c = solution["CCPM Branch C mass"].entries
m_tot = solution["CCPM Total mass"].entries

time = solution.t

print("\n=== CCPM branch masses over time ===")
""" for i, t in enumerate(time):
    print(
        f"t = {t:8.3f} s | "
        f"M_a = {m_a[i]:.8f}, "
        f"M_b = {m_b[i]:.8f}, "
        f"M_c = {m_c[i]:.8f}, "
        f"M_tot = {m_tot[i]:.8f}"
    ) """

# Simple checks
tol_mass = 1e-6
tol_zero = 1e-6

print("\n=== Checks ===")
print(f"max |M_tot - 1| = {np.max(np.abs(m_tot - 1.0)):.3e}")
print(f"max |M_a|       = {np.max(np.abs(m_a)):.3e}")
print(f"max |M_b|       = {np.max(np.abs(m_b)):.3e}")
print(f"max |M_c - 1|   = {np.max(np.abs(m_c - 1.0)):.3e}")

# Example expectation: all mass initially in branch C
branch_a_near_zero = np.all(np.abs(m_a) < tol_zero)
branch_b_near_zero = np.all(np.abs(m_b) < tol_zero)
branch_c_near_one = np.all(np.abs(m_c - 1.0) < tol_mass)
total_near_one = np.all(np.abs(m_tot - 1.0) < tol_mass)

print(f"Branch A near 0 for all time: {branch_a_near_zero}")
print(f"Branch B near 0 for all time: {branch_b_near_zero}")
print(f"Branch C near 1 for all time: {branch_c_near_one}")
print(f"Total mass near 1 for all time: {total_near_one}") 