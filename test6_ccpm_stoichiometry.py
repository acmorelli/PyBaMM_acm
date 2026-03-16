import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

model = DFN_CCPM()
param = model.default_parameter_values

sim = pybamm.Simulation(model, parameter_values=param)
solution = sim.solve([0, 100])

theta = solution["Positive CCPM stoichiometry"].entries[-1]
theta_xav = solution["X-averaged positive CCPM stoichiometry"].entries[-1]

print("Solved successfully.")
print("Positive CCPM stoichiometry final:", theta)
print("X-averaged positive CCPM stoichiometry final:", theta_xav)
