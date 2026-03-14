import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

model = DFN_CCPM()
param = model.default_parameter_values

sim = pybamm.Simulation(model, parameter_values=param)
solution = sim.solve([0, 100])

print("Model name:", model.name)
print("Solved successfully.")
print("Final voltage:", solution["Voltage [V]"].entries[-1])