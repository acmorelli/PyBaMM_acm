import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

model = DFN_CCPM()
param = model.default_parameter_values

sim = pybamm.Simulation(model, parameter_values=param)
solution = sim.solve([0, 100])

print("Solved successfully.")
print("g_a final:", solution["Positive CCPM branch a density"].entries[-1])
print("g_b final:", solution["Positive CCPM branch b density"].entries[-1])
print("g_c final:", solution["Positive CCPM branch c density"].entries[-1])
print("sum final:", solution["Positive CCPM total density"].entries[-1])
