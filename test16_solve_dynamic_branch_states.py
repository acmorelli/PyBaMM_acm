import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

model = DFN_CCPM()
sim = pybamm.Simulation(model, parameter_values=model.default_parameter_values)
sol = sim.solve([0, 1000])

g_a = sol["X-averaged positive CCPM interface branch a fraction"].entries[-1]
g_b = sol["X-averaged positive CCPM interface branch b fraction"].entries[-1]
g_c = sol["X-averaged positive CCPM interface branch c fraction"].entries[-1]

print("Solved successfully.\n")
print("g_a final:", g_a)
print("g_b final:", g_b)
print("g_c final:", g_c)
print("sum final:", g_a + g_b + g_c)