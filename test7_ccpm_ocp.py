import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

model = DFN_CCPM()
param = model.default_parameter_values

sim = pybamm.Simulation(model, parameter_values=param)
solution = sim.solve([0, 100])

U = solution["Positive CCPM open-circuit potential [V]"].entries[-1]
U_xav = solution["X-averaged positive CCPM open-circuit potential [V]"].entries[-1]

print("Solved successfully.")
print("Positive CCPM OCP final:", U)
print("X-averaged positive CCPM OCP final:", U_xav)
