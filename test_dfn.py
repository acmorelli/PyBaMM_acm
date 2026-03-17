import pybamm

# 1. Create a standard DFN model to access its parameter object
model = pybamm.lithium_ion.DFN()
import pybamm

# 1. Load a standard parameter set (e.g., Marquis 2019 for DFN)
param_values = pybamm.ParameterValues("Prada2013")

# 2. Search for the reaction rate string in the positive electrode
print("=== Searching for Reaction Rate Strings ===")
for key in sorted(param_values.keys()):
    if "reaction rate" in key.lower() and "positive" in key.lower():
        print(f"FOUND: '{key}'")

# 3. Double check the c_max string
print("\n=== Searching for c_max String ===")
for key in sorted(param_values.keys()):
    if "maximum concentration" in key.lower() and "positive" in key.lower():
        print(f"FOUND: '{key}'")
