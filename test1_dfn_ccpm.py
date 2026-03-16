import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

model = DFN_CCPM(build=False)

for key, submodel in model.submodels.items():
    if "particle" in key.lower():
        print(key, "->", type(submodel).__name__)
