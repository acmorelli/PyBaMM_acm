import pybamm

from .base_lithium_ion_model import BaseModel
from .dfn import DFN

class DFN_CCPM(DFN):
        
    def __init__(self, options=None, name="Doyle-Fuller-Newman CCPM model", build=True):
        # For degradation models we use the full form since this is a full-order model
        super().__init__(options, name, build=build)

    def set_particle_submodel(self):
        return super().set_particle_submodel()
    def set_intercalation_kinetics_submodel(self):
        return super().set_intercalation_kinetics_submodel()