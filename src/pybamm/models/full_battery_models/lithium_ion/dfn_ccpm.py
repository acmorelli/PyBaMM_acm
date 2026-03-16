import pybamm

from .base_lithium_ion_model import BaseModel
from .dfn import DFN
from ...submodels.particle.ccpm_positive_particle import CCPMPositiveParticle

class DFN_CCPM(DFN):
        
    def __init__(self, options=None, name="Doyle-Fuller-Newman CCPM model", build=True):
        # inherit other init
        super().__init__(options, name, build=build)

    def set_particle_submodel(self):
        super().set_particle_submodel()
        
        # replace only cathode submodel
        self.submodels["positive primary particle"] = CCPMPositiveParticle(
            self.param,
            domain="positive",
            options=self.options,
            phase="primary",
            x_average=False,
        )


    def set_intercalation_kinetics_submodel(self):
        return super().set_intercalation_kinetics_submodel()
    
