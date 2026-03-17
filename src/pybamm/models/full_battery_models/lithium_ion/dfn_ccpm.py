# C:\Users\dottavianomo\programming\PyBaMM_acm\src\pybamm\models\full_battery_models\lithium_ion\dfn_ccpm.py
import pybamm

from .base_lithium_ion_model import BaseModel
from .dfn import DFN
from ...submodels.particle.ccpm_positive_particle import CCPMPositiveParticle
from ...submodels.interface.kinetics.ccpm_positive_interface import CCPMPositiveInterface
class DFN_CCPM(DFN):
        
    def __init__(self, options=None, name="Doyle-Fuller-Newman CCPM model", build=True):
        # inherit other init
        super().__init__(options, name, build=build)
        
    @property
    def default_geometry(self):
        geometry = super().default_geometry
        eps_c = pybamm.Scalar(1e-6) * self.param.p.prim.c_max
        geometry["positive particle concentration"] = {
            "c_p": {        
                "min": eps_c,
                "max": self.param.p.prim.c_max - eps_c, #TODO is this truncation the best strategy?
    }
        }

        return geometry

    @property
    def default_var_pts(self):
        var_pts = super().default_var_pts
        var_pts.update({"c_p": 1000})
        return var_pts

    @property
    def default_submesh_types(self):
        submesh_types = super().default_submesh_types
        submesh_types.update(
            {"positive particle concentration": pybamm.Uniform1DSubMesh}
        )
        return submesh_types

    @property
    def default_spatial_methods(self):
        spatial_methods = super().default_spatial_methods
        spatial_methods.update(
            {"positive particle concentration": pybamm.FiniteVolume()}
        )
        return spatial_methods
    

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
        # Set kinetics for negative electrode (standard kinetics)
        domain = "negative"
        electrode_type = self.options.electrode_types[domain]
        if electrode_type == "porous":
            intercalation_kinetics = self.get_intercalation_kinetics(domain)
            phases = self.options.phases[domain]
            for phase in phases:
                submod = intercalation_kinetics(
                    self.param, domain, "lithium-ion main", self.options, phase
                )
                self.submodels[f"{domain} {phase} interface"] = submod

            if len(phases) > 1:
                self.submodels[f"total {domain} interface"] = (
                    pybamm.kinetics.TotalMainKinetics(
                        self.param, domain, "lithium-ion main", self.options
                    )
                )

        # Set kinetics for positive electrode (CCPM kinetics)
        domain = "positive"
        electrode_type = self.options.electrode_types[domain]
        if electrode_type == "porous":
            phases = self.options.phases[domain]
            for phase in phases:
                submod = CCPMPositiveInterface(
                    self.param, domain, "lithium-ion main", self.options, phase
                )
                self.submodels[f"{domain} {phase} interface"] = submod

            if len(phases) > 1:
                self.submodels[f"total {domain} interface"] = (
                    pybamm.kinetics.TotalMainKinetics(
                        self.param, domain, "lithium-ion main", self.options
                    )
                )

    
