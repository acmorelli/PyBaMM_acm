import pybamm

from .dfn import DFN
from ...submodels.particle.ccpm_positive_particle import CCPMPositiveParticle
from ...submodels.interface.kinetics.ccpm_positive_interface import CCPMPositiveInterface
from ...submodels.interface.open_circuit_potential.ccpm_ocp import CCPMOpenCircuitPotential


class DFN_CCPM(DFN):
    def __init__(
        self,
        options=None,
        name="Doyle-Fuller-Newman CCPM model",
        build=True,
        initial_branch="A",
        source_method="flux_form",
    ):
        self.initial_branch = initial_branch.upper()
        self.source_method = source_method
        super().__init__(options=options, name=name, build=build)

    @property
    def default_geometry(self):
        geometry = super().default_geometry

        c_max = self.param.p.prim.c_max
        eps_c = pybamm.Scalar(1e-8) * c_max #TODO debug

        geometry["CCPM positive particle concentration"] = {
            "c_p": {        
                "min": eps_c,
                "max": self.param.p.prim.c_max - eps_c, #TODO is this truncation the best strategy?
                    }
            }
        return geometry

    @property
    def default_var_pts(self):
        var_pts = super().default_var_pts
        var_pts.update({"c_p": 300})
        return var_pts

    @property
    def default_submesh_types(self):
        submesh_types = super().default_submesh_types
        submesh_types.update(
            {"CCPM positive particle concentration": pybamm.Uniform1DSubMesh}
        )
        return submesh_types

    @property
    def default_spatial_methods(self):
        spatial_methods = super().default_spatial_methods
        spatial_methods.update(
            {"CCPM positive particle concentration": pybamm.FiniteVolume()}
        )
        return spatial_methods

    def set_particle_submodel(self):
        super().set_particle_submodel()
        self.submodels["positive primary particle"] = CCPMPositiveParticle(
            self.param,
            domain="positive",
            options=self.options,
            phase="primary",
            initial_branch=self.initial_branch,
            source_method=self.source_method,
        )

    def set_open_circuit_potential_submodel(self):
        super().set_open_circuit_potential_submodel()
        # Replace positive electrode OCP with CCPM-specific version
        self.submodels["positive primary open-circuit potential"] = (
            CCPMOpenCircuitPotential(
                self.param,
                "positive",
                "lithium-ion main",
                self.options,
                "primary",
                self.x_average,
            )
        )

    def set_intercalation_kinetics_submodel(self):
        # keep standard negative-electrode kinetics
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

        # replace standard positive-electrode kinetics by CCPM kinetics
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