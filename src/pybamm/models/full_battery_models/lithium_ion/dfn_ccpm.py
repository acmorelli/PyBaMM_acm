import pybamm

from .dfn import DFN
from ...submodels.particle.ccpm_positive_particle import CCPMPositiveParticle
from ...submodels.interface.kinetics.ccpm_positive_interface import CCPMPositiveInterface
from pybamm.geometry import ccpm_spatial_vars as ccpm_vars


class DFN_CCPM(DFN):
    def __init__(
        self,
        options=None,
        name="Doyle-Fuller-Newman CCPM model",
        build=True,
        initial_branch="A",
    ):
        self.initial_branch = initial_branch.upper()
        super().__init__(options=options, name=name, build=build)

    @property
    def default_geometry(self):
        geometry = super().default_geometry

        c_max = self.param.p.prim.c_max
        eps_c = pybamm.Scalar(1e-6) * c_max

        c1_star = pybamm.Scalar(0.071) * c_max
        c2_star = pybamm.Scalar(0.929) * c_max
        c_sp1 = pybamm.Scalar(0.2113) * c_max
        c_sp2 = pybamm.Scalar(0.7887) * c_max

        geometry.update(
            {
                "CCPM positive particle branch a": {
                    ccpm_vars.c_a: {
                        "min": eps_c,
                        "max": c_sp1,
                    }
                },
                "CCPM positive particle branch b": {
                    ccpm_vars.c_b: {
                        "min": c1_star,
                        "max": c2_star,
                    }
                },
                "CCPM positive particle branch c": {
                    ccpm_vars.c_c: {
                        "min": c_sp2,
                        "max": c_max - eps_c,
                    }
                },
            }
        )
        return geometry

    @property
    def default_var_pts(self):
        var_pts = super().default_var_pts
        var_pts.update(
            {
                "c_a": 30,
                "c_b": 40,
                "c_c": 30,
            }
        )
        return var_pts

    @property
    def default_submesh_types(self):
        submesh_types = super().default_submesh_types
        submesh_types.update(
            {
                "CCPM positive particle branch a": pybamm.Uniform1DSubMesh,
                "CCPM positive particle branch b": pybamm.Uniform1DSubMesh,
                "CCPM positive particle branch c": pybamm.Uniform1DSubMesh,
            }
        )
        return submesh_types

    @property
    def default_spatial_methods(self):
        spatial_methods = super().default_spatial_methods
        spatial_methods.update(
            {
                "CCPM positive particle branch a": pybamm.FiniteVolume(),
                "CCPM positive particle branch b": pybamm.FiniteVolume(),
                "CCPM positive particle branch c": pybamm.FiniteVolume(),
            }
        )
        return spatial_methods

    def set_particle_submodel(self):
        super().set_particle_submodel()
        self.submodels["positive primary particle"] = CCPMPositiveParticle(
            self.param,
            domain="positive",
            options=self.options,
            phase="primary",
            x_average=False,
            initial_branch=self.initial_branch,
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