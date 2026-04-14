"""
CCPM half-cell radius sweep: 2C charge & discharge, baseline vs core-shell.

Usage:
    python run_radius_sweep.py <R_nm> <mode> [core_shell]

    R_nm       : particle radius in nm (e.g. 30, 50, 100, 300, 500, 1000)
    mode       : "charge" or "discharge"
    core_shell : optional flag, if present runs core-shell model

Examples:
    python run_radius_sweep.py 50 charge              # baseline charge 50nm
    python run_radius_sweep.py 500 discharge core_shell  # core-shell discharge 500nm
"""

import sys
import numpy as np
import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM


def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3


def arr(sol, name):
    return np.asarray(sol[name].entries)


# ── Parse arguments ───────────────────────────────────────────────────
R_nm = int(sys.argv[1])
mode = sys.argv[2]           # "charge" or "discharge"
use_core_shell = len(sys.argv) > 3 and sys.argv[3] == "core_shell"

R_p = R_nm * 1e-9            # [m]
D_s = 1e-14                  # intra-phase diffusivity [m²/s] (Malik et al. 2010)
c_max = 22806.0

tag = "cs" if use_core_shell else "bl"
label = f"R{R_nm}nm_{mode}_{tag}"
pkl_name = f"sweep_{label}.pkl"

print(f"=== {label} ===")
print(f"  R_p = {R_nm} nm, D_s = {D_s:.0e}, core_shell = {use_core_shell}")

# ── Parameters ────────────────────────────────────────────────────────
parameter_values = pybamm.ParameterValues("Prada2013")

updates = {
    "Negative electrode OCP [V]": 0.0,
    "Negative electrode conductivity [S.m-1]": 1.0776e7,
    "Negative electrode thickness [m]": 7e-4,
    "Negative electrode OCP entropic change [V.K-1]": 0.0,
    "Negative electrode charge transfer coefficient": 0.5,
    "Negative electrode double-layer capacity [F.m-2]": 0.2,
    "Lithium metal partial molar volume [m3.mol-1]": 1.3e-5,
    "Exchange-current density for lithium metal electrode [A.m-2]"
    "": li_metal_electrolyte_exchange_current_density_Xu2019,
    "Lower voltage cut-off [V]": 1.0,
    "Positive particle radius [m]": R_p,
    "Positive particle diffusivity [m2.s-1]": D_s,
}

if mode == "charge":
    updates["Initial concentration in positive electrode [mol.m-3]"] = 0.996 * c_max

parameter_values.update(updates, check_already_exists=False)

# ── Model ─────────────────────────────────────────────────────────────
model_kwargs = dict(
    options={"working electrode": "positive"},
    initial_branch="C" if mode == "charge" else "A",
    mode=mode,
    parameter_values=parameter_values,
)
if use_core_shell:
    model_kwargs["solid_diffusion"] = "core_shell"

model = DFN_CCPM(**model_kwargs)

# ── Experiment & solve ────────────────────────────────────────────────
exp_str = f"{'Charge' if mode == 'charge' else 'Discharge'} at 2C for 3000 seconds"
experiment = pybamm.Experiment([exp_str])

sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"),
)

print(f"Solving {label} …")
solution = sim.solve()
sim.save(pkl_name)
print(f"Done. Saved to {pkl_name}\n")

# ── Diagnostics ───────────────────────────────────────────────────────
time = solution["Time [s]"].entries
voltage = solution["Terminal voltage [V]"].entries
theta = arr(solution, "X-averaged positive CCPM stoichiometry")
m_tot = arr(solution, "X-averaged CCPM Total mass")

print(f"  Time: {time[0]:.1f} – {time[-1]:.1f} s ({len(time)} pts)")
print(f"  V:    [{voltage.min():.4f}, {voltage.max():.4f}] V")
print(f"  θ:    [{theta.min():.6f}, {theta.max():.6f}]")
print(f"  Mass drift: {m_tot[-1] - m_tot[0]:.4e}")
