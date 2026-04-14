"""
Test: transient core-shell CCPM — 2C discharge, Prada2013 defaults.
Runs both baseline and core-shell, saves pickles for postprocessing.
"""

import numpy as np
import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM


def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3


def arr(sol, name):
    return np.asarray(sol[name].entries)


# ── Parameters (Prada2013 defaults, no R_p/D_s override) ─────────────
parameter_values = pybamm.ParameterValues("Prada2013")
parameter_values.update(
    {
        "Negative electrode OCP [V]": 0.0,
        "Negative electrode conductivity [S.m-1]": 1.0776e7,
        "Negative electrode thickness [m]": 7e-4,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-5,
        "Exchange-current density for lithium metal electrode [A.m-2]"
        "": li_metal_electrolyte_exchange_current_density_Xu2019,
    },
    check_already_exists=False,
)

R_p = float(parameter_values["Positive particle radius [m]"])
D_s = float(parameter_values["Positive particle diffusivity [m2.s-1]"])
c_max = float(parameter_values["Maximum concentration in positive electrode [mol.m-3]"])
tau_diff = R_p**2 / D_s

print(f"Prada2013 defaults:")
print(f"  R_p  = {R_p*1e9:.1f} nm")
print(f"  D_s  = {D_s:.2e} m2/s")
print(f"  c_max = {c_max:.0f} mol/m3")
print(f"  tau_diff = R_p^2/D_s = {tau_diff:.3f} s")
print()

experiment = pybamm.Experiment(["Discharge at 2C for 3000 seconds"])
solver = pybamm.CasadiSolver(mode="safe")
solver_ida = pybamm.IDAKLUSolver()

# ── Baseline (no core-shell) ─────────────────────────────────────────
"""
print("=== BASELINE ===")
model_bl = DFN_CCPM(
    options={"working electrode": "positive"},
    initial_branch="A",
    mode="discharge",
    parameter_values=parameter_values,
)
sim_bl = pybamm.Simulation(model_bl, parameter_values=parameter_values,
                           experiment=experiment, solver=solver)
sol_bl = sim_bl.solve()
sim_bl.save("test_transient_cs_baseline.pkl")
print("  Saved test_transient_cs_baseline.pkl")
"""
# ── Core-shell ───────────────────────────────────────────────────────
print("\n=== CORE-SHELL ===")
model_cs = DFN_CCPM(
    options={"working electrode": "positive"},
    initial_branch="A",
    mode="discharge",
    parameter_values=parameter_values,
    solid_diffusion="core_shell",
    N_shell=3,
)
sim_cs = pybamm.Simulation(model_cs, parameter_values=parameter_values,
                           experiment=experiment, solver=solver_ida)
sol_cs = sim_cs.solve()
sim_cs.save("test_transient_cs_coreshell.pkl")
print("  Saved test_transient_cs_coreshell.pkl")
"""
# ── Quick diagnostics ────────────────────────────────────────────────
print("\n=== DIAGNOSTICS ===")
for label, sol in [("BL", sol_bl), ("CS", sol_cs)]:
    t = sol["Time [s]"].entries
    V = sol["Terminal voltage [V]"].entries
    theta = arr(sol, "X-averaged positive CCPM stoichiometry")
    m_tot = arr(sol, "X-averaged CCPM Total mass")
    print(f"  {label}: t=[{t[0]:.0f}, {t[-1]:.0f}]s, "
          f"V=[{V[-1]:.4f}, {V[0]:.4f}]V, "
          f"theta=[{theta[0]:.4f}, {theta[-1]:.4f}], "
          f"mass_drift={m_tot[-1]-m_tot[0]:.2e}")

# Voltage difference
t_bl = sol_bl["Time [s]"].entries
V_bl = sol_bl["Terminal voltage [V]"].entries
t_cs = sol_cs["Time [s]"].entries
V_cs = sol_cs["Terminal voltage [V]"].entries
t_end = min(t_bl[-1], t_cs[-1])
t_com = np.linspace(0, t_end, 500)
dV = np.interp(t_com, t_cs, V_cs) - np.interp(t_com, t_bl, V_bl)
print(f"\n  max|DeltaV| = {np.max(np.abs(dV))*1e3:.2f} mV")
print(f"  RMS(DeltaV) = {np.sqrt(np.mean(dV**2))*1e3:.2f} mV")
print("\nDone.")
"""