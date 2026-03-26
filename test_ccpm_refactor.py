"""
Smoke-test for the refactored DFN_CCPM model.
Tests build → variable checks → C/20 discharge → mass conservation.
Uses Prada2013 parameters and default 300-point c_p grid.
"""
import numpy as np
import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

# ---------- 1. Build ----------
print("1. Building model …")
model = DFN_CCPM(build=True, initial_branch="A")
print(f"   OK — {len(model.variables)} variables registered")

# ---------- 2. Check key CCPM variables exist ----------
print("2. Checking variable names …")
required_ccpm = [
    "Branch A PDF CCPM",
    "Branch B PDF CCPM",
    "Branch C PDF CCPM",
    "Positive CCPM stoichiometry",
    "CCPM Branch A mass",
    "CCPM Total mass",
    "CCPM Branch A lithiation rate",
    "CCPM Branch B lithiation rate",
    "CCPM Branch C lithiation rate",
    "CCPM source mass balance error",
]
required_standard = [
    "Positive electrode interfacial current density [A.m-2]",
    "Positive electrode volumetric interfacial current density [A.m-3]",
    "Positive electrode open-circuit potential [V]",
    "Positive electrode reaction overpotential [V]",
    "Positive electrode entropic change [V.K-1]",
    "Voltage [V]",
    "R-averaged positive particle concentration [mol.m-3]",
]
missing = [n for n in required_ccpm + required_standard if n not in model.variables]
if missing:
    print("   MISSING variables:")
    for m in missing:
        print(f"     - {m}")
    raise RuntimeError("Missing variables — aborting.")
print("   OK — all required variables present")

# ---------- 3. Fickian variables should NOT exist ----------
print("3. Verifying Fickian variables are removed …")
fickian_gone = [
    "Positive particle concentration [mol.m-3]",
    "Positive particle surface concentration [mol.m-3]",
    "Positive particle flux [mol.m-2.s-1]",
]
leaked = [v for v in fickian_gone if v in model.variables]
if leaked:
    print(f"   WARNING — Fickian variables still present: {leaked}")
else:
    print("   OK — no Fickian leak")

# ---------- 4. Discharge simulation ----------
print("4. Setting up C/30 discharge (Prada2013, 300 c_p pts) \u2026")
param = pybamm.ParameterValues("Prada2013")
experiment = pybamm.Experiment(["Discharge at C/30 for 100 seconds"])

sim = pybamm.Simulation(
    model,
    parameter_values=param,
    experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"),
)

print("5. Solving (100 s at C/30, 300 c_p pts) \u2026")
sol = sim.solve()

# ---------- 6. Results ----------
time = sol["Time [s]"].entries
V = sol["Voltage [V]"].entries
theta = sol["X-averaged positive CCPM stoichiometry"].entries
m_a = sol["X-averaged CCPM Branch A mass"].entries
m_b = sol["X-averaged CCPM Branch B mass"].entries
m_c = sol["X-averaged CCPM Branch C mass"].entries
m_tot = sol["X-averaged CCPM Total mass"].entries

print(f"\n   Time span:  {time[0]:.1f} – {time[-1]:.1f} s  ({len(time)} steps)")
print(f"   Voltage:    {V[0]:.4f} → {V[-1]:.4f} V")
print(f"   θ_CCPM:     {theta[0]:.6f} → {theta[-1]:.6f}")
print(f"   Masked mass (t=0):   m_a={m_a[0]:.6f}  m_b={m_b[0]:.6f}  m_c={m_c[0]:.6f}  total={m_tot[0]:.6f}")
print(f"   Masked mass (t=end): m_a={m_a[-1]:.6f}  m_b={m_b[-1]:.6f}  m_c={m_c[-1]:.6f}  total={m_tot[-1]:.6f}")
print(f"   Mass drift (masked):   {m_tot[-1] - m_tot[0]:.6e}")

# ---------- 7. Mass conservation check ----------
drift = abs(m_tot[-1] - m_tot[0])
if drift < 0.05:
    print(f"\n   MASS CONSERVATION OK (drift = {drift:.6e})")
else:
    print(f"\n   WARNING: significant mass drift = {drift:.6e}")

print("\nDone.")
