"""C/30 full discharge test — no plots, just mass diagnostics + pickle."""
import numpy as np, pickle, pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

model = DFN_CCPM(build=True, initial_branch="A", source_method="flux_form")
param = pybamm.ParameterValues("Prada2013")
experiment = pybamm.Experiment(["Discharge at C/30 until 2.0 V"])
sim = pybamm.Simulation(
    model, parameter_values=param, experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"),
)
print("Solving C/30 …")
sol = sim.solve()
print("Done.\n")

time  = sol["Time [s]"].entries
V     = sol["Voltage [V]"].entries
theta = sol["X-averaged positive CCPM stoichiometry"].entries
m_a   = sol["X-averaged CCPM Branch A mass"].entries
m_b   = sol["X-averaged CCPM Branch B mass"].entries
m_c   = sol["X-averaged CCPM Branch C mass"].entries
m_tot = sol["X-averaged CCPM Total mass"].entries

print(f"  Voltage:   {V[0]:.4f} -> {V[-1]:.4f} V")
print(f"  theta:     {theta[0]:.6f} -> {theta[-1]:.6f}")
print(f"  m_tot:     {m_tot[0]:.8f} -> {m_tot[-1]:.8f}  drift={m_tot[-1]-m_tot[0]:+.3e}")
print(f"  m_A:       {m_a[-1]:.6f}")
print(f"  m_B:       {m_b[-1]:.6f}")
print(f"  m_C:       {m_c[-1]:.6f}")
print()

# Time trajectory
idx = np.linspace(0, len(time)-1, 12, dtype=int)
for i in idx:
    t_h = time[i] / 3600
    print(f"  t={t_h:7.2f}h  theta={theta[i]:.6f}  m_tot={m_tot[i]:.6f}  "
          f"mA={m_a[i]:.4f}  mB={m_b[i]:.4f}  mC={m_c[i]:.4f}")

# Save pickle
with open("ccpm_c30_localized.pkl", "wb") as f:
    pickle.dump(sol, f)
print(f"\nPickle saved: ccpm_c30_localized.pkl")
