"""C/10 discharge to 2V - medium test to verify flux fix."""
import numpy as np, pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

model = DFN_CCPM(build=True, initial_branch="A", source_method="flux_form")
param = pybamm.ParameterValues("Prada2013")
experiment = pybamm.Experiment(
    ["Discharge at C/10 until 2.0 V"], termination="2.0 V"
)
sim = pybamm.Simulation(
    model, parameter_values=param, experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"),
)
print("Solving C/10 discharge ...")
sol = sim.solve()
print("Done.")

time = sol["Time [s]"].entries
V = sol["Voltage [V]"].entries
theta = sol["X-averaged positive CCPM stoichiometry"].entries
m_tot = sol["X-averaged CCPM Total mass"].entries
m_a = sol["X-averaged CCPM Branch A mass"].entries
m_b = sol["X-averaged CCPM Branch B mass"].entries
m_c = sol["X-averaged CCPM Branch C mass"].entries

t_h = time[-1] / 3600
print(f"\n  Duration: {t_h:.2f}h  ({len(time)} timesteps)")
print(f"  Voltage:  {V[0]:.4f} -> {V[-1]:.4f} V")
print(f"  theta:    {theta[0]:.6f} -> {theta[-1]:.6f}")
print(f"  m_tot:    {m_tot[0]:.8f} -> {m_tot[-1]:.8f}  drift={m_tot[-1]-m_tot[0]:+.3e}")
print(f"  m_A end:  {m_a[-1]:.6f}  m_B: {m_b[-1]:.6f}  m_C: {m_c[-1]:.6f}")

# Snapshots
n = len(time)
print("\n  Timeline:")
for frac in np.linspace(0, 1, 11):
    i = min(int(frac * (n - 1)), n - 1)
    print(f"    t={time[i]/3600:6.2f}h  V={V[i]:6.4f}  theta={theta[i]:.6f}  "
          f"m_tot={m_tot[i]:.6f}  mA={m_a[i]:.4f}  mB={m_b[i]:.4f}  mC={m_c[i]:.4f}")
