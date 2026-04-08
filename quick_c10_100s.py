import numpy as np, pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

param = pybamm.ParameterValues("Prada2013")
model = DFN_CCPM(initial_branch="A", parameter_values=param)
experiment = pybamm.Experiment(["Discharge at C/10 for 100 seconds"])
sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"))
print("solving...")
sol = sim.solve()

time = sol["Time [s]"].entries
m_a = sol["X-averaged CCPM Branch A mass"].entries
m_b = sol["X-averaged CCPM Branch B mass"].entries
m_c = sol["X-averaged CCPM Branch C mass"].entries
m_tot = sol["X-averaged CCPM Total mass"].entries
theta = sol["X-averaged positive CCPM stoichiometry"].entries
V = sol["Terminal voltage [V]"].entries

n = len(time)
print(f"\n=== C/10 for 100s ===")
print(f"Time points: {n}")
for frac in np.linspace(0, 1, 11):
    i = min(int(frac*(n-1)), n-1)
    print(f"t={time[i]:6.1f}s  V={V[i]:.4f}  theta={theta[i]:.6f}  m_tot={m_tot[i]:.8f}  "
          f"mA={m_a[i]:.6f}  mB={m_b[i]:.6f}  mC={m_c[i]:.6f}  dm={m_tot[i]-m_tot[0]:+.2e}")

print(f"\nMass drift: {m_tot[-1]-m_tot[0]:+.4e} ({100*(m_tot[-1]-m_tot[0])/m_tot[0]:.4f}%)")
ga = sol["Branch A PDF CCPM"].entries
gb = sol["Branch B PDF CCPM"].entries
gc = sol["Branch C PDF CCPM"].entries
print(f"min(g_a)={np.min(ga):.4e}")
print(f"min(g_b)={np.min(gb):.4e}")
print(f"min(g_c)={np.min(gc):.4e}")
