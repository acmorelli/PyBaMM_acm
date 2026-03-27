"""Test EvaluateAt behavior from actual CCPM solution."""
import numpy as np
import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

model = DFN_CCPM(build=True, initial_branch="A", source_method="flux_form")
param = pybamm.ParameterValues("Prada2013")
experiment = pybamm.Experiment(["Discharge at 1C for 2100 seconds"])
sim = pybamm.Simulation(
    model, parameter_values=param, experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"),
)
sol = sim.solve()

c_max = float(param["Maximum concentration in positive electrode [mol.m-3]"])
c_sp1 = 0.2113 * c_max
cp_a = sim.mesh["CCPM positive particle branch A"].nodes
dc = cp_a[1] - cp_a[0]

gA = np.squeeze(sol["X-averaged Branch A PDF CCPM"].entries)
if gA.ndim > 1 and gA.shape[0] != len(cp_a):
    gA = gA.T

J_AB = np.squeeze(sol["Branch A to B scalar flux"].entries)
if J_AB.ndim > 1:
    J_AB = J_AB.mean(axis=0)

R_a = np.squeeze(sol["X-averaged CCPM Branch A lithiation rate"].entries)
if R_a.ndim > 1 and R_a.shape[0] != len(cp_a):
    R_a = R_a.T

times = sol["Time [s]"].entries
n = len(times)

print("Comparing J_A_to_B: g_last*R vs g_extrap*R vs model J_AB")
print("(g_last = FV upwind, g_extrap = linear extrap to boundary)")
print()
for frac in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
    i = min(int(frac * (n - 1)), n - 1)
    gl = gA[-1, i]
    gp = gA[-2, i]
    ge = 1.5 * gl - 0.5 * gp
    rl = R_a[-1, i]
    rp = R_a[-2, i]
    re = 1.5 * rl - 0.5 * rp  # R at boundary (edge extrapolation)
    j_cell = gl * max(re, 0)
    j_extrap = ge * max(re, 0)
    jab = J_AB[i]
    match_cell = "<<< MATCH cell" if abs(jab - j_cell) < abs(jab - j_extrap) else ""
    match_extrap = "<<< MATCH extrap" if abs(jab - j_extrap) < abs(jab - j_cell) else ""
    print(f"t={times[i]:6.0f}s  g[-1]={gl:.4e}  g_ext={ge:.4e}  "
          f"J_cell={j_cell:.4e}  J_ext={j_extrap:.4e}  J_model={jab:.4e}  "
          f"{match_cell}{match_extrap}")
