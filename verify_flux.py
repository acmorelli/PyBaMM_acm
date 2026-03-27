"""Check FV flux vs wall correction: verify they match at runtime."""
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

cp_a = sim.mesh["CCPM positive particle branch A"].nodes
cp_a_edges = sim.mesh["CCPM positive particle branch A"].edges
dc_a = cp_a[1] - cp_a[0]
n_a = len(cp_a)

time = sol["Time [s]"].entries
n_t = len(time)

gA = np.squeeze(sol["X-averaged Branch A PDF CCPM"].entries)
if gA.ndim > 1 and gA.shape[0] != n_a:
    gA = gA.T

RA = np.squeeze(sol["X-averaged CCPM Branch A lithiation rate"].entries)
if RA.ndim > 1 and RA.shape[0] != n_a:
    RA = RA.T

# Get model-reported quantities
J_AB = np.squeeze(sol["Branch A to B scalar flux"].entries)
if J_AB.ndim > 1:
    J_AB = J_AB.mean(axis=0)

m_a = sol["X-averaged CCPM Branch A mass"].entries
m_tot = sol["X-averaged CCPM Total mass"].entries

# Reconstruct what FV actually does at the right boundary of domain A
print("Comparing FV boundary flux vs our J_A_to_B + wall_corr")
print(f"{'time':>6s}  {'g[-1]':>10s}  {'g[-2]':>10s}  {'R[-1]':>10s}  {'R[-2]':>10s}  "
      f"{'R_edge':>10s}  {'FV_flux':>10s}  {'J_AB':>10s}  {'corr_AR':>10s}  "
      f"{'J+corr':>10s}  {'diff':>10s}")

for frac in [0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)

    g_last = gA[-1, i]
    g_prev = gA[-2, i]
    r_last = RA[-1, i]
    r_prev = RA[-2, i]

    # Our R_edge (linear extrapolation)
    r_edge = 1.5 * r_last - 0.5 * r_prev

    # FV's actual boundary treatment:
    # node_to_edge(R_pos)[N] and node_to_edge(R_neg)[N]
    rp_last = max(r_last, 0)
    rp_prev = max(r_prev, 0)
    rn_last = min(r_last, 0)
    rn_prev = min(r_prev, 0)

    n2e_rpos = 1.5 * rp_last - 0.5 * rp_prev  # node_to_edge(max(R,0))
    n2e_rneg = 1.5 * rn_last - 0.5 * rn_prev  # node_to_edge(min(R,0))

    # FV flux: F = g[-1]*n2e_rpos + (-g[-1])*n2e_rneg = g[-1]*(n2e_rpos - n2e_rneg)
    fv_flux = g_last * (n2e_rpos - n2e_rneg)

    # Our decomposition
    j_ab = g_last * max(r_edge, 0)
    corr = g_last * max(-r_edge, 0)
    total = j_ab + corr

    diff = fv_flux - total

    j_model = J_AB[i] if isinstance(J_AB, np.ndarray) and J_AB.ndim == 1 else float(J_AB)

    print(f"{time[i]:6.0f}  {g_last:10.3e}  {g_prev:10.3e}  {r_last:10.3e}  {r_prev:10.3e}  "
          f"{r_edge:10.3e}  {fv_flux:10.3e}  {j_ab:10.3e}  {corr:10.3e}  "
          f"{total:10.3e}  {diff:10.3e}")

# Also check: does R change sign near boundary?
print("\nR sign check at last 5 cells, selected times:")
for frac in [0.5, 0.7, 0.9, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    print(f"  t={time[i]:.0f}s: R[-5:]={RA[-5:, i]}")

# Compute total mass flux through right edge of A over all time
# Midpoint integration over time
dt = np.diff(time)
fv_total = 0
our_total = 0
for i in range(len(time) - 1):
    g_last = 0.5 * (gA[-1, i] + gA[-1, i+1])
    r_last = 0.5 * (RA[-1, i] + RA[-1, i+1])
    r_prev = 0.5 * (RA[-2, i] + RA[-2, i+1])
    r_edge = 1.5 * r_last - 0.5 * r_prev
    rp_l = max(0.5*(max(RA[-1,i],0)+max(RA[-1,i+1],0)), 0)
    rp_p = max(0.5*(max(RA[-2,i],0)+max(RA[-2,i+1],0)), 0)
    rn_l = min(0.5*(min(RA[-1,i],0)+min(RA[-1,i+1],0)), 0)
    rn_p = min(0.5*(min(RA[-2,i],0)+min(RA[-2,i+1],0)), 0)
    n2erp = 1.5*rp_l - 0.5*rp_p
    n2ern = 1.5*rn_l - 0.5*rn_p
    fv = g_last * (n2erp - n2ern)
    ours = g_last * abs(r_edge)
    fv_total += fv * dt[i]
    our_total += ours * dt[i]

print(f"\nCumulative boundary flux (time-integrated):")
print(f"  FV approx:  {fv_total:.6e}")
print(f"  Our approx: {our_total:.6e}")
print(f"  Difference: {fv_total - our_total:.6e}")
print(f"  m_A drift:  {m_a[-1] - m_a[0]:.6e}")
print(f"  m_tot drift: {m_tot[-1] - m_tot[0]:.6e}")
