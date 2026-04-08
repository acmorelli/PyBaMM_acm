"""
Debug script: verify Branch C initial distribution.

Sets c_init deep in Branch C (θ = 0.90) and checks:
  1. PDF is non-negative everywhere
  2. Mean of g_c equals c_init
  3. Tails go to zero well before domain boundaries
  4. g_c is zero (or negligible) below c_sp2
  5. g_a and g_b are identically zero
  6. Total mass = 1
  7. Nothing is clipped at domain edges
"""
import numpy as np
from numpy import trapezoid as trapz
import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

# ---------- parameters ----------
theta_init = 0.90  # well inside Branch C (c_sp2 ~ 0.79, c2_star ~ 0.93)
parameter_values = pybamm.ParameterValues("Prada2013")
c_max_raw = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
c_init_target = theta_init * c_max_raw
eps_c = 1e-8 * c_max_raw

# Override c_init for positive electrode
parameter_values["Initial concentration in positive electrode [mol.m-3]"] = c_init_target

# Also need a consistent negative electrode init — use a manual value
# (for init-only test the negative side doesn't matter much, but solver wants it)
parameter_values["Initial concentration in negative electrode [mol.m-3]"] = (
    0.10 * parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
)

print(f"c_max_raw        = {c_max_raw:.2f}")
print(f"c_init (target)  = {c_init_target:.2f}  (theta = {theta_init})")
print(f"eps_c            = {eps_c:.6e}")
print()

# ---------- domain/mesh geometry ----------
npts = 300
c_min = eps_c
c_max = c_max_raw - eps_c
L = c_max - c_min

c1_star = c_min + (21 / npts) * L
c_sp1   = c_min + (63 / npts) * L
c_sp2   = c_min + (237 / npts) * L
c2_star = c_min + (279 / npts) * L

print("Transition points (concentration / theta):")
print(f"  c1_star = {c1_star:10.2f}  (theta = {c1_star/c_max_raw:.4f})")
print(f"  c_sp1   = {c_sp1:10.2f}  (theta = {c_sp1/c_max_raw:.4f})")
print(f"  c_sp2   = {c_sp2:10.2f}  (theta = {c_sp2/c_max_raw:.4f})")
print(f"  c2_star = {c2_star:10.2f}  (theta = {c2_star/c_max_raw:.4f})")
print()

# Cell edges (uniform grid)
cell_edges = np.array([c_min + (k / npts) * L for k in range(npts + 1)])
cell_centres = 0.5 * (cell_edges[:-1] + cell_edges[1:])
dc = cell_edges[1] - cell_edges[0]  # uniform spacing

print(f"Cell width dc = {dc:.4f}")
print(f"Domain: [{c_min:.4f}, {c_max:.4f}]")
print(f"First 3 centres: {cell_centres[:3]}")
print(f"Last  3 centres: {cell_centres[-3:]}")
print()

# ---------- build model and solve for t=0 only ----------
model = DFN_CCPM(initial_branch="C", parameter_values=parameter_values)

# Tiny discharge just to get solver to evaluate t=0
experiment = pybamm.Experiment(["Discharge at C/30 for 1 seconds"])
sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"),
)
sol = sim.solve()

# ---------- extract initial PDFs ----------
g_a_all = sol["Branch A PDF CCPM"].entries  # shape (npts, n_x, n_t)
g_b_all = sol["Branch B PDF CCPM"].entries
g_c_all = sol["Branch C PDF CCPM"].entries

# Take t=0, x-average (first time index, average over electrode)
g_a_0 = g_a_all[:, :, 0].mean(axis=1)  # (npts,)
g_b_0 = g_b_all[:, :, 0].mean(axis=1)
g_c_0 = g_c_all[:, :, 0].mean(axis=1)

print("=" * 70)
print("INITIAL DISTRIBUTION CHECKS")
print("=" * 70)

# --- Check 1: g_a and g_b should be zero ---
max_g_a = np.max(np.abs(g_a_0))
max_g_b = np.max(np.abs(g_b_0))
print(f"\n[1] g_a max abs value: {max_g_a:.4e}  {'PASS' if max_g_a < 1e-15 else 'FAIL'}")
print(f"    g_b max abs value: {max_g_b:.4e}  {'PASS' if max_g_b < 1e-15 else 'FAIL'}")

# --- Check 2: g_c non-negative ---
min_g_c = np.min(g_c_0)
print(f"\n[2] g_c min value: {min_g_c:.4e}  {'PASS' if min_g_c >= -1e-15 else 'FAIL (NEGATIVE!)'}")

# --- Check 3: total mass = 1 ---
mass_c = trapz(g_c_0, cell_centres)
print(f"\n[3] Total mass (integral of g_c): {mass_c:.10f}  {'PASS' if abs(mass_c - 1.0) < 1e-4 else 'FAIL'}")

# --- Check 4: mean = c_init ---
mean_c = trapz(g_c_0 * cell_centres, cell_centres) / mass_c
theta_mean = mean_c / c_max_raw
print(f"\n[4] Mean of g_c: {mean_c:.4f}  (theta = {theta_mean:.6f})")
print(f"    Target:      {c_init_target:.4f}  (theta = {theta_init:.6f})")
print(f"    Error:       {abs(mean_c - c_init_target):.4e}  "
      f"({'PASS' if abs(theta_mean - theta_init) < 0.002 else 'FAIL'})")

# --- Check 5: g_c is zero below c_sp2 ---
i_sp2 = np.searchsorted(cell_centres, c_sp2)
max_below_sp2 = np.max(np.abs(g_c_0[:i_sp2]))
print(f"\n[5] g_c max below c_sp2 (cell {i_sp2}): {max_below_sp2:.4e}  "
      f"{'PASS' if max_below_sp2 < 1e-12 else 'FAIL'}")

# --- Check 6: tails approach zero near c_sp2 and near c_max ---
# Find first non-negligible cell from the left (above c_sp2)
thresh = 1e-6 * np.max(g_c_0)
first_nonzero = next((i for i in range(npts) if g_c_0[i] > thresh), npts)
last_nonzero = next((i for i in range(npts - 1, -1, -1) if g_c_0[i] > thresh), 0)

print(f"\n[6] Tail analysis (threshold = {thresh:.4e}):")
print(f"    First non-negligible cell: {first_nonzero}  "
      f"(c = {cell_centres[first_nonzero]:.2f}, theta = {cell_centres[first_nonzero]/c_max_raw:.4f})")
print(f"    Last  non-negligible cell: {last_nonzero}  "
      f"(c = {cell_centres[last_nonzero]:.2f}, theta = {cell_centres[last_nonzero]/c_max_raw:.4f})")
print(f"    c_sp2 is at cell ~{i_sp2}  (theta ~ 0.79)")
print(f"    Right boundary at cell {npts-1}  (theta ~ 1.0)")

# Cells between c_sp2 and first nonzero
gap_left = first_nonzero - i_sp2
print(f"    Gap: {gap_left} cells between c_sp2 and first nonzero g_c")

# Cells between last nonzero and right boundary
gap_right = (npts - 1) - last_nonzero
print(f"    Gap: {gap_right} cells between last nonzero g_c and right boundary")

# Check explicit values at boundaries
print(f"\n    g_c at domain edges:")
print(f"      cell 0   (left  boundary): {g_c_0[0]:.4e}")
print(f"      cell {npts-1} (right boundary): {g_c_0[-1]:.4e}")
print(f"      cell {i_sp2}  (at c_sp2):       {g_c_0[i_sp2]:.4e}")
if i_sp2 + 1 < npts:
    print(f"      cell {i_sp2+1}  (c_sp2 + 1):     {g_c_0[i_sp2 + 1]:.4e}")

boundary_clipped = (g_c_0[-1] > 0.01 * np.max(g_c_0))
print(f"\n    Right boundary clipping: {'FAIL (clipped!)' if boundary_clipped else 'PASS (tail decayed)'}")

# --- Check 7: peak location ---
i_peak = np.argmax(g_c_0)
c_peak = cell_centres[i_peak]
print(f"\n[7] Peak at cell {i_peak}, c = {c_peak:.2f}  (theta = {c_peak/c_max_raw:.4f})")
print(f"    Expected peak near c_init = {c_init_target:.2f}  (theta = {theta_init})")

# --- Print detailed profile around the support ---
print(f"\n{'='*70}")
print("DETAILED g_c PROFILE (non-zero region +/- 5 cells)")
print(f"{'='*70}")
i_start = max(0, first_nonzero - 5)
i_end = min(npts, last_nonzero + 6)
print(f"{'cell':>5s}  {'c':>10s}  {'theta':>8s}  {'g_c':>14s}  {'note'}")
for i in range(i_start, i_end):
    c_val = cell_centres[i]
    th = c_val / c_max_raw
    note = ""
    if i == i_peak:
        note = "<-- PEAK"
    if abs(c_val - c_sp2) < dc:
        note += " ~c_sp2"
    if abs(c_val - c2_star) < dc:
        note += " ~c2_star"
    if i == 0:
        note += " LEFT-BOUNDARY"
    if i == npts - 1:
        note += " RIGHT-BOUNDARY"
    print(f"{i:5d}  {c_val:10.2f}  {th:8.4f}  {g_c_0[i]:14.6e}  {note}")

print(f"\n{'='*70}")
print("ALL CHECKS COMPLETE")
print(f"{'='*70}")
