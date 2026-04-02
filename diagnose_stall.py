"""
Diagnostic script: why does the C/30 discharge stall?
- Mass stuck in branch A (~60%), branch C only ~39%
- theta_CCPM stuck near 0.50
- g_a frozen between 80640s and 107540s
- R_a looks the same at all x nodes
- OCP on plateau, no sharp drop at 2V cutoff
"""
import numpy as np
import pybamm

# ── Load saved solution ────────────────────────────────────────────────
sim = pybamm.load("C30_2V_modeDch.pkl")
solution = sim.solution

def arr(name):
    return np.asarray(solution[name].entries)

time = solution["Time [s]"].entries
n_t = len(time)

# ── 1. Mass budget ────────────────────────────────────────────────────
m_a = arr("X-averaged CCPM Branch A mass")
m_b = arr("X-averaged CCPM Branch B mass")
m_c = arr("X-averaged CCPM Branch C mass")
m_tot = arr("X-averaged CCPM Total mass")

print("=" * 70)
print("1. MASS BUDGET")
print("=" * 70)
for frac in [0, 0.25, 0.5, 0.75, 0.9, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    print(f"  t={time[i]:8.0f}s  mA={m_a[i]:.6f} ({100*m_a[i]/m_tot[i]:.1f}%)  "
          f"mB={m_b[i]:.4e} ({100*m_b[i]/m_tot[i]:.1f}%)  "
          f"mC={m_c[i]:.4e} ({100*m_c[i]/m_tot[i]:.1f}%)  "
          f"total={m_tot[i]:.6f}")

print(f"\n  Total mass drift: {m_tot[-1]-m_tot[0]:+.4e} ({100*(m_tot[-1]-m_tot[0])/m_tot[0]:+.4f}%)")

# ── 2. Theta, voltage, OCP ───────────────────────────────────────────
theta = arr("X-averaged positive CCPM stoichiometry")
voltage = arr("Terminal voltage [V]")
ocp = arr("X-averaged positive electrode open-circuit potential [V]")

print("\n" + "=" * 70)
print("2. THETA / VOLTAGE / OCP")
print("=" * 70)
for frac in [0, 0.25, 0.5, 0.75, 0.9, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    print(f"  t={time[i]:8.0f}s  theta={theta[i]:.6f}  V={voltage[i]:.4f}  OCP={ocp[i]:.4f}")

# ── 3. Transition fluxes J_AB, J_BC ─────────────────────────────────
J_AB = arr("Branch A to B scalar flux")
J_BC = arr("Branch B to C scalar flux")

def mean_over_x(a, i):
    if a.ndim > 1:
        return np.mean(a[:, i])
    return a[i]

print("\n" + "=" * 70)
print("3. TRANSITION FLUXES (x-averaged)")
print("=" * 70)
for frac in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    jab = mean_over_x(J_AB, i)
    jbc = mean_over_x(J_BC, i)
    print(f"  t={time[i]:8.0f}s  J_AB={jab:+.6e}  J_BC={jbc:+.6e}")

# ── 4. g_a at c_sp1 boundary (is the PDF reaching the transition?) ──
g_a_raw = arr("Branch A PDF CCPM")
g_b_raw = arr("Branch B PDF CCPM")
g_c_raw = arr("Branch C PDF CCPM")

c_p_max_raw = 22806.0  # Prada2013
eps_c = 1e-8 * c_p_max_raw
c_max_eff = c_p_max_raw - eps_c
n_cp = 300
cp_edges = np.linspace(eps_c, c_max_eff, n_cp + 1)
cp_nodes = 0.5 * (cp_edges[:-1] + cp_edges[1:])
dc = cp_edges[1] - cp_edges[0]

# Branch boundary indices
idx_c1star = 21   # cell edge index for c1_star
idx_csp1 = 63     # cell edge index for c_sp1
idx_csp2 = 237    # cell edge index for c_sp2
idx_c2star = 279  # cell edge index for c2_star

# Cell centres near boundaries
# Cell i has edges [i, i+1], so cell 62 is the last A cell, cell 63 is first B cell
last_A_cell = idx_csp1 - 1  # cell 62
first_B_cell = idx_csp1     # cell 63
last_B_cell_near_c2 = idx_c2star - 1  # cell 278
first_C_cell = idx_c2star   # cell 279

n_flat = g_a_raw.shape[0] if g_a_raw.ndim == 2 else g_a_raw.size // n_t
n_x = n_flat // n_cp

print(f"\n  Mesh: {n_cp} cells, {n_x} x-nodes, dc={dc:.2f} mol/m3")
print(f"  c_sp1 = cell edge {idx_csp1}, c_node[{last_A_cell}]={cp_nodes[last_A_cell]:.1f}, c_node[{first_B_cell}]={cp_nodes[first_B_cell]:.1f}")
print(f"  c2_star = cell edge {idx_c2star}, c_node[{last_B_cell_near_c2}]={cp_nodes[last_B_cell_near_c2]:.1f}, c_node[{first_C_cell}]={cp_nodes[first_C_cell]:.1f}")

def reshape_3d(flat):
    flat = np.squeeze(flat)
    if flat.ndim == 2 and flat.shape[0] == n_cp * n_x:
        return flat.reshape((n_cp, n_x, n_t), order="F")
    elif flat.ndim == 2:
        return flat[:, np.newaxis, :] * np.ones((1, n_x, 1))
    return flat

ga_3d = reshape_3d(g_a_raw)
gb_3d = reshape_3d(g_b_raw)
gc_3d = reshape_3d(g_c_raw)

x_mid = n_x // 2  # mid electrode node

print("\n" + "=" * 70)
print("4. g_a NEAR c_sp1 BOUNDARY (x=mid)")
print("=" * 70)
print(f"  {'time':>10s}  {'g_a[cell60]':>12s}  {'g_a[cell61]':>12s}  {'g_a[cell62]':>12s}  {'g_a[cell63]':>12s}  {'g_a[cell64]':>12s}")
for frac in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    vals = [ga_3d[c, x_mid, i] for c in range(60, 65)]
    print(f"  {time[i]:10.0f}s  " + "  ".join(f"{v:12.6e}" for v in vals))

print("\n" + "=" * 70)
print("5. g_a PEAK LOCATION AND MOVEMENT (x=mid)")
print("=" * 70)
for frac in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    ga_slice = ga_3d[:, x_mid, i]
    peak_idx = np.argmax(ga_slice)
    peak_val = ga_slice[peak_idx]
    # also check mass in A region (cells 0..62)
    ga_mass_A = np.sum(ga_slice[:idx_csp1]) * dc
    ga_mass_outside = np.sum(ga_slice[idx_csp1:]) * dc
    print(f"  t={time[i]:8.0f}s  peak_cell={peak_idx:3d}  peak_theta={cp_nodes[peak_idx]/c_p_max_raw:.4f}  "
          f"peak_g={peak_val:.4e}  mass_in_A={ga_mass_A:.6f}  mass_outside_A={ga_mass_outside:.4e}")

# ── 6. R_a profile: is R actually pushing mass toward c_sp1? ────────
R_a_raw = arr("CCPM Branch A lithiation rate")
R_b_raw = arr("CCPM Branch B lithiation rate")
R_c_raw = arr("CCPM Branch C lithiation rate")

Ra_3d = reshape_3d(R_a_raw)
Rb_3d = reshape_3d(R_b_raw)
Rc_3d = reshape_3d(R_c_raw)

print("\n" + "=" * 70)
print("6. R_a PROFILE (x=mid) — should be positive (rightward) during discharge")
print("=" * 70)
# Sample a few c-cells across branch A
sample_cells = [0, 10, 20, 30, 40, 50, 60, 62]
header = f"  {'time':>10s}  " + "  ".join(f"R_a[c{c}]" for c in sample_cells)
print(header)
for frac in [0, 0.25, 0.5, 0.75, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    vals = [Ra_3d[c, x_mid, i] for c in sample_cells]
    print(f"  {time[i]:10.0f}s  " + "  ".join(f"{v:+.3e}" for v in vals))

# Check R_a sign across the full branch A at final time
Ra_final = Ra_3d[:idx_csp1, x_mid, -1]
print(f"\n  At final time, R_a in branch A (cells 0..{idx_csp1-1}):")
print(f"    min={Ra_final.min():+.4e}  max={Ra_final.max():+.4e}  mean={Ra_final.mean():+.4e}")
n_positive = np.sum(Ra_final > 0)
n_negative = np.sum(Ra_final < 0)
print(f"    R_a>0: {n_positive} cells, R_a<0: {n_negative} cells")

# ── 7. Check if g_a has leaked outside branch A mask ─────────────────
print("\n" + "=" * 70)
print("7. g_a LEAKAGE OUTSIDE BRANCH A REGION (cells >= c_sp1)")
print("=" * 70)
for frac in [0, 0.5, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    ga_outside = ga_3d[idx_csp1:, x_mid, i]
    print(f"  t={time[i]:8.0f}s  max(g_a outside A) = {ga_outside.max():.4e}  "
          f"sum*dc = {np.sum(ga_outside)*dc:.4e}")

# ── 8. RHS balance: is divergence being properly cancelled by source? ─
print("\n" + "=" * 70)
print("8. RHS INTEGRATED (should show flux leaving A = source entering B)")
print("=" * 70)
try:
    rhs_int_a = arr("Branch A PDF CCPM RHS Integrated on c_p")
    rhs_int_b = arr("Branch B PDF CCPM RHS Integrated on c_p")
    rhs_int_c = arr("Branch C PDF CCPM RHS Integrated on c_p")
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        i = min(int(frac * (n_t - 1)), n_t - 1)
        ra = mean_over_x(rhs_int_a, i)
        rb = mean_over_x(rhs_int_b, i)
        rc = mean_over_x(rhs_int_c, i)
        print(f"  t={time[i]:8.0f}s  ∫rhs_a={ra:+.4e}  ∫rhs_b={rb:+.4e}  ∫rhs_c={rc:+.4e}  sum={ra+rb+rc:+.4e}")
except Exception as e:
    print(f"  Could not extract RHS integrals: {e}")

# ── 9. Source mass check ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("9. SOURCE TERMS — checking S_a, S_b, S_c integrated")
print("=" * 70)
try:
    src_err = arr("CCPM source mass balance error")
    for frac in [0, 0.5, 1.0]:
        i = min(int(frac * (n_t - 1)), n_t - 1)
        se = mean_over_x(src_err, i)
        jab = mean_over_x(J_AB, i)
        jbc = mean_over_x(J_BC, i)
        print(f"  t={time[i]:8.0f}s  J_AB={jab:+.6e}  J_BC={jbc:+.6e}  source_balance_err={se:+.4e}")
except Exception as e:
    print(f"  Could not extract source error: {e}")

# ── 10. Key insight: does the advective flux F_a actually reach c_sp1? ─
print("\n" + "=" * 70)
print("10. ADVECTIVE FLUX F_a: div(F_a) telescoping check")
print("=" * 70)
print("  The transition J_A_to_B = Integral(div(F_a) * mask_a, c_p)")
print("  This telescopes to F_a at c_sp1 boundary (right edge of A)")
print("  If F_a(c_sp1) ≈ 0, no mass transfers A→B!")
print()
print("  F_a = Upwind(g_a)*R_pos + Downwind(g_a)*R_neg")
print("  At c_sp1 edge: Upwind(g_a) = g_a[cell 62], R_pos = smooth_max(R_a, 0)")
print()
print("  Checking g_a at last A cell (cell 62) and R_a there:")
for frac in [0, 0.25, 0.5, 0.75, 0.9, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    ga_62 = ga_3d[last_A_cell, x_mid, i]
    Ra_62 = Ra_3d[last_A_cell, x_mid, i]
    Ra_pos = max(Ra_62, 0)
    flux_est = ga_62 * Ra_pos
    print(f"  t={time[i]:8.0f}s  g_a[62]={ga_62:.6e}  R_a[62]={Ra_62:+.6e}  "
          f"R_pos={Ra_pos:.6e}  est_F_a(csp1)={flux_est:.6e}")

# ── 11. Rate of mass change: dm_a/dt vs J_AB ─────────────────────────
print("\n" + "=" * 70)
print("11. dm_a/dt vs J_AB (should be dm_a/dt ≈ -J_AB for discharge)")
print("=" * 70)
dt = np.diff(time)
dm_a = np.diff(m_a)
dm_a_dt = dm_a / dt
for frac in [0.25, 0.5, 0.75, 0.9]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    if i >= len(dm_a_dt):
        i = len(dm_a_dt) - 1
    jab = mean_over_x(J_AB, i)
    print(f"  t={time[i]:8.0f}s  dm_a/dt={dm_a_dt[i]:+.6e}  -J_AB={-jab:+.6e}  "
          f"ratio={dm_a_dt[i]/(-jab) if abs(jab)>1e-15 else float('nan'):.4f}")

# ── 12. CRITICAL: Check if mask_a is KILLING the RHS ─────────────────
print("\n" + "=" * 70)
print("12. MASK ANALYSIS — rhs_a = -div(F_a) * mask_a")
print("=" * 70)
print("  mask_a = (c_p <= c_sp1) → cells 0..62 are 1, cells 63..299 are 0")
print("  BUT: div(F_a) is computed on the FULL domain (300 cells)")
print("  Then multiplied by mask_a, zeroing everything beyond cell 62")
print()
print("  The problem: F_a is computed from Upwind(g_a)*R_pos globally.")
print("  ∂g_a/∂t = (-div(F_a) + S_a) on all 300 cells")
print("  But rhs_a = -div(F_a)*mask_a + S_a")
print("  So g_a dynamics are KILLED for cells 63-299 by the mask.")
print("  This is intentional — g_a should only live in branch A.")
print()
print("  But the issue is that g_a CAN'T exit A. Let's verify if J_AB")
print("  is extracting any flux:")

# Verify the telescoping: sum of div(F_a) over mask_a cells
# This should equal F_a at c_sp1 boundary
print("\n  Checking J_AB magnitude over time:")
if J_AB.ndim > 1:
    jab_mean = np.mean(J_AB, axis=0)
else:
    jab_mean = J_AB
print(f"  J_AB range: [{jab_mean.min():.6e}, {jab_mean.max():.6e}]")
print(f"  J_AB at end: {jab_mean[-1]:.6e}")

# ── 13. THE STALLING HYPOTHESIS ──────────────────────────────────────
print("\n" + "=" * 70)
print("13. STALLING HYPOTHESIS TEST")
print("=" * 70)
print("  If g_a peak has moved to c_sp1 but R_a sign changes (from Clarke eq),")
print("  particles at high c in branch A may have R_a < 0 (delithiation),")
print("  which means they move LEFT, not right toward c_sp1.")
print()
# Check R_a sign near c_sp1 at different times
print("  R_a near c_sp1 boundary (cells 55-62, x=mid):")
cells_near_csp1 = list(range(55, 63))
for frac in [0.5, 0.75, 0.9, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    vals = [Ra_3d[c, x_mid, i] for c in cells_near_csp1]
    print(f"  t={time[i]:8.0f}s  " + "  ".join(f"{v:+.2e}" for v in vals))

# ── 14. g_a shape evolution: is it broadening or translating? ────────
print("\n" + "=" * 70)
print("14. g_a DISTRIBUTION MOMENTS (x=mid, cells 0..62)")
print("=" * 70)
for frac in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
    i = min(int(frac * (n_t - 1)), n_t - 1)
    ga_A = ga_3d[:idx_csp1, x_mid, i]
    c_A = cp_nodes[:idx_csp1]
    mass = np.sum(ga_A) * dc
    if mass > 1e-15:
        mean_c = np.sum(ga_A * c_A) * dc / mass
        var_c = np.sum(ga_A * (c_A - mean_c)**2) * dc / mass
        std_c = np.sqrt(max(var_c, 0))
        mean_theta = mean_c / c_p_max_raw
    else:
        mean_theta = 0
        std_c = 0
    print(f"  t={time[i]:8.0f}s  mass_A={mass:.6f}  <theta_A>={mean_theta:.6f}  "
          f"std_c={std_c:.1f} mol/m3  (std_theta={std_c/c_p_max_raw:.6f})")

print("\n" + "=" * 70)
print("DONE — check output above for the stalling root cause")
print("=" * 70)
