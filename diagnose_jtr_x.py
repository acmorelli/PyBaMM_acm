"""
Diagnose: why does jtr_a have different signs at x=0, x=mid, x=end?
Focus on the macroscopic fields (phi_s, phi_e, T) that drive j_tr_a.
"""
import numpy as np
import pybamm

sim = pybamm.load("C30_2V_modeDch.pkl")
solution = sim.solution

def arr(name):
    return np.asarray(solution[name].entries)

time = solution["Time [s]"].entries
n_t = len(time)

# ── Mesh info ──────────────────────────────────────────────────────────
c_p_max_raw = 22806.0
eps_c = 1e-8 * c_p_max_raw
c_max_eff = c_p_max_raw - eps_c
n_cp = 300
cp_edges = np.linspace(eps_c, c_max_eff, n_cp + 1)
cp_nodes = 0.5 * (cp_edges[:-1] + cp_edges[1:])
dc = cp_edges[1] - cp_edges[0]

# Branch boundaries
npts = 300
L = c_max_eff - eps_c
c_sp1 = eps_c + (63/npts) * L

# ── Extract 3D arrays ─────────────────────────────────────────────────
jtr_a_raw = arr("CCPM Branch A interfacial current density [A.m-2]")
R_a_raw = arr("CCPM Branch A lithiation rate")

n_flat = jtr_a_raw.shape[0] if jtr_a_raw.ndim == 2 else jtr_a_raw.size // n_t
n_x = n_flat // n_cp
print(f"Mesh: {n_cp} c-cells, {n_x} x-nodes, {n_t} time steps")

def reshape_3d(flat):
    flat = np.squeeze(flat)
    if flat.ndim == 2 and flat.shape[0] == n_cp * n_x:
        return flat.reshape((n_cp, n_x, n_t), order="F")
    elif flat.ndim == 2:
        return flat[:, np.newaxis, :] * np.ones((1, n_x, 1))
    return flat

jtra_3d = reshape_3d(jtr_a_raw)
Ra_3d = reshape_3d(R_a_raw)

# x-node indices
x0, xmid, xend = 0, n_x // 2, n_x - 1

# ── 1. j_tr_a at key c-cells across x positions ──────────────────────
print("\n" + "=" * 80)
print("1. j_tr_a(c, x, t) — comparing x=0, x=mid, x=end")
print("=" * 80)

# Pick cells near the R_a sign change (cells 55-62) and a few lower ones
sample_cells = [5, 20, 40, 55, 58, 60, 61, 62]

for tidx_frac in [0.5, 0.75, 1.0]:
    ti = min(int(tidx_frac * (n_t - 1)), n_t - 1)
    print(f"\n  t = {time[ti]:.0f}s ({tidx_frac*100:.0f}% of sim)")
    print(f"  {'cell':>6s}  {'theta':>8s}  {'jtr_a(x=0)':>14s}  {'jtr_a(x=mid)':>14s}  {'jtr_a(x=end)':>14s}  {'jtr_a spread':>14s}")
    for c in sample_cells:
        j0 = jtra_3d[c, x0, ti]
        jm = jtra_3d[c, xmid, ti]
        je = jtra_3d[c, xend, ti]
        spread = max(j0, jm, je) - min(j0, jm, je)
        print(f"  {c:6d}  {cp_nodes[c]/c_p_max_raw:8.4f}  {j0:+14.6e}  {jm:+14.6e}  {je:+14.6e}  {spread:14.6e}")

# ── 2. R_a at key c-cells across x positions ─────────────────────────
print("\n" + "=" * 80)
print("2. R_a(c, x, t) — comparing x=0, x=mid, x=end")
print("=" * 80)

for tidx_frac in [0.5, 0.75, 1.0]:
    ti = min(int(tidx_frac * (n_t - 1)), n_t - 1)
    print(f"\n  t = {time[ti]:.0f}s")
    print(f"  {'cell':>6s}  {'theta':>8s}  {'R_a(x=0)':>14s}  {'R_a(x=mid)':>14s}  {'R_a(x=end)':>14s}")
    for c in sample_cells:
        r0 = Ra_3d[c, x0, ti]
        rm = Ra_3d[c, xmid, ti]
        re = Ra_3d[c, xend, ti]
        print(f"  {c:6d}  {cp_nodes[c]/c_p_max_raw:8.4f}  {r0:+14.6e}  {rm:+14.6e}  {re:+14.6e}")

# ── 3. Macroscopic potentials: phi_s, phi_e, delta_phi, T ────────────
print("\n" + "=" * 80)
print("3. MACROSCOPIC POTENTIALS across positive electrode x-nodes")
print("=" * 80)

phi_s = arr("Positive electrode potential [V]")
phi_e = arr("Positive electrolyte potential [V]")
T_p = arr("Positive electrode temperature [K]")
c_e = arr("Positive electrolyte concentration [mol.m-3]")
delta_phi = arr("Positive electrode surface potential difference [V]")

# These are (n_x, n_t) arrays
print(f"  phi_s shape: {phi_s.shape}")
print(f"  phi_e shape: {phi_e.shape}")

for tidx_frac in [0.5, 0.75, 1.0]:
    ti = min(int(tidx_frac * (n_t - 1)), n_t - 1)
    print(f"\n  t = {time[ti]:.0f}s")

    if phi_s.ndim == 2:
        n_x_macro = phi_s.shape[0]
        x_indices = [0, n_x_macro // 2, n_x_macro - 1]
        print(f"  {'x_idx':>6s}  {'phi_s [V]':>12s}  {'phi_e [V]':>12s}  {'dphi [V]':>12s}  {'T [K]':>10s}  {'c_e [mol/m3]':>14s}")
        for xi in x_indices:
            ps = phi_s[xi, ti]
            pe = phi_e[xi, ti]
            dp = delta_phi[xi, ti] if delta_phi.ndim == 2 else delta_phi[ti]
            tp = T_p[xi, ti] if T_p.ndim == 2 else T_p[ti]
            ce = c_e[xi, ti] if c_e.ndim == 2 else c_e[ti]
            print(f"  {xi:6d}  {ps:12.6f}  {pe:12.6f}  {dp:12.6f}  {tp:10.4f}  {ce:14.4f}")
    else:
        print(f"  phi_s = {phi_s[ti]:.6f}, phi_e = {phi_e[ti]:.6f}")

# ── 4. Compute the algebraic overpotential argument fed to sinh ──────
print("\n" + "=" * 80)
print("4. SINH ARGUMENT = (F*(phi_s - phi_e) - F*U_eq_p0 + mu(c)) / (2*R*T)")
print("   at the R_a zero-crossing (cells 60-62) for each x position")
print("=" * 80)

F_const = 96485.33212
R_const = 8.314462618
U_eq_p0 = 3.397
omega = 3.0

for tidx_frac in [0.75, 1.0]:
    ti = min(int(tidx_frac * (n_t - 1)), n_t - 1)
    print(f"\n  t = {time[ti]:.0f}s")

    if phi_s.ndim == 2:
        n_x_macro = phi_s.shape[0]
        # Map CCPM x-indices to macro x-indices
        # CCPM has n_x nodes, macro has n_x_macro nodes
        macro_x = [0, n_x_macro // 2, n_x_macro - 1]
        ccpm_x = [x0, xmid, xend]
        x_labels = ["x=0", "x=mid", "x=end"]

        for xi_macro, xi_ccpm, xlabel in zip(macro_x, ccpm_x, x_labels):
            ps = phi_s[xi_macro, ti]
            pe = phi_e[xi_macro, ti]
            if T_p.ndim == 2:
                tp = T_p[xi_macro, ti]
            else:
                tp = T_p[ti]
            if c_e.ndim == 2:
                ce = c_e[xi_macro, ti]
            else:
                ce = c_e[ti]
            c_e_init = 1000.0  # typical Prada2013

            print(f"\n  {xlabel}: phi_s={ps:.6f}, phi_e={pe:.6f}, T={tp:.2f}K, c_e={ce:.2f}")
            print(f"  {'cell':>6s}  {'theta':>8s}  {'mu/(RT)':>12s}  {'F*dphi/(2RT)':>14s}  {'sinh_arg':>12s}  {'j_tr_a':>14s}  {'R_a':>14s}")

            for c in [55, 58, 60, 61, 62]:
                theta = cp_nodes[c] / c_p_max_raw
                # mu_a = RT * [ln(theta/(1-theta)) + omega*(1 - 2*theta)]
                mu_over_RT = np.log(theta / (1 - theta)) + omega * (1 - 2 * theta)
                # overpotential argument for sinh
                # j_tr_a = ... * sinh((F*(phi_s-phi_e) - F*U_eq_p0 + mu) / (2RT))
                # = ... * sinh((F*dphi - F*U_eq + RT*mu_over_RT) / (2RT))
                dphi = ps - pe
                sinh_arg = (F_const * dphi - F_const * U_eq_p0 + R_const * tp * mu_over_RT) / (2 * R_const * tp)

                jtr = jtra_3d[c, xi_ccpm, ti]
                ra = Ra_3d[c, xi_ccpm, ti]
                print(f"  {c:6d}  {theta:8.4f}  {mu_over_RT:12.6f}  {F_const*dphi/(2*R_const*tp):14.6f}  {sinh_arg:12.6f}  {jtr:+14.6e}  {ra:+14.6e}")

# ── 5. j_tr_a sign flip: where exactly does it happen per x? ────────
print("\n" + "=" * 80)
print("5. j_tr_a SIGN-FLIP CELL INDEX per x-node (at final time)")
print("=" * 80)

ti = n_t - 1
print(f"  t = {time[ti]:.0f}s")
for xi in range(n_x):
    jtr_slice = jtra_3d[:63, xi, ti]  # branch A cells only
    # Find where j_tr_a changes sign
    sign_changes = np.where(np.diff(np.sign(jtr_slice)))[0]
    if len(sign_changes) > 0:
        flip_cell = sign_changes[-1]  # last sign change
        flip_theta = cp_nodes[flip_cell] / c_p_max_raw
        jtr_before = jtr_slice[flip_cell]
        jtr_after = jtr_slice[flip_cell + 1]
    else:
        flip_cell = -1
        flip_theta = 0
        jtr_before = jtr_after = 0
    if xi in [0, n_x//4, n_x//2, 3*n_x//4, n_x-1]:
        print(f"  x={xi:3d}  flip_cell={flip_cell:3d}  flip_theta={flip_theta:.4f}  "
              f"jtr[before]={jtr_before:+.4e}  jtr[after]={jtr_after:+.4e}")

# ── 6. R_a sign-flip cell index per x-node ──────────────────────────
print("\n" + "=" * 80)
print("6. R_a SIGN-FLIP CELL INDEX per x-node (at final time)")
print("=" * 80)

for xi in range(n_x):
    Ra_slice = Ra_3d[:63, xi, ti]
    sign_changes = np.where(np.diff(np.sign(Ra_slice)))[0]
    if len(sign_changes) > 0:
        flip_cell = sign_changes[-1]
        flip_theta = cp_nodes[flip_cell] / c_p_max_raw
        ra_before = Ra_slice[flip_cell]
        ra_after = Ra_slice[flip_cell + 1]
    else:
        flip_cell = -1
        flip_theta = 0
        ra_before = ra_after = 0
    if xi in [0, n_x//4, n_x//2, 3*n_x//4, n_x-1]:
        print(f"  x={xi:3d}  flip_cell={flip_cell:3d}  flip_theta={flip_theta:.4f}  "
              f"R_a[before]={ra_before:+.4e}  R_a[after]={ra_after:+.4e}")

# ── 7. delta_phi spread across x ────────────────────────────────────
print("\n" + "=" * 80)
print("7. DELTA_PHI = phi_s - phi_e SPREAD across x over time")
print("=" * 80)
if phi_s.ndim == 2:
    n_x_macro = phi_s.shape[0]
    for tidx_frac in [0.25, 0.5, 0.75, 1.0]:
        ti = min(int(tidx_frac * (n_t - 1)), n_t - 1)
        dphi_all = phi_s[:, ti] - phi_e[:, ti]
        print(f"  t={time[ti]:8.0f}s  dphi: min={dphi_all.min():.8f}  max={dphi_all.max():.8f}  "
              f"spread={dphi_all.max()-dphi_all.min():.4e}  mean={dphi_all.mean():.8f}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
