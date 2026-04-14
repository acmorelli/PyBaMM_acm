"""
Debug script: extract arrays from pickle and inspect r_p and related quantities.
"""
import numpy as np
import pybamm


def arr(sol, name):
    return np.asarray(sol[name].entries)


def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3


# ── Load ─────────────────────────────────────────────────────────────
print("Loading core-shell pickle ...")
sim_cs = pybamm.load("test_transient_cs_coreshell.pkl")
sol = sim_cs.solution

# ── Basic shapes ─────────────────────────────────────────────────────
t = arr(sol, "Time [s]")
print(f"\nTime: shape={t.shape}, range=[{t[0]:.2f}, {t[-1]:.2f}] s")

# List all variable names containing 'Shell' or 'r_p' or 'phase'
print("\n=== Variables containing 'shell', 'r_p', 'phase', 'radius' ===")
model = sim_cs.model if hasattr(sim_cs, 'model') else None
var_keys = sorted(model.variables.keys()) if model else sorted(sol.all_models[0].variables.keys()) if hasattr(sol, 'all_models') else []
if not var_keys:
    # Try to guess from known names
    var_keys = [
        "Shell phase front r_p",
        "Shell concentration chi_1", "Shell concentration chi_2",
        "Shell concentration chi_3", "Shell concentration chi_4",
        "CCPM positive particle concentration",
        "X-averaged positive CCPM stoichiometry",
        "X-averaged CCPM Total mass",
        "Time [s]",
    ]
for name in var_keys:
    low = name.lower()
    if any(kw in low for kw in ["shell", "r_p", "phase", "radius", "particle rad"]):
        try:
            data = arr(sol, name)
            print(f"  {name:60s}  shape={str(data.shape):20s}  "
                  f"min={np.min(data):.6e}  max={np.max(data):.6e}")
        except Exception as e:
            print(f"  {name:60s}  ERROR: {e}")

# ── Detailed inspection of r_p ───────────────────────────────────────
print("\n=== Shell phase front r_p ===")
rp = arr(sol, "Shell phase front r_p")
print(f"  shape: {rp.shape}")
print(f"  dtype: {rp.dtype}")
print(f"  min:   {np.min(rp):.6e}")
print(f"  max:   {np.max(rp):.6e}")
print(f"  mean:  {np.mean(rp):.6e}")

# Check R_p from parameters
param = sim_cs.parameter_values
R_p = float(param["Positive particle radius [m]"])
c_max = float(param["Maximum concentration in positive electrode [mol.m-3]"])
D_s = float(param["Positive particle diffusivity [m2.s-1]"])
print(f"\n  R_p (param) = {R_p:.6e} m = {R_p*1e9:.1f} nm")
print(f"  c_max       = {c_max:.0f} mol/m3")
print(f"  D_s         = {D_s:.2e} m2/s")

# ratio r_p / R_p
ratio = rp / R_p
print(f"\n  r_p / R_p:")
print(f"    min:  {np.min(ratio):.6f}")
print(f"    max:  {np.max(ratio):.6f}")
print(f"    mean: {np.mean(ratio):.6f}")

# Check if r_p is uniform along c_p at t=0
if rp.ndim == 3:
    print(f"\n  r_p at t=0, x=0: shape={(rp[:, 0, 0]).shape}")
    rp_t0 = rp[:, 0, 0]
    print(f"    unique values: {len(np.unique(rp_t0))}")
    print(f"    first 10: {rp_t0[:10]}")
    print(f"    last  10: {rp_t0[-10:]}")

    # r_p at last time
    rp_tend = rp[:, 0, -1]
    print(f"\n  r_p at t_end, x=0:")
    print(f"    unique values: {len(np.unique(rp_tend))}")
    print(f"    first 10: {rp_tend[:10]}")
    print(f"    last  10: {rp_tend[-10:]}")

    # r_p at mid time
    t_mid = rp.shape[2] // 2
    rp_tmid = rp[:, 0, t_mid]
    print(f"\n  r_p at t_mid (idx={t_mid}), x=0:")
    print(f"    first 10: {rp_tmid[:10]}")
    print(f"    last  10: {rp_tmid[-10:]}")
elif rp.ndim == 2:
    print(f"\n  r_p at t=0: {rp[:10, 0]}")
    print(f"  r_p at t_end: {rp[:10, -1]}")

# ── c_p grid ─────────────────────────────────────────────────────────
print("\n=== c_p grid ===")
cp_raw = arr(sol, "CCPM positive particle concentration")
print(f"  shape: {cp_raw.shape}")
if cp_raw.ndim > 1:
    cp = np.unique(cp_raw)
else:
    cp = cp_raw
print(f"  n_cp: {len(cp)}")
print(f"  c_p/c_max range: [{cp[0]/c_max:.6f}, {cp[-1]/c_max:.6f}]")
print(f"  c_p first 5: {cp[:5]}")
print(f"  c_p last  5: {cp[-5:]}")

# ── branch constants ─────────────────────────────────────────────────
from pybamm.models.submodels.particle.ccpm_positive_particle import CCPMPositiveParticle
consts = CCPMPositiveParticle.compute_branch_constants(R_p, 300, c_max)
c_sp1 = float(consts["c_sp1"].value)
c2_star = float(consts["c2_star"].value)
c1_star = float(consts["c1_star"].value)
print(f"\n=== Branch constants ===")
print(f"  c_sp1   = {c_sp1:.2f}  (c_sp1/c_max = {c_sp1/c_max:.6f})")
print(f"  c2_star = {c2_star:.2f}  (c2*/c_max  = {c2_star/c_max:.6f})")
print(f"  c1_star = {c1_star:.2f}  (c1*/c_max  = {c1_star/c_max:.6f})")

# ── Check shell concentrations ───────────────────────────────────────
print("\n=== Shell concentrations ===")
N_shell = 4
for j in range(1, N_shell + 1):
    name = f"Shell concentration chi_{j}"
    try:
        c_j = arr(sol, name)
        print(f"  {name}: shape={c_j.shape}, "
              f"min={np.min(c_j):.2f}, max={np.max(c_j):.2f}, "
              f"min/c_max={np.min(c_j)/c_max:.6f}, max/c_max={np.max(c_j)/c_max:.6f}")
    except Exception as e:
        print(f"  {name}: ERROR {e}")

# ── Check the r_p equation source ────────────────────────────────────
print("\n=== Checking how r_p is computed ===")
# Look for the r_p variable in the model
_model = sim_cs.model if hasattr(sim_cs, 'model') else (sol.all_models[0] if hasattr(sol, 'all_models') else None)
if _model:
    for name, var in _model.variables.items():
        if "r_p" in name.lower() and "shell" in name.lower():
            print(f"  Variable: {name}")
            try:
                print(f"    pybamm expression: {var}")
            except:
                pass
else:
    print("  (model not accessible from pickle)")

# ── Stoichiometry and mass ────────────────────────────────────────────
print("\n=== Stoichiometry ===")
theta = arr(sol, "X-averaged positive CCPM stoichiometry")
print(f"  shape: {theta.shape}")
print(f"  range: [{theta[0]:.6f}, {theta[-1]:.6f}]")

print("\n=== Mass ===")
m_tot = arr(sol, "X-averaged CCPM Total mass")
print(f"  shape: {m_tot.shape}")
print(f"  range: [{m_tot[0]:.6e}, {m_tot[-1]:.6e}]")
print(f"  drift: {m_tot[-1]-m_tot[0]:.6e}")

# ── Check r_p dimensions: is it in m or nm or something else? ─────────
print("\n=== r_p dimensional analysis ===")
if rp.ndim == 3:
    # Pick a few bins and examine r_p vs expected physical range
    for cp_idx in [0, rp.shape[0]//4, rp.shape[0]//2, 3*rp.shape[0]//4, rp.shape[0]-1]:
        rp_series = rp[cp_idx, 0, :]
        cp_val = cp[cp_idx] if cp_idx < len(cp) else float('nan')
        print(f"  cp_idx={cp_idx:3d}, c_p/c_max={cp_val/c_max:.4f}: "
              f"r_p range=[{rp_series.min():.6e}, {rp_series.max():.6e}] m, "
              f"r_p/R_p=[{rp_series.min()/R_p:.4f}, {rp_series.max()/R_p:.4f}]")

print("\nDone.")
