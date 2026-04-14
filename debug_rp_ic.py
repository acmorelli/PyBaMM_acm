"""Quick check: what does PyBaMM compute for the r_p initial condition?"""
import numpy as np
import pybamm

# Load the simulation to access the model internals
def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3

sim = pybamm.load("test_transient_cs_coreshell.pkl")
sol = sim.solution

# Check the actual state vector at t=0
print("=== State vector at t=0 ===")
y0 = sol.y[:, 0]
print(f"  y0 shape: {y0.shape}")

# Find which indices correspond to r_p
model = sim.model
print(f"\n=== Model state variables ===")
for var_name, var in model.variables.items():
    if hasattr(var, 'y_slices') and 'r_p' in var_name.lower():
        print(f"  {var_name}: y_slices = {var.y_slices}")

# Check all state variables and their slices
print(f"\n=== All state variables ===")
for i, (var, ic) in enumerate(model.initial_conditions.items()):
    print(f"  [{i}] {var.name}")

# Look at the initial conditions dict
print(f"\n=== Initial conditions evaluation ===")
for var, ic_expr in model.initial_conditions.items():
    if 'r_p' in var.name.lower():
        print(f"  Variable: {var.name}")
        print(f"  IC expression: {ic_expr}")
        # Try to find where this maps in y0
        if hasattr(var, 'y_slices'):
            for sl in var.y_slices:
                vals = y0[sl]
                print(f"    y0[{sl}]: min={vals.min():.6e}, max={vals.max():.6e}, "
                      f"shape={vals.shape}")
                print(f"    first 5: {vals[:5]}")
        break

# Also check the concatenated initial conditions
print(f"\n=== Concatenated IC ===")
try:
    ic_concat = model.concatenated_initial_conditions
    print(f"  shape: {ic_concat.shape}")
    print(f"  min: {ic_concat.min():.6e}")
    print(f"  max: {ic_concat.max():.6e}")
except Exception as e:
    print(f"  Error: {e}")

# Check nondimensionalization
print(f"\n=== Checking parameter values ===")
pv = sim.parameter_values
R_p = float(pv["Positive particle radius [m]"])
c_max = float(pv["Maximum concentration in positive electrode [mol.m-3]"])
print(f"  R_p = {R_p:.6e} m")
print(f"  c_max = {c_max:.0f} mol/m3")

# Check mesh
print(f"\n=== Mesh ===")
mesh = sim._mesh if hasattr(sim, '_mesh') else None
if mesh is None and hasattr(sol, 'all_meshes'):
    mesh = sol.all_meshes[0]
if mesh:
    for domain_name, submesh in mesh.items():
        dn = str(domain_name)
        if 'ccpm' in dn.lower() or 'c_p' in dn.lower() or 'particle' in dn.lower():
            print(f"  Domain: {domain_name}")
            sms = submesh if isinstance(submesh, list) else [submesh]
            for sm in sms:
                print(f"    nodes: min={sm.nodes.min():.6e}, max={sm.nodes.max():.6e}, "
                      f"n={len(sm.nodes)}")
                print(f"    first 5: {sm.nodes[:5]}")
                print(f"    last 5:  {sm.nodes[-5:]}")

# Also: manually evaluate the IC for r_p at known mesh points
print(f"\n=== Manual IC evaluation ===")
R_p_val = 5e-8  # Positive particle radius [m] 
c_sp1_val = 3724.98
c2_star_val = 22121.82

# Check if c_p mesh is nondimensionalized
# Compute what r_p_init would be for dimensional c_p values
for c_p_test in [38.0, 3725.0, 11000.0, 22121.0, 22768.0]:
    frac_eq = (c2_star_val - c_p_test) / (c2_star_val - c_sp1_val)
    frac_eq = max(frac_eq, 1e-4)
    frac_eq = min(frac_eq, 1.0 - 1e-4)
    r_p_init = R_p_val * frac_eq ** (1.0/3.0)
    r_p_init = max(r_p_init, 0.01 * R_p_val)
    r_p_init = min(r_p_init, R_p_val - 0.01 * R_p_val)
    print(f"  c_p={c_p_test:10.1f} (dim):  frac={frac_eq:.6f}, r_p={r_p_init:.6e} m, r_p/R_p={r_p_init/R_p_val:.6f}")

# And for nondimensionalized c_p (divided by c_max)
print()
for c_p_test_nd in [38.0/c_max, 0.1, 0.5, 0.97, 22768.0/c_max]:
    frac_eq = (c2_star_val - c_p_test_nd) / (c2_star_val - c_sp1_val)
    frac_eq = max(frac_eq, 1e-4)
    frac_eq = min(frac_eq, 1.0 - 1e-4)
    r_p_init = R_p_val * frac_eq ** (1.0/3.0)
    r_p_init = max(r_p_init, 0.01 * R_p_val)
    r_p_init = min(r_p_init, R_p_val - 0.01 * R_p_val)
    print(f"  c_p={c_p_test_nd:10.6f} (nd): frac={frac_eq:.6f}, r_p={r_p_init:.6e} m, r_p/R_p={r_p_init/R_p_val:.6f}")
