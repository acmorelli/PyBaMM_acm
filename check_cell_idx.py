"""Check if EvaluateAt picks the correct cell index."""
import numpy as np
import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

model = DFN_CCPM(build=True, initial_branch="A", source_method="flux_form")
param = pybamm.ParameterValues("Prada2013")
# Process model and geometry to get mesh
param.process_model(model)
geom = model.default_geometry
param.process_geometry(geom)
mesh = pybamm.Mesh(geom, model.default_submesh_types, model.default_var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# Domain A mesh
nodes_a = mesh["CCPM positive particle branch A"].nodes
edges_a = mesh["CCPM positive particle branch A"].edges
n_a = len(nodes_a)
dc_a = nodes_a[1] - nodes_a[0]

c_max = float(param["Maximum concentration in positive electrode [mol.m-3]"])
eps_c = 1e-8 * c_max
c_sp1 = 0.2113 * c_max

print(f"n_a = {n_a}")
print(f"c_max = {c_max}")
print(f"eps_c = {eps_c}")
print(f"c_sp1 = {c_sp1}")
print(f"dc_a = {dc_a}")
print()

# Our symbolic c_aN
c_aN_symbolic = c_sp1 - dc_a / 2
c_aN1_symbolic = c_sp1 - 3 * dc_a / 2

print(f"Mesh node[-1]  = {nodes_a[-1]:.18f}")
print(f"Our c_aN       = {c_aN_symbolic:.18f}")
print(f"Difference     = {nodes_a[-1] - c_aN_symbolic:.3e}")
print()
print(f"Mesh node[-2]  = {nodes_a[-2]:.18f}")
print(f"Our c_aN1      = {c_aN1_symbolic:.18f}")
print(f"Difference     = {nodes_a[-2] - c_aN1_symbolic:.3e}")
print()

# Check which cell argmin picks
for pos_label, pos_val in [("c_aN (last)", c_aN_symbolic),
                            ("c_aN1 (second-to-last)", c_aN1_symbolic)]:
    idx = np.argmin(np.abs(nodes_a - pos_val))
    print(f"EvaluateAt({pos_label} = {pos_val:.10f}): picks cell {idx} "
          f"(node={nodes_a[idx]:.10f}, expected={'N-1' if 'last' in pos_label else 'N-2'})")

# Now compute dc_a the same way PyBaMM does symbolically
# In the model code: dc_a = (c_sp1 - eps_c) / self.n_pts_a
dc_a_model = (c_sp1 - eps_c) / 200
print(f"\ndc_a from mesh: {dc_a:.18f}")
print(f"dc_a from model: {dc_a_model:.18f}")
print(f"Difference: {dc_a - dc_a_model:.3e}")

# Check left boundary too
c_a0_symbolic = eps_c + dc_a / 2
c_a1_symbolic = eps_c + 3 * dc_a / 2
print(f"\nMesh node[0]   = {nodes_a[0]:.18f}")
print(f"Our c_a0       = {c_a0_symbolic:.18f}")
print(f"Difference     = {nodes_a[0] - c_a0_symbolic:.3e}")

idx0 = np.argmin(np.abs(nodes_a - c_a0_symbolic))
idx1 = np.argmin(np.abs(nodes_a - c_a1_symbolic))
print(f"EvaluateAt(c_a0): picks cell {idx0} (expected 0)")
print(f"EvaluateAt(c_a1): picks cell {idx1} (expected 1)")

# Check domain B and C too
nodes_b = mesh["CCPM positive particle branch B"].nodes
nodes_c = mesh["CCPM positive particle branch C"].nodes
c1_star = 0.0710 * c_max
c2_star = 0.9290 * c_max
c_sp2 = 0.7887 * c_max
dc_b = (c2_star - c1_star) / 258
dc_c = (c_max - eps_c - c_sp2) / 63

print("\n--- Domain B ---")
for label, pos, expected_idx in [("B first", c1_star + dc_b/2, 0),
                                   ("B last", c2_star - dc_b/2, 257)]:
    idx = np.argmin(np.abs(nodes_b - pos))
    print(f"  {label}: pos={pos:.10f}, mesh_node[{expected_idx}]={nodes_b[expected_idx]:.10f}, "
          f"picked={idx}, diff={nodes_b[expected_idx]-pos:.3e}")

print("\n--- Domain C ---")
for label, pos, expected_idx in [("C first", c_sp2 + dc_c/2, 0),
                                   ("C last", (c_max-eps_c) - dc_c/2, 62)]:
    idx = np.argmin(np.abs(nodes_c - pos))
    print(f"  {label}: pos={pos:.10f}, mesh_node[{expected_idx}]={nodes_c[expected_idx]:.10f}, "
          f"picked={idx}, diff={nodes_c[expected_idx]-pos:.3e}")
