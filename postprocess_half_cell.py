"""
Post-process half-cell C/30 discharge: load pickle, plot voltage vs θ.
"""

import numpy as np
import pybamm
import matplotlib.pyplot as plt


def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    """Needed for pickle unpickling — must match the original simulation."""
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3


def arr(sol, name):
    return np.asarray(sol[name].entries)


# ── Load pickle ────────────────────────────────────────────────────────
path = r"C:\Users\dottavianomo\programming\ccpm_snap_mesh_Ra_smooth\half_cell_C30_discharge.pkl"
sim = pybamm.load(path)
solution = sim.solution

# ── Extract arrays ─────────────────────────────────────────────────────
voltage = solution["Terminal voltage [V]"].entries
theta = arr(solution, "X-averaged positive CCPM stoichiometry")
ocp = arr(solution, "X-averaged positive electrode open-circuit potential [V]")

U_eq_p0 = 3.397  # from ccpm_positive_interface.py
ocp_shifted = ocp - U_eq_p0

print(f"Terminal V range:  [{voltage.min():.4f}, {voltage.max():.4f}] V")
print(f"OCP range:         [{ocp.min():.4f}, {ocp.max():.4f}] V")
print(f"U_eq_p0:           {U_eq_p0} V")
print(f"OCP - U_eq_p0 range: [{ocp_shifted.min():.6f}, {ocp_shifted.max():.6f}] V")
print(f"θ range:           [{theta.min():.6f}, {theta.max():.6f}]")

# ── Plot: OCP - U_eq_p0 vs θ (Clarke-style) ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

# Full range
axes[0].plot(theta, ocp_shifted, linewidth=1.5)
axes[0].set_xlabel("Electrode SOC (θ)")
axes[0].set_ylabel("OCP − U_eq_p0 [V]")
axes[0].set_title("OCP − U_eq_p0 vs θ — C/30 discharge")
axes[0].axhline(0, color="k", linewidth=0.5, linestyle="--")
axes[0].grid(True, alpha=0.3)

# Zoom: hook region (A→B transition)
mask_hook = (theta >= 0.10) & (theta <= 0.35)
axes[1].plot(theta[mask_hook], ocp_shifted[mask_hook], linewidth=1.5)
axes[1].set_xlabel("Electrode SOC (θ)")
axes[1].set_ylabel("OCP − U_eq_p0 [V]")
axes[1].set_title("Zoom: hook region (θ ≈ 0.10–0.35)")
axes[1].axhline(0, color="k", linewidth=0.5, linestyle="--")
axes[1].grid(True, alpha=0.3)

# Zoom: shoulder region (B→C transition)
mask_shoulder = (theta >= 0.75) & (theta <= 0.95)
axes[2].plot(theta[mask_shoulder], ocp_shifted[mask_shoulder], linewidth=1.5)
axes[2].set_xlabel("Electrode SOC (θ)")
axes[2].set_ylabel("OCP − U_eq_p0 [V]")
axes[2].set_title("Zoom: shoulder region (θ ≈ 0.75–0.95)")
axes[2].axhline(0, color="k", linewidth=0.5, linestyle="--")
axes[2].grid(True, alpha=0.3)

plt.savefig("half_cell_voltage_vs_theta.png", dpi=150)
plt.show()
print("\nPlot saved to half_cell_voltage_vs_theta.png")
