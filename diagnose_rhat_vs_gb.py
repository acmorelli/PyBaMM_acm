"""
Diagnostic: overlay r_hat and g_b on the c_p grid at several time snapshots.
Checks whether r_hat is physical (0 <= r_hat <= 1) wherever g_b has mass.

Usage:
    python diagnose_rhat_vs_gb.py [cs_pkl]
    Defaults to test_transient_cs_coreshell.pkl
"""
import sys
import numpy as np
import pybamm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3


pkl = sys.argv[1] if len(sys.argv) > 1 else "test_transient_cs_coreshell.pkl"
sim = pybamm.load(pkl)
sol = sim.solution

# ── Grids ────────────────────────────────────────────────────────────
# entries shape: (n_cp=300, n_x=20, n_t)
c_p_3d = np.asarray(sol["CCPM positive particle concentration"].entries)
c_p = c_p_3d[:, 0, 0]  # 1D c_p grid
t = sol["Time [s]"].entries
R_p = float(np.asarray(sol["Positive particle radius [m]"].entries).flat[0])

# Pick ~8 time snapshots spread across the simulation
n_snaps = 8
t_indices = np.linspace(0, len(t) - 1, n_snaps, dtype=int)

# Use x-node = 0 (separator side)
x_idx = 0

fig, axes = plt.subplots(n_snaps, 1, figsize=(10, 3 * n_snaps), sharex=True)

for i, ti in enumerate(t_indices):
    ax = axes[i]

    # Extract at this time index and x-node: shape (n_cp,)
    g_b_all = np.asarray(sol["Branch B PDF CCPM"].entries)  # (n_cp, n_x, n_t)
    r_hat_all = np.asarray(sol["Shell phase front r_hat"].entries)

    g_b_slice = g_b_all[:, x_idx, ti]
    r_hat_slice = r_hat_all[:, x_idx, ti]

    # Plot g_b
    ax_gb = ax
    color_gb = "tab:blue"
    ax_gb.fill_between(c_p, 0, g_b_slice, alpha=0.3, color=color_gb, label="$g_b$")
    ax_gb.plot(c_p, g_b_slice, color=color_gb, linewidth=0.8)
    ax_gb.set_ylabel("$g_b$ [1/(mol/m³)]", color=color_gb)
    ax_gb.tick_params(axis="y", labelcolor=color_gb)

    # Plot r_hat on twin axis
    ax_rh = ax_gb.twinx()
    color_rh = "tab:red"
    ax_rh.plot(c_p, r_hat_slice, color=color_rh, linewidth=1.2, label="$\\hat{r}$")
    ax_rh.axhline(1.0, color="gray", ls="--", lw=0.5, label="$\\hat{r}=1$ (full core)")
    ax_rh.axhline(0.0, color="gray", ls=":", lw=0.5, label="$\\hat{r}=0$ (no core)")
    ax_rh.set_ylabel("$\\hat{r} = r_p/R_p$", color=color_rh)
    ax_rh.tick_params(axis="y", labelcolor=color_rh)

    # Highlight where g_b > threshold AND r_hat is outside [0, 1]
    threshold = 1e-3 * np.max(g_b_slice) if np.max(g_b_slice) > 0 else 1e-10
    bad = (g_b_slice > threshold) & ((r_hat_slice < 0) | (r_hat_slice > 1.0))
    if np.any(bad):
        ax_rh.scatter(c_p[bad], r_hat_slice[bad], c="black", marker="x", s=40,
                      zorder=5, label=f"UNPHYSICAL ({np.sum(bad)} bins)")

    ax_gb.set_title(f"t = {t[ti]:.1f} s  (snap {i+1}/{n_snaps}),  x_idx={x_idx}")

    # Print summary for this snapshot
    has_mass = g_b_slice > threshold
    if np.any(has_mass):
        rh_where_mass = r_hat_slice[has_mass]
        print(f"t={t[ti]:7.1f}s: g_b>0 in {np.sum(has_mass):3d} bins, "
              f"r_hat range [{rh_where_mass.min():.4f}, {rh_where_mass.max():.4f}]"
              f"  {'OK' if rh_where_mass.min() >= -0.01 and rh_where_mass.max() <= 1.01 else 'BAD!'}")
    else:
        print(f"t={t[ti]:7.1f}s: no g_b mass")

axes[-1].set_xlabel("$c_p$ [mol/m³]")
fig.suptitle(f"r_hat vs g_b diagnostic  |  R_p = {R_p*1e9:.0f} nm", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig("diagnose_rhat_vs_gb.png", dpi=150, bbox_inches="tight")
print(f"\nSaved diagnose_rhat_vs_gb.png")
