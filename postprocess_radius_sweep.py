"""
Postprocess radius sweep: compare baseline vs core-shell at each R_p.

Auto-discovers sweep_R*nm_*_{bl|cs}.pkl via regex.
"""

import re
import glob
import numpy as np
import pybamm
import matplotlib.pyplot as plt


def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3


def arr(sol, name):
    return np.asarray(sol[name].entries)


# ── Auto-discover sweep pkl files ─────────────────────────────────────
PKL_PATTERN = re.compile(
    r"sweep_R(\d+)nm_(charge|discharge)_(bl|cs)\.pkl$"
)

results = {}
radii_found = set()
modes_found = set()

for path in sorted(glob.glob("sweep_R*nm_*_*.pkl")):
    m = PKL_PATTERN.search(path)
    if not m:
        continue
    R_nm, mode, tag = int(m.group(1)), m.group(2), m.group(3)
    key = (R_nm, mode, tag)
    try:
        sim = pybamm.load(path)
        results[key] = sim.solution
        radii_found.add(R_nm)
        modes_found.add(mode)
        print(f"  Loaded {path}")
    except Exception as e:
        print(f"  FAILED {path}: {e}")

RADII_NM = sorted(radii_found)
MODES = sorted(modes_found)
D_s = 1e-14  # assumed for tau calculation

print(f"\nDiscovered {len(results)} solutions")
print(f"  Radii: {RADII_NM}")
print(f"  Modes: {MODES}")

# ── Figure 1: Voltage vs time, one subplot per radius ─────────────────
for mode in MODES:
    ncols = min(len(RADII_NM), 3)
    nrows = int(np.ceil(len(RADII_NM) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                             tight_layout=True, squeeze=False)
    fig.suptitle(f"Core-shell vs Baseline — 2C {mode}", fontsize=14)

    for ax, R_nm in zip(axes.flat, RADII_NM):
        for tag, ls, color, lbl in [
            ("bl", "--", "C0", "Baseline"),
            ("cs", "-", "C1", "Core-shell"),
        ]:
            key = (R_nm, mode, tag)
            if key not in results:
                continue
            sol = results[key]
            t = sol["Time [s]"].entries
            V = sol["Terminal voltage [V]"].entries
            ax.plot(t, V, ls, color=color, label=lbl)

        tau = (R_nm * 1e-9) ** 2 / D_s
        ax.set_title(f"R = {R_nm} nm  (τ_diff = {tau:.1f} s)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage [V]")
        ax.legend(fontsize=8)

    # Hide unused axes
    for ax in axes.flat[len(RADII_NM):]:
        ax.set_visible(False)

    plt.savefig(f"sweep_voltage_{mode}.png", dpi=150)
    print(f"Saved sweep_voltage_{mode}.png")

# ── Figure 2: Branch masses, one subplot per radius ───────────────────
for mode in MODES:
    ncols = min(len(RADII_NM), 3)
    nrows = int(np.ceil(len(RADII_NM) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                             tight_layout=True, squeeze=False)
    fig.suptitle(f"Branch masses — 2C {mode}", fontsize=14)

    for ax, R_nm in zip(axes.flat, RADII_NM):
        for tag, ls in [("bl", "--"), ("cs", "-")]:
            key = (R_nm, mode, tag)
            if key not in results:
                continue
            sol = results[key]
            t = sol["Time [s]"].entries
            ma = arr(sol, "X-averaged CCPM Branch A mass")
            mb = arr(sol, "X-averaged CCPM Branch B mass")
            mc = arr(sol, "X-averaged CCPM Branch C mass")
            pfx = "BL" if tag == "bl" else "CS"
            ax.plot(t, ma, ls, color="C0", label=f"{pfx} A")
            ax.plot(t, mb, ls, color="C1", label=f"{pfx} B")
            ax.plot(t, mc, ls, color="C2", label=f"{pfx} C")

        ax.set_title(f"R = {R_nm} nm")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Branch mass")
        ax.legend(fontsize=7, ncol=2)

    for ax in axes.flat[len(RADII_NM):]:
        ax.set_visible(False)

    plt.savefig(f"sweep_branches_{mode}.png", dpi=150)
    print(f"Saved sweep_branches_{mode}.png")

# ── Figure 3: Voltage difference (core-shell minus baseline) ──────────
fig, axes = plt.subplots(1, len(MODES), figsize=(7 * len(MODES), 5), tight_layout=True)
if len(MODES) == 1:
    axes = [axes]
fig.suptitle("ΔV = V_core-shell − V_baseline at 2C", fontsize=14)

colors = plt.cm.viridis(np.linspace(0, 0.9, len(RADII_NM)))
for ax, mode in zip(axes, MODES):
    for R_nm, color in zip(RADII_NM, colors):
        bl_key = (R_nm, mode, "bl")
        cs_key = (R_nm, mode, "cs")
        if bl_key not in results or cs_key not in results:
            continue
        sol_bl = results[bl_key]
        sol_cs = results[cs_key]

        t_bl = sol_bl["Time [s]"].entries
        V_bl = sol_bl["Terminal voltage [V]"].entries
        t_cs = sol_cs["Time [s]"].entries
        V_cs = sol_cs["Terminal voltage [V]"].entries

        # Interpolate to common time grid
        t_common = np.linspace(
            max(t_bl[0], t_cs[0]), min(t_bl[-1], t_cs[-1]), 200
        )
        V_bl_i = np.interp(t_common, t_bl, V_bl)
        V_cs_i = np.interp(t_common, t_cs, V_cs)
        dV = V_cs_i - V_bl_i

        ax.plot(t_common, dV * 1e3, color=color, label=f"{R_nm} nm")

    ax.set_title(mode.capitalize())
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ΔV [mV]")
    ax.legend()
    ax.axhline(0, color="k", lw=0.5, ls=":")

plt.savefig("sweep_deltaV.png", dpi=150)
print("Saved sweep_deltaV.png")

# ── Summary table ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'R_p':>8} {'mode':>10} {'tag':>4} {'V_min':>8} {'V_max':>8} {'θ_range':>16} {'m_drift':>12}")
print("=" * 80)
for R_nm in RADII_NM:
    for mode in MODES:
        for tag in ["bl", "cs"]:
            key = (R_nm, mode, tag)
            if key not in results:
                continue
            sol = results[key]
            V = sol["Terminal voltage [V]"].entries
            th = arr(sol, "X-averaged positive CCPM stoichiometry")
            mt = arr(sol, "X-averaged CCPM Total mass")
            print(
                f"{R_nm:>6}nm {mode:>10} {tag:>4} "
                f"{V.min():8.4f} {V.max():8.4f} "
                f"[{th.min():.4f}, {th.max():.4f}] "
                f"{mt[-1]-mt[0]:+.3e}"
            )

# ── Figure 4: Core-shell geometry evolution (s/R vs θ for each R_p) ──
# Reconstruct the lever-rule core radius from θ_avg for branch-B particles
# at selected c_avg values across the two-phase region
from scipy.optimize import brentq

def compute_binodals(omega):
    """Return (theta_b1, theta_b2) binodal boundaries for given omega."""
    def g(theta):
        return np.log(theta / (1 - theta)) + omega * (1 - 2 * theta)
    theta_b1 = brentq(lambda t: g(t) - g(0.5), 1e-6, 0.5 - 1e-6)
    theta_b2 = 1.0 - theta_b1
    return theta_b1, theta_b2

fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)
fig.suptitle("Core-shell geometry: core fraction & surface concentration", fontsize=14)

# Sample particles at different positions in branch B
theta_samples = np.linspace(0.05, 0.95, 200)

for R_nm, color in zip(RADII_NM, plt.cm.viridis(np.linspace(0, 0.9, len(RADII_NM)))):
    R_p = R_nm * 1e-9
    omega = 4.5 - (2.5 / 0.18e9) * 3 / R_p
    if omega <= 2:
        continue
    tb1, tb2 = compute_binodals(omega)

    # Core fraction for discharge: frac = (θ_b2 - θ_avg) / (θ_b2 - θ_b1)
    in_B = (theta_samples >= tb1) & (theta_samples <= tb2)
    frac = np.clip((tb2 - theta_samples[in_B]) / (tb2 - tb1), 0.01, 1.0)
    s_over_R = frac ** (1.0 / 3.0)

    axes[0].plot(theta_samples[in_B], s_over_R, color=color,
                 label=f"R={R_nm}nm (Ω={omega:.2f})")

    # c_surf deviation: Δc / c_max = -(j × R²) / (F × D_s × c_max) × (1/s - 1/R)
    # Assume j = 10 A/m² as representative 2C current density
    j_rep = 10.0
    F = 96485.0
    c_max = 22806.0
    geom = 1.0 / (s_over_R * R_p) - 1.0 / R_p
    delta_theta = -j_rep * R_p**2 / (F * D_s * c_max) * geom
    delta_theta_clipped = np.clip(delta_theta, -0.5, 0.5)

    axes[1].plot(theta_samples[in_B], delta_theta_clipped * 100, color=color,
                 label=f"R={R_nm}nm")

axes[0].set_xlabel("θ_avg (branch B)")
axes[0].set_ylabel("s / R  (core fraction¹ᐟ³)")
axes[0].set_title("Core radius vs lithiation")
axes[0].legend(fontsize=7)

axes[1].set_xlabel("θ_avg (branch B)")
axes[1].set_ylabel("Δθ_surf  [% of c_max]")
axes[1].set_title("Surface concentration shift (j = 10 A/m²)")
axes[1].legend(fontsize=7)
axes[1].axhline(0, color="k", lw=0.5, ls=":")

plt.savefig("sweep_core_shell_geometry.png", dpi=150)
print("Saved sweep_core_shell_geometry.png")

# ── Figure 5: Voltage vs θ (all radii overlaid) ──────────────────────
fig, axes = plt.subplots(1, len(MODES), figsize=(7 * len(MODES), 5), tight_layout=True)
if len(MODES) == 1:
    axes = [axes]
fig.suptitle("Voltage vs θ — all radii, 2C", fontsize=14)

for ax, mode in zip(axes, MODES):
    for R_nm, color in zip(RADII_NM, plt.cm.viridis(np.linspace(0, 0.9, len(RADII_NM)))):
        for tag, ls, alpha in [("bl", "--", 0.5), ("cs", "-", 1.0)]:
            key = (R_nm, mode, tag)
            if key not in results:
                continue
            sol = results[key]
            th = arr(sol, "X-averaged positive CCPM stoichiometry")
            V = sol["Terminal voltage [V]"].entries
            lbl = f"R={R_nm}nm {'CS' if tag == 'cs' else 'BL'}"
            ax.plot(th, V, ls, color=color, alpha=alpha, label=lbl)

    ax.set_title(mode.capitalize())
    ax.set_xlabel("θ")
    ax.set_ylabel("Voltage [V]")
    ax.legend(fontsize=6, ncol=2)

plt.savefig("sweep_voltage_vs_theta.png", dpi=150)
print("Saved sweep_voltage_vs_theta.png")

plt.show()
