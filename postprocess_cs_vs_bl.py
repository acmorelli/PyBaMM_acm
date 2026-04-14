"""
Compare Core-Shell vs Sheer (baseline) CCPM across particle radii.

Generates cross-radius comparison figures:
  1. Voltage curves overlaid for all radii (BL dashed, CS solid)
  2. ΔV vs time for each radius on one axis → shows diffusion effect scaling
  3. Max |ΔV| vs radius → quantifies when core-shell matters
  4. Voltage vs θ overlaid for all radii
  5. Branch-B stoichiometry spread (θ_avg - θ_surf) vs time per radius
  6. Summary bar chart: capacity-weighted RMSE(ΔV) vs radius
"""

import re
import glob
import numpy as np
import pybamm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Helper ────────────────────────────────────────────────────────────
def arr(sol, name):
    return np.asarray(sol[name].entries)

def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3

D_s = 1e-14  # m²/s

# ── Load data ─────────────────────────────────────────────────────────
PKL_RE = re.compile(r"sweep_R(\d+)nm_(charge|discharge)_(bl|cs)\.pkl$")

results = {}
radii_set, modes_set = set(), set()

for path in sorted(glob.glob("sweep_R*nm_*_*.pkl")):
    m = PKL_RE.search(path)
    if not m:
        continue
    R_nm, mode, tag = int(m.group(1)), m.group(2), m.group(3)
    try:
        sim = pybamm.load(path)
        results[(R_nm, mode, tag)] = sim.solution
        radii_set.add(R_nm)
        modes_set.add(mode)
        print(f"  Loaded {path}")
    except Exception as e:
        print(f"  SKIP  {path}: {e}")

RADII = sorted(radii_set)
MODES = sorted(modes_set)

# Only keep radii that have BOTH bl and cs for at least one mode
paired_radii = {mode: [R for R in RADII
                       if (R, mode, "bl") in results and (R, mode, "cs") in results]
                for mode in MODES}
all_paired = sorted(set(R for rr in paired_radii.values() for R in rr))

print(f"\nLoaded {len(results)} solutions")
print(f"  Radii with BL+CS pairs: {all_paired}")
print(f"  Modes: {MODES}")

if not all_paired:
    print("No BL+CS pairs found — nothing to compare.")
    raise SystemExit(0)

# ── Color map for radii ──────────────────────────────────────────────
cmap = plt.cm.plasma
norm = plt.Normalize(vmin=min(all_paired), vmax=max(all_paired))
radius_color = {R: cmap(norm(R)) for R in all_paired}

# ═════════════════════════════════════════════════════════════════════
# Figure 1: All voltage curves on one axis (BL dashed, CS solid)
# ═════════════════════════════════════════════════════════════════════
for mode in MODES:
    if not paired_radii[mode]:
        continue
    fig, ax = plt.subplots(figsize=(9, 5))
    for R_nm in paired_radii[mode]:
        c = radius_color[R_nm]
        for tag, ls, alpha in [("bl", "--", 0.55), ("cs", "-", 1.0)]:
            sol = results[(R_nm, mode, tag)]
            t = sol["Time [s]"].entries
            V = sol["Terminal voltage [V]"].entries
            lbl = f"R={R_nm}nm {'CS' if tag == 'cs' else 'BL'}"
            ax.plot(t, V, ls=ls, color=c, alpha=alpha, lw=1.4, label=lbl)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title(f"Core-Shell vs Baseline — 2C {mode}", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    fname = f"compare_voltage_{mode}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Figure 2: ΔV = V_CS − V_BL vs time, all radii on one axis
# ═════════════════════════════════════════════════════════════════════
for mode in MODES:
    if not paired_radii[mode]:
        continue
    fig, ax = plt.subplots(figsize=(9, 5))
    for R_nm in paired_radii[mode]:
        sol_bl = results[(R_nm, mode, "bl")]
        sol_cs = results[(R_nm, mode, "cs")]
        t_bl = sol_bl["Time [s]"].entries
        V_bl = sol_bl["Terminal voltage [V]"].entries
        t_cs = sol_cs["Time [s]"].entries
        V_cs = sol_cs["Terminal voltage [V]"].entries
        t_end = min(t_bl[-1], t_cs[-1])
        t_com = np.linspace(max(t_bl[0], t_cs[0]), t_end, 500)
        dV = np.interp(t_com, t_cs, V_cs) - np.interp(t_com, t_bl, V_bl)
        tau = (R_nm * 1e-9) ** 2 / D_s
        ax.plot(t_com, dV * 1e3, color=radius_color[R_nm], lw=1.6,
                label=f"R={R_nm}nm (τ={tau:.1f}s)")

    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ΔV  [mV]  (CS − BL)")
    ax.set_title(f"Voltage shift from core-shell — 2C {mode}", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fname = f"compare_deltaV_{mode}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Figure 3: Max |ΔV| and RMSE(ΔV) vs particle radius (log scale)
# ═════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
fig.suptitle("Core-shell effect vs particle radius — 2C", fontsize=13)

for mode in MODES:
    if not paired_radii[mode]:
        continue
    r_arr, dv_max, dv_rms = [], [], []
    for R_nm in paired_radii[mode]:
        sol_bl = results[(R_nm, mode, "bl")]
        sol_cs = results[(R_nm, mode, "cs")]
        t_bl = sol_bl["Time [s]"].entries
        V_bl = sol_bl["Terminal voltage [V]"].entries
        t_cs = sol_cs["Time [s]"].entries
        V_cs = sol_cs["Terminal voltage [V]"].entries
        t_end = min(t_bl[-1], t_cs[-1])
        t_com = np.linspace(max(t_bl[0], t_cs[0]), t_end, 500)
        dV = np.interp(t_com, t_cs, V_cs) - np.interp(t_com, t_bl, V_bl)
        r_arr.append(R_nm)
        dv_max.append(np.max(np.abs(dV)) * 1e3)
        dv_rms.append(np.sqrt(np.mean(dV**2)) * 1e3)

    marker = "o" if mode == "charge" else "s"
    axes[0].plot(r_arr, dv_max, f"-{marker}", label=mode.capitalize(), lw=2, ms=8)
    axes[1].plot(r_arr, dv_rms, f"-{marker}", label=mode.capitalize(), lw=2, ms=8)

for ax, ylabel, title in zip(
    axes,
    ["max |ΔV|  [mV]", "RMSE(ΔV)  [mV]"],
    ["Peak voltage deviation", "RMS voltage deviation"],
):
    ax.set_xlabel("Particle radius [nm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    # Add τ_diff on secondary x-axis
    ax2 = ax.secondary_xaxis("top", functions=(
        lambda R: (R * 1e-9)**2 / D_s,
        lambda tau: np.sqrt(tau * D_s) * 1e9,
    ))
    ax2.set_xlabel("τ_diff = R²/D_s  [s]")

fig.savefig("compare_effect_vs_radius.png", dpi=150, bbox_inches="tight")
print("Saved compare_effect_vs_radius.png")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Figure 4: V vs θ overlaid, all radii (BL dashed, CS solid)
# ═════════════════════════════════════════════════════════════════════
for mode in MODES:
    if not paired_radii[mode]:
        continue
    fig, ax = plt.subplots(figsize=(9, 5))
    for R_nm in paired_radii[mode]:
        c = radius_color[R_nm]
        for tag, ls, alpha in [("bl", "--", 0.55), ("cs", "-", 1.0)]:
            sol = results[(R_nm, mode, tag)]
            th = arr(sol, "X-averaged positive CCPM stoichiometry")
            V = sol["Terminal voltage [V]"].entries
            lbl = f"R={R_nm}nm {'CS' if tag == 'cs' else 'BL'}"
            ax.plot(th, V, ls=ls, color=c, alpha=alpha, lw=1.4, label=lbl)

    ax.set_xlabel("θ  (cell-average stoichiometry)")
    ax.set_ylabel("Voltage [V]")
    ax.set_title(f"V vs θ — Core-Shell vs Baseline, 2C {mode}", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    fname = f"compare_V_vs_theta_{mode}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Figure 5: Branch population dynamics — per mode, all radii overlaid
# ═════════════════════════════════════════════════════════════════════
branch_names = {
    "A": "X-averaged CCPM Branch A mass",
    "B": "X-averaged CCPM Branch B mass",
    "C": "X-averaged CCPM Branch C mass",
}

for mode in MODES:
    if not paired_radii[mode]:
        continue
    fig, axes_b = plt.subplots(1, 3, figsize=(16, 4.5), tight_layout=True)
    fig.suptitle(f"Branch populations — 2C {mode}", fontsize=13)

    for ax, (br_label, br_var) in zip(axes_b, branch_names.items()):
        for R_nm in paired_radii[mode]:
            c = radius_color[R_nm]
            for tag, ls, alpha in [("bl", "--", 0.55), ("cs", "-", 1.0)]:
                sol = results[(R_nm, mode, tag)]
                t = sol["Time [s]"].entries
                m = arr(sol, br_var)
                ax.plot(t, m, ls=ls, color=c, alpha=alpha, lw=1.2,
                        label=f"R={R_nm}nm {'CS' if tag=='cs' else 'BL'}")
        ax.set_title(f"Branch {br_label}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mass fraction")
        ax.grid(alpha=0.3)

    # Single legend for rightmost panel
    axes_b[-1].legend(fontsize=6, ncol=2, loc="best")
    fname = f"compare_branches_{mode}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Figure 6: ΔV vs θ (normalised progress) — isolates shape effect
# ═════════════════════════════════════════════════════════════════════
for mode in MODES:
    if not paired_radii[mode]:
        continue
    fig, ax = plt.subplots(figsize=(9, 5))
    for R_nm in paired_radii[mode]:
        sol_bl = results[(R_nm, mode, "bl")]
        sol_cs = results[(R_nm, mode, "cs")]
        th_bl = arr(sol_bl, "X-averaged positive CCPM stoichiometry")
        V_bl = sol_bl["Terminal voltage [V]"].entries
        th_cs = arr(sol_cs, "X-averaged positive CCPM stoichiometry")
        V_cs = sol_cs["Terminal voltage [V]"].entries

        # Interpolate to common θ grid
        th_lo = max(th_bl.min(), th_cs.min())
        th_hi = min(th_bl.max(), th_cs.max())
        th_com = np.linspace(th_lo, th_hi, 500)

        # Sort for interp (charge goes high→low in θ, need monotonic)
        idx_bl = np.argsort(th_bl)
        idx_cs = np.argsort(th_cs)
        V_bl_i = np.interp(th_com, th_bl[idx_bl], V_bl[idx_bl])
        V_cs_i = np.interp(th_com, th_cs[idx_cs], V_cs[idx_cs])
        dV = (V_cs_i - V_bl_i) * 1e3

        tau = (R_nm * 1e-9) ** 2 / D_s
        ax.plot(th_com, dV, color=radius_color[R_nm], lw=1.6,
                label=f"R={R_nm}nm (τ={tau:.1f}s)")

    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("θ  (cell-average stoichiometry)")
    ax.set_ylabel("ΔV  [mV]  (CS − BL)")
    ax.set_title(f"Voltage shift vs state-of-lithiation — 2C {mode}", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fname = f"compare_deltaV_vs_theta_{mode}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Summary table
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print(f"{'R [nm]':>8} {'τ_diff [s]':>10} {'mode':>10} "
      f"{'max|ΔV| [mV]':>14} {'RMSE(ΔV) [mV]':>14} {'V_BL range':>14} {'V_CS range':>14}")
print("=" * 90)

for mode in MODES:
    for R_nm in paired_radii[mode]:
        sol_bl = results[(R_nm, mode, "bl")]
        sol_cs = results[(R_nm, mode, "cs")]
        t_bl = sol_bl["Time [s]"].entries
        V_bl = sol_bl["Terminal voltage [V]"].entries
        t_cs = sol_cs["Time [s]"].entries
        V_cs = sol_cs["Terminal voltage [V]"].entries
        t_end = min(t_bl[-1], t_cs[-1])
        t_com = np.linspace(max(t_bl[0], t_cs[0]), t_end, 500)
        dV = np.interp(t_com, t_cs, V_cs) - np.interp(t_com, t_bl, V_bl)
        tau = (R_nm * 1e-9) ** 2 / D_s
        print(f"{R_nm:>6}nm {tau:>10.1f} {mode:>10} "
              f"{np.max(np.abs(dV))*1e3:>14.3f} {np.sqrt(np.mean(dV**2))*1e3:>14.3f} "
              f"[{V_bl.min():.3f},{V_bl.max():.3f}] "
              f"[{V_cs.min():.3f},{V_cs.max():.3f}]")

print("\nDone.")
