"""
Explore potential variables for replicating Clarke's Figure 7.

Clarke plots  U_eq_p0 − V_electrode  vs  θ  in a half-cell (Li counter
electrode @ 0 V).  As θ → 1 the OCP → −∞  so  U_eq_p0 − OCP → +∞,
confirming that the plotted quantity is the *positive electrode potential*
(not the full battery terminal voltage, which includes electrolyte losses).

PyBaMM voltage hierarchy (half-cell, Li ref = 0 V):
  - "Voltage [V]" / "Terminal voltage [V]"
        = φ_s(pos tab) − φ_s(neg tab) − contact drop
        ≈ φ_s_pos                              (full cell voltage incl. all losses)
  - "Battery voltage [V]"
        = "Voltage [V]" × num_cells_in_series   (for pack; = Voltage for 1 cell)
  - "X-averaged positive electrode potential [V]"
        = <φ_s>_x                               (solid potential, x-averaged)
  - "X-averaged positive electrode surface potential difference [V]"
        = <φ_s − φ_e>_x                        (driving potential at particle surface)
  - "X-averaged positive electrode reaction overpotential [V]"
        = <Δφ − OCP>_x                          (kinetic departure from equilibrium)

For Clarke's figure the correct variable is  φ_s  (positive electrode
potential), which in a half-cell equals the terminal voltage.  But the
surface potential difference  Δφ = φ_s − φ_e  isolates the electrode
from the electrolyte and may show cleaner hooks/shoulders.

This script plots all three key variables at C/5 and C/30 to compare.
"""

import numpy as np
import pybamm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    """Needed for pickle unpickling."""
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3


def arr(sol, name):
    return np.asarray(sol[name].entries)


# ── Load pickles ───────────────────────────────────────────────────────
base = r"C:\Users\dottavianomo\programming\ccpm_snap_mesh_Ra_smooth"
rates = {
    "C/30": ("half_cell_C30_charge.pkl", "half_cell_C30_discharge.pkl"),
    "C/5":  ("half_cell_C5_charge.pkl",  "half_cell_C5_discharge.pkl"),
}

data = {}
for rate, (chg_pkl, dis_pkl) in rates.items():
    print(f"Loading {rate} charge …", flush=True)
    sim_c = pybamm.load(f"{base}\\{chg_pkl}")
    print(f"Loading {rate} discharge …", flush=True)
    sim_d = pybamm.load(f"{base}\\{dis_pkl}")
    data[rate] = {"charge": sim_c.solution, "discharge": sim_d.solution}

U_eq_p0 = 3.397  # LFP reference potential [V]

theta_var = "X-averaged positive CCPM stoichiometry"

# Variables to plot: (label, PyBaMM name, transform)
#   transform: "U-v" → plot U_eq_p0 − v,  "v-U" → plot v − U_eq_p0,  "raw" → plot v
pot_vars = [
    ("U_eq_p0 − φ_s",   "X-averaged positive electrode potential [V]",               "U-v"),
    ("OCP − U_eq_p0",    "X-averaged positive electrode open-circuit potential [V]",   "v-U"),
    ("U_eq_p0 − OCP",    "X-averaged positive electrode open-circuit potential [V]",   "U-v"),
]

# ── Check availability ─────────────────────────────────────────────────
print("\n── Variable availability ──")
sol0 = data["C/5"]["charge"]
for label, vname, _ in pot_vars:
    try:
        v = arr(sol0, vname)
        print(f"  ✓  {vname}  shape={v.shape}")
    except KeyError:
        print(f"  ✗  {vname}  [NOT FOUND]")

# ── Figure: 3 rows (one per variable) × 3 cols (full, hook, shoulder) ──
# Each panel overlays C/30 and C/5 charge+discharge (4 curves)
n_rows = len(pot_vars)
fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5 * n_rows),
                         tight_layout=True, squeeze=False)

colors = {"C/30": {"charge": "tab:blue", "discharge": "tab:orange"},
          "C/5":  {"charge": "tab:green", "discharge": "tab:red"}}
styles = {"charge": "-", "discharge": "--"}

for row, (label, vname, transform) in enumerate(pot_vars):
    for rate in ["C/30", "C/5"]:
        for mode in ["charge", "discharge"]:
            sol = data[rate][mode]
            theta = arr(sol, theta_var)
            v = arr(sol, vname)

            if transform == "U-v":
                y = U_eq_p0 - v
                ylabel = f"{label} [V]"
            elif transform == "v-U":
                y = v - U_eq_p0
                ylabel = f"{label} [V]"
            else:
                y = v
                ylabel = f"{label} [V]"

            lbl = f"{rate} {mode}"
            c = colors[rate][mode]
            ls = styles[mode]

            # Full range
            axes[row, 0].plot(theta, y, color=c, ls=ls, lw=1.5, label=lbl)

            # Hook zoom (θ ≈ 0.05–0.35)
            m = (theta >= 0.05) & (theta <= 0.35)
            axes[row, 1].plot(theta[m], y[m], color=c, ls=ls, lw=1.5, label=lbl)

            # Shoulder zoom (θ ≈ 0.65–0.95)
            m = (theta >= 0.65) & (theta <= 0.95)
            axes[row, 2].plot(theta[m], y[m], color=c, ls=ls, lw=1.5, label=lbl)

    for col, title_sfx in enumerate(["full range", "hook (θ ≈ 0.05–0.35)",
                                      "shoulder (θ ≈ 0.65–0.95)"]):
        axes[row, col].set_xlabel("Electrode SOC (θ)")
        axes[row, col].set_ylabel(ylabel)
        axes[row, col].set_title(f"{label} — {title_sfx}")
        axes[row, col].axhline(0, color="k", lw=0.5, ls="--")
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(True, alpha=0.3)

    # Print diagnostics
    for rate in ["C/30", "C/5"]:
        sol_c = data[rate]["charge"]
        sol_d = data[rate]["discharge"]
        th_c, th_d = arr(sol_c, theta_var), arr(sol_d, theta_var)
        v_c, v_d = arr(sol_c, vname), arr(sol_d, vname)
        if transform == "U-v":
            y_c, y_d = U_eq_p0 - v_c, U_eq_p0 - v_d
        elif transform == "v-U":
            y_c, y_d = v_c - U_eq_p0, v_d - U_eq_p0
        else:
            y_c, y_d = v_c, v_d

        plat_c = (th_c >= 0.3) & (th_c <= 0.7)
        plat_d = (th_d >= 0.3) & (th_d <= 0.7)
        if plat_c.any() and plat_d.any():
            hyst = np.mean(y_d[plat_d]) - np.mean(y_c[plat_c])
            print(f"  {label} @ {rate}: plateau hysteresis = {hyst*1e3:.2f} mV")

out = "half_cell_fig7_potential_comparison_rates.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nPlot saved to {out}")
