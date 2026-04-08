"""
Explore which potential variable best replicates Clarke's Figure 7.

Clarke Fig. 7 shows  U_eq_p0 − V  vs  θ  where V is the *electrode*
potential measured in a half-cell at C/30.  The hooks (A↔B transition)
and shoulders (B↔C transition) are thermodynamic/kinetic features of
the phase-change population balance.

In our CCPM half-cell model the terminal voltage reads:

    V_cell = φ_s(pos) − φ_s(neg=Li)
           = [φ_e + Δφ] − 0                (Li ref ≈ 0 V)

where  Δφ = φ_s − φ_e  is the *surface potential difference*.

Liquid-phase ohmic and concentration drops (φ_e variations across the
separator / electrode) are lumped into V_cell but NOT into Δφ.

At finite rate the hierarchy is:
    OCP(θ)  ←  pure thermodynamics (no current)
    Δφ      =  OCP + η_r            (adds reaction overpotential)
    V_cell  =  Δφ + electrolyte IR   (adds liquid-phase losses)

Clarke measured the *cell voltage* in a half-cell (thin electrode,
dilute electrolyte), so electrolyte losses are small but non-zero.
At C/30 they should be negligible, so OCP, Δφ, and V_cell should
nearly coincide.  If the hooks/shoulders are washed out in V_cell
but visible in Δφ or OCP, the electrolyte model is the culprit.

This script plots all three (plus the reaction overpotential) to
identify which captures the features best.
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

print("Loading charge pickle …", flush=True)
sim_chg = pybamm.load(f"{base}\\half_cell_C30_charge.pkl")
sol_chg = sim_chg.solution
print("  done.", flush=True)

print("Loading discharge pickle …", flush=True)
sim_dis = pybamm.load(f"{base}\\half_cell_C30_discharge.pkl")
sol_dis = sim_dis.solution
print("  done.", flush=True)

U_eq_p0 = 3.397  # LFP reference potential [V]

# ── Variable names to try ──────────────────────────────────────────────
var_labels = {
    "OCP": "X-averaged positive electrode open-circuit potential [V]",
    "Δφ = φ_s − φ_e": "X-averaged positive electrode surface potential difference [V]",
    "V_terminal": "Terminal voltage [V]",
    "η_r (reaction)": "X-averaged positive electrode reaction overpotential [V]",
}

theta_var = "X-averaged positive CCPM stoichiometry"

# ── Probe which variables exist in the solution ────────────────────────
print("\n── Checking variable availability ─────────────────")
available = {}
for label, vname in var_labels.items():
    try:
        _ = arr(sol_chg, vname)
        available[label] = vname
        print(f"  ✓  {label:25s}  →  {vname}")
    except KeyError:
        print(f"  ✗  {label:25s}  →  {vname}  [NOT FOUND]")

# Also try alternate names
alt_names = [
    "Positive electrode surface potential difference [V]",
    "X-averaged positive electrode surface potential difference [V]",
    "Voltage [V]",
    "Terminal voltage [V]",
    "Battery voltage [V]",
    "Positive electrode potential [V]",
    "X-averaged positive electrode potential [V]",
    "Positive electrode reaction overpotential [V]",
    "X-averaged positive electrode reaction overpotential [V]",
]
print("\n── Trying alternate variable names ────────────────")
for vname in alt_names:
    try:
        v = arr(sol_chg, vname)
        print(f"  ✓  {vname:60s}  shape={v.shape}")
    except KeyError:
        print(f"  ✗  {vname}")

# ── Extract stoichiometry ──────────────────────────────────────────────
theta_chg = arr(sol_chg, theta_var)
theta_dis = arr(sol_dis, theta_var)

# ── Plot: one row per variable ─────────────────────────────────────────
n_vars = len(available)
fig, axes = plt.subplots(n_vars, 3, figsize=(18, 5 * n_vars),
                         tight_layout=True, squeeze=False)

for row, (label, vname) in enumerate(available.items()):
    v_chg = arr(sol_chg, vname)
    v_dis = arr(sol_dis, vname)

    # For η_r the sign convention is: η_r > 0 during discharge
    # For potentials: we plot U_eq_p0 − V  (to match Clarke)
    if "overpotential" in label.lower() or "η" in label:
        # Overpotential is already a small quantity; plot directly
        y_chg, y_dis = v_chg, v_dis
        ylabel = f"{label} [V]"
    else:
        y_chg = U_eq_p0 - v_chg
        y_dis = U_eq_p0 - v_dis
        ylabel = f"U_eq_p0 − {label} [V]"

    # Full range
    axes[row, 0].plot(theta_chg, y_chg, lw=1.5, label="charge")
    axes[row, 0].plot(theta_dis, y_dis, lw=1.5, label="discharge")
    axes[row, 0].set_xlabel("θ")
    axes[row, 0].set_ylabel(ylabel)
    axes[row, 0].set_title(f"{label} — full range")
    axes[row, 0].axhline(0, color="k", lw=0.5, ls="--")
    axes[row, 0].legend()
    axes[row, 0].grid(True, alpha=0.3)

    # Hook zoom  (θ ≈ 0.05–0.35)
    for th, y, lbl in [(theta_chg, y_chg, "charge"),
                        (theta_dis, y_dis, "discharge")]:
        m = (th >= 0.05) & (th <= 0.35)
        axes[row, 1].plot(th[m], y[m], lw=1.5, label=lbl)
    axes[row, 1].set_xlabel("θ")
    axes[row, 1].set_ylabel(ylabel)
    axes[row, 1].set_title(f"{label} — hook region")
    axes[row, 1].axhline(0, color="k", lw=0.5, ls="--")
    axes[row, 1].legend()
    axes[row, 1].grid(True, alpha=0.3)

    # Shoulder zoom  (θ ≈ 0.65–0.95)
    for th, y, lbl in [(theta_chg, y_chg, "charge"),
                        (theta_dis, y_dis, "discharge")]:
        m = (th >= 0.65) & (th <= 0.95)
        axes[row, 2].plot(th[m], y[m], lw=1.5, label=lbl)
    axes[row, 2].set_xlabel("θ")
    axes[row, 2].set_ylabel(ylabel)
    axes[row, 2].set_title(f"{label} — shoulder region")
    axes[row, 2].axhline(0, color="k", lw=0.5, ls="--")
    axes[row, 2].legend()
    axes[row, 2].grid(True, alpha=0.3)

    # Print diagnostics
    print(f"\n=== {label} ===")
    plat_c = (theta_chg >= 0.3) & (theta_chg <= 0.7)
    plat_d = (theta_dis >= 0.3) & (theta_dis <= 0.7)
    if plat_c.any() and plat_d.any():
        hyst = np.mean(y_dis[plat_d]) - np.mean(y_chg[plat_c])
        print(f"  Plateau hysteresis ΔV = {hyst*1e3:.2f} mV")
    print(f"  Charge  y-range: [{y_chg.min():.6f}, {y_chg.max():.6f}] V")
    print(f"  Discharge y-range: [{y_dis.min():.6f}, {y_dis.max():.6f}] V")

out = "half_cell_fig7_potential_comparison.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nPlot saved to {out}")
