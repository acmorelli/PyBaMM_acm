"""
Terminal voltage vs stoichiometry for all half-cell C-rates:
C/2, C/5, C/10, C/30 (charge + discharge).
"""

import numpy as np
import pybamm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    """Needed for pickle unpickling — must match the original simulation."""
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3


def arr(sol, name):
    return np.asarray(sol[name].entries)


base = r"C:\Users\dottavianomo\programming\ccpm_snap_mesh_Ra_smooth"

rates = ["C2", "C5", "C10", "C30"]
rate_labels = {"C2": "C/2", "C5": "C/5", "C10": "C/10", "C30": "C/30"}

# ── Load all pickles ───────────────────────────────────────────────────
data = {}
for rate in rates:
    for direction in ["charge", "discharge"]:
        key = f"{rate}_{direction}"
        pkl = f"{base}\\half_cell_{rate}_{direction}.pkl"
        print(f"Loading {key} …", flush=True)
        sim = pybamm.load(pkl)
        sol = sim.solution
        theta = arr(sol, "X-averaged positive CCPM stoichiometry")
        voltage = sol["Terminal voltage [V]"].entries
        data[key] = {
            "theta": theta,
            "voltage": voltage,
        }
        print(f"  θ=[{theta.min():.4f}, {theta.max():.4f}]  "
              f"V=[{voltage.min():.4f}, {voltage.max():.4f}] V")

# ── Colours per rate ───────────────────────────────────────────────────
colors = {"C2": "tab:red", "C5": "tab:orange", "C10": "tab:green", "C30": "tab:blue"}

# ── Plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

# --- Panel 1: full range ---
for rate in rates:
    c = colors[rate]
    lbl = rate_labels[rate]
    d_chg = data[f"{rate}_charge"]
    d_dis = data[f"{rate}_discharge"]
    axes[0].plot(d_chg["theta"], d_chg["voltage"],
                 linewidth=1.5, color=c, linestyle="--", label=f"{lbl} charge")
    axes[0].plot(d_dis["theta"], d_dis["voltage"],
                 linewidth=1.5, color=c, linestyle="-", label=f"{lbl} discharge")

axes[0].set_xlabel("Electrode SOC (θ)")
axes[0].set_ylabel("Terminal voltage [V]")
axes[0].set_title("Terminal voltage vs θ — all C-rates")
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(True, alpha=0.3)

# --- Panel 2: discharge only, hook region ---
for rate in rates:
    c = colors[rate]
    lbl = rate_labels[rate]
    d = data[f"{rate}_discharge"]
    mask = (d["theta"] >= 0.10) & (d["theta"] <= 0.35)
    axes[1].plot(d["theta"][mask], d["voltage"][mask],
                 linewidth=1.5, color=c, linestyle="-", label=f"{lbl} discharge")

axes[1].set_xlabel("Electrode SOC (θ)")
axes[1].set_ylabel("Terminal voltage [V]")
axes[1].set_title("Discharge only: hook region (θ ≈ 0.10–0.35)")
axes[1].legend(fontsize=7, ncol=1)
axes[1].grid(True, alpha=0.3)

# --- Panel 3: charge only, shoulder region (θ 0.5–1.0) ---
for rate in rates:
    c = colors[rate]
    lbl = rate_labels[rate]
    d = data[f"{rate}_charge"]
    mask = (d["theta"] >= 0.50) & (d["theta"] <= 1.0)
    axes[2].plot(d["theta"][mask], d["voltage"][mask],
                 linewidth=1.5, color=c, linestyle="--", label=f"{lbl} charge")

axes[2].set_xlabel("Electrode SOC (θ)")
axes[2].set_ylabel("Terminal voltage [V]")
axes[2].set_title("Charge only: shoulder region (θ ≈ 0.50–1.0)")
axes[2].legend(fontsize=7, ncol=1)
axes[2].grid(True, alpha=0.3)

plt.savefig("half_cell_voltage_vs_theta_all_rates.png", dpi=150)
plt.close()
print("\nPlot saved to half_cell_voltage_vs_theta_all_rates.png")
