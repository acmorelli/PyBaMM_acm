"""
OCP − U_eq_p0 vs stoichiometry for all half-cell C-rates.
Auto-discovers all half_cell_omega_*_{charge,discharge}.pkl files.
"""

import re
import glob
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
U_eq_p0 = 3.397  # from ccpm_positive_interface.py

# ── Auto-discover C-rate pkl files ────────────────────────────────────
# Expected naming: half_cell_omega_{RATE}_{charge|discharge}.pkl
# Examples: half_cell_omega_C2_discharge.pkl, half_cell_omega_2C_charge.pkl
_pattern = re.compile(
    r"half_cell_omega_([A-Za-z0-9]+)_(charge|discharge)\.pkl$"
)
_discovered = set()
for path in glob.glob(f"{base}\\half_cell_omega_*.pkl"):
    m = _pattern.search(path.replace("/", "\\"))
    if m:
        _discovered.add(m.group(1).lower())  # lowercase to dedup c5/C5

def _rate_to_numeric(tag):
    """Convert rate tag to numeric C-rate value.
    '2c' → 2.0 (2C),  'c2' → 0.5 (C/2),  'c10' → 0.1 (C/10)"""
    t = tag.lower()
    # Pattern: number followed by c → NC rate (multiplier)
    m = re.match(r"^(\d+(?:p\d+)?)c$", t)
    if m:
        return float(m.group(1).replace("p", "."))
    # Pattern: c followed by number → C/N rate (divisor)
    m = re.match(r"^c(\d+(?:p\d+)?)$", t)
    if m:
        return 1.0 / float(m.group(1).replace("p", "."))
    return 1e6  # unknown → sort last

def _rate_to_label(tag):
    """Convert rate tag to display label: 'c2'→'C/2', '2c'→'2C', 'c10'→'C/10'."""
    t = tag.lower()
    # Pattern: number followed by c → NC format (e.g. 2c → 2C)
    m = re.match(r"^(\d+(?:p\d+)?)c$", t)
    if m:
        num_str = m.group(1).replace("p", ".")
        return f"{num_str}C"
    # Pattern: c followed by number → C/N format (e.g. c2 → C/2)
    m = re.match(r"^c(\d+(?:p\d+)?)$", t)
    if m:
        num_str = m.group(1).replace("p", ".")
        num = float(num_str)
        return f"C/{int(num)}" if num == int(num) else f"C/{num_str}"
    return tag  # fallback

# Sort by numeric C-rate (ascending = slowest first)
rates = sorted(_discovered, key=_rate_to_numeric)
rate_labels = {r: _rate_to_label(r) for r in rates}
rate_divisor = {r: _rate_to_numeric(r) for r in rates}

print(f"Discovered C-rates: {[rate_labels[r] for r in rates]}")

# ── Load all pickles (skip missing) ───────────────────────────────────
data = {}
found_rates = []
for rate in rates:
    for direction in ["charge", "discharge"]:
        key = f"{rate}_{direction}"
        pkl = f"{base}\\half_cell_omega_{rate}_{direction}.pkl"
        try:
            print(f"Loading {key} …", flush=True)
            sim = pybamm.load(pkl)
            sol = sim.solution
            theta = arr(sol, "X-averaged positive CCPM stoichiometry")
            ocp = arr(sol, "X-averaged positive electrode open-circuit potential [V]")
            phi_s = arr(sol, "X-averaged positive electrode potential [V]")
            data[key] = {
                "theta": theta,
                "ocp_shift": U_eq_p0 - ocp,
                "phi_s_shift": U_eq_p0 - phi_s,
            }
            print(f"  θ=[{theta.min():.4f}, {theta.max():.4f}]  "
                  f"U_eq_p0−OCP=[{(U_eq_p0-ocp).min():.4f}, {(U_eq_p0-ocp).max():.4f}] V")
            if rate not in found_rates:
                found_rates.append(rate)
        except FileNotFoundError:
            print(f"  ⚠ file not found, skipping {key}")

# Keep only rates that have at least one direction loaded
rates = found_rates
if not rates:
    raise SystemExit("No data files found — nothing to plot.")

# ── Colours per rate: red = slowest (smallest numeric), blue = fastest ──
sorted_rates = sorted(rates, key=lambda r: rate_divisor[r])  # ascending: slowest first
n = len(sorted_rates)
if n == 1:
    cmap_vals = [0.0]
else:
    cmap_vals = [i / (n - 1) for i in range(n)]
from matplotlib.colors import LinearSegmentedColormap
_rb_cmap = LinearSegmentedColormap.from_list(
    "red_purple_blue", ["tab:red", "mediumorchid", "tab:blue"]
)
colors = {r: _rb_cmap(v) for r, v in zip(sorted_rates, cmap_vals)}  # slowest(0)→red, fastest(1)→blue

# ── Plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10), tight_layout=True)

# --- Panel 1: full range ---
for rate in rates:
    c = colors[rate]
    lbl = rate_labels[rate]
    if f"{rate}_charge" in data:
        d_chg = data[f"{rate}_charge"]
        axes[0, 0].plot(d_chg["theta"], d_chg["ocp_shift"],
                        linewidth=1.5, color=c, linestyle="--", label=f"{lbl} charge")
    if f"{rate}_discharge" in data:
        d_dis = data[f"{rate}_discharge"]
        axes[0, 0].plot(d_dis["theta"], d_dis["ocp_shift"],
                        linewidth=1.5, color=c, linestyle="-", label=f"{lbl} discharge")

axes[0, 0].set_xlabel("Electrode SOC (θ)")
axes[0, 0].set_ylabel("U_eq_p0 − OCP [V]")
axes[0, 0].set_title("U_eq_p0 − OCP vs θ — all C-rates")
axes[0, 0].axhline(0, color="k", linewidth=0.5, linestyle="--")
axes[0, 0].set_ylim(-0.04, 0.04)
axes[0, 0].legend(fontsize=8, ncol=2)
axes[0, 0].grid(True, alpha=0.3)

# --- Panel 2: discharge only, hook region ---
y_max_p2 = -np.inf
for rate in rates:
    if f"{rate}_discharge" not in data:
        continue
    c = colors[rate]
    lbl = rate_labels[rate]
    d = data[f"{rate}_discharge"]
    mask = (d["theta"] >= 0.10) & (d["theta"] <= 0.35)
    axes[0, 1].plot(d["theta"][mask], d["ocp_shift"][mask],
                    linewidth=1.5, color=c, linestyle="-", label=f"{lbl} discharge")
    if mask.any():
        y_max_p2 = max(y_max_p2, d["ocp_shift"][mask].max())

axes[0, 1].set_xlabel("Electrode SOC (θ)")
axes[0, 1].set_ylabel("U_eq_p0 − OCP [V]")
axes[0, 1].set_title("Discharge only: hook region (θ ≈ 0.10–0.35)")
axes[0, 1].axhline(0, color="k", linewidth=0.5, linestyle="--")
if y_max_p2 > -np.inf:
    axes[0, 1].set_ylim(0.0025, y_max_p2 * 1.05)
axes[0, 1].legend(fontsize=7, ncol=1)
axes[0, 1].grid(True, alpha=0.3)

# --- Panel 3: charge only, shoulder region (θ 0.5–1.0) ---
y_min_p3 = np.inf
for rate in rates:
    if f"{rate}_charge" not in data:
        continue
    c = colors[rate]
    lbl = rate_labels[rate]
    d = data[f"{rate}_charge"]
    mask = (d["theta"] >= 0.50) & (d["theta"] <= 1.0)
    axes[0, 2].plot(d["theta"][mask], d["ocp_shift"][mask],
                    linewidth=1.5, color=c, linestyle="--", label=f"{lbl} charge")
    if mask.any():
        y_min_p3 = min(y_min_p3, d["ocp_shift"][mask].min())

axes[0, 2].set_xlabel("Electrode SOC (θ)")
axes[0, 2].set_ylabel("U_eq_p0 − OCP [V]")
axes[0, 2].set_title("Charge only: shoulder region (θ ≈ 0.50–1.0)")
axes[0, 2].axhline(0, color="k", linewidth=0.5, linestyle="--")
if y_min_p3 < np.inf:
    axes[0, 2].set_ylim(y_min_p3 * 1.05, -0.0050)
axes[0, 2].legend(fontsize=7, ncol=1)
axes[0, 2].grid(True, alpha=0.3)

# --- Panel 4 (row 2, col 0): U_eq_p0 − φ_s, full range ---
for rate in rates:
    c = colors[rate]
    lbl = rate_labels[rate]
    if f"{rate}_charge" in data:
        d_chg = data[f"{rate}_charge"]
        axes[1, 0].plot(d_chg["theta"], d_chg["phi_s_shift"],
                        linewidth=1.5, color=c, linestyle="--", label=f"{lbl} charge")
    if f"{rate}_discharge" in data:
        d_dis = data[f"{rate}_discharge"]
        axes[1, 0].plot(d_dis["theta"], d_dis["phi_s_shift"],
                        linewidth=1.5, color=c, linestyle="-", label=f"{lbl} discharge")

axes[1, 0].set_xlabel("Electrode SOC (θ)")
axes[1, 0].set_ylabel("U_eq_p0 − φ_s [V]")
axes[1, 0].set_title("U_eq_p0 − φ_s vs θ — all C-rates")
axes[1, 0].axhline(0, color="k", linewidth=0.5, linestyle="--")
axes[1, 0].set_ylim(-0.04, 0.04)
axes[1, 0].legend(fontsize=8, ncol=2)
axes[1, 0].grid(True, alpha=0.3)

# --- Panel 5 (row 2, col 1): U_eq_p0 − φ_s, discharge hook ---
y_max_p5 = -np.inf
for rate in rates:
    if f"{rate}_discharge" not in data:
        continue
    c = colors[rate]
    lbl = rate_labels[rate]
    d = data[f"{rate}_discharge"]
    mask = (d["theta"] >= 0.10) & (d["theta"] <= 0.35)
    axes[1, 1].plot(d["theta"][mask], d["phi_s_shift"][mask],
                    linewidth=1.5, color=c, linestyle="-", label=f"{lbl} discharge")
    if mask.any():
        y_max_p5 = max(y_max_p5, d["phi_s_shift"][mask].max())

axes[1, 1].set_xlabel("Electrode SOC (θ)")
axes[1, 1].set_ylabel("U_eq_p0 − φ_s [V]")
axes[1, 1].set_title("U_eq_p0 − φ_s discharge: hook (θ ≈ 0.10–0.35)")
axes[1, 1].axhline(0, color="k", linewidth=0.5, linestyle="--")
if y_max_p5 > -np.inf:
    axes[1, 1].set_ylim(0.0025, y_max_p5 * 1.05)
axes[1, 1].legend(fontsize=7, ncol=1)
axes[1, 1].grid(True, alpha=0.3)

# --- Panel 6 (row 2, col 2): U_eq_p0 − φ_s, charge shoulder ---
y_min_p6 = np.inf
for rate in rates:
    if f"{rate}_charge" not in data:
        continue
    c = colors[rate]
    lbl = rate_labels[rate]
    d = data[f"{rate}_charge"]
    mask = (d["theta"] >= 0.50) & (d["theta"] <= 1.0)
    axes[1, 2].plot(d["theta"][mask], d["phi_s_shift"][mask],
                    linewidth=1.5, color=c, linestyle="--", label=f"{lbl} charge")
    if mask.any():
        y_min_p6 = min(y_min_p6, d["phi_s_shift"][mask].min())

axes[1, 2].set_xlabel("Electrode SOC (θ)")
axes[1, 2].set_ylabel("U_eq_p0 − φ_s [V]")
axes[1, 2].set_title("U_eq_p0 − φ_s charge: shoulder (θ ≈ 0.50–1.0)")
axes[1, 2].axhline(0, color="k", linewidth=0.5, linestyle="--")
if y_min_p6 < np.inf:
    axes[1, 2].set_ylim(y_min_p6 * 1.05, -0.0050)
axes[1, 2].legend(fontsize=7, ncol=1)
axes[1, 2].grid(True, alpha=0.3)

plt.savefig("half_cell_fig7_omega_all_rates.png", dpi=150)
plt.close()
print("\nPlot saved to half_cell_fig7_omega_all_rates.png")
