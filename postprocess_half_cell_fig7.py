"""
Replicate Clarke's Figure 7: solid potential φ_s − U_eq_p0 vs stoichiometry
for half-cell C/30 charge and discharge.

Can be used standalone (loads pickles) or as a library via plot_fig7().
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


def plot_fig7(sol_chg=None, sol_dis=None, save_path="half_cell_fig7_phi_s_vs_theta.png"):
    """Plot Clarke Fig. 7: U_eq_p0 − φ_s vs θ.

    Parameters
    ----------
    sol_chg : pybamm.Solution or None
        Charge solution. Omitted curves if None.
    sol_dis : pybamm.Solution or None
        Discharge solution. Omitted curves if None.
    save_path : str
        Output file path for the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    U_eq_p0 = 3.397

    datasets = []
    if sol_chg is not None:
        theta_chg = arr(sol_chg, "X-averaged positive CCPM stoichiometry")
        phi_s_chg = arr(sol_chg, "X-averaged positive electrode potential [V]")
        phi_s_shift_chg = U_eq_p0 - phi_s_chg
        datasets.append(("charge", theta_chg, phi_s_shift_chg))
    if sol_dis is not None:
        theta_dis = arr(sol_dis, "X-averaged positive CCPM stoichiometry")
        phi_s_dis = arr(sol_dis, "X-averaged positive electrode potential [V]")
        phi_s_shift_dis = U_eq_p0 - phi_s_dis
        datasets.append(("discharge", theta_dis, phi_s_shift_dis))

    if not datasets:
        raise ValueError("At least one of sol_chg or sol_dis must be provided")

    # Diagnostics
    for label, theta, shift in datasets:
        print(f"=== {label.title()} ===")
        print(f"  θ range:           [{theta.min():.6f}, {theta.max():.6f}]")
        print(f"  U_eq_p0-φ_s range: [{shift.min():.6f}, {shift.max():.6f}] V")
        print()

    # Compute plateau hysteresis ΔV if both available
    if len(datasets) == 2:
        _, th_c, sh_c = datasets[0]
        _, th_d, sh_d = datasets[1]
        plat_mask_c = (th_c >= 0.3) & (th_c <= 0.7)
        plat_mask_d = (th_d >= 0.3) & (th_d <= 0.7)
        if plat_mask_c.any() and plat_mask_d.any():
            plateau_c = np.mean(sh_c[plat_mask_c])
            plateau_d = np.mean(sh_d[plat_mask_d])
            delta_V = plateau_d - plateau_c
            print(f"Plateau hysteresis ΔV = {delta_V*1e3:.1f} mV  "
                  f"(charge={plateau_c*1e3:.1f} mV, discharge={plateau_d*1e3:.1f} mV)")
        else:
            delta_V = None
    else:
        delta_V = None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

    # --- Panel 1: full range ---
    for label, theta, shift in datasets:
        axes[0].plot(theta, shift, linewidth=1.5, label=label)
    axes[0].set_xlabel("Electrode SOC (θ)")
    axes[0].set_ylabel("U_eq_p0 − φ_s [V]")
    axes[0].set_title("U_eq_p0 − φ_s vs θ (Clarke Fig. 7)")
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[0].set_ylim(-0.04, 0.04)
    if delta_V is not None:
        axes[0].text(0.5, 0.95, f"ΔV hysteresis = {delta_V*1e3:.1f} mV",
                     transform=axes[0].transAxes, ha="center", va="top",
                     fontsize=11, bbox=dict(boxstyle="round,pad=0.3",
                                            fc="wheat", alpha=0.8))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: zoom hook region (A→B transition) ---
    for label, theta, shift in datasets:
        mask = (theta >= 0.10) & (theta <= 0.35)
        if mask.any():
            axes[1].plot(theta[mask], shift[mask], linewidth=1.5, label=label)
    axes[1].set_xlabel("Electrode SOC (θ)")
    axes[1].set_ylabel("U_eq_p0 − φ_s [V]")
    axes[1].set_title("Zoom: hook region (θ ≈ 0.10–0.35)")
    axes[1].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: zoom shoulder region (B→C transition) ---
    for label, theta, shift in datasets:
        mask = (theta >= 0.65) & (theta <= 0.93)
        if mask.any():
            axes[2].plot(theta[mask], shift[mask], linewidth=1.5, label=label)
    axes[2].set_xlabel("Electrode SOC (θ)")
    axes[2].set_ylabel("U_eq_p0 − φ_s [V]")
    axes[2].set_title("Zoom: shoulder region (θ ≈ 0.65–0.93)")
    axes[2].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    return fig


# ── Standalone: load pickles and plot ──────────────────────────────────
if __name__ == "__main__":
    base = r"C:\Users\dottavianomo\programming\ccpm_snap_mesh_Ra_smooth"

    print("Loading charge pickle …", flush=True)
    sim_chg = pybamm.load(f"{base}\\half_cell_C30_charge.pkl")
    print("  done.", flush=True)

    print("Loading discharge pickle …", flush=True)
    sim_dis = pybamm.load(f"{base}\\half_cell_C30_discharge.pkl")
    print("  done.", flush=True)

    fig = plot_fig7(sim_chg.solution, sim_dis.solution)
    plt.close(fig)
