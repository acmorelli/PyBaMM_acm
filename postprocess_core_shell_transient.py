"""
Post-process core-shell vs baseline CCPM simulation.

Load two pickles (baseline + core-shell) and call all plot functions.

Usage:
    python postprocess_core_shell_transient.py <bl_pkl> <cs_pkl> [R_nm]

Example:
    python postprocess_core_shell_transient.py sweep_R300nm_discharge_bl.pkl sweep_R300nm_discharge_cs.pkl 300
"""

import sys
import numpy as np
import pybamm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_core_shell_lib import (
    arr, get_cp_grid, find_time_index,
    plot_cavg_consistency,
    plot_particle_cross_section,
    plot_j_tr_b_comparison,
    plot_mass_evolution,
    plot_solid_potential,
    plot_phase_front_evolution,
    plot_csurf_vs_time,
    plot_shell_profile_snapshots,
    plot_theta_surf_vs_avg,
)

# ── Unpickling helper (must match simulation) ────────────────────────
def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3


# ── Configuration ────────────────────────────────────────────────────
# Edit these to change the exemplary snapshot
THETA_TARGET = 0.5   # SOC at which to take snapshots
X_IDX = 0            # electrode position index
N_SHELL = 3          # must match the simulation
C_MAX = 22806.0      # mol/m3

# ── Parse arguments ──────────────────────────────────────────────────
if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

bl_pkl = sys.argv[1]
cs_pkl = sys.argv[2]
R_nm = int(sys.argv[3]) if len(sys.argv) > 3 else 300
R_p = R_nm * 1e-9

print(f"Loading baseline:   {bl_pkl}")
sim_bl = pybamm.load(bl_pkl)
sol_bl = sim_bl.solution

print(f"Loading core-shell: {cs_pkl}")
sim_cs = pybamm.load(cs_pkl)
sol_cs = sim_cs.solution

# ── Extract constants from compute_branch_constants ──────────────────
from pybamm.models.submodels.particle.ccpm_positive_particle import CCPMPositiveParticle
consts = CCPMPositiveParticle.compute_branch_constants(R_p, 300, C_MAX)
c_sp1 = float(consts["c_sp1"].value)
c2_star = float(consts["c2_star"].value)
c1_star = float(consts["c1_star"].value)

print(f"\nR_p = {R_nm} nm,  c_sp1/c_max = {c_sp1/C_MAX:.4f},  "
      f"c2*/c_max = {c2_star/C_MAX:.4f}")

# ── Time indices ─────────────────────────────────────────────────────
t_idx_bl = find_time_index(sol_bl, THETA_TARGET)
t_idx_cs = find_time_index(sol_cs, THETA_TARGET)
t_bl = arr(sol_bl, "Time [s]")[t_idx_bl]
t_cs = arr(sol_cs, "Time [s]")[t_idx_cs]
print(f"Snapshot at theta ~ {THETA_TARGET}:  BL t={t_bl:.1f}s (idx {t_idx_bl}), "
      f"CS t={t_cs:.1f}s (idx {t_idx_cs})")

# Find a branch-B bin near the middle of [c_sp1, c2_star]
c_p = get_cp_grid(sol_cs)
c_mid = 0.5 * (c_sp1 + c2_star)
cp_idx = int(np.argmin(np.abs(c_p - c_mid)))
print(f"Exemplary bin: cp_idx={cp_idx}, c_p/c_max={c_p[cp_idx]/C_MAX:.4f}")

prefix = f"cs_R{R_nm}nm"

# ═════════════════════════════════════════════════════════════════════
# Plot 1: c_avg consistency
# ═════════════════════════════════════════════════════════════════════
print("\nPlot 1: c_avg consistency ...")
fig, _ = plot_cavg_consistency(sol_cs, N_SHELL, t_idx_cs, X_IDX,
                                c_sp1, c2_star, R_p, C_MAX)
fname = f"{prefix}_01_cavg_consistency.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

# ═════════════════════════════════════════════════════════════════════
# Plot 2: Particle cross-section
# ═════════════════════════════════════════════════════════════════════
print("Plot 2: Particle cross-section ...")
fig, _ = plot_particle_cross_section(sol_cs, N_SHELL, t_idx_cs, cp_idx,
                                      c_sp1, c2_star, R_p, C_MAX, X_IDX)
fname = f"{prefix}_02_particle_cross_section.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

# ═════════════════════════════════════════════════════════════════════
# Plot 3: j_tr_b comparison
# ═════════════════════════════════════════════════════════════════════
print("Plot 3: j_tr_b comparison ...")
fig, _ = plot_j_tr_b_comparison(sol_bl, sol_cs, t_idx_bl, t_idx_cs, C_MAX, X_IDX)
fname = f"{prefix}_03_j_tr_b_comparison.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

# ═════════════════════════════════════════════════════════════════════
# Plot 4: Mass evolution
# ═════════════════════════════════════════════════════════════════════
print("Plot 4: Mass evolution ...")
fig, _ = plot_mass_evolution(sol_bl, sol_cs)
fname = f"{prefix}_04_mass_evolution.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

# ═════════════════════════════════════════════════════════════════════
# Plot 5: Solid potential
# ═════════════════════════════════════════════════════════════════════
print("Plot 5: Solid potential ...")
fig, _ = plot_solid_potential(sol_bl, sol_cs)
fname = f"{prefix}_05_solid_potential.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

# ═════════════════════════════════════════════════════════════════════
# Plot 6: Phase front evolution
# ═════════════════════════════════════════════════════════════════════
print("Plot 6: Phase front r_p(c_p) ...")
n_t = len(arr(sol_cs, "Time [s]"))
t_snap = np.linspace(0, n_t - 1, 6, dtype=int)
fig, _ = plot_phase_front_evolution(sol_cs, C_MAX, R_p, c_sp1, c2_star,
                                     t_indices=t_snap, x_idx=X_IDX)
fname = f"{prefix}_06_phase_front.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

# ═════════════════════════════════════════════════════════════════════
# Plot 7: c_surf(t) for exemplary bin
# ═════════════════════════════════════════════════════════════════════
print("Plot 7: c_surf(t) ...")
fig, _ = plot_csurf_vs_time(sol_cs, N_SHELL, cp_idx, c2_star, C_MAX, X_IDX)
fname = f"{prefix}_07_csurf_vs_time.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

# ═════════════════════════════════════════════════════════════════════
# Plot 8: Shell profile snapshots
# ═════════════════════════════════════════════════════════════════════
print("Plot 8: Shell profile c(chi) ...")
fig, _ = plot_shell_profile_snapshots(sol_cs, N_SHELL, cp_idx, c_sp1, c2_star,
                                       C_MAX, R_p, t_indices=t_snap, x_idx=X_IDX)
fname = f"{prefix}_08_shell_profile.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

# ═════════════════════════════════════════════════════════════════════
# Plot 9: theta_surf vs theta_avg scatter
# ═════════════════════════════════════════════════════════════════════
print("Plot 9: theta_surf vs theta_avg ...")
fig, _ = plot_theta_surf_vs_avg(sol_cs, N_SHELL, t_idx_cs, c_sp1, c2_star,
                                 C_MAX, X_IDX)
fname = f"{prefix}_09_theta_surf_vs_avg.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

print(f"\nAll 9 plots saved with prefix '{prefix}_'")
