"""
Animate radial concentration cross-sections for core-shell particles.

For each discovered particle radius, shows the quasi-steady concentration
profile c(r) inside a branch-B particle at several time snapshots during
discharge (lithiation): α-core at c_α*, β-shell at c_β*, with c_surf
deviating from c_β* due to diffusion limitation.

Produces:
  - Static multi-panel PNG with time snapshots
  - Animated GIF (if imageio available)
"""

import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from scipy.optimize import brentq

# ── Physical constants ────────────────────────────────────────────────
F = 96485.0
c_max = 22806.0
D_s = 1e-14  # intra-phase diffusivity [m²/s]

# ── Ferguson & Bazant Omega ───────────────────────────────────────────
def omega_FB(R_p):
    return 4.5 - (2.5 / 0.18e9) * 3 / R_p

def compute_binodals(omega):
    def g(theta):
        return np.log(theta / (1 - theta)) + omega * (1 - 2 * theta)
    theta_b1 = brentq(lambda t: g(t) - g(0.5), 1e-6, 0.5 - 1e-6)
    return theta_b1, 1.0 - theta_b1


# ── Auto-discover available radii from sweep files ────────────────────
PKL_PATTERN = re.compile(r"sweep_R(\d+)nm_")
radii_found = sorted({
    int(PKL_PATTERN.search(p).group(1))
    for p in glob.glob("sweep_R*nm_*.pkl")
    if PKL_PATTERN.search(p)
})
print(f"Discovered radii: {radii_found} nm")

# ── Select radii where core-shell matters (τ_diff > 1s) ──────────────
RADII_NM = [R for R in radii_found if (R * 1e-9)**2 / D_s > 0.5]
if not RADII_NM:
    RADII_NM = radii_found[-3:]  # fallback: largest 3
print(f"Animating radii: {RADII_NM} nm")

# ── Time snapshots (fraction of lithiation for discharge) ─────────────
theta_avg_snapshots = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85]

# ── Build cross-section data ─────────────────────────────────────────
r_norm = np.linspace(0, 1, 500)  # r/R

fig, axes = plt.subplots(
    len(RADII_NM), len(theta_avg_snapshots),
    figsize=(3 * len(theta_avg_snapshots), 3.5 * len(RADII_NM)),
    tight_layout=True, squeeze=False,
)
fig.suptitle("Radial concentration profile — discharge (β-shell around α-core)", fontsize=12, y=1.01)

for row, R_nm in enumerate(RADII_NM):
    R_p = R_nm * 1e-9
    omega = omega_FB(R_p)
    tb1, tb2 = compute_binodals(omega)
    c_alpha = tb1 * c_max
    c_beta = tb2 * c_max
    tau = R_p**2 / D_s

    for col, theta_avg in enumerate(theta_avg_snapshots):
        ax = axes[row, col]

        # Lever rule: core fraction (discharge: core=α)
        if theta_avg < tb1 or theta_avg > tb2:
            # Outside branch B — single phase
            c_profile = np.full_like(r_norm, theta_avg * c_max)
            ax.plot(r_norm, c_profile / c_max, "C0", lw=2)
            ax.set_facecolor("#f0f0f0")
        else:
            frac_core = np.clip((tb2 - theta_avg) / (tb2 - tb1), 0.01, 1.0)
            s_over_R = frac_core ** (1.0 / 3.0)

            # Quasi-steady shell: c(r) = c_beta + (c_surf - c_beta) * (1/s - 1/r) / (1/s - 1/R)
            # where c_surf = c_beta - j*R²/(F*D_s) * (1/s - 1/R)
            # Use representative j for 2C
            j_rep = 10.0  # A/m²
            geom = 1.0 / (s_over_R * R_p) - 1.0 / R_p
            delta_c = -j_rep * R_p**2 / (F * D_s) * geom
            c_surf = c_beta + delta_c
            c_surf = np.clip(c_surf, 0.01 * c_max, 0.99 * c_max)

            # Build profile
            c_profile = np.full_like(r_norm, np.nan)
            core_mask = r_norm <= s_over_R
            shell_mask = r_norm > s_over_R

            c_profile[core_mask] = c_alpha

            # Shell: c(r) = c_beta + (c_surf - c_beta) * (1/s - 1/r_phys) / (1/s - 1/R)
            r_shell = r_norm[shell_mask]
            if geom > 0:
                inv_s = 1.0 / s_over_R
                c_profile[shell_mask] = c_beta + (c_surf - c_beta) * (
                    (inv_s - 1.0 / r_shell) / (inv_s - 1.0)
                )
            else:
                c_profile[shell_mask] = c_beta

            # Plot
            ax.fill_between(r_norm[core_mask], 0, 1, alpha=0.15, color="C0", label="α-core")
            ax.fill_between(r_norm[shell_mask], 0, 1, alpha=0.1, color="C1", label="β-shell")
            ax.axvline(s_over_R, color="k", ls="--", lw=0.8, alpha=0.5)
            ax.plot(r_norm, c_profile / c_max, "C3", lw=2)
            ax.axhline(tb2, color="C1", ls=":", lw=0.7, alpha=0.6)
            ax.axhline(tb1, color="C0", ls=":", lw=0.7, alpha=0.6)

            # Annotate c_surf
            ax.plot(1.0, c_surf / c_max, "rv", ms=6, clip_on=False)

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 1.05)

        if row == 0:
            ax.set_title(f"θ_avg = {theta_avg:.2f}", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"R={R_nm}nm\n(τ={tau:.1f}s)\nθ = c/c_max", fontsize=8)
        if row == len(RADII_NM) - 1:
            ax.set_xlabel("r / R", fontsize=9)

plt.savefig("sweep_cross_sections.png", dpi=150, bbox_inches="tight")
print("Saved sweep_cross_sections.png")

# ── Particle circle visualization ─────────────────────────────────────
# Show particles as circles with α-core (green) and β-shell (blue)
# at different θ_avg for the largest radius

R_nm_show = RADII_NM[-1]
R_p = R_nm_show * 1e-9
omega = omega_FB(R_p)
tb1, tb2 = compute_binodals(omega)

fig2, axes2 = plt.subplots(1, len(theta_avg_snapshots),
                            figsize=(2.5 * len(theta_avg_snapshots), 3),
                            tight_layout=True)
fig2.suptitle(f"Particle cross-section — R = {R_nm_show} nm, discharge", fontsize=11)

for ax, theta_avg in zip(axes2, theta_avg_snapshots):
    frac = np.clip((tb2 - theta_avg) / (tb2 - tb1), 0.01, 1.0)
    s_over_R = frac ** (1.0 / 3.0)

    # c_surf color intensity
    j_rep = 10.0
    geom = 1.0 / (s_over_R * R_p) - 1.0 / R_p
    delta_c = -j_rep * R_p**2 / (F * D_s) * geom
    c_surf_theta = np.clip(tb2 + delta_c / c_max, 0.01, 0.99)

    # Draw shell (outer circle)
    shell_color = plt.cm.Blues(0.3 + 0.5 * c_surf_theta)
    shell = Circle((0, 0), 1.0, fc=shell_color, ec="k", lw=1.5)
    ax.add_patch(shell)

    # Draw core (inner circle)
    core = Circle((0, 0), s_over_R, fc="#66bb6a", ec="k", lw=1.0)
    ax.add_patch(core)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title(f"θ={theta_avg:.2f}\ns/R={s_over_R:.2f}", fontsize=8)
    ax.axis("off")

plt.savefig("sweep_particle_circles.png", dpi=150, bbox_inches="tight")
print("Saved sweep_particle_circles.png")

# ── GIF animation: particles evolving over θ_avg ─────────────────────
try:
    import imageio.v2 as imageio
    import tempfile
    import os

    theta_anim = np.linspace(tb1 + 0.01, tb2 - 0.01, 60)
    frames = []

    for theta_avg in theta_anim:
        fig_f, ax_f = plt.subplots(1, 1, figsize=(4, 4.5))

        frac = np.clip((tb2 - theta_avg) / (tb2 - tb1), 0.01, 1.0)
        s_over_R = frac ** (1.0 / 3.0)

        j_rep = 10.0
        geom = 1.0 / (s_over_R * R_p) - 1.0 / R_p
        delta_c = -j_rep * R_p**2 / (F * D_s) * geom
        c_surf_theta = np.clip(tb2 + delta_c / c_max, 0.01, 0.99)

        # Shell
        shell_color = plt.cm.Blues(0.3 + 0.5 * c_surf_theta)
        shell = Circle((0, 0), 1.0, fc=shell_color, ec="k", lw=2)
        ax_f.add_patch(shell)

        # Core
        core = Circle((0, 0), s_over_R, fc="#66bb6a", ec="k", lw=1.5)
        ax_f.add_patch(core)

        # Phase boundary annotation
        ax_f.plot([0, s_over_R], [0, 0], "k--", lw=0.8)
        ax_f.text(s_over_R / 2, 0.05, f"s/R={s_over_R:.2f}", ha="center", fontsize=8)

        ax_f.set_xlim(-1.4, 1.4)
        ax_f.set_ylim(-1.4, 1.6)
        ax_f.set_aspect("equal")
        ax_f.set_title(
            f"R = {R_nm_show} nm — discharge\n"
            f"θ_avg = {theta_avg:.3f}  |  Δθ_surf = {delta_c/c_max:.4f}",
            fontsize=10,
        )
        ax_f.axis("off")

        # Render to image
        tmpf = os.path.join(tempfile.gettempdir(), f"frame_{len(frames):03d}.png")
        fig_f.savefig(tmpf, dpi=80, bbox_inches="tight")
        frames.append(imageio.imread(tmpf))
        plt.close(fig_f)
        os.remove(tmpf)

    imageio.mimsave("sweep_core_shell_animation.gif", frames, duration=0.1, loop=0)
    print("Saved sweep_core_shell_animation.gif")
except ImportError:
    print("imageio not available — skipping GIF animation")

print("Done.")
