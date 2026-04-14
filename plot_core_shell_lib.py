"""
Plot library for core-shell CCPM postprocessing.

All functions take solution objects (or extracted arrays) and return fig, ax.

Data layout from PyBaMM: variables on the c_p grid have shape
  (n_cp, n_x, n_t)
where n_cp=300, n_x=20 (electrode nodes), n_t=number of time steps.
Scalars (X-averaged, Time) have shape (n_t,).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


# ── Helpers ───────────────────────────────────────────────────────────
def arr(sol, name):
    return np.asarray(sol[name].entries)


def get_cp_grid(sol):
    """Return the c_p FV cell-centre array (1D, length n_cp)."""
    raw = arr(sol, "CCPM positive particle concentration")
    if raw.ndim > 1:
        return np.unique(raw)
    return raw


def find_time_index(sol, theta_target=0.5):
    """Find the time index closest to X-averaged theta = theta_target."""
    theta = arr(sol, "X-averaged positive CCPM stoichiometry")
    return int(np.argmin(np.abs(theta - theta_target)))


def _extract_cp_field(sol, name, t_idx, x_idx=0):
    """Extract a (n_cp,) vector from a variable at fixed (t, x).

    Data layout: (n_cp, n_x, n_t).
    """
    data = arr(sol, name)
    if data.ndim == 3:
        return data[:, x_idx, t_idx]
    elif data.ndim == 2:
        return data[:, t_idx]
    return data


def _extract_cp_timeseries(sol, name, cp_idx, x_idx=0):
    """Extract a (n_t,) timeseries from a variable at fixed (cp, x).

    Data layout: (n_cp, n_x, n_t).
    """
    data = arr(sol, name)
    if data.ndim == 3:
        return data[cp_idx, x_idx, :]
    elif data.ndim == 2:
        return data[cp_idx, :]
    return data


def _extract_scalar(sol, name, cp_idx, t_idx, x_idx=0):
    """Extract a single scalar from a variable at fixed (cp, x, t).

    Data layout: (n_cp, n_x, n_t).
    """
    data = arr(sol, name)
    if data.ndim == 3:
        return data[cp_idx, x_idx, t_idx]
    elif data.ndim == 2:
        return data[cp_idx, t_idx]
    elif data.ndim == 1:
        return data[t_idx]
    return float(data)


# ═════════════════════════════════════════════════════════════════════
# Plot 1: c_avg consistency check (shell volume integral vs c_p)
# ═════════════════════════════════════════════════════════════════════
def plot_cavg_consistency(sol_cs, N_shell, t_idx, x_idx=0,
                          c_sp1=None, c2_star=None, R_p=None, c_max=None):
    """Compare c_avg from shell volume integral with c_p from the PDF grid."""
    c_p = get_cp_grid(sol_cs)
    n_cp = len(c_p)

    # Read r_p(c_p) at this (t, x)
    r_p_vec = _extract_cp_field(sol_cs, "Shell phase front r_p", t_idx, x_idx)

    # Shell concentrations c_j(c_p) at this (t, x)
    shell_c = np.zeros((N_shell + 1, n_cp))
    shell_c[0, :] = c2_star  # Dirichlet BC
    for j in range(1, N_shell + 1):
        shell_c[j, :] = _extract_cp_field(sol_cs, f"Shell concentration chi_{j}",
                                            t_idx, x_idx)

    # Volume-average: c_avg = (r_p/R_p)^3 * c_sp1 + 3/R_p^3 * int_{r_p}^{R_p} c r^2 dr
    chi = np.linspace(0, 1, N_shell + 1)
    c_avg_shell = np.zeros(n_cp)
    for i in range(n_cp):
        ell = R_p - r_p_vec[i]
        if ell <= 0:
            c_avg_shell[i] = c_sp1
            continue
        r_nodes = r_p_vec[i] + chi * ell
        integrand = shell_c[:, i] * r_nodes**2
        shell_integral = np.trapezoid(integrand, r_nodes)
        core_vol = (r_p_vec[i] / R_p) ** 3
        c_avg_shell[i] = core_vol * c_sp1 + 3 / R_p**3 * shell_integral

    time = arr(sol_cs, "Time [s]")
    t_val = time[t_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(c_p / c_max, c_p / c_max, "k--", lw=0.8, label="$c_p$ (identity)")
    ax.plot(c_p / c_max, c_avg_shell / c_max, "o", ms=3, color="#E74C3C",
            label=r"$\bar{c}$ from shell integral")
    ax.set_xlabel(r"$c_p / c_{\max}$ (PDF grid)")
    ax.set_ylabel(r"$\bar{c} / c_{\max}$ (shell volume integral)")
    ax.set_title(f"Consistency check: $\\bar{{c}}$ vs $c_p$  |  t = {t_val:.1f} s")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ═════════════════════════════════════════════════════════════════════
# Plot 2: Particle cross-section c(r) visualisation
# ═════════════════════════════════════════════════════════════════════
def plot_particle_cross_section(sol_cs, N_shell, t_idx, cp_idx,
                                 c_sp1, c2_star, R_p, c_max, x_idx=0):
    """Draw quarter-circle particle cross-section with c(r) profile."""
    # r_p at this bin
    r_p_val = _extract_scalar(sol_cs, "Shell phase front r_p",
                               cp_idx, t_idx, x_idx)

    # Shell c at this bin
    c_shell = np.zeros(N_shell + 1)
    c_shell[0] = c2_star
    for j in range(1, N_shell + 1):
        c_shell[j] = _extract_scalar(sol_cs, f"Shell concentration chi_{j}",
                                       cp_idx, t_idx, x_idx)

    chi = np.linspace(0, 1, N_shell + 1)
    ell = R_p - r_p_val
    r_shell = r_p_val + chi * ell

    # Build full c(r): core (uniform) + shell (from FDM)
    r_core = np.linspace(0, r_p_val, 30)
    c_core = np.full_like(r_core, c_sp1)

    r_full = np.concatenate([r_core, r_shell])
    c_full = np.concatenate([c_core, c_shell])

    time = arr(sol_cs, "Time [s]")
    t_val = time[t_idx]
    c_p = get_cp_grid(sol_cs)
    theta_bin = c_p[cp_idx] / c_max

    fig, (ax_circ, ax_prof) = plt.subplots(1, 2, figsize=(13, 5),
                                            gridspec_kw={"width_ratios": [1, 1.3]})

    # ── Left: quarter circle ──
    core_color = "#3498DB"
    shell_color = "#E74C3C"
    boundary_color = "#2C3E50"

    core_wedge = Wedge((0, 0), r_p_val * 1e9, 0, 90,
                       facecolor=core_color, alpha=0.6, edgecolor="none",
                       label=r"$\alpha$-core ($c_{sp1}$)")
    shell_wedge = Wedge((0, 0), R_p * 1e9, 0, 90,
                        facecolor=shell_color, alpha=0.35, edgecolor="none",
                        label=r"$\beta$-shell")
    ax_circ.add_patch(shell_wedge)
    ax_circ.add_patch(core_wedge)

    theta_arc = np.linspace(0, np.pi / 2, 50)
    ax_circ.plot(r_p_val * 1e9 * np.cos(theta_arc),
                 r_p_val * 1e9 * np.sin(theta_arc),
                 color=boundary_color, lw=2, ls="--",
                 label=f"$r_p$ = {r_p_val*1e9:.1f} nm")
    ax_circ.plot(R_p * 1e9 * np.cos(theta_arc),
                 R_p * 1e9 * np.sin(theta_arc), color="k", lw=1.5)
    ax_circ.plot([0, R_p * 1e9], [0, 0], "k-", lw=1.5)
    ax_circ.plot([0, 0], [0, R_p * 1e9], "k-", lw=1.5)

    ax_circ.set_xlim(-R_p * 1e9 * 0.05, R_p * 1e9 * 1.15)
    ax_circ.set_ylim(-R_p * 1e9 * 0.05, R_p * 1e9 * 1.15)
    ax_circ.set_aspect("equal")
    ax_circ.set_xlabel("r [nm]")
    ax_circ.set_ylabel("r [nm]")
    ax_circ.legend(fontsize=8, loc="upper right")
    ax_circ.set_title(f"Particle cross-section\nt = {t_val:.1f} s, "
                      r"$\theta_{bin}$" + f" = {theta_bin:.3f}")

    # ── Right: c(r) profile ──
    ax_prof.axvspan(0, r_p_val * 1e9, alpha=0.15, color=core_color)
    ax_prof.axvspan(r_p_val * 1e9, R_p * 1e9, alpha=0.12, color=shell_color)

    ax_prof.plot(r_full * 1e9, c_full / c_max, "k-", lw=2)
    ax_prof.axvline(r_p_val * 1e9, color=boundary_color, ls="--", lw=1.5,
                    label=f"$r_p$ = {r_p_val*1e9:.1f} nm")
    ax_prof.axhline(c2_star / c_max, color=shell_color, ls=":", lw=1,
                    label=r"$c_2^*$")
    ax_prof.axhline(c_sp1 / c_max, color=core_color, ls=":", lw=1,
                    label=r"$c_{sp1}$")

    c_surf = c_shell[-1]
    ax_prof.plot(R_p * 1e9, c_surf / c_max, "s", ms=8, color="#E67E22",
                 zorder=5, label=f"$c_{{surf}}$ = {c_surf/c_max:.4f}")

    ax_prof.set_xlabel("r [nm]")
    ax_prof.set_ylabel(r"$c / c_{\max}$")
    ax_prof.set_title("Radial concentration profile")
    ax_prof.legend(fontsize=8)
    ax_prof.grid(alpha=0.3)

    fig.tight_layout()
    return fig, (ax_circ, ax_prof)


# ═════════════════════════════════════════════════════════════════════
# Plot 3: j_tr_b baseline vs core-shell
# ═════════════════════════════════════════════════════════════════════
def plot_j_tr_b_comparison(sol_bl, sol_cs, t_idx_bl, t_idx_cs, c_max,
                            x_idx=0):
    """Compare branch-B interfacial current density: baseline vs core-shell."""
    c_p_bl = get_cp_grid(sol_bl)
    c_p_cs = get_cp_grid(sol_cs)

    j_bl_t = _extract_cp_field(sol_bl, "CCPM Branch B interfacial current density [A.m-2]",
                                t_idx_bl, x_idx)
    j_cs_t = _extract_cp_field(sol_cs, "CCPM Branch B interfacial current density [A.m-2]",
                                t_idx_cs, x_idx)

    t_bl = arr(sol_bl, "Time [s]")[t_idx_bl]
    t_cs = arr(sol_cs, "Time [s]")[t_idx_cs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(c_p_bl / c_max, j_bl_t, "b-", lw=1.5, label=f"Baseline (t={t_bl:.1f}s)")
    ax1.plot(c_p_cs / c_max, j_cs_t, "r-", lw=1.5, label=f"Core-shell (t={t_cs:.1f}s)")
    ax1.set_ylabel(r"$j_{tr,b}$ [A/m$^2$]")
    ax1.set_title("Branch-B interfacial current density: baseline vs core-shell")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Difference
    if len(c_p_bl) == len(c_p_cs):
        dj = j_cs_t - j_bl_t
        ax2.plot(c_p_bl / c_max, dj, "k-", lw=1.5)
    else:
        dj = np.interp(c_p_bl / c_max, c_p_cs / c_max, j_cs_t) - j_bl_t
        ax2.plot(c_p_bl / c_max, dj, "k-", lw=1.5)

    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel(r"$c_p / c_{\max}$")
    ax2.set_ylabel(r"$\Delta j_{tr,b}$ [A/m$^2$]  (CS $-$ BL)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig, (ax1, ax2)


# ═════════════════════════════════════════════════════════════════════
# Plot 4: PDF mass evolution over time (X-averaged)
# ═════════════════════════════════════════════════════════════════════
def plot_mass_evolution(sol_bl, sol_cs):
    """Compare X-averaged branch masses over time: baseline vs core-shell."""
    t_bl = arr(sol_bl, "Time [s]")
    t_cs = arr(sol_cs, "Time [s]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    colors = {"A": "#3498DB", "B": "#E74C3C", "C": "#27AE60"}

    for i, branch in enumerate(["A", "B", "C"]):
        key = f"X-averaged CCPM Branch {branch} mass"
        m_bl = arr(sol_bl, key)
        m_cs = arr(sol_cs, key)

        axes[i].plot(t_bl, m_bl, "--", color=colors[branch], lw=1.5,
                     label="Baseline")
        axes[i].plot(t_cs, m_cs, "-", color=colors[branch], lw=1.5,
                     label="Core-shell")
        axes[i].set_xlabel("Time [s]")
        axes[i].set_title(f"Branch {branch} mass")
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

    axes[0].set_ylabel("Mass fraction")
    fig.suptitle("PDF mass evolution: baseline vs core-shell", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, axes


# ═════════════════════════════════════════════════════════════════════
# Plot 5: Solid potential comparison
# ═════════════════════════════════════════════════════════════════════
def plot_solid_potential(sol_bl, sol_cs, U_eq_p0=3.397):
    """Compare phi_s - U_eq_p0 vs theta: baseline vs core-shell."""
    theta_bl = arr(sol_bl, "X-averaged positive CCPM stoichiometry")
    theta_cs = arr(sol_cs, "X-averaged positive CCPM stoichiometry")

    phi_s_bl = arr(sol_bl, "X-averaged positive electrode potential [V]")
    phi_s_cs = arr(sol_cs, "X-averaged positive electrode potential [V]")

    eta_bl = phi_s_bl - U_eq_p0
    eta_cs = phi_s_cs - U_eq_p0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(theta_bl, eta_bl * 1e3, "b-", lw=1.5, label="Baseline")
    ax1.plot(theta_cs, eta_cs * 1e3, "r-", lw=1.5, label="Core-shell")
    ax1.set_xlabel(r"$\theta$ (SOC)")
    ax1.set_ylabel(r"$\phi_s - U_{eq,p0}$ [mV]")
    ax1.set_title("Solid potential overpotential vs SOC")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Delta
    theta_com = np.linspace(max(theta_bl.min(), theta_cs.min()),
                            min(theta_bl.max(), theta_cs.max()), 500)
    eta_bl_interp = np.interp(theta_com, theta_bl, eta_bl)
    eta_cs_interp = np.interp(theta_com, theta_cs, eta_cs)
    delta_eta = (eta_cs_interp - eta_bl_interp) * 1e3

    ax2.plot(theta_com, delta_eta, "k-", lw=1.5)
    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel(r"$\theta$ (SOC)")
    ax2.set_ylabel(r"$\Delta(\phi_s - U_{eq,p0})$ [mV]  (CS $-$ BL)")
    ax2.set_title("Solid potential shift from core-shell")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig, (ax1, ax2)


# ═════════════════════════════════════════════════════════════════════
# Plot 6: Phase front r_p(c_p) at time snapshots
# ═════════════════════════════════════════════════════════════════════
def plot_phase_front_evolution(sol_cs, c_max, R_p, c_sp1, c2_star,
                                t_indices=None, x_idx=0):
    """Line plot of r_p(c_p) at several time snapshots."""
    c_p = get_cp_grid(sol_cs)
    time = arr(sol_cs, "Time [s]")

    if t_indices is None:
        t_indices = np.linspace(0, len(time) - 1, 5, dtype=int)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=time[t_indices[0]], vmax=time[t_indices[-1]])

    fig, ax = plt.subplots(figsize=(9, 5))

    for ti in t_indices:
        rp = _extract_cp_field(sol_cs, "Shell phase front r_p", ti, x_idx)
        ax.plot(c_p / c_max, rp * 1e9, color=cmap(norm(time[ti])), lw=1.5,
                label=f"t = {time[ti]:.0f} s")

    ax.axhline(R_p * 1e9, color="k", ls=":", lw=0.8,
               label=f"$R_p$ = {R_p*1e9:.0f} nm")
    ax.axvline(c_sp1 / c_max, color="#3498DB", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(c2_star / c_max, color="#E74C3C", ls="--", lw=0.8, alpha=0.6)

    ax.set_xlabel(r"$c_p / c_{\max}$")
    ax.set_ylabel(r"$r_p$ [nm]")
    ax.set_title("Phase front position across branch-B bins")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ═════════════════════════════════════════════════════════════════════
# Plot 7: c_surf(t) vs c2* for a chosen bin
# ═════════════════════════════════════════════════════════════════════
def plot_csurf_vs_time(sol_cs, N_shell, cp_idx, c2_star, c_max, x_idx=0):
    """Plot surface concentration over time for one branch-B bin."""
    time = arr(sol_cs, "Time [s]")
    c_surf_t = _extract_cp_timeseries(sol_cs, f"Shell concentration chi_{N_shell}",
                                       cp_idx, x_idx)

    c_p = get_cp_grid(sol_cs)
    theta_bin = c_p[cp_idx] / c_max

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, c_surf_t / c_max, "r-", lw=1.5, label=r"$c_{surf}(t)$")
    ax.axhline(c2_star / c_max, color="#E74C3C", ls=":", lw=1,
               label=r"$c_2^*$" + f" = {c2_star/c_max:.4f}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$c_{surf} / c_{\max}$")
    ax.set_title(r"Surface concentration vs time  |  $\theta_{bin}$"
                 + f" = {theta_bin:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ═════════════════════════════════════════════════════════════════════
# Plot 8: Shell profile c(chi) snapshots for one bin
# ═════════════════════════════════════════════════════════════════════
def plot_shell_profile_snapshots(sol_cs, N_shell, cp_idx, c_sp1, c2_star,
                                  c_max, R_p, t_indices=None, x_idx=0):
    """Shell radial profile c(chi) at several time snapshots for one bin."""
    time = arr(sol_cs, "Time [s]")
    c_p = get_cp_grid(sol_cs)
    theta_bin = c_p[cp_idx] / c_max

    if t_indices is None:
        t_indices = np.linspace(0, len(time) - 1, 4, dtype=int)

    chi = np.linspace(0, 1, N_shell + 1)
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=time[t_indices[0]], vmax=time[t_indices[-1]])

    fig, ax = plt.subplots(figsize=(8, 5))

    for ti in t_indices:
        c_shell = np.zeros(N_shell + 1)
        c_shell[0] = c2_star
        for j in range(1, N_shell + 1):
            c_shell[j] = _extract_scalar(sol_cs, f"Shell concentration chi_{j}",
                                          cp_idx, ti, x_idx)

        ax.plot(chi, c_shell / c_max, "o-", color=cmap(norm(time[ti])),
                lw=1.5, ms=5, label=f"t = {time[ti]:.0f} s")

    ax.axhline(c2_star / c_max, color="#E74C3C", ls=":", lw=0.8,
               label=r"$c_2^*$")
    ax.axhline(c_sp1 / c_max, color="#3498DB", ls=":", lw=0.8,
               label=r"$c_{sp1}$")

    ax.set_xlabel(r"$\chi = (r - r_p)/(R_p - r_p)$")
    ax.set_ylabel(r"$c / c_{\max}$")
    ax.set_title(r"Shell profile snapshots  |  $\theta_{bin}$"
                 + f" = {theta_bin:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ═════════════════════════════════════════════════════════════════════
# Plot 9: theta_surf vs theta_avg scatter across branch-B bins
# ═════════════════════════════════════════════════════════════════════
def plot_theta_surf_vs_avg(sol_cs, N_shell, t_idx, c_sp1, c2_star,
                            c_max, x_idx=0):
    """Scatter of theta_surf vs theta_avg = c_p/c_max across branch-B bins."""
    c_p = get_cp_grid(sol_cs)
    theta_avg = c_p / c_max

    c_surf = _extract_cp_field(sol_cs, f"Shell concentration chi_{N_shell}",
                                t_idx, x_idx)
    theta_surf = c_surf / c_max

    # Branch-B mask
    mask = (c_p >= c_sp1) & (c_p <= c2_star)

    time = arr(sol_cs, "Time [s]")
    t_val = time[t_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="identity (no diffusion)")
    ax.scatter(theta_avg[mask], theta_surf[mask], c="#E74C3C", s=12, alpha=0.8,
               label="branch-B bins")
    ax.scatter(theta_avg[~mask], theta_surf[~mask], c="gray", s=5, alpha=0.3,
               label="outside branch B")

    ax.axvline(c_sp1 / c_max, color="#3498DB", ls=":", lw=0.8)
    ax.axvline(c2_star / c_max, color="#E74C3C", ls=":", lw=0.8)

    ax.set_xlabel(r"$\theta_{avg} = c_p / c_{\max}$")
    ax.set_ylabel(r"$\theta_{surf} = c_{surf} / c_{\max}$")
    ax.set_title(r"$\theta_{surf}$ vs $\theta_{avg}$" + f"  |  t = {t_val:.1f} s")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax
