"""
Manual first-timestep calculation for CCPM Branch A.
Goal: understand why j_tr_a changes sign across the c_p grid.
"""
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Physical constants & Prada2013 parameters
# =============================================================================
F = 96485.33212       # C/mol
R_gas = 8.314462618   # J/(mol·K)
T = 298.0             # K
RT = R_gas * T        # J/mol
RTF = RT / F          # V  (~0.02569 V)

c_p_max_raw = 22806.0  # mol/m³
eps_frac = 1e-8
c_p_max = c_p_max_raw * (1 - eps_frac)   # effective max

c_e_init = 1200.0     # mol/m³ (initial electrolyte conc)
c_e = c_e_init        # assume electrolyte hasn't changed at t=0

R_p = 5e-08           # positive particle radius [m]
beta = 3 / (F * R_p)  # A/FV prefactor for R_a

theta_init = 0.0038
c_init = theta_init * c_p_max_raw  # ≈ 86.66 mol/m³

# Branch boundaries
c1_star = 0.0710 * c_p_max
c_sp1   = 0.2113 * c_p_max
c_sp2   = 0.7887 * c_p_max
c2_star = 0.9290 * c_p_max

# CCPM kinetic parameters (from code)
j_prime_p0 = 2.0      # A/m² (hardcoded constant)
U_eq_p0 = 3.42        # V (LFP plateau reference)
omega = 3.0            # regular solution parameter

print("="*70)
print("KEY PARAMETERS")
print("="*70)
print(f"c_p_max       = {c_p_max:.2f} mol/m³")
print(f"c_init        = {c_init:.2f} mol/m³  (θ = {theta_init})")
print(f"R_p           = {R_p:.2e} m")
print(f"beta = 3/(F·R_p) = {beta:.4e} mol/(A·m²·s)")
print(f"RT/F          = {RTF:.6f} V")
print(f"c_sp1         = {c_sp1:.2f} mol/m³  (θ_sp1 = 0.2113)")
print(f"c1_star       = {c1_star:.2f} mol/m³  (θ1* = 0.071)")
print()

# =============================================================================
# 2. Concentration grid (Branch A: eps_c to c_sp1)
# =============================================================================
n_cp = 300
eps_c = eps_frac * c_p_max_raw
c_p = np.linspace(eps_c, c_sp1, n_cp)
theta_p = c_p / c_p_max
dc = c_p[1] - c_p[0]

print(f"Branch A grid: [{eps_c:.4f}, {c_sp1:.2f}], n={n_cp}, dc={dc:.4f}")
print()

# =============================================================================
# 3. Chemical potential μ(c_p) on the grid
# =============================================================================
mu_over_RT = np.log(theta_p / (1 - theta_p)) + omega * (1 - 2*theta_p)
mu = RT * mu_over_RT  # J/mol

print("="*70)
print("CHEMICAL POTENTIAL μ(c_p)")
print("="*70)
print(f"μ/RT at c_p[0]  (θ={theta_p[0]:.6f}):  {mu_over_RT[0]:.4f}")
print(f"μ/RT at c_p[-1] (θ={theta_p[-1]:.6f}): {mu_over_RT[-1]:.4f}")
print(f"μ at c_p[0]:  {mu[0]:.2f} J/mol")
print(f"μ at c_p[-1]: {mu[-1]:.2f} J/mol")

# Find where μ = 0
idx_mu_zero = np.argmin(np.abs(mu))
print(f"μ = 0 at c_p ≈ {c_p[idx_mu_zero]:.2f} (θ ≈ {theta_p[idx_mu_zero]:.6f})")
print()

# =============================================================================
# 4. Macroscopic OCP at θ_CCPM = 0.0038
# =============================================================================
theta_CCPM = theta_init  # at t=0
U_eq_eff = U_eq_p0 - RTF * (np.log(theta_CCPM / (1 - theta_CCPM)) + omega * (1 - 2*theta_CCPM))

print("="*70)
print("MACROSCOPIC OCP")
print("="*70)
print(f"θ_CCPM = {theta_CCPM}")
print(f"ln(θ/(1-θ)) = {np.log(theta_CCPM/(1-theta_CCPM)):.4f}")
print(f"Ω*(1-2θ)    = {omega*(1-2*theta_CCPM):.4f}")
print(f"μ(θ_CCPM)/RT = {np.log(theta_CCPM/(1-theta_CCPM)) + omega*(1-2*theta_CCPM):.4f}")
print(f"U_eq_eff(θ=0.0038) = {U_eq_eff:.6f} V")
print()

# =============================================================================
# 5. Discharge at C/30: estimate phi_p - phi_e at t=0
# =============================================================================
# At C/30, the cell operates at very low current. At t=0, the voltage is
# approximately the OCV. The positive electrode potential difference:
#   phi_p - phi_e ≈ U_eq_eff(θ_CCPM) - η_activation
#
# For very first instant, let's assume phi_p - phi_e ≈ U_eq_eff
# (the solver will adjust it, but at the very start it's near OCV)
phi_p_minus_phi_e = U_eq_eff  # V

print("="*70)
print("OVERPOTENTIAL ANALYSIS AT t=0")
print("="*70)
print(f"Assuming phi_p - phi_e ≈ U_eq_eff = {phi_p_minus_phi_e:.6f} V at t≈0")
print()

# =============================================================================
# 6. Branch-specific overpotential η_sp(c_p)
# =============================================================================
# η_sp = F*(phi_p - phi_e) - F*U_eq_p0 + μ(c_p)
#       = F*(phi_p - phi_e - 3.42) + μ(c_p)
eta_sp = F * (phi_p_minus_phi_e - U_eq_p0) + mu  # J/mol

# Normalized: η_sp / (2RT) = argument of sinh
eta_norm = eta_sp / (2 * RT)

print(f"F*(phi_p - phi_e - U_eq_p0) = F*{phi_p_minus_phi_e - U_eq_p0:.6f}")
print(f"  = {F*(phi_p_minus_phi_e - U_eq_p0):.2f} J/mol")
print()
print(f"η_sp(c_p[0])  = {eta_sp[0]:.2f} J/mol  →  sinh arg = {eta_norm[0]:.4f}")
print(f"η_sp(c_p[-1]) = {eta_sp[-1]:.2f} J/mol  →  sinh arg = {eta_norm[-1]:.4f}")
print()

# Find where η_sp = 0 (sign change)
idx_eta_zero = np.argmin(np.abs(eta_sp))
print(f"η_sp = 0 at c_p ≈ {c_p[idx_eta_zero]:.2f} (θ ≈ {theta_p[idx_eta_zero]:.6f})")
print(f"  This is where j_tr_a changes sign!")
print()

# =============================================================================
# 7. Compute j_tr_a(c_p) at t=0
# =============================================================================
common_factor = j_prime_p0 * (c_e / c_e_init) ** 0.5  # = j_prime_p0 = 2.0
conc_factor = np.sqrt(theta_p * (1 - theta_p))
j_tr_a = common_factor * conc_factor * np.sinh(eta_norm)  # A/m²

print("="*70)
print("j_tr_a(c_p) AT t=0")
print("="*70)
print(f"j_tr_a at c_p[0]   (θ={theta_p[0]:.6f}):  {j_tr_a[0]:.6e} A/m²")
print(f"j_tr_a at c_p[mid] (θ={theta_p[n_cp//2]:.6f}): {j_tr_a[n_cp//2]:.6e} A/m²")
print(f"j_tr_a at c_p[-1]  (θ={theta_p[-1]:.6f}): {j_tr_a[-1]:.6e} A/m²")
print(f"min(j_tr_a) = {np.min(j_tr_a):.6e} at θ ≈ {theta_p[np.argmin(j_tr_a)]:.6f}")
print(f"max(j_tr_a) = {np.max(j_tr_a):.6e} at θ ≈ {theta_p[np.argmax(j_tr_a)]:.6f}")
print()

idx_j_zero = np.argmin(np.abs(j_tr_a))
print(f"j_tr_a = 0 at θ ≈ {theta_p[idx_j_zero]:.6f} (c_p ≈ {c_p[idx_j_zero]:.2f})")
print()

# =============================================================================
# 8. Compute R_a(c_p) = -beta * j_tr_a
# =============================================================================
R_a = -beta * j_tr_a  # mol/(m³·s)

print("="*70)
print("LITHIATION RATE R_a(c_p) AT t=0")
print("="*70)
print(f"R_a at c_p[0]:   {R_a[0]:.6e} mol/(m³·s)  {'(lithiation)' if R_a[0] > 0 else '(DELITHIATION!)'}")
print(f"R_a at c_p[mid]: {R_a[n_cp//2]:.6e} mol/(m³·s)  {'(lithiation)' if R_a[n_cp//2] > 0 else '(DELITHIATION!)'}")
print(f"R_a at c_p[-1]:  {R_a[-1]:.6e} mol/(m³·s)  {'(lithiation)' if R_a[-1] > 0 else '(DELITHIATION!)'}")
print()
n_pos = np.sum(R_a > 0)
n_neg = np.sum(R_a < 0)
print(f"R_a > 0 (lithiation) at {n_pos}/{n_cp} grid points")
print(f"R_a < 0 (delithiation) at {n_neg}/{n_cp} grid points")
print()

# =============================================================================
# 9. KEY INSIGHT: decompose η_sp
# =============================================================================
print("="*70)
print("ROOT CAUSE ANALYSIS")
print("="*70)
print()
print("The branch-specific overpotential at each c_p is:")
print("  η_sp(c_p) = F*(ϕ_p - ϕ_e - U_eq_p0) + μ(c_p)")
print()
print("At equilibrium (t=0), ϕ_p - ϕ_e = U_eq_eff(θ_CCPM), so:")
print("  η_sp(c_p) = -μ(θ_CCPM) + μ(c_p)")
print(f"  η_sp(c_p) = {-mu[np.argmin(np.abs(theta_p - theta_CCPM))]:.2f} + μ(c_p)")
print()
print("This means η_sp is NEGATIVE where μ(c_p) < μ(θ_CCPM) and")
print("           POSITIVE where μ(c_p) > μ(θ_CCPM).")
print()
print("Since μ is monotonically increasing on [0, θ_sp1=0.2113],")
print("  → ALL c_p > c_init have μ(c_p) > μ(θ_CCPM) → η_sp > 0 → j_tr_a > 0 → R_a < 0")
print("  → ALL c_p < c_init have μ(c_p) < μ(θ_CCPM) → η_sp < 0 → j_tr_a < 0 → R_a > 0")
print()
print(f"Since θ_CCPM = {theta_CCPM} ≈ θ_p[{np.argmin(np.abs(theta_p - theta_CCPM))}],")
print(f"  {n_pos} out of {n_cp} points have R_a > 0 (lithiation)")
print(f"  {n_neg} out of {n_cp} points have R_a < 0 (DELITHIATION)")
print()
print("THIS IS THE BUG: j_tr_a depends on c_p through μ(c_p), creating")
print("an internal redistribution force that acts like diffusion within the PDF.")
print("All particles above the mean θ delithiate, all below lithiate → PDF sharpens.")
print()
print("In Clarke 2026, the lithiation rate R should be UNIFORM across the")
print("concentration grid. The reaction kinetics determine a SINGLE rate per branch,")
print("not a spatially-varying rate. The c_p-dependent BV equation should be")
print("integrated against g(c_p) to produce one scalar j_macro per branch:")
print()
print("  j_macro_A = ∫ g_A(c_p) · j_BV(c_p) dc_p")
print("  R_A = -(3/FR_p) · j_macro_A / m_A")
print()
print("Then R_A is CONSTANT across c_p, and ALL particles in branch A")
print("advect in the same direction during discharge.")

# =============================================================================
# 10. What the correct formulation should give
# =============================================================================
print()
print("="*70)
print("CORRECTED R_a (uniform rate)")
print("="*70)

# Initial PDF: Gamma(2, λ) on branch A
lam = 1.3681e-3 * c_p_max_raw
shifted = c_p - eps_c
g_a_init = shifted * 2 * np.exp(-shifted / lam) * (c_p <= c_sp1)
g_a_init = g_a_init / np.trapezoid(g_a_init, c_p)

# Weighted integral of j_tr_a against g_a (macroscopic current for branch A)
m_a = np.trapezoid(g_a_init, c_p)  # should be 1.0
j_macro_A = np.trapezoid(g_a_init * j_tr_a, c_p)
R_A_uniform = -beta * j_macro_A / m_a

print(f"m_A (integral of g_A) = {m_a:.6f}")
print(f"j_macro_A = ∫ g_A · j_tr_a dc_p = {j_macro_A:.6e} A/m²")
print(f"R_A (uniform) = {R_A_uniform:.6e} mol/(m³·s)")
print(f"  Sign: {'lithiation ✓' if R_A_uniform > 0 else 'DELITHIATION ✗'}")
print()

# =============================================================================
# PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Chemical potential
ax = axes[0, 0]
ax.plot(theta_p, mu_over_RT, 'b-', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(theta_CCPM, color='red', linestyle='--', label=f'θ_CCPM = {theta_CCPM}')
ax.set_xlabel('θ_p = c_p / c_max')
ax.set_ylabel('μ / RT')
ax.set_title('Chemical potential μ(c_p) on branch A grid')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Overpotential
ax = axes[0, 1]
ax.plot(theta_p, eta_norm, 'r-', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(theta_CCPM, color='red', linestyle='--', label=f'θ_CCPM = {theta_CCPM}')
ax.set_xlabel('θ_p = c_p / c_max')
ax.set_ylabel('η_sp / (2RT)')
ax.set_title('Overpotential (sinh argument) at t=0')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: j_tr_a
ax = axes[1, 0]
ax.plot(theta_p, j_tr_a, 'g-', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(theta_CCPM, color='red', linestyle='--', label=f'θ_CCPM = {theta_CCPM}')
ax.set_xlabel('θ_p = c_p / c_max')
ax.set_ylabel('j_tr_a [A/m²]')
ax.set_title('j_tr_a(c_p) at t=0  (BUG: sign changes!)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: R_a and initial PDF
ax1 = axes[1, 1]
color_R = 'tab:blue'
ax1.plot(theta_p, R_a, color=color_R, linewidth=2, label='R_a(c_p) [current]')
ax1.axhline(R_A_uniform, color='tab:green', linestyle='--', linewidth=2, label=f'R_A uniform = {R_A_uniform:.3e}')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('θ_p = c_p / c_max')
ax1.set_ylabel('R_a [mol/(m³·s)]', color=color_R)
ax1.tick_params(axis='y', labelcolor=color_R)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color_g = 'tab:orange'
ax2.plot(theta_p, g_a_init, color=color_g, linewidth=1.5, linestyle=':', label='g_A(c_p) initial')
ax2.set_ylabel('g_A(c_p)', color=color_g)
ax2.tick_params(axis='y', labelcolor=color_g)
ax2.legend(loc='upper right')

ax1.set_title('R_a: current (c_p-dependent) vs corrected (uniform)')

fig.suptitle('Manual First-Timestep Analysis: Why j_tr_a changes sign', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('manual_first_step_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print()
print("Plot saved to manual_first_step_analysis.png")
