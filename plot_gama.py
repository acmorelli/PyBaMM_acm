import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.optimize import brentq

c_max = 22806.0
c_init = 0.0038 * c_max
eps_c = 1e-8 * c_max
c_min = eps_c
c_sp1 = 0.2113 * c_max

dc = (c_max - 2*eps_c) / 299.0
c_p = np.linspace(eps_c, c_max - eps_c, 300)

def make_g(lam_val):
    shifted = c_p - c_min
    gv = shifted * np.exp(-shifted / lam_val) * (c_p <= c_sp1)
    norm = trapezoid(gv, c_p)
    return gv / norm if norm > 1e-30 else gv

def discrete_mean(lam_val):
    return trapezoid(make_g(lam_val) * c_p, c_p)

# Solve for lam such that the discrete mean = c_init exactly
lam = brentq(lambda l: discrete_mean(l) - c_init, dc * 0.01, c_sp1)
g = make_g(lam)

# verify mean
mean_c = trapezoid(g * c_p, c_p)
print(f'c_max  = {c_max:.0f}')
print(f'c_init = {c_init:.1f}  (theta={c_init/c_max:.4f})')
print(f'lambda = {lam:.1f}')
print(f'dc     = {dc:.1f}')
print(f'mean   = {mean_c:.1f}  (theta={mean_c/c_max:.4f})')




fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(c_p / c_max, g * c_max, 'b-', lw=2)
ax.axvline(c_init / c_max, color='r', ls='--', label=f'c_init (theta={c_init/c_max:.4f})')
ax.axvline(c_sp1 / c_max, color='g', ls='--', alpha=0.5, label=f'c_sp1 (theta={c_sp1/c_max:.4f})')
ax.set_xlim(-0.002, 0.04)
ax.set_xlabel('c / c_max (stoichiometry)')
ax.set_ylabel('g(c) * c_max (normalised PDF)')
ax.set_title(f'Gamma(2, lam) initial PDF  |  lam={lam:.1f}, mean={mean_c:.1f} mol/m3')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('initial_pdf_gamma.png', dpi=150)
print('Saved initial_pdf_gamma.png')