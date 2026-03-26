"""
Visualise the CCPM regularized-delta kernels used in _get_transition_sources.

For each of the four transitions (A→B, B→A, B→C, C→B) this script:
  1. Prints the extraction region (where the scalar flux J is integrated from).
  2. Plots the extraction kernel and the deposition kernel side-by-side.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Grid setup (mirrors the PyBaMM discretisation) ---
c_max = 1.0  # normalised; replace with real c_max if desired
N = 300
eps_c = 1e-8 * c_max
c_min = eps_c
c_max_eff = c_max - eps_c
c_p = np.linspace(c_min, c_max_eff, N)
dc = (c_max_eff - c_min) / (N - 1)
sigma = 1.5 * dc

# --- Branch geometry ---
c1_star = 0.0710 * c_max
c_sp1 = 0.2113 * c_max
c_sp2 = 0.7887 * c_max
c2_star = 0.9290 * c_max

mask_a = (c_p <= c_sp1).astype(float)
mask_b = ((c_p >= c1_star) & (c_p <= c2_star)).astype(float)
mask_c = (c_p >= c_sp2).astype(float)


def regularized_delta(c_target, mask):
    ker = np.exp(-((c_p - c_target) ** 2) / (2 * sigma ** 2)) * mask
    norm = np.trapezoid(ker, c_p)
    if norm < 1e-30:
        return ker * 0.0
    return ker / norm


# ── Extraction kernels (regions integrated to get scalar flux J) ──
# These are the δ(c - c_target) * mask used inside the J integrals
extract = {
    "A→B  (extract from A at c_sp1)": (c_sp1, mask_a),
    "B→A  (extract from B at c1*)":   (c1_star, mask_b),
    "B→C  (extract from B at c2*)":   (c2_star, mask_b),
    "C→B  (extract from C at c_sp2)": (c_sp2, mask_c),
}

# ── Deposition kernels (regions where flux is re-introduced) ──
deposit = {
    "A→B  (deposit into B at c_sp1)": (c_sp1, mask_b),
    "B→A  (deposit into A at c1*)":   (c1_star, mask_a),
    "B→C  (deposit into C at c2*)":   (c2_star, mask_c),
    "C→B  (deposit into B at c_sp2)": (c_sp2, mask_b),
}

# ── Print summary ──
print("=" * 70)
print("EXTRACTION REGIONS  (where scalar flux J is integrated)")
print("=" * 70)
for label, (c_target, mask) in extract.items():
    kern = regularized_delta(c_target, mask)
    nonzero = c_p[kern > 1e-6 * kern.max()] if kern.max() > 0 else []
    if len(nonzero):
        lo, hi = nonzero[0], nonzero[-1]
    else:
        lo, hi = np.nan, np.nan
    peak_idx = np.argmax(kern)
    print(f"  {label}")
    print(f"    target c  = {c_target:.4f}")
    print(f"    support   ≈ [{lo:.4f}, {hi:.4f}]  (where kernel > 0.1% of peak)")
    print(f"    peak at c = {c_p[peak_idx]:.4f},  peak value = {kern[peak_idx]:.2f}")
    print()

print("=" * 70)
print("DEPOSITION REGIONS  (where scalar flux is re-introduced)")
print("=" * 70)
for label, (c_target, mask) in deposit.items():
    kern = regularized_delta(c_target, mask)
    nonzero = c_p[kern > 1e-6 * kern.max()] if kern.max() > 0 else []
    if len(nonzero):
        lo, hi = nonzero[0], nonzero[-1]
    else:
        lo, hi = np.nan, np.nan
    peak_idx = np.argmax(kern)
    print(f"  {label}")
    print(f"    target c  = {c_target:.4f}")
    print(f"    support   ≈ [{lo:.4f}, {hi:.4f}]")
    print(f"    peak at c = {c_p[peak_idx]:.4f},  peak value = {kern[peak_idx]:.2f}")
    print()

# ── Plot ──
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
transitions = ["A→B", "B→A", "B→C", "C→B"]
extract_list = list(extract.items())
deposit_list = list(deposit.items())

for ax, trans, (elabel, (ec, emask)), (dlabel, (dcc, dmask)) in zip(
    axes, transitions, extract_list, deposit_list
):
    e_kern = regularized_delta(ec, emask)
    d_kern = regularized_delta(dcc, dmask)

    ax.fill_between(c_p, e_kern, alpha=0.35, color="tab:red", label=f"Extraction: {elabel}")
    ax.plot(c_p, e_kern, color="tab:red", lw=1.5)
    ax.fill_between(c_p, d_kern, alpha=0.35, color="tab:blue", label=f"Deposition: {dlabel}")
    ax.plot(c_p, d_kern, color="tab:blue", lw=1.5)

    # Mark branch boundaries
    for bc, ls, name in [
        (c1_star, "--", "c1*"),
        (c_sp1, "-.", "c_sp1"),
        (c_sp2, "-.", "c_sp2"),
        (c2_star, "--", "c2*"),
    ]:
        ax.axvline(bc, color="grey", ls=ls, lw=0.8, alpha=0.7)
        ax.text(bc, ax.get_ylim()[1] * 0.5, f" {name}", fontsize=7, color="grey")

    ax.set_ylabel("kernel value")
    ax.set_title(f"Transition {trans}", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

axes[-1].set_xlabel("c / c_max")
fig.suptitle(
    "CCPM regularized-delta kernels: extraction (red) vs deposition (blue)",
    fontsize=13,
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("transition_kernels.png", dpi=150)
plt.show()
print("\nSaved figure to transition_kernels.png")
