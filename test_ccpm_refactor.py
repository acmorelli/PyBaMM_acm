"""
GAUSSIAN source method diagnostic run.
Runs 1C discharge for 2100 s (crosses c_sp1, tests A->B switch).
Uses Prada2013 parameters, 300 c_p points.
Prints mass conservation and branch-switch diagnostics, plots key variables.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

PICKLE_PATH = "ccpm_gaussian_2100s.pkl"


# ── Build & solve ──────────────────────────────────────────────────────────────
print("Building model (gaussian) …")
model = DFN_CCPM(build=True, initial_branch="A", source_method="gaussian")
param = pybamm.ParameterValues("Prada2013")
experiment = pybamm.Experiment(["Discharge at 1C for 2100 seconds"])
sim = pybamm.Simulation(
    model,
    parameter_values=param,
    experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"),
)
print("Solving …")
sol = sim.solve()
print("Done.")

# ── Save solution ──────────────────────────────────────────────────────────────
with open(PICKLE_PATH, "wb") as f:
    pickle.dump({"sol": sol, "sim": sim, "param": param}, f)
print(f"Solution saved to {PICKLE_PATH}")

# ── Extract ────────────────────────────────────────────────────────────────────
time  = sol["Time [s]"].entries
V     = sol["Voltage [V]"].entries
theta = sol["X-averaged positive CCPM stoichiometry"].entries

m_a   = sol["X-averaged CCPM Branch A mass"].entries
m_b   = sol["X-averaged CCPM Branch B mass"].entries
m_c   = sol["X-averaged CCPM Branch C mass"].entries
m_tot = sol["X-averaged CCPM Total mass"].entries
m_tot_raw = np.mean(sol["CCPM Unmasked Total Mass"].entries, axis=0)

J_AB  = sol["Branch A to B scalar flux"].entries
J_BA  = sol["Branch B to A scalar flux"].entries
J_BC  = sol["Branch B to C scalar flux"].entries
J_CB  = sol["Branch C to B scalar flux"].entries

S_err = sol["CCPM source mass balance error"].entries  # should be ~0

# X-averaged PDFs: shape (n_cp, n_t) or (n_cp,) for scalar time
cp = sim.mesh["CCPM positive particle concentration"].nodes
c_max = float(param["Maximum concentration in positive electrode [mol.m-3]"])
c1_star = 0.0710 * c_max;  c_sp1 = 0.2113 * c_max
c_sp2   = 0.7887 * c_max;  c2_star = 0.9290 * c_max

def squeeze2d(arr, n_cp):
    """Return (n_cp, n_t) array regardless of solver output shape."""
    a = np.asarray(arr).squeeze()
    if a.ndim == 1:
        return a[:, None] if a.shape[0] == n_cp else a[None, :]
    return a if a.shape[0] == n_cp else a.T

n_cp = len(cp)
gA = squeeze2d(sol["X-averaged Branch A PDF CCPM"].entries, n_cp)
gB = squeeze2d(sol["X-averaged Branch B PDF CCPM"].entries, n_cp)
gC = squeeze2d(sol["X-averaged Branch C PDF CCPM"].entries, n_cp)

# ── Print diagnostics ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  GAUSSIAN — 2100 s 1C discharge  ({len(time)} timesteps)")
print(f"{'='*65}")
print(f"  Voltage:        {V[0]:.4f} → {V[-1]:.4f} V")
print(f"  θ_CCPM:         {theta[0]:.6f} → {theta[-1]:.6f}  (c_sp1/c_max={c_sp1/c_max:.4f})")
print(f"  MASKED   mass:  {m_tot[0]:.6f} → {m_tot[-1]:.6f}   drift = {m_tot[-1]-m_tot[0]:+.3e}")
print(f"  UNMASKED mass:  {m_tot_raw[0]:.6f} → {m_tot_raw[-1]:.6f}   drift = {m_tot_raw[-1]-m_tot_raw[0]:+.3e}")
print(f"  Source error:   max|∫(Sa+Sb+Sc)|= {np.max(np.abs(S_err)):.3e}")
print(f"\n  Branch masses at t=end:  A={m_a[-1]:.6f}  B={m_b[-1]:.6f}  C={m_c[-1]:.6f}")

print(f"\n  Transition fluxes (x-avg, time-max):")
print(f"    J_AB  max = {np.max(np.abs(J_AB)):.3e}   (A→B at c_sp1, should be >0 during discharge)")
print(f"    J_BA  max = {np.max(np.abs(J_BA)):.3e}   (B→A at c1*,  should be ~0 here)")
print(f"    J_BC  max = {np.max(np.abs(J_BC)):.3e}   (B→C at c2*,  should be ~0 here)")
print(f"    J_CB  max = {np.max(np.abs(J_CB)):.3e}   (C→B at c_sp2,should be ~0 here)")

switched_AB = m_b[-1] > 1e-6
switched_BC = m_c[-1] > 1e-6
print(f"\n  A→B switch occurred: {switched_AB}   (m_B final = {m_b[-1]:.3e})")
print(f"  B→C switch occurred: {switched_BC}   (m_C final = {m_c[-1]:.3e})")

# ── Plots ──────────────────────────────────────────────────────────────────────
i0, imid, iend = 0, len(time)//2, len(time)-1

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("CCPM Gaussian — 2100 s 1C discharge", fontsize=13)

# 1) Branch masses
ax = axes[0, 0]
ax.plot(time, m_a, label="m_A"); ax.plot(time, m_b, label="m_B")
ax.plot(time, m_c, label="m_C"); ax.plot(time, m_tot, "k--", label="m_tot (masked)")
ax.plot(time, m_tot_raw, "k:", label="m_tot (unmasked)")
ax.set_xlabel("Time [s]"); ax.set_ylabel("X-avg branch mass"); ax.set_title("Branch masses")
ax.legend(fontsize=8)

# 2) Transition fluxes
ax = axes[0, 1]
ax.plot(time, np.squeeze(J_AB), label="J_AB (A→B)")
ax.plot(time, np.squeeze(J_BA), label="J_BA (B→A)")
ax.plot(time, np.squeeze(J_BC), label="J_BC (B→C)")
ax.plot(time, np.squeeze(J_CB), label="J_CB (C→B)")
ax.set_xlabel("Time [s]"); ax.set_ylabel("Scalar flux"); ax.set_title("Transition fluxes")
ax.legend(fontsize=8)

# 3) θ and source error
ax = axes[0, 2]
ax2 = ax.twinx()
ax.plot(time, theta, "b-", label="θ_CCPM"); ax.axhline(c_sp1/c_max, ls="--", color="gray", label="c_sp1/c_max")
ax2.plot(time, np.abs(np.squeeze(S_err)), "r-", alpha=0.6, label="|source err|")
ax.set_xlabel("Time [s]"); ax.set_ylabel("θ", color="b"); ax2.set_ylabel("|∫S|", color="r")
ax.set_title("Stoichiometry & source error"); ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

# 4-6) PDFs at 3 snapshots
vlines = [(c1_star, "c1*"), (c_sp1, "csp1"), (c_sp2, "csp2"), (c2_star, "c2*")]
for idx_ax, (gX, label) in enumerate([(gA, "A"), (gB, "B"), (gC, "C")]):
    ax = axes[1, idx_ax]
    ax.plot(cp, gX[:, i0],   label=f"t={time[i0]:.0f}s")
    ax.plot(cp, gX[:, imid], label=f"t={time[imid]:.0f}s")
    ax.plot(cp, gX[:, iend], label=f"t={time[iend]:.0f}s")
    for xv, lbl in vlines:
        ax.axvline(xv, ls="--", lw=0.8, alpha=0.6, label=lbl)
    ax.set_xlabel("c_p [mol/m³]"); ax.set_ylabel("X-avg PDF")
    ax.set_title(f"Branch {label} PDF"); ax.legend(fontsize=7)

plt.tight_layout()
plt.show()

