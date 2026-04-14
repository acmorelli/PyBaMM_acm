"""Debug core-shell charge result — analyze mass instability."""
import numpy as np
import pybamm

def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    m_ref = 3.5e-8 * pybamm.constants.F
    return m_ref * c_Li**0.7 * c_e**0.3

def arr(sol, name):
    return np.asarray(sol[name].entries)

# Load core-shell and baseline results
sim_cs = pybamm.load("half_cell_omega_2c_charge_core_shell.pkl")
sol_cs = sim_cs.solution

sim_bl = pybamm.load("half_cell_omega_2c_charge.pkl")
sol_bl = sim_bl.solution

for label, sol in [("CORE-SHELL", sol_cs), ("BASELINE", sol_bl)]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    time = sol["Time [s]"].entries
    V = sol["Terminal voltage [V]"].entries
    theta = arr(sol, "X-averaged positive CCPM stoichiometry")
    m_a = arr(sol, "X-averaged CCPM Branch A mass")
    m_b = arr(sol, "X-averaged CCPM Branch B mass")
    m_c = arr(sol, "X-averaged CCPM Branch C mass")
    m_tot = arr(sol, "X-averaged CCPM Total mass")

    print(f"Time: {time[0]:.1f} to {time[-1]:.1f} s, {len(time)} pts")
    print(f"V:     [{V.min():.4f}, {V.max():.4f}]")
    print(f"theta: [{theta.min():.6f}, {theta.max():.6f}]")
    print(f"m_tot drift: {m_tot[-1]-m_tot[0]:.4e}")

    # Sampled time series
    indices = np.linspace(0, len(time)-1, 25, dtype=int)
    print(f"\n{'t':>8} {'m_a':>10} {'m_b':>10} {'m_c':>10} {'theta':>8} {'V':>8}")
    for i in indices:
        print(f"{time[i]:8.1f} {m_a[i]:10.4e} {m_b[i]:10.4e} {m_c[i]:10.4e} {theta[i]:8.5f} {V[i]:8.4f}")

    # Mass rate analysis
    dt = np.diff(time)
    dm_a = np.diff(m_a) / dt
    dm_b = np.diff(m_b) / dt
    dm_c = np.diff(m_c) / dt

    print(f"\ndm_a/dt range: [{dm_a.min():.4e}, {dm_a.max():.4e}]")
    print(f"dm_b/dt range: [{dm_b.min():.4e}, {dm_b.max():.4e}]")
    print(f"dm_c/dt range: [{dm_c.min():.4e}, {dm_c.max():.4e}]")

    # Detect staircase: where mass changes direction
    sign_changes_a = np.sum(np.diff(np.sign(dm_a)) != 0)
    sign_changes_b = np.sum(np.diff(np.sign(dm_b)) != 0)
    sign_changes_c = np.sum(np.diff(np.sign(dm_c)) != 0)
    print(f"\nSign changes: dm_a={sign_changes_a}, dm_b={sign_changes_b}, dm_c={sign_changes_c}")

    # Locate largest oscillations in m_b
    idx_big = np.argsort(np.abs(dm_b))[-10:]
    print(f"\nLargest |dm_b/dt| events:")
    for i in sorted(idx_big):
        print(f"  t={time[i]:8.1f}s  dm_b/dt={dm_b[i]:+.4e}  m_a={m_a[i]:.4e}  m_b={m_b[i]:.4e}  m_c={m_c[i]:.4e}  theta={theta[i]:.5f}")

# ── Plot comparison ────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(14, 8), tight_layout=True)
fig.suptitle("Core-shell vs Baseline — 2C charge")

for label, sol, ls in [("Baseline", sol_bl, "--"), ("Core-shell", sol_cs, "-")]:
    t = sol["Time [s]"].entries
    V = sol["Terminal voltage [V]"].entries
    th = arr(sol, "X-averaged positive CCPM stoichiometry")
    ma = arr(sol, "X-averaged CCPM Branch A mass")
    mb = arr(sol, "X-averaged CCPM Branch B mass")
    mc = arr(sol, "X-averaged CCPM Branch C mass")
    mt = arr(sol, "X-averaged CCPM Total mass")

    axes[0,0].plot(t, V, ls, label=label)
    axes[0,1].plot(t, th, ls, label=label)
    axes[0,2].plot(t, mt, ls, label=label)
    axes[1,0].plot(t, ma, ls, label=label)
    axes[1,1].plot(t, mb, ls, label=label)
    axes[1,2].plot(t, mc, ls, label=label)

for ax, title in zip(axes.flat, ["Voltage", "theta", "Total mass", "m_a", "m_b", "m_c"]):
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("Time [s]")

plt.savefig("debug_core_shell_vs_baseline.png", dpi=150)
print("\nPlot saved to debug_core_shell_vs_baseline.png")
