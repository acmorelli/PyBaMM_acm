"""
CCPM half-cell test: Li metal || LFP (positive working electrode).

Uses DFN_CCPM with options={"working electrode": "positive"} to remove the
graphite negative electrode and replace it with a lithium metal counter
electrode (fixed OCV = 0 V).
"""

import numpy as np
import pybamm
import matplotlib.pyplot as plt
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM


def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    """
    Exchange-current density for Butler-Volmer reactions between Li metal and
    LiPF6 in EC:DMC [Xu2019].
    """
    m_ref = 3.5e-8 * pybamm.constants.F  # (A/m2)(mol/m3)
    return m_ref * c_Li**0.7 * c_e**0.3


def arr(sol, name):
    return np.asarray(sol[name].entries)


# ── Parameters: start from Prada2013 (LFP cell) then add Li metal param ──
parameter_values = pybamm.ParameterValues("Prada2013")

# Patch Prada2013 graphite-anode params → physically correct Li metal foil.
# Values from Ecker2015_graphite_halfcell / Xu2019 reference sets.
parameter_values.update(
    {
        # Li metal electrode properties
        "Negative electrode OCP [V]": 0.0,
        "Negative electrode conductivity [S.m-1]": 1.0776e7,   # bulk Li metal
        "Negative electrode thickness [m]": 7e-4,              # 700 µm Li foil
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-5,
        "Exchange-current density for lithium metal electrode [A.m-2]"
        "": li_metal_electrolyte_exchange_current_density_Xu2019,
    },
    check_already_exists=False,
)

# ── Model ──────────────────────────────────────────────────────────────
model = DFN_CCPM(
    options={"working electrode": "positive"},
    initial_branch="A",
    mode="discharge",
    parameter_values=parameter_values,
)

# ── Experiment ─────────────────────────────────────────────────────────
experiment = pybamm.Experiment(
    ["Discharge at C/2 for 12000 seconds"],
)

# ── Simulation ─────────────────────────────────────────────────────────
sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"),
)

print("Building & solving half-cell CCPM …")
solution = sim.solve() #1543
sim.save("half_cell_omega_c2_discharge.pkl")
print("Done. Saved to half_cell_omega_c2_discharge.pkl\n")

# ── Diagnostics ────────────────────────────────────────────────────────
time = solution["Time [s]"].entries
voltage = solution["Terminal voltage [V]"].entries
theta = arr(solution, "X-averaged positive CCPM stoichiometry")

m_a = arr(solution, "X-averaged CCPM Branch A mass")
m_b = arr(solution, "X-averaged CCPM Branch B mass")
m_c = arr(solution, "X-averaged CCPM Branch C mass")
m_tot = arr(solution, "X-averaged CCPM Total mass")

print("=== MASS CHECK ===")
print(f"  Initial: A={m_a[0]:.4e}  B={m_b[0]:.4e}  C={m_c[0]:.4e}  tot={m_tot[0]:.6e}")
print(f"  Final:   A={m_a[-1]:.4e}  B={m_b[-1]:.4e}  C={m_c[-1]:.4e}  tot={m_tot[-1]:.6e}")
print(f"  Drift:   {m_tot[-1] - m_tot[0]:.6e}")
print()
print("=== VOLTAGE / θ ===")
print(f"  V range: [{voltage.min():.4f}, {voltage.max():.4f}] V")
print(f"  θ range: [{theta.min():.6f}, {theta.max():.6f}]")
print(f"  Time:    {time[-1]:.1f} s")

# ── Plots ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)

axes[0, 0].plot(time, voltage)
axes[0, 0].set(xlabel="Time [s]", ylabel="Voltage [V]", title="Half-cell voltage")

axes[0, 1].plot(time, theta)
axes[0, 1].set(xlabel="Time [s]", ylabel="θ_CCPM", title="Stoichiometry")

axes[1, 0].plot(time, m_a, label="A")
axes[1, 0].plot(time, m_b, label="B")
axes[1, 0].plot(time, m_c, label="C")
axes[1, 0].legend()
axes[1, 0].set(xlabel="Time [s]", ylabel="Branch mass", title="Branch masses")

axes[1, 1].plot(time, m_tot)
axes[1, 1].set(xlabel="Time [s]", ylabel="Total mass", title="Total mass (conservation)")

plt.savefig("half_cell_discharge_omega_c2.png", dpi=150)
#plt.show()
print("\nPlot saved to half_cell_discharge_omega_c2.png")

# ── Fig 7: OCP shift vs θ ─────────────────────────────────────────────
from postprocess_half_cell_fig7 import plot_fig7

fig7 = plot_fig7(sol_dis=solution, save_path="half_cell_discharge_omega_c2_fig7.png")
plt.show()
plt.close(fig7)
