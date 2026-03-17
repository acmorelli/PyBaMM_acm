import numpy as np
import pybamm
import matplotlib.pyplot as plt

from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM


# -----------------------------
# Helper
# -----------------------------
def arr(sol, name):
    return np.asarray(sol[name].entries)


def sign_status(a):
    a_min = np.nanmin(a)
    a_max = np.nanmax(a)
    if a_min < 0 and a_max > 0:
        return f"mixed sign   (min={a_min:.3e}, max={a_max:.3e})"
    elif a_max <= 0:
        return f"negative only (min={a_min:.3e}, max={a_max:.3e})"
    else:
        return f"positive only (min={a_min:.3e}, max={a_max:.3e})"


# -----------------------------
# Model / parameters
# -----------------------------
parameter_values = pybamm.ParameterValues("Prada2013")
model = DFN_CCPM(initial_branch="A")

parameter_values.update({
    "Current function [A]": -2.0,
    "Lower voltage cut-off [V]": 2.0,
    "Upper voltage cut-off [V]": 5.0,
})

sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    solver=pybamm.CasadiSolver(mode="safe"),
)

# Short time window first
t_eval = np.linspace(0, 50, 21)

print("solving...")
solution = sim.solve(t_eval=t_eval)
print("Solved.\n")


# -----------------------------
# Extract key diagnostics
# -----------------------------
time = solution["Time [s]"].entries

m_a = arr(solution, "X-averaged CCPM Branch A mass")
m_b = arr(solution, "X-averaged CCPM Branch B mass")
m_c = arr(solution, "X-averaged CCPM Branch C mass")
m_tot = arr(solution, "X-averaged CCPM Total mass")

theta_ccpm = arr(solution, "X-averaged positive CCPM stoichiometry")
voltage = arr(solution, "Terminal voltage [V]")

g_a = arr(solution, "Branch A PDF CCPM")
g_b = arr(solution, "Branch B PDF CCPM")
g_c = arr(solution, "Branch C PDF CCPM")

R_a = arr(solution, "CCPM Branch A lithiation rate")
R_b = arr(solution, "CCPM Branch B lithiation rate")
R_c = arr(solution, "CCPM Branch C lithiation rate")

j_ccpm = arr(solution, "CCPM Positive electrode interfacial current density [A.m-2]")


# -----------------------------
# Print sanity checks
# -----------------------------
print("=== MASS CHECK ===")
print(f"Initial masses:  A={m_a[0]:.6e}, B={m_b[0]:.6e}, C={m_c[0]:.6e}, total={m_tot[0]:.6e}")
print(f"Final masses:    A={m_a[-1]:.6e}, B={m_b[-1]:.6e}, C={m_c[-1]:.6e}, total={m_tot[-1]:.6e}")
print(f"Total mass drift: {m_tot[-1] - m_tot[0]:.6e}")
print(f"Min/Max total mass over time: {np.min(m_tot):.6e} / {np.max(m_tot):.6e}")
print()

print("=== NON-NEGATIVITY CHECK ===")
print(f"min(g_a) = {np.nanmin(g_a):.6e}")
print(f"min(g_b) = {np.nanmin(g_b):.6e}")
print(f"min(g_c) = {np.nanmin(g_c):.6e}")
print()

print("=== RATE SIGN CHECK ===")
print(f"R_a: {sign_status(R_a)}")
print(f"R_b: {sign_status(R_b)}")
print(f"R_c: {sign_status(R_c)}")
print()

print("=== FINITE VALUE CHECK ===")
print(f"Voltage finite?   {np.all(np.isfinite(voltage))}")
print(f"Theta finite?     {np.all(np.isfinite(theta_ccpm))}")
print(f"j_ccpm finite?    {np.all(np.isfinite(j_ccpm))}")
print()

print("=== QUICK STATE SNAPSHOT ===")
print(f"theta_ccpm(0)   = {theta_ccpm[0]:.6e}")
print(f"theta_ccpm(end) = {theta_ccpm[-1]:.6e}")
print(f"V(0)            = {voltage[0]:.6e}")
print(f"V(end)          = {voltage[-1]:.6e}")
print()


# -----------------------------
# Basic pass/fail hints
# -----------------------------
tol_mass = 5e-2
tol_neg = -1e-8

print("=== INTERPRETATION ===")
if abs(m_tot[-1] - m_tot[0]) < tol_mass:
    print("Mass conservation: roughly okay for first debug pass.")
else:
    print("Mass conservation: suspicious -> check advection/source discretisation.")

if np.nanmin(g_a) > tol_neg and np.nanmin(g_b) > tol_neg and np.nanmin(g_c) > tol_neg:
    print("PDF positivity: roughly okay.")
else:
    print("PDF positivity: negative values detected -> likely transport/source issue.")

if np.all(np.isfinite(voltage)) and np.all(np.isfinite(theta_ccpm)) and np.all(np.isfinite(j_ccpm)):
    print("No NaN/Inf in key variables.")
else:
    print("NaN/Inf detected -> inspect kinetics / log / sinh / boundary truncation.")


# -----------------------------
# Plots
# -----------------------------
plt.figure()
plt.plot(time, m_a, label="m_a")
plt.plot(time, m_b, label="m_b")
plt.plot(time, m_c, label="m_c")
plt.plot(time, m_tot, label="m_tot")
plt.xlabel("Time [s]")
plt.ylabel("X-averaged branch mass")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(time, theta_ccpm, label="theta_ccpm")
plt.xlabel("Time [s]")
plt.ylabel("X-averaged CCPM stoichiometry")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(time, voltage, label="Terminal voltage [V]")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(time, j_ccpm[-1,:], label="j_ccpm")
plt.xlabel("Time [s]")
plt.ylabel("Interfacial current density [A.m-2]")
plt.legend()
plt.tight_layout()

plt.show()