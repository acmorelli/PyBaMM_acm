import pybamm
from pybamm.models.submodels.interface.kinetics import ccpm_positive_interface
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM

import numpy as np
import matplotlib.pyplot as plt
param = pybamm.ParameterValues("Prada2013")
model = DFN_CCPM(parameter_values=param)
print("Model built.")


# 1. Create the model
model = DFN_CCPM(parameter_values=param)

""" print("=== keys containing CCPM ===")
for k in sorted(model.variables.keys()):
    if "CCPM" in k:
        print(k) """
        
# solve
""" 
""" 
param = pybamm.ParameterValues("Chen2020")
param.update({"Current function [A]": 0})  # 1 A constant current charge?
solver = pybamm.CasadiSolver(mode="safe")
t_eval = np.linspace(0, 10, 21)  # 0 to 100 s

sim = pybamm.Simulation(model, solver=solver, parameter_values=param)
solution = sim.solve(t_eval=t_eval)

# Retrieve CCPM mass variables
m_a = solution["CCPM Branch A mass"].entries
m_b = solution["CCPM Branch B mass"].entries
m_c = solution["CCPM Branch C mass"].entries
m_tot = solution["CCPM Total mass"].entries
cbar_c=solution["CCPM Branch C mean concentration"].entries

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t = solution.t
cp = sim.mesh["CCPM positive particle concentration"].nodes
g = solution["X-averaged Branch C PDF CCPM"].entries

# time derivative
dgdt = np.gradient(g, t, axis=1)

# plot start / mid / end
plt.figure(figsize=(7,4))
plt.plot(cp, dgdt[:, 0], label="dg/dt of x-avg g_c at t=0")
plt.xlabel("c_p")
plt.ylabel("d/dt (x-avg g_c)")
plt.title("Time derivative of x-averaged Branch C PDF")
plt.legend()
plt.grid(True)
plt.show()

""" time=t
plt.figure(figsize=(7,4))
plt.plot(time, m_a.mean(axis=0), label="avg(M_a)")
plt.plot(time, m_b.mean(axis=0), label="avg(M_b)")
plt.plot(time, m_c.mean(axis=0), label="avg(M_c)")
plt.plot(time, m_tot.mean(axis=0), "--", label="avg(M_tot)")
plt.xlabel("Time (s)")
plt.ylabel("x-averaged mass")
plt.title("CCPM branch masses over time")
plt.legend()
plt.grid(True)
plt.show()  """

""" 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t = solution.t
g = solution["Branch C PDF CCPM"].entries  # flattened space x time

cp = sim.built_model.mesh["CCPM positive particle concentration"].nodes
x  = sim.built_model.mesh["positive electrode"].nodes

n_cp = len(cp)
n_x = len(x)
n_t = len(t)

# reshape to (c_p, x, time)
g = g.reshape(n_cp, n_x, n_t)

# x-average -> shape (c_p, time)
g_avg = g.mean(axis=1)

fig, ax = plt.subplots()
line, = ax.plot(cp, g_avg[:, 0])
ax.set_xlabel("c_p")
ax.set_ylabel("g_c(c_p)")
ax.set_title("Branch C PDF CCPM")
txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

def update(k):
    line.set_ydata(g_avg[:, k])
    txt.set_text(f"t = {t[k]:.1f} s")
    return line, txt

ani = FuncAnimation(fig, update, frames=n_t, interval=200, blit=True)
plt.show() """
"""
for i, t in enumerate(time):
    print(
        f"t = {t:8.3f} s | "
        f"M_a = {m_a[i]:.8f}, "
        f"M_b = {m_b[i]:.8f}, "
        f"M_c = {m_c[i]:.8f}, "
        f"M_tot = {m_tot[i]:.8f}"
    ) """

# Simple checks
tol_mass = 1e-6
tol_zero = 1e-6

print("\n=== Checks ===")
print(f"max |M_tot - 1| = {np.max(np.abs(m_tot - 1.0)):.3e}")
print(f"max |M_a|       = {np.max(np.abs(m_a)):.3e}")
print(f"max |M_b|       = {np.max(np.abs(m_b)):.3e}")
print(f"max |M_c - 1|   = {np.max(np.abs(m_c - 1.0)):.3e}")

# Example expectation: all mass initially in branch C
branch_a_near_zero = np.all(np.abs(m_a) < tol_zero)
branch_b_near_zero = np.all(np.abs(m_b) < tol_zero)
branch_c_near_one = np.all(np.abs(m_c - 1.0) < tol_mass)
total_near_one = np.all(np.abs(m_tot - 1.0) < tol_mass)

print(f"Branch A near 0 for all time: {branch_a_near_zero}")
print(f"Branch B near 0 for all time: {branch_b_near_zero}")
print(f"Branch C near 1 for all time: {branch_c_near_one}")
print(f"Total mass near 1 for all time: {total_near_one}") 