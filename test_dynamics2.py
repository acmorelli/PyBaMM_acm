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
model = DFN_CCPM()
print('Model built.')
sim = pybamm.Simulation(
    model,
)

# Short time window first
t_eval = np.linspace(0, 100, 21)

print("solving...")
solution = sim.solve(t_eval=t_eval)
print("Solved.\n")


# -----------------------------
# Extract key diagnostics
# -----------------------------
c_nodes = sim.mesh["CCPM positive particle concentration"].nodes
cp = sim.mesh["CCPM positive particle concentration"].nodes
# -----------------------------
# Extract x-averaged Branch C PDF
# -----------------------------
g_c_xav = solution["X-averaged Branch C PDF CCPM"].entries
# expected shape: (N_c, N_t)

time = solution["Time [s]"].entries

# Pick three times: start / middle / end
i0 = 0
imid = len(time) // 2
iend = len(time) - 1

# profiles in c-space
g0 = g_c_xav[:, i0]
gmid = g_c_xav[:, imid]
gend = g_c_xav[:, iend]

# -----------------------------
# Compute mean concentration of Branch C
# cbar_C(t) = int(g_c * c dc) / int(g_c dc)
# -----------------------------
mass_c = np.trapezoid(g_c_xav, cp, axis=0)
num_c = np.trapezoid(g_c_xav * cp[:, None], cp, axis=0)
cbar_c = num_c / np.maximum(mass_c, 1e-16)

print("=== BRANCH C MEAN CONCENTRATION CHECK ===")
print(f"cbar_C(t=0)     = {cbar_c[i0]:.6e}")
print(f"cbar_C(t=mid)   = {cbar_c[imid]:.6e}")
print(f"cbar_C(t=end)   = {cbar_c[iend]:.6e}")
print(f"Delta cbar_C    = {cbar_c[iend] - cbar_c[i0]:.6e}")
print()

# -----------------------------
# Plot Branch C PDF at 3 times
# -----------------------------
plt.figure()
plt.plot(c_nodes, g0, label=f"t = {time[i0]:.3f} s")
plt.plot(c_nodes, gmid, label=f"t = {time[imid]:.3f} s")
plt.plot(c_nodes, gend, label=f"t = {time[iend]:.3f} s")
plt.xlabel("Concentration c_p [mol m$^{-3}$]")
plt.ylabel("X-averaged Branch C PDF")
plt.title("Branch C PDF in concentration space")
plt.legend()
plt.tight_layout()

# -----------------------------
# Plot mean concentration vs time
# -----------------------------
plt.figure()
plt.plot(time, cbar_c)
plt.xlabel("Time [s]")
plt.ylabel("Branch C mean concentration")
plt.title("Mean concentration of Branch C")
plt.tight_layout()

plt.show()